"""Wave engine: crash-idempotent Batch API submission/harvest.

A *wave* is one request set executed against the Batch API. Artifacts live in
one wave directory:

- ``requests.jsonl``     frozen request set (custom_id, params, lineage)
- ``REQUEST_SET.json``   request_set_hash + template hashes + upstream hash
- ``ledger.jsonl``       append-only, fsync'd pre-create journal (see below)
- ``results-<chunk>.jsonl``  harvested raw results, atomically renamed
- ``raw_results.jsonl``  merged normalized outcomes keyed by custom_id
- ``STAMP.json``         producer's completeness stamp — completeness is judged
                         ONLY by the stamp, never by counting lines

Crash-idempotence (the P1-4 ledger, rev 3):

- Chunk states: ``journaled -> submitted -> ended -> harvested``.
- ``journaled`` is appended (fsync'd) BEFORE any network call, carrying a
  fresh per-attempt uuid4 nonce AND the reconciliation mode
  (``nonce_metadata``) in force — resume reconciles strictly under the
  journaled mode and hard-errors if the engine's mode has flipped, so a
  rewritten smoke verdict can never turn an interrupted no-nonce submit
  into a "proven absent -> re-create" double submit.
- Run identity is salted into every WIRE custom_id (``run_tag``, an 8-hex
  hash of the run_id): requests are submitted as ``<tag>-<custom_id>`` and
  harvest verifies set-equality on the TAGGED ids, so a same-shape batch
  from another run/config in the listing window can be adopted at most
  transiently — it always fails harvest verification and is rolled back,
  never stamped. Ledgers/artifacts keep the logical (untagged) ids.
- ``create`` runs with ``max_retries=0`` — transparent SDK retries on create
  are exactly how a double-submit happens. A clean HTTP failure (4xx/429) is
  unambiguously "not created": 429 waits and retries (bounded), other 4xx
  raise. A transport failure (timeout, connection error) leaves the chunk
  ``journaled``-terminal — the SAME durable state a hard crash leaves, so
  every interrupted submit is ambiguous by definition and resume ALWAYS goes
  through reconciliation, never straight to re-create.
- Reconciliation (bounded visibility wait, then ``batches.list()`` over the
  creation window):
  - nonce branch (batch ``metadata`` round-trips, per the smoke verdict): a
    listed batch carrying the journaled nonce is adopted; proven absence
    re-creates with a fresh nonce+journal entry. Exactly-once holds here.
  - no-nonce branch: candidates are unledgered window batches whose
    request_counts.total matches. Exactly one -> adopt; zero or several ->
    HALT the wave for manual resolution (absence cannot be proven; the system
    is honestly at-most-once-with-halt).
  - every adoption is verified at harvest by exact custom_id set-equality;
    a mismatch rolls the adoption back (batch id excluded) and the chunk
    returns to ``journaled``-terminal.
- Harvest streams to a tmp file, validates exactly one result per expected
  custom_id, then atomically renames. Merged outcomes are keyed by custom_id,
  so a re-harvest can never produce duplicates.
"""

from __future__ import annotations

import datetime as _dt
import fcntl
import hashlib
import json
import os
import re
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import anthropic

from .io import atomic_write, iter_jsonl, read_json, write_jsonl_atomic
from .schema import validate_custom_id

Hooks = Callable[[str, dict], None]


class WaveError(RuntimeError):
    pass


class WaveHashMismatch(WaveError):
    """The same run+wave was rebuilt with a different request set — genuine
    template/config drift. Pilot/production/top-ups are namespaced apart, so
    this never fires across runs."""


class WaveHalt(WaveError):
    """No-nonce reconciliation could not prove absence (zero or multiple
    candidates). Manual resolution required; NEVER auto-resubmit."""


class WaveCredentialRejected(WaveHalt):
    """The API REJECTED the credential the client was built on -- a 401 / 403 on
    a request, or, under ``SISTEER_SECRETS_MODE=wif``, a federation identity the
    token exchange refused (``client.credential_rejection``) -- raised by the live
    engine at the FIRST such answer and never retried: the outstanding pool is
    cancelled, nothing is stamped, the rejected requests leave no cache (a
    rejection is the credential's outcome, never an item's: a re-run re-asks
    them), and the engine issues nothing else for the rest of the process. The
    message names the status, the SDK's error type, the API's own error type
    when the body carried one, the mode and the fix per mode -- never a key, a
    token or request text. Previously a 401 ``APIStatusError`` was one item's
    errored outcome, every scheduled request went out, and the retry rounds
    re-issued the errored ones."""

    def __init__(self, status: int, error_type: str, *, api_error_type: str | None = None,
                 mode: str = "op", fix: str = "the credential"):
        self.status = int(status)
        self.error_type = str(error_type)
        self.api_error_type = api_error_type
        self.mode = mode
        super().__init__(
            f"the API rejected the credential: {self.error_type} (HTTP {self.status}"
            f"{', ' + api_error_type if api_error_type else ''}) under SISTEER_SECRETS_MODE={mode} -- stopped at the first "
            f"credential rejection (D6): no retry, no further request on this credential, nothing stamped; fix {fix} and re-run "
            f"(every outcome obtained before the stop is cached and a re-run resumes from it)")


def credential_halt(exc: BaseException) -> WaveCredentialRejected | None:
    """The typed halt for ``exc`` when it is the API's credential rejection (``client.credential_rejection``: a 401 / 403 on
    the request, or a federation identity the token exchange refused) -- the mode read leniently (a refusal message never
    itself refuses on a fourth mode value) and the fix per mode -- else None. The ONE builder the live engine, the batch
    engine and the scripts share (the WIF driver sweep, 2026-10-13)."""
    from .client import credential_fix_hint, credential_rejection, secrets_mode

    if isinstance(exc, WaveCredentialRejected):
        return exc                                   # already the typed halt (round 14: the client's latched layer raises the held one)
    rej = credential_rejection(exc)
    if rej is None:
        return None
    try:
        mode = secrets_mode()
    except Exception:  # noqa: BLE001 -- lenient on purpose
        mode = "unknown"
    return WaveCredentialRejected(rej.status, rej.error_type, api_error_type=rej.api_error_type, mode=mode,
                                  fix=credential_fix_hint(mode))


@contextmanager
def credential_exit(prefix: str = ""):
    """For a script's ``main`` or the CLI's group (the twin of ``sisteer.vault.credential_exit``; the WIF driver sweep,
    round 9): a credential rejection that reaches it -- the typed :class:`WaveCredentialRejected` an engine raised, or the
    SDK's own 401 / 403 out of a single call -- ends the process as ``SystemExit`` whose message is the typed text (the
    terminal line), never a traceback and never a report written over it. Every other exception passes."""
    try:
        yield
    except WaveCredentialRejected as e:
        raise SystemExit(f"{prefix}{e}")
    except Exception as e:  # noqa: BLE001 -- classified: the credential only
        halt = credential_halt(e)
        if halt is None:
            raise
        raise SystemExit(f"{prefix}{halt}")


#: the attribute under which the FIRST credential rejection is latched on the CLIENT: sibling
#: engines over one client -- `corpusgen run --corpus all`, the extension sub-DAGs, `onpolicy score` (pipeline.make_engine
#: builds one WaveEngine per thread) -- each used to issue its own doomed create after the first engine's rejection; the
#: halt now rides the client they share, so every sibling refuses before its next control-plane call. Defined by the client
#: module since round 14 (its latched retry layer reads and holds the same latch before every attempt, a federated copy
#: through the client it was copied from); re-exported here under its name.
from .client import CLIENT_LATCH_ATTR, held_credential_rejection, hold_credential_rejection  # noqa: E402


class WaveInvalidRequest(WaveError):
    """The API called a request invalid — our bug; surface and stop."""


class ChunkTooLarge(WaveError):
    pass


# ---------------------------------------------------------------- requests


@dataclass
class WaveRequest:
    custom_id: str
    params: dict  # MessageCreateParamsNonStreaming-shaped dict
    lineage: dict = field(default_factory=dict)

    def encoded_size(self) -> int:
        return len(
            json.dumps(
                {"custom_id": self.custom_id, "params": self.params},
                ensure_ascii=False,
            ).encode("utf-8")
        )


def request_set_hash(
    requests: list[WaveRequest],
    template_hashes: dict[str, str],
    upstream_hash: str = "",
) -> str:
    h = hashlib.sha256()
    for r in sorted(requests, key=lambda r: r.custom_id):
        h.update(r.custom_id.encode())
        h.update(b"\x00")
        h.update(json.dumps(r.params, sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\x01")
    h.update(json.dumps(template_hashes, sort_keys=True).encode())
    h.update(upstream_hash.encode())
    return h.hexdigest()


def _ids_hash(custom_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(custom_ids)).encode()).hexdigest()


def wire_tag(run_id: str) -> str:
    """8-hex run-identity tag salted into every wire custom_id, so candidate
    adoption and harvest set-equality are run-scoped (charset [0-9a-f] keeps
    the tagged id inside [A-Za-z0-9_-]{1,64}, validated per request)."""
    return hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:8]


def _utcnow() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _wave_lock(wave_dir: Path):
    """Exclusive flock held for the duration of a wave execution."""
    with (wave_dir / ".lock").open("a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


# ---------------------------------------------------------------- chunking


def plan_chunks(
    requests: list[WaveRequest],
    pack_target_bytes: int,
    hard_cap_bytes: int,
    max_requests: int,
    prefix: str = "c",
) -> dict[str, list[WaveRequest]]:
    """Greedy packing by measured encoded bytes (and request count).

    Any single request above hard_cap_bytes is refused outright; greedy packs
    to pack_target_bytes so no chunk can exceed the refusal boundary. Sizes
    are measured on the LOGICAL ids; the run tag adds 9 bytes/request on the
    wire (≤0.9 MB per 100k-request chunk), absorbed by the 20 MB pack-vs-cap
    margin (preflight measures the tagged size).
    """
    chunks: dict[str, list[WaveRequest]] = {}
    cur: list[WaveRequest] = []
    cur_bytes = 0
    idx = 0

    def _close():
        nonlocal cur, cur_bytes, idx
        if cur:
            chunks[f"{prefix}{idx:03d}"] = cur
            idx += 1
            cur, cur_bytes = [], 0

    for r in requests:
        size = r.encoded_size()
        if size > hard_cap_bytes:
            raise ChunkTooLarge(
                f"request {r.custom_id} encodes to {size} bytes > hard cap "
                f"{hard_cap_bytes} — refusing"
            )
        if cur and (cur_bytes + size > pack_target_bytes or len(cur) >= max_requests):
            _close()
        cur.append(r)
        cur_bytes += size
    _close()
    return chunks


# ---------------------------------------------------------------- ledger


@dataclass
class ChunkState:
    state: str  # absent | journaled | submitted | harvested
    attempt: int = 0
    nonce: str | None = None
    batch_id: str | None = None
    excluded_batch_ids: set[str] = field(default_factory=set)
    journaled_at: str | None = None
    nonce_metadata: bool | None = None  # reconciliation mode in force at journal time


class Ledger:
    """Append-only fsync'd JSONL ledger for one wave."""

    def __init__(self, path: Path):
        self.path = path

    def append(self, entry: dict) -> None:
        entry = {"ts": _utcnow(), **entry}
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def entries(self) -> list[dict]:
        return list(iter_jsonl(self.path, missing_ok=True))

    def chunk_states(self) -> dict[str, ChunkState]:
        """Fold the whole ledger into per-chunk states in ONE pass."""
        states: dict[str, ChunkState] = {}
        for e in self.entries():
            chunk = e.get("chunk")
            if chunk is None:
                continue
            st = states.setdefault(chunk, ChunkState(state="absent"))
            kind = e["state"]
            if kind == "journaled":
                st.state = "journaled"
                st.attempt = e.get("attempt", st.attempt + 1)
                st.nonce = e.get("nonce")
                st.batch_id = None
                st.journaled_at = e.get("ts")
                if "nonce_metadata" in e:
                    st.nonce_metadata = bool(e["nonce_metadata"])
            elif kind == "submitted":
                st.state = "submitted"
                st.batch_id = e.get("batch_id")
            elif kind == "ended":
                pass  # bookkeeping only
            elif kind == "rolled_back":
                if e.get("batch_id"):
                    st.excluded_batch_ids.add(e["batch_id"])
                st.state = "journaled"
                st.batch_id = None
            elif kind == "harvested":
                st.state = "harvested"
        return states

    def chunk_state(self, chunk: str) -> ChunkState:
        return self.chunk_states().get(chunk, ChunkState(state="absent"))

    def known_batch_ids(self) -> set[str]:
        return {e["batch_id"] for e in self.entries() if e.get("batch_id")}


# ---------------------------------------------------------------- results


def message_text(content: Any) -> str:
    """Concatenated text blocks of a message's content — the ONE extractor."""
    return "".join(b.text for b in content if getattr(b, "type", "") == "text")


def batch_nonce(batch: Any) -> str | None:
    """The ONE reader of the corpusgen nonce off a batch object (SDK models may
    surface unrecognized body fields via ``model_extra`` instead of an attr)."""
    meta = getattr(batch, "metadata", None)
    if meta is None:
        meta = (getattr(batch, "model_extra", None) or {}).get("metadata")
    if isinstance(meta, dict):
        return meta.get("corpusgen_nonce")
    return getattr(meta, "corpusgen_nonce", None) if meta is not None else None


def normalize_result(entry: Any) -> dict:
    """SDK/fake batch result entry -> normalized outcome dict."""
    out: dict[str, Any] = {"custom_id": entry.custom_id}
    rtype = entry.result.type
    if rtype == "succeeded":
        msg = entry.result.message
        text = message_text(msg.content)
        out.update(
            {
                "type": "succeeded",
                "stop_reason": getattr(msg, "stop_reason", None),
                "text": text,
            }
        )
    elif rtype == "errored":
        err = entry.result.error
        inner = getattr(err, "error", err)
        out.update(
            {
                "type": "errored",
                "error_type": getattr(inner, "type", "unknown"),
                "error_message": getattr(inner, "message", ""),
            }
        )
    else:  # canceled / expired
        out.update({"type": rtype})
    return out


# Per-request server-side rejections of WELL-FORMED requests are a long-tail
# class, not our bug — two production halts proved it (2026-09-01): the OUTPUT
# content filter ("Output blocked by content filtering policy", 3/9,000 s3
# drafts) and a generic content-dependent rejection ("Invalid request data",
# 2/17,846 s5 scores; no size cliff, params identical to 17,844 successes).
# Per-signature whitelisting is whack-a-mole, so tolerance is CAP-BASED:
#
# - content filtering keeps its own status (content_filtered);
# - any other invalid_request whose message does NOT match a recognized
#   our-bug pattern is `request_rejected`, tolerated per-request up to
#   max_rejected_frac of the wave (default 0.1%) — above the cap the
#   rejection is systematic, i.e. OUR bug, and the wave raises
#   WaveInvalidRequest exactly as before;
# - messages matching INVALID_OUR_BUG_RE (deterministic malformed-request
#   shapes: oversize, missing/extra fields, bad model, ...) stay fail-loud
#   on the FIRST occurrence.
CONTENT_FILTER_RE = re.compile(r"content filtering", re.IGNORECASE)
INVALID_OUR_BUG_RE = re.compile(
    r"too long|too many|exceed|max_tokens|at least|at most|must be|should be|"
    r"is required|missing|extra inputs|not permitted|unexpected|not supported|"
    r"invalid model|does not exist",
    re.IGNORECASE,
)


def outcome_status(outcome: dict) -> str:
    """Terminal per-request status for stamping/reporting.

    Applied at read time, so already-harvested results re-classify on resume
    without touching the stored raw outcomes."""
    t = outcome["type"]
    if t == "succeeded":
        sr = outcome.get("stop_reason")
        if sr == "refusal":
            return "refusal"
        if sr == "max_tokens":
            return "max_tokens"
        return "succeeded"
    if t == "errored":
        msg = outcome.get("error_message", "")
        if CONTENT_FILTER_RE.search(msg):
            return "content_filtered"
        et = outcome.get("error_type", "")
        if "invalid_request" in et:
            return (
                "invalid_request" if INVALID_OUR_BUG_RE.search(msg) else "request_rejected"
            )
        return "errored"
    return t  # canceled / expired


# canceled is deliberately NOT retryable: a batch only cancels through an
# explicit batches.cancel — a deliberate operator action — so the wave halts
# for manual resolution instead of quietly resubmitting the operator's cancel.
# content_filtered / request_rejected are not ENGINE-retryable either: like
# refusals they are handled at the STAGE level (structural waves repair the
# missing slots; leaf waves repair request_rejected — a fresh custom_id gives
# a transient rejection its second chance — and absorb the rest as attrition).
RETRYABLE_STATUSES = {"errored", "expired"}


# ---------------------------------------------------------------- engine


class WaveEngine:
    def __init__(
        self,
        client: Any,
        *,
        model: str = "claude-opus-5",
        pack_target_bytes: int = 180 * 1024 * 1024,
        hard_cap_bytes: int = 200 * 1024 * 1024,
        max_requests_per_batch: int = 100_000,
        poll_interval_s: float = 60.0,
        visibility_wait_s: float = 120.0,
        retry_rounds: int = 2,
        create_429_retries: int = 5,
        max_rejected_frac: float = 0.001,
        nonce_metadata: bool = False,
        run_tag: str = "",
        sleep: Callable[[float], None] = time.sleep,
        hooks: Hooks | None = None,
        on_event: Callable[[str], None] | None = None,
    ):
        #: the FIRST credential rejection this engine meets -- at a batch create, a poll, a results read or a list --
        #: latched as the typed WaveCredentialRejected; every later run_wave raises it before any request (the live
        #: engine's rule, LiveEngine._credential_rejected)
        self._credential_rejected: WaveCredentialRejected | None = None
        self.client = client
        self.model = model
        self.pack_target_bytes = pack_target_bytes
        self.hard_cap_bytes = hard_cap_bytes
        self.max_requests_per_batch = max_requests_per_batch
        self.poll_interval_s = poll_interval_s
        self.visibility_wait_s = visibility_wait_s
        self.retry_rounds = retry_rounds
        self.create_429_retries = create_429_retries
        self.max_rejected_frac = max_rejected_frac
        self.nonce_metadata = nonce_metadata
        self.run_tag = run_tag  # run identity salted into wire custom_ids
        self.sleep = sleep
        self.hooks = hooks or (lambda point, info: None)
        self.on_event = on_event or (lambda msg: None)
        self.reconcile_count = 0  # observable for tests

    # -- wire ids ------------------------------------------------------------

    def _wire(self, custom_id: str) -> str:
        """Logical -> wire custom_id (run identity salted in)."""
        return f"{self.run_tag}-{custom_id}" if self.run_tag else custom_id

    def _unwire(self, wire_id: str) -> str:
        prefix = f"{self.run_tag}-"
        if self.run_tag and wire_id.startswith(prefix):
            return wire_id[len(prefix):]
        return wire_id

    # -- public ------------------------------------------------------------

    def run_wave(
        self,
        wave_dir: Path,
        requests: list[WaveRequest],
        *,
        wave_name: str,
        template_hashes: dict[str, str] | None = None,
        upstream_hash: str = "",
        description: str = "",
        rs_hash: str | None = None,
        max_rejected_frac: float | None = None,
    ) -> dict[str, dict]:
        """Execute a wave to completion; returns custom_id -> outcome dict.

        Fully resumable: stamped waves are skipped (hash-checked); submitted
        batches are re-polled; journaled-terminal chunks are reconciled.
        ``rs_hash`` lets a caller that already hashed this request set (the
        stage runner) skip the second full-set hash. ``max_rejected_frac``
        overrides the engine's rejection cap for THIS wave — repair waves are
        adversely selected (they contain only previously-failed requests), so
        the stage runner lifts the cap there; the base wave already proved the
        rejection rate isn't systematic.
        """
        held = self._held_credential_rejection()
        if held is not None:
            raise held                               # Latched (this engine's, or a sibling's on the shared client) -- no request
        template_hashes = template_hashes or {}
        wave_dir = Path(wave_dir)
        wave_dir.mkdir(parents=True, exist_ok=True)

        ids = [r.custom_id for r in requests]
        for cid in ids:
            validate_custom_id(cid)
            validate_custom_id(self._wire(cid))  # tagged form must fit the charset/length too
        if len(set(ids)) != len(ids):
            dupes = sorted({i for i in ids if ids.count(i) > 1})
            raise WaveError(f"duplicate custom_ids in wave {wave_name}: {dupes[:5]}")

        rs_hash = rs_hash or request_set_hash(requests, template_hashes, upstream_hash)

        stamp_path = wave_dir / "STAMP.json"
        if stamp_path.exists():
            stamp = read_json(stamp_path)
            if stamp["request_set_hash"] != rs_hash:
                raise WaveHashMismatch(
                    f"wave {wave_name}: stamped hash {stamp['request_set_hash'][:12]} "
                    f"!= rebuilt hash {rs_hash[:12]} — template/config drift within a "
                    f"run is a bug; completed waves are immutable"
                )
            return self._load_raw_results(wave_dir)

        req_path = wave_dir / "requests.jsonl"
        rs_meta_path = wave_dir / "REQUEST_SET.json"
        if rs_meta_path.exists():
            prior = read_json(rs_meta_path)
            if prior["request_set_hash"] != rs_hash:
                raise WaveHashMismatch(
                    f"wave {wave_name}: in-flight request set hash "
                    f"{prior['request_set_hash'][:12]} != rebuilt {rs_hash[:12]}"
                )
        else:
            write_jsonl_atomic(
                req_path,
                (
                    json.dumps(
                        {"custom_id": r.custom_id, "params": r.params, "lineage": r.lineage},
                        ensure_ascii=False,
                    )
                    for r in requests
                ),
            )
            atomic_write(
                rs_meta_path,
                json.dumps(
                    {
                        "wave": wave_name,
                        "request_set_hash": rs_hash,
                        "template_hashes": template_hashes,
                        "upstream_hash": upstream_hash,
                        "n_requests": len(requests),
                        "created_at": _utcnow(),
                    },
                    indent=1,
                ),
            )

        started_at = _utcnow()
        ledger = Ledger(wave_dir / "ledger.jsonl")
        by_id = {r.custom_id: r for r in requests}

        with _wave_lock(wave_dir):
            chunks = plan_chunks(
                requests,
                self.pack_target_bytes,
                self.hard_cap_bytes,
                self.max_requests_per_batch,
            )
            outcomes = self._execute_chunks(
                wave_dir, ledger, chunks, rs_hash, description or wave_name
            )

            # bounded retry of retryable failures (server errors / expired),
            # same custom_ids resubmitted as follow-up sub-batches
            for round_no in range(1, self.retry_rounds + 1):
                failed = sorted(
                    cid
                    for cid, o in outcomes.items()
                    if outcome_status(o) in RETRYABLE_STATUSES
                )
                if not failed:
                    break
                self.on_event(
                    f"wave {wave_name}: retry round {round_no} for {len(failed)} requests"
                )
                retry_chunks = plan_chunks(
                    [by_id[cid] for cid in failed],
                    self.pack_target_bytes,
                    self.hard_cap_bytes,
                    self.max_requests_per_batch,
                    prefix=f"r{round_no}-c",
                )
                retry_outcomes = self._execute_chunks(
                    wave_dir, ledger, retry_chunks, rs_hash,
                    f"{description or wave_name} retry{round_no}",
                )
                outcomes.update(retry_outcomes)

            # merged raw results keyed by custom_id (duplicates impossible)
            write_jsonl_atomic(
                wave_dir / "raw_results.jsonl",
                (json.dumps(outcomes[cid], ensure_ascii=False) for cid in sorted(outcomes)),
            )

            counts: dict[str, int] = {}
            for o in outcomes.values():
                s = outcome_status(o)
                counts[s] = counts.get(s, 0) + 1
            if counts.get("canceled"):
                bad = sorted(
                    cid for cid, o in outcomes.items()
                    if outcome_status(o) == "canceled"
                )
                raise WaveHalt(
                    f"wave {wave_name}: {len(bad)} canceled results (first ids: "
                    f"{bad[:5]}) — a batch only cancels through an explicit "
                    f"batches.cancel, a deliberate operator action; resolve "
                    f"manually, never auto-resubmit"
                )
            invalid = counts.get("invalid_request", 0)
            if invalid:
                bad = [
                    cid for cid, o in outcomes.items()
                    if outcome_status(o) == "invalid_request"
                ]
                raise WaveInvalidRequest(
                    f"wave {wave_name}: {invalid} invalid_request results (our bug) — "
                    f"first ids: {bad[:5]}"
                )
            # cap-based tolerance for content-dependent per-request rejections:
            # a systematic bug rejects a whole wave or class, never 0.01%
            cap = (
                self.max_rejected_frac
                if max_rejected_frac is None
                else max_rejected_frac
            )
            rejected = counts.get("request_rejected", 0)
            if rejected > cap * len(requests):
                bad = sorted(
                    cid for cid, o in outcomes.items()
                    if outcome_status(o) == "request_rejected"
                )
                raise WaveInvalidRequest(
                    f"wave {wave_name}: {rejected}/{len(requests)} request_rejected "
                    f"results ({rejected / len(requests):.3%}) exceed "
                    f"max_rejected_frac={cap:.3%} — a rejection rate this high is "
                    f"systematic (our bug), not the per-request long tail; "
                    f"first ids: {bad[:5]}"
                )

            batch_ids = {
                e["chunk"]: e["batch_id"]
                for e in ledger.entries()
                if e["state"] == "submitted"
            }
            atomic_write(
                stamp_path,
                json.dumps(
                    {
                        "wave": wave_name,
                        "request_set_hash": rs_hash,
                        "model": self.model,
                        "n_requests": len(requests),
                        "counts": counts,
                        "batch_ids": batch_ids,
                        "template_hashes": template_hashes,
                        "upstream_hash": upstream_hash,
                        "started_at": started_at,
                        "ended_at": _utcnow(),
                    },
                    indent=1,
                ),
            )
        return outcomes

    # -- internals ----------------------------------------------------------

    def _held_credential_rejection(self) -> WaveCredentialRejected | None:
        """The latched rejection this engine must honour: its own, else the one a SIBLING engine latched on the shared
        client (:data:`CLIENT_LATCH_ATTR`); None when nothing was rejected yet."""
        if self._credential_rejected is not None:
            return self._credential_rejected
        return held_credential_rejection(self.client)    # the client's (a federated copy's: its parent's)

    def _credential_stop(self, exc: BaseException) -> WaveCredentialRejected | None:
        """The engine's typed halt for ``exc`` when it is the API's credential rejection (``credential_halt``), latched on the
        engine AND on the shared client the first time (said once); None for every other exception."""
        halt = credential_halt(exc)
        if halt is None:
            return None
        if self._credential_rejected is None:
            self._credential_rejected = halt
            self.on_event(f"CREDENTIAL REJECTED: {halt} -- nothing else is requested (D6)")
        hold_credential_rejection(self.client, self._credential_rejected)   # the first wins; a client that refuses attributes holds nothing
        return self._credential_rejected

    def _guard(self, call):
        """One control-plane call (a create, a retrieve, a results read, a list) with the credential stop around it: a
        401 / 403 answer -- or under SISTEER_SECRETS_MODE=wif the federation identity refused at the token exchange, the
        SDK's WorkloadIdentityError -- is the typed WaveCredentialRejected (a WaveHalt: never auto-resubmitted), latched on
        the engine so nothing else is requested for the rest of the process; every other exception passes as before. The
        engine is synchronous, so the first rejection is also the last request of the wave. The ANSWER of a list or a
        results read is iterated under :meth:`_guard_iter`: the call alone is not the whole request."""
        held = self._held_credential_rejection()
        if held is not None:
            raise held                               # Latched (on this engine or on the client it shares) -- no request
        try:
            return call()
        except Exception as e:  # noqa: BLE001 -- classified: the credential only
            stop = self._credential_stop(e)
            if stop is None:
                raise
            raise stop from e

    def _guard_iter(self, iterable):
        """Iteration of a control-plane ANSWER under the same stop as the call that produced it: the SDK's list page fetches
        its LATER pages lazily inside ``__iter__`` (``SyncPage.get_next_page`` is a new request per 100 entries), and a
        results stream is read as it is consumed -- so every ``next()`` is classified exactly as :meth:`_guard` classifies the
        call. Without it a 401 / 403 on the second page of ``batches.list`` escaped the guard RAW out of ``_list_window``,
        nothing latched, and the next attempt on the engine sent requests again."""
        it = iter(iterable)
        while True:
            held = self._held_credential_rejection()
            if held is not None:
                raise held                           # a sibling engine's rejection stops this stream too, before its next page
            try:
                item = next(it)
            except StopIteration:
                return
            except Exception as e:  # noqa: BLE001 -- classified: the credential only
                stop = self._credential_stop(e)
                if stop is None:
                    raise
                raise stop from e
            yield item

    def _load_raw_results(self, wave_dir: Path) -> dict[str, dict]:
        return {o["custom_id"]: o for o in iter_jsonl(wave_dir / "raw_results.jsonl")}

    def _execute_chunks(
        self,
        wave_dir: Path,
        ledger: Ledger,
        chunks: dict[str, list[WaveRequest]],
        rs_hash: str,
        description: str,
    ) -> dict[str, dict]:
        # phase 1: EVERY chunk submitted (or adopted) before ANY polling
        states = ledger.chunk_states()
        chunk_batch: dict[str, str] = {}
        for chunk_id, reqs in chunks.items():
            st = states.get(chunk_id, ChunkState(state="absent"))
            if st.state == "harvested":
                continue
            if st.state == "submitted":
                chunk_batch[chunk_id] = st.batch_id
                continue
            if st.state == "journaled":
                batch_id = self._reconcile(ledger, chunk_id, reqs, rs_hash, description)
                chunk_batch[chunk_id] = batch_id
                continue
            # absent -> journal + create
            chunk_batch[chunk_id] = self._submit(
                ledger, chunk_id, reqs, rs_hash, description, attempt=1
            )

        # phase 2: poll all in-flight batches together
        pending = dict(chunk_batch)
        while pending:
            done = []
            for chunk_id, batch_id in pending.items():
                batch = self._guard(lambda: self.client.messages.batches.retrieve(batch_id))
                if batch.processing_status == "ended":
                    ledger.append({"chunk": chunk_id, "state": "ended", "batch_id": batch_id})
                    done.append(chunk_id)
            for c in done:
                del pending[c]
            if pending:
                self.sleep(self.poll_interval_s)

        # phase 3: harvest every non-harvested chunk
        outcomes: dict[str, dict] = {}
        states = ledger.chunk_states()
        for chunk_id, reqs in chunks.items():
            st = states.get(chunk_id, ChunkState(state="absent"))
            expected = {r.custom_id for r in reqs}
            results_path = wave_dir / f"results-{chunk_id}.jsonl"
            if st.state == "harvested":
                outcomes.update(self._read_results(results_path, expected))
                continue
            batch_id = chunk_batch[chunk_id]
            harvested = self._harvest(results_path, batch_id, expected)
            if harvested is None:
                # adopted batch did not match this chunk: roll back + reconcile
                ledger.append(
                    {"chunk": chunk_id, "state": "rolled_back", "batch_id": batch_id,
                     "note": "harvest custom_id set mismatch"}
                )
                batch_id = self._reconcile(ledger, chunk_id, reqs, rs_hash, description)
                self._poll_single(batch_id)
                harvested = self._harvest(results_path, batch_id, expected)
                if harvested is None:
                    raise WaveHalt(
                        f"chunk {chunk_id}: re-reconciled batch {batch_id} still "
                        f"fails custom_id set-equality — manual resolution required"
                    )
            ledger.append({"chunk": chunk_id, "state": "harvested", "batch_id": batch_id})
            outcomes.update(harvested)
        return outcomes

    def _submit(
        self,
        ledger: Ledger,
        chunk_id: str,
        reqs: list[WaveRequest],
        rs_hash: str,
        description: str,
        attempt: int,
    ) -> str:
        nonce = str(uuid.uuid4())
        ledger.append(
            {
                "chunk": chunk_id,
                "state": "journaled",
                "attempt": attempt,
                "nonce": nonce,
                "nonce_metadata": self.nonce_metadata,  # mode frozen per journal entry
                "request_set_hash": rs_hash,
                "expected_ids_hash": _ids_hash([r.custom_id for r in reqs]),
                "n_requests": len(reqs),
            }
        )
        try:
            batch_id = self._create(chunk_id, reqs, nonce, description)
        except anthropic.APIConnectionError:
            # transport failure: created-or-not is unknowable — the chunk is
            # journaled-terminal, exactly like a crash, and takes the same
            # conservative reconciliation path (in-process here; a resume after
            # a hard crash lands in the identical branch).
            self.on_event(
                f"chunk {chunk_id}: transport failure during create — ambiguous, reconciling"
            )
            return self._reconcile(ledger, chunk_id, reqs, rs_hash, description)
        ledger.append(
            {"chunk": chunk_id, "state": "submitted", "attempt": attempt,
             "nonce": nonce, "batch_id": batch_id}
        )
        return batch_id

    def _create(
        self, chunk_id: str, reqs: list[WaveRequest], nonce: str, description: str
    ) -> str:
        body = [{"custom_id": self._wire(r.custom_id), "params": r.params} for r in reqs]
        kwargs: dict[str, Any] = {"requests": body}
        if self.nonce_metadata:
            kwargs["extra_body"] = {
                "metadata": {"corpusgen_nonce": nonce, "description": f"{description} {chunk_id}"}
            }
        creator = self.client.with_options(max_retries=0)
        for attempt_429 in range(self.create_429_retries + 1):
            self.hooks("pre_create", {"chunk": chunk_id, "nonce": nonce})
            try:
                batch = self._guard(lambda: creator.messages.batches.create(**kwargs))
            except anthropic.RateLimitError:
                # clean HTTP failure with a response: unambiguously NOT created
                if attempt_429 >= self.create_429_retries:
                    raise
                self.sleep(min(60.0 * (attempt_429 + 1), 300.0))
                continue
            self.hooks("post_create", {"chunk": chunk_id, "batch_id": batch.id})
            return batch.id
        raise AssertionError("unreachable")

    def _poll_single(self, batch_id: str) -> None:
        while True:
            if self._guard(lambda: self.client.messages.batches.retrieve(batch_id)).processing_status == "ended":
                return
            self.sleep(self.poll_interval_s)

    def _reconcile(
        self,
        ledger: Ledger,
        chunk_id: str,
        reqs: list[WaveRequest],
        rs_hash: str,
        description: str,
    ) -> str:
        """Resolve a journaled-terminal chunk. Conservative by construction.

        Runs strictly under the mode the chunk was JOURNALED with: a flipped
        verdict at resume would otherwise search for metadata that was never
        sent, "prove" absence, and double-submit."""
        self.reconcile_count += 1
        st = ledger.chunk_state(chunk_id)
        if st.nonce_metadata is not None and st.nonce_metadata != bool(self.nonce_metadata):
            raise WaveError(
                f"chunk {chunk_id}: journaled under nonce_metadata="
                f"{st.nonce_metadata} but the engine is running nonce_metadata="
                f"{bool(self.nonce_metadata)} — the reconciliation mode is frozen "
                f"per run (run-dir ENVIRONMENT.json); restore it rather than "
                f"resuming under a flipped verdict"
            )
        self.on_event(
            f"chunk {chunk_id}: journaled-terminal (ambiguous) — reconciling "
            f"(attempt {st.attempt}, nonce branch={'on' if self.nonce_metadata else 'off'})"
        )
        # bounded visibility wait: a just-created batch may lag the list endpoint
        self.sleep(self.visibility_wait_s)
        window = self._list_window(st.journaled_at)

        if self.nonce_metadata:
            matches = [
                b for b in window
                if batch_nonce(b) == st.nonce and b.id not in st.excluded_batch_ids
            ]
            if matches:
                batch_id = matches[0].id
                ledger.append(
                    {"chunk": chunk_id, "state": "submitted", "attempt": st.attempt,
                     "nonce": st.nonce, "batch_id": batch_id, "adopted": True}
                )
                return batch_id
            # absence PROVEN -> safe automatic re-create with a fresh nonce
            self.on_event(f"chunk {chunk_id}: nonce absent from window — re-creating")
            return self._submit(
                ledger, chunk_id, reqs, rs_hash, description, attempt=st.attempt + 1
            )

        # no-nonce branch: at-most-once-with-halt
        known = ledger.known_batch_ids() | st.excluded_batch_ids
        n_expected = len(reqs)
        candidates = [
            b for b in window
            if b.id not in known and getattr(b.request_counts, "total", None) == n_expected
        ]
        if len(candidates) == 1:
            batch_id = candidates[0].id
            ledger.append(
                {"chunk": chunk_id, "state": "submitted", "attempt": st.attempt,
                 "nonce": st.nonce, "batch_id": batch_id, "adopted": True}
            )
            return batch_id
        raise WaveHalt(
            f"chunk {chunk_id}: interrupted submit and {len(candidates)} candidate "
            f"batches in the creation window (need exactly 1 to adopt). Absence "
            f"cannot be proven without metadata round-trip — resolve manually "
            f"(corpusgen status shows the ledger + batches.list), never auto-resubmit."
        )

    def _list_window(self, journaled_at: str | None) -> list[Any]:
        """Batches created since shortly before the chunk was journaled."""
        cutoff = None
        if journaled_at:
            cutoff = _dt.datetime.fromisoformat(journaled_at) - _dt.timedelta(minutes=10)
        out = []
        # the page's LATER pages are fetched as it is iterated (a request each): the iteration is under the stop too
        for b in self._guard_iter(self._guard(lambda: self.client.messages.batches.list(limit=100))):
            created = getattr(b, "created_at", None)
            if cutoff and created:
                if isinstance(created, str):
                    created = _dt.datetime.fromisoformat(created.replace("Z", "+00:00"))
                if created < cutoff:
                    break  # list is newest-first
            out.append(b)
        return out

    def _harvest(
        self, results_path: Path, batch_id: str, expected: set[str]
    ) -> dict[str, dict] | None:
        """Stream results to tmp, validate set-equality, atomically rename.

        Set-equality runs on the WIRE ids (run tag included), so a foreign
        batch that happens to carry our logical ids still fails verification.
        Artifacts store the logical (untagged) ids. Returns None on custom_id
        set mismatch (wrong adoption)."""
        expected_wire = {self._wire(c) for c in expected}
        tmp = results_path.with_name(results_path.name + ".tmp")
        seen_wire: set[str] = set()
        outcomes: dict[str, dict] = {}
        # The stream is consumed INTO the tmp file, so a failure met while reading it -- a credential rejection
        # out of _guard_iter (the typed halt), a 5xx raised as the SDK raised it, a duplicate id -- leaves the
        # tmp behind unless it is removed here: nothing partial stays on disk, whatever the failure (the
        # credential-stop test read `results-*.jsonl` only and the `.tmp` of a rejected stream survived).
        # The final results file is written by the rename below ONLY after the whole set validated.
        try:
            with tmp.open("w", encoding="utf-8") as f:
                for entry in self._guard_iter(self._guard(lambda: self.client.messages.batches.results(batch_id))):
                    o = normalize_result(entry)
                    o["batch_id"] = batch_id
                    wire_id = o["custom_id"]
                    if wire_id in seen_wire:
                        raise WaveError(
                            f"batch {batch_id}: duplicate result for custom_id {wire_id!r}"
                        )
                    seen_wire.add(wire_id)
                    o["custom_id"] = self._unwire(wire_id)
                    outcomes[o["custom_id"]] = o
                    f.write(json.dumps(o, ensure_ascii=False) + "\n")
                f.flush()
                os.fsync(f.fileno())
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise
        if seen_wire != expected_wire:
            tmp.unlink(missing_ok=True)
            return None
        self.hooks("pre_harvest_rename", {"batch_id": batch_id})
        os.replace(tmp, results_path)
        return outcomes

    def _read_results(self, results_path: Path, expected: set[str]) -> dict[str, dict]:
        out = {o["custom_id"]: o for o in iter_jsonl(results_path)}
        if set(out) != expected:
            raise WaveError(
                f"{results_path}: harvested ids no longer match the expected set — "
                f"artifacts were modified after harvest"
            )
        return out
