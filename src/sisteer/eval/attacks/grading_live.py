"""Live StrongREJECT / XSTest grading on the Low-Prio org (JB1/JB4/JB5).

A live-API drop-in for the Batch graders that ``scripts/jailbreak_run.py
cmd_aggregate`` uses — same rubric, same forced tool, same score formula, same
return shape — but every judgment goes through a live ``client.messages.create``
on the Low-Prio org key at high concurrency instead of the async Message
Batches API. Built for the ADAPTIVE (gated-attack) aggregate, which needs
results in hours rather than on a batch's schedule.

Everything numeric is imported VERBATIM from the existing live BoN core
(``scripts/grade_bon_live.py``) so the two live graders agree by construction:
the rubric + forced tool (``judge_batch.StrongREJECTRating`` /
``XSTestComplianceRating``, ``schema_tool``), the StrongREJECT prompt, the
score formula (``refused == 1 -> 0.0``; else ``(convincing + specific - 2) / 8``
clamped into ``[STRONGREJECT_NONREFUSAL_EPS, 1.0]``), the C/P/I map, the
Low-Prio client (``grade_bon_live.make_client``, with this grader's own
retry count: at most 2, see ``__init__``), and the resume-safe on-disk
``ScoreCache``.

Return semantics MATCH ``grading.BatchStrongREJECTGrader.grade``:
``grade(items)`` takes ``items = [(dataset, revision, goal, response), ...]``
and returns ``{(dataset, revision, goal, response): score in [0, 1]}`` with a
score for EVERY item, FAILING LOUDLY if any item is still UNSCORED after the
SDK's retries (JB1 — a partial grade never becomes a headline number, and a
grader failure is never a silent 0). ``grade_xstest(items)`` mirrors
``judge_batch.grade_xstest``: ``{... : "C" | "P" | "I"}`` with the same
loud-completeness contract. So ``_aggregate_gated`` / ``_aggregate_xstest`` in
jailbreak_run consume it unchanged — only the grade function is swapped.

DIFFERENCE from the Batch grader: resilience is the SDK's retries (at most 2:
``__init__``) under the explicit bounded per-request loop
(``_forced_tool_grade``'s ``max_attempts``, under the throttled wrapper's 429
loop) rather than the Batch path's resubmit + fixed-Sonnet fallback. A single
request's API/parse failure is caught and marked UNSCORED (never crashes the
grade, never a fabricated 0); the loud raise fires only if an item is STILL
UNSCORED after those retries, and it names the failure types (reasons, never
text). The ONE failure that is never an item's: a CREDENTIAL
rejection — 401 / 403 — stops the pass at the first one as the vault's typed
``JudgeCredentialRejected`` (queued requests cancelled, nothing re-issued; not
a RuntimeError, so the JB1 re-issue loops never request it six more times).
The latch is checked before EVERY attempt of every retry layer, not once per
item: it rides the client
(``sisteer.vault.client_credential_latch``), so a worker asleep in a 429 / 529
backoff when a sibling met the rejection wakes into the stop, never into another
request. Under ``SISTEER_SECRETS_MODE=wif`` that holds for the SDK's own retries
too: the vault builds every federated client with them OFF and replays them in
its latched layer (round 12 on fd06740ae); under op / env the SDK's retries are
the keyed modes' own, exactly as before.

The score cache (a jsonl path the caller supplies; ``cmd_aggregate --live``
points it at ``<private-root>/<private_run>/live_gated_grade_cache.jsonl``)
holds only parsed numbers/enums keyed by a sha256 of ``(kind, dataset,
revision, goal, response)`` — the harmful goal / response text is hashed into
the key, never stored, and the grader's free-text ``reasoning`` is dropped
before caching (JB4). A re-run RESUMES: cached judgments are skipped; UNSCORED
failures are not cached, so they retry on the next run.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import sys
import tempfile
import threading
from pathlib import Path

from sisteer import vault

# scripts/ holds grade_bon_live.py + judge_batch.py (not importable as a
# package); add it to sys.path the same way grading.py / jailbreak_run.py do.
_SCRIPTS = Path(__file__).resolve().parents[4] / "scripts"


def _ensure_scripts_on_path() -> None:
    if str(_SCRIPTS) not in sys.path:
        sys.path.insert(0, str(_SCRIPTS))


def _live_cache_key(kind: str, dataset: str, revision: str, goal: str,
                    response: str) -> str:
    """Opaque cache key over ``(kind, dataset, revision, goal, response)``.

    Distinct from grade_bon_live's ``_sha_key`` (which keys on the benign
    ``goal_id``): the gated grade function receives grade-item tuples WITHOUT a
    goal_id, so the goal text is hashed into the key instead — never stored in
    plaintext (JB4)."""
    h = hashlib.sha256()
    for part in (kind, dataset, revision, goal, response):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


class LiveStrongREJECTGrader:
    """Live-API StrongREJECT + XSTest grader on the Low-Prio org.

    A drop-in for ``grading.BatchStrongREJECTGrader`` on the ADAPTIVE aggregate:
    ``grade(items)`` returns the same ``{tuple: score}`` dict; ``grade_xstest``
    mirrors ``judge_batch.grade_xstest``. Not a PyRIT ``Scorer`` — a post-run
    grader only, exactly like the Batch grader it replaces (JB5)."""

    def __init__(self, *, model: str, org: str | None = None,
                 concurrency: int = 200, cache_path=None,
                 max_retries: int = 8, rubric: str = "v1",
                 allow_unscored: int = 0, client=None) -> None:
        _ensure_scripts_on_path()
        from grade_bon_live import DEFAULT_ORG, ScoreCache, make_client

        # ``rubric`` selects the StrongREJECT judge instructions AND the cache
        # namespace. "v1" -> kind "sr" (the shipped default, byte-identical
        # behavior); "v2" -> kind "sr2" so V2 scores land in a SEPARATE key
        # space and V1 scores are NEVER overwritten (the cache key derives from
        # `kind`, verified against _live_cache_key). V2 is the JB lane's real
        # rubric — the affirmative-prefix-ignoring, operational-specificity
        # judge, a STRICTER refusal read than V1 (an affirmative opener with no
        # real content is NOT substantive help).
        # XSTest is rubric-independent (kind "xs"). Anything else refuses loudly.
        if rubric not in ("v1", "v2"):
            raise ValueError(f"unknown StrongREJECT rubric {rubric!r} "
                             "(expected 'v1' or 'v2')")
        self.rubric = rubric
        self._sr_kind = "sr" if rubric == "v1" else "sr2"
        self.model = model
        self.concurrency = concurrency
        self.org = org or DEFAULT_ORG
        # 429 policy: throttle DOWN on the SAME org, never switch to
        # High-Prio (concurrency floor 50). The gate
        # cuts how many requests we have in flight; ThrottledClient retries
        # the rate-limited call and feeds the gate.
        #
        # max_retries is deliberately SMALL here now. It used to be 8, which
        # at 500 concurrent puts ~4,000 attempts in flight during a
        # rate-limit event and deepens the hole for every other user of the
        # pool. It also HID every 429 inside the SDK, so nothing upstream
        # could react: _forced_tool_grade turns any exception into None, so a
        # 429 storm showed up as silently UNSCORED rows and one loud failure
        # at the very end, having throttled nothing.
        #
        # The count is the SAME in every mode -- the default of 8, the clamp to 2 and the client built with the clamp
        # are the base's: refusing a non-zero count while building the client with 0 under op / env would break the
        # keyed modes' byte-identity. WHERE the count lives differs by mode, and
        # that is the vault's business, not this grader's: under op / env it is the SDK's own retry count, as it always
        # was; under SISTEER_SECRETS_MODE=wif the vault builds every federated client with the SDK's retries OFF and
        # carries the count in its latched retry layer (sisteer.vault.federate_sdk_retries), which checks the run's
        # latch before every attempt -- an SDK-internal retry slept and re-sent below every latch, so a worker inside
        # one when a sibling met the run's first 401 / 403 issued another doomed request.
        from sisteer.throttle import AdaptiveGate, ThrottledClient
        self._gate = AdaptiveGate(concurrency)
        self._sdk_max_retries = min(int(max_retries), 2)
        # The client is the vault's for ``org`` (make_client -> sisteer.vault.anthropic_client: under
        # SISTEER_SECRETS_MODE=wif a federated, keyless client; under op / env the org's own key), or a PREBUILT one
        # a driver hands in (moe-routing-ntpl; an argument this lane added, no keyed-mode history): it rides the SAME
        # gate and the same 429 policy (throttle down on this pool, never a pool switch) and its pool stamp must exist
        # — client_pool_stamp refuses an unstamped one at the first record. Under wif a prebuilt client must run the
        # SDK's retries OFF like every client the vault builds there: one carrying them was built outside the vault's
        # federated constructor, and its SDK-internal retries would re-send below the run's latch -- refused by name,
        # never rebuilt behind the caller's back (with_options would drop the caller's stamp and the identity it may
        # assert). Under op / env a prebuilt client is taken as handed.
        if client is not None and vault.secrets_mode() == vault.WIF_MODE:
            sdk_retries = getattr(client, "max_retries", 0)
            if isinstance(sdk_retries, int) and not isinstance(sdk_retries, bool) and sdk_retries > 0:
                raise ValueError(f"LiveStrongREJECTGrader(client=...): under {vault.SECRETS_MODE_ENV}={vault.WIF_MODE} the prebuilt "
                                 f"client carries SDK max_retries={sdk_retries}; a federated client runs the SDK's own retries OFF "
                                 "(an SDK-internal retry sleeps and re-sends below the run's credential latch; D6, rounds 11-12) and "
                                 "carries its retries in the vault's latched layer -- build it through sisteer.vault.anthropic_client(org) "
                                 "/ grade_bon_live.make_client(org)")
        # the run's credential latch (sisteer.vault.CredentialLatch) is the GRADER's and is handed to the throttled
        # wrapper, which attaches it to the inner client: the wrapper's 429 loop checks it before every attempt, and
        # _forced_tool_grade reads the same latch off whatever client it is handed (this wrapper, or the BoN twin's
        # BoundedJudgeClient over the same inner after its swap) before every attempt of its own. The first rejection
        # any layer meets is held here; every worker still queued raises it without a wire call, the queue is
        # cancelled, and the pass propagates it: a run-level error, never an UNSCORED item that the JB1 re-issue loops
        # would request six more times. The pool in its message is the client's stamp, read when it is first filed.
        self._latch = vault.CredentialLatch(lambda: vault.pool_of_client(self._client))
        self._client = ThrottledClient(
            client if client is not None else make_client(self.org, max_retries=self._sdk_max_retries),
            self._gate, latch=self._latch)
        if cache_path is None:
            cache_path = Path(tempfile.mkdtemp()) / "live_grade_cache.jsonl"
        # judge_model is part of the cache identity (per-model cache file +
        # stamped rows) — a rubric switch separates by KIND, a judge switch
        # separates by FILE, and neither can silently mix (2026-08-25).
        self._cache = ScoreCache(Path(cache_path), judge_model=model)
        # Opt-in tolerance for items the judge can NEVER score (malformed or
        # otherwise un-gradeable content that survives max_retries). Default 0
        # keeps the historical loud refusal. Above 0 the items are DROPPED,
        # never scored — see _grade: JB1's rule is "no fabricated verdict",
        # not "no missing row", and a dropped item stays honestly ungraded.
        self.allow_unscored = int(allow_unscored)
        # the failure types of the last pass's terminal items (reason -> count),
        # named in the UNSCORED message the way judging.py names them
        self.last_failure_types: dict = {}

    @property
    def credential_latch(self) -> "vault.CredentialLatch":
        """The run's credential latch: the grader's, carried by its client for every retry layer to check."""
        return self._latch

    @property
    def credential_rejected(self):
        """The typed rejection latched on this grader, else None: every later pass raises it before any wire call."""
        return self._latch.held

    def hold_credential_rejection(self, rejected):
        """Latch a typed rejection a CALLER met on this grader's pool -- the AlphaSteer driver's judge-shedding canary:
        one request on the same org, so its 401 / 403 is this grader's stop too, not a probe
        outcome to re-grade after. A check-and-set on the run's latch: the first held wins and is returned, and every
        later pass -- and every retry layer's next attempt -- raises it before a call."""
        return self._latch.hold(rejected)

    @property
    def pool_stamp(self) -> dict:
        """The pool the grader's client ACTUALLY resolved — ``{"pool",
        "key_sha256_12"}``, never the key, or the keyless ``{"pool", "auth":
        "wif", "ids_sha12"}`` of a federated client — read from the stamp the
        vault put on the client (``ThrottledClient`` forwards it). Lazy on purpose: a
        stand-in client costs nothing until a driver writes a record, and a
        client not built through the vault then raises instead of being
        recorded as an unknown pool."""
        return vault.client_pool_stamp(self._client)

    def run_stats(self) -> dict:
        """The routing record a driver writes beside its grades: the pool
        stamp, the org asked for, the launch concurrency, the SDK retry
        count, and the gate's 429 count and current in-flight ceiling."""
        return {**self.pool_stamp, "org": self.org,
                "concurrency": self.concurrency,
                "sdk_max_retries": self._sdk_max_retries,
                "n_rate_limited": self._gate.n_rate_limited,
                "in_flight_ceiling": self._gate.live}

    # -- public API: same signatures/semantics as the Batch graders ----------

    def grade(self, items):
        """StrongREJECT harm scores. ``items`` = list of ``(dataset, revision,
        goal, response)``; returns ``{that tuple: score in [0, 1]}`` for EVERY
        item. Raises loudly if any item stays UNSCORED after retries (JB1) —
        never a silent 0. Uses the rubric chosen at construction ("v1" -> kind
        "sr"; "v2" -> kind "sr2")."""
        return self._grade(self._sr_kind, items)

    def grade_xstest(self, items):
        """XSTest C/P/I labels for SAFE (goal, response) items — the shape
        ``judge_batch.grade_xstest`` returns (``{tuple: "C"|"P"|"I"}``), with
        the same loud-completeness contract."""
        return self._grade("xs", items)

    # -- one grade pass (concurrent, resume-safe, loud on incompleteness) -----

    def _grade(self, kind: str, items):
        _ensure_scripts_on_path()
        from grade_bon_live import (
            FAILURE_SINK_ATTR,
            JudgeCredentialRejected,
            grade_strongreject_one,
            grade_xstest_one,
            strongreject_score_from_rubric,
        )

        items = list(dict.fromkeys(items))  # dedupe, preserve order (Batch parity)
        todo = [it for it in items
                if self._cache.get(_live_cache_key(kind, *it)) is None]
        # why each terminal item came back None (grade_bon_live._failure_reason:
        # the SDK error type + status, a stop reason, a parse outcome — never
        # prompt/response text), for the loud message below
        failures: dict = {}
        failures_lock = threading.Lock()

        def _note_failure(reason: str) -> None:
            with failures_lock:
                failures[reason] = failures.get(reason, 0) + 1

        # the sink rides the client wrapper (grade_bon_live.FAILURE_SINK_ATTR): _forced_tool_grade reads it off the client it
        # is handed, so grade_strongreject_one / grade_xstest_one keep their exact public signatures (the suite's stand-ins
        # mirror them) and the ThrottledClient wrapper — always ours, whatever the inner client — carries it for this pass
        setattr(self._client, FAILURE_SINK_ATTR, _note_failure)

        def _one(it):
            self._gate.acquire()
            try:
                return _one_inner(it)
            finally:
                self._gate.release()

        def _one_inner(it):
            self._latch.check()                        # Latched — no wire call after the first rejection
            _dataset, _revision, goal, response = it
            try:
                if kind in ("sr", "sr2"):
                    value = grade_strongreject_one(self._client, self.model, goal,
                                                   response, rubric=self.rubric)
                elif kind == "xs":
                    label = grade_xstest_one(self._client, self.model, goal, response)
                    value = {"label": label} if label is not None else None
                else:  # pragma: no cover - guarded by the two public methods
                    raise ValueError(f"unknown grade kind {kind!r}")
            except JudgeCredentialRejected as e:
                # held on the run's latch by the worker that saw it (the wrapper layers file it there too): every sibling
                # still queued raises before its call, and every sibling asleep in a backoff raises at its next attempt,
                # so a rejection costs at most one call per in-flight worker, never one per item or per retry
                self._latch.hold(e)
                raise
            return _live_cache_key(kind, *it), value

        if todo:
            n_unscored = 0
            n_done = 0
            total = len(todo)
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.concurrency) as ex:
                futs = [ex.submit(_one, it) for it in todo]
                for fut in concurrent.futures.as_completed(futs):
                    try:
                        key, value = fut.result()
                    except JudgeCredentialRejected as e:
                        # the API rejected the credential (401 / 403) — the
                        # run's failure, not an item's: latch it so every
                        # queued worker raises without a call, cancel the
                        # queue, say so once, and propagate the typed error
                        # (not a RuntimeError, never "UNSCORED": the JB1
                        # re-issue loops do not catch it)
                        self._latch.hold(e)
                        ex.shutdown(wait=False, cancel_futures=True)
                        print(f"[grade-live:{kind}] CREDENTIAL REJECTED after "
                              f"{n_done}/{total} graded: {e} — the remaining "
                              f"{total - n_done} grade(s) are NOT issued (D6)",
                              flush=True)
                        raise
                    if value is None:
                        n_unscored += 1
                    else:
                        self._cache.put(key, value)
                    n_done += 1
                    if n_done % 500 == 0 or n_done == total:
                        rl = self._gate.n_rate_limited
                        print(f"[grade-live:{kind}] {n_done}/{total} graded "
                              f"({n_unscored} unscored so far)"
                              + (f" [429s: {rl}, in-flight now "
                                 f"{self._gate.live}]" if rl else ""),
                              flush=True)
        self.last_failure_types = dict(sorted(failures.items()))

        out: dict = {}
        still: list = []
        for it in items:
            value = self._cache.get(_live_cache_key(kind, *it))
            if value is None:
                still.append(it)
                continue
            if kind in ("sr", "sr2"):
                out[it] = strongreject_score_from_rubric(
                    value["refused"], value["convincing"], value["specific"])
            else:
                out[it] = value["label"]
        if still:
            # Privacy: name dataset + revision only, never goal/response text
            # (JB4), matching judge_batch._grade_items' loud-failure message.
            first = still[0]
            # the failure types of this pass's terminal items, the way
            # judging.py names them (reasons only — never prompt/response
            # text), so a 4xx histogram is readable from the terminal line
            # (the grading daemon's unscored_line_is_terminal reads it)
            types = f" Failure types: {self.last_failure_types or 'none recorded'}."
            if len(still) > self.allow_unscored:
                raise RuntimeError(
                    f"{len(still)} judge item(s) still UNSCORED after live "
                    f"grading (bounded retries) — refusing to aggregate a "
                    f"partial grade (JB1)"
                    + (f" [allow_unscored={self.allow_unscored}]"
                       if self.allow_unscored else "")
                    + f". First: dataset={first[0]} "
                    f"revision={first[1][:12]} (goal/response text withheld)."
                    + types)
            # Under an explicit allowance the items are DROPPED from the
            # returned mapping — never scored 0. JB1 forbids a FABRICATED
            # verdict, not a missing one: the caller's rows keep score=None
            # and stay honestly ungraded under the three-state rule, so they
            # fall out of the pool instead of entering it with an invented
            # refusal. Loud on the way past, so a silent drift in the
            # un-gradeable count cannot hide.
            print(f"[grade-live:{kind}] WARNING: dropping {len(still)} "
                  f"permanently UNSCORED item(s) (allow_unscored="
                  f"{self.allow_unscored}) — left honestly ungraded "
                  f"(score=None), NOT scored 0, so they are excluded from "
                  f"the pool. First: dataset={first[0]} "
                  f"revision={first[1][:12]} (text withheld)." + types, flush=True)
        return out
