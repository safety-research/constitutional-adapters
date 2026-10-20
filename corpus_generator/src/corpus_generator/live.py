"""Live wave engine: the SAME ``run_wave`` contract and artifact formats as
the Batch ``WaveEngine``, executed over the live Messages API with asyncio.

EXCEPTION (2026-09-01): two post-review fixup campaigns — the
refusal-censored story lanes regenerated on ``claude-opus-4-8`` and the
reframed advice principle subtrees on ``claude-opus-5`` — run LIVE on the
``lowprio`` credential instead of the ``batch`` one. Batch discipline does not apply to
these two campaigns; everything else still goes through batch.py. Bulk
generation on the ``highprio`` credential remains impossible (client.py guard).

Artifact parity — downstream stages/filters/materializers read these files
and must not care which engine produced them:

- ``requests.jsonl`` / ``REQUEST_SET.json``  frozen request set + hash
- ``live/<custom_id>.json``  per-request cached outcome — the RESUME unit;
  a re-run re-issues only requests with no cached outcome
- ``raw_results.jsonl``  merged normalized outcomes keyed by custom_id
- ``STAMP.json``  completeness stamp, hash-checked exactly like batch waves

Outcome dicts are byte-compatible with ``batch.normalize_result`` output, so
``outcome_status`` classifies refusals / content-filter blocks / rejections
identically. ``batch_id`` carries a ``live-<wave>`` marker into lineage.

Concurrency: a SHRINK-ONLY limiter starts at ``max_concurrency`` (default
200) and HALVES on 429 (200 -> 100 -> 50 ...) with a cool-down so one
saturation burst halves once and only PERSISTING 429s halve again — throttle
down, never switch credential labels. It never grows back within a wave: a
saturated rate limit may be shared with other users of the same key, and re-widening
mid-run just re-creates the event the halving answered.

``max_retries`` is deliberate: the SDK client is built with max_retries=0
(client.build_async_client) and THIS module owns the bounded retry loop —
429 / 5xx / 529 / transport / timeout, exponential backoff with jitter.
Refusals, content-filter blocks, and other 4xx rejections are OUTCOMES, not
retries. The CREDENTIAL is neither: a 401 / 403 on a request, or under
``SISTEER_SECRETS_MODE=wif`` a federation identity the token exchange refuses,
stops the wave at the FIRST such answer (``WaveCredentialRejected``, a
``WaveHalt``: the outstanding pool cancelled, nothing stamped, no cache for the
rejected requests, nothing else issued by this engine for the rest of the
process).
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from pathlib import Path
from typing import Any, Callable

import anthropic
import httpx2

from .batch import (
    RETRYABLE_STATUSES,
    WaveCredentialRejected,
    credential_halt,
    WaveHalt,
    WaveHashMismatch,
    WaveInvalidRequest,
    WaveError,
    WaveRequest,
    message_text,
    outcome_status,
    request_set_hash,
    _utcnow,
)
from .io import atomic_write, iter_jsonl, read_json, write_jsonl_atomic
from .schema import validate_custom_id

# statuses live.py retries itself (bounded); everything else is an outcome
_RETRYABLE_HTTP = {408, 409, 500, 502, 503, 504, 529}


class WaveDegraded(WaveHalt):
    """The provider stayed degraded (errored fraction above the cap on every
    resume) past the total budget — the wave is NOT stamped; every outcome so
    far is cached, so a later invocation resumes from where it stopped."""


class ShrinkLimiter:
    """Semaphore that only shrinks. ``halve()`` permanently retires half the
    outstanding permits (down to ``floor``); a cool-down makes a burst of
    429s from one saturation event count as ONE halving, while 429s that
    persist past the cool-down halve again."""

    def __init__(
        self,
        start: int,
        floor: int = 10,
        cooldown_s: float = 20.0,
        on_event: Callable[[str], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.limit = start
        self.floor = floor
        self.cooldown_s = cooldown_s
        self.on_event = on_event or (lambda m: None)
        self._clock = clock
        self._sem = asyncio.Semaphore(start)
        self._last_halve = float("-inf")
        self._holders: list[asyncio.Task] = []

    async def __aenter__(self):
        await self._sem.acquire()

    async def __aexit__(self, *exc):
        self._sem.release()

    async def _hold_forever(self):
        await self._sem.acquire()  # deliberately never released

    def halve(self) -> None:
        now = self._clock()
        if now - self._last_halve < self.cooldown_s:
            return  # same saturation event — already halved for it
        new = max(self.floor, self.limit // 2)
        retired = self.limit - new
        if retired <= 0:
            return
        self._last_halve = now
        self.limit = new
        self.on_event(f"429: halving live concurrency -> {new}")
        for _ in range(retired):
            self._holders.append(asyncio.ensure_future(self._hold_forever()))

    def close(self) -> None:
        for t in self._holders:
            t.cancel()


def _error_fields(exc: anthropic.APIStatusError) -> tuple[str, str]:
    """(error_type, error_message) off an SDK status error, matching what the
    Batch results endpoint reports for the same server-side rejection."""
    body = getattr(exc, "body", None)
    err = body.get("error") if isinstance(body, dict) else None
    if isinstance(err, dict):
        return str(err.get("type", f"http_{exc.status_code}")), str(err.get("message", ""))
    return f"http_{exc.status_code}", str(exc)


class LiveEngine:
    """Drop-in for ``WaveEngine`` where the stage runner is concerned: the
    ``run_wave`` signature, artifacts, halt semantics, and rejection caps are
    identical; only the transport is the live Messages API."""

    def __init__(
        self,
        client_factory: Callable[[], Any],
        *,
        model: str = "claude-opus-5",
        max_concurrency: int = 200,
        min_concurrency: int = 10,
        halve_cooldown_s: float = 20.0,
        max_attempts: int = 6,
        backoff_base_s: float = 5.0,
        backoff_cap_s: float = 120.0,
        retry_rounds: int = 2,
        retry_pause_s: float = 60.0,
        max_rejected_frac: float = 0.001,
        run_tag: str = "",
        sleep: Callable[[float], Any] | None = None,  # async sleeper (tests)
        on_event: Callable[[str], None] | None = None,
        # DEGRADED-PROVIDER guard (added 2026-09-03, with Opus degraded per
        # the status page): a wave whose errored fraction exceeds
        # ``degraded_error_frac`` after the per-request + retry-round budgets is
        # NOT stamped — the engine pauses ``degraded_pause_s`` and resumes from
        # the per-request cache (errored outcomes re-issued), until the wave
        # clears or ``degraded_max_s`` of pausing has accumulated, when it stops
        # with WaveDegraded (nothing lost: the cache is the resume unit). None
        # = the earlier behaviour (stamp whatever the retry rounds left).
        degraded_error_frac: float | None = None,
        degraded_pause_s: float = 600.0,
        degraded_max_s: float = 2700.0,
        sync_sleep: Callable[[float], None] = time.sleep,
    ):
        self.client_factory = client_factory
        self.degraded_error_frac = degraded_error_frac
        self.degraded_pause_s = degraded_pause_s
        self.degraded_max_s = degraded_max_s
        self.sync_sleep = sync_sleep
        self.degraded_pauses = 0  # observable for tests
        self.model = model
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self.halve_cooldown_s = halve_cooldown_s
        self.max_attempts = max_attempts
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.retry_rounds = retry_rounds
        self.retry_pause_s = retry_pause_s
        self.max_rejected_frac = max_rejected_frac
        self.run_tag = run_tag  # kept for interface parity; live ids are local
        self.sleep = sleep or asyncio.sleep
        self.on_event = on_event or (lambda m: None)
        #: The FIRST credential rejection of this engine, latched for the rest of the process
        #: -- one client factory, one credential: every later request fails the same way, so ``_request`` raises it before
        #: every later wire call (the pool's other tasks, the retry rounds, a later wave) and ``_execute`` refuses before it
        #: builds a client. The first rejection filed is the run's; every task raises that one object.
        self._credential_rejected: WaveCredentialRejected | None = None

    # -- public ---------------------------------------------------------------

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
        template_hashes = template_hashes or {}
        wave_dir = Path(wave_dir)
        wave_dir.mkdir(parents=True, exist_ok=True)

        ids = [r.custom_id for r in requests]
        for cid in ids:
            validate_custom_id(cid)
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
            return {o["custom_id"]: o for o in iter_jsonl(wave_dir / "raw_results.jsonl")}

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
                wave_dir / "requests.jsonl",
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
                        "engine": "live",
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
        live_marker = f"live-{wave_name}"
        paused_total = 0.0
        while True:
            outcomes = asyncio.run(self._execute(wave_dir, requests, wave_name, live_marker))
            if self.degraded_error_frac is None:
                break
            n_err = sum(1 for o in outcomes.values() if outcome_status(o) in RETRYABLE_STATUSES)
            frac = n_err / len(requests) if requests else 0.0
            if frac <= self.degraded_error_frac:
                break
            if paused_total + self.degraded_pause_s > self.degraded_max_s:
                raise WaveDegraded(
                    f"wave {wave_name}: {n_err}/{len(requests)} requests still errored "
                    f"({frac:.0%} > {self.degraded_error_frac:.0%}) after {paused_total / 60:.0f} min of "
                    f"pausing — the provider stayed degraded past the {self.degraded_max_s / 60:.0f} min "
                    f"budget; STOPPED without stamping (every outcome is cached; re-run to resume)"
                )
            self.degraded_pauses += 1
            self.on_event(
                f"wave {wave_name}: {n_err}/{len(requests)} errored ({frac:.0%} > "
                f"{self.degraded_error_frac:.0%}) — provider degraded; pausing "
                f"{self.degraded_pause_s / 60:.0f} min, then resuming from the cache"
            )
            self.sync_sleep(self.degraded_pause_s)
            paused_total += self.degraded_pause_s

        write_jsonl_atomic(
            wave_dir / "raw_results.jsonl",
            (json.dumps(outcomes[cid], ensure_ascii=False) for cid in sorted(outcomes)),
        )

        counts: dict[str, int] = {}
        for o in outcomes.values():
            s = outcome_status(o)
            counts[s] = counts.get(s, 0) + 1
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
        cap = self.max_rejected_frac if max_rejected_frac is None else max_rejected_frac
        rejected = counts.get("request_rejected", 0)
        if rejected > cap * len(requests):
            bad = sorted(
                cid for cid, o in outcomes.items()
                if outcome_status(o) == "request_rejected"
            )
            raise WaveInvalidRequest(
                f"wave {wave_name}: {rejected}/{len(requests)} request_rejected "
                f"results ({rejected / len(requests):.3%}) exceed "
                f"max_rejected_frac={cap:.3%} — systematic (our bug); "
                f"first ids: {bad[:5]}"
            )

        atomic_write(
            stamp_path,
            json.dumps(
                {
                    "wave": wave_name,
                    "engine": "live",
                    "request_set_hash": rs_hash,
                    "model": self.model,
                    "n_requests": len(requests),
                    "counts": counts,
                    "batch_ids": {"live": live_marker},
                    "template_hashes": template_hashes,
                    "upstream_hash": upstream_hash,
                    "started_at": started_at,
                    "ended_at": _utcnow(),
                },
                indent=1,
            ),
        )
        return outcomes

    # -- internals --------------------------------------------------------------

    async def _execute(
        self,
        wave_dir: Path,
        requests: list[WaveRequest],
        wave_name: str,
        live_marker: str,
    ) -> dict[str, dict]:
        live_dir = wave_dir / "live"
        live_dir.mkdir(exist_ok=True)

        outcomes: dict[str, dict] = {}
        todo: list[WaveRequest] = []
        for r in requests:
            cache = live_dir / f"{r.custom_id}.json"
            if cache.exists():
                outcomes[r.custom_id] = read_json(cache)  # resume unit
            else:
                todo.append(r)
        if len(outcomes):
            self.on_event(
                f"wave {wave_name}: {len(outcomes)} cached outcomes reused, "
                f"{len(todo)} to issue"
            )

        by_id = {r.custom_id: r for r in requests}
        # a cached ERRORED outcome (a 529/transport window that outlasted the
        # per-request retry budget) is not terminal across invocations — a
        # resume re-issues it, exactly like the batch engine's retry rounds
        retryable_cached = [
            by_id[cid] for cid, o in outcomes.items()
            if outcome_status(o) in RETRYABLE_STATUSES
        ]
        if not todo and not retryable_cached:
            return outcomes

        limiter = ShrinkLimiter(
            self.max_concurrency,
            floor=self.min_concurrency,
            cooldown_s=self.halve_cooldown_s,
            on_event=self.on_event,
        )
        if self._credential_rejected is not None:
            # this engine's credential was rejected earlier in the process -- no client, no request (the latch)
            raise self._credential_rejected
        client = self.client_factory()
        lock = asyncio.Lock()

        async def _run_pool(reqs: list[WaveRequest], label: str) -> None:
            done = 0

            async def _one(req: WaveRequest) -> None:
                nonlocal done
                out = await self._request(client, limiter, req)
                out["batch_id"] = live_marker
                atomic_write(
                    live_dir / f"{req.custom_id}.json", json.dumps(out, ensure_ascii=False)
                )
                async with lock:
                    outcomes[req.custom_id] = out
                    done += 1
                    if done % 25 == 0 or done == len(reqs):
                        self.on_event(f"wave {wave_name}{label}: {done}/{len(reqs)} live results")

            tasks = [asyncio.ensure_future(_one(r)) for r in reqs]
            try:
                await asyncio.gather(*tasks)
            except WaveCredentialRejected as e:
                # the API rejected the CREDENTIAL -- the wave's failure, not an item's. The latch
                # (set where the rejection was classified, before this handler runs) has every task that reaches the wire check
                # raise without a request; every task not yet done -- queued behind the limiter or in flight -- is cancelled here
                # (an answer a cancelled in-flight request would have brought is not cached: a re-run re-asks it), the pool is
                # awaited to the end, and the typed halt propagates: nothing stamped, no retry round, no further request on this
                # credential
                pending = [t for t in tasks if not t.done()]
                for t in pending:
                    t.cancel()
                settled = await asyncio.gather(*tasks, return_exceptions=True)
                n_never = sum(1 for r in settled if isinstance(r, (asyncio.CancelledError, WaveCredentialRejected)))
                self.on_event(
                    f"wave {wave_name}{label}: CREDENTIAL REJECTED after {done}/{len(reqs)} results: {e} -- "
                    f"{n_never} request(s) of this pool were never issued or never answered; the wave is NOT stamped (D6)"
                )
                raise

        try:
            # cached errored outcomes are re-issued in the FIRST pool of a
            # resume (not left to the retry rounds, which a caller may set to
            # 0): a resume after a degraded window must always re-ask
            initial = todo + [r for r in retryable_cached if r.custom_id not in {t.custom_id for t in todo}]
            if initial:
                await _run_pool(initial, "" if not retryable_cached else " resume")
            # engine-level retry of retryable failures (server errors — the
            # 529/transport long tail), mirroring WaveEngine.retry_rounds;
            # the pause lets an overload window pass before re-issuing
            for round_no in range(1, self.retry_rounds + 1):
                failed = sorted(
                    cid for cid, o in outcomes.items()
                    if outcome_status(o) in RETRYABLE_STATUSES
                )
                if not failed:
                    break
                self.on_event(
                    f"wave {wave_name}: retry round {round_no} for {len(failed)} requests"
                )
                await self.sleep(self.retry_pause_s)
                await _run_pool([by_id[cid] for cid in failed], f" retry{round_no}")
        finally:
            limiter.close()
            close = getattr(client, "close", None)
            if close is not None:
                res = close()
                if asyncio.iscoroutine(res):
                    await res
        return outcomes

    def _latch_credential_rejection(self, exc: BaseException) -> WaveCredentialRejected | None:
        """The typed halt for ``exc`` when it is the API's credential rejection (``client.credential_rejection``: a 401 / 403
        on the request, or a federation identity the token exchange refused), latched on this engine -- the FIRST one filed
        is the run's and every later task raises that same object; None for every other error."""
        halt = credential_halt(exc)
        if halt is None:
            return None
        if self._credential_rejected is None:
            self._credential_rejected = halt
        return self._credential_rejected

    async def _request(self, client, limiter: ShrinkLimiter, req: WaveRequest) -> dict:
        """One request to a terminal outcome dict (never raises for
        per-request outcomes; raises for cancellation and for the credential:
        ``WaveCredentialRejected`` at the first 401 / 403 of the engine)."""
        attempt = 0
        while True:
            delay = None
            async with limiter:
                held = self._credential_rejected
                if held is not None:
                    raise held                   # Latched -- no wire call after the first rejection of this engine
                try:
                    async with client.messages.stream(**req.params) as stream:
                        msg = await stream.get_final_message()
                    return {
                        "custom_id": req.custom_id,
                        "type": "succeeded",
                        "stop_reason": getattr(msg, "stop_reason", None),
                        "text": message_text(msg.content),
                    }
                except anthropic.RateLimitError:
                    limiter.halve()
                    attempt += 1
                    if attempt >= self.max_attempts:
                        return self._errored(req, "rate_limit_error", "429 after retries")
                    delay = self._backoff(attempt)
                except Exception as e:  # noqa: BLE001 -- classified below, the credential FIRST
                    # a 401 / 403 on the request, or a federation identity the token exchange
                    # refused (``WorkloadIdentityError``: an AnthropicError that matched no branch here and escaped the pool
                    # raw), is the CREDENTIAL's outcome, never this item's -- typed, latched, raised: no cache, no retry round
                    rejected = self._latch_credential_rejection(e)
                    if rejected is not None:
                        raise rejected from e
                    if isinstance(e, anthropic.APIStatusError):
                        if e.status_code in _RETRYABLE_HTTP:
                            if e.status_code == 529:
                                # Overloaded is a capacity signal like 429: shed
                                # load (cooldown makes one burst halve once)
                                limiter.halve()
                            attempt += 1
                            if attempt >= self.max_attempts:
                                et, em = _error_fields(e)
                                return self._errored(req, et or "api_error", em)
                            delay = self._backoff(attempt)
                        else:
                            # 4xx-class server-side verdict on a well-formed submit:
                            # an OUTCOME (content_filtered / request_rejected /
                            # invalid_request), classified by outcome_status exactly
                            # like a batch result
                            et, em = _error_fields(e)
                            return self._errored(req, et, em)
                    elif isinstance(e, (
                        anthropic.APIConnectionError,
                        anthropic.APITimeoutError,
                        # a transport error raised while READING the stream body
                        # (peer closed mid-stream: httpx2.RemoteProtocolError,
                        # ReadError, ...) surfaces raw from httpx2, not wrapped by
                        # the SDK — it killed the r2 edit wave at 1,615/3,958 on
                        # 2026-09-03; it is the same retryable transport class
                        httpx2.TransportError,
                        # the token exchange's 5xx or an unreachable token endpoint
                        # (a WorkloadIdentityError WITHOUT a credential status): the
                        # same transient class -- the rejection was classified above
                        getattr(anthropic, "WorkloadIdentityError", ()),
                    )):
                        attempt += 1
                        if attempt >= self.max_attempts:
                            return self._errored(
                                req, "api_error", "transport failure after retries"
                            )
                        delay = self._backoff(attempt)
                    else:
                        raise
            if delay is not None:
                await self.sleep(delay)  # back off OUTSIDE the limiter slot

    def _backoff(self, attempt: int) -> float:
        base = min(self.backoff_base_s * (2 ** (attempt - 1)), self.backoff_cap_s)
        return base * (0.5 + random.random())

    @staticmethod
    def _errored(req: WaveRequest, error_type: str, message: str) -> dict:
        return {
            "custom_id": req.custom_id,
            "type": "errored",
            "error_type": error_type,
            "error_message": message,
        }
