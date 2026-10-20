"""``sisteer.capeval.tinycap.sandbox_backoff`` — retry with backoff on Modal's sandbox-creation rate limit
(tinycap-verify-rate-limit, 2026-09-30). The live failure this pins closed: the gemma (c) row (tree 082f583) spawned
nine CPU verifiers ~40 ms apart, each opening eight sandboxes; Modal refused the creates with
``ResourceExhaustedError: Sandbox creation rate limit (5/s) exceeded``; inspect-sandboxes' own ``_standard_retry``
(five attempts, waits capped at 10 s) exhausted inside ~15 s, inspect aborted every task and the row read VERIFY_FAILED
with its generations intact. The module is PURE (no ``modal``), so this runs in every venv.

Pinned: a create that is refused N times then succeeds is retried with the 1, 2, 4, … s sequence (capped, jittered) and
every retry is reported; a create that is never accepted is re-raised AS ITSELF once the next wait would pass the total
bound (no silent fallback); a refusal that is not the rate limit re-raises at once with no sleep; the predicate reads
both Modal's exception type and the raw gRPC status; the policy comes from the environment."""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

import pytest

from sisteer.capeval.tinycap import contracts as C
from sisteer.capeval.tinycap import sandbox_backoff as BK


class ResourceExhaustedError(Exception):
    """A stand-in named like Modal's (``sandbox_backoff.is_rate_limited`` reads the class name when the real type is absent)."""


class _GRPCError(Exception):
    def __init__(self, status):
        super().__init__(f"grpc {status}")
        self.status = status


def _run(coro):
    return asyncio.run(coro)


def _harness(refusals, *, policy=None, rng_seed=0, exc_factory=None):
    """A fake create that raises ``refusals`` times then returns a sandbox, a recording sleep, a fixed clock that
    advances by each sleep, and the retry / exhausted reports."""
    exc_factory = exc_factory or (lambda i: ResourceExhaustedError(f"Sandbox creation rate limit (5/s) exceeded #{i}"))
    calls, sleeps, retries, exhausted = [], [], [], []
    now = {"t": 100.0}

    async def create(command, kwargs):
        calls.append((list(command), dict(kwargs)))
        if len(calls) <= refusals:
            raise exc_factory(len(calls))
        return SimpleNamespace(object_id=f"sb-{len(calls)}")

    async def sleep(s):
        sleeps.append(s)
        now["t"] += s

    def on_retry(attempt, wait, elapsed, exc):
        retries.append((attempt, wait, elapsed, type(exc).__name__))

    def on_exhausted(attempts, elapsed, exc):
        exhausted.append((attempts, elapsed, type(exc).__name__))

    kw = dict(policy=policy or BK.BackoffPolicy(jitter_s=0.0), sleep=sleep, rng=random.Random(rng_seed), clock=lambda: now["t"],
              on_retry=on_retry, on_exhausted=on_exhausted)
    return create, kw, calls, sleeps, retries, exhausted


def test_a_create_refused_n_times_then_accepted_is_retried_with_exponential_backoff_and_every_retry_reported():
    create, kw, calls, sleeps, retries, exhausted = _harness(3)
    sb = _run(BK.create_with_backoff(create, ["tail", "-f", "/dev/null"], {"name": "n"}, **kw))
    assert sb.object_id == "sb-4" and len(calls) == 4                         # three refusals, the fourth create lands
    assert calls[0] == (["tail", "-f", "/dev/null"], {"name": "n"}) == calls[3]  # the same create each time
    assert sleeps == [1.0, 2.0, 4.0]                                            # base 1 s doubling, no jitter
    assert [(a, w, e) for a, w, e, _ in retries] == [(0, 1.0, 0.0), (1, 2.0, 1.0), (2, 4.0, 3.0)]
    assert all(name == "ResourceExhaustedError" for *_, name in retries) and exhausted == []


def test_the_wait_is_capped_and_jittered_and_the_defaults_are_the_documented_ones():
    pol = BK.BackoffPolicy(base_s=1.0, cap_s=30.0, total_s=300.0, jitter_s=0.0)
    assert [BK.wait_before_attempt(pol, n) for n in range(7)] == [1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 30.0]
    with_jitter = BK.BackoffPolicy(jitter_s=1.0)
    rng = random.Random(7)
    waits = [BK.wait_before_attempt(with_jitter, n, rng) for n in range(3)]
    assert all(2.0 ** n <= w < 2.0 ** n + 1.0 for n, w in enumerate(waits))
    assert BK.DEFAULT_POLICY == BK.BackoffPolicy(base_s=1.0, cap_s=30.0, total_s=300.0, jitter_s=1.0)
    assert C.SANDBOX_CREATE_RATE_LIMIT_PER_S == 5
    with pytest.raises(ValueError, match="attempt must be >= 0"):
        BK.wait_before_attempt(pol, -1)
    with pytest.raises(ValueError, match="cap_s"):
        BK.BackoffPolicy(base_s=5.0, cap_s=1.0)
    with pytest.raises(ValueError, match="base_s must be > 0"):
        BK.BackoffPolicy(base_s=0.0)


def test_a_create_never_accepted_is_reraised_as_itself_once_the_next_wait_would_pass_the_total_bound():
    # total 10 s: waits 1 + 2 + 4 = 7 land; the next (8 s) would carry the total to 15 > 10 -> give up, re-raise the last refusal
    create, kw, calls, sleeps, retries, exhausted = _harness(10 ** 6, policy=BK.BackoffPolicy(total_s=10.0, jitter_s=0.0))
    with pytest.raises(ResourceExhaustedError, match=r"exceeded #4"):
        _run(BK.create_with_backoff(create, [], {}, **kw))
    assert sleeps == [1.0, 2.0, 4.0] and len(calls) == 4                      # four creates: the fourth's refusal is the one raised
    assert exhausted == [(3, 7.0, "ResourceExhaustedError")]                   # 3 retries spent, 7 s elapsed, the bound named
    assert len(retries) == 3
    # a bound of zero: no retry at all, the FIRST refusal comes straight back
    create, kw, calls, sleeps, retries, exhausted = _harness(10 ** 6, policy=BK.BackoffPolicy(total_s=0.0, jitter_s=0.0))
    with pytest.raises(ResourceExhaustedError, match=r"exceeded #1"):
        _run(BK.create_with_backoff(create, [], {}, **kw))
    assert sleeps == [] and len(calls) == 1 and exhausted == [(0, 0.0, "ResourceExhaustedError")]


def test_a_refusal_that_is_not_the_rate_limit_reraises_at_once_with_no_sleep():
    create, kw, calls, sleeps, retries, exhausted = _harness(2, exc_factory=lambda i: RuntimeError(f"image build failed #{i}"))
    with pytest.raises(RuntimeError, match="image build failed #1"):
        _run(BK.create_with_backoff(create, [], {}, **kw))
    assert len(calls) == 1 and sleeps == [] and retries == [] and exhausted == []
    # a cancellation is never retried either
    async def cancelled(command, kwargs):
        raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        _run(BK.create_with_backoff(cancelled, [], {}, **kw))
    assert sleeps == []


def test_the_predicate_reads_modals_type_name_and_the_raw_grpc_status():
    assert BK.is_rate_limited(ResourceExhaustedError("x"))
    class Sub(ResourceExhaustedError):
        pass
    assert BK.is_rate_limited(Sub("x"))                                        # any class in the MRO
    assert BK.is_rate_limited(_GRPCError(SimpleNamespace(name="RESOURCE_EXHAUSTED", value=8)))
    assert BK.is_rate_limited(_GRPCError(8))                                    # the numeric code
    assert not BK.is_rate_limited(_GRPCError(SimpleNamespace(name="UNAVAILABLE", value=14)))
    assert not BK.is_rate_limited(RuntimeError("Sandbox creation rate limit (5/s) exceeded"))   # the message alone is not the type
    assert not BK.is_rate_limited(TimeoutError())


def test_the_policy_comes_from_the_environment_and_a_non_number_refuses_by_name(monkeypatch):
    for var in (BK.ENV_BASE_S, BK.ENV_CAP_S, BK.ENV_TOTAL_S, BK.ENV_JITTER_S):
        monkeypatch.delenv(var, raising=False)
    assert BK.policy_from_env() == BK.DEFAULT_POLICY
    assert BK.policy_from_env({BK.ENV_TOTAL_S: "12", BK.ENV_JITTER_S: "0"}) == BK.BackoffPolicy(total_s=12.0, jitter_s=0.0)
    assert BK.policy_from_env({BK.ENV_BASE_S: "0.5", BK.ENV_CAP_S: "3"}) == BK.BackoffPolicy(base_s=0.5, cap_s=3.0)
    assert BK.policy_from_env({BK.ENV_TOTAL_S: "  "}) == BK.DEFAULT_POLICY  # blank = unset
    with pytest.raises(ValueError, match=f"{BK.ENV_TOTAL_S}='ten' is not a number"):
        BK.policy_from_env({BK.ENV_TOTAL_S: "ten"})
    assert BK.ENV_TOTAL_S == "TINYCAP_SANDBOX_CREATE_BACKOFF_TOTAL_S"
