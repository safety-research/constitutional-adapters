"""Live wave engine (live.py): artifact parity with the batch engine, resume
from per-request caches, deliberate retry ownership (429 halving, 5xx retry
rounds, cached-errored re-issue), and the same halt semantics."""

from __future__ import annotations

import asyncio
import json

import pytest

from corpus_generator.batch import WaveCredentialRejected, WaveHalt, WaveHashMismatch, WaveInvalidRequest, WaveRequest, outcome_status
from corpus_generator.io import read_json
from corpus_generator.live import LiveEngine, ShrinkLimiter
from fakes import (
    FakeLiveClient,
    exchange_rejecting_async_client,
    federation_rejected,
    live_bad_request,
    live_message,
    live_unauthorized,
    overloaded,
    rate_limited,
    transport_error,
)


def reqs(n: int) -> list[WaveRequest]:
    return [
        WaveRequest(custom_id=f"req-{i:03d}", params={"model": "m", "max_tokens": 10,
                    "messages": [{"role": "user", "content": f"prompt {i}"}]})
        for i in range(n)
    ]


async def _noop(_s: float) -> None:
    return None


def make_engine(client, **kw) -> LiveEngine:
    kw.setdefault("sleep", _noop)
    kw.setdefault("max_concurrency", 8)
    kw.setdefault("min_concurrency", 1)  # tests run tiny waves; don't floor out halving
    return LiveEngine(lambda: client, **kw)


def echo(params, attempt):
    return live_message(f"echo {params['messages'][0]['content']}")


# ---------------------------------------------------------------- happy path


def test_wave_artifacts_and_outcome_shape(tmp_path):
    client = FakeLiveClient(echo)
    engine = make_engine(client)
    out = engine.run_wave(tmp_path / "w", reqs(5), wave_name="w")
    assert set(out) == {f"req-{i:03d}" for i in range(5)}
    o = out["req-000"]
    assert o["type"] == "succeeded" and o["text"] == "echo prompt 0"
    assert o["stop_reason"] == "end_turn" and o["batch_id"] == "live-w"
    assert outcome_status(o) == "succeeded"
    # batch-engine artifact parity
    wave = tmp_path / "w"
    assert (wave / "STAMP.json").exists() and (wave / "raw_results.jsonl").exists()
    assert (wave / "REQUEST_SET.json").exists() and (wave / "requests.jsonl").exists()
    stamp = read_json(wave / "STAMP.json")
    assert stamp["counts"] == {"succeeded": 5}
    assert stamp["batch_ids"] == {"live": "live-w"}
    assert len(list((wave / "live").glob("*.json"))) == 5
    assert client.closed


def test_stamped_wave_skips_api_and_hash_guard(tmp_path):
    client = FakeLiveClient(echo)
    engine = make_engine(client)
    engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    n = client.calls
    again = engine.run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert client.calls == n and len(again) == 3
    with pytest.raises(WaveHashMismatch):
        engine.run_wave(
            tmp_path / "w",
            [WaveRequest(custom_id="req-000", params={"model": "other"})],
            wave_name="w",
        )


def test_refusal_and_max_tokens_pass_through(tmp_path):
    def responder(params, attempt):
        i = int(params["messages"][0]["content"].split()[-1])
        if i == 0:
            return live_message("", stop_reason="refusal")
        if i == 1:
            return live_message("trunc", stop_reason="max_tokens")
        return live_message("fine")

    out = make_engine(FakeLiveClient(responder)).run_wave(
        tmp_path / "w", reqs(3), wave_name="w"
    )
    assert outcome_status(out["req-000"]) == "refusal"
    assert outcome_status(out["req-001"]) == "max_tokens"
    assert outcome_status(out["req-002"]) == "succeeded"


# ---------------------------------------------------------------- resume


def test_resume_reuses_cached_and_reissues_only_missing(tmp_path):
    boom = {"n": 0}

    def responder(params, attempt):
        boom["n"] += 1
        if boom["n"] > 3:
            raise AssertionError("interrupted run")
        return echo(params, attempt)

    client = FakeLiveClient(responder)
    with pytest.raises(AssertionError):
        make_engine(client, max_concurrency=1).run_wave(tmp_path / "w", reqs(6), wave_name="w")
    cached = len(list((tmp_path / "w" / "live").glob("*.json")))
    assert 0 < cached < 6 and not (tmp_path / "w" / "STAMP.json").exists()

    client2 = FakeLiveClient(echo)
    out = make_engine(client2).run_wave(tmp_path / "w", reqs(6), wave_name="w")
    assert len(out) == 6 and client2.calls == 6 - cached


def test_cached_errored_outcome_is_reissued_on_resume(tmp_path):
    # a 529 window that outlasted the in-run budget must not be terminal
    # across invocations (the batch engine would have retried server errors)
    wave = tmp_path / "w"
    (wave / "live").mkdir(parents=True)
    stale = {"custom_id": "req-000", "type": "errored",
             "error_type": "overloaded_error", "error_message": "Overloaded",
             "batch_id": "live-w"}
    (wave / "live" / "req-000.json").write_text(json.dumps(stale))
    client = FakeLiveClient(echo)
    out = make_engine(client).run_wave(wave, reqs(2), wave_name="w")
    assert out["req-000"]["type"] == "succeeded"
    assert read_json(wave / "STAMP.json")["counts"] == {"succeeded": 2}


# ---------------------------------------------------------------- retries


def test_retry_rounds_recover_transient_5xx(tmp_path):
    def responder(params, attempt):
        i = int(params["messages"][0]["content"].split()[-1])
        if i < 2 and attempt <= 6:  # exhaust the per-request budget once
            raise overloaded()
        return echo(params, attempt)

    client = FakeLiveClient(responder)
    engine = make_engine(client, max_attempts=3, retry_rounds=2)
    out = engine.run_wave(tmp_path / "w", reqs(4), wave_name="w")
    # attempts 1-3 errored (round 0), attempts 4-6 errored (retry 1),
    # attempt 7+ succeeds (retry 2)
    assert all(o["type"] == "succeeded" for o in out.values())


def test_transport_failures_bounded_then_errored(tmp_path):
    def responder(params, attempt):
        raise transport_error()

    out = make_engine(
        FakeLiveClient(responder), max_attempts=2, retry_rounds=1
    ).run_wave(tmp_path / "w", reqs(1), wave_name="w")
    assert outcome_status(out["req-000"]) == "errored"
    assert read_json(tmp_path / "w" / "STAMP.json")["counts"] == {"errored": 1}


def test_429_halves_concurrency_once_per_burst(tmp_path):
    events: list[str] = []

    def responder(params, attempt):
        if attempt == 1:
            raise rate_limited()
        return echo(params, attempt)

    engine = make_engine(
        FakeLiveClient(responder), max_concurrency=8, halve_cooldown_s=3600.0,
        on_event=events.append,
    )
    out = engine.run_wave(tmp_path / "w", reqs(8), wave_name="w")
    assert all(o["type"] == "succeeded" for o in out.values())
    halvings = [e for e in events if "halving live concurrency" in e]
    assert halvings == ["429: halving live concurrency -> 4"]  # burst = ONE halving


def test_529_also_halves(tmp_path):
    events: list[str] = []

    def responder(params, attempt):
        if attempt == 1:
            raise overloaded()
        return echo(params, attempt)

    engine = make_engine(
        FakeLiveClient(responder), max_concurrency=4, halve_cooldown_s=3600.0,
        on_event=events.append,
    )
    engine.run_wave(tmp_path / "w", reqs(4), wave_name="w")
    assert any("halving live concurrency -> 2" in e for e in events)


def test_shrink_limiter_cooldown_and_floor():
    async def run():
        t = {"now": 0.0}
        lim = ShrinkLimiter(16, floor=2, cooldown_s=10.0, clock=lambda: t["now"])
        lim.halve()
        assert lim.limit == 8
        lim.halve()  # same burst: inside cooldown, no-op
        assert lim.limit == 8
        t["now"] = 11.0
        lim.halve()  # persisting: halves again
        assert lim.limit == 4
        t["now"] = 22.0
        lim.halve()
        t["now"] = 33.0
        lim.halve()  # floor binds
        assert lim.limit == 2
        lim.close()

    asyncio.run(run())


def test_concurrency_respects_limit(tmp_path):
    async def responder_sleepy(params, attempt):
        return echo(params, attempt)

    client = FakeLiveClient(echo)
    make_engine(client, max_concurrency=3).run_wave(tmp_path / "w", reqs(12), wave_name="w")
    assert client.max_in_flight <= 3


# ---------------------------------------------------------------- rejections


def test_content_filter_is_per_record_attrition(tmp_path):
    def responder(params, attempt):
        i = int(params["messages"][0]["content"].split()[-1])
        if i == 0:
            raise live_bad_request("Output blocked by content filtering policy")
        return echo(params, attempt)

    out = make_engine(FakeLiveClient(responder)).run_wave(tmp_path / "w", reqs(3), wave_name="w")
    assert outcome_status(out["req-000"]) == "content_filtered"


def test_our_bug_invalid_request_halts(tmp_path):
    def responder(params, attempt):
        raise live_bad_request("max_tokens: must be at least 1")

    with pytest.raises(WaveInvalidRequest):
        make_engine(FakeLiveClient(responder)).run_wave(tmp_path / "w", reqs(2), wave_name="w")


def test_rejected_cap_enforced_and_liftable(tmp_path):
    def responder(params, attempt):
        i = int(params["messages"][0]["content"].split()[-1])
        if i == 0:
            raise live_bad_request("Invalid request data")  # content-dependent
        return echo(params, attempt)

    with pytest.raises(WaveInvalidRequest):
        make_engine(FakeLiveClient(responder), max_rejected_frac=0.001).run_wave(
            tmp_path / "w1", reqs(4), wave_name="w1"
        )
    out = make_engine(FakeLiveClient(responder)).run_wave(
        tmp_path / "w2", reqs(4), wave_name="w2", max_rejected_frac=1.0
    )
    assert outcome_status(out["req-000"]) == "request_rejected"


def test_mid_stream_transport_error_is_retried_not_fatal(tmp_path):
    """2026-09-03: httpx2.RemoteProtocolError ("peer closed connection without
    sending complete message body") surfaced raw from get_final_message and
    took the whole wave down. Any httpx2 transport error is a retryable
    per-request failure, exactly like the SDK's connection errors."""
    import httpx2

    def responder(params, attempt):
        if attempt == 1:
            raise httpx2.RemoteProtocolError("peer closed connection without sending complete message body")
        return echo(params, attempt)

    engine = make_engine(FakeLiveClient(responder), max_attempts=3)
    out = engine.run_wave(tmp_path / "w", reqs(4), wave_name="w")
    assert all(o["type"] == "succeeded" for o in out.values())

    def always(params, attempt):
        raise httpx2.ReadError("read failed")

    engine2 = make_engine(FakeLiveClient(always), max_attempts=2, retry_rounds=0)
    out2 = engine2.run_wave(tmp_path / "w2", reqs(2), wave_name="w2")
    assert all(o["type"] == "errored" and "transport failure" in o["error_message"] for o in out2.values())


# ---------------------------------------------------------------- the credential (stop at the FIRST rejection)


def _raising(exc):
    """A responder that raises ``exc`` on every call."""
    def responder(params, attempt):
        raise exc
    return responder


def _assert_stopped_unstamped(wave, client_calls: int, limit: int, outcomes_cached: int = 0) -> None:
    """The wave stopped at the first rejected request: at most one request per in-flight slot, nothing stamped, no raw results, and the
    rejected requests left NO per-request cache (a rejection is the credential's outcome, never an item's; a re-run re-asks them)."""
    assert client_calls <= limit, client_calls
    assert not (wave / "STAMP.json").exists() and not (wave / "raw_results.jsonl").exists()
    cached = sorted((wave / "live").glob("*.json")) if (wave / "live").exists() else []
    assert len(cached) == outcomes_cached, [c.name for c in cached]


@pytest.mark.parametrize("status", [401, 403])
def test_a_credential_rejection_stops_the_wave_at_the_first_rejected_request_and_cancels_the_rest(tmp_path, monkeypatch, status):
    """``_request`` treated a 401 / 403 ``APIStatusError`` as ONE item's errored outcome, ``_run_pool`` had already scheduled every
    request, and the engine's retry rounds then re-issued the errored outcomes -- a revoked credential cost a whole wave of doomed
    requests (a register-edit wave is thousands) and left `errored` caches that a resume re-asked. Now the first rejection is the typed
    ``WaveCredentialRejected`` (a ``WaveHalt``: never auto-resubmitted), the outstanding pool is cancelled, nothing is stamped, the
    rejected requests leave no cache, and the engine issues nothing else for the rest of the process (the latch)."""
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    client = FakeLiveClient(_raising(live_unauthorized(status)))
    events: list[str] = []
    eng = make_engine(client, max_concurrency=3, retry_rounds=2, on_event=events.append)
    with pytest.raises(WaveCredentialRejected) as ei:
        eng.run_wave(tmp_path / "w", reqs(12), wave_name="w")
    e = ei.value
    assert isinstance(e, WaveHalt) and e.status == status
    assert e.error_type == ("AuthenticationError" if status == 401 else "PermissionDeniedError")
    assert e.api_error_type == ("authentication_error" if status == 401 else "permission_error")
    assert f"HTTP {status}" in str(e) and "SISTEER_SECRETS_MODE=wif" in str(e) and "first credential rejection" in str(e) and "D6" in str(e)
    assert "the federation ids file / the VM identity" in str(e) and "prompt 0" not in str(e)       # never request text
    _assert_stopped_unstamped(tmp_path / "w", client.calls, limit=3)
    assert client.closed                                                                 # the finally still runs
    assert sum("CREDENTIAL REJECTED" in m for m in events) == 1 and not any("retry round" in m for m in events)
    (said,) = [m for m in events if "CREDENTIAL REJECTED" in m]
    assert "never issued" in said and "prompt" not in said
    # the latch: a second wave on the same engine issues NOTHING and raises the same rejection before any request
    before = client.calls
    with pytest.raises(WaveCredentialRejected) as again:
        eng.run_wave(tmp_path / "w2", reqs(5), wave_name="w2")
    assert again.value is e and client.calls == before
    assert not (tmp_path / "w2" / "live").exists() or not list((tmp_path / "w2" / "live").glob("*.json"))


def test_a_rejection_mid_wave_keeps_the_outcomes_obtained_before_it_and_a_fixed_credential_resumes_them(tmp_path, monkeypatch):
    """The first four requests answer, the fifth is the rejection: the four verdicts stay cached (the resume unit), the halt is raised
    with at most one more request per in-flight slot, and a NEW engine (a fixed credential: a new process) resumes the wave from the
    cache, re-asking only what was never answered -- every request answered exactly once overall."""
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "op")
    answered = {"n": 0}

    def responder(params, attempt):
        answered["n"] += 1
        if answered["n"] > 4:
            raise live_unauthorized(401)
        return echo(params, attempt)

    client = FakeLiveClient(responder)
    with pytest.raises(WaveCredentialRejected) as ei:
        make_engine(client, max_concurrency=2).run_wave(tmp_path / "w", reqs(12), wave_name="w")
    assert "the pool's vault item" in str(ei.value) and "SISTEER_SECRETS_MODE=op" in str(ei.value)
    cached = sorted((tmp_path / "w" / "live").glob("*.json"))
    assert len(cached) == 4 and all(read_json(c)["type"] == "succeeded" for c in cached)
    assert client.calls <= 4 + 2 and not (tmp_path / "w" / "STAMP.json").exists()
    # the credential fixed: a fresh engine resumes
    fixed = FakeLiveClient(echo)
    out = make_engine(fixed, max_concurrency=2).run_wave(tmp_path / "w", reqs(12), wave_name="w")
    assert len(out) == 12 and all(outcome_status(o) == "succeeded" for o in out.values())
    assert fixed.calls == 8 and (tmp_path / "w" / "STAMP.json").exists()


def test_a_federation_identity_the_token_exchange_refuses_stops_the_wave_through_the_real_async_client(tmp_path, monkeypatch):
    """Under wif the credential is the federation identity and the API refuses it at the token exchange: the SDK raises
    ``WorkloadIdentityError(status_code=401 | 403)`` from the request that needed the bearer -- an ``AnthropicError``, NOT an
    ``APIStatusError``, so before the fix it matched no branch of ``_request`` and escaped the pool raw. Through a REAL AsyncAnthropic
    (offline: a mock token endpoint; the API is never reached): one exchange per in-flight slot at most, the typed halt, nothing stamped.
    The body shape does not matter (an OAuth error string, a non-JSON text). NOT this: the exchange's 5xx and an unreachable token
    endpoint are retried like a transport failure, bounded, then the verdict stands or the outcome is `errored`."""
    monkeypatch.setenv("SISTEER_SECRETS_MODE", "wif")
    for status, body in ((401, {"error": "invalid_grant", "error_description": "synthetic"}), (403, "synthetic non-json body")):
        client = exchange_rejecting_async_client(status, body)
        events: list[str] = []
        with pytest.raises(WaveCredentialRejected) as ei:
            make_engine(client, max_concurrency=3, on_event=events.append).run_wave(tmp_path / f"w{status}", reqs(12), wave_name=f"w{status}")
        assert (ei.value.status, ei.value.error_type, ei.value.api_error_type) == (status, "WorkloadIdentityError", None)
        assert "header.payload.signature" not in str(ei.value)
        _assert_stopped_unstamped(tmp_path / f"w{status}", client.request_hits["n"], limit=3)
        assert client.request_hits["n"] >= 1 and client.exchange_hits["n"] <= 2 * client.request_hits["n"]   # the SDK retries a failed exchange once
        assert sum("CREDENTIAL REJECTED" in m for m in events) == 1
    # the API's own error object at the exchange still yields its type
    client = exchange_rejecting_async_client(401, {"type": "error", "error": {"type": "authentication_error", "message": "synthetic"}})
    with pytest.raises(WaveCredentialRejected) as ei:
        make_engine(client, max_concurrency=3).run_wave(tmp_path / "wt", reqs(3), wave_name="wt")
    assert ei.value.api_error_type == "authentication_error"
    # the fake responder shapes: a 401 exchange is the same stop; a 503 exchange and an unreachable endpoint are transport retries
    client = FakeLiveClient(_raising(federation_rejected(401)))
    with pytest.raises(WaveCredentialRejected):
        make_engine(client, max_concurrency=3).run_wave(tmp_path / "wf", reqs(12), wave_name="wf")
    _assert_stopped_unstamped(tmp_path / "wf", client.calls, limit=3)
    def flaky_exchange(params, attempt):
        if attempt == 1:
            raise federation_rejected(503)
        if attempt == 2:
            raise federation_rejected(None)
        return echo(params, attempt)

    client = FakeLiveClient(flaky_exchange)
    out = make_engine(client, max_attempts=4, retry_rounds=0).run_wave(tmp_path / "w5", reqs(3), wave_name="w5")
    assert all(outcome_status(o) == "succeeded" for o in out.values()) and client.calls == 9
    client = FakeLiveClient(_raising(federation_rejected(None)))
    out = make_engine(client, max_attempts=2, retry_rounds=0).run_wave(tmp_path / "w6", reqs(2), wave_name="w6")
    assert all(outcome_status(o) == "errored" for o in out.values()) and client.calls == 4
