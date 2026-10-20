"""transport_refusal — the third judge-abstention class (2026-09-02).

What was observed (live, the Qwen3.5 benign-controls sonnet re-grade,
2026-09-02): for 7 of ~2,370 OR-Bench (prompt, generation) bodies per cell,
claude-sonnet-5 returned a FAST (0.7-1.3 s) ``529 overloaded_error`` on every
one of ~18 attempts (SDK x3 inside the 6-step, ~12-minute ladder) while every
sibling request in the same batch succeeded and zero 429s were seen. The same
body still 529'd with tool_choice=auto, with no thinking param, with 4x
max_tokens, with NO tools at all, and with a trailing newline — and cleared
on claude-haiku-4-5 in 1.3 s (stop=tool_use, an ordinary verdict). That is
the judge declining to grade that content, surfaced as a transport status
instead of stop_reason=refusal: a refusal by another channel. (The haiku
cross-check is DOCUMENTATION of the diagnosis; nothing at runtime depends on
a second model.)

Before this, every exhausted ladder was "transport exhausted" -> hard cell
failure — right for an outage, wrong here: re-running fails identically, so
the cell could never be graded at all, and a pipeline that treats exhausted
529s as pure infrastructure is silently unable to grade exactly the
borderline content a refusal-tolerant judge exists to bound over.

The discriminator (BOTH halves required, else the hard failure stands):
  (i)  the request's ladder saw NOTHING but 529s (a 429 or any other status
       in it disqualifies), AND
  (ii) the org was SERVING during that ladder — a sibling request in the same
       judge_all batch was answered inside [t_first, t_last]; failing that (a
       cache-warm re-run whose only live requests ARE the exhausted ones),
       ONE liveness canary to the same model on the same org succeeds.
A batch-wide 529 fails (ii) both ways and stays the loud JudgingError. Same
org throughout — the org-selection seam refuses anything else.
"""
import pathlib
import sys
import time

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sisteer.capeval.judging as j                       # noqa: E402
from sisteer.capeval.judging import (                     # noqa: E402
    ClaudeJudge, JudgeCache, JudgingError)

from test_capeval_judging import (                        # noqa: E402
    ORBENCH_RUBRIC_V1, _Client, _Msg, _StatusError, _cpi_msg, _single_req)


def _fast_ladder(monkeypatch):
    """Under test is WHICH outcome a ladder produces, not how long it naps."""
    monkeypatch.setattr(j, "_429_SLEEP_S", 0)
    monkeypatch.setattr(j, "_429_MAX_SLEEP_S", 0)
    monkeypatch.setattr(j, "_529_SLEEP_S", 0)
    monkeypatch.setattr(j, "_529_MAX_SLEEP_S", 0)


class _Overloaded(_StatusError):
    """The SDK's OverloadedError shape: status 529 + the request id it
    exposes off the ``request-id`` header."""

    def __init__(self, request_id="req_synthetic0001"):
        super().__init__(529)
        self.request_id = request_id


def _never(_params):
    raise AssertionError("the High-Prio org must never be called")


def _is_canary(params):
    # a judge request always carries the forced tool; the canary never does
    return "tools" not in params


def _judge(tmp_path, low_fn, **kw):
    return ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"),
                       clients={"lowprio": _Client(low_fn),
                                "highprio": _Client(_never)},
                       concurrency=kw.pop("concurrency", 4), **kw)


# --- the ladder reports what it saw -------------------------------------------


def test_create_exhaustion_carries_the_whole_ladder(tmp_path, monkeypatch):
    _fast_ladder(monkeypatch)

    def low(_params):
        raise _Overloaded("req_synthetic_ladder")

    judge = _judge(tmp_path, low)
    with pytest.raises(j.JudgeTransportExhausted) as ei:
        judge._create({"model": "m", "messages": []})
    e = ei.value
    assert isinstance(e, JudgingError)  # callers catching JudgingError still do
    assert e.statuses == (529,) * j._529_MAX_ATTEMPTS
    assert e.all_529
    assert e.request_id == "req_synthetic_ladder"
    assert e.t_last >= e.t_first
    assert "transport exhausted" in str(e) and "highprio" not in str(e)


def test_request_id_is_kept_only_as_a_plain_token():
    class _E(Exception):
        pass

    e = _E()
    e.request_id = "req_011CefJuD6VpfmWHygBUqoyQ"
    assert j._request_id_of(e) == "req_011CefJuD6VpfmWHygBUqoyQ"
    e.request_id = "has whitespace"
    assert j._request_id_of(e) is None  # never free text into the cache
    e.request_id = None
    e.body = {"request_id": "req_from_body"}
    assert j._request_id_of(e) == "req_from_body"
    assert j._request_id_of(None) is None


# --- (a) deterministic per-body 529 with serving siblings -> abstention -------


def test_all_529_body_with_serving_siblings_files_transport_refusal(
        tmp_path, monkeypatch):
    """The live 2026-09-02 shape: one body 529s on every attempt while its
    siblings in the same batch are answered. Filed as an abstention under
    its OWN reason, cached like every abstention sentinel, siblings
    verdicted, the run completes, no canary needed, no org hop."""
    _fast_ladder(monkeypatch)
    calls = {"cursed": 0, "canary": 0}

    def low(params):
        if _is_canary(params):
            calls["canary"] += 1
            return _Msg([])
        body = params["messages"][0]["content"]
        if "cursed" in body:
            calls["cursed"] += 1
            time.sleep(0.05)  # the ladder spans the siblings' answers
            raise _Overloaded("req_synthetic_cursed")
        time.sleep(0.1)  # answered INSIDE the cursed ladder's window
        return _cpi_msg("full_compliance")

    judge = _judge(tmp_path, low)
    cursed = _single_req("synthetic cursed prompt", "an ordinary answer")
    fine = [_single_req(f"synthetic fine prompt {i}", "ok") for i in range(3)]
    out = judge.judge_all([(cursed, ORBENCH_RUBRIC_V1)]
                          + [(r, ORBENCH_RUBRIC_V1) for r in fine],
                          allow_refusal_abstention=True)
    v = out[cursed.key()]
    assert v["abstained"] is True
    assert v["reason"] == "transport_refusal"
    assert v["reason"] not in ("refusal", "max_tokens")  # never masquerades
    assert v["attempts"] == j._529_MAX_ATTEMPTS == calls["cursed"]
    assert v["evidence"] == "siblings" and v["siblings_succeeded"] >= 1
    assert v["request_id"] == "req_synthetic_cursed"
    for r in fine:
        assert out[r.key()] == {"label": "full_compliance"}
    assert judge.n_abstained == 1 and judge.n_transport_refusals == 1
    assert calls["canary"] == 0 and judge.n_canary_calls == 0
    assert judge.cache.get(cursed.key()) == v  # stable bounds on re-read
    assert judge.n_calls_by_org["highprio"] == 0
    assert judge.n_calls_by_org["lowprio"] == 4  # 3 verdicts + 1 abstention


def test_transport_refusal_requires_the_optin(tmp_path, monkeypatch):
    """Strict mode is unchanged: an all-529 ladder is unverdicted and the
    pass raises; nothing is cached, no canary is sent."""
    _fast_ladder(monkeypatch)

    def low(params):
        if _is_canary(params):
            raise AssertionError("no canary in strict mode")
        if "cursed" in params["messages"][0]["content"]:
            raise _Overloaded()
        return _cpi_msg("full_compliance")

    judge = _judge(tmp_path, low)
    cursed = _single_req("synthetic cursed prompt", "an ordinary answer")
    fine = _single_req("synthetic fine prompt", "ok")
    with pytest.raises(JudgingError, match="unverdicted"):
        judge.judge_all([(cursed, ORBENCH_RUBRIC_V1), (fine, ORBENCH_RUBRIC_V1)])
    assert judge.cache.get(cursed.key()) is None
    assert judge.n_transport_refusals == 0 and judge.n_canary_calls == 0


def test_a_ladder_with_any_non_529_is_not_a_transport_refusal(
        tmp_path, monkeypatch):
    """Half (i): a 429 anywhere in the ladder disqualifies it, siblings or
    not — only an all-529 ladder can be the judge declining a body."""
    _fast_ladder(monkeypatch)
    seen = {"n": 0}

    def low(params):
        if _is_canary(params):
            return _Msg([])
        if "cursed" in params["messages"][0]["content"]:
            seen["n"] += 1
            raise _StatusError(429) if seen["n"] == 1 else _Overloaded()
        return _cpi_msg("full_compliance")

    judge = _judge(tmp_path, low)
    cursed = _single_req("synthetic cursed prompt", "an ordinary answer")
    fine = _single_req("synthetic fine prompt", "ok")
    with pytest.raises(JudgingError, match="unverdicted") as ei:
        judge.judge_all([(cursed, ORBENCH_RUBRIC_V1), (fine, ORBENCH_RUBRIC_V1)],
                        allow_refusal_abstention=True)
    assert "JudgeTransportExhausted" in str(ei.value)
    assert judge.n_transport_refusals == 0
    assert judge.cache.get(cursed.key()) is None
    assert judge.cache.get(fine.key()) == {"label": "full_compliance"}


# --- (b) batch-wide 529 -> the hard failure stands ----------------------------


def test_batch_wide_529_stays_a_hard_failure(tmp_path, monkeypatch):
    """Nothing in the batch is answered and the canary 529s too: the org
    was NOT serving, so every exhausted request stays unverdicted and the
    pass raises — an outage is never laundered into abstentions."""
    _fast_ladder(monkeypatch)
    n = {"canary": 0}

    def low(params):
        if _is_canary(params):
            n["canary"] += 1
        raise _Overloaded()

    judge = _judge(tmp_path, low)
    reqs = [_single_req(f"synthetic prompt {i}", "reply") for i in range(3)]
    with pytest.raises(JudgingError, match="unverdicted") as ei:
        judge.judge_all([(r, ORBENCH_RUBRIC_V1) for r in reqs],
                        allow_refusal_abstention=True)
    msg = str(ei.value)
    assert "JudgeTransportExhausted" in msg and "not shown serving" in msg
    assert n["canary"] == 1 == judge.n_canary_calls  # asked ONCE per batch
    assert judge.n_abstained == 0 and judge.n_transport_refusals == 0
    for r in reqs:
        assert judge.cache.get(r.key()) is None
    assert judge.n_calls_by_org["highprio"] == 0


# --- (ii) fallback: the cache-warm re-run has no siblings -> canary -----------


def test_cache_warm_rerun_with_serving_canary_files_transport_refusal(
        tmp_path, monkeypatch):
    """The re-run shape: every sibling is a cache hit, so the batch's ONLY
    live requests are the exhausted ones and no sibling can testify. One
    liveness canary to the same model on the same org answers -> the org
    was serving -> transport_refusal (evidence: canary)."""
    _fast_ladder(monkeypatch)
    n = {"canary": 0}
    judge_ref = {}

    def low(params):
        if _is_canary(params):
            n["canary"] += 1
            assert params["model"] == judge_ref["j"].model
            assert params["max_tokens"] == j.CANARY_MAX_TOKENS
            assert params["thinking"] == dict(j.THINKING)
            return _Msg([])
        raise _Overloaded("req_synthetic_rerun")

    judge = _judge(tmp_path, low)
    judge_ref["j"] = judge
    warm = _single_req("synthetic warm prompt", "ok")
    judge.cache.put(warm.key(), {"label": "full_compliance"}, "lowprio")
    cursed = _single_req("synthetic cursed prompt", "an ordinary answer")
    out = judge.judge_all([(warm, ORBENCH_RUBRIC_V1), (cursed, ORBENCH_RUBRIC_V1)],
                          allow_refusal_abstention=True)
    assert out[warm.key()] == {"label": "full_compliance"}
    assert judge.n_cache_hits == 1
    v = out[cursed.key()]
    assert v["reason"] == "transport_refusal"
    assert v["evidence"] == "canary" and v["siblings_succeeded"] == 0
    assert v["request_id"] == "req_synthetic_rerun"
    assert n["canary"] == 1 == judge.n_canary_calls
    assert judge.cache.get(cursed.key()) == v


def test_cache_warm_rerun_with_failing_canary_hard_fails(tmp_path, monkeypatch):
    """Same re-run shape, but the canary 529s as well: not shown serving,
    the request stays unverdicted, the pass raises."""
    _fast_ladder(monkeypatch)

    def low(_params):
        raise _Overloaded()

    judge = _judge(tmp_path, low)
    warm = _single_req("synthetic warm prompt", "ok")
    judge.cache.put(warm.key(), {"label": "full_compliance"}, "lowprio")
    cursed = _single_req("synthetic cursed prompt", "an ordinary answer")
    with pytest.raises(JudgingError, match="not shown serving"):
        judge.judge_all([(warm, ORBENCH_RUBRIC_V1), (cursed, ORBENCH_RUBRIC_V1)],
                        allow_refusal_abstention=True)
    assert judge.n_canary_calls == 1
    assert judge.cache.get(cursed.key()) is None


def test_canary_is_sent_once_per_batch_and_never_cached(tmp_path, monkeypatch):
    _fast_ladder(monkeypatch)
    n = {"canary": 0}

    def low(params):
        if _is_canary(params):
            n["canary"] += 1
            return _Msg([])
        raise _Overloaded()

    judge = _judge(tmp_path, low)
    reqs = [_single_req(f"synthetic cursed prompt {i}", "answer") for i in range(3)]
    out = judge.judge_all([(r, ORBENCH_RUBRIC_V1) for r in reqs],
                          allow_refusal_abstention=True)
    assert all(v["reason"] == "transport_refusal" for v in out.values())
    assert n["canary"] == 1  # three candidates, one canary
    assert len(judge.cache) == 3  # the three sentinels; the canary left nothing


# --- the sentinel's reason is one of the five named classes -------------------


def test_abstention_reasons_are_the_five_named_classes():
    assert j.ABSTENTION_REASONS == ("refusal", "max_tokens", "transport_refusal",
                                    "content_filter", "malformed_verdict")


def test_coordination_post_names_the_class():
    post = j.coordination_post(n_unique_requests=10)
    assert "transport_refusal" in post
    assert "same-org backoff" in post and "highprio" not in post
