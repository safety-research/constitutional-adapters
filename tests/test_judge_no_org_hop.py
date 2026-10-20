"""A 429 must THROTTLE DOWN, never hop Low-Prio -> High-Prio.

A standing rule of 2026-08-21 replaced the old "429 -> High-Prio fallback"
policy and said explicitly not to reinstate it: at
200-500 concurrent, a saturated Low-Prio pool is most likely OUR doing, and
moving that load to a second shared pool relocates the contention onto the
pool's other users instead of easing it. These tests exist because the old behaviour was
already implemented once and would otherwise be easy to reintroduce.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sisteer.capeval.judging as j                       # noqa: E402
from sisteer.capeval.judging import (                     # noqa: E402
    ClaudeJudge, JudgeCache, JudgingError)

from test_capeval_judging import (                        # noqa: E402
    ORBENCH_RUBRIC_V1, _Client, _StatusError, _cpi_msg, _single_req)


def _fast_backoff(monkeypatch):
    """The policy under test is which ORG gets called, not how long we nap."""
    monkeypatch.setattr(j, "_429_SLEEP_S", 0)
    monkeypatch.setattr(j, "_429_MAX_SLEEP_S", 0)
    monkeypatch.setattr(j, "_529_SLEEP_S", 0)
    monkeypatch.setattr(j, "_529_MAX_SLEEP_S", 0)


def _judge(tmp_path, low_fn, high_fn, **kw):
    return ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"),
                       clients={"lowprio": _Client(low_fn),
                                "highprio": _Client(high_fn)},
                       concurrency=kw.pop("concurrency", 4), **kw)


def test_429_never_reaches_highprio(tmp_path, monkeypatch):
    _fast_backoff(monkeypatch)
    judge = _judge(tmp_path,
                   lambda p: (_ for _ in ()).throw(_StatusError(429)),
                   lambda p: _cpi_msg("full_refusal"))
    with pytest.raises(JudgingError):
        judge.judge_all([(_single_req("synthetic prompt", "no."),
                          ORBENCH_RUBRIC_V1)])
    assert judge._clients["highprio"].messages.calls == [], (
        "a 429 on Low-Prio sent traffic to High-Prio — that is the D17 "
        "behaviour the 2026-08-21 rule replaced")
    assert judge._clients["lowprio"].messages.calls, "never retried at all"


def test_429_throttles_the_in_flight_ceiling(tmp_path, monkeypatch):
    _fast_backoff(monkeypatch)
    judge = _judge(tmp_path,
                   lambda p: (_ for _ in ()).throw(_StatusError(429)),
                   lambda p: _cpi_msg("full_refusal"), concurrency=64)
    with pytest.raises(JudgingError):
        judge.judge_all([(_single_req("synthetic prompt", "no."),
                          ORBENCH_RUBRIC_V1)])
    assert judge._live_concurrency < 64, (
        "backoff alone only spaces out ONE worker's retries; the ceiling has "
        "to come down for our footprint on the shared pool to shrink")
    assert judge._live_concurrency >= 8, "throttled below the floor"
    assert judge.n_rate_limited > 0


def test_a_throttle_down_is_permanent(tmp_path):
    """A swallowed permit must not come back when the worker finishes.

    Concurrency 100, not 16: the throttle floor is 50, so at 16 the gate is
    already at its floor and there is nothing to cut. (Floor semantics are
    covered directly in tests/test_throttle_gate.py.)
    """
    judge = _judge(tmp_path, lambda p: _cpi_msg("full_refusal"),
                   lambda p: _cpi_msg("full_refusal"), concurrency=100)
    judge._acquire()
    judge._note_rate_limit()          # 100 -> 50, owing 50 permits
    judge._release()                  # this one is swallowed, not handed back
    assert judge._live_concurrency == 50
    assert judge._permit_debt == 49


def test_source_has_no_org_fallback_loop():
    """Grep the source: catch the bypass before it ships, not at runtime."""
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src/sisteer/capeval/judging.py").read_text()
    body = src[src.index("def _create("):src.index("def _note_rate_limit(")]
    assert "highprio" not in body, (
        "_create names highprio — it must call exactly one org, the one "
        "pinned at construction")
    assert 'for org in (' not in body, "_create still loops over orgs"
    # the org-selection seam must sit IN the 429 path, so a reinstated hop
    # cannot route around the guard without deleting the seam (loudly)
    assert 'cause="rate_limit"' in body, (
        "_create's 429 handler no longer re-selects its org through "
        "_select_org — the guard is unwired")


# --- the org-selection seam: refuse a 429-caused escalation ------------------


def test_429_recovers_on_the_same_org_and_throttles(tmp_path, monkeypatch):
    """The reinstatement canary in the SUCCESS path: one 429, then a verdict.

    A reinstated fallback would satisfy the exhaustion tests above by
    never succeeding; this one pins the recovery shape — the verdict lands,
    the org recorded in provenance is the pinned one, High-Prio saw zero
    traffic, and the in-flight ceiling came down."""
    _fast_backoff(monkeypatch)
    calls = {"n": 0}

    def low(p):
        calls["n"] += 1
        if calls["n"] == 1:
            raise _StatusError(429)
        return _cpi_msg("full_refusal")

    judge = _judge(tmp_path, low, lambda p: _cpi_msg("full_refusal"),
                   concurrency=200)
    out = judge.judge_all([(_single_req("synthetic prompt", "no."),
                            ORBENCH_RUBRIC_V1)])
    (key,) = out
    assert out[key] == {"label": "full_refusal"}
    assert judge._clients["highprio"].messages.calls == [], (
        "the 429 moved traffic to High-Prio — the retired D17 fallback")
    assert judge.n_calls_by_org == {"lowprio": 1, "highprio": 0}
    assert judge.cache.get_org(key) == "lowprio"   # provenance: pinned org
    assert judge.n_rate_limited == 1
    assert judge._live_concurrency == 100, (
        "one 429 must halve the in-flight ceiling (200 -> 100): backoff "
        "alone does not shrink our footprint on the shared pool")


def test_select_org_refuses_a_rate_limit_caused_escalation(tmp_path):
    """The seam itself: a High-Prio selection whose CAUSE is a rate limit
    raises loudly, naming the rule that forbids it."""
    judge = _judge(tmp_path, lambda p: _cpi_msg("full_refusal"),
                   lambda p: _cpi_msg("full_refusal"))
    with pytest.raises(JudgingError) as e:
        judge._select_org("highprio", cause="rate_limit")
    msg = str(e.value)
    assert "2026-08-21" in msg and "D17" in msg, (
        "the refusal must name the rule so the reader learns WHY, not just "
        "that it broke")
    assert judge._clients["highprio"].messages.calls == []


def test_select_org_rate_limit_may_only_stay_put(tmp_path):
    judge = _judge(tmp_path, lambda p: _cpi_msg("full_refusal"),
                   lambda p: _cpi_msg("full_refusal"))
    assert judge._select_org("lowprio", cause="rate_limit") == "lowprio"


def test_select_org_preserves_deliberate_highprio_routing(tmp_path):
    """The guard keys on the CAUSE, not the org: a run explicitly
    constructed on High-Prio still works end to end."""
    judge = _judge(tmp_path, lambda p: _cpi_msg("full_refusal"),
                   lambda p: _cpi_msg("full_refusal"), org="highprio")
    assert judge._select_org("highprio", cause="deliberate") == "highprio"
    out = judge.judge_all([(_single_req("synthetic prompt", "no."),
                            ORBENCH_RUBRIC_V1)])
    (key,) = out
    assert judge._clients["highprio"].messages.calls, "no traffic flowed"
    assert judge._clients["lowprio"].messages.calls == []
    assert judge.cache.get_org(key) == "highprio"


def test_select_org_refuses_any_midrun_switch_whatever_the_label(tmp_path):
    """Mislabeling the hop 'deliberate' does not get around the guard:
    ONE org per run, pinned at construction."""
    judge = _judge(tmp_path, lambda p: _cpi_msg("full_refusal"),
                   lambda p: _cpi_msg("full_refusal"))
    with pytest.raises(JudgingError, match="ONE org per run"):
        judge._select_org("highprio", cause="deliberate")


def test_select_org_unknown_cause_raises(tmp_path):
    """The seam is total: a call site that will not name its cause cannot
    select an org at all."""
    judge = _judge(tmp_path, lambda p: _cpi_msg("full_refusal"),
                   lambda p: _cpi_msg("full_refusal"))
    with pytest.raises(JudgingError, match="unknown org-selection cause"):
        judge._select_org("lowprio", cause="because")


def test_client_refuses_a_foreign_org(tmp_path):
    """The backstop under the seam: even a caller that skips _select_org
    cannot get a client for an org the run is not pinned to."""
    judge = _judge(tmp_path, lambda p: _cpi_msg("full_refusal"),
                   lambda p: _cpi_msg("full_refusal"))
    with pytest.raises(JudgingError, match="pinned to"):
        judge._client("highprio")
