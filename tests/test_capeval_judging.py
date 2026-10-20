"""Judging spine tests.

No network, no GPU, no real Claude: transports are stubs. The golden-sha
tests are the rubric PRE-REGISTRATION gate: they were
committed BEFORE any control generation or judging ran, so any rubric
edit is a loud, versioned event — a changed sha here means a NEW rubric
version, never an in-place fix.
"""

import os
import stat

import pytest

from sisteer.capeval import judging as j
from sisteer.capeval.judging import (
    ALPACA_PAIR_RUBRIC_V1,
    ORBENCH_RUBRIC_V1,
    XSTEST_RUBRIC_V1,
    JudgeCache,
    JudgeRequest,
    JudgingError,
    Rubric,
    ClaudeJudge,
    coordination_post,
    measure_token_bounds,
    protocol_sha,
    render_request,
)

# --- golden shas (pre-registration pins) ------------------------------------

GOLDEN_RUBRIC_SHAS = {
    ("or_bench_cpi", "v1"): "15f29334703c31a8",
    ("xstest_cpi", "v1"): "288e8c08b0c7bbfa",
    ("alpaca_pair", "v1"): "2eb6e409cabce3ca",
}
# The wrapper hash at the frozen defaults (no sampling params — removed on
# claude-sonnet-5; thinking disabled; max_tokens 1024; judged_protocol_v1).
# or_bench/xstest share a tool schema, hence a protocol sha; the pair not
# colliding with alpaca is itself pinned.
GOLDEN_PROTOCOL_SHAS = {
    "or_bench_cpi": "b8d28441cf71caec",
    "xstest_cpi": "b8d28441cf71caec",
    "alpaca_pair": "3d45b24d5f57fe8f",
}


def test_rubric_shas_are_golden():
    for rubric in j.RUBRICS:
        assert rubric.sha == GOLDEN_RUBRIC_SHAS[(rubric.name, rubric.version)], (
            f"rubric {rubric.name}/{rubric.version} sha drifted — rubric "
            "text/schema is FROZEN after pre-registration (r1 #10); a "
            "deliberate change must be a NEW version + a decisions.md entry, "
            "and anything judged under the old sha is invalidated")


def test_protocol_shas_are_golden():
    for rubric in j.RUBRICS:
        assert protocol_sha(rubric) == GOLDEN_PROTOCOL_SHAS[rubric.name], (
            f"protocol sha for {rubric.name} drifted — the request wrapper "
            "(system framing / tool_choice / temperature / max_tokens / "
            "protocol code version) changed; that re-keys the shared cache "
            "and must be deliberate (r2 #3)")


def test_rubric_shas_are_distinct_and_content_sensitive():
    shas = {r.sha for r in j.RUBRICS}
    assert len(shas) == len(j.RUBRICS)
    edited = Rubric(name=ORBENCH_RUBRIC_V1.name,
                    version=ORBENCH_RUBRIC_V1.version,
                    text=ORBENCH_RUBRIC_V1.text + " ",
                    tool_schema=ORBENCH_RUBRIC_V1.tool_schema)
    assert edited.sha != ORBENCH_RUBRIC_V1.sha
    bumped = Rubric(name=ORBENCH_RUBRIC_V1.name, version="v2",
                    text=ORBENCH_RUBRIC_V1.text,
                    tool_schema=ORBENCH_RUBRIC_V1.tool_schema)
    assert bumped.sha != ORBENCH_RUBRIC_V1.sha


# --- JudgeRequest canonical key (r1 #5/#20) ----------------------------------


def _pair_req(a, b, model=j.JUDGE_MODEL):
    return JudgeRequest(prompt="synthetic instruction", candidate_a=a,
                        candidate_b=b, roles=("A", "B"),
                        rubric_sha=ALPACA_PAIR_RUBRIC_V1.sha,
                        judge_model=model,
                        protocol_sha=protocol_sha(ALPACA_PAIR_RUBRIC_V1))


def test_swapped_pair_is_a_different_key():
    ab = _pair_req("alpha text", "beta text")
    ba = _pair_req("beta text", "alpha text")
    assert ab.key() != ba.key()


def test_identical_candidate_swap_collapses_to_one_key():
    """Byte-identical candidates (the (base, base) tie canary): the
    swapped order renders the SAME effective request, so it is the SAME
    key — a cache hit by construction (r1 #20). Roles record the labels
    AS PRESENTED, never arm/base attribution."""
    ab = _pair_req("same text", "same text")
    ba = _pair_req("same text", "same text")
    assert ab.key() == ba.key()


def test_key_sensitive_to_every_component():
    base = _pair_req("alpha", "beta")
    assert _pair_req("alpha!", "beta").key() != base.key()
    other_model = _pair_req("alpha", "beta", model="claude-other")
    assert other_model.key() != base.key()
    diff_rubric = JudgeRequest(
        prompt="synthetic instruction", candidate_a="alpha",
        candidate_b="beta", roles=("A", "B"),
        rubric_sha="0" * 16, judge_model=j.JUDGE_MODEL,
        protocol_sha=protocol_sha(ALPACA_PAIR_RUBRIC_V1))
    assert diff_rubric.key() != base.key()
    diff_proto = JudgeRequest(
        prompt="synthetic instruction", candidate_a="alpha",
        candidate_b="beta", roles=("A", "B"),
        rubric_sha=ALPACA_PAIR_RUBRIC_V1.sha, judge_model=j.JUDGE_MODEL,
        protocol_sha="f" * 16)
    assert diff_proto.key() != base.key()
    diff_roles = JudgeRequest(
        prompt="synthetic instruction", candidate_a="alpha",
        candidate_b="beta", roles=("First", "Second"),
        rubric_sha=ALPACA_PAIR_RUBRIC_V1.sha, judge_model=j.JUDGE_MODEL,
        protocol_sha=protocol_sha(ALPACA_PAIR_RUBRIC_V1))
    assert diff_roles.key() != base.key()


def test_request_shape_refusals():
    with pytest.raises(JudgingError, match="both"):
        JudgeRequest(prompt="p", candidate_a="a", candidate_b="b", roles=None,
                     rubric_sha="x", judge_model="m", protocol_sha="y")
    with pytest.raises(JudgingError, match="both"):
        JudgeRequest(prompt="p", candidate_a="a", candidate_b=None,
                     roles=("A", "B"), rubric_sha="x", judge_model="m",
                     protocol_sha="y")
    with pytest.raises(JudgingError, match="mandatory"):
        JudgeRequest(prompt="p", candidate_a="a", candidate_b=None,
                     roles=None, rubric_sha="x", judge_model="m",
                     protocol_sha="")


# --- protocol_sha sensitivity (r2 #3) ----------------------------------------


def test_protocol_sha_sensitivity_each_wrapper_dimension_rekeys():
    base = protocol_sha(ORBENCH_RUBRIC_V1)
    assert protocol_sha(ORBENCH_RUBRIC_V1, system="other framing") != base
    assert protocol_sha(ORBENCH_RUBRIC_V1, temperature=0.1) != base
    assert protocol_sha(ORBENCH_RUBRIC_V1, max_tokens=2048) != base
    assert protocol_sha(ORBENCH_RUBRIC_V1,
                        thinking={"type": "adaptive"}) != base
    assert protocol_sha(ORBENCH_RUBRIC_V1, code_version="judged_protocol_v2") != base
    # tool schema + tool_choice: a renamed tool changes both
    renamed_schema = dict(ORBENCH_RUBRIC_V1.tool_schema, name="other_tool")
    renamed = Rubric(name=ORBENCH_RUBRIC_V1.name,
                     version=ORBENCH_RUBRIC_V1.version,
                     text=ORBENCH_RUBRIC_V1.text, tool_schema=renamed_schema)
    assert protocol_sha(renamed) != base
    # and rubric CONTENT (the text) stays protocol-invariant: rubric_sha
    # covers it, protocol_sha does not
    text_edit = Rubric(name=ORBENCH_RUBRIC_V1.name,
                       version=ORBENCH_RUBRIC_V1.version,
                       text=ORBENCH_RUBRIC_V1.text + " ",
                       tool_schema=ORBENCH_RUBRIC_V1.tool_schema)
    assert protocol_sha(text_edit) == base
    assert text_edit.sha != ORBENCH_RUBRIC_V1.sha


# --- JudgeCache ----------------------------------------------------------------


def test_cache_hit_miss_put_and_org(tmp_path):
    cache = JudgeCache(tmp_path / "state" / "judge_cache.sqlite")
    assert cache.get("k1") is None
    cache.put("k1", {"label": "full_refusal"}, org_used="lowprio")
    assert cache.get("k1") == {"label": "full_refusal"}
    assert cache.get_org("k1") == "lowprio"
    # write-once (2026-09-09): an agreeing repeat put is a no-op — the FIRST
    # producer's stamp stands, so the row's provenance never drifts to a
    # later caller (test_judge_cache_write_once.py has the disagreeing case)
    assert cache.put("k1", {"label": "full_refusal"}, org_used="highprio") == {"label": "full_refusal"}
    assert cache.get_org("k1") == "lowprio"
    assert len(cache) == 1
    # the cache dir is private (unsafe-prompt keys are hashes, but 0700
    # anyway)
    mode = stat.S_IMODE(os.stat(tmp_path / "state").st_mode)
    assert mode == 0o700


def test_cache_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv(j.CACHE_ENV_VAR, str(tmp_path / "over" / "c.sqlite"))
    cache = JudgeCache()
    assert cache.path == tmp_path / "over" / "c.sqlite"


def test_cache_refuses_free_text(tmp_path):
    cache = JudgeCache(tmp_path / "c.sqlite")
    with pytest.raises(JudgingError, match="JB4"):
        cache.put("k", {"reasoning": "the model said something long"},
                  org_used="lowprio")
    with pytest.raises(JudgingError, match="JB4"):
        cache.put("k", {"label": "x" * 65}, org_used="lowprio")


# --- ClaudeJudge with stub transports ------------------------------------------


class _ToolUse:
    type = "tool_use"

    def __init__(self, name, tool_input):
        self.name = name
        self.input = tool_input


class _Msg:
    stop_reason = "tool_use"

    def __init__(self, blocks):
        self.content = blocks


class _Messages:
    def __init__(self, fn):
        self._fn = fn
        self.calls = []

    def create(self, **params):
        self.calls.append(params)
        return self._fn(params)


class _Client:
    def __init__(self, fn):
        self.messages = _Messages(fn)


class _StatusError(Exception):
    def __init__(self, status):
        super().__init__(str(status))
        self.status_code = status


def _cpi_msg(label):
    return _Msg([_ToolUse("compliance_rating",
                          {"reasoning": "synthetic reasoning", "label": label})])


def _single_req(prompt, response):
    return JudgeRequest(prompt=prompt, candidate_a=response, candidate_b=None,
                        roles=None, rubric_sha=ORBENCH_RUBRIC_V1.sha,
                        judge_model=j.JUDGE_MODEL,
                        protocol_sha=protocol_sha(ORBENCH_RUBRIC_V1))


def _judge(tmp_path, low_fn, high_fn=None, **kw):
    clients = {"lowprio": _Client(low_fn)}
    if high_fn is not None:
        clients["highprio"] = _Client(high_fn)
    return ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"),
                       clients=clients, concurrency=4, **kw)


def _credential_error(status: int = 401):
    """The SDK's credential rejection as the API returns it (401 authentication_error / 403 permission_error): body
    shape only, no key anywhere."""
    import anthropic
    import httpx
    req = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    if status == 401:
        body = {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}
        return anthropic.AuthenticationError(f"Error code: 401 - {body}", response=httpx.Response(401, request=req), body=body)
    body = {"type": "error", "error": {"type": "permission_error", "message": "synthetic"}}
    return anthropic.PermissionDeniedError(f"Error code: 403 - {body}", response=httpx.Response(403, request=req), body=body)


def _raiser(exc_factory):
    def _fn(params):
        raise exc_factory()
    return _fn


def test_judge_all_stops_at_the_first_credential_rejection_and_latches_the_shared_client(tmp_path, capsys):
    """Auth is per client, so the FIRST 401 is the run's failure — _create raises the typed JudgeCredentialRejected (a
    JudgingError AND the vault's rejection) after that one call, never a retry; judge_all stops the batch at once (at
    most one call per in-flight worker for 20 requests, the queue cancelled, nothing cached) and names the org, the
    status and the error type; the latch rides the SHARED client, so a second judge over it (served_capability_judge:
    one judge per cell, one client) and the liveness canary refuse before their first call. A 403 is the same kind; a
    404 stays the hard failure it always was. See grading_diagnosis.md (2026-10-11)."""
    judge = _judge(tmp_path, _raiser(lambda: _credential_error(401)))
    client = judge._clients["lowprio"]
    pairs = [(_single_req(f"synthetic prompt {i}", "no."), ORBENCH_RUBRIC_V1) for i in range(20)]
    with pytest.raises(j.JudgeCredentialRejected) as ei:
        judge.judge_all(pairs)
    e = ei.value
    assert isinstance(e, JudgingError) and isinstance(e, j.vault.JudgeCredentialRejected)
    assert (e.pool, e.status, e.error_type, e.api_error_type) == ("lowprio", 401, "AuthenticationError", "authentication_error")
    assert 1 <= len(client.messages.calls) <= 4, len(client.messages.calls)          # never 20
    assert judge._credential_rejected is e and getattr(client, j.CREDENTIAL_LATCH_ATTR) is e
    assert judge.n_api_calls == 0 and len(judge.cache) == 0
    assert "401" in str(e) and "AuthenticationError" in str(e) and "lowprio" in str(e) and "x-api-key" not in str(e)
    out = capsys.readouterr().out
    assert "CREDENTIAL REJECTED on lowprio" in out and "NOT issued (D6)" in out and "synthetic prompt" not in out
    n_before = len(client.messages.calls)
    # the same judge again, a second judge over the same client, the canary: no wire call after the latch
    with pytest.raises(j.JudgeCredentialRejected):
        judge.judge_all(pairs[:3])
    judge2 = ClaudeJudge(cache=JudgeCache(tmp_path / "c2.sqlite"), clients={"lowprio": client}, concurrency=4)
    with pytest.raises(j.JudgeCredentialRejected):
        judge2.judge_all(pairs[:3])
    # the canary over the latched client: the run's typed rejection, never "not serving" (the WIF driver sweep, 2026-10-13 --
    # before it, a rejection met at the canary was swallowed into "any failure = not shown serving"); still no wire call
    with pytest.raises(j.JudgeCredentialRejected) as eic:
        judge._liveness_canary()
    assert eic.value is e
    assert len(client.messages.calls) == n_before
    # 403: the same kind, from one call
    (tmp_path / "b").mkdir()
    judge3 = _judge(tmp_path / "b", _raiser(lambda: _credential_error(403)))
    with pytest.raises(j.JudgeCredentialRejected) as ei3:
        judge3.judge_all(pairs[:5])
    assert (ei3.value.status, ei3.value.error_type) == (403, "PermissionDeniedError") and len(judge3._clients["lowprio"].messages.calls) <= 4
    # 404: a deterministic hard failure per request, as before — one histogram at the end, every request issued
    (tmp_path / "c").mkdir()
    judge4 = _judge(tmp_path / "c", _raiser(lambda: _StatusError(404)))
    with pytest.raises(JudgingError) as ei4:
        judge4.judge_all(pairs[:5])
    assert not isinstance(ei4.value, j.JudgeCredentialRejected) and "Failure types" in str(ei4.value)
    assert len(judge4._clients["lowprio"].messages.calls) == 5 and judge4._credential_rejected is None


def test_the_first_credential_rejection_filed_is_the_one_every_worker_raises(tmp_path):
    """Round 3 on 545c56eda, surfaced by the test above on a 4-CPU Slurm node: _latch_credential_rejection minted a NEW
    JudgeCredentialRejected per 401 and the last writer won the latch, so two workers meeting a 401 in the same instant raised
    different objects and judge_all propagated one while the latch held the other -- a code tie (a check-free set), not a flake.
    Now the latch is a check-and-set: the first rejection filed is THE run's rejection; a later one -- even of another status --
    returns the held object; a sibling judge over the shared client adopts it rather than minting its own; a non-rejection
    latches nothing; and four workers released by a barrier into a 401 at once raise the one object the latch holds."""
    import threading
    judge = _judge(tmp_path, _raiser(lambda: _credential_error(401)))
    client = judge._clients["lowprio"]
    first = judge._latch_credential_rejection("lowprio", _credential_error(401))
    second = judge._latch_credential_rejection("lowprio", _credential_error(403))      # a later worker, even a different status
    assert first is not None and second is first and first.status == 401               # the first filed, never the last
    assert judge._credential_rejected is first and getattr(client, j.CREDENTIAL_LATCH_ATTR) is first
    judge2 = ClaudeJudge(cache=JudgeCache(tmp_path / "c2.sqlite"), clients={"lowprio": client}, concurrency=4)
    assert judge2._latch_credential_rejection("lowprio", _credential_error(401)) is first and judge2._credential_rejected is first
    (tmp_path / "d").mkdir()
    judge3 = _judge(tmp_path / "d", _raiser(lambda: _StatusError(404)))
    assert judge3._latch_credential_rejection("lowprio", _StatusError(404)) is None and judge3._credential_rejected is None
    # under contention: the gate opens at the full width (AdaptiveGate(initial)), so four requests at concurrency 4 are four
    # workers in flight; the barrier releases them into the 401 together (best effort -- a broken barrier still raises the 401)
    barrier = threading.Barrier(4)

    def rejecting(params):
        try:
            barrier.wait(timeout=10)
        except threading.BrokenBarrierError:
            pass
        raise _credential_error(401)

    (tmp_path / "e").mkdir()
    judge4 = _judge(tmp_path / "e", rejecting)
    pairs = [(_single_req(f"synthetic prompt {i}", "no."), ORBENCH_RUBRIC_V1) for i in range(4)]
    with pytest.raises(j.JudgeCredentialRejected) as ei:
        judge4.judge_all(pairs)
    assert judge4._credential_rejected is ei.value and getattr(judge4._clients["lowprio"], j.CREDENTIAL_LATCH_ATTR) is ei.value
    assert len(judge4._clients["lowprio"].messages.calls) <= 4 and judge4.n_api_calls == 0


def test_judge_all_cache_first_and_replay_zero_transport(tmp_path):
    judge = _judge(tmp_path, lambda p: _cpi_msg("full_refusal"))
    pairs = [(_single_req("synthetic prompt one", "no."), ORBENCH_RUBRIC_V1),
             (_single_req("synthetic prompt two", "sure: ..."), ORBENCH_RUBRIC_V1)]
    out = judge.judge_all(pairs)
    assert len(out) == 2
    assert judge.n_api_calls == 2 and judge.n_cache_hits == 0
    assert all(v == {"label": "full_refusal"} for v in out.values())
    # reasoning is DROPPED before caching (JB4)
    for key in out:
        assert "reasoning" not in judge.cache.get(key)
        assert judge.cache.get_org(key) == "lowprio"

    # warm-cache replay through a FRESH judge: nonzero hits, ZERO
    # transport calls (r1 #9 — the acceptance cache demo's unit form)
    replay_low = _Messages(lambda p: (_ for _ in ()).throw(
        AssertionError("transport must not be reached on a warm cache")))
    fresh = ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"),
                        clients={"lowprio": type("C", (), {"messages": replay_low})()},
                        concurrency=4)
    out2 = fresh.judge_all(pairs)
    assert out2 == out
    assert fresh.n_cache_hits == 2 and fresh.n_api_calls == 0
    assert replay_low.calls == []


def test_judge_429_throttles_down_and_never_switches_org(tmp_path, monkeypatch):
    """SUPERSEDES test_judge_429_falls_back_to_highprio_and_logs_org.

    An earlier policy sent a rate-limited request to High-Prio. The rule
    of 2026-08-21 replaced that and said not to reinstate it: at 200-500
    concurrent, a saturated Low-Prio pool is most likely our own doing, and
    moving the load to a second shared pool makes it another user's 429
    rather than easing anything. A 429 now backs off on the SAME org and
    cuts our in-flight ceiling. Full policy coverage lives in
    tests/test_judge_no_org_hop.py.
    """
    monkeypatch.setattr(j, "_429_SLEEP_S", 0)
    monkeypatch.setattr(j, "_429_MAX_SLEEP_S", 0)

    def low(_params):
        raise _StatusError(429)

    def high(_params):
        raise AssertionError("a 429 must never reach the High-Prio org")

    # built directly, not via _judge: that helper pins concurrency=4, and the
    # throttle floor is 8, so a cut would be invisible at that width.
    judge = ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"),
                        clients={"lowprio": _Client(low),
                                 "highprio": _Client(high)},
                        concurrency=64)
    req = _single_req("synthetic prompt", "answer text")
    with pytest.raises(JudgingError):
        judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert judge.n_calls_by_org == {"lowprio": 0, "highprio": 0}
    assert judge._live_concurrency < 64      # throttled, not relocated


def test_judge_529_backs_off_same_org(tmp_path, monkeypatch):
    monkeypatch.setattr(j, "_529_SLEEP_S", 0)
    monkeypatch.setattr(j, "_529_MAX_SLEEP_S", 0)
    state = {"n": 0}

    def low(_params):
        state["n"] += 1
        if state["n"] == 1:
            raise _StatusError(529)
        return _cpi_msg("partial_refusal")

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic prompt", "hedged answer")
    out = judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert out[req.key()] == {"label": "partial_refusal"}
    assert judge.n_calls_by_org["lowprio"] == 1  # retried on the SAME org
    assert state["n"] == 2


def test_judge_raises_loudly_on_unverdicted(tmp_path):
    def low(params):
        body = params["messages"][0]["content"]
        if "poison" in body:
            return _Msg([])  # no tool_use block -> unverdicted
        return _cpi_msg("full_compliance")

    judge = _judge(tmp_path, low)
    pairs = [(_single_req("fine prompt", "ok"), ORBENCH_RUBRIC_V1),
             (_single_req("poison prompt", "??"), ORBENCH_RUBRIC_V1)]
    with pytest.raises(JudgingError, match="unverdicted"):
        judge.judge_all(pairs)
    # the failure message never carries prompt/response text
    try:
        judge.judge_all(pairs)
    except JudgingError as e:
        assert "poison" not in str(e)


def test_judge_bad_enum_is_unverdicted_not_cached(tmp_path):
    judge = _judge(tmp_path, lambda p: _cpi_msg("not_a_label"))
    req = _single_req("synthetic", "text")
    with pytest.raises(JudgingError, match="unverdicted"):
        judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert judge.cache.get(req.key()) is None


def test_judge_client_ignores_the_session_gateway_env(tmp_path,
                                                      monkeypatch):
    """A Claude Code session exports ANTHROPIC_BASE_URL at a
    session-local gateway; the SDK would inherit it and silently reroute
    judge traffic off the org pool the run named (observed to hang,
    2026-08-28). The real client pins the API base_url; the deliberate
    override is CAPEVAL_JUDGE_BASE_URL."""
    monkeypatch.setenv(j.POOL_SECRETS["lowprio"][0], "ORG-KEY-LOWPRIO")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:8787/t/xyz")
    monkeypatch.delenv(j.JUDGE_BASE_URL_ENV, raising=False)
    judge = ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"))
    client = judge._client("lowprio")  # real anthropic client, no network
    assert str(client.base_url).startswith(j.DEFAULT_JUDGE_BASE_URL)
    monkeypatch.setenv(j.JUDGE_BASE_URL_ENV, "https://example.test")
    fresh = ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"))
    assert str(fresh._client("lowprio").base_url).startswith(
        "https://example.test")


def test_judge_max_tokens_truncation_abstains_with_its_own_reason(tmp_path):
    """stop_reason=max_tokens + an unparseable forced-tool verdict = the
    judge deterministically burned its budget mid-reasoning (seen on a
    character-level case-puzzle item, both alpaca orders, 2026-08-28).
    Same abstention class as a refusal — bounded over by the caller,
    cached for stable bounds — NEVER the same label: the sentinel's
    ``reason`` says which. A max_tokens stop whose verdict still parses
    stays a real verdict."""
    def low(params):
        body = params["messages"][0]["content"]
        if "puzzle" in body:
            m = _Msg([_ToolUse("compliance_rating", {})])  # truncated JSON
            m.stop_reason = "max_tokens"
            return m
        m = _cpi_msg("full_compliance")
        m.stop_reason = "max_tokens"  # complete verdict, budget exhausted
        return m

    judge = _judge(tmp_path, low)
    trunc = _single_req("synthetic case puzzle prompt", "aLtErNaTiNg")
    fine = _single_req("plain synthetic prompt", "an answer")
    out = judge.judge_all([(trunc, ORBENCH_RUBRIC_V1),
                           (fine, ORBENCH_RUBRIC_V1)],
                          allow_refusal_abstention=True)
    assert out[trunc.key()] == {"abstained": True, "reason": "max_tokens"}
    assert out[fine.key()] == {"label": "full_compliance"}
    assert judge.n_abstained == 1
    # cached WITH the reason (abstention sentinels cache for stable bounds)
    assert judge.cache.get(trunc.key()) == {"abstained": True,
                                            "reason": "max_tokens"}


def test_judge_refusal_sentinel_carries_reason(tmp_path):
    """The refusal abstention sentinel names its reason too, so callers
    can always split refusals from truncations without special-casing."""
    def low(params):
        if "borderline" in params["messages"][0]["content"]:
            m = _Msg([])
            m.stop_reason = "refusal"
            return m
        return _cpi_msg("full_compliance")

    judge = _judge(tmp_path, low)
    refused = _single_req("borderline synthetic prompt", "a hedged answer")
    out = judge.judge_all([(refused, ORBENCH_RUBRIC_V1)],
                          allow_refusal_abstention=True)
    assert out[refused.key()] == {"abstained": True, "reason": "refusal"}


def test_judge_max_tokens_truncation_without_optin_still_fails_loud(
        tmp_path):
    """Without allow_refusal_abstention the strict contract holds: a
    deterministic truncation is unverdicted and the pass raises."""
    def low(params):
        m = _Msg([_ToolUse("compliance_rating", {})])
        m.stop_reason = "max_tokens"
        return m

    judge = _judge(tmp_path, low)
    req = _single_req("synthetic case puzzle prompt", "aLtErNaTiNg")
    with pytest.raises(JudgingError, match="unverdicted"):
        judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert judge.cache.get(req.key()) is None


def test_judge_refuses_stale_protocol_sha(tmp_path):
    judge = _judge(tmp_path, lambda p: _cpi_msg("full_compliance"))
    stale = JudgeRequest(prompt="p", candidate_a="r", candidate_b=None,
                         roles=None, rubric_sha=ORBENCH_RUBRIC_V1.sha,
                         judge_model=j.JUDGE_MODEL, protocol_sha="a" * 16)
    with pytest.raises(JudgingError, match="protocol"):
        judge.judge_all([(stale, ORBENCH_RUBRIC_V1)])


def test_render_request_shapes():
    req = _single_req("the prompt", "the response")
    params = render_request(ORBENCH_RUBRIC_V1, req)
    # claude-sonnet-5 rejects sampling params — the wrapper sends none,
    # and thinking is explicitly disabled (both verified live 2026-08-19)
    assert "temperature" not in params
    assert params["thinking"] == {"type": "disabled"}
    assert params["tool_choice"] == {"type": "tool", "name": "compliance_rating"}
    assert params["system"] == j.SYSTEM_FRAMING
    assert "the prompt" in params["messages"][0]["content"]
    assert "the response" in params["messages"][0]["content"]
    pair = _pair_req("resp A text", "resp B text")
    pp = render_request(ALPACA_PAIR_RUBRIC_V1, pair)
    body = pp["messages"][0]["content"]
    assert body.index("resp A text") < body.index("resp B text")
    single_on_pair = JudgeRequest(
        prompt="p", candidate_a="r", candidate_b=None, roles=None,
        rubric_sha=ALPACA_PAIR_RUBRIC_V1.sha, judge_model=j.JUDGE_MODEL,
        protocol_sha=protocol_sha(ALPACA_PAIR_RUBRIC_V1))
    with pytest.raises(JudgingError, match="paired"):
        render_request(ALPACA_PAIR_RUBRIC_V1, single_on_pair)
    with pytest.raises(JudgingError, match="rubric_sha"):
        render_request(ALPACA_PAIR_RUBRIC_V1, req)  # wrong rubric for request


# --- token bounds + coordination post ----------------------------------------


def test_token_bounds_offline_arithmetic():
    pairs = [(_single_req("aaaa", "bbbb"), ORBENCH_RUBRIC_V1),
             (_single_req("cccc", "dddd"), ORBENCH_RUBRIC_V1),
             (_single_req("aaaa", "bbbb"), ORBENCH_RUBRIC_V1)]  # dup deduped
    bounds = measure_token_bounds(pairs, max_tokens=1024)
    assert bounds["n_unique_requests"] == 2
    assert bounds["attempts"] == 3  # 1 + max_retries=2, retry-adjusted
    assert bounds["input_tokens_bound"] == bounds["input_tokens_once"] * 3
    assert bounds["output_tokens_bound"] == 1024 * 2 * 3
    assert "CONSERVATIVE" in bounds["input_bound_method"]
    # the byte bound really is bytes/2.5-ish: recompute one rendered request
    from sisteer.capeval.manifest import canonical_json
    rendered = render_request(ORBENCH_RUBRIC_V1, pairs[0][0])
    per = int(len(canonical_json(rendered).encode()) / 2.5) + 1
    assert bounds["input_tokens_once"] >= per


def test_token_bounds_count_tokens_endpoint_path():
    class _CountMessages:
        def count_tokens(self, **kw):
            assert kw["model"] == j.JUDGE_MODEL and kw["tools"]
            return type("R", (), {"input_tokens": 100})()

    client = type("C", (), {"messages": _CountMessages()})()
    pairs = [(_single_req("aaaa", "bbbb"), ORBENCH_RUBRIC_V1),
             (_single_req("cccc", "dddd"), ORBENCH_RUBRIC_V1)]
    bounds = measure_token_bounds(pairs, client=client, max_tokens=512)
    assert bounds["input_tokens_once"] == 200
    assert bounds["input_tokens_bound"] == 600
    assert bounds["output_tokens_bound"] == 512 * 2 * 3
    assert "exact" in bounds["input_bound_method"]


def test_coordination_post_format():
    bounds = {"attempts": 3, "input_tokens_bound": 36_000_000,
              "output_tokens_bound": 18_000_000,
              "input_bound_method": "anthropic count-tokens endpoint (exact)",
              "max_tokens": 1024, "n_unique_requests": 5953}
    post = coordination_post(n_unique_requests=5953, bounds=bounds)
    # the channel convention: `<N> <model> <prio> <duration>` leads
    assert post.startswith("200 sonnet-5 lowprio ~1h")
    assert "5953 unique requests" in post
    assert "36,000,000" in post and "18,000,000" in post
    # provenance must describe what the code DOES (2026-08-21 rule): a 429
    # backs off on the same org and throttles down; the retired
    # "highprio fallback" line must never come back
    assert "highprio" not in post
    assert "same-org backoff" in post and "throttle down" in post
    assert "floor 50" in post  # derived from sisteer.throttle.DEFAULT_FLOOR


# --- per-run judge model: no cross-model cache contamination --------------------


def test_judge_model_is_per_run_and_keys_never_cross_contaminate(tmp_path):
    """The judge model is a constructor parameter, resolved into every
    request key — a sonnet-keyed verdict can never serve a haiku run."""
    def make_req(model):
        return JudgeRequest(prompt="synthetic prompt", candidate_a="reply",
                            candidate_b=None, roles=None,
                            rubric_sha=ORBENCH_RUBRIC_V1.sha,
                            judge_model=model,
                            protocol_sha=protocol_sha(ORBENCH_RUBRIC_V1))

    assert make_req("claude-sonnet-5").key() != make_req(
        "claude-haiku-4-5").key()

    cache = JudgeCache(tmp_path / "c.sqlite")
    sonnet = ClaudeJudge(cache=cache, model="claude-sonnet-5", concurrency=2,
                         clients={"lowprio": _Client(
                             lambda p: _cpi_msg("full_refusal"))})
    sonnet.judge_all([(make_req(sonnet.model), ORBENCH_RUBRIC_V1)])
    assert sonnet.n_api_calls == 1

    calls = []

    def haiku_transport(params):
        calls.append(params["model"])
        return _cpi_msg("full_compliance")

    haiku = ClaudeJudge(cache=cache, model="claude-haiku-4-5", concurrency=2,
                        clients={"lowprio": _Client(haiku_transport)})
    out = haiku.judge_all([(make_req(haiku.model), ORBENCH_RUBRIC_V1)])
    # the sonnet verdict was cached but MUST NOT serve the haiku run
    assert haiku.n_cache_hits == 0 and haiku.n_api_calls == 1
    assert calls == ["claude-haiku-4-5"]  # the wire request names the model
    assert list(out.values()) == [{"label": "full_compliance"}]


def test_judge_529_exhaustion_never_switches_org(tmp_path, monkeypatch):
    """529 is Anthropic-side capacity, not an org limit: backoff runs on
    the SAME org and exhaustion fails the request loudly. (429 no longer
    switches org either — see the 2026-08-21 rule; nothing switches org.)"""
    monkeypatch.setattr(j, "_529_SLEEP_S", 0)
    monkeypatch.setattr(j, "_529_MAX_SLEEP_S", 0)

    def low(_params):
        raise _StatusError(529)

    def high(_params):
        raise AssertionError("a 529 must never reach the High-Prio org")

    judge = _judge(tmp_path, low, high)
    req = _single_req("synthetic prompt", "reply")
    with pytest.raises(JudgingError, match="unverdicted"):
        judge.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert judge.n_calls_by_org == {"lowprio": 0, "highprio": 0}


# --------------------------------------------------------------------------- #
# The client's per-request timeout                                              #
# --------------------------------------------------------------------------- #


def _captured_client_kwargs(monkeypatch, tmp_path, *, backend, model, org):
    """Build a ClaudeJudge with the anthropic module stubbed and return the
    kwargs it passed to the client constructor."""
    import sys
    import types as _types

    seen = {}

    class _FakeClient:
        def __init__(self, **kw):
            seen.update(kw)

    fake = _types.ModuleType("anthropic")
    fake.Anthropic = _FakeClient
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    # the openrouter pool resolves its key through _pool_key; an Anthropic pool through the vault (op mode here:
    # the stubbed key, never the secret-manager CLI) — both stubbed so no credential resolves, whichever backend runs
    monkeypatch.setattr(j, "_pool_key", lambda pool: "k")
    monkeypatch.delenv(j.vault.SECRETS_MODE_ENV, raising=False)
    monkeypatch.setattr(j.vault, "anthropic_api_key", lambda org, **kw: "k")
    judge = ClaudeJudge(model=model, backend=backend,
                        cache=JudgeCache(tmp_path / "c.sqlite"))
    judge._client(org)
    return seen


@pytest.mark.parametrize("backend,model,org", [
    ("anthropic", j.JUDGE_MODEL, "lowprio"),
    ("openrouter", "x-ai/grok-4.6", "openrouter"),
])
def test_client_always_sets_an_explicit_request_timeout(monkeypatch, tmp_path,
                                                        backend, model, org):
    """Both constructors must bound the wire call.

    The SDK's default is 600 s against a ~1.8 s judge call — a 330x margin
    that cannot tell a slow call from a dead socket. MEASURED 2026-09-11: two
    sockets ESTAB with empty send/recv queues stalled a 5,953-item run, and
    because legs run serially the whole run waited behind them; the run's
    watchdog then read the silence as pool degradation and throttled 200->50,
    which is the wrong remedy for the wrong cause (both pools answered in
    1.8 s when probed at max_retries=0). `sisteer.vault` already set a timeout
    and documented this exact failure; THIS module did not, and that gap is
    what bit. Both backends are pinned because otherwise only one of the two
    client constructors is covered.
    """
    kw = _captured_client_kwargs(monkeypatch, tmp_path,
                                 backend=backend, model=model, org=org)
    assert "timeout" in kw, (
        f"{backend} client built with no explicit timeout — it inherits the "
        "SDK's 600 s default and a few hung sockets become a silently idle pool")
    assert 0 < float(kw["timeout"]) <= 300, (
        f"timeout {kw['timeout']!r} is not a usable bound")


def test_the_timeout_is_far_above_the_measured_median_latency():
    """~1.8 s measured; a bound near that would clip genuinely slow live calls.
    This is the other error direction, and a too-tight timeout is the classic
    over-correction after a hang."""
    assert j.REQUEST_TIMEOUT_S >= 36.0
