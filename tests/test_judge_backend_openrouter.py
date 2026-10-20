"""``backend="openrouter"`` — the judge's second backend (2026-09-02).

Why it exists: claude-sonnet-5 deterministically declines ~8% of a
constitution-steered Qwen OR-Bench cell's bodies (transport_refusal +
content_filter; 108 of 1319, over the 5% ceiling), so those cells cannot be
sonnet-graded; ``x-ai/grok-4.6`` on OpenRouter verdicts them (10/10 of a
probed refused set) and is the campaign's fallback judge. Until now that ran
through an in-process shim — runpy-wrapping the driver, monkeypatching
``_read_key`` and ``messages.create``, hand-rolling a "drop <=3 unverdicted"
tolerance. This is the shim as first-class code, MINUS the tolerance: main's
abstention classes replace it, and anything they do not name stays LOUD.

Observed live against OpenRouter's Anthropic-compatible skin
(``https://openrouter.ai/api`` + the SDK's ``/v1/messages``; a synthetic
body; 2026-09-02):

* success: 200, ``stop_reason=tool_use``, blocks ``[thinking,
  redacted_thinking, tool_use]``, the rubric's tool with its label — the
  forced tool_use survives the skin.
* ``thinking={"type": "disabled"}``: 400 ``invalid_request_error``
  "Reasoning is mandatory for this endpoint and cannot be disabled",
  ``error_type: invalid_request``.
* ``temperature=None`` (the SDK serializes an explicit None as null): 400
  "Invalid Anthropic Messages API request", ``metadata.raw`` names the field.
* an unknown model id: 400 ``invalid_request_error`` "<id> is not a valid
  model ID", ``error_type: invalid_request`` — a hard failure.
* a bad key: 401 ``authentication_error``, ``error_type: authentication``.
* ``max_tokens=5``: 200 with 148 output tokens — the cap is not enforced on
  this skin, so a ``stop_reason=max_tokens`` abstention is unlikely there.

The error ENVELOPE, every time: ``{"type": "error", "error": {"type":
<anthropic type>, "message": ..., "error_type": <OpenRouter typed code>},
"request_id": ..., "metadata": {...}}``. The DECLINE codes mapped here (400
``refusal`` / ``content_policy_violation`` / ``max_tokens_exceeded``; a 403
with moderation metadata) come from OpenRouter's errors reference and are
documented non-retryable; grok declined nothing we could provoke, so those
fixtures are docs-shaped codes inside the observed envelope — stated here,
not hidden. Anything else (401/402/403-without-moderation/404, an
``invalid_request`` 400) is a hard failure by construction.
"""
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import sisteer.capeval.judging as j                       # noqa: E402
from sisteer.capeval.judging import (                     # noqa: E402
    ClaudeJudge, JudgeCache, JudgeRequest, JudgingError, coordination_post,
    protocol_sha, render_request, required_backend, translate_request)

from test_capeval_judging import (                        # noqa: E402
    ORBENCH_RUBRIC_V1, _Client, _Msg, _StatusError, _ToolUse, _cpi_msg,
    _single_req)

GROK = "x-ai/grok-4.6"
SONNET = "claude-sonnet-5"


# --- fixtures ------------------------------------------------------------------


def _fast_ladder(monkeypatch):
    monkeypatch.setattr(j, "_429_SLEEP_S", 0)
    monkeypatch.setattr(j, "_429_MAX_SLEEP_S", 0)
    monkeypatch.setattr(j, "_529_SLEEP_S", 0)
    monkeypatch.setattr(j, "_529_MAX_SLEEP_S", 0)


def _grok_req(prompt, response, model=GROK):
    return JudgeRequest(prompt=prompt, candidate_a=response, candidate_b=None,
                        roles=None, rubric_sha=ORBENCH_RUBRIC_V1.sha,
                        judge_model=model,
                        protocol_sha=protocol_sha(ORBENCH_RUBRIC_V1))


def _or_judge(tmp_path, fn, **kw):
    return ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"), model=GROK,
                       backend="openrouter",
                       clients={"openrouter": _Client(fn)},
                       concurrency=kw.pop("concurrency", 4), **kw)


class _Thinking:
    type = "thinking"
    thinking = "synthetic reasoning"


class _RedactedThinking:
    type = "redacted_thinking"


def _grok_msg(label):
    """The observed success shape: thinking blocks AHEAD of the forced
    tool_use (grok's reasoning cannot be disabled on this skin)."""
    return _Msg([_Thinking(), _RedactedThinking(),
                 _ToolUse("compliance_rating",
                          {"reasoning": "synthetic", "label": label})])


class _OpenRouterError(_StatusError):
    """The SDK status error as raised against OpenRouter's Anthropic-
    compatible skin: the observed envelope, with OpenRouter's typed
    ``error_type`` beside Anthropic's ``type``/``message``."""

    def __init__(self, status, anthropic_type, message, error_type,
                 request_id="gen-1788400000-synthetic0001", metadata=None):
        super().__init__(status)
        self.request_id = request_id
        self.body = {"type": "error",
                     "error": {"type": anthropic_type, "message": message,
                               "error_type": error_type},
                     "request_id": request_id}
        if metadata is not None:
            self.body["metadata"] = metadata


def _decline_400(error_type, rid="gen-1788400000-decline0001"):
    return _OpenRouterError(400, "invalid_request_error",
                            f"synthetic {error_type}", error_type,
                            request_id=rid)


# --- backend selection: one per run, pinned, model-compatible --------------------


def test_backend_constants_and_defaults(tmp_path):
    assert j.JUDGE_BACKENDS == ("anthropic", "openrouter")
    assert j.DEFAULT_JUDGE_BACKEND == "anthropic"
    default = ClaudeJudge(cache=JudgeCache(tmp_path / "a.sqlite"),
                          clients={"lowprio": _Client(lambda p: None)})
    assert default.backend == "anthropic" and default.org == "lowprio"
    assert default.n_calls_by_org == {"lowprio": 0, "highprio": 0}
    orj = _or_judge(tmp_path, lambda p: None)
    assert orj.backend == "openrouter" and orj.org == j.OPENROUTER_POOL
    assert orj.n_calls_by_org == {"openrouter": 0}
    # the key convention: the vault (env var, else the secret-manager item),
    # never a file — every pool, including the two anthropic orgs
    assert set(j.POOL_SECRETS) == {"lowprio", "highprio", "openrouter"}
    assert j.POOL_SECRETS["openrouter"] == ("OPENROUTER_API_KEY",
                                            "OpenRouter API Key")
    assert not hasattr(j, "OPENROUTER_KEY_FILE")
    assert j.OPENROUTER_BASE_URL == "https://openrouter.ai/api"


def test_unknown_backend_and_crossed_pool_refused(tmp_path):
    cache = JudgeCache(tmp_path / "c.sqlite")
    with pytest.raises(ValueError, match="backend must be one of"):
        ClaudeJudge(cache=cache, model=GROK, backend="together")
    # openrouter has exactly one pool; the anthropic orgs are not it
    with pytest.raises(ValueError, match="org must be one of"):
        ClaudeJudge(cache=cache, model=GROK, backend="openrouter",
                    org="lowprio")
    with pytest.raises(ValueError, match="org must be one of"):
        ClaudeJudge(cache=cache, model=SONNET, backend="anthropic",
                    org="openrouter")


def test_required_backend_table():
    assert required_backend(GROK) == "openrouter"
    assert required_backend("grok-4.6") == "openrouter"       # any grok id
    assert required_backend("anthropic/claude-sonnet-4.6") == "openrouter"
    assert required_backend(SONNET) == "anthropic"
    assert required_backend("claude-haiku-4-5") == "anthropic"
    assert required_backend("some-other-judge") is None      # no claim
    assert required_backend("") is None


def test_model_backend_crossed_pairs_refused(tmp_path):
    """The model IMPLIES the backend in every cache/store key only because
    the crossed pairs cannot be constructed."""
    cache = JudgeCache(tmp_path / "c.sqlite")
    with pytest.raises(ValueError, match="requires backend 'openrouter'"):
        ClaudeJudge(cache=cache, model=GROK, backend="anthropic")
    with pytest.raises(ValueError, match="requires backend 'anthropic'"):
        ClaudeJudge(cache=cache, model=SONNET, backend="openrouter")
    # an id naming neither family takes the named backend as is
    ClaudeJudge(cache=cache, model="some-other-judge", backend="openrouter",
                clients={"openrouter": _Client(lambda p: None)})


def test_mid_run_pool_switch_refused(tmp_path):
    """The org discipline generalizes to pools: an openrouter run refuses
    every anthropic org whatever the claimed cause, and vice versa."""
    orj = _or_judge(tmp_path, lambda p: None)
    with pytest.raises(JudgingError, match="pinned"):
        orj._client("lowprio")
    with pytest.raises(JudgingError, match="mid-run refused"):
        orj._select_org("lowprio", cause="deliberate")
    with pytest.raises(JudgingError, match="REFUSED"):
        orj._select_org("highprio", cause="rate_limit")
    assert orj._select_org("openrouter", cause="rate_limit") == "openrouter"
    anth = ClaudeJudge(cache=JudgeCache(tmp_path / "a.sqlite"),
                       clients={"lowprio": _Client(lambda p: None)})
    with pytest.raises(JudgingError, match="pinned"):
        anth._client("openrouter")


# --- request shaping per backend ---------------------------------------------------


def test_translate_request_openrouter_drops_exactly_the_rejected_params():
    req = _grok_req("synthetic prompt", "synthetic answer")
    params = render_request(ORBENCH_RUBRIC_V1, req)
    assert "thinking" in params and "temperature" not in params  # the declared protocol
    wire = translate_request("openrouter", params)
    # dropped: thinking (400 "Reasoning is mandatory ... cannot be disabled")
    assert "thinking" not in wire
    # everything else passes through byte-for-byte — the forced verdict included
    assert set(wire) == set(params) - {"thinking"}
    for k in wire:
        assert wire[k] == params[k]
    assert wire["tool_choice"] == {"type": "tool", "name": "compliance_rating"}
    assert wire["model"] == GROK
    # never mutates its input
    assert "thinking" in params
    # an explicit None temperature (the SDK would send null -> 400) is dropped;
    # a real number is kept
    assert "temperature" not in translate_request(
        "openrouter", {**params, "temperature": None})
    assert translate_request(
        "openrouter", {**params, "temperature": 0.3})["temperature"] == 0.3


def test_translate_request_anthropic_is_the_identity():
    req = _single_req("synthetic prompt", "synthetic answer")
    params = render_request(ORBENCH_RUBRIC_V1, req)
    assert translate_request("anthropic", params) == params
    assert translate_request("anthropic", params)["thinking"] == {
        "type": "disabled"}
    with pytest.raises(JudgingError, match="unknown judge backend"):
        translate_request("together", params)


def test_translation_does_not_touch_protocol_sha():
    """The declared protocol is the same on both backends: the cache key's
    protocol_sha is model-independent, and the backend rides the MODEL
    (required_backend) plus the summary provenance, not the sha."""
    assert protocol_sha(ORBENCH_RUBRIC_V1) == "b8d28441cf71caec"  # golden


def test_openrouter_wire_params_and_parse_via_judge_all(tmp_path):
    orj = _or_judge(tmp_path, lambda p: _grok_msg("partial_refusal"))
    req = _grok_req("synthetic prompt", "a hedged answer")
    out = orj.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert out[req.key()] == {"label": "partial_refusal"}
    calls = orj._clients["openrouter"].messages.calls
    assert len(calls) == 1
    wire = calls[0]
    assert "thinking" not in wire and "temperature" not in wire
    assert wire["model"] == GROK
    assert wire["tool_choice"] == {"type": "tool", "name": "compliance_rating"}
    assert wire["tools"][0]["name"] == "compliance_rating"
    # the pool actually used is recorded per row, like the org
    assert orj.cache.get_org(req.key()) == "openrouter"
    assert orj.n_calls_by_org == {"openrouter": 1}
    assert orj.n_api_calls == 1


def test_canary_goes_through_the_same_translation(tmp_path):
    orj = _or_judge(tmp_path, lambda p: _Msg([]))
    assert orj._liveness_canary() is True
    (call,) = orj._clients["openrouter"].messages.calls
    assert "thinking" not in call and call["model"] == GROK
    assert orj.n_canary_calls == 1


# --- OpenRouter declines -> main's abstention classes -------------------------------


@pytest.mark.parametrize("error_type,reason", [
    ("content_policy_violation", "content_filter"),
    ("refusal", "refusal"),
    ("max_tokens_exceeded", "max_tokens"),
])
def test_typed_400_decline_files_as_an_abstention_once(tmp_path, error_type,
                                                       reason):
    seen = {"n": 0}

    def fn(params):
        seen["n"] += 1
        raise _decline_400(error_type, rid=f"gen-1788400000-{error_type[:8]}")

    orj = _or_judge(tmp_path, fn)
    req = _grok_req("synthetic prompt", "an answer")
    out = orj.judge_all([(req, ORBENCH_RUBRIC_V1)],
                        allow_refusal_abstention=True)
    assert out[req.key()] == {
        "abstained": True, "reason": reason,
        "signature": f"openrouter:400:{error_type}",
        "request_id": f"gen-1788400000-{error_type[:8]}"}
    assert seen["n"] == 1                       # deterministic: never retried
    assert orj.n_abstained == 1 and orj.n_transport_refusals == 0
    assert orj.n_canary_calls == 0
    # cached like any verdict, under the openrouter pool
    assert orj.cache.get(req.key())["reason"] == reason
    assert orj.cache.get_org(req.key()) == "openrouter"


def test_403_with_moderation_metadata_is_content_filter(tmp_path):
    seen = {"n": 0}

    def fn(params):
        seen["n"] += 1
        raise _OpenRouterError(
            403, "permission_error", "synthetic: input flagged",
            "permission_denied", request_id="gen-1788400000-moderation1",
            metadata={"reasons": ["synthetic_reason"],
                      "flagged_input": "synthetic...", "provider_name": "P",
                      "model_slug": GROK})

    orj = _or_judge(tmp_path, fn)
    req = _grok_req("synthetic prompt", "an answer")
    out = orj.judge_all([(req, ORBENCH_RUBRIC_V1)],
                        allow_refusal_abstention=True)
    assert out[req.key()] == {
        "abstained": True, "reason": "content_filter",
        "signature": "openrouter:403:moderation",
        "request_id": "gen-1788400000-moderation1"}
    assert seen["n"] == 1


def test_classifier_reads_every_documented_metadata_placement():
    meta = {"reasons": ["r"], "flagged_input": "x"}
    for placement in ("metadata", "openrouter_metadata"):
        e = _OpenRouterError(403, "permission_error", "m", "permission_denied")
        e.body[placement] = meta
        assert j._classify_openrouter_decline(e) == (
            "content_filter", "openrouter:403:moderation")
    e = _OpenRouterError(403, "permission_error", "m", "permission_denied")
    e.body["error"]["metadata"] = meta
    assert j._classify_openrouter_decline(e) == (
        "content_filter", "openrouter:403:moderation")


@pytest.mark.parametrize("exc,label", [
    # observed: an unknown model id is an invalid_request 400
    (_OpenRouterError(400, "invalid_request_error",
                      "x-ai/grok-does-not-exist-99 is not a valid model ID",
                      "invalid_request"), "unknown model"),
    # observed: a bad key
    (_OpenRouterError(401, "authentication_error", "User not found.",
                      "authentication", request_id=None), "bad key"),
    # documented: insufficient credits — must never spend a ladder on it
    (_OpenRouterError(402, "billing_error", "Insufficient credits",
                      "payment_required"), "402"),
    # a 403 WITHOUT moderation metadata is a permission error on the key
    (_OpenRouterError(403, "permission_error", "key lacks permission",
                      "permission_denied"), "bare 403"),
    (_OpenRouterError(404, "not_found_error", "no such resource",
                      "not_found"), "404"),
    # the Anthropic content-filter MESSAGE without OpenRouter's typed code
    # is not an OpenRouter signature — backend signatures never cross
    (_OpenRouterError(400, "invalid_request_error",
                      "Output blocked by content filtering policy",
                      "invalid_request"), "anthropic-shaped message"),
    # a 400 whose typed code is not a decline
    (_OpenRouterError(400, "invalid_request_error", "too long",
                      "context_length_exceeded"), "context_length_exceeded"),
])
def test_unmapped_statuses_stay_hard_failures_after_one_attempt(
        tmp_path, monkeypatch, exc, label):
    _fast_ladder(monkeypatch)
    seen = {"n": 0}

    def fn(params):
        seen["n"] += 1
        raise exc

    orj = _or_judge(tmp_path, fn)
    req = _grok_req("synthetic prompt", "an answer")
    with pytest.raises(JudgingError, match="unverdicted") as ei:
        orj.judge_all([(req, ORBENCH_RUBRIC_V1)],
                      allow_refusal_abstention=True)
    assert seen["n"] == 1, f"{label}: a deterministic 4xx must not be retried"
    assert orj.n_abstained == 0
    assert orj.cache.get(req.key()) is None      # nothing silently filed
    assert "_OpenRouterError" in str(ei.value)   # the failure TYPE is named


def test_strict_mode_raises_on_a_typed_decline(tmp_path):
    orj = _or_judge(tmp_path, lambda p: (_ for _ in ()).throw(
        _decline_400("refusal")))
    req = _grok_req("synthetic prompt", "an answer")
    with pytest.raises(JudgingError) as ei:
        orj.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert "JudgeApiDecline" in str(ei.value)
    assert orj.cache.get(req.key()) is None


def test_openrouter_signatures_are_not_consulted_on_anthropic(tmp_path):
    """A typed OpenRouter 400 cannot arrive from api.anthropic.com; if a
    body shaped like one did, the anthropic backend must not file it."""
    anth = ClaudeJudge(cache=JudgeCache(tmp_path / "a.sqlite"),
                       clients={"lowprio": _Client(lambda p: (
                           _ for _ in ()).throw(_decline_400("refusal")))},
                       concurrency=2)
    req = _single_req("synthetic prompt", "an answer")
    with pytest.raises(JudgingError, match="unverdicted"):
        anth.judge_all([(req, ORBENCH_RUBRIC_V1)],
                       allow_refusal_abstention=True)
    assert anth.n_abstained == 0


# --- capacity / transport: back off, then FAIL — never an abstention --------------


@pytest.mark.parametrize("status", [502, 503, 504, 529])
def test_capacity_status_runs_the_ladder_then_fails(tmp_path, monkeypatch,
                                                    status):
    _fast_ladder(monkeypatch)
    seen = {"n": 0}

    def fn(params):
        seen["n"] += 1
        raise _OpenRouterError(status, "api_error", "synthetic provider down",
                               "provider_unavailable")

    orj = _or_judge(tmp_path, fn)
    req = _grok_req("synthetic prompt", "an answer")
    with pytest.raises(JudgingError, match="unverdicted") as ei:
        orj.judge_all([(req, ORBENCH_RUBRIC_V1)],
                      allow_refusal_abstention=True)
    assert seen["n"] == j._529_MAX_ATTEMPTS      # the same ladder as 529
    assert "transport exhausted on openrouter" in str(ei.value)
    assert orj.n_abstained == 0 and orj.n_transport_refusals == 0


def test_all_529_ladder_on_openrouter_is_never_a_transport_refusal(
        tmp_path, monkeypatch):
    """The transport_refusal ruling's evidence is sonnet-5's per-body 529
    on the Anthropic API; nothing like it was observed on OpenRouter/grok,
    so an all-529 ladder there stays a hard failure EVEN WITH siblings
    succeeding in-window and the opt-in on — and no canary is spent."""
    _fast_ladder(monkeypatch)

    def fn(params):
        if "bad" in params["messages"][0]["content"]:
            raise _OpenRouterError(529, "overloaded_error", "overloaded",
                                   "provider_overloaded")
        return _grok_msg("full_compliance")

    orj = _or_judge(tmp_path, fn)
    good = _grok_req("synthetic prompt", "a good answer")
    bad = _grok_req("synthetic prompt", "a bad answer")
    with pytest.raises(JudgingError, match="1 of 2") as ei:
        orj.judge_all([(good, ORBENCH_RUBRIC_V1), (bad, ORBENCH_RUBRIC_V1)],
                      allow_refusal_abstention=True)
    assert orj.n_transport_refusals == 0 and orj.n_canary_calls == 0
    assert orj.cache.get(good.key()) == {"label": "full_compliance"}
    assert orj.cache.get(bad.key()) is None
    assert "transport exhausted" in str(ei.value)
    assert "anthropic" in j._TRANSPORT_REFUSAL_BACKENDS
    assert "openrouter" not in j._TRANSPORT_REFUSAL_BACKENDS


def test_429_throttles_down_on_the_same_pool(tmp_path, monkeypatch):
    _fast_ladder(monkeypatch)
    orj = _or_judge(tmp_path, lambda p: (_ for _ in ()).throw(
        _OpenRouterError(429, "rate_limit_error", "rate limited",
                         "rate_limit_exceeded")), concurrency=64)
    req = _grok_req("synthetic prompt", "an answer")
    with pytest.raises(JudgingError):
        orj.judge_all([(req, ORBENCH_RUBRIC_V1)])
    assert orj._live_concurrency < 64 and orj.n_rate_limited > 0
    assert orj.n_calls_by_org == {"openrouter": 0}


# --- cache discrimination -----------------------------------------------------------


def test_cache_keys_discriminate_backends_through_the_model(tmp_path):
    """A grok verdict can never serve a sonnet request or vice versa: the
    model is in every key, and the crossed model/backend pairs cannot be
    constructed (test_model_backend_crossed_pairs_refused), so the backend
    is implied by the model in the key."""
    body = ("synthetic prompt", "the same answer text")
    assert _grok_req(*body).key() != _single_req(*body).key()
    cache = JudgeCache(tmp_path / "shared.sqlite")
    sonnet = ClaudeJudge(cache=cache, model=SONNET, concurrency=2,
                         clients={"lowprio": _Client(
                             lambda p: _cpi_msg("full_refusal"))})
    sonnet.judge_all([(_single_req(*body), ORBENCH_RUBRIC_V1)])
    calls = []

    def grok(params):
        calls.append(params["model"])
        return _grok_msg("full_compliance")

    orj = ClaudeJudge(cache=cache, model=GROK, backend="openrouter",
                      concurrency=2, clients={"openrouter": _Client(grok)})
    out = orj.judge_all([(_grok_req(*body), ORBENCH_RUBRIC_V1)])
    assert orj.n_cache_hits == 0 and orj.n_api_calls == 1
    assert calls == [GROK]
    assert list(out.values()) == [{"label": "full_compliance"}]
    assert cache.get_org(_grok_req(*body).key()) == "openrouter"
    assert cache.get_org(_single_req(*body).key()) == "lowprio"


# --- the key: the vault, never a file ----------------------------------------------


def test_pool_key_comes_from_the_environment_or_the_vault(monkeypatch):
    """There is no key FILE. A pool resolves from its own env var, else its
    secret-manager item; a pool with neither raises a NAMED JudgingError that
    never carries the value."""
    monkeypatch.setenv(j.POOL_SECRETS["openrouter"][0], "OPENROUTER-KEY-VALUE")
    assert j._pool_key("openrouter") == "OPENROUTER-KEY-VALUE"

    monkeypatch.delenv(j.POOL_SECRETS["openrouter"][0], raising=False)

    def _boom(*a, **k):
        raise FileNotFoundError("fake-secret-cmd")
    monkeypatch.setattr(j.vault.subprocess, "check_output", _boom)
    with pytest.raises(JudgingError) as exc:
        j._pool_key("openrouter")
    assert "openrouter" in str(exc.value)
    assert "OPENROUTER-KEY-VALUE" not in str(exc.value)

    with pytest.raises(JudgingError, match="unknown judge pool"):
        j._pool_key("nope")


def test_the_module_reads_no_credential_from_disk():
    """A source-level pin, because the failure it guards against is a NEW
    call site rather than a behaviour change in an existing one: no key path
    and no file read may come back into this module (vault-only rule)."""
    src = pathlib.Path(j.__file__).read_text()
    for banned in ("~/.config/", "anthropic-key", "openrouter-key",
                   "_read_key("):
        assert banned not in src, f"{banned!r} is back in judging.py"


def test_real_client_is_built_from_the_vault_against_the_openrouter_url(
        tmp_path, monkeypatch):
    """No test seam: the client is constructed lazily from the VAULT-resolved
    key and the backend's base URL — the shell's ANTHROPIC_BASE_URL and the
    CAPEVAL_JUDGE_BASE_URL override (anthropic-only) are both ignored."""
    monkeypatch.setenv(j.POOL_SECRETS["openrouter"][0], "OPENROUTER-KEY-VALUE")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://127.0.0.1:1/hijack")
    monkeypatch.setenv(j.JUDGE_BASE_URL_ENV, "http://127.0.0.1:1/override")
    built = {}

    class _FakeAnthropic:
        def __init__(self, **kw):
            built.update(kw)

    import anthropic
    monkeypatch.setattr(anthropic, "Anthropic", _FakeAnthropic)
    orj = ClaudeJudge(cache=JudgeCache(tmp_path / "c.sqlite"), model=GROK,
                      backend="openrouter")
    orj._client("openrouter")
    assert built["api_key"] == "OPENROUTER-KEY-VALUE"
    assert built["base_url"] == "https://openrouter.ai/api"
    assert built["max_retries"] == j.MAX_RETRIES


# --- the coordination line: which pool a run hits -----------------------------------


def test_coordination_post_names_openrouter_and_the_changed_obligation():
    post = coordination_post(n_unique_requests=2574, model=GROK,
                             backend="openrouter")
    assert post.startswith("200 grok-4.6 openrouter ~1h")
    assert "NOT the shared Anthropic" in post
    assert "needs no post there" in post
    assert "lowprio" not in post and "highprio" not in post
    assert "no transport_refusal ruling" in post
    assert "2574 unique requests" in post
    # the default is untouched (tests/test_capeval_judging.py pins it)
    assert coordination_post(n_unique_requests=5).startswith(
        "200 sonnet-5 lowprio ~1h")
    with pytest.raises(JudgingError, match="unknown judge backend"):
        coordination_post(n_unique_requests=1, backend="together")
