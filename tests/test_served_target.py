"""The SERVED jailbreak target (scripts/jailbreak_run.py --target-engine served).

CPU only, NO server: a stub chat transport stands in for the RoutingProxy.
Covers the invariants the served-target path exists to hold:

- the ``LLMHFSteered`` surface the gated attacks drive (chat / generate /
  n_target_forwards) is presented unchanged, greedy unless the attack samples;
- NO proxy-owned key is ever sent (cache_salt / selectors are the proxy's);
- a prompted cell's SI is the AUTHORITATIVE system message (attack-supplied
  system content folds into user text, exactly as the HF path does);
- every call is timed into ``timing_log`` (the target-leg decomposition);
- the positive controls: salt isolation over the FULL route set and
  steering-bites vs the scale-0 twin — an inert arm REFUSES, a missing twin
  refuses, a cell absent from the routing table refuses (never a silent fall
  back to the in-process target).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# the sibling tests' idiom: src/ for the package (the cluster gate venv installs no sisteer; the suite imports it from the tree) and
# scripts/ for the bare `import jailbreak_run` below -- a scripts/ module, which failed with ModuleNotFoundError at line ~688 whenever
# the file ran without PYTHONPATH=scripts (the backlog's "bare import"; tests/test_served_target_decode.py inserts the same path)
REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from sisteer.capeval.serving.fidelity import FidelityError  # noqa: E402
from sisteer.capeval.serving.proxy import FORBIDDEN_CLIENT_KEYS, RouteEntry  # noqa: E402
from sisteer.eval.attacks import served_target as st  # noqa: E402
from sisteer.capeval.serving import served_window as sw  # noqa: E402

LENS = "proxy_vllm+lens+interleaved"
PASS = "proxy_vllm+passthrough+interleaved"


def _routes():
    return [
        RouteEntry(virtual_model="petri__base", cell_id="none", cell_class="base",
                   serving_identity=PASS, intervention_hash="aa" * 8),
        RouteEntry(virtual_model="petri__none__lens0", cell_id="none__lens0",
                   cell_class="control", serving_identity=LENS,
                   intervention_hash="bb" * 8, control_of="none"),
        RouteEntry(virtual_model="petri__si_constitution__psr_kl_all",
                   cell_id="si_constitution__psr_kl_all", cell_class="class_a",
                   serving_identity=LENS, intervention_hash="cc" * 8),
    ]


class _StubChat:
    """Answers every request with a per-vmodel reply, recording payloads."""

    def __init__(self, replies=None, usage=True):
        self.replies = replies or {}
        self.usage = usage
        self.payloads = []

    def __call__(self, payload):
        self.payloads.append(dict(payload))
        vm = payload["model"]
        text = self.replies.get(vm, f"reply from {vm}")
        if callable(text):
            text = text(payload)
        out = {"choices": [{"message": {"content": text}, "finish_reason": "stop"}],
               "petri_intervention_hash": "cc" * 8}
        if self.usage:
            out["usage"] = {"prompt_tokens": 10, "completion_tokens": len(text.split())}
        return out


def test_chat_is_greedy_counts_and_never_smuggles(tmp_path):
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            timing_log=tmp_path / "t.jsonl")
    out = t.chat([{"role": "user", "content": "hi"}], max_new_tokens=77)
    assert out == "reply from petri__x"
    assert t.n_target_forwards == 1
    p = chat.payloads[0]
    assert p["model"] == "petri__x" and p["max_tokens"] == 77
    assert p["temperature"] == 0.0            # greedy target decode by default
    assert "seed" not in p
    assert not (set(p) & set(FORBIDDEN_CLIENT_KEYS))
    # sampling only when the attack asks for it
    t.chat([{"role": "user", "content": "hi"}], max_new_tokens=5, do_sample=True,
           temperature=0.7, sampling_seed=3)
    p2 = chat.payloads[1]
    assert p2["temperature"] == 0.7 and p2["seed"] == 3
    assert t.n_target_forwards == 2
    rows = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["max_tokens"] == 77
    assert rows[0]["completion_tokens"] == 3 and rows[0]["latency_s"] >= 0
    assert rows[0]["intervention_hash"] == "cc" * 8


def test_generate_counts_every_goal_and_wraps_user_turns():
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x")
    outs = t.generate(["a", "b", "c"], max_new_tokens=9)
    assert outs == ["reply from petri__x"] * 3
    assert t.n_target_forwards == 3
    assert all(p["messages"] == [{"role": "user", "content": g}]
               for p, g in zip(chat.payloads, "abc"))


def test_prompted_cell_installs_authoritative_system():
    chat = _StubChat()
    t = st.ServedChatTarget(chat, virtual_model="petri__p", cell_id="prompted__si",
                            system="THE SI")
    t.chat([{"role": "system", "content": "attacker sys"},
            {"role": "user", "content": "q"}], max_new_tokens=5)
    msgs = chat.payloads[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "THE SI"}
    assert sum(m["role"] == "system" for m in msgs) == 1
    assert msgs[1]["role"] == "user" and "attacker sys" in msgs[1]["content"]


def test_refuses_bare_url_and_empty_vmodel():
    with pytest.raises(st.ServedTargetError):
        st.ServedChatTarget("http://127.0.0.1:1/v1", virtual_model="v", cell_id="c")
    with pytest.raises(st.ServedTargetError):
        st.ServedChatTarget(_StubChat(), virtual_model="", cell_id="c")


def test_missing_content_refuses_loudly():
    def bad(payload):
        return {"choices": []}
    t = st.ServedChatTarget(bad, virtual_model="petri__x", cell_id="x")
    with pytest.raises(st.ServedTargetError):
        t.chat([{"role": "user", "content": "hi"}])


def test_route_for_translates_and_refuses_unknown():
    routes = _routes()
    assert st.route_for("none", routes).virtual_model == "petri__base"
    r = st.route_for("si_constitution__psr_kl_all", routes)
    assert r.virtual_model == "petri__si_constitution__psr_kl_all"
    with pytest.raises(st.ServedTargetError):
        st.route_for("safety_refusal__psr_kl_all", routes)


def test_evidence_bites_verified_when_twin_differs():
    routes = _routes()
    route = st.route_for("si_constitution__psr_kl_all", routes)
    chat = _StubChat(replies={"petri__si_constitution__psr_kl_all": "steered text",
                              "petri__none__lens0": "plain text"})
    ev = st.served_evidence(chat, route, routes)
    assert ev["salt_isolation"]["ok"] and ev["salt_isolation"]["n_routes"] == 3
    sb = ev["steering_bites"]
    assert sb["applicable"] and sb["verified"] and sb["twin"] == "none__lens0"
    assert sb["n_probes"] == len(st.PROBE_PROMPTS) and sb["n_differing"] == sb["n_probes"]
    # the probe went through the SAME serving path, greedy, no smuggled keys
    for p in chat.payloads:
        assert p["temperature"] == 0.0 and p["max_tokens"] == st.PROBE_MAX_TOKENS
        assert not (set(p) & set(FORBIDDEN_CLIENT_KEYS))


def test_evidence_refuses_inert_arm():
    routes = _routes()
    route = st.route_for("si_constitution__psr_kl_all", routes)
    chat = _StubChat()  # every vmodel answers with its own name... make them equal
    chat.replies = {"petri__si_constitution__psr_kl_all": "same",
                    "petri__none__lens0": "same"}
    with pytest.raises(FidelityError):
        st.served_evidence(chat, route, routes)


def test_evidence_refuses_without_twin():
    routes = [r for r in _routes() if r.cell_id != "none__lens0"]
    route = st.route_for("si_constitution__psr_kl_all", routes)
    with pytest.raises(st.ServedTargetError):
        st.served_evidence(_StubChat(), route, routes)


def test_evidence_passthrough_not_applicable():
    routes = _routes()
    ev = st.served_evidence(_StubChat(), st.route_for("none", routes), routes)
    assert ev["steering_bites"]["applicable"] is False


def _write_table(path: Path, routes):
    from dataclasses import asdict
    path.write_text(json.dumps({
        "version": 1, "base_model": "org/model", "serving_mode": "interleaved",
        "exact_replay_strategy": None, "base_revision": "r",
        "routes": [asdict(r) for r in routes], "prompted_prefixes": []}))


def test_build_served_target_end_to_end(monkeypatch, tmp_path):
    routes = _routes()
    table = tmp_path / "routing_table.json"
    _write_table(table, routes)
    chat = _StubChat(replies={"petri__si_constitution__psr_kl_all": "steered",
                              "petri__none__lens0": "plain"})

    class FakeClient:
        def __init__(self, base_url, *, identity):
            self.base_url = base_url.rstrip("/") + ("" if base_url.endswith("/v1") else "/v1")
            self.identity = identity
            self.served_models = ()
            self.proxy_identity = {"impl": "fake"}
            self.loaded_with = None
            self.chat_completions = chat

        def load(self, *, require_virtual_models=()):
            self.loaded_with = tuple(require_virtual_models)
            self.served_models = tuple(r.virtual_model for r in routes)

    monkeypatch.setattr(st, "ProxyClientEngine", FakeClient)
    t = st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id="si_constitution__psr_kl_all",
                               timing_log=tmp_path / "tl.jsonl")
    assert t.virtual_model == "petri__si_constitution__psr_kl_all"
    assert t.route_info["intervention_hash"] == "cc" * 8
    assert t.route_info["base_model"] == "org/model"
    assert t.evidence["steering_bites"]["verified"]
    assert t.system is None
    out = tmp_path / "out"
    p = st.write_target_manifest(out, t)
    doc = json.loads(p.read_text())
    assert doc["target_engine"] == "served" and doc["evidence"]["steering_bites"]["verified"]
    assert doc["virtual_model"] == t.virtual_model


def test_build_served_target_refuses_unrouted_cell(monkeypatch, tmp_path):
    table = tmp_path / "rt.json"
    _write_table(table, _routes())
    with pytest.raises(st.ServedTargetError):
        st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id="si_irrelevant__psr_kl_all")


# ---------------------------------------------------------------------------- #
# Serve variants: the attested declaration                                      #
# is REQUIRED for a variant cell; render refusals are TERMINAL per goal.       #
# Synthetic ids, a stub transport; the literals are PINNED in                   #
# served_target (tests/test_fst_integration_contract.py checks them against   #
# proxy.py on the integration branch).                                         #
# ---------------------------------------------------------------------------- #

import asyncio  # noqa: E402

from sisteer.capeval.serving.proxy import ProxyError, ProxyRefusal  # noqa: E402
from sisteer.capeval.serving.serve_declaration import variant_declarations  # noqa: E402
from sisteer.capeval.serving.served_window import PromptOverWindow  # noqa: E402
from sisteer.eval.attacks.static_common import load_covered_goals  # noqa: E402
from sisteer.eval.jailbreak_artifacts import (REFUSED_OUTCOME_PREFIX,  # noqa: E402
                                              exclude_errored_goals, refused_outcome)

TT_PSR = "safety_refusal__psr_kl_all"
H_PSR = "si_safety__psr_kl_all"
#: what WS-0 attests per variant (serve_declaration.variant_declarations) — the
#: values the client compares the proxy's map against
DECL_ALL_TURNS = {"serve_experiment": "fit_response_serve_all_turns"}
DECL_ALL_POSITIONS = {"serve_experiment": "fit_response_serve_all_positions"}
DECL_THINKING = {"serve_out_of_support": ["thinking"]}
ATTESTED_EXP2 = {f"{H_PSR}@all_turns": DECL_ALL_TURNS,
                 f"{H_PSR}@all_positions": DECL_ALL_POSITIONS}


def _variant_routes():
    return _routes() + [
        RouteEntry(virtual_model=f"petri__{H_PSR}", cell_id=H_PSR, cell_class="class_a",
                   serving_identity=LENS, intervention_hash="d1" * 8),
        RouteEntry(virtual_model=f"petri__{H_PSR}@all_turns", cell_id=f"{H_PSR}@all_turns",
                   cell_class="class_a", serving_identity=LENS, intervention_hash="d2" * 8),
        RouteEntry(virtual_model=f"petri__{H_PSR}@all_positions",
                   cell_id=f"{H_PSR}@all_positions", cell_class="class_a",
                   serving_identity=LENS, intervention_hash="d3" * 8),
    ]


def _wire_refusal(code, status=400):
    """What ProxyClientEngine raises for a proxy refusal: the JSON error body in
    the transport's text (``proxy._error``'s shape)."""
    body = {"error": {"message": f"the render check said no ({code})",
                      "type": "invalid_request_error", "code": code}}
    return ProxyError(f"proxy http://127.0.0.1:1/v1/chat/completions returned {status}: "
                      + json.dumps(body))


def _context_400():
    body = {"error": {"message": "upstream vLLM returned 400: This model's maximum context "
                                 "length is 9984 tokens. However, you requested 2048 output "
                                 "tokens and your prompt contains 9000 input tokens, for a "
                                 "total of 11048 tokens. Please reduce the length of the "
                                 "input prompt or the number of requested output tokens.",
                      "type": "invalid_request_error", "code": "upstream_error"}}
    return ProxyError("proxy http://127.0.0.1:1/v1/chat/completions returned 400: "
                      + json.dumps(body))


def test_pinned_all_turns_outcomes_carry_one_prefix():
    """S3: ``refused:<code>`` with the COMPLETE proxy code — never a doubled
    ``refused:all_turns_all_turns_…``; the four exact strings."""
    assert st.ALL_TURNS_REFUSAL_CODES == (
        "all_turns_span_mismatch", "all_turns_render_refused",
        "all_turns_expected_unavailable", "all_turns_render_failed")
    assert st.ALL_TURNS_REFUSED_OUTCOMES == (
        "refused:all_turns_span_mismatch", "refused:all_turns_render_refused",
        "refused:all_turns_expected_unavailable", "refused:all_turns_render_failed")
    for code, outcome in zip(st.ALL_TURNS_REFUSAL_CODES, st.ALL_TURNS_REFUSED_OUTCOMES):
        assert refused_outcome(code) == outcome == f"{REFUSED_OUTCOME_PREFIX}{code}"
        assert outcome.count("all_turns_") == 1
        assert not outcome.startswith("error:")
        assert st.AllTurnsRenderRefused("x", code=code).outcome == outcome
    with pytest.raises(ValueError):
        st.AllTurnsRenderRefused("x", code="not_a_code")
    with pytest.raises(ValueError):
        refused_outcome("has:colon")


def test_variant_declarations_are_what_the_client_expects():
    """The WS-0 values the client compares the attested map against."""
    assert variant_declarations("all_turns") == DECL_ALL_TURNS
    assert variant_declarations("all_positions") == DECL_ALL_POSITIONS
    assert variant_declarations("thinking") == DECL_THINKING
    assert st.ATTEST_SERVE_DECLARATIONS_KEY == "lens_serve_declarations"
    assert st.LENS_SALT_POLICIES == ("attested", "request_unique_all")


def test_all_turns_refusal_code_recognises_by_code_only():
    for code in st.ALL_TURNS_REFUSAL_CODES:
        assert st.all_turns_refusal_code(_wire_refusal(code)) == code
        assert st.all_turns_refusal_code(ProxyRefusal("m", code=code)) == code
        # a 409 / 500 carrying the code text is NOT a render refusal
        assert st.all_turns_refusal_code(_wire_refusal(code, status=409)) is None
        assert st.all_turns_refusal_code(ProxyRefusal("m", code=code, http_status=409)) is None
    # the python-repr body some transports write (the fallback regex)
    assert st.all_turns_refusal_code(ProxyError(
        "proxy x returned 400: {'error': {'message': 'm', 'code': 'all_turns_render_refused'}}"
    )) == "all_turns_render_refused"
    # a context-length 400, another code, a transport error, the window's error
    assert st.all_turns_refusal_code(_context_400()) is None
    assert st.all_turns_refusal_code(_wire_refusal("steered_fork_unsupported")) is None
    assert st.all_turns_refusal_code(ProxyError("cannot reach the petri proxy")) is None
    assert st.all_turns_refusal_code(PromptOverWindow(
        prompt_tokens=1, requested_max_tokens=1, window=1, server_window=1,
        effective_window=1, reason="r")) is None
    # prose that merely mentions a code is not a refusal
    assert st.all_turns_refusal_code(ProxyError(
        "proxy x returned 500: all_turns_render_failed happened")) is None


def test_target_raises_typed_refusal_counts_per_code_and_logs(tmp_path):
    codes = iter(["all_turns_expected_unavailable", "all_turns_render_refused"])

    def chat(payload):
        raise _wire_refusal(next(codes))
    t = st.ServedChatTarget(chat, virtual_model=f"petri__{H_PSR}@all_turns",
                            cell_id=f"{TT_PSR}@all_turns", timing_log=tmp_path / "t.jsonl")
    with pytest.raises(st.AllTurnsRenderRefused) as ei:
        t.chat([{"role": "user", "content": "hi"}], max_new_tokens=32)
    err = ei.value
    assert err.code == "all_turns_expected_unavailable"
    assert err.outcome == "refused:all_turns_expected_unavailable"
    assert isinstance(err, st.ServedTargetError) and isinstance(err.__cause__, ProxyError)
    with pytest.raises(st.AllTurnsRenderRefused):
        t.chat([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "a"},
                {"role": "user", "content": "b"}], max_new_tokens=32)
    assert t.n_all_turns_refused == {"all_turns_expected_unavailable": 1,
                                     "all_turns_render_refused": 1}
    assert t.n_all_turns_refused_total == 2
    assert t.n_target_forwards == 2 and t.n_prompt_over_window == 0
    rows = [json.loads(l) for l in (tmp_path / "t.jsonl").read_text().splitlines()]
    assert [r["error"] for r in rows] == ["all_turns_expected_unavailable",
                                          "all_turns_render_refused"]
    assert rows[1]["n_messages"] == 3 and rows[1]["completion_tokens"] == 0
    assert rows[1]["max_tokens"] == 32 and rows[1]["finish_reason"] is None
    assert t.n_timing_rows == 2


def test_clamp_check_stays_first_and_a_refused_retry_is_typed(tmp_path):
    """A context-length 400 is the clamp check's business and that check stays FIRST — but on the
    multi-turn chat() path the policy is STRICT (since 2026-09-10): room below
    the requested cap is the typed PromptOverWindow, never a clamped retry that would silently shorten
    the reply (the clamped path used to send 512 then 204 and record n_prompt_over_window=0). So the
    wire refusal staged for a second call is never reached: exactly one request, the over-window
    counter, no render-refusal count. The once-only clamp stays on the single-turn path
    (tests/test_served_window_clamp.py); tests/test_review_p1_4_multiturn_overflow.py pins this policy."""
    calls = []

    def chat(payload):
        calls.append(dict(payload))
        if len(calls) == 1:
            raise _context_400()
        raise _wire_refusal("all_turns_span_mismatch")
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x",
                            max_model_len=9984, timing_log=tmp_path / "t.jsonl")
    with pytest.raises(sw.PromptOverWindow):
        t.chat([{"role": "user", "content": "hi"}], max_new_tokens=2048)
    assert len(calls) == 1 and calls[0]["max_tokens"] == 2048     # the 400 was classified first; no clamped retry
    assert t.n_prompt_over_window == 1
    assert t.n_all_turns_refused == {}


def test_non_refusal_errors_still_propagate_unchanged():
    def chat(payload):
        raise ProxyError("proxy x returned 409: salt discipline")
    t = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x")
    with pytest.raises(ProxyError):
        t.chat([{"role": "user", "content": "hi"}])
    assert t.n_all_turns_refused == {}


def test_attested_declaration_for_matrix():
    hid, base = f"{H_PSR}@all_turns", H_PSR
    ok = {st.ATTEST_SERVE_DECLARATIONS_KEY: {**ATTESTED_EXP2,
                                             "none__lens0": DECL_THINKING}}   # a sibling entry
    assert st.attested_declaration_for(ok, harness_id=hid, base_harness_id=base,
                                       variant="all_turns") == DECL_ALL_TURNS
    assert st.attested_declaration_for(ok, harness_id=f"{H_PSR}@all_positions",
                                       base_harness_id=base,
                                       variant="all_positions") == DECL_ALL_POSITIONS
    # no map at all -> refuse (setting (2) undeclared is refused like thinking-ON)
    with pytest.raises(st.ServedTargetError, match="attests NO"):
        st.attested_declaration_for({"impl": "fake"}, harness_id=hid, base_harness_id=base,
                                    variant="all_turns")
    with pytest.raises(st.ServedTargetError, match="attests NO"):
        st.attested_declaration_for(None, harness_id=hid, base_harness_id=base,
                                    variant="all_turns")
    # an error beside the map -> refuse
    with pytest.raises(st.ServedTargetError, match="lens_serve_declarations_error"):
        st.attested_declaration_for(
            {**ok, st.ATTEST_SERVE_DECLARATIONS_ERROR_KEY: "workers disagree"},
            harness_id=hid, base_harness_id=base, variant="all_turns")
    # the cell is missing from the map -> refuse
    with pytest.raises(st.ServedTargetError, match="no entry"):
        st.attested_declaration_for(
            {st.ATTEST_SERVE_DECLARATIONS_KEY: {f"{H_PSR}@all_positions": DECL_ALL_POSITIONS}},
            harness_id=hid, base_harness_id=base, variant="all_turns")
    # a DIFFERENT value -> refuse, naming both
    with pytest.raises(st.ServedTargetError) as ei:
        st.attested_declaration_for(
            {st.ATTEST_SERVE_DECLARATIONS_KEY: {hid: DECL_ALL_POSITIONS}},
            harness_id=hid, base_harness_id=base, variant="all_turns")
    assert "fit_response_serve_all_turns" in str(ei.value)
    assert "fit_response_serve_all_positions" in str(ei.value)
    # the BASE cell must be ABSENT
    with pytest.raises(st.ServedTargetError, match="BASE cell"):
        st.attested_declaration_for(
            {st.ATTEST_SERVE_DECLARATIONS_KEY: {**ATTESTED_EXP2, base: DECL_ALL_TURNS}},
            harness_id=hid, base_harness_id=base, variant="all_turns")
    # a malformed map -> refuse
    with pytest.raises(st.ServedTargetError, match="not a per-cell map"):
        st.attested_declaration_for({st.ATTEST_SERVE_DECLARATIONS_KEY: ["x"]},
                                    harness_id=hid, base_harness_id=base, variant="all_turns")
    # a NON-variant cell: nothing required ...
    assert st.attested_declaration_for(ok, harness_id=base, base_harness_id=base,
                                       variant=None) is None
    assert st.attested_declaration_for({"impl": "fake"}, harness_id=base,
                                       base_harness_id=base, variant=None) is None
    # ... but a map that DECLARES it refuses (the arm would be mislabelled)
    with pytest.raises(st.ServedTargetError, match="DECLARED out of support"):
        st.attested_declaration_for(
            {st.ATTEST_SERVE_DECLARATIONS_KEY: {base: DECL_THINKING}},
            harness_id=base, base_harness_id=base, variant=None)


class _FakeVariantClient:
    """ProxyClientEngine stand-in whose /v1/models identity block is set by the test."""
    routes = ()
    chat = None
    identity_block = None

    def __init__(self, base_url, *, identity):
        self.base_url = base_url.rstrip("/") + ("" if base_url.endswith("/v1") else "/v1")
        self.identity = identity
        self.served_models = ()
        self.proxy_identity = None
        self.chat_completions = type(self).chat

    def load(self, *, require_virtual_models=()):
        self.served_models = tuple(r.virtual_model for r in type(self).routes)
        self.proxy_identity = type(self).identity_block


def _install_client(monkeypatch, routes, chat, identity_block):
    _FakeVariantClient.routes = routes
    _FakeVariantClient.chat = staticmethod(chat)
    _FakeVariantClient.identity_block = identity_block
    monkeypatch.setattr(st, "ProxyClientEngine", _FakeVariantClient)


def test_build_served_target_variant_end_to_end(monkeypatch, tmp_path):
    routes = _variant_routes()
    table = tmp_path / "rt.json"
    _write_table(table, routes)
    chat = _StubChat(replies={f"petri__{H_PSR}@all_turns": "steered",
                              "petri__none__lens0": "plain"})
    _install_client(monkeypatch, routes, chat, {
        "version": 1, st.ATTEST_SERVE_DECLARATIONS_KEY: ATTESTED_EXP2,
        st.ATTEST_LENS_SALT_POLICY_KEY: "request_unique_all",
        st.ATTEST_OBJECT_SHA256_BY_CELL_KEY: {H_PSR: "ab" * 32, f"{H_PSR}@all_turns": "ab" * 32,
                                              f"{H_PSR}@all_positions": "ab" * 32}})
    t = st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id=f"{TT_PSR}@all_turns", timing_log=tmp_path / "tl.jsonl")
    assert t.virtual_model == f"petri__{H_PSR}@all_turns"
    assert t.cell_id == f"{TT_PSR}@all_turns"
    assert t.route_info["harness_cell_id"] == f"{H_PSR}@all_turns"
    assert t.route_info["serve_variant"] == "all_turns"
    assert t.route_info["base_cell_id"] == TT_PSR
    assert t.route_info["positions"] == "response_all_turns"      # the variant's overlay
    assert t.route_info["serve_declarations"] == DECL_ALL_TURNS
    assert t.route_info["lens_salt_policy"] == "request_unique_all"
    assert t.route_info["object_artifact_sha256_by_cell"][f"{H_PSR}@all_turns"] == "ab" * 32
    assert t.route_info["intervention_hash"] == "d2" * 8
    assert t.evidence["steering_bites"]["verified"] and t.evidence["steering_bites"]["twin"] == "none__lens0"
    doc = json.loads(st.write_target_manifest(tmp_path / "out", t).read_text())
    assert doc["serve_variant"] == "all_turns" and doc["serve_declarations"] == DECL_ALL_TURNS
    assert doc["all_turns_check"] == {
        "n_refused_total": 0,
        "by_code": {c: 0 for c in st.ALL_TURNS_REFUSAL_CODES},
        "codes": list(st.ALL_TURNS_REFUSAL_CODES), "outcome_prefix": "refused:"}
    # the base cell on the SAME standup: no declaration required, none recorded
    b = st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id=TT_PSR, controls=False)
    assert b.route_info["serve_variant"] is None and b.route_info["serve_declarations"] is None
    assert b.route_info["positions"] is None        # no informational field on this table


def test_build_served_target_refuses_an_unattested_variant_before_any_call(monkeypatch, tmp_path):
    routes = _variant_routes()
    table = tmp_path / "rt.json"
    _write_table(table, routes)
    chat = _StubChat()
    # (a) a proxy that predates the landing: no map at all
    _install_client(monkeypatch, routes, chat, {"version": 1})
    with pytest.raises(st.ServedTargetError, match="attests NO"):
        st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id=f"{TT_PSR}@all_turns")
    assert chat.payloads == []                       # refused BEFORE the controls ran
    # (b) attested, but for the OTHER variant only
    _install_client(monkeypatch, routes, chat, {
        st.ATTEST_SERVE_DECLARATIONS_KEY: {f"{H_PSR}@all_positions": DECL_ALL_POSITIONS}})
    with pytest.raises(st.ServedTargetError, match="no entry"):
        st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id=f"{TT_PSR}@all_turns")
    # (c) the base is declared too -> the standup mislabels its cells
    _install_client(monkeypatch, routes, chat, {
        st.ATTEST_SERVE_DECLARATIONS_KEY: {**ATTESTED_EXP2, H_PSR: DECL_ALL_TURNS}})
    with pytest.raises(st.ServedTargetError, match="BASE cell"):
        st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id=f"{TT_PSR}@all_positions")
    # (d) the base addressed while the map declares it
    with pytest.raises(st.ServedTargetError, match="DECLARED out of support"):
        st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id=TT_PSR)
    # (e) a malformed variant spelling refuses at the door
    with pytest.raises(st.ServedTargetError):
        st.build_served_target(proxy_url="http://127.0.0.1:9/v1", routing_table=table,
                               cell_id=f"{TT_PSR}@bogus")
    assert chat.payloads == []


def test_append_target_attempt_accumulates_one_row_per_process(tmp_path):
    def chat(payload):
        raise _wire_refusal("all_turns_render_failed")
    t1 = st.ServedChatTarget(chat, virtual_model="petri__v", cell_id="c",
                             route_info={"tt_cell_id": "c", "harness_cell_id": "h",
                                         "intervention_hash": "ii", "serve_variant": "all_turns"})
    with pytest.raises(st.AllTurnsRenderRefused):
        t1.chat([{"role": "user", "content": "hi"}])
    t2 = st.ServedChatTarget(_StubChat(), virtual_model="petri__v", cell_id="c")
    t2.chat([{"role": "user", "content": "hi"}])
    p1 = st.append_target_attempt(tmp_path, t1)
    p2 = st.append_target_attempt(tmp_path, t2)
    assert p1 == p2 == tmp_path / st.TARGET_ATTEMPTS_FILE
    rows = [json.loads(l) for l in p1.read_text().splitlines()]
    assert [r["stage"] for r in rows] == ["final", "final"]
    assert rows[0]["all_turns_check"]["by_code"]["all_turns_render_failed"] == 1
    assert rows[0]["all_turns_check"]["n_refused_total"] == 1
    assert rows[0]["serve_variant"] == "all_turns" and rows[0]["harness_cell_id"] == "h"
    assert rows[1]["all_turns_check"]["n_refused_total"] == 0
    assert rows[1]["n_target_forwards"] == 1 and rows[1]["window_clamp"]["n_target_forwards"] == 1
    # the analysis SUMS the final rows
    assert sum(r["all_turns_check"]["n_refused_total"] for r in rows) == 1


# --------------------------------------------------------------------------- #
# The PyRIT runner: a render refusal is a TERMINAL refused: goal (F4 / S2)     #
# --------------------------------------------------------------------------- #


class _Scripted:
    """A proxy transport scripted per objective: the k-th call for an
    objective returns ``script[objective][k]`` (a str reply, or an exception to
    raise). Persisted replies are what ``responses.jsonl`` should hold."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = {}

    def __call__(self, payload):
        obj = payload["messages"][0]["content"]
        k = self.calls.get(obj, 0)
        self.calls[obj] = k + 1
        steps = self.script.get(obj, ["ok"])
        step = steps[min(k, len(steps) - 1)]
        if isinstance(step, BaseException):
            raise step
        return {"choices": [{"message": {"content": step}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def _two_turn_executor_cls():
    from pyrit.models import Message

    class _TwoTurns:
        """A Crescendo-like executor: two target sends per goal; the target's
        exception surfaces through PyRIT's bare-Exception wrap."""
        def __init__(self, target):
            self.target = target

        async def execute_async(self, *, objective):
            conv = [Message.from_prompt(prompt=objective, role="user")]
            try:
                r1 = await self.target._send_prompt_to_target_async(normalized_conversation=conv)
                conv = conv + [Message.from_prompt(prompt=r1[0].get_value(), role="assistant"),
                               Message.from_prompt(prompt="and then?", role="user")]
                # keep the objective first so the scripted transport keys on it
                conv[0] = Message.from_prompt(prompt=objective, role="user")
                await self.target._send_prompt_to_target_async(normalized_conversation=conv)
            except Exception as ex:  # noqa: BLE001 — the normalizer's wrap
                raise Exception("Error sending prompt with conversation ID: x") from ex
            from types import SimpleNamespace
            return SimpleNamespace(outcome="failure", executed_turns=2)
    return _TwoTurns


def _run(defense, goals, out_dir, *, resume):
    from sisteer.eval.attacks.pyrit_adapters import (AttackerChatTarget,
                                                     run_pyrit_attack_async)
    cls = _two_turn_executor_cls()
    return asyncio.run(run_pyrit_attack_async(
        defense=defense, goals=goals,
        attacker_factory=lambda lock: AttackerChatTarget(gpu_lock=lock,
                                                         generate_fn=lambda chat: "x"),
        scorer_fn=lambda p: "1",
        attack_factory=lambda *, target, adv_cfg, scorers: cls(target),
        out_dir=out_dir, memory_db_path=out_dir / "memory.db", resume=resume))


def _goals(names):
    from types import SimpleNamespace
    return [SimpleNamespace(goal_id=n, text=f"objective {n}", category="c") for n in names]


def test_pyrit_runner_records_refused_terminal_for_every_code(tmp_path):
    from sisteer.eval.attacks.pyrit_adapters import typed_error_outcome
    for i, code in enumerate(st.ALL_TURNS_REFUSAL_CODES):
        chat = _Scripted({"objective g0": ["turn one", _wire_refusal(code)]})
        d = st.ServedChatTarget(chat, virtual_model="petri__x", cell_id="x")
        out = _run(d, _goals(["g0"]), tmp_path / f"r{i}", resume=False)
        (rec,) = out["per_goal"]
        assert rec["outcome"] == f"refused:{code}"
        assert set(rec) == {"goal_id", "category", "n_target_forwards", "outcome", "turns", "capped"}
        assert rec["n_target_forwards"] == 2
        # covered on resume: a refused goal is TERMINAL
        assert load_covered_goals(tmp_path / f"r{i}" / "progress.jsonl") == {"g0"}
        # its partial rows are persisted (the graders drop them, not the runner)
        rows = [json.loads(l) for l in (tmp_path / f"r{i}" / "responses.jsonl").read_text().splitlines()]
        assert [r["response"] for r in rows] == ["turn one"]
        assert d.n_all_turns_refused == {code: 1}
    assert typed_error_outcome(st.AllTurnsRenderRefused("m", code="all_turns_render_refused")) \
        == "refused:all_turns_render_refused"
    assert typed_error_outcome(PromptOverWindow(
        prompt_tokens=1, requested_max_tokens=1, window=1, server_window=1,
        effective_window=1, reason="r")) == "error:PromptOverWindow"


def test_resume_keeps_refused_out_of_todo_and_retries_errored(tmp_path):
    """F4 / S2: a partial run with one goal refused
    (``refused:all_turns_expected_unavailable``), one ``error:PromptOverWindow``
    and one completed, INTERRUPTED before a fourth goal, then resumed twice:
    the refused goal stays out of ``todo``, keeps its rows, and is excluded +
    counted by code; the errored goal is retried; the fourth runs; the counts
    do not depend on the number of resumes."""
    out = tmp_path / "shard00"
    script = {"objective A": ["A turn one", _wire_refusal("all_turns_expected_unavailable")],
              "objective B": [_context_400()],            # unfittable -> error:PromptOverWindow
              "objective C": ["C one", "C two"]}
    d1 = st.ServedChatTarget(_Scripted(script), virtual_model="petri__x", cell_id="x",
                             max_model_len=9984)
    r1 = _run(d1, _goals(["A", "B", "C"]), out, resume=False)       # interrupted before D
    by = {r["goal_id"]: r["outcome"] for r in r1["per_goal"]}
    assert by == {"A": "refused:all_turns_expected_unavailable",
                  "B": "error:PromptOverWindow", "C": "failure"}
    assert load_covered_goals(out / "progress.jsonl") == {"A", "C"}

    def snapshot():
        import jailbreak_run
        prog = jailbreak_run._load_progress(out / "progress.jsonl")
        resp = {}
        for l in (out / "responses.jsonl").read_text().splitlines():
            r = json.loads(l)
            resp.setdefault(r["goal_id"], []).append(r["response"])
        ids, clean, rep = exclude_errored_goals(resp, prog, ["A", "B", "C", "D"], label="t",
                                                allow_errored=True)
        return prog, resp, ids, clean, rep

    # resume 1: everything now succeeds; only B (errored) and D (never run) are todo
    healthy = {k: ["one", "two"] for k in ("objective A", "objective B", "objective C",
                                            "objective D")}
    d2 = st.ServedChatTarget(_Scripted(healthy), virtual_model="petri__x", cell_id="x",
                             max_model_len=9984)
    r2 = _run(d2, _goals(["A", "B", "C", "D"]), out, resume=True)
    assert sorted(r["goal_id"] for r in r2["per_goal"]) == ["B", "D"]
    assert d2.n_target_forwards == 4
    prog, resp, ids, clean, rep = snapshot()
    assert prog["A"]["outcome"] == "refused:all_turns_expected_unavailable"   # untouched
    assert prog["B"]["outcome"] == "failure"                                # retried
    assert resp["A"] == ["A turn one"]                                      # rows kept
    assert rep["refused_goal_ids"] == ["A"] and rep["n_refused"] == 1
    assert rep["refused_by_code"] == {"all_turns_expected_unavailable": 1}
    assert rep["n_errored"] == 0 and rep["errored_goal_ids"] == []
    assert ids == ["B", "C", "D"] and "A" not in clean                        # dropped from grading
    assert rep["n_graded"] == 3
    rows_A = sum(1 for l in (out / "progress.jsonl").read_text().splitlines()
                 if json.loads(l)["goal_id"] == "A")
    assert rows_A == 1
    # resume 2: nothing to do; the counts are identical
    d3 = st.ServedChatTarget(_Scripted(healthy), virtual_model="petri__x", cell_id="x")
    r3 = _run(d3, _goals(["A", "B", "C", "D"]), out, resume=True)
    assert r3["n_goals_run"] == 0 and d3.n_target_forwards == 0
    prog3, resp3, ids3, clean3, rep3 = snapshot()
    assert (ids3, rep3) == (ids, rep) and resp3["A"] == ["A turn one"]
    assert prog3["A"] == prog["A"]
