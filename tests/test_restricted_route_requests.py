"""RESTRICTED-MODE native routes at the proxy: the request ALLOWLIST before
forwarding, the request-scoped response-boundary refusal after the upstream
completion, and the completions endpoint — with the standup-level half over
the scripted upstream.

A restricted-mode route = a lora route whose ATTESTED adapter carries a residual
payload at a position mode other than ``all``. Against a stub backend (no server):

  (i)   ``truncate_prompt_tokens=..., truncation_side="right"`` on a restricted-mode
        route is REFUSED by the allowlist BEFORE forwarding (the backend never sees it);
        the allowlist is EXACT (every listed key admitted, everything else refused,
        ``chat_template_kwargs`` only with ``enable_thinking`` alone, the seeded shape
        only in full);
  (ii)  a request the worker marks UNLOCATED has its completion REFUSED with
        ``response_boundary_unlocated`` (500) — never returned stamped as the cell;
  (iii) the same two requests on an ALL-POSITIONS PSR route and on a LoRA route are
        served unchanged (forwarded wholesale, no claim, cell-stable salt);
  (iv)  a response without ``id`` refuses (502, the lens's shape);
  (v)   ledger overflow AFTER readiness, modelled with the worker's OWN
        ``RequestFailureLedger``: the fault payload refuses every fresh completion
        (``response_boundary_ledger_fault``), the OLDEST failed request's late
        completion is still refused (its entry is still present), an all-positions
        route keeps serving; the rev-3 KNOWN-BAD (an evicting ledger returning None for
        the oldest id) is pinned as the completion the proxy WOULD return — the failure
        the latch forbids.

Plus: the audit row and the ``petri`` stamp carry the position mode and the claim
label; ``/v1/completions`` refuses (restricted code on a restricted route); the
literal twins are pinned. The standup-level tests (bottom; skip without the cached
gemma tokenizer) drive the REAL ``bring_up_lora`` over a synthetic view with a
RESPONSE-fit PSR beside an all-positions PSR and a LoRA cell, the stage-2 flags
armed by monkeypatch, against the scripted upstream whose ledger is scripted.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import pytest

from sisteer.capeval.serving import native_standup as ns
from sisteer.capeval.serving import proxy as px

from test_lora_route_adapter_block import LORA, PASS, _attestation, _sha  # noqa: E402
from test_petri_proxy import StubBackend, _TailRender, _ask, _chat  # noqa: E402

REPO = Path(__file__).resolve().parents[1]

RESP_CELL = "si_safety__psr_kl_all"        # a RESPONSE-fit PSR (restricted mode)
ALL_CELL = "si_ctl__psr_ce_all"            # an all-positions PSR (stage 1)
LORA_CELL = "si_safety__lora_r1"           # a plain LoRA
ZERO_CELL = "none__lora0"
V = {c: f"petri__{c}" for c in (RESP_CELL, ALL_CELL, LORA_CELL, ZERO_CELL)}
V["none"] = "petri__base"


def _routes():
    return [
        px.RouteEntry(virtual_model=V["none"], cell_id="none", cell_class="base",
                      serving_identity=PASS, intervention_hash="aa" * 8),
        px.RouteEntry(virtual_model=V[RESP_CELL], cell_id=RESP_CELL, cell_class="class_b",
                      serving_identity=LORA, intervention_hash="bb" * 8),
        px.RouteEntry(virtual_model=V[ALL_CELL], cell_id=ALL_CELL, cell_class="class_b",
                      serving_identity=LORA, intervention_hash="cc" * 8),
        px.RouteEntry(virtual_model=V[LORA_CELL], cell_id=LORA_CELL, cell_class="class_a",
                      serving_identity=LORA, intervention_hash="dd" * 8),
        px.RouteEntry(virtual_model=V[ZERO_CELL], cell_id=ZERO_CELL, cell_class="control",
                      serving_identity=LORA, intervention_hash="aa" * 8, control_of="none"),
    ]


def _blocks(routes):
    kinds = {RESP_CELL: "psr", ALL_CELL: "psr", LORA_CELL: "lora_all", ZERO_CELL: None}
    out = {}
    for r in routes:
        if r.cell_id not in kinds:
            continue
        zero = r.cell_id == ZERO_CELL
        out[r.cell_id] = {"name": r.cell_id, "dir": f"/bank/{r.cell_id}",
                          "config_sha256": _sha(f"cfg:{r.cell_id}"),
                          "weights_sha256": _sha(f"w:{r.cell_id}"),
                          "r": 1 if (zero or kinds[r.cell_id] == "psr") else 4,
                          "petri_intervention_hash": r.intervention_hash,
                          "petri_zero_adapter": zero, "petri_source_kind": kinds[r.cell_id]}
    return out


def _att(blocks, *, resp_mode="response"):
    att = _attestation(blocks, residual_slot={"armed": True, "n_workers": 1})   # an ARMED slot attests its worker count
    att["adapters"][RESP_CELL].update(residual_fingerprint=_sha("rf:resp"), position_mode=resp_mode)
    att["adapters"][ALL_CELL].update(residual_fingerprint=_sha("rf:all"), position_mode="all")
    return att


def _proxy(backend=None, *, resp_mode="response", attestation=True):
    routes = _routes()
    blocks = _blocks(routes)
    backend = backend if backend is not None else StubBackend()
    kw = {"engine_attestation": _att(blocks, resp_mode=resp_mode)} if attestation else {}
    proxy = px.RoutingProxy(routes, backend, base_model="org/model", serving_mode="interleaved",
                            adapter_blocks=blocks, agent_loop_render=_TailRender(), **kw)
    return proxy, backend


def _response(rid, content="a served reply"):
    return {"id": rid, "object": "chat.completion", "created": 0, "model": "x",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}


def _entry(rid, cell=RESP_CELL):
    return {"req_id": rid, "lora_int_id": 2, "cell": cell, "error_type": "ResponseBoundaryUnlocated",
            "prompt_len": 7, "header_len": 4,
            "detail": "the generation header occurs NOWHERE in the prompt ids"}


def _seeded(vmodel):
    return {"model": vmodel, "max_tokens": 4,
            "messages": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "Answer:"}],
            "continue_final_message": True, "add_generation_prompt": False}


# --------------------------------------------------------------------------- #
# the predicate and the literal twins                                          #
# --------------------------------------------------------------------------- #


def test_restricted_mode_is_read_off_the_attested_adapter_entry():
    proxy, _ = _proxy()
    by_v = {r.virtual_model: r for r in _routes()}
    assert proxy.native_position_mode(by_v[V[RESP_CELL]]) == "response"
    assert proxy.restricted_position_mode(by_v[V[RESP_CELL]]) == "response"
    assert proxy.native_position_mode(by_v[V[ALL_CELL]]) == "all"
    assert proxy.restricted_position_mode(by_v[V[ALL_CELL]]) is None       # all-positions: not restricted
    assert proxy.native_position_mode(by_v[V[LORA_CELL]]) is None          # a plain LoRA
    assert proxy.restricted_position_mode(by_v[V[ZERO_CELL]]) is None
    assert proxy.restricted_position_mode(by_v[V["none"]]) is None         # not a lora route
    for mode in ("prompt_last", "prefill_all"):
        p, _ = _proxy(resp_mode=mode)
        assert p.restricted_position_mode(by_v[V[RESP_CELL]]) == mode
    # no attestation: no mode is known (a lens standup, a v2 table admitted bare)
    bare, _ = _proxy(attestation=False)
    assert bare.native_position_mode(by_v[V[RESP_CELL]]) is None


def test_the_allowlist_constant_and_the_literal_twins_are_pinned():
    from sisteer.capeval.serving import native_residual as nr
    assert px.RESTRICTED_ROUTE_ALLOWED_KEYS == (
        "model", "messages", "max_tokens", "temperature", "top_p", "top_k", "seed", "stop", "n",
        "logprobs", "top_logprobs", "return_token_ids", "return_tokens_as_token_ids", "stream",
        "continue_final_message", "add_generation_prompt",
        # the closed loop's two exact-length controls, and nothing else
        "min_tokens", "ignore_eos")
    assert px.RESTRICTED_ROUTE_GOVERNED_KEYS == (px.CHAT_TEMPLATE_KWARGS_KEY,)
    assert "cache_salt" in px.FORBIDDEN_CLIENT_KEYS            # the proxy's own, never a client key
    assert ns.NATIVE_POSITION_MODES == nr.POSITION_MODES
    assert px.NATIVE_POSITION_MODE_ALL == ns.NATIVE_POSITION_MODE_ALL == "all"
    assert px.RESPONSE_BOUNDARY_UNLOCATED == "response_boundary_unlocated"
    assert px.RESPONSE_BOUNDARY_LEDGER_FAULT == "response_boundary_ledger_fault"
    assert px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED == "restricted_route_option_unsupported"
    # the standup's logprob-liveness knobs are the driver's (a script, not importable here)
    text = (REPO / "scripts" / "served_capability_run.py").read_text()
    for name in ("LOGPROB_LIVENESS_TOP_LOGPROBS", "LOGPROB_LIVENESS_MIN_COMMON_TOKENS",
                 "LOGPROB_LIVENESS_MIN_CONCLUSIVE_PROBES", "LOGPROB_BITES_MIN_RATIO",
                 "LOGPROB_BITES_ABS_FLOOR"):
        m = re.search(rf"^{name} = (.+)$", text, re.M)
        assert m, name
        assert float(getattr(ns, name)) == float(eval(m.group(1))), name  # noqa: S307 — a literal


# --------------------------------------------------------------------------- #
# (i) the allowlist, before forwarding                                         #
# --------------------------------------------------------------------------- #


def test_truncate_prompt_tokens_is_refused_before_forwarding_on_a_restricted_route():
    proxy, backend = _proxy()
    req = {**_chat(V[RESP_CELL]), "truncate_prompt_tokens": 8, "truncation_side": "right"}
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, req)
    assert ei.value.http_status == 400 and ei.value.code == px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED
    assert "truncate_prompt_tokens" in str(ei.value) and "truncation_side" in str(ei.value)
    assert backend.payloads == [] and backend.failure_probes == []        # never forwarded
    assert proxy.request_log.maxlen and len(proxy.request_log) == 0       # no audit row: refused first


@pytest.mark.parametrize("key,value", [
    ("chat_template", "{{ messages }}"), ("prompt_embeds", "AAAA"), ("documents", [{"text": "x"}]),
    ("tools", [{"type": "function", "function": {"name": "f"}}]), ("echo", True),
    ("truncation_side", "left"), ("min_p", 0.1), ("repetition_penalty", 1.1),
    ("presence_penalty", 0.5), ("best_of", 2), ("skip_special_tokens", False),
    (px.RESPONSE_PREFILL_KEY, True), (px.AGENT_LOOP_KEY, True), (px.THINKING_ON_KEY, True),
])
def test_every_key_outside_the_allowlist_refuses(key, value):
    proxy, backend = _proxy()
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, {**_chat(V[RESP_CELL]), key: value})
    assert ei.value.code == px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED and ei.value.http_status == 400
    assert repr(key) in str(ei.value) or key in str(ei.value)
    assert backend.payloads == []


def test_the_exact_length_controls_pass_and_a_render_changing_key_beside_them_still_refuses():
    """The closed loop's EXACT-LENGTH body (``min_tokens ==
    max_tokens`` + ``ignore_eos``, what ``closed_loop_via_proxy(exact_length=True)`` and the
    certification's ``_exact_length_supported`` probe send) is forwarded on a restricted-mode route
    — both are length controls, neither changes the render or the response boundary — and the
    known-bad: the SAME body with a render-changing key beside them (``truncate_prompt_tokens``)
    is still refused before forwarding, under the restricted-route code."""
    proxy, backend = _proxy()
    exact = {**_chat(V[RESP_CELL]), "max_tokens": 4, "min_tokens": 4, "ignore_eos": True,
             "temperature": 0.0, "seed": 0}
    _ask(proxy, exact)
    sent = backend.payloads[-1]
    assert sent["min_tokens"] == 4 and sent["ignore_eos"] is True and sent["max_tokens"] == 4
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, {**exact, "truncate_prompt_tokens": 8, "truncation_side": "right"})
    assert ei.value.http_status == 400 and ei.value.code == px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED
    assert "truncate_prompt_tokens" in str(ei.value) and len(backend.payloads) == 1   # never forwarded


def test_every_allowlisted_key_is_forwarded_on_a_restricted_route():
    proxy, backend = _proxy()
    req = {"model": V[RESP_CELL], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 4,
           "temperature": 0.0, "top_p": 1.0, "top_k": -1, "seed": 0, "stop": ["\n"], "n": 1,
           "logprobs": True, "top_logprobs": 5, "return_token_ids": True,
           "return_tokens_as_token_ids": True, "stream": False, "min_tokens": 4, "ignore_eos": True,
           px.CHAT_TEMPLATE_KWARGS_KEY: {px.ENABLE_THINKING_KEY: False}}
    out = _ask(proxy, req)
    sent = backend.payloads[-1]
    for k in req:
        assert k in sent, k
    assert sent["model"] == RESP_CELL and "cache_salt" in sent
    assert out["petri"][px.LEDGER_KEY_POSITION_MODE] == "response"
    assert out["petri"][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_CLEAN
    assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE


def test_chat_template_kwargs_beyond_enable_thinking_refuses_under_the_restricted_code():
    proxy, backend = _proxy()
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, {**_chat(V[RESP_CELL]),
                     px.CHAT_TEMPLATE_KWARGS_KEY: {px.ENABLE_THINKING_KEY: False, "custom": 1}})
    assert ei.value.code == px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED and "custom" in str(ei.value)
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, {**_chat(V[RESP_CELL]), px.CHAT_TEMPLATE_KWARGS_KEY: "not-an-object"})
    assert ei.value.code == px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED
    assert backend.payloads == []
    # the same bodies on the ALL-positions route fall to the thinking gate's own code (unchanged)
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, {**_chat(V[ALL_CELL]),
                     px.CHAT_TEMPLATE_KWARGS_KEY: {px.ENABLE_THINKING_KEY: False, "custom": 1}})
    assert ei.value.code == px.THINKING_KWARGS_INVALID


def test_the_seeded_shape_is_admitted_only_in_full():
    proxy, backend = _proxy()
    out = _ask(proxy, _seeded(V[RESP_CELL]))                       # the MMLU control's shape
    assert backend.payloads[-1]["continue_final_message"] is True
    assert backend.payloads[-1]["add_generation_prompt"] is False
    assert out["petri"][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_CLEAN
    bad = []
    r = _seeded(V[RESP_CELL]); del r["add_generation_prompt"]; bad.append(r)          # header not stated off
    r = _seeded(V[RESP_CELL]); r["messages"][-1]["role"] = "user"; bad.append(r)      # no assistant seed
    r = _seeded(V[RESP_CELL]); r["continue_final_message"] = False; bad.append(r)     # the default, stated
    r = _seeded(V[RESP_CELL]); r["continue_final_message"] = 1; bad.append(r)         # a truthy coercion
    r = _chat(V[RESP_CELL]); r["add_generation_prompt"] = False; bad.append(r)        # no header, no seed
    n_before = len(backend.payloads)
    for req in bad:
        with pytest.raises(px.ProxyRefusal) as ei:
            _ask(proxy, req)
        assert ei.value.code == px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED, req
    assert len(backend.payloads) == n_before


# --------------------------------------------------------------------------- #
# (ii) the request-scoped refusal; (iii) untouched routes; (iv) no id          #
# --------------------------------------------------------------------------- #


class _LedgerBackend(StubBackend):
    """A backend whose post-check IS the worker RPC's body: ``native_worker.sisteer_take_request_failure`` ->
    ``native_residual.take_request_failure`` -> ``RequestFailureLedger.claim`` on a REAL ledger."""

    def __init__(self, ledger, **kw):
        super().__init__(**kw)
        self.ledger = ledger

    async def take_request_failure(self, request_id):
        self.failure_probes.append(request_id)
        return self.ledger.claim(request_id)


def test_the_boundary_refusal_control_key_admits_exactly_the_headerless_shape_and_is_stripped_upstream():
    """O49: the proxy-only marker admits ONE body the allowlist otherwise refuses — a single user turn under
    add_generation_prompt: false — on a restricted-mode route only; it never travels upstream; with the worker's
    entry ledgered under the ENGINE id the completion is refused `response_boundary_unlocated`, and a 200 (no entry)
    is exactly the control's failure signal. Any other shape, or the key on an all-positions route, is a 400."""
    import secrets
    from sisteer.capeval.serving import native_residual as nr
    from sisteer.capeval.serving import serving_attestation as sa
    key = px.BOUNDARY_REFUSAL_CONTROL_KEY
    probe = {"model": V[RESP_CELL], "messages": [dict(m) for m in sa.BOUNDARY_REFUSAL_CONTROL_MESSAGES],
             **sa.BOUNDARY_REFUSAL_CONTROL_KWARGS}
    headerless = {**probe, "messages": [{"role": "user", "content": "x"}]}
    # without the marker: the seeded-shape rule refuses it (unchanged)
    proxy, backend = _proxy()
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, headerless)
    assert ei.value.http_status == 400 and ei.value.code == px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED
    # with the marker and the worker's ENGINE-id entry: refused by name; the marker never reached upstream
    led = nr.RequestFailureLedger()
    facts = {k: v for k, v in _entry("chatcmpl-ctl").items() if k != "req_id"}

    class _Backend(_LedgerBackend):
        async def chat_completions(self, payload):
            led.record(f"chatcmpl-ctl-{secrets.token_hex(4)}", **facts)     # the worker ledgers the header-less render
            return await super().chat_completions(payload)

    backend = _Backend(led, responses=[_response("chatcmpl-ctl")])
    proxy, _ = _proxy(backend)
    nonce = proxy.boundary_control_nonce
    assert isinstance(nonce, str) and len(nonce) == 32
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, {**probe, key: nonce})
    assert ei.value.http_status == 500 and ei.value.code == px.RESPONSE_BOUNDARY_UNLOCATED
    assert ei.value.extra["worker_error_type"] == "ResponseBoundaryUnlocated"       # the structured field the control reads
    assert key not in backend.payloads[-1] and backend.payloads[-1]["add_generation_prompt"] is False
    assert backend.failure_probes == ["chatcmpl-ctl"] and len(led) == 0
    # the broken join / worker: nothing ledgered -> the completion comes back 200 (the control's FAILURE signal)
    backend2 = _LedgerBackend(nr.RequestFailureLedger(), responses=[_response("chatcmpl-ctl2")])
    proxy2, _ = _proxy(backend2)
    out = _ask(proxy2, {**probe, key: proxy2.boundary_control_nonce})
    body = out if isinstance(out, dict) else out[-1]
    assert body["id"] == "chatcmpl-ctl2"
    # KNOWN-BAD: the marker beside ANY other body is refused by name — above all the
    # [user, assistant, user] transcript whose EARLIER assistant header the locator would find (served steered,
    # no boundary failure, a completion stamped as the cell): the body is compared WHOLE against the constant
    escape = {**probe, "messages": [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
                                    {"role": "user", "content": sa.BOUNDARY_REFUSAL_CONTROL_PROMPT}]}
    for bad in (escape, headerless, {**probe, "messages": [{"role": "user", "content": "x"}]},
                {**probe, "add_generation_prompt": True}, {**probe, "continue_final_message": True},
                {**probe, "messages": [{"role": "assistant", "content": sa.BOUNDARY_REFUSAL_CONTROL_PROMPT}]},
                {**probe, "n": 2}, {**probe, "max_tokens": 64}, {**probe, "temperature": 0.7}, {**probe, "stop": ["x"]},
                {**probe, "messages": [dict(m) for m in sa.BOUNDARY_REFUSAL_CONTROL_MESSAGES] * 2}):
        proxy3, backend3 = _proxy(_LedgerBackend(nr.RequestFailureLedger(), responses=[_response("chatcmpl-esc")]))
        with pytest.raises(px.ProxyRefusal) as ei3:
            _ask(proxy3, {**bad, key: proxy3.boundary_control_nonce})
        assert ei3.value.http_status == 400 and ei3.value.code == px.BOUNDARY_CONTROL_SHAPE_INVALID, bad
        assert backend3.payloads == []                                              # nothing reached the engine
    # the exact body with a marker that is not THIS proxy's nonce: 400, named — a client cannot present it
    for token in (True, "yes", "", proxy.boundary_control_nonce + "0", nonce[::-1]):
        proxy5, backend5 = _proxy(_LedgerBackend(nr.RequestFailureLedger(), responses=[_response("chatcmpl-n")]))
        with pytest.raises(px.ProxyRefusal) as ei5:
            _ask(proxy5, {**probe, key: token})
        assert ei5.value.http_status == 400 and ei5.value.code == px.BOUNDARY_CONTROL_NONCE_INVALID, token
        assert backend5.payloads == []
    # the marker on an all-positions route: 400 boundary_control_not_restricted
    proxy4, _ = _proxy()
    with pytest.raises(px.ProxyRefusal) as ei4:
        _ask(proxy4, {**probe, key: proxy4.boundary_control_nonce, "model": V[ALL_CELL]})
    assert ei4.value.http_status == 400 and ei4.value.code == px.BOUNDARY_CONTROL_NOT_RESTRICTED


def test_end_to_end_a_failure_ledgered_under_the_engine_id_refuses_the_completion_claimed_by_its_own_id():
    """The worker ledgers a boundary failure under the SCHEDULER's id — at vLLM 0.25.1 the
    completion id plus a random `-<8 hex>` suffix (InputProcessor.assign_request_id), `<i>_`-prefixed for an n>1
    child (ParentRequest.get_child_info) — while the proxy claims by the completion's own `id`. With the old
    exact-key pop every claim read clean and the UNSTEERED completion went out stamped as the restricted cell;
    now the completion is REFUSED, named (500 response_boundary_unlocated), and a completion that owns no failure
    is forwarded while another's failure stays ledgered."""
    import secrets
    from sisteer.capeval.serving import native_residual as nr
    led = nr.RequestFailureLedger()
    facts = {k: v for k, v in _entry("chatcmpl-x").items() if k != "req_id"}
    eng_x = f"chatcmpl-x-{secrets.token_hex(4)}"
    led.record(eng_x, **facts)                                              # n == 1
    led.record(f"1_chatcmpl-y-{secrets.token_hex(4)}", **facts)             # one failed child of an n == 2 completion
    assert "chatcmpl-x" not in led._entries and eng_x in led._entries       # the exact-key world would have read clean
    backend = _LedgerBackend(led, responses=[_response("chatcmpl-x")])
    proxy, _ = _proxy(backend)
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, _chat(V[RESP_CELL]))
    assert ei.value.http_status == 500 and ei.value.code == px.RESPONSE_BOUNDARY_UNLOCATED
    assert "chatcmpl-x" in str(ei.value) and "ResponseBoundaryUnlocated" in str(ei.value) and "never returned" in str(ei.value)
    assert backend.failure_probes == ["chatcmpl-x"] and len(led) == 1       # x claimed by the completion id; y's stays
    # the n>1 completion: its failed child condemns the whole completion
    backend2 = _LedgerBackend(led, responses=[_response("chatcmpl-y")])
    proxy2, _ = _proxy(backend2)
    with pytest.raises(px.ProxyRefusal) as ei2:
        _ask(proxy2, _chat(V[RESP_CELL]))
    assert ei2.value.code == px.RESPONSE_BOUNDARY_UNLOCATED and len(led) == 0
    # a completion that owns no failure is forwarded (the ledger is empty; a fresh claim is None, not a fault)
    backend3 = _LedgerBackend(led, responses=[_response("chatcmpl-z")])
    proxy3, _ = _proxy(backend3)
    out = _ask(proxy3, _chat(V[RESP_CELL]))
    body = out if isinstance(out, dict) else out[-1]
    assert body["id"] == "chatcmpl-z" and backend3.failure_probes == ["chatcmpl-z"]


def test_an_unlocated_completion_is_refused_never_returned():
    backend = StubBackend(responses=[_response("chatcmpl-unlocated")],
                          request_failures={"chatcmpl-unlocated": _entry("chatcmpl-unlocated")})
    proxy, _ = _proxy(backend)
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, _chat(V[RESP_CELL]))
    assert ei.value.http_status == 500 and ei.value.code == px.RESPONSE_BOUNDARY_UNLOCATED
    msg = str(ei.value)
    assert "ResponseBoundaryUnlocated" in msg and "NOWHERE" in msg and "never returned" in msg
    assert "prompt_len=7" in msg and "header_len=4" in msg
    assert backend.failure_probes == ["chatcmpl-unlocated"]         # claimed by the completion id
    # the audit row is the refused request's record: it names why, and the mode
    row = proxy.request_log[-1]
    assert row[px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_UNLOCATED
    assert row[px.LEDGER_KEY_POSITION_MODE] == "response"
    assert row[px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE


def test_a_clean_restricted_completion_is_claimed_and_served_with_the_stamp():
    backend = StubBackend(responses=[_response("chatcmpl-clean", "steered")])
    proxy, _ = _proxy(backend)
    out = _ask(proxy, _chat(V[RESP_CELL]))
    assert out["choices"][0]["message"]["content"] == "steered"
    assert backend.failure_probes == ["chatcmpl-clean"]
    assert out["model"] == V[RESP_CELL] and out["petri"]["cell_id"] == RESP_CELL
    assert out["petri"][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_CLEAN
    assert out["petri"][px.LEDGER_KEY_POSITION_MODE] == "response"
    assert proxy.request_log[-1][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_CLEAN


def test_all_positions_and_lora_routes_are_untouched():
    """(iii): the same two requests — truncate_prompt_tokens, and one the worker
    would mark unlocated — are served unchanged on an all-positions PSR route and
    on a LoRA route: forwarded wholesale, NO claim, cell-stable salt."""
    for cell, mode in ((ALL_CELL, "all"), (LORA_CELL, None)):
        backend = StubBackend(responses=[_response("chatcmpl-a"), _response("chatcmpl-b")],
                              request_failures={"chatcmpl-b": _entry("chatcmpl-b", cell)})
        proxy, _ = _proxy(backend)
        out_a = _ask(proxy, {**_chat(V[cell]), "truncate_prompt_tokens": 8, "truncation_side": "right"})
        assert backend.payloads[-1]["truncate_prompt_tokens"] == 8          # forwarded as before
        assert backend.payloads[-1]["truncation_side"] == "right"
        out_b = _ask(proxy, _chat(V[cell]))
        assert out_b["id"] == "chatcmpl-b"                                  # served: no claim ran
        assert backend.failure_probes == []
        assert backend.request_failures == {"chatcmpl-b": _entry("chatcmpl-b", cell)}   # never popped
        for out in (out_a, out_b):
            assert out["petri"][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_NOT_APPLICABLE
            assert out["petri"][px.LEDGER_KEY_POSITION_MODE] == mode
            assert out["petri"][px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_CELL_STABLE
        assert backend.payloads[0]["cache_salt"] == backend.payloads[1]["cache_salt"]
    # the base route: no mode, no claim
    proxy, backend = _proxy()
    out = _ask(proxy, _chat(V["none"]))
    assert out["petri"][px.LEDGER_KEY_POSITION_MODE] is None
    assert out["petri"][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_NOT_APPLICABLE


def test_a_restricted_response_without_an_id_refuses():
    resp = _response("chatcmpl-x")
    del resp["id"]
    backend = StubBackend(responses=[resp])
    proxy, _ = _proxy(backend)
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, _chat(V[RESP_CELL]))
    assert ei.value.http_status == 502 and ei.value.code == px.RESPONSE_BOUNDARY_POSTCHECK_FAILED
    assert "no 'id'" in str(ei.value)
    assert proxy.request_log[-1][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_UNVERIFIABLE
    # the same id-less response on the all-positions route is served (no claim to make)
    backend = StubBackend(responses=[dict(resp)])
    proxy, _ = _proxy(backend)
    assert _ask(proxy, _chat(V[ALL_CELL]))["petri"]["cell_id"] == ALL_CELL


@pytest.mark.parametrize("payload", [None, "", {}, [], {"results": []}, {"results": None}, {"results": "x"},
                                     {"results": [1]}, {"results": ["ok"]}, {"result": [None]}])
def test_known_bad_a_claim_reply_without_a_per_worker_result_list_refuses(payload):
    """The collective_rpc reply used to read as "no failure" for an empty body, null, {},
    {"results": []} and every other shape without a per-worker list — a fail-closed hole on the restricted
    route (FAILURE_CLAIM_CLEAN recorded). Only a NON-EMPTY list whose entries are None or a Mapping
    establishes a claim; anything else refuses, named."""
    with pytest.raises(px.ProxyRefusal) as ei:
        px.parse_failure_claim(payload, url="http://u/collective_rpc", request_id="chatcmpl-x")
    assert ei.value.http_status == 502 and ei.value.code == "lens_postcheck_failed"
    assert "without a per-worker result list" in str(ei.value)


@pytest.mark.parametrize("results", [[None], [None, None, None], [None] * 5, [None, None, None, "x"], [None, None, None, 1],
                                     [None, None, {"error_type": "x"}, 2], [{"error_type": "x"}]])
def test_known_bad_a_claim_reply_short_of_the_attested_worker_count_or_with_a_foreign_entry_refuses(results):
    """On a four-worker server {"results": [null]} is ONE worker's word — the list must carry exactly
    the attested worker count, every entry None (that worker's "no failure") or a Mapping (its claim); fewer, more or
    a foreign entry refuses, named, 502 lens_postcheck_failed (the native route re-labels it postcheck_failed)."""
    with pytest.raises(px.ProxyRefusal) as ei:
        px.parse_failure_claim({"results": results}, url="u", request_id="r", n_workers=4)
    assert ei.value.http_status == 502 and ei.value.code == "lens_postcheck_failed"
    assert ("attestation names 4 worker(s)" in str(ei.value)) or ("without a per-worker result list" in str(ei.value))


def test_a_claim_reply_matching_the_attested_worker_count_is_read_and_the_proxy_binds_the_count_from_the_attestation():
    # every worker's None IS the contract's "no failure"; the first non-None entry is the claim
    assert px.parse_failure_claim({"results": [None] * 4}, url="u", request_id="r", n_workers=4) is None
    entry = {"error_type": "ResponseBoundaryUnlocated"}
    assert px.parse_failure_claim({"results": [None, entry, None, None]}, url="u", request_id="r", n_workers=4) == entry
    # the RoutingProxy hands the attested count to a backend that exposes `expected_workers`
    routes = _routes()
    blocks = _blocks(routes)
    att = _att(blocks)
    att["residual_slot"] = {"armed": True, "n_workers": 4}
    backend = StubBackend()
    px.RoutingProxy(routes, backend, base_model="org/model", serving_mode="interleaved",
                    adapter_blocks=blocks, engine_attestation=att, agent_loop_render=_TailRender())
    assert backend.expected_workers == 4
    # KNOWN-BAD: an ARMED slot WITHOUT the count is refused at start — the count-free shape is
    # the UNARMED schema-1 identity's only (nothing on such an engine ever raises a claim to read)
    armed_no_count = _att(blocks); armed_no_count["residual_slot"] = {"armed": True}
    with pytest.raises(px.ProxyError, match="armed but carries no n_workers"):
        px.RoutingProxy(routes, StubBackend(), base_model="org/model", serving_mode="interleaved",
                        adapter_blocks=blocks, engine_attestation=armed_no_count, agent_loop_render=_TailRender())
    # (the UNARMED count-free shape {"armed": False} stays admitted — pinned by test_lora_route_adapter_block's
    # fixtures, whose adapters carry no residual payload)
    # the attestation's shape: {armed False} or {armed, n_workers >= 1}; anything else refuses by name
    for bad in ({"armed": True}, {"armed": True, "n_workers": 0}, {"armed": True, "n_workers": True}, {"armed": True, "n_workers": "4"},
                {"armed": True, "workers": 2}):
        bad_att = _att(blocks); bad_att["residual_slot"] = bad
        with pytest.raises(px.ProxyError, match="residual_slot"):
            px.validated_engine_attestation(bad_att)


def test_a_claim_reply_with_a_per_worker_list_is_read_first_non_none_wins():
    assert px.parse_failure_claim({"results": [None]}, url="u", request_id="r") is None
    assert px.parse_failure_claim({"results": [None, None, None, None]}, url="u", request_id="r") is None
    entry = {"error_type": "ResponseBoundaryUnlocated", "prompt_len": 5}
    assert px.parse_failure_claim({"results": [None, entry]}, url="u", request_id="r") == entry
    fault = {"fault": "ledger_overflow", "unclaimed": 1024}
    assert px.parse_failure_claim({"results": [fault, None]}, url="u", request_id="r") == fault


def test_an_unverifiable_claim_refuses_the_restricted_completion_under_its_own_code():
    """The backend's refusal (a malformed reply, a transport failure) reaches the restricted route as
    response_boundary_postcheck_failed (502) — never as a clean claim, never forwarded."""
    class MalformedReplyBackend(StubBackend):
        async def take_request_failure(self, request_id):
            self.failure_probes.append(request_id)
            return px.parse_failure_claim({"results": []}, url="http://u/collective_rpc", request_id=request_id)

    backend = MalformedReplyBackend()
    proxy, _ = _proxy(backend)
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, _chat(V[RESP_CELL]))
    assert ei.value.http_status == 502 and ei.value.code == px.RESPONSE_BOUNDARY_POSTCHECK_FAILED
    assert "without a per-worker result list" in str(ei.value) and backend.failure_probes
    _ask(proxy, _chat(V[ALL_CELL]))                                  # no claim on an all-positions route


def test_a_backend_without_the_claim_refuses_restricted_routes_only():
    class BareBackend(StubBackend):
        take_request_failure = None

    backend = BareBackend()
    proxy, _ = _proxy(backend)
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, _chat(V[RESP_CELL]))
    assert ei.value.code == px.RESPONSE_BOUNDARY_POSTCHECK_FAILED and "post-check" in str(ei.value)
    _ask(proxy, _chat(V[ALL_CELL]))
    _ask(proxy, _chat(V[LORA_CELL]))
    _ask(proxy, _chat(V["none"]))


# --------------------------------------------------------------------------- #
# (v) ledger overflow AFTER readiness — the worker's own ledger class            #
# --------------------------------------------------------------------------- #


def test_ledger_overflow_after_readiness_refuses_fresh_and_oldest_alike_and_all_positions_keeps_serving():
    from sisteer.capeval.serving import native_residual as nr

    cap = 4
    ledger = nr.RequestFailureLedger(cap=cap)
    assert ledger.report() == {"ledger_size": 0, "ledger_fault": False, "ledger_cap": cap}   # readiness
    # the worker records cap failures nobody claims (the proxy post-check not running), then one more
    oldest = "chatcmpl-old-0"
    for i in range(cap):
        ledger.record(f"chatcmpl-old-{i}", cell=RESP_CELL, error_type="ResponseBoundaryUnlocated",
                      prompt_len=7, header_len=4, detail="header NOWHERE")
    ledger.record("chatcmpl-old-overflow", cell=RESP_CELL, error_type="ResponseBoundaryUnlocated",
                  prompt_len=7, header_len=4, detail="header NOWHERE")
    assert ledger.fault is True and len(ledger) == cap                    # latched; nothing evicted
    backend = _LedgerBackend(ledger, responses=[_response("chatcmpl-fresh"), _response(oldest),
                                                _response("chatcmpl-all-1"), _response("chatcmpl-all-2")])
    proxy, _ = _proxy(backend)
    # a FRESH request on the restricted route: the fault payload -> refused
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, _chat(V[RESP_CELL]))
    assert ei.value.http_status == 500 and ei.value.code == px.RESPONSE_BOUNDARY_LEDGER_FAULT
    assert "ledger_overflow" in str(ei.value) and "unclaimed=4" in str(ei.value)
    assert proxy.request_log[-1][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_LEDGER_FAULT
    # the OLDEST failed request's LATE completion: still refused — its entry is still present
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, _chat(V[RESP_CELL]))
    assert ei.value.code == px.RESPONSE_BOUNDARY_LEDGER_FAULT
    assert oldest in ledger._entries and len(ledger) == cap               # never evicted, never expired
    # an ALL-POSITIONS route keeps serving through the fault (no claim on it)
    out = _ask(proxy, _chat(V[ALL_CELL]))
    assert out["id"] == "chatcmpl-all-1" and out["petri"]["cell_id"] == ALL_CELL
    assert _ask(proxy, _chat(V[LORA_CELL]))["id"] == "chatcmpl-all-2"
    assert backend.failure_probes == ["chatcmpl-fresh", oldest]
    # the readiness fact the standup reads after the controls
    assert ledger.report() == {"ledger_size": cap, "ledger_fault": True, "ledger_cap": cap}
    assert ns.ledger_problems([ledger.report()]) and len(ns.ledger_problems([ledger.report()])) == 2


class _EvictingLedger:
    """rev 3's hole: a bounded ledger that DROPS the oldest unclaimed
    entry on overflow and counts the drop — a later claim for that request reads
    None, "no failure", and the unsteered completion would be filed as served."""

    def __init__(self, cap):
        self.cap, self.entries, self.dropped = cap, {}, 0

    def record(self, rid, **facts):
        if len(self.entries) >= self.cap:
            self.entries.pop(next(iter(self.entries)))
            self.dropped += 1
        self.entries[rid] = {"req_id": rid, **facts}

    def claim(self, rid):
        return self.entries.pop(rid, None)


def test_known_bad_an_evicting_ledger_lets_the_oldest_unsteered_completion_through():
    """rev 3's ``ledger_dropped`` counter, pinned as the failure the design forbids:
    with an EVICTING ledger the proxy reads None for the oldest failed request and
    RETURNS its completion stamped as the cell — the proxy alone cannot tell an
    evicted failure from no failure, which is why the worker's ledger LATCHES."""
    cap = 4
    ledger = _EvictingLedger(cap)
    oldest = "chatcmpl-old-0"
    for i in range(cap + 1):
        ledger.record(f"chatcmpl-old-{i}", cell=RESP_CELL, error_type="ResponseBoundaryUnlocated")
    assert ledger.dropped == 1 and oldest not in ledger.entries
    backend = _LedgerBackend(ledger, responses=[_response(oldest, "UNSTEERED text")])
    proxy, _ = _proxy(backend)
    out = _ask(proxy, _chat(V[RESP_CELL]))                 # WOULD be returned — the hole
    assert out["id"] == oldest and out["choices"][0]["message"]["content"] == "UNSTEERED text"
    assert out["petri"]["cell_id"] == RESP_CELL and out["petri"][px.LEDGER_KEY_FAILURE_CLAIM] == "clean"
    # ... and the still-present entries DO refuse: the proxy's half works, the ledger's half is the guard
    backend.responses.append(_response("chatcmpl-old-1"))
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, _chat(V[RESP_CELL]))
    assert ei.value.code == px.RESPONSE_BOUNDARY_UNLOCATED


# --------------------------------------------------------------------------- #
# the completions endpoint; the lens rows carry the label too                  #
# --------------------------------------------------------------------------- #


def test_completions_endpoint_refuses_with_the_restricted_code_on_a_restricted_route():
    proxy, backend = _proxy()

    def ask(body):
        return asyncio.run(proxy.handle_completions(body))

    with pytest.raises(px.ProxyRefusal) as ei:
        ask({"model": V[RESP_CELL], "prompt": "raw"})
    assert ei.value.http_status == 400 and ei.value.code == px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED
    for body in ({"model": V[ALL_CELL], "prompt": "raw"}, {"model": V["none"], "prompt": "raw"},
                 {"model": "petri__unknown"}, {}):
        with pytest.raises(px.ProxyRefusal) as ei:
            ask(body)
        assert ei.value.http_status == 404 and ei.value.code == px.COMPLETIONS_UNSUPPORTED, body
    assert backend.payloads == []
    web = pytest.importorskip("aiohttp.web")  # noqa: F841 — the HTTP surface mounts the route
    app = px.make_app(proxy)
    paths = {r.resource.canonical for r in app.router.routes() if hasattr(r, "resource")}
    assert "/v1/completions" in paths


def test_completions_endpoint_holds_the_token_rule():
    proxy, _ = _proxy()
    proxy._api_token = "tok"
    with pytest.raises(px.ProxyRefusal) as ei:
        asyncio.run(proxy.handle_completions({"model": V[RESP_CELL]}, authorization="Bearer wrong"))
    assert ei.value.code == "bad_token" and ei.value.http_status == 401


def test_a_lens_row_records_a_clean_claim_and_no_native_mode(tmp_path):
    from test_petri_proxy import _proxy as lens_proxy
    backend = StubBackend(responses=[_response("chatcmpl-lens")])
    proxy, _, _ = lens_proxy(tmp_path, backend=backend)
    out = _ask(proxy, _chat("petri__si_constitution__cast"))
    assert out["petri"][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_CLEAN
    assert out["petri"][px.LEDGER_KEY_POSITION_MODE] is None
    assert proxy.request_log[-1][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_CLEAN
    # a poisoned lens response keeps its codes and labels its row
    from test_petri_proxy import _FAILURE_RECORD
    backend = StubBackend(responses=[_response("chatcmpl-p")],
                          request_failures={"chatcmpl-p": dict(_FAILURE_RECORD)})
    proxy, _, _ = lens_proxy(tmp_path, backend=backend)
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, _chat("petri__si_constitution__cast"))
    assert ei.value.code == "lens_request_failed"
    assert proxy.request_log[-1][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_UNLOCATED


# =========================================================================== #
# the standup half: the REAL bring-up over the scripted upstream               #
# =========================================================================== #

from test_native_standup_stage2 import (RESTRICTED_PSR_CELL, RESTRICTED_PSR_HCELL,  # noqa: E402
                                        ALL_PSR_CELL, ALL_PSR_HCELL, UNLOCATED_MARKER,
                                        bring_up_stage2_standup)
from test_lora_serving_up import GEMMA_HCELL, lsu  # noqa: E402,F401


def _post(su, payload):
    """One chat through the live proxy; returns ``(status, body)`` — the refusal's
    JSON error body when the proxy refused (``LoraStandupError`` carries it)."""
    try:
        return 200, su.chat(payload)
    except ns.ProxyHTTPRefusal as e:           # the structured refusal: status + the JSON error body
        return e.status, json.loads(e.body_text)
    except ns.LoraStandupError as e:
        m = re.search(r"HTTP (\d+): (.*)$", str(e), re.S)
        assert m, str(e)
        return int(m.group(1)), json.loads(m.group(2))


def _code(body):
    return body["error"]["code"]


def test_stage2_standup_serves_a_restricted_psr_and_the_proxy_holds_it_to_the_allowlist_and_the_ledger(
        lsu, tmp_path, monkeypatch):
    su, cap = bring_up_stage2_standup(lsu, tmp_path, monkeypatch,
                                      failures_by_model={RESTRICTED_PSR_HCELL: {"marker": UNLOCATED_MARKER}})
    try:
        assert su.residual_slot["cells"][RESTRICTED_PSR_HCELL]["position_mode"] == "response"
        assert su.residual_slot["cells"][ALL_PSR_HCELL]["position_mode"] == "all"
        assert su.proxy.restricted_position_mode(su.routes_by_cell[RESTRICTED_PSR_HCELL]) == "response"
        assert su.proxy.restricted_position_mode(su.routes_by_cell[ALL_PSR_HCELL]) is None
        # the controls passed: the restricted cell's bites statistic is the logprob liveness
        pc = json.loads((cap["out"] / "positive_controls.json").read_text())
        bites = pc["controls"]["steering_bites"]
        assert bites["passed"] is True
        assert bites["cells"][RESTRICTED_PSR_HCELL]["statistic"] == ns.BITES_STATISTIC_LOGPROB
        assert bites["cells"][RESTRICTED_PSR_HCELL]["logprob_liveness"]["verified"] is True
        assert bites["cells"][ALL_PSR_HCELL]["statistic"] == ns.BITES_STATISTIC_TEXT
        assert bites["cells"][GEMMA_HCELL]["statistic"] == ns.BITES_STATISTIC_TEXT
        # the ledger read after the controls: clean, recorded in both records
        ep = json.loads((cap["out"] / "endpoint.json").read_text())
        cs = json.loads((cap["out"] / "cells_served.json").read_text())
        for rec in (ep, cs):
            led = rec[ns.RESIDUAL_LEDGER_AFTER_CONTROLS_KEY]
            assert led["checked"] is True and led["unclaimed"] == 0 and led["ledger_fault"] is False
            assert led["problems"] == []
        rv, av, lv = (su.vmodels[c] for c in (RESTRICTED_PSR_HCELL, ALL_PSR_HCELL, GEMMA_HCELL))
        chat_log = cap["chat_log"]
        n_seen = len(chat_log.read_text().splitlines())
        # (i) the allowlist refuses BEFORE forwarding — the upstream never saw the body
        status, body = _post(su, {"model": rv, "messages": [{"role": "user", "content": "hi"}],
                                  "max_tokens": 2, "truncate_prompt_tokens": 8, "truncation_side": "right"})
        assert status == 400 and _code(body) == px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED
        assert len(chat_log.read_text().splitlines()) == n_seen
        # (ii) a request the worker marks UNLOCATED: refused, never returned as the cell
        status, body = _post(su, {"model": rv, "max_tokens": 2,
                                  "messages": [{"role": "user", "content": f"hi {UNLOCATED_MARKER}"}]})
        assert status == 500 and _code(body) == px.RESPONSE_BOUNDARY_UNLOCATED
        assert "NOWHERE" in body["error"]["message"]
        # the worker's ledger is clean again: the proxy claimed the entry
        assert ns.unclaimed_failures_problems(su) == []
        # (iii) the same two bodies on the all-positions PSR route and the LoRA route: served
        for vm in (av, lv):
            status, body = _post(su, {"model": vm, "messages": [{"role": "user", "content": "hi"}],
                                      "max_tokens": 2, "truncate_prompt_tokens": 8, "truncation_side": "right"})
            assert status == 200 and body["petri"][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_NOT_APPLICABLE
            last = json.loads(chat_log.read_text().splitlines()[-1])
            assert last["payload"]["truncate_prompt_tokens"] == 8          # forwarded wholesale
            status, body = _post(su, {"model": vm, "max_tokens": 2,
                                      "messages": [{"role": "user", "content": f"hi {UNLOCATED_MARKER}"}]})
            assert status == 200 and body["petri"]["cell_id"] in (ALL_PSR_HCELL, GEMMA_HCELL)
        # the seeded shape (the MMLU control's) is served on the restricted route, stamped
        status, body = _post(su, {"model": rv, "max_tokens": 2,
                                  "messages": [{"role": "user", "content": "Q?"},
                                               {"role": "assistant", "content": "Answer:"}],
                                  "continue_final_message": True, "add_generation_prompt": False})
        assert status == 200
        assert body["petri"][px.LEDGER_KEY_SALT_REGIME] == px.SALT_REGIME_REQUEST_UNIQUE
        assert body["petri"][px.LEDGER_KEY_POSITION_MODE] == "response"
        assert body["petri"][px.LEDGER_KEY_FAILURE_CLAIM] == px.FAILURE_CLAIM_CLEAN
        # two identical requests on the restricted route mint DISTINCT salts; the all route one salt
        for vm, distinct in ((rv, True), (av, False)):
            for _ in range(2):
                assert _post(su, {"model": vm, "messages": [{"role": "user", "content": "same"}],
                                  "max_tokens": 2})[0] == 200
            salts = [json.loads(l)["payload"]["cache_salt"] for l in chat_log.read_text().splitlines()[-2:]]
            assert (salts[0] != salts[1]) is distinct, (vm, salts)
        # the completions endpoint through the live app
        import urllib.error
        import urllib.request
        req = urllib.request.Request(f"{su.proxy_url}/completions", method="POST",
                                     data=json.dumps({"model": rv, "prompt": "raw"}).encode(),
                                     headers={"Content-Type": "application/json"})
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(req, timeout=30)
        assert ei.value.code == 400
        assert json.loads(ei.value.read())["error"]["code"] == px.RESTRICTED_ROUTE_OPTION_UNSUPPORTED
    finally:
        su.tear_down()


def test_stage2_ledger_fault_after_readiness_refuses_every_restricted_completion(lsu, tmp_path, monkeypatch):
    """(v) end to end: readiness read a clean ledger; the worker latches its fault
    afterwards (every claim answers the fault payload) — the controls' first
    completion on the restricted route is refused ``response_boundary_ledger_fault``
    and the standup FAILS its controls rather than publishing."""
    with pytest.raises(lsu.PositiveControlsFailed) as ei:
        bring_up_stage2_standup(lsu, tmp_path, monkeypatch, ledger_fault=True)
    assert px.RESPONSE_BOUNDARY_LEDGER_FAULT in str(ei.value)
    rec = ei.value.record
    assert rec["failed_control"] == "steering_bites"          # the first control that reaches the psr route
    assert not (tmp_path / "out" / "endpoint.json").exists()
