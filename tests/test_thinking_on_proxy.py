"""The proxy half of thinking-ON serving (docs/served_thinking_on.md): the
``thinking_mode`` constructor contract, the ``petri_thinking_on`` refusal
matrix on ON and OFF proxies, the selector injection rule (declared cells on a
moved-header template only), the ``render_regime`` stamp, the identity
block's new fields, the routing table's fact, and — load-bearing for the live
wave — the OFF proxy's BYTE-IDENTITY: a pre-landing request produces the same
upstream body, audit row and response as before.

All offline, all synthetic (the proxy tests' canonical cell set, invented
header ids, placeholder digests — no real recorded data).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.serving import serve_declaration as sd  # noqa: E402

from test_petri_proxy import (_TailRender, StubBackend, _behavior_files,  # noqa: E402
                              _chat, _source, _spec)

OFF = [901, 902, 903, 904, 905, 906, 907]
ON = [901, 902, 903, 904, 908]
MOVES = {"switch": True, "moves": True, "off_ids": OFF, "on_ids": ON}
STAYS = {"switch": True, "moves": False, "off_ids": OFF, "on_ids": OFF}
NO_SWITCH = {"switch": False, "moves": None, "off_ids": OFF, "on_ids": None}
PARTIAL = {"switch": True, "moves": None, "off_ids": OFF, "on_ids": None}
LENS = "si_safety__psr_kl_all"          # a lens-served cell of the canonical set
LENS2 = "si_constitution__psr_kl_all"


def _ask(proxy, request):
    return asyncio.run(proxy.handle_chat_completions(request))


def _cells(tmp_path):
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = dc.enumerate_cells(spec, source, behavior_files=_behavior_files(spec),
                               prompt_files={"auditor": b"# a\n", "judge": b"# j\n"})
    return spec, source, cells


def _vmodel(cells, cell_id):
    return next(c.virtual_model_name for c in cells.canonical + cells.controls
                if c.cell_id == cell_id)


def _proxy(tmp_path, **kw):
    spec, source, cells = _cells(tmp_path)
    backend = StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend, base_model=spec.target_model,
        serving_mode="interleaved", prompted_prefixes=px.prompted_prefix_specs(cells, source),
        agent_loop_render=_TailRender(), **kw)
    return proxy, backend, cells


def _on_proxy(tmp_path, header=MOVES, declared=(LENS, dc.NOOP_LENS_CELL_ID), **kw):
    decls = {c: sd.thinking_declaration() for c in declared}
    return _proxy(tmp_path, thinking_mode="on", thinking_header=header,
                  lens_serve_declarations=decls, **kw)


def _req(vm, **extra):
    r = _chat(vm)
    r.update(extra)
    return r


def _refuses(proxy, request, code):
    with pytest.raises(px.ProxyRefusal) as ei:
        _ask(proxy, request)
    assert ei.value.code == code, (ei.value.code, str(ei.value))
    return ei.value


# --------------------------------------------------------------------------- #
# the constructor contract                                                      #
# --------------------------------------------------------------------------- #


def test_off_is_the_default_and_on_needs_a_certified_fact_and_a_map(tmp_path):
    proxy, _, _ = _proxy(tmp_path)
    assert proxy.thinking_mode == "off"
    with pytest.raises(px.ProxyError, match="not one of"):
        _proxy(tmp_path, thinking_mode="maybe")
    with pytest.raises(px.ProxyError, match="thinking-header fact"):
        _proxy(tmp_path, thinking_mode="on", lens_serve_declarations={})
    with pytest.raises(px.ProxyError, match="thinking-header fact"):
        _proxy(tmp_path, thinking_mode="on", thinking_header=NO_SWITCH, lens_serve_declarations={})
    with pytest.raises(px.ProxyError, match="UNAVAILABLE"):
        _proxy(tmp_path, thinking_mode="on", thinking_header=PARTIAL,
               thinking_header_error="ChatMarkerError: x", lens_serve_declarations={})
    with pytest.raises(px.ProxyError, match="no attested"):
        _proxy(tmp_path, thinking_mode="on", thinking_header=MOVES)
    # an EMPTY attested map is a stated fact — legal (this wave's all-fit cells)
    proxy, _, _ = _proxy(tmp_path, thinking_mode="on", thinking_header=MOVES, lens_serve_declarations={})
    assert proxy.thinking_mode == "on" and proxy.lens_serve_declarations == {}
    # the fact's shape is validated on every proxy
    with pytest.raises(px.ProxyError, match="exactly the keys"):
        _proxy(tmp_path, thinking_header={"switch": True})
    with pytest.raises(px.ProxyError, match="moves must be"):
        _proxy(tmp_path, thinking_header={**MOVES, "moves": False})
    with pytest.raises(px.ProxyError, match="needs moves None"):
        _proxy(tmp_path, thinking_header={**PARTIAL, "moves": True}, thinking_header_error="e")
    # the fact/error pairing, both directions
    with pytest.raises(px.ProxyError, match="carries the reason"):
        _proxy(tmp_path, thinking_header=PARTIAL)
    with pytest.raises(px.ProxyError, match="contradiction"):
        _proxy(tmp_path, thinking_header=MOVES, thinking_header_error="why")
    # the map is validated
    with pytest.raises(px.ProxyError, match="serve_out_of_support"):
        _proxy(tmp_path, lens_serve_declarations={LENS: {"serve_out_of_support": ["nope"]}})
    with pytest.raises(px.ProxyError, match="contradiction"):
        _proxy(tmp_path, lens_serve_declarations={}, lens_serve_declarations_error="why")


# --------------------------------------------------------------------------- #
# the OFF proxy: byte-identical, refuses only the new key                       #
# --------------------------------------------------------------------------- #


def test_off_proxy_is_byte_identical_for_pre_landing_requests(tmp_path):
    """The OFF proxy (the default) is the per-request-regime proxy main serves
    every instrument through: a request that says nothing about thinking is
    forwarded with no declaration and no selector fact, and its audit row and
    response stamp record the regime as UNSPECIFIED (main's contract since the
    fit-support-transfer landing — the stamp is always present, never absent)."""
    proxy, backend, cells = _proxy(tmp_path)
    for cell in (LENS, "none", "prompted__safety", dc.NOOP_LENS_CELL_ID):
        vm = _vmodel(cells, cell)
        req = _chat(vm, system="synthetic safety instruction: be safe (test bytes)"
                    if cell == "prompted__safety" else None)
        out = _ask(proxy, req)
        body = backend.payloads[-1]
        assert px.THINKING_ON_KEY not in body
        if px.UPSTREAM_EXTRA_ARGS_KEY in body:
            assert "thinking_on" not in body[px.UPSTREAM_EXTRA_ARGS_KEY]
        assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == px.RENDER_REGIME_UNSPECIFIED
        assert proxy.request_log[-1][px.LEDGER_KEY_RENDER_REGIME] == px.RENDER_REGIME_UNSPECIFIED
        assert {"virtual_model", "cell_id", "serving_identity", "intervention_hash"} <= set(out["petri"])
    # an undeclared ON render on a lens route: main's OFF proxy FAILS CLOSED without a
    # served-positions fact for the cell (the sampled lineage passed it through), and admits
    # it — forwarding chat_template_kwargs UNCHANGED — once the routing table states the
    # cell's served mask is not header-located (`all`; the narrowed §2.5.4 rule); a
    # malformed kwargs object refuses the SHAPE validation on every proxy
    vm = _vmodel(cells, LENS)
    _refuses(proxy, _req(vm, chat_template_kwargs={"enable_thinking": True}), px.THINKING_ON_UNDECLARED)
    _refuses(proxy, _req(vm, chat_template_kwargs="garbage"), px.THINKING_KWARGS_INVALID)
    stated, backend2, _ = _proxy(tmp_path, lens_served_positions={LENS: ["all"]})
    _ask(stated, _req(vm, chat_template_kwargs={"enable_thinking": True}))
    assert backend2.payloads[-1]["chat_template_kwargs"] == {"enable_thinking": True}


def test_off_proxy_validates_the_declaration_key_per_request(tmp_path):
    """On the OFF proxy the declaration key is NOT refused wholesale (the
    sampled-rerun lineage's `thinking_on_standup_off` was dropped at the
    merge: main's default serves declared thinking-ON requests per request —
    the paired-GSM8K / MATH-500 instruments' path); it is validated by main's
    matrix: a declaration without the ON render is a MISMATCH, the key on a
    non-lens route is refused whatever its value, a non-boolean is invalid."""
    proxy, backend, cells = _proxy(tmp_path)
    n = len(backend.payloads)
    for cell in (LENS, dc.NOOP_LENS_CELL_ID):
        _refuses(proxy, _req(_vmodel(cells, cell), petri_thinking_on=True), px.THINKING_ON_MISMATCH)
        for value in (None, "yes"):
            _refuses(proxy, _req(_vmodel(cells, cell), petri_thinking_on=value), px.THINKING_ON_INVALID)
    # the matrix's order on a non-lens route: a declared True without the ON render is
    # a MISMATCH first; the key with any other value is refused for the route
    _refuses(proxy, _req(_vmodel(cells, "none"), petri_thinking_on=True), px.THINKING_ON_MISMATCH)
    _refuses(proxy, _req(_vmodel(cells, "none"), petri_thinking_on=False), px.THINKING_ON_NOT_LENS)
    assert len(backend.payloads) == n           # nothing reached upstream


# --------------------------------------------------------------------------- #
# the ON proxy: the refusal matrix                                              #
# --------------------------------------------------------------------------- #


def test_on_proxy_requires_the_declaration_on_lens_routes_and_injects_for_declared_cells(tmp_path):
    proxy, backend, cells = _on_proxy(tmp_path)
    vm = _vmodel(cells, LENS)
    _refuses(proxy, _chat(vm), px.THINKING_ON_UNDECLARED)
    _refuses(proxy, _req(vm, petri_thinking_on=False), px.THINKING_ON_UNDECLARED)
    out = _ask(proxy, _req(vm, petri_thinking_on=True))
    body = backend.payloads[-1]
    sel = body[px.UPSTREAM_EXTRA_ARGS_KEY]
    assert sel["thinking_on"] == 1 and not isinstance(sel["thinking_on"], bool)
    assert px.THINKING_ON_KEY not in body                      # never upstream
    assert "chat_template_kwargs" not in body                  # the LAUNCH pins the render
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    assert proxy.request_log[-1][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    # the twin (declared through its donor) injects too, at scale 0
    _ask(proxy, _req(_vmodel(cells, dc.NOOP_LENS_CELL_ID), petri_thinking_on=True))
    sel = backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    assert sel["thinking_on"] == 1 and sel["scale"] == 0.0
    # an UNDECLARED lens cell (an all-position fit): the declaration is
    # required and validated, but nothing is injected
    _ask(proxy, _req(_vmodel(cells, LENS2), petri_thinking_on=True))
    assert "thinking_on" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    _refuses(proxy, _chat(_vmodel(cells, LENS2)), px.THINKING_ON_UNDECLARED)


def test_on_proxy_non_lens_routes_serve_undeclared_and_refuse_the_key(tmp_path):
    proxy, backend, cells = _on_proxy(tmp_path)
    for cell, system in (("none", None),
                         ("prompted__safety", "synthetic safety instruction: be safe (test bytes)")):
        vm = _vmodel(cells, cell)
        out = _ask(proxy, _chat(vm, system=system))
        assert px.UPSTREAM_EXTRA_ARGS_KEY not in backend.payloads[-1]
        assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"   # the regime is recorded
        for value in (True, False):
            req = _chat(vm, system=system)
            req[px.THINKING_ON_KEY] = value
            _refuses(proxy, req, px.THINKING_ON_NOT_LENS)


def test_on_proxy_kwargs_and_declaration_shapes(tmp_path):
    proxy, backend, cells = _on_proxy(tmp_path)
    vm = _vmodel(cells, LENS)
    ok = dict(petri_thinking_on=True)
    _refuses(proxy, _req(vm, **ok, chat_template_kwargs={"enable_thinking": False}),
             px.THINKING_REGIME_MISMATCH)
    _refuses(proxy, _req(vm, **ok, chat_template_kwargs="garbage"), px.THINKING_KWARGS_INVALID)
    _refuses(proxy, _req(vm, **ok, chat_template_kwargs={"enable_thinking": True, "x": 1}),
             px.THINKING_KWARGS_INVALID)
    _refuses(proxy, _req(vm, **ok, chat_template_kwargs={"enable_thinking": "true"}),
             px.THINKING_KWARGS_INVALID)
    _refuses(proxy, _req(vm, **ok, chat_template_kwargs=None), px.THINKING_KWARGS_INVALID)
    # a redundant explicit true is accepted and forwarded
    _ask(proxy, _req(vm, **ok, chat_template_kwargs={"enable_thinking": True}))
    assert backend.payloads[-1]["chat_template_kwargs"] == {"enable_thinking": True}
    # the regime flip is refused on non-lens routes too
    _refuses(proxy, _req(_vmodel(cells, "none"), chat_template_kwargs={"enable_thinking": False}),
             px.THINKING_REGIME_MISMATCH)
    for bad in (None, "true", 1):
        _refuses(proxy, _req(vm, petri_thinking_on=bad), px.THINKING_ON_INVALID)
    # declared beside a non-tail declaration on a MOVED header: mismatch
    _refuses(proxy, _req(vm, **ok, petri_response_prefill=True), px.THINKING_ON_MISMATCH)
    _refuses(proxy, _req(vm, **ok, petri_agent_loop=True), px.THINKING_ON_MISMATCH)


def test_on_proxy_on_an_unmoved_header_validates_but_never_injects(tmp_path):
    """gemma-4: the header does not move, so the declaration is required and
    recorded, nothing is injected, and the non-tail declarations fall
    through to their own validators."""
    proxy, backend, cells = _on_proxy(tmp_path, header=STAYS, declared=())
    vm = _vmodel(cells, LENS)
    _refuses(proxy, _chat(vm), px.THINKING_ON_UNDECLARED)
    out = _ask(proxy, _req(vm, petri_thinking_on=True))
    assert "thinking_on" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    assert out["petri"][px.LEDGER_KEY_RENDER_REGIME] == "thinking_on"
    # beside a non-tail declaration the thinking gate stands aside (the
    # response-prefill validator then judges the shape on its own terms)
    e = _refuses(proxy, _req(vm, petri_thinking_on=True, petri_response_prefill=True),
                 "response_prefill_mismatch")
    assert e.code != px.THINKING_ON_MISMATCH


def test_tokenize_on_an_on_proxy_refuses_a_regime_flip(tmp_path):
    class TokBackend(StubBackend):
        async def tokenize(self, payload):
            self.payloads.append(json.loads(json.dumps(payload)))
            return {"count": 3, "tokens": [1, 2, 3]}
    spec, source, cells = _cells(tmp_path)
    proxy = px.RoutingProxy(px.routes_from_cells(cells), TokBackend(),
                            base_model=spec.target_model, serving_mode="interleaved",
                            prompted_prefixes=px.prompted_prefix_specs(cells, source),
                            agent_loop_render=_TailRender(), thinking_mode="on",
                            thinking_header=MOVES, lens_serve_declarations={})
    vm = _vmodel(cells, LENS)
    with pytest.raises(px.ProxyRefusal) as ei:
        asyncio.run(proxy.handle_tokenize(_req(vm, chat_template_kwargs={"enable_thinking": False})))
    assert ei.value.code == px.THINKING_REGIME_MISMATCH
    out = asyncio.run(proxy.handle_tokenize(_chat(vm)))
    assert out["count"] == 3


# --------------------------------------------------------------------------- #
# the identity block and the routing table                                      #
# --------------------------------------------------------------------------- #


def test_identity_block_states_the_mode_and_the_facts(tmp_path):
    off, _, _ = _proxy(tmp_path)
    meta = off.models_payload()["petri_proxy"]
    assert meta[px.THINKING_MODE_KEY] == "off"
    assert meta[px.THINKING_DECLARATION_KEY_KEY] == "petri_thinking_on"
    assert px.THINKING_HEADER_KEY not in meta and px.SERVE_DECLARATIONS_KEY not in meta
    # an OFF proxy handed the fact echoes it (a reader sees off_ids and why ON is unavailable)
    partial, _, _ = _proxy(tmp_path, thinking_header=PARTIAL, thinking_header_error="ChatMarkerError: x")
    meta = partial.models_payload()["petri_proxy"]
    assert meta[px.THINKING_HEADER_KEY] == PARTIAL and meta[px.THINKING_HEADER_ERROR_KEY] == "ChatMarkerError: x"
    on, _, _ = _on_proxy(tmp_path)
    meta = on.models_payload()["petri_proxy"]
    assert meta[px.THINKING_MODE_KEY] == "on"
    assert meta[px.THINKING_HEADER_KEY] == MOVES and meta[px.THINKING_HEADER_ERROR_KEY] is None
    assert meta[px.SERVE_DECLARATIONS_KEY] == {LENS: sd.thinking_declaration(),
                                               dc.NOOP_LENS_CELL_ID: sd.thinking_declaration()}
    assert meta[px.SERVE_DECLARATIONS_ERROR_KEY] is None
    # the pre-existing identity fields are all still there
    assert {"version", "serving_mode", "exact_replay_strategy", "base_model",
            "petri_agent_loop_entry", "agent_loop_render"} <= set(meta)


def test_routing_table_carries_the_fact_only_when_given(tmp_path):
    spec, source, cells = _cells(tmp_path)
    plain = tmp_path / "plain.json"
    px.export_routing_table(cells, source, plain, base_model=spec.target_model,
                            serving_mode="interleaved", base_revision="rev0")
    doc = json.loads(plain.read_text())
    assert px.THINKING_HEADER_KEY not in doc and px.THINKING_HEADER_ERROR_KEY not in doc
    table = px.load_routing_table(plain)
    assert table[px.THINKING_HEADER_KEY] is None and table[px.THINKING_HEADER_ERROR_KEY] is None
    with_fact = tmp_path / "fact.json"
    px.export_routing_table(cells, source, with_fact, base_model=spec.target_model,
                            serving_mode="interleaved", base_revision="rev0", thinking_header=MOVES)
    table = px.load_routing_table(with_fact)
    assert table[px.THINKING_HEADER_KEY] == MOVES and table[px.THINKING_HEADER_ERROR_KEY] is None
    partial = tmp_path / "partial.json"
    px.export_routing_table(cells, source, partial, base_model=spec.target_model,
                            serving_mode="interleaved", thinking_header=PARTIAL,
                            thinking_header_error="ChatMarkerError: why")
    table = px.load_routing_table(partial)
    assert table[px.THINKING_HEADER_KEY] == PARTIAL
    assert table[px.THINKING_HEADER_ERROR_KEY] == "ChatMarkerError: why"
    with pytest.raises(px.ProxyError, match="carries the reason"):
        px.export_routing_table(cells, source, tmp_path / "bad.json", base_model=spec.target_model,
                                serving_mode="interleaved", thinking_header=PARTIAL)
    # a proxy rebuilt from the table holds the same fact and mode
    proxy = px.RoutingProxy(table["routes"], StubBackend(), base_model=table["base_model"],
                            serving_mode=table["serving_mode"],
                            prompted_prefixes=table["prompted_prefixes"],
                            agent_loop_render=_TailRender(),
                            thinking_header=table[px.THINKING_HEADER_KEY],
                            thinking_header_error=table[px.THINKING_HEADER_ERROR_KEY])
    assert proxy.models_payload()["petri_proxy"][px.THINKING_HEADER_ERROR_KEY] == "ChatMarkerError: why"


def test_refusal_codes_are_the_documented_closed_set():
    """Main's §2.5.6 matrix plus the thinking-ON standup's ONE addition, the
    regime flip. The sampled-rerun lineage's `thinking_on_standup_off` (refuse
    the declaration key wholesale on an OFF proxy) is NOT in the set: the OFF
    proxy is main's per-request-regime proxy, which validates the key by the
    matrix (test_off_proxy_validates_the_declaration_key_per_request)."""
    assert px.THINKING_REFUSAL_CODES == (
        "thinking_kwargs_invalid", "thinking_kwargs_inert",
        "thinking_on_undeclared", "thinking_on_invalid", "thinking_on_mismatch",
        "thinking_on_not_lens", "thinking_on_cell_not_declared", "thinking_on_unattested",
        "thinking_regime_mismatch")
    assert not hasattr(px, "THINKING_ON_STANDUP_OFF")
    assert px.THINKING_ON_KEY == "petri_thinking_on"
    assert px.THINKING_MODES == ("off", "on")
    assert cm  # imported for the cell set's manifest vocabulary
