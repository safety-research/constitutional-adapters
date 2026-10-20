"""BFCL on a STEERED (lens-served) cell — U4(b) / PW18: the request shapes
inspect_evals' BFCL solver produces, against the proxy's own agent-loop
classifier, its Petri-facing entry and the real gemma-4 render.

Everything here is offline and synthetic (invented prompts, the proxy
tests' canonical cell set, placeholder tool schemas); the one real object
is the pinned ``google/gemma-4-E4B-it`` tokenizer, read from the local HF
cache only (``local_files_only`` — no test may download a model); those
tests skip when it is not cached.

What is established (the conditions under which a BFCL request is served
correctly on a lens route):

* the SINGLE-TURN categories the plan runs (V1 u V2, P9) send ONE
  generation-prompt request per item — system / user (a live entry may
  carry a plain user / assistant history) plus ``tools``; the model's tool
  calls are captured and AST-scored, never executed, never sent back
  (``single_turn_solve``: ``generate(tool_calls="none")``). Those bodies
  are NOT agent-loop shapes (``proxy.agent_loop_shape`` is None), the real
  gemma-4 render ENDS at the generation header, the armed entry attaches
  nothing, and the lens serves them undeclared like any generation task —
  mask = header + every generated position (the tool call);
* a tool TURN (the assistant's pending tool call; a tool result awaiting
  the continuation — what V3's ``tool_calls="loop"`` would send) IS an
  agent-loop shape, its real render does NOT end at the header, and the
  armed entry declares it, so the lens steers it from the LAST header
  (header + tool call + tool response + generated — the completed
  gemma-E4B Petri audit's regime); with the entry OFF the same turn is
  forwarded undeclared and refused downstream (the pinned direction);
* the two declarations are mutually exclusive: a BFCL body never carries
  ``petri_response_prefill`` or ``continue_final_message``, and a request
  stating both is a 400;
* the proxy's ``/v1/models`` identity block spells the entry under the key
  the tinycap contracts read (``petri_proxy.petri_agent_loop_entry``).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sisteer.capeval.serving import proxy as px  # noqa: E402
from sisteer.capeval.tinycap import contracts as C  # noqa: E402

from test_petri_proxy import StubBackend, _cells, _source, _spec  # noqa: E402

LENS_CELL = "si_safety__psr_kl_all"

#: BFCL-style tool schemas as inspect_evals' ``create_tool_info_from_dict``
#: -> the openai client put them on the wire (nested object parameters, an
#: enum, a required list) — invented, in the dataset's shape
BFCL_TOOLS = [
    {"type": "function", "function": {
        "name": "calculate_triangle_area",
        "description": "Calculate the area of a triangle given its base and height.",
        "parameters": {"type": "object",
                       "properties": {"base": {"type": "integer", "description": "The base."},
                                      "height": {"type": "integer", "description": "The height."},
                                      "unit": {"type": "string", "enum": ["cm", "m"],
                                               "description": "The unit."}},
                       "required": ["base", "height"]},
        "strict": False}},
    {"type": "function", "function": {
        "name": "geometry_circle",
        "description": "Compute circle facts.",
        "parameters": {"type": "object",
                       "properties": {"circle": {"type": "object", "description": "The circle.",
                                                 "properties": {"radius": {"type": "number",
                                                                           "description": "r"}},
                                                 "required": ["radius"]}},
                       "required": ["circle"]},
        "strict": False}},
]
SYSTEM = {"role": "system", "content": "You are an expert in composing functions."}
USER = {"role": "user", "content": "Find the area of a triangle with base 10 and height 5."}
LIVE_HISTORY = [
    SYSTEM,
    {"role": "user", "content": "Hi, can you help with geometry?"},
    {"role": "assistant", "content": "Of course. What do you need?"},
    {"role": "user", "content": "Area of a 3 by 4 triangle, in cm."},
]
PENDING_CALL = {"role": "assistant", "content": "",
                "tool_calls": [{"id": "call_1", "type": "function",
                                "function": {"name": "calculate_triangle_area",
                                             "arguments": json.dumps({"base": 10, "height": 5})}}]}
TOOL_RESULT = {"role": "tool", "tool_call_id": "call_1", "content": "25"}

#: the shapes BFCL's single-turn solver sends (one request per item)
SINGLE_TURN_SHAPES = {
    "system_user": [SYSTEM, USER],
    "user_only": [USER],
    "live_history": LIVE_HISTORY,
}
#: the shapes a tool ROUND-TRIP would send (V3's loop; Petri's audit)
TOOL_TURN_SHAPES = {
    "pending_tool_call": ([SYSTEM, USER, PENDING_CALL], px.AGENT_LOOP_SHAPE_TOOL_CALL),
    "tool_result": ([SYSTEM, USER, PENDING_CALL, TOOL_RESULT], px.AGENT_LOOP_SHAPE_TOOL_RESULT),
}


def _body(vm, msgs, **over):
    body = {"model": vm, "messages": list(msgs), "tools": BFCL_TOOLS, "tool_choice": "auto",
            "max_tokens": 64, "temperature": 0.0,
            C.CHAT_TEMPLATE_KWARGS_KEY: {C.ENABLE_THINKING_KEY: False}}
    body.update(over)
    return body


# --------------------------------------------------------------------------- #
# the classifier and the structural entry                                      #
# --------------------------------------------------------------------------- #


def test_single_turn_bfcl_shapes_are_not_agent_loops_and_tool_turns_are():
    for name, msgs in SINGLE_TURN_SHAPES.items():
        body = _body("petri__x", msgs)
        assert px.agent_loop_shape(body) is None, name
        assert C.AGENT_LOOP_KEY not in px.declare_agent_loop(body, lens_route=True), name
    for name, (msgs, shape) in TOOL_TURN_SHAPES.items():
        body = _body("petri__x", msgs)
        assert px.agent_loop_shape(body) == shape, name
        assert px.declare_agent_loop(body, lens_route=True)[C.AGENT_LOOP_KEY] is True, name
        # never on a non-lens route (the prompted / base legs serve tool turns natively)
        assert C.AGENT_LOOP_KEY not in px.declare_agent_loop(body, lens_route=False), name


# --------------------------------------------------------------------------- #
# the proxy with the entry ON (the standup's default) and OFF                  #
# --------------------------------------------------------------------------- #


class _Render:
    """A scripted render probe: an agent-loop STRUCTURE renders to synthetic
    non-tail ids, everything else to an ordinary tail-header render (what
    the real gemma-4 template does — pinned against the real tokenizer
    below)."""

    header_ids = (901, 902, 903)

    def render(self, messages, tools):
        if px.agent_loop_shape({"messages": list(messages)}) is None:
            return (1, 2, 3, *self.header_ids)
        return (1, 2, 3, *self.header_ids, 41, 42, 77, 78)


def _proxy(tmp_path, *, entry: bool):
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = _cells(spec, source)
    backend = StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend,
        base_model=spec.target_model, serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cells, source),
        exact_replay_strategy=None,
        agent_loop_render=_Render(), petri_agent_loop_entry=entry)
    vms = {r.cell_id: r.virtual_model for r in px.routes_from_cells(cells)}
    return proxy, backend, vms


def _ask(proxy, body):
    return asyncio.run(proxy.handle_chat_completions(body))


def test_entry_on_serves_single_turn_bfcl_without_a_fact_and_tool_turns_with_one(tmp_path):
    proxy, backend, vms = _proxy(tmp_path, entry=True)
    lens = vms[LENS_CELL]
    meta = proxy.models_payload()[C.PROXY_IDENTITY_KEY]
    assert meta[C.PROXY_IDENTITY_AGENT_LOOP_ENTRY] is True     # the key the tinycap driver reads
    for name, msgs in SINGLE_TURN_SHAPES.items():
        out = _ask(proxy, _body(lens, msgs))
        assert out["model"] == lens, name
        sent = backend.payloads[-1]
        sel = sent[px.UPSTREAM_EXTRA_ARGS_KEY]
        assert "agent_loop" not in sel and "response_prefill" not in sel, name
        assert C.AGENT_LOOP_KEY not in sent and C.RESPONSE_PREFILL_KEY not in sent, name
        assert sent["tools"] == BFCL_TOOLS and sent["tool_choice"] == "auto", name   # forwarded untouched
        # the lens stamps: what the Inspect provider's on_response verifies
        assert out["petri"]["cell_id"] == LENS_CELL and out["petri"]["serving_identity"].startswith(
            "proxy_vllm+lens+"), name
    for name, (msgs, _) in TOOL_TURN_SHAPES.items():
        out = _ask(proxy, _body(lens, msgs))
        assert out["model"] == lens, name
        sel = backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
        assert sel["agent_loop"] == 1 and not isinstance(sel["agent_loop"], bool), name
        assert C.AGENT_LOOP_KEY not in backend.payloads[-1], name       # the key never travels upstream
    # the base (passthrough) route: tool turns serve natively, no fact ever
    base = vms["none"]
    for msgs, _ in TOOL_TURN_SHAPES.values():
        _ask(proxy, _body(base, msgs))
        assert "agent_loop" not in backend.payloads[-1].get(px.UPSTREAM_EXTRA_ARGS_KEY, {})


def test_entry_off_forwards_a_tool_turn_undeclared_and_the_client_cannot_declare(tmp_path):
    """The opt-out (``--no-petri-agent-loop``): a tool turn on a lens route
    is forwarded UNDECLARED (it would refuse at the adapter as a contained
    500 — the pinned direction, exactly what the tinycap plan-time gate
    exists to refuse before); single-turn BFCL is unaffected either way; and
    the client declaring the key itself is verified against the render
    like any client declaration (a tail render claiming a relocation is a
    400), so the declaration stays the served path's business."""
    proxy, backend, vms = _proxy(tmp_path, entry=False)
    lens = vms[LENS_CELL]
    assert proxy.models_payload()[C.PROXY_IDENTITY_KEY][C.PROXY_IDENTITY_AGENT_LOOP_ENTRY] is False
    for msgs, _ in TOOL_TURN_SHAPES.values():
        _ask(proxy, _body(lens, msgs))
        assert "agent_loop" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    for msgs in SINGLE_TURN_SHAPES.values():
        _ask(proxy, _body(lens, msgs))
        assert "agent_loop" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    with pytest.raises(px.ProxyRefusal, match="not a turn of the agent loop"):
        _ask(proxy, _body(lens, SINGLE_TURN_SHAPES["system_user"], **{C.AGENT_LOOP_KEY: True}))
    with pytest.raises(px.ProxyRefusal, match="not a turn of the agent loop"):
        _ask(proxy, _body(lens, SINGLE_TURN_SHAPES["live_history"], **{C.AGENT_LOOP_KEY: True}))


def test_the_two_declarations_are_mutually_exclusive(tmp_path):
    proxy, _, vms = _proxy(tmp_path, entry=True)
    body = _body(vms[LENS_CELL], TOOL_TURN_SHAPES["tool_result"][0],
                 **{C.AGENT_LOOP_KEY: True, C.RESPONSE_PREFILL_KEY: True})
    with pytest.raises(px.ProxyRefusal, match="BOTH"):
        _ask(proxy, body)


# --------------------------------------------------------------------------- #
# the real gemma-4 render: single-turn BFCL ends at the header, a tool turn    #
# does not                                                                     #
# --------------------------------------------------------------------------- #

MODEL = "google/gemma-4-E4B-it"
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"   # the served pin (test_gemma4_petri_multiturn_boundary)


def _tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True)
    except Exception as e:  # not cached here — skip, never fetch
        pytest.skip(f"pinned {MODEL} tokenizer not in the local HF cache: {type(e).__name__}")


def test_real_gemma4_render_single_turn_bfcl_is_a_tail_render_and_tool_turns_are_not():
    tok = _tok()
    probe = px.TokenizerAgentLoopRender(tok)
    header = tuple(probe.header_ids)
    H = len(header)
    for name, msgs in SINGLE_TURN_SHAPES.items():
        ids = tuple(int(t) for t in probe.render(msgs, BFCL_TOOLS))
        assert ids[-H:] == header, name          # an ordinary tail render: served undeclared
        assert C.AGENT_LOOP_KEY not in px.declare_agent_loop(
            _body("petri__x", msgs), lens_route=True, probe=probe), name
    for name, (msgs, _) in TOOL_TURN_SHAPES.items():
        ids = tuple(int(t) for t in probe.render(msgs, BFCL_TOOLS))
        assert ids[-H:] != header, name          # the open model turn continues past the header
        assert header in {ids[i:i + H] for i in range(len(ids) - H + 1)}, name
        assert px.declare_agent_loop(_body("petri__x", msgs), lens_route=True,
                                     probe=probe)[C.AGENT_LOOP_KEY] is True, name


def test_real_proxy_with_the_gemma4_probe_serves_bfcl_shapes_on_the_lens_route(tmp_path):
    tok = _tok()
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = _cells(spec, source)
    backend = StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend,
        base_model=spec.target_model, serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cells, source),
        exact_replay_strategy=None,
        agent_loop_render=px.TokenizerAgentLoopRender(tok), petri_agent_loop_entry=True)
    lens = next(r.virtual_model for r in px.routes_from_cells(cells) if r.cell_id == LENS_CELL)
    for name, msgs in SINGLE_TURN_SHAPES.items():
        assert _ask(proxy, _body(lens, msgs))["model"] == lens, name
        assert "agent_loop" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY], name
    for name, (msgs, _) in TOOL_TURN_SHAPES.items():
        assert _ask(proxy, _body(lens, msgs))["model"] == lens, name
        assert backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]["agent_loop"] == 1, name
