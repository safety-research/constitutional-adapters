"""The gemma-4 Petri multi-turn boundary, against the REAL pinned
tokenizer + canonical chat template (2026-08-30 EngineCore incident).

Everything here is skipped unless the pinned ``google/gemma-4-E4B-it``
tokenizer is already in the local HF cache (``local_files_only`` — no
test may download a model). Gemma is load-bearing for this suite: llama
HIDES both traps pinned here (its user turn ends at the special
``<|eot_id|>`` with no trailing scaffold, so located == appended and
every render tail is the header) — a llama-only run would false-pass.

Two distinct defects are pinned, and they must not be conflated:

* the HF-leg width-vs-located disagreement (fixed by 03184be):
  ``generation_header_ids`` (what ``add_generation_prompt`` appends, 3
  ids on this pin) vs ``derive_assistant_markers().header`` (anchored at
  the previous turn's terminator, absorbs the user turn's trailing
  ``\n`` — 4 ids). The guard here keeps a future change from quietly
  re-introducing the comparison of the two definitions;
* the vLLM-leg multi-turn tail: Petri agent-loop renders legitimately
  do not END at the generation header even under
  ``add_generation_prompt=True`` — the canonical template continues the
  model turn after tool traffic — so the serve-time boundary must be
  the LAST occurrence of the header, never a tail-equality check. Under
  main's declared-only policy that boundary is served through the
  ``petri_agent_loop`` declaration (proxy-verified -> selector fact ->
  ``HookContext.agent_loop`` -> the adapter's declared arm), and the
  three real shapes are pinned in BOTH directions below: they REFUSE
  undeclared and SERVE declared, at the adapter and through the proxy
  with the real tokenizer as the render probe.
"""

import asyncio
import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from sisteer.capeval.serving import lens_adapters as la  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402

MODEL = "google/gemma-4-E4B-it"
#: the revision the Petri audit server pinned (cluster B's gemma-audit,
#: vllm_server.log 2026-08-30) — the render shapes are template bytes,
#: so the pin is part of what is being tested.
REVISION = "ee0ef6023621cff504d758262d4e04895a5af4a2"

D = 8


def _tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, local_files_only=True)
    except Exception as e:  # not cached on this machine — skip, never fetch
        pytest.skip(f"pinned {MODEL} tokenizer not in the local HF cache: "
                    f"{type(e).__name__}")


def _render(tok, msgs, tools=None):
    ids = tok.apply_chat_template(
        list(msgs), tokenize=True, tools=tools, add_generation_prompt=True)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return tuple(int(i) for i in ids)


_TOOLS = [{"type": "function", "function": {
    "name": "get_weather", "description": "Get weather",
    "parameters": {"type": "object",
                   "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]

_USER = {"role": "user", "content": "Hi there. Please check the weather."}
_TOOL_RESULT = {"role": "tool", "tool_call_id": "call1",
                "content": "Sunny, 72F."}


def _assistant_tc(content):
    return {"role": "assistant", "content": content,
            "tool_calls": [{"id": "call1", "type": "function",
                            "function": {"name": "get_weather",
                                         "arguments": {"city": "Boston"}}}]}


def _assistant_tc_openai(content):
    """The same turn as the OpenAI client actually sends it: ``arguments``
    is a JSON STRING (vLLM deserializes it before the template runs; the
    pinned gemma template refuses a string outright)."""
    m = _assistant_tc(content)
    m["tool_calls"][0]["function"]["arguments"] = json.dumps(
        {"city": "Boston"})
    return m


def _petri_shapes(tok):
    """The agent-loop request shapes Petri actually sends (read out of
    the 2026-08-30 audit's inspect logs), rendered by the real template.
    Every one is a legitimate /v1/chat/completions request with
    ``add_generation_prompt=True``."""
    return {
        "pending_tool_call": _render(
            tok, [_USER, _assistant_tc("")], tools=_TOOLS),
        "tool_response_continuation": _render(
            tok, [_USER, _assistant_tc(""), _TOOL_RESULT], tools=_TOOLS),
        # the shape that killed the engine: assistant text + tool calls,
        # tool response in, nothing after -> tail [.., '.', <turn|>, \n]
        "content_and_tool_response": _render(
            tok, [_USER, _assistant_tc("Let me check that for you."),
                  _TOOL_RESULT], tools=_TOOLS),
    }


def _find_last(ids, header):
    H = len(header)
    for start in range(len(ids) - H, -1, -1):
        if tuple(ids[start:start + H]) == tuple(header):
            return start
    return None


def _response_op(header):
    return la.HookOp(op=la.OP_ADD_VECTOR, layer=0,
                     vector=torch.ones(D, dtype=torch.float32),
                     positions=la.POSITIONS_RESPONSE,
                     header_ids=tuple(header))


def _prefill_ctx(ids):
    return la.HookContext(phase=la.PHASE_PREFILL,
                          positions=tuple(range(len(ids))),
                          token_ids=tuple(ids))


# --- the two header definitions (the HF-leg guard, 03184be) --------------------


def test_generation_header_is_three_ids_and_located_header_is_four():
    """The regression guard for 03184be: the two definitions of "the
    assistant header" genuinely differ on this pin, and the serve/fit
    boundary vocabulary is the GENERATION header (3), never the
    terminator-anchored located header (4). A change that makes these
    agree again has changed one of the definitions."""
    from sisteer import hooks
    from sisteer.chat_markers import (derive_assistant_markers,
                                      generation_header_ids)
    tok = _tok()
    gen = generation_header_ids(tok)
    assert len(gen) == 3
    assert hooks.response_header_len(tok) == 3
    located = derive_assistant_markers(tok).header
    assert located is not None and len(located) == 4
    assert tuple(located[1:]) == tuple(gen)  # leading '\n' is the difference


# --- the multi-turn render shapes (the vLLM-leg fix) ---------------------------


def test_ordinary_renders_end_at_the_generation_header():
    from sisteer.chat_markers import generation_header_ids
    tok = _tok()
    header = generation_header_ids(tok)
    single = _render(tok, [{"role": "system", "content": "Be safe."}, _USER])
    multi = _render(tok, [
        _USER, {"role": "assistant", "content": "Hello."},
        {"role": "user", "content": "Tell me more."}])
    assert single[-len(header):] == tuple(header)
    assert multi[-len(header):] == tuple(header)


def test_petri_agent_loop_renders_do_not_end_at_the_header():
    """The trap itself: all three real Petri shapes render with the
    final model turn still open — none ends at the generation header,
    and the fatal one ends at a CLOSED turn (the 2026-08-30 tail)."""
    from sisteer.chat_markers import generation_header_ids
    tok = _tok()
    header = tuple(generation_header_ids(tok))
    shapes = _petri_shapes(tok)
    for name, ids in shapes.items():
        assert ids[-len(header):] != header, name
    # the observed crash signature: '.', <turn|>, '\n'
    tail = shapes["content_and_tool_response"][-2:]
    assert tail == (tok.convert_tokens_to_ids("<turn|>"),
                    tok.convert_tokens_to_ids("\n"))


def test_petri_shapes_refuse_undeclared_at_the_adapter():
    """The UNDECLARED half of the pair — kept verbatim.

    The 2026-08-30 petrimask fix served every agent-loop render from the
    LAST header occurrence, unconditionally. main's 2026-09-01 rework
    (f352d5e) landed the narrower policy instead — a non-tail render is
    served only when the request DECLARES it — and that policy won the
    2026-09-02 merge of the Petri lineage. So the real Petri shapes,
    arriving with no declaration, REFUSE at the adapter (contained
    per-request by the lens worker's failure ledger, never engine-fatal).
    The declared half (next test) is how they serve; this half pins that
    serving them is a declaration, never drift back to the unconditional
    locator (still reproducible at its frozen serving_impl_commit on
    origin/petri-gemma-e4b).
    """
    from sisteer.chat_markers import generation_header_ids
    tok = _tok()
    header = tuple(generation_header_ids(tok))
    op = _response_op(header)
    for name, ids in _petri_shapes(tok).items():
        assert _find_last(ids, header) is not None, name  # header IS present
        h = torch.randn(len(ids), D, dtype=torch.float32)
        with pytest.raises(la.AdapterError, match="DECLARES"):
            la.apply_hook_ops(h, (op,), 1.0, layer=0, ctx=_prefill_ctx(ids))


#: where the generation header starts in all three real renders at this
#: pin (the shapes share a user turn + tools preamble; the boundary is
#: located, and this number is the pin of what "located" must find)
EXPECTED_BOUNDARY = 58


def test_petri_shapes_serve_when_declared_agent_loop():
    """The DECLARED half of the pair: under ``HookContext.agent_loop`` the
    same three renders serve, with the boundary at exactly
    ``_find_last(ids, header)`` (= 58 for all three at this pin) and the
    masked row set = the header + everything after it. Everything after
    it means, on these shapes, the assistant's tool call, the
    tool-response syntax and body, and every generated position — the
    completed audit's regime (lens_adapters module docstring), asserted
    here so nobody reads the masked region as ordinary assistant text.
    The three TAILS all differ (tool-response opener / closed tool
    response / closed turn), which is why the invariant is "header
    present and not at the tail", never a particular tail token."""
    from sisteer.chat_markers import generation_header_ids
    tok = _tok()
    header = tuple(generation_header_ids(tok))
    op = _response_op(header)
    shapes = _petri_shapes(tok)
    tails = set()
    for name, ids in shapes.items():
        expected = _find_last(ids, header)
        assert expected == EXPECTED_BOUNDARY, (name, expected)
        ctx = la.HookContext(phase=la.PHASE_PREFILL,
                             positions=tuple(range(len(ids))),
                             token_ids=tuple(ids), agent_loop=True)
        assert la._response_start(op, ctx, len(ids)) == expected, name
        mask = la._row_mask(op, ctx, torch.zeros(len(ids), D))
        assert mask.tolist() == ([False] * expected
                                 + [True] * (len(ids) - expected)), name
        assert bool(mask[-1]), name  # the forward that emits token 1
        # the public entry serves it: rows before the boundary untouched,
        # every row from the header on edited by the unit vector
        h = torch.randn(len(ids), D, dtype=torch.float32)
        out = la.apply_hook_ops(h.clone(), (op,), 1.0, layer=0, ctx=ctx)
        out = h if out is None else out
        edited = (out != h).any(dim=-1).tolist()
        assert edited == mask.tolist(), name
        tails.add(ids[-4:])
    assert len(tails) == 3  # three shapes, three different tails


# --- the proxy with the REAL tokenizer as its render probe ---------------------


def _openai_shapes():
    """The three shapes as chat-completions REQUEST BODIES, in the OpenAI
    wire form (string tool-call arguments)."""
    return {
        "pending_tool_call": [_USER, _assistant_tc_openai("")],
        "tool_response_continuation": [_USER, _assistant_tc_openai(""),
                                       _TOOL_RESULT],
        "content_and_tool_response": [
            _USER, _assistant_tc_openai("Let me check that for you."),
            _TOOL_RESULT],
    }


def test_tokenizer_render_probe_mirrors_the_served_render():
    """``TokenizerAgentLoopRender`` on the real tokenizer: the header is
    the generation header, and rendering the OpenAI wire form (string
    arguments) reproduces the template's own render of the deserialized
    form token for token, for all three shapes."""
    from sisteer.chat_markers import generation_header_ids
    tok = _tok()
    probe = px.TokenizerAgentLoopRender(tok)
    assert probe.header_ids == tuple(generation_header_ids(tok))
    assert len(probe.header_ids) == 3
    shapes = _petri_shapes(tok)
    for name, msgs in _openai_shapes().items():
        assert probe.render(msgs, _TOOLS) == shapes[name], name


def _real_proxy(tmp_path, tok, *, entry):
    from test_petri_proxy import StubBackend, _cells, _source, _spec
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = _cells(spec, source)
    backend = StubBackend()
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend,
        base_model=spec.target_model, serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cells, source),
        exact_replay_strategy=None,
        agent_loop_render=px.TokenizerAgentLoopRender(tok),
        petri_agent_loop_entry=entry)
    lens_vm = next(r.virtual_model for r in px.routes_from_cells(cells)
                   if r.cell_id == "si_safety__psr_kl_all")
    return proxy, backend, lens_vm


def _body(vm, msgs, *, declare):
    body = {"model": vm, "messages": msgs, "tools": _TOOLS,
            "max_tokens": 32, "temperature": 0.0}
    if declare:
        body[px.AGENT_LOOP_KEY] = True
    return body


def test_proxy_accepts_the_three_real_shapes_declared_and_refuses_single_turn(
        tmp_path):
    """The verifier against the REAL render: each declared shape passes
    (header present, not at the tail) and travels as the selector fact;
    a declared single-turn request — whose render ENDS at the header —
    is a 400."""
    tok = _tok()
    proxy, backend, vm = _real_proxy(tmp_path, tok, entry=False)
    for name, msgs in _openai_shapes().items():
        out = asyncio.run(proxy.handle_chat_completions(
            _body(vm, msgs, declare=True)))
        assert out["model"] == vm, name
        sel = backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
        assert sel["agent_loop"] == 1 and not isinstance(sel["agent_loop"],
                                                         bool), name
        assert px.AGENT_LOOP_KEY not in backend.payloads[-1]
    single = {"model": vm, "max_tokens": 32,
              "messages": [{"role": "system", "content": "Be safe."}, _USER],
              px.AGENT_LOOP_KEY: True}
    with pytest.raises(px.ProxyRefusal, match="not a turn of the agent loop"):
        asyncio.run(proxy.handle_chat_completions(single))
    # and a multi-turn PLAIN transcript (no tool traffic) renders to the
    # tail header — the verifier's render check refuses that declaration
    plain = _body(vm, [_USER, {"role": "assistant", "content": "Hello."},
                       {"role": "user", "content": "Tell me more."},
                       _assistant_tc_openai("")], declare=True)
    plain["messages"][-1] = {"role": "assistant", "content": "Sure.",
                             "tool_calls": []}
    with pytest.raises(px.ProxyRefusal, match="not a turn of the agent loop"):
        asyncio.run(proxy.handle_chat_completions(plain))


def test_proxy_entry_declares_the_real_shapes_for_petri(tmp_path):
    """The Petri-facing entry with the real render probe: the three shapes
    arriving WITHOUT the key (as inspect_ai's client sends them) are
    declared at entry, verified against the real render, and served with
    the fact; an ordinary single-turn request through the same entry
    carries no fact."""
    tok = _tok()
    proxy, backend, vm = _real_proxy(tmp_path, tok, entry=True)
    for name, msgs in _openai_shapes().items():
        out = asyncio.run(proxy.handle_chat_completions(
            _body(vm, msgs, declare=False)))
        assert out["model"] == vm, name
        assert backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY][
            "agent_loop"] == 1, name
    asyncio.run(proxy.handle_chat_completions(
        {"model": vm, "max_tokens": 32, "messages": [_USER]}))
    assert "agent_loop" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
