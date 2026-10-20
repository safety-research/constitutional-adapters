"""The declared Petri agent-loop path:
the declaration's whole path — Petri-facing entry -> proxy verifier ->
selector fact -> HookContext -> located boundary — plus a positive control
that can FAIL. All offline, all synthetic (invented prompts/tokens,
placeholder ids, the proxy tests' own canonical cell set — no real recorded
data). The three REAL gemma-4 agent-loop shapes are exercised against the
pinned tokenizer in ``tests/test_gemma4_petri_multiturn_boundary.py``; this
module is the synthetic matrix around them, modeled line for line on
``tests/test_response_prefill_serving.py`` (the prefill declaration whose
path this one shares).

What is covered, end to end:

* ``lens_adapters._response_start``: an UNDECLARED non-tail render still
  refuses (the pinned direction, verbatim); a DECLARED agent-loop render
  locates the boundary at the LAST header occurrence and masks everything
  from it — the header, the assistant's tool call, the tool-response
  scaffold and body, and the final prefill row (the forward that emits
  token 1); header-nowhere refuses; ``positions="last"`` refuses under the
  declaration; a context claiming both declarations refuses.
* ``lens_steer.parse_selector`` accepts the fact strictly ({True, False,
  1, 0}) and refuses both facts on one selector.
* ``lens_worker.build_request_slots`` threads the fact into exactly the
  flagged request's HookContext.
* ``proxy``: ``agent_loop_shape`` / ``declare_agent_loop`` (the
  Petri-facing entry); the ``petri_agent_loop`` verifier's full refusal
  matrix against a scripted render probe — structure AND render; the
  verified fact travelling as the selector's ``agent_loop`` (transport
  int 1) while the client key never goes upstream; the UNDECLARED shape
  forwarded untouched and refused by the lens post-check when the worker
  ledger reports the contained failure (the pinned direction, at the
  proxy); the entry armed on a proxy — declares on structure, still
  verified against the render; smuggling still refuses.
* the wire round trip's CPU half: proxy payload -> JSON -> the worker's
  ``extract_selector`` reads ``agent_loop`` as exactly True from the
  transport int (the vllm half — the pinned ChatCompletionRequest —
  runs in ``scripts/modal/capctrl_served.py``'s preflight, where vllm is
  importable).
* a positive control in ``fidelity._check_steering_bites``'s shape: on a
  declared agent-loop render a steered arm's token 1 diverges from its
  scale-0 twin's, because the declared mask reaches the final prefill
  row; an INERT arm produces identical ids and the check RAISES — the
  control can fail, proven.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from sisteer.capeval.serving import fidelity as fd  # noqa: E402
from sisteer.capeval.serving import lens_adapters as la  # noqa: E402
from sisteer.capeval.serving import lens_steer as ls  # noqa: E402
from sisteer.capeval.serving import lens_worker as lw  # noqa: E402
from sisteer.capeval.serving import proxy as px  # noqa: E402

from test_petri_proxy import StubBackend, _cells, _source, _spec  # noqa: E402

HEADER = (901, 902, 903, 904)  # synthetic assistant-header ids
USER = (1, 2, 3)               # a synthetic user turn
TOOL_CALL = (41, 42, 43)       # the assistant's tool call, after the header
TOOL_SCAFFOLD = (77, 78)       # tool-response opener/body/closer scaffold
#: the synthetic agent-loop render: header present, prompt NOT ending at it
AGENT_IDS = (*USER, *HEADER, *TOOL_CALL, *TOOL_SCAFFOLD)
#: an ordinary render: the header IS the tail
TAIL_IDS = (*USER, *HEADER)
D = 8


def _resp_op(**over):
    kw = dict(op=la.OP_ADD_VECTOR, layer=0, positions=la.POSITIONS_RESPONSE,
              header_ids=HEADER, vector=torch.zeros(D, dtype=torch.float32))
    kw.update(over)
    return la.HookOp(**kw)


def _prefill_ctx(ids, **facts):
    return la.HookContext(la.PHASE_PREFILL, tuple(range(len(ids))),
                          tuple(ids), **facts)


# --------------------------------------------------------------------------- #
# _response_start: the located boundary under the agent-loop declaration       #
# --------------------------------------------------------------------------- #


def test_undeclared_agent_loop_render_still_refuses():
    """The pinned direction, verbatim: a render that does not END at the
    header refuses unless the request declared it — the agent-loop
    landing adds a declaration, never a relaxation."""
    with pytest.raises(la.AdapterError, match="DECLARES"):
        la._response_start(_resp_op(), _prefill_ctx(AGENT_IDS),
                           len(AGENT_IDS))


def test_declared_agent_loop_locates_last_header_and_masks_through_tail():
    """The declared arm: boundary at the header's LAST occurrence, mask
    from there through the tail. Stated in the assertion, because it is
    the semantic decision of this landing: the masked region is header +
    the assistant's TOOL CALL + the TOOL-RESPONSE SCAFFOLD (external input,
    not model output) + every generated position — the completed audit's
    regime, kept for comparability — and, in particular, the FINAL
    prefill row, the forward that emits token 1."""
    ctx = _prefill_ctx(AGENT_IDS, agent_loop=True)
    op = _resp_op()
    start = la._response_start(op, ctx, len(AGENT_IDS))
    assert start == len(USER) == 3
    mask = la._row_mask(op, ctx, torch.zeros(len(AGENT_IDS), D))
    assert mask.tolist() == ([False] * len(USER)
                             + [True] * (len(HEADER) + len(TOOL_CALL)
                                         + len(TOOL_SCAFFOLD)))
    n_scaffold = len(TOOL_SCAFFOLD)
    assert mask[-n_scaffold:].all()      # the tool-response scaffold: INSIDE
    assert mask[len(USER) + len(HEADER):len(USER) + len(HEADER)
                + len(TOOL_CALL)].all()  # the tool call: INSIDE
    assert bool(mask[-1])                # the forward that emits token 1


def test_declared_agent_loop_takes_the_last_header_occurrence():
    """Multi-turn: an earlier assistant turn's header must not win."""
    ids = (*HEADER, 7, 8, 9, *USER, *HEADER, *TOOL_CALL, *TOOL_SCAFFOLD)
    start = la._response_start(_resp_op(),
                               _prefill_ctx(ids, agent_loop=True), len(ids))
    assert start == len(HEADER) + 3 + len(USER)


def test_declared_agent_loop_with_no_header_refuses():
    ids = (1, 2, 3, 4, 5, 6)
    with pytest.raises(la.AdapterError, match="occur NOWHERE") as e:
        la._response_start(_resp_op(), _prefill_ctx(ids, agent_loop=True),
                           len(ids))
    assert "agent_loop" in str(e.value)  # the refusal names the declaration


def test_declared_agent_loop_on_a_tail_render_is_the_tail_boundary():
    """At the ADAPTER the declaration on an ordinary render is harmless
    (rfind finds the tail occurrence — the same boundary as undeclared);
    it is the PROXY that refuses such a declaration, as a claim the render
    does not support (test_declaration_render_mismatches_refuse)."""
    op = _resp_op()
    declared = la._response_start(op, _prefill_ctx(TAIL_IDS, agent_loop=True),
                                  len(TAIL_IDS))
    plain = la._response_start(op, _prefill_ctx(TAIL_IDS), len(TAIL_IDS))
    assert declared == plain == len(USER)


def test_positions_last_refuses_a_declared_agent_loop():
    op = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, positions=la.POSITIONS_LAST,
                   vector=torch.zeros(D, dtype=torch.float32))
    with pytest.raises(la.AdapterError, match="positions='last'") as e:
        la._row_mask(op, _prefill_ctx(AGENT_IDS, agent_loop=True),
                     torch.zeros(len(AGENT_IDS), D))
    assert "agent_loop" in str(e.value)


def test_decode_rows_are_all_inside_the_mask_under_the_declaration():
    ctx = la.HookContext(la.PHASE_DECODE, (len(AGENT_IDS),),
                         (5,), agent_loop=True)
    mask = la._row_mask(_resp_op(), ctx, torch.zeros(1, D))
    assert mask.tolist() == [True]


def test_hookcontext_agent_loop_is_a_strict_bool_and_exclusive():
    with pytest.raises(la.AdapterError, match="agent_loop must be a bool"):
        la.HookContext(la.PHASE_PREFILL, (0, 1), (1, 2),
                       agent_loop=1)  # type: ignore[arg-type]
    with pytest.raises(la.AdapterError, match="BOTH"):
        la.HookContext(la.PHASE_PREFILL, (0, 1), (1, 2),
                       response_prefill=True, agent_loop=True)


def test_apply_hook_ops_serves_the_declared_render_and_refuses_undeclared():
    """The public entry, both directions on the same ids: declared serves
    (rows before the boundary untouched, rows from it edited); undeclared
    refuses with the pinned message."""
    op = _resp_op(vector=torch.ones(D, dtype=torch.float32))
    torch.manual_seed(0)
    h = torch.randn(len(AGENT_IDS), D)
    out = la.apply_hook_ops(h.clone(), (op,), 1.0, layer=0,
                            ctx=_prefill_ctx(AGENT_IDS, agent_loop=True))
    out = h if out is None else out  # in-place or returned, either way
    edited = (out != h).any(dim=-1)
    assert edited.tolist() == [False] * len(USER) + [True] * (
        len(AGENT_IDS) - len(USER))
    with pytest.raises(la.AdapterError, match="DECLARES"):
        la.apply_hook_ops(h.clone(), (op,), 1.0, layer=0,
                          ctx=_prefill_ctx(AGENT_IDS))


# --------------------------------------------------------------------------- #
# selector parsing + worker threading                                          #
# --------------------------------------------------------------------------- #


def test_parse_selector_accepts_agent_loop_strictly():
    for wire in (True, 1):
        sel = ls.parse_selector({"petri_object": "c", "expect_hash": "h",
                                 "agent_loop": wire})
        assert sel["agent_loop"] is True
    for wire in (False, 0):
        sel = ls.parse_selector({"petri_object": "c", "expect_hash": "h",
                                 "agent_loop": wire})
        assert sel["agent_loop"] is False
    sel = ls.parse_selector({"petri_object": "c", "expect_hash": "h"})
    assert "agent_loop" not in sel
    for bad in ("yes", 2, 1.0):
        with pytest.raises(ls.LensSteerError, match="agent_loop"):
            ls.parse_selector({"petri_object": "c", "expect_hash": "h",
                               "agent_loop": bad})
    with pytest.raises(ls.LensSteerError, match="BOTH"):
        ls.parse_selector({"petri_object": "c", "expect_hash": "h",
                           "agent_loop": 1, "response_prefill": 1})
    assert "agent_loop" in ls.SELECTOR_KEYS
    # the closed fact vocabulary: the two non-tail declarations plus the
    # thinking-ON render bit (docs/served_thinking_on.md)
    assert set(ls.SELECTOR_FACT_KEYS) == {"response_prefill", "agent_loop", "thinking_on"}


class _FakeRegistry:
    """Just enough of WorkerObjectRegistry for build_request_slots: one
    plan-needing object per selector."""

    def __init__(self):
        self.obj = ls.LoadedObject(
            cell_id="cell", kind="obj_psr_kl_all", expect_hash="ih",
            ops=(_resp_op(),), effective_scale=1.0)

    def select(self, petri_object, expect_hash, scale=None):
        return self.obj


def test_slot_builder_threads_agent_loop_per_request():
    """Two co-batched prefills, only one flagged: exactly that one's ctx
    carries agent_loop, and its boundary is the located one."""
    ids_a = list(AGENT_IDS)  # flagged: an agent-loop render
    ids_b = list(TAIL_IDS)   # unflagged: an ordinary render
    q = [len(ids_a), len(ids_b)]
    qsl = [0, q[0], q[0] + q[1]]
    positions = list(range(q[0])) + list(range(q[1]))
    toks = {"a": ids_a, "b": ids_b}
    slots = lw.build_request_slots(
        query_start_loc=qsl, seq_lens=q, num_actual_tokens=sum(q),
        positions=positions, req_ids=["a", "b"],
        raw_selectors=[
            {"petri_object": "cell", "expect_hash": "ih",
             "agent_loop": 1},  # the wire form, as vllm_xargs lands
            {"petri_object": "cell", "expect_hash": "ih"},
        ],
        token_lookup=lambda rid: toks[rid],
        prompt_len_lookup=lambda rid: len(toks[rid]),
        registry=_FakeRegistry())
    by_id = {s.req_id: s for s in slots.slots}
    assert by_id["a"].ctx is not None and by_id["a"].ctx.agent_loop
    assert not by_id["a"].ctx.response_prefill
    assert by_id["b"].ctx is not None and not by_id["b"].ctx.agent_loop
    assert la._response_start(_resp_op(), by_id["a"].ctx, q[0]) == len(USER)


# --------------------------------------------------------------------------- #
# the proxy: entry, verifier, transport                                        #
# --------------------------------------------------------------------------- #

LENS_CELL = "si_safety__psr_kl_all"

_TOOLS = [{"type": "function", "function": {
    "name": "lookup", "description": "Look something up",
    "parameters": {"type": "object",
                   "properties": {"q": {"type": "string"}},
                   "required": ["q"]}}}]


def _tool_call_msg(content=""):
    # OpenAI wire form: arguments is a JSON STRING
    return {"role": "assistant", "content": content,
            "tool_calls": [{"id": "call_0001", "type": "function",
                            "function": {"name": "lookup",
                                         "arguments": json.dumps(
                                             {"q": "weather"})}}]}


def _agent_chat(model, *, shape="tool_result", declare=True, tools=True,
                agp=None, system=None):
    msgs = []
    if system is not None:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": "Please look up the weather."})
    msgs.append(_tool_call_msg())
    if shape == "tool_result":
        msgs.append({"role": "tool", "tool_call_id": "call_0001",
                     "content": "Sunny."})
    body = {"model": model, "messages": msgs, "max_tokens": 16,
            "temperature": 0.0}
    if tools:
        body["tools"] = _TOOLS
    if agp is not None:
        body["add_generation_prompt"] = agp
    if declare:
        body[px.AGENT_LOOP_KEY] = True
    return body


def _single_chat(model, *, declare=False):
    body = {"model": model, "max_tokens": 16,
            "messages": [{"role": "user", "content": "Hello there."}]}
    if declare:
        body[px.AGENT_LOOP_KEY] = True
    return body


class _FakeRender:
    """A scripted AgentLoopRender: fixed header ids, and a render that is a
    function of the message STRUCTURE — an agent-loop structure renders as
    the synthetic non-tail agent-loop ids, anything else as an ordinary
    tail-header render — unless a test scripts ``ids_for`` itself."""

    header_ids = HEADER

    def __init__(self, ids_for=None):
        self.calls = []
        self._ids_for = ids_for

    def render(self, messages, tools):
        self.calls.append((list(messages), tools))
        if self._ids_for is not None:
            return self._ids_for(messages, tools)
        if px.agent_loop_shape({"messages": list(messages)}) is None:
            return TAIL_IDS
        return AGENT_IDS


_UNSET = object()


def _agent_proxy(tmp_path, *, render=_UNSET, entry=False, backend=None,
                 base_revision=None):
    """A proxy with the entry OFF unless asked (the verifier tests send
    client declarations and must see exactly what the client sent — the
    proxy's own default is ON, tested separately)."""
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = _cells(spec, source)
    backend = backend or StubBackend()
    render = _FakeRender() if render is _UNSET else render
    proxy = px.RoutingProxy(
        px.routes_from_cells(cells), backend,
        base_model=spec.target_model, serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cells, source),
        exact_replay_strategy=None,
        agent_loop_render=render, petri_agent_loop_entry=entry,
        base_revision=base_revision)
    return proxy, backend, cells


def _vmodel(cells, cell_id):
    return next(r.virtual_model for r in px.routes_from_cells(cells)
                if r.cell_id == cell_id)


def _ask(proxy, request):
    return asyncio.run(proxy.handle_chat_completions(request))


def test_agent_loop_shape_classifier():
    """The structural half, both shapes and every non-shape."""
    m = "petri__x"
    assert px.agent_loop_shape(_agent_chat(m)) == px.AGENT_LOOP_SHAPE_TOOL_RESULT
    assert px.agent_loop_shape(_agent_chat(m, shape="tool_call")) \
        == px.AGENT_LOOP_SHAPE_TOOL_CALL
    assert px.agent_loop_shape(_agent_chat(m, system="Be safe.")) \
        == px.AGENT_LOOP_SHAPE_TOOL_RESULT
    assert px.agent_loop_shape(_agent_chat(m, agp=True)) \
        == px.AGENT_LOOP_SHAPE_TOOL_RESULT
    # non-shapes
    assert px.agent_loop_shape(_single_chat(m)) is None
    assert px.agent_loop_shape(_agent_chat(m, agp=False)) is None
    seeded = _agent_chat(m)
    seeded["continue_final_message"] = True
    assert px.agent_loop_shape(seeded) is None
    plain_assistant = _agent_chat(m, shape="tool_call")
    plain_assistant["messages"][-1] = {"role": "assistant",
                                       "content": "Done."}
    assert px.agent_loop_shape(plain_assistant) is None
    empty_calls = _agent_chat(m, shape="tool_call")
    empty_calls["messages"][-1]["tool_calls"] = []
    assert px.agent_loop_shape(empty_calls) is None
    # system + a single non-system message is single-turn
    one = {"messages": [{"role": "system", "content": "s"},
                        {"role": "tool", "tool_call_id": "c",
                         "content": "x"}]}
    assert px.agent_loop_shape(one) is None
    assert px.agent_loop_shape({"messages": "nope"}) is None
    assert px.agent_loop_shape({"messages": [1, 2]}) is None


def test_declare_agent_loop_entry_attaches_only_on_shape():
    m = "petri__x"
    req = _agent_chat(m, declare=False)
    out = px.declare_agent_loop(req)
    assert out[px.AGENT_LOOP_KEY] is True
    assert px.AGENT_LOOP_KEY not in req              # a copy, never a mutation
    assert px.AGENT_LOOP_KEY not in px.declare_agent_loop(_single_chat(m))
    # an explicit client statement, either way, is respected
    explicit = _agent_chat(m, declare=False)
    explicit[px.AGENT_LOOP_KEY] = False
    assert px.declare_agent_loop(explicit)[px.AGENT_LOOP_KEY] is False
    assert px.declare_agent_loop(_agent_chat(m))[px.AGENT_LOOP_KEY] is True


def test_declared_agent_loop_travels_as_a_selector_fact(tmp_path):
    proxy, backend, cells = _agent_proxy(tmp_path)
    vm = _vmodel(cells, LENS_CELL)
    for shape in ("tool_result", "tool_call"):
        resp = _ask(proxy, _agent_chat(vm, shape=shape))
        assert resp["model"] == vm
        up = backend.payloads[-1]
        sel = up[px.UPSTREAM_EXTRA_ARGS_KEY]
        # the transport form: int 1, never a bool (vllm_xargs lax-coerces
        # JSON bools to ints; the proxy sends the one honest spelling)
        assert sel["agent_loop"] == 1 and not isinstance(sel["agent_loop"],
                                                         bool)
        assert "response_prefill" not in sel
        assert sel["petri_object"] == LENS_CELL
        # the client key never travels upstream; the transcript does
        assert px.AGENT_LOOP_KEY not in up
        assert up["tools"] == _TOOLS
        assert up["messages"][-1]["role"] == ("tool" if shape == "tool_result"
                                              else "assistant")
    # the verifier rendered the request's OWN messages + tools
    render = proxy._agent_loop_render
    assert len(render.calls) == 2
    assert render.calls[-1][1] == _TOOLS


def test_undeclared_agent_loop_shape_is_forwarded_and_contained_downstream(
        tmp_path):
    """THE PINNED DIRECTION AT THE PROXY: an undeclared agent-loop shape is
    not touched by the gate — it goes upstream with no fact, the worker
    refuses it at the adapter and serves it unsteered (contained), and the
    proxy's lens post-check then refuses the completion as a 500. Restoring
    multi-turn serving is the declaration, never a relaxation."""
    backend = StubBackend(request_failures={
        "cmpl-stub": {"cell_id": LENS_CELL, "error_type": "AdapterError",
                      "error": "this prompt does not end at the header ... "
                               "only served when the request DECLARES it"}})
    proxy, backend, cells = _agent_proxy(tmp_path, backend=backend)
    vm = _vmodel(cells, LENS_CELL)
    with pytest.raises(px.ProxyRefusal, match="served unsteered") as e:
        _ask(proxy, _agent_chat(vm, declare=False))
    assert e.value.http_status == 500
    assert e.value.code == "lens_request_failed"
    up = backend.payloads[-1]                      # it DID go upstream ...
    assert "agent_loop" not in up[px.UPSTREAM_EXTRA_ARGS_KEY]  # ... undeclared
    assert proxy._agent_loop_render.calls == []    # and was never rendered


def test_declaration_on_a_non_lens_route_refuses(tmp_path):
    proxy, backend, cells = _agent_proxy(tmp_path)
    vm = _vmodel(cells, "none")  # passthrough: no mask to re-locate
    with pytest.raises(px.ProxyRefusal, match="not lens-served") as e:
        _ask(proxy, _agent_chat(vm))
    assert e.value.code == "agent_loop_not_lens"
    assert e.value.http_status == 400
    # ... but an undeclared agent-loop turn on a MASKLESS route is fine
    resp = _ask(proxy, _agent_chat(vm, declare=False))
    assert resp["model"] == vm
    assert backend.payloads[-1]["messages"][-1]["role"] == "tool"


def test_declaration_structure_mismatches_refuse(tmp_path):
    proxy, backend, cells = _agent_proxy(tmp_path)
    vm = _vmodel(cells, LENS_CELL)

    def refused(req, code, match):
        with pytest.raises(px.ProxyRefusal, match=match) as e:
            _ask(proxy, req)
        assert e.value.code == code
        assert e.value.http_status == 400

    # single-turn: a declaration with no agent loop behind it
    refused(_single_chat(vm, declare=True), "agent_loop_mismatch",
            "not a turn of the agent loop")
    refused(_agent_chat(vm, agp=False), "agent_loop_mismatch",
            "not a turn of the agent loop")
    seeded = _agent_chat(vm)
    seeded["continue_final_message"] = True
    refused(seeded, "agent_loop_mismatch", "not a turn of the agent loop")
    plain = _agent_chat(vm, shape="tool_call")
    plain["messages"][-1] = {"role": "assistant", "content": "Done."}
    refused(plain, "agent_loop_mismatch", "not a turn of the agent loop")
    both = _agent_chat(vm)
    both[px.RESPONSE_PREFILL_KEY] = True
    refused(both, "agent_loop_mismatch", "BOTH")
    bad = _agent_chat(vm)
    bad[px.AGENT_LOOP_KEY] = "true"
    refused(bad, "agent_loop_invalid", "JSON boolean")
    assert backend.payloads == []  # every one refused BEFORE the engine


def test_declaration_render_mismatches_refuse(tmp_path):
    """The render half of the verifier — the part the entry cannot see:
    a structurally valid declaration whose render has no header anywhere,
    or ENDS at the header (an ordinary render), or does not render at
    all, is a 400."""
    def check(render, code, match):
        proxy, backend, cells = _agent_proxy(tmp_path, render=render)
        vm = _vmodel(cells, LENS_CELL)
        with pytest.raises(px.ProxyRefusal, match=match) as e:
            _ask(proxy, _agent_chat(vm))
        assert e.value.code == code and e.value.http_status == 400
        assert backend.payloads == []

    check(_FakeRender(lambda m, t: (1, 2, 3, 4, 5, 6)),
          "agent_loop_mismatch", "occur NOWHERE")
    check(_FakeRender(lambda m, t: TAIL_IDS),
          "agent_loop_mismatch", "ENDS at the assistant header")

    def boom(m, t):
        raise ValueError("template refused the transcript")
    check(_FakeRender(boom), "agent_loop_render_failed", "does not render")


def test_declaration_with_an_unloadable_probe_refuses(tmp_path, monkeypatch):
    """No explicit probe and the served tokenizer cannot be loaded -> the
    declaration cannot be verified -> refused, never honoured on structure
    alone; undeclared traffic still flows, and the refusal is per request
    (a later successful load would serve)."""
    monkeypatch.setattr(px, "load_served_tokenizer",
                        lambda *a, **k: (_ for _ in ()).throw(
                            OSError("offline: no such tokenizer")))
    proxy, backend, cells = _agent_proxy(tmp_path, render=None,
                                         base_revision="rev0")
    vm = _vmodel(cells, LENS_CELL)
    with pytest.raises(px.ProxyRefusal,
                       match="could not load the served tokenizer") as e:
        _ask(proxy, _agent_chat(vm))
    assert e.value.code == "agent_loop_unverifiable"
    assert e.value.http_status == 400
    assert "@rev0" in str(e.value)  # the pin it tried is named
    assert backend.payloads == []
    _ask(proxy, _single_chat(vm))
    assert "agent_loop" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    meta = proxy.models_payload()["petri_proxy"]
    assert meta["agent_loop_render"] == {
        "mode": "lazy", "loaded": False,
        "source": f"{proxy.base_model}@rev0"}
    assert meta["petri_agent_loop_entry"] is False


def test_lazy_probe_builds_once_from_the_table_pin(tmp_path, monkeypatch):
    """No explicit probe: the first agent-loop turn builds one from
    base_model @ base_revision via load_served_tokenizer, exactly once,
    and later turns reuse it."""
    calls = []

    class _LazyStub(_FakeRender):
        def __init__(self, tok):
            super().__init__()
            self.tok = tok

    def load_served_tokenizer(name, *, revision=None, chat_template=None,
                              local_files_only=False):
        calls.append((name, revision, chat_template, local_files_only))
        return "TOK"

    monkeypatch.setattr(px, "load_served_tokenizer", load_served_tokenizer)
    monkeypatch.setattr(px, "TokenizerAgentLoopRender", _LazyStub)
    proxy, backend, cells = _agent_proxy(tmp_path, render=None, entry=True,
                                         base_revision="rev0")
    vm = _vmodel(cells, LENS_CELL)
    assert proxy.models_payload()["petri_proxy"]["agent_loop_render"][
        "loaded"] is False
    for _ in range(2):
        _ask(proxy, _agent_chat(vm, declare=False))  # the entry declares
        assert backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY][
            "agent_loop"] == 1
    assert calls == [(proxy.base_model, "rev0", None, True)]  # built ONCE,
    # from the table pin, local files only (never a request-time download)
    assert proxy.models_payload()["petri_proxy"]["agent_loop_render"] == {
        "mode": "lazy", "loaded": True, "source": f"{proxy.base_model}@rev0"}


def test_the_entry_is_on_by_default(tmp_path, monkeypatch):
    """A proxy built with NO agent-loop arguments — every driver in
    scripts/ — has the entry armed and a lazy probe: the review's finding
    (correct in the tested entry point, disabled in the launcher) cannot
    recur by omission."""
    spec = _spec(serving_mode="interleaved")
    source = _source(tmp_path)
    cells = _cells(spec, source)
    backend = StubBackend()
    proxy = px.RoutingProxy(  # BARE: no agent-loop arguments at all
        px.routes_from_cells(cells), backend,
        base_model=spec.target_model, serving_mode="interleaved",
        prompted_prefixes=px.prompted_prefix_specs(cells, source),
        exact_replay_strategy=None)
    meta = proxy.models_payload()["petri_proxy"]
    assert meta["petri_agent_loop_entry"] is True
    assert meta["agent_loop_render"] == {
        "mode": "lazy", "loaded": False,
        "source": f"{spec.target_model}@unpinned"}
    # and it declares: swap in a scripted probe and send a real shape
    proxy._agent_loop_render = _FakeRender()
    vm = _vmodel(cells, LENS_CELL)
    _ask(proxy, _agent_chat(vm, declare=False))
    assert backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]["agent_loop"] == 1
    # single-turn traffic through the same default proxy is untouched
    _ask(proxy, _single_chat(vm))
    assert "agent_loop" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]


def test_entry_armed_declares_and_serves_agent_loop_turns(tmp_path):
    """The Petri-facing entry on: an agent-loop turn arriving WITHOUT the
    key (as Petri's client sends it) is declared at entry, verified, and
    travels as the fact; a single-turn request is left alone; an explicit
    client False is respected (forwarded undeclared)."""
    proxy, backend, cells = _agent_proxy(tmp_path, entry=True)
    vm = _vmodel(cells, LENS_CELL)
    for shape in ("tool_result", "tool_call"):
        _ask(proxy, _agent_chat(vm, shape=shape, declare=False))
        sel = backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
        assert sel["agent_loop"] == 1 and not isinstance(sel["agent_loop"],
                                                         bool)
        assert px.AGENT_LOOP_KEY not in backend.payloads[-1]
    _ask(proxy, _single_chat(vm))
    assert "agent_loop" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    explicit = _agent_chat(vm, declare=False)
    explicit[px.AGENT_LOOP_KEY] = False
    _ask(proxy, explicit)
    assert "agent_loop" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    meta = proxy.models_payload()["petri_proxy"]
    assert meta["petri_agent_loop_entry"] is True
    assert meta["agent_loop_render"]["mode"] == "explicit"
    assert meta["agent_loop_render"]["loaded"] is True


def test_entry_leaves_tail_renders_undeclared(tmp_path):
    """The render half of the entry (Llama-3.1's case): an agent-loop
    STRUCTURE whose render ENDS at the header is an ordinary render — the
    entry attaches nothing and the tail arm serves it undeclared, exactly
    as before this landing. A structure-only entry would have 400'd every
    llama tool turn (test_petri_proxy's round-trip pin caught it)."""
    proxy, backend, cells = _agent_proxy(
        tmp_path, entry=True, render=_FakeRender(lambda m, t: TAIL_IDS))
    vm = _vmodel(cells, LENS_CELL)
    resp = _ask(proxy, _agent_chat(vm, declare=False))
    assert resp["model"] == vm
    assert "agent_loop" not in backend.payloads[-1][px.UPSTREAM_EXTRA_ARGS_KEY]
    assert px.AGENT_LOOP_KEY not in backend.payloads[-1]
    # the pure entry states the same rule
    req = _agent_chat("petri__x", declare=False)
    assert px.AGENT_LOOP_KEY not in px.declare_agent_loop(
        req, probe=_FakeRender(lambda m, t: TAIL_IDS))
    assert px.declare_agent_loop(
        req, probe=_FakeRender(lambda m, t: AGENT_IDS))[px.AGENT_LOOP_KEY] \
        is True
    assert px.declare_agent_loop(req)[px.AGENT_LOOP_KEY] is True  # structural half


def test_entry_declaration_is_still_verified_against_the_render(tmp_path):
    """Two layers, both required: an entry-attached declaration on a
    render whose header occurs NOWHERE is refused by the verifier exactly
    like a client's (a loud 400 instead of the adapter's contained
    refusal) — the entry never bypasses verification."""
    proxy, backend, cells = _agent_proxy(
        tmp_path, entry=True,
        render=_FakeRender(lambda m, t: (1, 2, 3, 4, 5, 6)))
    vm = _vmodel(cells, LENS_CELL)
    with pytest.raises(px.ProxyRefusal, match="occur NOWHERE") as e:
        _ask(proxy, _agent_chat(vm, declare=False))
    assert e.value.code == "agent_loop_mismatch"
    assert backend.payloads == []


def test_entry_is_loud_on_a_turn_it_cannot_render(tmp_path):
    """An agent-loop-structured lens turn the probe cannot render (a
    non-text part) is a 400 at the entry, never a silent forward."""
    def boom(m, t):
        raise ValueError("image part")
    proxy, backend, cells = _agent_proxy(tmp_path, entry=True,
                                         render=_FakeRender(boom))
    vm = _vmodel(cells, LENS_CELL)
    with pytest.raises(px.ProxyRefusal, match="could not render") as e:
        _ask(proxy, _agent_chat(vm, declare=False))
    assert e.value.code == "agent_loop_render_failed"
    assert backend.payloads == []
    _ask(proxy, _single_chat(vm))  # single-turn never touches the probe
    assert backend.payloads[-1]["model"]


def test_entry_never_declares_on_a_non_lens_route_by_accident(tmp_path):
    """The verifier refuses a declaration on a maskless route, so an entry
    that declared on structure alone would break every non-lens leg of an
    audit (the undefended base and the LoRA cells talk through the same
    proxy). With the entry armed, an agent-loop turn to the PASSTHROUGH
    vmodel and to a LORA vmodel must still serve, undeclared — the entry
    consults the route, not just the structure. (This test caught exactly
    that defect in the first draft of the entry.)"""
    proxy, backend, cells = _agent_proxy(tmp_path, entry=True)
    for cell_id in ("none", "si_safety__lora_r1"):
        vm = _vmodel(cells, cell_id)
        resp = _ask(proxy, _agent_chat(vm, declare=False))
        assert resp["model"] == vm, cell_id
        assert px.UPSTREAM_EXTRA_ARGS_KEY not in backend.payloads[-1], cell_id
        assert px.AGENT_LOOP_KEY not in backend.payloads[-1], cell_id
    # the pure entry function states the same rule directly
    req = _agent_chat("petri__x", declare=False)
    assert px.AGENT_LOOP_KEY not in px.declare_agent_loop(req, lens_route=False)
    assert px.declare_agent_loop(req, lens_route=True)[px.AGENT_LOOP_KEY] is True


def test_selector_smuggling_of_the_fact_still_refuses(tmp_path):
    proxy, _, cells = _agent_proxy(tmp_path)
    vm = _vmodel(cells, LENS_CELL)
    req = _agent_chat(vm)
    req[px.UPSTREAM_EXTRA_ARGS_KEY] = {"agent_loop": 1}
    with pytest.raises(px.ProxyRefusal, match="reserved selector"):
        _ask(proxy, req)


def test_wire_roundtrip_cpu_half(tmp_path):
    """proxy payload -> JSON transport -> the worker's extractor: the fact
    arrives as the transport int and parses as exactly True (the vllm
    half — the pinned ChatCompletionRequest's dict[str, str|int|float]
    lax coercion — is capctrl_served.py's preflight, where vllm imports)."""
    proxy, backend, cells = _agent_proxy(tmp_path)
    vm = _vmodel(cells, LENS_CELL)
    _ask(proxy, _agent_chat(vm))
    wire = json.loads(json.dumps(backend.payloads[-1]))
    xargs = wire[px.UPSTREAM_EXTRA_ARGS_KEY]
    assert xargs["agent_loop"] == 1 and type(xargs["agent_loop"]) is int
    assert all(isinstance(v, (str, int, float)) and not isinstance(v, bool)
               for v in xargs.values())  # the transport's value types
    sel = lw.extract_selector(xargs)
    assert sel is not None and sel["agent_loop"] is True
    assert "response_prefill" not in sel


# --------------------------------------------------------------------------- #
# the positive control CAN FAIL                                                #
# --------------------------------------------------------------------------- #

_VOCAB = 5


def _token1(ids, vector, *, declared, readout, hidden):
    """A toy served step for one declared agent-loop render: run the
    prefill rows through the hook, read the FINAL prefill row through a
    fixed linear readout, and return its argmax — 'token 1'. The forward
    that emits token 1 IS the final prefill row, so whether steering can
    change token 1 is exactly whether the mask covers that row."""
    op = _resp_op(vector=vector)
    ctx = _prefill_ctx(ids, agent_loop=declared)
    out = la.apply_hook_ops(hidden.clone(), (op,), 1.0, layer=0, ctx=ctx)
    out = hidden if out is None else out
    return int((out[-1] @ readout).argmax())


def _bites_evidence(ids, vector, *, readout, hidden):
    steered = _token1(ids, vector, declared=True, readout=readout,
                      hidden=hidden)
    scale0 = _token1(ids, torch.zeros_like(vector), declared=True,
                     readout=readout, hidden=hidden)
    return {"arms": {"psr_kl_all": {"steered_ids": [[steered]],
                                    "scale0_ids": [[scale0]]}}}


def test_positive_control_passes_on_a_live_arm_and_fails_on_an_inert_one():
    """fidelity._check_steering_bites over a declared agent-loop render:
    a LIVE arm (a vector aimed at the readout's least-likely token, large
    enough to flip the argmax) changes token 1 relative to its scale-0
    twin — the declared mask reaches the final prefill row — so the check
    passes; an INERT arm (zero vector) leaves token 1 identical and the
    check RAISES. The failing direction is proven, not assumed."""
    torch.manual_seed(0)
    hidden = torch.randn(len(AGENT_IDS), D)
    readout = torch.randn(D, _VOCAB)
    base_logits = hidden[-1] @ readout
    target = int(base_logits.argmin())
    live = 50.0 * readout[:, target]  # aims the final row at `target`
    ev = _bites_evidence(AGENT_IDS, live, readout=readout, hidden=hidden)
    assert ev["arms"]["psr_kl_all"]["steered_ids"] == [[target]]
    assert ev["arms"]["psr_kl_all"]["steered_ids"] != \
        ev["arms"]["psr_kl_all"]["scale0_ids"]
    diag: dict = {}
    fd._check_steering_bites(ev, diag)          # passes: the arm bites
    assert diag["arms"]["psr_kl_all"]["n_differing"] == 1

    inert = torch.zeros(D)
    ev0 = _bites_evidence(AGENT_IDS, inert, readout=readout, hidden=hidden)
    with pytest.raises(fd.FidelityError, match="NO difference"):
        fd._check_steering_bites(ev0, {})       # fails: nothing changed


def test_positive_control_cannot_run_undeclared():
    """The same live arm on the same render, UNDECLARED, does not produce
    a wrong reading — it refuses outright (the pinned direction), which is
    why the control is meaningful only through the declared path."""
    torch.manual_seed(0)
    hidden = torch.randn(len(AGENT_IDS), D)
    readout = torch.randn(D, _VOCAB)
    with pytest.raises(la.AdapterError, match="DECLARES"):
        _token1(AGENT_IDS, 50.0 * readout[:, 0], declared=False,
                readout=readout, hidden=hidden)
