"""The DECLARED THINKING-ON header, end to end, mirroring
``test_response_prefill_serving.py``: the locator picks the ON header
under the ``thinking_on`` fact and refuses without the op's second header;
the tail rule holds under both headers; ``parse_selector`` strictness and
the exclusivity of the three facts; the worker threads the fact into
exactly the flagged request; ``SelectionRecord`` files ``facts`` and
``prefill_masked_rows`` (7 vs 5 on a two-header op); the six rules of
``attach_thinking_header``; the export/load round trip through
``export_worker_registry`` -> ``WorkerObjectRegistry.load`` with the
``assistant_header_ids_thinking_on`` index key; the no-op twin's DONOR rule
(a ``none`` + ``@thinking`` registry gives the twin the declaration and the
ON header, reported under ``none__lens0``; a ``none`` + base registry
reports no twin entry; a position variant is never a donor);
``_assert_ops_express_declarations`` for all three declarations and their
refusals; ``verify_assistant_header_thinking_on`` on a synthetic Qwen-shaped
template and, cache-permitting, the REAL pinned Qwen3.5-4B tokenizer
(``HF_HUB_OFFLINE``, local files only — a clone without the cache skips).

All offline, all synthetic: invented token ids, the proxy tests' canonical
cell set, placeholder digests — no real recorded data.
"""

from __future__ import annotations

import dataclasses
import io
import json
import os
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sisteer import chat_markers as cmk  # noqa: E402
from sisteer.capeval import manifest as cm  # noqa: E402
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402
from sisteer.capeval.serving import lens_adapters as la  # noqa: E402
from sisteer.capeval.serving import lens_steer as ls  # noqa: E402
from sisteer.capeval.serving import lens_worker as lw  # noqa: E402
from sisteer.capeval.serving import serve_declaration as sd  # noqa: E402

from test_petri_proxy import _behavior_files, _spec  # noqa: E402

#: a synthetic Qwen-shaped pair: the OFF header ends in a CLOSED empty think
#: block (7 tokens), the ON header in an OPEN think tag (5 tokens) — they
#: share the first 4 ids, exactly as the real template's do
OFF = (901, 902, 903, 904, 905, 906, 907)
ON = (901, 902, 903, 904, 908)
D = 8
PSR_BASE = "si_safety__psr_kl_all"
THINKING = f"{PSR_BASE}@thinking"
QWEN_4B = ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")


def _resp_op(**over):
    kw = dict(op=la.OP_ADD_VECTOR, layer=0, positions=la.POSITIONS_RESPONSE,
              header_ids=OFF, vector=torch.ones(D, dtype=torch.float32))
    kw.update(over)
    return la.HookOp(**kw)


def _two_header_op(**over):
    return _resp_op(thinking_header_ids=ON, **over)


def _prefill_ctx(ids, **facts):
    return la.HookContext(la.PHASE_PREFILL, tuple(range(len(ids))), tuple(ids),
                          **facts)


def _cached_tok(model, revision):
    transformers = pytest.importorskip("transformers")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    try:
        return transformers.AutoTokenizer.from_pretrained(
            model, revision=revision, local_files_only=True)
    except Exception as e:  # noqa: BLE001 — not cached here: skip, never fetch
        pytest.skip(f"pinned {model} tokenizer @ {revision[:8]} not in the local HF "
                    f"cache ({type(e).__name__})")


# --------------------------------------------------------------------------- #
# HookOp / HookContext: the second header and the fact                        #
# --------------------------------------------------------------------------- #


def test_hookop_second_header_rules():
    op = _two_header_op()
    assert op.thinking_header_ids == ON and op.header_ids == OFF
    # legal on header-located modes only
    with pytest.raises(la.AdapterError, match="does not take thinking_header_ids"):
        la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D),
                  thinking_header_ids=ON)
    with pytest.raises(la.AdapterError, match="does not take thinking_header_ids"):
        la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D),
                  positions=la.POSITIONS_LAST, thinking_header_ids=ON)
    # non-empty, non-negative ints
    with pytest.raises(la.AdapterError, match="is empty"):
        _resp_op(thinking_header_ids=())
    with pytest.raises(la.AdapterError, match="non-negative ints"):
        _resp_op(thinking_header_ids=(1, -2))
    # a second header EQUAL to the first is the header-INVARIANT template's case
    # (gemma-4): attached like any other (plan P9 withdrawn)
    assert _resp_op(thinking_header_ids=OFF).thinking_header_ids == OFF
    # a materialized copy keeps it (dataclasses.replace carries the field)
    (m,) = la.materialize_ops((op,), device=torch.device("cpu"), dtype=torch.float32)
    assert m.thinking_header_ids == ON


def test_hookcontext_thinking_on_is_a_strict_exclusive_bool():
    ctx = _prefill_ctx((1, 2, *ON), thinking_on=True)
    assert ctx.thinking_on is True
    with pytest.raises(la.AdapterError, match="must be a bool"):
        la.HookContext(la.PHASE_PREFILL, (0, 1), (1, 2), thinking_on=1)  # type: ignore[arg-type]
    for other in ("response_prefill", "agent_loop"):
        with pytest.raises(la.AdapterError, match="beside a non-tail declaration"):
            la.HookContext(la.PHASE_PREFILL, (0, 1), (1, 2), thinking_on=True,
                           **{other: True})
    # DecodeRows is UNCHANGED: thinking is not a decode-phase fact
    assert {f.name for f in dataclasses.fields(la.DecodeRows)} == {"response_prefill", "agent_loop"}


# --------------------------------------------------------------------------- #
# the locator: WHICH header, under WHICH fact                                 #
# --------------------------------------------------------------------------- #


def test_locator_picks_the_on_header_under_the_fact():
    op = _two_header_op()
    ids_on = (1, 2, 3, *ON)
    ids_off = (1, 2, 3, *OFF)
    # the declared ON render locates the ON header at the tail
    assert la._response_start(op, _prefill_ctx(ids_on, thinking_on=True), len(ids_on)) == 3
    # the OFF render, undeclared, locates the OFF header exactly as before
    assert la._response_start(op, _prefill_ctx(ids_off), len(ids_off)) == 3
    # the tail rule holds against the OTHER header in both directions
    with pytest.raises(la.AdapterError, match="does not end at the"):
        la._response_start(op, _prefill_ctx(ids_on), len(ids_on))
    with pytest.raises(la.AdapterError, match="does not end at the"):
        la._response_start(op, _prefill_ctx(ids_off, thinking_on=True), len(ids_off))
    # the masked rows: 7 (OFF) vs 5 (ON) on the two-header op
    assert la.prefill_mask_rows((op,), _prefill_ctx(ids_off)) == len(OFF)
    assert la.prefill_mask_rows((op,), _prefill_ctx(ids_on, thinking_on=True)) == len(ON)
    # an `all` op counts the whole prompt; a decode ctx is None
    assert la.prefill_mask_rows((la.HookOp(op=la.OP_ADD_VECTOR, layer=0,
                                           vector=torch.ones(D)),), _prefill_ctx(ids_on)) == len(ids_on)
    assert la.prefill_mask_rows((op,), la.HookContext(la.PHASE_DECODE, (9,), (5,))) is None


def test_locator_refuses_the_fact_on_an_op_without_a_second_header():
    ids_on = (1, 2, 3, *ON)
    with pytest.raises(la.AdapterError, match="does not declare") as ei:
        la._response_start(_resp_op(), _prefill_ctx(ids_on, thinking_on=True), len(ids_on))
    assert "@thinking" in str(ei.value)          # names the missing CELL declaration


def test_undeclared_thinking_on_render_message_names_the_gate():
    """The tail refusal on an ON render arriving UNDECLARED names the
    petri_thinking_on gate beside the two non-tail gates."""
    ids_on = (1, 2, 3, *ON)
    with pytest.raises(la.AdapterError, match="petri_thinking_on"):
        la._response_start(_two_header_op(), _prefill_ctx(ids_on), len(ids_on))


def test_resumed_clause_checks_both_headers():
    op = _two_header_op()
    for cached in ((1, *OFF, 5), (1, *ON, 5)):
        ctx = la.HookContext(la.PHASE_PREFILL, tuple(range(len(cached), len(cached) + len(ON))),
                             ON, cached=len(cached), cached_ids=cached, thinking_on=True)
        with pytest.raises(la.ResumedPrefillRefusal, match="not header-free"):
            la._response_start(op, ctx, len(ON))


# --------------------------------------------------------------------------- #
# attach_thinking_header: the six rules                                        #
# --------------------------------------------------------------------------- #

DECL = {"kind": "psr", "positions": "response", sd.SERVE_OUT_OF_SUPPORT_KEY: ["thinking"]}
UNDECL = {"kind": "psr", "positions": "response"}


def test_attach_rules():
    resp, other = _resp_op(), la.HookOp(op=la.OP_ADD_VECTOR, layer=1, vector=torch.ones(D))
    # 1+2: declares -> required, attached to every response op, others untouched
    out = la.attach_thinking_header((resp, other), DECL, ON)
    assert out[0].thinking_header_ids == ON and out[1].thinking_header_ids is None
    with pytest.raises(la.AdapterError, match="no thinking-ON header was supplied"):
        la.attach_thinking_header((resp,), DECL, None)
    # 3: no header-located op -> vacuous
    with pytest.raises(la.AdapterError, match="VACUOUS"):
        la.attach_thinking_header((other,), DECL, ON)
    # 4: never combined with all-turns
    all_turns = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D),
                          positions=la.POSITIONS_RESPONSE_ALL_TURNS, header_ids=OFF,
                          turn_end_ids=(77, 78))
    with pytest.raises(la.AdapterError, match="never combined"):
        la.attach_thinking_header((all_turns,), DECL, ON)
    # 5: undeclared -> ids must be None; None is a pass-through
    with pytest.raises(la.AdapterError, match="does not declare"):
        la.attach_thinking_header((resp,), UNDECL, ON)
    assert la.attach_thinking_header((resp, other), UNDECL, None) == (resp, other)
    # 6: equal headers are the header-invariant case — attached; a malformed declaration refuses
    assert la.attach_thinking_header((resp,), DECL, OFF)[0].thinking_header_ids == OFF
    with pytest.raises(la.AdapterError, match="malformed serve declaration"):
        la.attach_thinking_header((resp,), {**UNDECL, sd.SERVE_OUT_OF_SUPPORT_KEY: "thinking"}, ON)
    # the stacked wrapper: the declaration lives on the object leg
    assert la.declares_thinking({"kind": la.STACKED_KIND, "object": DECL})
    assert not la.declares_thinking({"kind": la.STACKED_KIND, "object": UNDECL})


# --------------------------------------------------------------------------- #
# selector + worker threading + the record                                     #
# --------------------------------------------------------------------------- #


def test_parse_selector_accepts_the_fact_strictly_and_exclusively():
    for wire in (True, 1):
        assert ls.parse_selector({"petri_object": "c", "expect_hash": "h",
                                  "thinking_on": wire})["thinking_on"] is True
    for wire in (False, 0):
        assert ls.parse_selector({"petri_object": "c", "expect_hash": "h",
                                  "thinking_on": wire})["thinking_on"] is False
    for bad in ("yes", 2, 1.0):
        with pytest.raises(ls.LensSteerError, match="thinking_on"):
            ls.parse_selector({"petri_object": "c", "expect_hash": "h", "thinking_on": bad})
    for other in ("response_prefill", "agent_loop"):
        with pytest.raises(ls.LensSteerError, match="BOTH"):
            ls.parse_selector({"petri_object": "c", "expect_hash": "h",
                               "thinking_on": 1, other: 1})
    assert ls.SELECTOR_FACT_KEYS == ("response_prefill", "agent_loop", "thinking_on")
    assert set(ls.SELECTOR_FACT_KEYS) <= set(ls.SELECTOR_KEYS)


class _FakeRegistry:
    def __init__(self):
        self.obj = ls.LoadedObject(cell_id="cell", kind="obj_psr_kl_all", expect_hash="ih",
                                   ops=(_two_header_op(),), effective_scale=1.0,
                                   serve_declarations=dict(sd.variant_declarations("thinking")))

    def select(self, petri_object, expect_hash, scale=None):
        return self.obj


def test_slot_builder_threads_the_fact_per_request_and_the_record_files_it():
    ids_a = [1, 2, 3, *ON]     # flagged: the declared ON render
    ids_b = [9, 8, 7, *OFF]    # unflagged: the OFF render
    q = [len(ids_a), len(ids_b)]
    qsl = [0, q[0], q[0] + q[1]]
    toks = {"a": ids_a, "b": ids_b}
    reg = _FakeRegistry()
    slots = lw.build_request_slots(
        query_start_loc=qsl, seq_lens=q, num_actual_tokens=sum(q),
        positions=list(range(q[0])) + list(range(q[1])), req_ids=["a", "b"],
        raw_selectors=[{"petri_object": "cell", "expect_hash": "ih", "thinking_on": 1},
                       {"petri_object": "cell", "expect_hash": "ih"}],
        token_lookup=lambda rid: toks[rid], prompt_len_lookup=lambda rid: len(toks[rid]),
        registry=reg)
    by_id = {s.req_id: s for s in slots.slots}
    assert by_id["a"].ctx.thinking_on and not by_id["b"].ctx.thinking_on
    assert la._response_start(_two_header_op(), by_id["a"].ctx, q[0]) == 3
    # the SelectionRecord files the facts and the masked-row count per request
    rec = lw.SelectionRecord()
    for rid in ("a", "b"):
        rec.observe(req_id=rid, layer=0, requested="cell", requested_hash="ih",
                    requested_scale=None, applied=reg.obj, modified=True,
                    path=lw.SELECTION_PATH_PER_SLOT, ctx=by_id[rid].ctx)
    d = rec.as_dict()["requests"]
    assert d["a"]["facts"] == {"response_prefill": False, "agent_loop": False, "thinking_on": True}
    assert d["b"]["facts"] == {"response_prefill": False, "agent_loop": False, "thinking_on": False}
    assert d["a"]["prefill_masked_rows"] == len(ON) and d["b"]["prefill_masked_rows"] == len(OFF)
    # the instrument never raises: an unlocatable boundary files null
    rec2 = lw.SelectionRecord()
    rec2.observe(req_id="x", layer=0, requested="cell", requested_hash="ih", requested_scale=None,
                 applied=reg.obj, modified=False, ctx=_prefill_ctx((1, 2, 3, 4, 5, 6, 7, 8)))
    assert rec2.as_dict()["requests"]["x"]["prefill_masked_rows"] is None
    assert rec2.n_record_errors == 0


def _by_id_registry(*objs):
    return ls.WorkerObjectRegistry({o.cell_id: o for o in objs}, assistant_header_ids=OFF,
                                   assistant_header_ids_thinking_on=ON)


def _build_complete_prefills(reg, specs):
    """``build_request_slots`` over ((req_id, ids, raw selector) ...), every
    request a complete cold prefill in batch order."""
    q = [len(ids) for _r, ids, _s in specs]
    qsl = [0]
    for n in q:
        qsl.append(qsl[-1] + n)
    toks = {rid: list(ids) for rid, ids, _s in specs}
    return lw.build_request_slots(
        query_start_loc=qsl, seq_lens=q, num_actual_tokens=sum(q),
        positions=[p for n in q for p in range(n)], req_ids=[rid for rid, _i, _s in specs],
        raw_selectors=[sel for _r, _i, sel in specs], token_lookup=lambda rid: toks[rid],
        prompt_len_lookup=lambda rid: len(toks[rid]), registry=reg)


def _record_as_the_worker_does(fwd):
    """Observe every selector slot exactly as the worker's apply loop does
    (``resumed``, ``ctx`` AND the slot's own diagnostics), and read back the
    two fields finding A1 is about."""
    rec = lw.SelectionRecord()
    for s in fwd.slots:
        rec.observe(req_id=s.req_id, layer=0, requested=s.selector["petri_object"],
                    requested_hash=s.selector["expect_hash"], requested_scale=s.selector.get("scale"),
                    applied=s.obj, modified=False, path=lw.SELECTION_PATH_PER_SLOT,
                    resumed=s.resumed, ctx=s.ctx, facts=s.facts,
                    prefill_masked_rows=s.prefill_masked_rows)
    assert rec.n_record_errors == 0
    return {rid: (e["facts"], e["prefill_masked_rows"]) for rid, e in rec.as_dict()["requests"].items()}


def test_declared_twin_files_the_same_fact_and_width_alone_and_co_batched():
    """Code-review finding A1: the ``none__lens0`` twin (scale 0, its donor's
    two-header op) never needs a positional plan, so alone it takes the
    row-local path with ``ctx=None`` — and the record used to read
    ``thinking_on`` and the masked-row count off the context alone, filing
    ``thinking_on=False`` against a selector that carries the fact, and null
    against a mask the ON header locates. The gate's ``thinking_twin_facts``
    / ``thinking_twin_masked_rows`` checks read exactly these two fields. The
    facts come from the selector, the width from the request's own prefill,
    on BOTH paths."""
    arm = ls.LoadedObject(cell_id="cell", kind="obj_psr_kl_all", expect_hash="ih", ops=(_two_header_op(),),
                          effective_scale=1.0, serve_declarations=dict(sd.variant_declarations("thinking")))
    twin = dataclasses.replace(arm, cell_id="none__lens0", expect_hash="ih0", effective_scale=0.0)
    reg = _by_id_registry(arm, twin)
    ids_on = [1, 2, 3, *ON]
    twin_on = {"petri_object": "none__lens0", "expect_hash": "ih0", "scale": 0.0, "thinking_on": 1}
    twin_off = {"petri_object": "none__lens0", "expect_hash": "ih0", "scale": 0.0}
    arm_on = {"petri_object": "cell", "expect_hash": "ih", "thinking_on": 1}

    alone_on = _build_complete_prefills(reg, [("t", ids_on, twin_on)])
    assert alone_on.plan is None and alone_on.slots[0].ctx is None          # the row-local path
    alone_off = _build_complete_prefills(reg, [("u", [9, 8, 7, *OFF], twin_off)])
    mixed = _build_complete_prefills(reg, [("t", ids_on, twin_on), ("u", [9, 8, 7, *OFF], twin_off),
                                           ("a", ids_on, arm_on)])
    assert mixed.plan is not None and all(s.ctx is not None for s in mixed.slots)   # the plan path
    want_on = ({"response_prefill": False, "agent_loop": False, "thinking_on": True}, len(ON))
    want_off = ({"response_prefill": False, "agent_loop": False, "thinking_on": False}, len(OFF))
    assert _record_as_the_worker_does(alone_on)["t"] == _record_as_the_worker_does(mixed)["t"] == want_on
    assert _record_as_the_worker_does(alone_off)["u"] == _record_as_the_worker_does(mixed)["u"] == want_off
    assert _record_as_the_worker_does(mixed)["a"] == want_on                # the steered arm, as before
    # the slot itself carries the diagnostics, identically on both paths
    for a, m in ((alone_on.slots[0], mixed.slots[0]), (alone_off.slots[0], mixed.slots[1])):
        assert (a.facts, a.prefill_masked_rows) == (m.facts, m.prefill_masked_rows)


def test_observe_files_explicit_diagnostics_without_a_context_and_they_win():
    """The record's contract for the worker's call: ``facts`` and
    ``prefill_masked_rows`` handed in explicitly are filed as given with
    ``ctx=None`` (the row-local path), are normalised to the closed fact
    vocabulary, win over a context's reading, and a later decode observation
    never overwrites them; the context-only reading stays the fallback for a
    caller that hands none (the tests above)."""
    obj = ls.LoadedObject(cell_id="cell", kind="obj_psr_kl_all", expect_hash="ih", ops=(_two_header_op(),),
                          effective_scale=0.0)
    rec = lw.SelectionRecord()
    rec.observe(req_id="r", layer=0, requested="cell", requested_hash="ih", requested_scale=0.0,
                applied=obj, modified=False, path=lw.SELECTION_PATH_PER_SLOT, ctx=None,
                facts={"thinking_on": True, "not_a_fact": True}, prefill_masked_rows=len(ON))
    e = rec.as_dict()["requests"]["r"]
    assert e["facts"] == {"response_prefill": False, "agent_loop": False, "thinking_on": True}
    assert e["prefill_masked_rows"] == len(ON)
    # a decode observation with a group context carrying no facts: nothing overwritten
    rec.observe(req_id="r", layer=1, requested="cell", requested_hash="ih", requested_scale=0.0,
                applied=obj, modified=False, path=lw.SELECTION_PATH_GROUPED, ctx=la.DecodeRows(),
                facts={"thinking_on": True}, prefill_masked_rows=None)
    e = rec.as_dict()["requests"]["r"]
    assert e["facts"]["thinking_on"] is True and e["prefill_masked_rows"] == len(ON)
    # explicit facts win over a context that disagrees (the selector is the authority)
    rec.observe(req_id="s", layer=0, requested="cell", requested_hash="ih", requested_scale=0.0,
                applied=obj, modified=False, ctx=_prefill_ctx([1, 2, 3, *OFF]),
                facts={"thinking_on": True}, prefill_masked_rows=len(ON))
    e = rec.as_dict()["requests"]["s"]
    assert e["facts"]["thinking_on"] is True and e["prefill_masked_rows"] == len(ON)
    assert rec.n_record_errors == 0
    # the closed vocabulary is the record's, pinned
    assert lw.selector_facts({"petri_object": "cell", "expect_hash": "ih", "thinking_on": True}) == {
        "response_prefill": False, "agent_loop": False, "thinking_on": True}
    assert lw.selector_facts(None) == {"response_prefill": False, "agent_loop": False, "thinking_on": False}


# --------------------------------------------------------------------------- #
# _assert_ops_express_declarations                                              #
# --------------------------------------------------------------------------- #


def test_ops_express_declarations_all_three_and_refusals():
    resp, two = _resp_op(), _two_header_op()
    allop = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D))
    turns = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(D),
                      positions=la.POSITIONS_RESPONSE_ALL_TURNS, header_ids=OFF,
                      turn_end_ids=(77, 78))
    raw = lambda spec: {"resolved_spec": spec}  # noqa: E731
    thinking = sd.apply_variant(UNDECL, "thinking")
    all_turns = sd.apply_variant(UNDECL, "all_turns")
    all_pos = sd.apply_variant(UNDECL, "all_positions")
    assert ls._assert_ops_express_declarations("c", raw(thinking), [two]) == sd.variant_declarations("thinking")
    assert ls._assert_ops_express_declarations("c", raw(all_turns), [turns]) == sd.variant_declarations("all_turns")
    assert ls._assert_ops_express_declarations("c", raw(all_pos), [allop]) == sd.variant_declarations("all_positions")
    assert ls._assert_ops_express_declarations("c", raw(UNDECL), [resp]) == {}
    # unnamed all-turns / all cells are the ordinary cells they always were
    assert ls._assert_ops_express_declarations("c", raw({"kind": "psr", "positions": "response_all_turns"}), [turns]) == {}
    assert ls._assert_ops_express_declarations("c", raw({"kind": "psr"}), [allop]) == {}
    with pytest.raises(ls.LensSteerError, match="do not all carry the ON header"):
        ls._assert_ops_express_declarations("c", raw(thinking), [resp])
    with pytest.raises(ls.LensSteerError, match="undeclared second header"):
        ls._assert_ops_express_declarations("c", raw(UNDECL), [two])
    with pytest.raises(ls.LensSteerError, match="fit_response_serve_all_turns"):
        ls._assert_ops_express_declarations("c", raw(all_turns), [resp])
    with pytest.raises(ls.LensSteerError, match="fit_response_serve_all_positions"):
        ls._assert_ops_express_declarations("c", raw(all_pos), [resp])
    with pytest.raises(ls.LensSteerError, match="malformed"):
        ls._assert_ops_express_declarations("c", raw({**UNDECL, sd.SERVE_EXPERIMENT_KEY: "nope"}), [resp])


# --------------------------------------------------------------------------- #
# export / load round trip + the twin donor rule                               #
# --------------------------------------------------------------------------- #


def _psr_blob(hdr_len=len(OFF)) -> bytes:
    g = torch.Generator().manual_seed(11)
    blob = {"kind": "psr", "layers": [0], "positions": "response",
            "response_header_len": hdr_len,
            "directions": {"0": torch.randn(D, generator=g) * 0.05},
            "gate_w": {"0": torch.randn(D, generator=g) * 0.3},
            "gate_b": {"0": torch.randn(1, generator=g) * 0.5}}
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _world(tmp_path, cell_ids):
    """A source + enumerated cells for ``cell_ids`` (harness ids; a
    ``@variant`` entry re-serves PSR_BASE's artifact under the overlay)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    artifact = tmp_path / "psr.pt"
    artifact.write_bytes(_psr_blob())
    base_spec = {"kind": "psr", "si_id": "safety", "positions": "response"}
    entries = {"none": dc.RegistryEntry(cell_id="none", kind="none"),
               "prompted__safety": dc.RegistryEntry(cell_id="prompted__safety", kind="prompted",
                                                    si_id="safety", si_prefix=b"synthetic safety"),
               "prompted__constitution": dc.RegistryEntry(cell_id="prompted__constitution",
                                                          kind="prompted", si_id="constitution",
                                                          si_prefix=b"synthetic constitution")}
    for cid in cell_ids:
        if cid in entries:
            continue
        base, variant = sd.split_variant(cid)
        spec = sd.apply_variant(base_spec, variant) if variant else dict(base_spec)
        entries[cid] = dc.RegistryEntry(cell_id=cid, kind="psr", si_id="safety",
                                        object_path=str(artifact), serving_path="lens",
                                        resolved_spec=spec)
    source = dc.MappingCellSource(entries)
    spec = _spec(serving_mode="interleaved", cells=list(cell_ids),
                 prompted_prefix_hashes={"safety": cm.sha256_hex16(b"synthetic safety"),
                                         "constitution": cm.sha256_hex16(b"synthetic constitution")})
    cells = dc.enumerate_cells(spec, source, behavior_files=_behavior_files(spec),
                               prompt_files={"auditor": b"# a\n", "judge": b"# j\n"})
    assert not cells.absent
    return cells, source


def test_export_load_round_trip_and_the_declared_twin(tmp_path):
    cells, source = _world(tmp_path, ["none", THINKING])
    reg = tmp_path / "reg"
    entries = ls.export_worker_registry(cells, source, reg, assistant_header_ids=OFF,
                                        assistant_header_ids_thinking_on=ON,
                                        zero_op_hidden_size=D)
    index = json.loads((reg / ls.REGISTRY_INDEX).read_text())
    assert index[ls.REGISTRY_THINKING_HEADER_KEY] == list(ON)
    twin = next(e for e in entries if e.cell_id == dc.NOOP_LENS_CELL_ID)
    assert twin.donor_cell == THINKING
    assert sd.declarations_of(twin.resolved_spec) == {"serve_out_of_support": ["thinking"]}
    loaded = ls.WorkerObjectRegistry.load(reg)
    assert loaded.serves_thinking_on()
    assert loaded.assistant_header_ids_thinking_on == ON
    for cid in (THINKING, dc.NOOP_LENS_CELL_ID):
        obj = loaded.select(cid, next(o.expect_hash for o in loaded.objects() if o.cell_id == cid),
                            0.0 if cid == dc.NOOP_LENS_CELL_ID else None)
        assert all(op.thinking_header_ids == ON for op in obj.ops)
    # the map: the declared cell AND its twin under its OWN id (Q7)
    assert loaded.serve_declarations_by_cell() == {
        THINKING: {"serve_out_of_support": ["thinking"]},
        dc.NOOP_LENS_CELL_ID: {"serve_out_of_support": ["thinking"]}}
    assert lw.serve_declarations_cells(loaded) == loaded.serve_declarations_by_cell()
    # materialize carries everything
    m = loaded.materialize(device=torch.device("cpu"), dtype=torch.float32)
    assert m.serves_thinking_on() and m.serve_declarations_by_cell() == loaded.serve_declarations_by_cell()
    assert m.assistant_header_ids_thinking_on == ON


def test_none_plus_base_registry_reports_no_twin_entry(tmp_path):
    cells, source = _world(tmp_path, ["none", PSR_BASE])
    reg = tmp_path / "reg"
    ls.export_worker_registry(cells, source, reg, assistant_header_ids=OFF, zero_op_hidden_size=D)
    assert json.loads((reg / ls.REGISTRY_INDEX).read_text())[ls.REGISTRY_THINKING_HEADER_KEY] is None
    loaded = ls.WorkerObjectRegistry.load(reg)
    assert not loaded.serves_thinking_on()
    assert loaded.serve_declarations_by_cell() == {}
    assert lw.serve_declarations_cells(loaded) == {}


def test_mixed_registry_hands_the_on_header_to_the_declared_cell_only(tmp_path):
    """The `declared_thinking` standup (none, base, base@thinking): the
    undeclared base carries no second header, the declared cell and the
    twin (donor = the render-declaring cell, P12) do."""
    cells, source = _world(tmp_path, ["none", PSR_BASE, THINKING])
    reg = tmp_path / "reg"
    entries = ls.export_worker_registry(cells, source, reg, assistant_header_ids=OFF,
                                        assistant_header_ids_thinking_on=ON,
                                        zero_op_hidden_size=D)
    assert next(e for e in entries if e.cell_id == dc.NOOP_LENS_CELL_ID).donor_cell == THINKING
    loaded = ls.WorkerObjectRegistry.load(reg)
    by = {o.cell_id: o for o in loaded.objects()}
    assert all(op.thinking_header_ids is None for op in by[PSR_BASE].ops)
    assert all(op.thinking_header_ids == ON for op in by[THINKING].ops)
    assert all(op.thinking_header_ids == ON for op in by[dc.NOOP_LENS_CELL_ID].ops)
    assert set(loaded.serve_declarations_by_cell()) == {THINKING, dc.NOOP_LENS_CELL_ID}
    # the policy map keys the declared cell under its VARIANT spelling with
    # the op's policy (a response object is conditional) — Q11
    assert loaded.resumed_prefill_policy_by_cell()[THINKING] == la.RESUMED_CONDITIONAL


def test_export_refusals_for_the_on_header(tmp_path):
    cells, source = _world(tmp_path, ["none", THINKING])
    with pytest.raises(ls.LensSteerError, match="no assistant_header_ids_thinking_on"):
        ls.export_worker_registry(cells, source, tmp_path / "r1", assistant_header_ids=OFF,
                                  zero_op_hidden_size=D)
    # an ON header EQUAL to the OFF header is the header-INVARIANT template's case (gemma-4):
    # exported, recorded as the second header, and the loaded registry serves thinking-ON
    # (plan P9's refusal withdrawn)
    ls.export_worker_registry(cells, source, tmp_path / "r2", assistant_header_ids=OFF,
                              assistant_header_ids_thinking_on=OFF, zero_op_hidden_size=D)
    idx = json.loads((tmp_path / "r2" / ls.REGISTRY_INDEX).read_text())
    assert tuple(idx[ls.REGISTRY_THINKING_HEADER_KEY]) == OFF == tuple(idx["assistant_header_ids"])
    eq = ls.WorkerObjectRegistry.load(tmp_path / "r2")
    assert eq.serves_thinking_on() and eq.assistant_header_ids_thinking_on == OFF
    cells2, source2 = _world(tmp_path / "b", ["none", PSR_BASE])
    with pytest.raises(ls.LensSteerError, match="no lens cell declares"):
        ls.export_worker_registry(cells2, source2, tmp_path / "r3", assistant_header_ids=OFF,
                                  assistant_header_ids_thinking_on=ON, zero_op_hidden_size=D)


def test_noop_twin_donor_rule():
    def entry(cid, spec):
        return ls.RegistryExportEntry(cell_id=cid, kind="psr", resolved_spec=spec, expect_hash="h",
                                      artifact="a", sidecar=None, artifact_sha16="s",
                                      artifact_intervention_hash="h", forced_scale=None,
                                      donor_cell=None, validated_by=None)
    base = entry(PSR_BASE, UNDECL)
    thinking = entry(THINKING, sd.apply_variant(UNDECL, "thinking"))
    turns = entry(f"{PSR_BASE}@all_turns", sd.apply_variant(UNDECL, "all_turns"))
    allp = entry(f"{PSR_BASE}@all_positions", sd.apply_variant(UNDECL, "all_positions"))
    other = entry("si_constitution__psr_kl_all", UNDECL)
    # prefer the RENDER-declaring cell over the min-id base
    assert ls.noop_twin_donor([base, thinking]).cell_id == THINKING
    assert ls.noop_twin_donor([thinking, base]).cell_id == THINKING
    # experiment 2's mixed registry: the undeclared base, never a position variant
    assert ls.noop_twin_donor([base, turns, allp]).cell_id == PSR_BASE
    assert ls.noop_twin_donor([turns, allp, other]).cell_id == "si_constitution__psr_kl_all"
    # position variants only -> no donor (the caller arms the zero op)
    assert ls.noop_twin_donor([turns, allp]) is None
    assert ls.noop_twin_donor([]) is None
    # ties break lexicographically within a rank
    assert ls.noop_twin_donor([other, base]).cell_id == "si_constitution__psr_kl_all"


def test_position_variants_only_arm_the_zero_op(tmp_path):
    cells, source = _world(tmp_path, ["none", f"{PSR_BASE}@all_positions"])
    entries = ls.export_worker_registry(cells, source, tmp_path / "reg", assistant_header_ids=OFF,
                                        zero_op_hidden_size=D)
    twin = next(e for e in entries if e.cell_id == dc.NOOP_LENS_CELL_ID)
    assert twin.donor_cell is None and twin.kind == la.ZERO_KIND
    with pytest.raises(ls.LensSteerError, match="never a donor"):
        ls.export_worker_registry(cells, source, tmp_path / "reg2", assistant_header_ids=OFF)


# --------------------------------------------------------------------------- #
# the worker preflight + install report                                        #
# --------------------------------------------------------------------------- #


def test_preflight_runs_the_on_verifier_only_for_a_thinking_registry():
    from test_petri_lens_worker import FakeLayer, FakeView, _obj, _ok_header
    calls = []

    def on_verifier(tok, recorded):
        calls.append(tuple(recorded))

    def make(ops, on_header):
        reg = ls.WorkerObjectRegistry({"c": _obj("c", ops)}, assistant_header_ids=OFF,
                                      assistant_header_ids_thinking_on=on_header)
        view = FakeView(layers=[FakeLayer() for _ in range(2)])
        st = lw.LensWorkerSteering(view=view, registry=reg)
        st.preflight(object(), header_verifier=_ok_header, thinking_header_verifier=on_verifier)
        return st

    make((_resp_op(),), None)
    assert calls == []                              # an ordinary registry never touches it
    st = make((_two_header_op(),), ON)
    assert calls == [ON]
    report = st.install()
    payload = lw.install_report_payload(report)
    assert payload[lw.LENS_SERVE_DECLARATIONS_KEY] == {}   # _obj carries no declaration
    assert lw.DEFAULT_THINKING_HEADER_VERIFIER is ls.verify_assistant_header_thinking_on


# --------------------------------------------------------------------------- #
# verify_assistant_header_thinking_on                                          #
# --------------------------------------------------------------------------- #


class _QwenShapedTok:
    """A synthetic tokenizer whose chat template branches on enable_thinking
    and whose generation prompt stops INSIDE the completed header when ON."""

    chat_template = "{% if enable_thinking %}on{% else %}off{% endif %}"

    def __init__(self):
        self._sisteer_template_kwargs = {"enable_thinking": False}


def test_verify_on_header_refusals_synthetic(monkeypatch):
    tok = _QwenShapedTok()
    monkeypatch.setattr(cmk, "generation_header_ids", lambda t, m=None: OFF)
    monkeypatch.setattr(cmk, "generation_header_ids_under",
                        lambda t, *, enable_thinking: ON if enable_thinking else OFF)
    ls.verify_assistant_header_thinking_on(tok, ON)                  # match
    with pytest.raises(ls.LensSteerError, match="recorded no"):
        ls.verify_assistant_header_thinking_on(tok, None)
    with pytest.raises(ls.LensSteerError, match="mismatch"):
        ls.verify_assistant_header_thinking_on(tok, (1, 2, 3))
    # header-invariant template (gemma-4): the live ON header equals the OFF header — a
    # registry that recorded the equal header passes; one that recorded a MOVING header
    # against this template is the mismatch the guard still catches
    monkeypatch.setattr(cmk, "generation_header_ids_under", lambda t, *, enable_thinking: OFF)
    ls.verify_assistant_header_thinking_on(tok, OFF)
    with pytest.raises(ls.LensSteerError, match="mismatch"):
        ls.verify_assistant_header_thinking_on(tok, ON)
    # no switch: the derivation refuses, re-raised
    def _no_switch(t, *, enable_thinking):
        raise cmk.ChatMarkerError("no thinking switch")
    monkeypatch.setattr(cmk, "generation_header_ids_under", _no_switch)
    with pytest.raises(ls.LensSteerError, match="no thinking-ON render"):
        ls.verify_assistant_header_thinking_on(tok, ON)


def test_verify_on_header_on_the_real_qwen35_4b_tokenizer():
    from sisteer.models import pin_reasoning_template_off
    tok = _cached_tok(*QWEN_4B)
    pin_reasoning_template_off(tok)
    facts = cmk.thinking_header_facts(tok)
    assert facts["switch"] and facts["moves"] and len(facts["off_ids"]) == 7 and len(facts["on_ids"]) == 5
    ls.verify_assistant_header_thinking_on(tok, facts["on_ids"])
    with pytest.raises(ls.LensSteerError, match="mismatch"):
        ls.verify_assistant_header_thinking_on(tok, facts["off_ids"][:5])
    # the OFF verifier is untouched by the ON derivation (the pin is restored)
    ls.verify_assistant_header(tok, facts["off_ids"])
    assert tok._sisteer_template_kwargs == {"enable_thinking": False}


# --------------------------------------------------------------------------- #
# the sampled-rerun lineage's extras, on main's API (merged 2026-09-11)         #
# --------------------------------------------------------------------------- #


def test_serve_declaration_thinking_helpers():
    """The standup-wide standup's spellings of the declaration, built on the
    variant overlay: the value, the overlay, the predicate, and the overlay's
    exactly-once rule (a base already carrying ANY declaration key refuses)."""
    assert sd.thinking_declaration() == {"serve_out_of_support": ["thinking"]}
    assert sd.thinking_declaration() == sd.variant_declarations(sd.VARIANT_THINKING)
    assert sd.declarations_of(UNDECL) == {}
    assert sd.declarations_of(DECL) == {"serve_out_of_support": ["thinking"]}
    assert sd.declares_thinking(DECL) and not sd.declares_thinking(UNDECL)
    assert sd.declare_thinking(UNDECL) == DECL and UNDECL == {"kind": "psr", "positions": "response"}
    with pytest.raises(sd.ServeDeclarationError, match="exactly once"):
        sd.declare_thinking(DECL)
    with pytest.raises(sd.ServeDeclarationError, match="exactly once"):
        sd.declare_thinking({**UNDECL, sd.SERVE_EXPERIMENT_KEY: sd.EXPERIMENT_FIT_RESPONSE_SERVE_ALL_TURNS})
    for bad in ("thinking", [], ["thinking", "thinking"], ["nope"], [1]):
        with pytest.raises(sd.ServeDeclarationError):
            sd.declarations_of({**UNDECL, sd.SERVE_OUT_OF_SUPPORT_KEY: bad})


def test_off_path_is_byte_identical_without_the_fact():
    """A pre-landing op (no second header) under an undeclared context is
    the exact old locator: the same boundary, the same refusals."""
    op = _resp_op()
    ids = (1, 2, 3, *OFF)
    assert la._response_start(op, _prefill_ctx(ids), len(ids)) == 3
    seeded = (1, 2, 3, *OFF, 55, 56)
    assert la._response_start(op, _prefill_ctx(seeded, response_prefill=True), len(seeded)) == 3
    with pytest.raises(la.AdapterError, match="DECLARES"):
        la._response_start(op, _prefill_ctx(seeded), len(seeded))


def test_declared_cell_is_a_distinct_served_identity(tmp_path):
    """The intervention hash covers the declaration: the same bytes under the
    declared spelling hash differently from the OFF base (a store cannot
    confuse the two regimes by hash) — and, declared BY NAME on this tree,
    the vmodel NAME differs too (`petri__<cell>@thinking`), so the drivers
    address the regime they mean."""
    on_cells, _ = _world(tmp_path / "on", ["none", THINKING])
    off_cells, _ = _world(tmp_path / "off", ["none", PSR_BASE])
    h_on = next(c.intervention_hash for c in on_cells.canonical if c.cell_id == THINKING)
    h_off = next(c.intervention_hash for c in off_cells.canonical if c.cell_id == PSR_BASE)
    assert h_on != h_off
    vm_on = next(c.virtual_model_name for c in on_cells.canonical if c.cell_id == THINKING)
    vm_off = next(c.virtual_model_name for c in off_cells.canonical if c.cell_id == PSR_BASE)
    assert vm_on == f"petri__{THINKING}" and vm_off == f"petri__{PSR_BASE}" and vm_on != vm_off


def test_undeclared_registry_index_states_no_second_header(tmp_path):
    """An OFF export states the second header as null (main's index always
    carries the key) and loads with no thinking-ON service anywhere: no
    declared cell, no op with a second header, the twin's donor the min-id
    lens cell exactly as before."""
    cells, source = _world(tmp_path, ["none", PSR_BASE])
    reg = tmp_path / "reg"
    ls.export_worker_registry(cells, source, reg, assistant_header_ids=OFF, zero_op_hidden_size=D)
    index = json.loads((reg / ls.REGISTRY_INDEX).read_text())
    assert index[ls.REGISTRY_THINKING_HEADER_KEY] is None
    loaded = ls.WorkerObjectRegistry.load(reg)
    assert not loaded.serves_thinking_on() and loaded.assistant_header_ids_thinking_on is None
    assert loaded.serve_declarations_by_cell() == {} and lw.serve_declarations_cells(loaded) == {}
    assert all(op.thinking_header_ids is None for o in loaded.objects() for op in o.ops)
    twin = next(o for o in loaded.objects() if o.cell_id == dc.NOOP_LENS_CELL_ID)
    assert twin.effective_scale == 0.0 and twin.serve_declarations == {}


def test_strict_attestation_readers():
    """The standup's two install-report readers: the attested declarations map
    and the attested ON header (`lens_worker.thinking_header_attested`, this
    merge's addition) — every worker must carry the key and agree; a worker
    from an older tree is a REASON, never read as "none"."""
    rep = {"cells": ["c"], lw.LENS_SERVE_DECLARATIONS_KEY: {"c": sd.thinking_declaration()},
           lw.LENS_THINKING_HEADER_KEY: list(ON)}
    att = {"worker_reports": [dict(rep), dict(rep)]}
    assert lw.serve_declarations_attested(att) == ({"c": sd.thinking_declaration()}, None)
    assert lw.thinking_header_attested(att) == (ON, None)
    none = {"worker_reports": [{lw.LENS_SERVE_DECLARATIONS_KEY: {}, lw.LENS_THINKING_HEADER_KEY: None}]}
    assert lw.serve_declarations_attested(none) == ({}, None)
    assert lw.thinking_header_attested(none) == (None, None)
    old = {"worker_reports": [{"cells": ["c"]}]}
    assert lw.serve_declarations_attested(old)[0] is None and "predates" in lw.serve_declarations_attested(old)[1]
    assert lw.thinking_header_attested(old)[0] is None and "predates" in lw.thinking_header_attested(old)[1]
    disagree = {"worker_reports": [dict(rep), {**rep, lw.LENS_THINKING_HEADER_KEY: list(OFF)}]}
    assert "disagree" in lw.thinking_header_attested(disagree)[1]
    bad = {"worker_reports": [{**rep, lw.LENS_SERVE_DECLARATIONS_KEY: {"c": {"serve_out_of_support": "x"}}}]}
    assert "malformed" in lw.serve_declarations_attested(bad)[1]
    assert lw.thinking_header_attested({"worker_reports": []})[0] is None
    assert lw.serve_declarations_attested("nope")[0] is None
    # the install payload carries the header ONLY when an install computed it
    from sisteer.capeval.serving.lens_worker import InstallReport, install_report_payload
    base = dict(registration_count=1, layers=(0,), plan_layer=0, cells=("c",), ledger_cells=())
    assert lw.LENS_THINKING_HEADER_KEY not in install_report_payload(InstallReport(**base))
    assert install_report_payload(InstallReport(**base, thinking_header_attested=True))[lw.LENS_THINKING_HEADER_KEY] is None
    assert install_report_payload(InstallReport(**base, thinking_header_attested=True,
                                                assistant_header_ids_thinking_on=ON))[lw.LENS_THINKING_HEADER_KEY] == list(ON)
