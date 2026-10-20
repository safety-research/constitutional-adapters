"""The vectorized lens apply (lens-vectorize, 2026-09-04): one apply per
(object, declared facts) group of decode rows instead of one per request
per layer — F3 of the hook-performance landing — plus the sync-free row
selection it rests on and the per-row REFERENCE switch the GPU gate
compares it against. CPU-only, over the worker's injectable view seam.

EXACTNESS ON THE CPU. Every tensor here is INTEGER-VALUED (small ints
stored as float32), so every dot product and sum is exact whatever the
reduction order — a grouped [R, d] @ [d] gemv and R single-row gemvs
must then agree BIT for BIT on any backend, and `torch.equal` is the
right assertion. On the GPU the two are NOT bit-identical (the 2026-09-04
gate, job 382416, measured stacked fp32 reductions reducing in an
M-dependent order), and since the 2026-09-04 decision the STACKED call is
the default served arithmetic inside a group (``lens_adapters.
_row_reduction``; ``SISTEER_LENS_STACKED_REDUCTIONS=0`` opts back to the
per-row calls, the bit-identical reference). Values cannot distinguish
the two here, so TestRowwiseReductions pins the CALL SHAPES through
``la._matmul`` in BOTH modes — one [R, d] call by default, R single-row
calls under the opt-out; the rest of these tests pin the row bookkeeping
— which rows are read, which are written, which are never touched —
where a mistake would be a wrong answer, not a rounding difference.
scripts/lens_vectorize_gate.py holds the served stacked arithmetic to its
bounded bf16-kernel-variance contract on hardware
(docs/lens_stacked_reductions_scoping.md).

The load-bearing tests: the MIXED BATCH (steered decode rows of two
cells, a passthrough request, the scale-0 twin, and a prefill in one
forward) equals the per-row reference bit for bit on these exact
integer fixtures in BOTH reduction modes, with the untouched rows
untouched; a contained AdapterError
on a group records the IDENTICAL failure for every member that the
per-row loop records for each; and `_row_selection` agrees with the
dense `_row_mask` on every (op, context) pair, returning None exactly
where the dense mask was all-ones."""

import types

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_steer as ls
from sisteer.capeval.serving import lens_worker as lw
from test_petri_lens_worker import (HEADER, HIDDEN, FakeLayer, FakeView,
                                    _forward, _obj, _ok_header, _registry,
                                    _sel)

# --------------------------------------------------------------------------
# integer-valued fixtures (exact arithmetic on the CPU)
# --------------------------------------------------------------------------


def _ints(*shape, lo=-3, hi=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(lo, hi, shape, generator=g).to(torch.float32)


def _gated_int_op(layer, seed=1, positions=la.POSITIONS_ALL, header=None,
                  skip_bos=False):
    return la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=layer,
                     vector=_ints(HIDDEN, seed=seed),
                     gate_w=_ints(HIDDEN, seed=seed + 1),
                     gate_b=torch.tensor([100.0]), positions=positions,
                     header_ids=header, skip_bos=skip_bos)


def _add_int_op(layer, seed=5, positions=la.POSITIONS_ALL, header=None,
                skip_bos=False, phase=la.PHASE_BOTH):
    return la.HookOp(op=la.OP_ADD_VECTOR, layer=layer,
                     vector=_ints(HIDDEN, seed=seed), positions=positions,
                     header_ids=header, skip_bos=skip_bos, phase=phase)


def _lowrank_int_op(layer, seed=9):
    return la.HookOp(op=la.OP_LOWRANK_RELU_GATED, layer=layer,
                     lowrank_a=_ints(2, HIDDEN, seed=seed),
                     lowrank_b=_ints(HIDDEN, 2, seed=seed + 1))


def _affine_int_op(layer, seed=13, rank=2):
    return la.HookOp(op=la.OP_AFFINE_SUBSPACE, layer=layer,
                     rotate=_ints(rank, HIDDEN, seed=seed),
                     affine_w=_ints(rank, HIDDEN, seed=seed + 1),
                     affine_b=_ints(rank, seed=seed + 2))


def _hr(nat, seed=21):
    return _ints(nat, HIDDEN, seed=seed), _ints(nat, HIDDEN, seed=seed + 1)


def _steering(registry, n_layers=4, per_row_apply=None):
    view = FakeView(layers=[FakeLayer() for _ in range(n_layers)])
    st = lw.LensWorkerSteering(view=view, registry=registry,
                               per_row_apply=per_row_apply)
    st.preflight(object(), header_verifier=_ok_header)
    st.install()
    return st, view


def _dctx(pos=7, **facts):
    return la.HookContext(la.PHASE_DECODE, (pos,), token_ids=(42,), **facts)


def _pctx(ids, **facts):
    return la.HookContext(la.PHASE_PREFILL, tuple(range(len(ids))),
                          token_ids=tuple(ids), **facts)


# --------------------------------------------------------------------------
# Part A — _row_selection agrees with the dense _row_mask everywhere
# --------------------------------------------------------------------------


def _dense(sel, T):
    """Densify a selection the way the dense builder would express it."""
    if sel is None:
        return torch.ones(T, dtype=torch.bool)
    if sel is la.NO_ROWS:
        return torch.zeros(T, dtype=torch.bool)
    return sel


class TestRowSelection:
    OPS = {
        "all": _add_int_op(0),
        "all+bos": _add_int_op(0, skip_bos=True),
        "last": _add_int_op(0, positions=la.POSITIONS_LAST),
        "response": _add_int_op(0, positions=la.POSITIONS_RESPONSE,
                                header=HEADER),
        "response+bos": _add_int_op(0, positions=la.POSITIONS_RESPONSE,
                                    header=HEADER, skip_bos=True),
    }

    @pytest.mark.parametrize("name", sorted(OPS))
    def test_decode_context_selects_every_row_without_a_mask(self, name):
        op = self.OPS[name]
        h = torch.zeros(1, HIDDEN)
        dense = la._row_mask(op, _dctx(), h)
        sel = la._row_selection(op, _dctx(), h)
        # the dense builder's decode mask is ALL-ONES for every op; the
        # selection says so as None (no multiply) — never a tensor
        assert sel is None
        assert dense is None or bool(dense.all())

    @pytest.mark.parametrize("name", sorted(OPS))
    def test_decode_rows_group_context_matches(self, name):
        op = self.OPS[name]
        h = torch.zeros(5, HIDDEN)
        assert la._row_selection(op, la.DecodeRows(), h) is None
        dense = la._row_mask(op, la.DecodeRows(), h)
        assert dense is None or dense.tolist() == [True] * 5

    @pytest.mark.parametrize("ids", [
        (1, 2, 3) + HEADER,            # an ordinary render: header at the tail
        (1,) + HEADER,                 # short prompt
        HEADER,                        # the header IS the prompt
    ])
    @pytest.mark.parametrize("name", sorted(OPS))
    def test_prefill_selection_equals_the_dense_mask(self, name, ids):
        op = self.OPS[name]
        h = torch.zeros(len(ids), HIDDEN)
        ctx = _pctx(ids)
        dense = la._row_mask(op, ctx, h)
        sel = la._row_selection(op, ctx, h)
        if dense is None:
            assert sel is None
        else:
            assert torch.equal(_dense(sel, len(ids)), dense)
            if sel is not None and sel is not la.NO_ROWS:
                assert bool(sel.any())   # never a tensor with no True row

    def test_fully_masked_prefill_is_no_rows(self):
        op = _add_int_op(0, skip_bos=True)
        h = torch.zeros(1, HIDDEN)
        ctx = la.HookContext(la.PHASE_PREFILL, (0,))
        assert la._row_mask(op, ctx, h).tolist() == [False]
        assert la._row_selection(op, ctx, h) is la.NO_ROWS

    def test_last_under_a_declared_render_refuses_identically(self):
        op = self.OPS["last"]
        h = torch.zeros(1, HIDDEN)
        msgs = []
        for ctx in (_dctx(agent_loop=True), la.DecodeRows(agent_loop=True),
                    _pctx((1, 2, 3), response_prefill=True)):
            for fn in (la._row_mask, la._row_selection):
                with pytest.raises(la.AdapterError, match="positions='last'") as e:
                    fn(op, ctx, torch.zeros(3 if ctx.phase == la.PHASE_PREFILL
                                            else 1, HIDDEN))
                msgs.append(str(e.value))
        # the same op, the same declaration -> the same message, whichever
        # builder and whichever context shape
        assert len({m for m in msgs[:4]}) == 1     # decode: ctx + group
        assert "agent_loop" in msgs[0]

    def test_response_prefill_boundary_refusal_is_shared(self):
        op = self.OPS["response"]
        ids = (1, 2, 3, 4)   # no header anywhere: undeclared -> refuses
        for fn in (la._row_mask, la._row_selection):
            with pytest.raises(la.AdapterError, match="assistant header"):
                fn(op, _pctx(ids), torch.zeros(len(ids), HIDDEN))

    def test_decode_rows_context_validation(self):
        assert la.DecodeRows().phase == la.PHASE_DECODE
        with pytest.raises(la.AdapterError, match="BOTH"):
            la.DecodeRows(response_prefill=True, agent_loop=True)
        with pytest.raises(la.AdapterError, match="bools"):
            la.DecodeRows(agent_loop=1)  # type: ignore[arg-type]
        assert la.DecodeRows(agent_loop=True) == la.DecodeRows(agent_loop=True)
        assert hash(la.DecodeRows()) == hash(la.DecodeRows())


# --------------------------------------------------------------------------
# Part B — apply_hook_ops on a DecodeRows group == the per-row applies
# --------------------------------------------------------------------------


def _per_row(fn, rows, ops, *, layer, ctx, **kw):
    return torch.cat([fn(rows[i:i + 1], ops, 1.0, layer=layer, ctx=ctx, **kw)
                      for i in range(rows.shape[0])])


class TestGroupedApplyEquality:
    @pytest.mark.parametrize("make", [
        lambda: _gated_int_op(0),
        lambda: _gated_int_op(0, positions=la.POSITIONS_RESPONSE, header=HEADER),
        lambda: _add_int_op(0),
        lambda: _add_int_op(0, positions=la.POSITIONS_LAST),
        lambda: _add_int_op(0, skip_bos=True),
        lambda: _lowrank_int_op(0),
        lambda: _affine_int_op(0),
    ])
    @pytest.mark.parametrize("R", [1, 3, 8])
    def test_stacked_decode_rows_equal_row_by_row(self, make, R):
        op = make()
        rows = _ints(R, HIDDEN, seed=R)
        group = la.apply_hook_ops(rows, (op,), 1.0, layer=0,
                                  ctx=la.DecodeRows())
        single = _per_row(la.apply_hook_ops, rows, (op,), layer=0,
                          ctx=_dctx())
        legacy = _per_row(la.apply_hook_ops, rows, (op,), layer=0,
                          ctx=_dctx(), legacy_masks=True)
        assert torch.equal(group, single)
        assert torch.equal(group, legacy)
        assert not torch.equal(group, rows)   # something was applied

    def test_phase_skipped_op_is_not_invoked_for_the_group(self):
        op = _add_int_op(0, phase=la.PHASE_PREFILL)
        rows = _ints(4, HIDDEN)
        assert la.apply_hook_ops(rows, (op,), 1.0, layer=0,
                                 ctx=la.DecodeRows()) is rows

    def test_scale_zero_group_is_the_same_tensor(self):
        rows = _ints(4, HIDDEN)
        assert la.apply_hook_ops(rows, (_gated_int_op(0),), 0.0, layer=0,
                                 ctx=la.DecodeRows()) is rows

    def test_group_context_needs_no_row_count_check(self):
        # a HookContext pins its row count; a DecodeRows carries none —
        # the group's rows are whatever the caller stacked
        op = _gated_int_op(0)
        rows = _ints(6, HIDDEN)
        assert la.apply_hook_ops(rows, (op,), 1.0, layer=0,
                                 ctx=la.DecodeRows()).shape == rows.shape
        with pytest.raises(la.AdapterError, match="rows"):
            la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=_dctx())

    def test_context_needing_op_still_refuses_without_a_context(self):
        op = _add_int_op(0, positions=la.POSITIONS_LAST)
        with pytest.raises(la.AdapterError, match="needs a HookContext"):
            la.apply_hook_ops(_ints(2, HIDDEN), (op,), 1.0, layer=0)

    def test_oproj_group_equals_row_by_row(self):
        d_in, d_out, k = 6, HIDDEN, 2
        op = la.HookOp(op=la.OP_LORA_OPROJ, layer=0,
                       lowrank_a=_ints(k, d_in, seed=31),
                       lowrank_b=_ints(d_out, k, seed=32),
                       positions=la.POSITIONS_RESPONSE, header_ids=HEADER)
        x, out = _ints(5, d_in, seed=33), _ints(5, d_out, seed=34)
        group = la.apply_oproj_ops(x, out, (op,), 1.0, layer=0,
                                   ctx=la.DecodeRows())
        single = torch.cat([
            la.apply_oproj_ops(x[i:i + 1], out[i:i + 1], (op,), 1.0, layer=0,
                               ctx=_dctx()) for i in range(5)])
        legacy = torch.cat([
            la.apply_oproj_ops(x[i:i + 1], out[i:i + 1], (op,), 1.0, layer=0,
                               ctx=_dctx(), legacy_masks=True)
            for i in range(5)])
        assert torch.equal(group, single) and torch.equal(group, legacy)
        assert not torch.equal(group, out)


def _mode(monkeypatch, stacked: bool) -> None:
    """Pin the reduction mode process-locally (the attribute the env sets at
    import): True = stacked (the default), False = the per-row opt-out."""
    monkeypatch.setattr(la, "_STACKED_REDUCTIONS", bool(stacked))


def _calls(shape_stacked, shape_row, R, stacked):
    """The expected call list for a group of R rows: one stacked call, or R
    single-row calls."""
    return [shape_stacked] if stacked else [shape_row] * R


@pytest.mark.parametrize("stacked", [True, False], ids=["stacked", "per_row"])
class TestRowwiseReductions:
    """The 2026-09-04 gate (job 382416) measured the stacked fp32 matmul of
    a vectorized group NOT bit-identical to the per-request single-row
    calls: the psr gate gemv at K=2560 (gemma-4, Qwen3.5) and K=3136
    (Nemotron) from B=7 up, the rank-4 LoRA GEMMs at every K, and every K
    once prefill row counts were stacked — always one bf16 rounding of the
    edit or of the sum (a different summation order). Inside a group every
    reduction goes through ``la._row_reduction`` under the REDUCTION MODE:
    ONE stacked call by default (the 2026-09-04 decision; a bounded
    bf16-kernel-variance contract on hardware, docs/
    lens_stacked_reductions_scoping.md), or the per-request single-row call
    per row under ``SISTEER_LENS_STACKED_REDUCTIONS=0`` — the bit-identical
    reference. Integer values cannot tell the two apart on the CPU, so these
    tests pin the CALL SHAPES ``la._matmul`` sees in BOTH modes, and that a
    single row, one request's rows, and the values are the same in either."""

    def _spy(self, monkeypatch):
        calls = []
        real = la._matmul

        def spy(x, w):
            calls.append((tuple(x.shape), tuple(w.shape)))
            return real(x, w)
        monkeypatch.setattr(la, "_matmul", spy)
        return calls

    def _spy_linear(self, monkeypatch):
        calls = []
        real = la._linear

        def spy(x, w, b=None):
            calls.append((tuple(x.shape), tuple(w.shape), b is not None))
            return real(x, w, b)
        monkeypatch.setattr(la, "_linear", spy)
        return calls

    @pytest.mark.parametrize("rank", [1, 2])
    @pytest.mark.parametrize("R", [1, 3, 8])
    def test_affine_subspace_is_issued_per_row_for_a_group(self, monkeypatch, R, rank, stacked):
        """Round-8 review finding: the OP_AFFINE_SUBSPACE branch ignored the
        group's row independence — both F.linear calls took [B, d] and the
        back-projection [B, rank]. loreft_both registers this op in BOTH
        phases, so its decode rows ARE grouped. All three reductions go
        through _row_reduction: one stacked call each by default, per row
        under the opt-out; on integers the values equal the single-row calls
        bit for bit — at rank 1 (the gate's rank1 kernel-shape class) and
        rank 2 alike."""
        _mode(monkeypatch, stacked)
        op = _affine_int_op(0, rank=rank)          # rotate / affine_w [rank, H], affine_b [rank]
        rows = _ints(R, HIDDEN, seed=R)
        mm, lin = self._spy(monkeypatch), self._spy_linear(monkeypatch)
        grouped = la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=la.DecodeRows())
        one = stacked or R == 1
        n_lin, n_mm = (2, 1) if one else (2 * R, R)
        r_call = R if one else 1
        assert lin == [((r_call, HIDDEN), (rank, HIDDEN), True), ((r_call, HIDDEN), (rank, HIDDEN), False)] * (1 if one else R)
        assert mm == [((r_call, rank), (rank, HIDDEN))] * (1 if one else R)
        mm.clear(); lin.clear()
        also = la.apply_hook_ops(rows, (op,), 1.0, layer=0, independent_rows=True)
        assert len(lin) == n_lin and len(mm) == n_mm and torch.equal(grouped, also)
        mm.clear(); lin.clear()
        # one request's rows (per-slot prefill context, or ctx None): ONE call each
        la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=_pctx(range(R)))
        la.apply_hook_ops(rows, (op,), 1.0, layer=0)
        assert lin == [((R, HIDDEN), (rank, HIDDEN), True), ((R, HIDDEN), (rank, HIDDEN), False)] * 2
        assert mm == [((R, rank), (rank, HIDDEN))] * 2
        # grouped == the single-row calls, row by row, on both the serving and the
        # legacy-mask reference entry points
        for i in range(R):
            single = la.apply_hook_ops(rows[i:i + 1], (op,), 1.0, layer=0, ctx=_dctx())
            legacy = la.apply_hook_ops(rows[i:i + 1], (op,), 1.0, layer=0, ctx=_dctx(),
                                       legacy_masks=True)
            assert torch.equal(grouped[i:i + 1], single) and torch.equal(grouped[i:i + 1], legacy)
        assert not torch.equal(grouped, rows)

    @pytest.mark.parametrize("R", [1, 3, 8])
    def test_gated_gemv_is_issued_per_row_for_a_group(self, monkeypatch, R, stacked):
        _mode(monkeypatch, stacked)
        op = _gated_int_op(0)
        rows = _ints(R, HIDDEN, seed=R)
        calls = self._spy(monkeypatch)
        grouped = la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=la.DecodeRows())
        want = _calls(((R, HIDDEN), (HIDDEN,)), ((1, HIDDEN), (HIDDEN,)), R, stacked or R == 1)
        assert calls == want                       # one stacked gemv, or R single-row gemvs
        calls.clear()
        # a row-local group (ctx None) declares its rows independent explicitly
        also = la.apply_hook_ops(rows, (op,), 1.0, layer=0, independent_rows=True)
        assert calls == want
        assert torch.equal(grouped, also)
        calls.clear()
        # ONE request's rows (a prefill HookContext, or ctx None on the per-slot
        # path) are ONE call — the per-request path's own arithmetic, untouched
        la.apply_hook_ops(rows, (op,), 1.0, layer=0)
        la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=_pctx(range(R)))
        assert calls == [((R, HIDDEN), (HIDDEN,))] * 2
        # and each per-row result is exactly the per-request call on that row
        for i in range(R):
            single = la.apply_hook_ops(rows[i:i + 1], (op,), 1.0, layer=0, ctx=_dctx())
            assert torch.equal(grouped[i:i + 1], single)

    def test_lowrank_gated_block_op_is_issued_per_row(self, monkeypatch, stacked):
        _mode(monkeypatch, stacked)
        op = _lowrank_int_op(0)
        rows = _ints(4, HIDDEN, seed=3)
        calls = self._spy(monkeypatch)
        la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=la.DecodeRows())
        # read GEMM [r, d] @ [d, k], then write GEMM [r, k] @ [k, d]: r = 4 once, or 1 x4
        r = 4 if stacked else 1
        assert calls == [((r, HIDDEN), (HIDDEN, 2)), ((r, 2), (2, HIDDEN))] * (1 if stacked else 4)
        calls.clear()
        la.apply_hook_ops(rows, (op,), 1.0, layer=0)
        assert calls == [((4, HIDDEN), (HIDDEN, 2)), ((4, 2), (2, HIDDEN))]

    def test_oproj_gemms_are_issued_per_row_for_a_group(self, monkeypatch, stacked):
        _mode(monkeypatch, stacked)
        d_in, d_out, k = 6, HIDDEN, 2
        op = la.HookOp(op=la.OP_LORA_OPROJ, layer=0,
                       lowrank_a=_ints(k, d_in, seed=31), lowrank_b=_ints(d_out, k, seed=32),
                       positions=la.POSITIONS_RESPONSE, header_ids=HEADER)
        x, out = _ints(5, d_in, seed=33), _ints(5, d_out, seed=34)
        calls = self._spy(monkeypatch)
        grouped = la.apply_oproj_ops(x, out, (op,), 1.0, layer=0, ctx=la.DecodeRows())
        r = 5 if stacked else 1
        assert calls == [((r, d_in), (d_in, k)), ((r, k), (k, d_out))] * (1 if stacked else 5)
        calls.clear()
        la.apply_oproj_ops(x, out, (op,), 1.0, layer=0, ctx=la.DecodeRows(),
                           independent_rows=True)
        assert len(calls) == (2 if stacked else 10)
        calls.clear()
        # one request's prefill rows: one read GEMM, one write GEMM
        la.apply_oproj_ops(x, out, (op,), 1.0, layer=0, ctx=_pctx((1,) + HEADER))
        assert calls == [((5, d_in), (d_in, k)), ((5, k), (k, d_out))]
        for i in range(5):
            single = la.apply_oproj_ops(x[i:i + 1], out[i:i + 1], (op,), 1.0, layer=0,
                                        ctx=_dctx())
            assert torch.equal(grouped[i:i + 1], single)

    def test_a_single_row_is_one_call_either_way(self, monkeypatch, stacked):
        """B=1 is exact by construction in BOTH modes: a one-row group is the
        single-row call itself (the gate's pure_B1 / controls_B1 stay hard and
        exact for every kind; job 382416 filed pure_B1=PASS on every stacked
        case)."""
        _mode(monkeypatch, stacked)
        op = _gated_int_op(0)
        row = _ints(1, HIDDEN)
        calls = self._spy(monkeypatch)
        la.apply_hook_ops(row, (op,), 1.0, layer=0, ctx=la.DecodeRows())
        la.apply_hook_ops(row, (op,), 1.0, layer=0, independent_rows=True)
        assert calls == [((1, HIDDEN), (HIDDEN,))] * 2

    def test_group_apply_sites_declare_independent_rows(self, monkeypatch, stacked):
        """The worker's grouped block apply of a 3-row decode group goes
        through _row_reduction at every hooked layer — ONE [3, d] call by
        default, three single-row calls under the opt-out — for a row-local
        gated object (ctx None), a context-needing one (DecodeRows), and a
        both-phase affine-subspace object (the loreft_both shape) alike."""
        _mode(monkeypatch, stacked)
        r, n = (3, 1) if stacked else (1, 3)
        cases = [
            ("gated/all", _gated_int_op(1)),
            ("gated/response", _gated_int_op(1, positions=la.POSITIONS_RESPONSE, header=HEADER)),
            ("affine/both", _affine_int_op(1)),
        ]
        for name, op in cases:
            assert op.phase == la.PHASE_BOTH
            a = _obj("cell-a", [op])
            st, view = _steering(_registry(a, header=HEADER))
            qsl, seq_lens, nat, pos = _forward([(1, 5), (1, 6), (1, 4)])
            rids = ["r0", "r1", "r2"]
            view.set_batch(rids, {rid: _sel(a) for rid in rids},
                           {rid: list(range(c + 1)) for rid, (q, c) in zip(rids, [(1, 5), (1, 6), (1, 4)])},
                           (qsl, seq_lens, nat), prompt_lens={rid: 3 for rid in rids})
            h, res = _hr(nat)
            mm, lin = self._spy(monkeypatch), self._spy_linear(monkeypatch)
            st.hook_body(1, (pos, h, res), (h, res))
            if name.startswith("gated"):
                assert mm == [((r, HIDDEN), (HIDDEN,))] * n and lin == [], name
            else:
                assert lin == [((r, HIDDEN), (2, HIDDEN), True), ((r, HIDDEN), (2, HIDDEN), False)] * n
                assert mm == [((r, 2), (2, HIDDEN))] * n, name
            assert st.selection_record.as_dict()["n_applications_grouped"] == 3
            assert st.selection_record.as_dict()["lens_reduction_mode"] == (
                la.REDUCTION_MODE_STACKED if stacked else la.REDUCTION_MODE_PER_ROW)
            assert lw.ops_vectorize_in_decode(a.ops)


# --------------------------------------------------------------------------
# Part C — grouping: which slots form a group, which stay per slot
# --------------------------------------------------------------------------


def _slots(spec, ctxs=True):
    """RequestSlots from (req_id, rows slice, obj-or-None, ctx-or-None)."""
    out = []
    for rid, rows, obj, ctx in spec:
        sel = None if obj is None else _sel(obj, scale=(0.0 if obj.effective_scale == 0.0 else None))
        out.append(lw.RequestSlot(req_id=rid, rows=rows, selector=sel,
                                  obj=obj, ctx=ctx))
    return lw.ForwardSlots(slots=tuple(out), plan=None, tracked_plan=None)


class TestBuildSlotGroups:
    def setup_method(self):
        self.a = _obj("cell-a", [_gated_int_op(1)])
        self.b = _obj("cell-b", [_add_int_op(1)])
        self.zero = _obj("none__lens0", [_gated_int_op(1)], scale=0.0)

    def test_contiguous_decode_rows_form_one_sliced_group(self):
        fwd = _slots([("r0", slice(0, 1), self.a, _dctx(5)),
                      ("r1", slice(1, 2), self.a, _dctx(9)),
                      ("r2", slice(2, 3), self.a, _dctx(7))])
        groups, per_slot = lw.build_slot_groups(fwd)
        assert per_slot == ()
        (g,) = groups
        assert g.cell_id == "cell-a" and g.rows == slice(0, 3)
        assert g.contiguous and g.ctx == la.DecodeRows()
        assert [s.req_id for s in g.slots] == ["r0", "r1", "r2"]

    def test_single_decode_slot_is_a_one_row_group(self):
        fwd = _slots([("r0", slice(0, 1), self.a, _dctx(5))])
        groups, per_slot = lw.build_slot_groups(fwd)
        assert per_slot == () and len(groups) == 1
        assert groups[0].rows == slice(0, 1)

    def test_scattered_rows_become_an_index(self):
        fwd = _slots([("a0", slice(0, 1), self.a, _dctx(5)),
                      ("u", slice(1, 2), None, _dctx(5)),       # unhooked
                      ("b0", slice(2, 3), self.b, _dctx(5)),
                      ("a1", slice(3, 4), self.a, _dctx(6)),
                      ("z", slice(4, 5), self.zero, _dctx(6)),
                      ("a2", slice(5, 6), self.a, _dctx(8))])
        groups, per_slot = lw.build_slot_groups(fwd)
        by_cell = {g.cell_id: g for g in groups}
        assert set(by_cell) == {"cell-a", "cell-b", "none__lens0"}
        assert by_cell["cell-a"].rows == (0, 3, 5)
        assert not by_cell["cell-a"].contiguous
        assert by_cell["cell-b"].rows == slice(2, 3)
        assert by_cell["none__lens0"].rows == slice(4, 5)
        assert by_cell["none__lens0"].scale == 0.0
        assert per_slot == ()
        # the unhooked request appears nowhere
        assert all("u" != s.req_id for g in groups for s in g.slots)
        idx = by_cell["cell-a"].index_on(torch.device("cpu"))
        assert idx.tolist() == [0, 3, 5] and idx.dtype == torch.long
        assert by_cell["cell-a"].index_on(torch.device("cpu")) is idx  # once

    def test_prefill_slots_stay_per_slot(self):
        fwd = _slots([("p", slice(0, 4), self.a, _pctx((1, 2, 3, 4))),
                      ("d", slice(4, 5), self.a, _dctx(4))])
        groups, per_slot = lw.build_slot_groups(fwd)
        assert [s.req_id for s in per_slot] == ["p"]
        assert [g.rows for g in groups] == [slice(4, 5)]

    def test_declared_facts_split_groups(self):
        fwd = _slots([("plain", slice(0, 1), self.a, _dctx(5)),
                      ("agent", slice(1, 2), self.a, _dctx(5, agent_loop=True)),
                      ("seed", slice(2, 3), self.a,
                       _dctx(5, response_prefill=True)),
                      ("plain2", slice(3, 4), self.a, _dctx(6))])
        groups, _ = lw.build_slot_groups(fwd)
        ctxs = sorted((g.ctx.response_prefill, g.ctx.agent_loop, g.rows)
                      for g in groups)
        assert ctxs == [(False, False, (0, 3)), (False, True, slice(1, 2)),
                        (True, False, slice(2, 3))]

    def test_multi_row_slots_never_join_a_group(self):
        """The 2026-09-04 gate finding (job 382416): on the row-local path
        (ctx None) a prefill's rows used to be stacked with the decode
        rows of the same object — and the stacked fp32 gemv over 328 /
        2519 / 13601 rows was not bit-identical to the per-slot call on
        llama and GLM (exact for every decode-only group there). A group
        is a stack of SINGLE rows: the 4-row prefill stays per slot, the
        one-row slots of the same object form the group."""
        fwd = _slots([("p", slice(0, 4), self.b, None),
                      ("d", slice(4, 5), self.b, None),
                      ("d2", slice(5, 6), self.b, None)])
        groups, per_slot = lw.build_slot_groups(fwd)
        (g,) = groups
        assert g.rows == slice(4, 6) and g.ctx is None
        assert [s.req_id for s in per_slot] == ["p"]
        # a prefill-only forward of row-local slots (the B=32 wave's first
        # step: 32 prompts, ~13.6k rows) forms NO group at all
        fwd = _slots([("p0", slice(0, 4), self.b, None),
                      ("p1", slice(4, 9), self.b, None)])
        groups, per_slot = lw.build_slot_groups(fwd)
        assert groups == () and [s.req_id for s in per_slot] == ["p0", "p1"]
        # a one-row PREFILL slot with a context (a 1-token prompt) stays per slot too
        fwd = _slots([("p1", slice(0, 1), self.a, _pctx((1,))),
                      ("d", slice(1, 2), self.a, _dctx(4))])
        groups, per_slot = lw.build_slot_groups(fwd)
        assert [s.req_id for s in per_slot] == ["p1"]
        assert [g.rows for g in groups] == [slice(1, 2)]

    def test_per_row_all_puts_every_selector_slot_per_slot(self):
        fwd = _slots([("a", slice(0, 1), self.a, _dctx(5)),
                      ("u", slice(1, 2), None, _dctx(5)),
                      ("b", slice(2, 3), self.b, _dctx(5))])
        groups, per_slot = lw.build_slot_groups(fwd, per_row_all=True)
        assert groups == () and [s.req_id for s in per_slot] == ["a", "b"]

    @pytest.mark.parametrize("names", [
        frozenset({"synthetic_vector"}),        # every _obj's kind
        frozenset({"relu_gated_vector"}),       # cell-a's op
    ])
    def test_per_row_names_force_only_the_named_objects(self, names):
        fwd = _slots([("a", slice(0, 1), self.a, _dctx(5)),
                      ("b", slice(1, 2), self.b, _dctx(5))])
        groups, per_slot = lw.build_slot_groups(fwd, per_row_names=names)
        if "synthetic_vector" in names:      # both objects carry the kind
            assert groups == () and len(per_slot) == 2
        else:                                # only cell-a carries the op
            assert [s.req_id for s in per_slot] == ["a"]
            assert [g.cell_id for g in groups] == ["cell-b"]

    def test_parse_per_row_apply(self):
        assert lw.parse_per_row_apply(None) == (False, frozenset())
        assert lw.parse_per_row_apply("") == (False, frozenset())
        assert lw.parse_per_row_apply("0") == (False, frozenset())
        assert lw.parse_per_row_apply("1") == (True, frozenset())
        assert lw.parse_per_row_apply("all") == (True, frozenset())
        assert lw.parse_per_row_apply("psr, lora_oproj") == (
            False, frozenset({"psr", "lora_oproj"}))
        assert lw.parse_per_row_apply("psr,all") == (True, frozenset())


# --------------------------------------------------------------------------
# Part D — the hook body: grouped == per-row reference, untouched rows
# --------------------------------------------------------------------------


def _mixed_batch(view, a, b, zero, *, prefill_ids):
    """Batch order (rows): a decode, unhooked decode, b decode, a decode,
    scale-0 decode, a decode, a PREFILL (len(prefill_ids) rows), b decode.
    Returns (nat, positions, slices-by-req)."""
    P = len(prefill_ids)
    shapes = [(1, 9), (1, 4), (1, 6), (1, 11), (1, 5), (1, 8), (P, 0), (1, 7)]
    qsl, seq_lens, nat, pos = _forward(shapes)
    rids = ["a0", "u", "b0", "a1", "z", "a2", "ap", "b1"]
    sel = {"a0": _sel(a), "b0": _sel(b), "a1": _sel(a), "z": _sel(zero, 0.0),
           "a2": _sel(a), "ap": _sel(a), "b1": _sel(b)}
    toks = {}
    for rid, (q, computed) in zip(rids, shapes):
        toks[rid] = list(range(100, 100 + computed + q))
    toks["ap"] = list(prefill_ids)
    prompt_lens = {rid: 3 for rid in rids}
    prompt_lens["ap"] = P
    view.set_batch(rids, sel, toks, (qsl, seq_lens, nat),
                   prompt_lens=prompt_lens)
    rows = {rid: slice(s, e) for rid, (s, e) in zip(rids, zip(qsl, qsl[1:]))}
    return nat, pos, rows


class TestHookBodyGroupedVsReference:
    LAYERS = (1, 2)

    def _objs(self):
        a = _obj("cell-a", [_gated_int_op(l, seed=l, positions=la.POSITIONS_RESPONSE,
                                          header=HEADER) for l in self.LAYERS])
        b = _obj("cell-b", [_add_int_op(l, seed=40 + l) for l in self.LAYERS])
        zero = _obj("none__lens0", [op for op in a.ops], scale=0.0)
        return a, b, zero

    def _run(self, per_row_apply, with_residual=True):
        a, b, zero = self._objs()
        reg = _registry(a, b, zero, header=HEADER)
        st, view = _steering(reg, per_row_apply=per_row_apply)
        prefill = (1, 2) + HEADER
        nat, pos, rows = _mixed_batch(view, a, b, zero, prefill_ids=prefill)
        h, r = _hr(nat)
        h0, r0 = h.clone(), r.clone()
        outs = {}
        for layer in self.LAYERS:
            if with_residual:
                out = st.hook_body(layer, (pos, h, r), (h, r))
                assert out is not None
                new_h, out_r = out
                assert out_r is r
            else:
                new_h = st.hook_body(layer, (pos, h), h)
            assert torch.equal(h, h0) and torch.equal(r, r0)  # inputs untouched
            outs[layer] = new_h
        return st, outs, h0, r0, rows

    @pytest.mark.parametrize("with_residual", [True, False])
    def test_grouped_equals_per_row_reference_on_a_mixed_batch(self, with_residual):
        st_v, vec, h0, r0, rows = self._run(None, with_residual)
        st_r, ref, _, _, _ = self._run("1", with_residual)
        assert st_v.apply_path == lw.APPLY_PATH_VECTORIZED_STACKED and st_r.apply_path == "per_row"
        for layer in self.LAYERS:
            assert torch.equal(vec[layer], ref[layer])
            # untouched rows: the unhooked request and the scale-0 twin
            for rid in ("u", "z"):
                assert torch.equal(vec[layer][rows[rid]], h0[rows[rid]])
            # steered rows moved, and each cell's rows moved by ITS object
            true = h0 + (r0 if with_residual else 0)
            for rid, obj in (("a0", st_v._registry.select("cell-a", "hash-cell-a")),
                             ("b0", st_v._registry.select("cell-b", "hash-cell-b"))):
                expect = la.apply_hook_ops(
                    true[rows[rid]], obj.ops, 1.0, layer=layer, ctx=_dctx())
                assert torch.equal(vec[layer][rows[rid]],
                                   h0[rows[rid]] + (expect - true[rows[rid]]))
            assert not torch.equal(vec[layer][rows["ap"]], h0[rows["ap"]])

    def test_selection_record_is_identical_between_paths(self):
        st_v, _, _, _, _ = self._run(None)
        st_r, _, _, _, _ = self._run("1")
        rv, rr = st_v.selection_record.as_dict(), st_r.selection_record.as_dict()
        assert rv["n_observations"] == rr["n_observations"] > 0
        assert rv["requests"] == rr["requests"]
        assert rv["pairs"] == rr["pairs"]
        assert rv["all_observations_matched"]

    def test_groups_were_actually_formed(self, monkeypatch):
        """The vectorized path calls registry.apply ONCE per group per
        layer (3 groups: cell-a decodes, cell-b decodes, the scale-0 twin)
        plus once for the prefill slot; the reference once per slot."""
        counts = {}

        def _count(path):
            a, b, zero = self._objs()
            reg = _registry(a, b, zero, header=HEADER)
            n = {"apply": 0}
            orig = reg.apply

            def wrapped(*args, **kw):
                n["apply"] += 1
                return orig(*args, **kw)
            monkeypatch.setattr(reg, "apply", wrapped)
            st, view = _steering(reg, per_row_apply=path)
            nat, pos, rows = _mixed_batch(view, a, b, zero,
                                          prefill_ids=(1, 2) + HEADER)
            h, r = _hr(nat)
            st.hook_body(1, (pos, h, r), (h, r))
            return n["apply"]
        assert _count("1") == 7          # a0 b0 a1 z a2 ap b1
        assert _count(None) == 4         # {a decodes} {b decodes} {z} + ap

    def test_scattered_group_rows_are_written_by_index(self):
        """cell-a's decode rows sit at 0, 3, 5 (scattered): the write goes
        through index_select / index_copy_ and lands on exactly those."""
        st, vec, h0, r0, rows = self._run(None)
        moved = [(vec[1][i] != h0[i]).any().item() for i in range(h0.shape[0])]
        expect = [True] * h0.shape[0]
        for rid in ("u", "z"):
            for i in range(rows[rid].start, rows[rid].stop):
                expect[i] = False
        # the prefill's two pre-header rows sit OUTSIDE cell-a's response
        # mask (positions 0, 1 of (1, 2) + HEADER): untouched by design
        expect[rows["ap"].start] = expect[rows["ap"].start + 1] = False
        assert moved == expect

    def test_single_request_forward_is_the_degenerate_group(self):
        a, b, zero = self._objs()
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        qsl, seq_lens, nat, pos = _forward([(1, 9)])
        view.set_batch(["a0"], {"a0": _sel(a)}, {"a0": list(range(10))},
                       (qsl, seq_lens, nat), prompt_lens={"a0": 3})
        h, r = _hr(nat)
        out = st.hook_body(1, (pos, h, r), (h, r))
        true = h + r
        expect = la.apply_hook_ops(true, a.ops, 1.0, layer=1, ctx=_dctx(9))
        assert torch.equal(out[0], h + (expect - true))

    def test_env_switch_is_read_at_construction(self, monkeypatch):
        a, b, zero = self._objs()
        monkeypatch.setenv(lw.PER_ROW_APPLY_ENV, "1")
        st, _ = _steering(_registry(a, b, zero, header=HEADER))
        assert st.apply_path == "per_row"
        monkeypatch.setenv(lw.PER_ROW_APPLY_ENV, "psr,lora_oproj")
        st, _ = _steering(_registry(a, b, zero, header=HEADER))
        assert st.apply_path == "vectorized+stacked(per_row: lora_oproj,psr)"
        monkeypatch.delenv(lw.PER_ROW_APPLY_ENV)
        st, _ = _steering(_registry(a, b, zero, header=HEADER))
        assert st.apply_path == "vectorized+stacked" == lw.APPLY_PATH_VECTORIZED_STACKED

    def test_per_kind_fallback_keeps_that_kind_per_row(self, monkeypatch):
        """The gate's per-kind fallback: cell-a (relu_gated_vector) per
        row, cell-b still grouped — and the outputs still equal the
        all-per-row reference."""
        st_r, ref, _, _, _ = self._run("1")
        a, b, zero = self._objs()
        reg = _registry(a, b, zero, header=HEADER)
        n = {"apply": 0}
        orig = reg.apply

        def wrapped(*args, **kw):
            n["apply"] += 1
            return orig(*args, **kw)
        monkeypatch.setattr(reg, "apply", wrapped)
        st, view = _steering(reg, per_row_apply="relu_gated_vector")
        nat, pos, rows = _mixed_batch(view, a, b, zero,
                                      prefill_ids=(1, 2) + HEADER)
        h, r = _hr(nat)
        out, _ = st.hook_body(1, (pos, h, r), (h, r))
        assert torch.equal(out, ref[1])
        # a0 a1 a2 ap z (relu-gated objects, per slot) + one cell-b group
        assert n["apply"] == 6


class TestGroupFailureContainment:
    def test_group_refusal_records_every_member_with_the_per_row_message(self):
        """A LAST-position op under a declared agent-loop render refuses
        per request; for a decode group the one refusal is recorded for
        every member — the same message each would have produced alone."""
        arm = _obj("arm-last", [la.HookOp(op=la.OP_ADD_VECTOR, layer=1,
                                          vector=_ints(HIDDEN),
                                          positions=la.POSITIONS_LAST)])
        msgs = {}
        for path in (None, "1"):
            st, view = _steering(_registry(arm), per_row_apply=path)
            qsl, seq_lens, nat, pos = _forward([(1, 5), (1, 6), (1, 4)])
            rids = ["r0", "r1", "r2"]
            view.set_batch(rids, {r: dict(_sel(arm), agent_loop=1) for r in rids},
                           {r: list(range(c + 1)) for r, (q, c) in
                            zip(rids, [(1, 5), (1, 6), (1, 4)])},
                           (qsl, seq_lens, nat),
                           prompt_lens={r: 3 for r in rids})
            h, r = _hr(nat)
            h0 = h.clone()
            out = st.hook_body(1, (pos, h, r), (h, r))
            assert out is None or torch.equal(out[0], h0)   # nothing steered
            st._require_alive()                              # not condemned
            msgs[path] = {rid: st.take_request_failure(rid) for rid in rids}
            assert all(m is not None for m in msgs[path].values())
        for rid in ("r0", "r1", "r2"):
            assert msgs[None][rid] == msgs["1"][rid]
            assert "positions='last'" in msgs[None][rid]["error"]

    def test_failed_member_is_skipped_at_later_layers_and_forwards(self):
        """A group member that failed at layer 1 must not be re-attempted
        at layer 2 of the same forward (the group's cached rows no longer
        describe it) nor at a later forward — exactly the per-slot skip."""
        good = _obj("cell-a", [_gated_int_op(1), _gated_int_op(2)])
        st, view = _steering(_registry(good))
        qsl, seq_lens, nat, pos = _forward([(1, 5), (1, 6)])
        view.set_batch(["r0", "r1"], {"r0": _sel(good), "r1": _sel(good)},
                       {"r0": list(range(6)), "r1": list(range(7))},
                       (qsl, seq_lens, nat), prompt_lens={"r0": 3, "r1": 3})
        # plant a failure for r0 the way an earlier layer would have
        st._record_request_failure(req_id="r0", layer=0, site=la.HOOK_SITE_BLOCK,
                                   cell_id="cell-a", error=la.AdapterError("x"))
        h, r = _hr(nat)
        h0 = h.clone()
        out, _ = st.hook_body(1, (pos, h, r), (h, r))
        assert torch.equal(out[0], h0[0])          # r0 untouched
        assert not torch.equal(out[1], h0[1])      # r1 served (per slot)
        out2, _ = st.hook_body(2, (pos, h, r), (h, r))
        assert torch.equal(out2[0], h0[0]) and not torch.equal(out2[1], h0[1])


# --------------------------------------------------------------------------
# Part E — the o_proj site: grouped == per-row, untouched rows untouched
# --------------------------------------------------------------------------


class _OprojModule:
    def __init__(self):
        self.hooks = []

    def register_forward_hook(self, fn):
        self.hooks.append(fn)
        return object()


class _OprojLayer(FakeLayer):
    class _Attn:
        def __init__(self):
            self.o_proj = _OprojModule()

    def __init__(self):
        super().__init__()
        self.pre_hooks = []
        self.self_attn = self._Attn()

    def register_forward_pre_hook(self, fn, *, with_kwargs=False):
        self.pre_hooks.append(fn)
        return object()


D_IN, D_OUT = 6, HIDDEN


def _lora_int_obj(cell_id, layers=(1,), scale=1.0, seed=50):
    ops = tuple(la.HookOp(op=la.OP_LORA_OPROJ, layer=l,
                          lowrank_a=_ints(2, D_IN, seed=seed + l),
                          lowrank_b=_ints(D_OUT, 2, seed=seed + 10 + l),
                          positions=la.POSITIONS_RESPONSE, header_ids=HEADER)
                for l in layers)
    return ls.LoadedObject(cell_id=cell_id, kind="lora_all",
                           expect_hash=f"hash-{cell_id}", ops=ops,
                           effective_scale=float(scale))


def _oproj_steering(registry, per_row_apply=None, n_layers=4):
    view = FakeView(layers=[_OprojLayer() for _ in range(n_layers)])
    view.tensor_parallel_size = lambda: 1
    st = lw.LensWorkerSteering(view=view, registry=registry,
                               per_row_apply=per_row_apply)
    st.preflight(object(), header_verifier=_ok_header)
    st.install()
    return st, view


class TestOprojGrouped:
    def _run(self, path):
        a = _lora_int_obj("lora-a")
        b = _lora_int_obj("lora-b", seed=70)
        zero = _lora_int_obj("none__lens0", scale=0.0)
        st, view = _oproj_steering(_registry(a, b, zero, header=HEADER), path)
        shapes = [(1, 9), (1, 4), (1, 6), (1, 11), (1, 5), (1, 8)]
        qsl, seq_lens, nat, pos = _forward(shapes)
        rids = ["a0", "u", "b0", "a1", "z", "a2"]
        sel = {"a0": _sel(a), "b0": _sel(b), "a1": _sel(a),
               "z": _sel(zero, 0.0), "a2": _sel(a)}
        view.set_batch(rids, sel,
                       {r: list(range(c + 1)) for r, (q, c) in zip(rids, shapes)},
                       (qsl, seq_lens, nat), prompt_lens={r: 3 for r in rids})
        for layer in view.layers:
            for pre in layer.pre_hooks:
                pre(None, (pos,), {})
        x, out = _ints(nat, D_IN, seed=80), _ints(nat, D_OUT, seed=81)
        x0, out0 = x.clone(), out.clone()
        new = st.oproj_hook_body(1, (x,), out)
        assert torch.equal(x, x0) and torch.equal(out, out0)
        return st, new, out0

    def test_grouped_equals_per_row_and_leaves_other_rows(self):
        st_v, vec, out0 = self._run(None)
        st_r, ref, _ = self._run("1")
        assert torch.equal(vec, ref)
        assert torch.equal(vec[1], out0[1]) and torch.equal(vec[4], out0[4])
        for i in (0, 2, 3, 5):
            assert not torch.equal(vec[i], out0[i])
        assert (st_v.selection_record.as_dict()["requests"]
                == st_r.selection_record.as_dict()["requests"])


# --------------------------------------------------------------------------
# Part F — the precomputed (cell, site, layer) op index (F4)
# --------------------------------------------------------------------------


class TestOpsIndex:
    def test_ops_for_equals_the_scan(self):
        ops = (_gated_int_op(1), _add_int_op(1), _gated_int_op(3),
               la.HookOp(op=la.OP_LORA_OPROJ, layer=1,
                         lowrank_a=_ints(2, D_IN), lowrank_b=_ints(D_OUT, 2),
                         positions=la.POSITIONS_RESPONSE, header_ids=HEADER))
        obj = ls.LoadedObject(cell_id="c", kind="stacked", expect_hash="h",
                              ops=ops, effective_scale=1.0)
        reg = _registry(obj, header=HEADER)
        for site in (la.HOOK_SITE_BLOCK, la.HOOK_SITE_OPROJ):
            for layer in range(5):
                want = tuple(op for op in la.ops_at_site(ops, site)
                             if op.layer == layer)
                assert reg.ops_for("c", site, layer) == want
        assert reg.ops_for("c", la.HOOK_SITE_BLOCK, 1) == (ops[0], ops[1])
        assert reg.ops_for("nope", la.HOOK_SITE_BLOCK, 1) == ()
        with pytest.raises(la.AdapterError, match="unknown hook site"):
            reg.ops_for("c", "attn", 1)

    def test_materialize_rebuilds_the_index(self):
        obj = _obj("c", [_gated_int_op(1)])
        reg = _registry(obj).materialize(device=torch.device("cpu"),
                                        dtype=torch.float32)
        (op,) = reg.ops_for("c", la.HOOK_SITE_BLOCK, 1)
        assert op is reg.select("c", "hash-c").ops[0]

    def test_apply_matches_the_unindexed_arithmetic(self):
        obj = _obj("c", [_gated_int_op(1), _gated_int_op(2, seed=3)])
        reg = _registry(obj)
        rows = _ints(4, HIDDEN)
        for layer in (1, 2, 3):
            got = reg.apply("c", "hash-c", rows, layer=layer, ctx=la.DecodeRows())
            want = la.apply_hook_ops(rows, la.ops_at_site(obj.ops, la.HOOK_SITE_BLOCK),
                                     1.0, layer=layer, ctx=la.DecodeRows())
            assert torch.equal(got, want)
            if layer == 3:
                assert got is rows   # no op here: the same tensor


# --------------------------------------------------------------------------
# Part G — the co-scheduling record (observation only; the multigroup gate's
# proof that distinct groups shared a forward)
# --------------------------------------------------------------------------


class TestCoScheduleRecord:
    def _objs(self):
        a = _obj("cell-a", [_gated_int_op(1)])
        b = _obj("cell-b", [_add_int_op(1)])
        zero = _obj("none__lens0", [_gated_int_op(1)], scale=0.0)
        return a, b, zero

    def test_two_cells_same_render_in_one_forward(self):
        a, b, zero = self._objs()
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        qsl, seq_lens, nat, pos = _forward([(1, 9), (1, 4), (1, 6), (1, 5)])
        view.set_batch(["a0", "u", "b0", "z"],
                       {"a0": _sel(a), "b0": _sel(b), "z": _sel(zero, 0.0)},
                       {r: list(range(c + 1)) for r, (q, c) in
                        zip(["a0", "u", "b0", "z"], [(1, 9), (1, 4), (1, 6), (1, 5)])},
                       (qsl, seq_lens, nat), prompt_lens={r: 3 for r in ["a0", "u", "b0", "z"]})
        h, r = _hr(nat)
        st.hook_body(1, (pos, h, r), (h, r))
        rec = st.coschedule_record.as_dict()
        assert rec["n_forwards"] == 1 and rec["n_forwards_with_groups"] == 1
        assert rec["max_distinct_nonzero_groups_in_one_forward"] == 2   # the twin is not nonzero
        assert rec["max_distinct_nonzero_cells_in_one_forward"] == 2
        assert rec["n_forwards_with_ge2_groups"] == 1
        assert rec["n_forwards_with_same_render_pair"] == 1             # both plain decode
        assert rec["n_record_errors"] == 0

    def test_plain_and_seeded_rows_of_one_cell_are_two_groups_not_a_pair(self):
        # a RESPONSE-masked (plan-needing) object: the declared render fact
        # reaches the HookContext and splits the DecodeRows grouping key
        a = _obj("cell-a", [_gated_int_op(1, positions=la.POSITIONS_RESPONSE, header=HEADER)])
        _, b, zero = self._objs()
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        qsl, seq_lens, nat, pos = _forward([(1, 9), (1, 6)])
        view.set_batch(["plain", "seeded"],
                       {"plain": _sel(a), "seeded": dict(_sel(a), response_prefill=1)},
                       {"plain": list(range(10)), "seeded": list(range(7))},
                       (qsl, seq_lens, nat), prompt_lens={"plain": 3, "seeded": 3})
        h, r = _hr(nat)
        st.hook_body(1, (pos, h, r), (h, r))
        rec = st.coschedule_record.as_dict()
        assert rec["max_distinct_nonzero_groups_in_one_forward"] == 2   # distinct DecodeRows keys
        assert rec["max_distinct_nonzero_cells_in_one_forward"] == 1
        assert rec["n_forwards_with_same_render_pair"] == 0             # one cell, two renders

    def test_single_cell_and_reset(self):
        a, b, zero = self._objs()
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        qsl, seq_lens, nat, pos = _forward([(1, 9), (1, 5)])
        view.set_batch(["a0", "a1"], {"a0": _sel(a), "a1": _sel(a)},
                       {"a0": list(range(10)), "a1": list(range(6))}, (qsl, seq_lens, nat),
                       prompt_lens={"a0": 3, "a1": 3})
        h, r = _hr(nat)
        st.hook_body(1, (pos, h, r), (h, r))
        rec = st.coschedule_record.as_dict()
        assert rec["max_distinct_nonzero_groups_in_one_forward"] == 1
        assert rec["n_forwards_with_ge2_groups"] == 0 and rec["n_forwards_with_same_render_pair"] == 0
        st.coschedule_record.reset()
        assert st.coschedule_record.as_dict()["n_forwards"] == 0

    def test_observe_never_raises(self):
        rec = lw.CoScheduleRecord()
        rec.observe([object()])        # not a group at all: counted as an error, nothing else moves
        assert rec.n_record_errors == 1 and rec.first_record_error
        assert rec.n_forwards == 0 and rec.n_forwards_with_groups == 0
        rec.observe(())
        assert rec.n_forwards == 1 and rec.n_forwards_with_groups == 0

    def test_per_row_reference_path_records_no_groups(self):
        """Under SISTEER_LENS_PER_ROW_APPLY=1 every slot is per slot — the
        record sees forwards but no vectorized groups (the reference tree,
        older still, has no record at all)."""
        a, b, zero = self._objs()
        st, view = _steering(_registry(a, b, zero, header=HEADER), per_row_apply="1")
        qsl, seq_lens, nat, pos = _forward([(1, 9), (1, 6)])
        view.set_batch(["a0", "b0"], {"a0": _sel(a), "b0": _sel(b)},
                       {"a0": list(range(10)), "b0": list(range(7))}, (qsl, seq_lens, nat),
                       prompt_lens={"a0": 3, "b0": 3})
        h, r = _hr(nat)
        st.hook_body(1, (pos, h, r), (h, r))
        rec = st.coschedule_record.as_dict()
        assert rec["n_forwards"] == 1 and rec["n_forwards_with_groups"] == 0
        assert rec["max_distinct_nonzero_groups_in_one_forward"] == 0


# --------------------------------------------------------------------------
# Part H — the gate-only in-worker shadow compare (SISTEER_LENS_SHADOW_COMPARE)
# --------------------------------------------------------------------------


class Vllm0251BatchInvariant:
    """A module shaped EXACTLY like the state surface of vllm 0.25.1's
    ``vllm/model_executor/layers/batch_invariant.py`` — read off the installed
    ``vllm-0.25.1+cu129`` wheel the serving environment pins (2026-09-04): the
    private ``_batch_invariant_MODE`` / ``_batch_invariant_LIB`` flags, and the
    three public functions ``enable_batch_invariant_mode`` / ``override_envs_
    for_invariance`` / ``init_batch_invariance`` (the last is what
    ``gpu_worker.init_worker_distributed_environment`` calls inside
    ``init_device``). There is NO ``vllm_is_batch_invariant`` — the surface an
    earlier version of the worker test invented and the pinned vLLM never had."""

    def __init__(self, envs):
        self._batch_invariant_MODE = False
        self._batch_invariant_LIB = None
        self._envs = envs

    def enable_batch_invariant_mode(self):
        if self._batch_invariant_MODE:
            return
        self._batch_invariant_MODE = True
        self._batch_invariant_LIB = object()

    def override_envs_for_invariance(self):
        pass

    def init_batch_invariance(self):
        if self._envs.VLLM_BATCH_INVARIANT:
            self.override_envs_for_invariance()
            self.enable_batch_invariant_mode()


class TestRoutingClass:
    """``lens_worker.ops_vectorize_in_decode`` / ``phase_vectorizes_in_decode``
    — the predicate the GPU gate classifies its inventory with — IS the served
    behaviour: an object whose every op is phase=prefill is applied only on the
    per-slot path (its decode groups exist — build_slot_groups routes by SLOT
    phase — but are unmodified on both sides of the shadow: never a counted
    comparison), while a both/decode-phase op modifies its decode groups. The
    SelectionRecord's per-path counters say which path applied what."""

    #: the prefill slot's prompt: two tokens then the assistant header (so a
    #: positions="response" op has a located boundary)
    PREFILL_IDS = (1, 2) + HEADER
    P = len(PREFILL_IDS)

    def _drive(self, monkeypatch, obj, *, shadow=True, per_row_apply=None):
        if shadow:
            monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        else:
            monkeypatch.delenv(lw.SHADOW_COMPARE_ENV, raising=False)
        st, view = _steering(_registry(obj, header=HEADER), per_row_apply=per_row_apply)
        # rows: a decode slot, a PREFILL slot (P rows), a decode slot — all of obj
        shapes = [(1, 9), (self.P, 0), (1, 6)]
        qsl, seq_lens, nat, pos = _forward(shapes)
        rids = ["d0", "p", "d1"]
        toks = {rid: list(range(100, 100 + c + q)) for rid, (q, c) in zip(rids, shapes)}
        toks["p"] = list(self.PREFILL_IDS)
        view.set_batch(rids, {rid: _sel(obj) for rid in rids}, toks, (qsl, seq_lens, nat),
                       prompt_lens={"d0": 3, "p": self.P, "d1": 3})
        h, r = _hr(nat)
        st.hook_body(1, (pos, h, r), (h, r))
        return st

    # (phase, positions) -> vectorizes?, grouped slots, per-slot slots, modifying per path
    CASES = [
        # every op prefill-only: the decode group exists (slot-phase routing) but is
        # phase-skipped on both sides; the prefill slot carries the whole application
        (la.PHASE_PREFILL, la.POSITIONS_ALL, False, 2, 1, 0, 1),
        # a ROW-LOCAL op (both phases, all positions, no BOS skip) needs no context —
        # but the PREFILL slot is multi-row and stays per slot (job 382416: stacking
        # it broke bit-identity); the two decode rows form the group
        (la.PHASE_BOTH, la.POSITIONS_ALL, True, 2, 1, 2, 1),
        # a context-needing both-phase op: prefill per slot (its own mask), decodes grouped
        (la.PHASE_BOTH, la.POSITIONS_RESPONSE, True, 2, 1, 2, 1),
        # decode-only: the prefill slot is applied-but-skipped, the decode group modified
        (la.PHASE_DECODE, la.POSITIONS_ALL, True, 2, 1, 2, 0),
    ]

    @pytest.mark.parametrize("phase,positions,vec,n_grouped,n_per_slot,mod_grouped,mod_per_slot",
                             CASES, ids=[f"{c[0]}-{c[1]}" for c in CASES])
    def test_predicate_matches_the_served_grouping(self, monkeypatch, phase, positions, vec,
                                                   n_grouped, n_per_slot, mod_grouped, mod_per_slot):
        header = HEADER if positions == la.POSITIONS_RESPONSE else None
        obj = _obj("cell-x", [_add_int_op(1, phase=phase, positions=positions, header=header)])
        vectorizes = lw.ops_vectorize_in_decode(obj.ops)
        assert vectorizes is vec
        assert vectorizes == lw.phase_vectorizes_in_decode(phase) == (phase != la.PHASE_PREFILL)
        st = self._drive(monkeypatch, obj)
        rec = st.shadow_record.as_dict()
        sel = st.selection_record.as_dict()
        # exactly ONE vectorized group formed, whatever the op phase (slot-phase routing) ...
        assert rec["n_groups_compared"] + rec["n_unmodified_groups"] == 1
        assert sel["n_applications_grouped"] == n_grouped
        assert sel["n_applications_per_slot"] == n_per_slot
        assert sel["n_applications_unattributed"] == 0 and sel["n_modifying_unattributed"] == 0
        assert sel["n_modifying_grouped"] == mod_grouped
        assert sel["n_modifying_per_slot"] == mod_per_slot
        key, = list(rec["by_key"])
        if vectorizes:
            # ... and was MODIFIED -> a counted shadow comparison, grouped-path modification
            assert rec["n_groups_compared"] == 1 and rec["n_equal"] == 1
            assert rec["by_key"][key]["n_compared"] == 1
            assert key.endswith(f":{phase}:B2")          # the two decode rows, never the prefill
        else:
            # ... and was phase-skipped on BOTH sides: unmodified, never compared
            assert rec["n_groups_compared"] == 0 and rec["n_unmodified_groups"] == 1
            assert key.endswith(":prefill:B2") and rec["by_key"][key]["n_unmodified"] == 1
            assert rec["by_key"][key]["n_compared"] == 0
        # the predicate is exactly "a counted comparison can exist for this object"
        assert vectorizes == (rec["n_groups_compared"] > 0) == (sel["n_modifying_grouped"] > 0)

    def test_mixed_phase_object_vectorizes(self):
        ops = (_add_int_op(1, phase=la.PHASE_PREFILL), _add_int_op(2, phase=la.PHASE_BOTH))
        assert lw.ops_vectorize_in_decode(ops) is True
        assert lw.ops_vectorize_in_decode((_add_int_op(1, phase=la.PHASE_PREFILL),)) is False
        assert lw.ops_vectorize_in_decode(()) is False
        with pytest.raises(lw.LensWorkerError, match="unknown op phase"):
            lw.phase_vectorizes_in_decode("sometimes")

    def test_reference_path_counts_everything_per_slot(self, monkeypatch):
        obj = _obj("cell-x", [_add_int_op(1, positions=la.POSITIONS_RESPONSE, header=HEADER)])
        st_ref = self._drive(monkeypatch, obj, shadow=False, per_row_apply="1")
        st_vec = self._drive(monkeypatch, obj, shadow=False)
        ref, vec = st_ref.selection_record.as_dict(), st_vec.selection_record.as_dict()
        assert ref["n_applications_per_slot"] == 3 and ref["n_applications_grouped"] == 0
        assert ref["n_modifying_per_slot"] == 3
        assert vec["n_applications_per_slot"] == 1 and vec["n_applications_grouped"] == 2
        assert vec["n_modifying_grouped"] == 2 and vec["n_modifying_per_slot"] == 1
        # the per-request entries read the same on either path (record-level counters only)
        assert ref["requests"] == vec["requests"] and ref["n_observations"] == vec["n_observations"]
        # a caller naming no path is counted as unattributed, never as a path
        rec = lw.SelectionRecord()
        rec.observe(req_id="r", layer=1, requested="c", requested_hash="h", requested_scale=1.0,
                    applied=obj, modified=True)
        d = rec.as_dict()
        assert d["n_applications_unattributed"] == 1 and d["n_modifying_unattributed"] == 1
        assert d["n_applications_per_slot"] == d["n_applications_grouped"] == 0
        rec.observe(req_id="r", layer=1, requested="c", requested_hash="h", requested_scale=1.0,
                    applied=obj, modified=False, path="sideways")
        assert rec.as_dict()["n_applications_unattributed"] == 2


class TestShadowCompare:
    def _objs(self):
        a = _obj("cell-a", [_gated_int_op(1, positions=la.POSITIONS_RESPONSE, header=HEADER),
                            _gated_int_op(2, seed=3, positions=la.POSITIONS_RESPONSE, header=HEADER)])
        b = _obj("cell-b", [_add_int_op(1), _add_int_op(2, seed=41)])
        zero = _obj("none__lens0", [op for op in a.ops], scale=0.0)
        return a, b, zero

    def _drive(self, st, view, a, b, zero, layers=(1, 2)):
        nat, pos, rows = _mixed_batch(view, a, b, zero, prefill_ids=(1, 2) + HEADER)
        h, r = _hr(nat)
        outs = [st.hook_body(layer, (pos, h, r), (h, r))[0] for layer in layers]
        return outs, h.clone(), rows

    def test_off_by_default_no_work_no_counters(self, monkeypatch):
        monkeypatch.delenv(lw.SHADOW_COMPARE_ENV, raising=False)
        a, b, zero = self._objs()
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        assert st.apply_path == lw.APPLY_PATH_VECTORIZED_STACKED and st._shadow_compare is False
        spy = {"n": 0}
        orig = st._slot_apply

        def counting(*args, **kw):
            spy["n"] += 1
            return orig(*args, **kw)
        monkeypatch.setattr(st, "_slot_apply", counting)
        self._drive(st, view, a, b, zero)
        # only the prefill slot (per slot on both paths) ever called it: no
        # shadow applies, no clones, nothing recorded
        assert spy["n"] == 2                                  # the prefill slot, at 2 layers
        rec = st.shadow_record.as_dict()
        assert rec["enabled"] is False and rec["n_groups_compared"] == 0
        assert rec["n_layer_calls"] == 0 and rec["by_key"] == {}

    def test_on_equal_counts_and_serves_per_row(self, monkeypatch):
        a, b, zero = self._objs()
        st_ref, view_ref = _steering(_registry(a, b, zero, header=HEADER), per_row_apply="1")
        ref_outs, _, _ = self._drive(st_ref, view_ref, a, b, zero)
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        assert st.apply_path == lw.APPLY_PATH_SHADOW and st._shadow_compare is True
        outs, h0, rows = self._drive(st, view, a, b, zero)
        for layer_i in range(2):
            assert torch.equal(outs[layer_i], ref_outs[layer_i])   # the per-row result is served
        rec = st.shadow_record.as_dict()
        # groups per layer: cell-a decode (3 rows), cell-b decode (2 rows),
        # the scale-0 twin (unmodified on both paths) -> 2 compared + 1 unmodified per layer
        assert rec["n_groups_compared"] == 4 and rec["n_equal"] == 4 and rec["n_unequal"] == 0
        assert rec["n_unmodified_groups"] == 2 and rec["n_shadow_errors"] == 0
        assert rec["first_unequal"] is None and rec["max_abs_diff"] == 0.0
        # the tolerance denominator counts EVERY compared element (equal groups too):
        # cell-a 3 rows + cell-b 2 rows, at 2 layers
        assert rec["n_elems_compared"] == 10 * HIDDEN and rec["n_unequal_elems"] == 0
        assert rec["max_ulps_edit"] == 0.0 and rec["n_flag_disagreements"] == 0
        keys = set(rec["by_key"])
        assert any(k.startswith("synthetic_vector:relu_gated_vector:gemv:response:skipbos0:both:B3") for k in keys)
        assert any(k.startswith("synthetic_vector:add_vector:vec:all:skipbos0:both:B2") for k in keys)
        assert rec["n_layer_calls"] == 2 and rec["n_layer_calls_with_ge2_cells_compared"] == 2
        assert rec["max_distinct_cells_compared_in_one_layer_call"] == 2
        assert sorted(rec["cells_compared"]) == ["cell-a", "cell-b"]
        # selection record: identical observations to the reference
        assert st.selection_record.as_dict()["requests"] == st_ref.selection_record.as_dict()["requests"]

    def test_injected_inequality_is_counted_and_per_row_is_served(self, monkeypatch):
        a, b, zero = self._objs()
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        st_ref, view_ref = _steering(_registry(a, b, zero, header=HEADER), per_row_apply="1")
        # perturb the GROUPED apply only (registry.apply without legacy masks)
        orig = st._registry.apply

        def perturbed(*args, **kw):
            out = orig(*args, **kw)
            if not kw.get("legacy_masks") and isinstance(kw.get("ctx"), la.DecodeRows) \
                    and args[0] == "cell-a":
                out = out + 0.5
            return out
        monkeypatch.setattr(st._registry, "apply", perturbed)
        outs, _, _ = self._drive(st, view, a, b, zero)
        ref_outs, _, _ = self._drive(st_ref, view_ref, a, b, zero)
        for i in range(2):
            assert torch.equal(outs[i], ref_outs[i])            # the per-row result is served anyway
        rec = st.shadow_record.as_dict()
        assert rec["n_unequal"] == 2 and rec["n_equal"] == 2    # cell-a unequal at both layers
        fu = rec["first_unequal"]
        assert fu["cell"] == "cell-a" and fu["layer"] == 1 and fu["rows"] == 3
        assert fu["max_abs_diff"] == 0.5 and rec["max_abs_diff"] == 0.5
        assert fu["key"].startswith("synthetic_vector:relu_gated_vector:gemv:response:skipbos0:both:B3")
        k = fu["key"]
        assert rec["by_key"][k]["n_unequal"] == 2 and rec["by_key"][k]["max_abs_diff"] == 0.5
        # every element of both cell-a groups moved; the denominator still counts cell-b's
        assert fu["n_unequal_elems"] == 3 * HIDDEN and fu["n_elems"] == 3 * HIDDEN
        assert rec["n_unequal_elems"] == 6 * HIDDEN and rec["n_elems_compared"] == 10 * HIDDEN
        assert rec["by_key"][k]["n_unequal_elems"] == 6 * HIDDEN
        assert rec["by_key"][k]["n_elems_compared"] == 6 * HIDDEN
        assert fu["max_ulps_edit"] is not None and fu["max_ulps_edit"] > 0
        assert rec["max_ulps_edit"] == rec["by_key"][k]["max_ulps_edit"] >= fu["max_ulps_edit"]
        assert rec["n_flag_disagreements"] == 0 and fu["note"] == ""

    def test_modified_flag_disagreement_is_unequal(self, monkeypatch):
        a, b, zero = self._objs()
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        orig = st._registry.apply

        def grouped_noop(*args, **kw):
            if not kw.get("legacy_masks") and isinstance(kw.get("ctx"), la.DecodeRows) \
                    and args[0] == "cell-b":
                return args[2]                                  # "unmodified" on the grouped side
            return orig(*args, **kw)
        monkeypatch.setattr(st._registry, "apply", grouped_noop)
        self._drive(st, view, a, b, zero)
        rec = st.shadow_record.as_dict()
        assert rec["n_unequal"] == 2
        assert "modified flags differ" in rec["first_unequal"]["note"]
        # a flag disagreement is a routing/masking defect, counted apart from kernel
        # variance so the gate can fail it in either reduction mode
        assert rec["n_flag_disagreements"] == 2
        k = rec["first_unequal"]["key"]
        assert rec["by_key"][k]["n_flag_disagreements"] == 2

    def test_shadow_uses_the_legacy_entry_point(self, monkeypatch):
        """The per-row path the shadow runs IS the legacy flag's function:
        one method object, called with legacy_masks=True by both."""
        a, b, zero = self._objs()
        calls = []
        orig = lw.LensWorkerSteering._slot_apply

        def spy(self, layer, slot, rows, *, legacy_masks):
            calls.append((self.apply_path, legacy_masks))
            return orig(self, layer, slot, rows, legacy_masks=legacy_masks)
        monkeypatch.setattr(lw.LensWorkerSteering, "_slot_apply", spy)
        st_ref, view_ref = _steering(_registry(a, b, zero, header=HEADER), per_row_apply="1")
        self._drive(st_ref, view_ref, a, b, zero)
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        self._drive(st, view, a, b, zero)
        legacy_calls = [c for c in calls if c[0] == lw.APPLY_PATH_PER_ROW]
        shadow_calls = [c for c in calls if c[0] == lw.APPLY_PATH_SHADOW]
        assert legacy_calls and all(lm is True for _, lm in legacy_calls)
        # the shadow path: every GROUP MEMBER (cell-a x3 + cell-b x2 + the scale-0
        # twin x1, at 2 layers) through the SAME function with the legacy masks;
        # the prefill slot is a per-slot leftover on both paths and keeps the
        # new masks (2 calls)
        assert sum(1 for _, lm in shadow_calls if lm is True) == 12
        assert sum(1 for _, lm in shadow_calls if lm is False) == 2
        assert lw.LensWorkerSteering._slot_apply is spy                     # one object, spied once
        assert st._shadow_block.__func__ is lw.LensWorkerSteering._shadow_block

    def test_shadow_error_is_contained_and_recorded(self, monkeypatch):
        a, b, zero = self._objs()
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        orig = st._registry.apply

        def legacy_raises(*args, **kw):
            if kw.get("legacy_masks") and args[0] == "cell-b":
                raise la.AdapterError("shadow-only refusal")
            return orig(*args, **kw)
        monkeypatch.setattr(st._registry, "apply", legacy_raises)
        outs, h0, rows = self._drive(st, view, a, b, zero, layers=(1,))
        rec = st.shadow_record.as_dict()
        assert rec["n_shadow_errors"] == 1 and rec["first_shadow_error"]["cell"] == "cell-b"
        assert "shadow-only refusal" in rec["first_shadow_error"]["error"]
        # cell-b's rows served unsteered and its requests recorded failed
        for rid in ("b0", "b1"):
            assert torch.equal(outs[0][rows[rid]], h0[rows[rid]])
            assert st.take_request_failure(rid) is not None
        st._require_alive()                                             # not condemned

    def test_per_row_flag_disables_shadow(self, monkeypatch):
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        a, b, zero = self._objs()
        st, _ = _steering(_registry(a, b, zero, header=HEADER), per_row_apply="1")
        assert st.apply_path == lw.APPLY_PATH_PER_ROW and st._shadow_compare is False

    def test_oproj_shadow(self, monkeypatch):
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        a = _lora_int_obj("lora-a")
        b = _lora_int_obj("lora-b", seed=70)
        zero = _lora_int_obj("none__lens0", scale=0.0)
        outs = {}
        for path, env in ((None, "1"), ("1", "1")):
            st, view = _oproj_steering(_registry(a, b, zero, header=HEADER), path)
            shapes = [(1, 9), (1, 4), (1, 6), (1, 11), (1, 5), (1, 8)]
            qsl, seq_lens, nat, pos = _forward(shapes)
            rids = ["a0", "u", "b0", "a1", "z", "a2"]
            sel = {"a0": _sel(a), "b0": _sel(b), "a1": _sel(a), "z": _sel(zero, 0.0), "a2": _sel(a)}
            view.set_batch(rids, sel, {r: list(range(c + 1)) for r, (q, c) in zip(rids, shapes)},
                           (qsl, seq_lens, nat), prompt_lens={r: 3 for r in rids})
            for layer in view.layers:
                for pre in layer.pre_hooks:
                    pre(None, (pos,), {})
            x, out = _ints(nat, D_IN, seed=80), _ints(nat, D_OUT, seed=81)
            outs[path] = st.oproj_hook_body(1, (x,), out)
            if path is None:
                rec = st.shadow_record.as_dict()
                assert st.apply_path == lw.APPLY_PATH_SHADOW
                assert rec["n_groups_compared"] == 2 and rec["n_unequal"] == 0
                assert rec["n_unmodified_groups"] == 1                 # the twin
                assert any(k.startswith("lora_all:lora_oproj:rankK:response:skipbos0:both:B3")
                           for k in rec["by_key"])
                assert rec["n_layer_calls_with_ge2_cells_compared"] == 1
        assert torch.equal(outs[None], outs["1"])

    def test_record_never_raises_and_key_vocabulary(self):
        rec = lw.ShadowCompareRecord(enabled=True)
        rec.observe(key="k", layer="x", cell=None, rows=None, site="block", equal=False,
                    max_abs_diff=float("nan"))
        assert rec.n_unequal == 1 and rec.first_unequal["layer"] is None
        assert rec.max_abs_diff == 0.0                                  # nan never becomes the max
        rec.observe_layer_call(object())                                # not iterable: counted, not raised
        assert rec.n_record_errors == 1
        op = _gated_int_op(3, positions=la.POSITIONS_RESPONSE, header=HEADER)
        assert lw.shadow_key("psr", (op,), 32) == "psr:relu_gated_vector:gemv:response:skipbos0:both:B32"
        # a key names the op(s) it buckets — there is no placeholder form (the
        # 2026-09-04 multigroup FAIL keyed op-less groups "x:-:-:-:-:-:B1", which
        # the gate cannot classify); an op-less group is recorded UNSERVED instead
        with pytest.raises(lw.LensWorkerError, match="unserved, never keyed"):
            lw.shadow_key("x", (), 1)
        # an unserved shadow error carries no key and enters by_key under none
        rec.observe_error(key=None, layer=1, cell="c", rows=2, site="block",
                          error=la.AdapterError("boom"))
        assert rec.n_shadow_errors == 1 and rec.first_shadow_error["key"] is None
        assert "None" not in rec.by_key and rec.n_record_errors == 1

    def test_batch_invariant_report_on_the_pinned_0_25_1_surface(self):
        """The report reads the state off the surface vllm 0.25.1 ACTUALLY has
        (:class:`Vllm0251BatchInvariant`): the private ``_batch_invariant_MODE``
        flag that ``init_batch_invariance()`` (run at ``init_device``, before
        model load) sets iff ``envs.VLLM_BATCH_INVARIANT`` — there is no
        ``vllm_is_batch_invariant`` at that version."""
        envs = types.SimpleNamespace(VLLM_BATCH_INVARIANT=False)
        bi = Vllm0251BatchInvariant(envs)
        assert not hasattr(bi, "vllm_is_batch_invariant")
        bi.init_batch_invariance()                                     # what gpu_worker.init_device does
        rep = lw.batch_invariant_report(envs, bi, version="0.25.1")
        assert rep == {"batch_invariant_env": False, "batch_invariant_active": False,
                       "batch_invariant_surface": "_batch_invariant_MODE (vllm 0.25.1)",
                       "batch_invariant_error": None}
        envs1 = types.SimpleNamespace(VLLM_BATCH_INVARIANT=True)
        bi1 = Vllm0251BatchInvariant(envs1)
        bi1.init_batch_invariance()
        rep = lw.batch_invariant_report(envs1, bi1, version="0.25.1")
        assert rep["batch_invariant_env"] is True and rep["batch_invariant_active"] is True
        assert rep["batch_invariant_error"] is None
        # env set but init NOT run (the flag still False): env and active disagree — reported
        # as they are, never reconciled; the gate rejects the pair
        bi2 = Vllm0251BatchInvariant(envs1)
        rep = lw.batch_invariant_report(envs1, bi2, version="0.25.1")
        assert rep["batch_invariant_env"] is True and rep["batch_invariant_active"] is False
        assert lw.BATCH_INVARIANT_MODE_ATTR_VERSIONS == ("0.25.1",)

    def test_batch_invariant_active_is_version_guarded_and_fails_loud(self):
        envs = types.SimpleNamespace(VLLM_BATCH_INVARIANT=False)
        bi = Vllm0251BatchInvariant(envs)
        # the private flag is read ONLY at a verified version
        rep = lw.batch_invariant_report(envs, bi, version="0.26.0")
        assert rep["batch_invariant_env"] is False and rep["batch_invariant_active"] is None
        assert rep["batch_invariant_surface"] is None
        assert "active:" in rep["batch_invariant_error"]
        assert "unverified private flag" in rep["batch_invariant_error"]
        with pytest.raises(lw.LensWorkerError, match="BATCH_INVARIANT_MODE_ATTR_VERSIONS"):
            lw.batch_invariant_active(bi, None)
        # a module with NEITHER surface: an explicit error naming what it carries
        rep = lw.batch_invariant_report(envs, types.SimpleNamespace(rms_norm_batch_invariant=1),
                                        version="0.25.1")
        assert rep["batch_invariant_active"] is None and "neither" in rep["batch_invariant_error"]
        assert "rms_norm_batch_invariant" in rep["batch_invariant_error"]
        # a non-bool flag is not a state
        rep = lw.batch_invariant_report(envs, types.SimpleNamespace(_batch_invariant_MODE=None),
                                        version="0.25.1")
        assert rep["batch_invariant_active"] is None and "not a bool" in rep["batch_invariant_error"]
        # a public reader (newer vLLM) is preferred at ANY version, over the private flag
        pub = types.SimpleNamespace(vllm_is_batch_invariant=lambda: True, _batch_invariant_MODE=False)
        rep = lw.batch_invariant_report(envs, pub, version="0.99.0")
        assert rep["batch_invariant_active"] is True
        assert rep["batch_invariant_surface"] == "vllm_is_batch_invariant()"
        assert rep["batch_invariant_error"] is None
        bad_pub = types.SimpleNamespace(vllm_is_batch_invariant=lambda: 1)
        rep = lw.batch_invariant_report(envs, bad_pub, version="0.99.0")
        assert rep["batch_invariant_active"] is None and "not a bool" in rep["batch_invariant_error"]
        # a non-bool env is not a state either
        rep = lw.batch_invariant_report(types.SimpleNamespace(VLLM_BATCH_INVARIANT="1"), bi,
                                        version="0.25.1")
        assert rep["batch_invariant_env"] is None and "env:" in rep["batch_invariant_error"]
        rep = lw.batch_invariant_report()                              # no vllm on this box
        assert rep["batch_invariant_env"] is None and "env:" in rep["batch_invariant_error"]
        assert rep["batch_invariant_active"] is None and "active:" in rep["batch_invariant_error"]


class TestShadowCrossSite:
    """The multigroup EMISSION path of the shadow record: a BLOCK-site object
    and an O_PROJ-site object co-scheduled in one forward — the gate's
    ``multigroup`` case (psr + lora_r4 + a second psr). ``site_layers``
    installs the registry-wide UNION of sites x layers and a forward's groups
    are built once for every hook, so at the block hook the LoRA group is
    applied with NO op (``ops_for`` = ``()``, the identity) and at the o_proj
    hook the gated group is — and until 2026-09-04 each such group was keyed
    with placeholder dashes (``lora_all:-:-:-:-:-:B2``), a key the gate
    cannot classify: the llama8b multigroup case FAILED on 8 of them while
    its 20192 comparisons sat an order of magnitude inside the contract.
    Every single-cell case escapes this because its registry (one cell +
    its scale-0 twin, the same ops) hooks exactly that cell's own sites, so
    ``ops_for`` is never empty at a hooked point. Pinned: every key names
    its op (classifiable by the adapters' own rule), an op-less group is
    counted UNSERVED per kind@site, and a side that modifies rows it serves
    no op at is a flag disagreement (a routing defect), never a dash key."""

    def _objs(self):
        a = _obj("cell-a", [_gated_int_op(1, positions=la.POSITIONS_RESPONSE, header=HEADER),
                            _gated_int_op(2, seed=3, positions=la.POSITIONS_RESPONSE, header=HEADER)])
        b = _lora_int_obj("lora-b", layers=(1,))                 # o_proj site, layer 1 only
        zero = _obj("none__lens0", [op for op in a.ops], scale=0.0)
        return a, b, zero

    def _drive(self, st, view, a, b, zero):
        nat, pos, rows = _mixed_batch(view, a, b, zero, prefill_ids=(1, 2) + HEADER)
        for layer in view.layers:
            for pre in layer.pre_hooks:
                pre(None, (pos,), {})
        h, r = _hr(nat)
        x, out = _ints(nat, D_IN, seed=80), _ints(nat, D_OUT, seed=81)
        blk = [st.hook_body(layer, (pos, h, r), (h, r))[0] for layer in (1, 2)]
        opj = st.oproj_hook_body(1, (x,), out)
        return blk, opj, rows

    @staticmethod
    def _assert_every_key_names_its_op(rec):
        assert rec["by_key"], "no keys recorded"
        for k in rec["by_key"]:
            parts = k.split(":")
            assert len(parts) == 7 and parts[-1].startswith("B"), k
            assert all(p and p != "-" for p in parts), f"placeholder axis in shadow key {k!r}"
            for op in parts[1].split("+"):
                la.op_reduces_in_group(op)      # the adapters' classifier accepts it (else raises)

    def test_every_key_names_its_op_and_op_less_groups_are_unserved(self, monkeypatch):
        a, b, zero = self._objs()
        reg = _registry(a, b, zero, header=HEADER)
        assert lw.site_layers(reg) == ((1, 2), (1,))
        st_ref, view_ref = _oproj_steering(_registry(a, b, zero, header=HEADER), "1")
        ref_blk, ref_opj, _ = self._drive(st_ref, view_ref, a, b, zero)
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        st, view = _oproj_steering(reg)
        assert st.apply_path == lw.APPLY_PATH_SHADOW
        blk, opj, _ = self._drive(st, view, a, b, zero)
        for i in range(2):
            assert torch.equal(blk[i], ref_blk[i])                 # the per-row result is served
        assert torch.equal(opj, ref_opj)
        rec = st.shadow_record.as_dict()
        self._assert_every_key_names_its_op(rec)
        keys = set(rec["by_key"])
        assert keys == {"synthetic_vector:relu_gated_vector:gemv:response:skipbos0:both:B3",   # cell-a, compared
                        "synthetic_vector:relu_gated_vector:gemv:response:skipbos0:both:B1",   # the twin, unmodified
                        "lora_all:lora_oproj:rankK:response:skipbos0:both:B2"}                # lora-b at o_proj
        # cell-a compared at block layers 1 + 2, lora-b at o_proj layer 1; the twin
        # unmodified (scale 0) at the two block layers
        assert rec["n_groups_compared"] == 3 and rec["n_equal"] == 3 and rec["n_unequal"] == 0
        assert rec["n_unmodified_groups"] == 2 and rec["n_shadow_errors"] == 0
        # UNSERVED: lora-b's group at the two block-site hooks (no block op), cell-a's
        # and the twin's groups at the o_proj hook (no o_proj op) — counted per
        # kind@site, entered under NO key
        assert rec["n_unserved_groups"] == 4
        assert rec["unserved"] == {"lora_all@block": 2, "synthetic_vector@oproj": 2}
        assert rec["first_unserved"] == {"kind": "lora_all", "cell": "lora-b", "site": "block",
                                         "layer": 1, "rows": 2}
        assert rec["n_unserved_modified"] == 0 and rec["first_unserved_modified"] is None
        assert rec["n_flag_disagreements"] == 0 and rec["n_record_errors"] == 0
        assert rec["n_layer_calls"] == 3
        # the selection record is the reference's, observation for observation
        assert st.selection_record.as_dict()["requests"] == st_ref.selection_record.as_dict()["requests"]

    def test_modifying_rows_with_no_op_at_the_site_is_a_flag_disagreement(self, monkeypatch):
        a, b, zero = self._objs()
        st_ref, view_ref = _oproj_steering(_registry(a, b, zero, header=HEADER), "1")
        ref_blk, _, _ = self._drive(st_ref, view_ref, a, b, zero)
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        st, view = _oproj_steering(_registry(a, b, zero, header=HEADER))
        orig = st._registry.apply

        def grouped_leaks(*args, **kw):
            out = orig(*args, **kw)
            # the GROUPED block-site apply "modifies" lora-b's rows — a cell with no
            # block-site op (a routing defect; the per-row side stays the identity)
            if not kw.get("legacy_masks") and args[0] == "lora-b":
                out = out + 1.0
            return out
        monkeypatch.setattr(st._registry, "apply", grouped_leaks)
        blk, _, _ = self._drive(st, view, a, b, zero)
        for i in range(2):
            assert torch.equal(blk[i], ref_blk[i])                 # the per-row result is served
        rec = st.shadow_record.as_dict()
        self._assert_every_key_names_its_op(rec)                    # still no dash key
        assert rec["n_unserved_groups"] == 4 and rec["n_unserved_modified"] == 2
        assert rec["n_flag_disagreements"] == 2                     # the gate fails this in every mode
        fum = rec["first_unserved_modified"]
        assert fum["cell"] == "lora-b" and fum["site"] == "block" and fum["layer"] == 1
        assert "no op at this site, yet modified: grouped=True, per_row=False" in fum["note"]
        assert rec["n_unequal"] == 0 and rec["first_unequal"] is None     # not a keyed comparison
