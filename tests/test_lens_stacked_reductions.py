"""The REDUCTION MODE inside a vectorized lens group (lens_adapters.
STACKED_REDUCTIONS_ENV = ``SISTEER_LENS_STACKED_REDUCTIONS``) — the
2026-09-04 decision (docs/lens_stacked_reductions_scoping.md): STACKED
reductions are the default served arithmetic on every family and every
reduction kind; ``0`` / ``per_row`` opts back to the bit-identical per-row
path.

CONTRACT PINNED HERE. (1) MODE VOCABULARY: unset / empty / 1 / stacked =
stacked; 0 / per_row / off = per_row; anything else refuses (AdapterError) —
served arithmetic is never picked by a typo. The value is read once at import
(``la._STACKED_REDUCTIONS``); tests flip that attribute, never the env.
(2) STACKED DEFAULT: every reduction inside a multi-row group (psr gate gemv,
LoRA rank-1 / rank-K GEMM pair, LoReFT affine pair + back-projection, o_proj
LoRA) is ONE ``[B, K]`` call; the worker attests ``vectorized+stacked`` (and
``vectorized+stacked+shadow(serves per_row)`` / ``vectorized+stacked(per_row:
<names>)``), ``lens_reduction_mode == "stacked"`` in its SelectionRecord, and
on integer-valued CPU tensors — exact arithmetic — the stacked result equals
the per-row one bit for bit (values cannot tell the two apart here; the GPU
gate does that). (3) PER-ROW OPT-OUT: ``B`` single-row calls — the pre-landing
call sequence, restated so a change to either side shows up — attested as
``vectorized+per_row`` (+ the same suffixes), ``lens_reduction_mode ==
"per_row"``. (4) B=1 IS EXACT IN BOTH MODES BY CONSTRUCTION: a one-row group is
the single-row call itself, so the gate's pure_B1 / controls_B1 stay hard and
exact for every kind. (5) OP CLASSIFICATION: ``la.REDUCTION_OPS`` is exactly
the set of ops whose grouped apply goes through ``_row_reduction`` with
``rowwise=True``; every other registered op is elementwise (or prefill-only,
never grouped) — the split the gate's tolerance reads off a shadow key; since
2026-09-06 the reduction ops split further into GATED (``la.GATED_REDUCTION_
OPS``: the gated-vector and low-rank pairs, judged AT THE GATE) and LINEAR
(LoRA, LoReFT, judged on the edit) classes, defined ONCE in the adapters and
mirrored by the gate's literals, which are pinned equal here.
(6) THE DEVIATION UNIT: ``edit_excess_ulps(applied, ref, base)`` reads a
kernel-variance flip as <= ~1 ulp of the EDIT beyond one rounding of the sum
(the cancellation case that is 128 ulps of the output reads 1), a dropped
one-output-ulp edit as 0 (indistinguishable per element — the concentration
fields judge it) and a dropped eight-output-ulp edit as ~100; the shadow
record carries ``n_elems_compared`` / ``max_ulps_edit`` / ``max_frac_row`` /
``max_frac_group`` / ``n_flag_disagreements`` per key and overall. (7) PLANTED
BUGS: a sign flip, a row shift, a permuted operand and a wrong-precision read
in the grouped path each fail the gate's reduction tolerance by ORDERS OF
MAGNITUDE (fraction of unequal elements >= 0.1 against the 1e-4 bound); one
row's edit dropped, halved, or swapped with another row's fails it through the
ROW concentration bound on the real worker record, and in a construction that
defeats the first form of the contract (every element within 4 output ulps,
case-wide fraction 5e-6) still fails through the row and group bounds — the
swap also through the edit bound; a single moved element of an elementwise op
fails it (exact); the unperturbed stacked run passes with zero unequal groups.
(8) PLUMBING: the gate sets the env explicitly per
reduction mode and scrubs it otherwise; the A/B driver's sides name both
modes; the microbench leaves the process mode as it found it."""

import copy
import dataclasses
import math
import sys
from pathlib import Path

import json

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_worker as lw
from test_lens_vectorized_apply import (HEADER, HIDDEN, TestHookBodyGroupedVsReference,
                                        TestShadowCompare, _add_int_op, _affine_int_op, _dctx,
                                        _gated_int_op, _hr, _ints, _lowrank_int_op, _mixed_batch,
                                        _obj, _registry, _steering)

REPO = Path(__file__).resolve().parents[1]
# scripts/ goes to the FRONT unconditionally: scripts/modal/lens_vectorize_gate.py shares the bare module name with the
# reviewed gate in scripts/, and 52 test modules put scripts/modal on sys.path — whichever directory is in front at the
# FIRST bare import wins for the whole process. A conditional insert left scripts/ wherever an earlier module had put
# it, and a sharded gate whose shard collected such a module first imported the Modal launcher here (AttributeError:
# module 'modal' has no attribute 'App' under a modal stub; cpu-suite-8968828 shard 1, 2026-09-16).
_SCRIPTS_DIR = str(REPO / "scripts")
if _SCRIPTS_DIR in sys.path:
    sys.path.remove(_SCRIPTS_DIR)
sys.path.insert(0, _SCRIPTS_DIR)

import lens_stacked_ab as ab  # noqa: E402
import lens_stacked_microbench as mb  # noqa: E402
import lens_vectorize_gate as g  # noqa: E402
assert Path(g.__file__).resolve() == (REPO / "scripts" / "lens_vectorize_gate.py").resolve(), \
    f"the bare name resolved to {g.__file__}, not the reviewed gate in scripts/ (sys.path order)"

STACKED, PER_ROW = la.REDUCTION_MODE_STACKED, la.REDUCTION_MODE_PER_ROW


@pytest.fixture
def stacked(monkeypatch):
    """The default mode, pinned process-locally (the attribute the env sets at import)."""
    monkeypatch.setattr(la, "_STACKED_REDUCTIONS", True)
    yield


@pytest.fixture
def per_row_mode(monkeypatch):
    """The opt-out (SISTEER_LENS_STACKED_REDUCTIONS=0), pinned process-locally."""
    monkeypatch.setattr(la, "_STACKED_REDUCTIONS", False)
    yield


def _spy(monkeypatch):
    calls = []
    real = la._matmul

    def spy(x, w):
        calls.append((tuple(x.shape), tuple(w.shape)))
        return real(x, w)
    monkeypatch.setattr(la, "_matmul", spy)
    return calls


def _spy_linear(monkeypatch):
    calls = []
    real = la._linear

    def spy(x, w, b=None):
        calls.append((tuple(x.shape), tuple(w.shape), b is not None))
        return real(x, w, b)
    monkeypatch.setattr(la, "_linear", spy)
    return calls


def _oproj_op(d_in=6, d_out=HIDDEN, k=2):
    return la.HookOp(op=la.OP_LORA_OPROJ, layer=0,
                     lowrank_a=_ints(k, d_in, seed=31), lowrank_b=_ints(d_out, k, seed=32),
                     positions=la.POSITIONS_RESPONSE, header_ids=HEADER)


# --------------------------------------------------------------------------
# 1. the mode vocabulary; the default
# --------------------------------------------------------------------------


class TestModeVocabulary:
    def test_parse_reduction_mode(self):
        for default in (None, "", "   "):
            assert la.parse_reduction_mode(default) == STACKED
        for on in ("1", "true", "ON", " yes ", "all", "True", "stacked", " Stacked "):
            assert la.parse_reduction_mode(on) == STACKED, on
            assert la.parse_stacked_reductions(on) is True
        for off in ("0", "false", "off", "no", "per_row", " PER_ROW "):
            assert la.parse_reduction_mode(off) == PER_ROW, off
            assert la.parse_stacked_reductions(off) is False
        for bad in ("psr", "maybe", "2", "stacked,psr"):
            with pytest.raises(la.AdapterError, match="not a reduction mode"):
                la.parse_reduction_mode(bad)
        assert la.parse_stacked_reductions(None) is True

    def test_default_is_stacked_in_this_process(self, monkeypatch):
        monkeypatch.delenv(la.STACKED_REDUCTIONS_ENV, raising=False)
        assert la.parse_reduction_mode(None) == STACKED
        assert la.STACKED_REDUCTIONS_ENV == "SISTEER_LENS_STACKED_REDUCTIONS"
        assert la.REDUCTION_MODES == (STACKED, PER_ROW) == ("stacked", "per_row")
        # the import-time read (no env in the test process) left the default on
        assert la.stacked_reductions_enabled() is True and la.reduction_mode() == STACKED

    def test_reduction_mode_follows_the_switch(self, monkeypatch):
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", False)
        assert la.reduction_mode() == PER_ROW and la.stacked_reductions_enabled() is False
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", True)
        assert la.reduction_mode() == STACKED and la.stacked_reductions_enabled() is True

    def test_apply_path_vocabulary(self):
        assert lw.apply_path_string(STACKED) == "vectorized+stacked" == lw.APPLY_PATH_VECTORIZED_STACKED
        assert lw.apply_path_string(PER_ROW) == "vectorized+per_row" == lw.APPLY_PATH_VECTORIZED_PER_ROW
        assert lw.apply_path_string(STACKED, shadow=True) == lw.APPLY_PATH_SHADOW \
            == "vectorized+stacked+shadow(serves per_row)"
        assert lw.apply_path_string(PER_ROW, shadow=True) == lw.APPLY_PATH_SHADOW_PER_ROW \
            == "vectorized+per_row+shadow(serves per_row)"
        assert lw.apply_path_string(STACKED, per_row_names=["psr", "lora_oproj"]) \
            == "vectorized+stacked(per_row: lora_oproj,psr)"
        assert lw.APPLY_PATH_PER_ROW == "per_row" and lw.APPLY_PATH_VECTORIZED == "vectorized"
        assert lw.LENS_REDUCTION_MODE_KEY == "lens_reduction_mode"
        with pytest.raises(lw.LensWorkerError, match="unknown reduction mode"):
            lw.apply_path_string("bmm")


# --------------------------------------------------------------------------
# 2. the stacked default: one call per reduction, equal values on exact ints
# --------------------------------------------------------------------------


class TestStackedDefault:
    @pytest.mark.parametrize("R", [2, 5])
    def test_one_stacked_call_per_reduction(self, monkeypatch, stacked, R):
        mm = _spy(monkeypatch)
        rows = _ints(R, HIDDEN, seed=R)
        la.apply_hook_ops(rows, (_gated_int_op(0),), 1.0, layer=0, ctx=la.DecodeRows())
        assert mm == [((R, HIDDEN), (HIDDEN,))]
        mm.clear()
        la.apply_hook_ops(rows, (_lowrank_int_op(0),), 1.0, layer=0, independent_rows=True)
        assert mm == [((R, HIDDEN), (HIDDEN, 2)), ((R, 2), (2, HIDDEN))]
        mm.clear()
        lin = _spy_linear(monkeypatch)
        la.apply_hook_ops(rows, (_affine_int_op(0),), 1.0, layer=0, ctx=la.DecodeRows())
        assert lin == [((R, HIDDEN), (2, HIDDEN), True), ((R, HIDDEN), (2, HIDDEN), False)]
        assert mm == [((R, 2), (2, HIDDEN))]
        mm.clear()
        x, out = _ints(R, 6, seed=33), _ints(R, HIDDEN, seed=34)
        la.apply_oproj_ops(x, out, (_oproj_op(),), 1.0, layer=0, ctx=la.DecodeRows())
        assert mm == [((R, 6), (6, 2)), ((R, 2), (2, HIDDEN))]

    @pytest.mark.parametrize("make", [
        lambda: _gated_int_op(0),
        lambda: _gated_int_op(0, positions=la.POSITIONS_RESPONSE, header=HEADER),
        lambda: _lowrank_int_op(0),
        lambda: _affine_int_op(0),
    ])
    def test_stacked_equals_per_row_on_exact_integers(self, monkeypatch, make):
        op = make()
        rows = _ints(6, HIDDEN, seed=6)
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", True)
        st = la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=la.DecodeRows())
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", False)
        pr = la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=la.DecodeRows())
        single = torch.cat([la.apply_hook_ops(rows[i:i + 1], (op,), 1.0, layer=0, ctx=_dctx())
                            for i in range(6)])
        assert torch.equal(st, pr) and torch.equal(st, single) and not torch.equal(st, rows)

    def test_one_requests_rows_stay_one_call(self, monkeypatch, stacked):
        """The mode governs GROUPS only: a per-request (non-rowwise) apply of
        R rows is one [R, d] call in either mode, exactly as before."""
        mm = _spy(monkeypatch)
        rows = _ints(4, HIDDEN, seed=4)
        la.apply_hook_ops(rows, (_gated_int_op(0),), 1.0, layer=0)
        assert mm == [((4, HIDDEN), (HIDDEN,))]
        mm.clear()
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", False)
        la.apply_hook_ops(rows, (_gated_int_op(0),), 1.0, layer=0)
        assert mm == [((4, HIDDEN), (HIDDEN,))]

    def test_worker_attests_the_mode(self, monkeypatch, stacked):
        a = _obj("cell-a", [_gated_int_op(1)])
        st, _ = _steering(_registry(a, header=HEADER))
        assert st.apply_path == "vectorized+stacked" and st._stacked_reductions is True
        assert st.reduction_mode == STACKED
        assert st.selection_record.as_dict()[lw.LENS_REDUCTION_MODE_KEY] == STACKED
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        st, _ = _steering(_registry(a, header=HEADER))
        assert st.apply_path == "vectorized+stacked+shadow(serves per_row)"
        monkeypatch.delenv(lw.SHADOW_COMPARE_ENV)
        st, _ = _steering(_registry(a, header=HEADER), per_row_apply="relu_gated_vector")
        assert st.apply_path == "vectorized+stacked(per_row: relu_gated_vector)"
        # the all-per-row reference forms no group: the path stays "per_row", the
        # mode is still attested (what a group WOULD get)
        st, _ = _steering(_registry(a, header=HEADER), per_row_apply="1")
        assert st.apply_path == "per_row" and st._stacked_reductions is False
        assert st.reduction_mode == STACKED
        assert st.selection_record.as_dict()[lw.LENS_REDUCTION_MODE_KEY] == STACKED
        # the A/B driver names the same strings
        assert ab.SIDES[ab.STACKED][1] == "vectorized+stacked"
        assert ab.SIDES[ab.STACKED_SHADOW][1] == "vectorized+stacked+shadow(serves per_row)"
        assert ab.SIDES[ab.PER_ROW][1] == "vectorized+per_row"

    def test_selection_record_reset_keeps_the_mode(self):
        rec = lw.SelectionRecord(reduction_mode=PER_ROW)
        rec.reset()
        assert rec.as_dict()[lw.LENS_REDUCTION_MODE_KEY] == PER_ROW
        assert lw.SelectionRecord().as_dict()[lw.LENS_REDUCTION_MODE_KEY] is None

    def test_worker_mixed_batch_equals_the_per_row_reference_on_integers(self, monkeypatch):
        """The hook body's grouped apply under the default: stacked calls, and on
        exact integers the same bytes as the all-per-row reference."""
        t = TestHookBodyGroupedVsReference()
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", True)
        mm = _spy(monkeypatch)
        st_v, vec, _, _, rows = t._run(None)
        assert st_v.apply_path == "vectorized+stacked"
        # cell-a's 3-row decode group at each of the two layers: ONE [3, d] gemv
        assert ((3, HIDDEN), (HIDDEN,)) in mm
        st_r, ref, _, _, _ = t._run("1")
        assert st_r.apply_path == "per_row"
        for layer in t.LAYERS:
            assert torch.equal(vec[layer], ref[layer])


# --------------------------------------------------------------------------
# 3. the per-row opt-out: the pre-landing call sequence, attested as such
# --------------------------------------------------------------------------


class TestPerRowOptOut:
    @pytest.mark.parametrize("R", [2, 5])
    def test_issues_the_per_row_call_sequence(self, monkeypatch, per_row_mode, R):
        """Byte-identical to the pre-landing `_row_reduction`: R single-row
        gemvs for the psr gate, R (read, write) GEMM pairs for the low-rank
        ops, R (F.linear, F.linear, matmul) triples for the affine subspace,
        R (read, write) pairs at the o_proj site — never a [R, d] call."""
        mm = _spy(monkeypatch)
        rows = _ints(R, HIDDEN, seed=R)
        la.apply_hook_ops(rows, (_gated_int_op(0),), 1.0, layer=0, ctx=la.DecodeRows())
        assert mm == [((1, HIDDEN), (HIDDEN,))] * R
        mm.clear()
        la.apply_hook_ops(rows, (_lowrank_int_op(0),), 1.0, layer=0, independent_rows=True)
        assert mm == [((1, HIDDEN), (HIDDEN, 2)), ((1, 2), (2, HIDDEN))] * R
        mm.clear()
        lin = _spy_linear(monkeypatch)
        la.apply_hook_ops(rows, (_affine_int_op(0),), 1.0, layer=0, ctx=la.DecodeRows())
        assert lin == [((1, HIDDEN), (2, HIDDEN), True), ((1, HIDDEN), (2, HIDDEN), False)] * R
        assert mm == [((1, 2), (2, HIDDEN))] * R
        mm.clear()
        x, out = _ints(R, 6, seed=33), _ints(R, HIDDEN, seed=34)
        la.apply_oproj_ops(x, out, (_oproj_op(),), 1.0, layer=0, ctx=la.DecodeRows())
        assert mm == [((1, 6), (6, 2)), ((1, 2), (2, HIDDEN))] * R

    def test_worker_attests_per_row(self, monkeypatch, per_row_mode):
        a = _obj("cell-a", [_gated_int_op(1)])
        st, _ = _steering(_registry(a, header=HEADER))
        assert st.apply_path == "vectorized+per_row" and st._stacked_reductions is False
        assert st.reduction_mode == PER_ROW
        assert st.selection_record.as_dict()[lw.LENS_REDUCTION_MODE_KEY] == PER_ROW
        monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
        st, _ = _steering(_registry(a, header=HEADER))
        assert st.apply_path == "vectorized+per_row+shadow(serves per_row)"
        rec = st.shadow_record.as_dict()
        assert rec["max_ulps"] == 0.0 and rec["max_abs_ref"] == 0.0 and rec["n_unequal_elems"] == 0
        assert rec["n_elems_compared"] == 0 and rec["max_ulps_edit"] == 0.0
        assert rec["max_frac_row"] == 0.0 and rec["max_frac_group"] == 0.0
        monkeypatch.delenv(lw.SHADOW_COMPARE_ENV)
        st, _ = _steering(_registry(a, header=HEADER), per_row_apply="relu_gated_vector")
        assert st.apply_path == "vectorized+per_row(per_row: relu_gated_vector)"

    def test_worker_mixed_batch_equals_the_reference_on_integers(self, monkeypatch, per_row_mode):
        t = TestHookBodyGroupedVsReference()
        mm = _spy(monkeypatch)
        st_v, vec, _, _, _ = t._run(None)
        assert st_v.apply_path == "vectorized+per_row"
        assert ((3, HIDDEN), (HIDDEN,)) not in mm and ((1, HIDDEN), (HIDDEN,)) in mm
        st_r, ref, _, _, _ = t._run("1")
        for layer in t.LAYERS:
            assert torch.equal(vec[layer], ref[layer])


# --------------------------------------------------------------------------
# 4. B=1 is exact in both modes by construction
# --------------------------------------------------------------------------


class TestSingleRowIsExactInBothModes:
    @pytest.mark.parametrize("mode_on", [True, False], ids=["stacked", "per_row"])
    def test_row_reduction_on_one_row_ignores_the_mode(self, monkeypatch, mode_on):
        """`_row_reduction` on a one-row group calls `fn` ONCE on the row
        itself in either mode — the single-row call the per-request path
        makes — so a B=1 request cannot differ between the modes, and the
        gate's pure_B1 / controls_B1 / steering_bites_B1 stay exact for every
        kind (job 382416, the stacked candidate: pure_B1=PASS on all 26 cases that
        reached a verdict)."""
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", mode_on)
        seen = []

        def fn(x):
            seen.append(x)
            return x * 2
        one = _ints(1, HIDDEN)
        out = la._row_reduction(one, fn, True)
        assert len(seen) == 1 and seen[0] is one and torch.equal(out, one * 2)
        seen.clear()
        three = _ints(3, HIDDEN, seed=3)
        out3 = la._row_reduction(three, fn, True)
        if mode_on:
            assert len(seen) == 1 and seen[0] is three           # ONE stacked call
        else:
            assert len(seen) == 3 and all(s.shape[0] == 1 for s in seen)   # per row
        assert torch.equal(out3, three * 2)
        seen.clear()
        la._row_reduction(three, fn, False)                       # one request's rows
        assert len(seen) == 1 and seen[0] is three

    def test_single_row_is_one_call_either_way(self, monkeypatch):
        for mode_on in (True, False):
            monkeypatch.setattr(la, "_STACKED_REDUCTIONS", mode_on)
            mm = _spy(monkeypatch)
            row = _ints(1, HIDDEN)
            la.apply_hook_ops(row, (_gated_int_op(0),), 1.0, layer=0, ctx=la.DecodeRows())
            la.apply_hook_ops(row, (_gated_int_op(0),), 1.0, layer=0)
            assert mm == [((1, HIDDEN), (HIDDEN,))] * 2


# --------------------------------------------------------------------------
# 5. which ops the mode governs (the gate's tolerance class)
# --------------------------------------------------------------------------


class TestReductionOpClassification:
    def test_every_registered_op_is_classified_exactly_once(self):
        ops = set(la.OPS)
        assert ops == la.REDUCTION_OPS | la.ELEMENTWISE_OPS
        assert not (la.REDUCTION_OPS & la.ELEMENTWISE_OPS)
        assert la.REDUCTION_OPS == {la.OP_RELU_GATED_VECTOR, la.OP_LINEAR_GATED_VECTOR, la.OP_AFFINE_SUBSPACE,
                                    la.OP_LOWRANK_RELU_GATED, la.OP_LOWRANK_LINEAR_GATED, la.OP_LORA_OPROJ,
                                    la.OP_RELU_GATED_VECTOR_PAIR}   # the paired write (2026-09-17)
        for op in la.REDUCTION_OPS:
            assert la.op_reduces_in_group(op) is True
        for op in la.ELEMENTWISE_OPS:
            assert la.op_reduces_in_group(op) is False
        with pytest.raises(la.AdapterError, match="classified neither"):
            la.op_reduces_in_group("bogus_op")
        # the two reduction CLASSES (2026-09-06) partition the reduction ops — defined ONCE here
        assert la.GATED_REDUCTION_OPS == {la.OP_RELU_GATED_VECTOR, la.OP_LINEAR_GATED_VECTOR,
                                          la.OP_LOWRANK_RELU_GATED, la.OP_LOWRANK_LINEAR_GATED,
                                          la.OP_RELU_GATED_VECTOR_PAIR}   # two gated gemvs per row
        assert la.LINEAR_REDUCTION_OPS == {la.OP_AFFINE_SUBSPACE, la.OP_LORA_OPROJ}
        assert la.GATED_REDUCTION_OPS | la.LINEAR_REDUCTION_OPS == la.REDUCTION_OPS
        assert not (la.GATED_REDUCTION_OPS & la.LINEAR_REDUCTION_OPS)
        # the apply's own pairs, plus the paired write (one op, two gated-vector gates per row)
        assert set(la._GATED_OPS) | set(la._LOWRANK_OPS) | {la.OP_RELU_GATED_VECTOR_PAIR} == la.GATED_REDUCTION_OPS
        # every registered op lands in exactly one of the three classes
        classes = {op: la.reduction_op_class(op) for op in ops}
        assert len(la.OP_CLASSES) == 3 and set(classes.values()) == set(la.OP_CLASSES)
        for op in la.GATED_REDUCTION_OPS:
            assert classes[op] == la.OP_CLASS_REDUCTION_GATED == "reduction_gated", op
        for op in la.LINEAR_REDUCTION_OPS:
            assert classes[op] == la.OP_CLASS_REDUCTION_LINEAR == "reduction_linear", op
        for op in la.ELEMENTWISE_OPS:
            assert classes[op] == la.OP_CLASS_ELEMENTWISE == "elementwise", op
        with pytest.raises(la.AdapterError, match="classified neither"):
            la.reduction_op_class("bogus_op")
        # the gate's literal mirrors equal the adapters' definition (the way _REFUSAL_KINDS mirrors SHED_KINDS)
        assert g.GATED_REDUCTION_OPS == la.GATED_REDUCTION_OPS and g.LINEAR_REDUCTION_OPS == la.LINEAR_REDUCTION_OPS
        assert (g.OP_CLASS_REDUCTION_GATED, g.OP_CLASS_REDUCTION_LINEAR, g.OP_CLASS_ELEMENTWISE) == (
            la.OP_CLASS_REDUCTION_GATED, la.OP_CLASS_REDUCTION_LINEAR, la.OP_CLASS_ELEMENTWISE)
        tail = "gemv:response:skipbos0:both:B32"
        for op in ops:
            assert g.shadow_key_op_class(f"kind:{op}:{tail}") == classes[op], op
        # no gated op is served at the o_proj site, so the block-site shadow carries every gate stat
        # (lens_worker._shadow_oproj records none on purpose)
        assert not (set(la.OPROJ_SITE_OPS) & la.GATED_REDUCTION_OPS)

    def test_reduction_ops_go_through_row_reduction_and_elementwise_do_not(self, monkeypatch, stacked):
        seen = []
        real = la._row_reduction

        def spy(x, fn, rowwise):
            seen.append((tuple(x.shape), rowwise))
            return real(x, fn, rowwise)
        monkeypatch.setattr(la, "_row_reduction", spy)
        rows = _ints(3, HIDDEN, seed=3)
        # block-site reduction ops: relu / linear gated, lowrank relu / linear, affine
        block_ops = [
            _gated_int_op(0),
            la.HookOp(op=la.OP_LINEAR_GATED_VECTOR, layer=0, vector=_ints(HIDDEN, seed=1),
                      gate_w=_ints(HIDDEN, seed=2), gate_b=torch.tensor([100.0])),
            _lowrank_int_op(0),
            la.HookOp(op=la.OP_LOWRANK_LINEAR_GATED, layer=0, lowrank_a=_ints(2, HIDDEN, seed=9),
                      lowrank_b=_ints(HIDDEN, 2, seed=10)),
            _affine_int_op(0),
        ]
        for op in block_ops:
            seen.clear()
            la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=la.DecodeRows())
            assert seen and all(s == ((3, HIDDEN), True) for s in seen), op.op
            assert la.op_reduces_in_group(op.op) is True
        seen.clear()
        la.apply_oproj_ops(_ints(3, 6, seed=33), _ints(3, HIDDEN, seed=34), (_oproj_op(),), 1.0,
                           layer=0, ctx=la.DecodeRows())
        assert seen == [((3, 6), True)] and la.op_reduces_in_group(la.OP_LORA_OPROJ) is True
        seen.clear()
        la.apply_hook_ops(rows, (_add_int_op(0),), 1.0, layer=0, ctx=la.DecodeRows())
        assert seen == [] and la.op_reduces_in_group(la.OP_ADD_VECTOR) is False
        # the two prefill_last ops are prefill-only by validation: never grouped
        for op_name in (la.OP_PREFILL_LAST_MATMUL, la.OP_PREFILL_LAST_COSINE_GATED):
            assert la.op_reduces_in_group(op_name) is False
        matmul = la.HookOp(op=la.OP_PREFILL_LAST_MATMUL, layer=0, matrix=torch.eye(HIDDEN),
                           phase=la.PHASE_PREFILL)
        assert matmul.phase == la.PHASE_PREFILL and lw.phase_vectorizes_in_decode(matmul.phase) is False


# --------------------------------------------------------------------------
# 6. ulps at the edit scale; the shadow record's tolerance fields
# --------------------------------------------------------------------------


def _adjacent(b: torch.Tensor, steps: int = 1) -> torch.Tensor:
    """``b`` moved ``steps`` representable values away from zero."""
    it = torch.int16 if b.dtype == torch.bfloat16 else torch.int32
    bits = b.view(it)
    return (bits + torch.where(bits >= 0, steps, -steps).to(it)).view(b.dtype)


class TestUlps:
    @pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
    def test_adjacent_values_are_one_ulp(self, dtype):
        b = torch.tensor([1.0, 2.0, 100.0, -0.375, 3e-3], dtype=dtype)
        assert torch.equal(la.ulp_distance(_adjacent(b), b), torch.ones(5))
        assert torch.equal(la.ulp_distance(_adjacent(b, 2), b), torch.full((5,), 2.0))
        assert torch.equal(la.ulp_distance(b, b), torch.zeros(5))

    def test_bf16_ulp_is_two_to_the_minus_seven_at_one(self):
        b = torch.tensor([1.0], dtype=torch.bfloat16)
        a = torch.tensor([1.0 + 2 ** -7], dtype=torch.bfloat16)
        assert float(la.ulp_distance(a, b)) == 1.0
        # the 382416 residual-stream diffs: 2^-6 at |x| in [2, 4), 2^-2 at [32, 64)
        assert float(la.ulp_distance(torch.tensor([2.0 + 2 ** -6], dtype=torch.bfloat16),
                                     torch.tensor([2.0], dtype=torch.bfloat16))) == 1.0
        assert float(la.ulp_distance(torch.tensor([32.25], dtype=torch.bfloat16),
                                     torch.tensor([32.0], dtype=torch.bfloat16))) == 1.0

    def test_edit_excess_ulps_is_the_contracts_unit(self):
        """The contract's per-element unit: |applied - ref|
        beyond one rounding of the output, in bf16 ulps of the per-row edit.
        The microbench's 128-ulp tail (an edit of ~0.75 landing the output at
        0.0039; one rounding of the edit is 128 ulps of the output) reads ~1; a
        dropped edit of one output ulp reads 0 (per element it IS a flipped sum
        rounding — the concentration bounds judge it); a dropped edit of eight
        output ulps reads 96; an edit written where per-row wrote none reads
        astronomically."""
        bf = torch.bfloat16
        base = torch.tensor([[-0.75, 1000.0, 1000.0, 1000.0, 1.0]], dtype=bf)
        ref = torch.tensor([[0.0039, 1004.0, 1004.0, 1032.0, 1.0]], dtype=bf)     # edits .75, 4, 4, 32, 0
        app = torch.tensor([[0.0039 + 2 ** -8, 1000.0, 1004.0, 1000.0, 1.5]], dtype=bf)
        ex = la.edit_excess_ulps(app, ref, base)[0]
        assert la.ulp_distance(app, ref)[0, 0] == 128.0 and 0.99 <= float(ex[0]) <= 1.0   # the tail, as 1
        assert float(ex[1]) == 0.0                                  # dropped 1-output-ulp edit: invisible here
        assert float(ex[2]) == 0.0                                  # equal
        assert float(ex[3]) == 96.0                                 # dropped 8-output-ulp edit: (32-8)/0.25
        assert float(ex[4]) == la.EDIT_ULPS_CAP                     # an edit per-row never made: capped, finite
        # halving an edit of 8 output ulps: (16 - 8) / ulp(32) = 32 excess ulps
        assert float(la.edit_excess_ulps(torch.tensor([[1016.0]], dtype=bf), torch.tensor([[1032.0]], dtype=bf),
                                         torch.tensor([[1000.0]], dtype=bf))) == 32.0
        assert la.ulp_spacing(torch.tensor([1000.0]), bf).item() == 4.0
        assert la.ulp_spacing(torch.tensor([0.75]), bf).item() == 2 ** -8

    def test_mismatch_refuses(self):
        with pytest.raises(la.AdapterError):
            la.ulp_distance(torch.zeros(2), torch.zeros(2, dtype=torch.bfloat16))
        with pytest.raises(la.AdapterError):
            la.ulp_distance(torch.zeros(2), torch.zeros(3))
        with pytest.raises(la.AdapterError, match="magnitude tensor"):
            la.ulp_distance(torch.zeros(2), torch.zeros(2), at=torch.zeros(3))
        with pytest.raises(la.AdapterError, match="edit_excess_ulps"):
            la.edit_excess_ulps(torch.zeros(2), torch.zeros(2), torch.zeros(2, dtype=torch.bfloat16))
        with pytest.raises(la.AdapterError, match="edit_excess_ulps"):
            la.edit_excess_ulps(torch.zeros(2), torch.zeros(3), torch.zeros(3))


class TestShadowRecordFields:
    def test_unequal_stats_helper(self):
        base = torch.tensor([[0.0, 0.0, -1.5]], dtype=torch.bfloat16)
        ref = torch.tensor([[1.0, 0.0, 2.0]], dtype=torch.bfloat16)                 # edit = (1, 0, 3.5)
        app = torch.tensor([[1.0 + 2 ** -7, 0.5, 2.0]], dtype=torch.bfloat16)
        s = lw._unequal_stats(app, ref, base)
        assert s["n_unequal_elems"] == 2 and s["max_abs_diff"] == 0.5 and s["max_abs_ref"] == 2.0
        assert s["max_ulps"] == 1.0          # the zero-reference element is excluded from the ulp read
        # element 0: one rounding of the output -> 0 excess; element 1: an edit per-row never made -> the cap
        assert s["max_ulps_edit"] == la.EDIT_ULPS_CAP
        # concentration: 2 of the row's 3 elements, 2 of the group's 3
        assert s["max_frac_row"] == pytest.approx(2 / 3) and s["frac_group"] == pytest.approx(2 / 3)
        # without the base rows the edit unit is simply absent; the concentration fields are not
        s_nb = lw._unequal_stats(app, ref)
        assert "max_ulps_edit" not in s_nb and s_nb["max_frac_row"] == pytest.approx(2 / 3)
        s0 = lw._unequal_stats(torch.tensor([[0.5]]), torch.tensor([[0.0]]), torch.tensor([[0.0]]))
        assert "max_ulps" not in s0 and s0["n_unequal_elems"] == 1 and s0["max_ulps_edit"] > 0
        assert s0["max_frac_row"] == 1.0 and s0["frac_group"] == 1.0
        # a 3-row group with one row's elements all moved: row fraction 1, group fraction 1/3
        b3 = torch.zeros(3, 4, dtype=torch.bfloat16); r3 = b3 + 2.0; a3 = r3.clone(); a3[1] = 0.0
        s3 = lw._unequal_stats(a3, r3, b3)
        assert s3["max_frac_row"] == 1.0 and s3["frac_group"] == pytest.approx(1 / 3) and s3["n_unequal_elems"] == 4

    def test_record_accumulates_the_tolerance_fields(self):
        rec = lw.ShadowCompareRecord(enabled=True)
        rec.observe(key="k:B3", layer=1, cell="c", rows=3, site="block", equal=True, n_elems=300)
        rec.observe(key="k:B3", layer=2, cell="c", rows=3, site="block", equal=False, max_abs_diff=0.5,
                    max_ulps=1.0, max_abs_ref=3.0, n_unequal_elems=4, n_elems=300, max_ulps_edit=1.0,
                    max_frac_row=0.02, frac_group=4 / 300)
        rec.observe(key="k:B3", layer=3, cell="c", rows=3, site="block", equal=False, max_abs_diff=0.25,
                    max_ulps=2.0, max_abs_ref=1.0, n_unequal_elems=6, n_elems=300, max_ulps_edit=2.0,
                    max_frac_row=0.05, frac_group=6 / 300)
        rec.observe(key="k:B3", layer=4, cell="c", rows=3, site="block", equal=False, max_abs_diff=0.125,
                    note="modified flags differ: grouped=True, per_row=False", n_elems=300)
        rec.observe(key="e:B2", layer=1, cell="e", rows=2, site="block", equal=None)
        d = rec.as_dict()
        assert d["n_groups_compared"] == 4 and d["n_unequal"] == 3 and d["n_unmodified_groups"] == 1
        assert d["n_elems_compared"] == 1200 and d["n_unequal_elems"] == 10
        assert d["max_ulps_edit"] == 2.0 and d["max_ulps"] == 2.0 and d["max_abs_ref"] == 3.0
        assert d["max_frac_row"] == 0.05 and d["max_frac_group"] == pytest.approx(6 / 300)
        assert d["n_flag_disagreements"] == 1
        e = d["by_key"]["k:B3"]
        assert e["n_compared"] == 4 and e["n_elems_compared"] == 1200 and e["n_unequal_elems"] == 10
        assert e["max_ulps_edit"] == 2.0 and e["n_flag_disagreements"] == 1
        assert e["max_frac_row"] == 0.05 and e["max_frac_group"] == pytest.approx(6 / 300)
        fu = d["first_unequal"]
        assert fu["max_ulps_edit"] == 1.0 and fu["n_elems"] == 300 and fu["n_unequal_elems"] == 4
        assert fu["max_frac_row"] == 0.02 and fu["frac_group"] == pytest.approx(4 / 300)
        assert d["by_key"]["e:B2"]["n_unmodified"] == 1 and d["by_key"]["e:B2"]["n_elems_compared"] == 0
        # an unequal group recorded without the fields leaves them alone
        rec2 = lw.ShadowCompareRecord(enabled=True)
        rec2.observe(key="k", layer=1, cell="c", rows=1, site="block", equal=False, max_abs_diff=0.5)
        d2 = rec2.as_dict()
        assert d2["first_unequal"]["max_ulps_edit"] is None and d2["max_ulps_edit"] == 0.0
        assert d2["first_unequal"]["max_frac_row"] is None and d2["max_frac_row"] == 0.0
        assert d2["n_elems_compared"] == 0 and d2["n_flag_disagreements"] == 0

    def test_block_site_records_every_compared_element(self, monkeypatch, stacked):
        rec = _shadow_drive(monkeypatch)
        # cell-a 3 rows + cell-b 2 rows compared at 2 layers; the twin unmodified
        assert rec["n_groups_compared"] == 4 and rec["n_unequal"] == 0
        assert rec["n_elems_compared"] == 10 * HIDDEN and rec["n_unequal_elems"] == 0
        assert rec["max_ulps_edit"] == 0.0 and rec["n_flag_disagreements"] == 0
        assert rec["max_frac_row"] == 0.0 and rec["max_frac_group"] == 0.0
        a_key = next(k for k in rec["by_key"] if ":relu_gated_vector:" in k and k.endswith(":B3"))
        b_key = next(k for k in rec["by_key"] if ":add_vector:" in k and k.endswith(":B2"))
        assert rec["by_key"][a_key]["n_elems_compared"] == 6 * HIDDEN
        assert rec["by_key"][b_key]["n_elems_compared"] == 4 * HIDDEN

    def test_oproj_site_records_the_edit_unit_and_concentration(self):
        """The o_proj shadow passes the UNSTEERED output rows as the base, so
        an o_proj-site inequality carries max_ulps_edit and the concentration
        fields too (the helper's contract; the site is exercised end to end by
        test_lens_vectorized_apply TestShadowCompare.test_oproj_shadow)."""
        base = torch.full((3, HIDDEN), 1.0, dtype=torch.bfloat16)          # unsteered o_proj rows
        ref = base + 2.0                                                   # per-row output 3.0 (edit 2.0)
        app = ref.clone()
        app[1, 0] = 3.0 + 2 ** -6                                          # one element, 1 ulp at [2, 4)
        s = lw._unequal_stats(app, ref, base)
        assert s["n_unequal_elems"] == 1 and s["max_ulps"] == 1.0
        assert s["max_ulps_edit"] == 0.0                                   # one rounding of the sum: no excess
        assert s["max_frac_row"] == pytest.approx(1 / HIDDEN) and s["frac_group"] == pytest.approx(1 / (3 * HIDDEN))


# --------------------------------------------------------------------------
# 7. planted bugs fail the gate's tolerance by orders of magnitude
# --------------------------------------------------------------------------


def _shadow_drive(monkeypatch, *, plant=None, scale_hidden: float = 1.0):
    """Drive the mixed decode batch (cell-a relu-gated x3 rows, cell-b
    add_vector x2, the scale-0 twin, an unhooked request, a prefill) through a
    shadow-compare worker in the STACKED mode and return the record. ``plant``
    installs a bug in the GROUPED path only; ``scale_hidden`` scales the
    residual so a wrong-precision read has something to round."""
    monkeypatch.setattr(la, "_STACKED_REDUCTIONS", True)
    monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
    t = TestShadowCompare()
    a, b, zero = t._objs()
    st, view = _steering(_registry(a, b, zero, header=HEADER))
    assert st.apply_path == lw.APPLY_PATH_SHADOW
    if plant is not None:
        plant(monkeypatch, st)
    nat, pos, rows = _mixed_batch(view, a, b, zero, prefill_ids=(1, 2) + HEADER)
    h, r = _hr(nat)
    if scale_hidden != 1.0:
        h, r = h * scale_hidden, r * scale_hidden
    for layer in (1, 2):
        st.hook_body(layer, (pos, h, r), (h, r))
    return st.shadow_record.as_dict()


def _plant_at_row_reduction(bug: str):
    """A bug in the GROUPED reduction only: `_row_reduction` on a multi-row
    group (rowwise, > 1 row — the stacked call) returns a wrong result; the
    per-row reference slices (one row each) are untouched."""
    def plant(monkeypatch, st):
        real = la._row_reduction

        def bugged(x, fn, rowwise):
            if rowwise and x.dim() == 2 and x.shape[0] > 1:
                if bug == "sign":
                    return -fn(x)                              # the edit's sign flipped
                if bug == "shift":
                    return fn(x).roll(1, 0)                    # right values on the wrong rows
                if bug == "operand":
                    return fn(x.flip(-1))                      # the operand read permuted / transposed
                if bug == "dtype":
                    return fn(x.to(torch.bfloat16).float())    # the group read at bf16 precision
                raise AssertionError(bug)
            return real(x, fn, rowwise)
        monkeypatch.setattr(la, "_row_reduction", bugged)
    return plant


def _plant_row_fault(bug: str):
    """A ROW-LEVEL fault in the grouped reduction only: for the fixture's
    relu-gated op ``fn(x)`` is the gate pre-activation per row (the fixture's
    ``gate_b`` is 100), so row 0's edit is dropped by driving its gate below
    zero, halved by halving its post-bias gate, or swapped with row 1's."""
    def plant(monkeypatch, st):
        real = la._row_reduction

        def bugged(x, fn, rowwise):
            out = real(x, fn, rowwise)
            if rowwise and x.dim() == 2 and x.shape[0] > 1 and out.dim() == 1:
                out = out.clone()
                if bug == "drop_row":
                    out[0] = -1e9                                   # relu(g + b) -> 0: no edit on row 0
                elif bug == "halve_row":
                    out[0] = (out[0] + 100.0) / 2.0 - 100.0          # relu(g + b) halved on row 0
                elif bug == "swap_rows":
                    out[[0, 1]] = out[[1, 0]]                       # row 0 gets row 1's edit and vice versa
                else:
                    raise AssertionError(bug)
            return out
        monkeypatch.setattr(la, "_row_reduction", bugged)
    return plant


LINEAR_KEY = "lora_all:lora_oproj:rankK:response:skipbos0:both:B32"
PSR_KEY = "psr:relu_gated_vector:gemv:response:skipbos0:both:B32"
LOWRANK_KEY = "lowrank_gated:lowrank_relu_gated:rankK:all:skipbos0:both:B32"     # the registered low-rank kind's key
#: the gate stats a current worker files on an unequal PSR group of the measured class
GATE_OK = {"gate_n_groups_with_stats": 1, "gate_n_groups_observed": 1, "gate_max_diff_rel_sum": 0.8 * (2.0 ** -24),
           "gate_max_flip_rel_sum": None, "gate_n_relu_flips": 0, "gate_n_errors": 0,
           "gate_edit_consistency_max_ulps": 0.5, "gate_exact_rows_n": 30, "gate_exact_rows_unequal_elems": 0,
           "gate_lowrank_zero_delta_max_ulps_edit": None,
           "gate_n_groups_recon": 1, "gate_write_recon_mismatch_elems": 0}


def _synthetic_case(stats: dict, *, n_groups: int = 6000, rows: int = 32, width: int = 4096,
                    key: str = LINEAR_KEY, gate: dict = None) -> dict:
    """A shadow record twin: ``n_groups`` compared 32 x 4096 groups under one
    reduction key (a LINEAR one by default — the edit-ulp rule's class; pass
    ``key=PSR_KEY`` with ``gate=GATE_OK`` for the GATED class), exactly ONE of
    them unequal with the deviation ``stats`` (``lw._unequal_stats`` of a
    planted group) — so the case-wide fraction is a few e-6 whatever the
    row-level fault did."""
    n_elems = n_groups * rows * width
    e = {"n_compared": n_groups, "n_equal": n_groups - 1, "n_unequal": 1, "n_unmodified": 0,
         "n_shadow_errors": 0, "max_abs_diff": stats["max_abs_diff"], "max_ulps": stats.get("max_ulps", 0.0),
         "max_abs_ref": stats["max_abs_ref"], "n_unequal_elems": stats["n_unequal_elems"],
         "n_elems_compared": n_elems, "max_ulps_edit": stats.get("max_ulps_edit", 0.0),
         "max_frac_row": stats["max_frac_row"], "max_frac_group": stats["frac_group"],
         "n_flag_disagreements": 0, "first_unequal": None, **(gate or {})}
    return {"enabled": True, "n_groups_compared": n_groups, "n_equal": n_groups - 1, "n_unequal": 1,
            "n_unmodified_groups": 0, "n_shadow_errors": 0, "n_record_errors": 0, "n_keys_dropped": 0,
            "n_flag_disagreements": 0, "max_abs_diff": e["max_abs_diff"], "max_ulps": e["max_ulps"],
            "max_abs_ref": e["max_abs_ref"], "n_unequal_elems": e["n_unequal_elems"], "n_elems_compared": n_elems,
            "max_ulps_edit": e["max_ulps_edit"], "max_frac_row": e["max_frac_row"],
            "max_frac_group": e["max_frac_group"],
            "first_unequal": {"key": key, "note": "", "n_unequal_elems": e["n_unequal_elems"],
                              "max_ulps_edit": e["max_ulps_edit"], "max_frac_row": e["max_frac_row"],
                              "frac_group": e["max_frac_group"]},
            "by_key": {key: e}, "n_layer_calls_with_ge2_cells_compared": 1}


def _plant_elementwise_one_element(monkeypatch, st):
    """One element of cell-b's (add_vector) grouped result moved by 1.0."""
    orig = st._registry.apply

    def perturbed(*args, **kw):
        out = orig(*args, **kw)
        if not kw.get("legacy_masks") and isinstance(kw.get("ctx"), la.DecodeRows) and args[0] == "cell-b":
            out = out.clone()
            out[0, 0] = out[0, 0] + 1.0
        return out
    monkeypatch.setattr(st._registry, "apply", perturbed)


def _waves(rec):
    return [{"B": 3, "compared_in_wave": rec["n_groups_compared"], "unequal_in_wave": rec["n_unequal"]}]


class TestPlantedBugs:
    def test_unperturbed_stacked_run_is_exact_on_integers_and_passes(self, monkeypatch):
        rec = _shadow_drive(monkeypatch)
        assert rec["n_unequal"] == 0
        tol = g.reduction_tolerance(rec, reduction_mode=STACKED)
        assert tol["ok"] is True
        ok, detail = g.shadow_check(rec, (3,), routing=g.ROUTING_VECTORIZED, waves=_waves(rec),
                                    reduction_mode=STACKED)
        assert ok is True, detail

    @pytest.mark.parametrize("bug", ["sign", "shift", "operand", "dtype"])
    def test_planted_reduction_bug_fails_the_tolerance_by_orders_of_magnitude(self, monkeypatch, bug):
        # the precision bug needs values bf16 cannot hold: thirds (the integer
        # fixtures are exact in bf16, and x257 happened to cancel under the
        # integer gate weights)
        rec = _shadow_drive(monkeypatch, plant=_plant_at_row_reduction(bug),
                            scale_hidden=(1.0 / 3.0 if bug == "dtype" else 1.0))
        # only cell-a (the reduction op) is affected; both of its groups at both layers differ
        assert rec["n_unequal"] == 2 and rec["n_flag_disagreements"] == 0
        a_key = rec["first_unequal"]["key"]
        assert ":relu_gated_vector:" in a_key and a_key.endswith(":B3")
        assert g.shadow_key_op_class(a_key) == g.OP_CLASS_REDUCTION_GATED
        tol = g.reduction_tolerance(rec, reduction_mode=STACKED)
        assert tol["ok"] is False, tol["detail"]
        # the real worker filed OBSERVED gate stats on every unequal gated group, so this is a NUMERIC
        # fail, not missing evidence — and the observed grouped gate carries the planted corruption
        assert a_key in tol["gated_keys_unequal"] and tol["keys_lacking_gate_stats"] == []
        assert "fraction of unequal reduction-op elements" in tol["detail"]
        fu_gate = rec["first_unequal"]["gate"]
        assert fu_gate["gate_source"] == "observed" and fu_gate["gate_max_diff_rel_sum"] > g.REDUCTION_GATE_REL_SUM_TOL
        assert fu_gate["recomputed_max_diff_rel_sum"] == 0.0            # a recomputation cannot see the plant
        # the deviation is >= 1000x the bound: a defect, never kernel variance
        assert tol["frac_unequal_elems"] >= 0.1 >= 1000 * g.REDUCTION_FRAC_TOL
        assert "1e-04" in tol["detail"] or "1e-4" in tol["detail"] or str(g.REDUCTION_FRAC_TOL) in tol["detail"]
        ok, detail = g.shadow_check(rec, (3,), routing=g.ROUTING_VECTORIZED, waves=_waves(rec),
                                    reduction_mode=STACKED)
        assert ok is False and a_key.split(":B")[0] in detail or ok is False
        ok_mg, _ = g.shadow_multigroup_check(rec, reduction_mode=STACKED)
        assert ok_mg is False

    @pytest.mark.parametrize("bug", ["drop_row", "halve_row", "swap_rows"])
    def test_planted_row_fault_fails_through_the_row_bound_on_the_worker(self, monkeypatch, bug):
        """One row's edit dropped, halved, or swapped with another's, in the
        grouped path only: the real shadow record shows nearly every element
        of that row unequal (max_frac_row ~ 1 against the 2e-2 bound) and the
        tolerance fails naming the ROW bound."""
        rec = _shadow_drive(monkeypatch, plant=_plant_row_fault(bug))
        assert rec["n_unequal"] == 2 and rec["n_flag_disagreements"] == 0        # cell-a, both layers
        fu = rec["first_unequal"]
        assert ":relu_gated_vector:" in fu["key"] and fu["rows"] == 3
        # the fixture's steering vector has integer entries, some zero: those elements carry no
        # edit and stay equal, so "nearly every element of the row" is >= half the row here
        assert rec["max_frac_row"] >= 0.5 >= 10 * g.REDUCTION_ROW_FRAC_TOL
        expect_group = (2 / 3) if bug == "swap_rows" else (1 / 3)
        assert rec["max_frac_group"] >= expect_group / 2
        tol = g.reduction_tolerance(rec, reduction_mode=STACKED)
        assert tol["ok"] is False and "largest unequal fraction in one ROW" in tol["detail"]
        ok, detail = g.shadow_check(rec, (3,), routing=g.ROUTING_VECTORIZED, waves=_waves(rec),
                                    reduction_mode=STACKED)
        assert ok is False and "in one ROW" in detail

    @pytest.mark.parametrize("bug", ["drop_row", "halve_row", "swap_rows"])
    def test_row_fault_that_defeats_the_first_contract_still_fails(self, bug):
        """Review round 13's hole, reproduced and closed. A 32 x 4096 bf16 group
        with |output| ~ 1000 (bf16 ulp 4) and per-row edits of ONE output ulp
        (+4; -4 on row 6): one row's edit dropped, halved (2 rounds back to
        the unsteered value in bf16 — a halved one-ulp edit IS a dropped one)
        or two rows' edits swapped moves every affected element by <= 2
        OUTPUT ulps and, as the only unequal group of 6000, is 5e-6 to 1e-5
        of the case — inside the first form of the contract (case fraction
        <= 1e-4, <= 4 ulps of max(|output|, |edit|)). The concentration bounds
        fail it (row 1.0 > 2e-2, group 1/32 or 2/32 > 2e-3); the drop and the
        halving are invisible per element (max_ulps_edit 0: a one-ulp move of
        the output), the swap additionally fails the edit bound (an 8-unit
        difference is 2 output ulps but 128 ulps of the 4-unit edit)."""
        bf = torch.bfloat16
        base = torch.full((32, 4096), 1000.0, dtype=bf)
        edit = torch.full((32, 4096), 4.0, dtype=bf)                     # one output ulp
        edit[6] = -4.0
        ref = base + edit
        app = ref.clone()
        if bug == "drop_row":
            app[5] = base[5]                                              # stacked added nothing on row 5
        elif bug == "halve_row":
            app[5] = (base[5].float() + 2.0).to(bf)                       # half the edit: rounds to unsteered
        else:
            app[5], app[6] = ref[6].clone(), ref[5].clone()               # rows 5 and 6 got each other's edit
        # the first form of the contract: every element within 4 ulps of max(|output|, |edit|)
        diff = (app.float() - ref.float()).abs()
        old_unit = diff / la.ulp_spacing(torch.maximum(ref.float().abs(), edit.float().abs()), bf)
        assert float(old_unit.max()) <= 4.0
        stats = lw._unequal_stats(app, ref, base)
        snap = _synthetic_case(stats)
        tol = g.reduction_tolerance(snap, reduction_mode=STACKED)
        assert tol["frac_unequal_elems"] <= 2 / (6000 * 32) < g.REDUCTION_FRAC_TOL / 5   # the case fraction hides it
        assert tol["ok"] is False
        assert "largest unequal fraction in one ROW 1.00e+00 > 0.02" in tol["detail"]
        assert "largest unequal fraction in one GROUP" in tol["detail"] and "> 0.002" in tol["detail"]
        if bug == "swap_rows":
            assert stats["max_ulps_edit"] == 128.0 and "bf16 ulps of the per-row edit beyond one output ulp > 4" in tol["detail"]
        else:
            assert stats["max_ulps_edit"] == 0.0 and "> 4" not in tol["detail"]   # invisible per element
        assert stats["max_frac_row"] == 1.0 and stats["frac_group"] == pytest.approx((2 if bug == "swap_rows" else 1) / 32)
        ok, detail = g.shadow_check(snap, (32,), routing=g.ROUTING_VECTORIZED,
                                    waves=[{"B": 32, "compared_in_wave": 6000}], reduction_mode=STACKED)
        assert ok is False and "in one ROW" in detail
        ok_mg, detail_mg = g.shadow_multigroup_check(snap, reduction_mode=STACKED)
        assert ok_mg is False and "in one ROW" in detail_mg
        # the same group with kernel-variance-sized deviation (3 single flips in 3 rows) passes
        app2 = ref.clone()
        for r_ in (1, 9, 17):
            app2[r_, r_] = app2[r_, r_] + 4.0                             # one output ulp each
        good = _synthetic_case(lw._unequal_stats(app2, ref, base))
        assert g.reduction_tolerance(good, reduction_mode=STACKED)["ok"] is True
        # THE GATED CLASS (2026-09-06): the same row fault on a psr key WITH gate stats still fails
        # through the concentration bounds — they apply to both classes unchanged — while the edit
        # unit is informational there (the swap's 128 edit ulps no longer names "> 4")
        snap_g = _synthetic_case(stats, key=PSR_KEY, gate=GATE_OK)
        tol_g = g.reduction_tolerance(snap_g, reduction_mode=STACKED)
        assert tol_g["ok"] is False and "largest unequal fraction in one ROW 1.00e+00 > 0.02" in tol_g["detail"]
        assert "largest unequal fraction in one GROUP" in tol_g["detail"]
        assert "> 4" not in tol_g["detail"] and "> 16 x 2^-24" not in tol_g["detail"]
        assert tol_g["max_ulps_edit_gated"] == stats["max_ulps_edit"] and tol_g["max_ulps_edit_linear"] is None
        assert g.reduction_tolerance(_synthetic_case(lw._unequal_stats(app2, ref, base), key=PSR_KEY, gate=GATE_OK),
                                     reduction_mode=STACKED)["ok"] is True
        # ... and the gated key WITHOUT gate stats is missing evidence, whatever the numbers say
        tol_ng = g.reduction_tolerance(_synthetic_case(lw._unequal_stats(app2, ref, base), key=PSR_KEY),
                                       reduction_mode=STACKED)
        assert tol_ng["ok"] is False and "lacks gate stats" in tol_ng["detail"]

    def test_one_moved_element_of_an_elementwise_op_fails_exactly(self, monkeypatch):
        rec = _shadow_drive(monkeypatch, plant=_plant_elementwise_one_element)
        assert rec["n_unequal"] == 2 and rec["n_unequal_elems"] == 2           # one element per layer
        b_key = rec["first_unequal"]["key"]
        assert ":add_vector:" in b_key and g.shadow_key_op_class(b_key) == "elementwise"
        tol = g.reduction_tolerance(rec, reduction_mode=STACKED)
        assert tol["ok"] is False and b_key in tol["elementwise_keys_unequal"]
        ok, detail = g.shadow_check(rec, (3,), routing=g.ROUTING_VECTORIZED, waves=_waves(rec),
                                    reduction_mode=STACKED)
        assert ok is False and "exact" in detail.lower()

    def test_measured_class_passes_stacked_and_fails_per_row(self):
        """The nemotron3nano4b A/B (388415) shadow table's B=32 key: 3747
        unequal elements over 7812 x 32 x 3136 = 7.84e8 compared (frac 4.8e-6),
        every flip one rounding of the edit — inside the stacked contract; the
        same record under the per-row mode (which promises exactness) is a
        FAIL, and so is the record with a 4.5-ulp element."""
        key = "psr:relu_gated_vector:gemv:response:skipbos0:both:B32"
        row, grp = 2 / 3136, 3 / (32 * 3136)          # 2 flips in one row, 3 in one group: the measured class
        snap = {"enabled": True, "n_groups_compared": 7812, "n_unequal": 1809, "n_unmodified_groups": 0,
                "n_shadow_errors": 0, "n_record_errors": 0, "n_flag_disagreements": 0, "n_keys_dropped": 0,
                "n_unequal_elems": 3747, "n_elems_compared": 784_000_000, "max_ulps_edit": 1.0,
                "max_frac_row": row, "max_frac_group": grp,
                "max_ulps": 128.0, "max_abs_ref": 992.0, "max_abs_diff": 0.25,
                "first_unequal": {"key": key, "note": "", "n_unequal_elems": 3, "max_ulps_edit": 1.0,
                                  "max_frac_row": row, "frac_group": grp},
                "by_key": {key: {"n_compared": 7812, "n_unequal": 1809, "n_unmodified": 0, "n_shadow_errors": 0,
                                 "n_unequal_elems": 3747, "n_elems_compared": 784_000_000,
                                 "max_ulps_edit": 1.0, "max_frac_row": row, "max_frac_group": grp,
                                 "max_ulps": 128.0, "max_abs_ref": 992.0,
                                 "max_abs_diff": 0.25, "n_flag_disagreements": 0, "first_unequal": None,
                                 # the gated class's evidence (2026-09-06): stats on every unequal group
                                 "gate_n_groups_with_stats": 1809, "gate_n_groups_observed": 1809,
                                 "gate_max_diff_rel_sum": 0.6 * g.REDUCTION_GATE_UNIT,
                                 "gate_max_flip_rel_sum": None, "gate_n_relu_flips": 0, "gate_n_errors": 0,
                                 "gate_edit_consistency_max_ulps": 0.9, "gate_exact_rows_n": 1809 * 29,
                                 "gate_exact_rows_unequal_elems": 0, "gate_lowrank_zero_delta_max_ulps_edit": None,
                                 "gate_n_groups_recon": 1809, "gate_write_recon_mismatch_elems": 0}}}
        tol = g.reduction_tolerance(snap, reduction_mode=STACKED)
        assert tol["frac_unequal_elems"] == pytest.approx(3747 / 784_000_000)
        assert tol["ok"] is True and tol["max_frac_row"] == row and tol["max_frac_group"] == grp
        assert g.reduction_tolerance(snap, reduction_mode=PER_ROW)["ok"] is False

        def variant(**over):
            v = dict(snap, **over)
            v["by_key"] = {key: dict(snap["by_key"][key], **over)}
            return g.reduction_tolerance(v, reduction_mode=STACKED)
        # a GATED key: the edit unit is recorded and reported, not a rule (2026-09-06) ...
        assert variant(max_ulps_edit=4.5)["ok"] is True
        assert "edit-ulps 4.5 recorded, not a rule" in variant(max_ulps_edit=4.5)["detail"]
        # ... the gate bound is
        over = variant(gate_max_diff_rel_sum=20 * g.REDUCTION_GATE_UNIT)
        assert over["ok"] is False and "20 x 2^-24 > 16 x 2^-24" in over["detail"]
        # ... and so is the gate-edit identity
        broken = variant(gate_edit_consistency_max_ulps=4.5)
        assert broken["ok"] is False and "gate-edit identity of a GATED reduction op breaks by 4.5" in broken["detail"]
        # ... a single unequal element where the gates agreed ...
        inexact = variant(gate_exact_rows_unequal_elems=1)
        assert inexact["ok"] is False and "in rows whose observed gates are bit-identical or masked off" in inexact["detail"]
        # ... and one group whose gate stats were not observed
        unsourced = variant(gate_n_groups_observed=1808)
        assert unsourced["ok"] is False and "gate_n_groups_observed == n_unequal" in unsourced["detail"]
        # ... and one element that is not the bit-exact reconstruction from the observed gate (round 4 P1)
        recon = variant(gate_write_recon_mismatch_elems=1)
        assert recon["ok"] is False and "not the bit-exact reconstruction base + eff * g * v" in recon["detail"]
        assert variant(gate_n_groups_recon=1808)["ok"] is False                       # a vector-gated key not fully rebuilt
        assert variant(max_frac_row=0.5)["ok"] is False and "in one ROW" in variant(max_frac_row=0.5)["detail"]
        assert variant(max_frac_group=0.01)["ok"] is False and "in one GROUP" in variant(max_frac_group=0.01)["detail"]


# --------------------------------------------------------------------------
# 8. plumbing: the gate's env, the A/B sides, the microbench
# --------------------------------------------------------------------------


class TestPlumbing:
    def test_gate_standup_env_sets_or_scrubs_the_mode(self):
        base = {"PATH": "/bin", "PYTHONPATH": "/cand/src", g.STACKED_ENV: "1", g.SHADOW_ENV: "1"}
        env = g.standup_env(base, Path("/cand"), batch_invariant=0)
        assert g.STACKED_ENV not in env and g.SHADOW_ENV not in env      # the reference: scrubbed
        on = g.standup_env(base, Path("/cand"), batch_invariant=0, reduction_mode=STACKED)
        assert on[g.STACKED_ENV] == "1" and g.SHADOW_ENV not in on
        off = g.standup_env(base, Path("/cand"), batch_invariant=0, reduction_mode=PER_ROW)
        assert off[g.STACKED_ENV] == "0"
        # the A/B driver's sides set the env explicitly for BOTH modes
        pr = g.standup_env(base, Path("/cand"), batch_invariant=0, env_extra=ab.SIDES[ab.PER_ROW][0])
        assert pr[g.STACKED_ENV] == "0" and g.SHADOW_ENV not in pr
        st = g.standup_env(base, Path("/cand"), batch_invariant=0, env_extra=ab.SIDES[ab.STACKED][0])
        assert st[g.STACKED_ENV] == "1"
        both = g.standup_env(base, Path("/cand"), batch_invariant=0, env_extra=ab.SIDES[ab.STACKED_SHADOW][0])
        assert both[g.STACKED_ENV] == "1" and both[g.SHADOW_ENV] == "1"
        assert g.STACKED_ENV == la.STACKED_REDUCTIONS_ENV == ab.STACKED_ENV
        for side, (extra, path) in ab.SIDES.items():
            assert la.parse_reduction_mode(extra[g.STACKED_ENV]) in path, side

    def test_microbench_cpu_schema_and_exact_profile(self):
        before = la.reduction_mode()
        recs = mb.run_microbench(models=("tiny",), batches=(1, 2, 3), profiles=("ints",), iters=2, warmup=1,
                                 trials=2, device="cpu")
        kinds = {(r["kind"], r["oproj_shape"]) for r in recs}
        assert kinds == {(k, None if s == la.HOOK_SITE_BLOCK else "attn") for k, s in mb.KINDS}
        assert len(recs) == len(mb.KINDS) * 3
        for r in recs:
            assert set(r["timing"]) == set(mb.ARMS)
            for arm in mb.ARMS:
                assert r["timing"][arm]["median_us"] >= 0 and r["timing"][arm]["n"] == 2
                assert r["timing"][arm]["ms_per_step_all_layers"] == pytest.approx(
                    r["timing"][arm]["median_us"] * r["n_layers"] / 1000.0)
            s = r["deviation_summary"]
            assert s["trials"] == 2 and s["n_unequal"] == 0 and s["max_ulps"] == 0.0     # exact arithmetic
            assert s["legacy_equals_per_row_all_trials"] is True
            if r["kind"] == "psr_gemv":
                assert s["gate_scalar"]["frac_rows_gate_unequal"] == 0.0
        # the bench leaves the process mode as it found it (the default)
        assert la.reduction_mode() == before == STACKED
        md = mb.render_markdown(recs)
        assert md.count("\n") == len(recs) + 1 and "| tiny | psr_gemv |" in md

    def test_microbench_deviation_reads_ulps(self):
        pr = torch.tensor([[1.0, 2.0, 0.0, 4.0]], dtype=torch.bfloat16)
        st = torch.tensor([[1.0 + 2 ** -7, 2.0 + 2 ** -5, 0.0, 4.0]], dtype=torch.bfloat16)
        d = mb.deviation(st, pr)
        assert d["n_unequal"] == 2 and d["n_rows_unequal"] == 1 and d["max_ulps"] == 2.0
        assert d["ulps_hist"] == {"<=1": 1, "<=2": 1, ">2": 0} and d["max_abs_diff"] == 2 ** -5
        assert mb.deviation(pr, pr)["n_unequal"] == 0

    def test_ab_agreement_and_pool(self):
        def sig(ids, lps):
            return {"token_ids": list(ids), "logprobs": list(lps), "top": [[("a", -0.1)]] * len(ids)}
        run_a = [sig((1, 2, 3), (-0.1, -0.2, -0.3)), sig((4, 5, 6), (-0.1, -0.2, -0.3)), sig((7, 8, 9), (-0.5, -0.5, -0.5))]
        run_b = [sig((1, 2, 3), (-0.1, -0.2, -0.3)), sig((4, 5, 7), (-0.1, -0.2, -0.9)), sig((7, 8, 9), (-0.5, -0.5, -0.6))]
        a = ab.agreement(run_a, run_b)
        assert a["n"] == 3 and a["same_tokens"] == 2 and a["same_all"] == 1 and a["first_div"] == [2, ("lp", 2)]
        p = ab.pool([a, ab.agreement(run_a, run_a)])
        assert p["requests"] == 6 and p["same_all"] == 4 and p["same_tokens"] == 5
        assert p["frac_same_all"] == pytest.approx(4 / 6) and p["median_first_token_div"] == 2
        assert ab.pairwise([run_a, run_b, []]) == [a]
        assert len(ab.cross([run_a, run_b], [run_a])) == 2
        with pytest.raises(g.GateError):
            ab.agreement(run_a, run_a[:2])

    def test_ab_shadow_rows_table(self):
        snap = {"by_key": {"psr:relu_gated_vector:gemv:response:skipbos0:both:B32":
                           {"n_compared": 10, "n_unequal": 3, "n_unmodified": 0, "max_abs_diff": 0.0156,
                            "max_ulps": 1.0, "max_abs_ref": 3.5, "n_unequal_elems": 7},
                           "psr:relu_gated_vector:gemv:response:skipbos0:both:B8":
                           {"n_compared": 4, "n_unequal": 0, "n_unmodified": 0, "max_abs_diff": 0.0}}}
        rows = ab.shadow_rows_table(snap)
        assert [r["rows"] for r in rows] == [8, 32] and rows[1]["max_ulps"] == 1.0 and rows[0]["max_ulps"] is None
        assert ab.shadow_rows_table({"rpc": "absent"}) == [] and ab.shadow_rows_table(None) == []
        doc = {"model": {"key": "m", "hf": "h", "rev": "r" * 12}, "tree": "/t", "sha": "s" * 12,
               "geometry": {"max_model_len": 1, "max_num_seqs": 1}, "batches": [8], "repeats": 2, "max_tokens": 64,
               "ignore_eos": True, "timing_batches": [8],
               "cases": [{"case_id": "c", "objects": "synthetic", "cell": "x",
                          "apply_path": {s: ab.SIDES[s][1] for s in ab.SIDES}, "b1_sanity": {"equal": True},
                          "comparisons": {"8": {n: ab.pool([]) for n in ab.COMPARISONS}},
                          "shadow": {"n_groups_compared": 1}, "shadow_rows": rows,
                          "timing": {ab.PER_ROW: {"8": {"step_ms": 1.0}}, ab.STACKED: {"8": {"step_ms": 0.5}}}},
                         {"case_id": "d", "error": "boom"}]}
        md = ab.render_report(doc)
        assert "| 8 | stacked vs per_row |" in md and "ERROR: boom" in md and "| per_row | 1.0 |" in md


# --------------------------------------------------------------------------
# 6. WHERE the unit peaks: worst_elem / gate-scalar evidence (2026-09-05)
# --------------------------------------------------------------------------


class TestShadowEvidenceFields:
    """The fields added so a tolerance violation can be READ, not just counted:
    ``worst_elem`` (the peak element and its row's edit ratio / cosine) from
    ``_unequal_stats``; ``_gate_stats`` (the gated op's gate scalar both ways);
    and the record's ``worst_unequal`` + gate summary. None of them is a rule."""

    def test_worst_elem_locates_the_peak_and_reads_its_row(self):
        bf = torch.bfloat16
        base = torch.zeros(3, 4, dtype=bf)
        edit = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=bf)
        ref = base + edit                                      # every row's per-row edit = (1, 2, 4, 8)
        # a gate perturbation: row 1's whole edit scaled 1.125x (exact in bf16) — uniform, cosine 1
        app = ref.clone(); app[1] = edit * 1.125
        s = lw._unequal_stats(app, ref, base)
        w = s["worst_elem"]
        assert w["row"] == 1 and w["row_n_unequal"] == 4 and w["row_n_elems"] == 4
        assert w["row_edit_ratio"] == pytest.approx(1.125, rel=1e-3)
        assert w["row_edit_cos"] == pytest.approx(1.0, abs=1e-6)
        assert w["ulps_edit"] == s["max_ulps_edit"] and w["ulps_edit"] > 4.0      # 15 edit ulps: a violation
        assert w["edit_applied"] == pytest.approx(w["edit_ref"] * 1.125, rel=1e-3)
        assert w["base"] == 0.0 and w["ref"] == w["edit_ref"] and w["abs_diff"] == pytest.approx(w["edit_ref"] * 0.125)
        assert w["ulps_out"] is not None and w["row_edit_norm"] == pytest.approx(float(edit.float().norm()))
        # a MISPLACED edit: row 2 receives another direction — same norm, cosine well below 1
        app2 = ref.clone(); app2[2] = torch.tensor([8.0, 4.0, 2.0, 1.0], dtype=bf)
        w2 = lw._unequal_stats(app2, ref, base)["worst_elem"]
        assert w2["row"] == 2 and w2["row_edit_ratio"] == pytest.approx(1.0, rel=1e-3) and w2["row_edit_cos"] < 0.8
        # a DROPPED edit: row 0 back at base — ratio 0, no cosine
        app3 = ref.clone(); app3[0] = base[0]
        w3 = lw._unequal_stats(app3, ref, base)["worst_elem"]
        assert w3["row"] == 0 and w3["row_edit_ratio"] == 0.0 and w3["row_edit_cos"] is None
        assert w3["applied"] == 0.0 and w3["ulps_edit"] == s["max_ulps_edit"] or w3["ulps_edit"] > 60
        # without base rows there is no edit, hence no worst element
        assert "worst_elem" not in lw._unequal_stats(app, ref)
        # a 1-D group (one row) still reads
        w1 = lw._unequal_stats(app[1], ref[1], base[1])["worst_elem"]
        assert w1["row"] == 0 and w1["row_n_elems"] == 4 and w1["row_edit_ratio"] == pytest.approx(1.125, rel=1e-3)

    def test_gate_stats_reads_the_gate_both_ways(self, monkeypatch):
        d = 8
        w = torch.tensor([1.0, -1.0, 0.5, -0.5, 0.25, -0.25, 2.0, -2.0])
        op = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=torch.ones(d), gate_w=w,
                       gate_b=torch.tensor([0.0]))
        rows = torch.tensor([[1.0] * d, [0.5] * d, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                            dtype=torch.bfloat16)
        # per-row gates, exact: rows 0 and 1 sum w to 0 -> ReLU 0; row 2 -> 2.0
        real = la._matmul

        def perturbed(x, y):   # the STACKED gate call (M > 1 against gate_w) lands 2^-10 high
            out = real(x, y)
            return out + 2 ** -10 if (x.dim() == 2 and x.shape[0] > 1 and y is w) else out
        monkeypatch.setattr(la, "_matmul", perturbed)
        g = lw._gate_stats([op], rows)
        assert g["gate_ops"] == [la.OP_RELU_GATED_VECTOR] and g["gate_n_rows"] == 3
        # rows 0 / 1: per-row 0, stacked 2^-10 -> two ReLU flips; row 2: 2.0 vs 2.0 + 2^-10
        assert g["gate_n_rows_unequal"] == 3 and g["gate_n_relu_flips"] == 2
        assert g["gate_max_abs_diff"] == pytest.approx(2 ** -10)
        assert g["gate_max_rel_diff"] == pytest.approx(1.0)              # a flipped gate is entirely new
        assert g["gate_min_abs_at_diff"] == pytest.approx(2 ** -10)      # the smallest gate that differed
        assert g["gate_max_abs_ref"] == pytest.approx(2.0)
        # |delta| over sum|h_i w_i|: rows 0 / 1 / 2 have sum|hw| = 7.5 / 3.75 / 2.0 — row 2 carries the max
        assert g["gate_max_diff_rel_sum"] == pytest.approx(2 ** -10 / 2.0, rel=1e-4)
        # THE FLIP RULE'S NUMBER (2026-09-06): the fired side (stacked, 2^-10) over sum|hw| — rows 0 / 1
        # flipped with sum|hw| 7.5 / 3.75, so row 1 carries the max; one gate element per row here
        assert g["gate_max_flip_rel_sum"] == pytest.approx(2 ** -10 / 3.75, rel=1e-4)
        assert g["gate_n_elems"] == 3
        assert len(g["gate_rows_unequal_sample"]) == 3
        assert g["gate_rows_unequal_sample"][2][:2] == pytest.approx([2.0, 2.0 + 2 ** -10])
        assert g["gate_rows_unequal_sample"][0][2] == pytest.approx(7.5)  # sum|hw| of row 0
        # the linear-gated op keeps the negative half: no ReLU, no flips, still a deviation
        lin = la.HookOp(op=la.OP_LINEAR_GATED_VECTOR, layer=0, vector=torch.ones(d), gate_w=w,
                        gate_b=torch.tensor([0.0]))
        gl = lw._gate_stats([lin], rows)
        assert gl["gate_n_relu_flips"] == 0 and gl["gate_n_rows_unequal"] == 3
        assert gl["gate_max_flip_rel_sum"] is None                            # no ReLU, no flip, no ratio
        # no gated op, a one-row group, or 3-D rows: nothing to read
        vec = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(d))
        assert lw._gate_stats([vec], rows) == {} and lw._gate_stats([op], rows[:1]) == {}
        assert lw._gate_stats([op], rows.unsqueeze(1)) == {}
        # with the real matmul on CPU both ways agree: an empty deviation, not an error
        monkeypatch.setattr(la, "_matmul", real)
        g0 = lw._gate_stats([op], rows)
        assert g0["gate_n_rows_unequal"] == 0 and g0["gate_min_abs_at_diff"] is None and "gate_error" not in g0
        assert g0["gate_max_diff_rel_sum"] == 0.0 and g0["gate_max_flip_rel_sum"] is None
        assert g0["gate_source"] == "recomputed" == g["gate_source"]        # a cross-check, never the evidence

    def test_gate_stats_reads_the_lowrank_gate_per_latent(self, monkeypatch):
        """The low-rank gated pair (2026-09-06): the gate is the k-vector of
        latents ``h @ A^T`` per row, the reduction magnitude ``|h| @ |A|^T``
        per latent, and every ratio is per (row, latent) element."""
        d, k = 8, 2
        a = torch.tensor([[1.0, -1.0] * 4, [1.0] * 8])                    # latent 0 cancels, latent 1 sums
        op = la.HookOp(op=la.OP_LOWRANK_RELU_GATED, layer=0, lowrank_a=a, lowrank_b=torch.ones(d, k))
        rows = torch.tensor([[1.0] * d, [0.5] * d, [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
                            dtype=torch.bfloat16)
        # per-row latents, exact: rows 0 / 1 -> (0, 8) / (0, 4); row 2 -> (2, 2)
        real = la._matmul

        def perturbed(x, y):   # the STACKED read through A (M > 1 against A's storage) lands latent 0 2^-10 high
            out = real(x, y)
            if x.dim() == 2 and x.shape[0] > 1 and y.data_ptr() == a.data_ptr():
                out = out + torch.tensor([2 ** -10, 0.0])
            return out
        monkeypatch.setattr(la, "_matmul", perturbed)
        gs = lw._gate_stats([op], rows)
        assert gs["gate_ops"] == [la.OP_LOWRANK_RELU_GATED] and gs["gate_n_rows"] == 3 and gs["gate_n_elems"] == 6
        # latent 0 differs on every row (3 elements); latent 1 is untouched
        assert gs["gate_n_rows_unequal"] == 3 and gs["gate_n_relu_flips"] == 2       # rows 0 / 1: 0 -> 2^-10
        assert gs["gate_max_abs_diff"] == pytest.approx(2 ** -10)
        assert gs["gate_max_abs_ref"] == pytest.approx(8.0)
        # |delta| / sum|h a_j|: latent 0's sum|hw| is 8 / 4 / 2 on rows 0 / 1 / 2 — row 2 carries the max
        assert gs["gate_max_diff_rel_sum"] == pytest.approx(2 ** -10 / 2.0, rel=1e-4)
        # the fired side over sum|hw| on the two flipped elements: 2^-10 / 8 and 2^-10 / 4
        assert gs["gate_max_flip_rel_sum"] == pytest.approx(2 ** -10 / 4.0, rel=1e-4)
        assert gs["gate_min_abs_at_diff"] == pytest.approx(2 ** -10)
        assert gs["gate_rows_unequal_sample"][2][:2] == pytest.approx([2.0, 2.0 + 2 ** -10])
        # the linear low-rank gate keeps the negative half: same deviation, no flips
        lin = la.HookOp(op=la.OP_LOWRANK_LINEAR_GATED, layer=0, lowrank_a=a, lowrank_b=torch.ones(d, k))
        gl = lw._gate_stats([lin], rows)
        assert gl["gate_n_rows_unequal"] == 3 and gl["gate_n_relu_flips"] == 0 and gl["gate_max_flip_rel_sum"] is None
        # a group serving both gated pairs sums the elements and lists both ops
        vec = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=torch.ones(d), gate_w=torch.ones(d),
                        gate_b=torch.tensor([0.0]))
        both = lw._gate_stats([vec, op], rows)
        assert both["gate_ops"] == [la.OP_RELU_GATED_VECTOR, la.OP_LOWRANK_RELU_GATED] and both["gate_n_elems"] == 9
        # a raise inside is returned as gate_error (the gate FAILS it as missing evidence)
        monkeypatch.setattr(la, "_matmul", lambda x, y: (_ for _ in ()).throw(RuntimeError("boom")))
        assert lw._gate_stats([op], rows) == {"gate_error": "RuntimeError: boom"}

    def test_record_tracks_the_worst_unequal_and_the_gate(self):
        rec = lw.ShadowCompareRecord(enabled=True)
        g1 = {"gate_source": "observed", "gate_n_rows_unequal": 2, "gate_n_relu_flips": 0, "gate_max_abs_diff": 1e-3,
              "gate_max_rel_diff": 0.01, "gate_min_abs_at_diff": 0.1,
              "gate_max_diff_rel_sum": 1e-8, "gate_max_flip_rel_sum": None, "gate_edit_consistency_max_ulps": 0.5,
              "gate_exact_rows_n": 1, "gate_exact_rows_unequal_elems": 0, "gate_lowrank_zero_delta_max_ulps_edit": None,
              "gate_write_recon_elems": 24, "gate_write_recon_mismatch_elems": 0}
        g2 = {"gate_source": "observed", "gate_n_rows_unequal": 1, "gate_n_relu_flips": 1, "gate_max_abs_diff": 2e-3,
              "gate_max_rel_diff": 0.2, "gate_min_abs_at_diff": 0.01,
              "gate_max_diff_rel_sum": 3e-8, "gate_max_flip_rel_sum": 2e-8, "gate_edit_consistency_max_ulps": 1.5,
              "gate_exact_rows_n": 2, "gate_exact_rows_unequal_elems": 0, "gate_lowrank_zero_delta_max_ulps_edit": 0.5,
              "gate_write_recon_elems": 24, "gate_write_recon_mismatch_elems": 2}
        common = dict(key="k:B3", cell="c", rows=3, site="block", equal=False, n_unequal_elems=1,
                      n_elems=30, max_frac_row=0.1, frac_group=1 / 30)
        rec.observe(layer=1, max_abs_diff=0.5, max_ulps_edit=1.0, worst_elem={"row": 0, "col": 1}, gate=g1, **common)
        rec.observe(layer=2, max_abs_diff=0.25, max_ulps_edit=5.0, worst_elem={"row": 2, "col": 0}, gate=g2, **common)
        rec.observe(layer=3, max_abs_diff=0.25, max_ulps_edit=2.0, **common)     # no evidence sub-records
        d = rec.as_dict()
        assert d["first_unequal"]["layer"] == 1 and d["worst_unequal"]["layer"] == 2
        assert d["worst_unequal"]["worst_elem"] == {"row": 2, "col": 0} and d["worst_unequal"]["gate"] == g2
        assert d["first_unequal"]["gate"] == g1 and d["first_unequal"]["worst_elem"] == {"row": 0, "col": 1}
        e = d["by_key"]["k:B3"]
        assert e["first_unequal"]["layer"] == 1 and e["worst_unequal"]["layer"] == 2
        assert d["gate_n_rows_unequal"] == 3 and d["gate_n_relu_flips"] == 1
        assert d["gate_max_abs_diff"] == 2e-3 and d["gate_max_rel_diff"] == 0.2 and d["gate_min_abs_at_diff"] == 0.01
        assert e["gate_max_rel_diff"] == 0.2 and e["gate_n_relu_flips"] == 1 and e["gate_n_rows_unequal"] == 3
        # THE GATED RULE'S INPUTS (2026-09-06), case-wide and per key: the maxima of the two ratios (the
        # flip ratio None until a flip is seen), how many unequal groups carried stats, and gate errors
        assert d["gate_max_diff_rel_sum"] == 3e-8 and d["gate_max_flip_rel_sum"] == 2e-8
        assert d["gate_n_groups_with_stats"] == 2 and d["gate_n_errors"] == 0 and d["first_gate_error"] is None
        assert e["gate_max_diff_rel_sum"] == 3e-8 and e["gate_max_flip_rel_sum"] == 2e-8
        assert e["gate_n_groups_with_stats"] == 2 and e["gate_n_errors"] == 0
        assert d["gate_edit_consistency_max_ulps"] == 1.5 == e["gate_edit_consistency_max_ulps"]
        # provenance and exactness: observed groups counted, exact rows summed
        assert d["gate_n_groups_observed"] == 2 == e["gate_n_groups_observed"]
        assert d["gate_exact_rows_n"] == 3 == e["gate_exact_rows_n"]
        assert d["gate_exact_rows_unequal_elems"] == 0 == e["gate_exact_rows_unequal_elems"]
        assert d["gate_lowrank_zero_delta_max_ulps_edit"] == 0.5 == e["gate_lowrank_zero_delta_max_ulps_edit"]
        # write reconstruction (round 4 P1): groups rebuilt counted, mismatches summed (None = not rebuilt)
        assert d["gate_n_groups_recon"] == 2 == e["gate_n_groups_recon"]
        assert d["gate_write_recon_mismatch_elems"] == 2 == e["gate_write_recon_mismatch_elems"]
        # a tie keeps the earlier group; a later, larger peak replaces it
        rec.observe(layer=4, max_abs_diff=0.25, max_ulps_edit=5.0, **common)
        assert rec.as_dict()["worst_unequal"]["layer"] == 2
        rec.observe(layer=5, max_abs_diff=0.25, max_ulps_edit=6.0, **common)
        assert rec.as_dict()["worst_unequal"]["layer"] == 5 and rec.as_dict()["by_key"]["k:B3"]["worst_unequal"]["layer"] == 5
        # a gate_error is counted, named, and is NOT a group with stats
        rec.observe(layer=6, max_abs_diff=0.25, max_ulps_edit=0.5, gate={"gate_error": "RuntimeError: boom"}, **common)
        d = rec.as_dict()
        assert d["gate_n_errors"] == 1 and d["gate_n_groups_with_stats"] == 2 and d["gate_n_groups_observed"] == 2
        assert d["first_gate_error"] == {"key": "k:B3", "layer": 6, "cell": "c", "error": "RuntimeError: boom"}
        assert d["by_key"]["k:B3"]["gate_n_errors"] == 1 and d["by_key"]["k:B3"]["gate_n_groups_with_stats"] == 2
        assert d["worst_unequal"]["layer"] == 5 and d["gate_max_diff_rel_sum"] == 3e-8
        # a record whose gated groups never flipped keeps the flip ratio None (the gate reads "no flips")
        rec3 = lw.ShadowCompareRecord(enabled=True)
        rec3.observe(layer=1, max_abs_diff=0.5, max_ulps_edit=1.0, gate=g1, **common)
        rec3.observe(layer=2, max_abs_diff=0.5, max_ulps_edit=1.0, gate=dict(g1, gate_max_diff_rel_sum=2e-8), **common)
        d3 = rec3.as_dict()
        assert d3["gate_max_flip_rel_sum"] is None and d3["gate_max_diff_rel_sum"] == 2e-8 and d3["gate_n_groups_with_stats"] == 2
        assert d3["gate_edit_consistency_max_ulps"] == 0.5
        assert d3["by_key"]["k:B3"]["gate_max_flip_rel_sum"] is None and d3["by_key"]["k:B3"]["gate_n_groups_with_stats"] == 2
        # the payload stays JSON (the gate reads it over the RPC)
        json.dumps(rec.as_dict(), allow_nan=False)
        # an unequal group without the edit unit never becomes the worst; the fields read None
        rec2 = lw.ShadowCompareRecord(enabled=True)
        rec2.observe(key="k", layer=1, cell="c", rows=1, site="block", equal=False, max_abs_diff=0.5)
        d2 = rec2.as_dict()
        assert d2["worst_unequal"] is None and d2["first_unequal"]["worst_elem"] is None
        assert d2["first_unequal"]["gate"] is None and d2["gate_min_abs_at_diff"] is None
        assert d2["by_key"]["k"]["worst_unequal"] is None
        assert d2["gate_max_diff_rel_sum"] == 0.0 and d2["gate_max_flip_rel_sum"] is None
        assert d2["gate_n_groups_with_stats"] == 0 and d2["gate_n_errors"] == 0
        assert d2["by_key"]["k"]["gate_n_groups_with_stats"] == 0 and d2["by_key"]["k"]["gate_max_flip_rel_sum"] is None
        assert d2["gate_edit_consistency_max_ulps"] is None and d2["by_key"]["k"]["gate_edit_consistency_max_ulps"] is None
        assert d2["gate_n_groups_observed"] == 0 == d2["by_key"]["k"]["gate_n_groups_observed"]
        assert d2["gate_exact_rows_unequal_elems"] == 0 and d2["gate_lowrank_zero_delta_max_ulps_edit"] is None
        assert d2["gate_n_groups_recon"] == 0 and d2["gate_write_recon_mismatch_elems"] == 0
        # a group not rebuilt (None) counts in neither
        rec5 = lw.ShadowCompareRecord(enabled=True)
        rec5.observe(layer=1, max_abs_diff=0.5, max_ulps_edit=1.0,
                     gate=dict(g1, gate_write_recon_mismatch_elems=None, gate_write_recon_skipped="x"), **common)
        assert rec5.as_dict()["gate_n_groups_recon"] == 0 and rec5.as_dict()["gate_n_groups_observed"] == 1
        # an unsourced or recomputed gate record has stats but is NOT observed
        rec4 = lw.ShadowCompareRecord(enabled=True)
        rec4.observe(layer=1, max_abs_diff=0.5, max_ulps_edit=1.0, gate={k_: v_ for k_, v_ in g1.items() if k_ != "gate_source"}, **common)
        rec4.observe(layer=2, max_abs_diff=0.5, max_ulps_edit=1.0, gate=dict(g1, gate_source="recomputed"), **common)
        d4 = rec4.as_dict()
        assert d4["gate_n_groups_with_stats"] == 2 and d4["gate_n_groups_observed"] == 0
        assert d4["by_key"]["k:B3"]["gate_n_groups_with_stats"] == 2 and d4["by_key"]["k:B3"]["gate_n_groups_observed"] == 0


# --------------------------------------------------------------------------
# 9. the gates are OBSERVED, not recomputed
# --------------------------------------------------------------------------


def _sparse_wide(monkeypatch, *, width: int = 4096, coord: int = 7):
    """Make the mixed-batch fixture ``width`` wide with cell-a's relu-gated
    direction SPARSE (one nonzero coordinate): a defect on one row's gate then
    moves ONE element per row — 1/4096 of the row, 1/(3 * 4096) of the group,
    ~5e-5 of the case — under every concentration bound."""
    import test_lens_vectorized_apply as tlva
    monkeypatch.setattr(tlva, "HIDDEN", width)
    real = tlva._gated_int_op

    def sparse(layer, seed=1, positions=la.POSITIONS_ALL, header=None, skip_bos=False):
        op = real(layer, seed=seed, positions=positions, header=header, skip_bos=skip_bos)
        v = torch.zeros(width)
        v[coord] = 3.0
        return dataclasses.replace(op, vector=v)
    monkeypatch.setattr(tlva, "_gated_int_op", sparse)


def _plant_gate_row0_gets_row1(monkeypatch, st):
    """GROUPED-ONLY: the stacked gate pre-activation of row 0 is replaced by
    row 1's (the reviewer's false-green construction). The per-row reference
    is untouched, and a RECOMPUTATION of the gate from the rows sees nothing."""
    real = la._row_reduction

    def bugged(x, fn, rowwise):
        out = real(x, fn, rowwise)
        if rowwise and x.dim() == 2 and x.shape[0] > 1 and out.dim() == 1:
            out = out.clone()
            out[0] = out[1]
        return out
    monkeypatch.setattr(la, "_row_reduction", bugged)


def _plant_gate_nudge_row0(monkeypatch, st, eps: float):
    """GROUPED-ONLY: row 0's stacked gate pre-activation moved by ``eps`` — a
    legitimate kernel-variance-sized delta (a few 2^-24 of sum|hw| when ``eps``
    is ~1e-4 against a sum|hw| in the thousands)."""
    real = la._row_reduction

    def nudged(x, fn, rowwise):
        out = real(x, fn, rowwise)
        if rowwise and x.dim() == 2 and x.shape[0] > 1 and out.dim() == 1:
            out = out.clone()
            out[0] = out[0] + eps
        return out
    monkeypatch.setattr(la, "_row_reduction", nudged)


def _plant_write_one_ulp_at(monkeypatch, st, *, col: int):
    """GROUPED-ONLY WRITE defect of ONE ulp at ``col`` of cell-a's row 0 (the
    gate the apply used is whatever it was — captured faithfully)."""
    orig = st._registry.apply

    def perturbed(*args, **kw):
        out = orig(*args, **kw)
        if (not kw.get("legacy_masks") and isinstance(kw.get("ctx"), la.DecodeRows)
                and args[0] == "cell-a" and out is not args[2]):
            out = out.clone()
            out[0, col] = torch.nextafter(out[0, col], torch.tensor(float("inf"), dtype=out.dtype))
        return out
    monkeypatch.setattr(st._registry, "apply", perturbed)


def _plant_write_one_ulp(monkeypatch, st):
    """GROUPED-ONLY WRITE defect of ONE ulp: cell-a's grouped output on row 0,
    coordinate 7, moved to the next representable value — the gate the apply
    used is right and captured equal; only exactness where the gates agree
    can see this (the fixture's rows are fp32, so one fp32 ulp)."""
    orig = st._registry.apply

    def perturbed(*args, **kw):
        out = orig(*args, **kw)
        if (not kw.get("legacy_masks") and isinstance(kw.get("ctx"), la.DecodeRows)
                and args[0] == "cell-a" and out is not args[2]):
            out = out.clone()
            out[0, 7] = torch.nextafter(out[0, 7], torch.tensor(float("inf"), dtype=out.dtype))
        return out
    monkeypatch.setattr(st._registry, "apply", perturbed)


def _host_numeric_state():
    """What the bf16 fixture's flip count depends on besides its seeded
    inputs: the process's thread count (the BLAS partition, hence the fp32
    summation order), the CPU's vector ISA, and the two torch defaults the
    arithmetic assumes. For assertion messages, so a failure on another host
    names its state instead of reading as an unexplained numerics mismatch."""
    return {"threads": torch.get_num_threads(), "cpu": torch.backends.cpu.get_cpu_capability(),
            "default_dtype": str(torch.get_default_dtype()),
            "fp32_matmul_precision": torch.get_float32_matmul_precision()}


def _wide_bf16_drive(monkeypatch, *, kind: str, n_rows: int = 32, width: int = 4096, seed: int = 0):
    """A 32-row bf16 decode batch of cell-a (non-integer rows, a non-integer
    gated op — ``kind`` 'vector' or 'lowrank') through a shadow-compare worker
    in the stacked mode at two layers; returns the record. Unlike the integer
    fixtures this produces REAL kernel variance on the CPU: the stacked
    reduction and the single-row ones round differently in fp32. Every random
    tensor is drawn from the seeded generator; the thread count is the host's
    (never pinned — the pooled flip count is judged at a scale that absorbs
    it, ``_case_frac_tol``)."""
    import test_lens_vectorized_apply as tlva
    # the arithmetic assumes torch's process defaults; a setting another test leaked (a bf16 default dtype,
    # a reduced-precision fp32 matmul) fails here by name rather than downstream as an unexplained flip count
    assert torch.get_default_dtype() is torch.float32, _host_numeric_state()
    assert torch.get_float32_matmul_precision() == "highest", _host_numeric_state()
    monkeypatch.setattr(la, "_STACKED_REDUCTIONS", True)
    monkeypatch.setenv(lw.SHADOW_COMPARE_ENV, "1")
    gen = torch.Generator().manual_seed(seed)
    if kind == "vector":
        # gate weights small enough that the fp32 gate variance flips only a few output roundings per
        # group. The pooled fraction of this 2-group, 2.6e5-element case is a host-dependent flip count,
        # judged at _case_frac_tol and never at the case-wide 1e-4 (/64 read below 1e-4 on one CPU and
        # 1.68e-4 on another for the byte-identical file; /8 read 1.5e-4)
        ops = [la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=l, vector=torch.randn(width, generator=gen),
                         gate_w=torch.randn(width, generator=gen) / 64.0, gate_b=torch.tensor([0.25]))
               for l in (1, 2)]
    else:
        ops = [la.HookOp(op=la.OP_LOWRANK_RELU_GATED, layer=l, lowrank_a=torch.randn(4, width, generator=gen) / 8.0,
                         lowrank_b=torch.randn(width, 4, generator=gen)) for l in (1, 2)]
    a = _obj("cell-a", ops)
    zero = _obj("none__lens0", list(ops), scale=0.0)
    st, view = _steering(_registry(a, zero, header=HEADER))
    assert st.apply_path == lw.APPLY_PATH_SHADOW
    shapes = [(1, 5 + i) for i in range(n_rows)]
    qsl, seq_lens, nat, pos = tlva._forward(shapes)
    rids = [f"a{i}" for i in range(n_rows)]
    toks = {rid: list(range(100, 100 + c + q)) for rid, (q, c) in zip(rids, shapes)}
    view.set_batch(rids, {rid: tlva._sel(a) for rid in rids}, toks, (qsl, seq_lens, nat),
                   prompt_lens={rid: 3 for rid in rids})
    h = torch.randn(nat, width, generator=gen).to(torch.bfloat16)
    r = torch.randn(nat, width, generator=gen).to(torch.bfloat16)
    for layer in (1, 2):
        st.hook_body(layer, (pos, h, r), (h, r))
    return st.shadow_record.as_dict()


#: the group count from which the case-wide pooled-fraction bound is the honest
#: readout: REDUCTION_FRAC_TOL's measured basis is cases of thousands of groups
#: (the 1600-trial microbench, the A/B by-rows tables) — see _case_frac_tol
MANY_GROUPS = 1000


def _case_frac_tol(rec) -> float:
    """The pooled-fraction bound honest for a record of THIS size, derived
    from its group count and the contract's own constants.

    The pooled fraction is the element-weighted mean of the per-group
    fractions, so it never exceeds the largest one — which the contract bounds
    at REDUCTION_GROUP_FRAC_TOL (2e-3) for every healthy group. The case-wide
    REDUCTION_FRAC_TOL (1e-4) is a statement about that mean over THOUSANDS of
    groups (its measured basis: 0 to 3.9e-5 on the microbench, 0 to 1.35e-5 on
    the A/B). Over a handful of groups the same mean is a few host-dependent
    flip counts, not a rate: how many bf16 output roundings a fp32 gate delta
    flips follows the host's BLAS summation order, and the byte-identical
    2-group ``_wide_bf16_drive`` file (2.6e5 elements) read 1.68e-4 on one CPU
    and below 1e-4 on another. The production bound is not a stable readout at
    that size, so a test asserting it there measures the host, not the path.
    Below MANY_GROUPS the honest bound is therefore the group bound (exact
    arithmetic, no model); from MANY_GROUPS up it is the production constant,
    so a fixture widened into the measured regime is judged at the measured
    bound. The production constant itself is untouched: the gate's callers
    never pass ``frac_tol``."""
    n_groups = int(rec["n_groups_compared"])
    return g.REDUCTION_FRAC_TOL if n_groups >= MANY_GROUPS else g.REDUCTION_GROUP_FRAC_TOL


def _production_scale_differs_only_on_the_pooled_clause(rec, tol):
    """The same record judged at the production scale: every verdict field
    but the pooled clause's own (``ok``, ``detail``, ``frac_tol``) is
    identical, and where the production verdict disagrees the RULES it names
    are the pooled fraction and nothing else. Pinned so the case-scale
    evaluation can never hide a gate, identity, reconstruction, exactness or
    concentration failure behind the looser pooled bound — on a host where the
    pooled fraction lands under 1e-4 both verdicts are ok; on one where it does
    not, exactly that clause differs.

    A FAIL detail has three parts (``reduction_tolerance``): the rules that
    fired, ``"; "``-joined; the case totals (``; N unequal group(s) of M
    compared, max |diff| ...; first: ...``); and ``worst_brief``'s evidence
    tail, which READS the worst group's gate stats whatever rule fired —
    ``|delta gate|/sum|hw| 0.432 x 2^-24`` (the rule's bound is 16 x 2^-24: a
    37x margin), ``gate-edit identity 0.941 ulps (worst {...})`` (bound 4; one
    bf16 output rounding reads below 1 by construction — |delta_out| <=
    ulp(ref) against a unit of ulp(ref) + ulp(edit)). Only the first part says
    which clauses failed, so that part alone is read. Until 2026-09-17 the
    check scanned the whole detail for each other rule's signature phrase, and
    on the Modal CPU suite — 16 vCPU, another BLAS summation order, the pooled
    fraction above 1e-4 — it found ``|delta gate|`` in the evidence tail of a
    verdict whose only rule was the pooled fraction (2026-09-15 twice,
    2026-09-17 once; byte-identical message on three trees): a reading taken
    for a rule. ``test_production_scale_companion_reads_the_rules_not_the_
    evidence_tail`` replays that host's record on every host."""
    prod = g.reduction_tolerance(rec, reduction_mode=STACKED)
    assert prod["frac_tol"] == g.REDUCTION_FRAC_TOL == 1e-4                    # the production bound, untouched
    assert {k for k in tol if tol[k] != prod[k]} <= {"ok", "detail", "frac_tol"}
    if prod["ok"] is not True:
        totals = f"; {prod['n_unequal']} unequal group(s) of {prod['n_groups_compared']} compared, "
        rules, sep, evidence = prod["detail"].partition(totals)
        assert sep, (prod["detail"], _host_numeric_state())                    # rules / totals / evidence layout
        pooled = (f"fraction of unequal reduction-op elements {prod['frac_unequal_elems']:.2e} > "
                  f"{g.REDUCTION_FRAC_TOL:g} ({prod['n_unequal_elems']} of {prod['n_elems_compared']} elements)")
        assert rules == pooled, (rules, pooled, _host_numeric_state())         # ONLY the pooled clause, as a rule
        for absent in ("in one GROUP", "in one ROW", "|delta gate|", "gate-edit identity",
                       "bit-exact reconstruction", "bit-identical or masked off", "bf16 ulps of the per-row edit"):
            assert absent not in rules, absent                                 # no other rule fired
        # the evidence tail is where those phrases legitimately live: readings, inside their production bounds
        worst_gate = (rec.get("worst_unequal") or {}).get("gate")
        if isinstance(worst_gate, dict) and "gate_error" not in worst_gate:
            assert "|delta gate|/sum|hw|" in evidence and "gate-edit identity" in evidence, evidence
        assert prod["max_gate_diff_rel_sum"] <= g.REDUCTION_GATE_REL_SUM_TOL
        assert prod["max_gate_edit_consistency_ulps"] <= g.REDUCTION_GATE_CONSISTENCY_ULPS_TOL
    return prod


def _at_the_other_hosts_pooled_fraction(rec, frac: float = 1.68e-4):
    """A copy of ``rec`` whose pooled unequal fraction reads what the
    byte-identical fixture read on another host's CPU: 1.68e-4, the reading
    recorded when the case scale was introduced (the vector case, another
    machine), above the production 1e-4 and under the group 2e-3. The Modal
    CPU suite's own low-rank reading is above 1e-4 too — its production
    verdict failed on the pooled clause, six runs, one byte-identical message
    — but pytest's truncation of the detail left the number itself
    unrecorded; any value in (1e-4, 2e-3] replays the branch. The first
    unequal reduction
    key's ``n_unequal_elems`` is raised (never lowered) until the pooled count
    is ceil(frac x the elements the production verdict compares). Every other
    field — the gate stats, the concentration fields, the worst group and its
    evidence — is the local record's, so the production-scale verdict fails
    on the pooled clause and nothing else, and its detail carries the same
    evidence tail that host's did."""
    out = copy.deepcopy(rec)
    base = g.reduction_tolerance(rec, reduction_mode=STACKED)
    keys = [k for k, e in out["by_key"].items() if int(e.get("n_unequal") or 0) > 0]
    assert keys, rec["n_unequal"]
    want = math.ceil(frac * int(base["n_elems_compared"]))
    e = out["by_key"][keys[0]]
    e["n_unequal_elems"] = int(e.get("n_unequal_elems") or 0) + max(0, want - int(base["n_unequal_elems"]))
    return out


def _plant_write_one_coordinate_over(monkeypatch, st):
    """GROUPED-ONLY WRITE defect: cell-a's grouped edit on row 0 lands one
    coordinate over — the gate the apply used is right, the write is not."""
    orig = st._registry.apply

    def perturbed(*args, **kw):
        out = orig(*args, **kw)
        if (not kw.get("legacy_masks") and isinstance(kw.get("ctx"), la.DecodeRows)
                and args[0] == "cell-a" and out is not args[2]):
            rows = args[2]
            out = out.clone()
            out[0] = rows[0] + (out[0] - rows[0]).roll(1, 0)
        return out
    monkeypatch.setattr(st._registry, "apply", perturbed)


class TestObservedGates:
    """Review P1 of the gated contract (2026-09-06): the rule must read the
    gates the two applies ACTUALLY USED, not a recomputation — a grouped-only
    defect that hands row 0 row 1's gate is invisible to a recomputation, and
    with a sparse direction also to every concentration bound. The seam
    (``la.GateCapture`` / ``la.gate_capture``), the observed comparison
    (``lw._gate_stats_observed``: the gate delta AND the gate-edit identity),
    and the public verdict end to end through the worker's shadow path. The
    healthy-but-UNEQUAL observed case is proven at the unit level (a 32 x 4096
    bf16 group): the mixed-batch fixture drives fp32 rows, in which any gate
    nudge is resolvable on every touched element, so a kernel-variance twin
    cannot be staged end to end at that size without tripping the pooled
    fraction of a 24k-element case; the end-to-end healthy run is the exact one."""

    def test_capture_seam_records_the_gates_the_apply_used(self, monkeypatch, stacked):
        d = 8
        vec = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=torch.tensor([1.0] + [0.0] * 7),
                        gate_w=torch.ones(d), gate_b=torch.tensor([0.0]))
        low = la.HookOp(op=la.OP_LOWRANK_LINEAR_GATED, layer=0, lowrank_a=torch.tensor([[1.0] * d, [1.0, -1.0] * 4]),
                        lowrank_b=torch.ones(d, 2))
        rows = torch.tensor([[1.0] * d, [0.5] * d, [2.0] + [0.0] * 7], dtype=torch.bfloat16)
        assert la._GATE_CAPTURE is None
        cap = la.GateCapture()
        with la.gate_capture(cap):
            assert la._GATE_CAPTURE is cap
            la.apply_hook_ops(rows, (vec, low), 1.0, layer=0, ctx=la.DecodeRows())
        assert la._GATE_CAPTURE is None                                   # restored
        assert [e["op"] for e in cap.entries] == [la.OP_RELU_GATED_VECTOR, la.OP_LOWRANK_LINEAR_GATED]
        v, lo = cap.entries
        assert v["kind"] == "vector" and v["gate"].tolist() == [8.0, 4.0, 2.0] and v["sum_abs"].tolist() == [8.0, 4.0, 2.0]
        assert v["eff"] == 1.0 and v["mask"] is None and v["vector"] is vec.vector
        assert lo["kind"] == "lowrank" and lo["gate"].shape == (3, 2) == lo["sum_abs"].shape and lo["write"] is low.lowrank_b
        assert lo["gate"].tolist() == [[8.0, 0.0], [4.0, 0.0], [2.0, 2.0]]
        # unarmed: nothing recorded, nothing allocated on the apply's side
        la.apply_hook_ops(rows, (vec, low), 1.0, layer=0, ctx=la.DecodeRows())
        assert len(cap.entries) == 2
        # the per-row mode captures the same shapes (concatenated per row)
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", False)
        cap2 = la.GateCapture()
        with la.gate_capture(cap2):
            la.apply_hook_ops(rows, (vec, low), 1.0, layer=0, ctx=la.DecodeRows())
        assert torch.equal(cap2.entries[0]["gate"], v["gate"]) and torch.equal(cap2.entries[1]["gate"], lo["gate"])
        # exception-safe: the sink is restored when the apply raises
        with pytest.raises(la.AdapterError):
            with la.gate_capture(la.GateCapture()):
                la.apply_hook_ops(rows, (la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=torch.ones(d + 1),
                                                   gate_w=torch.ones(d + 1), gate_b=torch.tensor([0.0])),),
                                  1.0, layer=0, ctx=la.DecodeRows())
        assert la._GATE_CAPTURE is None

    @staticmethod
    def _captures(op, rows):
        grouped = la.GateCapture()
        with la.gate_capture(grouped):
            applied = la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=la.DecodeRows())
        per_row = []
        for i in range(rows.shape[0]):
            c = la.GateCapture()
            with la.gate_capture(c):
                la.apply_hook_ops(rows[i:i + 1], (op,), 1.0, layer=0, ctx=la.DecodeRows())
            per_row.append(c)
        return grouped, per_row, applied

    def test_observed_stats_compare_the_two_captures_and_bind_gate_to_edit(self):
        d = 8
        op = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=torch.tensor([1.0] + [0.0] * 7),
                       gate_w=torch.ones(d), gate_b=torch.tensor([0.0]))
        rows = torch.tensor([[1.0] * d, [0.5] * d, [2.0] + [0.0] * 7], dtype=torch.bfloat16)
        grouped, per_row, out = self._captures(op, rows)
        ref = out                                                         # exact on the CPU: both paths agree
        gs = lw._gate_stats_observed((op,), grouped, per_row, out, ref, rows)
        assert gs["gate_source"] == "observed" and gs["gate_n_slots"] == 3 and gs["gate_n_elems"] == 3
        assert gs["gate_n_rows_unequal"] == 0 and gs["gate_max_diff_rel_sum"] == 0.0
        assert gs["gate_edit_consistency_max_ulps"] == 0.0 and gs["gate_edit_consistency_worst"]["residual"] == 0.0
        assert gs["gate_exact_rows_n"] == 3 and gs["gate_exact_rows_unequal_elems"] == 0
        assert gs["gate_lowrank_zero_delta_max_ulps_edit"] is None       # no low-rank op on the group
        assert gs["gate_write_recon_elems"] == 2 * 3 * d and gs["gate_write_recon_mismatch_elems"] == 0
        assert gs["gate_write_recon_skipped"] is None
        # THE REVIEWER'S CONSTRUCTION: the grouped apply used row 1's gate on row 0 (gates 8, 4, 2)
        bad = la.GateCapture()
        bad.entries = [dict(grouped.entries[0], gate=grouped.entries[0]["gate"].clone())]
        bad.entries[0]["gate"][0] = bad.entries[0]["gate"][1]
        applied = out.clone()
        applied[0, 0] = (rows[0, 0].float() + 4.0).to(torch.bfloat16)      # the wrong gate, faithfully written
        s = lw._gate_stats_observed((op,), bad, per_row, applied, ref, rows)
        assert s["gate_n_rows_unequal"] == 1 and s["gate_max_diff_rel_sum"] == pytest.approx(4.0 / 8.0)   # |8 - 4| / sum|hw| 8
        assert s["gate_max_diff_rel_sum"] > 1e5 * g.REDUCTION_GATE_REL_SUM_TOL
        assert s["gate_edit_consistency_max_ulps"] <= 1.0                 # the (wrong) gate explains the output
        # a WRONG WRITE: the gate the apply used is right, the edit landed one coordinate over
        applied = out.clone()
        applied[0, 1] = (rows[0, 1].float() + 8.0).to(torch.bfloat16)
        applied[0, 0] = rows[0, 0]
        s = lw._gate_stats_observed((op,), grouped, per_row, applied, ref, rows)
        assert s["gate_max_diff_rel_sum"] == 0.0                           # nothing at the gate ...
        assert s["gate_edit_consistency_max_ulps"] > 4.0                   # ... the identity breaks
        assert s["gate_edit_consistency_worst"]["row"] == 0 and s["gate_edit_consistency_worst"]["col"] in (0, 1)
        assert s["gate_exact_rows_n"] == 3 and s["gate_exact_rows_unequal_elems"] == 2   # ... and so does exactness
        assert s["gate_write_recon_mismatch_elems"] == 2                                  # ... and the reconstruction
        # the reviewer's construction (the row is exempt from exactness): the grouped gate used a wrong gate,
        # written faithfully -> the reconstruction from the OBSERVED (wrong) gate matches: no mismatch there
        s_bad = lw._gate_stats_observed((op,), bad, per_row, torch.where(torch.tensor([[True] + [False] * 7] * 3),
                                                                            (rows.float() + 4.0).to(torch.bfloat16), out)
                                        * 0 + out.clone().index_put_((torch.tensor([0]), torch.tensor([0])),
                                                                      (rows[0, 0].float() + 4.0).to(torch.bfloat16)),
                                        ref, rows)
        assert s_bad["gate_write_recon_mismatch_elems"] == 0 and s_bad["gate_max_diff_rel_sum"] == pytest.approx(0.5)
        # a low-rank group is never rebuilt: the field is None with the reason
        low_ = la.HookOp(op=la.OP_LOWRANK_RELU_GATED, layer=0, lowrank_a=torch.tensor([[1.0] * d, [1.0, -1.0] * 4]),
                         lowrank_b=torch.ones(d, 2))
        gl_, pl_, outl_ = self._captures(low_, rows)
        sl_ = lw._gate_stats_observed((low_,), gl_, pl_, outl_, outl_, rows)
        assert sl_["gate_write_recon_mismatch_elems"] is None and "lowrank" in sl_["gate_write_recon_skipped"]
        # mismatched captures are missing evidence, never a pass
        assert "gate_error" in lw._gate_stats_observed((op,), None, per_row, out, ref, rows)
        assert "gate_error" in lw._gate_stats_observed((op,), grouped, per_row[:2], out, ref, rows)
        assert "gate_error" in lw._gate_stats_observed((op,), la.GateCapture(), per_row, out, ref, rows)
        off = la.GateCapture()
        off.entries = [dict(grouped.entries[0], eff=2.0)]
        assert "different effective scales" in lw._gate_stats_observed((op,), off, per_row, out, ref, rows)["gate_error"]
        # no gated op / one row: nothing to read
        add = la.HookOp(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(d))
        assert lw._gate_stats_observed((add,), grouped, per_row, out, ref, rows) == {}
        assert lw._gate_stats_observed((op,), grouped, per_row[:1], out[:1], ref[:1], rows[:1]) == {}
        # the low-rank pair: latents [B, k], the write through B in the identity
        low = la.HookOp(op=la.OP_LOWRANK_RELU_GATED, layer=0, lowrank_a=torch.tensor([[1.0] * d, [1.0, -1.0] * 4]),
                        lowrank_b=torch.ones(d, 2))
        gl, pl, outl = self._captures(low, rows)
        sl = lw._gate_stats_observed((low,), gl, pl, outl, outl, rows)
        assert sl["gate_n_elems"] == 6 and sl["gate_n_rows_unequal"] == 0 and sl["gate_edit_consistency_max_ulps"] == 0.0
        badl = la.GateCapture()
        badl.entries = [dict(gl.entries[0], gate=gl.entries[0]["gate"].clone())]
        badl.entries[0]["gate"][0, 0] = badl.entries[0]["gate"][1, 0]      # latent 0 of row 0 <- row 1's
        appl = outl.clone()
        appl[0] = (rows[0].float() + (badl.entries[0]["gate"][0] @ low.lowrank_b.t())).to(torch.bfloat16)
        sl = lw._gate_stats_observed((low,), badl, pl, appl, outl, rows)
        assert sl["gate_n_rows_unequal"] == 1 and sl["gate_max_diff_rel_sum"] == pytest.approx(4.0 / 8.0)
        assert sl["gate_edit_consistency_max_ulps"] <= 1.0

    def test_healthy_observed_deviation_passes_the_public_rule(self):
        """A kernel-variance twin through the observed path and the public
        record + rule: one row's stacked gate a few 2^-24 of sum|hw| away and
        the one element that rounding moved, in a 32 x 4096 group — ok."""
        bf = torch.bfloat16
        d = 4096
        v = torch.zeros(d)
        v[7] = 2.0 + 2.0 ** -7                                             # so that 1.0 + v[7] is an exact bf16 tie
        op = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=v, gate_w=torch.full((d,), 2.0 ** -12),
                       gate_b=torch.tensor([0.0]))
        rows = _ints(32, d, lo=0, hi=3, seed=5).to(bf)                     # non-negative: the gate is mean(h)
        rows[2] = 1.0                                                      # row 2's gate is EXACTLY 1.0 (4096 x 2^-12)
        grouped, per_row, out = self._captures(op, rows)
        ref = out
        assert float(grouped.entries[0]["gate"][2]) == 1.0 and float(ref[2, 7]) == 3.0     # 3.0078125 rounded to even
        sum_hw = float(grouped.entries[0]["sum_abs"][2])                   # = 1.0
        eps = 2.0 * g.REDUCTION_GATE_UNIT * sum_hw                         # 2 x 2^-24 of the reduction: 1 + 2^-23, exact
        grouped.entries[0]["gate"][2] += eps
        # the grouped side WROTE from the moved gate — the same write the apply uses — and the tie tips over:
        # one bf16 ulp at [2, 7], nothing else
        applied = la.gated_vector_write(rows, grouped.entries[0]["gate"], 1.0, v, None, bf)
        assert float(applied[2, 7]) == 3.015625 and int((applied != ref).sum()) == 1
        gs = lw._gate_stats_observed((op,), grouped, per_row, applied, ref, rows)
        assert gs["gate_n_rows_unequal"] == 1 and gs["gate_max_diff_rel_sum"] == pytest.approx(2 * g.REDUCTION_GATE_UNIT, rel=1e-6)
        assert gs["gate_max_diff_rel_sum"] < g.REDUCTION_GATE_REL_SUM_TOL / 4
        assert gs["gate_edit_consistency_max_ulps"] <= 1.0
        assert gs["gate_exact_rows_n"] == 31 and gs["gate_exact_rows_unequal_elems"] == 0   # row 2's gate moved; the rest agree
        stats = lw._unequal_stats(applied, ref, rows)
        assert gs["gate_write_recon_mismatch_elems"] == 0                 # the rounding flip IS what the moved gate writes
        gate = {k_: gs[k_] for k_ in GATE_OK if k_ in gs}
        gate.update(gate_n_groups_with_stats=1, gate_n_groups_observed=1, gate_n_errors=0, gate_n_groups_recon=1)
        tol = g.reduction_tolerance(_synthetic_case(stats, key=PSR_KEY, gate=gate), reduction_mode=STACKED)
        assert tol["ok"] is True, tol["detail"]
        assert "gates observed on all 1 unequal group(s)" in tol["detail"] and "writes exact on the 31 row(s)" in tol["detail"]
        assert "both outputs the bit-exact reconstruction from the observed gates on all 1 vector-gated group(s)" in tol["detail"]
        # the same record through a real ShadowCompareRecord: the aggregates the gate reads
        rec = lw.ShadowCompareRecord(enabled=True)
        rec.observe(key=PSR_KEY, layer=1, cell="c", rows=32, site="block", equal=False, n_elems=ref.numel(),
                    gate=gs, **stats)
        e = rec.as_dict()["by_key"][PSR_KEY]
        assert e["gate_n_groups_with_stats"] == 1 and e["gate_edit_consistency_max_ulps"] == gs["gate_edit_consistency_max_ulps"]
        assert e["gate_max_diff_rel_sum"] == gs["gate_max_diff_rel_sum"]

    def test_sparse_gate_misplacement_fails_through_the_observed_gate_not_the_bounds(self, monkeypatch):
        """END TO END through the worker's shadow path and the public rule:
        a grouped-only defect handing row 0 row 1's gate, with a sparse
        direction. Every concentration bound and the pooled fraction stay
        under threshold, the edit-ulp unit is informational for a gated op,
        and a recomputed gate reads 0.0 — the pre-fix false green. The
        OBSERVED grouped gate carries the defect and fails the gate bound by
        orders of magnitude."""
        _sparse_wide(monkeypatch)
        rec = _shadow_drive(monkeypatch, plant=_plant_gate_row0_gets_row1)
        assert rec["n_unequal"] >= 1 and rec["n_flag_disagreements"] == 0
        fu = rec["first_unequal"]
        assert ":relu_gated_vector:" in fu["key"] and fu["rows"] == 3
        # under every concentration / fraction bound: one element per affected row
        assert rec["max_frac_row"] <= 1 / 4096 + 1e-12 < g.REDUCTION_ROW_FRAC_TOL / 50
        assert rec["max_frac_group"] <= 1 / (3 * 4096) + 1e-12 < g.REDUCTION_GROUP_FRAC_TOL / 20
        tol = g.reduction_tolerance(rec, reduction_mode=STACKED)
        assert tol["frac_unequal_elems"] < g.REDUCTION_FRAC_TOL
        # the recomputation is blind; the observation is not
        gate = fu["gate"]
        assert gate["gate_source"] == "observed" and gate["recomputed_max_diff_rel_sum"] == 0.0
        assert gate["gate_max_diff_rel_sum"] > 1e4 * g.REDUCTION_GATE_REL_SUM_TOL
        assert gate["gate_n_rows_unequal"] == 1                             # exactly row 0's gate
        assert tol["ok"] is False and tol["keys_lacking_gate_stats"] == []
        assert "largest |delta gate|/sum|hw| of a GATED reduction op" in tol["detail"]
        assert "> 16 x 2^-24 (the stacked gate is not the per-row gate" in tol["detail"]
        for absent in ("in one ROW", "in one GROUP", "> 0.0001", "bf16 ulps of the per-row edit beyond"):
            assert absent not in tol["detail"], absent                        # ONLY the observed gate fired
        ok, detail = g.shadow_check(rec, (3,), routing=g.ROUTING_VECTORIZED, waves=_waves(rec), reduction_mode=STACKED)
        assert ok is False and "|delta gate|" in detail
        assert g.shadow_multigroup_check(rec, reduction_mode=STACKED)[0] is False

    def test_sparse_write_defect_fails_through_the_gate_edit_identity(self, monkeypatch):
        """END TO END: the gate the grouped apply used is right, its edit
        landed one coordinate over. The observed gate delta is 0, the bounds
        are all under threshold, and the gate-edit identity fails the gate."""
        _sparse_wide(monkeypatch)
        rec = _shadow_drive(monkeypatch, plant=_plant_write_one_coordinate_over)
        assert rec["n_unequal"] >= 1 and rec["n_flag_disagreements"] == 0
        fu = rec["first_unequal"]
        assert rec["max_frac_row"] <= 2 / 4096 + 1e-12 < g.REDUCTION_ROW_FRAC_TOL / 20
        tol = g.reduction_tolerance(rec, reduction_mode=STACKED)
        assert tol["frac_unequal_elems"] < g.REDUCTION_FRAC_TOL
        assert fu["gate"]["gate_max_diff_rel_sum"] == 0.0 and fu["gate"]["gate_n_rows_unequal"] == 0
        assert fu["gate"]["gate_edit_consistency_max_ulps"] > g.REDUCTION_GATE_CONSISTENCY_ULPS_TOL
        assert fu["gate"]["gate_exact_rows_unequal_elems"] == 2            # row 0's gate agreed: its write must be exact
        assert fu["gate"]["gate_write_recon_mismatch_elems"] == 2          # ... and it is not the reconstruction
        assert tol["ok"] is False and "the gate-edit identity of a GATED reduction op breaks by" in tol["detail"]
        assert "in rows whose observed gates are bit-identical or masked off" in tol["detail"]
        assert "largest |delta gate|" not in tol["detail"] and "in one ROW" not in tol["detail"]

    def test_sparse_one_ulp_wrong_write_where_gates_agree_fails_end_to_end(self, monkeypatch):
        """Review P1-b: a grouped-only write of ONE ulp at one coordinate,
        gates captured equal. The identity residual is ~0.5 (inside its
        4-ulp tolerance — the pre-fix false green), every bound is under
        threshold, the gate delta is 0; exactness where the gates agree is
        what fails it."""
        _sparse_wide(monkeypatch)
        rec = _shadow_drive(monkeypatch, plant=_plant_write_one_ulp)
        assert rec["n_unequal"] >= 1 and rec["n_flag_disagreements"] == 0
        fu = rec["first_unequal"]
        gate = fu["gate"]
        assert gate["gate_source"] == "observed" and gate["gate_max_diff_rel_sum"] == 0.0
        assert gate["gate_edit_consistency_max_ulps"] <= g.REDUCTION_GATE_CONSISTENCY_ULPS_TOL   # the identity alone is blind
        assert gate["gate_exact_rows_n"] == 3 and gate["gate_exact_rows_unequal_elems"] == 1
        assert gate["gate_write_recon_mismatch_elems"] == 1                                    # the general rule sees it too
        tol = g.reduction_tolerance(rec, reduction_mode=STACKED)
        assert tol["frac_unequal_elems"] < g.REDUCTION_FRAC_TOL and tol["max_frac_row"] <= 1 / 4096 + 1e-12
        assert tol["ok"] is False
        assert "unequal element(s) of a GATED reduction op in rows whose observed gates are bit-identical or masked off" in tol["detail"]
        assert "not the bit-exact reconstruction" in tol["detail"]
        assert "a wrong write, never kernel variance" in tol["detail"]
        for absent in ("gate-edit identity of a GATED reduction op breaks", "largest |delta gate|", "in one ROW", "> 0.0001"):
            assert absent not in tol["detail"], absent                        # only the two write rules fired

    def test_legit_gate_delta_plus_write_outside_the_support_fails_on_write_consistency(self, monkeypatch):
        """Round 4 P1, end to end: a legitimate nonzero gate delta on row 0
        (inside the bound) exempts the row from exactness, and a grouped-only
        ONE-ulp write on a coordinate where the direction is exactly ZERO
        leaves the identity residual at ~1 — the pre-fix false green. The
        bit-exact reconstruction from the observed gate fails it alone."""
        # 8192 wide: the nudge moves col 7 on row 0 at both layers and the write adds col 8 — a handful
        # of elements against 8e4 compared, so the pooled fraction stays under 1e-4 (at 4096 it read 1.2e-4)
        _sparse_wide(monkeypatch, width=8192)
        rec = _shadow_drive(monkeypatch, plant=lambda mp, st: (_plant_gate_nudge_row0(mp, st, 2.0 ** -13),
                                                               _plant_write_one_ulp_at(mp, st, col=8)))
        assert rec["n_unequal"] >= 1 and rec["n_flag_disagreements"] == 0
        gate = rec["first_unequal"]["gate"]
        assert gate["gate_source"] == "observed"
        assert 0 < gate["gate_max_diff_rel_sum"] <= g.REDUCTION_GATE_REL_SUM_TOL          # a legitimate, nonzero delta
        assert gate["gate_edit_consistency_max_ulps"] <= g.REDUCTION_GATE_CONSISTENCY_ULPS_TOL   # the envelope is blind
        assert gate["gate_exact_rows_unequal_elems"] == 0                                  # row 0 is exempt (dg != 0)
        assert gate["gate_write_recon_mismatch_elems"] >= 1                                # the reconstruction is not
        tol = g.reduction_tolerance(rec, reduction_mode=STACKED)
        assert tol["frac_unequal_elems"] < g.REDUCTION_FRAC_TOL and tol["max_frac_row"] <= 2 / 8192 + 1e-12
        assert tol["ok"] is False
        assert "is not the bit-exact reconstruction base + eff * g * v from that side's OBSERVED gate" in tol["detail"]
        for absent in ("gate-edit identity of a GATED reduction op breaks", "largest |delta gate|", "in one ROW",
                       "> 0.0001", "bit-identical or masked off"):
            assert absent not in tol["detail"], absent                                    # write consistency ALONE
        # the same delta with a wrong write ON the nonzero coordinate (one ulp beyond what the gate implies)
        rec2 = _shadow_drive(monkeypatch, plant=lambda mp, st: (_plant_gate_nudge_row0(mp, st, 2.0 ** -13),
                                                                _plant_write_one_ulp_at(mp, st, col=7)))
        gate2 = rec2["first_unequal"]["gate"]
        assert 0 < gate2["gate_max_diff_rel_sum"] <= g.REDUCTION_GATE_REL_SUM_TOL
        assert gate2["gate_write_recon_mismatch_elems"] >= 1
        tol2 = g.reduction_tolerance(rec2, reduction_mode=STACKED)
        assert tol2["ok"] is False and "not the bit-exact reconstruction" in tol2["detail"]
        # and the nudge alone — a legitimate delta, faithfully written — is ok
        rec3 = _shadow_drive(monkeypatch, plant=lambda mp, st: _plant_gate_nudge_row0(mp, st, 2.0 ** -13))
        tol3 = g.reduction_tolerance(rec3, reduction_mode=STACKED)
        assert tol3["ok"] is True, tol3["detail"]
        if rec3["n_unequal"]:
            assert rec3["first_unequal"]["gate"]["gate_write_recon_mismatch_elems"] == 0

    def test_one_ulp_write_on_a_masked_off_row_fails_through_the_public_rule(self):
        """Review P1-b, the masked variant: a row every gated op masks off must
        come back untouched by both paths; one ulp written there is a wrong
        write whatever the gate did. Decode groups never mask a row (every
        decode row is a response row), so this is pinned through the observed
        comparison, the public record and the rule with a synthetic mask."""
        bf = torch.bfloat16
        d = 4096
        v = torch.zeros(d)
        v[7] = 3.0
        op = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=v, gate_w=torch.ones(d), gate_b=torch.tensor([0.0]))
        rows = _ints(32, d, lo=0, hi=3, seed=5).to(bf)
        grouped, per_row, out = self._captures(op, rows)
        grouped.entries[0]["mask"] = torch.ones(32, dtype=torch.bool)
        grouped.entries[0]["mask"][5] = False                              # row 5 receives no edit
        ref = out.clone()
        ref[5] = rows[5]                                                   # ... on the per-row path
        applied = ref.clone()
        applied[5, 7] = (ref[5, 7].float() + la.ulp_spacing(ref[5, 7], bf)).to(bf)   # one ulp written anyway
        gs = lw._gate_stats_observed((op,), grouped, per_row, applied, ref, rows)
        assert gs["gate_exact_rows_unequal_elems"] == 1 and gs["gate_exact_rows_n"] == 32
        assert gs["gate_edit_consistency_max_ulps"] <= g.REDUCTION_GATE_CONSISTENCY_ULPS_TOL
        stats = lw._unequal_stats(applied, ref, rows)
        gate = {k_: gs[k_] for k_ in GATE_OK if k_ in gs}
        gate.update(gate_n_groups_with_stats=1, gate_n_groups_observed=1, gate_n_errors=0, gate_n_groups_recon=1)
        tol = g.reduction_tolerance(_synthetic_case(stats, key=PSR_KEY, gate=gate), reduction_mode=STACKED)
        assert tol["ok"] is False and "bit-identical or masked off" in tol["detail"]
        assert gs["gate_write_recon_mismatch_elems"] == 1 and "not the bit-exact reconstruction" in tol["detail"]
        # the same row untouched on both paths: ok
        gs0 = lw._gate_stats_observed((op,), grouped, per_row, ref, ref, rows)
        assert gs0["gate_exact_rows_unequal_elems"] == 0

    def test_lowrank_zero_delta_rows_take_the_linear_unit(self):
        """A low-rank write z @ B^T reduces over k, so an equal latent vector
        does not promise an exact write: such rows are judged by the LINEAR
        edit-ulp unit (one output-ulp flip reads 0 excess and passes; a
        wrong write of many edit ulps fails), never by exactness — and never
        by the reconstruction, which the low-rank pair does not get. Round 5
        P3: the demonstration runs on the REGISTERED low-rank key
        (lowrank_gated:lowrank_relu_gated), where the one-ulp PASS is the
        accepted class — BOUNDED, not exact — not on the vector key."""
        bf = torch.bfloat16
        d = 4096
        low = la.HookOp(op=la.OP_LOWRANK_RELU_GATED, layer=0,
                        lowrank_a=torch.stack([torch.ones(d), torch.tensor([1.0, -1.0] * (d // 2))]),
                        lowrank_b=torch.ones(d, 2))
        rows = _ints(32, d, lo=0, hi=3, seed=5).to(bf)
        gl, pl, outl = self._captures(low, rows)
        one_ulp = outl.clone()
        one_ulp[0, 3] = (outl[0, 3].float() + la.ulp_spacing(outl[0, 3], bf)).to(bf)
        sl = lw._gate_stats_observed((low,), gl, pl, one_ulp, outl, rows)
        assert sl["gate_exact_rows_n"] == 0 and sl["gate_exact_rows_unequal_elems"] == 0     # not an exactness row
        assert sl["gate_lowrank_zero_delta_max_ulps_edit"] == 0.0                            # one output ulp: no excess
        wrong = outl.clone()
        wrong[0, 3] = (outl[0, 3].float() + 64.0 * la.ulp_spacing(outl[0, 3] - rows[0, 3], bf)).to(bf)
        sw = lw._gate_stats_observed((low,), gl, pl, wrong, outl, rows)
        assert sw["gate_lowrank_zero_delta_max_ulps_edit"] > g.REDUCTION_ULPS_EDIT_TOL
        assert sw["gate_exact_rows_unequal_elems"] == 0                                      # never judged as exactness
        # through the public rule on the REGISTERED low-rank key (one element of one 32 x 4096 group:
        # every concentration bound holds); a low-rank key is never rebuilt and needs no reconstruction
        assert g.shadow_key_op_class(LOWRANK_KEY) == g.OP_CLASS_REDUCTION_GATED and not g.shadow_key_has_vector_gate(LOWRANK_KEY)
        assert sw["gate_write_recon_mismatch_elems"] is None                                 # never rebuilt
        stats = lw._unequal_stats(wrong, outl, rows)
        gate = dict(GATE_OK, gate_lowrank_zero_delta_max_ulps_edit=sw["gate_lowrank_zero_delta_max_ulps_edit"],
                    gate_exact_rows_n=0, gate_n_groups_recon=0, gate_write_recon_mismatch_elems=None)
        tol = g.reduction_tolerance(_synthetic_case(stats, key=LOWRANK_KEY, gate=gate), reduction_mode=STACKED)
        assert tol["ok"] is False and "the low-rank write on rows whose observed latents are bit-identical departs by" in tol["detail"]
        assert "in one ROW" not in tol["detail"] and "bit-identical or masked off" not in tol["detail"]
        assert "bit-exact reconstruction" not in tol["detail"]
        # THE ACCEPTED CLASS: identical latents + one wrong bf16 output ulp PASSES on the low-rank key — the
        # write is a k-term reduction, so those rows are BOUNDED (0 excess edit ulps <= 4), never exact
        gate_ok = dict(gate, gate_lowrank_zero_delta_max_ulps_edit=0.0)
        tol_ok = g.reduction_tolerance(_synthetic_case(lw._unequal_stats(one_ulp, outl, rows), key=LOWRANK_KEY, gate=gate_ok),
                                       reduction_mode=STACKED)
        assert tol_ok["ok"] is True and "low-rank zero-delta rows within 0 edit ulps (<= 4)" in tol_ok["detail"]
        assert tol_ok["vector_gated_keys_unequal"] == [] and tol_ok["n_gate_groups_recon"] == 0
        assert "writes exact on the 0 row(s)" in tol_ok["detail"]                   # no row of it is held exact
        # the same one-ulp write on the VECTOR key is a wrong write (exactness + reconstruction), for contrast
        assert g.reduction_tolerance(_synthetic_case(lw._unequal_stats(one_ulp, outl, rows), key=PSR_KEY,
                                                     gate=dict(GATE_OK, gate_exact_rows_unequal_elems=1,
                                                               gate_write_recon_mismatch_elems=1)),
                                     reduction_mode=STACKED)["ok"] is False

    def test_gate_magnitude_histogram_on_a_planted_distribution(self):
        """Measurement only (2026-09-06, the small-gates question): per-row
        gates planted at 0, 3 f, 100 f and 1e5 f (f = 2^-24 sum|hw| = 2^-24 with
        sum|hw| = 1) bucket 1 / 1 / 1 / 1; the near-threshold dose share is
        3 / (3 + 100 + 1e5) by hand (eff and ||v|| cancel); g/f spans 3 to 1e5.
        The record sums the counts and the doses and recomputes the share."""
        f = 2.0 ** -24
        d = 4
        v = torch.tensor([3.0, 0.0, 4.0, 0.0])                                  # ||v|| = 5
        op = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=v, gate_w=torch.ones(d), gate_b=torch.tensor([0.0]))
        gates = torch.tensor([0.0, 3 * f, 100 * f, 1e5 * f])
        per_row = []
        for gv in gates:
            c = la.GateCapture()
            c.entries = [{"op": op.op, "kind": "vector", "gate": gv.reshape(1), "sum_abs": torch.ones(1),
                          "eff": 2.0, "mask": None, "vector": v}]
            per_row.append(c)
        grouped = la.GateCapture()
        grouped.entries = [{"op": op.op, "kind": "vector", "gate": gates.clone(), "sum_abs": torch.ones(4),
                            "eff": 2.0, "mask": None, "vector": v}]
        rows = torch.zeros(4, d, dtype=torch.bfloat16)
        gs = lw._gate_histogram_observed((op,), per_row)
        assert gs["gate_hist_n_groups"] == 1 and gs["gate_hist_n_rows"] == 4
        assert (gs["gate_hist_zero"], gs["gate_hist_near"], gs["gate_hist_mid"], gs["gate_hist_far"]) == (1, 1, 1, 1)
        # the unequal-only stats carry no histogram: it is a separate, every-group measurement
        assert "gate_hist_zero" not in lw._gate_stats_observed((op,), grouped, per_row, rows, rows, rows)
        assert gs["gate_dose_near"] == pytest.approx(2.0 * 3 * f * 5.0, rel=1e-6)
        assert gs["gate_dose_total"] == pytest.approx(2.0 * (3 + 100 + 1e5) * f * 5.0, rel=1e-6)
        assert gs["gate_dose_share_near_threshold"] == pytest.approx(3 / (3 + 100 + 1e5), rel=1e-6)
        assert gs["gate_g_over_f_min_nonzero"] == pytest.approx(3.0) and gs["gate_g_over_f_max"] == pytest.approx(1e5)
        # the bucket edges are inclusive on the upper side: exactly 10 f is near, exactly 1000 f is mid
        edge = la.GateCapture()
        edge.entries = [dict(grouped.entries[0], gate=torch.tensor([10 * f, 1000 * f, 1000.5 * f, 0.0]))]
        per_edge = []
        for gv in edge.entries[0]["gate"]:
            c = la.GateCapture()
            c.entries = [dict(per_row[0].entries[0], gate=gv.reshape(1))]
            per_edge.append(c)
        ge = lw._gate_histogram_observed((op,), per_edge)
        assert (ge["gate_hist_zero"], ge["gate_hist_near"], ge["gate_hist_mid"], ge["gate_hist_far"]) == (1, 1, 1, 1)
        # the low-rank pair: one element per row x latent, dose through ||B[:, j]||
        low = la.HookOp(op=la.OP_LOWRANK_RELU_GATED, layer=0, lowrank_a=torch.ones(2, d),
                        lowrank_b=torch.tensor([[3.0, 0.0], [4.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))   # ||B0|| 5, ||B1|| 1
        z = torch.tensor([[0.0, 3 * f], [100 * f, 1e5 * f]])
        gl = la.GateCapture()
        gl.entries = [{"op": low.op, "kind": "lowrank", "gate": z.clone(), "sum_abs": torch.ones(2, 2), "eff": 1.0,
                       "mask": None, "write": low.lowrank_b}]
        pl = []
        for i in range(2):
            c = la.GateCapture()
            c.entries = [dict(gl.entries[0], gate=z[i:i + 1].clone(), sum_abs=torch.ones(1, 2))]
            pl.append(c)
        gs2 = lw._gate_histogram_observed((low,), pl)
        assert (gs2["gate_hist_zero"], gs2["gate_hist_near"], gs2["gate_hist_mid"], gs2["gate_hist_far"]) == (1, 1, 1, 1)
        assert gs2["gate_dose_share_near_threshold"] == pytest.approx((3 * f * 1.0) / (3 * f * 1.0 + 100 * f * 5.0 + 1e5 * f * 1.0), rel=1e-6)
        # THE RECORD SUMS EVERY COMPARED GATED GROUP, EQUAL OR UNEQUAL (round 4 P2-a): an EQUAL group of
        # far gates and an UNEQUAL group of the planted distribution both land in the denominator, so the
        # share is the planted 3e-5 diluted by the far group's dose — never ~100 % near-threshold
        far = lw._gate_histogram_observed((op,), [dict_cap for dict_cap in per_row[3:4] * 4])   # four rows at 1e5 f
        rec = lw.ShadowCompareRecord(enabled=True)
        common = dict(key=PSR_KEY, cell="c", rows=4, site="block", n_elems=16)
        rec.observe(layer=1, equal=True, gate_hist=far, **common)                        # equal: no gate stats, a histogram
        rec.observe(layer=2, equal=False, n_unequal_elems=1, max_frac_row=0.25, frac_group=1 / 16, max_abs_diff=0.5,
                    max_ulps_edit=1.0, gate=dict(GATE_OK, gate_source="observed"), gate_hist=gs, **common)
        d_ = rec.as_dict()
        for scope in (d_, d_["by_key"][PSR_KEY]):
            assert scope["gate_hist_n_groups"] == 2
            assert (scope["gate_hist_zero"], scope["gate_hist_near"], scope["gate_hist_mid"], scope["gate_hist_far"]) == (1, 1, 1, 5)
            assert scope["gate_dose_share_near_threshold"] == pytest.approx(3 / (3 + 100 + 5 * 1e5), rel=1e-6)
            assert scope["gate_g_over_f_min_nonzero"] == pytest.approx(3.0) and scope["gate_g_over_f_max"] == pytest.approx(1e5)
        assert d_["n_equal"] == 1 and d_["n_unequal"] == 1 and d_["gate_n_groups_observed"] == 1
        # informational: the public rule reads ok on it and the report clause names the numbers
        snap = _synthetic_case(lw._unequal_stats(rows, rows, rows) | {"n_unequal_elems": 1, "max_frac_row": 1 / 4096,
                                                                       "frac_group": 1 / (32 * 4096), "max_abs_ref": 1.0,
                                                                       "max_abs_diff": 0.5, "max_ulps_edit": 1.0},
                               key=PSR_KEY, gate=GATE_OK)
        snap.update({k_: d_[k_] for k_ in g.GATE_HIST_KEYS})
        tol = g.reduction_tolerance(snap, reduction_mode=STACKED)
        assert tol["ok"] is True
        assert g.gate_hist_brief(d_) == ("gate magnitudes over 2 compared group(s), equal and unequal (per-row, in noise floors f): "
                                         "zero 1, near (0 < g <= 10 f) 1, mid (<= 1000 f) 1, far 5; near-threshold dose share "
                                         "6e-06; g/f nonzero min 3, max 100000")
        assert g.gate_hist_brief({"gate_ops": ["relu_gated_vector"]}) == "" and g.gate_hist_brief(None) == ""

    def test_histogram_covers_equal_groups_end_to_end(self, monkeypatch):
        """Round 4 P2-a through the worker: the integer sparse fixture is EXACT
        (every compared cell-a group equal, far gates at gate_b = 100), yet the
        histogram counts every compared gated group; the planted misplacement
        makes them unequal and the count is the same two."""
        _sparse_wide(monkeypatch)
        rec = _shadow_drive(monkeypatch)
        assert rec["n_unequal"] == 0 and rec["gate_hist_n_groups"] == 2         # cell-a at two layers, both equal
        assert rec["gate_hist_far"] + rec["gate_hist_mid"] + rec["gate_hist_near"] + rec["gate_hist_zero"] == 6
        assert rec["gate_hist_zero"] + rec["gate_hist_far"] == 6                # relu(g + 100): fired far, or not at all
        assert rec["gate_n_groups_observed"] == 0                               # no unequal group: no gate stats at all
        a_key = next(k for k in rec["by_key"] if ":relu_gated_vector:" in k)
        assert rec["by_key"][a_key]["gate_hist_n_groups"] == 2
        rec2 = _shadow_drive(monkeypatch, plant=_plant_gate_row0_gets_row1)
        assert rec2["n_unequal"] >= 1 and rec2["gate_hist_n_groups"] == 2       # the same groups, now unequal too

    def test_public_rule_requires_observed_provenance_on_every_group(self):
        """Review P1-a: three unequal groups, two observed and one recomputed
        (with the smallest edit-ulps, so the WORST nested record still says
        'observed'): the public rule fails on the per-key observed count,
        not on any nested record; a group without a source fails the same way;
        all observed passes."""
        def obs(**over):
            base = dict(GATE_OK, gate_source="observed", gate_n_rows=32, gate_n_rows_unequal=1, gate_n_elems=32)
            for k_ in ("gate_n_groups_with_stats", "gate_n_groups_observed", "gate_n_errors", "gate_n_groups_recon"):
                base.pop(k_)
            return dict(base, **over)
        common = dict(key=PSR_KEY, cell="c", rows=32, site="block", equal=False, n_unequal_elems=1,
                      n_elems=32 * 4096, max_frac_row=1 / 4096, frac_group=1 / (32 * 4096), max_abs_diff=0.03125)

        def record(third):
            rec = lw.ShadowCompareRecord(enabled=True)
            rec.observe(layer=1, max_ulps_edit=1.0, gate=obs(), **common)
            rec.observe(layer=2, max_ulps_edit=1.0, gate=obs(), **common)
            rec.observe(layer=3, max_ulps_edit=0.0, gate=third, **common)
            return rec.as_dict()
        d = record(obs(gate_source="recomputed"))
        assert d["worst_unequal"]["gate"]["gate_source"] == "observed"     # the worst record alone would mislead
        assert d["gate_n_groups_with_stats"] == 3 and d["gate_n_groups_observed"] == 2
        tol = g.reduction_tolerance(d, reduction_mode=STACKED)
        assert tol["ok"] is False and tol["keys_lacking_gate_stats"] == [PSR_KEY]
        assert "gate_n_groups_observed == n_unequal" in tol["detail"] and "gates observed on all" not in tol["detail"]
        # no source at all on one group
        unsourced = obs()
        unsourced.pop("gate_source")
        tol = g.reduction_tolerance(record(unsourced), reduction_mode=STACKED)
        assert tol["ok"] is False and tol["keys_lacking_gate_stats"] == [PSR_KEY]
        # every group observed: ok, and the detail may say so
        tol = g.reduction_tolerance(record(obs()), reduction_mode=STACKED)
        assert tol["ok"] is True and "gates observed on all 3 unequal group(s)" in tol["detail"]

    def test_disabled_shadow_compare_never_enters_the_seam(self, monkeypatch):
        """Review P2: the production apply path never enters the capture seam
        — the seam refuses a None sink, and a worker without the shadow
        compare drives the whole mixed batch without calling gate_capture."""
        real = la.gate_capture
        with pytest.raises(la.AdapterError, match="needs a GateCapture"):
            with real(None):
                pass                                                       # pragma: no cover

        def boom(sink):
            raise AssertionError("gate_capture entered with the shadow compare disabled")
        monkeypatch.setattr(la, "gate_capture", boom)
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", True)
        monkeypatch.delenv(lw.SHADOW_COMPARE_ENV, raising=False)
        t = TestShadowCompare()
        a, b, zero = t._objs()
        st, view = _steering(_registry(a, b, zero, header=HEADER))
        assert st.apply_path == lw.APPLY_PATH_VECTORIZED_STACKED and not st._shadow_compare
        nat, pos, rows = _mixed_batch(view, a, b, zero, prefill_ids=(1, 2) + HEADER)
        h, r = _hr(nat)
        for layer in (1, 2):
            st.hook_body(layer, (pos, h, r), (h, r))                       # no capture, no sink, no raise
        assert la._GATE_CAPTURE is None
        # the low-rank branch allocates its latent list only when armed
        low = la.HookOp(op=la.OP_LOWRANK_RELU_GATED, layer=0, lowrank_a=_ints(2, HIDDEN, seed=9),
                        lowrank_b=_ints(HIDDEN, 2, seed=10))
        seen = []
        real_cat = torch.cat

        def spy_cat(ts, *a_, **k_):
            seen.append(len(ts))
            return real_cat(ts, *a_, **k_)
        monkeypatch.setattr(torch, "cat", spy_cat)
        la.apply_hook_ops(_ints(3, HIDDEN, seed=3), (low,), 1.0, layer=0, ctx=la.DecodeRows())
        assert seen == []                                                  # nothing gathered for a capture
        # the claim is the truth per op class, not "zero" and not "one branch" for every class
        for text in (la.GateCapture.__doc__, la.gate_capture.__doc__ or ""):
            assert "zero" not in text.lower().replace("not zero", "")
        assert "a gated-VECTOR op does one module-global lookup and one branch" in la.GateCapture.__doc__
        assert "three branches in the\n    stacked mode" in la.GateCapture.__doc__ and "B + 2 in the per-row mode" in la.GateCapture.__doc__
        assert "not zero" in la.GateCapture.__doc__.lower()

    def test_unarmed_guard_count_per_op_class_is_exactly_the_stated_one(self, monkeypatch, stacked):
        """Round 5 P2, executably: the capture guards of ``_apply_one`` (the
        ``cap = _GATE_CAPTURE`` lookups and the ``is not None`` branches),
        traced line by line while an UNARMED apply runs. A gated-VECTOR op: one
        lookup, one branch. A LOW-RANK op: one lookup and three branches in the
        stacked mode (the list guard, the latent guard inside the reduction
        closure, the record guard), B + 2 in the per-row mode — the closure runs
        once per row — and three again on the per-request (non-rowwise) path.
        The PAIRED-WRITE op (2026-09-17): one lookup and one branch of its own —
        its seam records ONE entry carrying both gates — in either mode; the
        trace below shows that third lookup line is the pair op's, not a
        duplicate of either other class's."""
        import inspect
        import sys
        src, start = inspect.getsourcelines(la._apply_one)
        lookups = {start + i for i, line in enumerate(src) if "cap = _GATE_CAPTURE" in line}
        branches = {start + i for i, line in enumerate(src) if "if cap is not None" in line or "if zs is not None" in line}
        # vector: 1 guard line; pair: 1 guard line; low-rank: 3 guard lines
        assert len(lookups) == 3 and len(branches) == 5
        fname = la._apply_one.__code__.co_filename

        def count(fn):
            hits = {"lookup": 0, "branch": 0}

            def tracer(frame, event, arg):
                if frame.f_code.co_filename != fname:
                    return None
                if event == "line":
                    if frame.f_lineno in lookups:
                        hits["lookup"] += 1
                    elif frame.f_lineno in branches:
                        hits["branch"] += 1
                return tracer
            sys.settrace(tracer)
            try:
                fn()
            finally:
                sys.settrace(None)
            return hits

        def lookup_lines(fn):
            """WHICH census lines an apply executed — the per-class attribution of the lookups."""
            hit = set()

            def tracer(frame, event, arg):
                if frame.f_code.co_filename != fname:
                    return None
                if event == "line" and frame.f_lineno in lookups:
                    hit.add(frame.f_lineno)
                return tracer
            sys.settrace(tracer)
            try:
                fn()
            finally:
                sys.settrace(None)
            return hit
        d, B = 8, 3
        rows = _ints(B, d, seed=3)
        vec = la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=_ints(d, seed=1), gate_w=_ints(d, seed=2),
                        gate_b=torch.tensor([100.0]))
        pair = la.HookOp(op=la.OP_RELU_GATED_VECTOR_PAIR, layer=0, vector=_ints(d, seed=1), gate_w=_ints(d, seed=2),
                         gate_b=torch.tensor([100.0]), vector2=_ints(d, seed=5), gate_w2=_ints(d, seed=6),
                         gate_b2=torch.tensor([100.0]))
        low = la.HookOp(op=la.OP_LOWRANK_RELU_GATED, layer=0, lowrank_a=_ints(2, d, seed=9), lowrank_b=_ints(d, 2, seed=10))
        assert la._GATE_CAPTURE is None
        # stacked mode (the served default)
        assert count(lambda: la.apply_hook_ops(rows, (vec,), 1.0, layer=0, ctx=la.DecodeRows())) == {"lookup": 1, "branch": 1}
        assert count(lambda: la.apply_hook_ops(rows, (pair,), 1.0, layer=0, ctx=la.DecodeRows())) == {"lookup": 1, "branch": 1}
        assert count(lambda: la.apply_hook_ops(rows, (low,), 1.0, layer=0, ctx=la.DecodeRows())) == {"lookup": 1, "branch": 3}
        # the census attributed: the three classes execute three DISTINCT lookup lines that
        # together are the whole census — so the third line is the pair op's own seam
        per_class = [lookup_lines(lambda op=op: la.apply_hook_ops(rows, (op,), 1.0, layer=0, ctx=la.DecodeRows()))
                     for op in (vec, pair, low)]
        assert all(len(s) == 1 for s in per_class) and len(set().union(*per_class)) == 3
        assert set().union(*per_class) == lookups
        # the per-request path (one reduction call, not rowwise): the same three
        assert count(lambda: la.apply_hook_ops(rows, (low,), 1.0, layer=0)) == {"lookup": 1, "branch": 3}
        # per-row mode: the latent guard runs once per row
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", False)
        assert count(lambda: la.apply_hook_ops(rows, (low,), 1.0, layer=0, ctx=la.DecodeRows())) == {"lookup": 1, "branch": B + 2}
        assert count(lambda: la.apply_hook_ops(rows, (vec,), 1.0, layer=0, ctx=la.DecodeRows())) == {"lookup": 1, "branch": 1}
        assert count(lambda: la.apply_hook_ops(rows, (pair,), 1.0, layer=0, ctx=la.DecodeRows())) == {"lookup": 1, "branch": 1}
        # an elementwise op touches no guard at all
        assert count(lambda: la.apply_hook_ops(rows, (_add_int_op(0),), 1.0, layer=0, ctx=la.DecodeRows())) == {"lookup": 0, "branch": 0}

    def test_wide_bf16_healthy_unequal_vector_passes_end_to_end(self, monkeypatch):
        """THE COVERAGE GAP: a 32-row bf16 decode batch of non-integer rows
        with a non-integer gated-vector op, through the worker's shadow path.
        The stacked [32, 4096] @ [4096] and the 32 single-row gemvs differ in
        fp32 rounding, so the record is genuinely UNEQUAL — observed gate
        deltas a few 2^-24 of sum|hw|, the identity within a ulp, writes exact
        wherever the gates agree — and the public rule reads ok.

        The pooled fraction is NOT a stable readout at this size: 2 groups,
        2.6e5 elements, and the flip count follows the host's fp32 summation
        order (the byte-identical file read 1.68e-4 on one CPU and below 1e-4
        on another). The case-wide 1e-4 is a many-group bound, so the rule is
        evaluated with the pooled clause at the case-appropriate scale
        (``_case_frac_tol``: the group bound, exact arithmetic), every other
        clause at its production tolerance, and the production-scale verdict
        is pinned to differ on that one clause at most — so the test reads
        the same on every CPU."""
        rec = _wide_bf16_drive(monkeypatch, kind="vector")
        assert rec["n_groups_compared"] == 2 and rec["n_unequal"] >= 1, rec["n_unequal"]
        frac_tol = _case_frac_tol(rec)
        assert frac_tol == g.REDUCTION_GROUP_FRAC_TOL                              # 2 groups: the group bound is the case bound
        tol = g.reduction_tolerance(rec, reduction_mode=STACKED, frac_tol=frac_tol)
        assert tol["ok"] is True, (tol["detail"], _host_numeric_state())
        assert tol["frac_tol"] == frac_tol and f"<= {frac_tol:g}" in tol["detail"]
        assert 0 < tol["frac_unequal_elems"] <= frac_tol, (tol["frac_unequal_elems"], _host_numeric_state())
        # what the pooled fraction IS here: every unequal element sits in a row whose observed gate differed
        assert rec["gate_n_rows_unequal"] > 0 and tol["n_unequal_elems"] <= rec["gate_n_rows_unequal"] * 4096
        _production_scale_differs_only_on_the_pooled_clause(rec, tol)
        assert tol["n_gate_groups_observed"] == rec["n_unequal"] and tol["gate_exact_rows_unequal_elems"] == 0
        assert tol["n_gate_groups_recon"] == rec["n_unequal"] and tol["gate_write_recon_mismatch_elems"] == 0
        assert rec["gate_hist_n_groups"] == 2                                     # every compared group, equal or not
        assert 0 < tol["max_gate_diff_rel_sum"] <= g.REDUCTION_GATE_REL_SUM_TOL
        # one bf16 output rounding: |delta_out| <= ulp(ref) over a unit of ulp(ref) + ulp(edit), so < 1 wherever the
        # predicted fp32 delta is under an edit ulp (production bounds it at 4; 0.941 read on the Modal CPU)
        assert tol["max_gate_edit_consistency_ulps"] <= 1.0, (tol["max_gate_edit_consistency_ulps"], _host_numeric_state())
        gate = rec["first_unequal"]["gate"]
        assert gate["gate_source"] == "observed" and gate["gate_n_rows_unequal"] >= 1
        assert "gates observed on all" in tol["detail"] and "writes exact on the" in tol["detail"]

    def test_wide_bf16_healthy_unequal_lowrank_passes_end_to_end(self, monkeypatch):
        """The low-rank pair through the same fixture; the pooled clause at
        the case-appropriate scale for the same reason as the vector case."""
        rec = _wide_bf16_drive(monkeypatch, kind="lowrank")
        assert rec["n_groups_compared"] == 2 and rec["n_unequal"] >= 1, rec["n_unequal"]
        tol = g.reduction_tolerance(rec, reduction_mode=STACKED, frac_tol=_case_frac_tol(rec))
        assert tol["ok"] is True, (tol["detail"], _host_numeric_state())
        assert 0 < tol["frac_unequal_elems"] <= tol["frac_tol"] == g.REDUCTION_GROUP_FRAC_TOL, \
            (tol["frac_unequal_elems"], _host_numeric_state())
        _production_scale_differs_only_on_the_pooled_clause(rec, tol)
        assert tol["n_gate_groups_observed"] == rec["n_unequal"]
        assert tol["max_gate_diff_rel_sum"] <= g.REDUCTION_GATE_REL_SUM_TOL
        assert tol["max_gate_edit_consistency_ulps"] <= 1.0, (tol["max_gate_edit_consistency_ulps"], _host_numeric_state())  # see the vector case
        lz = tol["max_gate_lowrank_zero_delta_ulps_edit"]
        assert lz is None or lz <= g.REDUCTION_ULPS_EDIT_TOL
        assert rec["first_unequal"]["gate"]["gate_source"] == "observed"
        assert tol["n_gate_groups_recon"] == 0 and tol["vector_gated_keys_unequal"] == []   # low-rank: never rebuilt, none needed

    def test_production_scale_companion_reads_the_rules_not_the_evidence_tail(self, monkeypatch):
        """The Modal CPU suite's intermittent failure of the low-rank case
        (2026-09-15, twice, and 2026-09-17; three trees, one node, one
        message): on that host the 2-group pooled fraction lands above the
        production 1e-4, so the production-scale verdict FAILS on the pooled
        clause — correctly — and its detail carries ``worst_brief``'s evidence
        tail, which READS the worst group's gate stats on every FAIL detail
        whatever rule fired: ``|delta gate|/sum|hw| 0.432 x 2^-24`` (bound 16)
        and ``gate-edit identity 0.941 ulps`` (bound 4; one bf16 output
        rounding). The companion scanned the whole detail for each other
        rule's signature phrase and took a reading for a rule. Replayed here
        on every host — the local record with its pooled count raised to that
        host's, everything else as measured — the companion must accept it,
        and must still refuse a record where a second RULE fires."""
        rec = _wide_bf16_drive(monkeypatch, kind="lowrank")
        other = _at_the_other_hosts_pooled_fraction(rec)
        tol = g.reduction_tolerance(other, reduction_mode=STACKED, frac_tol=_case_frac_tol(other))
        assert tol["ok"] is True, tol["detail"]
        prod = _production_scale_differs_only_on_the_pooled_clause(other, tol)
        assert prod["ok"] is False and prod["frac_unequal_elems"] > g.REDUCTION_FRAC_TOL
        # the phrases ARE in that detail — in the evidence tail, as readings inside their production bounds
        assert "|delta gate|/sum|hw|" in prod["detail"] and "gate-edit identity" in prod["detail"], prod["detail"]
        assert prod["max_gate_diff_rel_sum"] <= g.REDUCTION_GATE_REL_SUM_TOL
        assert prod["max_gate_edit_consistency_ulps"] <= g.REDUCTION_GATE_CONSISTENCY_ULPS_TOL
        # a second rule on the same record — the group bound — is still refused, by name
        worse = copy.deepcopy(other)
        key = next(k for k, e in worse["by_key"].items() if int(e.get("n_unequal") or 0) > 0)
        worse["by_key"][key]["max_frac_group"] = 2 * g.REDUCTION_GROUP_FRAC_TOL
        tol2 = g.reduction_tolerance(worse, reduction_mode=STACKED, frac_tol=_case_frac_tol(worse))
        assert tol2["ok"] is False and "in one GROUP" in tol2["detail"]
        with pytest.raises(AssertionError, match="in one GROUP"):
            _production_scale_differs_only_on_the_pooled_clause(worse, tol2)

    def test_sparse_healthy_run_is_exact_and_ok(self, monkeypatch):
        _sparse_wide(monkeypatch)
        rec = _shadow_drive(monkeypatch)
        assert rec["n_unequal"] == 0 and rec["n_groups_compared"] == 4
        assert g.reduction_tolerance(rec, reduction_mode=STACKED)["ok"] is True
