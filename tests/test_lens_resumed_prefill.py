"""Resumed prefills for per-position ops (the lens half, 2026-09-06).

vLLM keys KV blocks on token ids + cache_salt and never on the steering
selector, and cached tokens skip the forward, so a prefix cached under the
SAME cell and intervention carries exactly the edit a cold prefill would
have given it — position by position — iff every op of the cell computes
each row's edit from that row alone, at every position, in both phases.
For such cells (policy ``always``) the lens now ACCEPTS a resumed prefill
(the prompt tail after a prefix-cache hit, or the fully-cached prompt's one
scheduled last row) and applies the ops to the rows in the forward with the
bookkeeping against the FULL prompt. A ``response``-mode cell of per-position
ops (policy ``conditional``) is decided PER REQUEST: served iff the cached
region carries no assistant header, full or partial at the boundary, else a
CONTAINED per-request refusal (the reviewer's ruling, 2026-09-06). Every
other cell (policy ``never``) keeps the engine-fatal typed refusal.

Covers, CPU-only (synthetic ops, invented ids, no vLLM):

- the op-NAME partition (``PER_POSITION_OPS`` / ``BROADCAST_OPS``) is a
  partition of ``OPS`` and an unclassified name refuses;
- ``resumed_prefill_ok`` over every servable variant every REGISTERED
  adapter emits (the gate's own probes through the real adapters), pinned
  to a hand-written table so a new variant cannot land unclassified;
- the resumed ``HookContext`` shape rules;
- the apply: the resumed slice equals the tail of the cold apply exactly
  for the elementwise op and to fp32 tolerance for the reduction ops
  (a T-row and an (T-L)-row matmul reduce in different orders even on
  the CPU — the same class the stacked-reduction contract tolerates), and
  every non-resumable op meets the typed ``ResumedPrefillRefusal``;
- the lattice (``check_tracked_segment`` / ``build_segment_contexts``):
  which tracked shapes become RESUMED PREFILLS and which still refuse;
- the worker: acceptance / refusal on both paths, the end-to-end hook body
  on a resumed forward against the cold forward, the per-request
  attestation in the SelectionRecord, the install payload's per-cell map
  riding through the frozen launcher validator, and the registry helper.
"""

import sys
from pathlib import Path

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import lens_batch as lb
from sisteer.capeval.serving import lens_steer as ls
from sisteer.capeval.serving import lens_worker as lw
from sisteer.capeval.serving import vllm_server as vs

from test_petri_lens_worker import (HIDDEN, FakeLayer, FakeView, _forward,
                                    _make_steering, _obj, _registry,
                                    _row_local_op, _sel)

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

import lens_vectorize_gate as g  # noqa: E402  (the adapter probes)
assert Path(g.__file__).resolve() == (REPO / "scripts" / "lens_vectorize_gate.py").resolve(), \
    f"the bare name resolved to {g.__file__}, not the reviewed gate in scripts/ (sys.path order)"

HEADER = (900, 901, 902, 903)


def _vec(**over):
    kw = dict(op=la.OP_ADD_VECTOR, layer=0, vector=torch.ones(HIDDEN))
    kw.update(over)
    return la.HookOp(**kw)


def _gated(op=la.OP_RELU_GATED_VECTOR, seed=3, **over):
    gen = torch.Generator().manual_seed(seed)
    kw = dict(op=op, layer=0, vector=torch.randn(HIDDEN, generator=gen),
              gate_w=torch.randn(HIDDEN, generator=gen), gate_b=torch.zeros(1))
    kw.update(over)
    return la.HookOp(**kw)


def _affine(seed=5, **over):
    gen = torch.Generator().manual_seed(seed)
    R = torch.linalg.qr(torch.randn(HIDDEN, 2, generator=gen))[0].t().contiguous()
    kw = dict(op=la.OP_AFFINE_SUBSPACE, layer=0, rotate=R,
              affine_w=torch.randn(2, HIDDEN, generator=gen) * 0.1,
              affine_b=torch.randn(2, generator=gen) * 0.1)
    kw.update(over)
    return la.HookOp(**kw)


def _lowrank(op=la.OP_LOWRANK_RELU_GATED, seed=7, **over):
    gen = torch.Generator().manual_seed(seed)
    kw = dict(op=op, layer=0, lowrank_a=torch.randn(2, HIDDEN, generator=gen),
              lowrank_b=torch.randn(HIDDEN, 2, generator=gen) * 0.1)
    kw.update(over)
    return la.HookOp(**kw)


def _lora(seed=9, **over):
    gen = torch.Generator().manual_seed(seed)
    kw = dict(op=la.OP_LORA_OPROJ, layer=0, lowrank_a=torch.randn(2, HIDDEN, generator=gen),
              lowrank_b=torch.randn(HIDDEN, 2, generator=gen) * 0.1)
    kw.update(over)
    return la.HookOp(**kw)


def _alphasteer():
    return la.HookOp(op=la.OP_PREFILL_LAST_MATMUL, layer=0, matrix=torch.eye(HIDDEN),
                     phase=la.PHASE_PREFILL)


def _cosine():
    return la.HookOp(op=la.OP_PREFILL_LAST_COSINE_GATED, layer=0, vector=torch.ones(HIDDEN),
                     cond=torch.ones(HIDDEN), threshold=-1.0, positions=la.POSITIONS_LAST,
                     phase=la.PHASE_PREFILL)


def _cold_ctx(P, ids=None):
    return la.HookContext(phase=la.PHASE_PREFILL, positions=tuple(range(P)),
                          token_ids=ids)


def _resumed_ctx(P, L, ids=None):
    return la.HookContext(phase=la.PHASE_PREFILL, positions=tuple(range(L, P)),
                          token_ids=(None if ids is None else tuple(ids[L:P])), cached=L)


# --------------------------------------------------------------------------
# A — the predicate
# --------------------------------------------------------------------------


class TestPartition:
    def test_every_op_is_classified_exactly_once(self):
        assert la.PER_POSITION_OPS | la.BROADCAST_OPS == set(la.OPS)
        assert not (la.PER_POSITION_OPS & la.BROADCAST_OPS)
        for name in la.OPS:
            assert la.op_name_per_position(name) is (name in la.PER_POSITION_OPS)
        # the broadcast ops are exactly the two last-prompt-token readers
        assert la.BROADCAST_OPS == {la.OP_PREFILL_LAST_MATMUL, la.OP_PREFILL_LAST_COSINE_GATED}
        # and the partition is derived from the same vocabulary the reduction contract splits
        assert la.PER_POSITION_OPS == la.REDUCTION_OPS | {la.OP_ADD_VECTOR}
        assert la.BROADCAST_OPS == la.ELEMENTWISE_OPS - {la.OP_ADD_VECTOR}

    def test_an_unclassified_op_name_refuses(self, monkeypatch):
        """Adding an op to OPS without classifying it makes the partition pin
        above fail AND the name-level predicate refuse — never a default."""
        with pytest.raises(la.AdapterError, match="classified neither"):
            la.op_name_per_position("brand_new_op")
        monkeypatch.setattr(la, "OPS", tuple(la.OPS) + ("brand_new_op",))
        assert set(la.OPS) - (la.PER_POSITION_OPS | la.BROADCAST_OPS) == {"brand_new_op"}

    @pytest.mark.parametrize("op, ok", [
        pytest.param(_vec(), True, id="add_vector-all-both"),
        pytest.param(_vec(skip_bos=True), True, id="add_vector-all-skipbos"),
        pytest.param(_vec(positions=la.POSITIONS_LAST), False, id="add_vector-last"),
        pytest.param(_vec(positions=la.POSITIONS_RESPONSE, header_ids=HEADER), False,
                     id="add_vector-response"),
        pytest.param(_vec(phase=la.PHASE_PREFILL), False, id="add_vector-prefill-only"),
        pytest.param(_vec(phase=la.PHASE_DECODE), False, id="add_vector-decode-only"),
        pytest.param(_gated(), True, id="relu_gated-all-both"),
        pytest.param(_gated(op=la.OP_LINEAR_GATED_VECTOR), True, id="linear_gated-all-both"),
        pytest.param(_gated(positions=la.POSITIONS_RESPONSE, header_ids=HEADER), False,
                     id="relu_gated-response"),
        pytest.param(_affine(), True, id="affine-all-both"),
        pytest.param(_affine(phase=la.PHASE_PREFILL), False, id="affine-prefill-only(loreft)"),
        pytest.param(_lowrank(), True, id="lowrank_relu-all-both"),
        pytest.param(_lowrank(op=la.OP_LOWRANK_LINEAR_GATED), True, id="lowrank_linear-all-both"),
        pytest.param(_lora(), True, id="lora_oproj-all"),
        pytest.param(_lora(positions=la.POSITIONS_RESPONSE, header_ids=HEADER), False,
                     id="lora_oproj-response"),
        pytest.param(_alphasteer(), False, id="prefill_last_matmul"),
        pytest.param(_cosine(), False, id="prefill_last_cosine_gated"),
    ])
    def test_op_level_predicate(self, op, ok):
        assert la.op_resumed_prefill_ok(op) is ok
        assert la.resumed_prefill_ok(op) is ok
        assert la.resumed_prefill_ok([op]) is ok
        policy = la.op_resumed_prefill_policy(op)
        assert (policy == la.RESUMED_ALWAYS) is ok
        # the three-valued policy: response-mode per-position ops in both phases are CONDITIONAL
        want = (la.RESUMED_ALWAYS if ok else la.RESUMED_CONDITIONAL
                if (la.op_name_per_position(op.op) and op.phase == la.PHASE_BOTH
                    and op.positions == la.POSITIONS_RESPONSE) else la.RESUMED_NEVER)
        assert policy == want and policy in la.RESUMED_POLICIES

    def test_a_mix_is_resumable_only_when_every_op_is(self):
        assert la.resumed_prefill_ok([_vec(), _gated(), _lora()]) is True
        assert la.resumed_prefill_ok([_vec(), _alphasteer()]) is False
        assert la.resumed_prefill_ok([_gated(), _vec(positions=la.POSITIONS_LAST)]) is False
        assert la.resumed_prefill_ok(()) is False          # nothing to vouch for
        with pytest.raises(la.AdapterError):
            la.op_resumed_prefill_ok("add_vector")           # a name is not an op
        with pytest.raises(la.AdapterError):
            la.resumed_prefill_ok(["add_vector"])

    def test_cell_policy_is_never_over_conditional_over_always(self):
        resp = _gated(positions=la.POSITIONS_RESPONSE, header_ids=HEADER)
        assert la.resumed_prefill_policy([_vec(), _gated(), _lora()]) == la.RESUMED_ALWAYS
        assert la.resumed_prefill_policy([resp, resp]) == la.RESUMED_CONDITIONAL
        assert la.resumed_prefill_policy([_vec(), resp]) == la.RESUMED_CONDITIONAL      # any conditional op
        assert la.resumed_prefill_policy([resp, _alphasteer()]) == la.RESUMED_NEVER      # any never op wins
        assert la.resumed_prefill_policy([resp, _vec(positions=la.POSITIONS_LAST)]) == la.RESUMED_NEVER
        assert la.resumed_prefill_policy([_vec(phase=la.PHASE_PREFILL)]) == la.RESUMED_NEVER
        assert la.resumed_prefill_policy(()) == la.RESUMED_NEVER
        assert la.resumed_prefill_policy(resp) == la.RESUMED_CONDITIONAL
        assert la.resumed_prefill_ok([resp]) is False       # the bool form is "always" only
        with pytest.raises(la.AdapterError):
            la.resumed_prefill_policy(["relu_gated_vector"])
        assert set(la.RESUMED_POLICIES) == {"never", "always", "conditional"}


class TestCachedRegionHeaderConflict:
    """The conditional cell's per-request rule: the cached region [0, cached)
    must carry NO occurrence of the header — full anywhere (a prior writer's
    boundary), or partial at the cache boundary (a boundary inside the
    header)."""

    H = (900, 901, 902, 903)

    @pytest.mark.parametrize("cached, safe", [
        pytest.param((1, 2, 3, 4, 5, 6, 7), True, id="header-free"),
        pytest.param((), True, id="empty-region"),
        pytest.param((1, 2, 900, 901, 902, 903, 7), False, id="full-occurrence-inside"),
        pytest.param((900, 901, 902, 903), False, id="full-occurrence-is-the-region"),
        pytest.param((1, 2, 3, 900, 901, 902, 903), False, id="full-occurrence-at-the-tail"),
        pytest.param((1, 2, 3, 900), False, id="partial-1-of-4-at-the-boundary"),
        pytest.param((1, 2, 900, 901, 902), False, id="partial-3-of-4-at-the-boundary"),
        pytest.param((900, 901), False, id="partial-shorter-than-the-header"),
        pytest.param((1, 900, 901, 5), True, id="an-interior-partial-is-not-the-header"),
        pytest.param((1, 901, 902, 903), True, id="a-header-suffix-is-not-a-prefix"),
    ])
    def test_rule(self, cached, safe):
        why = la.cached_region_header_conflict(self.H, cached)
        assert (why is None) is safe, why
        if not safe:
            assert ("prior writer's response boundary" in why) or ("INSIDE the assistant header" in why)

    def test_empty_header_refuses_and_cell_helpers(self):
        with pytest.raises(la.AdapterError, match="non-empty header"):
            la.cached_region_header_conflict((), (1, 2))
        resp = _gated(positions=la.POSITIONS_RESPONSE, header_ids=self.H)
        assert la.response_header_ids([_vec(), resp]) == self.H
        assert la.response_header_ids([_vec(), _gated()]) is None
        assert la.resumed_header_conflict([_vec()], (900, 901, 902, 903)) is None    # no response op: nothing to conflict
        assert la.resumed_header_conflict([resp], (1, 2, 3)) is None
        assert la.resumed_header_conflict([resp], (1, 900, 901, 902, 903)) is not None
        other = _gated(positions=la.POSITIONS_RESPONSE, header_ids=(7, 8))
        with pytest.raises(la.AdapterError, match="disagree on their assistant header"):
            la.response_header_ids([resp, other])


#: THE POLICY TABLE over every servable variant the REGISTERED adapters
#: emit, keyed by the gate's case id (kind:op:shape:positions:skipbos:phase).
#: Hand-written from each kind's documented semantics, so the assertion is
#: not the predicate checked against itself; the completeness assertion
#: below makes a new servable variant fail until it is filed here.
RESUMED_POLICY_BY_CASE = {
    # per-position ops at all positions in both phases: ALWAYS resumable
    "psr:relu_gated_vector:gemv:all:skipbos0:both": "always",
    # the paired write (kind psr_paired_write, 2026-09-17): ONE op per layer carrying
    # TWO per-token ReLU gates, both read from the SAME pre-steering hidden of one row,
    # writing scale * (g_pos d_pos - g_neg d_neg) to that row alone. Each term is the
    # psr op's row-local computation and a difference of two row-local edits is row-
    # local, so the op is a PER_POSITION_OPS member; positions=all by construction (the
    # class refuses a masked parent and the adapter a masked leg, hence no `response`
    # twin here) and both phases — ALWAYS, for the psr row's reason twice over: a
    # reader resuming from the cache recovers rows this writer steered identically.
    "psr_paired_write:relu_gated_vector_pair:gemv:all:skipbos0:both": "always",
    "vector_all:add_vector:vec:all:skipbos0:both": "always",
    "vector_all:add_vector:vec:all:skipbos1:both": "always",
    "lora:lora_oproj:rank1:all:skipbos0:both": "always",
    "lora_all:lora_oproj:rankK:all:skipbos0:both": "always",
    # the LoRA-subtraction object: two lora_oproj ops per layer (+1 / -1), the same
    # per-position op as lora_all, so the same resumed-prefill policy (bcb-bfcl-200)
    "lora_subtraction:lora_oproj:rankK:all:skipbos0:both": "always",
    "alphakl:linear_gated_vector:gemv:all:skipbos0:both": "always",
    "alphakl:relu_gated_vector:gemv:all:skipbos0:both": "always",
    "lowrank_gated:lowrank_linear_gated:rank1:all:skipbos0:both": "always",
    "lowrank_gated:lowrank_linear_gated:rankK:all:skipbos0:both": "always",
    "lowrank_gated:lowrank_relu_gated:rank1:all:skipbos0:both": "always",
    "lowrank_gated:lowrank_relu_gated:rankK:all:skipbos0:both": "always",
    "loreft_both:affine_subspace:rank1:all:skipbos0:both": "always",
    "loreft_both:affine_subspace:rankK:all:skipbos0:both": "always",
    "vector:add_vector:vec:all:skipbos0:both": "always",
    "vector:add_vector:vec:all:skipbos1:both": "always",
    # the response mask, per-position ops in both phases: CONDITIONAL — decided
    # per request on the cached region being header-free (the scale-12 cells)
    "psr:relu_gated_vector:gemv:response:skipbos0:both": "conditional",
    "vector_all:add_vector:vec:response:skipbos0:both": "conditional",
    "vector_all:add_vector:vec:response:skipbos1:both": "conditional",
    "lora:lora_oproj:rank1:response:skipbos0:both": "conditional",
    "lora_all:lora_oproj:rankK:response:skipbos0:both": "conditional",
    "lora_subtraction:lora_oproj:rankK:response:skipbos0:both": "conditional",
    "vector:add_vector:vec:response:skipbos0:both": "conditional",
    "vector:add_vector:vec:response:skipbos1:both": "conditional",
    "static_r:add_vector:vec:response:skipbos0:both": "conditional",
    # positions="last": placed by the prompt length — NEVER
    "antidote:add_vector:vec:last:skipbos0:both": "never",
    "static_r:add_vector:vec:last:skipbos0:prefill": "never",
    # phase-restricted: the writer's decode rows were not edited — NEVER
    "loreft:affine_subspace:rankK:all:skipbos0:prefill": "never",
    "static_r:add_vector:vec:all:skipbos0:prefill": "never",
    # caa_static_r (its own adapter, deliberately not routed to static_r's)
    # emits OP_ADD_VECTOR at positions=all in PHASE_PREFILL, and
    # PHASE-RESTRICTION alone decides this row: `op_resumed_prefill_policy`
    # returns never for any phase != both, before positions is even looked
    # at. Same shape as the static_r row above and never for the same
    # reason — its generated rows were not edited, so a reader resuming its
    # prompt rows would not be steered the way the writer was. The
    # `resid_pre` fit site the adapter also pins is NOT the reason: it moves
    # which block the edit is served at, not which rows carry it.
    "caa_static_r:add_vector:vec:all:skipbos0:prefill": "never",
    # broadcast ops: one edit from the last prompt token — NEVER
    "alphasteer:prefill_last_matmul:dxd:all:skipbos0:prefill": "never",
    "cosine_gated_r:prefill_last_cosine_gated:other:last:skipbos0:prefill": "never",
}
#: the bool form: unconditionally resumable
RESUMED_OK_BY_CASE = {k: v == "always" for k, v in RESUMED_POLICY_BY_CASE.items()}


class TestPredicateOverTheRegisteredAdapters:
    def _probe_all(self, tmp_path):
        """(case_id -> ops) over every servable variant of every registered
        kind, through the REAL adapters (the gate's own probes)."""
        out = {}
        for arm, kind in g.servable_arms().items():
            for cid, _variant in g.probe_variants(arm, kind, len(HEADER), tmp_path / "arms"):
                _cell, ddir = g.write_synthetic_object(arm, g.INVENTORY_DIMS, tmp_path / "re" / cid.replace(":", "_"),
                                                       len(HEADER), **_variant)
                _k, ops = g.adapt_object_dir(ddir / _cell, arm=arm, header_len=len(HEADER))
                out[cid] = ops
        for kind in g.NON_PETRI_KINDS:
            for cid, _variant, ops in g.probe_kind_variants(kind, len(HEADER), tmp_path / "kinds"):
                out[cid] = ops
        return out

    def test_table_is_complete_and_the_predicate_agrees_with_it(self, tmp_path):
        probed = self._probe_all(tmp_path)
        assert set(probed) == set(RESUMED_OK_BY_CASE), (
            sorted(set(probed) ^ set(RESUMED_OK_BY_CASE)))
        for cid, ops in probed.items():
            assert la.resumed_prefill_ok(ops) is RESUMED_OK_BY_CASE[cid], cid
            assert la.resumed_prefill_policy(ops) == RESUMED_POLICY_BY_CASE[cid], cid
            # the documented positions / phase of the case id are exactly what decide it
            c = g.parse_case_id(cid)
            per_pos = la.op_name_per_position(c["op"]) and c["phase"] == "both"
            want = ("always" if per_pos and c["positions"] == "all"
                    else "conditional" if per_pos and c["positions"] == "response" else "never")
            assert RESUMED_POLICY_BY_CASE[cid] == want, cid
        assert sorted(set(RESUMED_POLICY_BY_CASE.values())) == ["always", "conditional", "never"]

    def test_internal_kinds(self):
        """The synthetic test kind and the built-in zero op are add_vector at
        every position in both phases: resumable (the scale-0 twin reads
        as its donor, which is the row-local path's whole point)."""
        zero = la.get_adapter(la.ZERO_KIND)({"kind": la.ZERO_KIND, "hidden_size": HIDDEN, "layer": 0},
                                            b"", b"")
        assert la.resumed_prefill_ok(zero) is True
        assert la.resumed_prefill_policy(zero) == la.RESUMED_ALWAYS


# --------------------------------------------------------------------------
# A' — the resumed HookContext
# --------------------------------------------------------------------------


class TestResumedContext:
    def test_resumed_shape(self):
        ctx = _resumed_ctx(9, 5, ids=list(range(200, 209)))
        assert ctx.is_resumed and ctx.cached == 5 and ctx.positions == (5, 6, 7, 8)
        assert ctx.token_ids == (205, 206, 207, 208)
        assert not _cold_ctx(9).is_resumed
        # the fully-cached prompt's one scheduled row is a one-row resumed prefill
        one = la.HookContext(phase=la.PHASE_PREFILL, positions=(8,), cached=8)
        assert one.is_resumed and one.positions == (8,)

    def test_shape_refusals(self):
        with pytest.raises(la.AdapterError, match="RESUMED prefill HookContext"):
            la.HookContext(phase=la.PHASE_PREFILL, positions=(0, 1, 2), cached=5)
        with pytest.raises(la.AdapterError, match="RESUMED prefill HookContext"):
            la.HookContext(phase=la.PHASE_PREFILL, positions=(5, 7, 8), cached=5)
        # the old complete-prefill message now names the resumed alternative
        with pytest.raises(la.AdapterError, match="cached=<n>"):
            la.HookContext(phase=la.PHASE_PREFILL, positions=(5, 6, 7))
        with pytest.raises(la.AdapterError, match="decode.*does not take cached"):
            la.HookContext(phase=la.PHASE_DECODE, positions=(9,), cached=3)
        for bad in (-1, True, 2.0, "3"):
            with pytest.raises(la.AdapterError, match="cached must be"):
                la.HookContext(phase=la.PHASE_PREFILL, positions=(0, 1), cached=bad)
        # the cached region's ids: exactly `cached` of them, on a resumed context only
        ctx = la.HookContext(phase=la.PHASE_PREFILL, positions=(3, 4), cached=3, cached_ids=[1, 2, 3])
        assert ctx.cached_ids == (1, 2, 3)
        with pytest.raises(la.AdapterError, match="mis-sized region"):
            la.HookContext(phase=la.PHASE_PREFILL, positions=(3, 4), cached=3, cached_ids=[1, 2])
        with pytest.raises(la.AdapterError, match="cached == 0"):
            la.HookContext(phase=la.PHASE_PREFILL, positions=(0, 1), cached_ids=[])
        # a decode context still refuses BOTH declarations, unchanged
        with pytest.raises(la.AdapterError, match="BOTH"):
            la.HookContext(phase=la.PHASE_PREFILL, positions=(0, 1), response_prefill=True, agent_loop=True)


# --------------------------------------------------------------------------
# A'' — the apply on a resumed slice
# --------------------------------------------------------------------------


P, L = 9, 5   # a 9-token prompt, its first 5 positions served from the cache


def _hidden(seed=11, rows=P):
    return torch.randn(rows, HIDDEN, generator=torch.Generator().manual_seed(seed))


class TestResumedApply:
    @pytest.mark.parametrize("op", [
        pytest.param(_vec(), id="add_vector"),
        pytest.param(_vec(skip_bos=True), id="add_vector-skipbos"),
    ])
    def test_elementwise_tail_is_exact(self, op):
        h = _hidden()
        cold = la.apply_hook_ops(h, [op], 1.0, layer=0, ctx=_cold_ctx(P))
        warm = la.apply_hook_ops(h[L:], [op], 1.0, layer=0, ctx=_resumed_ctx(P, L))
        assert torch.equal(warm, cold[L:])
        # the fully-cached shape: the one last row, bit-identical too
        one = la.apply_hook_ops(h[P - 1:], [op], 1.0, layer=0,
                                ctx=la.HookContext(phase=la.PHASE_PREFILL, positions=(P - 1,), cached=P - 1))
        assert torch.equal(one, cold[P - 1:])

    def test_skip_bos_masks_nothing_on_a_resumed_slice(self):
        """skip_bos is a mask on ABSOLUTE position 0, which a resumed slice
        never contains: every row of the slice is edited — the bookkeeping
        is against the full prompt, not the slice (a slice-relative mask
        would leave the first resumed row unsteered)."""
        op = _vec(skip_bos=True)
        h = _hidden()
        warm = la.apply_hook_ops(h[L:], [op], 1.0, layer=0, ctx=_resumed_ctx(P, L))
        assert torch.equal(warm, h[L:] + 1.0)
        cold = la.apply_hook_ops(h, [op], 1.0, layer=0, ctx=_cold_ctx(P))
        assert torch.equal(cold[0], h[0]) and torch.equal(cold[1:], h[1:] + 1.0)
        # the dense reference mask builder agrees (no row of the slice is masked)
        sel = la._row_selection(op, _resumed_ctx(P, L), h[L:])
        assert sel is not None and bool(sel.all())
        assert bool(la._row_mask(op, _resumed_ctx(P, L), h[L:]).all())

    @pytest.mark.parametrize("op", [
        pytest.param(_gated(), id="relu_gated"),
        pytest.param(_gated(op=la.OP_LINEAR_GATED_VECTOR), id="linear_gated"),
        pytest.param(_affine(), id="affine_subspace"),
        pytest.param(_lowrank(), id="lowrank_relu"),
        pytest.param(_lowrank(op=la.OP_LOWRANK_LINEAR_GATED), id="lowrank_linear"),
    ])
    def test_reduction_tail_matches_to_fp32_tolerance(self, op):
        """A reduction op's per-row result at M = P rows vs M = P - L rows:
        the same arithmetic, reduced in an M-dependent order (the stacked-
        reduction contract's class) — equal to fp32 tolerance, and each row's
        edit is a function of that row alone (a permuted slice gives the
        permuted result)."""
        h = _hidden()
        cold = la.apply_hook_ops(h, [op], 1.0, layer=0, ctx=_cold_ctx(P))
        warm = la.apply_hook_ops(h[L:], [op], 1.0, layer=0, ctx=_resumed_ctx(P, L))
        assert torch.allclose(warm, cold[L:], atol=1e-5, rtol=1e-5)
        assert not torch.equal(warm, h[L:])          # steering happened on the slice
        perm = torch.tensor([3, 0, 2, 1])
        again = la.apply_hook_ops(h[L:][perm], [op], 1.0, layer=0, ctx=_resumed_ctx(P, L))
        assert torch.allclose(again, warm[perm], atol=1e-5, rtol=1e-5)

    def test_oproj_tail(self):
        op = _lora()
        x = _hidden(seed=21)
        out = _hidden(seed=22)
        cold = la.apply_oproj_ops(x, out, [op], 1.0, layer=0, ctx=_cold_ctx(P))
        warm = la.apply_oproj_ops(x[L:], out[L:], [op], 1.0, layer=0, ctx=_resumed_ctx(P, L))
        assert torch.allclose(warm, cold[L:], atol=1e-5, rtol=1e-5)
        with pytest.raises(la.ResumedPrefillRefusal, match="RESUMED prefill"):
            la.apply_oproj_ops(x[L:], out[L:],
                               [_lora(positions=la.POSITIONS_RESPONSE, header_ids=HEADER)],
                               1.0, layer=0, ctx=_resumed_ctx(P, L, ids=list(range(P))))

    @pytest.mark.parametrize("op", [
        pytest.param(_vec(positions=la.POSITIONS_LAST), id="last"),
        pytest.param(_vec(phase=la.PHASE_PREFILL), id="prefill-only"),
        pytest.param(_vec(phase=la.PHASE_DECODE), id="decode-only"),
        pytest.param(_alphasteer(), id="alphasteer"),
        pytest.param(_cosine(), id="cosine"),
    ])
    def test_never_policy_ops_meet_the_typed_refusal(self, op):
        h = _hidden()
        ids = list(range(100, 100 + P - len(HEADER))) + list(HEADER)
        with pytest.raises(la.ResumedPrefillRefusal) as ei:
            la.apply_hook_ops(h[L:], [op], 1.0, layer=0, ctx=_resumed_ctx(P, L, ids=ids))
        assert isinstance(ei.value, la.AdapterError)     # contained per request by the worker
        assert "resumed_prefill_ok is False" in str(ei.value)
        # the same ops apply normally on a COLD context (no regression)
        la.apply_hook_ops(h, [op], 1.0, layer=0, ctx=_cold_ctx(P, ids=ids))

    def test_mask_builders_refuse_last_under_resume(self):
        ctx = _resumed_ctx(P, L, ids=list(range(P)))
        op = _vec(positions=la.POSITIONS_LAST)
        with pytest.raises(la.ResumedPrefillRefusal, match="RESUMED prefill"):
            la._row_selection(op, ctx, _hidden(rows=P - L))
        with pytest.raises(la.ResumedPrefillRefusal, match="RESUMED prefill"):
            la._row_mask(op, ctx, _hidden(rows=P - L))


def _resumed_ctx_region(P, L, ids):
    """A resumed context that also carries the cached region's ids (what the
    worker builds for a CONDITIONAL cell it decided to serve)."""
    return la.HookContext(phase=la.PHASE_PREFILL, positions=tuple(range(L, P)),
                          token_ids=tuple(ids[L:P]), cached=L, cached_ids=tuple(ids[:L]))


class TestConditionalApply:
    """A response-mode op under a resumed context: the locator VERIFIES the
    cached region header-free from ``cached_ids`` (never trusts a flag),
    then masks from the header located in the slice — the tail of the cold
    apply."""

    @pytest.mark.parametrize("op", [
        pytest.param(_vec(positions=la.POSITIONS_RESPONSE, header_ids=HEADER), id="add_vector-response"),
        pytest.param(_gated(positions=la.POSITIONS_RESPONSE, header_ids=HEADER), id="gated-response"),
    ])
    def test_safe_resume_is_the_tail_of_the_cold_apply(self, op):
        ids = list(range(100, 100 + P - len(HEADER))) + list(HEADER)   # header at the tail
        h = _hidden()
        cold = la.apply_hook_ops(h, [op], 1.0, layer=0, ctx=_cold_ctx(P, ids=ids))
        # the cached region [0, L) is header-free (L = 5 <= header start = 5): served
        warm = la.apply_hook_ops(h[L:], [op], 1.0, layer=0, ctx=_resumed_ctx_region(P, L, ids))
        if op.op == la.OP_ADD_VECTOR:
            assert torch.equal(warm, cold[L:])
        else:
            assert torch.allclose(warm, cold[L:], atol=1e-5, rtol=1e-5)
        # the mask: only the header rows of the slice are edited (positions 5..8 = the header)
        sel = la._row_selection(op, _resumed_ctx_region(P, L, ids), h[L:])
        assert sel.tolist() == [True, True, True, True]
        # a smaller cached region: the slice holds pre-header rows too, unedited
        ctx3 = _resumed_ctx_region(P, 3, ids)
        assert la._row_selection(op, ctx3, h[3:]).tolist() == [False, False, True, True, True, True]
        assert torch.equal(la._row_mask(op, ctx3, h[3:]), la._row_selection(op, ctx3, h[3:]))

    def test_unsafe_regions_refuse_contained(self):
        op = _gated(positions=la.POSITIONS_RESPONSE, header_ids=HEADER)
        h = _hidden()
        # the cache boundary INSIDE the header (cached = 6: header[:1] at the boundary)
        ids = list(range(100, 105)) + list(HEADER)
        with pytest.raises(la.ResumedPrefillRefusal, match="INSIDE the assistant header"):
            la.apply_hook_ops(h[6:], [op], 1.0, layer=0, ctx=_resumed_ctx_region(P, 6, ids))
        # an earlier header inside the cached region (a prior writer's boundary)
        ids2 = [100] + list(HEADER) + [105, 106, 107, 108]
        ctx2 = la.HookContext(phase=la.PHASE_PREFILL, positions=(7, 8), token_ids=(107, 108), cached=7,
                              cached_ids=tuple(ids2[:7]))
        with pytest.raises(la.ResumedPrefillRefusal, match="prior writer's response boundary"):
            la.apply_hook_ops(h[7:], [op], 1.0, layer=0, ctx=ctx2)
        # no cached ids at all: the region cannot be verified — refused, never assumed safe
        with pytest.raises(la.ResumedPrefillRefusal, match="no cached_ids"):
            la.apply_hook_ops(h[L:], [op], 1.0, layer=0, ctx=_resumed_ctx(P, L, ids=ids))
        for bad in (ctx2, _resumed_ctx(P, L, ids=ids)):
            with pytest.raises(la.ResumedPrefillRefusal):
                la._row_selection(op, bad, _hidden(rows=P - bad.cached))
        # a NEVER op in the same cell still refuses regardless of the region
        with pytest.raises(la.ResumedPrefillRefusal, match="resumed_prefill_ok is False"):
            la.apply_hook_ops(h[L:], [_alphasteer()], 1.0, layer=0, ctx=_resumed_ctx_region(P, L, ids))

    def test_scale_zero_and_no_op_layers_stay_bit_identical(self):
        h = _hidden(rows=P - L)
        ctx = _resumed_ctx(P, L)
        assert la.apply_hook_ops(h, [_vec()], 0.0, layer=0, ctx=ctx) is h
        assert la.apply_hook_ops(h, [_vec(layer=3)], 1.0, layer=0, ctx=ctx) is h
        # a non-resumable op at ANOTHER layer is not consulted at this layer
        assert la.apply_hook_ops(h, [_alphasteer().__class__(**{**_alphasteer().__dict__, "layer": 2})],
                                 1.0, layer=0, ctx=ctx) is h


# --------------------------------------------------------------------------
# B — the lattice
# --------------------------------------------------------------------------


def _seg(q, computed, start=0):
    return lb.Segment(start=start, end=start + q, q=q, computed=computed,
                      seq_len=computed + q)


class TestTrackedSegmentClasses:
    def test_resumed_slice_ending_at_the_prompt_end(self):
        seg = _seg(4, 5)   # positions 5..8 of a 9-token prompt
        assert lb.check_tracked_segment(seg, prompt_len=9, req_id="r", resumed_ok=True) == lb.SEG_RESUMED_PREFILL
        with pytest.raises(lb.LensBatchError, match="(?i)cannot resume"):
            lb.check_tracked_segment(seg, prompt_len=9, req_id="r", resumed_ok=False)
        with pytest.raises(lb.LensBatchError, match="(?i)cannot resume"):
            lb.check_tracked_segment(seg, prompt_len=9, req_id="r")        # the default

    def test_resumed_slice_overrunning_the_prompt_is_a_recompute(self):
        # positions 5..8 of a 7-token prompt: rows 7, 8 are GENERATED tokens
        with pytest.raises(lb.LensBatchError, match="preemption recompute under caching"):
            lb.check_tracked_segment(_seg(4, 5), prompt_len=7, req_id="r", resumed_ok=True)

    def test_resumed_slice_stopping_short_is_a_chunk(self):
        with pytest.raises(lb.LensBatchError, match="chunked-prefill continuation"):
            lb.check_tracked_segment(_seg(4, 5), prompt_len=12, req_id="r", resumed_ok=True)

    def test_fully_cached_prompt(self):
        # q == 1 at computed == N-1: the fully-cached prompt's last row
        assert lb.check_tracked_segment(_seg(1, 8), prompt_len=9, req_id="r",
                                        resumed_ok=True) == lb.SEG_RESUMED_PREFILL
        with pytest.raises(lb.LensBatchError, match="(?i)INSIDE.*cannot resume"):
            lb.check_tracked_segment(_seg(1, 8), prompt_len=9, req_id="r")
        # a decode-shaped row deeper inside the prompt is a chunk remnant either way
        with pytest.raises(lb.LensBatchError, match="INSIDE.*exactly the last prompt token"):
            lb.check_tracked_segment(_seg(1, 4), prompt_len=9, req_id="r", resumed_ok=True)
        with pytest.raises(lb.LensBatchError, match="INSIDE"):
            lb.check_tracked_segment(_seg(1, 4), prompt_len=9, req_id="r")

    def test_the_other_classes_are_unchanged(self):
        assert lb.check_tracked_segment(_seg(9, 0), prompt_len=9, req_id="r", resumed_ok=True) == lb.SEG_PREFILL
        assert lb.check_tracked_segment(_seg(1, 9), prompt_len=9, req_id="r", resumed_ok=True) == lb.SEG_DECODE
        assert lb.check_tracked_segment(_seg(1, 40), prompt_len=9, req_id="r") == lb.SEG_DECODE
        with pytest.raises(lb.LensBatchError, match="preemption"):
            lb.check_tracked_segment(_seg(12, 0), prompt_len=9, req_id="r", resumed_ok=True)
        with pytest.raises(lb.LensBatchError, match="SHORTER"):
            lb.check_tracked_segment(_seg(5, 0), prompt_len=9, req_id="r", resumed_ok=True)
        assert set(lb.SEGMENT_CLASSES) == {"prefill", "decode", "resumed_prefill"}


class TestResumedContexts:
    def test_filed_resumed_segment_gets_a_resumed_prefill_context(self):
        # one complete prefill (rows 0..4), one resumed hit (5 cached, rows at 5..8), one decode
        qsl, seqs, nat = [0, 5, 9, 10], [5, 9, 20], 10
        plan = lb.derive_batch_plan(qsl, seqs, nat, allow_resumed=True)
        pos = list(range(5)) + [5, 6, 7, 8] + [19]
        ids = list(range(100, 105)) + [205, 206, 207, 208] + [319]
        ctxs = lb.build_segment_contexts(plan, positions=pos, token_ids=ids, resumed_prefill=[1])
        assert ctxs[0].ctx.positions == (0, 1, 2, 3, 4) and ctxs[0].ctx.cached == 0
        r = ctxs[1].ctx
        assert r.phase == la.PHASE_PREFILL and r.is_resumed and r.cached == 5
        assert r.positions == (5, 6, 7, 8) and r.token_ids == (205, 206, 207, 208)
        assert ctxs[2].ctx.phase == la.PHASE_DECODE and ctxs[2].ctx.positions == (19,)
        # not filed: the pre-landing None (a selector-less hit)
        assert lb.build_segment_contexts(plan, positions=pos, token_ids=ids)[1].ctx is None

    def test_cached_ids_ride_into_the_filed_context(self):
        qsl, seqs, nat = [0, 5, 9], [5, 9], 9
        plan = lb.derive_batch_plan(qsl, seqs, nat, allow_resumed=True)
        pos = list(range(5)) + [5, 6, 7, 8]
        ids = list(range(100, 105)) + [205, 206, 207, 208]
        ctxs = lb.build_segment_contexts(plan, positions=pos, token_ids=ids, resumed_prefill=[1],
                                         cached_ids={1: [200, 201, 202, 203, 204]})
        assert ctxs[1].ctx.cached_ids == (200, 201, 202, 203, 204) and ctxs[0].ctx.cached_ids is None
        with pytest.raises(lb.LensBatchError, match="unattached cached region"):
            lb.build_segment_contexts(plan, positions=pos, token_ids=ids, resumed_prefill=[1],
                                      cached_ids={0: [1, 2, 3, 4, 5]})
        with pytest.raises(la.AdapterError, match="mis-sized"):
            lb.build_segment_contexts(plan, positions=pos, token_ids=ids, resumed_prefill=[1],
                                      cached_ids={1: [1, 2]})

    def test_fully_cached_row_filed_as_a_one_row_resumed_prefill(self):
        qsl, seqs, nat = [0, 1], [9], 1     # q=1 at position 8 of a 9-prompt
        plan = lb.derive_batch_plan(qsl, seqs, nat, allow_resumed=True)
        assert plan.n_decode == 1 and plan.n_resumed == 0   # decode-SHAPED to the lattice
        ctx = lb.build_segment_contexts(plan, positions=[8], token_ids=[208], resumed_prefill=[0])[0].ctx
        assert ctx.is_resumed and ctx.positions == (8,) and ctx.cached == 8 and ctx.token_ids == (208,)

    def test_filing_refusals(self):
        qsl, seqs, nat = [0, 5], [5], 5
        plan = lb.derive_batch_plan(qsl, seqs, nat, allow_resumed=True)
        with pytest.raises(lb.LensBatchError, match="COMPLETE prefill"):
            lb.build_segment_contexts(plan, positions=list(range(5)), resumed_prefill=[0])
        with pytest.raises(lb.LensBatchError, match="names segment 3"):
            lb.build_segment_contexts(plan, positions=list(range(5)), resumed_prefill=[3])
        # the positions cross-check still runs first on a filed resume
        qsl, seqs, nat = [0, 4], [9], 4
        plan = lb.derive_batch_plan(qsl, seqs, nat, allow_resumed=True)
        with pytest.raises(lb.LensBatchError, match="positions tensor"):
            lb.build_segment_contexts(plan, positions=[0, 1, 2, 3], resumed_prefill=[0])


# --------------------------------------------------------------------------
# C — the worker
# --------------------------------------------------------------------------


def _skipbos_op(layer=1, value=1.0):
    """positions=all, both phases, skip_bos — plan-needing AND resumable."""
    return la.HookOp(op=la.OP_ADD_VECTOR, layer=layer,
                     vector=torch.full((HIDDEN,), float(value)), skip_bos=True)


def _last_op(layer=1):
    return la.HookOp(op=la.OP_ADD_VECTOR, layer=layer, vector=torch.ones(HIDDEN),
                     positions=la.POSITIONS_LAST)


def _slots(arm, segs, tokens, prompt_lens, *, extra=None, allow_resumed=True):
    qsl, seq_lens, nat, pos = _forward(segs)
    reg = _registry(arm, *(extra or ()))
    sels = [(_sel(o, scale=(0.0 if o.effective_scale == 0.0 else None)) if o is not None else None)
            for o in [arm] + list(extra or ())][:len(tokens)]
    return lw.build_request_slots(
        query_start_loc=qsl, seq_lens=seq_lens, num_actual_tokens=nat, positions=pos,
        req_ids=list(tokens), raw_selectors=sels,
        token_lookup=tokens.__getitem__, prompt_len_lookup=prompt_lens.__getitem__,
        registry=reg, allow_resumed=allow_resumed)


class TestBuildRequestSlotsResumed:
    def test_tracked_resumable_resume_is_a_resumed_prefill_slot(self):
        arm = _obj("arm-skipbos", [_skipbos_op()])
        fwd = _slots(arm, [(4, 5)], {"steered": list(range(200, 209))}, {"steered": 9})
        (slot,) = fwd.slots
        assert slot.ctx.is_resumed and slot.ctx.positions == (5, 6, 7, 8) and slot.ctx.cached == 5
        assert slot.ctx.token_ids == (205, 206, 207, 208)
        assert slot.resumed == lw.ResumedPrefill(cached=5, computed=4)
        # the ledger projection: no complete-prefill credit, one resumed
        assert fwd.tracked_plan.full_prefills == () and fwd.tracked_plan.n_decode == 0
        assert fwd.tracked_plan.n_resumed == 1

    def test_tracked_non_resumable_resume_refuses(self):
        arm = _obj("arm-last", [_last_op()])
        with pytest.raises(lb.LensBatchError, match="(?i)cannot resume"):
            _slots(arm, [(4, 5)], {"steered": list(range(200, 209))}, {"steered": 9})

    def test_fully_cached_prompt_of_a_resumable_cell(self):
        arm = _obj("arm-skipbos", [_skipbos_op()])
        fwd = _slots(arm, [(1, 8)], {"steered": list(range(200, 209))}, {"steered": 9})
        (slot,) = fwd.slots
        assert slot.ctx.is_resumed and slot.ctx.positions == (8,) and slot.ctx.cached == 8
        assert slot.resumed == lw.ResumedPrefill(cached=8, computed=1)
        # ... and a GENUINE decode step of the same request is a decode context, no facts
        fwd = _slots(arm, [(1, 9)], {"steered": list(range(200, 210))}, {"steered": 9})
        assert fwd.slots[0].ctx.phase == la.PHASE_DECODE and fwd.slots[0].resumed is None
        # a one-row resumed prefill stays PER SLOT (never a decode group)
        groups, per_slot = lw.build_slot_groups(_slots(arm, [(1, 8)], {"steered": list(range(200, 209))},
                                                       {"steered": 9}))
        assert groups == () and len(per_slot) == 1

    def test_untracked_selector_carrying_resume_on_the_plan_path(self):
        """A row-local (untracked) object co-scheduled with a tracked one:
        served with recorded facts when resumable; the scale-0 twin of a
        non-resumable kind in the same forward still refuses — the salt
        discipline holds on the plan path too."""
        tracked = _obj("arm-skipbos", [_skipbos_op()])
        rowlocal = _obj("arm-rowlocal", [_row_local_op(layer=1)])
        fwd = _slots(tracked, [(9, 0), (4, 5)],
                     {"steered": list(range(100, 109)), "other": list(range(200, 209))},
                     {"steered": 9, "other": 9}, extra=[rowlocal])
        assert fwd.slots[0].resumed is None and fwd.slots[0].ctx.cached == 0
        assert fwd.slots[1].ctx is None        # a row-local op needs no context
        assert fwd.slots[1].resumed == lw.ResumedPrefill(cached=5, computed=4)
        twin = _obj("static__scale0", [la.HookOp(op=la.OP_ADD_VECTOR, layer=1, vector=torch.ones(HIDDEN),
                                                  phase=la.PHASE_PREFILL)], scale=0.0)
        with pytest.raises(lw.LensWorkerError, match="(?i)cannot resume"):
            _slots(tracked, [(9, 0), (4, 5)],
                   {"steered": list(range(100, 109)), "twin": list(range(200, 209))},
                   {"steered": 9, "twin": 9}, extra=[twin])

    def test_caching_off_engine_is_unchanged(self):
        arm = _obj("arm-skipbos", [_skipbos_op()])
        with pytest.raises(lb.LensBatchError, match="prefix-cache hit"):
            _slots(arm, [(4, 5)], {"steered": list(range(200, 209))}, {"steered": 9}, allow_resumed=False)


def _response_op(layer=1):
    return la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=layer, vector=torch.ones(HIDDEN),
                     gate_w=torch.ones(HIDDEN), gate_b=torch.zeros(1),
                     positions=la.POSITIONS_RESPONSE, header_ids=HEADER)


#: a 12-token single-turn prompt: 8 content ids then the 4-id header at the tail
SINGLE_TURN = list(range(100, 108)) + list(HEADER)
#: a 14-token two-turn prompt: content, header (a prior turn's boundary), more
#: content, the final header — the multi-turn shape
TWO_TURN = [100, 101] + list(HEADER) + [106, 107, 108, 109] + list(HEADER)

#: THE RULING'S TABLE: (policy, prompt, (q, computed), expected) where expected is
#: "served" (a filed, served resumed prefill), "contained" (a per-request
#: ResumedPrefillRefusal on the slot, no context, the engine untouched),
#: "fatal" (the engine-fatal salt-discipline refusal), "decode" / "prefill"
#: (not a resume shape).
RESUME_TABLE = [
    # conditional (response-mode op), one final header at 8..11
    pytest.param("conditional", SINGLE_TURN, (4, 8), "served", id="cond-cached<=header_start-served"),
    pytest.param("conditional", SINGLE_TURN, (7, 5), "served", id="cond-cached-deep-before-header-served"),
    pytest.param("conditional", SINGLE_TURN, (3, 9), "contained", id="cond-boundary-inside-header-1of4"),
    pytest.param("conditional", SINGLE_TURN, (1, 11), "contained", id="cond-boundary-inside-header-3of4"),
    pytest.param("conditional", TWO_TURN, (4, 10), "contained", id="cond-full-earlier-header-cached"),
    pytest.param("conditional", TWO_TURN, (8, 6), "contained", id="cond-earlier-header-is-the-region-tail"),
    pytest.param("conditional", TWO_TURN, (10, 4), "contained", id="cond-partial-earlier-header-2of4"),
    pytest.param("conditional", TWO_TURN, (12, 2), "served", id="cond-two-turn-cached-before-any-header"),
    pytest.param("conditional", SINGLE_TURN, (12, 0), "prefill", id="cond-complete-prefill"),
    pytest.param("conditional", SINGLE_TURN, (1, 12), "decode", id="cond-decode-step"),
    # always: served whatever the region holds
    pytest.param("always", SINGLE_TURN, (3, 9), "served", id="always-boundary-inside-header-served"),
    pytest.param("always", TWO_TURN, (4, 10), "served", id="always-earlier-header-served"),
    pytest.param("always", SINGLE_TURN, (1, 11), "served", id="always-fully-cached-served"),
    # never: the engine-fatal refusal, unchanged
    pytest.param("never", SINGLE_TURN, (4, 8), "fatal", id="never-refuses-fatal"),
    pytest.param("never", TWO_TURN, (12, 2), "fatal", id="never-refuses-fatal-two-turn"),
]


def _cell_for(policy):
    if policy == "conditional":
        return _obj("cell-response", [_response_op()])
    if policy == "always":
        return _obj("cell-skipbos", [_skipbos_op()])
    return _obj("cell-last", [_last_op()])


class TestConditionalResumeTable:
    """The per-request decision for a CONDITIONAL (response-mode) cell,
    table-driven (the reviewer's pins), next to the unchanged always / never
    behaviour — through build_request_slots, on the plan path."""

    @pytest.mark.parametrize("policy, prompt, shape, expected", RESUME_TABLE)
    def test_build_request_slots(self, policy, prompt, shape, expected):
        arm = _cell_for(policy)
        assert la.resumed_prefill_policy(arm.ops) == policy
        q, computed = shape
        n = len(prompt)
        tokens = {"r": list(prompt) + ([999] if computed >= n else [])}   # a decode step knows one generated id
        if expected == "fatal":
            with pytest.raises((lb.LensBatchError, lw.LensWorkerError)) as ei:
                _slots(arm, [shape], tokens, {"r": n})
            assert "resumed_prefill_ok is False" in str(ei.value)
            return
        fwd = _slots(arm, [shape], tokens, {"r": n})
        (slot,) = fwd.slots
        if expected == "served":
            assert slot.refusal is None
            assert slot.ctx is not None and slot.ctx.is_resumed and slot.ctx.cached == computed
            assert slot.ctx.positions == tuple(range(computed, computed + q))
            assert slot.resumed == lw.ResumedPrefill(cached=computed, computed=q)
            if policy == "conditional":
                assert slot.ctx.cached_ids == tuple(prompt[:computed])   # the region the op re-verifies
                # and the apply on that context serves the response mask from the header in the slice
                out = la.apply_hook_ops(torch.zeros(q, HIDDEN), arm.ops, 1.0, layer=1, ctx=slot.ctx)
                assert out.shape == (q, HIDDEN)
            else:
                assert slot.ctx.cached_ids is None
        elif expected == "contained":
            assert isinstance(slot.refusal, la.ResumedPrefillRefusal)
            assert isinstance(slot.refusal, la.AdapterError)
            assert "resumed_prefill_policy is conditional" in str(slot.refusal)
            assert ("prior writer's response boundary" in str(slot.refusal)
                    or "INSIDE the assistant header" in str(slot.refusal))
            assert slot.ctx is None                       # nothing is applied to a refused request
            assert slot.resumed == lw.ResumedPrefill(cached=computed, computed=q)   # the shape is still recorded
        elif expected == "prefill":
            assert slot.refusal is None and slot.resumed is None and slot.ctx.cached == 0
        else:
            assert slot.refusal is None and slot.resumed is None and slot.ctx.phase == la.PHASE_DECODE

    def test_decision_on_the_untracked_paths_too(self):
        """A conditional cell's scale-0 twin (untracked: scale 0) on the
        row-local path, and co-scheduled with a tracked request on the plan
        path: the same per-request rule, contained."""
        twin = _obj("cell-response__scale0", [_response_op()], scale=0.0)
        n = len(SINGLE_TURN)
        # row-local path (only this request): safe -> served, unsafe -> contained
        fwd = _slots(twin, [(4, 8)], {"t": list(SINGLE_TURN)}, {"t": n})
        assert fwd.slots[0].refusal is None and fwd.slots[0].resumed == lw.ResumedPrefill(cached=8, computed=4)
        fwd = _slots(twin, [(3, 9)], {"t": list(SINGLE_TURN)}, {"t": n})
        assert isinstance(fwd.slots[0].refusal, la.ResumedPrefillRefusal)
        # plan path, untracked next to a tracked cold prefill
        tracked = _obj("arm-skipbos", [_skipbos_op()])
        fwd = _slots(tracked, [(9, 0), (3, 9)], {"s": list(range(300, 309)), "t": list(SINGLE_TURN)},
                     {"s": 9, "t": n}, extra=[twin])
        assert fwd.slots[0].refusal is None and isinstance(fwd.slots[1].refusal, la.ResumedPrefillRefusal)
        fwd = _slots(tracked, [(9, 0), (4, 8)], {"s": list(range(300, 309)), "t": list(SINGLE_TURN)},
                     {"s": 9, "t": n}, extra=[twin])
        assert fwd.slots[1].refusal is None and fwd.slots[1].resumed == lw.ResumedPrefill(cached=8, computed=4)

    def test_resume_decision_kinds(self):
        arm = _cell_for("conditional")
        d = lw.resume_decision(arm, computed=8, q=4, seq_len=12, prompt_len=12, tokens=SINGLE_TURN, req_id="r")
        assert d.kind == lw.RESUME_SERVE and d.policy == "conditional" and d.cached_ids == tuple(SINGLE_TURN[:8])
        d = lw.resume_decision(arm, computed=9, q=3, seq_len=12, prompt_len=12, tokens=SINGLE_TURN, req_id="r")
        assert d.kind == lw.RESUME_REFUSE_CONTAINED and d.reason and d.refusal is not None
        assert d.facts == lw.ResumedPrefill(cached=9, computed=3)
        assert lw.resume_decision(arm, computed=0, q=12, seq_len=12, prompt_len=12, tokens=SINGLE_TURN,
                                  req_id="r").kind == lw.RESUME_NOT_A_RESUME
        assert lw.resume_decision(arm, computed=12, q=1, seq_len=13, prompt_len=12, tokens=SINGLE_TURN + [1],
                                  req_id="r").kind == lw.RESUME_NOT_A_RESUME
        assert lw.resume_decision(_cell_for("never"), computed=8, q=4, seq_len=12, prompt_len=12,
                                  tokens=SINGLE_TURN, req_id="r").kind == lw.RESUME_REFUSE_FATAL
        d = lw.resume_decision(_cell_for("always"), computed=9, q=3, seq_len=12, prompt_len=12,
                               tokens=SINGLE_TURN, req_id="r")
        assert d.kind == lw.RESUME_SERVE and d.cached_ids is None
        with pytest.raises(lw.LensWorkerError, match="behind the cache state"):
            lw.resume_decision(arm, computed=8, q=4, seq_len=12, prompt_len=12, tokens=SINGLE_TURN[:5], req_id="r")


class TestContainedRefusalEndToEnd:
    def test_refused_request_serves_unsteered_and_the_engine_keeps_serving(self):
        """The conditional cell's unsafe resume through the hook body: the
        request is failed (recorded, claimable by the proxy as a
        ResumedPrefillRefusal), its rows stay bit-identical, the worker is
        NOT condemned, and the next request — a safe resume of the same cell
        — serves and attests resumed."""
        arm = _obj("cell-response", [_response_op(layer=1)])
        steering, view = _two_layer_steering(arm)
        n = len(SINGLE_TURN)
        # unsafe: the cache boundary inside the header (q=3, computed=9)
        h = _hidden(seed=41, rows=3)
        out = _run(steering, view, arm, [(3, 9)], {"unsafe": list(SINGLE_TURN)}, h.clone(), prompt_lens={"unsafe": n})
        assert out is None                                    # nothing applied: a bitwise passthrough
        rec = steering.take_request_failure("unsafe")
        assert rec is not None and rec["error_type"] == "ResumedPrefillRefusal" and rec["cell_id"] == "cell-response"
        assert rec["layer"] == 1 and rec["site"] == la.HOOK_SITE_BLOCK
        assert "resumed_prefill_policy is conditional" in rec["error"]
        assert steering.take_request_failure("unsafe") is None   # claimed once
        # the engine is alive: a SAFE resume of the same cell serves and attests
        h2 = _hidden(seed=42, rows=4)
        new_h, _ = _run(steering, view, arm, [(4, 8)], {"safe": list(SINGLE_TURN)}, h2.clone(), prompt_lens={"safe": n})
        assert not torch.equal(new_h, h2)                     # the header rows were steered
        e = steering.selection_record.as_dict()["requests"]["safe"]
        assert e["prefill_resumed"] is True and e["prefill_cached_positions"] == 8 and e["prefill_computed_positions"] == 4
        assert e["n_applications_modifying_resumed"] == 1
        assert "unsafe" not in steering.selection_record.as_dict()["requests"]   # a refused request is never observed as applied
        # and a cold complete prefill of the same cell still serves
        h3 = _hidden(seed=43, rows=n)
        new_h3, _ = _run(steering, view, arm, [(n, 0)], {"cold": list(SINGLE_TURN)}, h3.clone())
        assert torch.equal(new_h3[:8], h3[:8]) and not torch.equal(new_h3[8:], h3[8:])

    def test_refusal_is_recorded_once_per_forward_and_skipped_at_every_layer(self):
        arm = _obj("cell-response", [_response_op(layer=1), _response_op(layer=2)])
        view = FakeView(layers=[FakeLayer() for _ in range(3)], prefix_caching=True)
        steering = lw.LensWorkerSteering(view=view, registry=_registry(arm))
        steering.preflight(object(), header_verifier=lambda tok, rec: None)
        steering.install()
        n = len(TWO_TURN)
        qsl, seq_lens, nat, pos = _forward([(4, 10)])
        view.set_batch(["r"], {"r": _sel(arm)}, {"r": list(TWO_TURN)}, (qsl, seq_lens, nat), prompt_lens={"r": n})
        h, r = _hidden(seed=44, rows=4), torch.zeros(4, HIDDEN)
        assert steering.hook_body(1, (pos, h, r), (h, r)) is None
        assert steering.hook_body(2, (pos, h, r), (h, r)) is None    # the cached classification: no re-record
        assert len(steering._request_failures) == 1 and "r" in steering._failed_req_ids
        rec = steering.take_request_failure("r")
        assert rec["error_type"] == "ResumedPrefillRefusal" and "prior writer's response boundary" in rec["error"]
        steering._require_alive()                                    # never condemned


def _two_layer_steering(arm, n_layers=3):
    view = FakeView(layers=[FakeLayer() for _ in range(n_layers)], prefix_caching=True)
    steering = lw.LensWorkerSteering(view=view, registry=_registry(arm))
    steering.preflight(object(), header_verifier=lambda tok, rec: None)
    steering.install()
    return steering, view


def _run(steering, view, arm, segs, tokens, h, *, layer=1, prompt_lens=None, sel=None):
    qsl, seq_lens, nat, pos = _forward(segs)
    view.set_batch(list(tokens), {rid: (sel or _sel(arm)) for rid in tokens}, tokens,
                   (qsl, seq_lens, nat), prompt_lens=prompt_lens)
    r = torch.zeros(nat, HIDDEN)
    out = steering.hook_body(layer, (pos, h, r), (h, r))
    return out


class TestHookBodyOnAResumedForward:
    def test_resumed_forward_equals_the_tail_of_the_cold_forward(self):
        """The same 9-token prompt served cold (one complete prefill) and
        warm (5 positions cached, rows 5..8 in the forward): the warm hook
        output is bit-identical to rows 5..8 of the cold output, and the
        SelectionRecord attests the resume."""
        arm = _obj("arm-skipbos", [_skipbos_op(layer=1, value=0.25)])
        steering, view = _two_layer_steering(arm)
        prompt = list(range(200, 209))
        h_full = _hidden(seed=31, rows=9)
        cold = _run(steering, view, arm, [(9, 0)], {"cold": prompt}, h_full.clone())
        new_cold, _ = cold
        assert torch.equal(new_cold[0], h_full[0])                 # BOS skipped
        assert torch.equal(new_cold[1:], h_full[1:] + 0.25)
        warm = _run(steering, view, arm, [(4, 5)], {"warm": prompt}, h_full[5:].clone(),
                    prompt_lens={"warm": 9})
        new_warm, _ = warm
        assert torch.equal(new_warm, new_cold[5:])
        snap = steering.selection_record.as_dict()
        assert snap["requests"]["cold"]["prefill_resumed"] is False
        assert snap["requests"]["cold"]["prefill_cached_positions"] == 0
        w = snap["requests"]["warm"]
        assert w["prefill_resumed"] is True
        assert w["prefill_cached_positions"] == 5 and w["prefill_computed_positions"] == 4
        assert w["n_applications_modifying"] == 1 and w["n_applications_modifying_resumed"] == 1
        assert snap["n_requests_resumed_prefill"] == 1
        assert "prefill_resumed" in lw.SELECTION_RECORD_COVERAGE

    def test_fully_cached_row_is_steered_as_the_last_prompt_row(self):
        arm = _obj("arm-skipbos", [_skipbos_op(layer=1, value=0.25)])
        steering, view = _two_layer_steering(arm)
        prompt = list(range(200, 209))
        h_full = _hidden(seed=32, rows=9)
        new_cold, _ = _run(steering, view, arm, [(9, 0)], {"cold": prompt}, h_full.clone())
        new_one, _ = _run(steering, view, arm, [(1, 8)], {"one": prompt}, h_full[8:].clone(),
                          prompt_lens={"one": 9})
        assert torch.equal(new_one, new_cold[8:])
        e = steering.selection_record.as_dict()["requests"]["one"]
        assert e["prefill_resumed"] and e["prefill_cached_positions"] == 8 and e["prefill_computed_positions"] == 1

    def test_non_resumable_cell_resume_condemns_the_worker(self):
        """The engine-side salt-discipline backstop is UNCHANGED for a cell
        that cannot resume: the refusal propagates and condemns."""
        arm = _obj("arm-last", [_last_op(layer=1)])
        steering, view = _two_layer_steering(arm)
        prompt = list(range(200, 209))
        with pytest.raises(lb.LensBatchError, match="(?i)cannot resume"):
            _run(steering, view, arm, [(4, 5)], {"warm": prompt}, _hidden(rows=4), prompt_lens={"warm": 9})
        with pytest.raises(lw.LensWorkerError, match="CONDEMNED"):
            _run(steering, view, arm, [(9, 0)], {"cold": prompt}, _hidden(rows=9))

    def test_mixed_forward_isolation(self):
        """A resumed steered slice next to an unhooked complete prefill: the
        unhooked rows stay bit-identical, the resumed rows get their edit."""
        arm = _obj("arm-skipbos", [_skipbos_op(layer=1, value=0.5)])
        steering, view = _two_layer_steering(arm)
        qsl, seq_lens, nat, pos = _forward([(4, 5), (3, 0)])
        view.set_batch(["warm", "plain"], {"warm": _sel(arm), "plain": None},
                       {"warm": list(range(200, 209)), "plain": [1, 2, 3]}, (qsl, seq_lens, nat),
                       prompt_lens={"warm": 9})
        h = _hidden(seed=33, rows=nat)
        r = torch.zeros(nat, HIDDEN)
        new_h, _ = steering.hook_body(1, (pos, h, r), (h, r))
        assert torch.equal(new_h[:4], h[:4] + 0.5)
        assert torch.equal(new_h[4:], h[4:])


class TestInstallPayloadAndRegistryHelper:
    def test_install_payload_carries_the_per_cell_predicate_through_the_frozen_validator(self):
        arm = _obj("arm-skipbos", [_skipbos_op(layer=1)])
        last = _obj("arm-last", [_last_op(layer=1)])
        twin = _obj("none__lens0", [_skipbos_op(layer=1)], scale=0.0)
        steering, _view = _make_steering(_registry(arm, last, twin))
        report = steering.install() if steering._report is None else steering._report
        assert report.resumed_prefill_ok_cells == {"arm-skipbos": True, "arm-last": False, "none__lens0": True}
        assert report.resumed_prefill_policy_cells == {"arm-skipbos": "always", "arm-last": "never",
                                                       "none__lens0": "always"}
        payload = lw.install_report_payload(report)
        assert payload[lw.RESUMED_PREFILL_OK_CELLS_KEY] == {"arm-last": False, "arm-skipbos": True,
                                                            "none__lens0": True}
        assert payload[lw.RESUMED_PREFILL_POLICY_CELLS_KEY] == {"arm-last": "never", "arm-skipbos": "always",
                                                                "none__lens0": "always"}
        expected = {"registration_count": 1, "layers": [1], "oproj_layers": [], "plan_layer": 1,
                    "cells": ["arm-last", "arm-skipbos", "none__lens0"], "ledger_cells": ["arm-last", "arm-skipbos"]}
        # the production validator (frozen vllm_server), verbatim: the extra key rides through
        assert vs.validate_install_reports([payload, dict(payload)], expected) == [payload, payload]
        # ... and the attestation reader recovers it
        att = {"worker_reports": [payload, dict(payload)]}
        assert lw.resumed_prefill_ok_cells_attested(att) == (payload[lw.RESUMED_PREFILL_OK_CELLS_KEY], None)
        old = {k: v for k, v in payload.items() if k != lw.RESUMED_PREFILL_OK_CELLS_KEY}
        got, why = lw.resumed_prefill_ok_cells_attested({"worker_reports": [payload, old]})
        assert got is None and "predates" in why
        other = dict(payload, **{lw.RESUMED_PREFILL_OK_CELLS_KEY: {"arm-last": True}})
        got, why = lw.resumed_prefill_ok_cells_attested({"worker_reports": [payload, other]})
        assert got is None and "disagree" in why
        assert lw.resumed_prefill_ok_cells_attested({})[0] is None
        # the three-valued reader, the proxy half's: present everywhere, agreed, a known policy
        assert lw.resumed_prefill_policy_cells_attested(att) == (payload[lw.RESUMED_PREFILL_POLICY_CELLS_KEY], None)
        old_p = {k: v for k, v in payload.items() if k != lw.RESUMED_PREFILL_POLICY_CELLS_KEY}
        got, why = lw.resumed_prefill_policy_cells_attested({"worker_reports": [payload, old_p]})
        assert got is None and "predates" in why
        junk = dict(payload, **{lw.RESUMED_PREFILL_POLICY_CELLS_KEY: {"arm-last": "sometimes"}})
        got, why = lw.resumed_prefill_policy_cells_attested({"worker_reports": [junk]})
        assert got is None and "policy mapping" in why
        # a report built outside install() carries no map and emits no key
        bare = lw.InstallReport(registration_count=1, layers=(1,), plan_layer=1, cells=("c",), ledger_cells=())
        assert lw.RESUMED_PREFILL_OK_CELLS_KEY not in lw.install_report_payload(bare)
        assert lw.RESUMED_PREFILL_POLICY_CELLS_KEY not in lw.install_report_payload(bare)

    def test_registry_helper_matches_the_worker(self):
        arm = _obj("arm-skipbos", [_skipbos_op(layer=1)])
        last = _obj("arm-last", [_last_op(layer=1)])
        resp = _obj("arm-response", [_response_op(layer=1)])
        reg = _registry(arm, last, resp)
        assert reg.resumed_prefill_ok_by_cell() == {"arm-last": False, "arm-response": False, "arm-skipbos": True}
        assert reg.resumed_prefill_ok_by_cell() == lw.resumed_prefill_ok_cells(reg)
        assert reg.resumed_prefill_policy_by_cell() == {"arm-last": "never", "arm-response": "conditional",
                                                        "arm-skipbos": "always"}
        assert reg.resumed_prefill_policy_by_cell() == lw.resumed_prefill_policy_cells(reg)
