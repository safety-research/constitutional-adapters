"""The two sdf-base-steering DERIVED objects, made servable (2026-09-06):
the full-corpus fit MINUS its register-matched control fit, as cells the
full22 wave can stand up. CPU-only, synthetic artifacts throughout.

THE DESIGN CLAIM THESE TESTS PIN, because the whole change rests on it:
NEITHER OBJECT IS A NEW SERVED SHAPE.

- ``psr_ce_all_minus_ctl_distilled`` distils the functional residual back
  into its parents' own geometry, so its artifact IS a PSRObject blob
  (kind ``psr``) and it serves through the committed psr adapter with no
  new code at all — the arm name carries the derivation, exactly as the
  psr_xfer_* transferred units do. Registration, not translation.
- ``lora_r4_ce_minus_ctl`` stores BOTH parents' (A, B) factors bit-exact
  and applies each parent's OWN o_proj hook at +scale / -scale
  (multilayer.LoRASubtractionObject.apply). So it needs a new blob SCHEMA
  translation — and nothing else: the ops it emits are the existing
  OP_LORA_OPROJ at the existing HOOK_SITE_OPROJ, two per layer, at coeff
  +1 / -1. No new op, no new hook site, no new reduction class.

The load-bearing test is therefore the same one the lora lane used for
the o_proj site: adapt the REAL class's own saved bytes and prove
``apply_oproj_ops`` reproduces ``LoRASubtractionObject.apply``'s
transformation on a real tiny model, across the prefill and a KV-cached
decode step. A structural test would pass on a translation that served
only one parent, or served the difference with the sign flipped.

Two things are checked at DIFFERENT strengths there, and the difference
is the honest part. The forward agreement is at the o_proj lane's
existing fp32 kernel-order tolerance, not bit-exact: the op stores the
read basis transposed (``lowrank_a = Aᵀ``), so the served GEMM reads a
differently-laid-out buffer than the class's ``x @ A``. Measured at
~1e-8 absolute here, and inherited unchanged from the committed
OP_LORA_OPROJ — this kind adds no numerical slack of its own, and the
test also requires the deviation to sit orders of magnitude below the
edit itself. What IS exact is the algebra: the positive-parent ops are
bitwise the corpus parent's own tensors, so dropping the control term
recovers that parent exactly.

Also pinned here: the derived cells reach the explicit registry (their
sidecar is derived_meta.json, never train_budget_meta.json), the ops are
classified into the reduction contract so the STACKED mode covers them,
and stacked == per_row on exact integers for this kind's op sequence.

Section 6 carries main's port of this file (bcb-bfcl-200, 2026-09-10; merged
2026-09-11): the registry's derived record, the FIXED statistical family
(object-only methods outside it), the REAL registry source over the dedicated
two-SI config, and the derived preflight. Its pins that named main's ONE arm
and ONE corpus SI are widened to the merged vocabulary (two derived arms, two
corpus SIs, four object-only methods); everything else is main's verbatim.
"""

from __future__ import annotations

import io
import json
from dataclasses import replace as dataclasses_replace
from pathlib import Path

import pytest
import torch
import yaml

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.serving import lens_adapters as la
from sisteer.extraction.multilayer import (LoRASubtractionObject,
                                           MultiLayerLoRAObject)
from sisteer.methods import ARMS, DERIVED_ARMS, is_derived_arm

KIND = "lora_subtraction"
PSR_ARM = "psr_ce_all_minus_ctl_distilled"
LORA_ARM = "lora_r4_ce_minus_ctl"
#: a synthetic corpus SI, in the OBJECT-ONLY pseudo-instruction shape the
#: real cells use (never a system prompt)
SI = "constv2-mix-full"
D_IN, D_OUT, RANK = 10, 6, 3      # non-square on purpose: a transposed
HEADER = (7, 8, 9)                # basis is then a shape error, not silent
#: main's port of this file (bcb-bfcl-200, 2026-09-10): the dedicated two-SI
#: config that is its roster, and the ONE cell it serves, in both spellings
REPO = Path(__file__).resolve().parents[1]
SDFBASE_CFG = REPO / "configs" / "two_si_gemma4e4b_sdfbase.yaml"
HSI = "constv2_mix_full"
CELL = f"{SI}__{LORA_ARM}"
HCELL = f"si_{HSI}__{LORA_ARM}"
GEMMA_IT = "google/gemma-4-E4B-it"
GEMMA_IT_REV = "ee0ef6023621cff504d758262d4e04895a5af4a2"


def _blob_bytes(payload) -> bytes:
    buf = io.BytesIO()
    torch.save(payload, buf)
    return buf.getvalue()


def _sub_blob(layers=(0, 2), rank=RANK, seed=5, positions="all",
              d_in=D_IN, d_out=D_OUT, **extra):
    """A lora_subtraction artifact in the class's own save schema."""
    g = torch.Generator().manual_seed(seed)
    blob = {"kind": KIND, "layers": list(layers), "rank": rank,
            "positions": positions, "response_header_len": (
                len(HEADER) if positions == "response" else None)}
    for side in ("pos", "neg"):
        blob[f"A_{side}"] = {str(l): torch.randn(d_in, rank, generator=g)
                             for l in layers}
        blob[f"B_{side}"] = {str(l): torch.randn(d_out, rank, generator=g) * 0.5
                             for l in layers}
    blob.update(extra)
    return blob


def _adapt(blob, spec=None, header=HEADER, turn_end=None):
    return la.get_adapter(KIND)(dict(spec or {"kind": KIND}),
                                _blob_bytes(blob), b"",
                                header_ids=tuple(header) if header else None,
                                turn_end_ids=turn_end)


# ---------------------------------------------------------------------------
# 1. registration: two arms, one new kind, no new op
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_both_arms_are_registered_and_refuse_to_train_or_build(self):
        for arm, builder in ((PSR_ARM, "psr_residual_distill.py"),
                             (LORA_ARM, "lora_subtract.py")):
            assert arm in ARMS, arm
            with pytest.raises(RuntimeError, match="DERIVED"):
                ARMS[arm].train(None, None, None, None)
            with pytest.raises(RuntimeError, match="no untrained form"):
                ARMS[arm].build(None, None, 0)
            # the refusal names the builder that DOES produce the artifact
            with pytest.raises(RuntimeError, match=builder):
                ARMS[arm].train(None, None, None, None)

    def test_object_kinds_are_the_real_classes_and_the_psr_one_is_reused(self):
        from sisteer.extraction.psr import PSRObject

        # the distilled residual: its parents' own class, so the committed
        # psr adapter serves it with NO new translation
        assert ra.ARM_OBJECT_KINDS[PSR_ARM] == PSRObject.kind == "psr"
        assert ra.ARM_OBJECT_KINDS["psr_ce_all"] == ra.ARM_OBJECT_KINDS[PSR_ARM]
        # the difference object: its own class, because it holds two
        # parents' factors
        assert ra.ARM_OBJECT_KINDS[LORA_ARM] == LoRASubtractionObject.kind == KIND
        assert LoRASubtractionObject.kind not in (
            MultiLayerLoRAObject.kind, PSRObject.kind)

    def test_petri_vocabulary_names_both_and_round_trips(self):
        for arm in (PSR_ARM, LORA_ARM):
            assert arm in dc.CLASS_A_METHODS
            tt = f"{SI}__{arm}"
            harness = ra.harness_cell_id(tt)
            assert harness == f"si_constv2_mix_full__{arm}"
            assert ra.traintest_cell_id(harness) == tt
            parsed = dc.parse_cell_id(harness)
            assert parsed.cell_class == "class_a" and parsed.method == arm

    def test_the_corpus_si_still_refuses_a_prompted_or_stacked_spelling(self):
        """The derived cells are named through an OBJECT-ONLY
        pseudo-instruction, so widening CLASS_A_METHODS must not open a
        prompted/stacked door for it."""
        for bad in ("prompted__constv2_mix_full",
                    f"stacked__constv2_mix_full__{LORA_ARM}"):
            with pytest.raises(dc.DefenseCellError):
                dc.parse_cell_id(bad)

    def test_the_kind_is_adapted_and_no_new_op_entered_the_vocabulary(self):
        assert KIND in la.registered_kinds()
        assert KIND not in la.UNSERVABLE_KIND_REASONS
        ops = _adapt(_sub_blob())
        assert {op.op for op in ops} == {la.OP_LORA_OPROJ}
        assert {la.op_site(op) for op in ops} == {la.HOOK_SITE_OPROJ}


# ---------------------------------------------------------------------------
# 2. the translation: two ops per layer, the parents' own bytes, +1 / -1
# ---------------------------------------------------------------------------


class TestTranslation:
    def test_two_ops_per_layer_carry_each_parents_factors_bit_exact(self):
        blob = _sub_blob(layers=(0, 2))
        ops = _adapt(blob)
        assert [op.layer for op in ops] == [0, 0, 2, 2]
        assert [op.coeff for op in ops] == [1.0, -1.0, 1.0, -1.0]
        for i, layer in enumerate((0, 2)):
            pos_op, neg_op = ops[2 * i], ops[2 * i + 1]
            for op, side in ((pos_op, "pos"), (neg_op, "neg")):
                A = blob[f"A_{side}"][str(layer)]
                B = blob[f"B_{side}"][str(layer)]
                # read basis Aᵀ [k, d_in], write basis B [d_out, k] — the
                # parent's OWN tensors, unscaled and unswapped
                assert torch.equal(op.lowrank_a, A.t())
                assert torch.equal(op.lowrank_b, B)
                assert op.lowrank_a.dtype == torch.float32

    def test_the_sign_rides_the_coefficient_not_a_negated_basis(self):
        """The class negates the SCALE (neg.register_hooks(-scale)); the
        adapter must mirror that, keeping the write basis the parent's own
        bytes — otherwise "both parents' factors ride bit-exact" stops
        being checkable at the served ops."""
        blob = _sub_blob()
        ops = _adapt(blob)
        neg = [op for op in ops if op.coeff == -1.0]
        assert neg, "no negative-parent op"
        for op in neg:
            B = blob["B_neg"][str(op.layer)]
            assert torch.equal(op.lowrank_b, B)          # NOT -B
            assert not torch.equal(op.lowrank_b, -B)

    def test_fit_leg_is_the_default_and_the_cell_can_remask(self):
        all_ops = _adapt(_sub_blob(positions="all"))
        assert {op.positions for op in all_ops} == {la.POSITIONS_ALL}
        assert all(op.header_ids is None for op in all_ops)
        resp = _adapt(_sub_blob(positions="response"))
        assert {op.positions for op in resp} == {la.POSITIONS_RESPONSE}
        assert all(op.header_ids == HEADER for op in resp)
        # the MODE belongs to the cell: an all-position fit served under a
        # cross-phase cell's response override
        over = _adapt(_sub_blob(positions="all"),
                      spec={"kind": KIND, "positions": "response"})
        assert {op.positions for op in over} == {la.POSITIONS_RESPONSE}

    def test_recorded_fit_positions_reads_the_declared_schema(self):
        for pos in ("all", "response"):
            assert la.recorded_fit_positions(
                KIND, _blob_bytes(_sub_blob(positions=pos)), b"") == pos

    @pytest.mark.parametrize("mutate,match", [
        (lambda b: b.pop("A_neg"), "A_neg"),
        (lambda b: b.pop("B_pos"), "B_pos"),
        (lambda b: b.__setitem__("layers", [0, 5]), "layers"),
        (lambda b: b.__setitem__("rank", 9), "rank"),
        (lambda b: b["A_neg"].pop("2"), "layers"),
        (lambda b: b["A_neg"].__setitem__("0", torch.randn(D_IN, RANK + 1)),
         "not same-shape"),
        (lambda b: b["B_pos"].__setitem__("0", torch.randn(D_OUT + 1, RANK)),
         "not same-shape"),
    ])
    def test_self_inconsistent_artifacts_refuse(self, mutate, match):
        blob = _sub_blob()
        mutate(blob)
        with pytest.raises(la.AdapterError, match=match):
            _adapt(blob)

    def test_wrong_kind_stray_sidecar_and_spec_drift_refuse(self):
        with pytest.raises(la.AdapterError, match="sidecar"):
            la.get_adapter(KIND)({"kind": KIND}, _blob_bytes(_sub_blob()),
                                 b"stray")
        with pytest.raises(la.AdapterError, match="wrong adapter"):
            la.get_adapter("lora_all")({"kind": "lora_all"},
                                       _blob_bytes(_sub_blob()), b"")
        with pytest.raises(la.AdapterError, match="cross-check"):
            _adapt(_sub_blob(), spec={"kind": KIND, "layers": [1, 4]})
        with pytest.raises(la.AdapterError, match="cross-check"):
            _adapt(_sub_blob(), spec={"kind": KIND, "rank": RANK + 1})

    def test_response_without_a_header_refuses_rather_than_widening(self):
        with pytest.raises(la.AdapterError, match="header"):
            _adapt(_sub_blob(positions="response"), header=None)

    def test_a_nonfinite_factor_refuses(self):
        blob = _sub_blob()
        blob["B_neg"]["0"][0, 0] = float("nan")
        with pytest.raises(la.AdapterError, match="non-finite"):
            _adapt(blob)


# ---------------------------------------------------------------------------
# 3. THE LOAD-BEARING ONE: the ops ARE the class's transformation
# ---------------------------------------------------------------------------


def _tiny_parent(model, layers, seed, rank=2):
    obj = MultiLayerLoRAObject(model, layers=list(layers), rank=rank)
    g = torch.Generator().manual_seed(seed)
    for l in layers:
        obj.A[l].data = torch.randn(obj.A[l].shape, generator=g) * 0.3
        obj.B[l].data = torch.randn(obj.B[l].shape, generator=g) * 0.3
    return obj


def test_ops_reproduce_the_authoritative_difference_object(tiny_llama):
    """Adapt the REAL LoRASubtractionObject's own saved bytes, then run a
    prefill + a KV-cached decode step twice: once under the class's own
    ``apply`` (both parents' hooks, +scale / -scale), once with the
    adapter's ops applied at the same o_proj modules through
    ``apply_oproj_ops``. The o_proj outputs must agree at every steered
    layer, in both phases.

    THE TOLERANCE IS THE lora_all LANE'S, and for the same reason (its
    ``test_ops_reproduce_the_authoritative_class_forward``): the op stores
    the read basis TRANSPOSED (``lowrank_a = Aᵀ``, the vocabulary's
    convention), so the served GEMM reads a differently-laid-out buffer
    than the class's ``x @ A`` and the two differ by fp32 kernel-order
    noise — measured here at ~1e-8 absolute on a 16-dim tiny model. That
    is a property of the committed o_proj op, inherited unchanged; this
    kind adds none of its own. What must be EXACT is the algebra, and the
    companion test below pins that: the positive-parent ops are bitwise
    the corpus parent's own.

    Non-vacuity is asserted, not assumed: the steered output must differ
    from the unsteered one, and from what the POSITIVE parent alone would
    produce — otherwise a translation that dropped the control term, or
    served it with the wrong sign, would pass."""
    model, _tok = tiny_llama
    layers = [0, 1]
    pos = _tiny_parent(model, layers, seed=11)
    neg = _tiny_parent(model, layers, seed=23)
    sub = LoRASubtractionObject(pos, neg)

    buf = io.BytesIO()
    torch.save(LoRASubtractionObject.compose_blob(pos.to_blob(),
                                                  neg.to_blob()), buf)
    ops = la.get_adapter(KIND)({"kind": KIND}, buf.getvalue(), b"")
    assert len(ops) == 2 * len(layers)

    ids = torch.randint(3, 90, (1, 6), generator=torch.Generator().manual_seed(3))
    P = ids.shape[1]
    decoder = model.model.layers

    def run(mode):
        """(o_proj outputs per layer per phase) under `mode`.

        HOOK ORDER IS THE WHOLE POINT: forward hooks fire in registration
        order, so the object's own hooks must be installed BEFORE the
        capture hooks — otherwise the capture reads the o_proj output the
        object has not edited yet and the comparison is vacuous (it was,
        the first time this test ran)."""
        import contextlib

        seen = {l: [] for l in layers}
        state = {"n": 0}

        def capture(l):
            def hook(_mod, args, output):
                x = args[0]
                x2 = x[0] if x.dim() == 3 else x
                out2 = output[0] if output.dim() == 3 else output
                if mode == "ops":
                    T = x2.shape[0]
                    ctx = (la.HookContext(la.PHASE_PREFILL, tuple(range(T)))
                           if T > 1 else
                           la.HookContext(la.PHASE_DECODE, (state["n"],)))
                    new = la.apply_oproj_ops(x2, out2, ops, 1.0, layer=l,
                                             ctx=ctx)
                    seen[l].append(new.detach().clone())
                    return new.unsqueeze(0) if output.dim() == 3 else new
                seen[l].append(out2.detach().clone())
                return output
            return hook

        ctx = (sub.apply(model, scale=1.0) if mode == "class"
               else pos.apply(model, scale=1.0) if mode == "pos_only"
               else contextlib.nullcontext())
        with torch.no_grad(), ctx:          # the object's hooks go on first
            handles = [decoder[l].self_attn.o_proj.register_forward_hook(
                capture(l)) for l in layers]
            try:
                state["n"] = 0
                out = model(input_ids=ids, use_cache=True)
                state["n"] = P
                model(input_ids=ids[:, -1:],
                      past_key_values=out.past_key_values, use_cache=True)
            finally:
                for h in handles:
                    h.remove()
        return seen

    ref = run("class")
    mine = run("ops")
    bare = run("bare")
    pos_only = run("pos_only")
    for l in layers:
        for phase in (0, 1):
            got, want = mine[l][phase], ref[l][phase]
            assert torch.allclose(got, want, rtol=1e-5, atol=1e-6), (
                f"layer {l} phase {phase}: the adapter's ops do not "
                "reproduce LoRASubtractionObject's own transformation "
                f"(max abs {float((got - want).abs().max())})")
            # the deviation is kernel-order noise, not a different edit:
            # it must be orders of magnitude below the edit ITSELF
            edit = (want - bare[l][phase]).abs().max()
            assert float((got - want).abs().max()) < 1e-3 * float(edit)
            # non-vacuous: the steer fired, and it is the DIFFERENCE, not
            # the positive parent alone (a dropped or sign-flipped control
            # term fails here)
            assert not torch.equal(want, bare[l][phase])
            assert not torch.allclose(want, pos_only[l][phase],
                                      rtol=1e-4, atol=1e-6)


def test_dropping_the_control_term_recovers_the_corpus_parent(tiny_llama):
    """The algebra the artifact exists to support, at the SERVED ops: the
    positive-parent ops of the difference object are bit-identical to the
    ops the corpus parent serves under on its own. That is what makes
    "corpus minus control" a decomposition rather than a re-fit."""
    model, _tok = tiny_llama
    layers = [0, 1]
    pos = _tiny_parent(model, layers, seed=11)
    neg = _tiny_parent(model, layers, seed=23)
    buf = io.BytesIO()
    torch.save(LoRASubtractionObject.compose_blob(pos.to_blob(),
                                                  neg.to_blob()), buf)
    sub_ops = la.get_adapter(KIND)({"kind": KIND}, buf.getvalue(), b"")
    pbuf = io.BytesIO()
    torch.save(pos.to_blob(), pbuf)
    parent_ops = la.get_adapter("lora_all")({"kind": "lora_all"},
                                            pbuf.getvalue(), b"")
    positive = [op for op in sub_ops if op.coeff == 1.0]
    assert len(positive) == len(parent_ops)
    for a, b in zip(positive, parent_ops):
        assert (a.op, a.layer, a.positions, a.phase) == \
               (b.op, b.layer, b.positions, b.phase)
        assert torch.equal(a.lowrank_a, b.lowrank_a)
        assert torch.equal(a.lowrank_b, b.lowrank_b)


# ---------------------------------------------------------------------------
# 4. the stacked-reduction contract covers the new ops
# ---------------------------------------------------------------------------


class TestReductionContract:
    def test_ops_are_classified_as_linear_reductions(self):
        ops = _adapt(_sub_blob())
        for op in ops:
            assert la.op_reduces_in_group(op.op) is True
            assert la.reduction_op_class(op.op) == la.OP_CLASS_REDUCTION_LINEAR
        # the o_proj site carries no GATED op, so this kind adds no new
        # gate evidence requirement to the worker's shadow compare
        assert not (set(la.OPROJ_SITE_OPS) & la.GATED_REDUCTION_OPS)

    @pytest.mark.parametrize("stacked", [True, False])
    def test_stacked_and_per_row_agree_on_exact_integers(self, monkeypatch,
                                                         stacked):
        """The reduction-mode contract for a LINEAR reduction op: on
        exactly-representable integers the two modes are bit-identical, so
        this kind is servable under the wave's stacked mode with the same
        guarantee its lora_all parents have."""
        monkeypatch.setattr(la, "_STACKED_REDUCTIONS", stacked)
        blob = _sub_blob()
        for side in ("pos", "neg"):
            for l in ("0", "2"):
                blob[f"A_{side}"][l] = torch.randint(
                    -3, 4, (D_IN, RANK)).float()
                blob[f"B_{side}"][l] = torch.randint(
                    -3, 4, (D_OUT, RANK)).float()
        ops = _adapt(blob)
        x = torch.randint(-3, 4, (4, D_IN)).float()
        out = torch.randint(-3, 4, (4, D_OUT)).float()
        got = la.apply_oproj_ops(x, out, ops, 1.0, layer=0,
                                 independent_rows=not stacked)
        # the reference: the arithmetic written out, per row, per parent
        ref = out.clone()
        for op in ops:
            if op.layer != 0:
                continue
            delta = (x @ op.lowrank_a.t()) @ op.lowrank_b.t()
            ref = ref + (op.coeff * delta)
        assert torch.equal(got, ref)

    def test_the_reduction_mode_attestation_vocabulary_is_untouched(self):
        """This kind rides the EXISTING op classes, so it must not have
        widened the sets the mode attestation is defined over."""
        assert la.REDUCTION_OPS == (la.GATED_REDUCTION_OPS
                                    | la.LINEAR_REDUCTION_OPS)
        assert la.OP_LORA_OPROJ in la.LINEAR_REDUCTION_OPS
        assert la.REDUCTION_MODES == ("stacked", "per_row")


# ---------------------------------------------------------------------------
# 4b. the worker installs ONE hook per layer, not one per op
# ---------------------------------------------------------------------------


class TestWorkerInstall:
    """THE shape-specific worker risk of this kind, and the reason it is
    pinned separately: every other o_proj kind emits at most ONE op per
    layer, so nothing before now distinguished "a hook per layer" from "a
    hook per op". This kind emits two. If the install registered per op,
    each layer's hook would fire twice and apply BOTH ops each time — the
    edit would be doubled, which no output-difference check would catch
    (a doubled steer still differs from the control)."""

    def _registry_for(self, layers=(1, 3)):
        import io as _io

        from sisteer.capeval.serving import lens_steer as ls

        blob = _sub_blob(layers=layers, d_in=8, d_out=8)
        buf = _io.BytesIO()
        torch.save(blob, buf)
        ops = la.get_adapter(KIND)({"kind": KIND}, buf.getvalue(), b"")
        obj = ls.LoadedObject(cell_id="derived-cell", kind=KIND,
                              expect_hash="hash-derived", ops=tuple(ops),
                              effective_scale=1.0)
        return ls.WorkerObjectRegistry({obj.cell_id: obj},
                                       assistant_header_ids=None), ops

    def test_two_ops_per_layer_install_one_hook_per_layer(self):
        from test_lora_lens_serving import _steering

        registry, ops = self._registry_for(layers=(1, 3))
        assert len(ops) == 4                      # two layers, two parents
        steering, view = _steering(registry)
        rep = steering._report
        # the layer set is DEDUPLICATED: two ops at layer 1, one hook
        assert rep.oproj_layers == (1, 3) and rep.layers == ()
        for i, layer in enumerate(view.layers):
            want = 1 if i in (1, 3) else 0
            assert len(layer.self_attn.o_proj.hooks) == want, i
            assert len(layer.pre_hooks) == want

    def test_the_layers_ops_index_returns_both_parents_ops_in_order(self):
        registry, _ = self._registry_for(layers=(1,))
        got = registry.ops_for("derived-cell", la.HOOK_SITE_OPROJ, 1)
        assert [op.coeff for op in got] == [1.0, -1.0]
        assert registry.ops_for("derived-cell", la.HOOK_SITE_BLOCK, 1) == ()


# ---------------------------------------------------------------------------
# 5. the explicit registry accepts a derived cell (derived_meta.json)
# ---------------------------------------------------------------------------


def _stage_derived(bank: Path, cell_id: str, *, kind: str = KIND,
                   derived_id: str | None = None, blob=None,
                   also_budget_sidecar: bool = False,
                   break_sha: bool = False) -> Path:
    """A staged DERIVED cell dir: object.pt + derived_meta.json (+
    staged_from.json), shaped like the real staging layout — the sidecar
    binds the object bytes to two complete parent identities."""
    from sisteer.eval.defense_cell import sha256_file

    d = bank / cell_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "object.pt"
    payload = blob if blob is not None else _sub_blob()
    torch.save(payload, p)
    sha = sha256_file(p)

    def parent(cell, obj_sha):
        return {"cell_id": cell, "model_key": "tiny_base",
                "object_sha256": obj_sha,
                "budget_meta_sha256": "bb" * 32,
                "model_id": "synthetic-org/tiny", "model_revision": "0" * 40,
                "positions": "all", "response_header_len": None,
                "target_layers": [0, 2], "template_hash": "1111222233334444"}
    prov = {"kind": kind,
            "parent_pos": parent(f"{SI}__lora_r4_ce", "aa" * 32),
            "parent_neg": parent("constv2ctl2-full__lora_r4_ce", "cc" * 32)}
    # the SAME provenance rides inside the bytes (the verdict cross-checks)
    payload = torch.load(p, map_location="cpu", weights_only=True)
    payload["provenance"] = prov
    torch.save(payload, p)
    sha = sha256_file(p)
    meta = {**prov, "derived_id": derived_id or cell_id,
            "object_sha256": ("00" * 32) if break_sha else sha}
    (d / "derived_meta.json").write_text(json.dumps(meta))
    (d / "staged_from.json").write_text(json.dumps({"source": "synthetic"}))
    if also_budget_sidecar:
        (d / "train_budget_meta.json").write_text(
            json.dumps({"cell_id": cell_id, "completion_tokens": 512,
                        "object_sha256": sha}))
    return p


class TestExplicitRegistry:
    def _cfg(self, bank):
        return {"explicit_cells": ["none", f"{SI}__{LORA_ARM}"],
                "defenses_dir": str(bank)}

    def test_a_derived_cell_builds_without_a_budget_sidecar(self, tmp_path):
        from sisteer.eval.defense_cell import registry_from_explicit_cells

        bank = tmp_path / "cells_view" / "main"
        cid = f"{SI}__{LORA_ARM}"
        p = _stage_derived(bank, cid)
        assert not (p.parent / "train_budget_meta.json").exists()
        cells = registry_from_explicit_cells(self._cfg(bank))
        assert [c.id for c in cells] == ["none", cid]
        cell = cells[1]
        assert cell.class_ == "A" and cell.arm == LORA_ARM and cell.si_id == SI
        assert Path(cell.object_path) == p

    def test_a_stale_or_mislabelled_derived_object_refuses(self, tmp_path):
        from sisteer.eval.defense_cell import registry_from_explicit_cells

        cid = f"{SI}__{LORA_ARM}"
        # sidecar sha != the bytes
        bank = tmp_path / "stale"
        _stage_derived(bank, cid, break_sha=True)
        with pytest.raises(ValueError, match="object_sha256 mismatch"):
            registry_from_explicit_cells(self._cfg(bank))
        # sidecar names ANOTHER derivation
        bank2 = tmp_path / "wrongid"
        _stage_derived(bank2, cid, derived_id="constv2ctl2-full__lora_r4_ce_minus_ctl")
        with pytest.raises(ValueError, match="wrong-identity artifact"):
            registry_from_explicit_cells(self._cfg(bank2))
        # TWO identity sidecars: ambiguous, never guessed
        bank3 = tmp_path / "ambiguous"
        _stage_derived(bank3, cid, also_budget_sidecar=True)
        with pytest.raises(ValueError, match="more than one identity sidecar"):
            registry_from_explicit_cells(self._cfg(bank3))

    def test_a_fitted_cell_still_needs_its_budget_sidecar(self, tmp_path):
        """The derived branch must not have relaxed the fitted one."""
        from sisteer.eval.defense_cell import registry_from_explicit_cells

        bank = tmp_path / "cells_view" / "main"
        d = bank / f"{SI}__psr_ce_all"
        d.mkdir(parents=True)
        torch.save({"kind": "psr", "positions": "all", "layers": [0],
                    "directions": {"0": torch.zeros(4)},
                    "gate_w": {"0": torch.zeros(4)},
                    "gate_b": {"0": torch.zeros(1)}}, d / "object.pt")
        cfg = {"explicit_cells": ["none", f"{SI}__psr_ce_all"],
               "defenses_dir": str(bank)}
        # the refusal names every spelling a fitted cell may prove itself
        # under, and the bare dir carries none of them
        with pytest.raises(ValueError, match="no identity sidecar") as err:
            registry_from_explicit_cells(cfg)
        for spelling in ("budget sidecar", "fit_provenance.json", "derived_meta.json"):
            assert spelling in str(err.value)

    def test_the_derived_psr_cell_needs_no_new_kind_at_all(self, tmp_path):
        """The distilled residual rides the psr adapter: a staged cell
        whose blob is an ordinary PSRObject artifact builds and adapts
        under the derived ARM name with no lora_subtraction anywhere."""
        from sisteer.eval.defense_cell import registry_from_explicit_cells

        bank = tmp_path / "cells_view" / "main"
        cid = f"{SI}__{PSR_ARM}"
        psr_blob = {"kind": "psr", "layers": [0, 2], "train_direction": True,
                    "positions": "all", "response_header_len": None,
                    "directions": {"0": torch.randn(8), "2": torch.randn(8)},
                    "gate_w": {"0": torch.randn(8), "2": torch.randn(8)},
                    "gate_b": {"0": torch.zeros(1), "2": torch.zeros(1)}}
        p = _stage_derived(bank, cid, kind="psr", blob=psr_blob)
        cells = registry_from_explicit_cells(
            {"explicit_cells": ["none", cid], "defenses_dir": str(bank)})
        assert [c.id for c in cells] == ["none", cid]
        ops = la.get_adapter(ra.ARM_OBJECT_KINDS[PSR_ARM])(
            {"kind": "psr"}, p.read_bytes(), b"")
        assert {op.op for op in ops} == {la.OP_RELU_GATED_VECTOR}
        assert [op.layer for op in ops] == [0, 2]


# ---------------------------------------------------------------------------
# 6. main's port (bcb-bfcl-200, 2026-09-10), carried across the math500 merge:
#    the registry's derived record, the fixed statistical family, the REAL
#    registry source over the dedicated config, and the derived preflight.
#    Pins widened to the merged vocabulary are marked `merged:`.
# ---------------------------------------------------------------------------


class TestMainPortRegistration:
    def test_the_arm_is_registered_last_and_refuses_to_train_or_build(self):
        assert LORA_ARM in ARMS
        # appended LAST at its merge; the paired write (2026-09-17) was appended after it, so
        # the TAIL is the three derived arms in registration order — still a positional pin
        assert list(ARMS)[-3:] == [PSR_ARM, LORA_ARM, "psr_ce_all_minus_ctl"]
        assert ARMS[LORA_ARM].suites == frozenset({"two_si"})
        with pytest.raises(RuntimeError, match="DERIVED"):
            ARMS[LORA_ARM].train(None, None, None, None)
        with pytest.raises(RuntimeError, match="no untrained form"):
            ARMS[LORA_ARM].build(None, None, 0)
        # the refusal names the builder that DOES produce the artifact
        with pytest.raises(RuntimeError, match="lora_subtract.py"):
            ARMS[LORA_ARM].train(None, None, None, None)

    def test_the_derived_record_is_the_registry_s_own(self):
        assert DERIVED_ARMS == (PSR_ARM, LORA_ARM, "psr_ce_all_minus_ctl")   # + the paired write (2026-09-17)
        assert is_derived_arm(LORA_ARM)
        assert not is_derived_arm("lora_r4") and not is_derived_arm("psr_kl_all")
        assert set(DERIVED_ARMS) <= set(ARMS)
        # NOT in TrainTest's default Class A matrix: its cell is enumerated
        # by the dedicated config's class_a_arms, never crossed with the
        # safety SIs (a default-matrix membership would fabricate cells and
        # move measurement_inputs_sha for every config)
        from sisteer.eval.defense_cell import CLASS_A_ARMS
        assert LORA_ARM not in CLASS_A_ARMS

    def test_the_kind_loader_hydrates_the_class(self):
        from sisteer.methods import _KIND_LOADERS
        # a bound classmethod is re-bound on every access, so compare by
        # equality (same function, same class), never identity
        assert _KIND_LOADERS[KIND] == LoRASubtractionObject.load_blob

    def test_object_kind_is_the_real_class_and_distinct(self):
        from sisteer.extraction.psr import PSRObject
        assert ra.ARM_OBJECT_KINDS[LORA_ARM] == LoRASubtractionObject.kind == KIND
        assert LoRASubtractionObject.kind not in (
            MultiLayerLoRAObject.kind, PSRObject.kind)
        # MOVED PIN: this asserted
        # `LORA_ARM not in ra.LORA_EXPORTED_ARMS` — the per-arm set that
        # decided serving_path, empty for its whole life. The serving path is
        # a POLICY of the cell source now (ra.SERVING_PATH_POLICIES): by KIND
        # the subtraction IS natively exportable, and WHICH path a standup
        # serves it on is the standup's call — TestRealRegistrySource pins
        # both answers (lens under the default, lora under lora_native).
        assert not hasattr(ra, "LORA_EXPORTED_ARMS")
        assert KIND in dc.NATIVE_LORA_SERVABLE_KINDS
        assert dc.native_lora_unservable_reason(
            KIND, {"kind": KIND, "positions": "all"}) is None

    def test_petri_vocabulary_names_it_last_and_round_trips(self):
        # named LAST at its merge; the paired write was appended after it (2026-09-17)
        assert dc.CLASS_A_METHODS[-2:] == (LORA_ARM, "psr_ce_all_minus_ctl")
        assert ra.HARNESS_TO_TRAINTEST_SI[HSI] == SI
        assert ra.TRAINTEST_TO_HARNESS_SI[SI] == HSI
        assert ra.harness_cell_id(CELL) == HCELL
        assert ra.traintest_cell_id(HCELL) == CELL
        parsed = dc.parse_cell_id(HCELL)
        assert parsed.cell_class == "class_a" and parsed.method == LORA_ARM
        assert parsed.si_id == HSI

    def test_the_si_vocabulary_is_promptable_plus_object_only(self):
        # + constitution_full: the FULL 34k constitution as a prompted appendix arm, registered by the sdfbase-llama8b lane
        # — an APPENDIX arm, not a paper arm; promptable, no object
        assert cm.PETRI_PROMPTABLE_SI_IDS == ("safety", "constitution", "irrelevant", "constitution_full")
        assert cm.PETRI_OBJECT_ONLY_SI_IDS == (HSI, "constv2ctl2_full")   # merged: both corpora
        assert cm.PETRI_SI_IDS == cm.PETRI_PROMPTABLE_SI_IDS + cm.PETRI_OBJECT_ONLY_SI_IDS
        assert not (set(cm.PETRI_PROMPTABLE_SI_IDS) & set(cm.PETRI_OBJECT_ONLY_SI_IDS))
        # every object-only SI is translated, and only promptable SIs have
        # a variant behavior dir
        assert set(cm.PETRI_OBJECT_ONLY_SI_IDS) <= set(ra.HARNESS_TO_TRAINTEST_SI)
        from sisteer.capeval.petri import scenarios as psc
        assert set(psc.SI_VARIANT_DIRS) == set(cm.PETRI_PROMPTABLE_SI_IDS)
        assert len(psc.VARIANT_DIRS) == len(cm.PETRI_PROMPTABLE_SI_IDS)

    def test_the_corpus_si_refuses_every_prompted_or_stacked_spelling(self):
        """The derived cell is named through an OBJECT-ONLY pseudo-
        instruction, so widening CLASS_A_METHODS must not open a
        prompted/stacked door for it — anywhere the id grammar lives."""
        for bad in (f"prompted__{HSI}", f"stacked__{HSI}__{LORA_ARM}",
                    f"stacked__{HSI}__psr_kl_all"):
            with pytest.raises(dc.DefenseCellError, match="OBJECT-ONLY"):
                dc.parse_cell_id(bad)
            with pytest.raises(dc.DefenseCellError):
                ra.traintest_cell_id(bad)
        # the canonical vocabulary lists the object-only cell (an object-only
        # SI x an object-only method) but no prompted one, no prompt-family
        # method under the corpus SI, and no object-only method under a
        # prompt SI
        from sisteer.capeval.petri.petri_stats import (canonical_cell_ids,
                                                       object_only_cell_ids)
        ids = canonical_cell_ids()
        assert f"prompted__{HSI}" not in ids
        assert HCELL in ids and HCELL in object_only_cell_ids()
        # merged: every object-only cell is a corpus SI x an object-only method, nothing else
        assert set(object_only_cell_ids()) == {f"si_{si}__{m}" for si in cm.PETRI_OBJECT_ONLY_SI_IDS
                                              for m in dc.OBJECT_ONLY_METHODS}
        assert f"si_{HSI}__psr_kl_all" not in ids
        assert f"si_safety__{LORA_ARM}" not in ids
        assert ids.count(HCELL) == 1

    def test_the_serving_vocabulary_is_not_the_statistical_family(self):
        """The arm is admitted by the SERVING grammar
        (CLASS_A_METHODS) but is NOT a leg of any prompt-SI
        hypothesis — petri_stats' fixed family is
        PROMPTABLE_CLASS_A_METHODS, unchanged at 13."""
        assert dc.OBJECT_ONLY_METHODS == ("psr_ce_all", "lora_r4_ce", PSR_ARM, LORA_ARM,
                                          "psr_ce_all_minus_ctl")   # merged; + the paired write (2026-09-17)
        assert LORA_ARM in dc.CLASS_A_METHODS
        assert LORA_ARM not in dc.PROMPTABLE_CLASS_A_METHODS
        assert len(dc.PROMPTABLE_CLASS_A_METHODS) == 13
        assert tuple(dc.PROMPTABLE_CLASS_A_METHODS) + dc.OBJECT_ONLY_METHODS == dc.CLASS_A_METHODS
        # the grammar still admits the cell under its corpus SI...
        assert dc.parse_cell_id(HCELL).method == LORA_ARM
        # ...and, unchanged, a prompt-SI spelling parses as a cell id (the
        # grammar is the serving vocabulary; the STATISTICS exclude it)
        assert dc.parse_cell_id(f"si_safety__{LORA_ARM}").cell_class == "class_a"

    def test_the_stacked_constructor_and_hash_refuse_the_object_only_si(self):
        """The grammar refused `stacked__constv2_mix_full__…`, but the
        lower-level wrapper constructor and the hash boundary validated
        against the BROAD PETRI_SI_IDS and could build and hash a stack
        the grammar cannot name. Both now admit PROMPTABLE SIs only."""
        obj = {"kind": KIND, "layers": [0], "rank": RANK, "positions": "all"}
        with pytest.raises(dc.DefenseCellError, match="OBJECT-ONLY"):
            dc.stacked_resolved_spec(HSI, b"never a system prompt", obj)
        ok = dc.stacked_resolved_spec("safety", b"a real prefix", obj)    # unchanged
        assert ok["si_id"] == "safety"
        forged = {**ok, "si_id": HSI}
        with pytest.raises(cm.ManifestError, match="promptable"):
            cm.petri_intervention_hash(forged, artifact_bytes=b"x", sidecar_bytes=b"",
                                       si_prefix_bytes=b"a real prefix")
        assert len(cm.petri_intervention_hash(ok, artifact_bytes=b"x", sidecar_bytes=b"",
                                              si_prefix_bytes=b"a real prefix")) == 16

    def test_the_manifest_refuses_a_prompted_leg_over_the_object_only_si(self):
        spec = _spec(cells=["none", HCELL, f"prompted__{HSI}"])
        with pytest.raises(cm.ManifestError, match="OBJECT-ONLY"):
            cm.petri_expected_behavior_keys(spec)
        # and a prefix hash keyed on it is not a promptable SI's
        with pytest.raises(cm.ManifestError, match="non-promptable"):
            cm.validate_petri_spec(_spec(prompted_prefix_hashes={
                "safety": "ab" * 8, HSI: "cd" * 8}))
        # the clean spec (object cell only) is fine and adds no variant dir
        clean = _spec(cells=["none", HCELL])
        cm.validate_petri_spec(clean)
        assert cm.petri_expected_behavior_keys(clean) == sorted(clean.behaviors)

    def test_the_corpus_pseudo_instruction_resolves_as_a_corpus_cell(self):
        from sisteer.data.instructions import BY_ID, resolve_si
        from sisteer.eval.jailbreak import resolve_instruction
        inst = resolve_si(SI)
        assert inst.category == "corpus_cell"
        assert inst.checker == "external"
        assert inst.checker_kwargs == {"suite": "corpus_cell"}
        assert resolve_instruction(SI) is inst is BY_ID[SI]
        # the control corpus resolves too (a parent; on this lineage a served cell as well)
        assert resolve_si("constv2ctl2-full").category == "corpus_cell"

    def test_the_kind_is_adapted_and_no_new_op_entered_the_vocabulary(self):
        assert KIND in la.registered_kinds()
        assert KIND not in la.UNSERVABLE_KIND_REASONS
        ops = _adapt(_sub_blob())
        assert {op.op for op in ops} == {la.OP_LORA_OPROJ}
        assert {la.op_site(op) for op in ops} == {la.HOOK_SITE_OPROJ}

    def test_the_launcher_cell_token_parses_through_the_one_grammar(self):
        """The tinycap ``--cells`` token is the TrainTest id, translated by
        the ONE vocabulary and validated by defense_cells.parse_cell_id."""
        C = pytest.importorskip("sisteer.capeval.tinycap.contracts")
        parsed = C.parse_launcher_cell(CELL)
        assert parsed.harness_id == HCELL
        assert parsed.si == SI and parsed.arm == LORA_ARM
        # the prompted spelling of the corpus cell is refused by the
        # authoritative validator the parser hands the translated id to
        # (a DefenseCellError, which the parser does not re-wrap)
        with pytest.raises((C.CellNameError, dc.DefenseCellError), match="OBJECT-ONLY"):
            C.parse_launcher_cell(f"prompted__{SI}")

    def test_the_safety_defense_builder_skips_the_derived_arm(self):
        """scripts/build_safety_defense.py sweeps `list(ARMS)` and calls
        every arm's trainer — the derived arm's refusal must be a deliberate
        SKIP, never a recorded 'native trainer rejected the pairs'."""
        src = (REPO / "scripts" / "build_safety_defense.py").read_text()
        assert "is_derived_arm(arm)" in src
        assert "from sisteer.methods import ARMS, is_derived_arm" in src


class TestMainPortTranslation:
    def test_turn_end_ids_ride_through_to_every_op(self):
        """Main's guard passes ``turn_end_ids`` to EVERY adapter (the
        all-turns mask, admitted for a KNOWN response fit only); the port
        must accept it and forward it to both parents' ops, exactly as
        lora_all does. (The real object is an all-position fit and serves
        at positions=all; this pins the API, not the lane's deployment.)"""
        turn_end = (11, 12)
        ops = _adapt(_sub_blob(positions="response"),
                     spec={"kind": KIND, "positions": la.POSITIONS_RESPONSE_ALL_TURNS},
                     turn_end=turn_end)
        assert {op.positions for op in ops} == {la.POSITIONS_RESPONSE_ALL_TURNS}
        assert all(op.turn_end_ids == turn_end for op in ops)
        assert all(op.header_ids == HEADER for op in ops)
        blob = _sub_blob(positions="response")
        ref = la.get_adapter("lora_all")(
            {"kind": "lora_all", "positions": la.POSITIONS_RESPONSE_ALL_TURNS},
            _blob_bytes({"kind": "lora_all", "layers": [0, 2], "rank": RANK,
                         "A": blob["A_pos"], "B": blob["B_pos"],
                         "positions": "response",
                         "response_header_len": len(HEADER)}),
            b"", header_ids=HEADER, turn_end_ids=turn_end)
        # an all-position fit is NOT admitted under the all-turns mask (the
        # shared guard's rule, inherited unchanged by this kind)
        with pytest.raises(la.AdapterError, match="all-turns"):
            _adapt(_sub_blob(positions="all"),
                   spec={"kind": KIND, "positions": la.POSITIONS_RESPONSE_ALL_TURNS},
                   turn_end=turn_end)
        assert [(o.positions, o.header_ids, o.turn_end_ids) for o in ops[::2]] == \
               [(o.positions, o.header_ids, o.turn_end_ids) for o in ref]


def _spec(**over) -> cm.PetriSpec:
    kwargs = dict(
        audit="derived-serving-test",
        behaviors=list(cm.PETRI_BEHAVIORS),
        cells=["none", HCELL],
        scenario_bundle_hash="ab" * 8,
        auditor_model="anthropic/claude-sonnet-5",
        judge_model="anthropic/claude-opus-5",
        scenarios_model="anthropic/claude-sonnet-5",
        prompt_provenance="package-pinned@petri-bloom==0.2.6",
        target_model=GEMMA_IT,
        target_revision=GEMMA_IT_REV,
        serving_backend="proxy_vllm",
        serving_mode="exact_replay",
        serving_packages={"vllm": "0.25.1+cu129", "vllm-lens": "1.2.1"},
        serving_launch_flags=["--enforce-eager"],
        chat_template_hash="12" * 8,
        tool_parser="llama3_json",
        serving_impl_commit="fedcba9876" * 4,
        # a promptable SI's hash: validate_petri_spec wants a non-empty map,
        # and no prompted cell is in play here so nothing is verified against it
        prompted_prefix_hashes={"safety": "ac3a58f61d53cf24"},
        bloom_options=cm.BloomOptions(
            max_turns=15, enable_prefill=False, enable_rollback=False,
            compaction=False, realism_filter=False, epochs=1, seed=0),
    )
    kwargs.update(over)
    return cm.PetriSpec(**kwargs)


def _parent_block(cell, obj_sha, fp, budget_sha="bb" * 32):
    """One parent block in the real derived_meta.json's shape (survey §K)."""
    return {"cell_id": cell, "model_key": "gemma4e4b_base",
            "object_path": f"/vol/out/gemma4e4b_base/cells_all/{cell}/object.pt",
            "object_sha256": obj_sha, "budget_meta_sha256": budget_sha,
            "fit_fingerprint": fp,
            "model_id": "google/gemma-4-E4B", "model_revision": "4" * 40,
            "model_revision_source": "train_budget_meta.json",
            "positions": "all", "response_header_len": None,
            "target_layers": [0, 2], "template_hash": "e3b0c44298fc1c14",
            "training_budget_kind": "document_tokens"}


def _stage_derived_main(bank: Path, cell_id: str = CELL, *, blob=None,
                   derived_id: str | None = None, parents=None,
                   also_budget_sidecar: bool = False,
                   break_sha: bool = False, no_derived_id: bool = False) -> Path:
    """A staged DERIVED cell dir shaped like the volume's:
    ``<bank>/<cell>/{object.pt, derived_meta.json}`` — the sidecar binds
    the object bytes to two complete parent identities, and the SAME
    provenance rides inside the bytes."""
    from sisteer.eval.defense_cell import sha256_file

    d = bank / cell_id
    d.mkdir(parents=True, exist_ok=True)
    p = d / "object.pt"
    payload = dict(blob if blob is not None else _sub_blob())
    parents = parents or {
        "parent_pos": _parent_block(f"{SI}__lora_r4_ce", "aa" * 32, "d3" * 32),
        "parent_neg": _parent_block("constv2ctl2-full__lora_r4_ce", "cc" * 32, "39" * 32)}
    prov = {"kind": KIND, "derived_id": derived_id or cell_id, **parents}
    payload["provenance"] = prov
    torch.save(payload, p)
    sha = sha256_file(p)
    meta = {**prov, "model_key": "gemma4e4b_base", "model_id": "google/gemma-4-E4B",
            "model_revision": "4" * 40, "positions": payload["positions"],
            "response_header_len": payload.get("response_header_len"),
            "rank": payload["rank"],
            "layers": payload["layers"], "builder": "scripts/sdfbase/lora_subtract.py",
            "object_sha256": ("00" * 32) if break_sha else sha}
    if no_derived_id:
        meta.pop("derived_id")
    (d / "derived_meta.json").write_text(json.dumps(meta, indent=1))
    if also_budget_sidecar:
        (d / "train_budget_meta.json").write_text(
            json.dumps({"cell_id": cell_id, "completion_tokens": 512,
                        "object_sha256": sha}))
    return p


BEHAVIOR_FILES = {b: f"# behavior {b}\n".encode() for b in cm.PETRI_BEHAVIORS}
PROMPT_FILES = {"auditor": b"synthetic auditor prompt", "judge": b"synthetic judge prompt"}


class TestRealRegistrySource:
    def test_the_dedicated_config_is_the_roster(self):
        cfg = yaml.safe_load(SDFBASE_CFG.read_text())
        assert cfg["si_ids"] == [SI]
        assert cfg["class_a_arms"] == [LORA_ARM]
        assert cfg["class_b_arms"] == [] and cfg["class_c_arms"] == []
        assert cfg["model"] == GEMMA_IT and cfg["model_revision"] == GEMMA_IT_REV
        assert Path(cfg["defenses_dir"]).is_absolute()
        # every other key is the reduced config's, verbatim
        ref = yaml.safe_load((REPO / "configs/two_si_gemma4e4b_reduced.yaml").read_text())
        swapped = {"si_ids", "class_a_arms", "class_b_arms", "class_c_arms", "defenses_dir"}
        assert {k for k in set(cfg) | set(ref) if cfg.get(k) != ref.get(k)} == swapped

    def test_traintest_registry_enumerates_the_derived_cell(self, tmp_path):
        from sisteer.eval import defense_cell as tt
        cfg = yaml.safe_load(SDFBASE_CFG.read_text())
        cfg["defenses_dir"] = str(tmp_path / "bank")
        cells = tt.registry_from_cfg(cfg)
        ids = [c.id for c in cells]
        assert CELL in ids
        cell = tt.by_id(cells)[CELL]
        assert cell.class_ == "A" and cell.arm == LORA_ARM and cell.si_id == SI
        assert Path(cell.object_path) == tmp_path / "bank" / CELL / "object.pt"
        # main's Cartesian registry ALSO mints a prompted__<si> row for every
        # si_id — an INERT row here: the harness grammar refuses to translate
        # it (parse_cell_id), so no consumer can ever serve it, and the
        # launcher's cell parser refuses the token. Pinned so the
        # fabricated-baseline hazard the sdf-base lineage's explicit-cells
        # mode removed is at least visible on main.
        assert ids == ["none", f"prompted__{SI}", CELL]
        with pytest.raises(dc.DefenseCellError, match="OBJECT-ONLY"):
            ra.harness_cell_id(f"prompted__{SI}")

    def test_the_real_source_resolves_the_cell_from_a_synthetic_bank(self, tmp_path):
        bank = tmp_path / "bank"
        p = _stage_derived_main(bank)
        source = ra.traintest_cell_source(_spec(), repo_root=REPO,
                                          config_path=SDFBASE_CFG,
                                          defenses_dir=bank)
        entry = source.get(HCELL)
        assert entry is not None
        assert entry.kind == KIND and entry.si_id == HSI
        assert Path(entry.object_path) == p
        assert entry.sidecar_path is None            # derived_meta.json is NOT the
        assert entry.serving_path == "lens"          # <artifact>.json sidecar convention
        spec = dict(entry.resolved_spec)
        assert spec["kind"] == KIND and spec["arm"] == LORA_ARM and spec["si_id"] == HSI
        assert spec["layers"] == [0, 2] and spec["rank"] == RANK
        assert spec["positions"] == "all" and spec["response_header_len"] is None
        # the embedded provenance (parent cell ids + shas) IS part of the
        # hashed identity (survey §B.3)
        assert spec["provenance"]["parent_pos"]["cell_id"] == f"{SI}__lora_r4_ce"
        assert spec["provenance"]["parent_neg"]["object_sha256"] == "cc" * 32
        # the base cell resolves, the object-only SI has no prompted entry
        assert source.get("none").kind == cm.NONE_KIND
        with pytest.raises(dc.DefenseCellError, match="OBJECT-ONLY"):
            source.get(f"prompted__{HSI}")
        # ...and enumeration files the cell with an intervention hash
        cs = dc.enumerate_cells(_spec(), source, behavior_files=BEHAVIOR_FILES,
                                prompt_files=PROMPT_FILES)
        assert [a.cell_id for a in cs.absent] == []
        present = {c.cell_id: c for c in cs.canonical}
        assert set(present) >= {"none", HCELL}
        derived = present[HCELL]
        assert derived.kind == KIND and len(derived.intervention_hash) == 16
        # the route hash: artifact bytes || the (absent) sidecar, stated as
        # b"" || the resolved spec — the blob's embedded provenance included
        assert derived.intervention_hash == cm.petri_intervention_hash(
            entry.resolved_spec, artifact_bytes=p.read_bytes(), sidecar_bytes=b"")

    def test_the_serving_path_follows_the_source_policy(self, tmp_path):
        """MOVED PIN (was
        `LORA_ARM not in ra.LORA_EXPORTED_ARMS` at the registration test):
        the subtraction arm serves on the LENS under the default policy —
        byte-identical to before the policy existed — and as a NATIVE
        adapter under `lora_native`, where the table derives `none__lora0`
        and no `none__lens0`."""
        bank = tmp_path / "bank"
        _stage_derived_main(bank)
        lens_src = ra.traintest_cell_source(_spec(), repo_root=REPO,
                                            config_path=SDFBASE_CFG,
                                            defenses_dir=bank)
        native_src = ra.traintest_cell_source(
            _spec(), repo_root=REPO, config_path=SDFBASE_CFG,
            defenses_dir=bank, serving_path_policy="lora_native")
        assert lens_src.serving_path_policy == "lens"          # the default
        lens_entry, native_entry = lens_src.get(HCELL), native_src.get(HCELL)
        assert lens_entry.serving_path == "lens"
        assert native_entry.serving_path == "lora"
        # the policy decides the PATH and nothing else about the entry
        assert dataclasses_replace(native_entry, serving_path="lens") == lens_entry
        # lens standup: unchanged — the derived none__lens0, no lora leg
        cs = dc.enumerate_cells(_spec(), lens_src, behavior_files=BEHAVIOR_FILES,
                                prompt_files=PROMPT_FILES)
        assert [c.cell_id for c in cs.controls] == [dc.NOOP_LENS_CELL_ID]
        # lora-native standup: the zero-adapter leg only, hash == none's
        cs = dc.enumerate_cells(_spec(), native_src, behavior_files=BEHAVIOR_FILES,
                                prompt_files=PROMPT_FILES, lens_controls=False)
        assert [a.cell_id for a in cs.absent] == []
        by_id = {c.cell_id: c for c in cs.all_served()}
        assert [c.cell_id for c in cs.controls] == [dc.NOOP_LORA_CELL_ID]
        assert dc.NOOP_LENS_CELL_ID not in by_id
        lora0 = by_id[dc.NOOP_LORA_CELL_ID]
        assert lora0.control_of == "none"
        assert lora0.intervention_hash == by_id["none"].intervention_hash
        assert lora0.serving_identity == "proxy_vllm+lora+exact_replay"
        assert by_id[HCELL].serving_identity == "proxy_vllm+lora+exact_replay"

    def test_a_masked_fit_refuses_under_lora_native_by_name(self, tmp_path):
        """A response-fit subtraction is a lens object: `lora_native` refuses
        it at the SOURCE, naming the cell and the fit leg (native multi-LoRA
        has no position mask); the default policy still serves it on the lens."""
        bank = tmp_path / "bank"
        _stage_derived_main(bank, blob=_sub_blob(positions="response"))
        lens_src = ra.traintest_cell_source(_spec(), repo_root=REPO,
                                            config_path=SDFBASE_CFG,
                                            defenses_dir=bank)
        assert lens_src.get(HCELL).serving_path == "lens"
        native_src = ra.traintest_cell_source(
            _spec(), repo_root=REPO, config_path=SDFBASE_CFG,
            defenses_dir=bank, serving_path_policy="lora_native")
        with pytest.raises(ra.RegistryAdapterError) as exc:
            native_src.get(HCELL)
        msg = str(exc.value)
        assert HCELL in msg and "positions='response'" in msg
        assert "position-masked" in msg and "lora-native" in msg

    def test_an_unstaged_cell_is_absent_with_the_artifact_path(self, tmp_path):
        bank = tmp_path / "empty"
        bank.mkdir()
        source = ra.traintest_cell_source(_spec(), repo_root=REPO,
                                          config_path=SDFBASE_CFG, defenses_dir=bank)
        cs = dc.enumerate_cells(_spec(), source, behavior_files=BEHAVIOR_FILES,
                                prompt_files=PROMPT_FILES)
        absent = {a.cell_id: a.reason for a in cs.absent}
        assert set(absent) == {HCELL}
        assert str(bank / CELL / "object.pt") in absent[HCELL]

    def test_the_source_refuses_another_models_manifest(self, tmp_path):
        bank = tmp_path / "bank"
        _stage_derived_main(bank)
        with pytest.raises(ra.RegistryAdapterError, match="base model"):
            ra.traintest_cell_source(_spec(target_revision="9" * 40), repo_root=REPO,
                                     config_path=SDFBASE_CFG, defenses_dir=bank)
        # the model pin is checked BEFORE the SI vocabulary: a manifest that
        # also drags in an undeclared SI still names the model mismatch
        with pytest.raises(ra.RegistryAdapterError, match="base model"):
            ra.traintest_cell_source(
                _spec(target_revision="9" * 40,
                      cells=["none", HCELL, "si_safety__psr_kl_all"]),
                repo_root=REPO, config_path=SDFBASE_CFG, defenses_dir=bank)

    def test_a_config_without_the_corpus_si_refuses_vocabulary_drift(self, tmp_path):
        """The reduced config declares the three safety SIs only: asking it
        for the derived cell is vocabulary drift, refused by name."""
        with pytest.raises(ra.RegistryAdapterError, match="constv2-mix-full"):
            ra.traintest_cell_source(
                _spec(), repo_root=REPO,
                config_path=REPO / "configs/two_si_gemma4e4b_reduced.yaml",
                defenses_dir=tmp_path)

    def test_a_wrong_kind_artifact_under_the_cell_refuses(self, tmp_path):
        bank = tmp_path / "bank"
        d = bank / CELL
        d.mkdir(parents=True)
        torch.save({"kind": "lora_all", "layers": [0], "rank": 1,
                    "A": {"0": torch.zeros(4)}, "B": {"0": torch.zeros(4)},
                    "positions": "all", "response_header_len": None},
                   d / "object.pt")
        source = ra.traintest_cell_source(_spec(), repo_root=REPO,
                                          config_path=SDFBASE_CFG, defenses_dir=bank)
        with pytest.raises(ra.RegistryAdapterError, match="vocabulary drift"):
            source.get(HCELL)


# ---------------------------------------------------------------------------
# 6. the derived verdict + the superseded-parent preflight
# ---------------------------------------------------------------------------


def _lora_all_blob(seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    layers = [0, 2]
    return {"kind": "lora_all", "layers": layers, "rank": RANK,
            "A": {str(l): torch.randn(D_IN, RANK, generator=g) for l in layers},
            "B": {str(l): torch.randn(D_OUT, RANK, generator=g) for l in layers},
            "positions": "all", "response_header_len": None}


def _stage_parent(root: Path, cell_id: str, seed: int, fp: str) -> Path:
    """A FULLY staged synthetic parent in the volume's bank layout:
    ``<root>/gemma4e4b_base/cells_all/<cell>/{object.pt,
    train_budget_meta.json, COMPLETE}`` — a document-schema budget sidecar
    and a coherent COMPLETE manifest, what the staged-cell gate validates."""
    from sisteer.eval.defense_cell import sha256_file

    d = root / "gemma4e4b_base" / "cells_all" / cell_id
    d.mkdir(parents=True, exist_ok=True)
    torch.save(_lora_all_blob(seed), d / "object.pt")
    meta = {"cell_id": cell_id, "model_id": "google/gemma-4-E4B",
            "model_revision": "4" * 40,
            "object_sha256": sha256_file(d / "object.pt"),
            "training_budget_kind": "document_tokens",
            "sequence_max_len": 256, "completion_tokens": None,
            "sample_manifest_sha": "ab" * 32, "positions": "all",
            "response_header_len": None, "target_layers": [0, 2],
            "template_hash": "e3b0c44298fc1c14", "fit_fingerprint": fp}
    (d / "train_budget_meta.json").write_text(json.dumps(meta))
    files = {n: sha256_file(d / n) for n in ("object.pt", "train_budget_meta.json")}
    (d / "COMPLETE").write_text(json.dumps(
        {"cell_id": cell_id, "fit_fingerprint": fp, "files": files,
         "object_file": "object.pt", "object_sha256": files["object.pt"]}))
    return d


def _superseded_record(cell_id: str, *, prior_fp: str, announced_fp: str | None) -> dict:
    """The volume's ``superseded_by.json`` shape (survey §K; the sdf-base
    fitting lane's snapshot record): the EARLIER fit's fingerprint at the top
    and the continuation it announces under ``superseded_by``."""
    sb = {"reason": "cont4 re-issue at a longer cap", "continue_cap": 16,
          "continue_lr_mult": 0.25, "stopper_rule": "replay_exempt"}
    if announced_fp is not None:
        sb["expected_refit_fingerprint"] = announced_fp
    return {"cell_id": cell_id, "fit_fingerprint": prior_fp,
            "snapshot": f"/vol/out/gemma4e4b_base_r2cap4c/cells_all/{cell_id}",
            "snapshot_bank": "/vol/out/gemma4e4b_base_r2cap4c",
            "superseded_by": sb, "at": "2026-09-04T01:38:31+00:00"}


@pytest.fixture()
def volume(tmp_path):
    """A synthetic bank shaped like the volume: two fully staged parents
    under ``out/`` and the derived object under ``objects/`` whose sidecar
    binds them by sha AND by fit fingerprint."""
    from sisteer.eval.defense_cell import sha256_file

    root = tmp_path / "out"
    POS_FP, NEG_FP = "d3" * 32, "39" * 32
    pos_dir = _stage_parent(root, f"{SI}__lora_r4_ce", seed=1, fp=POS_FP)
    neg_dir = _stage_parent(root, "constv2ctl2-full__lora_r4_ce", seed=2, fp=NEG_FP)
    parents = {}
    for role, d, fp in (("parent_pos", pos_dir, POS_FP), ("parent_neg", neg_dir, NEG_FP)):
        parents[role] = _parent_block(
            d.name, sha256_file(d / "object.pt"), fp,
            budget_sha=sha256_file(d / "train_budget_meta.json"))
    pos_blob = torch.load(pos_dir / "object.pt", weights_only=True)
    neg_blob = torch.load(neg_dir / "object.pt", weights_only=True)
    blob = LoRASubtractionObject.compose_blob(pos_blob, neg_blob)
    obj = _stage_derived_main(tmp_path / "objects", CELL, blob=blob, parents=parents)
    import types
    return types.SimpleNamespace(root=root, obj=obj, pos_dir=pos_dir,
                                 neg_dir=neg_dir, POS_FP=POS_FP, NEG_FP=NEG_FP)


class TestDerivedVerdictAndPreflight:
    def test_the_sha_bound_verdict_passes_on_a_coherent_bank(self, volume):
        from sisteer.eval.train_budget import (derived_object_problems,
                                               object_identity_kind)
        assert object_identity_kind(volume.obj) == "derived"
        assert derived_object_problems(volume.obj, derived_id=CELL) == []
        assert derived_object_problems(volume.obj, derived_id=CELL,
                                       parents_root=volume.root) == []

    def test_the_verdict_refuses_stale_mislabelled_or_ambiguous_dirs(self, tmp_path):
        from sisteer.eval.train_budget import (derived_object_problems,
                                               object_identity_kind)
        p = _stage_derived_main(tmp_path / "stale", break_sha=True)
        assert any("object_sha256 mismatch" in s for s in derived_object_problems(p))
        p = _stage_derived_main(tmp_path / "wrongid",
                           derived_id="constv2ctl2-full__lora_r4_ce_minus_ctl")
        assert any("wrong-identity" in s
                   for s in derived_object_problems(p, derived_id=CELL))
        p = _stage_derived_main(tmp_path / "ambiguous", also_budget_sidecar=True)
        assert object_identity_kind(p) == "ambiguous"
        assert any("BOTH" in s for s in derived_object_problems(p))
        # a bare dir (no sidecar) is refused, never trusted
        d = tmp_path / "bare" / CELL
        d.mkdir(parents=True)
        torch.save(_sub_blob(), d / "object.pt")
        assert object_identity_kind(d / "object.pt") is None
        assert any("sidecar" in s for s in derived_object_problems(d / "object.pt"))

    def test_a_derived_object_is_bound_to_its_cell_id_everywhere(self, volume, tmp_path):
        """A coherent artifact requested under another cell, or a sidecar
        recording NO derived_id, refuses through the verdict, the preflight
        AND the served mirror — never a document fit under the wrong name."""
        from sisteer.eval import derived_preflight as dp
        from sisteer.eval import sweep
        from sisteer.eval.train_budget import derived_object_problems
        wrong = "constv2ctl2-full__lora_r4_ce_minus_ctl"
        # wrong id, coherent bytes
        probs = derived_object_problems(volume.obj, derived_id=wrong, parents_root=volume.root)
        assert any("wrong-identity" in s for s in probs)
        rec = dp.preflight_derived_object(volume.obj, derived_id=wrong, parents_root=volume.root)
        assert not rec["ok"] and any("wrong-identity" in s for s in rec["problems"])
        with pytest.raises(sweep.UnrecordedMirrorError, match="wrong-identity"):
            sweep.resolve_training_mirror(wrong, volume.obj)
        assert sweep.resolve_training_mirror(CELL, volume.obj).status == sweep.DOCUMENT_FIT_STATUS
        # no id recorded: refused even with the expected id supplied
        p = _stage_derived_main(tmp_path / "unnamed", no_derived_id=True)
        for got in (derived_object_problems(p), derived_object_problems(p, derived_id=CELL)):
            assert any("records no derived_id" in s for s in got)
        rec = dp.preflight_derived_object(p, derived_id=CELL, parents_root=volume.root)
        assert not rec["ok"] and any("records no derived_id" in s for s in rec["problems"])
        with pytest.raises(sweep.UnrecordedMirrorError, match="records no derived_id"):
            sweep.resolve_training_mirror(CELL, p)

    def test_the_verdict_refuses_a_parent_refit_under_the_same_id(self, volume):
        from sisteer.eval.train_budget import derived_object_problems
        _stage_parent(volume.root, volume.neg_dir.name, seed=77, fp="e" * 64)
        assert derived_object_problems(volume.obj, derived_id=CELL) == []   # bytes ok
        probs = derived_object_problems(volume.obj, derived_id=CELL,
                                        parents_root=volume.root)
        assert any("not the object staged under that id now" in p for p in probs)

    def test_the_sha_verdict_is_silent_about_a_supersession_marker(self, volume):
        """survey §K, exactly: the volume's positive parent carries a
        superseded_by.json announcing the continuation whose bytes the dir
        now holds; the sha-bound check neither reads nor needs it."""
        from sisteer.eval.train_budget import derived_object_problems
        (volume.pos_dir / "superseded_by.json").write_text(json.dumps(
            _superseded_record(volume.pos_dir.name, prior_fp="ba" * 32,
                               announced_fp=volume.POS_FP)))
        assert derived_object_problems(volume.obj, derived_id=CELL,
                                       parents_root=volume.root) == []
        # ...and would ALSO be silent about a marker announcing a DIFFERENT
        # continuation — the gap the preflight below closes
        (volume.pos_dir / "superseded_by.json").write_text(json.dumps(
            _superseded_record(volume.pos_dir.name, prior_fp="ba" * 32,
                               announced_fp="05" * 32)))
        assert derived_object_problems(volume.obj, derived_id=CELL,
                                       parents_root=volume.root) == []

    def test_the_preflight_records_the_marker_and_accepts_the_announced_parent(self, volume):
        from sisteer.eval import derived_preflight as dp
        rec = dp.preflight_derived_object(volume.obj, derived_id=CELL,
                                          parents_root=volume.root)
        assert rec["ok"] and rec["problems"] == []
        assert set(rec["parents"]) == {"parent_pos", "parent_neg"}
        for role in rec["parents"]:
            assert rec["parents"][role]["superseded_by"] is None
            assert rec["parents"][role]["superseded_check"] == dp.SUPERSEDED_ABSENT
        marker = _superseded_record(volume.pos_dir.name, prior_fp="ba" * 32,
                                    announced_fp=volume.POS_FP)
        (volume.pos_dir / "superseded_by.json").write_text(json.dumps(marker))
        rec = dp.preflight_derived_object(volume.obj, derived_id=CELL,
                                          parents_root=volume.root)
        assert rec["ok"], rec["problems"]
        pos = rec["parents"]["parent_pos"]
        assert pos["superseded_by"] == marker                      # recorded verbatim
        assert pos["superseded_check"] == dp.SUPERSEDED_CONTINUATION_IS_THIS_PARENT
        assert pos["fit_fingerprint"] == volume.POS_FP
        assert rec["parents"]["parent_neg"]["superseded_check"] == dp.SUPERSEDED_ABSENT
        assert dp.assert_derived_preflight(volume.obj, consumer="t", derived_id=CELL,
                                           parents_root=volume.root) == rec

    def test_the_preflight_refuses_a_marker_naming_another_continuation(self, volume):
        from sisteer.eval import derived_preflight as dp
        (volume.pos_dir / "superseded_by.json").write_text(json.dumps(
            _superseded_record(volume.pos_dir.name, prior_fp="ba" * 32,
                               announced_fp="05" * 32)))
        rec = dp.preflight_derived_object(volume.obj, derived_id=CELL,
                                          parents_root=volume.root)
        assert not rec["ok"]
        assert rec["parents"]["parent_pos"]["superseded_check"] == dp.SUPERSEDED_MISMATCH
        (problem,) = rec["problems"]
        # both fingerprints are NAMED
        assert "05" * 32 in problem and volume.POS_FP in problem
        assert "parent_pos" in problem
        with pytest.raises(SystemExit, match=r"\[preflight\].*another|parent_pos"):
            dp.assert_derived_preflight(volume.obj, consumer="preflight",
                                        derived_id=CELL, parents_root=volume.root)

    def test_the_preflight_refuses_incomplete_or_foreign_markers(self, volume):
        from sisteer.eval import derived_preflight as dp
        # no announced fingerprint: cannot be cross-checked, never passes by absence
        (volume.pos_dir / "superseded_by.json").write_text(json.dumps(
            _superseded_record(volume.pos_dir.name, prior_fp="ba" * 32,
                               announced_fp=None)))
        rec = dp.preflight_derived_object(volume.obj, derived_id=CELL,
                                          parents_root=volume.root)
        assert not rec["ok"] and any("announces no" in p for p in rec["problems"])
        # a marker belonging to another cell
        (volume.pos_dir / "superseded_by.json").write_text(json.dumps(
            _superseded_record("constv2-mix-1m__lora_r4_ce", prior_fp="ba" * 32,
                               announced_fp=volume.POS_FP)))
        rec = dp.preflight_derived_object(volume.obj, derived_id=CELL,
                                          parents_root=volume.root)
        assert not rec["ok"] and any("different cell" in p for p in rec["problems"])
        # a torn marker
        (volume.pos_dir / "superseded_by.json").write_text("{not json")
        rec = dp.preflight_derived_object(volume.obj, derived_id=CELL,
                                          parents_root=volume.root)
        assert not rec["ok"] and any("unreadable" in p for p in rec["problems"])
        assert rec["parents"]["parent_pos"]["superseded_check"] == dp.SUPERSEDED_MISMATCH

    def test_the_preflight_carries_the_sha_verdict_too(self, volume):
        from sisteer.eval import derived_preflight as dp
        _stage_parent(volume.root, volume.neg_dir.name, seed=77, fp="e" * 64)
        rec = dp.preflight_derived_object(volume.obj, derived_id=CELL,
                                          parents_root=volume.root)
        assert not rec["ok"]
        assert any("not the object staged under that id now" in p for p in rec["problems"])

    def test_the_served_mirror_reads_the_derived_sidecar_as_a_document_fit(self, volume):
        """survey §B.1: sweep.read_derived_provenance relays the parents'
        agreeing facts — base model id, the empty template hash, the
        document budget kind — so the instruct-target serve records the
        template check as inapplicable (a transfer), with no new rule."""
        from sisteer.eval import sweep
        prov = sweep.resolve_training_mirror(CELL, volume.obj)
        assert prov.source == f"derived_meta:{KIND}"
        assert prov.status == sweep.DOCUMENT_FIT_STATUS
        assert prov.model_id == "google/gemma-4-E4B"
        assert prov.trained_template_hash == "e3b0c44298fc1c14"
        assert prov.training_budget_kind == "document_tokens"
        assert prov.positions_at_fit == "all"
