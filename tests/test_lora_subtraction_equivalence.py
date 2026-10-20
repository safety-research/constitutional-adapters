"""The rank-8 FOLD as the equivalence test for the ``lora_subtraction``
serving mechanism.

Three ways to compute the derived edit at one o_proj site, for factors
``A[l] [d_in, r]``, ``B[l] [d_out, r]`` (hook ``x @ A @ Bᵀ``, so the dense
per-layer update is ``ΔW = B @ Aᵀ``):

1. the CLASS — ``LoRASubtractionObject.apply``: the positive parent's own
   MultiLayerLoRAObject hook at +scale, then the negative parent's at −scale
   (two casts to the model dtype, one per hook);
2. the served TWO-OP lens path — ``get_adapter("lora_subtraction")`` → two
   OP_LORA_OPROJ ops per layer (coeff +1 / −1) → ``apply_oproj_ops``, which
   casts each op's delta separately and adds sequentially (two casts);
3. the ONE-OP rank-8 FOLD — ``A_f = [A₊ | A₋]`` (d_in × 2r), ``B_f = [B₊ | −B₋]``
   (d_out × 2r), one OP_LORA_OPROJ op (ONE cast): the shape a native
   multi-LoRA export would serve.

In real arithmetic all three are identical: ``x A_f B_fᵀ = x A₊ B₊ᵀ − x A₋
B₋ᵀ``. In fp32 they agree to kernel-order noise (pinned here at
``allclose(atol=1e-5, rtol=1e-5)``); on INTEGER-valued factors and
activations the two lens paths are BITWISE equal (every intermediate is an
exactly representable integer). Under bf16 ACTIVATIONS the one-cast fold and
the two-cast paths differ by the second rounding — that deviation is
MEASURED here (max |diff|, in bf16 ulps of the output, the fraction of
elements that differ, the ratio to the edit itself) and printed, never
assumed: the plan admits the fold as a serving fallback only with these
numbers recorded.

The REAL staged object (``constv2-mix-full__lora_r4_ce_minus_ctl``, 42
layers, rank 4, gemma-4-E4B o_proj shapes) is exercised when its grading host copy
is present (sha-asserted ``be93b469…``): sidecar ↔ bytes ↔ embedded
provenance through ``derived_object_problems``, the adapter's 84 ops, and
the two-op vs fold vs class-arithmetic agreement at fp32 and bf16 on random
activations of the real shapes (the class's hook body is replicated on
tensors — hydrating it needs the 4B model, which a CPU test does not load).
"""

from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path

import pytest
import torch

from sisteer.capeval.serving import lens_adapters as la
from sisteer.extraction.multilayer import (LoRASubtractionObject,
                                           MultiLayerLoRAObject)

from _env import host_site_path  # tests/ helper: a host-only data path through the site config

KIND = "lora_subtraction"
REAL_OBJECT = (host_site_path("$STAGE/sdfbase/full22_objects/gemma4e4b/cells")
               / "derived__constv2-mix-full__lora_r4_ce_minus_ctl/object.pt")
REAL_SHA256 = "be93b469a647248c68b48482c42e5af8f24da0993ba789f35e7a3aeeb2663391"
REAL_DERIVED_ID = "constv2-mix-full__lora_r4_ce_minus_ctl"
FP32_TOL = dict(atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------
# helpers: blobs, ops, the fold
# ---------------------------------------------------------------------------


def _lora_all_blob(layers, shapes, rank, seed):
    """An all-position lora_all save blob over ``shapes[l] = (d_in, d_out)``
    with factors at 0.3 x unit scale."""
    g = torch.Generator().manual_seed(seed)
    return {"kind": "lora_all", "layers": list(layers), "rank": rank,
            "A": {str(l): torch.randn(shapes[l][0], rank, generator=g) * 0.3
                  for l in layers},
            "B": {str(l): torch.randn(shapes[l][1], rank, generator=g) * 0.3
                  for l in layers},
            "positions": "all",
            "response_header_len": None}


def _sub_blob(layers=(0, 1), shapes=None, rank=3, seeds=(11, 23)):
    shapes = shapes or {l: (10, 6) for l in layers}
    pos = _lora_all_blob(layers, shapes, rank, seeds[0])
    neg = _lora_all_blob(layers, shapes, rank, seeds[1])
    return LoRASubtractionObject.compose_blob(pos, neg)


def _bytes(blob) -> bytes:
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _two_ops(blob, header=None):
    return la.get_adapter(KIND)({"kind": KIND}, _bytes(blob), b"",
                                header_ids=header)


def _fold_factors(blob, layer):
    """``A_f = [A₊ | A₋]`` (d_in × 2r), ``B_f = [B₊ | −B₋]`` (d_out × 2r)."""
    l = str(layer)
    a_f = torch.cat([blob["A_pos"][l], blob["A_neg"][l]], dim=1)
    b_f = torch.cat([blob["B_pos"][l], -blob["B_neg"][l]], dim=1)
    return a_f.float(), b_f.float()


def _fold_op(blob, layer, positions=la.POSITIONS_ALL):
    """The ONE-op rank-2r fold in the served op vocabulary: read basis
    transposed (``lowrank_a = A_fᵀ`` [2r, d_in]), write basis ``B_f``."""
    a_f, b_f = _fold_factors(blob, layer)
    return la.HookOp(op=la.OP_LORA_OPROJ, layer=int(layer),
                     lowrank_a=a_f.t().contiguous(),
                     lowrank_b=b_f.contiguous(),
                     positions=positions, skip_bos=False, phase=la.PHASE_BOTH)


def _class_arithmetic(x, out, blob, layer, scale=1.0):
    """LoRASubtractionObject.apply's hook bodies, replicated on tensors:
    the positive parent's MultiLayerLoRAObject._make_hook at +scale, then
    the negative parent's at −scale, each ``coef = x.to(A.dtype) @ A;
    delta = coef @ B.T; out = out + (scale * delta).to(out.dtype)``."""
    l = str(layer)
    cur = out
    for side, sgn in (("pos", 1.0), ("neg", -1.0)):
        A, B = blob[f"A_{side}"][l], blob[f"B_{side}"][l]
        coef = x.to(A.dtype) @ A
        delta = coef @ B.T
        cur = cur + (sgn * scale * delta).to(out.dtype)
    return cur


def _bf16_ulp(magnitude: float) -> float:
    """One bf16 ulp at ``magnitude`` (8 significand bits, 7 stored)."""
    if magnitude == 0.0:
        return 2.0 ** -133
    return 2.0 ** (math.floor(math.log2(magnitude)) - 7)


def _measure(a: torch.Tensor, b: torch.Tensor, edit: torch.Tensor) -> dict:
    """The recorded deviation between two computations of the same edit."""
    diff = (a.float() - b.float()).abs()
    out_mag = float(max(a.float().abs().max(), b.float().abs().max()))
    max_diff = float(diff.max())
    return {"max_abs_diff": max_diff,
            "max_abs_diff_in_bf16_ulps_of_output": max_diff / _bf16_ulp(out_mag),
            "frac_elements_differing": float((diff > 0).float().mean()),
            "max_abs_edit": float(edit.float().abs().max()),
            "max_diff_over_max_edit": (max_diff / float(edit.float().abs().max())
                                       if float(edit.float().abs().max()) > 0
                                       else float("nan"))}


# ---------------------------------------------------------------------------
# 1. the fold is the difference, dimensionally and numerically (fp32)
# ---------------------------------------------------------------------------


def test_dense_delta_w_of_the_fold_is_pos_minus_neg():
    """``B_f @ A_fᵀ == B₊ @ A₊ᵀ − B₋ @ A₋ᵀ`` per layer — the dense update in
    the factor layout the hook uses (``ΔW = B @ Aᵀ``; a ``B @ A`` product
    would be dimensionally wrong on non-square o_proj)."""
    blob = _sub_blob(layers=(0, 2), shapes={0: (10, 6), 2: (10, 6)})
    for layer in (0, 2):
        l = str(layer)
        a_f, b_f = _fold_factors(blob, layer)
        assert a_f.shape == (10, 6) and b_f.shape == (6, 6)   # d_in x 2r, d_out x 2r
        dense_fold = b_f @ a_f.t()                              # [d_out, d_in]
        dense_diff = (blob["B_pos"][l] @ blob["A_pos"][l].t()
                      - blob["B_neg"][l] @ blob["A_neg"][l].t())
        assert dense_fold.shape == (6, 10)
        assert torch.allclose(dense_fold, dense_diff, **FP32_TOL)
        assert dense_fold.abs().max() > 0


def test_two_op_path_equals_the_fold_in_fp32_on_random_activations():
    blob = _sub_blob(layers=(0, 1), shapes={0: (12, 8), 1: (12, 8)}, rank=4)
    ops = _two_ops(blob)
    g = torch.Generator().manual_seed(1)
    for layer in (0, 1):
        x = torch.randn(7, 12, generator=g)
        out = torch.randn(7, 8, generator=g)
        two = la.apply_oproj_ops(x, out, ops, 1.0, layer=layer)
        fold = la.apply_oproj_ops(x, out, (_fold_op(blob, layer),), 1.0,
                                  layer=layer)
        cls = _class_arithmetic(x, out, blob, layer)
        assert not torch.equal(two, out)                       # non-vacuous
        assert torch.allclose(two, fold, **FP32_TOL)
        assert torch.allclose(two, cls, **FP32_TOL)
        assert torch.allclose(fold, cls, **FP32_TOL)
        # scale rides through identically on both lens paths
        two2 = la.apply_oproj_ops(x, out, ops, 2.0, layer=layer)
        fold2 = la.apply_oproj_ops(x, out, (_fold_op(blob, layer),), 2.0,
                                   layer=layer)
        assert torch.allclose(two2, fold2, **FP32_TOL)
        assert torch.allclose(two2 - out, 2.0 * (two - out), **FP32_TOL)


def test_integer_valued_factors_are_bitwise_equal_across_paths():
    """On exactly representable integers every intermediate of both lens
    paths and the class arithmetic is an integer, so the three computations
    are BIT-identical — in fp32 AND in bf16: with factors and activations in
    {−1, 0, 1}, d_in = 8 and r = 2 every partial sum is bounded by
    8 · 2 · 2 + 1 = 33 < 256, inside bf16's exactly-representable integers,
    so a one-cast and a two-cast path cannot round differently."""
    layers = (0,)
    shapes = {0: (8, 6)}
    blob = _sub_blob(layers=layers, shapes=shapes, rank=2)
    g = torch.Generator().manual_seed(4)
    for side in ("pos", "neg"):
        blob[f"A_{side}"]["0"] = torch.randint(-1, 2, (8, 2), generator=g).float()
        blob[f"B_{side}"]["0"] = torch.randint(-1, 2, (6, 2), generator=g).float()
    ops = _two_ops(blob)
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randint(-1, 2, (5, 8), generator=g).to(dtype)
        out = torch.randint(-1, 2, (5, 6), generator=g).to(dtype)
        two = la.apply_oproj_ops(x, out, ops, 1.0, layer=0)
        fold = la.apply_oproj_ops(x, out, (_fold_op(blob, 0),), 1.0, layer=0)
        cls = _class_arithmetic(x, out, blob, 0)
        assert two.dtype == fold.dtype == cls.dtype == dtype
        assert torch.equal(two, fold), dtype
        assert torch.equal(two, cls), dtype
        assert not torch.equal(two, out), dtype                 # non-vacuous


# ---------------------------------------------------------------------------
# 2. the class itself, on a real tiny model (prefill + KV-cached decode)
# ---------------------------------------------------------------------------


def _tiny_parent(model, layers, seed, rank=2):
    obj = MultiLayerLoRAObject(model, layers=list(layers), rank=rank)
    g = torch.Generator().manual_seed(seed)
    for l in layers:
        obj.A[l].data = torch.randn(obj.A[l].shape, generator=g) * 0.3
        obj.B[l].data = torch.randn(obj.B[l].shape, generator=g) * 0.3
    return obj


def test_class_apply_two_op_lens_path_and_fold_agree_on_the_tiny_model(tiny_llama):
    """``LoRASubtractionObject.apply`` vs the adapter's two ops vs the
    one-op fold, at every steered o_proj, across a prefill and a KV-cached
    decode step, fp32: ``allclose(atol=1e-5, rtol=1e-5)`` — and the edit is
    non-vacuous (differs from the bare model AND from the positive parent
    alone, so a dropped or sign-flipped control term fails)."""
    import contextlib

    model, _tok = tiny_llama
    layers = [0, 1]
    pos = _tiny_parent(model, layers, seed=11)
    neg = _tiny_parent(model, layers, seed=23)
    sub = LoRASubtractionObject(pos, neg)
    blob = LoRASubtractionObject.compose_blob(pos.to_blob(), neg.to_blob())
    two_ops = la.get_adapter(KIND)({"kind": KIND}, _bytes(blob), b"")
    fold_ops = tuple(_fold_op(blob, l) for l in layers)
    assert len(two_ops) == 2 * len(layers) and len(fold_ops) == len(layers)

    ids = torch.randint(3, 90, (1, 6), generator=torch.Generator().manual_seed(3))
    P = ids.shape[1]
    decoder = model.model.layers

    def run(mode):
        seen = {l: [] for l in layers}
        state = {"n": 0}
        ops = {"two": two_ops, "fold": fold_ops}.get(mode)

        def capture(l):
            def hook(_mod, args, output):
                x = args[0]
                x2 = x[0] if x.dim() == 3 else x
                out2 = output[0] if output.dim() == 3 else output
                if ops is not None:
                    T = x2.shape[0]
                    ctx = (la.HookContext(la.PHASE_PREFILL, tuple(range(T)))
                           if T > 1 else
                           la.HookContext(la.PHASE_DECODE, (state["n"],)))
                    new = la.apply_oproj_ops(x2, out2, ops, 1.0, layer=l, ctx=ctx)
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

    ref, two, fold = run("class"), run("two"), run("fold")
    bare, pos_only = run("bare"), run("pos_only")
    for l in layers:
        for phase in (0, 1):
            want = ref[l][phase]
            for name, got in (("two-op", two[l][phase]), ("fold", fold[l][phase])):
                assert torch.allclose(got, want, **FP32_TOL), (
                    f"layer {l} phase {phase}: the {name} path does not reproduce "
                    f"LoRASubtractionObject.apply (max abs "
                    f"{float((got - want).abs().max())})")
                edit = (want - bare[l][phase]).abs().max()
                assert float((got - want).abs().max()) < 1e-3 * float(edit)
            assert torch.allclose(two[l][phase], fold[l][phase], **FP32_TOL)
            assert not torch.equal(want, bare[l][phase])
            assert not torch.allclose(want, pos_only[l][phase], rtol=1e-4, atol=1e-6)


# ---------------------------------------------------------------------------
# 3. bf16 activations: MEASURED, printed, bounded
# ---------------------------------------------------------------------------


def _bf16_deviation(blob, layer, x_bf16, out_bf16):
    ops = _two_ops(blob)
    two = la.apply_oproj_ops(x_bf16, out_bf16, ops, 1.0, layer=layer)
    fold = la.apply_oproj_ops(x_bf16, out_bf16, (_fold_op(blob, layer),), 1.0,
                              layer=layer)
    cls = _class_arithmetic(x_bf16, out_bf16, blob, layer)
    assert two.dtype == fold.dtype == cls.dtype == torch.bfloat16
    edit = two.float() - out_bf16.float()
    return {"two_vs_fold": _measure(two, fold, edit),
            "two_vs_class": _measure(two, cls, edit),
            "fold_vs_class": _measure(fold, cls, edit)}, (two, fold, cls, edit)


def test_bf16_activations_two_op_vs_fold_deviation_is_measured_and_bounded(
        capsys):
    """The honest number behind the fallback decision: under bf16 activations
    (the served dtype) the one-cast fold and the two-cast lens path differ
    by the second rounding. Measured on synthetic factors at production-like
    magnitudes, PRINTED, and bounded: no element differs by more than 2
    bf16 ulps of the output magnitude (measured 1.00), while a quarter of
    the elements DO differ and — the edit being only a few output-ulps wide
    — the deviation is a large fraction of the edit (recorded, not
    bounded). The two-cast lens path and the class arithmetic (also two
    casts, in the same order) agree to kernel-order noise (measured 0)."""
    layers = (0, 1)
    shapes = {0: (64, 48), 1: (64, 48)}
    blob = _sub_blob(layers=layers, shapes=shapes, rank=4, seeds=(5, 6))
    # production-like factor magnitudes (the real object: max|A| ~0.19,
    # max|B| ~0.06 on unit-scale activations)
    for side in ("pos", "neg"):
        for l in map(str, layers):
            blob[f"A_{side}"][l] = blob[f"A_{side}"][l] / 3.0 * 0.6
            blob[f"B_{side}"][l] = blob[f"B_{side}"][l] / 3.0 * 0.2
    g = torch.Generator().manual_seed(9)
    worst = {"two_vs_fold": 0.0, "two_vs_class": 0.0}
    for layer in layers:
        x = torch.randn(256, 64, generator=g).to(torch.bfloat16)
        out = torch.randn(256, 48, generator=g).to(torch.bfloat16)
        m, (two, fold, cls, edit) = _bf16_deviation(blob, layer, x, out)
        print(f"[bf16 synthetic layer {layer}] two-op vs fold: "
              f"max|diff| {m['two_vs_fold']['max_abs_diff']:.3e} "
              f"({m['two_vs_fold']['max_abs_diff_in_bf16_ulps_of_output']:.2f} "
              f"bf16 ulps of output), {m['two_vs_fold']['frac_elements_differing']:.4f} "
              f"of elements differ, max|diff|/max|edit| "
              f"{m['two_vs_fold']['max_diff_over_max_edit']:.3e}; "
              f"two-op vs class: max|diff| {m['two_vs_class']['max_abs_diff']:.3e}")
        assert float(edit.abs().max()) > 0                       # non-vacuous
        # the bound that MEANS something at bf16: at most one rounding step
        # of the output dtype separates the one-cast fold from the two-cast
        # path (measured: exactly 1.00 ulp at the worst element)
        assert m["two_vs_fold"]["max_abs_diff_in_bf16_ulps_of_output"] <= 2.0
        # ...and the fold is NOT a drop-in at bf16: a quarter of the
        # elements differ (recorded, never assumed away), and because the
        # edit is itself only a few output-ulps wide the deviation is a
        # LARGE fraction of the edit (0.13 here; 0.11–0.43 on the real
        # object) — the number the fallback decision must be read against
        assert m["two_vs_fold"]["frac_elements_differing"] > 0
        assert m["two_vs_class"]["max_abs_diff_in_bf16_ulps_of_output"] <= 1.0
        worst["two_vs_fold"] = max(worst["two_vs_fold"],
                                   m["two_vs_fold"]["max_abs_diff_in_bf16_ulps_of_output"])
        worst["two_vs_class"] = max(worst["two_vs_class"],
                                    m["two_vs_class"]["max_abs_diff_in_bf16_ulps_of_output"])
    captured = capsys.readouterr()
    assert "two-op vs fold" in captured.out
    with capsys.disabled():
        print(f"\n[bf16 synthetic] worst two-op vs fold {worst['two_vs_fold']:.2f} ulps; "
              f"worst two-op vs class {worst['two_vs_class']:.2f} ulps")


# ---------------------------------------------------------------------------
# 4. the REAL staged object (skip unless the grading host copy is present)
# ---------------------------------------------------------------------------


needs_real_object = pytest.mark.skipif(
    not REAL_OBJECT.is_file(),
    reason=f"the staged derived object is not at {REAL_OBJECT}")


@pytest.fixture(scope="module")
def real_blob():
    sha = hashlib.sha256(REAL_OBJECT.read_bytes()).hexdigest()
    assert sha == REAL_SHA256, (
        f"{REAL_OBJECT} hashes to {sha}, not the pinned {REAL_SHA256} — a "
        "different object than the one the lane staged")
    return torch.load(REAL_OBJECT, map_location="cpu", weights_only=True)


@needs_real_object
def test_real_object_sidecar_binds_the_bytes_and_the_embedded_provenance(real_blob):
    from sisteer.eval.train_budget import derived_object_problems

    assert derived_object_problems(REAL_OBJECT, derived_id=REAL_DERIVED_ID) == []
    assert real_blob["kind"] == KIND
    assert real_blob["layers"] == list(range(42)) and real_blob["rank"] == 4
    assert real_blob["positions"] == "all" and real_blob["response_header_len"] is None
    prov = real_blob["provenance"]
    assert prov["parent_pos"]["cell_id"] == "constv2-mix-full__lora_r4_ce"
    assert prov["parent_neg"]["cell_id"] == "constv2ctl2-full__lora_r4_ce"
    assert prov["model_id"] == "google/gemma-4-E4B"          # FIT on the base model
    for l in range(42):
        for side in ("pos", "neg"):
            assert real_blob[f"A_{side}"][str(l)].shape[1] == 4
            assert real_blob[f"B_{side}"][str(l)].shape == (2560, 4)
            assert real_blob[f"A_{side}"][str(l)].dtype == torch.float32


@needs_real_object
def test_real_object_is_bound_to_its_cell_id_in_the_served_mirror():
    """Resolving the real staged artifact under a wrong cell id used to
    succeed as a document fit."""
    from sisteer.eval import sweep
    prov = sweep.resolve_training_mirror(REAL_DERIVED_ID, REAL_OBJECT)
    assert prov.status == sweep.DOCUMENT_FIT_STATUS and prov.source == f"derived_meta:{KIND}"
    with pytest.raises(sweep.UnrecordedMirrorError, match="wrong-identity"):
        sweep.resolve_training_mirror("definitely-the-wrong-cell", REAL_OBJECT)


@needs_real_object
def test_real_object_adapts_to_84_ops_and_the_fold_agrees(real_blob, capsys):
    """Two ops per layer at the existing site, coeff +1 / −1, the parents'
    own bytes; then, on random activations of the REAL shapes (d_in 2048 on
    the sliding layers, 4096 on the global ones, d_out 2560): the two-op
    path == the fold == the class arithmetic at fp32 (allclose 1e-5), and
    the bf16 deviation MEASURED and printed per probed layer."""
    ops = la.get_adapter(KIND)({"kind": KIND}, REAL_OBJECT.read_bytes(), b"")
    assert len(ops) == 84
    assert [op.layer for op in ops] == [l for l in range(42) for _ in (0, 1)]
    assert [op.coeff for op in ops] == [1.0, -1.0] * 42
    assert {op.positions for op in ops} == {la.POSITIONS_ALL}
    for i, l in enumerate(range(42)):
        assert torch.equal(ops[2 * i].lowrank_a, real_blob["A_pos"][str(l)].t())
        assert torch.equal(ops[2 * i + 1].lowrank_b, real_blob["B_neg"][str(l)])
    g = torch.Generator().manual_seed(2026)
    worst = 0.0
    for layer in (0, 20, 41):
        d_in = real_blob["A_pos"][str(layer)].shape[0]
        x = torch.randn(32, d_in, generator=g)
        out = torch.randn(32, 2560, generator=g)
        two = la.apply_oproj_ops(x, out, ops, 1.0, layer=layer)
        fold = la.apply_oproj_ops(x, out, (_fold_op(real_blob, layer),), 1.0,
                                  layer=layer)
        cls = _class_arithmetic(x, out, real_blob, layer)
        assert not torch.equal(two, out)
        assert torch.allclose(two, fold, **FP32_TOL)
        assert torch.allclose(two, cls, **FP32_TOL)
        m, (two_b, fold_b, cls_b, edit) = _bf16_deviation(
            real_blob, layer, x.to(torch.bfloat16), out.to(torch.bfloat16))
        print(f"[bf16 REAL object layer {layer} d_in={d_in}] two-op vs fold: "
              f"max|diff| {m['two_vs_fold']['max_abs_diff']:.3e} "
              f"({m['two_vs_fold']['max_abs_diff_in_bf16_ulps_of_output']:.2f} "
              f"bf16 ulps of output), {m['two_vs_fold']['frac_elements_differing']:.4f} "
              f"of elements differ, max|diff|/max|edit| "
              f"{m['two_vs_fold']['max_diff_over_max_edit']:.3e}; "
              f"max|edit| {m['two_vs_fold']['max_abs_edit']:.3e}; "
              f"two-op vs class max|diff| {m['two_vs_class']['max_abs_diff']:.3e}")
        assert m["two_vs_fold"]["max_abs_diff_in_bf16_ulps_of_output"] <= 2.0
        assert m["two_vs_class"]["max_abs_diff_in_bf16_ulps_of_output"] <= 1.0
        worst = max(worst, m["two_vs_fold"]["max_abs_diff_in_bf16_ulps_of_output"])
    with capsys.disabled():
        print(f"\n[bf16 REAL object] worst two-op vs fold over layers 0/20/41: "
              f"{worst:.2f} bf16 ulps of output")
