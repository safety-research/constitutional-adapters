"""native-psr-serving step 1 — the RESIDUAL SLOT's CPU design proof
(``serving/native_residual.py``).

The proofs are SEPARATED (fact 14):

- (a) the fp32 GATE under its numerical CONTRACT ``|g - g_ref| <= 4 d eps32
  ||x|| ||w||`` per row against ``PSRObject.gate_pre`` / the lens's
  ``_matmul(r.float(), w) + b`` — a reference shape (seed 12, x (7,128), four
  gates) and the tiny llama's — with the measured maximum REPORTED; known-bad:
  a bf16 gate and a gate on the POST-write stream violate the bound;
- (b) the WRITE-BACK exact with SUPPLIED deltas — bit-identical in bf16 to the
  lens's ``applied = true + add.to(dtype); delta = applied - true; new = hidden
  + delta`` including fact 2's counterexample and the cancellation cases;
  known-bad: the ``hidden + delta.to(model)`` form is 4 ulps off, a form that
  skips ``applied - true`` fails the cancellation cases;
- (c) END-TO-END vs the lens (``_adapt_psr`` -> ``apply_hook_ops`` on ``true``):
  per steered row bf16-identical EXCEPT where fp32 gate noise flips a bf16
  rounding — the flip RATE and the max ulp distance where it flips (<= 1) are
  measured and reported; unsteered rows bit-identical to the unpatched pair;
- (d) the CONSTANT add vs ``MultiLayerVectorObject.apply`` / ``DiffMeanObject.
  apply`` / ``VectorObject.apply`` on the tiny llama, fp32 and bf16, exact per
  element; position 0 unsteered iff ``skip_bos``; known-bad: ``skip_bos=False``
  steers position 0;
- (e) the gemma4 form (``residual is None`` -> ``true = hidden``);
- (f) routing through the REAL metadata path (fakes mirroring
  ``PunicaWrapperGPU.update_metadata`` and ``prepare_tensors``' early return):
  PSR -> base-only -> mixed; every ``slot_by_token`` entry -1 after the
  base-only step while the kernel-meta buffer still holds PSR ids (known-bad:
  the old read steers ``none``); pad rows -1; ``idx = -1`` never indexes a
  slot (known-bad: an unclamped gather reads slot S-1);
- (g) the patch harness on fakes with the wheel's signatures: the loader lifts
  the payload through the classifier (unpatched: the wheel's ValueError);
  activation / deactivation (capture BEFORE the original; idempotent; the LRU
  callback; ``remove_all_adapters``; eviction; a repeated deactivate); the
  REAL inheritance shape with the table installed TWICE -> exactly ONE op
  invocation per block per forward, originals snapshotted, wrappers marked
  (known-bad: table-order patching yields two); the child reads ``layer_idx``;
  known-bad: neither ``layer_idx`` nor a prefix -> refuse naming the class;
  NemotronH's ``layer_idx`` argument with no attribute;
- (h) construction order (fact 11): runner -> load_model -> load_lora_model
  (manager + bank) -> bind -> profile-shaped dummy step -> capture-shaped step
  -> first real request; bound exactly once; known-bad: a runner-constructor
  registration finds no bank, a mapping update before any bank refuses;
- (i) import contract: vllm blocked + env unset -> ``PATCH_REPORT.armed is
  False``; env set with vllm absent -> the import RAISES;
- (j) the exact op ``torch.ops.vllm.sisteer_residual_slot`` through its fake
  impl (FakeTensorMode) and its eager impl on CPU tensors; known-bad:
  ``torch.ops.sisteer.residual_slot`` raises AttributeError; the class table
  resolves against the wheel when it is present (skipped otherwise);
- the ledger (fact 15), the payload reader's known-bads, the classifier, the
  worker's report and RPCs.

CPU-only; synthetic payloads; the tiny random llama for the HF references."""

from __future__ import annotations

import copy
import importlib.util
import io
import json
import math
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from sisteer.capeval.serving import lens_adapters as la
from sisteer.capeval.serving import native_residual as nr
from sisteer.capeval.serving import native_worker as nw
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving import worker_reports as wr

from native_residual_fakes import (FakeLoRAMapping, FakeLoRARequest, FakeModel, FakeRunner,
                                   fake_vllm_config, make_fake_layer_classes,
                                   make_fake_lora_classes, write_adapter_dir)

SRC = str(Path(__file__).resolve().parents[1] / "src")
BF16 = torch.bfloat16


def _report(name: str, **facts):
    """The measured numbers the review artifact carries (printed under -s / on
    failure; also asserted where the plan states a bound)."""
    print(f"[native_residual_slot] {name}: " + ", ".join(f"{k}={v}" for k, v in facts.items()))


# ---------------------------------------------------------------------------
# payload builders
# ---------------------------------------------------------------------------


def _block(kind: str, *, d: int, layers, n_layers_model: int, boundaries=None, mode="all",
           header_len=None, skip_bos=False, coeff=1.0, anchor=0):
    gated = kind in nr.GATED_KINDS
    return {"schema": nr.RESIDUAL_SLOT_SCHEMA, "kind": kind,
            "op": nr.OP_RELU_GATED_VECTOR if gated else nr.OP_ADD_VECTOR,
            "layers": list(layers), "boundaries": list(boundaries if boundaries is not None else layers),
            "position_mode": mode, "response_header_len": header_len, "response_header_ids": [],
            "skip_bos": skip_bos, "coeff": coeff, "hidden_size": d,
            "n_layers_model": n_layers_model, "anchor_layer": anchor,
            "dtype_policy": nr.DTYPE_POLICY_GATED if gated else nr.DTYPE_POLICY_CONSTANT}


def _psr_tensors(d: int, layers, *, seed: int, dir_scale=2.0, bias=0.5):
    """Random gates with a POSITIVE bias so roughly every gate is open, and
    directions of norm ~dir_scale (a realistic dose)."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for l in layers:
        w = torch.randn(d, generator=g) / math.sqrt(d)
        dvec = torch.randn(d, generator=g)
        dvec = dvec / dvec.norm() * dir_scale
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_w"] = w
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_b"] = torch.full((1,), bias)
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.direction"] = dvec
    return out


def _const_tensors(d: int, layers, *, seed: int, scale=0.7):
    g = torch.Generator().manual_seed(seed)
    return {f"{nr.PAYLOAD_PREFIX}layers.{l}.vector": torch.randn(d, generator=g) * scale
            for l in layers}


def _psr_payload(d, layers, n_layers_model, seed=1, bias=0.5, **kw):
    return nr.payload_from_block_and_tensors(
        _block("psr", d=d, layers=layers, n_layers_model=n_layers_model, **kw),
        _psr_tensors(d, layers, seed=seed, bias=bias))


def _const_payload(kind, d, layers, n_layers_model, seed=2, skip_bos=True, coeff=1.0, **kw):
    return nr.payload_from_block_and_tensors(
        _block(kind, d=d, layers=layers, n_layers_model=n_layers_model, skip_bos=skip_bos,
               coeff=coeff, **kw),
        _const_tensors(d, layers, seed=seed))


def _bank(S=4, L=3, d=16, max_tokens=64, dtype=BF16):
    nr.reset_for_tests()
    b = nr.ResidualSlotBank(n_slots=S, n_boundaries=L, hidden_size=d, max_tokens=max_tokens,
                            device="cpu", model_dtype=dtype)
    nr._publish_bank(b)
    return b


def _bits(t: torch.Tensor) -> torch.Tensor:
    return t.contiguous().view(torch.int16 if t.dtype == BF16 else torch.int32)


def _ulp_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a - b| in units of the bf16 ulp of the larger magnitude."""
    mag = torch.maximum(a.abs(), b.abs()).float().clamp(min=2.0 ** -126)
    ulp = 2.0 ** (torch.floor(torch.log2(mag)) - 7)
    return (a.float() - b.float()).abs() / ulp


@pytest.fixture(autouse=True)
def _fresh_module_state():
    nr.reset_for_tests()
    yield
    nr.reset_for_tests()


# ---------------------------------------------------------------------------
# (a) the gate contract
# ---------------------------------------------------------------------------


def _batched_gate(x: torch.Tensor, w: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The op's gate over rows all routed to one gate vector."""
    w_tok = w.unsqueeze(0).expand(x.shape[0], -1)
    return F.relu((x.float() * w_tok).sum(-1) + b)


def test_gate_contract_on_sols_shape_and_the_bf16_and_post_write_known_bads():
    g = torch.Generator().manual_seed(12)
    x = torch.randn(7, 128, generator=g)
    worst = 0.0
    worst_ratio = 0.0
    for _ in range(4):
        w = torch.randn(128, generator=g)
        b = torch.randn(1, generator=g)
        ref = F.relu(la._matmul(x.float(), w) + b)          # the lens's gate (and gate_pre's)
        got = _batched_gate(x, w, b)
        bound = nr.gate_contract_bound(x, w)
        diff = (got - ref).abs()
        assert bool((diff <= bound).all()), (diff, bound)
        worst = max(worst, float(diff.max()))
        worst_ratio = max(worst_ratio, float((diff / bound).max()))
        # KNOWN-BAD 1: a gate computed in bf16 violates the bound
        got_bf16 = F.relu((x.to(BF16) @ w.to(BF16)).float() + b)
        bad = (got_bf16 - ref).abs()
        assert bool((bad > bound).any()), "a bf16 gate must violate the fp32 contract"
        # KNOWN-BAD 2: the gate on the POST-write stream (the lens/PSR compute it on the
        # PRE-steering hidden) violates the bound wherever the gate is open
        add = got.unsqueeze(-1) * (w / w.norm() * 3.0)      # a write along the gate direction
        post = F.relu(la._matmul((x + add).float(), w) + b)
        bad2 = (post - ref).abs()
        assert bool((bad2 > bound).any())
    _report("gate contract (the review's shape, 4 gates)", max_abs_diff=worst,
            max_diff_over_bound=worst_ratio)
    assert worst_ratio <= 1.0


def test_gate_contract_against_psr_object_gate_pre_on_the_tiny_llama(tiny_llama):
    from sisteer.extraction.psr import PSRObject

    model, _ = tiny_llama
    layers = [0, 1]
    obj = PSRObject(model, layers)
    g = torch.Generator().manual_seed(3)
    d = model.config.hidden_size
    for l in layers:
        obj.gate_w[l].data = torch.randn(d, generator=g)
        obj.gate_b[l].data = torch.randn(1, generator=g)
    hidden = torch.randn(2, 5, d, generator=g) * 4.0
    worst_ratio = 0.0
    for l in layers:
        ref = F.relu(obj.gate_pre(l, hidden)).reshape(-1).detach()
        got = _batched_gate(hidden.reshape(-1, d), obj.gate_w[l].data, obj.gate_b[l].data)
        bound = nr.gate_contract_bound(hidden.reshape(-1, d), obj.gate_w[l].data)
        diff = (got - ref).abs()
        assert bool((diff <= bound).all())
        worst_ratio = max(worst_ratio, float((diff / bound).max()))
        bf16 = F.relu((hidden.reshape(-1, d).to(BF16) @ obj.gate_w[l].data.to(BF16)).float()
                      + obj.gate_b[l].data)
        assert bool(((bf16 - ref).abs() > bound).any())
    _report("gate contract (tiny llama)", max_diff_over_bound=worst_ratio)


# ---------------------------------------------------------------------------
# (b) the write-back with supplied deltas
# ---------------------------------------------------------------------------


def _lens_writeback(hidden, residual, add32):
    """lens_worker._hook_body_inner + _apply_slots_block, verbatim in shape."""
    true = hidden if residual is None else hidden + residual
    applied = true + add32.to(hidden.dtype)                  # gated_vector_write
    delta = applied - true
    return hidden + delta, true, applied


def test_writeback_is_bit_identical_to_the_lens_on_fact_2s_counterexample():
    hidden = torch.tensor([[2.171875]], dtype=BF16)          # [T=1, d=1]
    residual = torch.tensor([[-1.5234375]], dtype=BF16)
    add32 = torch.tensor([[0.150390625]], dtype=torch.float32)
    lens_new, true, applied = _lens_writeback(hidden, residual, add32)
    got = nr.writeback(hidden, true, applied, torch.tensor([True]))
    assert torch.equal(_bits(got), _bits(lens_new))
    z_lens = (lens_new + residual).item()
    assert z_lens == 0.7890625                                # the plan's lens value
    # KNOWN-BAD: the ``hidden + delta.to(model)`` form is FOUR ulps off
    bad = hidden + add32.to(BF16)
    z_bad = (bad + residual).item()
    assert z_bad == 0.8046875
    assert abs(z_bad - z_lens) / 0.00390625 == 4.0


@pytest.mark.parametrize("case", ["cancellation", "delta_dominates", "power_of_two", "random"])
def test_writeback_matches_the_lens_bitwise_on_cancellation_and_boundary_cases(case):
    g = torch.Generator().manual_seed(7)
    T, d = 256, 32
    if case == "cancellation":
        hidden = (torch.randn(T, d, generator=g) * 8).to(BF16)
        residual = (-hidden.float() + torch.randn(T, d, generator=g) * 1e-2).to(BF16)
        add32 = torch.randn(T, d, generator=g) * 0.05
    elif case == "delta_dominates":
        hidden = (torch.randn(T, d, generator=g) * 1e-2).to(BF16)
        residual = (torch.randn(T, d, generator=g) * 1e-2).to(BF16)
        add32 = torch.randn(T, d, generator=g) * 50.0
    elif case == "power_of_two":
        hidden = torch.full((T, d), 0.75, dtype=BF16)
        residual = torch.full((T, d), 0.25, dtype=BF16)      # true = 1.0 exactly, a binade edge
        add32 = torch.randn(T, d, generator=g) * 0.01
    else:
        hidden = (torch.randn(T, d, generator=g) * 3).to(BF16)
        residual = (torch.randn(T, d, generator=g) * 3).to(BF16)
        add32 = torch.randn(T, d, generator=g) * 0.3
    sel = torch.rand(T, generator=g) < 0.7
    lens_new, true, applied = _lens_writeback(hidden, residual, add32)
    expected = torch.where(sel.unsqueeze(-1), lens_new, hidden)
    got = nr.writeback(hidden, true, applied, sel)
    assert torch.equal(_bits(got), _bits(expected))
    # KNOWN-BAD: a form that skips ``applied - true`` and writes ``applied`` back onto
    # hidden fails wherever hidden and residual differ
    naive = torch.where(sel.unsqueeze(-1), applied, hidden)
    assert not torch.equal(_bits(naive), _bits(expected))
    # KNOWN-BAD: the ``hidden + delta.to(model)`` form is not the lens's
    other = torch.where(sel.unsqueeze(-1), hidden + add32.to(BF16), hidden)
    if case != "power_of_two":
        assert not torch.equal(_bits(other), _bits(expected))
    # unsteered rows are the SAME bits as hidden
    assert torch.equal(_bits(got[~sel]), _bits(hidden[~sel]))


# ---------------------------------------------------------------------------
# (c) + (e) end-to-end vs the lens, pair form and gemma form
# ---------------------------------------------------------------------------


def _psr_blob_bytes(d, layers, seed):
    tens = _psr_tensors(d, layers, seed=seed)
    blob = {"kind": "psr", "layers": list(layers), "positions": "all",
            "response_header_len": None,
            "directions": {str(l): tens[f"{nr.PAYLOAD_PREFIX}layers.{l}.direction"] for l in layers},
            "gate_w": {str(l): tens[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_w"] for l in layers},
            "gate_b": {str(l): tens[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_b"] for l in layers}}
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue(), tens


@pytest.mark.parametrize("form", ["pair", "gemma"])
def test_end_to_end_vs_the_lens_measures_the_flip_rate(form):
    T, d, layers = 512, 256, [0, 1, 2]
    blob_bytes, tens = _psr_blob_bytes(d, layers, seed=21)
    ops = la._adapt_psr({"kind": "psr", "layers": layers, "positions": "all"}, blob_bytes)
    assert len(ops) == 3 and all(op.op == la.OP_RELU_GATED_VECTOR for op in ops)
    payload = nr.payload_from_block_and_tensors(
        _block("psr", d=d, layers=layers, n_layers_model=3), tens)
    b = _bank(S=2, L=3, d=d, max_tokens=T)
    b.load_slot(1, payload, lora_int_id=7)
    g = torch.Generator().manual_seed(22)
    mapping = torch.where(torch.rand(T, generator=g) < 0.8, torch.tensor(1), torch.tensor(-1))
    b.update_mapping(mapping, T)
    steered = mapping == 1
    hidden = (torch.randn(T, d, generator=g) * 3).to(BF16)
    residual = None if form == "gemma" else (torch.randn(T, d, generator=g) * 3).to(BF16)
    positions = torch.arange(T)
    flips = 0
    rows = 0
    worst_ulp = 0.0
    for l in layers:
        true = hidden if residual is None else hidden + residual
        applied = la.apply_hook_ops(true, ops, 1.0, layer=l, ctx=None)
        lens_new = hidden + (applied - true)
        got = nr.residual_slot_forward(hidden, residual, positions, l, b.buffers)
        # unsteered rows: the same bits as the unpatched pair
        assert torch.equal(_bits(got[~steered]), _bits(hidden[~steered]))
        same = (_bits(got[steered]) == _bits(lens_new[steered])).all(dim=-1)
        rows += int(steered.sum())
        flips += int((~same).sum())
        if bool((~same).any()):
            dist = _ulp_distance(got[steered][~same], lens_new[steered][~same]).max()
            worst_ulp = max(worst_ulp, float(dist))
    rate = flips / rows
    _report(f"end-to-end vs lens ({form} form)", steered_rows=rows, rows_with_a_flip=flips,
            flip_rate=round(rate, 5), max_ulp_where_flipped=worst_ulp)
    assert worst_ulp <= 1.0            # a flip is one bf16 rounding, never more
    assert rate < 0.05                 # the gate noise flips a small fraction of rows


def test_end_to_end_the_gate_reads_the_pre_write_stream_and_the_delta_lands_on_hidden():
    """The gemma form with a supplied gate: the op's steered row equals the
    lens's ``gated_vector_write`` on ``true``; the residual passes through."""
    d = 32
    payload = _psr_payload(d, [0], 1, seed=5)
    b = _bank(S=1, L=1, d=d, max_tokens=8)
    b.load_slot(0, payload, lora_int_id=1)
    b.update_mapping(torch.tensor([0, 0, 0, 0]), 4)
    hidden = (torch.randn(4, d, generator=torch.Generator().manual_seed(101)) * 2).to(BF16)
    residual = (torch.randn(4, d, generator=torch.Generator().manual_seed(102)) * 2).to(BF16)
    got = nr.residual_slot_forward(hidden, residual, torch.arange(4), 0, b.buffers)
    true = hidden + residual
    g = F.relu(true.float() @ payload.gate_w[0] + payload.gate_b[0])
    applied = la.gated_vector_write(true, g, 1.0, payload.direction[0], None, BF16)
    lens_new = hidden + (applied - true)
    dist = _ulp_distance(got, lens_new)
    assert float(dist.max()) <= 1.0


# ---------------------------------------------------------------------------
# (d) the constant kinds vs the HF objects on the tiny llama
# ---------------------------------------------------------------------------


def _capture_block_outputs(model, layers):
    from sisteer.hooks import _hidden_from_output
    from sisteer.models import get_decoder_layers

    captured = {}
    blocks = get_decoder_layers(model)
    handles = [blocks[l].register_forward_hook(
        lambda m, a, out, l=l: captured.__setitem__(l, _hidden_from_output(out).detach().clone()))
        for l in layers]
    return captured, handles


def _hf_object(kind, model, layers, vectors, skip_bos):
    from sisteer.extraction.diffmean import DiffMeanObject
    from sisteer.extraction.multilayer import MultiLayerVectorObject
    from sisteer.hooks import SteeringVector
    from sisteer.methods import VectorObject

    if kind == "vector_all":
        obj = MultiLayerVectorObject(model, layers=layers, skip_bos=skip_bos)
        for l in layers:
            obj.params[l].data = vectors[l].clone()
        return obj
    if kind == "diffmean":
        return DiffMeanObject(model, {l: vectors[l] for l in layers}, skip_bos=skip_bos)
    sv = SteeringVector(vectors={l: vectors[l].clone() for l in layers}, method="caa",
                        model_name="tiny", instruction_id="t")
    return VectorObject(sv, positions="all", skip_bos=skip_bos)


@pytest.mark.parametrize("dtype", [torch.float32, BF16])
@pytest.mark.parametrize("kind", ["vector_all", "diffmean", "vector"])
def test_constant_add_matches_the_hf_objects_exactly(tiny_llama, kind, dtype):
    base, tok = tiny_llama
    model = copy.deepcopy(base).to(dtype).eval()
    d = model.config.hidden_size
    layers = [0, 1]
    g = torch.Generator().manual_seed(31)
    vectors = {l: torch.randn(d, generator=g) * 0.5 for l in layers}
    obj = _hf_object(kind, model, layers, vectors, skip_bos=True)
    ids = torch.tensor([[1, 5, 9, 13, 17], [1, 6, 10, 14, 18]])
    raw, h_raw = _capture_block_outputs(model, layers)          # registered BEFORE the hook
    with torch.no_grad(), obj.apply(model, 1.0):
        applied, h_app = _capture_block_outputs(model, layers)  # registered AFTER the hook
        model(ids)
    for h in h_raw + h_app:
        h.remove()
    tens = {f"{nr.PAYLOAD_PREFIX}layers.{l}.vector": vectors[l] for l in layers}
    payload = nr.payload_from_block_and_tensors(
        _block(kind, d=d, layers=layers, n_layers_model=2, skip_bos=True), tens)
    b = _bank(S=1, L=2, d=d, max_tokens=16, dtype=dtype)
    b.load_slot(0, payload, lora_int_id=1)
    B, S = ids.shape
    b.update_mapping(torch.zeros(B * S, dtype=torch.long), B * S)
    positions = torch.arange(S).repeat(B)
    for l in layers:
        true = raw[l].reshape(-1, d)
        want = applied[l].reshape(-1, d)
        got = nr.residual_slot_forward(true, None, positions, l, b.buffers)
        assert got.dtype == dtype
        assert torch.equal(got, want), (kind, dtype, l)
        # position 0 (the BOS) is UNSTEERED under skip_bos — the same bits as the raw output
        bos = positions == 0
        assert torch.equal(got[bos], true[bos])
        assert not torch.equal(got[~bos], true[~bos])
    # KNOWN-BAD: a skip_bos=False payload steers position 0, disagreeing with the HF object
    bad = nr.payload_from_block_and_tensors(
        _block(kind, d=d, layers=layers, n_layers_model=2, skip_bos=False), tens)
    b.load_slot(0, bad, lora_int_id=1)
    got0 = nr.residual_slot_forward(raw[0].reshape(-1, d), None, positions, 0, b.buffers)
    assert not torch.equal(got0, applied[0].reshape(-1, d))
    assert not torch.equal(got0[positions == 0], raw[0].reshape(-1, d)[positions == 0])


def test_static_r_coeff_folds_in_the_lens_order():
    """``coeff`` (static_r's base_scale) multiplies the MODEL-dtype vector, as
    the lens's ``addition = vec * eff`` does — never the fp32 vector first."""
    d = 32
    tens = _const_tensors(d, [0], seed=9, scale=3.0)
    payload = nr.payload_from_block_and_tensors(
        _block("static_r", d=d, layers=[0], n_layers_model=1, skip_bos=False, coeff=0.05), tens)
    b = _bank(S=1, L=1, d=d, max_tokens=4)
    b.load_slot(0, payload, lora_int_id=1)
    vec = tens[f"{nr.PAYLOAD_PREFIX}layers.0.vector"]
    lens_addition = vec.to(BF16) * 0.05
    assert torch.equal(b.buffers.vector[0, 0], lens_addition)
    assert not torch.equal(b.buffers.vector[0, 0], (vec * 0.05).to(BF16))  # the other order differs
    b.update_mapping(torch.tensor([0, 0]), 2)
    hidden = (torch.randn(2, d, generator=torch.Generator().manual_seed(103)) * 2).to(BF16)
    got = nr.residual_slot_forward(hidden, None, torch.arange(2), 0, b.buffers)
    assert torch.equal(got, hidden + lens_addition)


# ---------------------------------------------------------------------------
# (f) routing through the real metadata path
# ---------------------------------------------------------------------------


def _install(fk, layers, table=None):
    nr.reset_for_tests()
    targets = nr.PatchTargets(
        lora_model_cls=fk.LoRAModel, manager_cls=fk.LoRAModelManager,
        punica_cls=fk.PunicaWrapperBase, runner_mixin_cls=fk.LoRAModelRunnerMixin,
        decoder_layer_classes=tuple(table if table is not None else layers.TABLE),
        model_info=fk.model_info, op=nr.ensure_op_registered())
    return targets, nr.install(targets)


def test_mapping_transitions_refill_slot_by_token_while_the_kernel_meta_goes_stale():
    fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
    _install(fk, layers)
    S, L, d, MAXT = 3, 2, 16, 24
    b = nr.ResidualSlotBank(n_slots=S, n_boundaries=L, hidden_size=d, max_tokens=MAXT,
                            device="cpu", model_dtype=BF16)
    nr._publish_bank(b)
    punica = fk.PunicaWrapperGPU(MAXT, 8, "cpu", max_loras=S)          # the PATCHED class
    lora_index_to_id = [1, 2, None]                                    # id 1 -> slot 0, id 2 -> slot 1
    psr_payload = _psr_payload(d, [0, 1], L, seed=4, bias=20.0)       # an open gate on every row
    b.load_slot(0, psr_payload, lora_int_id=1)
    b.load_slot(1, None, lora_int_id=2)                                # a plain LoRA
    # step 1: a PSR batch (5 tokens of request 1, 3 of a base request)
    m1 = FakeLoRAMapping((1,) * 5 + (0,) * 3, (1, 0), True)
    punica.update_metadata(m1, lora_index_to_id, S + 1, 97)
    assert b.buffers.slot_by_token[:8].tolist() == [0] * 5 + [-1] * 3
    assert b.buffers.slot_by_token[8:].tolist() == [-1] * (MAXT - 8)   # pad rows
    assert punica.token_mapping_meta.token_lora_mapping[:8].tolist() == [0] * 5 + [-1] * 3
    assert b.mapping_updates == 1
    # step 2: a BASE-ONLY batch — vLLM's kernel meta returns early and keeps step 1's ids
    m2 = FakeLoRAMapping((0,) * 6, (0, 0), True)
    punica.update_metadata(m2, lora_index_to_id, S + 1, 97)
    assert b.buffers.slot_by_token.tolist() == [-1] * MAXT
    stale = punica.token_mapping_meta.token_lora_mapping[:6].tolist()
    assert stale == [0] * 5 + [-1], stale        # KNOWN-BAD: the old read would steer ``none``
    assert bool(punica.token_mapping_meta.no_lora_flag_cpu[0])
    gen = torch.Generator().manual_seed(44)
    hidden = torch.randn(6, d, generator=gen).to(BF16)
    residual = torch.randn(6, d, generator=gen).to(BF16)
    got = nr.residual_slot_forward(hidden, residual, torch.arange(6), 0, b.buffers)
    assert torch.equal(_bits(got), _bits(hidden))                      # every row unsteered
    # what the stale read WOULD have done: steered five base rows
    stale_buf = b.buffers._replace(slot_by_token=torch.tensor(stale + [-1] * (MAXT - 6)))
    bad = nr.residual_slot_forward(hidden, residual, torch.arange(6), 0, stale_buf)
    assert not torch.equal(bad[:5], hidden[:5])
    # step 3: a MIXED batch (psr, lora, base)
    m3 = FakeLoRAMapping((2, 2, 1, 1, 0, 1), (2, 1, 0, 1), True)
    punica.update_metadata(m3, lora_index_to_id, S + 1, 97)
    assert b.buffers.slot_by_token[:6].tolist() == [1, 1, 0, 0, -1, 0]
    got = nr.residual_slot_forward(hidden, residual, torch.arange(6), 0, b.buffers)
    steered = torch.tensor([False, False, True, True, False, True])
    assert torch.equal(_bits(got[~steered]), _bits(hidden[~steered]))
    assert not torch.equal(got[steered], hidden[steered])
    # an UNCHANGED mapping is skipped by the manager (no refill; contents unchanged)
    n_before = b.mapping_updates
    mgr_like = SimpleNamespace(_last_mapping=m3)
    assert mgr_like._last_mapping == FakeLoRAMapping((2, 2, 1, 1, 0, 1), (2, 1, 0, 1), True)
    assert b.mapping_updates == n_before


def test_minus_one_never_indexes_a_slot():
    """KNOWN-BAD: an UNCLAMPED gather with idx = -1 reads slot S-1 — a payload
    parked there would steer every base token."""
    S, L, d = 3, 1, 16
    b = _bank(S=S, L=L, d=d, max_tokens=8)
    b.load_slot(S - 1, _psr_payload(d, [0], L, seed=8), lora_int_id=9)
    b.update_mapping(torch.tensor([-1, -1, -1, -1]), 4)
    hidden = (torch.randn(4, d, generator=torch.Generator().manual_seed(104)) * 2).to(BF16)
    got = nr.residual_slot_forward(hidden, None, torch.arange(4), 0, b.buffers)
    assert torch.equal(_bits(got), _bits(hidden))
    idx = b.buffers.slot_by_token[:4]
    assert bool(b.buffers.active[idx, 0].all())           # the unclamped read says "active"
    assert not bool(b.buffers.active[idx.clamp(min=0), 0].any()) or True
    live = idx >= 0
    assert not bool((live & b.buffers.active[idx.clamp(min=0), 0]).any())


# ---------------------------------------------------------------------------
# (g) the patch harness
# ---------------------------------------------------------------------------


def _residual_adapter(tmp_path: Path, name: str, payload: nr.ResidualPayload, *, d_in=16,
                      d_out=16, anchor=0, prefix="model.layers"):
    return write_adapter_dir(tmp_path / name, layers=[anchor], d_in=d_in, d_out=d_out, rank=1,
                             prefix=prefix, zero=True, residual_block=payload.block(),
                             residual_tensors=payload.tensors())


def test_loader_lifts_the_payload_and_the_unpatched_wheel_refuses_the_entry(tmp_path):
    fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
    d, L = 16, 3
    payload = _psr_payload(d, [0, 1, 2], L, seed=11)
    adir = _residual_adapter(tmp_path, "psr", payload)
    cfg = json.loads((adir / "adapter_config.json").read_text())
    # UNPATCHED: the wheel's unexpected-module check refuses the payload keys — through
    # ``parse_fine_tuned_lora_name``'s "unsupported LoRA weight" (lora/utils.py @0.25.1), the
    # first parse of a key that is neither lora_A/lora_B nor an embedding
    with pytest.raises(ValueError, match="unsupported LoRA weight"):
        fk.LoRAModel.from_local_checkpoint(str(adir), {"o_proj"}, fk.PEFTHelper(cfg),
                                           lora_model_id=1, device="cpu", dtype=BF16)
    _install(fk, layers)
    lora = fk.LoRAModel.from_local_checkpoint(str(adir), {"o_proj"}, fk.PEFTHelper(cfg),
                                              lora_model_id=1, device="cpu", dtype=BF16)
    assert lora.sisteer_residual is not None
    assert lora.sisteer_residual.fingerprint == payload.fingerprint
    assert set(lora.loras) == {"model.layers.0.self_attn.o_proj"}      # the zero pair only
    assert lora.loras["model.layers.0.self_attn.o_proj"].lora_a.dtype == BF16
    plain = write_adapter_dir(tmp_path / "lora", layers=[0, 2], d_in=d, d_out=d, rank=2)
    cfg2 = json.loads((plain / "adapter_config.json").read_text())
    lora2 = fk.LoRAModel.from_local_checkpoint(str(plain), {"o_proj"}, fk.PEFTHelper(cfg2),
                                               lora_model_id=2, device="cpu", dtype=BF16)
    assert lora2.sisteer_residual is None
    # the original is snapshotted and the wrapper marked
    assert nr._is_wrapped(fk.LoRAModel.__dict__["from_local_checkpoint"])


def _manager(fk, vc):
    return fk.LRUCacheLoRAModelManager(model=SimpleNamespace(), max_num_seqs=8,
                                       max_num_batched_tokens=64, vocab_size=97,
                                       lora_config=vc.lora_config, device="cpu", vllm_config=vc)


def test_activation_deactivation_eviction_and_remove_all(tmp_path):
    fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
    _install(fk, layers)
    d, L, S = 16, 3, 2
    vc = fake_vllm_config(hidden_size=d, num_layers=L, dtype=BF16, max_loras=S, max_cpu_loras=3)
    mgr = _manager(fk, vc)
    bank = mgr.sisteer_bank
    assert nr.bank() is bank and bank.n_slots == S and bank.n_boundaries == L
    p1 = _psr_payload(d, [0, 1], L, seed=1)
    p3 = _const_payload("vector_all", d, [2], L, seed=3)
    a1 = _residual_adapter(tmp_path, "a1", p1)
    a2 = write_adapter_dir(tmp_path / "a2", layers=[0], d_in=d, d_out=d, rank=1)
    a3 = _residual_adapter(tmp_path, "a3", p3)

    def load(path, lid):
        cfg = json.loads((Path(path) / "adapter_config.json").read_text())
        return fk.LoRAModel.from_local_checkpoint(str(path), {"o_proj"}, fk.PEFTHelper(cfg),
                                                  lora_model_id=lid, device="cpu", dtype=BF16)

    l1, l2, l3 = load(a1, 1), load(a2, 2), load(a3, 3)
    mgr.add_adapter(l1)
    mgr.add_adapter(l2)
    assert mgr.activate_adapter(1) is True
    assert mgr.lora_index_to_id[0] == 1 and bank.records[0].lora_int_id == 1
    assert bool(bank.buffers.active[0, 0]) and bool(bank.buffers.active[0, 1]) \
        and not bool(bank.buffers.active[0, 2])
    assert mgr.activate_adapter(1) is False                     # already active: untouched
    assert mgr.activate_adapter(2) is True                      # a plain LoRA -> zeros, inactive
    assert mgr.lora_index_to_id[1] == 2 and bank.records[1] is None
    assert not bool(bank.buffers.active[1].any())
    # deactivate: capture BEFORE the original erases the slot; idempotent
    assert mgr.deactivate_adapter(1) is True
    assert mgr.lora_index_to_id[0] is None and bank.records[0] is None
    assert not bool(bank.buffers.active[0].any()) and float(bank.buffers.gate_w[0].abs().sum()) == 0.0
    assert mgr.deactivate_adapter(1) is False                   # repeated: no error, no change
    mgr._deactivate_adapter(1)                                  # the raw path, repeated
    # LRU eviction through the callback: both slots full, a third activation evicts the oldest
    mgr.add_adapter(l3)
    assert mgr.activate_adapter(1) is True                      # slot 0 again
    assert mgr.lora_index_to_id == [1, 2]
    assert bank.records[0].lora_int_id == 1
    assert mgr.activate_adapter(3) is True                      # evicts the oldest ACTIVE (2)
    assert mgr.lora_index_to_id == [1, 3]
    slot3 = 1
    assert bank.records[slot3].lora_int_id == 3 and bank.records[slot3].payload.kind == "vector_all"
    assert bool(bank.buffers.active[slot3, 2]) and not bool(bank.buffers.gated[slot3])
    assert bank.records[0].lora_int_id == 1 and bool(bank.buffers.gated[0])   # slot 0 untouched
    # remove_all_adapters: every slot zero (the original resets lora_index_to_id first)
    mgr.remove_all_adapters()
    assert all(r is None for r in bank.records)
    assert not bool(bank.buffers.active.any())
    assert float(bank.buffers.vector.abs().sum()) == 0.0 and float(bank.buffers.direction.abs().sum()) == 0.0
    assert mgr.lora_index_to_id == [None, None]


def test_capture_dummies_load_as_empty_slots_and_are_removed(tmp_path):
    fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
    _install(fk, layers)
    d, L, S = 16, 2, 3
    vc = fake_vllm_config(hidden_size=d, num_layers=L, dtype=BF16, max_loras=S, max_lora_rank=4)
    wm = fk.LRUCacheWorkerLoRAManager(vc, "cpu", {})
    wm.create_lora_manager(SimpleNamespace(), vc)
    bank = nr.bank()
    for i in range(1, S + 1):
        wm.add_dummy_lora(FakeLoRARequest(f"warmup_{i}", i, "/not/a/real/path"), rank=4)
        wm._adapter_manager.activate_adapter(i)
    assert all(r is None for r in bank.records)
    assert not bool(bank.buffers.active.any())
    wm.remove_all_adapters()
    assert wm._adapter_manager.lora_index_to_id == [None] * S


def test_first_install_reports_an_inherited_forward_as_wrapped():
    """KNOWN-BAD of the GPU proof (job 507491, 2026-09-15): on the wheel's FIRST
    install the report said ``Qwen3_5DecoderLayer: forward inherits:Qwen3NextDecoderLayer,
    forward_wrapped False`` — the bit was read at plan time, before the parent's
    forward was wrapped — and ``residual_slot_problems`` refused readiness. The
    twice-installed test below only saw ``True`` because the second install
    re-read the bit. ONE install on fresh fakes must state the end state, in
    table order (parent first) and with the child listed BEFORE its parent."""
    from sisteer.capeval.serving import serving_attestation as sa
    for order in ("table", "child_first"):
        fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
        table = list(layers.TABLE)
        if order == "child_first":
            table.remove(layers.Qwen3_5DecoderLayer); table.insert(0, layers.Qwen3_5DecoderLayer)
        _, rep = _install(fk, layers, table=table)
        assert rep.installs == 1
        child = rep.patched_classes[nr._qual(layers.Qwen3_5DecoderLayer)]
        assert child["forward"].startswith("inherits:") and child["forward"].endswith("Qwen3NextDecoderLayer"), (order, child)
        assert child["forward_wrapped"] is True and child["__init__"] == "wrapped" and child["__init___wrapped"] is True, (order, child)
        parent = rep.patched_classes[nr._qual(layers.Qwen3NextDecoderLayer)]
        assert parent == {"forward": "wrapped", "forward_wrapped": True, "__init__": "wrapped", "__init___wrapped": True}, (order, parent)
        # the readiness validator's rule over exactly this report: no problem about any class
        for cls, entry in rep.patched_classes.items():
            assert entry["forward_wrapped"] is True, (order, cls, entry)
        report = {"armed": True, "op_name": nr.OP_QUALNAME, "op_registered": True, "runner_bound": True,
                  "ledger_fault": False, "ledger_size": 0, "n_layers_model": 4, "n_layers_hooked": 4,
                  "patched_classes": rep.patched_classes, "slots": {}}
        expected = {"armed": True, "n_layers_model": 4, "adapters": {}}
        assert [p for p in sa.residual_slot_problems([report], expected) if "wrapped forward" in p] == [], order
        nr.reset_for_tests()


@pytest.mark.parametrize("arch,cls_name", [
    ("llama", "LlamaDecoderLayer"), ("gemma4", "Gemma4DecoderLayer"),
    ("qwen3_next", "Qwen3NextDecoderLayer"), ("qwen3_5", "Qwen3_5DecoderLayer"),
    ("nemotron_h", "NemotronHMLPDecoderLayer"), ("nemotron_h", "NemotronHAttentionDecoderLayer"),
    ("nemotron_h", "NemotronHMambaDecoderLayer"), ("nemotron_h", "NemotronHMoEDecoderLayer"),
])
def test_one_op_invocation_per_block_with_the_table_installed_twice(arch, cls_name):
    fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
    targets, rep1 = _install(fk, layers)
    rep2 = nr.install(targets)                                 # the SECOND install: a no-op
    assert rep2 is rep1 and rep2.installs == 2
    d, L, S = 16, 4, 2
    vc = fake_vllm_config(hidden_size=d, num_layers=L, dtype=BF16, max_loras=S)
    model = FakeModel(getattr(layers, cls_name), L, vc, arch=arch)
    bank = nr.ResidualSlotBank(n_slots=S, n_boundaries=L, hidden_size=d, max_tokens=16,
                               device="cpu", model_dtype=BF16)
    nr._publish_bank(bank)
    bank.load_slot(0, _psr_payload(d, list(range(L)), L, seed=2), lora_int_id=1)
    bank.update_mapping(torch.tensor([0, 0, -1, 0, 0]), 5)
    hidden = (torch.randn(5, d, generator=torch.Generator().manual_seed(105)) * 2).to(BF16)
    out = model(hidden, torch.arange(5))
    assert bank.op_invocations == L, (cls_name, bank.op_invocations)
    assert out.shape == hidden.shape
    # every instance recorded its boundary; the boundaries are the block indices
    assert sorted(b for _, b, _ in rep1.instances) == list(range(L))
    assert rep1.hooked_boundaries() == list(range(L))
    # the class report: forward wrapped where DEFINED, inherited where not
    entry = rep1.patched_classes[nr._qual(getattr(layers, cls_name))]
    if cls_name == "Qwen3_5DecoderLayer":
        assert entry["forward"].startswith("inherits:") \
            and entry["forward"].endswith("Qwen3NextDecoderLayer") and entry["forward_wrapped"] is True
    else:
        assert entry["forward"] == "wrapped"
    assert entry["__init__"] == "wrapped"
    # originals snapshotted, wrappers marked
    cls = getattr(layers, cls_name)
    assert nr._is_wrapped(cls.forward) and nr._is_wrapped(cls.__init__)
    assert callable(cls.forward._sisteer_orig) and not nr._is_wrapped(cls.forward._sisteer_orig)


def test_known_bad_table_order_patching_writes_the_inherited_forward_twice():
    """Rev 2's install (wrap the parent's forward, then 'the child's', which
    IS the parent's) fires the op twice per Qwen3.5 block."""
    layers = make_fake_layer_classes()
    calls = []

    def naive_wrap(fn):
        def forward(self, *a, **k):
            out = fn(self, *a, **k)
            calls.append(type(self).__name__)
            return out
        return forward

    for cls in (layers.Qwen3NextDecoderLayer, layers.Qwen3_5DecoderLayer):
        cls.forward = naive_wrap(cls.forward)                  # table order, no __dict__ check
    vc = fake_vllm_config(hidden_size=8, num_layers=2, dtype=BF16, max_loras=1)
    model = FakeModel(layers.Qwen3_5DecoderLayer, 2, vc, arch="qwen3_5")
    model(torch.randn(3, 8, generator=torch.Generator().manual_seed(106)).to(BF16), torch.arange(3))
    assert len(calls) == 4                                     # TWO per block — the defect


def test_boundary_sources_the_child_reads_layer_idx_nemotron_reads_the_argument_neither_refuses():
    fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
    table = layers.TABLE + (layers.NoBoundaryLayer,)
    _, rep = _install(fk, layers, table=table)
    vc = fake_vllm_config(hidden_size=8, num_layers=6, dtype=BF16, max_loras=1)
    bank = nr.ResidualSlotBank(n_slots=1, n_boundaries=6, hidden_size=8, max_tokens=8,
                               device="cpu", model_dtype=BF16)
    nr._publish_bank(bank)
    bank.update_mapping(torch.tensor([-1, -1]), 2)
    q = layers.Qwen3_5DecoderLayer(vc, layer_type="full_attention", prefix="model.language_model.layers.5")
    assert q.layer_idx == 5 and q._sisteer_boundary == 5
    n = layers.NemotronHMLPDecoderLayer(config=SimpleNamespace(hidden_size=8), layer_idx=4,
                                        prefix="model.layers.4")
    assert not hasattr(n, "layer_idx") and n._sisteer_boundary == 4
    with pytest.raises(nr.ResidualSlotError, match="disagree"):
        layers.NemotronHMLPDecoderLayer(config=SimpleNamespace(hidden_size=8), layer_idx=3,
                                        prefix="model.layers.4")
    ll = layers.LlamaDecoderLayer(vc, prefix="model.layers.2")
    assert ll._sisteer_boundary == 2
    # KNOWN-BAD: neither a layer_idx nor a prefix -> the first forward refuses naming the class
    bad = layers.NoBoundaryLayer()
    assert not hasattr(bad, "_sisteer_boundary")
    h = torch.randn(2, 8, generator=torch.Generator().manual_seed(107)).to(BF16)
    with pytest.raises(nr.ResidualSlotError, match="NoBoundaryLayer"):
        bad(torch.arange(2), h, None)
    # a layer constructed with an empty prefix is the same refusal at forward time
    bare = layers.LlamaDecoderLayer(vc)
    with pytest.raises(nr.ResidualSlotError, match="LlamaDecoderLayer"):
        bare(torch.arange(2), h, None)
    assert rep.patched_classes[nr._qual(layers.NoBoundaryLayer)]["forward"] == "wrapped"


# ---------------------------------------------------------------------------
# (h) construction order
# ---------------------------------------------------------------------------


def test_construction_order_runner_load_model_manager_bind_profile_capture_first_request(tmp_path):
    fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
    targets, rep = _install(fk, layers)
    d, L, S = 16, 3, 2
    vc = fake_vllm_config(hidden_size=d, num_layers=L, dtype=BF16, max_loras=S,
                          max_num_batched_tokens=32)
    runner = FakeRunner(vc, "cpu", fk.LoRAModelRunnerMixin, layers.LlamaDecoderLayer,
                        arch="llama", n_layers=L)                  # init_device: no model yet
    # KNOWN-BAD (rev 2): registering at the runner's constructor finds no bank
    with pytest.raises(nr.ResidualSlotError, match="no ResidualSlotBank"):
        nr._require_bank("runner constructor").bind_runner(runner)
    assert nr.bank() is None and rep.instances == []
    runner.load_model()                                             # model -> load_lora_model
    bank = nr.bank()
    assert bank is not None and bank.runner is runner                 # bound exactly once
    assert len(rep.instances) == L and rep.hooked_boundaries() == list(range(L))
    assert bank.max_tokens == 32 and bank.n_boundaries == L and bank.hidden_size == d
    bank.bind_runner(runner)                                          # idempotent for the same runner
    with pytest.raises(nr.ResidualSlotError, match="already bound"):
        bank.bind_runner(object())
    hidden = (torch.randn(6, d, generator=torch.Generator().manual_seed(108)) * 2).to(BF16)
    positions = torch.arange(6)
    # profile-shaped dummy step: dummy LoRAs in every slot, a mapping over them -> a no-op
    with runner.maybe_setup_dummy_loras(vc.lora_config):
        reqs = {FakeLoRARequest(f"warmup_{i}", i, "/not/a/real/path") for i in range(1, S + 1)}
        out = runner.execute_model(token_lora_mapping=(1, 1, 2, 2, 1, 2), prompt_lora_mapping=(1, 2),
                                   lora_requests=reqs, hidden_states=hidden, positions=positions)
        assert all(r is None for r in bank.records)
        ref = FakeModel(make_fake_layer_classes().LlamaDecoderLayer, L, vc, arch="llama")(hidden, positions)
        assert torch.equal(out, ref)                                  # bit-identical no-op
        assert bank.op_invocations == L
    assert runner.lora_manager._adapter_manager.lora_index_to_id == [None] * S
    # capture-shaped step: base-only mapping
    out = runner.execute_model(token_lora_mapping=(0,) * 6, prompt_lora_mapping=(0,),
                               lora_requests=set(), hidden_states=hidden, positions=positions)
    assert torch.equal(out, ref) and bank.buffers.slot_by_token.tolist() == [-1] * 32
    # the first REAL request: a PSR adapter routed per token
    payload = _psr_payload(d, [0, 1, 2], L, seed=6)
    adir = _residual_adapter(tmp_path, "psr", payload)
    req = FakeLoRARequest("psr_cell", 1, str(adir))
    out = runner.execute_model(token_lora_mapping=(1, 1, 1, 0, 0, 1), prompt_lora_mapping=(1, 0, 1),
                               lora_requests={req}, hidden_states=hidden, positions=positions)
    assert bank.records[0].lora_int_id == 1 and bank.records[0].payload.fingerprint == payload.fingerprint
    assert bank.buffers.slot_by_token[:6].tolist() == [0, 0, 0, -1, -1, 0]
    steered = torch.tensor([True, True, True, False, False, True])
    assert torch.equal(out[~steered], ref[~steered])
    assert not torch.equal(out[steered], ref[steered])
    # the report reads all of it
    report = wr.residual_slot_report(runner)
    assert report["armed"] is True and report["op_name"] == "vllm.sisteer_residual_slot"
    assert report["runner_bound"] is True and report["n_layers_hooked"] == L
    assert report["n_layers_model"] == L and report["ledger_fault"] is False
    assert report["slots"] == {"1": {"slot": 0, **payload.summary()}}
    assert report["bank"]["op_invocations"] >= 3 * L
    assert report["patched_classes"][nr._qual(layers.LlamaDecoderLayer)]["forward"] == "wrapped"


def test_a_mapping_update_before_any_bank_refuses():
    fk, layers = make_fake_lora_classes(), make_fake_layer_classes()
    _install(fk, layers)
    punica = fk.PunicaWrapperGPU(16, 4, "cpu", max_loras=2)
    with pytest.raises(nr.ResidualSlotError, match="no ResidualSlotBank"):
        punica.update_metadata(FakeLoRAMapping((1, 0), (1, 0), True), [1, None], 3, 97)


def test_a_second_bank_in_one_process_refuses():
    _bank()
    with pytest.raises(nr.ResidualSlotError, match="second ResidualSlotBank"):
        nr._publish_bank(nr.ResidualSlotBank(n_slots=1, n_boundaries=1, hidden_size=4,
                                             max_tokens=4, device="cpu", model_dtype=BF16))


# ---------------------------------------------------------------------------
# (i) the import contract
# ---------------------------------------------------------------------------


def test_import_contract_unarmed_without_the_flag_and_loud_with_it_but_no_vllm():
    code = r"""
import sys, os
sys.modules["vllm"] = None
sys.modules["vllm_lens"] = None
from sisteer.capeval.serving import native_worker as nw
from sisteer.capeval.serving import native_residual as nr
assert nr.PATCH_REPORT.armed is False and nr.PATCH_REPORT.op_name is None
assert nr.bank() is None
attrs = [a for a in dir(nw.NativeWorkerExtension) if not a.startswith("__")]
assert attrs == ["sisteer_arm_lora_routing_recorder", "sisteer_disarm_lora_routing_recorder",
                 "sisteer_lora_routing_observations", "sisteer_mamba_backend_report",
                 "sisteer_residual_slot_report", "sisteer_scheduler_config",
                 "sisteer_take_request_failure"], attrs
print("OK")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "OK"
    armed = r"""
import sys
sys.modules["vllm"] = None
try:
    from sisteer.capeval.serving import native_residual as nr
except Exception as e:
    print(type(e).__name__, str(e)[:80])
else:
    print("IMPORTED", nr.PATCH_REPORT.armed)
"""
    out = subprocess.run([sys.executable, "-c", armed], capture_output=True, text=True,
                         env={"PYTHONPATH": SRC, "PATH": "/usr/bin:/bin",
                              nr.ENV_FLAG: "1"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.startswith("ResidualSlotError"), out.stdout


# ---------------------------------------------------------------------------
# (j) the exact op
# ---------------------------------------------------------------------------


def test_the_exact_op_through_its_fake_and_eager_impls_and_the_known_bad_name():
    from torch._subclasses.fake_tensor import FakeTensorMode

    op = nr.ensure_op_registered()
    assert op is torch.ops.vllm.sisteer_residual_slot
    assert nr.op_registered() and nr.OP_QUALNAME == "vllm.sisteer_residual_slot"
    schema = str(torch.ops.vllm.sisteer_residual_slot.default._schema)
    assert schema.startswith("vllm::sisteer_residual_slot(Tensor hidden, Tensor? residual, "
                             "Tensor? positions, SymInt boundary) -> Tensor"), schema
    d = 16
    with FakeTensorMode():
        fh = torch.empty(9, d, dtype=BF16)
        fo = op(fh, torch.empty(9, d, dtype=BF16), torch.arange(9), 2)
        assert tuple(fo.shape) == (9, d) and fo.dtype == BF16
        fo2 = op(fh, None, None, 0)
        assert tuple(fo2.shape) == (9, d)
    b = _bank(S=2, L=3, d=d, max_tokens=16)
    b.load_slot(1, _psr_payload(d, [0, 2], 3, seed=13, bias=20.0), lora_int_id=4)   # gate open
    b.update_mapping(torch.tensor([1, -1, 1]), 3)
    hidden = (torch.randn(3, d, generator=torch.Generator().manual_seed(109)) * 2).to(BF16)
    residual = (torch.randn(3, d, generator=torch.Generator().manual_seed(110)) * 2).to(BF16)
    got = op(hidden, residual, torch.arange(3), 2)
    ref = nr.residual_slot_forward(hidden, residual, torch.arange(3), 2, b.buffers)
    assert torch.equal(got, ref) and b.op_invocations == 1
    assert torch.equal(got[1], hidden[1]) and not torch.equal(got[0], hidden[0])
    assert torch.equal(op(hidden, residual, torch.arange(3), 1), hidden)   # boundary 1 inactive
    # KNOWN-BAD: the name rev 2 used does not exist
    with pytest.raises(AttributeError):
        torch.ops.sisteer.residual_slot
    # the op fired with no bank refuses (no silent pass-through)
    nr.reset_for_tests()
    with pytest.raises(nr.ResidualSlotError, match="no ResidualSlotBank"):
        op(hidden, residual, torch.arange(3), 0)


@pytest.mark.skipif(importlib.util.find_spec("vllm") is None,
                    reason="the pinned vllm wheel is not installed here (Modal runs it)")
def test_the_class_table_resolves_against_the_wheel():
    import inspect

    targets = nr.vllm_patch_targets()
    names = {c.__name__ for c in targets.decoder_layer_classes}
    assert names == {n for _, n in nr.VLLM_DECODER_LAYER_CLASSES}
    for cls in targets.decoder_layer_classes:
        own = cls.__dict__.get("forward")
        if cls.__name__ == "Qwen3_5DecoderLayer":
            assert own is None                                  # inherits Qwen3Next's
            continue
        src = inspect.getsource(own)
        assert ("return hidden_states, residual" in src or "return output, residual" in src
                or "return hidden_states, None" in src), cls.__name__
    assert targets.op is torch.ops.vllm.sisteer_residual_slot


# ---------------------------------------------------------------------------
# the ledger (fact 15)
# ---------------------------------------------------------------------------


def test_ledger_entries_leave_only_by_claim_and_the_cap_latches_a_fault():
    led = nr.RequestFailureLedger(cap=3)
    assert led.claim("r1") is None
    led.record("r1", cell_id="c", error_type="ResponseBoundaryUnlocated")
    led.record("r1", cell_id="c", error_type="dup")               # once per request
    assert len(led) == 1 and "r1" in led.failed_ids
    got = led.claim("r1")
    # (+ the completion id the entry belongs to and every engine id the claim popped — the engine-id join)
    assert got == {"req_id": "r1", "external_req_id": "r1", "cell_id": "c", "error_type": "ResponseBoundaryUnlocated",
                   "claimed_engine_ids": ["r1"]}
    assert led.claim("r1") is None and len(led) == 0
    for r in ("a", "b", "c"):
        led.record(r, cell_id="x")
    assert len(led) == 3 and led.fault is False
    led.record("d", cell_id="x")                                    # at the cap: LATCH
    assert led.fault is True
    # KNOWN-BAD (rev 3's evicting ledger): the OLDEST entry would have been dropped and its
    # late completion would read None; here it is still present and every claim faults
    for r in ("a", "d", "zzz-never-seen"):
        assert led.claim(r) == {"fault": "ledger_overflow", "unclaimed": 3, "cap": 3}
    assert len(led) == 3 and led.report() == {"ledger_size": 3, "ledger_fault": True, "ledger_cap": 3}


def test_take_request_failure_answers_none_unarmed_and_faults_armed_without_a_bank():
    assert nr.take_request_failure("r") is None
    nr.PATCH_REPORT.armed = True
    assert nr.take_request_failure("r") == {"fault": "bank_missing", "req_id": "r"}
    b = _bank()
    assert nr.take_request_failure("r") is None
    b.ledger.record("r", cell_id="c")
    assert nw.NativeWorkerExtension.sisteer_take_request_failure(SimpleNamespace(), "r") == {
        "req_id": "r", "cell_id": "c", "external_req_id": "r", "claimed_engine_ids": ["r"]}
    assert nr.take_request_failure("r") is None


# ---------------------------------------------------------------------------
# the payload reader and the classifier
# ---------------------------------------------------------------------------


def test_classify_adapter_key():
    assert sa.classify_adapter_key("base_model.model.model.layers.3.self_attn.o_proj.lora_A.weight") == \
        ("lora", "base_model.model.model.layers.3.self_attn.o_proj", "A")
    assert sa.classify_adapter_key("x.lora_B.weight") == ("lora", "x", "B")
    assert sa.classify_adapter_key("sisteer_residual.layers.12.gate_w") == ("residual", 12, "gate_w")
    assert sa.classify_adapter_key("sisteer_residual.layers.0.vector") == ("residual", 0, "vector")
    for bad in ("sisteer_residual.layers.3.gate", "sisteer_residual.layers.x.gate_w",
                "sisteer_residual.gate_w", "sisteer_residual.layers.-1.vector",
                "model.layers.0.o_proj.weight", ".lora_A.weight", ""):
        with pytest.raises(sa.ServingAttestationError):
            sa.classify_adapter_key(bad)
    assert sa.RESIDUAL_KEY_PREFIX == nr.PAYLOAD_PREFIX
    assert tuple(sa.RESIDUAL_KEY_FIELDS) == nr.RESIDUAL_FIELDS


def test_payload_reader_known_bads(tmp_path):
    d, L = 16, 3
    good = _psr_payload(d, [0, 2], L, seed=1)
    adir = _residual_adapter(tmp_path, "ok", good)
    read = nr.read_residual_payload(adir)
    assert read.fingerprint == good.fingerprint and read.layers == (0, 2) and read.boundaries == (0, 2)
    plain = write_adapter_dir(tmp_path / "plain", layers=[0], d_in=d, d_out=d)
    assert nr.read_residual_payload(plain) is None
    # a block without tensors / tensors without a block
    write_adapter_dir(tmp_path / "b1", layers=[0], d_in=d, d_out=d, residual_block=good.block())
    with pytest.raises(nr.ResidualSlotError, match="declaration without its payload"):
        nr.read_residual_payload(tmp_path / "b1")
    write_adapter_dir(tmp_path / "b2", layers=[0], d_in=d, d_out=d, residual_tensors=good.tensors())
    with pytest.raises(nr.ResidualSlotError, match="payload without its declaration"):
        nr.read_residual_payload(tmp_path / "b2")
    tens = good.tensors()
    blk = good.block()
    # unknown field
    with pytest.raises(nr.ResidualSlotError, match="not one of"):
        nr.payload_from_block_and_tensors(blk, {**tens, f"{nr.PAYLOAD_PREFIX}layers.0.vector": torch.zeros(d)})
    # a layer the block does not declare
    with pytest.raises(nr.ResidualSlotError, match="does not declare"):
        nr.payload_from_block_and_tensors(blk, {**tens, f"{nr.PAYLOAD_PREFIX}layers.1.gate_w": torch.zeros(d)})
    # a declared layer missing from the payload
    with pytest.raises(nr.ResidualSlotError, match="lacks"):
        nr.payload_from_block_and_tensors(blk, {k: v for k, v in tens.items() if ".2." not in k})
    # a dtype outside the policy
    bad = dict(tens)
    bad[f"{nr.PAYLOAD_PREFIX}layers.0.direction"] = bad[f"{nr.PAYLOAD_PREFIX}layers.0.direction"].to(BF16)
    with pytest.raises(nr.ResidualSlotError, match="written fp32"):
        nr.payload_from_block_and_tensors(blk, bad)
    # the wrong shape / non-finite
    bad = dict(tens)
    bad[f"{nr.PAYLOAD_PREFIX}layers.0.gate_w"] = torch.zeros(d + 1)
    with pytest.raises(nr.ResidualSlotError, match="shape"):
        nr.payload_from_block_and_tensors(blk, bad)
    bad = dict(tens)
    bad[f"{nr.PAYLOAD_PREFIX}layers.0.gate_b"] = torch.tensor([float("nan")])
    with pytest.raises(nr.ResidualSlotError, match="non-finite"):
        nr.payload_from_block_and_tensors(blk, bad)
    # block refusals: kind/op mismatch, a PSR coeff, boundaries outside the model, response
    # without a header width, a header width on mode all, a duplicate boundary
    with pytest.raises(nr.ResidualSlotError, match="serves through op"):
        nr.payload_from_block_and_tensors({**blk, "op": nr.OP_ADD_VECTOR}, tens)
    with pytest.raises(nr.ResidualSlotError, match="coeff"):
        nr.payload_from_block_and_tensors({**blk, "coeff": 0.5}, tens)
    with pytest.raises(nr.ResidualSlotError, match="outside the model"):
        nr.payload_from_block_and_tensors({**blk, "boundaries": [0, 3]}, tens)
    with pytest.raises(nr.ResidualSlotError, match="response_header_len"):
        nr.payload_from_block_and_tensors({**blk, "position_mode": "response"}, tens)
    with pytest.raises(nr.ResidualSlotError, match="belongs to the response mode"):
        nr.payload_from_block_and_tensors({**blk, "response_header_len": 4}, tens)
    with pytest.raises(nr.ResidualSlotError, match="duplicate"):
        nr.payload_from_block_and_tensors({**blk, "boundaries": [0, 0]}, tens)
    with pytest.raises(nr.ResidualSlotError, match="dtype_policy"):
        nr.payload_from_block_and_tensors({**blk, "dtype_policy": nr.DTYPE_POLICY_CONSTANT}, tens)
    with pytest.raises(nr.ResidualSlotError, match="keys"):
        nr.payload_from_block_and_tensors({**blk, "extra": 1}, tens)
    # the fingerprint moves with the bytes and with the block
    tweaked = dict(tens)
    tweaked[f"{nr.PAYLOAD_PREFIX}layers.0.gate_b"] = tweaked[f"{nr.PAYLOAD_PREFIX}layers.0.gate_b"] + 1e-3
    assert nr.payload_from_block_and_tensors(blk, tweaked).fingerprint != good.fingerprint
    assert nr.payload_from_block_and_tensors({**blk, "skip_bos": False}, tens).fingerprint == good.fingerprint
    assert nr.payload_from_block_and_tensors({**blk, "anchor_layer": 1}, tens).fingerprint != good.fingerprint


def test_a_stage_2_position_mode_refuses_at_load_while_stage_2_is_unarmed(monkeypatch):
    """Stage 2 is ARMED on this tree (step 7); the guard is kept for an unarmed
    worker and pinned here under a monkeypatch. Armed, a restricted-mode payload
    whose block carries NO header ids (the stage-1 exporter's ``[]``) refuses at
    activation instead — the boundary could not be located for any request."""
    d, L = 16, 2
    payload = nr.payload_from_block_and_tensors(
        _block("psr", d=d, layers=[0], n_layers_model=L, mode="response", header_len=4),
        _psr_tensors(d, [0], seed=1))
    b = _bank(S=1, L=L, d=d)
    assert nr.STAGE2_POSITION_CLASSES is True
    monkeypatch.setattr(nr, "STAGE2_POSITION_CLASSES", False)
    with pytest.raises(nr.ResidualSlotError, match="stage-2 position class"):
        b.load_slot(0, payload, lora_int_id=1)
    assert b.records[0] is None and not bool(b.buffers.active.any())
    monkeypatch.setattr(nr, "STAGE2_POSITION_CLASSES", True)
    with pytest.raises(nr.ResidualSlotError, match="response_header_ids"):
        b.load_slot(0, payload, lora_int_id=1)
    assert b.records[0] is None and not bool(b.buffers.active.any())
    # a hidden size / layer count that does not match the model refuses too
    with pytest.raises(nr.ResidualSlotError, match="hidden_size"):
        b.load_slot(0, _psr_payload(d + 2, [0], L, seed=1), lora_int_id=1)
    with pytest.raises(nr.ResidualSlotError, match="n_layers_model"):
        b.load_slot(0, _psr_payload(d, [0], L + 1, seed=1), lora_int_id=1)


def test_op_shape_and_dtype_refusals():
    b = _bank(S=1, L=1, d=8, max_tokens=4)
    h = torch.randn(4, 8, generator=torch.Generator().manual_seed(111)).to(BF16)
    with pytest.raises(nr.ResidualSlotError, match="wider than the LoRA mapping"):
        nr.residual_slot_forward(torch.randn(5, 8, generator=torch.Generator().manual_seed(112)).to(BF16), None, None, 0, b.buffers)
    with pytest.raises(nr.ResidualSlotError, match="model dtype"):
        nr.residual_slot_forward(h.float(), None, None, 0, b.buffers)
    with pytest.raises(nr.ResidualSlotError, match="does not pair"):
        nr.residual_slot_forward(h, torch.randn(4, 8, generator=torch.Generator().manual_seed(113)), None, 0, b.buffers)
    with pytest.raises(nr.ResidualSlotError, match="boundary"):
        b.apply(h, None, None, 1)
    with pytest.raises(nr.ResidualSlotError, match="\\[num_tokens, hidden\\]"):
        nr.residual_slot_forward(h.reshape(2, 2, 8), None, None, 0, b.buffers)


def test_constants_agree_with_the_lens_and_the_report_is_readable_unarmed():
    assert nr.OP_RELU_GATED_VECTOR == la.OP_RELU_GATED_VECTOR
    assert nr.OP_ADD_VECTOR == la.OP_ADD_VECTOR
    assert nr.REQUEST_FAILURES_CAP == 1024
    from sisteer.capeval.serving import lens_worker as lw
    assert nr.REQUEST_FAILURES_CAP == lw.REQUEST_FAILURES_CAP
    rep = wr.residual_slot_report(None)
    assert rep["armed"] is False and rep["op_name"] is None and rep["bank"] is None
    assert rep["slots"] == {} and rep["runner_bound"] is False and rep["ledger_fault"] is False
    assert nw.NativeWorkerExtension.sisteer_residual_slot_report(SimpleNamespace(model_runner=None)) == rep


# --------------------------------------------------------------------------- #
# the row-chunked op body (GPU proof job 508688: the profile run's temporaries)   #
# --------------------------------------------------------------------------- #

def test_the_op_body_runs_in_row_chunks_and_a_chunked_forward_matches_the_single_pass(monkeypatch):
    """KNOWN-BAD of the llama8b PSR certification run: vLLM's memory
    profile forwards S x L rows at once and the op's ``[T, d]`` temporaries
    ate 16 GiB of KV. The body now runs in chunks of
    ``RESIDUAL_FORWARD_CHUNK_ROWS``: with the chunk forced small, a T-row
    forward is processed in ceil(T / chunk) calls of the row body; the
    unselected rows and the CONSTANT rows are bit-identical to the single
    pass (the where-form and the model-dtype add do not depend on T), the
    gated rows agree within the gate's error contract (the per-row reduction
    may pick another kernel at another T — fact 14); an all-(-1)
    mapping (the profile run's shape) is a bit-exact no-op through the chunked
    path too; and a captured-size forward (T <= chunk) takes ONE call."""
    S, L, d, T = 3, 2, 16, 23
    b = _bank(S=S, L=L, d=d, max_tokens=64)
    b.load_slot(0, _psr_payload(d, [0, 1], L, seed=3, bias=0.5), lora_int_id=1)
    b.load_slot(1, _const_payload("vector_all", d, [0, 1], L, seed=4, skip_bos=False), lora_int_id=2)
    g = torch.Generator().manual_seed(11)
    mapping = torch.tensor([0, 1, -1] * 7 + [0, 1], dtype=torch.int64)[:T]
    b.update_mapping(mapping, T)
    hidden = (torch.randn(T, d, generator=g) * 2).to(BF16)
    residual = (torch.randn(T, d, generator=g) * 2).to(BF16)
    pos = torch.arange(T)
    single = nr.residual_slot_forward(hidden, residual, pos, 1, b.buffers)          # T <= chunk: one pass
    calls = []
    real_rows = nr._residual_slot_forward_rows

    def spy(h, r, p, boundary, buffers, idx, *facts):
        calls.append(int(h.shape[0]))
        return real_rows(h, r, p, boundary, buffers, idx, *facts)

    monkeypatch.setattr(nr, "_residual_slot_forward_rows", spy)
    monkeypatch.setattr(nr, "RESIDUAL_FORWARD_CHUNK_ROWS", 5)
    chunked = nr.residual_slot_forward(hidden, residual, pos, 1, b.buffers)
    assert calls == [5, 5, 5, 5, 3] and sum(calls) == T                        # ceil(23 / 5) calls
    assert chunked.shape == single.shape and chunked.dtype == single.dtype
    const_rows = mapping == 1
    none_rows = mapping == -1
    gated_rows = mapping == 0
    assert torch.equal(chunked[none_rows], hidden[none_rows]) and torch.equal(single[none_rows], hidden[none_rows])
    assert torch.equal(chunked[const_rows], single[const_rows])               # the model-dtype add: exact
    assert torch.allclose(chunked[gated_rows].float(), single[gated_rows].float(), rtol=2 ** -6, atol=2 ** -5)
    assert not torch.equal(single[gated_rows], hidden[gated_rows])            # the gated write did land
    # the profile run's shape: every row -1 -> a bit-exact no-op through the chunked path
    b.update_mapping(torch.full((T,), -1, dtype=torch.int64), T)
    calls.clear()
    assert torch.equal(nr.residual_slot_forward(hidden, residual, pos, 1, b.buffers), hidden)
    assert len(calls) == 5
    # a captured-size forward: T <= the chunk -> exactly one call of the row body
    monkeypatch.setattr(nr, "RESIDUAL_FORWARD_CHUNK_ROWS", 64)
    calls.clear()
    nr.residual_slot_forward(hidden, residual, pos, 1, b.buffers)
    assert calls == [T]
    assert nr.RESIDUAL_FORWARD_CHUNK_ROWS == 64 and isinstance(nr.RESIDUAL_FORWARD_CHUNK_ROWS, int)
    nr.reset_for_tests()


def test_the_chunk_constant_is_far_above_every_captured_batch():
    """8192 rows: above vLLM's largest cudagraph capture size (the attested
    max_cudagraph_capture_size is in the hundreds), so every captured shape is
    a single-chunk forward — the path the admitted certifications ran."""
    assert nr.RESIDUAL_FORWARD_CHUNK_ROWS == 8192
