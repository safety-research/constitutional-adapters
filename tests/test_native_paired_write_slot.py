"""The PAIRED-WRITE native route (relu-differenced-native-qwen; the native residual slot's
SECOND GATE TERM): ``relu(w1.z + b1) d1 - relu(w2.z + b2) d2`` on the SAME pre-write residual
state, one read, one write, served as op ``relu_gated_vector_pair`` of the existing slot for
the kind ``psr_paired_write`` (``extraction/paired_write.PSRPairedWriteObject``).

CPU-only. Written BEFORE the kernel existed (every test here failed on the single-gate tree):

- (a) IDENTITY: both parents hydrated through ``PSRPairedWriteObject`` on the tiny llama; the
  slot's write vs ``paired_write(layer, hidden)`` on the same random fp32 state, within the
  gate contract per TERM (``|g - g_ref| <= 4 d eps32 ||x|| ||w||`` each side, so 2x the
  single-gate bound in total) plus the write's own fp32 roundings; the measured worst ratio
  is reported;
- (b) KNOWN-BAD: the two sides SWAPPED must FAIL the identity tolerance;
- (c) NEG-ZEROED: a paired payload whose second side is all zeros writes BIT-IDENTICALLY to
  the single-gate write of its first side (bf16 and fp32; the llama and the gemma4 forms);
- (d) COMPOSITION == FITTED: the serve-time composition of two single-gate parent blobs
  (``lora_export.compose_paired_write_blob``) and a FITTED ``psr_paired_write`` artifact
  (``PSRPairedWriteObject.compose_blob`` + provenance, the builder's own shape) export to
  BYTE-IDENTICAL payloads and equal fingerprints — synthetic, and on the REAL llama8b teacher
  and Qwen3.5-4B batch-64 seed-1 objects when a read-only scratch copy is present
  (``SISTEER_TEST_OBJECTS`` or ``~/.cache/sisteer-test-objects``; skipped otherwise, never fetched);
- (e) the FOUR registry moves (``RESIDUAL_SLOT_KINDS`` x2, ``NATIVE_LORA_SERVABLE_KINDS``,
  ``CLASS_A_METHODS`` + ``OBJECT_ONLY_METHODS``, ``ARM_OBJECT_KINDS``), pinned equal across the
  three modules, a paired cell NOT refused by ``native_lora_unservable_reason`` and judged
  ``all`` by ``native_position_mode``; the harness id; the derived arm's refusing ArmSpec;
- (f) the FINGERPRINT covers both sides and both parent shas: distinct from either parent's,
  from the swapped payload's, and from the same tensors under other parent shas;
- (g) an UNPAIRED standup is UNCHANGED: ``SlotBuffers``' first eleven fields, the bank's
  allocation (no second-gate tensors unless armed), and the kernel's output bit-identical to
  the single-gate body as it stood before this change (inlined here verbatim) — on an unarmed
  bank AND on an armed bank with no paired slot resident;
- the validator's refusals (kind/op mismatch, a paired op with one side, a single op carrying
  a second side, coefficients other than (+1, -1), identical parents, a single block wearing
  the paired keys), the classifier, the safetensors round trip, the exporter's refusals
  (differing layer sets / widths between the sides, a masked position leg, a foreign
  application order, a fitted artifact without both parent shas), and the certification
  plumbing (the TARGETS row, the cross-bank pin file, the roster view, the generated config).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from sisteer.capeval import manifest as cm
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.serving import lora_export as lx
from sisteer.capeval.serving import native_residual as nr
from sisteer.capeval.serving import serving_attestation as sa

REPO = Path(__file__).resolve().parents[1]
BF16 = torch.bfloat16
EPS32 = float(torch.finfo(torch.float32).eps)
PAIRED = "psr_paired_write"
ARM = "psr_ce_all_minus_ctl"
CELL = f"constv2-mix-full__{ARM}"
HCELL = f"si_constv2_mix_full__{ARM}"
#: the read-only scratch copies of the real objects (pulled from the cluster by hand; never in
#: the repo, never fetched by a test): <root>/<tag>/object.pt
OBJECTS_ROOT = Path(os.environ.get("SISTEER_TEST_OBJECTS", Path.home() / ".cache" / "sisteer-test-objects"))
REAL_SETS = {
    # tag: (fitted paired dir, pos parent dir, neg parent dir, hidden size, expected shas)
    "llama8b": ("llama8b_teacher", "llama8b_pos", "llama8b_neg", 4096,
                ("3ee2adb94723ff9b6a0caada32d79d0946d449070462f7fdba07ef285eea81d5",
                 "4df62c74415737e2e910d4b3c7a55d169012b6cff538934d4a053ec7318db799",
                 "c30ec630a5e32eff916eda6142108e179e0374d307c8f9373641ed600f546da5")),
    "qwen35_4b_s1": ("qwen35_4b_s1_paired", "qwen35_4b_s1_pos", "qwen35_4b_s1_neg", 2560,
                     ("f98f0349673b7cb0cf85f54aee71f74bf9f0bbf981e989b52c8ba6599a492c2d",
                      "7b87767d91bd9787d22a23a80a5e78c53d16b506cc25e4d9483f74e88238fda1",
                      "f65425b851ab446b329fa93631462dd4f31fb16e11955867ef902dffd635c2cd")),
}


def _report(name: str, **facts):
    print(f"[native_paired_write] {name}: " + ", ".join(f"{k}={v}" for k, v in facts.items()))


@pytest.fixture(autouse=True)
def _fresh_module_state():
    nr.reset_for_tests()
    yield
    nr.reset_for_tests()


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _side(d: int, layers, *, seed: int, dir_scale: float = 2.0, bias: float = 0.5,
          zero: bool = False) -> dict:
    """One PSR side in ``PSRObject.save``'s per-layer shape (str keys, fp32): gates with a
    POSITIVE bias so they are open on most rows, directions of norm ``dir_scale``."""
    g = torch.Generator().manual_seed(seed)
    out = {"gate_w": {}, "gate_b": {}, "directions": {}}
    for l in layers:
        if zero:
            w, b, v = torch.zeros(d), torch.zeros(1), torch.zeros(d)
        else:
            w = torch.randn(d, generator=g) / (d ** 0.5)
            v = torch.randn(d, generator=g)
            v = v / v.norm() * dir_scale
            b = torch.full((1,), bias)
        out["gate_w"][str(l)] = w
        out["gate_b"][str(l)] = b
        out["directions"][str(l)] = v
    return out


def _psr_blob(side: dict, layers) -> dict:
    return {"kind": "psr", "layers": list(layers), "train_direction": False, "positions": "all",
            "response_header_len": None, **{k: dict(v) for k, v in side.items()}}


def _bytes(blob) -> bytes:
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fitted_blob(pos: dict, neg: dict, layers, pos_sha: str, neg_sha: str) -> dict:
    """The FITTED artifact exactly as scripts/sdfbase/psr_residual_distill.py writes it: the
    object's own ``compose_blob`` with the builder's provenance block (both parents' shas)."""
    from sisteer.extraction.paired_write import APPLICATION_ORDER, PSRPairedWriteObject

    prov = {"kind": PAIRED, "derived_id": "constv2-mix-full__psr_ce_all_minus_ctl",
            "application_order": APPLICATION_ORDER, "positions": "all",
            "parent_pos": {"cell_id": "constv2-mix-full__psr_ce_all", "object_sha256": pos_sha},
            "parent_neg": {"cell_id": "constv2ctl2-full__psr_ce_all", "object_sha256": neg_sha},
            "builder": "tests (synthetic)"}
    return PSRPairedWriteObject.compose_blob(_psr_blob(pos, layers), _psr_blob(neg, layers), provenance=prov)


def _single_tensors(side: dict, layers) -> dict:
    out = {}
    for l in layers:
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_w"] = side["gate_w"][str(l)]
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_b"] = side["gate_b"][str(l)].reshape(1)
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.direction"] = side["directions"][str(l)]
    return out


def _paired_tensors(pos: dict, neg: dict, layers) -> dict:
    out = _single_tensors(pos, layers)
    for l in layers:
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_w2"] = neg["gate_w"][str(l)]
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.gate_b2"] = neg["gate_b"][str(l)].reshape(1)
        out[f"{nr.PAYLOAD_PREFIX}layers.{l}.direction2"] = neg["directions"][str(l)]
    return out


def _single_block(d: int, layers, L: int, *, mode="all", anchor=0) -> dict:
    return {"schema": nr.RESIDUAL_SLOT_SCHEMA, "kind": "psr", "op": nr.OP_RELU_GATED_VECTOR,
            "layers": list(layers), "boundaries": list(layers), "position_mode": mode,
            "response_header_len": None, "response_header_ids": [], "skip_bos": False, "coeff": 1.0,
            "hidden_size": d, "n_layers_model": L, "anchor_layer": anchor,
            "dtype_policy": nr.DTYPE_POLICY_GATED}


POS_SHA = "a" * 64
NEG_SHA = "b" * 64


def _paired_block(d: int, layers, L: int, *, pos_sha=POS_SHA, neg_sha=NEG_SHA, mode="all", anchor=0,
                  pos_coeff=1.0, neg_coeff=-1.0) -> dict:
    return {**_single_block(d, layers, L, mode=mode, anchor=anchor), "kind": PAIRED,
            "op": nr.OP_RELU_GATED_VECTOR_PAIR,
            "parent_pos_sha256": pos_sha, "parent_neg_sha256": neg_sha,
            "parent_pos_coeff": pos_coeff, "parent_neg_coeff": neg_coeff}


def _single_payload(side, d, layers, L):
    return nr.payload_from_block_and_tensors(_single_block(d, layers, L), _single_tensors(side, layers))


def _paired_payload(pos, neg, d, layers, L, **kw):
    return nr.payload_from_block_and_tensors(_paired_block(d, layers, L, **kw), _paired_tensors(pos, neg, layers))


def _bank(*, S=3, L=3, d=16, max_tokens=64, dtype=BF16, paired=False):
    b = nr.ResidualSlotBank(n_slots=S, n_boundaries=L, hidden_size=d, max_tokens=max_tokens,
                            device="cpu", model_dtype=dtype, paired=paired)
    nr._publish_bank(b)
    return b


def _route(bank, slot: int, T: int) -> None:
    bank.buffers.slot_by_token.fill_(-1)
    bank.buffers.slot_by_token[:T] = slot


def _legacy_rows(hidden, residual, pos, boundary, buffers, idx, prompt_len_tok, response_start_tok):
    """THE SINGLE-GATE BODY AS IT STOOD BEFORE THE SECOND TERM (native_residual.
    _residual_slot_forward_rows @ 78e0fd90), inlined verbatim: the byte-identity reference for (g)."""
    live = idx >= 0
    idx_c = idx.clamp(min=0)
    sel = live & buffers.active[idx_c, boundary]
    if pos is not None:
        sel = sel & ~(buffers.skip_bos[idx_c] & (pos == 0))
    sel = sel & nr.position_class_selection(buffers.mode[idx_c], pos, prompt_len_tok, response_start_tok)
    true = hidden if residual is None else hidden + residual
    w_tok = buffers.gate_w[idx_c, boundary]
    b_tok = buffers.gate_b[idx_c, boundary]
    d_tok = buffers.direction[idx_c, boundary]
    v_tok = buffers.vector[idx_c, boundary]
    g = F.relu((true.float() * w_tok).sum(-1) + b_tok)
    add32 = (g * sel.to(torch.float32)).unsqueeze(-1) * d_tok
    applied_gated = true + add32.to(hidden.dtype)
    applied_const = true + v_tok
    applied = torch.where(buffers.gated[idx_c].unsqueeze(-1), applied_gated, applied_const)
    return nr.writeback(hidden, true, applied, sel)


def _legacy_forward(bank, hidden, residual, positions, boundary):
    T = hidden.shape[0]
    b = bank.buffers
    return _legacy_rows(hidden, residual, positions, boundary, b, b.slot_by_token[:T],
                        b.prompt_len_by_token[:T], b.response_start_by_token[:T])


# ---------------------------------------------------------------------------
# (a) / (b): identity against PSRPairedWriteObject on the tiny llama, and the swapped known-bad
# ---------------------------------------------------------------------------


def _hydrate_pair(model, layers, pos: dict, neg: dict):
    from sisteer.extraction.paired_write import PSRPairedWriteObject
    from sisteer.extraction.psr import PSRObject

    objs = []
    for side in (pos, neg):
        o = PSRObject(model, list(layers))
        for l in layers:
            o.gate_w[l].data = side["gate_w"][str(l)].clone()
            o.gate_b[l].data = side["gate_b"][str(l)].clone()
            o.directions[l].data = side["directions"][str(l)].clone()
        objs.append(o)
    return PSRPairedWriteObject(objs[0], objs[1])


def _identity_tolerance(x2d, pos, neg, layer, out, hidden):
    """Per element: the gate contract of EACH term times |d_term| (2x the single-gate bound's
    reach), plus the write's own fp32 roundings (applied, delta, new — 4 ulps of the larger)."""
    b1 = nr.gate_contract_bound(x2d, pos["gate_w"][str(layer)]).unsqueeze(-1) * pos["directions"][str(layer)].abs()
    b2 = nr.gate_contract_bound(x2d, neg["gate_w"][str(layer)]).unsqueeze(-1) * neg["directions"][str(layer)].abs()
    return b1 + b2 + 4.0 * EPS32 * torch.maximum(out.abs(), hidden.abs())


def test_identity_against_the_paired_write_object_on_the_tiny_llama_and_the_swapped_known_bad(tiny_llama):
    model, _ = tiny_llama
    d = model.config.hidden_size
    L = model.config.num_hidden_layers
    layers = list(range(L))
    pos = _side(d, layers, seed=11, dir_scale=2.0, bias=0.6)
    neg = _side(d, layers, seed=12, dir_scale=1.5, bias=0.4)
    obj = _hydrate_pair(model, layers, pos, neg)
    assert obj.application_order == nr.PAIRED_APPLICATION_ORDER
    g = torch.Generator().manual_seed(5)
    B, S = 2, 7
    hidden = torch.randn(B, S, d, generator=g) * 4.0
    x2d = hidden.reshape(-1, d)
    T = x2d.shape[0]
    bank = _bank(S=2, L=L, d=d, max_tokens=T, dtype=torch.float32, paired=True)
    bank.load_slot(0, _paired_payload(pos, neg, d, layers, L), lora_int_id=1)
    bank.load_slot(1, _paired_payload(neg, pos, d, layers, L, pos_sha=NEG_SHA, neg_sha=POS_SHA), lora_int_id=2)  # SWAPPED
    positions = torch.arange(S).repeat(B)
    worst = 0.0
    n_open_both = 0
    for l in layers:
        v_ref = obj.paired_write(l, hidden).detach().reshape(-1, d)   # fp32, the object's own combined write
        ref_new = nr.writeback(x2d, x2d, x2d + v_ref, torch.ones(T, dtype=torch.bool))
        _route(bank, 0, T)
        got = bank.apply(x2d, None, positions, l)
        tol = _identity_tolerance(x2d, pos, neg, l, got, x2d)
        diff = (got - ref_new).abs()
        assert bool((diff <= tol).all()), (l, float(diff.max()), float(tol.min()))
        worst = max(worst, float((diff / tol.clamp(min=1e-30)).max()))
        n_open_both += int(((obj._last_gate[0][l] > 0) & (obj._last_gate[1][l] > 0)).sum())
        # KNOWN-BAD (b): the sides swapped is a DIFFERENT function of the same state
        _route(bank, 1, T)
        bad = bank.apply(x2d, None, positions, l)
        assert bool(((bad - ref_new).abs() > tol).any()), l
    assert n_open_both > 0, "the fixture must exercise rows where BOTH gates are open"
    _report("identity (tiny llama, every layer)", worst_diff_over_tolerance=worst, rows_both_gates_open=n_open_both)
    assert worst <= 1.0


def test_identity_holds_in_bf16_up_to_the_stated_flip_class(tiny_llama):
    """The bf16 model dtype: the paired add is cast ONCE, so the write equals the object's
    ``hidden + add.to(bf16)`` except where the two fp32 gates' contract noise flips a bf16
    rounding — the single-gate route's stated class (fact 14); the rate is measured and reported."""
    model, _ = tiny_llama
    d, L = model.config.hidden_size, model.config.num_hidden_layers
    layers = list(range(L))
    pos, neg = _side(d, layers, seed=21), _side(d, layers, seed=22, dir_scale=1.2)
    obj = _hydrate_pair(model, layers, pos, neg)
    g = torch.Generator().manual_seed(9)
    hidden = (torch.randn(3, 5, d, generator=g) * 4.0).to(BF16)
    x2d = hidden.reshape(-1, d)
    T = x2d.shape[0]
    bank = _bank(S=1, L=L, d=d, max_tokens=T, dtype=BF16, paired=True)
    bank.load_slot(0, _paired_payload(pos, neg, d, layers, L), lora_int_id=1)
    flips = total = 0
    for l in layers:
        v_ref = obj.paired_write(l, hidden).detach().reshape(-1, d)
        ref_new = nr.writeback(x2d, x2d, x2d + v_ref.to(BF16), torch.ones(T, dtype=torch.bool))
        _route(bank, 0, T)
        got = bank.apply(x2d, None, None, l)
        ne = got != ref_new
        flips += int(ne.sum())
        total += ne.numel()
        if ne.any():   # where they differ, by at most one bf16 ulp
            mag = torch.maximum(got.abs(), ref_new.abs()).float().clamp(min=2.0 ** -126)
            ulp = 2.0 ** (torch.floor(torch.log2(mag)) - 7)
            assert float(((got.float() - ref_new.float()).abs() / ulp)[ne].max()) <= 1.0
    _report("bf16 flip rate (tiny llama)", flips=flips, elements=total, rate=flips / max(total, 1))


# ---------------------------------------------------------------------------
# (c) the NEG-zeroed paired payload equals the single-gate write bit for bit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [BF16, torch.float32])
@pytest.mark.parametrize("form", ["llama", "gemma4"])
def test_neg_zeroed_paired_payload_is_bit_identical_to_the_single_gate_write(dtype, form):
    d, L, T = 16, 3, 24
    layers = [0, 2]
    pos = _side(d, layers, seed=31, bias=0.7)
    zero = _side(d, layers, seed=0, zero=True)
    g = torch.Generator().manual_seed(2)
    hidden = (torch.randn(T, d, generator=g) * 3.0).to(dtype)
    residual = None if form == "gemma4" else (torch.randn(T, d, generator=g) * 3.0).to(dtype)
    positions = torch.arange(T)
    armed = _bank(S=2, L=L, d=d, max_tokens=T, dtype=dtype, paired=True)
    armed.load_slot(0, _paired_payload(pos, zero, d, layers, L), lora_int_id=1)   # pos - 0
    armed.load_slot(1, _single_payload(pos, d, layers, L), lora_int_id=2)         # pos alone, same bank
    nr.reset_for_tests()
    plain = _bank(S=1, L=L, d=d, max_tokens=T, dtype=dtype, paired=False)          # pos alone, UNARMED bank
    plain.load_slot(0, _single_payload(pos, d, layers, L), lora_int_id=3)
    for l in range(L):
        _route(armed, 0, T)
        paired_out = armed.apply(hidden, residual, positions, l)
        _route(armed, 1, T)
        single_same_bank = armed.apply(hidden, residual, positions, l)
        _route(plain, 0, T)
        single_plain = plain.apply(hidden, residual, positions, l)
        assert torch.equal(paired_out, single_same_bank), l
        assert torch.equal(paired_out, single_plain), l
        if l in layers:
            assert not torch.equal(paired_out, hidden)    # the fixture steers (an inert pair proves nothing)


# ---------------------------------------------------------------------------
# (d) composition == fitted, byte for byte
# ---------------------------------------------------------------------------


def _spec(kind=PAIRED, **over):
    return {"kind": kind, "positions": "all", **over}


def test_composed_payload_equals_the_fitted_objects_payload_byte_for_byte_synthetic():
    d, L = 16, 3
    layers = [0, 1, 2]
    pos, neg = _side(d, layers, seed=41), _side(d, layers, seed=42)
    pos_b, neg_b = _bytes(_psr_blob(pos, layers)), _bytes(_psr_blob(neg, layers))
    fitted = _fitted_blob(pos, neg, layers, _sha(pos_b), _sha(neg_b))
    composed = lx.compose_paired_write_blob(torch.load(io.BytesIO(pos_b), weights_only=True),
                                            torch.load(io.BytesIO(neg_b), weights_only=True),
                                            pos_sha256=_sha(pos_b), neg_sha256=_sha(neg_b))
    assert composed["kind"] == PAIRED and composed["application_order"] == nr.PAIRED_APPLICATION_ORDER
    a = lx._residual_psr_paired_write(_spec(), fitted, hidden_size=d)
    b = lx._residual_psr_paired_write(_spec(), composed, hidden_size=d)
    assert a["tensors"].keys() == b["tensors"].keys()
    for k in a["tensors"]:
        assert torch.equal(a["tensors"][k], b["tensors"][k]) and a["tensors"][k].dtype == torch.float32, k
    assert {k: v for k, v in a.items() if k != "tensors"} == {k: v for k, v in b.items() if k != "tensors"}
    assert a["op"] == nr.OP_RELU_GATED_VECTOR_PAIR and a["coeff"] == 1.0 and a["mode"] == "all"
    assert a["parent_pos_sha256"] == _sha(pos_b) and a["parent_neg_sha256"] == _sha(neg_b)
    assert (a["parent_pos_coeff"], a["parent_neg_coeff"]) == nr.PAIRED_TERM_COEFFS == (1.0, -1.0)
    # the same six keys per layer, every parent tensor bit-exact
    for l in layers:
        for fld, side, key in (("gate_w", pos, "gate_w"), ("direction", pos, "directions"), ("gate_w2", neg, "gate_w"),
                               ("direction2", neg, "directions")):
            assert torch.equal(a["tensors"][f"{nr.PAYLOAD_PREFIX}layers.{l}.{fld}"], side[key][str(l)])
    # through the worker's validator: one fingerprint
    blk = {**_paired_block(d, layers, L, pos_sha=_sha(pos_b), neg_sha=_sha(neg_b))}
    pa = nr.payload_from_block_and_tensors(blk, a["tensors"])
    pb = nr.payload_from_block_and_tensors(blk, b["tensors"])
    assert pa.fingerprint == pb.fingerprint and pa.paired and pa.gated
    # the artifact bytes differ (provenance), the PAYLOAD does not
    assert _bytes(fitted) != _bytes(composed)


def test_composition_refuses_parents_that_do_not_pair():
    d = 8
    pos = _psr_blob(_side(d, [0, 1], seed=1), [0, 1])
    with pytest.raises(lx.LoraExportError, match="layer sets differ"):
        lx.compose_paired_write_blob(pos, _psr_blob(_side(d, [0, 2], seed=2), [0, 2]), pos_sha256=POS_SHA, neg_sha256=NEG_SHA)
    with pytest.raises(lx.LoraExportError, match="shapes differ"):
        lx.compose_paired_write_blob(pos, _psr_blob(_side(d + 2, [0, 1], seed=2), [0, 1]), pos_sha256=POS_SHA, neg_sha256=NEG_SHA)
    masked = {**_psr_blob(_side(d, [0, 1], seed=2), [0, 1]), "positions": "response", "response_header_len": 3}
    with pytest.raises(lx.LoraExportError, match="positions"):
        lx.compose_paired_write_blob(pos, masked, pos_sha256=POS_SHA, neg_sha256=NEG_SHA)
    with pytest.raises(lx.LoraExportError, match="not 'psr'"):
        lx.compose_paired_write_blob(pos, {**_psr_blob(_side(d, [0, 1], seed=2), [0, 1]), "kind": "vector_all"},
                                     pos_sha256=POS_SHA, neg_sha256=NEG_SHA)
    with pytest.raises(lx.LoraExportError, match="same parent"):
        lx.compose_paired_write_blob(pos, pos, pos_sha256=POS_SHA, neg_sha256=POS_SHA)
    for bad in ("", "xyz", "A" * 64):
        with pytest.raises(lx.LoraExportError, match="sha256"):
            lx.compose_paired_write_blob(pos, _psr_blob(_side(d, [0, 1], seed=2), [0, 1]), pos_sha256=bad, neg_sha256=NEG_SHA)


@pytest.mark.parametrize("tag", sorted(REAL_SETS))
def test_composed_payload_equals_the_fitted_teacher_on_the_real_objects(tag):
    """The REAL objects (llama8b's fitted teacher 3ee2adb9...; Qwen3.5-4B batch-64 seed-1 V
    f98f0349... with its two parents), from a read-only scratch copy: the serve-time
    composition of the two parent blobs is BYTE-IDENTICAL to the fitted artifact's payload."""
    fitted_dir, pos_dir, neg_dir, d, shas = REAL_SETS[tag]
    paths = [OBJECTS_ROOT / x / "object.pt" for x in (fitted_dir, pos_dir, neg_dir)]
    if not all(p.is_file() for p in paths):
        pytest.skip(f"real objects for {tag} not present under {OBJECTS_ROOT} (read-only scratch; never fetched by a test)")
    raw = [p.read_bytes() for p in paths]
    assert [_sha(b) for b in raw] == list(shas), "the scratch copy is not the pinned object"
    fitted, pos_blob, neg_blob = (torch.load(io.BytesIO(b), map_location="cpu", weights_only=True) for b in raw)
    assert fitted["kind"] == PAIRED and fitted["provenance"]["parent_pos"]["object_sha256"] == shas[1]
    assert fitted["provenance"]["parent_neg"]["object_sha256"] == shas[2]
    composed = lx.compose_paired_write_blob(pos_blob, neg_blob, pos_sha256=shas[1], neg_sha256=shas[2])
    a = lx._residual_psr_paired_write(_spec(), fitted, hidden_size=d)
    b = lx._residual_psr_paired_write(_spec(), composed, hidden_size=d)
    assert a["layers"] == list(range(32)) == b["layers"]
    assert a["tensors"].keys() == b["tensors"].keys() and len(a["tensors"]) == 6 * 32
    for k in a["tensors"]:
        assert torch.equal(a["tensors"][k], b["tensors"][k]), k
    assert {k: v for k, v in a.items() if k != "tensors"} == {k: v for k, v in b.items() if k != "tensors"}
    blk = _paired_block(d, a["layers"], 36, pos_sha=shas[1], neg_sha=shas[2])
    pa = nr.payload_from_block_and_tensors(blk, a["tensors"])
    pb = nr.payload_from_block_and_tensors(blk, b["tensors"])
    assert pa.fingerprint == pb.fingerprint
    _report(f"real {tag}", fingerprint=pa.fingerprint[:16], layers=len(a["layers"]), hidden=d)


# ---------------------------------------------------------------------------
# the exporter: a fitted artifact's refusals and the end-to-end bank entry
# ---------------------------------------------------------------------------


def test_fitted_artifact_refusals():
    d, layers = 8, [0, 1]
    pos, neg = _side(d, layers, seed=1), _side(d, layers, seed=2)
    ok = _fitted_blob(pos, neg, layers, POS_SHA, NEG_SHA)
    assert lx._residual_psr_paired_write(_spec(), ok, hidden_size=d)["layers"] == layers
    # differing layer sets between the sides
    bad = {**ok, "neg": {k: {"0": v["0"]} for k, v in ok["neg"].items()}}
    with pytest.raises(lx.LoraExportError, match="layer sets differ"):
        lx._residual_psr_paired_write(_spec(), bad, hidden_size=d)
    # ... and a side lacking one layer's tensor in ONE group only (the per-layer read names the layer)
    bad = {**ok, "neg": {**ok["neg"], "gate_b": {"0": ok["neg"]["gate_b"]["0"]}}}
    with pytest.raises(lx.LoraExportError, match="layer 1"):
        lx._residual_psr_paired_write(_spec(), bad, hidden_size=d)
    # differing widths between the sides
    bad = {**ok, "neg": {**ok["neg"], "directions": {**ok["neg"]["directions"], "1": torch.zeros(d + 1)}}}
    with pytest.raises(lx.LoraExportError, match="dim"):
        lx._residual_psr_paired_write(_spec(), bad, hidden_size=d)
    # a masked position leg (the object refuses masked parents; the payload refuses it too)
    with pytest.raises(lx.LoraExportError, match="positions"):
        lx._residual_psr_paired_write(_spec(), {**ok, "positions": "response", "response_header_len": 3}, hidden_size=d)
    with pytest.raises(lx.LoraExportError, match="positions"):
        lx._residual_psr_paired_write(_spec(positions="response"), ok, hidden_size=d)
    # a foreign application order (a sequential composition is a different function)
    with pytest.raises(lx.LoraExportError, match="application_order"):
        lx._residual_psr_paired_write(_spec(), {**ok, "application_order": "pos_then_neg"}, hidden_size=d)
    # provenance without both parent shas: an ad-hoc object is never banked
    with pytest.raises(lx.LoraExportError, match="parent"):
        lx._residual_psr_paired_write(_spec(), {**ok, "provenance": {}}, hidden_size=d)
    with pytest.raises(lx.LoraExportError, match="parent"):
        lx._residual_psr_paired_write(_spec(), {**ok, "provenance": {**ok["provenance"], "parent_neg": {}}}, hidden_size=d)
    # a spec/blob layer disagreement (the psr exporter's F9-style cross-check)
    with pytest.raises(lx.LoraExportError, match="layers"):
        lx._residual_psr_paired_write(_spec(layers=[0]), ok, hidden_size=d)
    # the wrong kind
    with pytest.raises(lx.LoraExportError, match="kind"):
        lx._residual_psr_paired_write(_spec(), {**ok, "kind": "psr"}, hidden_size=d)
    # the export-kind registries moved together
    assert PAIRED in lx.RESIDUAL_EXPORT_KINDS and PAIRED in lx.GENUINE_EXPORT_KINDS


@pytest.fixture(scope="module")
def tiny_base(tmp_path_factory):
    """The tiny llama saved as a local snapshot with its tokenizer (the export test's hermetic
    fixture) — the exporter reads the base's module tree for the anchor and the width."""
    transformers = pytest.importorskip("transformers")
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, vocab_size=97, max_position_embeddings=64, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    d = tmp_path_factory.mktemp("tiny-llama-paired")
    model.save_pretrained(d)
    try:
        tok = transformers.AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-LlamaForCausalLM",
                                                         local_files_only=True)
    except Exception as e:  # noqa: BLE001 — not cached: skip, never fetch
        pytest.skip(f"tiny llama tokenizer not in the local HF cache ({type(e).__name__}: {e})")
    tok.save_pretrained(d)
    return str(d)


def test_end_to_end_export_of_fitted_and_composed_artifacts_yields_one_bank_entry(tiny_base, tmp_path):
    d, L, layers = 32, 2, [0, 1]
    pos, neg = _side(d, layers, seed=51), _side(d, layers, seed=52)
    pos_b, neg_b = _bytes(_psr_blob(pos, layers)), _bytes(_psr_blob(neg, layers))
    fitted = _fitted_blob(pos, neg, layers, _sha(pos_b), _sha(neg_b))
    composed = lx.compose_paired_write_blob(_psr_blob(pos, layers), _psr_blob(neg, layers),
                                            pos_sha256=_sha(pos_b), neg_sha256=_sha(neg_b))
    spec = {"kind": PAIRED, "positions": "all", "petri_intervention_hash": "cd" * 8, "layers": layers}
    out_f = lx.export_genuine_adapter(spec, _bytes(fitted), tmp_path / "fitted", base_model=tiny_base)
    out_c = lx.export_genuine_adapter(spec, _bytes(composed), tmp_path / "composed", base_model=tiny_base)
    wf, wc = (out_f / lx.ADAPTER_WEIGHTS).read_bytes(), (out_c / lx.ADAPTER_WEIGHTS).read_bytes()
    assert wf == wc, "the fitted and the composed payload must be BYTE-IDENTICAL"
    cf, cc = json.loads((out_f / lx.ADAPTER_CONFIG).read_text()), json.loads((out_c / lx.ADAPTER_CONFIG).read_text())
    assert cf["petri_residual_slot"] == cc["petri_residual_slot"]
    assert cf["petri_source_kind"] == PAIRED and cf["petri_fit_positions"] == "all" and cf["r"] == 1
    block = cf["petri_residual_slot"]
    assert block["kind"] == PAIRED and block["op"] == nr.OP_RELU_GATED_VECTOR_PAIR
    assert block["parent_pos_sha256"] == _sha(pos_b) and block["parent_neg_sha256"] == _sha(neg_b)
    assert (block["parent_pos_coeff"], block["parent_neg_coeff"]) == (1.0, -1.0)
    assert block["coeff"] == 1.0 and block["skip_bos"] is False and block["dtype_policy"] == nr.DTYPE_POLICY_GATED
    # the worker reads it back as a paired payload, both sides bit-exact, and the digests name the kind
    payload = nr.read_residual_payload(out_f)
    assert payload.paired and payload.layers == (0, 1)
    for l in layers:
        assert torch.equal(payload.gate_w[l], pos["gate_w"][str(l)]) and torch.equal(payload.gate_w2[l], neg["gate_w"][str(l)])
        assert torch.equal(payload.direction[l], pos["directions"][str(l)]) and torch.equal(payload.direction2[l], neg["directions"][str(l)])
        assert torch.equal(payload.gate_b2[l], neg["gate_b"][str(l)].reshape(1))
    dig = lx.adapter_digests(out_f)
    assert dig.petri_source_kind == PAIRED and dig.petri_position_mode == "all" and dig.petri_zero_adapter is False
    # a LoRA-side reader still sees exactly the rank-1 zero carrier
    keys = set(sa.adapter_module_dims(out_f))
    assert keys == {"base_model.model.model.layers.0.self_attn.o_proj"}
    # the composer would ARM the paired term for this launch (its torch-free reading of the block)
    from sisteer.capeval.serving import vllm_server as vs
    assert vs._residual_slot_armed({"x": str(out_f)}) and vs._residual_slot_paired({"x": str(out_f)})


# ---------------------------------------------------------------------------
# (e) the four registry moves and the lens the launcher would otherwise take
# ---------------------------------------------------------------------------


def _paired_resolved_spec(layers=(0, 1)):
    """What registry_adapter._resolve_torch_blob yields for a fitted paired artifact: the
    non-tensor fields (the two sides drop out as tensor payloads), plus kind / arm."""
    return {"layers": list(layers), "application_order": nr.PAIRED_APPLICATION_ORDER, "positions": "all",
            "response_header_len": None,
            "provenance": {"parent_pos": {"object_sha256": POS_SHA}, "parent_neg": {"object_sha256": NEG_SHA}},
            "kind": PAIRED, "arm": ARM, "si_id": "constv2_mix_full"}


def test_the_four_registry_moves_and_the_predicates():
    from sisteer.extraction.paired_write import APPLICATION_ORDER, PSRPairedWriteObject
    from sisteer.methods import ARMS, DERIVED_ARMS, is_derived_arm

    # one kind tuple, three spellings — every one carries the paired kind
    assert dc.RESIDUAL_SLOT_KINDS == nr.RESIDUAL_SLOT_KINDS == lx.RESIDUAL_EXPORT_KINDS
    assert PAIRED in dc.RESIDUAL_SLOT_KINDS and PAIRED in dc.NATIVE_LORA_SERVABLE_KINDS
    assert set(dc.NATIVE_LORA_SERVABLE_KINDS) == set(lx.GENUINE_EXPORT_KINDS)
    assert set(dc.RESIDUAL_SLOT_KINDS) < set(dc.NATIVE_LORA_SERVABLE_KINDS)
    assert PSRPairedWriteObject.kind == PAIRED and nr.PAIRED_KINDS == (PAIRED,)
    assert nr.PAIRED_APPLICATION_ORDER == APPLICATION_ORDER
    # the arm: Class A grammar, OBJECT-ONLY (the statistical family stays at 13), kind mapping
    assert ARM in dc.CLASS_A_METHODS and ARM in dc.OBJECT_ONLY_METHODS
    assert ARM not in dc.PROMPTABLE_CLASS_A_METHODS and len(dc.PROMPTABLE_CLASS_A_METHODS) == 13
    assert tuple(dc.PROMPTABLE_CLASS_A_METHODS) + dc.OBJECT_ONLY_METHODS == dc.CLASS_A_METHODS
    assert ra.ARM_OBJECT_KINDS[ARM] == PAIRED
    assert dc.parse_cell_id(HCELL).method == ARM and dc.parse_cell_id(HCELL).cell_class == "class_a"
    assert ra.harness_cell_id(CELL) == HCELL and ra.traintest_cell_id(HCELL) == CELL
    # the object-only SI has no prompted / stacked spelling; a variant on a prompt-SI spelling of an
    # object-only method is not a cell the vocabulary expresses either
    for bad in ("prompted__constv2_mix_full", f"stacked__constv2_mix_full__{ARM}"):
        with pytest.raises(dc.DefenseCellError):
            dc.parse_cell_id(bad)
    # a DERIVED arm — registered for hydration + serving, build and train refuse
    assert ARM in DERIVED_ARMS and is_derived_arm(ARM) and ARM in ARMS
    with pytest.raises(RuntimeError, match="derived object"):
        ARMS[ARM].build(None, None, 0)
    with pytest.raises(RuntimeError, match="never trained"):
        ARMS[ARM].train(None, None, None, [])
    # the predicates the launcher's lens hangs on: NOT refused, judged `all`, disposition serve
    spec = _paired_resolved_spec()
    assert dc.native_lora_unservable_reason(PAIRED, spec) is None
    assert dc.native_lora_unservable_reason(PAIRED, None) is None
    assert dc.native_position_mode(PAIRED, spec) == ("all", None)
    assert dc.native_position_mode(PAIRED, {k: v for k, v in spec.items() if k != "positions"}) == ("all", None)
    assert dc.native_serving_disposition(PAIRED, spec) == ("serve", None)
    # a paired artifact whose leg is masked is judged like a psr record (the mode is what it says)
    assert dc.native_position_mode(PAIRED, {**spec, "positions": "response", "response_header_len": 3}) == ("response", 3)
    with pytest.raises(dc.DefenseCellError):
        dc.native_position_mode(PAIRED, {**spec, "positions": "last"})
    # the composer's torch-free constants agree with the worker's
    from sisteer.capeval.serving import vllm_server as vs
    assert vs.RESIDUAL_SLOT_PAIRED_ENV == nr.ENV_FLAG_PAIRED and vs.RESIDUAL_SLOT_PAIRED_OP == nr.OP_RELU_GATED_VECTOR_PAIR
    assert nr.ENV_FLAG_PAIRED != nr.ENV_FLAG


# ---------------------------------------------------------------------------
# (f) the fingerprint covers both sides and both parent shas
# ---------------------------------------------------------------------------


def test_fingerprint_is_distinct_from_both_parents_the_swap_and_other_parent_shas():
    d, L, layers = 16, 3, [0, 1]
    pos, neg = _side(d, layers, seed=61), _side(d, layers, seed=62)
    paired = _paired_payload(pos, neg, d, layers, L)
    p_pos, p_neg = _single_payload(pos, d, layers, L), _single_payload(neg, d, layers, L)
    swapped = _paired_payload(neg, pos, d, layers, L, pos_sha=NEG_SHA, neg_sha=POS_SHA)
    other_shas = _paired_payload(pos, neg, d, layers, L, pos_sha="c" * 64, neg_sha="d" * 64)
    fps = [paired.fingerprint, p_pos.fingerprint, p_neg.fingerprint, swapped.fingerprint, other_shas.fingerprint]
    assert len(set(fps)) == len(fps), fps
    assert all(len(f) == 64 for f in fps)
    # deterministic, and the block alone (parent shas) moves it: same tensors, other shas
    assert _paired_payload(pos, neg, d, layers, L).fingerprint == paired.fingerprint
    assert paired.summary()["parent_pos_sha256"] == POS_SHA and paired.summary()["parent_neg_sha256"] == NEG_SHA
    assert "parent_pos_sha256" not in p_pos.summary() and "parent_pos_sha256" not in p_pos.block()
    # the block and the tensors round-trip through the payload's own writers
    assert nr.residual_fingerprint(paired.block(), paired.tensors()) == paired.fingerprint
    assert set(paired.tensors()) == {f"{nr.PAYLOAD_PREFIX}layers.{l}.{f}" for l in layers
                                     for f in nr.GATED_FIELDS + nr.PAIRED_FIELDS}


# ---------------------------------------------------------------------------
# (g) an unpaired standup is unchanged
# ---------------------------------------------------------------------------


def test_slot_buffers_first_eleven_fields_and_the_unarmed_allocation_are_unchanged():
    assert nr.SlotBuffers._fields[:11] == ("slot_by_token", "active", "gated", "skip_bos", "gate_w", "gate_b",
                                           "direction", "vector", "mode", "prompt_len_by_token",
                                           "response_start_by_token")
    assert nr.SlotBuffers._fields[11:] == ("paired", "gate_w2", "gate_b2", "direction2")
    b = _bank(S=4, L=3, d=16, max_tokens=32)
    assert b.buffers.gate_w2 is None and b.buffers.gate_b2 is None and b.buffers.direction2 is None
    assert b.buffers.paired is None and b.paired_armed is False
    rep = b.report()
    assert rep["paired_armed"] is False and rep["paired_slots"] == 0
    # an ARMED bank allocates the three tensors in the slot shapes, once, at construction
    nr.reset_for_tests()
    a = _bank(S=4, L=3, d=16, max_tokens=32, paired=True)
    assert a.paired_armed and tuple(a.buffers.gate_w2.shape) == (4, 3, 16) == tuple(a.buffers.direction2.shape)
    assert tuple(a.buffers.gate_b2.shape) == (4, 3) and tuple(a.buffers.paired.shape) == (4,)
    assert a.buffers.gate_w2.dtype == a.buffers.gate_b2.dtype == a.buffers.direction2.dtype == torch.float32
    assert a.buffers.paired.dtype == torch.bool and not a.buffers.paired.any()
    assert a.report()["paired_armed"] is True and a.report()["paired_slots"] == 0


def test_a_paired_payload_on_an_unarmed_bank_refuses_and_names_the_flag():
    d, L, layers = 16, 3, [0, 1]
    b = _bank(S=2, L=L, d=d, max_tokens=8)
    with pytest.raises(nr.ResidualSlotError, match=nr.ENV_FLAG_PAIRED):
        b.load_slot(0, _paired_payload(_side(d, layers, seed=1), _side(d, layers, seed=2), d, layers, L), lora_int_id=1)
    assert b.records[0] is None and not b.buffers.active[0].any()      # a refused load leaves an EMPTY slot


@pytest.mark.parametrize("dtype", [BF16, torch.float32])
@pytest.mark.parametrize("armed", [False, True])
def test_unpaired_slots_write_bit_identically_to_the_single_gate_body_as_it_stood(dtype, armed):
    d, L, T = 16, 3, 40
    b = _bank(S=4, L=L, d=d, max_tokens=T, dtype=dtype, paired=armed)
    b.load_slot(0, _single_payload(_side(d, [0, 1], seed=71, bias=0.3), d, [0, 1], L), lora_int_id=1)
    g = torch.Generator().manual_seed(4)
    const_block = {**_single_block(d, [1, 2], L), "kind": "vector_all", "op": nr.OP_ADD_VECTOR, "skip_bos": True,
                   "dtype_policy": nr.DTYPE_POLICY_CONSTANT}
    b.load_slot(1, nr.payload_from_block_and_tensors(
        const_block, {f"{nr.PAYLOAD_PREFIX}layers.{l}.vector": torch.randn(d, generator=g) for l in (1, 2)}), lora_int_id=2)
    b.load_slot(2, None, lora_int_id=3)                       # a plain LoRA
    hidden = (torch.randn(T, d, generator=g) * 3.0).to(dtype)
    residual = (torch.randn(T, d, generator=g) * 3.0).to(dtype)
    positions = torch.arange(T) % 7
    sbt = torch.tensor([0, 1, 2, -1] * (T // 4), dtype=torch.int64)   # PSR, constant, plain LoRA, base — interleaved
    b.buffers.slot_by_token[:T] = sbt
    for boundary in range(L):
        for res in (residual, None):
            got = b.apply(hidden, res, positions, boundary)
            want = _legacy_forward(b, hidden, res, positions, boundary)
            assert torch.equal(got, want), (boundary, res is None)
            assert not torch.equal(got, hidden) if boundary in (0, 1, 2) else True   # the fixture steers
    if armed:
        assert b.report()["paired_slots"] == 0 and not b.buffers.paired.any()
        # the second-gate rows of an unpaired slot are zero (a cleared slot never carries a tenant's bytes)
        assert not b.buffers.gate_w2.any() and not b.buffers.direction2.any() and not b.buffers.gate_b2.any()


class _OpRecorder(TorchDispatchMode):
    """The ORDERED aten dispatch sequence of a body. The op runs inside CUDA-graph
    capture as one opaque custom op, so the unarmed path's launch ORDER — not only
    its output bytes — is the certified graph."""

    def __init__(self):
        super().__init__()
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.ops.append(str(func))
        return func(*args, **(kwargs or {}))


def _hoisted_rows(hidden, residual, pos, boundary, buffers, idx, prompt_len_tok, response_start_tok):
    """KNOWN-BAD: the P1-1 body — the mask cast HOISTED before the gate and reused. Same bytes,
    a different dispatch order (the mask cast standing where the hidden-state cast was)."""
    live = idx >= 0
    idx_c = idx.clamp(min=0)
    sel = live & buffers.active[idx_c, boundary]
    if pos is not None:
        sel = sel & ~(buffers.skip_bos[idx_c] & (pos == 0))
    sel = sel & nr.position_class_selection(buffers.mode[idx_c], pos, prompt_len_tok, response_start_tok)
    true = hidden if residual is None else hidden + residual
    w_tok = buffers.gate_w[idx_c, boundary]
    b_tok = buffers.gate_b[idx_c, boundary]
    d_tok = buffers.direction[idx_c, boundary]
    v_tok = buffers.vector[idx_c, boundary]
    sel32 = sel.to(torch.float32)
    g = F.relu((true.float() * w_tok).sum(-1) + b_tok)
    add32 = (g * sel32).unsqueeze(-1) * d_tok
    applied_gated = true + add32.to(hidden.dtype)
    applied_const = true + v_tok
    applied = torch.where(buffers.gated[idx_c].unsqueeze(-1), applied_gated, applied_const)
    return nr.writeback(hidden, true, applied, sel)


@pytest.mark.parametrize("dtype", [BF16, torch.float32])
@pytest.mark.parametrize("form", ["llama", "gemma4"])
def test_unarmed_kernel_dispatch_sequence_equals_the_pre_change_body_and_the_hoist_is_caught(dtype, form):
    """The unarmed path's aten dispatch SEQUENCE equals the pre-change body's op for op (not
    only its bytes — the captured graph is the sequence), and the instrument catches the
    known-bad hoist P1-1 named."""
    d, L, T = 16, 3, 24
    b = _bank(S=3, L=L, d=d, max_tokens=T, dtype=dtype, paired=False)
    b.load_slot(0, _single_payload(_side(d, [0, 1], seed=81, bias=0.3), d, [0, 1], L), lora_int_id=1)
    g = torch.Generator().manual_seed(6)
    const_block = {**_single_block(d, [1, 2], L), "kind": "vector_all", "op": nr.OP_ADD_VECTOR, "skip_bos": True,
                   "dtype_policy": nr.DTYPE_POLICY_CONSTANT}
    b.load_slot(1, nr.payload_from_block_and_tensors(
        const_block, {f"{nr.PAYLOAD_PREFIX}layers.{l}.vector": torch.randn(d, generator=g) for l in (1, 2)}), lora_int_id=2)
    hidden = (torch.randn(T, d, generator=g) * 3.0).to(dtype)
    residual = None if form == "gemma4" else (torch.randn(T, d, generator=g) * 3.0).to(dtype)
    positions = torch.arange(T) % 7
    idx = torch.tensor([0, 1, -1] * (T // 3), dtype=torch.int64)
    bufs = b.buffers
    pl, rs = bufs.prompt_len_by_token[:T], bufs.response_start_by_token[:T]
    for boundary in range(L):
        args = (hidden, residual, positions, boundary, bufs, idx, pl, rs)
        with _OpRecorder() as new:
            got = nr._residual_slot_forward_rows(*args)
        with _OpRecorder() as old:
            want = _legacy_rows(*args)
        with _OpRecorder() as bad:
            hoisted = _hoisted_rows(*args)
        assert torch.equal(got, want) and torch.equal(hoisted, want)          # bytes: all three agree ...
        assert len(old.ops) > 20 and new.ops == old.ops, (boundary, new.ops, old.ops)   # ... the SEQUENCE: only the fix does
        assert bad.ops != old.ops, "the instrument must catch the hoisted mask cast (P1-1)"
    _report("dispatch sequence (unarmed)", n_ops=len(old.ops), dtype=str(dtype), form=form)


# ---------------------------------------------------------------------------
# the LENS adapter for the paired kind (P1-2): the cert's lens legs serve the real thing
# ---------------------------------------------------------------------------


def test_lens_adapter_reproduces_the_paired_write_objects_hook_bit_exact(tiny_llama):
    """``lens_adapters.get_adapter("psr_paired_write")`` -> one OP_RELU_GATED_VECTOR_PAIR op per
    layer whose ``apply_hook_ops`` output equals ``PSRPairedWriteObject``'s own hook body on a
    random state bit for bit (fp32; and bf16 through the worker's materialization), at scale 1
    and 0.5; scale 0 is the SAME tensor."""
    from sisteer.capeval.serving import lens_adapters as la

    model, _ = tiny_llama
    d, L = model.config.hidden_size, model.config.num_hidden_layers
    layers = list(range(L))
    pos, neg = _side(d, layers, seed=91, bias=0.6), _side(d, layers, seed=92, dir_scale=1.5, bias=0.4)
    obj = _hydrate_pair(model, layers, pos, neg)
    art = _bytes(_fitted_blob(pos, neg, layers, POS_SHA, NEG_SHA))
    ops = la.get_adapter(PAIRED)({"kind": PAIRED, "layers": layers, "positions": "all"}, art, b"")
    assert [op.layer for op in ops] == layers
    assert all(op.op == la.OP_RELU_GATED_VECTOR_PAIR and op.phase == la.PHASE_BOTH and not op.skip_bos
               and op.positions == la.POSITIONS_ALL and not la.op_needs_context(op) for op in ops)
    for l in layers:
        op = ops[l]
        assert torch.equal(op.vector, pos["directions"][str(l)]) and torch.equal(op.vector2, neg["directions"][str(l)])
        assert torch.equal(op.gate_w, pos["gate_w"][str(l)]) and torch.equal(op.gate_w2, neg["gate_w"][str(l)])
    g = torch.Generator().manual_seed(7)
    hidden = torch.randn(2, 6, d, generator=g) * 4.0
    n_changed = 0
    for l in layers:
        for scale in (1.0, 0.5):
            ref = obj._make_hook(l, scale)(None, None, hidden).detach()    # the object's own hook body
            got = la.apply_hook_ops(hidden, ops, scale, layer=l)
            assert torch.equal(got, ref), (l, scale, float((got - ref).abs().max()))
            n_changed += int(not torch.equal(got, hidden))
        assert la.apply_hook_ops(hidden, ops, 0.0, layer=l) is hidden
    assert n_changed == 2 * L, "the fixture must steer at every layer"
    # bf16 model dtype through the worker's own materialization: fp32 fields stay fp32, ONE cast
    mat = la.materialize_ops(ops, device="cpu", dtype=BF16)
    hb = hidden.to(BF16)
    for l in layers:
        assert torch.equal(la.apply_hook_ops(hb, mat, 1.0, layer=l), obj._make_hook(l, 1.0)(None, None, hb).detach())
    # the vocabulary: a gated reduction op, per-position, the ONE op name both routes spell
    assert la.op_reduces_in_group(la.OP_RELU_GATED_VECTOR_PAIR) and la.op_name_per_position(la.OP_RELU_GATED_VECTOR_PAIR)
    assert la.reduction_op_class(la.OP_RELU_GATED_VECTOR_PAIR) == la.OP_CLASS_REDUCTION_GATED
    assert la.OP_RELU_GATED_VECTOR_PAIR == nr.OP_RELU_GATED_VECTOR_PAIR
    assert la._PAIRED_APPLICATION_ORDER == nr.PAIRED_APPLICATION_ORDER
    assert PAIRED in la.registered_kinds() and PAIRED not in la.UNSERVABLE_KIND_REASONS


def test_lens_adapter_refusals_and_the_worker_gate_stats():
    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.capeval.serving import lens_worker as lw

    d, layers = 8, [0, 1]
    pos, neg = _side(d, layers, seed=1), _side(d, layers, seed=2)
    ok = _fitted_blob(pos, neg, layers, POS_SHA, NEG_SHA)
    adapt = la.get_adapter(PAIRED)
    assert len(adapt({"kind": PAIRED}, _bytes(ok), b"")) == 2
    # a served or fit position leg other than `all`
    with pytest.raises(la.AdapterError):
        adapt({"kind": PAIRED, "positions": "response"}, _bytes(ok), b"", header_ids=(1, 2, 3))
    with pytest.raises(la.AdapterError):
        adapt({"kind": PAIRED}, _bytes({**ok, "positions": "response", "response_header_len": 3}), b"",
              header_ids=(1, 2, 3))
    # a foreign application order; a side lacking a layer; a side of another width; the wrong kind
    with pytest.raises(la.AdapterError, match="application_order"):
        adapt({"kind": PAIRED}, _bytes({**ok, "application_order": "pos_then_neg"}), b"")
    with pytest.raises(la.AdapterError, match="layer"):
        adapt({"kind": PAIRED}, _bytes({**ok, "neg": {k: {"0": v["0"]} for k, v in ok["neg"].items()}}), b"")
    with pytest.raises(la.AdapterError, match="dim"):
        adapt({"kind": PAIRED}, _bytes({**ok, "neg": {**ok["neg"], "gate_w": {**ok["neg"]["gate_w"],
                                                                              "1": torch.zeros(d + 1)}}}), b"")
    with pytest.raises(la.AdapterError):
        adapt({"kind": PAIRED}, _bytes({**ok, "kind": "psr"}), b"")
    with pytest.raises(la.AdapterError, match="layers"):
        adapt({"kind": PAIRED, "layers": [0]}, _bytes(ok), b"")
    # HookOp's own field discipline: the pair needs its second side; a single op refuses it
    v, w, bb = pos["directions"]["0"], pos["gate_w"]["0"], pos["gate_b"]["0"]
    with pytest.raises(la.AdapterError, match="gate_w2|vector2|gate_b2"):
        la.HookOp(op=la.OP_RELU_GATED_VECTOR_PAIR, layer=0, vector=v, gate_w=w, gate_b=bb)
    with pytest.raises(la.AdapterError, match="gate_w2"):
        la.HookOp(op=la.OP_RELU_GATED_VECTOR, layer=0, vector=v, gate_w=w, gate_b=bb, gate_w2=w)
    with pytest.raises(la.AdapterError, match="positions"):
        la.HookOp(op=la.OP_RELU_GATED_VECTOR_PAIR, layer=0, vector=v, gate_w=w, gate_b=bb,
                  vector2=v, gate_w2=w, gate_b2=bb, positions="response", header_ids=(1, 2))
    # the worker's shape class and its recomputed gate stats see BOTH gates
    ops = adapt({"kind": PAIRED}, _bytes(ok), b"")
    assert lw._op_shape_class(ops[0]) == "gemv"
    rows = torch.randn(5, d, generator=torch.Generator().manual_seed(3)) * 3.0
    st = lw._gate_stats([ops[0]], rows)
    assert st["gate_ops"] == [la.OP_RELU_GATED_VECTOR_PAIR] and st["gate_n_elems"] == 2 * 5 and "gate_error" not in st
    assert lw._gate_stats([ops[0]], rows[:1]) == {}          # one row: nothing to compare


# ---------------------------------------------------------------------------
# the cert row's REQUIRED PHASES against its cells (P1-2): caught before a GPU is allocated
# ---------------------------------------------------------------------------


def test_every_target_rows_cells_are_servable_on_every_required_phase():
    import sys

    from sisteer.capeval.serving import lens_adapters as la
    from sisteer.eval.defense_cell import arm_of
    for p in (REPO / "scripts", REPO / "scripts" / "sdfbase"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import multilora_cert as mc

    # what the registry adapter resolves for each kind the rows serve (the fit leg the predicates read)
    specs = {"lora_all": {"kind": "lora_all", "positions": "all"},
             "lora_subtraction": {"kind": "lora_subtraction", "positions": "all"},
             "psr": {"kind": "psr", "positions": "all"},
             "static_r": {"kind": "static_r", "base_scale": 0.05, "steer_positions": "prefill_all", "site": "pre",
                          "layers": [19], "metadata": {}},
             PAIRED: _paired_resolved_spec()}
    t = mc.TARGETS["qwen35_4b_paired"]
    assert mc.required_phases(t) == ("oracle", "lora_native", "gate_canary", "throughput_native",
                                     "lens_stacked", "admission")
    assert mc.required_serving_paths(t) == ("lora_native", "lens")
    full = mc.TARGETS["qwen35_4b"]                                             # a numerics row: the same two paths
    assert mc.required_serving_paths(full) == ("lora_native", "lens")
    assert "lens_per_row" in mc.required_phases(full) and "probes" in mc.required_phases(full)
    for key, row in mc.TARGETS.items():
        for cell in row.cells:
            kind = ra.ARM_OBJECT_KINDS[arm_of(cell)]
            spec = dict(specs[kind])
            if kind == "psr" and "psr_kl_all" in cell:
                spec.update(positions="response", response_header_len=3)     # the response-fit canaries
            problems = mc.cell_serving_problems(kind, spec, paths=mc.required_serving_paths(row))
            assert problems == [], (key, cell, kind, problems)
    # the paired cell on BOTH paths, explicitly
    assert mc.cell_serving_problems(PAIRED, _paired_resolved_spec()) == []
    assert la.get_adapter(PAIRED) is not None and dc.native_lora_unservable_reason(PAIRED, _paired_resolved_spec()) is None
    # KNOWN-BADS: a native-only kind fails the lens phase by name; a lens-only kind fails the native one
    bad = mc.cell_serving_problems("circuit_breaker", {"kind": "circuit_breaker", "rank": 16, "lora_alpha": 16})
    assert len(bad) == 1 and "lens_stacked" in bad[0] and "circuit_breaker" in bad[0]
    bad = mc.cell_serving_problems("alphasteer", {"kind": "alphasteer"})
    assert any("lora_native" in p for p in bad) and not any("lens_stacked" in p for p in bad)
    # the phase table names every run_unit phase once, and the paths it stands up are exactly the two
    assert [n for n, _ in mc.UNIT_PHASES] == ["oracle", "lora_native", "gate_canary", "probes", "throughput_native",
                                             "lens_stacked", "lens_per_row", "admission"]
    assert set(mc.NUMERICS_ONLY_PHASES) == {"probes", "lens_per_row"}


def test_clear_slot_zeroes_the_second_side_and_the_paired_flag():
    d, L, layers = 16, 3, [0, 1]
    b = _bank(S=2, L=L, d=d, max_tokens=8, paired=True)
    b.load_slot(0, _paired_payload(_side(d, layers, seed=1), _side(d, layers, seed=2), d, layers, L), lora_int_id=1)
    assert bool(b.buffers.paired[0]) and b.buffers.gate_w2[0].any() and b.report()["paired_slots"] == 1
    assert b.slots_report()["1"]["parent_pos_sha256"] == POS_SHA
    assert b.clear_slot(0)
    assert not b.buffers.paired.any() and not b.buffers.gate_w2.any() and not b.buffers.gate_b2.any() \
        and not b.buffers.direction2.any() and b.report()["paired_slots"] == 0
    # a paired payload over a slot that held a single-gate one, and back
    b.load_slot(0, _single_payload(_side(d, layers, seed=3), d, layers, L), lora_int_id=2)
    assert not bool(b.buffers.paired[0]) and not b.buffers.gate_w2[0].any()


# ---------------------------------------------------------------------------
# the validator, the classifier, the reader
# ---------------------------------------------------------------------------


def test_validator_refusals():
    d, L, layers = 8, 2, [0, 1]
    pos, neg = _side(d, layers, seed=1), _side(d, layers, seed=2)
    good = _paired_block(d, layers, L)
    assert nr.payload_from_block_and_tensors(good, _paired_tensors(pos, neg, layers)).paired
    def refuse(block, tensors, needle):
        with pytest.raises(nr.ResidualSlotError, match=needle):
            nr.payload_from_block_and_tensors(block, tensors)
    refuse({**good, "kind": "psr"}, _paired_tensors(pos, neg, layers), "kind")                     # paired op, single kind
    refuse({**_single_block(d, layers, L), "kind": PAIRED}, _single_tensors(pos, layers), "kind")  # paired kind, single op
    refuse({k: v for k, v in good.items() if k != "parent_neg_sha256"}, _paired_tensors(pos, neg, layers), "keys")
    refuse({**_single_block(d, layers, L), "parent_pos_sha256": POS_SHA}, _single_tensors(pos, layers), "keys")
    refuse({**good, "parent_pos_coeff": 2.0}, _paired_tensors(pos, neg, layers), "coeff")
    refuse({**good, "parent_neg_coeff": 1.0}, _paired_tensors(pos, neg, layers), "coeff")
    refuse({**good, "parent_neg_sha256": POS_SHA}, _paired_tensors(pos, neg, layers), "same parent")
    refuse({**good, "parent_pos_sha256": "nope"}, _paired_tensors(pos, neg, layers), "sha256")
    refuse({**good, "coeff": 0.5}, _paired_tensors(pos, neg, layers), "coeff")
    refuse({**good, "skip_bos": True}, _paired_tensors(pos, neg, layers), "skip_bos")
    refuse({**good, "dtype_policy": nr.DTYPE_POLICY_CONSTANT}, _paired_tensors(pos, neg, layers), "dtype_policy")
    # a paired op with only ONE side present; a single op carrying a second side
    refuse(good, _single_tensors(pos, layers), "lacks")
    refuse(_single_block(d, layers, L), _paired_tensors(pos, neg, layers), "is not one of op")
    # a second side of the wrong width / dtype / a non-finite entry
    bad = _paired_tensors(pos, neg, layers)
    bad[f"{nr.PAYLOAD_PREFIX}layers.0.direction2"] = torch.zeros(d + 1)
    refuse(good, bad, "shape")
    bad = _paired_tensors(pos, neg, layers)
    bad[f"{nr.PAYLOAD_PREFIX}layers.0.gate_w2"] = torch.zeros(d, dtype=BF16)
    refuse(good, bad, "fp32")
    bad = _paired_tensors(pos, neg, layers)
    bad[f"{nr.PAYLOAD_PREFIX}layers.1.gate_b2"] = torch.tensor([float("nan")])
    refuse(good, bad, "non-finite")
    # op / kind tables
    assert nr.RESIDUAL_OPS == (nr.OP_RELU_GATED_VECTOR, nr.OP_RELU_GATED_VECTOR_PAIR, nr.OP_ADD_VECTOR)
    assert nr.DTYPE_POLICIES[nr.OP_RELU_GATED_VECTOR_PAIR] == nr.DTYPE_POLICY_GATED
    assert nr.op_for_kind(PAIRED) == nr.OP_RELU_GATED_VECTOR_PAIR and nr.op_for_kind("psr") == nr.OP_RELU_GATED_VECTOR
    assert nr.op_for_kind("static_r") == nr.OP_ADD_VECTOR
    with pytest.raises(nr.ResidualSlotError):
        nr.op_for_kind("lora_all")


def test_readiness_expectation_carries_paired_and_refuses_a_bank_that_disagrees(tmp_path):
    """The readiness belt beside the load-time refusal: the launcher's expectation says whether a
    registered payload is the paired op, and every worker's bank must carry its second-gate
    buffers iff it is; the composer's own cross-check refuses a launch whose paired flag
    disagrees with the adapters."""
    from native_residual_fakes import write_adapter_dir
    from sisteer.capeval.serving import vllm_server as vs

    d, L, layers = 8, 2, [0, 1]
    pos, neg = _side(d, layers, seed=1), _side(d, layers, seed=2)
    paired = _paired_payload(pos, neg, d, layers, L)
    single = _single_payload(pos, d, layers, L)
    dirs = {"pair": str(write_adapter_dir(tmp_path / "pair", layers=[0], d_in=d, d_out=d, zero=True,
                                          residual_block=paired.block(), residual_tensors=paired.tensors())),
            "one": str(write_adapter_dir(tmp_path / "one", layers=[0], d_in=d, d_out=d, zero=True,
                                         residual_block=single.block(), residual_tensors=single.tensors())),
            "plain": str(write_adapter_dir(tmp_path / "plain", layers=[0], d_in=d, d_out=d))}
    exp = sa.expected_residual_slot_state(["one", "pair", "plain"], dirs)
    assert exp["armed"] and exp["paired"] and exp["adapters"]["pair"]["op"] == nr.OP_RELU_GATED_VECTOR_PAIR
    exp_single = sa.expected_residual_slot_state(["one", "plain"], {k: dirs[k] for k in ("one", "plain")})
    assert exp_single["armed"] and not exp_single["paired"]
    # a worker whose bank lacks the buffers while a paired payload is registered is NAMED; the inverse too;
    # an older report without the key on a single-gate launch raises no paired problem
    base = {"armed": True, "ledger_fault": False, "ledger_size": 0, "slots": {}, "op_name": sa.RESIDUAL_SLOT_OP_NAME,
            "op_registered": True, "runner_bound": True, "n_layers_model": L, "n_layers_hooked": L}
    assert any("second-gate" in p for p in sa.residual_slot_problems([{**base, "bank": {"paired_armed": False}}], exp))
    assert any("second-gate" in p for p in sa.residual_slot_problems([{**base, "bank": {}}], exp))
    assert not any("second-gate" in p for p in sa.residual_slot_problems([{**base, "bank": {"paired_armed": True}}], exp))
    assert any("second-gate" in p for p in sa.residual_slot_problems([{**base, "bank": {"paired_armed": True}}], exp_single))
    assert not any("second-gate" in p for p in sa.residual_slot_problems([{**base, "bank": {"slots_loaded": 1}}], exp_single))
    # the composer's cross-check: the flag must follow the adapters
    assert vs._residual_slot_paired(dirs) and not vs._residual_slot_paired({k: dirs[k] for k in ("one", "plain")})
    fake_ok = SimpleNamespace(lora_native=True, lora_modules=["one", "pair", "plain"], lora_adapter_dirs=dirs,
                              residual_slot_armed=True, residual_slot_paired=True)
    assert vs.expected_residual_slot_state(fake_ok)["paired"] is True
    fake_bad = SimpleNamespace(**{**vars(fake_ok), "residual_slot_paired": False})
    with pytest.raises(vs.VllmServerError, match="PAIRED payload"):
        vs.expected_residual_slot_state(fake_bad)


def test_classifier_and_the_safetensors_round_trip(tmp_path):
    for fld in nr.PAIRED_FIELDS:
        assert sa.classify_adapter_key(f"{nr.PAYLOAD_PREFIX}layers.3.{fld}") == (sa.ADAPTER_KEY_RESIDUAL, 3, fld)
    assert sa.RESIDUAL_KEY_FIELDS == nr.RESIDUAL_FIELDS
    d, L, layers = 8, 2, [0, 1]
    pos, neg = _side(d, layers, seed=1), _side(d, layers, seed=2)
    payload = _paired_payload(pos, neg, d, layers, L)
    from native_residual_fakes import write_adapter_dir
    adir = write_adapter_dir(tmp_path / "a", layers=[0], d_in=d, d_out=d, zero=True,
                             residual_block=payload.block(), residual_tensors=payload.tensors())
    back = nr.read_residual_payload(adir)
    assert back.paired and back.fingerprint == payload.fingerprint and back.block() == payload.block()
    for l in layers:
        for fld in ("gate_w", "gate_b", "direction", "gate_w2", "gate_b2", "direction2"):
            assert torch.equal(getattr(back, fld)[l], getattr(payload, fld)[l])


# ---------------------------------------------------------------------------
# the certification plumbing: TARGETS row, the cross-bank pin file, the roster view, the config
# ---------------------------------------------------------------------------


def test_certification_plumbing_names_reseeds_objects_read_only_through_a_qualified_pin():
    import sys

    import yaml
    for p in (REPO / "scripts", REPO / "scripts" / "sdfbase"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    import multilora_admission as adm
    import multilora_cert as mc

    t = mc.TARGETS["qwen35_4b_paired"]
    assert t.model_key == "qwen35_4b" and t.view == "pairedcert" and t.tt_cell == CELL
    assert t.pin_key == f"reseed_paired_pins.json:qwen35_4b.{CELL}"
    assert t.extra_cells == ("constv2-mix-full__lora_r4_ce",) and t.extra_pin_keys == ("base__constv2-mix-full__lora_r4_ce",)
    assert t.numerics == adm.NUMERICS_NOT_MEASURED and not t.measures_numerics
    assert t.config == "configs/sdfbase_full22_qwen35_4b_pairedcert.yaml"
    assert t.manifest == "configs/capeval/manifests/petri_full22_qwen35_4b_pairedcert.yaml"
    assert "qwen35_4b_paired" in mc.PSR_TARGET_KEYS
    assert mc.harness_cell_id(CELL) == HCELL
    # the pin file: the STAGED COPY of the reseed object in the vLLM lane's own bank (`paired_write`;
    # the reseed-derived path is recorded as `reseed_source`, read-only), both parents, the kind, the
    # shas — and the object's OWN identity, `derived_id`, which IS the cell id (the driver-side cell
    # validator and the view builder both hold the derived sidecar to it; a second name for one
    # object was the wrong-identity refusal the rename retired)
    pins_path = REPO / "configs/reseed_paired_pins.json"
    pins = json.loads(pins_path.read_text())
    entry = pins["qwen35_4b"][CELL]
    assert entry["object_sha256"] == "f98f0349673b7cb0cf85f54aee71f74bf9f0bbf981e989b52c8ba6599a492c2d"
    assert entry["kind"] == PAIRED and entry["seed"] == 1 and entry["positions"] == "all"
    assert entry["application_order"] == nr.PAIRED_APPLICATION_ORDER and entry["layers"] == list(range(32))
    assert entry["derived_id"] == CELL == "constv2-mix-full__psr_ce_all_minus_ctl"
    assert entry["bank"] == "paired_write" and entry["staged_id"] == f"paired_s1__{CELL}"
    assert entry["source"] == f"paired_write/qwen35_4b/cells_view/s1/{CELL}"
    assert entry["reseed_source"].startswith("sdfbase/reseed/derived/psr_paired_write/qwen35_4b_base_eb64_s1/")
    assert pins["_cells"] == [CELL]
    assert entry["parent_pos"]["object_sha256"] == "7b87767d91bd9787d22a23a80a5e78c53d16b506cc25e4d9483f74e88238fda1"
    assert entry["parent_neg"]["object_sha256"] == "f65425b851ab446b329fa93631462dd4f31fb16e11955867ef902dffd635c2cd"
    assert entry["parent_pos"]["cell_id"] == "constv2-mix-full__psr_ce_all" and entry["parent_neg"]["cell_id"] == "constv2ctl2-full__psr_ce_all"
    assert entry["identity_sidecar"] == "derived_meta.json" and entry["read_only"] is True
    llama = pins["llama8b"][CELL]
    assert llama["object_sha256"] == "3ee2adb94723ff9b6a0caada32d79d0946d449070462f7fdba07ef285eea81d5"
    assert llama["wave_pin_key"] == "derived__constv2-mix-full__psr_ce_all_minus_ctl"
    wave = json.loads((REPO / mc.PINS_REL).read_text())
    assert wave["llama8b"][llama["wave_pin_key"]]["object_sha256"] == llama["object_sha256"]
    assert llama["parent_pos"]["object_sha256"] == wave["llama8b"]["base__constv2-mix-full__psr_ce_all"]["object_sha256"]
    assert llama["parent_neg"]["object_sha256"] == wave["llama8b"]["base__constv2ctl2-full__psr_ce_all"]["object_sha256"]
    # the driver resolves the qualified key against the committed file, beside the wave's pins
    ref = mc.resolve_pin_key(t.pin_key, pins_path=REPO / mc.PINS_REL, model_key=t.model_key)
    assert ref.qualified and ref.path == pins_path and ref.sha256 == entry["object_sha256"] and ref.key == CELL
    # the roster view names the cell through the CROSS-BANK PIN FORM (no rung, no staged id) and the LoRA twin
    roster = yaml.safe_load((REPO / "configs/sdfbase_full22_roster_psrcert.yaml").read_text())
    view = roster["targets"]["qwen35_4b"]["views"]["pairedcert"]
    assert view["cells"][CELL] == {"pins": "configs/reseed_paired_pins.json", "pin": f"qwen35_4b.{CELL}"}
    assert view["cells"]["constv2-mix-full__lora_r4_ce"] == "base__constv2-mix-full__lora_r4_ce"
    assert list(view["cells"]) == [CELL, "constv2-mix-full__lora_r4_ce"] and view["config"] == t.config
    # the generated pair lists exactly the row's cells, and the manifest their harness ids
    cfg = yaml.safe_load((REPO / t.config).read_text())
    assert cfg["explicit_cells"] == ["none", CELL, "constv2-mix-full__lora_r4_ce"]
    assert cfg["defenses_dir"] == "out/full22/qwen35_4b/cells_view/pairedcert"
    spec = cm.load_petri_spec(REPO / t.manifest)
    assert list(spec.cells) == ["none", HCELL, "si_constv2_mix_full__lora_r4_ce"]
    # the production roster never learned of it
    prod = (REPO / "configs/sdfbase_full22_roster.yaml").read_text()
    assert "pairedcert" not in prod and "reseed_paired_pins" not in prod
    # the undistilled arm as a WHOLE token (its `_distilled` sibling is a production cell and shares the prefix): until
    # 2026-09-21 the production roster never named it; since then (the composed difference_relu arm joins the batch-64
    # three-seed family through derived_admission's psr_paired_write branch) it names it on the three PAIRED
    # views `s<s>p` ONLY, as a plain STAGED id `paired_eb64s<s>__…` (the lane's own staged copy of the reseed derived
    # object, pinned in sdfbase_full22_pins.json) — never the cross-bank pin form, never on any other view, never
    # through this certification file
    import re
    prod_roster = yaml.safe_load(prod)
    homes = [(vn, v["cells"][CELL]) for vn, v in prod_roster["targets"]["qwen35_4b"]["views"].items() if CELL in (v.get("cells") or {})]
    assert [vn for vn, _ in homes] == ["s1p", "s2p", "s3p"], homes
    assert all(sid == f"paired_eb64s{vn[1]}__{CELL}" for vn, sid in homes), homes          # a string, not the {pins, pin} form
    # ... and since 2026-09-23 (b64-capability-agentic-queue, the all-models one-seed waves) on the OTHER three targets' PAIRED views
    # `s1p` (`paired_of: s1`), as the plain staged ids `paired_eb8__…` (gemma4e4b, nemotron3nano4b) / `paired_eb64__…` (llama8b) — the
    # parents' effective batch, one seed; still never on a non-paired view, never the cross-bank pin form
    for key, t in prod_roster["targets"].items():
        for vn, v in t["views"].items():
            cells = v.get("cells") or {}
            if key == "qwen35_4b" and re.match(r"^s[123]p$", vn):
                continue
            if re.match(r"^s[1-9][0-9]*p$", vn) and v.get("paired_of") == vn[:-1] and key in ("gemma4e4b", "llama8b", "nemotron3nano4b"):
                assert re.fullmatch(rf"paired_eb(8|64)__{re.escape(CELL)}", str(cells.get(CELL))), (key, vn, cells.get(CELL))
                continue
            # ... and since 2026-10-10 (c7e73cd04, the worked example's roster block) on the FRONT-DOOR target `qwen35_4b_demo`, whose
            # battery wave carries the difference_relu arm as the committed pins' DERIVED object `derived__…` (one of its ten launcher
            # tokens) and whose relu dose view `d0p75_relu` carries the dosed copy `d0p75__…` — plain staged ids both, on exactly those
            # two views, never the cross-bank pin form
            if key == "qwen35_4b_demo" and vn in ("battery", "d0p75_relu"):
                expect = f"derived__{CELL}" if vn == "battery" else f"d0p75__{CELL}"
                assert cells.get(CELL) == expect, (key, vn, cells.get(CELL))
                continue
            assert CELL not in cells, (key, vn)
