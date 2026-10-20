#!/usr/bin/env python
"""Make a DOSE-SCALED copy of a fitted LoRA object: the write basis B x scale, A untouched.

The LoRA analogue of scale_psr_dose.py (directions x scale, gate untouched). Two kinds:

  lora_all          (extraction/multilayer.MultiLayerLoRAObject — the ``lora_r4_ce`` cells): per stored
                    layer the served edit is ``out += (x @ A_l) @ B_lᵀ`` (lens_adapters._adapt_lora_all ->
                    OP_LORA_OPROJ, ``(x @ lowrank_aᵀ) @ lowrank_bᵀ``), LINEAR in B, so ``B_l <- scale * B_l``
                    makes the served delta exactly ``scale`` x the parent's — for rank 1 and rank r alike.
                    A, layers, rank, positions, response_header_len ride byte-identical. This is the same
                    locus the served registry's coefficient fold uses for these kinds
                    (registry_shim._COEFF_LOCUS["lora_all"] == "lora_b": "scaling A too would fold c²").
  lora_subtraction  (extraction/multilayer.LoRASubtractionObject — the derived ``lora_r4_ce_minus_ctl`` cells):
                    both parents' factors ride in the artifact and serve as two OP_LORA_OPROJ ops per layer
                    (pos at coeff +1, neg at coeff -1), so the edit is ``Δ_pos(x) − Δ_neg(x)``; scaling BOTH
                    write bases (``B_pos``, ``B_neg``) by ``scale`` scales the difference by ``scale``. A_pos,
                    A_neg, the embedded provenance (parent identities) and every non-tensor key are unchanged.

Why a copy and not a flag: the served lens bakes the strength (``effective_scale`` is binary 0/1 at serve
time) and the registry pins ``base_scale`` at 1.0 for these kinds, so a 0.5x dose has to be DIFFERENT BYTES
with its own hash and a recorded parent — never a serve-time selector.

TWO SERVED SHAPES, ONE RULE (the dose ladder on the front door's difference LoRA). The
o_proj-only blobs (integer layer keys) are served by the LENS HOOK path (lens_adapters._adapt_lora_all /
_adapt_lora_subtraction), and the GENERAL module-set blobs (``targets`` / ``lora_modules`` / ``per_expert_rank``; factor maps
keyed ``<layer>.<module>``, routed experts as 3-D ``[E, r_e, in]`` / ``[E, out, r_e]`` tensors) are served NATIVELY
(lora_export._export_lora_subtraction / _export_multilayer_lora -> PEFT ``lora_A`` / ``lora_B`` per target, ``lora_B =
cat([B_pos, -B_neg])`` for a difference, experts concatenated along the per-expert rank axis and flattened, scaling
exactly 1). Both are LINEAR in the write basis, so the SCALING is the same key-agnostic rule (``scale_blob``: every
``B`` / ``B_pos`` / ``B_neg`` tensor x scale — every layer, every module, every expert; A, provenance and every non-tensor
key untouched), and the VERIFICATION follows the blob's own serving path: levels 2 and 3 translate a general-shape blob
through the native exporter (never through the hook adapter, which the object is never served by) and a legacy blob
through the hook adapter, byte for byte as before. The sidecar records which path ran (``path``: ``lens_hook`` /
``lora_native``).

VERIFICATION (recorded in the sidecar's ``dose_scaled_from.verification`` block; any failure refuses):
  1. tensor-level: every scaled tensor equals scale x the parent's bit-exactly (torch.equal); every other
     tensor and non-tensor key is unchanged;
  2. served-op level: the WAVE'S OWN adapter (sisteer.capeval.serving.lens_adapters) translates the parent
     bytes and the copy's bytes into HookOps; per op ``lowrank_a`` / coeff / layer / positions are identical
     and ``lowrank_b`` is exactly scale x the parent's;
  3. served-activation level: on ONE fixed probe batch (seed 0, ``--probe-rows`` rows of N(0,1) o_proj inputs
     per layer) the summed op delta of the copy equals scale x the parent's — exactly in fp32 (a power-of-two
     scale commutes with IEEE rounding), and after the bf16 cast the served hook applies; the max abs
     difference is recorded. This is the guard against a transposed factor or a wrong-factor slip: a wrongly
     scaled LoRA would look plausible and produce silently wrong numbers.

Output dir (a bank-stageable cell dir, sdfbase full22 layout): object.pt, the parent's identity sidecar
(train_budget_meta.json for a fitted cell, derived_meta.json for the LoRA subtraction) re-bound to the new
bytes + ``dose_scaled_from``, object.pt.provenance.json when the parent has one, staged_from.json.

THE 1.0 ANCHOR (``--anchor --scale 1``; the strength frontier). The ladder's 1.0 rung is the
parent RE-SERVED in the same pass as its neighbours. Where the parent is already staged under the campaign's
own id, the roster maps the anchor view to that id. Where it is NOT — a parent consumed from its owning
lane's bank before that lane staged it (the Llama ladder) — the anchor is a BYTE-IDENTICAL copy of the parent's
object.pt (``shutil.copyfile``; the sha is asserted EQUAL, never re-serialised) staged under the ladder's own
``d1__`` / ``d1it__`` id, so the frontier never borrows an id another lane may still be staging. The same
three-level verification runs at scale 1 (every tensor equal, every served op identical, the served
activation identical on the probe batch) and is recorded like any rung's; ``dose_scaled_from.anchor`` is true,
``tensors_scaled`` is empty, and ``scaling_method`` says no tensor was touched.

PARENT PROVENANCE (``--parent-origin``, ``--parent-state``): when ``--parent`` is a pulled copy, ``--parent-origin``
names the authoritative path the bytes were read from, and ``--parent-state`` records what state that parent was
in when consumed (e.g. "final bytes; not yet staged by its producer"). Both ride in ``dose_scaled_from`` and
``staged_from.json`` beside the verified parent sha, so a reader can tell WHICH bytes of WHICH parent this copy
descends from without inferring anything about that parent's later staging.

  python scripts/sdfbase/scale_lora_dose.py --parent <cell dir> --scale 0.5 --out <cell dir> \
      --target qwen35_4b --staged-id halfdose__constv2-mix-full__lora_r4_ce \
      --parent-staged-id base__constv2-mix-full__lora_r4_ce
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _p in (str(REPO / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dose_scale_common import (BUDGET_META, DERIVED_META, check_parent_sidecar,  # noqa: E402
                               copy_provenance_json, fresh_out_dir, identity_of, now_utc,
                               producer_commit, rebind_identity, sha256_file, tensor_stats,
                               write_staged_from)

PRODUCER = "scripts/sdfbase/scale_lora_dose.py"
SCALED_KEYS = {"lora_all": ("B",), "lora_subtraction": ("B_pos", "B_neg")}
UNCHANGED_TENSOR_KEYS = {"lora_all": ("A",), "lora_subtraction": ("A_pos", "A_neg")}
LORA_ARMS = ("lora_r4_ce", "lora_r4", "lora_all", "lora_r1", "lora_r16")


def _lora_arm_ok(arm: str | None) -> bool:
    try:
        from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS
        if arm in ARM_OBJECT_KINDS:
            return ARM_OBJECT_KINDS[arm] in ("lora_all", "lora")
    except Exception:  # noqa: BLE001
        pass
    return arm in LORA_ARMS or (isinstance(arm, str) and arm.startswith("lora"))


def scale_blob(blob: dict, scale: float) -> tuple[dict, dict]:
    """The scaled blob (SAME key order as the parent's) and per-layer norm records."""
    kind = blob.get("kind")
    if kind not in SCALED_KEYS:
        raise SystemExit(f"parent blob kind {kind!r} is not a LoRA kind this producer scales ({sorted(SCALED_KEYS)})")
    norms: dict = {}
    new = {}
    for k, v in blob.items():
        if k in SCALED_KEYS[kind]:
            if not isinstance(v, dict) or not v:
                raise SystemExit(f"{kind}: {k!r} is not a non-empty layer map")
            scaled = {}
            norms[k] = {}
            for layer, t in v.items():
                t32 = t.detach().float().clone()
                s = t32 * float(scale)
                scaled[layer] = s
                norms[k][str(layer)] = {"before": float(t32.norm()), "after": float(s.norm())}
            new[k] = scaled
        else:
            new[k] = v
    return new, norms


def _adapt(kind: str, blob_bytes: bytes):
    from sisteer.capeval.serving import lens_adapters as la
    fn = {"lora_all": la._adapt_lora_all, "lora_subtraction": la._adapt_lora_subtraction}[kind]
    return fn({}, blob_bytes)


def _blob_bytes(blob: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def verify(parent_blob: dict, new_blob: dict, scale: float, *, probe_rows: int, seed: int) -> dict:
    """The three-level check (module docstring). Returns the verification record; raises on failure."""
    kind = parent_blob["kind"]
    rec: dict = {"scale": float(scale), "kind": kind, "checks": []}

    # 1. tensor-level
    for k in SCALED_KEYS[kind]:
        for layer in parent_blob[k]:
            want = parent_blob[k][layer].detach().float() * float(scale)
            if not torch.equal(new_blob[k][layer], want):
                raise SystemExit(f"tensor check failed: {k}[{layer}] != {scale} x parent")
    for k in UNCHANGED_TENSOR_KEYS[kind]:
        for layer in parent_blob[k]:
            if not torch.equal(new_blob[k][layer], parent_blob[k][layer]):
                raise SystemExit(f"tensor check failed: {k}[{layer}] changed")
    for k in parent_blob:
        if k in SCALED_KEYS[kind] or k in UNCHANGED_TENSOR_KEYS[kind]:
            continue
        if new_blob[k] != parent_blob[k]:
            raise SystemExit(f"non-tensor key {k!r} changed")
    rec["checks"].append({"check": "tensors", "status": "PASS",
                          "scaled": list(SCALED_KEYS[kind]), "unchanged": list(UNCHANGED_TENSOR_KEYS[kind]),
                          "rule": "torch.equal(new, parent * scale) per scaled tensor; torch.equal(new, parent) otherwise"})
    _verify_served(parent_blob, new_blob, scale, probe_rows=probe_rows, seed=seed, rec=rec)
    rec["status"] = "PASS"
    return rec


def verify_anchor(parent_blob: dict, new_blob: dict, *, probe_rows: int, seed: int) -> dict:
    """The three-level check for the 1.0 ANCHOR copy: level 1 is dtype-exact equality of EVERY tensor and
    every non-tensor key (the copy is the parent's bytes, so nothing may differ, not even a dtype); levels 2
    and 3 are the served-op and served-activation checks at scale 1 through the same code every rung uses."""
    kind = parent_blob["kind"]
    if kind not in SCALED_KEYS:
        raise SystemExit(f"parent blob kind {kind!r} is not a LoRA kind this producer anchors ({sorted(SCALED_KEYS)})")
    rec: dict = {"scale": 1.0, "kind": kind, "anchor": True, "checks": []}
    if set(new_blob) != set(parent_blob):
        raise SystemExit(f"anchor check failed: key sets differ {sorted(set(new_blob) ^ set(parent_blob))}")
    for k in parent_blob:
        pv, nv = parent_blob[k], new_blob[k]
        if isinstance(pv, dict) and pv and all(isinstance(t, torch.Tensor) for t in pv.values()):
            if set(pv) != set(nv):
                raise SystemExit(f"anchor check failed: {k} layer sets differ")
            for layer, t in pv.items():
                if nv[layer].dtype != t.dtype or not torch.equal(nv[layer], t):
                    raise SystemExit(f"anchor check failed: {k}[{layer}] differs from the parent (dtype or values)")
        elif nv != pv:
            raise SystemExit(f"anchor check failed: non-tensor key {k!r} differs")
    rec["checks"].append({"check": "tensors", "status": "PASS", "scaled": [], "unchanged": sorted(parent_blob),
                          "rule": "torch.equal(new, parent) AND equal dtype for every tensor; == for every other key"})
    _verify_served(parent_blob, new_blob, 1.0, probe_rows=probe_rows, seed=seed, rec=rec)
    rec["status"] = "PASS"
    return rec


#: the decoder prefix the native verification exports under (naming only: no skeleton is passed, so no site is held to a
#: tree; parent and copy export under the SAME prefix and are compared tensor by tensor)
NATIVE_DECODER_PREFIX = "model.layers"
#: relative tolerance of the native level-2 check when fp32 rounding keeps ``lora_B`` from being bitwise ``scale x parent``
#: (it IS bitwise for every scale: negation, cat, reshape and permute are exact; the tolerance is the record's ceiling)
NATIVE_REL_TOL = 1e-6


def _general_shape_of(blob: dict):
    """The blob's GENERAL module-set shape (lora_export._general_shape: the lora_native structure — ``targets`` /
    ``lora_modules`` / ``per_expert_rank``), validated by the exporter's own rules; None for the legacy o_proj hook shape."""
    from sisteer.capeval.serving import lora_export as LX
    kind = blob["kind"]
    maps = ("A", "B") if kind == "lora_all" else ("A_pos", "B_pos", "A_neg", "B_neg")
    return LX._general_shape(kind, blob, factor_maps=maps)


def _export_native(kind: str, blob: dict):
    """The wave's OWN native translator for the kind: the PEFT tensors + adapter config the served standup writes."""
    from sisteer.capeval.serving import lora_export as LX
    fn = {"lora_all": LX._export_multilayer_lora, "lora_subtraction": LX._export_lora_subtraction}[kind]
    return fn({}, blob, decoder_prefix=NATIVE_DECODER_PREFIX)


def _native_tensor_names(kind: str, shape) -> dict:
    """``{(layer, module): (lora_A name, lora_B name, meta)}`` exactly as lora_export._export_general names the exported
    tensors (linear targets at their site; routed-expert halves under the experts module path + PEFT suffix)."""
    from sisteer.capeval.serving import lora_export as LX
    out = {}
    for layer, module in shape.keys:
        meta = shape.targets[module]
        if meta["kind"] == "linear":
            base = f"base_model.model.{NATIVE_DECODER_PREFIX}.{layer}.{meta['site']}"
        else:
            path, param = LX._experts_module_path(kind, module, meta["site"])
            base = f"base_model.model.{NATIVE_DECODER_PREFIX}.{layer}.{path}{LX.EXPERTS_PEFT_SUFFIX[param]}"
        out[(int(layer), str(module))] = (f"{base}.lora_A.weight", f"{base}.lora_B.weight", meta)
    return out


def _verify_served_native(parent_blob: dict, new_blob: dict, scale: float, shape, *, probe_rows: int, seed: int, rec: dict) -> None:
    """Levels 2 and 3 for a GENERAL module-set blob, through the native exporter (module docstring TWO SERVED SHAPES).
    Level 2: parent and copy -> PEFT tensors + config; the configs equal; every ``lora_A.weight`` torch.equal; every
    ``lora_B.weight`` == scale x the parent's (bitwise, else within NATIVE_REL_TOL relative — recorded either way), the routed
    experts' flattened halves and any zero companion included. Level 3: per exported target the served delta on one fixed probe
    batch — linear ``(x @ lora_A^T) @ lora_B^T``; experts per expert through lora_export.experts_unflatten (vLLM's own reshape) —
    copy == scale x parent within 1e-6 relative, exactness recorded."""
    from sisteer.capeval.serving import lora_export as LX
    kind = parent_blob["kind"]
    translator = f"sisteer.capeval.serving.lora_export.{_export_native_name(kind)}"
    tf, cf = _export_native(kind, parent_blob)
    th, ch = _export_native(kind, new_blob)
    if cf != ch:
        diff = sorted(k for k in set(cf) | set(ch) if cf.get(k) != ch.get(k))
        raise SystemExit(f"served-op check failed (lora_native): the exported adapter configs differ on {diff}")
    if set(tf) != set(th):
        raise SystemExit(f"served-op check failed (lora_native): exported tensor sets differ {sorted(set(tf) ^ set(th))}")
    n_a = n_b = 0
    exact_b = True
    max_rel = 0.0
    for name in sorted(tf):
        pf, ph = tf[name], th[name]
        if pf.shape != ph.shape or pf.dtype != ph.dtype:
            raise SystemExit(f"served-op check failed (lora_native): {name} shape/dtype {tuple(ph.shape)}/{ph.dtype} != parent's {tuple(pf.shape)}/{pf.dtype}")
        if name.endswith(".lora_A.weight"):
            if not torch.equal(ph, pf):
                raise SystemExit(f"served-op check failed (lora_native): lora_A differs at {name}")
            n_a += 1
        elif name.endswith(".lora_B.weight"):
            want = pf * float(scale)
            if not torch.equal(ph, want):
                exact_b = False
                rel = ((ph - want).abs().max() / max(want.abs().max().item(), 1e-30)).item()
                if rel > NATIVE_REL_TOL:
                    raise SystemExit(f"served-op check failed (lora_native): lora_B != {scale} x parent at {name} (max rel err {rel})")
                max_rel = max(max_rel, rel)
            n_b += 1
        else:
            raise SystemExit(f"served-op check failed (lora_native): unexpected exported tensor {name}")
    names = _native_tensor_names(kind, shape)
    expected = {n for a, b, _ in names.values() for n in (a, b)}
    companions = sorted(set(tf) - expected)          # zero pairs for a missing experts half — level 2 covered them
    if expected - set(tf):
        raise SystemExit(f"served-op check failed (lora_native): the exporter wrote no tensor for {sorted(expected - set(tf))[:4]}")
    n_experts = sorted({int(m["num_experts"]) for m in shape.targets.values() if m["kind"] == "experts"})
    rec["checks"].append({"check": "served_ops", "status": "PASS", "path": "lora_native", "adapter": translator,
                          "n_tensors": len(tf), "n_lora_a": n_a, "n_lora_b": n_b, "n_companion_tensors": len(companions),
                          "exact_fp32": bool(exact_b), "max_rel_err": max_rel,
                          "shape": {"modules": list(shape.modules), "set_id": shape.set_id,
                                    "per_expert_rank": dict(shape.per_expert_rank) if shape.per_expert_rank is not None else None,
                                    "layers": len(shape.layers), "targets": len(shape.keys), "routed_experts": n_experts},
                          "rule": "parent and copy exported through the native translator under one prefix: configs equal; every "
                                  "lora_A.weight torch.equal; every lora_B.weight == parent * scale (bitwise, else <= 1e-6 relative, "
                                  "recorded), the routed experts' flattened halves included"})

    # 3. served-activation level, per exported target, on one fixed probe batch
    g = torch.Generator().manual_seed(int(seed))
    max_abs = 0.0
    max_abs_bf16 = 0.0
    exact_fp32 = True
    exact_bf16 = True
    ref_norm = 0.0
    n_linear = n_expert_targets = 0
    for (layer, module), (a_name, b_name, meta) in sorted(names.items()):
        A_f, B_f, A_h, B_h = tf[a_name], tf[b_name], th[a_name], th[b_name]
        if meta["kind"] == "linear":
            x = torch.randn(int(probe_rows), int(A_f.shape[1]), generator=g, dtype=torch.float32)
            d_full = (x @ A_f.t()) @ B_f.t()
            d_half = (x @ A_h.t()) @ B_h.t()
            n_linear += 1
        else:
            E = int(meta["num_experts"])
            A3f, B3f = LX.experts_unflatten(A_f, B_f, E)
            A3h, B3h = LX.experts_unflatten(A_h, B_h, E)
            x = torch.randn(int(probe_rows), int(A3f.shape[2]), generator=g, dtype=torch.float32)
            d_full = torch.einsum("enr,eor->eno", torch.einsum("ni,eri->enr", x, A3f), B3f)
            d_half = torch.einsum("enr,eor->eno", torch.einsum("ni,eri->enr", x, A3h), B3h)
            n_expert_targets += 1
        want = d_full * float(scale)
        diff = (d_half - want).abs().max().item()
        max_abs = max(max_abs, diff)
        ref_norm = max(ref_norm, want.abs().max().item())
        exact_fp32 &= torch.equal(d_half, want)
        diff16 = (d_half.to(torch.bfloat16).float() - want.to(torch.bfloat16).float()).abs().max().item()
        max_abs_bf16 = max(max_abs_bf16, diff16)
        exact_bf16 &= torch.equal(d_half.to(torch.bfloat16), want.to(torch.bfloat16))
    tol = 1e-6 * max(ref_norm, 1e-30)
    if max_abs > tol:
        raise SystemExit(f"served-activation check failed (lora_native): max |delta_scaled - scale*delta_parent| = {max_abs} "
                         f"(tolerance {tol}) on the fixed probe batch")
    rec["checks"].append({"check": "served_activation", "status": "PASS", "path": "lora_native", "probe_rows": int(probe_rows),
                          "seed": int(seed), "layers": len(shape.layers), "targets_linear": n_linear, "targets_experts": n_expert_targets,
                          "max_abs_diff_fp32": max_abs, "max_rel_diff_fp32": (max_abs / max(ref_norm, 1e-30)),
                          "exact_fp32": bool(exact_fp32), "max_abs_diff_bf16": max_abs_bf16, "exact_bf16": bool(exact_bf16),
                          "reference_max_abs": ref_norm,
                          "rule": "per exported target: linear (x @ lora_A^T) @ lora_B^T; routed experts per expert through "
                                  "lora_export.experts_unflatten (vLLM's reshape) on x ~ N(0,1): copy == scale * parent "
                                  "(fp32 within 1e-6 relative; exactness recorded)"})


def _export_native_name(kind: str) -> str:
    return {"lora_all": "_export_multilayer_lora", "lora_subtraction": "_export_lora_subtraction"}[kind]


def _verify_served(parent_blob: dict, new_blob: dict, scale: float, *, probe_rows: int, seed: int, rec: dict) -> None:
    """Levels 2 and 3 of the module docstring, appended to ``rec['checks']``; raises on failure. The blob's OWN serving path
    decides the translator (TWO SERVED SHAPES): a general module-set blob goes through the native exporter, a legacy o_proj
    blob through the lens hook adapter."""
    kind = parent_blob["kind"]
    shape = _general_shape_of(parent_blob)
    if shape is not None:
        _verify_served_native(parent_blob, new_blob, scale, shape, probe_rows=probe_rows, seed=seed, rec=rec)
        return
    # 2. served-op level, through the wave's adapter
    positions = parent_blob.get("positions", "all")
    if positions != "all":
        rec["checks"].append({"check": "served_ops", "status": "SKIPPED", "path": "lens_hook",
                              "reason": f"positions={positions!r}: the adapter needs the served header ids; the tensor "
                                        "and activation checks below cover the arithmetic"})
        ops_full = ops_half = None
    else:
        ops_full = _adapt(kind, _blob_bytes(parent_blob))
        ops_half = _adapt(kind, _blob_bytes(new_blob))
        if len(ops_full) != len(ops_half):
            raise SystemExit(f"served-op check failed: {len(ops_full)} vs {len(ops_half)} ops")
        for of, oh in zip(ops_full, ops_half):
            if (of.op, of.layer, of.coeff, of.positions, of.skip_bos, of.phase, of.header_ids) != \
                    (oh.op, oh.layer, oh.coeff, oh.positions, oh.skip_bos, oh.phase, oh.header_ids):
                raise SystemExit(f"served-op check failed: op identity differs at layer {of.layer}")
            if not torch.equal(of.lowrank_a, oh.lowrank_a):
                raise SystemExit(f"served-op check failed: lowrank_a differs at layer {of.layer}")
            if not torch.equal(oh.lowrank_b, of.lowrank_b * float(scale)):
                raise SystemExit(f"served-op check failed: lowrank_b != {scale} x parent at layer {of.layer}")
        rec["checks"].append({"check": "served_ops", "status": "PASS", "path": "lens_hook", "n_ops": len(ops_full),
                              "adapter": f"sisteer.capeval.serving.lens_adapters._adapt_{kind}",
                              "rule": "per op: lowrank_a / coeff / layer / positions identical; lowrank_b == parent * scale (torch.equal)"})

    # 3. served-activation level on one fixed probe batch
    g = torch.Generator().manual_seed(int(seed))
    if ops_full is None:   # build the op arithmetic from the tensors directly (same formula)
        from sisteer.capeval.serving import lens_adapters as la
        pairs_f = [(int(l), parent_blob[a][l], parent_blob[b][l]) for a, b in
                   (("A", "B"),) if kind == "lora_all" for l in parent_blob["A"]]
        if kind == "lora_subtraction":
            pairs_f = [(int(l), parent_blob["A_pos"][l], parent_blob["B_pos"][l]) for l in parent_blob["A_pos"]]
        ops_full = la._lora_pair_ops(kind, pairs_f, la.POSITIONS_ALL, None)
        pairs_h = [(l, new_blob["A" if kind == "lora_all" else "A_pos"][str(l)],
                    new_blob["B" if kind == "lora_all" else "B_pos"][str(l)]) for l, _, _ in pairs_f]
        ops_half = la._lora_pair_ops(kind, pairs_h, la.POSITIONS_ALL, None)
    layers = sorted({op.layer for op in ops_full})
    max_abs = 0.0
    max_abs_bf16 = 0.0
    exact_fp32 = True
    exact_bf16 = True
    ref_norm = 0.0
    for layer in layers:
        lf = [op for op in ops_full if op.layer == layer]
        lh = [op for op in ops_half if op.layer == layer]
        d_in = lf[0].lowrank_a.shape[1]
        x = torch.randn(int(probe_rows), d_in, generator=g, dtype=torch.float32)
        d_full = sum(op.coeff * ((x @ op.lowrank_a.t()) @ op.lowrank_b.t()) for op in lf)
        d_half = sum(op.coeff * ((x @ op.lowrank_a.t()) @ op.lowrank_b.t()) for op in lh)
        want = d_full * float(scale)
        diff = (d_half - want).abs().max().item()
        max_abs = max(max_abs, diff)
        ref_norm = max(ref_norm, want.abs().max().item())
        exact_fp32 &= torch.equal(d_half, want)
        diff16 = (d_half.to(torch.bfloat16).float() - want.to(torch.bfloat16).float()).abs().max().item()
        max_abs_bf16 = max(max_abs_bf16, diff16)
        exact_bf16 &= torch.equal(d_half.to(torch.bfloat16), want.to(torch.bfloat16))
    tol = 1e-6 * max(ref_norm, 1e-30)
    if max_abs > tol:
        raise SystemExit(f"served-activation check failed: max |delta_scaled - scale*delta_parent| = {max_abs} "
                         f"(tolerance {tol}) on the fixed probe batch")
    rec["checks"].append({"check": "served_activation", "status": "PASS", "path": "lens_hook", "probe_rows": int(probe_rows),
                          "seed": int(seed), "layers": len(layers), "max_abs_diff_fp32": max_abs,
                          "exact_fp32": bool(exact_fp32), "max_abs_diff_bf16": max_abs_bf16,
                          "exact_bf16": bool(exact_bf16), "reference_max_abs": ref_norm,
                          "rule": "sum over the layer's ops of coeff * (x @ lowrank_a^T) @ lowrank_b^T on x ~ N(0,1): "
                                  "copy == scale * parent (fp32 within 1e-6 relative; exactness recorded)"})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--parent", required=True, type=Path, help="the fitted / derived cell dir (object.pt + identity sidecar)")
    ap.add_argument("--scale", required=True, type=float)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target", required=True)
    ap.add_argument("--staged-id", required=True)
    ap.add_argument("--parent-staged-id", default=None)
    ap.add_argument("--probe-rows", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor", action="store_true",
                    help="the ladder's 1.0 rung as a BYTE-IDENTICAL re-serve copy of the parent (requires --scale 1; "
                         "module docstring THE 1.0 ANCHOR)")
    ap.add_argument("--parent-origin", default=None,
                    help="the authoritative path the parent bytes were read from, when --parent is a pulled copy")
    ap.add_argument("--parent-state", default=None,
                    help="what state the owning lane's parent was in when consumed (recorded verbatim in the provenance)")
    args = ap.parse_args(argv)
    # A NEGATIVE scale is admitted (the strength frontier, 2026-09-10): the served edit is linear in the
    # scaled tensor, so scale x (-1) is exactly the REVERSED edit — steering against the fitted direction —
    # and the same three-level verification covers it (torch.equal(new, parent * scale) holds for any
    # finite scale). Refused: 0 (no object; the `none` anchor is that cell) and 1.0 (the parent itself) —
    # EXCEPT under --anchor, where 1.0 is the only admitted scale and the copy is the parent's bytes.
    if args.anchor:
        if args.scale != 1.0:
            raise SystemExit(f"--anchor is the 1.0 rung only (got --scale {args.scale}); a scaled copy never anchors")
    elif not (args.scale == args.scale) or args.scale in (0.0, 1.0):
        raise SystemExit(f"--scale must be a finite non-zero number != 1.0 (got {args.scale}); "
                         "a 1.0 copy is the parent (pass --anchor to stage it as the re-served rung) and a 0 copy is the `none` cell")

    p_obj = args.parent / "object.pt"
    if not p_obj.is_file():
        raise SystemExit(f"parent lacks {p_obj}")
    identity, sidecar = identity_of(args.parent)
    parent_sha = sha256_file(p_obj)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    check_parent_sidecar(meta, parent_sha, sidecar)
    blob = torch.load(p_obj, map_location="cpu", weights_only=True)
    kind = blob.get("kind")
    if identity == "staged":
        if not _lora_arm_ok(meta.get("arm")):
            raise SystemExit(f"parent arm {meta.get('arm')!r} is not a LoRA arm")
        if kind != "lora_all":
            raise SystemExit(f"a fitted LoRA cell must store kind 'lora_all', got {kind!r}")
        cell_id = meta.get("cell_id")
    else:
        if kind != "lora_subtraction" or meta.get("kind") != kind:
            raise SystemExit(f"derived parent kind {kind!r} / sidecar {meta.get('kind')!r} — this producer scales "
                             "the derived 'lora_subtraction' only")
        cell_id = meta.get("derived_id")
    if cell_id != args.staged_id.split("__", 1)[1]:
        raise SystemExit(f"--staged-id {args.staged_id} does not end in the parent's cell id {cell_id!r}")

    fresh_out_dir(args.out)
    o_obj = args.out / "object.pt"
    if args.anchor:
        import shutil
        shutil.copyfile(p_obj, o_obj)                     # the parent's BYTES, never a re-serialisation
        new_sha = sha256_file(o_obj)
        if new_sha != parent_sha:
            raise SystemExit(f"anchor copy hashes to {new_sha}, parent to {parent_sha} — the bytes did not copy intact")
        rt = torch.load(o_obj, map_location="cpu", weights_only=True)
        verification = verify_anchor(blob, rt, probe_rows=args.probe_rows, seed=args.seed)
        norms = {k: {str(layer): {"before": float(t.detach().float().norm()), "after": float(t.detach().float().norm())}
                     for layer, t in blob[k].items()} for k in SCALED_KEYS[kind]}
    else:
        new, norms = scale_blob(blob, args.scale)
        torch.save(new, o_obj)
        rt = torch.load(o_obj, map_location="cpu", weights_only=True)
        verification = verify(blob, rt, args.scale, probe_rows=args.probe_rows, seed=args.seed)
        new_sha = sha256_file(o_obj)
        assert new_sha != parent_sha

    block = {
        "parent_object_sha256": parent_sha,
        "parent_dir": str(args.parent),
        "parent_origin": args.parent_origin or str(args.parent),
        "parent_consumed_state": args.parent_state,
        "parent_sidecar_object_sha256_matches_bytes": True,   # check_parent_sidecar refused otherwise
        "parent_sha256_verified_at": now_utc(),
        "parent_staged_id": args.parent_staged_id,
        "parent_identity": identity,
        "parent_fit_fingerprint": meta.get("fit_fingerprint"),
        "object_type": kind,
        "scale": float(args.scale),
        "anchor": bool(args.anchor),
        "scaling_method": ("NONE: byte-identical copy of the parent object.pt (sha256 equal); the ladder's re-served "
                           "1.0 anchor, staged under its own id" if args.anchor else
                           "write basis B x scale per layer; A untouched" if kind == "lora_all" else
                           "both parents' write bases B_pos, B_neg x scale per layer; A_pos, A_neg and the "
                           "embedded parent provenance untouched"),
        "tensors_scaled": [] if args.anchor else list(SCALED_KEYS[kind]),
        "tensors_unchanged": (list(SCALED_KEYS[kind]) if args.anchor else [])
                             + list(UNCHANGED_TENSOR_KEYS[kind]) + ["layers", "rank", "positions", "response_header_len"]
                             + (["provenance"] if "provenance" in blob else []),
        "norms_by_tensor_and_layer": norms,
        "verification": verification,
        "producer": PRODUCER,
        "producer_commit": producer_commit(Path(__file__)),
        "written_at": now_utc(),
        "note": (("the ladder's 1.0 ANCHOR: the parent object's own bytes re-served under this lane's staged id — the "
                  "record above is the PARENT's; nothing was scaled (strength-frontier D64)") if args.anchor else
                 ("a post-hoc dose-scaled copy of the parent object — the record above is the PARENT's (this object "
                  "was not trained or derived separately); the served o_proj edit is linear in the write basis, so "
                  "the copy's served delta is exactly scale x the parent's (lens_adapters OP_LORA_OPROJ)")),
    }
    meta_out = rebind_identity(meta, new_sha, block)
    (args.out / (BUDGET_META if identity == "staged" else DERIVED_META)).write_text(
        json.dumps(meta_out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    copy_provenance_json(args.parent, args.out, scale=args.scale, parent_sha=parent_sha,
                         fallback={"kind": kind, "layers": [int(l) for l in blob["layers"]]})
    write_staged_from(args.out, target=args.target, staged_id=args.staged_id, cell_id=cell_id, new_sha=new_sha,
                      parent_staged_id=args.parent_staged_id, parent=args.parent, parent_sha=parent_sha,
                      scale=args.scale, producer=PRODUCER, anchor=bool(args.anchor),
                      parent_origin=args.parent_origin, parent_state=args.parent_state)
    first = SCALED_KEYS[kind][0]
    print(json.dumps({"parent_object_sha256": parent_sha, "object_sha256": new_sha, "scale": args.scale, "kind": kind,
                      "anchor": bool(args.anchor),
                      "identity": identity, "layers": len(blob["layers"]), "rank": blob.get("rank"),
                      "norm_before_mean": sum(v["before"] for v in norms[first].values()) / len(norms[first]),
                      "norm_after_mean": sum(v["after"] for v in norms[first].values()) / len(norms[first]),
                      "verification": {c["check"]: c["status"] for c in verification["checks"]},
                      "max_abs_diff_fp32": verification["checks"][-1]["max_abs_diff_fp32"],
                      "out": str(args.out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
