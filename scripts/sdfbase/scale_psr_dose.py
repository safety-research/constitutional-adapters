#!/usr/bin/env python
"""Make a DOSE-SCALED copy of a fitted PSR object: directions x scale, gate untouched.

Why a copy and not a flag. The served lens bakes the steering strength into the
object (``effective_scale`` is binary 0/1 at serve time), so a served pass at any
strength other than 1.0 has to serve DIFFERENT BYTES. The campaign's 2x pass ran
through the HF path (``probe_score.py --scale 2.0`` -> ``obj.apply(model,
scale=2.0)`` -> the hook ``h + scale * g(h) * d``, exactly linear in ``scale`` at
the hooked module; tests/test_scale_semantics.py), which the served path cannot
take. This producer writes the file-level equivalent: ``d_l <- scale * d_l`` for
every stored layer, ``gate_w`` / ``gate_b`` / ``positions`` / ``response_header_len``
/ ``train_direction`` byte-identical, so the served op
``eff * relu(h @ gate_w + gate_b) * vector`` computes ``scale * g(h) * d`` up to
floating-point rounding order.

Why the DIRECTION and nothing else. A gated PSR has a gauge freedom:
``(w, b, d) -> (c w, c b, d / c)`` is a no-op (relu is positively homogeneous), so
the per-token add ``g(h) * d`` is what "dose" means; scaling the gate instead of
the direction changes the gate's FIRING RATE (which tokens are steered), not the
dose. ``||d||`` (equivalently ``||d|| * ||w||`` with the gate fixed) is the dose,
and this script moves exactly that.

TWO PARENT IDENTITIES (the halfdose extension, 2026-09-08). A fitted PSR cell carries
train_budget_meta.json (arm psr_ce_all / psr_kl_all / psr_mse_all); the DERIVED distilled
PSR residual (``constv2-mix-full__psr_ce_all_minus_ctl_distilled``) carries derived_meta.json
and stores kind ``psr`` with an embedded parent provenance. Both are the same served shape
(lens_adapters._adapt_psr) and take the same scaling; the copy keeps the parent's identity
contract, re-bound to the new bytes (dose_scale_common).

TWO KINDS (the paired-write extension, 2026-09-17). Besides ``psr`` (one gate) the producer
scales ``psr_paired_write`` (extraction/paired_write.PSRPairedWriteObject): the coherent virtual
teacher ``V(z) = u_pos(z) - u_neg(z) = relu(w_pos.z + b_pos) d_pos - relu(w_neg.z + b_neg) d_neg``,
both parents' gates stored bit-exact under the blob's ``pos`` / ``neg`` sides and applied by ONE
combined hook on the SHARED pre-write state z. Its dose is the common factor on BOTH sides'
directions: ``c V = c u_pos - c u_neg`` (each u is linear in its own d, and the gauge argument
above holds side by side), so the copy is ``pos.directions x scale`` AND ``neg.directions x scale``
- the SAME scale - with both gates, ``layers``, ``application_order``, ``positions``,
``response_header_len`` and the embedded ``provenance`` byte-identical. Scaling ONE side would be
``c u_pos - u_neg``: a different DIFFERENCE, i.e. a different intervention (at c -> 0 a lone
parent), not a dose of V. So there is deliberately no per-side entry point in this file:
:data:`PAIRED_SIDES` is iterated in one loop by the scaler (:func:`scale_paired_blob`) and the
verifier (:func:`verify_paired`), the scale is one float bound once, and the verifier refuses a
copy whose two sides do not carry the same factor. A paired parent is always a DERIVED object
(derived_meta.json with kind psr_paired_write and parent_pos / parent_neg blocks; the builder is
scripts/sdfbase/psr_residual_distill.py compose-v); a train_budget_meta.json identity is refused
for it. Any other kind - a LoRA, the non-persisted ``psr_shared_state_write`` construction -
still refuses cleanly before anything is written.

VERIFICATION (recorded in ``dose_scaled_from.verification``; any failure refuses): the directions
equal scale x the parent's bit-exactly and every other tensor is unchanged; the wave's adapter
translates both artifacts and every op's ``vector`` is scale x the parent's with gate_w / gate_b
identical (and every op's coeff exactly 1.0); on one fixed probe batch of hidden states the summed op
edit ``relu(h·w+b)·d`` of the copy equals scale x the parent's (exact in fp32 for a power-of-two
scale; max abs diff recorded). For the PAIRED kind the same three levels run at the paired shape:
tensors - both sides' directions exactly scale x the parent's, both gates and every non-side key
unchanged; served ops - through the REGISTERED paired adapter (lens_adapters._adapt_psr_paired_write,
kind psr_paired_write: ONE ``relu_gated_vector_pair`` op per layer carrying BOTH sides - ``vector`` /
``gate_w`` / ``gate_b`` the pos side, ``vector2`` / ``gate_w2`` / ``gate_b2`` the neg side; the single-
gate _adapt_psr check lifted to two sides, :func:`verify_paired_served_ops`): both artifacts are
translated and, per op, the identity (op, layer, positions, skip_bos, phase, header ids) is equal,
``coeff`` is exactly 1.0 on parent AND copy, BOTH gates are byte-identical (torch.equal), and BOTH
payload vectors equal the parent's x the ONE scale (torch.equal; the max abs diff recorded per side) -
the one-factor invariant surfaced at the served op, so a one-sided scale is refused at the op level as
well as at the tensor level, and an op of any other shape refuses (the check never guesses a payload
layout). On a tree with NO adapter registered for the kind the level is SKIPPED with the reason
recorded - the producer still runs on a pre-adapter checkout - never dropped silently; served
activation - the PAIRED write on ONE shared probe batch h per layer (SharedStateWriteObject.
paired_write's arithmetic), copy == scale x parent. Each SIDE additionally gets the full single-gate verification
(tensors, served ops through _adapt_psr, activation) on the kind-``psr`` view
PSRPairedWriteObject._parent_blob extracts, recorded under ``verification.per_side``.

FINGERPRINTS AND THE DOSE LOCUS (both kinds). Two identities are bound here. The family's CARRIER
identity is ``object_sha256`` - the file's sha256 (dose_scale_common.sha256_file; what
derived_object_problems, staged_from.json and the served registry bind) - which covers both sides of a
paired blob because both ride in the one torch.save payload. The served PAYLOAD identity is
``residual_fingerprint`` (native_residual.residual_fingerprint; attested per cell under
``residual_slot.cells.<cell>.residual_fingerprint``): sha256 over the canonical ``petri_residual_slot``
block PLUS every payload tensor ``sisteer_residual.layers.<l>.{gate_w, gate_b, direction}`` (key,
dtype, shape, raw bytes, sorted keys). It is NOT ``fingerprint_bf16``: every residual cell on one base
rides the SAME rank-1 zero LoRA carrier, so ``fingerprint_bf16`` is identical across PSR payloads by
design and adapter identity is the pair (fingerprint_bf16, residual_fingerprint) - never assert
distinctness on the carrier. The producer cannot compute ``residual_fingerprint`` itself: the block
half carries base-model facts (``hidden_size``, ``n_layers_model``, ``anchor_layer`` = the base's
first o_proj layer, ``boundaries``, the served ``response_header_ids``) that only a served standup /
base skeleton supplies (lora_export._derived_dims, _export_residual_slot), not the object bytes. So
beside the existing ``new_sha != parent_sha`` it computes the TENSOR HALF of that fingerprint
(:func:`payload_tensor_fingerprint`: residual_fingerprint's per-tensor recipe verbatim) over the
payload the exporter's own parser builds from each stored gate (lora_export._residual_psr on the
kind-``psr`` view - the object itself, or each side of a paired blob as _parent_blob extracts it), and
refuses unless the copy's all-gates payload fingerprint and EVERY gate's direction-key fingerprint
differ from the parent's while every gate's gate-key fingerprint is equal (the scale reached the bytes
of every side; the gates did not move). For one FIXED block ``residual_fingerprint`` differs exactly
when this tensor half differs, so this is distinctness of the served payload identity, up to the block
the standup adds. Whether ``psr_paired_write`` is a ``native_residual.RESIDUAL_SLOT_KINDS`` member is
READ off the tree and recorded (``paired_kind_in_residual_slot_kinds``), never assumed: where it is
(served natively as the two-gate op), its served residual_fingerprint hashes the two-gate payload
lora_export._residual_psr_paired_write builds, which this producer does not compute (the block half
needs a standup); where it is not, no paired residual payload exists. Either way the per-side payload
fingerprints are of the kind-psr views - each side's own payload identity, what a paired payload
carries per side. ``--sibling <rung dir>`` (repeatable) additionally refuses a copy whose sha or all-gates payload
fingerprint equals another rung's. The dose is baked into the bytes and never a serve-time
coefficient: the native residual slot refuses a gated payload with coeff != 1.0
(native_residual.validate_block), so the producer refuses a copy carrying any coefficient key
(:data:`COEFFICIENT_KEYS`) other than exactly 1.0 and pins every adapted op's coeff at 1.0 -
recorded under ``dose_scaled_from.dose_locus``.

Output dir (a bank-stageable cell dir, sdfbase full22 layout):
  object.pt                    the scaled blob (same keys as PSRObject.save / PSRPairedWriteObject._blob)
  object.pt.provenance.json    the parent's, plus dose_scale / parent_object_sha256
  train_budget_meta.json  OR   the parent's record (the training / derivation facts ARE the parent's:
  derived_meta.json            this object was not trained separately) with object_sha256 re-bound to
                               the scaled bytes and a ``dose_scaled_from`` block naming the parent
  staged_from.json             the staging record (source parent dir, producer)
No COMPLETE marker (this is not a publication of a fit); the view builder /
staged_cell_problems treat an absent marker as fine.

  python scripts/sdfbase/scale_psr_dose.py --parent <cell dir> --scale 0.5 \
      --out <cell dir> --target qwen35_4b --staged-id half__constv2-mix-full__psr_ce_all
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
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
                               producer_commit, rebind_identity, sha256_file, write_staged_from)

PRODUCER = "scripts/sdfbase/scale_psr_dose.py"
PSR_ARMS = ("psr_ce_all", "psr_kl_all", "psr_mse_all")
PSR_KIND = "psr"
#: extraction/paired_write.PSRPairedWriteObject.kind — V = +pos − neg, one combined hook per layer.
PAIRED_KIND = "psr_paired_write"
SUPPORTED_KINDS = (PSR_KIND, PAIRED_KIND)
#: The paired write's two sides, in the blob's own order. A TUPLE iterated in ONE loop by the scaler
#: and the verifier: neither takes a side argument, so a copy that scales one side cannot be produced
#: or admitted here (module docstring, TWO KINDS — scaling one side is a different difference).
PAIRED_SIDES = ("pos", "neg")
PSR_TENSOR_GROUPS = ("directions", "gate_w", "gate_b")
#: Spellings under which a served kind carries a serve-time coefficient (lens_adapters' base_scale /
#: base_alpha, the registry shim's coefficient, HookOp.coeff, PositionMaskedFit.apply's scale). A gated
#: PSR artifact must carry NONE of them, or exactly 1.0: its dose is ||direction|| in the bytes, and the
#: native residual slot refuses a gated payload with coeff != 1.0 (native_residual.validate_block).
COEFFICIENT_KEYS = ("coeff", "coefficient", "scale", "base_scale", "base_alpha", "effective_scale")


def _blob_bytes(blob: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


# --------------------------------------------------------------------------------------------------- #
# Fingerprints and the dose locus                                                                     #
# --------------------------------------------------------------------------------------------------- #


def residual_payload_tensors(psr_blob: dict) -> dict:
    """The residual-slot PAYLOAD of a kind-``psr`` blob — ``sisteer_residual.layers.<l>.{gate_w, gate_b,
    direction}`` — built by the EXPORTER's own parser (lora_export._residual_psr), so keys, dtypes and shapes
    are exactly what a bank entry carries and what native_residual.residual_fingerprint hashes. A parser
    refusal (a non-fp32 tensor, a missing layer) refuses here too: such an object could not be served."""
    from sisteer.capeval.serving import lora_export as le

    layers = [int(l) for l in psr_blob["layers"]]
    first = psr_blob["directions"][str(layers[0])]
    try:
        return le._residual_psr({}, psr_blob, hidden_size=int(first.shape[0]))["tensors"]
    except Exception as e:  # noqa: BLE001 — every parser refusal is a refusal here
        raise SystemExit(f"the residual exporter's parser (lora_export._residual_psr) refused the kind-psr view: {e}")


def payload_tensor_fingerprint(tensors: dict) -> str:
    """The TENSOR HALF of native_residual.residual_fingerprint — its per-tensor recipe verbatim (key, dtype,
    shape, raw bytes, in sorted key order) WITHOUT the ``petri_residual_slot`` block. The block carries
    base-model facts (hidden_size, n_layers_model, anchor_layer, boundaries, the served header ids) that only
    a served standup / base skeleton supplies, not the object bytes; for one fixed block residual_fingerprint
    differs exactly when this does (module docstring, FINGERPRINTS)."""
    h = hashlib.sha256()
    for key in sorted(tensors):
        t = tensors[key].detach().cpu().contiguous()
        h.update(key.encode("utf-8"))
        h.update(str(t.dtype).encode("utf-8"))
        h.update(json.dumps(list(t.shape)).encode("utf-8"))
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def _gates_of(blob: dict, kind: str) -> dict[str, dict]:
    """The stored gates of a blob as {tag: kind-psr view} — the blob itself (tag ``psr``) for the single-gate
    kind; each side as PSRPairedWriteObject._parent_blob extracts it for the paired kind."""
    if kind == PAIRED_KIND:
        from sisteer.extraction.paired_write import PSRPairedWriteObject
        return {side: PSRPairedWriteObject._parent_blob(blob, side) for side in PAIRED_SIDES}
    return {PSR_KIND: blob}


def payload_fingerprints(blob: dict, kind: str) -> dict:
    """Per stored gate: the payload's tensor fingerprint, and the same recipe over its direction keys alone
    and its gate keys (gate_w, gate_b) alone; plus ``all_gates`` over every gate's payload (keys prefixed by
    the gate tag, so a paired blob's covers BOTH sides)."""
    per_gate: dict = {}
    union: dict = {}
    for tag, view in _gates_of(blob, kind).items():
        tensors = residual_payload_tensors(view)
        per_gate[tag] = {
            "payload": payload_tensor_fingerprint(tensors),
            "direction_keys": payload_tensor_fingerprint({k: t for k, t in tensors.items() if k.endswith(".direction")}),
            "gate_keys": payload_tensor_fingerprint({k: t for k, t in tensors.items() if k.endswith((".gate_w", ".gate_b"))}),
            "n_tensors": len(tensors),
        }
        union.update({f"{tag}/{k}": t for k, t in tensors.items()})
    return {"per_gate": per_gate, "all_gates": payload_tensor_fingerprint(union)}


def fingerprint_record(parent: dict, copy: dict, kind: str) -> dict:
    """The parent's and the copy's payload fingerprints (:func:`payload_fingerprints`) with what they must
    satisfy REFUSED otherwise: the copy's all-gates fingerprint differs from the parent's; EVERY gate's
    direction-key fingerprint differs (the scale reached the bytes of every side — a matching one on any side
    means it did not); every gate's gate-key fingerprint is equal (the gauge, never the dose). Module
    docstring, FINGERPRINTS: this is the tensor half of the served residual_fingerprint, not the carrier's
    fingerprint_bf16 (identical across PSR payloads by design)."""
    fp_p, fp_c = payload_fingerprints(parent, kind), payload_fingerprints(copy, kind)
    if fp_p["all_gates"] == fp_c["all_gates"]:
        raise SystemExit("fingerprint check failed: the copy's residual-payload tensor fingerprint equals the "
                         "parent's — the scale never reached the bytes")
    gates: dict = {}
    for tag in fp_p["per_gate"]:
        where = f"side {tag!r}" if kind == PAIRED_KIND else "the object"
        pg, cg = fp_p["per_gate"][tag], fp_c["per_gate"][tag]
        if pg["direction_keys"] == cg["direction_keys"]:
            raise SystemExit(f"fingerprint check failed: {where}'s direction payload fingerprint equals the parent's — "
                             f"the scale never reached {where}'s bytes")
        if pg["gate_keys"] != cg["gate_keys"]:
            raise SystemExit(f"fingerprint check failed: {where}'s gate payload fingerprint differs from the parent's — "
                             "the gate is the gauge, never the dose, and must ride byte-identical")
        if pg["n_tensors"] != cg["n_tensors"]:
            raise SystemExit(f"fingerprint check failed: {where}'s payload has {cg['n_tensors']} tensors, the parent's {pg['n_tensors']}")
        gates[tag] = {"payload": {"parent": pg["payload"], "copy": cg["payload"]},
                      "direction_keys": {"parent": pg["direction_keys"], "copy": cg["direction_keys"]},
                      "gate_keys": {"parent": pg["gate_keys"], "copy": cg["gate_keys"]},
                      "n_tensors": pg["n_tensors"]}
    rec: dict = {
        "residual_payload_tensor_fingerprint": {
            "recipe": ("the TENSOR half of native_residual.residual_fingerprint — sha256 over (key, dtype, shape, raw bytes) "
                       "per payload tensor sisteer_residual.layers.<l>.{gate_w, gate_b, direction} in sorted key order, the "
                       "payload built by lora_export._residual_psr from the kind-psr view of each stored gate; all_gates "
                       "prefixes each tensor key with its gate tag"),
            "block_half": ("NOT computed: the petri_residual_slot block carries base-model facts (hidden_size, n_layers_model, "
                           "anchor_layer = the base's first o_proj layer, boundaries, the served response_header_ids) that a "
                           "served standup / base skeleton supplies (lora_export._derived_dims), not the object bytes; for one "
                           "fixed block residual_fingerprint differs exactly when the tensor half differs"),
            "not_fingerprint_bf16": ("fingerprint_bf16 is the rank-1 zero LoRA CARRIER's fingerprint, identical across every "
                                     "PSR payload on one base by design (serving_attestation: adapter identity is the pair "
                                     "(fingerprint_bf16, residual_fingerprint)); distinctness is asserted on the payload only"),
            "all_gates": {"parent": fp_p["all_gates"], "copy": fp_c["all_gates"]},
            "per_gate": gates,
        },
    }
    if kind == PAIRED_KIND:
        from sisteer.capeval.serving import native_residual as nr
        in_slot = PAIRED_KIND in nr.RESIDUAL_SLOT_KINDS          # read off the tree, never assumed
        rec["residual_payload_tensor_fingerprint"]["paired_kind_in_residual_slot_kinds"] = bool(in_slot)
        rec["residual_payload_tensor_fingerprint"]["paired_kind_note"] = (
            (f"{PAIRED_KIND!r} IS a native_residual.RESIDUAL_SLOT_KINDS member on this tree (served natively as the two-gate "
             "op); its served residual_fingerprint hashes the two-gate payload lora_export._residual_psr_paired_write builds, "
             "which this producer does not compute (the block half needs a standup); "
             if in_slot else
             f"{PAIRED_KIND!r} is not a native_residual.RESIDUAL_SLOT_KINDS member on this tree (its native serving is "
             "deferred), so no paired residual payload or residual_fingerprint exists here; ")
            + "per_gate.pos / per_gate.neg are the kind-psr views PSRPairedWriteObject._parent_blob extracts — each side's own "
              "payload identity, what a paired payload carries per side")
    return rec


def check_siblings(sibling_dirs: list[Path], new_sha: str, new_fp: str) -> list[dict]:
    """``--sibling`` rungs of the same ladder: the copy's object_sha256 AND all-gates payload fingerprint must differ
    from each (two rungs with the same bytes mean the scale never reached them). Returns what was compared."""
    out = []
    for d in sibling_dirs:
        p = Path(d) / "object.pt"
        if not p.is_file():
            raise SystemExit(f"--sibling {d} lacks object.pt")
        sha = sha256_file(p)
        sib = torch.load(p, map_location="cpu", weights_only=True)
        sib_kind = sib.get("kind")
        if sib_kind not in SUPPORTED_KINDS:
            raise SystemExit(f"--sibling {d} stores kind {sib_kind!r}, not one of {SUPPORTED_KINDS}")
        fp = payload_fingerprints(sib, sib_kind)["all_gates"]
        if sha == new_sha or fp == new_fp:
            raise SystemExit(f"the copy is identical to sibling rung {d} (object_sha256 {sha[:12]}…, residual-payload tensor "
                             f"fingerprint {fp[:12]}…) — two rungs of one ladder must differ; the scale never reached the bytes")
        out.append({"dir": str(d), "object_sha256": sha, "residual_payload_tensor_fingerprint": fp})
    return out


def assert_dose_in_bytes(blob: dict, kind: str, *, where: str = "the copy") -> dict:
    """The dose lives in ||direction||, never on the serve path: the artifact (top level, and each side of a
    paired blob) carries NONE of :data:`COEFFICIENT_KEYS`, or carries it at exactly 1.0. Anything else refuses —
    the native residual slot would refuse the gated payload too (coeff != 1.0), and a coefficient that
    quietly moved the dose onto the serve path would make the bytes lie about the dose they carry."""
    scopes = {"": blob}
    if kind == PAIRED_KIND:
        scopes.update({side: blob[side] for side in PAIRED_SIDES})
    present: dict = {}
    for scope, d in scopes.items():
        for k in COEFFICIENT_KEYS:
            if k in d:
                v = d[k]
                if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != 1.0:
                    raise SystemExit(f"{where} carries a serve-time coefficient {scope + '.' if scope else ''}{k}={v!r}; "
                                     "the PSR dose is baked into ||direction|| and served at coeff 1.0 exactly "
                                     "(native_residual refuses a gated payload with coeff != 1.0) — refusing")
                present[f"{scope}.{k}" if scope else k] = 1.0
    return {"locus": "bytes: ||directions|| per stored gate" + (" (both sides)" if kind == PAIRED_KIND else ""),
            "serve_time_coefficient": 1.0,
            "coefficient_keys_present": present,
            "rule": f"none of {list(COEFFICIENT_KEYS)} in the artifact (top level" +
                    (", pos, neg" if kind == PAIRED_KIND else "") +
                    "), or exactly 1.0; every adapted HookOp.coeff == 1.0 (served_ops)"}


# --------------------------------------------------------------------------------------------------- #
# Scaling                                                                                             #
# --------------------------------------------------------------------------------------------------- #


def scale_psr_blob(blob: dict, scale: float, layers: list[int]) -> tuple[dict, dict, dict]:
    """The single-gate copy: ``directions`` x scale per layer, everything else the parent's own objects,
    SAME key order as PSRObject.save. Returns (blob, norms_before, norms_after) by layer."""
    norms_before, norms_after = {}, {}
    new = {}
    for k, v in blob.items():          # SAME key order as PSRObject.save
        if k == "directions":
            scaled = {}
            for l in layers:
                d = v[str(l)].detach().float().clone()
                norms_before[l] = float(d.norm())
                scaled[str(l)] = d * float(scale)
                norms_after[l] = float(scaled[str(l)].norm())
            new[k] = scaled
        else:
            new[k] = v
    return new, norms_before, norms_after


def _check_paired_shape(blob: dict, layers: list[int]) -> None:
    for side in PAIRED_SIDES:
        if not isinstance(blob.get(side), dict):
            raise SystemExit(f"{PAIRED_KIND} parent lacks its {side!r} side — not a PSRPairedWriteObject blob")
        for g in PSR_TENSOR_GROUPS:
            if not isinstance(blob[side].get(g), dict):
                raise SystemExit(f"{PAIRED_KIND} parent side {side!r} lacks {g!r}")
            missing = [l for l in layers if str(l) not in blob[side][g]]
            if missing:
                raise SystemExit(f"{PAIRED_KIND} parent side {side!r} {g!r} lacks layers {missing}")


def scale_paired_blob(blob: dict, scale: float, layers: list[int]) -> tuple[dict, dict]:
    """The paired copy: BOTH sides' ``directions`` x the ONE ``scale`` per layer; both gates and every
    non-side key the parent's own objects; SAME key order as PSRPairedWriteObject._blob / _side.

    Why both sides, and why the same factor. V(z) = u_pos(z) − u_neg(z) with u = relu(w·z + b)·d linear in
    its d, so c·V = c·u_pos − c·u_neg: the dose of the paired write is the common factor on both sides'
    directions. Scaling ONE side gives c·u_pos − u_neg (or u_pos − c·u_neg) — a DIFFERENT DIFFERENCE, i.e. a
    different intervention (as c → 0 it degenerates to a lone parent), not a dose of V. Hence the
    structure: this function takes one scale and no side selector, PAIRED_SIDES is iterated in the one
    loop below, and verify_paired refuses a copy whose two sides do not carry the same factor.

    Returns (blob, norms) with norms = {side: {"before": {layer: ||d||}, "after": {...}}}."""
    norms = {side: {"before": {}, "after": {}} for side in PAIRED_SIDES}
    new = {}
    for k, v in blob.items():          # SAME key order as PSRPairedWriteObject._blob
        if k in PAIRED_SIDES:
            side = {}
            for g, tensors in v.items():     # SAME key order as PSRPairedWriteObject._side
                if g == "directions":
                    scaled = {}
                    for l in layers:
                        d = tensors[str(l)].detach().float().clone()
                        norms[k]["before"][str(l)] = float(d.norm())
                        scaled[str(l)] = d * float(scale)       # the ONE scale, this side — and the other
                        norms[k]["after"][str(l)] = float(scaled[str(l)].norm())
                    side[g] = scaled
                else:
                    side[g] = tensors                            # gate_w / gate_b: the parent's own objects
            new[k] = side
        else:
            new[k] = v
    return new, norms


# --------------------------------------------------------------------------------------------------- #
# Verification                                                                                        #
# --------------------------------------------------------------------------------------------------- #


def verify(blob: dict, rt: dict, scale: float, layers: list[int], *, probe_rows: int, seed: int) -> dict:
    rec: dict = {"scale": float(scale), "kind": "psr", "checks": []}
    # 1. tensors: directions exactly scale x parent, everything else bit-identical
    for l in layers:
        if not torch.equal(rt["directions"][str(l)], blob["directions"][str(l)].float() * float(scale)):
            raise SystemExit(f"tensor check failed: directions[{l}] != {scale} x parent")
        if not torch.equal(rt["gate_w"][str(l)], blob["gate_w"][str(l)]):
            raise SystemExit(f"tensor check failed: gate_w[{l}] changed")
        if not torch.equal(rt["gate_b"][str(l)], blob["gate_b"][str(l)]):
            raise SystemExit(f"tensor check failed: gate_b[{l}] changed")
    for k in blob:
        if k not in ("directions", "gate_w", "gate_b") and rt[k] != blob[k]:
            raise SystemExit(f"non-tensor key {k!r} changed")
    rec["checks"].append({"check": "tensors", "status": "PASS", "scaled": ["directions"],
                          "unchanged": ["gate_w", "gate_b"],
                          "rule": "torch.equal(new, parent * scale) for directions; torch.equal(new, parent) otherwise"})
    # 2. served ops through the wave's adapter
    positions = blob.get("positions", "all")
    if positions != "all":
        rec["checks"].append({"check": "served_ops", "status": "SKIPPED",
                              "reason": f"positions={positions!r}: the adapter needs the served header ids"})
    else:
        from sisteer.capeval.serving import lens_adapters as la
        ops_full = la._adapt_psr({}, _blob_bytes(blob))
        ops_half = la._adapt_psr({}, _blob_bytes(rt))
        if len(ops_full) != len(ops_half):
            raise SystemExit("served-op check failed: op count differs")
        for of, oh in zip(ops_full, ops_half):
            if (of.op, of.layer, of.coeff, of.positions, of.skip_bos, of.phase, of.header_ids) != \
                    (oh.op, oh.layer, oh.coeff, oh.positions, oh.skip_bos, oh.phase, oh.header_ids):
                raise SystemExit(f"served-op check failed: op identity differs at layer {of.layer}")
            # the dose is in ||vector||, never in the op's coefficient: the native residual slot refuses a
            # gated payload with coeff != 1.0, so a coeff anywhere but 1.0 here is a dose on the serve path
            if of.coeff != 1.0 or oh.coeff != 1.0:
                raise SystemExit(f"served-op check failed: coeff {of.coeff} / {oh.coeff} != 1.0 at layer {of.layer} — "
                                 "the PSR dose is baked into ||direction||, never carried as a serve-time coefficient")
            if not (torch.equal(of.gate_w, oh.gate_w) and torch.equal(of.gate_b, oh.gate_b)):
                raise SystemExit(f"served-op check failed: gate differs at layer {of.layer}")
            if not torch.equal(oh.vector, of.vector * float(scale)):
                raise SystemExit(f"served-op check failed: vector != {scale} x parent at layer {of.layer}")
        rec["checks"].append({"check": "served_ops", "status": "PASS", "n_ops": len(ops_full), "coeff": 1.0,
                              "adapter": "sisteer.capeval.serving.lens_adapters._adapt_psr",
                              "rule": "per op: gate_w / gate_b / coeff / layer / positions identical, coeff == 1.0; "
                                      "vector == parent * scale (torch.equal)"})
    # 3. served activation on one fixed probe batch: relu(h.w + b) * d
    g = torch.Generator().manual_seed(int(seed))
    max_abs = 0.0
    ref = 0.0
    exact = True
    exact_bf16 = True
    for l in layers:
        k = str(l)
        w, b = blob["gate_w"][k].float(), blob["gate_b"][k].float().reshape(1)
        d_full, d_half = blob["directions"][k].float(), rt["directions"][k]
        h = torch.randn(int(probe_rows), w.shape[0], generator=g, dtype=torch.float32)
        gate = torch.relu(h @ w + b)                    # [rows]
        e_full = gate.unsqueeze(-1) * d_full
        e_half = gate.unsqueeze(-1) * d_half
        want = e_full * float(scale)
        max_abs = max(max_abs, (e_half - want).abs().max().item())
        ref = max(ref, want.abs().max().item())
        exact &= torch.equal(e_half, want)
        exact_bf16 &= torch.equal(e_half.to(torch.bfloat16), want.to(torch.bfloat16))
    tol = 1e-6 * max(ref, 1e-30)
    if max_abs > tol:
        raise SystemExit(f"served-activation check failed: max |edit_scaled - scale*edit_parent| = {max_abs} (tol {tol})")
    rec["checks"].append({"check": "served_activation", "status": "PASS", "probe_rows": int(probe_rows),
                          "seed": int(seed), "layers": len(layers), "max_abs_diff_fp32": max_abs,
                          "exact_fp32": bool(exact), "exact_bf16": bool(exact_bf16), "reference_max_abs": ref,
                          "rule": "relu(h @ gate_w + gate_b) * directions on h ~ N(0,1): copy == scale * parent"})
    rec["status"] = "PASS"
    return rec


def _paired_write(blob: dict, h: torch.Tensor, key: str) -> torch.Tensor:
    """V(h) = u_pos(h) − u_neg(h) at one layer on ONE shared pre-write state h ([rows, d]) —
    SharedStateWriteObject.paired_write's arithmetic on the stored tensors: each gate on the same h in fp32,
    the signed terms summed in fp32, one write."""
    add = None
    for coef, side in ((1.0, "pos"), (-1.0, "neg")):
        sb = blob[side]
        pre = h.float() @ sb["gate_w"][key].float() + sb["gate_b"][key].float()      # PSRObject.gate_pre
        term = (coef * torch.relu(pre)).unsqueeze(-1) * sb["directions"][key].float()
        add = term if add is None else add + term
    return add


def verify_paired_served_ops(blob: dict, rt: dict, scale: float, layers: list[int]) -> dict:
    """The served-op level at the PAIRED shape: the single-gate ``_adapt_psr`` check of :func:`verify`
    lifted to two sides (module docstring, VERIFICATION). BOTH artifacts are translated through the
    REGISTERED adapter for :data:`PAIRED_KIND` (lens_adapters.get_adapter — the one door every serving
    path uses) and, per op: the op is :data:`lens_adapters.OP_RELU_GATED_VECTOR_PAIR` (any other shape
    refuses — this check knows that layout and guesses no other); the identity (op, layer, positions,
    skip_bos, phase, header ids) is equal parent vs copy; ``coeff`` is exactly 1.0 on BOTH (the dose never
    rides the serve path); BOTH gates are byte-identical — pos ``gate_w`` / ``gate_b`` and neg ``gate_w2``
    / ``gate_b2`` (torch.equal); and BOTH payload vectors equal the parent's x the ONE ``scale`` —
    ``vector`` (pos) and ``vector2`` (neg), torch.equal, the max abs diff recorded per side — the same
    factor on both sides AT THE OP, so a one-sided scale is refused here whichever side moved (the two
    sides ride as two vectors on one op, never as a fused blob that could hide it).

    On a tree where NO adapter is registered for the paired kind the level is SKIPPED with the reason
    recorded (the producer runs on a pre-adapter checkout), never dropped and never guessed. Raises
    SystemExit on any failure; returns the level's record (the single-gate ``served_ops`` schema:
    check / status / n_ops / coeff / adapter / rule, plus ``op`` and the per-side ``max_abs_diff_fp32``)."""
    from sisteer.capeval.serving import lens_adapters as la
    registered = la.registered_kinds()
    if PAIRED_KIND not in registered:
        return {"check": "served_ops", "status": "SKIPPED", "adapter": None,
                "reason": (f"no lens adapter is registered for kind {PAIRED_KIND!r} on this tree (lens_adapters."
                           f"registered_kinds() = {sorted(registered)}) — the paired kind has no served op to translate "
                           "here. Each SIDE's ops are translated through the parent kind's adapter (_adapt_psr) under "
                           "per_side.<side>.checks; the paired arithmetic is checked in served_activation")}
    adapter = la.get_adapter(PAIRED_KIND)
    adapter_name = f"{adapter.__module__}.{adapter.__qualname__}"
    ops_full = adapter({}, _blob_bytes(blob))
    ops_half = adapter({}, _blob_bytes(rt))
    if len(ops_full) != len(ops_half):
        raise SystemExit(f"served-op check failed ({PAIRED_KIND}): op count differs, parent {len(ops_full)} vs copy {len(ops_half)}")
    if len(ops_full) != len(layers):
        raise SystemExit(f"served-op check failed ({PAIRED_KIND}): {len(ops_full)} ops for {len(layers)} layers — "
                         "the paired adapter emits ONE op per stored layer")
    sides = (("pos", "vector", "gate_w", "gate_b"), ("neg", "vector2", "gate_w2", "gate_b2"))
    max_abs = {side: 0.0 for side, *_ in sides}
    for of, oh in zip(ops_full, ops_half):
        if of.op != la.OP_RELU_GATED_VECTOR_PAIR or oh.op != la.OP_RELU_GATED_VECTOR_PAIR:
            raise SystemExit(f"served-op check failed ({PAIRED_KIND}): the adapter emitted op {of.op!r} / {oh.op!r} at layer "
                             f"{of.layer}; this check knows the {la.OP_RELU_GATED_VECTOR_PAIR!r} layout (vector / gate_w / gate_b "
                             "the pos side, vector2 / gate_w2 / gate_b2 the neg side) and guesses no other — refusing")
        if (of.op, of.layer, of.positions, of.skip_bos, of.phase, of.header_ids) != \
                (oh.op, oh.layer, oh.positions, oh.skip_bos, oh.phase, oh.header_ids):
            raise SystemExit(f"served-op check failed ({PAIRED_KIND}): op identity differs at layer {of.layer}")
        # the dose is in ||vector|| on both sides, never in the op's coefficient: the native residual slot
        # refuses a gated payload with coeff != 1.0, so a coeff anywhere but 1.0 here is a dose on the serve path
        if of.coeff != 1.0 or oh.coeff != 1.0:
            raise SystemExit(f"served-op check failed ({PAIRED_KIND}): coeff {of.coeff} / {oh.coeff} != 1.0 at layer {of.layer} "
                             "— the paired dose is baked into both sides' ||direction||, never carried as a serve-time coefficient")
        for side, vec, gw, gb in sides:
            if not (torch.equal(getattr(of, gw), getattr(oh, gw)) and torch.equal(getattr(of, gb), getattr(oh, gb))):
                raise SystemExit(f"served-op check failed ({PAIRED_KIND}): the {side} gate ({gw} / {gb}) differs at layer "
                                 f"{of.layer} — the gate is the gauge, never the dose")
            want = getattr(of, vec) * float(scale)
            got = getattr(oh, vec)
            max_abs[side] = max(max_abs[side], float((got - want).abs().max().item()))
            if not torch.equal(got, want):
                raise SystemExit(f"served-op check failed ({PAIRED_KIND}): the {side} side's payload {vec} != {scale} x parent at "
                                 f"layer {of.layer} (max abs diff {max_abs[side]}) — both sides must carry the SAME factor at the "
                                 "served op (one side scaled is a different difference, not a dose)")
    return {"check": "served_ops", "status": "PASS", "n_ops": len(ops_full), "coeff": 1.0,
            "adapter": adapter_name, "op": la.OP_RELU_GATED_VECTOR_PAIR,
            "max_abs_diff_fp32": max_abs,
            "rule": ("per op (one relu_gated_vector_pair per layer, both artifacts through the registered paired adapter): "
                     "layer / positions / skip_bos / phase / header ids identical, coeff == 1.0 on parent and copy; pos gate "
                     "(gate_w, gate_b) and neg gate (gate_w2, gate_b2) torch.equal; vector == parent.vector * scale AND "
                     "vector2 == parent.vector2 * scale (torch.equal) — the ONE scale on both sides at the served op")}


def verify_paired(blob: dict, rt: dict, scale: float, layers: list[int], *, probe_rows: int, seed: int) -> dict:
    """The three-level check at the PAIRED shape (module docstring, TWO KINDS / VERIFICATION), plus the
    full single-gate :func:`verify` per side on the kind-``psr`` view PSRPairedWriteObject._parent_blob
    extracts. Raises on failure; returns the record."""
    from sisteer.extraction.paired_write import PSRPairedWriteObject

    rec: dict = {"scale": float(scale), "kind": PAIRED_KIND, "sides": list(PAIRED_SIDES), "checks": [], "per_side": {}}
    if set(rt) != set(blob):
        raise SystemExit(f"tensor check failed: key sets differ {sorted(set(rt) ^ set(blob))}")
    # 1. tensors: BOTH sides' directions exactly the ONE scale x parent; both gates and every other key unchanged
    for side in PAIRED_SIDES:
        if set(rt[side]) != set(blob[side]):
            raise SystemExit(f"tensor check failed: side {side!r} key sets differ {sorted(set(rt[side]) ^ set(blob[side]))}")
        for l in layers:
            k = str(l)
            if not torch.equal(rt[side]["directions"][k], blob[side]["directions"][k].float() * float(scale)):
                raise SystemExit(f"tensor check failed: {side}.directions[{l}] != {scale} x parent — both sides must "
                                 "carry the SAME factor (one side scaled is a different difference, not a dose)")
            if not torch.equal(rt[side]["gate_w"][k], blob[side]["gate_w"][k]):
                raise SystemExit(f"tensor check failed: {side}.gate_w[{l}] changed")
            if not torch.equal(rt[side]["gate_b"][k], blob[side]["gate_b"][k]):
                raise SystemExit(f"tensor check failed: {side}.gate_b[{l}] changed")
        for g in rt[side]:
            if g not in PSR_TENSOR_GROUPS and rt[side][g] != blob[side][g]:
                raise SystemExit(f"non-tensor key {side}.{g!r} changed")
    for k in blob:
        if k not in PAIRED_SIDES and rt[k] != blob[k]:
            raise SystemExit(f"non-tensor key {k!r} changed")
    rec["checks"].append({"check": "tensors", "status": "PASS",
                          "scaled": [f"{s}.directions" for s in PAIRED_SIDES],
                          "unchanged": [f"{s}.{g}" for s in PAIRED_SIDES for g in ("gate_w", "gate_b")],
                          "rule": "torch.equal(new, parent * scale) for BOTH sides' directions with the one scale; "
                                  "torch.equal(new, parent) for both gates; == for every other key"})
    # 2. served ops: BOTH artifacts through the REGISTERED paired adapter — the single-gate check lifted to two
    #    sides (verify_paired_served_ops); SKIPPED with the reason recorded on a tree without the adapter, never
    #    silently dropped, never guessed
    rec["checks"].append(verify_paired_served_ops(blob, rt, scale, layers))
    # 3. served activation: the PAIRED write on ONE shared probe batch per layer
    g = torch.Generator().manual_seed(int(seed))
    max_abs = 0.0
    ref = 0.0
    exact = True
    exact_bf16 = True
    for l in layers:
        k = str(l)
        d_model = blob["pos"]["gate_w"][k].shape[0]
        h = torch.randn(int(probe_rows), d_model, generator=g, dtype=torch.float32)
        v_full = _paired_write(blob, h, k)
        v_half = _paired_write(rt, h, k)
        want = v_full * float(scale)
        max_abs = max(max_abs, (v_half - want).abs().max().item())
        ref = max(ref, want.abs().max().item())
        exact &= torch.equal(v_half, want)
        exact_bf16 &= torch.equal(v_half.to(torch.bfloat16), want.to(torch.bfloat16))
    tol = 1e-6 * max(ref, 1e-30)
    if max_abs > tol:
        raise SystemExit(f"served-activation check failed: max |V_scaled - scale*V_parent| = {max_abs} (tol {tol})")
    rec["checks"].append({"check": "served_activation", "status": "PASS", "probe_rows": int(probe_rows),
                          "seed": int(seed), "layers": len(layers), "max_abs_diff_fp32": max_abs,
                          "exact_fp32": bool(exact), "exact_bf16": bool(exact_bf16), "reference_max_abs": ref,
                          "rule": "relu(h.w_pos + b_pos) d_pos - relu(h.w_neg + b_neg) d_neg on ONE shared h ~ N(0,1) "
                                  "(SharedStateWriteObject.paired_write's arithmetic): copy == scale * parent"})
    # 4. each side, as the kind-psr object it is, through the single-gate three-level check (served ops via _adapt_psr)
    for side in PAIRED_SIDES:
        rec["per_side"][side] = verify(PSRPairedWriteObject._parent_blob(blob, side),
                                       PSRPairedWriteObject._parent_blob(rt, side),
                                       scale, layers, probe_rows=probe_rows, seed=seed)
    rec["status"] = "PASS"
    return rec


# --------------------------------------------------------------------------------------------------- #
# main                                                                                                #
# --------------------------------------------------------------------------------------------------- #


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
    ap.add_argument("--sibling", action="append", default=[], type=Path,
                    help="another rung dir of the same ladder (holds object.pt); repeatable. The copy's object_sha256 "
                         "and tensor fingerprint must differ from every sibling's (module docstring, FINGERPRINTS)")
    args = ap.parse_args(argv)
    # A NEGATIVE scale is admitted (the strength frontier, 2026-09-10): the served edit is linear in the
    # scaled tensor, so scale x (-1) is exactly the REVERSED edit — steering against the fitted direction —
    # and the same three-level verification covers it (torch.equal(new, parent * scale) holds for any
    # finite scale). Refused: 0 (no object; the `none` anchor is that cell) and 1.0 (the parent itself).
    if not math.isfinite(args.scale) or args.scale in (0.0, 1.0):
        raise SystemExit(f"--scale must be a finite non-zero number != 1.0 (got {args.scale}); "
                         "a 1.0 copy is the parent and a 0 copy is the `none` cell")

    p_obj = args.parent / "object.pt"
    if not p_obj.is_file():
        raise SystemExit(f"parent lacks {p_obj}")
    identity, sidecar = identity_of(args.parent)
    parent_sha = sha256_file(p_obj)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    check_parent_sidecar(meta, parent_sha, sidecar)
    blob = torch.load(p_obj, map_location="cpu", weights_only=True)
    kind = blob.get("kind")
    if kind not in SUPPORTED_KINDS:
        raise SystemExit(f"parent blob kind {kind!r} is not one this producer scales {SUPPORTED_KINDS}")
    if identity == "staged":
        if kind != PSR_KIND:
            raise SystemExit(f"a {kind!r} parent is a DERIVED object (derived_meta.json with parent_pos / parent_neg); "
                             f"a {BUDGET_META} identity is not defined for it")
        if meta.get("arm") not in PSR_ARMS:
            raise SystemExit(f"parent arm {meta.get('arm')!r} is not a PSR arm")
        cell_id = meta.get("cell_id")
    else:
        if meta.get("kind") != kind:
            raise SystemExit(f"derived sidecar kind {meta.get('kind')!r} != blob kind {kind!r}")
        cell_id = meta.get("derived_id")
    if cell_id != args.staged_id.split("__", 1)[1]:
        raise SystemExit(f"--staged-id {args.staged_id} does not end in the parent's cell id {cell_id!r}")

    layers = [int(l) for l in blob["layers"]]
    if kind == PAIRED_KIND:
        _check_paired_shape(blob, layers)
        new, paired_norms = scale_paired_blob(blob, args.scale, layers)
    else:
        new, norms_before, norms_after = scale_psr_blob(blob, args.scale, layers)
    fresh_out_dir(args.out)
    o_obj = args.out / "object.pt"
    torch.save(new, o_obj)
    rt = torch.load(o_obj, map_location="cpu", weights_only=True)
    if kind == PAIRED_KIND:
        verification = verify_paired(blob, rt, args.scale, layers, probe_rows=args.probe_rows, seed=args.seed)
    else:
        verification = verify(blob, rt, args.scale, layers, probe_rows=args.probe_rows, seed=args.seed)
    dose_locus = assert_dose_in_bytes(rt, kind)
    new_sha = sha256_file(o_obj)
    assert new_sha != parent_sha
    fingerprints = {"object_sha256": {"parent": parent_sha, "copy": new_sha}, **fingerprint_record(blob, rt, kind)}
    fingerprints["siblings"] = check_siblings(
        args.sibling, new_sha, fingerprints["residual_payload_tensor_fingerprint"]["all_gates"]["copy"])

    if kind == PAIRED_KIND:
        shape_fields = {
            "scaling_method": ("BOTH sides' directions (pos.directions, neg.directions) x the SAME scale per layer; both "
                               "gates (gate_w, gate_b) untouched — c (u_pos - u_neg) = c V is the gauge-invariant dose of "
                               "the paired write; scaling one side would be a different difference, not a dose"),
            "tensors_scaled": [f"{s}.directions" for s in PAIRED_SIDES],
            "tensors_unchanged": [f"{s}.{g}" for s in PAIRED_SIDES for g in ("gate_w", "gate_b")]
                                 + ["layers", "application_order", "positions", "response_header_len"]
                                 + (["provenance"] if "provenance" in blob else []),
            "direction_norm_before_by_side_and_layer": {s: paired_norms[s]["before"] for s in PAIRED_SIDES},
            "direction_norm_after_by_side_and_layer": {s: paired_norms[s]["after"] for s in PAIRED_SIDES},
            "note": ("a post-hoc dose-scaled copy of the parent PAIRED WRITE — the record above is the PARENT's (this "
                     "object was not derived separately); the combined hook computes scale * (u_pos(z) - u_neg(z)), the "
                     "file-level equivalent of PSRPairedWriteObject.apply(model, scale=scale)"),
        }
    else:
        shape_fields = {
            "scaling_method": "directions x scale per layer; gate (gate_w, gate_b) untouched — the gauge-invariant dose",
            "tensors_scaled": ["directions"],
            "tensors_unchanged": ["gate_w", "gate_b", "layers", "positions", "response_header_len", "train_direction"]
                                 + (["provenance"] if "provenance" in blob else []),
            "direction_norm_before_by_layer": {str(l): norms_before[l] for l in layers},
            "direction_norm_after_by_layer": {str(l): norms_after[l] for l in layers},
            "note": ("a post-hoc dose-scaled copy of the parent fit — the record above is the "
                     "PARENT's (this object was not trained or derived separately); the served op computes "
                     "scale * g(h) * d, the file-level equivalent of obj.apply(model, scale=scale)"),
        }
    block = {
        "parent_object_sha256": parent_sha,
        "parent_dir": str(args.parent),
        "parent_staged_id": args.parent_staged_id,
        "parent_identity": identity,
        "parent_fit_fingerprint": meta.get("fit_fingerprint"),
        "object_type": kind,
        "scale": float(args.scale),
        **shape_fields,
        "verification": verification,
        "fingerprints": fingerprints,
        "dose_locus": dose_locus,
        "producer": PRODUCER,
        "producer_commit": producer_commit(Path(__file__)),
        "written_at": now_utc(),
    }
    meta_out = rebind_identity(meta, new_sha, block)
    (args.out / (BUDGET_META if identity == "staged" else DERIVED_META)).write_text(
        json.dumps(meta_out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    copy_provenance_json(args.parent, args.out, scale=args.scale, parent_sha=parent_sha,
                         fallback={"kind": kind, "layers": layers})
    write_staged_from(args.out, target=args.target, staged_id=args.staged_id, cell_id=cell_id, new_sha=new_sha,
                      parent_staged_id=args.parent_staged_id, parent=args.parent, parent_sha=parent_sha,
                      scale=args.scale, producer=PRODUCER)
    if kind == PAIRED_KIND:
        norm_summary = {
            "norm_before_mean": {s: sum(paired_norms[s]["before"].values()) / len(layers) for s in PAIRED_SIDES},
            "norm_after_mean": {s: sum(paired_norms[s]["after"].values()) / len(layers) for s in PAIRED_SIDES},
        }
    else:
        norm_summary = {"norm_before_mean": sum(norms_before.values()) / len(layers),
                        "norm_after_mean": sum(norms_after.values()) / len(layers)}
    print(json.dumps({"parent_object_sha256": parent_sha, "object_sha256": new_sha, "scale": args.scale, "kind": kind,
                      "identity": identity, "layers": len(layers),
                      **norm_summary,
                      "verification": {c["check"]: c["status"] for c in verification["checks"]},
                      "max_abs_diff_fp32": verification["checks"][-1]["max_abs_diff_fp32"],
                      "residual_payload_tensor_fingerprint": fingerprints["residual_payload_tensor_fingerprint"]["all_gates"]["copy"],
                      "out": str(args.out)}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
