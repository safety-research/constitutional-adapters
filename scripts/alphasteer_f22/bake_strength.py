#!/usr/bin/env python
"""alphasteer-full22 STRENGTH FRONTIER (paper Figure 5) — bake one target's strength-scaled AlphaSteer
cells for the frontier's dose ladder, with the campaign's OWN rung machinery (bake_rungs.py) and its
verification, and lay them out as a bank make_bank_view.py admits.

THE LADDER (as22_strength.STRENGTHS): nominal `base_scale` -0.5, 0.1, 0.3, 0.5, 1.0 — NOT a geometric grid
off the fit's rho (bake_rungs.lambda_grid), NOT multiples of the campaign's selected rung. The grid is the
ONE thing this script does not reuse; everything that turns a strength into an object is imported from
bake_rungs.py and never re-spelled here:

  * `Parent`        the Phase B lambda=1 fit at <root>/<target>/cells/alphasteer/, identity-checked
                    (fit_meta fingerprint + COMPLETE marker, fit_provenance.json as the ONE identity
                    sidecar, blob base_scale == 1.0 — so this bakes off the PARENT, never off the
                    campaign's pre-scaled cells/selected__alphasteer, which Parent refuses);
  * `scaled_blob`   the parent mapping in the parent's key order with ONLY `base_scale` rewritten
                    (tensors by reference: no copy, no arithmetic);
  * `verify_rung`   from the bytes on disk, through the real adapter (lens_adapters.get_adapter) and the
                    real request-time body (apply_hook_ops): Ms bit-identical, coeff == strength, hook
                    layers == fit layers - 1, served edit == strength x the parent's edit under
                    torch.equal in fp32 AND bf16.

Plus ONE check of this lane's own, appended to every scaled cell's verification block: `sign`. The
served coefficient must carry the strength's sign, and for a NEGATIVE strength the served edit must be
EXACTLY `-|s|` x the parent's edit AND exactly the NEGATION of the `+|s|` object's edit (the +|s| blob is
built through the same adapter for the comparison), torch.equal, fp32 and bf16. Negatives need no code
change in the serving path (`_finite_float` admits them; the coeff multiplies the identical h_last @ M),
so the check is what makes that a verified fact rather than an assumption.

`1.0` IS THE PARENT and is never re-saved: its cell dir holds `object.pt` as a SYMLINK to the parent's
object.pt (bit-identity by construction — the same bytes; asserted by sha and by resolve()), the
parent's fit_provenance.json byte-identical, and sidecars recording mechanism `parent_unscaled`. It gets
its own view so the frontier's 1.0 point is a same-pass anchor like every other rung. A strength of
1.0 handed to the SCALED path refuses; 0, non-finite and non-numeric strengths refuse before disk is
touched (as22_strength.validate_strength).

LAYOUT, per target (as22_strength; make_bank_view.py's shape — the VIEWS are built by make_bank_view.py
with configs/alphasteer_f22_strength_{roster.yaml,pins.json}, never here):
    <root>/<target>/strength/cells/<view>__alphasteer/
        object.pt              the strength blob (or the anchor's symlink to the parent's)
        fit_provenance.json    the parent's, byte-identical — the ONE identity sidecar
        meta.json              the parent's cell identity (the registry keys) + a `strength` block
        dose_scaled_from.json  parent sha, strength, mechanism, the full verification block
        staged_from.json       the staging record (stage_full22.sh hashcheck_and_stamp schema)
    <root>/<target>/strength/strength_bake.json      the bake record (written LAST; --check requires it)
Every file lands by tmp + os.replace; an in-progress cell carries a `.BAKING-<staged id>` marker in the
bank dir (written first, removed last) so an interrupted bake is ours to rebuild and a stranger's
object.pt is never removed (bake_rungs' discipline).

EXIT CODES (bake_rungs'): 0 every requested target baked / kept / verified; 1 a refusal; 3 a target with
NO fitted object yet (a clean per-target skip, never a traceback).

LANES (as22_strength.LANES, 2026-09-11): `--lane alphasteer_f22` (the campaign's parents, the default) or
`--lane alphasteer_f22_rebake` (the RE-BAKED, threshold-selected parents under the parallel root
$SHARED/exp/objects/alphasteer_f22_rebake). A lane's TARGET SET is its roster's `targets` keys — `--target`
defaults to all of them and refuses a target the roster does not name. `--root` must be the lane's bank root.
Every cell's verification block carries a `layers` check: the copy's layer list equals the parent's ELEMENT FOR
ELEMENT, its Ms keys are exactly that set, and the gaps of a NON-CONTIGUOUS set are recorded, never filled.

usage (CPU only — the objects are 0.2-1.2 GB each, so run it under a CPU srun, not on a login shell):
  python scripts/alphasteer_f22/bake_strength.py --target gemma4e4b --root $SHARED/exp/objects/alphasteer_f22
  python scripts/alphasteer_f22/bake_strength.py --lane alphasteer_f22_rebake --root $SHARED/exp/objects/alphasteer_f22_rebake
  python scripts/alphasteer_f22/bake_strength.py --target gemma4e4b,qwen35_4b --root ... --check
  options: --strengths=-0.5,0.1,0.3,0.5,1.0 (default: the ladder; use the `=` form — a value starting
           with `-` is otherwise read as an option)  --force  --probe-rows 6  --seed 0
"""
from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _p in (str(REPO / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import bake_rungs as br  # noqa: E402  (Parent, scaled_blob, verify_rung — imported, never copied)
from as22_strength import (ANCHOR_STRENGTH, DEFAULT_LANE, LANES, STRENGTHS, StrengthRefusal,  # noqa: E402
                           bake_record_path, is_anchor, lane as lane_of, lane_targets, layer_gaps, load_roster,
                           parse_strength, require_target, roster_parent_layers, staged_id, strength_bank_dir,
                           strength_cell_dir, view_name)
from as22_targets import CELL_ID  # noqa: E402
from bake_rungs import MissingObject, Refusal  # noqa: E402  (the same exceptions, the same exit codes)

#: /2 adds the `layers` check (element-for-element layer-list equality, gaps recorded) to every cell's verification
#: block and the lane / lineage fields to the record; /1 records (from the first bake) are still accepted on --check
SCHEMA = "alphasteer_f22_strength_bake/2"
ACCEPTED_SCHEMAS = ("alphasteer_f22_strength_bake/1", SCHEMA)
SIDECAR_SCHEMA = "alphasteer_f22_strength_scaled_from/1"
PRODUCER = "scripts/alphasteer_f22/bake_strength.py"
BAKE_RECORD = "strength_bake.json"
DOSE_SIDECAR = br.DOSE_SIDECAR                 # dose_scaled_from.json — the same name the rung bake writes
MECHANISM_SCALED = br.MECHANISM                # "scalar_base_scale"
MECHANISM_ANCHOR = "parent_unscaled"
IDENTITY_SIDECAR = "fit_provenance.json"
EXIT_OK, EXIT_REFUSED, EXIT_NO_OBJECT = br.EXIT_OK, br.EXIT_REFUSED, br.EXIT_NO_OBJECT


# --------------------------------------------------------------------------- strengths

def parse_strengths(values: List[str]) -> List[float]:
    """Comma-separated / repeated `--strengths`; every value validated (0, NaN, inf, non-numbers refuse),
    duplicates refused. Default: the ladder."""
    if not values:
        return list(STRENGTHS)
    out: List[float] = []
    for v in values:
        for tok in str(v).split(","):
            tok = tok.strip()
            if not tok:
                continue
            s = parse_strength(tok)
            if s in out:
                raise StrengthRefusal(f"strength {s!r} listed twice")
            out.append(s)
    if not out:
        raise StrengthRefusal("--strengths names no strength")
    for s in out:
        view_name(s)                           # every strength must have a canonical view name
    return out


def scaled_strength(s: float) -> float:
    """The strength handed to the SCALED path: never the parent's 1.0 (served as the anchor)."""
    if is_anchor(s):
        raise Refusal(f"strength {ANCHOR_STRENGTH!r} is the parent object bit-for-bit — it is served from its own "
                      "view as the anchor (bake_anchor), never re-saved through the scaled path")
    return float(s)


def record_sha(record: Dict[str, Any]) -> str:
    """bake_rungs' record convention: sha256 over json.dumps(record minus 'sha256', sort_keys=True)."""
    body = {k: v for k, v in record.items() if k != "sha256"}
    return br.sha256_bytes(json.dumps(body, sort_keys=True).encode("utf-8"))


# --------------------------------------------------------------------------- the layer-list check

def verify_layers(parent: "br.Parent", blob: dict, *, tag: str) -> dict:
    """The copy's layer list equals the parent's ELEMENT FOR ELEMENT (order included), its `Ms` keys are
    exactly that set (no key added, none dropped), and nothing about the set is assumed contiguous: the
    threshold-selected re-baked parents SKIP layers (gemma 23; qwen 10 and 11), so the gaps are recorded,
    never filled, and a contiguous set is just the case with no gaps. Raises Refusal."""
    got = [int(l) for l in (blob.get("layers") or [])]
    want = [int(l) for l in (parent.blob.get("layers") or [])]
    if not want:
        raise Refusal(f"{tag}: the parent blob declares no `layers` list")
    if got != want:
        raise Refusal(f"{tag}: layer list {got} != the parent's {want} (element for element, order included)")
    keys = sorted(int(k) for k in blob.get("Ms") or {})
    if keys != sorted(want) or len(blob["Ms"]) != len(want):
        raise Refusal(f"{tag}: Ms keys {keys} ({len(blob['Ms'])} entries) != the layer list {sorted(want)}")
    if sorted(want) != parent.layers:
        raise Refusal(f"{tag}: the parent's declared layers {sorted(want)} != its Ms keys {parent.layers}")
    gaps = layer_gaps(want)
    return {"check": "layers", "status": "PASS", "fit_layers": want, "n_layers": len(want),
            "hook_layers": [l - 1 for l in want], "gaps": gaps, "contiguous": not gaps,
            "rule": "blob.layers == parent.layers element for element (order included); sorted(Ms keys) == that set "
                    "with no extra or missing key; gaps between min and max recorded, never assumed absent"}


# --------------------------------------------------------------------------- the sign check

def _probe_edits(op_p, op_r, d: int, *, dtype, T: int, seed: int):
    """bake_rungs.verify_rung's step-4 probe, restated only as far as the arithmetic: a batch whose last row
    is h_last and every other row zero, through the REAL apply_hook_ops; the broadcast edit is out[:-1]."""
    import torch
    from sisteer.capeval.serving import lens_adapters as la

    ctx = la.HookContext(phase=la.PHASE_PREFILL, positions=tuple(range(T)))
    g = torch.Generator().manual_seed(int(seed))
    hidden = torch.zeros(T, d, dtype=dtype)
    hidden[-1] = torch.randn(d, generator=g, dtype=torch.float32).to(dtype)
    (mp,) = la.materialize_ops([op_p], device="cpu", dtype=dtype)
    (mr,) = la.materialize_ops([op_r], device="cpu", dtype=dtype)
    out_p = la.apply_hook_ops(hidden, [mp], 1.0, layer=mp.layer, ctx=ctx)
    out_r = la.apply_hook_ops(hidden, [mr], 1.0, layer=mr.layer, ctx=ctx)
    return out_p[:-1], out_r[:-1]


def verify_sign(parent: "br.Parent", ops_r, s: float, *, probe_rows: int, seed: int) -> dict:
    """The strength lane's own check. Every served op's coeff carries the strength's SIGN and equals it;
    the served edit equals `s` x the parent's edit; and for a negative strength it equals EXACTLY `-|s|` x
    the parent's edit AND the exact NEGATION of the `+|s|` object's edit (built through the same adapter),
    torch.equal, fp32 and bf16. Raises Refusal."""
    import math

    import torch

    s = float(s)
    mag = abs(s)
    if len(ops_r) != len(parent.ops):
        raise Refusal(f"s={s:.4g}: sign check has {len(ops_r)} served ops vs the parent's {len(parent.ops)}")
    for op in ops_r:
        if op.coeff != s or math.copysign(1.0, op.coeff) != math.copysign(1.0, s):
            raise Refusal(f"s={s:.4g}: served coeff {op.coeff!r} at hook layer {op.layer} does not carry the "
                          f"strength (value and sign) — refusing")
    # the +|s| object, through the SAME adapter door, for the negation comparison
    if s < 0:
        abs_blob = br.scaled_blob(parent.blob, mag)
        ops_abs = br.adapter_ops(br.resolved_spec_of(abs_blob), br.blob_bytes(abs_blob))
        if len(ops_abs) != len(ops_r):
            raise Refusal(f"s={s:.4g}: the +|s| object served {len(ops_abs)} ops vs {len(ops_r)}")
    else:
        ops_abs = ops_r
    T = max(2, int(probe_rows))
    per_dtype = {}
    for dtype in (torch.float32, torch.bfloat16):
        for k, (op_p, op_r, op_a) in enumerate(zip(parent.ops, ops_r, ops_abs)):
            edit_p, edit_r = _probe_edits(op_p, op_r, parent.d, dtype=dtype, T=T, seed=seed)
            if not torch.equal(edit_r, s * edit_p):
                raise Refusal(f"s={s:.4g}: served edit != s x parent edit at hook layer {op_r.layer} ({dtype})")
            if s < 0:
                _, edit_a = _probe_edits(op_p, op_a, parent.d, dtype=dtype, T=T, seed=seed)
                if not torch.equal(edit_r, -(mag * edit_p)):
                    raise Refusal(f"s={s:.4g}: served edit != -|s| x parent edit at hook layer {op_r.layer} "
                                  f"({dtype}) — the sign did not reach the served edit exactly")
                if not torch.equal(edit_r, -edit_a):
                    raise Refusal(f"s={s:.4g}: served edit is not the exact negation of the +{mag:.4g} object's "
                                  f"edit at hook layer {op_r.layer} ({dtype})")
                if torch.equal(edit_r, edit_a) and bool((edit_a != 0).any()):
                    raise Refusal(f"s={s:.4g}: served edit EQUALS the +{mag:.4g} object's edit at hook layer "
                                  f"{op_r.layer} ({dtype}) — the sign was dropped")
            else:
                if not torch.equal(edit_r, mag * edit_p):
                    raise Refusal(f"s={s:.4g}: served edit != |s| x parent edit at hook layer {op_r.layer} ({dtype})")
        per_dtype[str(dtype).replace("torch.", "")] = {"exact": True, "probe_rows": T}
    return {"check": "sign", "status": "PASS", "strength": s, "abs_strength": mag,
            "negative": s < 0, "coeff_sign": -1 if s < 0 else 1, "n_ops": len(ops_r), "seed": int(seed), **per_dtype,
            "rule": ("every served coeff == s (value AND sign); apply_hook_ops (prefill HookContext) edit == s * parent "
                     "edit; for s < 0 additionally edit == -(|s| * parent edit) AND edit == -(edit of the +|s| object "
                     "built through the same adapter), torch.equal, fp32 and bf16")}


# --------------------------------------------------------------------------- layout helpers

def _marker(root: Path, target: str, s: float) -> Path:
    return strength_bank_dir(root, target) / f".BAKING-{staged_id(s)}"


def _is_ours(root: Path, target: str, s: float) -> bool:
    cdir = strength_cell_dir(root, target, s)
    return (cdir / DOSE_SIDECAR).is_file() or _marker(root, target, s).is_file()


def _clear_cell(root: Path, target: str, s: float, *, force: bool) -> None:
    """Remove a previous cell of OURS (complete, or interrupted with the marker down); refuse to remove
    anything else — another writer's object.pt is never deleted."""
    cdir = strength_cell_dir(root, target, s)
    marker = _marker(root, target, s)
    if cdir.exists() or cdir.is_symlink():
        has_obj = (cdir / "object.pt").is_file() or (cdir / "object.pt").is_symlink()
        if has_obj and not _is_ours(root, target, s):
            raise Refusal(f"{cdir} holds an object.pt without our {DOSE_SIDECAR} (and no {marker.name} marker) — not "
                          "a previous bake of this producer; refusing to remove it")
        if cdir.is_symlink():
            cdir.unlink()
        else:
            shutil.rmtree(cdir)
    if marker.exists():
        marker.unlink()


def _write_common_sidecars(parent: "br.Parent", cdir: Path, s: float, *, mechanism: str, sha: str,
                           verification: dict, extra_block: dict) -> None:
    br.write_bytes_atomic(cdir / IDENTITY_SIDECAR, (parent.dir / IDENTITY_SIDECAR).read_bytes())
    meta = dict(parent.meta)
    meta["sidecars"] = [IDENTITY_SIDECAR, DOSE_SIDECAR, "staged_from.json"]
    meta["strength"] = {"strength": float(s), "view": view_name(s), "staged_id": staged_id(s),
                        "mechanism": mechanism, "parent_object_sha256": parent.sha, "producer": PRODUCER}
    br.write_json_atomic(cdir / "meta.json", meta)
    block = {
        "schema": SIDECAR_SCHEMA,
        "target": parent.target, "view": view_name(s), "cell_id": CELL_ID, "staged_id": staged_id(s),
        "strength": float(s), "lambda": float(s),          # `lambda` too: the rung bake's readers key on it
        "object_sha256": sha, "parent_object_sha256": parent.sha, "parent_dir": str(parent.dir),
        "parent_sidecar_sha256": parent.sidecar_shas, "parent_base_scale": 1.0,
        "n_layers": len(parent.layers), "layers": parent.layers, "d": parent.d,
        "mechanism": mechanism,
        "intervention_hash": verification["intervention_hash"],
        "verification": verification, "producer": PRODUCER, "producer_commit": br.code_sha(),
        "written_at": br.now_utc(),
        **extra_block,
    }
    br.write_json_atomic(cdir / DOSE_SIDECAR, block)
    staged_shas = {p.name: br.sha256_file(p) for p in sorted(cdir.iterdir())
                   if p.is_file() and p.name != "staged_from.json"}
    br.write_json_atomic(cdir / "staged_from.json", {
        "target": parent.target, "lane": view_name(s), "cell_id": CELL_ID, "staged_id": staged_id(s),
        "source": (f"strength-scaled copy (base_scale {s!r}) of {parent.dir} (parent object_sha256 {parent.sha})"
                   if mechanism == MECHANISM_SCALED else
                   f"the parent object itself (base_scale 1.0, object.pt -> {parent.object_path}; object_sha256 {parent.sha})"),
        "pulled_at": br.now_utc(), "object_sha256": sha, "staged_shas": staged_shas,
        "recorded_hash_check": br._recorded_hash_check(cdir, sha), "staged_by": PRODUCER}, sort_keys=True)


def _row(parent: "br.Parent", s: float, cdir: Path, sha: str, verification: dict, *, mechanism: str,
         status: str) -> dict:
    return {"strength": float(s), "view": view_name(s), "staged_id": staged_id(s), "status": status,
            "cell_dir": str(cdir), "object_path": str(cdir / "object.pt"),
            "object_is_symlink": (cdir / "object.pt").is_symlink(),
            "object_sha256": sha, "parent_object_sha256": parent.sha, "is_parent": mechanism == MECHANISM_ANCHOR,
            "intervention_hash": verification["intervention_hash"], "mechanism": mechanism,
            "n_layers": len(parent.layers), "layers": parent.layers, "d": parent.d,
            "identity_sidecar": IDENTITY_SIDECAR,
            "files": {p.name: br.sha256_file(p) for p in sorted(cdir.iterdir()) if p.is_file()},
            "verification": verification}


# --------------------------------------------------------------------------- the scaled cell

def bake_scaled(parent: "br.Parent", root: Path, s: float, *, force: bool, probe_rows: int, seed: int) -> dict:
    import torch

    s = scaled_strength(s)
    target = parent.target
    cdir = strength_cell_dir(root, target, s)
    if not force:                                  # --force rebuilds whatever is there (of ours); no re-verification first
        existing = _existing_cell(parent, root, s, probe_rows=probe_rows, seed=seed)
        if existing is not None:
            return existing
    _clear_cell(root, target, s, force=force)
    cdir.mkdir(parents=True, exist_ok=False)
    marker = _marker(root, target, s)
    br.write_bytes_atomic(marker, f"{PRODUCER} pid {os.getpid()} {br.now_utc()} strength {s!r}\n".encode())
    try:
        blob = br.scaled_blob(parent.blob, s)
        obj = cdir / "object.pt"
        br.torch_save_atomic(blob, obj)
        rung_bytes = obj.read_bytes()
        rung_blob = torch.load(io.BytesIO(rung_bytes), map_location="cpu", weights_only=True)
        verification = br.verify_rung(parent, rung_blob, rung_bytes, s, probe_rows=probe_rows, seed=seed)
        ops_r = br.adapter_ops(br.resolved_spec_of(rung_blob), rung_bytes)
        verification["checks"].append(verify_sign(parent, ops_r, s, probe_rows=probe_rows, seed=seed))
        verification["checks"].append(verify_layers(parent, rung_blob, tag=f"{target} {view_name(s)}"))
        verification["strength"] = s
        sha = br.sha256_bytes(rung_bytes)
        if sha == parent.sha:
            raise Refusal(f"s={s:.4g}: the scaled bytes hash to the parent's sha — nothing was scaled")
        f32, md = br.fit_rho_at(parent.fit_meta, abs(s))
        _write_common_sidecars(parent, cdir, s, mechanism=MECHANISM_SCALED, sha=sha, verification=verification,
                               extra_block={
                                   "mechanism_note": br.mechanism_note(),
                                   "tensors_scaled": [], "tensors_unchanged": ["Ms", "refusals"],
                                   "scalar_written": {"base_scale": s},
                                   "rho_fit_fp32_by_layer_at_abs_strength": f32,
                                   "rho_fit_model_dtype_by_layer_at_abs_strength": md,
                                   "note": ("a strength-scaled copy of the Phase B lambda=1 fit (bake_rungs.scaled_blob): the "
                                            "identity record beside it (fit_provenance.json) is the PARENT's, byte-identical; "
                                            "served at scale 1.0 its edit is `strength` x the parent's (base_scale is the served "
                                            "op's coeff; a negative strength flips the sign of the edit exactly). The fit-side "
                                            "rho table is |strength| x the lambda=1 value (rho is a norm ratio)")})
        marker.unlink()
        return _row(parent, s, cdir, sha, verification, mechanism=MECHANISM_SCALED, status="BAKED")
    except Refusal:
        shutil.rmtree(cdir, ignore_errors=True)
        marker.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- the anchor (1.0)

def _anchor_checks(parent: "br.Parent", cdir: Path) -> dict:
    """The anchor's verification block: object.pt is a symlink that RESOLVES to the parent's object.pt,
    its bytes hash to the parent's sha, and the bytes read at THIS path translate through the real adapter
    to ops identical to the parent's (coeff 1.0, matrices torch.equal, same hook layers)."""
    import torch

    obj = cdir / "object.pt"
    tag = f"{parent.target} {view_name(ANCHOR_STRENGTH)}"
    if not obj.is_symlink():
        raise Refusal(f"{tag}: {obj} is not a symlink — the anchor serves the PARENT's bytes, never a copy")
    if obj.resolve() != parent.object_path.resolve():
        raise Refusal(f"{tag}: {obj} -> {os.readlink(obj)} resolves to {obj.resolve()}, not the parent's "
                      f"{parent.object_path.resolve()}")
    data = obj.read_bytes()
    sha = br.sha256_bytes(data)
    if sha != parent.sha:
        raise Refusal(f"{tag}: bytes at {obj} hash to {sha[:12]}, the parent's are {parent.sha[:12]}")
    blob = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if float(blob.get("base_scale", float("nan"))) != ANCHOR_STRENGTH:
        raise Refusal(f"{tag}: base_scale {blob.get('base_scale')!r} != 1.0")
    ops = br.adapter_ops(br.resolved_spec_of(blob), data)
    if len(ops) != len(parent.ops):
        raise Refusal(f"{tag}: served op count {len(ops)} != parent's {len(parent.ops)}")
    for op_a, op_p in zip(ops, parent.ops):
        if br._op_identity(op_a) != br._op_identity(op_p) or op_a.coeff != 1.0 or op_p.coeff != 1.0 \
                or not torch.equal(op_a.matrix, op_p.matrix):
            raise Refusal(f"{tag}: the served ops of the anchor differ from the parent's at hook layer {op_a.layer}")
    ih = br.intervention_hash(br.resolved_spec_of(blob), data)
    parent_ih = br.intervention_hash(parent.spec, parent.bytes)
    if ih != parent_ih:
        raise Refusal(f"{tag}: intervention_hash {ih} != the parent's {parent_ih}")
    return {"strength": ANCHOR_STRENGTH, "lambda": ANCHOR_STRENGTH, "mechanism": MECHANISM_ANCHOR, "status": "PASS",
            "intervention_hash": ih,
            "checks": [{"check": "anchor_identity", "status": "PASS", "symlink_target": os.readlink(obj),
                        "resolves_to": str(obj.resolve()), "object_sha256": sha, "parent_object_sha256": parent.sha,
                        "rule": "object.pt is a symlink resolving to the parent's object.pt; sha256(bytes) == parent sha; "
                                "base_scale == 1.0"},
                       {"check": "served_ops", "status": "PASS", "n_ops": len(ops),
                        "hook_layers": [op.layer for op in ops], "fit_layers": parent.layers,
                        "coeff_rung": 1.0, "coeff_parent": 1.0,
                        "rule": "the bytes at the anchor path translate through lens_adapters.get_adapter('alphasteer') "
                                "to ops identical to the parent's (identity, matrix torch.equal, coeff 1.0)"},
                       verify_layers(parent, blob, tag=tag)]}


def bake_anchor(parent: "br.Parent", root: Path, *, force: bool, probe_rows: int, seed: int) -> dict:
    s = ANCHOR_STRENGTH
    target = parent.target
    cdir = strength_cell_dir(root, target, s)
    if not force:
        existing = _existing_cell(parent, root, s, probe_rows=probe_rows, seed=seed)
        if existing is not None:
            return existing
    _clear_cell(root, target, s, force=force)
    cdir.mkdir(parents=True, exist_ok=False)
    marker = _marker(root, target, s)
    br.write_bytes_atomic(marker, f"{PRODUCER} pid {os.getpid()} {br.now_utc()} strength {s!r} (anchor)\n".encode())
    try:
        obj = cdir / "object.pt"
        obj.symlink_to(os.path.relpath(parent.object_path, cdir))
        verification = _anchor_checks(parent, cdir)
        _write_common_sidecars(parent, cdir, s, mechanism=MECHANISM_ANCHOR, sha=parent.sha, verification=verification,
                               extra_block={
                                   "tensors_scaled": [], "tensors_unchanged": ["Ms", "refusals"], "scalar_written": {},
                                   "object_pt": {"symlink": True, "target": os.readlink(obj)},
                                   "note": ("the frontier's 1.0 point IS the Phase B lambda=1 parent object, bit-for-bit: "
                                            "object.pt is a symlink to the parent's object.pt (no copy, no re-save), given its "
                                            "own view so it is served in the same pass as its neighbours (strength-frontier "
                                            "D6); the identity record beside it is the parent's, byte-identical")})
        marker.unlink()
        return _row(parent, s, cdir, parent.sha, verification, mechanism=MECHANISM_ANCHOR, status="BAKED")
    except Refusal:
        shutil.rmtree(cdir, ignore_errors=True)
        marker.unlink(missing_ok=True)
        raise


# --------------------------------------------------------------------------- check / keep

def check_cell(parent: "br.Parent", root: Path, s: float, *, probe_rows: int, seed: int) -> dict:
    """Re-verify one cell dir on disk (no writes). Raises Refusal naming the cell."""
    import torch

    target = parent.target
    s = float(s)
    cdir = strength_cell_dir(root, target, s)
    tag = f"{target} {view_name(s)} (s={s:.4g})"
    if _marker(root, target, s).is_file():
        raise Refusal(f"{tag}: {_marker(root, target, s)} is down — an interrupted bake, not a cell (re-run the bake)")
    obj = cdir / "object.pt"
    if not obj.is_file():
        raise Refusal(f"{tag}: no object.pt at {obj}")
    for name in (IDENTITY_SIDECAR, "meta.json", DOSE_SIDECAR, "staged_from.json"):
        if not (cdir / name).is_file():
            raise Refusal(f"{tag}: {name} missing from {cdir}")
    present = [x for x in br.IDENTITY_SIDECARS if (cdir / x).is_file()]
    if present != [IDENTITY_SIDECAR]:
        raise Refusal(f"{tag}: identity sidecars {present} != ['{IDENTITY_SIDECAR}'] (exactly one)")
    if (cdir / IDENTITY_SIDECAR).read_bytes() != (parent.dir / IDENTITY_SIDECAR).read_bytes():
        raise Refusal(f"{tag}: {IDENTITY_SIDECAR} is not byte-identical to the parent's")
    block = br.load_json(cdir / DOSE_SIDECAR)
    if block.get("schema") != SIDECAR_SCHEMA:
        raise Refusal(f"{tag}: {DOSE_SIDECAR} schema {block.get('schema')!r} != {SIDECAR_SCHEMA!r}")
    if block.get("strength") != s or block.get("lambda") != s:
        raise Refusal(f"{tag}: {DOSE_SIDECAR} records strength {block.get('strength')!r} != {s!r}")
    if block.get("parent_object_sha256") != parent.sha:
        raise Refusal(f"{tag}: {DOSE_SIDECAR} names parent {str(block.get('parent_object_sha256'))[:12]} but the parent "
                      f"on disk is {parent.sha[:12]}")
    want_mech = MECHANISM_ANCHOR if is_anchor(s) else MECHANISM_SCALED
    if block.get("mechanism") != want_mech:
        raise Refusal(f"{tag}: {DOSE_SIDECAR} records mechanism {block.get('mechanism')!r} != {want_mech!r}")
    data = obj.read_bytes()
    sha = br.sha256_bytes(data)
    if sha != block.get("object_sha256"):
        raise Refusal(f"{tag}: object.pt hashes to {sha[:12]} but {DOSE_SIDECAR} records "
                      f"{str(block.get('object_sha256'))[:12]} — the bytes changed after the bake")
    staged = br.load_json(cdir / "staged_from.json")
    if staged.get("object_sha256") != sha:
        raise Refusal(f"{tag}: staged_from.json object_sha256 != on-disk sha")
    for name, want in (staged.get("staged_shas") or {}).items():
        p = cdir / name
        if not p.is_file() or br.sha256_file(p) != want:
            raise Refusal(f"{tag}: staged_from.json names {name} at {want[:12]} but the file is missing or differs")
    meta = br.load_json(cdir / "meta.json")
    for key, want in (("cell_id", CELL_ID), ("arm", CELL_ID), ("class", "B"), ("si_id", None)):
        if meta.get(key, None) != want:
            raise Refusal(f"{tag}: meta.json {key}={meta.get(key)!r} != {want!r}")
    if (meta.get("strength") or {}).get("strength") != s:
        raise Refusal(f"{tag}: meta.json strength block does not record {s!r}")
    if is_anchor(s):
        verification = _anchor_checks(parent, cdir)
        mech = MECHANISM_ANCHOR
    else:
        if obj.is_symlink():
            raise Refusal(f"{tag}: a scaled cell's object.pt is a symlink — only the anchor links to the parent")
        if sha == parent.sha:
            raise Refusal(f"{tag}: a scaled cell's bytes hash to the parent's sha")
        try:
            blob = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            raise Refusal(f"{tag}: object.pt does not load: {e}") from e
        verification = br.verify_rung(parent, blob, data, s, probe_rows=probe_rows, seed=seed)
        ops_r = br.adapter_ops(br.resolved_spec_of(blob), data)
        verification["checks"].append(verify_sign(parent, ops_r, s, probe_rows=probe_rows, seed=seed))
        verification["checks"].append(verify_layers(parent, blob, tag=tag))
        verification["strength"] = s
        mech = MECHANISM_SCALED
    if verification["intervention_hash"] != block.get("intervention_hash"):
        raise Refusal(f"{tag}: intervention_hash {verification['intervention_hash']} != recorded "
                      f"{block.get('intervention_hash')}")
    return _row(parent, s, cdir, sha, verification, mechanism=mech, status="VERIFIED")


def _existing_cell(parent: "br.Parent", root: Path, s: float, *, probe_rows: int, seed: int) -> Optional[dict]:
    """A complete previous bake of THIS strength off THIS parent — re-verified and KEPT; anything partial
    or different returns None (and the caller rebuilds)."""
    cdir = strength_cell_dir(root, parent.target, s)
    if _marker(root, parent.target, s).is_file():
        return None
    if not (cdir / DOSE_SIDECAR).is_file() or not (cdir / "object.pt").is_file():
        return None
    try:
        block = br.load_json(cdir / DOSE_SIDECAR)
    except Refusal:
        return None
    if block.get("strength") != float(s) or block.get("parent_object_sha256") != parent.sha:
        return None
    if br.sha256_file(cdir / "object.pt") != block.get("object_sha256"):
        return None
    row = check_cell(parent, root, s, probe_rows=probe_rows, seed=seed)
    row["status"] = "KEPT"
    return row


# --------------------------------------------------------------------------- per target

def bake_target(root: Path, target: str, strengths: List[float], *, lane_name: str, force: bool, check: bool,
                probe_rows: int, seed: int) -> dict:
    import torch

    ln = lane_of(lane_name)
    roster = load_roster(ln)
    tinfo = require_target(ln, target, roster)           # the lane's roster is THE target set
    parent = br.Parent(root, target)              # MissingObject / Refusal propagate
    view = tinfo.get("campaign_view")
    declared = roster_parent_layers(tinfo)
    if declared is not None and declared != parent.layers:
        raise Refusal(f"{target}: the lane roster declares parent_layers {declared} but the parent blob's layers are "
                      f"{parent.layers} — the roster and the object disagree about the fit; refusing")
    bank = strength_bank_dir(root, target)
    rec_path = bake_record_path(root, target)
    gaps = layer_gaps(parent.layers)
    print(f"[strength] {target} [lane {ln.name}, {ln.parent_lineage}]: parent {parent.sha[:12]} layers={len(parent.layers)} "
          f"{parent.layers} gaps={gaps or 'none (contiguous)'} d={parent.d} campaign view={view}; "
          f"strengths {[float(s) for s in strengths]} -> views {[view_name(s) for s in strengths]}", flush=True)
    rows: List[dict] = []
    if check:
        if not rec_path.is_file():
            raise Refusal(f"{target}: no {rec_path} — nothing to check (bake first)")
        prev = br.load_json(rec_path)
        if prev.get("schema") not in ACCEPTED_SCHEMAS:
            raise Refusal(f"{target}: {rec_path} schema {prev.get('schema')!r} not in {ACCEPTED_SCHEMAS!r}")
        if prev.get("lane", DEFAULT_LANE) != ln.name:
            raise Refusal(f"{target}: {rec_path} was baked under lane {prev.get('lane', DEFAULT_LANE)!r}, not {ln.name!r}")
        if prev.get("sha256") != record_sha(prev):
            raise Refusal(f"{target}: {rec_path} sha256 does not verify — an edited or torn record")
        if prev.get("parent", {}).get("object_sha256") != parent.sha:
            raise Refusal(f"{target}: {rec_path} was baked off parent {str(prev.get('parent', {}).get('object_sha256'))[:12]} "
                          f"but the parent on disk is {parent.sha[:12]}")
        if [float(x) for x in prev.get("strengths", [])] != [float(s) for s in strengths]:
            raise Refusal(f"{target}: {rec_path} strengths {prev.get('strengths')} != requested {strengths}")
        prev_rows = {r["staged_id"]: r for r in prev.get("cells", [])}
        for s in strengths:
            row = check_cell(parent, root, s, probe_rows=probe_rows, seed=seed)
            sid = staged_id(s)
            if sid not in prev_rows or prev_rows[sid].get("object_sha256") != row["object_sha256"]:
                raise Refusal(f"{target} {sid}: object sha {row['object_sha256'][:12]} != the bake record's "
                              f"{str(prev_rows.get(sid, {}).get('object_sha256'))[:12]}")
            rows.append(row)
            print(f"[check] {target} {view_name(s)} s={s:.4g} sha {row['object_sha256'][:12]} VERIFIED "
                  f"({', '.join(c['check'] + ':' + c['status'] for c in row['verification']['checks'])})", flush=True)
        return {"target": target, "status": "VERIFIED", "cells": rows, "record": str(rec_path)}
    bank.mkdir(parents=True, exist_ok=True)
    for s in strengths:
        if is_anchor(s):
            row = bake_anchor(parent, root, force=force, probe_rows=probe_rows, seed=seed)
        else:
            row = bake_scaled(parent, root, s, force=force, probe_rows=probe_rows, seed=seed)
        rows.append(row)
        print(f"[strength] {target} {view_name(s)} s={s:.4g} sha {row['object_sha256'][:12]} <- parent {parent.sha[:12]} "
              f"{row['status']} [{row['mechanism']}] "
              f"({', '.join(c['check'] + ':' + c['status'] for c in row['verification']['checks'])})", flush=True)
    record = {
        "schema": SCHEMA, "lane": ln.name, "parent_lineage": ln.parent_lineage, "roster": ln.roster_rel,
        "target": target, "campaign_view": view, "root": str(root), "bank": str(bank),
        "cell_id": CELL_ID, "parent": parent.record(),
        "parent_layers": parent.layers, "parent_layer_gaps": gaps, "parent_layers_contiguous": not gaps,
        "roster_parent_layers": declared,
        "strengths": [float(s) for s in strengths], "views": [view_name(s) for s in strengths],
        "anchor_strength": ANCHOR_STRENGTH,
        "mechanism": {"scaled": br.mechanism_note(),
                      "anchor": {"name": MECHANISM_ANCHOR,
                                 "what": "object.pt is a symlink to the parent's object.pt; no bytes are written or scaled"}},
        "cells": [{k: v for k, v in r.items() if k != "status"} for r in rows],
        "producer": PRODUCER, "code_sha": br.code_sha(), "torch_version": torch.__version__,
        "written_at": br.now_utc(),
    }
    record["sha256"] = record_sha(record)
    br.write_json_atomic(rec_path, record)
    print(f"[strength] {target}: record {rec_path} ({len(rows)} cells; "
          f"{sum(r['status'] == 'BAKED' for r in rows)} baked, {sum(r['status'] == 'KEPT' for r in rows)} kept)", flush=True)
    return {"target": target, "status": "BAKED", "cells": rows, "record": str(rec_path)}


def _targets(values: List[str], lane_name: str) -> List[str]:
    """`--target` values (repeat or comma-separate), or the lane roster's WHOLE target set when none is given;
    every named target must be in the lane's roster (refused loudly, before touching disk)."""
    ln = lane_of(lane_name)
    roster = load_roster(ln)
    known = lane_targets(ln, roster)
    out: List[str] = []
    for v in values:
        for t in v.split(","):
            t = t.strip()
            if t and t not in out:
                out.append(t)
    if not out:
        return known
    for t in out:
        require_target(ln, t, roster)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lane", default=DEFAULT_LANE, choices=sorted(LANES),
                    help="which lambda=1 parents (as22_strength.LANES; the lane's roster IS its target set)")
    ap.add_argument("--target", action="append", default=[],
                    help="target key (repeat, or comma-separate); default: every target of the lane's roster")
    ap.add_argument("--root", required=True, type=Path,
                    help="the objects root, e.g. $SHARED/exp/objects/alphasteer_f22[_rebake] (parent at "
                         "<root>/<target>/cells/alphasteer, the strength bank under <root>/<target>/strength/)")
    ap.add_argument("--strengths", action="append", default=[],
                    help="nominal base_scale values (repeat or comma-separate; write --strengths=-0.5,... since a value "
                         f"starting with `-` is otherwise read as an option); default: the ladder {list(STRENGTHS)}. "
                         "0, NaN, inf refuse; 1.0 is the anchor (the parent, symlinked)")
    ap.add_argument("--force", action="store_true", help="re-bake over a previous bake of ours")
    ap.add_argument("--check", action="store_true", help="re-verify an existing bake; write nothing")
    ap.add_argument("--probe-rows", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    targets = _targets(args.target, args.lane)
    strengths = parse_strengths(args.strengths)        # refuses 0 / NaN / inf / duplicates before touching disk
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[strength] REFUSED: --root {root} is not a directory", file=sys.stderr)
        return EXIT_REFUSED
    if root.name != lane_of(args.lane).bank_root_name:
        print(f"[strength] REFUSED: --root {root} is not the lane's bank root ({lane_of(args.lane).bank_root_rel}) — a "
              "lane's parents live under its own root; pass the matching --lane", file=sys.stderr)
        return EXIT_REFUSED
    outcomes: Dict[str, str] = {}
    rc = EXIT_OK
    for t in targets:
        try:
            res = bake_target(root, t, strengths, lane_name=args.lane, force=args.force, check=args.check,
                              probe_rows=args.probe_rows, seed=args.seed)
            outcomes[t] = res["status"]
        except MissingObject as e:
            outcomes[t] = "SKIPPED_NO_OBJECT"
            print(f"[strength] SKIP {e}", flush=True)
            if rc == EXIT_OK:
                rc = EXIT_NO_OBJECT
        except Refusal as e:
            outcomes[t] = "REFUSED"
            print(f"[strength] REFUSED {t}: {e}", file=sys.stderr, flush=True)
            rc = EXIT_REFUSED
    print("[strength] outcomes: " + ", ".join(f"{t}={s}" for t, s in outcomes.items()) + f"; exit {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
