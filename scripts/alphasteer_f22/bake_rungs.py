#!/usr/bin/env python
"""alphasteer-full22 Phase C — bake the six λ-scaled RUNG objects of one target's served λ ladder
and lay out one per-rung defenses dir each (LADDER_CONTRACT.md "Per-rung directory").

THE GRID (recomputed here from the fit's own record, never typed in):
    λ_1 = floor / min_l fit_meta.per_layer[l].rho_model_dtype.p05        (floor = 3e-3)
    λ_i = λ_1 · 2^(i-1),  i = 1..n_rungs (6)
The fit-side ρ per layer at each rung is λ_i × the λ=1 value (rho_note: "rho(lambda) = lambda *
rho(1)"), recorded per rung (fp32 AND model dtype, p05/p50/p95) as the ladder record's PROPOSAL
column. It decides nothing: condition (c) is read off the served ρ record on each rung's own
standup. The grid arithmetic assumes the parent was fit at base_scale 1.0 (fit_meta
base_scale_source "uncalibrated_fit"); any other parent scale refuses.

THE MECHANISM — verified against THIS tree's served adapter, not assumed. `lens_adapters.
_adapt_alphasteer` reads the blob's `base_scale` and emits, per fit layer, one
`HookOp(op=OP_PREFILL_LAST_MATMUL, matrix=M_l (fp32 at adapt, cast to the serving dtype at
materialize), coeff=base_scale, phase=prefill)`; the request-time body `_apply_one` computes
`eff = scale * op.coeff; out + (eff * (h_last @ M)).unsqueeze(-2)`. So `base_scale` reaches the
served op AS THE COEFF and the matrices are untouched: a rung is the parent blob with ONLY the
scalar `base_scale` rewritten to λ (same keys, same key order, every tensor bit-identical), and
the served edit is exactly λ × the λ=1 edit (the coeff multiplies the identical `h_last @ M`).
The matrices are NOT pre-multiplied — that would put a differently-rounded bf16 matrix under the
hook and change what "λ× the fitted edit" means. Should `base_scale` ever stop reaching the op,
the adapter-level check below fails and the bake refuses; it does not silently fall back to
scaling matrices.

VERIFICATION (every rung, reloaded FROM DISK; any failure refuses with a non-zero exit and no
bake record is written — the halfdose `scale_psr_dose.py` discipline):
  1. tensors: key set + ORDER equal to the parent's; every `Ms[l]` (and `refusals[l]`) torch.equal
     to the parent's, fp32; every non-tensor field equal EXCEPT `base_scale`;
  2. scalar: `base_scale == λ` exactly (the Python float that IS the grid value); parent's == 1.0;
  3. served translation: both blobs through `lens_adapters.get_adapter("alphasteer")` (the guarded
     registry door every serving path uses) under the resolved spec registry_adapter derives from
     the blob's own non-tensor fields: same op count; per op identical (op, layer, positions,
     skip_bos, phase, header_ids), matrix torch.equal, rung coeff == λ, parent coeff == 1.0;
  4. served edit on a fixed CPU probe batch through the REAL `apply_hook_ops` under a prefill
     HookContext, fp32 and bf16: the rung's edit == λ × the parent's, torch.equal (exact — the same
     `python-float × tensor` op on the identical `h_last @ M`).
  Plus the per-rung `petri_intervention_hash` (the resolved spec carries base_scale, so each rung
  hashes differently — the point of serving them side by side).

LAYOUT, per rung (LADDER_CONTRACT.md; what `mt_serving_up.py` / `lens_gate_standup.py` →
`registry_from_explicit_cells` → `train_budget.fit_provenance_problems` and `_bank_objects` read):
    <root>/<target>/ladder/lam<i>/cells_view/<view>/alphasteer/
        object.pt              the rung blob
        fit_provenance.json    the parent's, byte-identical WHEN the parent's record self-identifies
                               by NOTHING (no object sha, no cell id) — the ONE identity sidecar, and
                               nothing is re-bound; train_budget_meta.json and derived_meta.json are
                               never written here. WHEN THE PARENT'S RECORD CARRIES AN object_sha256
                               the byte copy is REFUSED: it would make the rung assert its parent's
                               identity, which is what a downstream mirror check reads. Such a rung
                               gets a REBOUND record instead — the child's own sha, the parent's
                               record preserved verbatim under `derived_from.record`, and the stage-3
                               keys carried to the top level. Met first on llama8b, whose parent came
                               from a slice producer that DOES bind a sha (2026-09-13).
        meta.json              the parent's cell identity (cell_id/arm/class/si_id — the
                               registry_adapter._check_meta_json keys) + a `ladder` block
        dose_scaled_from.json  the provenance sidecar: parent sha, λ, mechanism, verification
        staged_from.json       the staging record (stage_full22.sh hashcheck_and_stamp schema)
    <root>/<target>/ladder/lam<i>/cells_view/<view>/VIEW_MANIFEST.json
    <root>/<target>/ladder/ladder_bake.json     the bake record (written LAST; --check requires it)
Every file lands by tmp + os.replace. No COMPLETE marker (a rung is not a publication of a fit).

EXIT CODES: 0 every requested target baked / kept / verified; 1 a refusal (a failed check, a torn
parent, a dir that is not ours); 3 no refusal but at least one target had NO fitted object yet
(`<root>/<target>/cells/alphasteer/object.pt` absent) — a clean per-target SKIP, never a
traceback, so a multi-target run keeps going and the caller can tell "not yet" from "wrong".

usage:
  python scripts/alphasteer_f22/bake_rungs.py --target gemma4e4b --root $SHARED/exp/objects/alphasteer_f22
  python scripts/alphasteer_f22/bake_rungs.py --target gemma4e4b,qwen35_4b,glm46vflash --root ... --check
  options: --rungs 6  --floor 3e-3  --force (re-bake over a previous bake of ours)  --probe-rows 6  --seed 0
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_HERE = Path(__file__).resolve().parent
REPO = _HERE.parents[1]
for _p in (str(REPO / "src"), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from as22_targets import (CELL_ID, ladder_dir, parent_cell_dir, rung_cell_dir,  # noqa: E402
                          rung_dir, rung_view_dir, view_of)

SCHEMA = "alphasteer_f22_ladder_bake/1"
PRODUCER = "scripts/alphasteer_f22/bake_rungs.py"
BAKE_RECORD = "ladder_bake.json"
DOSE_SIDECAR = "dose_scaled_from.json"
IDENTITY_SIDECARS = ("train_budget_meta.json", "derived_meta.json", "fit_provenance.json")
STAGE3_FIT_REQUIRED_KEYS = ("resolved_model_id", "resolved_revision", "chat_template_hash", "writer")
FIT_META_SCHEMA = "alphasteer_fit_meta/1"
DEFAULT_FLOOR = 3e-3
DEFAULT_RUNGS = 6
MECHANISM = "scalar_base_scale"

EXIT_OK, EXIT_REFUSED, EXIT_NO_OBJECT = 0, 1, 3


class Refusal(Exception):
    """A verification / identity failure: the bake stops, exit 1."""


class MissingObject(Exception):
    """The target has no fitted object yet: a clean per-target skip, exit 3."""


# --------------------------------------------------------------------------- small helpers

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def code_sha() -> Optional[str]:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_HERE, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001
        return None


def _tmp_for(path: Path) -> Path:
    return path.with_name(f".tmp-{path.name}-{os.getpid()}")


def write_bytes_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(path)
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def write_json_atomic(path: Path, obj: Any, *, sort_keys: bool = True) -> None:
    write_bytes_atomic(path, (json.dumps(obj, indent=1, sort_keys=sort_keys) + "\n").encode("utf-8"))


def torch_save_atomic(blob: dict, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_for(path)
    with open(tmp, "wb") as f:
        torch.save(blob, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def blob_bytes(blob: dict) -> bytes:
    import torch

    buf = io.BytesIO()
    torch.save(blob, buf)
    return buf.getvalue()


def load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise Refusal(f"unreadable JSON {p}: {e}") from e


def lambda_4sf(lam: float) -> float:
    return float(f"{lam:.4g}")


# --------------------------------------------------------------------------- the grid (plan C.1)

def lambda_grid(fit_meta: dict, *, n_rungs: int = DEFAULT_RUNGS, floor: float = DEFAULT_FLOOR) -> dict:
    """λ_1 = floor / min_l rho_model_dtype.p05(λ=1); λ_i = λ_1 · 2^(i-1). Recomputed from the fit's
    own per-layer record; refuses a record whose per_layer keys are not exactly its layers, or a
    parent not fit at base_scale 1.0 (the λ=1 reference the proposal is stated against)."""
    if not isinstance(fit_meta, dict) or fit_meta.get("schema") != FIT_META_SCHEMA:
        raise Refusal(f"fit_meta schema {fit_meta.get('schema') if isinstance(fit_meta, dict) else None!r} "
                      f"!= {FIT_META_SCHEMA!r}")
    if float(fit_meta.get("base_scale", float("nan"))) != 1.0:
        raise Refusal(f"fit_meta base_scale {fit_meta.get('base_scale')!r} != 1.0 — the λ grid is stated "
                      "against the λ=1 fit-side ρ (rho(lambda) = lambda * rho(1)); refusing to bake a "
                      "ladder off a pre-scaled parent")
    layers = [int(l) for l in fit_meta.get("layers") or []]
    if not layers or len(set(layers)) != len(layers):
        raise Refusal(f"fit_meta.layers {fit_meta.get('layers')!r} is empty or has duplicates")
    per_layer = fit_meta.get("per_layer") or {}
    if sorted(int(k) for k in per_layer) != sorted(layers):
        raise Refusal(f"fit_meta.per_layer keys {sorted(per_layer)} != fit_meta.layers {sorted(layers)}")
    if int(n_rungs) < 1:
        raise Refusal(f"n_rungs must be >= 1, got {n_rungs}")
    if not (float(floor) > 0):
        raise Refusal(f"floor must be > 0, got {floor}")
    p05_md, p05_f32 = {}, {}
    for l in layers:
        rec = per_layer[str(l)]
        try:
            p05_md[l] = float(rec["rho_model_dtype"]["p05"])
            p05_f32[l] = float(rec["rho_fp32"]["p05"])
        except (KeyError, TypeError) as e:
            raise Refusal(f"fit_meta.per_layer[{l}] lacks rho_model_dtype/rho_fp32 p05: {e}") from e
        if not (p05_md[l] > 0) or not (p05_f32[l] > 0):
            raise Refusal(f"fit_meta.per_layer[{l}] rho p05 is not > 0 (model dtype {p05_md[l]}, fp32 "
                          f"{p05_f32[l]}) — a layer with no edit cannot set a floor")
    argmin = min(layers, key=lambda l: (p05_md[l], l))
    lam1 = float(floor) / p05_md[argmin]
    lambdas = [lam1 * (2.0 ** (i - 1)) for i in range(1, int(n_rungs) + 1)]
    return {
        "rule": "lambda_1 = floor / min_l fit_meta.per_layer[l].rho_model_dtype.p05 (lambda=1); "
                "lambda_i = lambda_1 * 2^(i-1), i = 1..n_rungs (plan C.1)",
        "floor": float(floor), "n_rungs": int(n_rungs),
        "argmin_layer": argmin,
        "min_rho_p05_model_dtype_at_unit": p05_md[argmin],
        "min_rho_p05_fp32_at_unit": min(p05_f32.values()),
        "argmin_layer_fp32": min(layers, key=lambda l: (p05_f32[l], l)),
        "lambda_1": lam1, "lambda_1_4sf": lambda_4sf(lam1),
        "lambdas": lambdas, "lambdas_4sf": [lambda_4sf(x) for x in lambdas],
        "proposal_only": "fit-side, UNSTEERED capture (plan B.7/C.1): decides nothing; condition (c) is "
                         "the SERVED per-layer rho on each rung's own standup (D25)",
    }


def fit_rho_at(fit_meta: dict, lam: float) -> Tuple[Dict[str, dict], Dict[str, dict]]:
    """Per fit layer, the fit-side ρ table at rung λ: λ × the λ=1 p05/p50/p95, fp32 and model dtype
    (`rho_fit_fp32_by_layer` / `rho_fit_model_dtype_by_layer`)."""
    f32, md = {}, {}
    for l in fit_meta["layers"]:
        rec = fit_meta["per_layer"][str(l)]
        f32[str(l)] = {q: float(lam) * float(rec["rho_fp32"][q]) for q in ("p05", "p50", "p95")}
        md[str(l)] = {q: float(lam) * float(rec["rho_model_dtype"][q]) for q in ("p05", "p50", "p95")}
    return f32, md


# --------------------------------------------------------------------------- the parent

def fit_meta_fingerprint(body: Dict[str, Any]) -> str:
    """The value the fit records as ``fit_meta.sha256``: alphasteer_screen.write_fingerprinted_json's
    sha256 over its canonical JSON (``sort_keys=True, indent=1``) of everything BUT ``sha256``, with the
    ``fit_meta.json.COMPLETE`` marker repeating it. It is the RECORD's fingerprint, not the object's
    hash: the publication records no object.pt sha at all, and the object<->record tie is the content
    cross-checks in Parent (layers, chosen.layers, base_scale, site, per-layer hidden_dim). Replicated
    here because this script is stdlib+torch by design and the writer lives on the fit branch; pinned
    against the three real Phase B records on 2026-09-09 after jobs 84542-84544 died on the misreading
    (fit_meta.sha256 compared to object.pt's bytes)."""
    b = dict(body)
    b.pop("sha256", None)
    return hashlib.sha256(json.dumps(b, sort_keys=True, indent=1, default=str).encode("utf-8")).hexdigest()


def fit_meta_fingerprint_ok(fit_meta: Dict[str, Any]) -> bool:
    want = fit_meta.get("sha256")
    return bool(want) and fit_meta_fingerprint(fit_meta) == want


class Parent:
    """The Phase B fit, loaded and identity-checked (every refusal names what disagreed)."""

    def __init__(self, root: Path, target: str):
        import torch

        self.target = target
        self.dir = parent_cell_dir(root, target)
        self.object_path = self.dir / "object.pt"
        if not self.object_path.is_file():
            raise MissingObject(f"{target}: no fitted object at {self.object_path} — the Phase B fit has not "
                                "landed for this target (skipped; nothing baked)")
        for name in ("fit_meta.json", "fit_provenance.json", "meta.json"):
            if not (self.dir / name).is_file():
                raise Refusal(f"{target}: object.pt present but {self.dir / name} missing — a torn fit "
                              "publication; refusing to bake off it")
        self.bytes = self.object_path.read_bytes()
        self.sha = sha256_bytes(self.bytes)
        self.fit_meta = load_json(self.dir / "fit_meta.json")
        self.fit_provenance = load_json(self.dir / "fit_provenance.json")
        self.meta = load_json(self.dir / "meta.json")
        self.sidecar_shas = {name: sha256_file(self.dir / name)
                             for name in ("fit_meta.json", "fit_provenance.json", "meta.json")}
        # the record's OWN fingerprint (fit_meta_fingerprint), and the publication marker that repeats it.
        # NOT compared to object.pt: the fit records no object sha, and the object<->record tie is the
        # content cross-checks below. The first submission compared the two and refused every target.
        rec_sha = self.fit_meta.get("sha256")
        if not fit_meta_fingerprint_ok(self.fit_meta):
            raise Refusal(f"{target}: fit_meta.sha256 {str(rec_sha)[:12]} is not the fingerprint of the record's "
                          f"own body ({fit_meta_fingerprint(self.fit_meta)[:12]}; write_fingerprinted_json's canonical "
                          "JSON) — an edited or torn fit_meta.json; refusing")
        marker = self.dir / "fit_meta.json.COMPLETE"
        if not marker.is_file():
            raise Refusal(f"{target}: no {marker.name} beside fit_meta.json — the fit writes the marker STRICTLY "
                          "after the payload, so its absence is an incomplete publication; refusing")
        content = marker.read_text(encoding="utf-8").strip()
        if content != rec_sha:
            raise Refusal(f"{target}: {marker.name} records {content[:12]} but fit_meta.sha256 is "
                          f"{str(rec_sha)[:12]} — torn publication; refusing")
        missing = [k for k in STAGE3_FIT_REQUIRED_KEYS if not self.fit_provenance.get(k)]
        if missing:
            raise Refusal(f"{target}: fit_provenance.json lacks (or carries null) {missing} — the stage-3 "
                          "identity record the bank view requires; refusing")
        others = [s for s in IDENTITY_SIDECARS if s != "fit_provenance.json" and (self.dir / s).is_file()]
        if others:
            raise Refusal(f"{target}: parent dir carries fit_provenance.json AND {others} — a dir declares "
                          "exactly one identity; refusing to pick one")
        for key, want in (("cell_id", CELL_ID), ("arm", CELL_ID), ("class", "B"), ("si_id", None)):
            if self.meta.get(key, None) != want:
                raise Refusal(f"{target}: meta.json {key}={self.meta.get(key)!r} != {want!r} — not the "
                              "alphasteer Class B cell (D10 mis-attribution); refusing")
        for key, want in (("cell_id", CELL_ID), ("arm", CELL_ID), ("site", "pre"),
                          ("steer_positions", "prefill")):
            if self.fit_meta.get(key) != want:
                raise Refusal(f"{target}: fit_meta.{key}={self.fit_meta.get(key)!r} != {want!r}")
        try:
            self.blob = torch.load(io.BytesIO(self.bytes), map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            raise Refusal(f"{target}: object.pt does not load as a weights-only torch blob: {e}") from e
        if not isinstance(self.blob, dict) or self.blob.get("kind") != CELL_ID:
            raise Refusal(f"{target}: object.pt is not an alphasteer blob (kind="
                          f"{self.blob.get('kind') if isinstance(self.blob, dict) else None!r})")
        Ms = self.blob.get("Ms")
        if not isinstance(Ms, dict) or not Ms:
            raise Refusal(f"{target}: object.pt carries no plural 'Ms' mapping — a legacy single-layer blob "
                          "is not what the multi-layer fit writes; refusing")
        self.layers = sorted(int(l) for l in Ms)
        declared = sorted(int(l) for l in self.blob.get("layers") or [])
        if declared != self.layers:
            raise Refusal(f"{target}: blob.layers {declared} != Ms keys {self.layers}")
        if sorted(int(l) for l in self.fit_meta["layers"]) != self.layers:
            raise Refusal(f"{target}: fit_meta.layers {sorted(self.fit_meta['layers'])} != blob layers "
                          f"{self.layers}")
        chosen = (self.fit_meta.get("chosen") or {}).get("layers")
        if chosen is not None and sorted(int(l) for l in chosen) != self.layers:
            raise Refusal(f"{target}: fit_meta.chosen.layers {sorted(chosen)} != blob layers {self.layers}")
        if float(self.blob.get("base_scale", float("nan"))) != 1.0:
            raise Refusal(f"{target}: blob base_scale {self.blob.get('base_scale')!r} != 1.0 (fit_meta says "
                          f"{self.fit_meta.get('base_scale')!r}) — the grid is stated against the λ=1 fit")
        if self.blob.get("site") != "pre" or self.blob.get("steer_positions") != "prefill":
            raise Refusal(f"{target}: blob site={self.blob.get('site')!r} steer_positions="
                          f"{self.blob.get('steer_positions')!r} — expected pre / prefill")
        d = None
        for l in self.layers:
            M = Ms[l] if l in Ms else Ms[str(l)]
            if not isinstance(M, torch.Tensor) or M.dim() != 2 or M.shape[0] != M.shape[1]:
                raise Refusal(f"{target}: Ms[{l}] is not a square 2-D tensor")
            if M.dtype != torch.float32:
                raise Refusal(f"{target}: Ms[{l}] dtype {M.dtype} != float32 (the fp32-on-CPU blob contract)")
            if not torch.isfinite(M).all():
                raise Refusal(f"{target}: Ms[{l}] carries non-finite values")
            d = int(M.shape[0]) if d is None else d
            if int(M.shape[0]) != d:
                raise Refusal(f"{target}: Ms[{l}] is {tuple(M.shape)} but d={d} at the first layer")
            hd = self.fit_meta["per_layer"][str(l)].get("hidden_dim")
            if hd is not None and int(hd) != d:
                raise Refusal(f"{target}: fit_meta.per_layer[{l}].hidden_dim {hd} != matrix d {d}")
        self.d = d
        self.spec = resolved_spec_of(self.blob)
        self.ops = adapter_ops(self.spec, self.bytes)

    def record(self) -> dict:
        fm = self.fit_meta
        return {"dir": str(self.dir), "object_sha256": self.sha, "sidecar_sha256": self.sidecar_shas,
                "base_scale": float(self.blob["base_scale"]), "base_scale_source": fm.get("base_scale_source"),
                "layers": self.layers, "n_layers": len(self.layers), "d": self.d,
                "site": self.blob["site"], "steer_positions": self.blob["steer_positions"],
                "model_id": fm.get("model_id"), "model_revision": fm.get("model_revision"),
                "chat_template_hash": fm.get("chat_template_hash"), "fit_code_sha": fm.get("code_sha"),
                "fit_mode": fm.get("fit_mode"), "screen_f9_sha256": fm.get("screen_f9_sha256"),
                "split_sha256": fm.get("split_sha256"), "leakage_report_sha256": fm.get("leakage_report_sha256"),
                "fit_provenance": dict(self.fit_provenance)}


# --------------------------------------------------------------------------- the served translation

def resolved_spec_of(blob: dict) -> dict:
    """The resolved spec registry_adapter derives for a torch-blob artifact (`_resolve_torch_blob`,
    restated over its own `_json_safe`): every non-tensor stored field + kind + arm. A field the
    adapter cannot serialise refuses here exactly as enumeration would."""
    from sisteer.capeval.petri import registry_adapter as ra

    spec: Dict[str, Any] = {}
    for key, value in blob.items():
        if key == "kind":
            continue
        safe = ra._json_safe(value, f"blob[{key!r}]")
        if safe is ra._TENSOR:
            continue
        spec[key] = safe
    spec.update(kind=CELL_ID, arm=CELL_ID)
    return spec


def adapter_ops(spec: dict, artifact: bytes):
    from sisteer.capeval.serving import lens_adapters as la

    return la.get_adapter(CELL_ID)(spec, artifact, b"")


def intervention_hash(spec: dict, artifact: bytes) -> str:
    from sisteer.capeval import manifest as cm

    return cm.petri_intervention_hash(spec, artifact_bytes=artifact, sidecar_bytes=b"")


def scaled_blob(parent_blob: dict, lam: float) -> dict:
    """The rung blob: the parent's mapping in the parent's key ORDER with only `base_scale` set to λ.
    Tensors are the parent's objects (no copy, no arithmetic)."""
    return {k: (float(lam) if k == "base_scale" else v) for k, v in parent_blob.items()}


def _op_identity(op) -> tuple:
    return (op.op, op.layer, op.positions, op.skip_bos, op.phase, op.header_ids)


def verify_rung(parent: Parent, rung_blob: dict, rung_bytes: bytes, lam: float, *,
                probe_rows: int, seed: int) -> dict:
    """The four checks of the module docstring over a rung blob RELOADED from disk. Raises Refusal."""
    import torch
    from sisteer.capeval.serving import lens_adapters as la

    rec: Dict[str, Any] = {"lambda": float(lam), "mechanism": MECHANISM, "checks": []}
    pb = parent.blob
    # 1. tensors + every non-scalar field
    if list(rung_blob.keys()) != list(pb.keys()):
        raise Refusal(f"λ={lam:.4g}: key order {list(rung_blob)} != parent {list(pb)}")
    for l in parent.layers:
        a, b = rung_blob["Ms"][l], pb["Ms"][l]
        if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
            raise Refusal(f"λ={lam:.4g}: Ms[{l}] is not bit-identical to the parent's (dtype {a.dtype} vs "
                          f"{b.dtype}, shape {tuple(a.shape)} vs {tuple(b.shape)})")
    ra_, rb_ = rung_blob.get("refusals"), pb.get("refusals")
    if (ra_ is None) != (rb_ is None):
        raise Refusal(f"λ={lam:.4g}: refusals presence differs from the parent")
    if ra_ is not None:
        if sorted(ra_) != sorted(rb_):
            raise Refusal(f"λ={lam:.4g}: refusals keys differ from the parent")
        for l in ra_:
            if not torch.equal(ra_[l], rb_[l]):
                raise Refusal(f"λ={lam:.4g}: refusals[{l}] is not bit-identical to the parent's")
    for k in pb:
        if k in ("Ms", "refusals", "base_scale"):
            continue
        if rung_blob[k] != pb[k]:
            raise Refusal(f"λ={lam:.4g}: non-tensor field {k!r} changed ({rung_blob[k]!r} != {pb[k]!r})")
    rec["checks"].append({"check": "tensors", "status": "PASS", "scaled": [], "unchanged": ["Ms", "refusals"],
                          "n_matrices": len(parent.layers), "d": parent.d,
                          "rule": "list(keys) equal; torch.equal(rung.Ms[l], parent.Ms[l]) for every layer "
                                  "(fp32); refusals equal; every non-tensor field equal except base_scale"})
    # 2. the scalar
    if type(rung_blob["base_scale"]) is not float or rung_blob["base_scale"] != float(lam):
        raise Refusal(f"λ={lam:.4g}: rung base_scale {rung_blob['base_scale']!r} != λ {float(lam)!r}")
    if float(pb["base_scale"]) != 1.0:
        raise Refusal(f"λ={lam:.4g}: parent base_scale {pb['base_scale']!r} != 1.0")
    rec["checks"].append({"check": "scalar", "status": "PASS", "base_scale": float(lam), "parent_base_scale": 1.0,
                          "rule": "rung.base_scale == lambda exactly (Python float equality); parent == 1.0"})
    # 3. served translation through the wave's adapter
    spec_r = resolved_spec_of(rung_blob)
    if spec_r.get("base_scale") != float(lam):
        raise Refusal(f"λ={lam:.4g}: resolved spec base_scale {spec_r.get('base_scale')!r} != λ")
    ops_p, ops_r = parent.ops, adapter_ops(spec_r, rung_bytes)
    if len(ops_p) != len(ops_r) or len(ops_r) != len(parent.layers):
        raise Refusal(f"λ={lam:.4g}: served op count parent={len(ops_p)} rung={len(ops_r)} layers={len(parent.layers)}")
    for op_p, op_r in zip(ops_p, ops_r):
        if _op_identity(op_p) != _op_identity(op_r):
            raise Refusal(f"λ={lam:.4g}: served op identity differs: {_op_identity(op_p)} vs {_op_identity(op_r)}")
        if op_p.op != la.OP_PREFILL_LAST_MATMUL or op_p.phase != la.PHASE_PREFILL:
            raise Refusal(f"λ={lam:.4g}: served op is {op_p.op}/{op_p.phase}, not the prefill last-token matmul")
        if not torch.equal(op_p.matrix, op_r.matrix):
            raise Refusal(f"λ={lam:.4g}: served matrix at hook layer {op_r.layer} differs from the parent's")
        if op_r.coeff != float(lam) or op_p.coeff != 1.0:
            raise Refusal(f"λ={lam:.4g}: served coeff rung={op_r.coeff!r} parent={op_p.coeff!r} — base_scale did "
                          "not reach the served op as the coeff; the scalar mechanism does not hold on this tree")
    hook_layers = [op.layer for op in ops_r]
    if hook_layers != [l - 1 for l in parent.layers]:
        raise Refusal(f"λ={lam:.4g}: served hook layers {hook_layers} != fit layers - 1 {[l - 1 for l in parent.layers]}")
    rec["checks"].append({"check": "served_ops", "status": "PASS", "n_ops": len(ops_r),
                          "adapter": "sisteer.capeval.serving.lens_adapters.get_adapter('alphasteer')",
                          "op": la.OP_PREFILL_LAST_MATMUL, "phase": la.PHASE_PREFILL,
                          "hook_layers": hook_layers, "fit_layers": parent.layers,
                          "coeff_rung": float(lam), "coeff_parent": 1.0, "coeff_ratio": float(lam) / 1.0,
                          "rule": "same op count; per op identical (op, layer, positions, skip_bos, phase, "
                                  "header_ids); matrix torch.equal; coeff == lambda (rung) / 1.0 (parent); "
                                  "hook layer == fit layer - 1 (site=pre)"})
    # 4. the served edit on a fixed CPU probe batch through the real apply_hook_ops
    T = max(2, int(probe_rows))
    ctx = la.HookContext(phase=la.PHASE_PREFILL, positions=tuple(range(T)))
    edit_rec = {}
    for dtype in (torch.float32, torch.bfloat16):
        g = torch.Generator().manual_seed(int(seed))
        exact = True
        max_abs = 0.0
        for op_p, op_r in zip(ops_p, ops_r):
            (mp,) = la.materialize_ops([op_p], device="cpu", dtype=dtype)
            (mr,) = la.materialize_ops([op_r], device="cpu", dtype=dtype)
            hidden = torch.zeros(T, parent.d, dtype=dtype)
            # every row but the last is ZERO, so out[:-1] IS the broadcast edit with no rounding from
            # the residual add; the last row carries h_last (the row the edit is computed from)
            hidden[-1] = torch.randn(parent.d, generator=g, dtype=torch.float32).to(dtype)
            out_p = la.apply_hook_ops(hidden, [mp], 1.0, layer=mp.layer, ctx=ctx)
            out_r = la.apply_hook_ops(hidden, [mr], 1.0, layer=mr.layer, ctx=ctx)
            edit_p, edit_r = out_p[:-1], out_r[:-1]
            want = float(lam) * edit_p
            if not torch.equal(edit_r, want):
                exact = False
                max_abs = max(max_abs, float((edit_r.float() - want.float()).abs().max()))
            if torch.equal(out_r, out_p) and float(lam) != 1.0:
                raise Refusal(f"λ={lam:.4g}: the rung's served output equals the parent's at hook layer "
                              f"{mr.layer} ({dtype}) — the edit did not scale")
        if not exact:
            raise Refusal(f"λ={lam:.4g}: served edit != lambda x parent edit in {dtype} (max |diff| {max_abs})")
        edit_rec[str(dtype).replace("torch.", "")] = {"exact": True, "probe_rows": T}
    rec["checks"].append({"check": "served_edit", "status": "PASS", "seed": int(seed), **edit_rec,
                          "rule": "apply_hook_ops (prefill HookContext) on a probe batch whose last row is "
                                  "h_last and every other row zero: rung edit == lambda * parent edit, "
                                  "torch.equal, in fp32 and bf16"})
    rec["intervention_hash"] = intervention_hash(spec_r, rung_bytes)
    rec["status"] = "PASS"
    return rec


# --------------------------------------------------------------------------- the layout

def _recorded_hash_check(d: Path, sha: str) -> dict:
    """stage_full22.sh hashcheck_and_stamp's verdict (dose_scale_common._recorded_hash_check): every
    sidecar json's 64-hex values under sha/hash/digest/fingerprint keys; only a TOP-LEVEL
    object_sha256 claims to be this object's hash."""
    hexre = re.compile(r"^[0-9a-f]{64}$")
    keyre = re.compile(r"sha|hash|digest|fingerprint", re.I)
    entries: List[dict] = []

    def walk(o, path, fname):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k, fname)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]", fname)
        elif isinstance(o, str) and hexre.match(o) and keyre.search(path):
            entries.append({"file": fname, "key": path, "value": o, "matches_object": o == sha})

    for f in sorted(d.iterdir()):
        if f.is_file() and f.suffix == ".json" and f.name != "staged_from.json":
            walk(json.loads(f.read_text(encoding="utf-8")), "", f.name)
    own = [e for e in entries if e["key"] == "object_sha256"]
    status = ("match" if any(e["matches_object"] for e in entries)
              else "recorded_hashes_are_not_this_object" if entries else "none_recorded")
    return {"status": status,
            "object_sha256_recorded_match": (all(e["matches_object"] for e in own) if own else None),
            "object_sha256_recorded_in": sorted({e["file"] for e in own}), "entries": entries}


BAKING_MARKER = ".BAKING"


def _is_our_rung_dir(rdir: Path, cell_dir: Path) -> bool:
    """A COMPLETE bake of ours carries dose_scaled_from.json in the cell dir; an in-progress or
    interrupted one carries the `.BAKING` marker in the rung dir (written before anything else,
    removed after the last file lands). Either is ours to remove; nothing else is."""
    return (cell_dir / DOSE_SIDECAR).is_file() or (rdir / BAKING_MARKER).is_file()


def _clear_rung(rdir: Path, cell_dir: Path, *, force: bool) -> None:
    """Remove a previous bake of OURS (complete or interrupted — see _is_our_rung_dir — or a rung dir
    holding no object.pt at all); refuse anything else: another writer's data is never deleted."""
    if not rdir.exists():
        return
    if cell_dir.exists() and (cell_dir / "object.pt").is_file() and not _is_our_rung_dir(rdir, cell_dir):
        raise Refusal(f"{cell_dir} holds an object.pt without our {DOSE_SIDECAR} (and no {BAKING_MARKER} "
                      "marker) — not a previous bake of this producer; refusing to remove it")
    serve = rdir / "serve"
    if serve.exists() and not force:
        raise Refusal(f"{rdir} carries a standup's serve/ dir — a rung that has been stood up is not "
                      "re-baked without --force")
    shutil.rmtree(rdir)


def _rung_record_paths(root: Path, target: str, i: int) -> Tuple[Path, Path, Path]:
    return rung_dir(root, target, i), rung_view_dir(root, target, i), rung_cell_dir(root, target, i)


def bake_rung(parent: Parent, root: Path, i: int, lam: float, *, force: bool, probe_rows: int,
              seed: int) -> dict:
    import torch

    target, view = parent.target, view_of(parent.target)
    rdir, vdir, cdir = _rung_record_paths(root, target, i)
    existing = _existing_rung(parent, root, i, lam)
    if existing is not None and not force:
        return existing
    _clear_rung(rdir, cdir, force=force)
    cdir.mkdir(parents=True, exist_ok=False)
    # the in-progress marker goes down FIRST: an interrupted or refused bake leaves a dir this
    # producer recognises as its own (and removes on the next run), never a stranger's object.pt
    write_bytes_atomic(rdir / BAKING_MARKER, f"{PRODUCER} pid {os.getpid()} {now_utc()} lambda {lam!r}\n".encode())
    try:
        return _bake_rung_files(parent, root, i, lam, rdir, vdir, cdir, probe_rows=probe_rows, seed=seed)
    except Refusal:
        shutil.rmtree(rdir, ignore_errors=True)      # a refused rung leaves nothing behind
        raise


def _bake_rung_files(parent: Parent, root: Path, i: int, lam: float, rdir: Path, vdir: Path, cdir: Path, *,
                     probe_rows: int, seed: int) -> dict:
    import torch

    target, view = parent.target, view_of(parent.target)
    blob = scaled_blob(parent.blob, lam)
    obj = cdir / "object.pt"
    torch_save_atomic(blob, obj)
    # identity: the parent's fit_provenance.json byte-identical WHEN it self-identifies by nothing;
    # otherwise REBOUND to this rung (written below, once the rung's own sha exists — a record that
    # binds a sha cannot be written before the bytes it binds)
    parent_binds_sha = bool(parent.fit_provenance.get("object_sha256"))
    if not parent_binds_sha:
        write_bytes_atomic(cdir / "fit_provenance.json", (parent.dir / "fit_provenance.json").read_bytes())
    meta = dict(parent.meta)
    meta["sidecars"] = ["fit_provenance.json", DOSE_SIDECAR, "staged_from.json"]
    meta["ladder"] = {"rung_i": int(i), "lambda": float(lam), "parent_object_sha256": parent.sha,
                      "producer": PRODUCER}
    write_json_atomic(cdir / "meta.json", meta)
    # verification over the bytes ON DISK
    rung_bytes = obj.read_bytes()
    rung_blob = torch.load(io.BytesIO(rung_bytes), map_location="cpu", weights_only=True)
    verification = verify_rung(parent, rung_blob, rung_bytes, lam, probe_rows=probe_rows, seed=seed)
    sha = sha256_bytes(rung_bytes)
    if sha == parent.sha:
        raise Refusal(f"λ={lam:.4g}: rung bytes hash to the parent's sha — nothing was scaled")
    if parent_binds_sha:
        write_json_atomic(cdir / "fit_provenance.json", rebound_identity(parent, sha, lam))
    f32, md = fit_rho_at(parent.fit_meta, lam)
    block = {
        "schema": "alphasteer_f22_dose_scaled_from/1",
        "target": target, "view": view, "cell_id": CELL_ID, "rung_i": int(i),
        "lambda": float(lam), "lambda_4sf": lambda_4sf(lam),
        "object_sha256": sha, "parent_object_sha256": parent.sha, "parent_dir": str(parent.dir),
        "parent_sidecar_sha256": parent.sidecar_shas, "parent_base_scale": 1.0,
        "n_layers": len(parent.layers), "layers": parent.layers, "d": parent.d,
        "mechanism": MECHANISM, "mechanism_note": mechanism_note(),
        "tensors_scaled": [], "tensors_unchanged": ["Ms", "refusals"], "scalar_written": {"base_scale": float(lam)},
        "intervention_hash": verification["intervention_hash"],
        "rho_fit_fp32_by_layer": f32, "rho_fit_model_dtype_by_layer": md,
        "verification": verification, "producer": PRODUCER, "producer_commit": code_sha(),
        "written_at": now_utc(),
        "note": "a λ-scaled copy of the Phase B fit — the identity record beside it (fit_provenance.json) is "
                "the PARENT's, byte-identical; this object was not fit separately; served at scale 1.0 its "
                "edit is lambda x the parent's (base_scale is the served op's coeff)",
    }
    write_json_atomic(cdir / DOSE_SIDECAR, block)
    write_json_atomic(vdir / "VIEW_MANIFEST.json", {
        "wave": "alphasteer_f22_ladder", "target": target, "view": view, "bank": str(rdir),
        "rung_i": int(i), "lambda": float(lam),
        "entries": [{"cell_id": CELL_ID, "staged_id": f"lam{i}__{CELL_ID}", "source": str(cdir),
                     "object_sha256": sha, "identity": "fit_provenance", "mode": "directory",
                     "fit_record": {k: parent.fit_provenance.get(k) for k in
                                    ("resolved_model_id", "resolved_revision", "chat_template_hash",
                                     "positions_at_fit", "writer")}}],
        "written_at": now_utc()})
    staged_shas = {p.name: sha256_file(p) for p in sorted(cdir.iterdir()) if p.is_file() and p.name != "staged_from.json"}
    write_json_atomic(cdir / "staged_from.json", {
        "target": target, "lane": f"lam{i}", "cell_id": CELL_ID, "staged_id": f"lam{i}__{CELL_ID}",
        "source": f"lambda-scaled copy (base_scale {lam!r}, rung {i}) of {parent.dir} (parent object_sha256 {parent.sha})",
        "pulled_at": now_utc(), "object_sha256": sha, "staged_shas": staged_shas,
        "recorded_hash_check": _recorded_hash_check(cdir, sha), "staged_by": PRODUCER}, sort_keys=True)
    (rdir / BAKING_MARKER).unlink()                  # every file has landed: the rung is complete
    return rung_row(target, i, lam, cdir, vdir, sha, parent, verification, f32, md, status="BAKED")


def rebound_identity(parent: "Parent", sha: str, lam: float) -> dict:
    """The identity record for a rung whose PARENT's record binds its own sha. The child's sha at
    the top level, the stage-3 keys the bank view requires carried up, and the parent's record
    preserved WHOLE under ``derived_from.record`` — never edited and never partially merged, since a
    patched provenance file is indistinguishable from a correct one, and the slice record already
    names ITS parent, which a flat ``parent_object_sha256`` here would clobber."""
    pfp = dict(parent.fit_provenance)
    rebound = {k: pfp.get(k) for k in STAGE3_FIT_REQUIRED_KEYS}
    rebound.update({
        "schema": "alphasteer_f22_rung_identity/1",
        "object_sha256": sha,
        "writer": f"{PRODUCER} (dose-scaled rung; the FIT's own record is preserved verbatim under "
                  f"derived_from.record, which is where {pfp.get('writer', 'the parent writer')!r} "
                  "and the parent's lineage live)",
        "derived_from": {
            "object_sha256": parent.sha,
            "record": pfp,
            "record_sha256": parent.sidecar_shas["fit_provenance.json"],
            "why_rebound": "the parent's identity sidecar binds an object_sha256, so a "
                           "byte-identical copy would make this rung claim the parent's identity",
        },
    })
    missing_rb = [k for k in STAGE3_FIT_REQUIRED_KEYS if not rebound.get(k)]
    if missing_rb:
        raise Refusal(f"λ={lam:.4g}: the parent's identity record lacks {missing_rb}, so a rebound "
                      "record cannot carry the stage-3 keys the bank view requires; refusing")
    return rebound


def repair_identity_if_misbound(parent: "Parent", cdir: Path, sha: str, lam: float) -> bool:
    """A KEPT rung is re-verified but never rewritten, so a rung baked before the rebind existed
    still asserts its PARENT's identity — and 'kept' hid that from every re-run (2026-09-13: six
    llama rungs, one refused pinned row downstream).

    The repair condition is EXACTLY the verifier's: when the parent's record binds a sha, this rung
    must carry the rebound record, so anything else on disk — the parent's record copied down (the
    llama case) or a stale copy from before the parent gained a sha — is rewritten from the
    producer. Repairing on a NARROWER condition than the verifier refuses on leaves states that
    refuse forever with nothing able to fix them, which a first cut of this did. Returns True when
    repaired. The object is NOT touched: only the record that describes it is wrong."""
    if not parent.fit_provenance.get("object_sha256"):
        return False                                    # the byte-copy contract still holds
    fp_path = cdir / "fit_provenance.json"
    want = rebound_identity(parent, sha, lam)
    if fp_path.is_file():
        try:
            if load_json(fp_path) == want:
                return False
        except Refusal:
            pass
    write_json_atomic(fp_path, want)
    return True


def rung_row(target, i, lam, cdir, vdir, sha, parent, verification, f32, md, *, status) -> dict:
    return {"i": int(i), "lambda": float(lam), "lambda_4sf": lambda_4sf(lam), "status": status,
            "dir": str(cdir.parents[2]), "defenses_dir": str(vdir), "cell_dir": str(cdir),
            "object_path": str(cdir / "object.pt"), "object_sha256": sha, "parent_object_sha256": parent.sha,
            "intervention_hash": verification["intervention_hash"],
            "n_layers": len(parent.layers), "layers": parent.layers, "d": parent.d,
            "mechanism": MECHANISM, "identity_sidecar": "fit_provenance.json",
            "files": {p.name: sha256_file(p) for p in sorted(cdir.iterdir()) if p.is_file()},
            "rho_fit_fp32_by_layer": f32, "rho_fit_model_dtype_by_layer": md,
            "rho_fit_min_p05_model_dtype": min(v["p05"] for v in md.values()),
            "rho_fit_argmin_layer_model_dtype": int(min(md, key=lambda k: (md[k]["p05"], int(k)))),
            "verification": verification}


def _existing_rung(parent: Parent, root: Path, i: int, lam: float) -> Optional[dict]:
    """A complete previous bake of THIS λ (our sidecar present, its λ equal, its object sha equal to
    the bytes on disk and its parent sha equal to the current parent) — re-verified and KEPT. Anything
    partial or different returns None (and the caller rebuilds)."""
    rdir, vdir, cdir = _rung_record_paths(root, parent.target, i)
    if (rdir / BAKING_MARKER).is_file():             # interrupted: rebuilt, never kept
        return None
    if not (cdir / DOSE_SIDECAR).is_file() or not (cdir / "object.pt").is_file():
        return None
    try:
        block = load_json(cdir / DOSE_SIDECAR)
    except Refusal:
        return None
    if block.get("lambda") != float(lam) or block.get("parent_object_sha256") != parent.sha:
        return None
    if sha256_file(cdir / "object.pt") != block.get("object_sha256"):
        return None
    sha_on_disk = sha256_file(cdir / "object.pt")
    repaired = repair_identity_if_misbound(parent, cdir, sha_on_disk, lam)
    if repaired:
        # the staging record stamps every sidecar's hash, so it must be re-stamped with the record
        sfp = cdir / "staged_from.json"
        if sfp.is_file():
            st = load_json(sfp)
            st["staged_shas"] = {p.name: sha256_file(p) for p in sorted(cdir.iterdir())
                                 if p.is_file() and p.name != "staged_from.json"}
            st["recorded_hash_check"] = _recorded_hash_check(cdir, sha_on_disk)
            st["identity_repaired_at"] = now_utc()
            write_json_atomic(sfp, st)
    row = check_rung(parent, root, i, lam, probe_rows=6, seed=0)
    row["status"] = "REPAIRED" if repaired else "KEPT"
    return row


def check_rung(parent: Parent, root: Path, i: int, lam: float, *, probe_rows: int, seed: int) -> dict:
    """Re-verify one rung dir on disk (no writes). Raises Refusal naming the rung."""
    import torch

    target = parent.target
    rdir, vdir, cdir = _rung_record_paths(root, target, i)
    tag = f"{target} lam{i} (λ={lam:.4g})"
    if (rdir / BAKING_MARKER).is_file():
        raise Refusal(f"{tag}: {rdir} carries the {BAKING_MARKER} marker — an interrupted bake, not a rung "
                      "(re-run the bake; it rebuilds the dir)")
    obj = cdir / "object.pt"
    if not obj.is_file():
        raise Refusal(f"{tag}: no object.pt at {obj}")
    for name in ("fit_provenance.json", "meta.json", DOSE_SIDECAR, "staged_from.json"):
        if not (cdir / name).is_file():
            raise Refusal(f"{tag}: {name} missing from {cdir}")
    if not (vdir / "VIEW_MANIFEST.json").is_file():
        raise Refusal(f"{tag}: VIEW_MANIFEST.json missing from {vdir}")
    present = [s for s in IDENTITY_SIDECARS if (cdir / s).is_file()]
    if present != ["fit_provenance.json"]:
        raise Refusal(f"{tag}: identity sidecars {present} != ['fit_provenance.json'] (exactly one)")
    # The identity sidecar is the parent's byte-identical ONLY when the parent's record binds no
    # sha of its own; when it does, the rung carries the REBOUND record instead (see
    # rebound_identity) and byte-identity would be the defect rather than the invariant.
    rung_fp_bytes = (cdir / "fit_provenance.json").read_bytes()
    if parent.fit_provenance.get("object_sha256"):
        rung_fp = json.loads(rung_fp_bytes)
        want = rebound_identity(parent, sha256_file(cdir / "object.pt"), lam)
        if rung_fp != want:
            raise Refusal(f"{tag}: the parent's identity record binds a sha, so this rung must carry the "
                          "REBOUND record (its own sha, the parent's record whole under derived_from); "
                          "what is on disk is neither that nor producible from this parent")
    elif rung_fp_bytes != (parent.dir / "fit_provenance.json").read_bytes():
        raise Refusal(f"{tag}: fit_provenance.json is not byte-identical to the parent's")
    extra = sorted(p.name for p in vdir.iterdir() if p.name not in (CELL_ID, "VIEW_MANIFEST.json"))
    if extra:
        raise Refusal(f"{tag}: view dir {vdir} carries unexpected entries {extra}")
    block = load_json(cdir / DOSE_SIDECAR)
    if block.get("lambda") != float(lam):
        raise Refusal(f"{tag}: {DOSE_SIDECAR} records lambda {block.get('lambda')!r} != {float(lam)!r}")
    if block.get("parent_object_sha256") != parent.sha:
        raise Refusal(f"{tag}: {DOSE_SIDECAR} names parent {str(block.get('parent_object_sha256'))[:12]} but the "
                      f"parent on disk is {parent.sha[:12]}")
    if block.get("mechanism") != MECHANISM:
        raise Refusal(f"{tag}: {DOSE_SIDECAR} records mechanism {block.get('mechanism')!r} != {MECHANISM!r}")
    rung_bytes = obj.read_bytes()
    sha = sha256_bytes(rung_bytes)
    if sha != block.get("object_sha256"):
        raise Refusal(f"{tag}: object.pt hashes to {sha[:12]} but {DOSE_SIDECAR} records "
                      f"{str(block.get('object_sha256'))[:12]} — the bytes changed after the bake")
    staged = load_json(cdir / "staged_from.json")
    if staged.get("object_sha256") != sha:
        raise Refusal(f"{tag}: staged_from.json object_sha256 != on-disk sha")
    for name, want in (staged.get("staged_shas") or {}).items():
        p = cdir / name
        if not p.is_file() or sha256_file(p) != want:
            raise Refusal(f"{tag}: staged_from.json names {name} at {want[:12]} but the file is missing or differs")
    meta = load_json(cdir / "meta.json")
    for key, want in (("cell_id", CELL_ID), ("arm", CELL_ID), ("class", "B"), ("si_id", None)):
        if meta.get(key, None) != want:
            raise Refusal(f"{tag}: meta.json {key}={meta.get(key)!r} != {want!r}")
    try:
        rung_blob = torch.load(io.BytesIO(rung_bytes), map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        raise Refusal(f"{tag}: object.pt does not load: {e}") from e
    verification = verify_rung(parent, rung_blob, rung_bytes, lam, probe_rows=probe_rows, seed=seed)
    if verification["intervention_hash"] != block.get("intervention_hash"):
        raise Refusal(f"{tag}: intervention_hash {verification['intervention_hash']} != recorded "
                      f"{block.get('intervention_hash')}")
    f32, md = fit_rho_at(parent.fit_meta, lam)
    if block.get("rho_fit_model_dtype_by_layer") != md or block.get("rho_fit_fp32_by_layer") != f32:
        raise Refusal(f"{tag}: the recorded fit-side rho table differs from λ x fit_meta's")
    return rung_row(target, i, lam, cdir, vdir, sha, parent, verification, f32, md, status="VERIFIED")


def mechanism_note() -> dict:
    from sisteer.capeval.serving import lens_adapters as la

    return {
        "name": MECHANISM,
        "what": "the rung blob is the parent blob with ONLY the scalar base_scale rewritten to lambda; every "
                "tensor (Ms, refusals) is bit-identical to the parent's; the matrices are NOT pre-multiplied",
        "why": "lens_adapters._adapt_alphasteer emits HookOp(op=OP_PREFILL_LAST_MATMUL, matrix=M_l, "
               "coeff=base_scale, phase=prefill) per fit layer, and _apply_one computes eff = scale * op.coeff; "
               "out + (eff * (h_last @ M)).unsqueeze(-2) — base_scale reaches the served op AS THE COEFF, so "
               "the served edit at scale 1.0 is exactly lambda x the lambda=1 edit on the identical h_last @ M",
        "evidence": {"adapter_module": la.__file__, "adapter_source_sha256": sha256_file(Path(la.__file__)),
                     "functions": ["_adapt_alphasteer", "_apply_one", "materialize_ops"],
                     "op": la.OP_PREFILL_LAST_MATMUL,
                     "matrix_dtype_policy": la._OP_TENSOR_FIELDS[la.OP_PREFILL_LAST_MATMUL],
                     "asserted_per_rung": ["served_ops: coeff_rung == lambda, coeff_parent == 1.0, matrix torch.equal",
                                           "served_edit: apply_hook_ops rung == lambda * parent, torch.equal, fp32 + bf16"]},
        "if_base_scale_stopped_reaching_the_op": "the served_ops check refuses (coeff != lambda); the bake never "
                                                  "falls back to scaling the matrices silently",
    }


# --------------------------------------------------------------------------- per target

def bake_target(root: Path, target: str, *, n_rungs: int, floor: float, force: bool, check: bool,
                probe_rows: int, seed: int) -> dict:
    import torch

    parent = Parent(root, target)          # MissingObject / Refusal propagate
    grid = lambda_grid(parent.fit_meta, n_rungs=n_rungs, floor=floor)
    view = view_of(target)
    ldir = ladder_dir(root, target)
    rec_path = ldir / BAKE_RECORD
    print(f"[bake] {target}: parent {parent.sha[:12]} layers={len(parent.layers)} d={parent.d} view={view}; "
          f"lambda_1 = {floor:g} / rho_p05[{grid['argmin_layer']}]={grid['min_rho_p05_model_dtype_at_unit']:.6g} "
          f"= {grid['lambda_1']:.6g}; grid {grid['lambdas_4sf']}", flush=True)
    rows = []
    if check:
        if not rec_path.is_file():
            raise Refusal(f"{target}: no {rec_path} — nothing to check (bake first)")
        prev = load_json(rec_path)
        if prev.get("schema") != SCHEMA:
            raise Refusal(f"{target}: {rec_path} schema {prev.get('schema')!r} != {SCHEMA!r}")
        if prev.get("parent", {}).get("object_sha256") != parent.sha:
            raise Refusal(f"{target}: {rec_path} was baked off parent {str(prev.get('parent', {}).get('object_sha256'))[:12]} "
                          f"but the parent on disk is {parent.sha[:12]}")
        if prev.get("grid", {}).get("lambdas") != grid["lambdas"]:
            raise Refusal(f"{target}: {rec_path} grid {prev.get('grid', {}).get('lambdas')} != recomputed {grid['lambdas']}")
        prev_rows = {int(r["i"]): r for r in prev.get("rungs", [])}
        for i, lam in enumerate(grid["lambdas"], start=1):
            row = check_rung(parent, root, i, lam, probe_rows=probe_rows, seed=seed)
            if i not in prev_rows or prev_rows[i].get("object_sha256") != row["object_sha256"]:
                raise Refusal(f"{target} lam{i}: object sha {row['object_sha256'][:12]} != the bake record's "
                              f"{str(prev_rows.get(i, {}).get('object_sha256'))[:12]}")
            rows.append(row)
            print(f"[check] {target} lam{i} λ={lam:.4g} sha {row['object_sha256'][:12]} VERIFIED", flush=True)
        return {"target": target, "status": "VERIFIED", "rungs": rows, "record": str(rec_path)}
    for i, lam in enumerate(grid["lambdas"], start=1):
        row = bake_rung(parent, root, i, lam, force=force, probe_rows=probe_rows, seed=seed)
        rows.append(row)
        print(f"[bake] {target} lam{i} λ={lam:.4g} sha {row['object_sha256'][:12]} <- parent {parent.sha[:12]} "
              f"{row['status']} ({', '.join(c['check'] + ':' + c['status'] for c in row['verification']['checks'])})",
              flush=True)
    record = {
        "schema": SCHEMA, "target": target, "view": view, "root": str(root), "ladder_dir": str(ldir),
        "cell_id": CELL_ID, "parent": parent.record(), "grid": grid,
        "mechanism": mechanism_note(),
        "rungs": [{k: v for k, v in r.items() if k != "status"} for r in rows],
        "producer": PRODUCER, "code_sha": code_sha(), "torch_version": torch.__version__,
        "written_at": now_utc(),
    }
    record["sha256"] = sha256_bytes(json.dumps(record, sort_keys=True).encode("utf-8"))
    write_json_atomic(rec_path, record)
    print(f"[bake] {target}: record {rec_path} ({len(rows)} rungs; "
          f"{sum(r['status'] == 'BAKED' for r in rows)} baked, {sum(r['status'] == 'KEPT' for r in rows)} kept, "
          f"{sum(r['status'] == 'REPAIRED' for r in rows)} identity-repaired)",
          flush=True)
    return {"target": target, "status": "BAKED", "rungs": rows, "record": str(rec_path)}


def _targets(values: List[str]) -> List[str]:
    out: List[str] = []
    for v in values:
        for t in v.split(","):
            t = t.strip()
            if t and t not in out:
                out.append(t)
    if not out:
        raise SystemExit("--target names no target")
    for t in out:
        view_of(t)     # refuses an unknown target loudly, before touching disk
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", required=True, action="append",
                    help="target key (repeat, or comma-separate): gemma4e4b qwen35_4b glm46vflash nemotron3nano4b")
    ap.add_argument("--root", required=True, type=Path,
                    help="the objects root, e.g. $SHARED/exp/objects/alphasteer_f22 (parent at "
                         "<root>/<target>/cells/alphasteer, rungs under <root>/<target>/ladder/)")
    ap.add_argument("--rungs", type=int, default=DEFAULT_RUNGS)
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR,
                    help="the bf16-faithfulness floor lambda_1 is set to (plan C.1; default 3e-3)")
    ap.add_argument("--force", action="store_true", help="re-bake over a previous bake of ours")
    ap.add_argument("--check", action="store_true", help="re-verify an existing bake; write nothing")
    ap.add_argument("--probe-rows", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    targets = _targets(args.target)
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        print(f"[bake] REFUSED: --root {root} is not a directory", file=sys.stderr)
        return EXIT_REFUSED
    outcomes: Dict[str, str] = {}
    rc = EXIT_OK
    for t in targets:
        try:
            res = bake_target(root, t, n_rungs=args.rungs, floor=args.floor, force=args.force,
                              check=args.check, probe_rows=args.probe_rows, seed=args.seed)
            outcomes[t] = res["status"]
        except MissingObject as e:
            outcomes[t] = "SKIPPED_NO_OBJECT"
            print(f"[bake] SKIP {e}", flush=True)
            if rc == EXIT_OK:
                rc = EXIT_NO_OBJECT
        except Refusal as e:
            outcomes[t] = "REFUSED"
            print(f"[bake] REFUSED {t}: {e}", file=sys.stderr, flush=True)
            rc = EXIT_REFUSED
    print("[bake] outcomes: " + ", ".join(f"{t}={s}" for t, s in outcomes.items()) + f"; exit {rc}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
