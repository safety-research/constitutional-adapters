"""Shared bookkeeping for the DOSE-SCALED copy producers (scale_psr_dose.py, scale_lora_dose.py).

A dose-scaled copy is a NEW object (its own bytes, its own sha) derived from ONE parent cell by a
linear rescale of the parent's write tensor — directions for a gated PSR, the write basis B for a
LoRA — with every other tensor byte-identical. The served lens bakes the steering strength
(``effective_scale`` is binary 0/1 at serve time), so a non-1x dose MUST be different bytes, never a
serve-time knob; this module makes every such copy carry the same record:

* the parent's IDENTITY SIDECAR, re-bound to the new bytes. Two identity contracts exist
  (train_budget.object_identity_kind): ``train_budget_meta.json`` (a training-lane fit) and
  ``derived_meta.json`` (a derived object — the distilled PSR residual, the LoRA subtraction). The
  copy keeps the parent's contract: the training facts / derivation facts ARE the parent's (the copy
  was not trained or derived separately), only ``object_sha256`` (and a ``files`` map's object.pt
  entry, when present) is re-bound, and a ``dose_scaled_from`` block names the parent, the scale,
  which tensors moved, the producer, and the VERIFICATION result.
* ``object.pt.provenance.json`` when the parent has one, annotated with ``dose_scale`` and
  ``parent_object_sha256``.
* ``staged_from.json`` in the staging agent's schema (stage_full22.sh hashcheck_and_stamp), so the
  bank-view builder admits the dir like any staged copy and the staging manifest / pins can be
  rebuilt from it.

No COMPLETE marker is written (a copy is not a publication of a fit); the view builder treats an
absent marker as fine.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

BUDGET_META = "train_budget_meta.json"
DERIVED_META = "derived_meta.json"
FIT_PROVENANCE = "fit_provenance.json"
IDENTITY_SIDECARS = (BUDGET_META, DERIVED_META, FIT_PROVENANCE)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def producer_commit(script_path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(script_path).resolve().parent,
                                       text=True).strip()
    except Exception:  # noqa: BLE001
        return None


def identity_of(parent: Path) -> tuple[str, Path]:
    """(identity, sidecar path) of the parent dir — exactly one of the identity sidecars must be
    present (the same dispatch train_budget.object_identity_kind makes). fit_provenance.json is
    refused: no dose-scaled copy of an HPO-promotion cell is defined by this producer (its record
    carries no sha and no cell id; re-binding it is not a mechanical step)."""
    present = [s for s in IDENTITY_SIDECARS if (parent / s).is_file()]
    if len(present) != 1:
        raise SystemExit(f"parent {parent} declares {present} identity sidecars — need exactly one of "
                         f"{BUDGET_META} / {DERIVED_META}")
    if present[0] == FIT_PROVENANCE:
        raise SystemExit(f"parent {parent} is an HPO-promotion cell ({FIT_PROVENANCE}); this producer "
                         "re-binds only train_budget_meta.json / derived_meta.json identities")
    return ("staged" if present[0] == BUDGET_META else "derived"), parent / present[0]


def check_parent_sidecar(meta: dict, parent_sha: str, sidecar: Path) -> None:
    if meta.get("object_sha256") != parent_sha:
        raise SystemExit(f"parent sidecar {sidecar} records object_sha256 {meta.get('object_sha256')} but "
                         f"object.pt hashes to {parent_sha}")


def rebind_identity(meta: dict, new_sha: str, block: dict) -> dict:
    out = dict(meta)
    out["object_sha256"] = new_sha
    if isinstance(out.get("files"), dict) and "object.pt" in out["files"]:
        out["files"] = dict(out["files"])
        out["files"]["object.pt"] = new_sha
    out["dose_scaled_from"] = block
    return out


def _recorded_hash_check(d: Path, sha: str) -> dict:
    """stage_full22.sh hashcheck_and_stamp's verdict, byte-for-byte the same rules: every sidecar
    json's 64-hex values under sha/hash/digest/fingerprint keys; only a TOP-LEVEL object_sha256
    claims to be this object's hash."""
    hexre = re.compile(r"^[0-9a-f]{64}$")
    keyre = re.compile(r"sha|hash|digest|fingerprint", re.I)
    entries: list[dict] = []

    def walk(o, path, fname):
        if isinstance(o, dict):
            for k, v in o.items():
                walk(v, f"{path}.{k}" if path else k, fname)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]", fname)
        elif isinstance(o, str) and hexre.match(o) and keyre.search(path):
            entries.append({"file": fname, "key": path, "value": o, "matches_object": o == sha})

    for root, _, files in os.walk(d):
        for f in sorted(files):
            p = Path(root) / f
            rel = str(p.relative_to(d))
            if f.endswith(".json") and f != "staged_from.json":
                try:
                    walk(json.loads(p.read_text(encoding="utf-8")), "", rel)
                except Exception as e:  # noqa: BLE001
                    entries.append({"file": rel, "key": "<unparseable>", "value": str(e), "matches_object": False})
    own = [e for e in entries if e["key"] == "object_sha256"]
    prior = [e for e in entries if e["key"].endswith("continued_from.prior_object_sha256")]
    if any(e["matches_object"] for e in entries):
        status = "match"
    elif entries:
        status = "recorded_hashes_are_not_this_object"
    else:
        status = "none_recorded"
    return {"status": status,
            "object_sha256_recorded_match": (all(e["matches_object"] for e in own) if own else None),
            "object_sha256_recorded_in": sorted({e["file"] for e in own}),
            "object_identical_to_continued_from_prior": (all(e["matches_object"] for e in prior) if prior else None),
            "entries": entries}


def write_staged_from(out: Path, *, target: str, staged_id: str, cell_id: str, new_sha: str,
                      parent_staged_id: str | None, parent: Path, parent_sha: str, scale: float,
                      producer: str, anchor: bool = False, parent_origin: str | None = None,
                      parent_state: str | None = None) -> dict:
    """``anchor``: the copy is the ladder's re-served 1.0 rung — byte-identical to the parent (same sha), staged
    under its own id so the frontier's 1.0 point is served in the same pass as its neighbours without borrowing
    another lane's staged id. ``parent_origin`` / ``parent_state``: where the parent bytes were authoritatively
    read from (when ``parent`` is a pulled copy) and what state the owning lane's parent was in when consumed
    (e.g. final bytes, not yet staged) — recorded so the provenance says which sha of which parent this copy
    descends from, without assuming anything about that parent's later staging."""
    lane = staged_id.split("__", 1)[0]
    staged_shas = {str(p.relative_to(out)): sha256_file(p)
                   for p in sorted(out.rglob("*")) if p.is_file() and p.name != "staged_from.json"}
    assert staged_shas["object.pt"] == new_sha
    who = parent_staged_id or str(parent)
    source = (f"re-served 1.0 anchor: byte-identical copy of {who} (parent object_sha256 {parent_sha} == this object's)"
              if anchor else
              f"dose-scaled copy (x{scale:g}) of {who} (parent object_sha256 {parent_sha})")
    rec = {
        "target": target, "lane": lane, "cell_id": cell_id, "staged_id": staged_id,
        "source": source,
        "pulled_at": now_utc(), "object_sha256": new_sha, "staged_shas": staged_shas,
        "recorded_hash_check": _recorded_hash_check(out, new_sha),
        "staged_by": producer,
    }
    if parent_origin is not None:
        rec["parent_origin"] = parent_origin
    if parent_state is not None:
        rec["parent_consumed_state"] = parent_state
    (out / "staged_from.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return rec


def copy_provenance_json(parent: Path, out: Path, *, scale: float, parent_sha: str, fallback: dict) -> None:
    p = parent / "object.pt.provenance.json"
    prov = json.loads(p.read_text(encoding="utf-8")) if p.is_file() else dict(fallback)
    prov = dict(prov)
    prov["dose_scale"] = float(scale)
    prov["parent_object_sha256"] = parent_sha
    (out / "object.pt.provenance.json").write_text(json.dumps(prov, indent=2) + "\n", encoding="utf-8")


def fresh_out_dir(out: Path) -> None:
    """The output dir is written from scratch (a re-run never leaves a stale sidecar beside a new
    object); refuses to remove anything that is not a previous dose-scaled copy of ours."""
    if out.exists():
        marker = None
        for s in (BUDGET_META, DERIVED_META):
            p = out / s
            if p.is_file():
                try:
                    marker = json.loads(p.read_text(encoding="utf-8")).get("dose_scaled_from")
                except ValueError:
                    marker = None
        if not marker and any(out.iterdir()):
            raise SystemExit(f"{out} exists and is not a previous dose-scaled copy — refusing to overwrite")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def tensor_stats(t) -> dict:
    import torch

    t = t.detach().float()
    return {"norm": float(t.norm()), "max_abs": float(t.abs().max()) if t.numel() else 0.0,
            "shape": list(t.shape)}
