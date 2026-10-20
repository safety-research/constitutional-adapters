#!/usr/bin/env python
"""ONE parent-admission resolver for the round-2 derived-object builders
(sdf-base-steering; scripts/sdfbase/lora_subtract.py and
scripts/sdfbase/psr_residual_distill.py both build from two fitted parent
cells and used to carry their own copies of this gate — the /simplify pass
folded them here so the two builders admit parents by exactly one rule).

A fitted cell is ADMITTED as a parent when, in this order:

1. its ``object.pt`` exists and a COMPLETE marker is present (a torn publish
   is never a parent);
2. its budget sidecar ``train_budget_meta.json`` exists and names exactly
   this cell (a wrong-identity artifact refuses);
3. the SHARED staged-cell gate passes (``train_budget.staged_cell_problems``:
   object bytes ↔ sidecar sha, document-budget schema, COMPLETE manifest
   coherence — resume-grade strictness on the sample manifest);
4. every :data:`PARENT_REQUIRED_FIELDS` entry is recorded non-null and every
   :data:`PARENT_RECORDED_FIELDS` KEY is present;
5. the bank's store identity (``_store_identity.json``, the F15 sentinel
   every writer into the bank had to match) exists and names the sidecar's
   model, and the parent's ``model_revision`` RESOLVES against it
   (:func:`resolve_model_revision`: the sidecar's own record when it carries
   one — it must then agree with the store — else the store's; the
   qwen35_4b_base bank's round-1 sidecars record it null although the fits
   config pins one, which is exactly why the store is the authoritative
   fallback). The SOURCE of the resolved revision is recorded in the
   provenance as ``model_revision_source``;
6. the blob loads weights-only and carries the ``kind`` the builder expects.

The returned provenance block carries every identity field the shared
derived-object verdict (``train_budget.derived_object_problems``) later
re-checks, plus the parent's own fit fingerprint (a refit under the same
cell id has a different one). Torch-free at import (torch is imported inside
:func:`admit_parent`, the way the builders do).
"""

from __future__ import annotations

import json
from pathlib import Path

from build_heldout_split import sha256_file
from sisteer.eval.object_identity import DERIVED_PARENT_MATCH_FIELDS

#: The bank-level identity sidecar (train_two_si_objects.assert_store_identity
#: writes it into ``<bank>/cells_<leg>/``).
STORE_IDENT = "_store_identity.json"
BUDGET_META = "train_budget_meta.json"

#: Parent sidecar fields that must be RECORDED (non-null) for a parent to be
#: usable: the model identity (``meta["model"]`` reads None — the sidecar's
#: key is ``model_id``) and the fit geometry the cross-parent match
#: compares.
PARENT_REQUIRED_FIELDS = ("model_id", "target_layers", "positions",
                          "template_hash", "training_budget_kind")

#: Parent sidecar fields whose KEY must be present but whose value is
#: resolved against the bank rather than required non-null in the sidecar
#: (:func:`resolve_model_revision`). A sidecar WITHOUT the key is a
#: pre-schema record and refuses.
PARENT_RECORDED_FIELDS = ("model_revision",)

#: Parent fields the two parents must AGREE on — the shared derived-object
#: verdict's own tuple, so a builder refuses BEFORE composing on exactly the
#: fields the verdict re-checks afterwards.
PARENT_MATCH_FIELDS = DERIVED_PARENT_MATCH_FIELDS


class ParentAdmissionError(RuntimeError):
    """Loud failure — a parent that cannot be proven is never composed."""


def resolve_model_revision(meta: dict, ident: dict, *, meta_path,
                           ident_path, error=ParentAdmissionError
                           ) -> tuple[str, str]:
    """``(model_revision, source)`` for a parent — the sidecar's own record
    when it carries one (it must then AGREE with the store identity), else
    the bank's store identity (the authoritative store-level record; the
    qwen35_4b_base sidecars record null). Refuses a contradiction and a
    revision recorded nowhere: a derived object never ships without the
    revision its parents were fit against."""
    sidecar_rev = meta.get("model_revision")
    store_rev = ident.get("model_revision")
    if sidecar_rev is not None:
        if store_rev is not None and str(sidecar_rev) != str(store_rev):
            raise error(
                f"{meta_path} records model_revision={sidecar_rev!r} but "
                f"the bank's store identity {ident_path} claims "
                f"{store_rev!r} — refusing a parent whose revision "
                "contradicts its own store")
        return str(sidecar_rev), BUDGET_META
    if store_rev is None:
        raise error(
            f"neither {meta_path} nor {ident_path} records a model "
            "revision — a parent fit against an unrecorded revision cannot "
            "be a derived object's parent")
    return str(store_rev), STORE_IDENT


def admit_parent(root, key: str, cell_id: str, *, kind: str,
                 error=ParentAdmissionError) -> tuple[Path, dict, dict, dict]:
    """(object path, blob, budget sidecar, provenance block) for one fitted
    parent cell at
    ``<root>/<key>/cells_all/<cell_id>/``, identity-checked per the module
    docstring; ``kind`` is the blob kind the builder expects (``lora_all``,
    ``psr``). ``error`` is the exception class raised on every refusal (the
    builder's own, so its CLI files one REFUSED verdict)."""
    import torch

    from sisteer.eval.train_budget import staged_cell_problems

    cell_dir = Path(root) / key / "cells_all" / cell_id
    obj_path = cell_dir / "object.pt"
    if not obj_path.is_file():
        raise error(f"{obj_path} missing — parent fit not harvested")
    marker_path = cell_dir / "COMPLETE"
    if not marker_path.is_file():
        raise error(
            f"{cell_dir} carries no COMPLETE marker — refusing a parent "
            "that may be a torn publish")
    meta_path = cell_dir / BUDGET_META
    if not meta_path.is_file():
        raise error(f"{meta_path} missing — no identity sidecar")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("cell_id") != cell_id:
        raise error(
            f"{meta_path} names cell_id={meta.get('cell_id')!r}, expected "
            f"{cell_id!r} — wrong-identity artifact, refusing")
    problems = staged_cell_problems(obj_path, cell_id)
    if problems:
        raise error(
            f"{cell_dir} failed the staged-cell identity gate:\n  - "
            + "\n  - ".join(problems))
    missing = ([f for f in PARENT_REQUIRED_FIELDS if meta.get(f) is None]
               + [f for f in PARENT_RECORDED_FIELDS if f not in meta])
    if missing:
        raise error(
            f"{meta_path} records no {missing} — a parent whose model "
            "identity or fit geometry is unrecorded cannot be a derived "
            "object's parent")
    ident_path = cell_dir.parent / STORE_IDENT
    if not ident_path.is_file():
        raise error(
            f"{ident_path} missing — {Path(root) / key} is not a bank (no "
            "store identity); a parent's model revision is resolved "
            "against it")
    ident = json.loads(ident_path.read_text(encoding="utf-8"))
    if ident.get("model") != meta["model_id"]:
        raise error(
            f"{meta_path} records model_id={meta['model_id']!r} but the "
            f"bank's store identity {ident_path} claims "
            f"{ident.get('model')!r} — a cell from another bank")
    revision, rev_source = resolve_model_revision(
        meta, ident, meta_path=meta_path, ident_path=ident_path, error=error)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    blob = torch.load(obj_path, map_location="cpu", weights_only=True)
    if blob.get("kind") != kind:
        raise error(f"{obj_path} kind={blob.get('kind')!r}, expected {kind!r}")
    prov = {
        "cell_id": cell_id,
        "model_key": key,
        "object_path": str(obj_path),
        "object_sha256": sha256_file(obj_path),
        "budget_meta_sha256": sha256_file(meta_path),
        # The parent's OWN fit identity (F15): the fingerprint its COMPLETE
        # marker carries binds model+revision+SI+arm+leg+pool+optimizer+
        # config — a refit under the same cell id has a different one.
        "fit_fingerprint": marker.get("fit_fingerprint"),
        "model_id": meta["model_id"],
        "model_revision": revision,
        "model_revision_source": rev_source,
        "positions": meta["positions"],
        "response_header_len": meta.get("response_header_len"),
        "target_layers": [int(l) for l in meta["target_layers"]],
        "template_hash": meta["template_hash"],
        "training_budget_kind": meta["training_budget_kind"],
        "sample_manifest_sha": meta.get("sample_manifest_sha"),
        "heldout_pool_sha": meta.get("heldout_pool_sha"),
    }
    return obj_path, blob, meta, prov


def require_parents_match(pos_prov: dict, neg_prov: dict, *, what: str,
                          error=ParentAdmissionError) -> None:
    """Refuse a pair whose parents differ on any :data:`PARENT_MATCH_FIELDS`
    entry — named, before any tensor is touched. ``what`` names the
    derived object ("a task-vector difference", "a paired write")."""
    for f in PARENT_MATCH_FIELDS:
        if pos_prov.get(f) != neg_prov.get(f):
            raise error(
                f"parents disagree on {f}: {pos_prov['cell_id']} has "
                f"{pos_prov.get(f)!r}, {neg_prov['cell_id']} has "
                f"{neg_prov.get(f)!r} — {what} across that is not a defined "
                "object")


def pool_budget(pools_dir, si: str) -> dict:
    """The parent's DATA SCALE from its staged pool manifest (budget +
    realized official tokens + manifest sha) — 'unavailable' recorded
    honestly when the local staging dir lacks the manifest, never a
    silent omission."""
    mp = Path(pools_dir) / f"pairs_{si}.manifest.json"
    if not mp.is_file():
        return {"si": si, "budget": None, "official_tokens": None,
                "source": f"unavailable ({mp} missing)"}
    m = json.loads(mp.read_text(encoding="utf-8"))
    return {"si": si, "budget": m.get("budget"),
            "official_tokens": m.get("official_tokens"),
            "manifest_sha256": sha256_file(mp), "source": str(mp)}
