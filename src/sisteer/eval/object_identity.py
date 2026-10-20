"""ONE consumer entry point for "may I load this artifact as <id>", for the
round-2 measurement consumers (``ce_score``, the residual builder, the
derived Modal app): the shared ``train_budget`` identity gates with the
one decision they deliberately leave to the caller — what to do with a
BARE artifact (no sidecar of either kind).

Dispatch is on the sidecar the artifact's dir carries
(:func:`train_budget.assert_object_identity`):

- ``train_budget_meta.json`` → the staged-cell gate (sidecar names the
  cell, object bytes hash to the sidecar's sha, document-budget schema,
  COMPLETE coherence);
- ``derived_meta.json`` → the derived-object verdict (sidecar ↔ object
  sha, ``derived_id``, ≥ 2 complete ``parent_*`` identities agreeing on
  model/revision/layers/fit leg, the artifact's EMBEDDED provenance naming
  the same parents and kind, and — with ``parents_root`` — every parent
  re-verified on disk with its recorded sha);
- BOTH → refused (an ambiguous dir); NEITHER → refused unless the caller
  passes ``allow_bare=True`` (scratch/test objects — never a production
  default; the probe scorer's own leniency lives in probe_score.py).

Everything here is a thin layer over ``train_budget`` so there is exactly
one verdict per identity kind.
"""

from __future__ import annotations

import json
from pathlib import Path

from .train_budget import (BUDGET_META_NAME, DERIVED_META_NAME,
                           DERIVED_PARENT_FIELDS, DERIVED_PARENT_MATCH_FIELDS,
                           FIT_PROVENANCE_NAME, IDENTITY_SIDECARS,
                           PROMOTION_PROVENANCE_NAME, assert_derived_object,
                           assert_fit_provenance_cell, assert_object_identity,
                           assert_staged_cell, derived_object_problems,
                           fit_provenance_problems, object_identity_kind)

__all__ = [
    "BUDGET_META_NAME", "DERIVED_META_NAME", "DERIVED_PARENT_FIELDS",
    "DERIVED_PARENT_MATCH_FIELDS", "FIT_PROVENANCE_NAME", "IDENTITY_SIDECARS",
    "PROMOTION_PROVENANCE_NAME", "assert_derived_object",
    "assert_fit_provenance_cell", "assert_object_identity",
    "assert_parents_root_for_derived", "assert_staged_cell",
    "derived_object_problems", "fit_provenance_problems",
    "object_identity_kind", "recorded_identity", "validate_artifact_identity",
]


def assert_parents_root_for_derived(object_path, parents_root, *,
                                    consumer: str) -> None:
    """A DERIVED artifact may only be scored with its parents re-verified
    on disk: without ``parents_root`` the derived verdict checks the sidecar
    against the bytes but never asks the LIVE bank whether the parents it
    names are still the objects staged under those cell ids — a refit parent
    under the same cell id would be paired with stale derived bytes and the
    row would read as a valid decomposition of the new fit. Consumers that
    score derived objects (probe_score, base_mmlu) call this before the gate;
    a staged or bare artifact is untouched (it has no parents)."""
    if parents_root is None and object_identity_kind(object_path) == "derived":
        raise SystemExit(
            f"[{consumer}] {Path(object_path)} is a DERIVED object (its dir "
            f"carries {DERIVED_META_NAME}) — pass --parents-root <fits bank "
            "root holding <model_key>/cells_all/> so its parents' object and "
            "sidecar shas are re-verified against the live bank before it is "
            "loaded. Without that check a refit parent under the same cell "
            "id would be paired with stale derived bytes.")


def recorded_identity(object_path) -> str | None:
    """The identity the artifact's own sidecar records — ``derived_id`` for
    a derived object, ``cell_id`` for a fitted cell — or the dir name when
    the sidecar is unreadable/unstamped (the bank layout's identity, which
    the validator then names as the problem). None for a bare dir."""
    obj_path = Path(object_path)
    kind = object_identity_kind(obj_path)
    if kind is None:
        return None
    if kind == "fit_provenance":
        # the HPO-promotion lineage's fit record names no cell; its
        # companion provenance.json (si + arm) does when present, else the
        # bank layout's dir name is the identity
        try:
            prov = json.loads((obj_path.parent / PROMOTION_PROVENANCE_NAME)
                              .read_text(encoding="utf-8"))
            rec = (f"{prov['si']}__{prov['arm']}"
                   if prov.get("si") and prov.get("arm") else None)
        except (ValueError, AttributeError, OSError, KeyError):
            rec = None
        return str(rec) if rec else obj_path.parent.name
    sidecar = obj_path.parent / (DERIVED_META_NAME if kind == "derived"
                                 else BUDGET_META_NAME)
    try:
        meta = json.loads(sidecar.read_text(encoding="utf-8"))
        rec = (meta.get("derived_id") if kind == "derived"
               else meta.get("cell_id"))
    except (ValueError, AttributeError, OSError):
        rec = None
    return str(rec) if rec else obj_path.parent.name


def validate_artifact_identity(object_path, cell_id: str | None, *,
                               consumer: str, parents_root=None,
                               allow_bare: bool = False,
                               require_parents_root: bool = False,
                               **staged_kw) -> str:
    """Validate ``object_path`` as ``cell_id`` (a fitted cell id or a
    derived id) through whichever contract its dir declares; returns the
    kind validated (``"staged"`` | ``"derived"`` | ``"bare"``). A dir with
    no sidecar refuses unless ``allow_bare``. With ``cell_id=None`` on a
    sidecared dir the sidecar's own recorded identity is the requested one
    (self-consistency: every sha/kind/parent check still runs). With
    ``require_parents_root`` a derived artifact refuses unless
    ``parents_root`` is given (:func:`assert_parents_root_for_derived`)."""
    if require_parents_root:
        assert_parents_root_for_derived(object_path, parents_root,
                                        consumer=consumer)
    kind = object_identity_kind(object_path)
    if kind is None:
        if allow_bare:
            return "bare"
        raise SystemExit(
            f"[{consumer}] {Path(object_path).parent} carries none of "
            f"{list(IDENTITY_SIDECARS)} — an unsidecared artifact cannot be "
            "proven to be anything; pass --allow-bare only for a "
            "deliberately ad-hoc object.")
    if cell_id is None and kind != "ambiguous":
        cell_id = recorded_identity(object_path)
    return assert_object_identity(object_path, cell_id, consumer=consumer,
                                  parents_root=parents_root, **staged_kw)
