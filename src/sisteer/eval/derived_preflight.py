"""The DERIVED-object preflight: the sha-bound parent verdict PLUS the
supersession cross-check.

``train_budget.derived_object_problems(parents_root=)`` proves a derived
object against its parents BY SHA: each parent dir under
``<parents_root>/<model_key>/cells_all/<cell_id>/`` must pass the staged-cell
gate and its ``object.pt`` / ``train_budget_meta.json`` must hash to what the
derived sidecar recorded. That check reads nothing else in the parent dir —
in particular not ``superseded_by.json``, the snapshot record the sdf-base
fitting lane writes BEFORE it re-issues a cell under the same id (a
continuation: the EARLIER fit is snapshotted to a sibling bank and the dir's
CURRENT bytes become the continuation's). The volume's positive parent
(``constv2-mix-full__lora_r4_ce``) carries exactly such a file, and the sidecar
of the served derived object records the CONTINUATION as its parent
(``parent_pos.fit_fingerprint == superseded_by.expected_refit_fingerprint``).

So the sha gate PASSES on the volume's bytes and is silent about the marker —
correct, and not enough on its own: nothing would notice a derived sidecar
that named the EARLIER fit (the superseded one, whose bytes moved to the
snapshot bank) while the sha check happened to pass against a re-staged copy.
This module records the marker and asserts, per parent, that the fingerprint
the marker ANNOUNCES as the current publication is the fingerprint the
derived sidecar says it consumed. A mismatch refuses NAMING BOTH; an absent
marker is recorded as absent (most parents have none); the marker's own
``cell_id`` must name the parent it sits beside.

Read-only, CPU-cheap, model-free: the launcher's preflight runs it in the
volume-mounted container before any GPU minute, and a test runs it on a
synthetic bank shaped like the real one.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .train_budget import (DERIVED_META_NAME, _parent_blocks,
                           derived_object_problems, parent_cell_dir)

#: the snapshot record a continuation / refit writes beside the cell it is
#: about to supersede (scripts/modal/sdfbase_fits.py on the sdf-base lineage;
#: read there by scripts/sdfbase/probe_score._superseded_state)
SUPERSEDED_NAME = "superseded_by.json"

#: the three verdicts one parent's supersession check can reach
SUPERSEDED_ABSENT = "absent"
SUPERSEDED_CONTINUATION_IS_THIS_PARENT = "continuation_is_this_parent"
SUPERSEDED_MISMATCH = "mismatch"


def read_superseded_record(parent_dir) -> dict | None:
    """``<parent_dir>/superseded_by.json`` as a dict, ``None`` when the
    file is absent. An unreadable or non-object file is a ``ValueError``
    (a torn marker is a fact about the bank, never silently "absent")."""
    p = Path(parent_dir) / SUPERSEDED_NAME
    if not p.is_file():
        return None
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except ValueError as e:
        raise ValueError(f"unreadable {p}: {e}") from e
    if not isinstance(doc, dict):
        raise ValueError(f"{p} is not a JSON object")
    return doc


def superseded_parent_problems(parent_block: dict, record: dict | None,
                               *, where: str) -> list[str]:
    """Problem strings (empty = consistent) for ONE parent's supersession
    marker against the derived sidecar's block for that parent (``where``
    is the parent's role, prefixed onto every problem):

    - no marker: nothing to check (the common case);
    - the marker's ``cell_id`` (when present) must be this parent's;
    - ``superseded_by.expected_refit_fingerprint`` — the fingerprint the
      marker announces as the publication that supersedes the snapshotted
      one — must equal the sidecar's ``fit_fingerprint`` for this parent
      (the object the derivation consumed IS the continuation the marker
      announced). A marker whose announcement carries no fingerprint, or a
      sidecar block with none, cannot be cross-checked and is a problem
      too: the check must never pass by absence."""
    if record is None:
        return []
    problems: list[str] = []
    tag = f"{where}: "
    m_cell = record.get("cell_id")
    if m_cell is not None and m_cell != parent_block.get("cell_id"):
        problems.append(
            f"{tag}{SUPERSEDED_NAME} names cell_id={m_cell!r} but sits "
            f"beside the parent {parent_block.get('cell_id')!r} — the "
            "marker belongs to a different cell")
    sb = record.get("superseded_by")
    announced = (sb.get("expected_refit_fingerprint")
                 if isinstance(sb, dict) else None)
    recorded = parent_block.get("fit_fingerprint")
    if not announced:
        problems.append(
            f"{tag}{SUPERSEDED_NAME} announces no "
            "superseded_by.expected_refit_fingerprint — the continuation "
            "it introduced cannot be identified, so the parent the "
            "derivation consumed cannot be confirmed as it")
    if not recorded:
        problems.append(
            f"{tag}the derived sidecar records no fit_fingerprint for this "
            "parent — nothing to compare the marker's announced "
            "continuation against")
    if announced and recorded and str(announced) != str(recorded):
        problems.append(
            f"{tag}{SUPERSEDED_NAME} announces the continuation "
            f"expected_refit_fingerprint={announced!r} (superseding the "
            f"snapshotted fit {record.get('fit_fingerprint')!r}, snapshot "
            f"{record.get('snapshot')!r}) but the derived sidecar records "
            f"parent fit_fingerprint={recorded!r} — the object this "
            "derivation consumed is NOT the publication the marker names as "
            "current; refusing rather than serving a derivation of a "
            "superseded (or unannounced) parent")
    return problems


def preflight_derived_object(object_path, *, derived_id: str | None = None,
                             parents_root) -> dict[str, Any]:
    """The full derived-object preflight record for ONE staged object:
    the sha-bound verdict (:func:`train_budget.derived_object_problems`
    with ``parents_root``) and, per parent, the recorded
    ``superseded_by.json`` (or its absence) with the fingerprint
    cross-check above. Returns::

        {"object": str, "derived_id": str | None, "parents_root": str,
         "ok": bool, "problems": [str, ...],
         "parents": {role: {"cell_id", "cell_dir", "fit_fingerprint",
                            "superseded_by": dict | None,
                            "superseded_check": <verdict>}}}

    ``ok`` is False iff ``problems`` is non-empty; a supersession mismatch
    is a problem (it refuses), an absent marker is not. The sidecar is read
    once here for the parent blocks; a sidecar the sha verdict already
    refused still yields a record, so the caller sees BOTH verdicts."""
    obj_path = Path(object_path)
    problems = derived_object_problems(obj_path, derived_id=derived_id,
                                       parents_root=parents_root)
    parents_out: dict[str, Any] = {}
    meta_path = obj_path.parent / DERIVED_META_NAME
    blocks: dict[str, dict] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta, dict):
                blocks = _parent_blocks(meta)
        except ValueError:
            pass  # already a problem from derived_object_problems
    for role, block in blocks.items():
        cell_dir = parent_cell_dir(parents_root, block)
        entry: dict[str, Any] = {
            "cell_id": block.get("cell_id"), "cell_dir": str(cell_dir),
            "fit_fingerprint": block.get("fit_fingerprint"),
            "superseded_by": None, "superseded_check": SUPERSEDED_ABSENT}
        try:
            record = read_superseded_record(cell_dir)
        except ValueError as e:
            problems.append(f"{role}: {e}")
            entry["superseded_check"] = SUPERSEDED_MISMATCH
        else:
            if record is not None:
                entry["superseded_by"] = record
                sp = superseded_parent_problems(block, record, where=role)
                problems.extend(sp)
                entry["superseded_check"] = (
                    SUPERSEDED_CONTINUATION_IS_THIS_PARENT if not sp
                    else SUPERSEDED_MISMATCH)
        parents_out[role] = entry
    return {"object": str(obj_path), "derived_id": derived_id,
            "parents_root": str(parents_root), "ok": not problems,
            "problems": problems, "parents": parents_out}


def assert_derived_preflight(object_path, *, consumer: str,
                             derived_id: str | None = None,
                             parents_root) -> dict[str, Any]:
    """:func:`preflight_derived_object`, raising ``SystemExit`` naming the
    consumer on any problem; returns the record otherwise (the caller
    writes it beside its run record)."""
    rec = preflight_derived_object(object_path, derived_id=derived_id,
                                   parents_root=parents_root)
    if not rec["ok"]:
        raise SystemExit(
            f"[{consumer}] derived object at {object_path}"
            + (f" ({derived_id!r})" if derived_id else "")
            + " failed the derived preflight:\n  - "
            + "\n  - ".join(rec["problems"])
            + "\nRefusing to serve a derived object whose parents cannot be "
              "proven to be the ones its sidecar names.")
    return rec
