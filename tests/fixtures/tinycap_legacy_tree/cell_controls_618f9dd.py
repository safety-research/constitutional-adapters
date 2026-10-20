"""The ``controls_summary.json`` WRITER of ``src/sisteer/eval/cell_controls.py``
at 618f9dd, verbatim (``write_controls_summary`` + ``_assert_counts_only`` and
the two names they read) — the pre-source-discipline writer: it OVERLAYS the
offered payloads onto the existing ``controls`` and rewrites the document
with no ``sources`` block. A test fixture standing in for that tree's
``sisteer.eval.cell_controls`` inside a synthetic staged tree (no torch
import chain: the rest of the module is not needed to reproduce E17 A1).
Do not modernise it.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

#: the writer reads one attribute off the cell, ``.id`` (the annotation below
#: is never evaluated: ``from __future__ import annotations``)
DefenseCell = object
SUMMARY_FILENAME = "controls_summary.json"
_SAFE_STRING = re.compile(r"^[A-Za-z0-9_.:/+-]{1,80}$")


def _assert_counts_only(payloads, path: str = "controls") -> None:
    """Every control payload is a dict of labels/fracs/counts/hashes ONLY
    (D15/JB4): numbers, bools, None, and short whitespace-free strings. The
    error names the JSON path and the leaf's type/length — never its content
    (the jailbreak_artifacts leak-free convention)."""

    def describe(p, v):
        extra = f", len={len(v)}" if isinstance(v, str) else ""
        return f"{p}: disallowed {type(v).__name__}{extra}"

    def walk(node, p):
        if isinstance(node, dict):
            for k, v in node.items():
                if not (isinstance(k, str) and _SAFE_STRING.match(k)):
                    raise ValueError(f"{describe(p, k)} (dict key)")
                walk(v, f"{p}.{k}")
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                walk(v, f"{p}[{i}]")
        elif isinstance(node, bool) or node is None or isinstance(node, (int, float)):
            return
        elif isinstance(node, str):
            if not _SAFE_STRING.match(node):
                raise ValueError(describe(p, node))
        else:
            raise ValueError(describe(p, node))

    if not isinstance(payloads, dict):
        raise ValueError(f"{path}: control payloads must be a dict")
    for control, payload in payloads.items():
        if not isinstance(payload, dict):
            raise ValueError(
                f"{path}.{control}: a control payload must be a dict of "
                "counts/fracs/hashes")
    walk(payloads, path)



def write_controls_summary(out_root, cell: DefenseCell, payloads: dict,
                           provenance: dict):
    """Atomically write/merge ``<out_root>/<cell.id>/controls_summary.json``
    ({control: payload} + provenance; labels/fracs/counts/hashes ONLY —
    asserted). Merging into an existing summary requires the IDENTICAL
    provenance — two provenances in one summary would be exactly the stale
    fill r1 #4 forbids."""
    _assert_counts_only(payloads)
    _assert_counts_only({"provenance": provenance}, path="")
    path = Path(out_root) / cell.id / SUMMARY_FILENAME
    merged = dict(payloads)
    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("provenance") != provenance:
            raise ValueError(
                f"[cell-controls] {path} exists with a DIFFERENT provenance "
                "— refusing to merge results across provenances (r1 #4); "
                "delete the stale summary deliberately if the cell really "
                "was regenerated")
        merged = {**existing.get("controls", {}), **payloads}
    doc = {"cell_id": cell.id, "controls": merged, "provenance": provenance}
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w") as f:
            f.write(json.dumps(doc, indent=1, sort_keys=True))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path
