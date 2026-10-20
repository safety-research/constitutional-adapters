#!/usr/bin/env python
"""petri_leg_select.py — the Petri driver's leg SELECTION from a FROZEN enumeration.

    petri_leg_select.py <cells.json> --cells <harness id>[,<harness id>...]        -> the legs' indices, space-separated
    petri_leg_select.py <cells.json> --indices "<i> <j> ..." --objects-only        -> the same indices back, or a refusal

``cells.json`` is the record ``petri_run.py print-cells`` freezes (``canonical`` + ``controls``, each entry a
``defense_cells.PetriCell`` as a dict); a leg's index is its position in ``canonical + controls`` — the SAME indexing
``petri_drive_full22.sh`` and ``petri_run.py run --cell-index`` use, so what this prints is what the driver drives.

WHY IT EXISTS. ``petri_drive_full22.sh ... ALL`` drives EVERY leg of the frozen record — the canonical cells AND the derived
controls leg (``none__lens0`` / ``none__lora0``) — and a lane that measures its steered objects and pairs them against the
campaign's EXISTING ``none`` anchor must never drive an unsteered leg (a fresh ``none`` audit is a new anchor, which the
2026-09-15 anchor ruling forbids). Selecting BY CELL from the frozen record, with the unsteered legs refused by construction,
is what makes "six objects only" a property of the command rather than of the operator's arithmetic:

* ``--cells``: every named id must be a CANONICAL leg whose class is an OBJECT class (``class_a`` / ``class_b`` / ``stacked``);
  ``none`` (class ``base``), a ``prompted__*`` leg and every controls leg refuse BY NAME; an id the record does not carry, or
  a duplicate, refuses. The indices come back in the record's order (not the argument's), one line, space-separated.
* ``--indices --objects-only``: an explicit index list (the plan's ``select_cells.py --petri-indices`` output, or a hand list)
  is HELD to the same rule — an index that names ``none``, a prompted leg or a controls leg refuses; out of range refuses.

The class is read from the entry's ``cell_class`` (what ``print-cells`` records); an entry without it (an older record) is
classified from its id through ``defense_cells.parse_cell_id`` — never from how the id looks.

Exit 0 with the indices on stdout; exit 2 with the reason on stderr for every refusal (the driver treats any non-zero as FATAL).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

_SRC = Path(__file__).resolve().parents[3] / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

#: the cell classes a MEASURED object leg may carry (defense_cells.CELL_CLASSES minus base / prompted / control)
OBJECT_CELL_CLASSES: Tuple[str, ...] = ("class_a", "class_b", "stacked")
#: the classes this selector refuses BY NAME (the unsteered anchor, the prompted baselines, the derived controls twin)
NON_OBJECT_CELL_CLASSES: Tuple[str, ...] = ("base", "prompted", "control")


class LegSelectError(ValueError):
    pass


def _cell_class(entry: Dict, *, is_control: bool) -> str:
    """The entry's class: what the frozen record says, else derived from its id through the harness grammar."""
    cc = entry.get("cell_class")
    if isinstance(cc, str) and cc:
        return cc
    if is_control:
        return "control"
    from sisteer.capeval.petri import defense_cells as dc  # lazy: torch-free, but keep the CLI light
    try:
        return dc.parse_cell_id(str(entry.get("cell_id", ""))).cell_class
    except Exception as e:  # noqa: BLE001 — the refusal names the entry
        raise LegSelectError(f"leg {entry.get('cell_id')!r} carries no cell_class and its id does not parse: {e}") from e


def load_legs(path: Path) -> List[Dict]:
    """``canonical + controls`` in the driver's index order, each entry annotated with ``_index`` / ``_class`` / ``_control``."""
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise LegSelectError(f"{path}: unreadable frozen enumeration ({e})") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("canonical"), list) or not isinstance(doc.get("controls"), list):
        raise LegSelectError(f"{path}: not a frozen enumeration record (needs `canonical` and `controls` lists)")
    legs: List[Dict] = []
    for i, entry in enumerate(list(doc["canonical"]) + list(doc["controls"])):
        if not isinstance(entry, dict) or not isinstance(entry.get("cell_id"), str) or not entry["cell_id"]:
            raise LegSelectError(f"{path}: leg {i} is not a cell record with a cell_id")
        is_control = i >= len(doc["canonical"])
        legs.append({**entry, "_index": i, "_control": is_control, "_class": _cell_class(entry, is_control=is_control)})
    return legs


def select_by_cells(legs: Sequence[Dict], cells: Sequence[str]) -> List[int]:
    """The indices of the OBJECT legs named in ``cells`` (record order); refuses an unknown id, a duplicate, and every
    non-object leg by name."""
    wanted = [c.strip() for c in cells if c.strip()]
    if not wanted:
        raise LegSelectError("--cells names no cell")
    dup = sorted({c for c in wanted if wanted.count(c) > 1})
    if dup:
        raise LegSelectError(f"--cells names a cell twice: {dup}")
    canonical = {leg["cell_id"]: leg for leg in legs if not leg["_control"]}
    control_ids = {leg["cell_id"] for leg in legs if leg["_control"]}
    out: List[int] = []
    for c in wanted:
        if c in control_ids and c not in canonical:
            raise LegSelectError(f"cell {c!r} is the derived CONTROLS leg — never a measured object leg; refusing")
        leg = canonical.get(c)
        if leg is None:
            raise LegSelectError(f"cell {c!r} is not a canonical leg of the frozen enumeration "
                                 f"(canonical: {sorted(canonical)})")
        if leg["_class"] not in OBJECT_CELL_CLASSES:
            raise LegSelectError(f"cell {c!r} is a {leg['_class']!r} leg, not an object leg "
                                 f"({OBJECT_CELL_CLASSES}) — the unsteered anchor and the prompted baselines are never "
                                 "measured by this selector; refusing")
        out.append(leg["_index"])
    return sorted(out)


def check_indices(legs: Sequence[Dict], indices: Sequence[int], *, objects_only: bool) -> List[int]:
    """Validate an explicit index list against the record: in range, distinct, and — under ``objects_only`` — every index an
    OBJECT leg (a ``none`` / prompted / controls index refuses by name)."""
    if not indices:
        raise LegSelectError("--indices names no leg")
    if len(set(indices)) != len(indices):
        raise LegSelectError(f"--indices names a leg twice: {sorted(indices)}")
    n = len(legs)
    for i in indices:
        if i < 0 or i >= n:
            raise LegSelectError(f"leg index {i} is outside the frozen enumeration (0..{n - 1})")
        leg = legs[i]
        if objects_only and leg["_class"] not in OBJECT_CELL_CLASSES:
            what = "the derived CONTROLS leg" if leg["_control"] else f"a {leg['_class']!r} leg"
            raise LegSelectError(f"leg {i} ({leg['cell_id']}) is {what}, not an object leg — refused under --objects-only")
    return sorted(indices)


def _parse_indices(text: str) -> List[int]:
    toks = [t for t in text.replace(",", " ").split() if t]
    out = []
    for t in toks:
        if not t.isdigit():
            raise LegSelectError(f"--indices: {t!r} is not a leg index")
        out.append(int(t))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("cells_json", type=Path, help="the frozen enumeration (petri_run.py print-cells --out-root <OUT>/cells.json)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--cells", default=None, help="comma list of HARNESS cell ids to select (object legs only)")
    g.add_argument("--indices", default=None, help="an explicit leg index list to validate (space / comma separated)")
    ap.add_argument("--objects-only", action="store_true",
                    help="with --indices: refuse an index that names `none`, a prompted leg or a controls leg (--cells always does)")
    a = ap.parse_args(argv)
    try:
        legs = load_legs(a.cells_json)
        if a.cells is not None:
            idx = select_by_cells(legs, a.cells.split(","))
        else:
            idx = check_indices(legs, _parse_indices(a.indices), objects_only=bool(a.objects_only))
    except LegSelectError as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(" ".join(str(i) for i in idx))
    return 0


if __name__ == "__main__":
    sys.exit(main())
