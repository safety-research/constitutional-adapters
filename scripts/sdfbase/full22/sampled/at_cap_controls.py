"""At-cap count of a SAMPLED served-controls store — the number the ladder's
caller reads to decide whether the next rung is launched at all
(scripts/sdfbase/full22/sampled/DESIGN.md, "At-cap counts").

  at_cap_controls.py --out-root ROOT --cell CELL [--control C] [--json]

Locates CELL's store under ROOT (``cells/*/manifest.json`` whose ``cell_id``
is CELL — exactly one, or a refusal), and per control reports ``n_at_cap``
(items whose finish_reason was ``length`` at this store's cap), ``n``, the cap
and the store dir. The count is read from BOTH records the producer wrote —
``gen_stats.json[control]`` (stamped only after ``assert_complete``) and the
per-item sidecar ``private/gens_<name>.meta.jsonl`` bound to the spooled
texts by sha256 — and the two must agree.

REFUSES, exit 2, never a number:
  * a store with no ``gen_stats.json`` (the run never completed);
  * a control absent from it, or one whose entry carries no ``decode`` — a
    GREEDY store has no per-item at-cap record and must never read as 0;
  * a store with no meta sidecar, a sidecar that does not bind every
    spooled item, or a count that disagrees with gen_stats.

Output: one line per control (``<control> n_at_cap=K n=N cap=C store=DIR``);
with ``--control C`` and no ``--json`` the bare integer K alone (for a shell
caller); with ``--json`` one JSON object. Deliberately self-contained (no
harness import — the driver's import costs seconds of torch); the driver's
own reader and this one are held to the same answers by
tests/test_sampled_controls.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

META_SUFFIX = ".meta.jsonl"
EXIT_REFUSED = 2


class Refusal(Exception):
    """A store this tool will not put a number on."""


def _text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def store_digest(texts: Dict[str, str]) -> str:
    """The content identity of ONE control's complete spool: sha256 over the sorted `item_id<TAB><text sha256>` lines. The ladder gate records
    it (ladder_decide: identity.store_digest) and the registry closer recomputes it, so a decision is bound to the GENERATION it judged — a
    regenerated spool at the same path no longer matches the old record."""
    h = hashlib.sha256()
    for iid in sorted(texts):
        h.update(f"{iid}\t{_text_sha(texts[iid])}\n".encode("utf-8"))
    return h.hexdigest()


def locate_store(out_root: Path, cell: str) -> Path:
    cells_dir = Path(out_root) / "cells"
    if not cells_dir.is_dir():
        raise Refusal(f"{out_root}: no cells/ dir — not a served controls root")
    matches: List[Path] = []
    for d in sorted(cells_dir.iterdir()):
        mp = d / "manifest.json"
        if not mp.is_file():
            continue
        try:
            m = json.loads(mp.read_text())
        except json.JSONDecodeError as e:
            raise Refusal(f"{mp}: unreadable manifest ({e})")
        if m.get("cell_id") == cell:
            matches.append(d)
    if len(matches) != 1:
        raise Refusal(f"{out_root}: {len(matches)} stores match cell {cell!r} "
                      f"(need exactly 1): {[str(d) for d in matches]}")
    return matches[0]


def _spool_texts(path: Path) -> Dict[str, str]:
    """{item_id: text} of a complete spool — header skipped, first write
    wins on an identical replay, a differing duplicate or a torn line
    refuses (a complete store has neither)."""
    if not path.is_file():
        raise Refusal(f"{path}: no spool")
    lines = path.read_text(encoding="utf-8").splitlines()
    out: Dict[str, str] = {}
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise Refusal(f"{path}: corrupt or torn line {i + 1}; not a complete store")
        if i == 0:
            if "__spool_header__" not in row:
                raise Refusal(f"{path}: first line is not a spool header")
            continue
        iid, text = row["item_id"], row["text"]
        if iid in out:
            if out[iid] != text:
                raise Refusal(f"{path}: item {iid!r} spooled twice with different text")
            continue
        out[iid] = text
    return out


def _meta_bound(meta_path: Path, texts: Dict[str, str]) -> Dict[str, dict]:
    """Per spooled item the LAST meta row whose text_sha256 matches the
    spooled text (the driver's GenMeta.bound rule); a spooled item with no
    matching row refuses."""
    if not meta_path.is_file():
        raise Refusal(f"{meta_path}: no meta sidecar — a greedy store, or one "
                      "written before the sidecar existed; no per-item at-cap "
                      "record, refusing to read it as 0")
    bound: Dict[str, dict] = {}
    for i, line in enumerate(meta_path.read_text(encoding="utf-8").splitlines()):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            raise Refusal(f"{meta_path}: corrupt or torn line {i + 1}; not a complete store")
        iid = row.get("item_id")
        if iid in texts and row.get("text_sha256") == _text_sha(texts[iid]):
            bound[iid] = row
    missing = [i for i in texts if i not in bound]
    if missing:
        raise Refusal(f"{meta_path}: {len(missing)} spooled item(s) have no bound meta "
                      f"row (first: {missing[:3]}); refusing")
    return bound


def read_control(cell_dir: Path, control: str) -> dict:
    cell_dir = Path(cell_dir)
    gs_path = cell_dir / "gen_stats.json"
    if not gs_path.is_file():
        raise Refusal(f"{cell_dir}: no gen_stats.json — the producer stamps it only after "
                      "every control is complete; this store is incomplete")
    all_gs = json.loads(gs_path.read_text())
    gs = all_gs.get(control)
    if not isinstance(gs, dict):
        raise Refusal(f"{gs_path}: no entry for control {control!r} "
                      f"(has {sorted(all_gs)})")
    if "decode" not in gs or "n_at_cap" not in gs or "at_cap_item_ids" not in gs:
        raise Refusal(f"{gs_path}: {control} carries no sampled decode record "
                      "(decode/n_at_cap/at_cap_item_ids) — a GREEDY store has no "
                      "per-item at-cap status and never reads as 0; refusing")
    spool_name = Path(str(gs.get("spool") or "")).name
    if not spool_name.endswith(".jsonl"):
        raise Refusal(f"{gs_path}: {control} names no spool")
    spool_path = cell_dir / "private" / spool_name
    texts = _spool_texts(spool_path)
    n_total = int(gs.get("n_total", -1))
    if len(texts) != n_total:
        raise Refusal(f"{spool_path}: {len(texts)} spooled items != gen_stats n_total="
                      f"{n_total}; incomplete or inconsistent store")
    meta_path = spool_path.with_name(spool_name[:-len(".jsonl")] + META_SUFFIX)
    bound = _meta_bound(meta_path, texts)
    at_cap = sorted(i for i, r in bound.items() if r.get("at_cap") is True)
    for i, r in bound.items():
        if not isinstance(r.get("at_cap"), bool):
            raise Refusal(f"{meta_path}: item {i!r} at_cap={r.get('at_cap')!r} is not a bool")
    if at_cap != sorted(gs["at_cap_item_ids"]) or len(at_cap) != int(gs["n_at_cap"]):
        raise Refusal(f"{cell_dir}: {control} meta sidecar says {len(at_cap)} at cap "
                      f"but gen_stats says n_at_cap={gs['n_at_cap']} "
                      f"({len(gs['at_cap_item_ids'])} ids); inconsistent store, refusing")
    cap = int(gs.get("cap", (gs.get("decode") or {}).get("cap", -1)))
    return {"control": control, "n_at_cap": len(at_cap), "n": n_total, "cap": cap,
            "at_cap_item_ids": at_cap,
            "n_regenerated": gs.get("n_regenerated"),
            "n_copied": gs.get("n_copied"),
            "n_floor_sample": gs.get("n_floor_sample"),
            "store": str(cell_dir)}


def count(out_root: Path, cell: str, control: Optional[str] = None) -> dict:
    """{"cell", "store", "controls": {control: read_control(...)}} for the
    requested control, else every sampled control gen_stats names."""
    cell_dir = locate_store(Path(out_root), cell)
    gs_path = cell_dir / "gen_stats.json"
    if not gs_path.is_file():
        raise Refusal(f"{cell_dir}: no gen_stats.json — the producer stamps it only after "
                      "every control is complete; this store is incomplete")
    all_gs = json.loads(gs_path.read_text())
    if control is not None:
        controls = [control]
    else:
        controls = [c for c, v in all_gs.items() if isinstance(v, dict) and "decode" in v]
        if not controls:
            raise Refusal(f"{gs_path}: no control carries a sampled decode record — a "
                          "greedy store never reads as 0 at cap; refusing")
    return {"cell": cell, "store": str(cell_dir),
            "controls": {c: read_control(cell_dir, c) for c in controls}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", required=True, help="the rung's controls out-root")
    ap.add_argument("--cell", required=True, help="the cell id (manifest cell_id)")
    ap.add_argument("--control", default=None,
                    help="one control (xstest | or_bench_hard_1k); alone, prints "
                         "the bare integer")
    ap.add_argument("--json", action="store_true", help="one JSON object instead")
    ap.add_argument("--share", action="store_true",
                    help="one JSON line per control {control, n_at_cap, n, cap, share} "
                         "(share = n_at_cap / n) — the ladder's share readout; with "
                         "--control a single line; the bare integer stays the default")
    args = ap.parse_args(argv)
    try:
        out = count(Path(args.out_root), args.cell, args.control)
    except Refusal as e:
        print(f"[at-cap-controls] REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    if args.json:
        print(json.dumps(out, sort_keys=True))
    elif args.share:
        for c, blk in out["controls"].items():
            print(json.dumps({"control": c, "n_at_cap": blk["n_at_cap"], "n": blk["n"],
                              "cap": blk["cap"],
                              "share": (blk["n_at_cap"] / blk["n"]) if blk["n"] else 0.0},
                             sort_keys=True))
    elif args.control is not None:
        print(out["controls"][args.control]["n_at_cap"])
    else:
        for c, blk in out["controls"].items():
            print(f"{c} n_at_cap={blk['n_at_cap']} n={blk['n']} cap={blk['cap']} "
                  f"store={blk['store']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
