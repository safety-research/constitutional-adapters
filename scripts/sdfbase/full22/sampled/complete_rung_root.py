#!/usr/bin/env python
"""Make a static RUNG root self-contained: every (family, cell) of the run gets a shard00, including the pairs the
ladder never escalated because their SOURCE run had ZERO at-cap rows. For such a pair the rung's text IS the source
text, so the completion is a pure COPY-FORWARD of the source rows (``regenerated: false``, ``escalated_from_cap`` =
the source cap, ``source_cap`` = the source cap) plus a progress record per goal and an ``escalation.json`` with
``n_at_cap_before: 0`` — exactly what escalate_static.py would have written had it run on that pair. A pair whose
source has at-cap rows is REFUSED (that pair needs a real escalation, never a copy). Existing rung dirs are left
untouched; idempotent. Never touches the source root.

usage: complete_rung_root.py --private-root R --rung-run <..._sampled_t<T>_cap<N>> --source-run <..._sampled_t<T>[_cap<M>]>
                             --cap-from M --cap-to N --attacks a,b --cells c1,c2 [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path


def _rows(path: Path) -> list:
    out = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{i + 1}: not JSON ({e}); refusing")
    return out


def _write(path: Path, rows: list) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    os.replace(tmp, path)


def complete_pair(private_root: Path, rung_run: str, source_run: str, attack: str, cell: str, *,
                  cap_from: int, cap_to: int, dry_run: bool) -> dict:
    rung_dir = private_root / rung_run / attack / cell
    src_dir = private_root / source_run / attack / cell
    if rung_dir.is_dir():
        return {"attack": attack, "cell": cell, "status": "present", "changed": False}
    shards = sorted(p for p in src_dir.glob("shard*") if p.is_dir())
    if not shards:
        raise SystemExit(f"{src_dir}: source pair has no shards; refusing")
    rows, prog = [], []
    for s in shards:
        rows += _rows(s / "responses.jsonl"); prog += _rows(s / "progress.jsonl")
    at_cap = [r for r in rows if r.get("at_cap") is True or r.get("finish_reason") == "length"]
    if any("finish_reason" not in r or "decode" not in r for r in rows):
        raise SystemExit(f"{src_dir}: greedy-shaped rows (no finish_reason/decode); not a sampled source; refusing")
    if at_cap:
        raise SystemExit(f"{src_dir}: {len(at_cap)} at-cap rows — this pair NEEDS a real escalation (escalate_static.py), "
                         "never a copy-forward; refusing")
    if not dry_run:
        out = rung_dir / "shard00"; out.mkdir(parents=True, exist_ok=False)
        os.chmod(rung_dir, 0o700); os.chmod(out, 0o700)
        copied = []
        for r in rows:
            r2 = dict(r); r2.update({"regenerated": False, "escalated_from_cap": cap_from, "source_cap": int(r.get("source_cap", cap_from))})
            copied.append(r2)
        _write(out / "responses.jsonl", copied)
        seen = set(); p2 = []
        for p in prog:
            if p["goal_id"] in seen:
                continue
            seen.add(p["goal_id"]); q = dict(p); q["n_target_forwards"] = 0; p2.append(q)
        _write(out / "progress.jsonl", p2)
        esc = {"attack": attack, "defense": cell, "cap_from": cap_from, "cap_to": cap_to, "n_rows": len(copied),
               "n_at_cap_before": 0, "n_regenerated": 0, "n_at_cap_after": 0, "n_copied": len(copied),
               "n_prefix_extension": 0, "n_fork": 0, "n_identical": 0, "n_boundary_landing": 0,
               "seed_base": rows[0]["decode"].get("seed_base"), "trial": rows[0].get("trial"),
               "completed_by": "complete_rung_root.py (source pair had zero at-cap rows; pure copy-forward)",
               "source_run": source_run, "at": datetime.now(timezone.utc).isoformat()}
        (out / "escalation.json").write_text(json.dumps(esc, indent=1) + "\n")
    return {"attack": attack, "cell": cell, "status": "completed" if not dry_run else "would_complete",
            "n_rows": len(rows), "n_goals": len({p["goal_id"] for p in prog}), "changed": not dry_run}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--private-root", required=True); ap.add_argument("--rung-run", required=True)
    ap.add_argument("--source-run", required=True); ap.add_argument("--cap-from", type=int, required=True)
    ap.add_argument("--cap-to", type=int, required=True); ap.add_argument("--attacks", required=True)
    ap.add_argument("--cells", required=True); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.private_root)
    recs = [complete_pair(root, a.rung_run, a.source_run, fam, cell, cap_from=a.cap_from, cap_to=a.cap_to, dry_run=a.dry_run)
            for fam in a.attacks.split(",") for cell in a.cells.split(",")]
    for r in recs:
        print(json.dumps(r))
    print(json.dumps({"TOTAL": {"present": sum(r["status"] == "present" for r in recs),
                                "completed": sum(r["status"] in ("completed", "would_complete") for r in recs)}, "dry_run": a.dry_run}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
