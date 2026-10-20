#!/usr/bin/env python3
"""Stamp `degenerate` / `degenerate_reason` on every row of a store.

FLAG, DO NOT FILTER. Rows stay in the pool so the pool
composition still matches the sibling models; the flag lets a consumer
condition on it. See sisteer.eval.degeneracy for what is detected and why.

Rewrites each family file atomically (tmp + rename) so an interruption cannot
leave a half-written store. Idempotent: re-running restamps to the same values.
Never run this against a store while a grader is rewriting it -- both do
atomic per-file rewrites and the later writer wins, silently dropping the
other's work.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sisteer.eval.degeneracy import classify                # noqa: E402

# The grade cache also lives in the store dir and is NOT a family file; a bare
# *.jsonl glob picks it up and corrupts the count (bitten twice on this lane).
FAMILIES = ["bon", "pasttense", "codeattack", "renellm", "grok420",
            "wildjailbreak_harmful", "phtest", "wildjailbreak_benign"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    store = Path(a.store_dir).expanduser()
    per = collections.defaultdict(lambda: [0, 0])
    cell = collections.defaultdict(lambda: [0, 0])
    total = flagged = 0

    for fam in FAMILIES:
        p = store / f"{fam}.jsonl"
        if not p.exists():
            print(f"[flag] missing {p.name} — skipping")
            continue
        rows = []
        for line in p.open():
            if not line.strip():
                continue
            r = json.loads(line)
            v = classify(r.get("response"))
            r.update(v)
            rows.append(r)
            c = r.get("si_condition")
            total += 1
            flagged += v["degenerate"]
            per[c][0] += 1
            per[c][1] += v["degenerate"]
            cell[(fam, c)][0] += 1
            cell[(fam, c)][1] += v["degenerate"]
        if not a.dry_run:
            tmp = p.with_suffix(".jsonl.tmp")
            with tmp.open("w") as fh:
                for r in rows:
                    fh.write(json.dumps(r) + "\n")
            os.replace(tmp, p)
        print(f"[flag] {fam:24s} {len(rows):6d} rows")

    print(f"\n[flag] {flagged}/{total} = {flagged/max(total,1):.2%} flagged"
          + ("  (DRY RUN — nothing written)" if a.dry_run else ""))
    print("[flag] by condition:")
    for c in sorted(per, key=str):
        n, d = per[c]
        print(f"    {str(c):18s} {d:5d} / {n:5d} = {d/n:6.2%}")
    worst = sorted(cell.items(), key=lambda kv: -kv[1][1] / max(kv[1][0], 1))[:5]
    print("[flag] worst cells:")
    for (fam, c), (n, d) in worst:
        if d:
            print(f"    {fam:22s} {str(c):16s} {d:4d}/{n:5d} = {d/n:6.2%}")


if __name__ == "__main__":
    main()
