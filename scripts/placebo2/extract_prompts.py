#!/usr/bin/env python3
"""Extract the paired prompt instances for the placebo-control regeneration.

Reads the labelled store READ-ONLY and writes one row per undefended
(`si_condition == "none"`, candidate_idx == 0) instance of the requested
families — exactly the instances the new arms must cover so every generated
row pairs 1:1 with a stored undefended row on
(goal_id, variant_name, sample_idx, trial).

Output rows carry only what generation needs (family, goal_id, sample_idx,
trial, category, source, side, variant_name, text). No responses and no
scores leave the store.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FAMILIES = ("pasttense", "amplegcg_plus")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--store-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--families", default=",".join(FAMILIES))
    args = ap.parse_args()

    rows = []
    for fam in [f.strip() for f in args.families.split(",") if f.strip()]:
        seen = set()
        with (Path(args.store_dir) / f"{fam}.jsonl").open() as fh:
            for line in fh:
                if not line.strip():
                    continue
                r = json.loads(line)
                if (r.get("si_condition") != "none"
                        or int(r.get("candidate_idx", 0)) != 0):
                    continue
                key = (r["goal_id"], int(r["sample_idx"]), int(r.get("trial", 0)))
                if key in seen:
                    raise SystemExit(f"duplicate none instance {fam}/{key}")
                seen.add(key)
                rows.append({
                    "family": r["attack"], "goal_id": r["goal_id"],
                    "sample_idx": int(r["sample_idx"]),
                    "trial": int(r.get("trial", 0)),
                    "category": r.get("category"), "source": r.get("source"),
                    "side": r.get("side", "harmful"),
                    "variant_name": r.get("variant_name"),
                    "text": r["attack_prompt"],
                })
        print(f"[extract] {fam}: {len(seen)} instances")
    rows.sort(key=lambda r: (r["family"], r["goal_id"],
                             r["sample_idx"], r["trial"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    print(f"[extract] wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
