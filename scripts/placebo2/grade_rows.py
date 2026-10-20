#!/usr/bin/env python3
"""Grade the placebo-control rows with the SAME judge the store was graded by.

LiveStrongREJECTGrader (rubric v1, kind "sr"), model from configs/two_si.yaml
grader.model (claude-haiku-4-5), Low-Prio org key, item shape
(family, goal_id, attack_prompt, response) — byte-identical machinery to
build_two_si_pool.py grade-store, so the new arms' scores are comparable to
the store's by construction.

The on-disk ScoreCache can be SEEDED from the store's own
store_grade_cache.jsonl (--seed-cache): the cache key is
sha256(kind|family|goal_id|attack_prompt|response), so any regenerated
response byte-identical to a stored one (the sr_greedy fidelity arm, mostly)
reuses the store's judgment and costs nothing. The store file is only read.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

_WT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_WT / "src"))
sys.path.insert(0, str(_WT / "scripts"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--seed-cache", default=None,
                    help="existing ScoreCache jsonl to copy into --cache "
                         "(read-only source; only used when --cache is absent)")
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--concurrency", type=int, default=200)
    args = ap.parse_args()

    cache = Path(args.cache)
    if args.seed_cache and not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(args.seed_cache, cache)
        print(f"[grade] seeded cache from {args.seed_cache}")

    rows = [json.loads(l) for l in Path(args.rows).open() if l.strip()]
    todo = [r for r in rows if r.get("score") is None]
    items = [(r["attack"], r["goal_id"], r["attack_prompt"], r["response"])
             for r in todo]
    print(f"[grade] {len(rows)} rows, {len(todo)} ungraded, "
          f"{len(set(items))} unique judge items")

    from sisteer.eval.attacks.grading_live import LiveStrongREJECTGrader
    grader = LiveStrongREJECTGrader(
        model=args.model, concurrency=args.concurrency, cache_path=cache)
    scores = grader.grade(items)

    for r, it in zip(todo, items):
        r["score"] = float(scores[it])
        r["graded"] = True

    out = Path(args.out)
    tmp = out.with_suffix(".tmp")
    with tmp.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    tmp.rename(out)
    print(f"[grade] wrote {len(rows)} graded rows -> {out}")


if __name__ == "__main__":
    # A credential rejection that reaches main -- the shared judge's / grader's typed
    # error -- ends the process with the typed text as its terminal line, never a traceback
    from sisteer import vault  # noqa: E402

    with vault.credential_exit(prefix="[grade] "):
        main()
