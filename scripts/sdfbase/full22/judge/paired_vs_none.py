"""Every cell's over-refusal PAIRED against its own view's unsteered `none`
cell, item by item (the same pinned prompts, so the pairing is exact).

Per (target, cell, control):
  * rate_cell, rate_none        full_refusal / n_judged_both, over the items
                                JUDGED in both cells (an item abstained in
                                either is excluded and counted);
  * delta = rate_cell - rate_none  with a paired Wald 95% CI from the
                                discordant counts (b = refused in cell only,
                                c = refused in none only):
                                SE = sqrt(b + c - (b - c)^2 / n) / n;
  * mcnemar_exact_p             two-sided exact binomial on (b, c).

Writes <root>/<target>/judge_paired_vs_none.json and prints a table.
usage: python paired_vs_none.py [--root DIR]
"""
import argparse
import json
import math
import sys
from pathlib import Path

ROOT = "$GRADE/sdfbase_full22/controls"   # under the grade root, $GRADE: expanded at use
CONTROLS = ("xstest", "or_bench_hard_1k")


def _site_path(value) -> Path:
    """``value`` with the grade root's placeholder ($GRADE) expanded through sisteer.site, ``~`` expanded."""
    src = str(Path(__file__).resolve().parents[4] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer import site
    return site.expand_path(value)


def load_items(cell_dir: Path):
    out = {}
    for line in open(cell_dir / "judge_items.jsonl"):
        r = json.loads(line)
        if r["control"] == "xstest" and r.get("subset") != "safe":
            continue  # over-refusal is the safe-subset measure
        out.setdefault(r["control"], {})[r["item_id"]] = r
    return out


def binom_two_sided_p(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    # exact two-sided: 2 * P(X <= k) under Binom(n, 0.5), capped at 1
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def paired(cell_items, none_items, terminated_only=False):
    ids = [i for i in cell_items if i in none_items]
    if terminated_only:
        # only items whose generation TERMINATED in BOTH cells (at_cap from cap_condition.py)
        ids = [i for i in ids if not cell_items[i].get("at_cap", False)
               and not none_items[i].get("at_cap", False)]
    both = [i for i in ids if cell_items[i]["status"] == "judged" and none_items[i]["status"] == "judged"]
    n = len(both)
    rc = {i: cell_items[i]["label"] == "full_refusal" for i in both}
    rn = {i: none_items[i]["label"] == "full_refusal" for i in both}
    b = sum(1 for i in both if rc[i] and not rn[i])
    c = sum(1 for i in both if rn[i] and not rc[i])
    a = sum(1 for i in both if rc[i] and rn[i])
    rate_cell = sum(rc.values()) / n if n else None
    rate_none = sum(rn.values()) / n if n else None
    delta = (b - c) / n if n else None
    se = math.sqrt(max(0.0, (b + c) - (b - c) ** 2 / n)) / n if n else None
    return {"n_pairs": len(ids), "n_judged_both": n,
            "n_excluded_abstained_either": len(ids) - n,
            "rate_cell": rate_cell, "rate_none": rate_none, "delta": delta,
            "delta_wald95": ([delta - 1.96 * se, delta + 1.96 * se] if n else None),
            "refused_both": a, "refused_cell_only": b, "refused_none_only": c,
            "mcnemar_exact_p": binom_two_sided_p(b, c)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--targets", default="", help="comma list of target dirs (default all)")
    args = ap.parse_args()
    root = _site_path(args.root)
    print(f"{'target':13s} {'cell':46s} {'control':16s} {'set':4s} {'cell%':>7s} {'none%':>7s} "
          f"{'delta pp':>9s} {'wald95 pp':>18s} {'n':>5s} {'b/c':>9s} {'p':>7s}")
    want = {x for x in args.targets.split(",") if x}
    for t in sorted(p.name for p in root.iterdir() if (p / "cells").is_dir()):
        if want and t not in want:
            continue
        cells = sorted((root / t / "cells").glob("*__controls__*"))
        none_dirs = [d for d in cells if d.name.startswith("none__controls__")]
        if len(none_dirs) != 1:
            print(f"{t}: expected exactly one none cell, found {[d.name for d in none_dirs]}")
            continue
        none_items = load_items(none_dirs[0])
        out = {"target": t, "none_cell": none_dirs[0].name, "cells": {}}
        for d in cells:
            if not (d / "judge_items.jsonl").exists():
                print(f"{t}/{d.name}: no judge_items.jsonl"); continue
            ci = load_items(d)
            cid = json.loads((d / "judge_summary.json").read_text())["cell_id"]
            out["cells"][d.name] = {"cell_id": cid}
            for control in CONTROLS:
                if control not in ci or control not in none_items:
                    continue
                r = paired(ci[control], none_items[control])
                rt = paired(ci[control], none_items[control], terminated_only=True)
                rt["definition"] = "same pairing restricted to items whose generation terminated in BOTH cells"
                out["cells"][d.name][control] = {**r, "terminated_in_both": rt}
                for tag, rr in (("all", r), ("term", rt)):
                    if not rr["n_judged_both"]:
                        print(f"{t:13s} {cid:46s} {control:16s} {tag:4s} no pairs"); continue
                    w = rr["delta_wald95"]
                    print(f"{t:13s} {cid:46s} {control:16s} {tag:4s} {100*rr['rate_cell']:7.2f} {100*rr['rate_none']:7.2f} "
                          f"{100*rr['delta']:+9.2f} [{100*w[0]:+7.2f},{100*w[1]:+7.2f}] {rr['n_judged_both']:5d} "
                          f"{rr['refused_cell_only']:4d}/{rr['refused_none_only']:<4d} {rr['mcnemar_exact_p']:7.4f}")
        (root / t / "judge_paired_vs_none.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
