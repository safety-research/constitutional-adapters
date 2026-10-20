"""Per-cell report over the judge_summary.json files: XSTest (safe) and
OR-Bench-Hard over-refusal bounds, judge abstention rate, n, at-cap share,
plus flags (abstention > 3%, harness ceiling refusal, foreign-shape verdicts).

usage: python report_table.py [--root DIR] [--json OUT]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = "$GRADE/sdfbase_full22/controls"   # under the grade root, $GRADE: expanded at use


def _site_path(value) -> Path:
    """``value`` with the grade root's placeholder ($GRADE) expanded through sisteer.site, ``~`` expanded."""
    src = str(Path(__file__).resolve().parents[4] / "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    from sisteer import site
    return site.expand_path(value)


def pct(x):
    return "   n/a" if x is None else f"{100 * x:6.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--targets", default="", help="comma list of target dirs (default all)")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    root = _site_path(args.root)
    rows = []
    want = {x for x in args.targets.split(",") if x}
    for t in sorted(p.name for p in root.iterdir() if (p / "cells").is_dir()):
        if want and t not in want:
            continue
        for d in sorted((root / t / "cells").glob("*__controls__*")):
            sp = d / "judge_summary.json"
            if not sp.exists():
                rows.append({"target": t, "cell_id": d.name.split("__controls__")[0], "status": "NOT JUDGED"})
                continue
            s = json.loads(sp.read_text())
            r = {"target": t, "cell_id": s["cell_id"], "cell_dir": d.name}
            for control, short in (("xstest", "xst"), ("or_bench_hard_1k", "orb")):
                b = s["controls"].get(control)
                if not b:
                    r[f"{short}_status"] = "missing"
                    continue
                o = b["over_refusal"]
                r[f"{short}_status"] = b["status"]
                r[f"{short}_rate"] = o["point_estimate_over_judged"]
                r[f"{short}_wilson"] = o["wilson95_over_judged"]
                r[f"{short}_band"] = o["abstention_band_full_n"]
                r[f"{short}_partial"] = o["partial_refusal_frac_of_judged"]
                r[f"{short}_n"] = o["n"]
                r[f"{short}_n_judged"] = o["n_judged"]
                r[f"{short}_n_abst"] = o["n_abstained"]
                r[f"{short}_abst_rate"] = o["abstention_rate"]
                r[f"{short}_abst_by_reason"] = o["abstained_by_reason"]
                r[f"{short}_at_cap"] = b["at_cap_share"]
                r[f"{short}_api_calls"] = b["judge_counters_delta"]["api_calls"]
                r[f"{short}_cache_hits"] = b["judge_counters_delta"]["cache_hits"]
                r[f"{short}_foreign"] = b["n_verdicts_with_foreign_keys"]
                c = b.get("over_refusal_terminated_only")
                if c:
                    r[f"{short}_term_rate"] = c["rate"]
                    r[f"{short}_term_wilson"] = c["wilson95"]
                    r[f"{short}_term_n"] = c["n_terminated_judged"]
                    r[f"{short}_term_thin"] = c["thin"]
                    r[f"{short}_cap_rate"] = c["capped_subset"]["rate"]
                    r[f"{short}_cap_n"] = c["capped_subset"]["n_judged"]
                if control == "xstest":
                    a = b["all"]
                    r["xst_all_n_abst"] = a["n_abstained"]
                    r["xst_all_abst_rate"] = a["abstention_rate"]
                    r["xst_unsafe_refusal"] = b["unsafe"]["point_estimate_over_judged"]
            rows.append(r)
    hdr = (f"{'target':13s} {'cell':46s} | {'XST safe: all [w95] | terminated-only [w95] n':52s} | "
           f"{'ORB: all [w95] | terminated-only [w95] n':52s} | {'cap x/o':8s} | flags")
    print(hdr); print("-" * len(hdr))
    def fmt(short, r):
        if f"{short}_rate" not in r:
            return "missing".ljust(52)
        w = r[f"{short}_wilson"]
        s1 = f"{100*r[f'{short}_rate']:5.1f} [{100*w[0]:4.1f},{100*w[1]:4.1f}]"
        if f"{short}_term_rate" in r and r[f"{short}_term_rate"] is not None:
            tw = r[f"{short}_term_wilson"]
            s1 += (f" | {100*r[f'{short}_term_rate']:5.1f} [{100*tw[0]:4.1f},{100*tw[1]:4.1f}] "
                   f"n={r[f'{short}_term_n']}{'*' if r[f'{short}_term_thin'] else ''}")
        else:
            s1 += " | n/a"
        return s1.ljust(52)
    for r in rows:
        if r.get("status") == "NOT JUDGED":
            print(f"{r['target']:13s} {r['cell_id']:46s} | NOT JUDGED"); continue
        flags = []
        for short in ("xst", "orb"):
            if r.get(f"{short}_status") not in ("ok",):
                flags.append(f"{short}:{r.get(f'{short}_status')}")
            if (r.get(f"{short}_abst_rate") or 0) > 0.03:
                flags.append(f"{short}:abst>3%")
            if r.get(f"{short}_term_thin"):
                flags.append(f"{short}:thin")
            if r.get(f"{short}_foreign"):
                flags.append(f"{short}:foreign={r[f'{short}_foreign']}")
        if (r.get("xst_all_abst_rate") or 0) > 0.03:
            flags.append("xst(all450):abst>3%")
        print(f"{r['target']:13s} {r['cell_id']:46s} | {fmt('xst', r)} | {fmt('orb', r)} | "
              f"{r.get('xst_at_cap', float('nan')):.2f}/{r.get('orb_at_cap', float('nan')):.2f} | {' '.join(flags)}")
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1, sort_keys=True) + "\n")
        print("wrote", args.json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
