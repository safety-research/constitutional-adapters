"""Read back the CLUSTER stores after push-back and report, per cell: judged
row counts per (task, metric), the bounded aggregates present, and whether
judge_summary.json / judge_items.jsonl landed. Runs ON the login host (pipe over
ssh stdin); prints one line per cell and a totals line. Read-only.
"""
import collections
import glob
import json
import os

# the controls root on the cluster: F22_CONTROLS_REMOTE, else under the cluster's shared root (SISTEER_SHARED_ROOT, else
# CLUSTER_SHARED_ROOT: the variables sisteer.site reads for $SHARED; this script runs from stdin on the cluster, outside a checkout)
_SHARED = os.environ.get("SISTEER_SHARED_ROOT") or os.environ.get("CLUSTER_SHARED_ROOT")
R = os.environ.get("F22_CONTROLS_REMOTE") or (f"{_SHARED}/exp/sdfbase_full22/controls" if _SHARED else None)
if not R:
    raise SystemExit("set F22_CONTROLS_REMOTE (the controls root on the cluster) or SISTEER_SHARED_ROOT (its shared root)")
tot = collections.Counter()
for t in sorted(d for d in os.listdir(R) if os.path.isdir(f"{R}/{d}/cells")):
    for c in sorted(glob.glob(f"{R}/{t}/cells/*__controls__*")):
        cnt = collections.Counter()
        aggs = {}
        for line in open(c + "/rows.jsonl"):
            r = json.loads(line)
            if r["task"].startswith("judged/"):
                cnt[(r["task"].split("/")[1], r["metric"])] += 1
                if r["item_id"] == "__aggregate__" and r["metric"].endswith("refused_bounds_aggregate"):
                    x = json.loads(r["extra_json"])
                    aggs[r["metric"]] = (x.get("refused_lo", x.get("orb_refused_lo")),
                                         x.get("refused_hi", x.get("orb_refused_hi")),
                                         x.get("n_abstained"))
        xst = cnt[("xstest_v2", "xst_label")]
        orb = cnt[("or_bench_hard_1k", "orb_label")]
        orb_r = cnt[("or_bench_hard_1k", "orb_refused")]
        has_sum = os.path.exists(c + "/judge_summary.json")
        has_items = os.path.exists(c + "/judge_items.jsonl")
        n_items = sum(1 for _ in open(c + "/judge_items.jsonl")) if has_items else 0
        tot["cells"] += 1
        tot["xst_label"] += xst
        tot["orb_label"] += orb
        tot["summaries"] += int(has_sum)
        print(f"{t:13s} {os.path.basename(c)[:58]:58s} xst_label={xst:3d} orb_label={orb:4d} "
              f"orb_refused={orb_r:4d} bounds={ {k.replace('_refused_bounds_aggregate', ''): v for k, v in aggs.items()} } "
              f"summary={has_sum} items={n_items}")
print("TOTAL", dict(tot))
