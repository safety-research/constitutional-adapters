"""Pre-judge snapshot, run ON the cluster (pipe over ssh stdin): audits every served
controls store (system_sha / judge model / gen config / spool counts / at-cap shares / no
judged rows yet) and writes .rows_sha_before_judge.json = {target/cell: {rows_sha, rows_bytes}}
next to the stores. pushback_judged.py refuses any cell whose remote rows.jsonl no longer
matches this snapshot or whose local rows.jsonl does not start with these bytes.
"""
import json, glob, os, collections, hashlib
# the controls root on the cluster: F22_CONTROLS_REMOTE, else under the cluster's shared root (SISTEER_SHARED_ROOT, else
# CLUSTER_SHARED_ROOT: the variables sisteer.site reads for $SHARED; this script runs from stdin on the cluster, outside a checkout)
_SHARED = os.environ.get("SISTEER_SHARED_ROOT") or os.environ.get("CLUSTER_SHARED_ROOT")
R = os.environ.get("F22_CONTROLS_REMOTE") or (f"{_SHARED}/exp/sdfbase_full22/controls" if _SHARED else None)
if not R:
    raise SystemExit("set F22_CONTROLS_REMOTE (the controls root on the cluster) or SISTEER_SHARED_ROOT (its shared root)")
out = {}
for t in sorted(d for d in os.listdir(R) if os.path.isdir(f"{R}/{d}/cells")):
    for c in sorted(glob.glob(f"{R}/{t}/cells/*controls*")):
        m = json.load(open(c + "/manifest.json")); rc = m["resolved_config"]
        gs = json.load(open(c + "/gen_stats.json"))
        def n(f):
            with open(c + "/private/" + f) as fh:
                next(fh); return sum(1 for _ in fh)
        fx = gs["xstest"]["finish_reasons"]; fo = gs["or_bench_hard_1k"]["finish_reasons"]
        rows = collections.Counter()
        for line in open(c + "/rows.jsonl"):
            r = json.loads(line); rows[r["task"].split("/")[0]] += 1
        sha = hashlib.sha256(open(c + "/rows.jsonl", "rb").read()).hexdigest()
        name = os.path.basename(c)
        out[f"{t}/{name}"] = {"rows_sha": sha, "rows_bytes": os.path.getsize(c + "/rows.jsonl")}
        print(t, name[:58].ljust(58), "sys", rc.get("system_sha"), "judge", rc.get("judge_model"),
              "limit", rc.get("limit"), "gen", rc["gen"]["max_new_tokens"], rc["gen"]["do_sample"], rc["gen"]["batch_size"],
              "rev", m["model_revision"][:8], "xst", n("gens_xstest_v2.jsonl"), "cap=%.2f" % (fx.get("length", 0) / 450),
              "orb", n("gens_or_bench_hard_1k.jsonl"), "cap=%.2f" % (fo.get("length", 0) / 1319),
              "rows", dict(rows), "judged?", any(k.startswith("judged") for k in rows))
snap = f"{R}/.rows_sha_before_judge.json"
prev = json.load(open(snap)) if os.path.exists(snap) else {}
prev.update(out)
json.dump(prev, open(snap, "w"), indent=1)
print("wrote .rows_sha_before_judge.json with", len(out), "cells")
