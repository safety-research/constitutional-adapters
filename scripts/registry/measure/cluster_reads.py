#!/usr/bin/env python3
"""cluster_reads.py -- per adapter of a cluster-read window, the read wall
against S10's <= 3 min bound, the refusals, and the inbox conflict count after the window.

    python scripts/registry/measure/cluster_reads.py [--window W] [--root ~/registry] [--state ~/.local/state/registry-landing] [--json]

Reads, for the named window (default the latest plan under <root>/state/cluster_reads/): the plan (routes, pins, refusals, stages),
the shard summaries pulled beside it (<plan>.summaries/<adapter>/<i>-of-<K>.json: wall per shard, groups seen / published / held /
refused / refused-known / not owned, inputs hashed and checked against the manifest, the admission timings), the window's stage stamps
from the chain log (the `-- cluster reads submit` .. `-- cluster reads pull` span; the `REFUSED` adapter stamps), and
`find <root>/records -name '*.conflict-*' | wc -l` (S11: 0 after a Slurm read). Per adapter: route, shards reported / expected, the
max shard wall (the adapter's read wall on the cluster), the refusal reasons, and the S10 verdict (<= 180 s, or `refused` when the
grading host read it, or `missing` when a shard never reported). sampled_static is reported as measured, never relaxed.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

S10_BOUND_S = 180.0
STAMP_RE = re.compile(r"^(?:-- |--- )(?P<label>.*?)\s(?P<hms>\d\d:\d\d:\d\d)")


def _load(path: str) -> Optional[Dict[str, Any]]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            d = json.load(fh)
        return d if isinstance(d, dict) else None
    except (OSError, ValueError):
        return None


def latest_plan_path(root: str) -> Optional[str]:
    d = os.path.join(root, "state", "cluster_reads")
    if not os.path.isdir(d):
        return None
    names = sorted(n for n in os.listdir(d) if n.endswith(".json"))
    return os.path.join(d, names[-1]) if names else None


def shard_summaries(plan_path: str) -> Dict[str, List[Dict[str, Any]]]:
    base = plan_path[:-len(".json")] + ".summaries"
    out: Dict[str, List[Dict[str, Any]]] = {}
    if not os.path.isdir(base):
        return out
    for a in sorted(os.listdir(base)):
        ad = os.path.join(base, a)
        if os.path.isdir(ad):
            out[a] = [d for fn in sorted(os.listdir(ad)) if fn.endswith(".json") for d in [_load(os.path.join(ad, fn))] if d]
    return out


def conflict_files(root: str) -> int:
    n = 0
    for _dp, _dn, fns in os.walk(os.path.join(root, "records")):
        n += sum(1 for f in fns if ".conflict-" in f)
    return n


def window_stamps(log_path: str, window: str) -> Dict[str, str]:
    """The cluster-read stage stamps (label -> HH:MM:SS) of the chain-log slice whose `cluster-reads plan --window <window>` call it holds;
    the last slice when the window is not found by name."""
    if not os.path.isfile(log_path):
        return {}
    slices: List[List[str]] = []
    for line in open(log_path, encoding="utf-8", errors="replace"):
        if line.startswith("=== ff chain start") or line.startswith("=== migrate chain start"):
            slices.append([])
        if slices:
            slices[-1].append(line.rstrip("\n"))
    chosen = next((s for s in reversed(slices) if any(window in l for l in s)), slices[-1] if slices else [])
    out: Dict[str, str] = {}
    for l in chosen:
        m = STAMP_RE.match(l)
        if m:
            out[m.group("label")] = m.group("hms")
    return out


def _span_s(stamps: Dict[str, str], start: str, end: str) -> Optional[float]:
    a, b = stamps.get(start), stamps.get(end)
    if not a or not b:
        return None
    ta = datetime.strptime(a, "%H:%M:%S"); tb = datetime.strptime(b, "%H:%M:%S")
    d = (tb - ta).total_seconds()
    return d if d >= 0 else d + 86400


def table(root: str, state: str, plan_path: Optional[str]) -> Dict[str, Any]:
    plan = _load(plan_path) if plan_path else None
    if plan is None:
        return {"error": "no plan", "conflict_files": conflict_files(root)}
    sums = shard_summaries(plan_path)
    stamps = window_stamps(os.path.join(state, "windows", "chain.log"), str(plan.get("window")))
    rows = []
    for a, e in sorted((plan.get("adapters") or {}).items()):
        docs = sums.get(a, [])
        walls = [float(d.get("wall_s") or 0) for d in docs if d.get("pass")]
        reported = len(docs)
        # C6: the plan records REFUSALS (an adapter the grading host never read, by reason); a pre-C6 plan file spelled the key `fallback`
        fb = ((plan.get("refusals") if plan.get("refusals") is not None else plan.get("fallback")) or {}).get(a) or []
        if e.get("route") != "cluster":
            verdict = "refused"
        elif reported < int(e.get("shards") or 0):
            verdict = "missing"
        elif fb:
            verdict = "refused"
        else:
            verdict = "ok" if walls and max(walls) <= S10_BOUND_S else "over"
        timings = [d["timings_s"] for d in docs if d.get("timings_s")]
        rows.append({"adapter": a, "route": e.get("route"), "shards": e.get("shards"), "reported": reported,
                     "read_wall_s": (round(max(walls), 1) if walls else None),
                     "pass_s_max": (round(max(float(t.get("pass", 0)) for t in timings), 1) if timings else None),
                     "admission_s_max": (round(max(float(t.get("admission_pre", 0)) + float(t.get("admission_post", 0)) for t in timings), 1) if timings else None),
                     "published": sum(len(d.get("published") or {}) for d in docs), "held": sum(len(d.get("held") or []) for d in docs),
                     "refused": sum(len(d.get("refused") or []) for d in docs), "refused_known": sum(len(d.get("refused_known") or []) for d in docs),
                     "not_owned": sum(int((d.get("pass") or {}).get("groups_not_owned") or 0) for d in docs),
                     "inputs_hashed": sum(int((d.get("pass") or {}).get("inputs_hashed") or 0) for d in docs),
                     "checked_against_manifest": sum(int((d.get("pass") or {}).get("inputs_checked_against_manifest") or 0) for d in docs),
                     "refused_roots": sorted({r.get("component") for d in docs for r in (d.get("refused_roots") or []) if r.get("component")}),
                     "refusals": fb, "s10": verdict, "bound_s": S10_BOUND_S})
    return {"window": plan.get("window"), "plan": plan_path, "tree_sha": plan.get("tree_sha"), "stages": plan.get("stages"),
            "array_span_s": _span_s(stamps, "cluster reads submit", "cluster reads revalidate"),
            "cluster_stages_span_s": _span_s(stamps, "reader-state export", "cluster reads pull"),
            "conflict_files": conflict_files(root), "rows": rows}


def render(t: Dict[str, Any]) -> str:
    if t.get("error"):
        return f"cluster reads: {t['error']}; conflict files under records/: {t['conflict_files']}"
    lines = [f"cluster reads window {t['window']} (tree {str(t.get('tree_sha') or '')[:9]}): array span {t['array_span_s']} s, cluster stages {t['cluster_stages_span_s']} s, "
             f"conflict files under records/: {t['conflict_files']} (S11 wants 0)",
             f"{'adapter':<20} {'route':<8} {'K':>2} {'rep':>3} {'wall s':>8} {'pass s':>8} {'adm s':>6} {'pub':>4} {'held':>4} {'ref':>4} {'known':>5} {'S10':<9} refusals"]
    for r in t["rows"]:
        lines.append(f"{r['adapter']:<20} {str(r['route']):<8} {str(r['shards']):>2} {r['reported']:>3} {str(r['read_wall_s']):>8} {str(r['pass_s_max']):>8} {str(r['admission_s_max']):>6} "
                     f"{r['published']:>4} {r['held']:>4} {r['refused']:>4} {r['refused_known']:>5} {r['s10']:<9} {'; '.join(r['refusals'])[:90]}")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=os.environ.get("REGISTRY_ROOT") or os.path.expanduser("~/registry"))
    ap.add_argument("--state", default=os.environ.get("REGISTRY_LANDING_STATE") or os.path.expanduser("~/.local/state/registry-landing"))
    ap.add_argument("--window", help="the window's name (default: the latest plan)")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    plan_path = os.path.join(a.root, "state", "cluster_reads", f"{a.window}.json") if a.window else latest_plan_path(a.root)
    t = table(a.root, a.state, plan_path)
    print(json.dumps(t, indent=1, default=str) if a.json else render(t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
