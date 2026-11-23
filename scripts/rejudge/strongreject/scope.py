#!/usr/bin/env python3
"""The paper-row scope of the StrongREJECT re-judge: one JOB per paper run, naming the transcript stores its Haiku judgment graded.

    PYTHONPATH=src python scripts/rejudge/strongreject/scope.py --out SCOPE_DIR --goals GOALS.json \
        --static-map STATIC_MAP.json --mtbon-map MTBON_MAP.json --mtbon-manifests MANIFEST.jsonl [...] \
        [--static-goals STATIC_GOALS.json] [--site NAME=PREFIX ...] [--path-rewrite FROM=TO ...] [--runs RUNS.txt]

SCOPE: paper rows only -- runs with ``v_run_current.in_current_paper = 1`` on the eight jailbreak evals (codeattack, grok420, pasttense,
renellm, actorattack, crescendo, pair, strongreject_bon). The registry is read READ-ONLY. ``--runs FILE`` (one run id per line) replaces
the paper-row selection with exactly those runs (a SUPPLEMENT, e.g. the contrast anchors the paper rows are measured against, which are
not paper rows themselves); every other rule is the same. A listed run the registry does not hold on one of the eight evals refuses the
scope.

WHICH TEXTS. The registry keeps per-goal results, not the judged texts, so each run's Haiku judgment (judge ``strongreject_v1_haiku45``
50095c69, or ``strongreject_v2`` c10c253f for the best-of-N runs graded with the v2 rubric) is joined to the stores its grader read
through three maps, built from the stores themselves (every judgment's stores checked to hold exactly its goals, and the multi-turn
per-turn counts against the producers' own records). This script joins them to the paper runs and writes ``jobs.jsonl``;
:mod:`enumerate_units` reads the stores and applies the graders' rules. The maps:

  --static-map       JSON list, one entry per static judgment: ``{judgment_id, units: [<store path>...], unresolved: [...]}``
  --mtbon-map        JSON object, judgment id -> ``{units: [<uid>:<manifest dir name>...], method, fold?: {store: [...], climb: [...]}}``
                     (``fold``: a best-of-N reading folded from a 512-token store and its climb, which :mod:`scope_runs` refuses)
  --mtbon-manifests  JSON lines, one per multi-turn or best-of-N store: ``{uid, kind, unit_path, files: [[<path>, ...]...],
                     errored_goals?, refused_goals?}``; a unit's key is ``<uid>:<the manifest's directory name>``
  --goals            ``{goals: {goal_id: {dataset, revision, text}}, goal_sets: {mt: [...], bon: [...]}}`` (``goals.py`` writes it)
  --static-goals     optional: a JSON list of ``{goal_id, dataset, revision, text}`` cross-checked against ``--goals``

For a run the registry ties to its stores by itself, ``scope_runs.py --runs RUNS.tsv`` with ``store=@judgment`` builds the same jobs from
the registry alone (no maps), except a folded best-of-N reading.

Each job: ``{run_id, campaign, eval_id, category (static|mt|bon), kind (sr|sr2), haiku_judge_id, haiku_judgment_id, units, fold,
sites}``; a unit is ``{type, path, files?, drop_goals?}``. A store's SITE is the first ``--site NAME=PREFIX`` its path starts with
(``local`` = the home directory, always declared); :mod:`enumerate_units` runs once per site. ``--path-rewrite FROM=TO`` maps a path a
map recorded through a copy back to the original store's path. ``goals.json`` holds every goal's (dataset, revision, text) -- the cache
key and the prompt use the ORIGINAL HarmBench / JBB goal text, never the attack prompt.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sqlite3
import sys

JAILBREAK_EVALS = ("codeattack", "grok420", "pasttense", "renellm", "actorattack", "crescendo", "pair", "strongreject_bon")
CATEGORY = {"codeattack": "static", "grok420": "static", "pasttense": "static", "renellm": "static",
            "actorattack": "mt", "crescendo": "mt", "pair": "mt", "strongreject_bon": "bon"}
HAIKU_V1 = "50095c69789db8b367cd38cb167ed9a768606bfac3aa843d035a4cfa5a233dfa"
HAIKU_V2 = "c10c253f156c334234682c802c7b412132b5878861d5b992eb129beddd0d8539"
KIND_OF_JUDGE = {HAIKU_V1: "sr", HAIKU_V2: "sr2"}
HOME = os.path.expanduser("~")


def site_of(path: str, sites) -> str:
    for name, prefix in sites:
        if path.startswith(prefix.rstrip("/") + "/"):
            return name
    return "unknown"


def rewrite(path: str, rules) -> str:
    for a, b in rules:
        if path.startswith(a):
            return b + path[len(a):]
    return path


def paper_runs(db: str, run_ids=None):
    """The scope's runs and their judgments: the paper rows, or with ``run_ids`` exactly those runs (refused unless every one is a run
    on one of the eight evals)."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    marks = ",".join("?" * len(JAILBREAK_EVALS))
    cols = ("SELECT r.run_id, r.campaign, r.eval_id, r.instrument_id, r.judge_id, r.model_id, r.object_id, r.lane_source_legacy, r.pass_label, "
            "r.trial_index, r.token_cap_tokens, vc.selection_state FROM v_run_current vc JOIN run r ON r.run_id = vc.run_id ")
    if run_ids is None:
        runs = [dict(r) for r in con.execute(cols + f"WHERE vc.in_current_paper = 1 AND r.eval_id IN ({marks}) ORDER BY r.run_id", JAILBREAK_EVALS)]
    else:
        want = sorted(set(run_ids))
        runs = []
        for i in range(0, len(want), 500):
            part = want[i:i + 500]
            runs += [dict(r) for r in con.execute(cols + f"WHERE r.run_id IN ({','.join('?' * len(part))}) AND r.eval_id IN ({marks})",
                                                  part + list(JAILBREAK_EVALS))]
        runs.sort(key=lambda r: r["run_id"])
        missing = sorted(set(want) - {r["run_id"] for r in runs})
        if missing:
            raise SystemExit(f"--runs: {len(missing)} run ids are not runs on the eight jailbreak evals in {db} (e.g. {missing[:3]})")
    ids = [r["run_id"] for r in runs]
    judg = collections.defaultdict(list)
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        for j in con.execute(f"SELECT judgment_id, run_id, judge_id, status, finished_utc, n_items_judged FROM judgment "
                             f"WHERE run_id IN ({','.join('?' * len(part))})", part):
            judg[j["run_id"]].append(dict(j))
    return runs, judg


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(HOME, "registry", "registry.sqlite"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--static-map", required=True)
    ap.add_argument("--mtbon-map", required=True)
    ap.add_argument("--mtbon-manifests", nargs="+", required=True)
    ap.add_argument("--goals", required=True, help="goals.json (goals: dataset, revision, text; goal_sets): goals.py writes it")
    ap.add_argument("--static-goals", help="optional: a JSON list of {goal_id, dataset, revision, text} cross-checked against --goals")
    ap.add_argument("--site", action="append", default=[], help="NAME=PREFIX: stores under PREFIX are read on site NAME")
    ap.add_argument("--path-rewrite", action="append", default=[], help="FROM=TO prefix rewrite of a scanned path")
    ap.add_argument("--runs", help="file of run ids: exactly these runs instead of the paper rows (a supplement)")
    args = ap.parse_args(argv)
    sites = [("local", HOME)] + [tuple(s.split("=", 1)) for s in args.site]
    rules = [tuple(s.split("=", 1)) for s in args.path_rewrite]
    os.makedirs(args.out, exist_ok=True)

    P = json.load(open(args.goals))
    goals = {g: {"dataset": v["dataset"], "revision": v["revision"], "text": v["text"]} for g, v in P["goals"].items()}
    for g in (json.load(open(args.static_goals)) if args.static_goals else []):
        v = goals.get(g["goal_id"])
        if v is None or (v["dataset"], v["revision"], v["text"]) != (g["dataset"], g["revision"], g["text"]):
            raise SystemExit(f"goal {g['goal_id']}: the two goal tables disagree")
    goal_sets = {"static": sorted(goals), "mt": sorted(P["goal_sets"]["mt"]), "bon": sorted(P["goal_sets"]["bon"])}
    json.dump({"goals": goals, "goal_sets": goal_sets}, open(os.path.join(args.out, "goals.json"), "w"), indent=0, sort_keys=True)

    static_map = {d["judgment_id"]: d for d in json.load(open(args.static_map))}
    mtbon_map = json.load(open(args.mtbon_map))
    units = {}
    for mf in args.mtbon_manifests:
        src = os.path.basename(os.path.dirname(mf))
        for line in open(mf):
            r = json.loads(line)
            path = rewrite(r["unit_path"], rules)
            files = [rewrite(f[0], rules) for f in r["files"]]
            units[r["uid"] + ":" + src] = {"type": r["kind"], "path": path, "files": sorted(files),
                                            "drop_goals": sorted(set(r.get("errored_goals") or []) | set(r.get("refused_goals") or []))}

    run_ids = [l.strip() for l in open(args.runs) if l.strip()] if args.runs else None
    runs, judg = paper_runs(args.registry, run_ids)
    jobs, problems = [], []
    for r in runs:
        haiku = [j for j in judg[r["run_id"]] if j["judge_id"] in KIND_OF_JUDGE and j["status"] == "complete"]
        if len(haiku) != 1:
            problems.append({"run_id": r["run_id"], "problem": f"{len(haiku)} complete Haiku judgments"})
            if not haiku:
                continue
            haiku.sort(key=lambda j: j["finished_utc"] or "")
        hj = haiku[-1]
        cat = CATEGORY[r["eval_id"]]
        job = {"run_id": r["run_id"], "campaign": r["campaign"], "eval_id": r["eval_id"], "category": cat,
               "kind": KIND_OF_JUDGE[hj["judge_id"]], "haiku_judge_id": hj["judge_id"], "haiku_judgment_id": hj["judgment_id"],
               "haiku_n_items_judged": hj["n_items_judged"], "lane_source_legacy": r["lane_source_legacy"], "pass_label": r["pass_label"],
               "model_id": r["model_id"], "object_id": r["object_id"], "units": [], "fold": None}
        if cat == "static":
            m = static_map.get(hj["judgment_id"])
            if not m or not m["units"] or m["unresolved"]:
                problems.append({"run_id": r["run_id"], "problem": "static mapping missing or unresolved", "map": m})
                continue
            job["units"] = [{"type": "static", "path": u} for u in m["units"]]
        else:
            m = mtbon_map.get(hj["judgment_id"])
            if not m or not (m["units"] or m.get("fold")):
                problems.append({"run_id": r["run_id"], "problem": "mt/bon mapping missing", "map": m})
                continue
            job["mapping_method"] = m.get("method")
            if m.get("fold"):
                job["fold"] = {"store": [units[u] for u in m["fold"]["store"]], "climb": [units[u] for u in m["fold"]["climb"]]}
            else:
                job["units"] = [units[u] for u in m["units"]]
        paths = [u["path"] for u in job["units"]] + ([u["path"] for u in job["fold"]["store"] + job["fold"]["climb"]] if job["fold"] else [])
        job["sites"] = sorted({site_of(p, sites) for p in paths})
        jobs.append(job)
    with open(os.path.join(args.out, "jobs.jsonl"), "w") as fh:
        for j in jobs:
            fh.write(json.dumps(j, sort_keys=True) + "\n")
    summary = {"paper_runs": len(runs), "jobs": len(jobs), "problems": problems,
               **({"selection": f"--runs {os.path.abspath(args.runs)} ({len(run_ids)} ids)"} if args.runs else {}),
               "by_category": dict(collections.Counter(j["category"] for j in jobs)),
               "by_kind": dict(collections.Counter(j["kind"] for j in jobs)),
               "by_sites": dict(collections.Counter("+".join(j["sites"]) for j in jobs))}
    json.dump(summary, open(os.path.join(args.out, "scope_summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "problems"}, indent=1), f"\nproblems: {len(problems)}")
    for p in problems[:20]:
        print("  ", p["run_id"], p["problem"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
