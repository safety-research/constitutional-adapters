#!/usr/bin/env python3
"""Read the transcript stores of the paper-row scope and write the StrongREJECT JUDGE UNITS and their TEXTS (stdlib only, so it runs on
a cluster node with a bare ``python3``; no text leaves the machine it runs on except through the files it writes).

    python3 enumerate_units.py --jobs jobs.jsonl --goals goals.json --site SITE --out DIR [--workers 6]

``--site`` names the machine it runs on, one of the site names :mod:`scope` assigned the jobs (``local`` for the grading host,
``cluster_a`` / ``cluster_b`` for the clusters' shared storage); only the jobs whose stores all live there are read.

A UNIT is one judged response: ``(run_id, item_id = goal_id, trial_index, sample_index, turn_index)`` -> its judge-input key, the live
grader's cache key ``sha256(kind \\0 dataset \\0 revision \\0 goal text \\0 response \\0)`` (``grading_live._live_cache_key``; kind
``sr`` = rubric v1, ``sr2`` = v2). One request is sent per distinct key; every unit it covers takes its verdict.

The graders' rules, as the stores' own records reproduce them (the per-turn counts equal the producers' records):
  static  every row of ``<cell>/shard*/responses.jsonl`` (shards in sorted order) whose goal is in the goal set; item key
          (goal, ``trial``, ``sample_idx``, -1) -- the variant index, as the Sonnet regrade twin's ``sample_index``; pooled runs list
          their t0 and t1 member stores, whose rows carry ``trial`` 0 / 1
  mt      the cell's per-turn replies, goals in the multi-turn set, minus goals whose latest progress outcome is ``error:`` or
          ``refused:`` (jailbreak_artifacts.exclude_errored_goals); a goal re-run in two shards keeps only the LAST shard's replies
          (jailbreak_run._load_gated_responses merges shards in sorted order with dict.update); a goal two located units share is taken
          from the first; item key (goal, 0, n, -1) where n is the reply's position among the goal's replies in the grader's
          order -- the Sonnet regrade twin's convention for multi-turn rows (``sample_index`` = the request ordinal within the goal), so
          ``v_judge_side_by_side`` pairs the judges -- with the reply's own ``order`` and its attack stream (ActorAttack's actors and
          Crescendo's backtracks restart ``order`` inside one shard) kept in the unit's note
  bon     every draw of ``bon_responses*.jsonl`` for the 100 JBB goals; item key (goal, ``trial``, ``sample_idx``, -1); a 512-store +
          4,096-climb reading judges the FOLD: the store's draws with each climbed (goal, sample_idx) slot replaced by its redraw
A key collision inside one run (the same item key twice) keeps the first row and is counted in the report.

Outputs under ``--out``:
  units.tsv.gz      run_id, item_id, trial, sample, turn, custom_id, kind, response_chars, note (multi-turn: order=<o>;stream=<s>)
  texts.jsonl.gz    one line per DISTINCT key on this site: {k, kind, dataset, revision, goal_id, response}
  report.json       per job: rows read, units, distinct keys, dropped goals, collisions, missing files
"""
from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from typing import Dict, List, Tuple

GOALS: Dict[str, Dict[str, str]] = {}
GOAL_SETS: Dict[str, set] = {}


def cache_key(kind: str, dataset: str, revision: str, goal: str, response: str) -> str:
    """The live grader's cache key (``sisteer.eval.attacks.grading_live._live_cache_key``), re-implemented stdlib-only; a unit test
    pins the two equal."""
    h = hashlib.sha256()
    for part in (kind, dataset, revision, goal, response):
        h.update(part.encode())
        h.update(b"\x00")
    return h.hexdigest()


def _read_jsonl(path: str):
    with open(path, "rb") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError:
                yield None


def _text(resp) -> str:
    if isinstance(resp, str):
        return resp
    return "" if resp is None else json.dumps(resp)


def static_rows(unit: dict, rep: dict):
    shards = sorted(glob.glob(os.path.join(unit["path"], "shard*", "responses.jsonl")))
    if not shards:
        rep["missing"].append(unit["path"])
    gs = GOAL_SETS["static"]
    for shard in shards:
        ordinal: Dict[Tuple[str, int], int] = {}
        for r in _read_jsonl(shard):
            if not isinstance(r, dict) or "goal_id" not in r:
                rep["bad_lines"] += 1
                continue
            g = str(r["goal_id"])
            if g not in gs:
                rep["unknown_goal"] += 1
                continue
            if not isinstance(r.get("response"), str):
                rep["nonstr_response"] += 1
            trial = int(r.get("trial") or 0)
            k = (g, trial)
            sample = r.get("sample_idx")
            sample = int(sample) if sample is not None else ordinal.get(k, 0)
            ordinal[k] = ordinal.get(k, 0) + 1
            yield g, trial, sample, -1, _text(r.get("response")), ""


def mt_rows(job_units: List[dict], rep: dict):
    gs = GOAL_SETS["mt"]
    seen_goals: set = set()
    for unit in job_units:
        drop = set(unit.get("drop_goals") or [])
        rows = []
        for fi, fp in enumerate(sorted(unit["files"])):
            if not os.path.exists(fp):
                rep["missing"].append(fp)
                continue
            for r in _read_jsonl(fp):
                if not isinstance(r, dict) or "goal_id" not in r:
                    rep["bad_lines"] += 1
                    continue
                if r.get("response") is None:
                    rep["no_response"] += 1
                    continue
                g = str(r["goal_id"])
                if g not in gs:
                    rep["unknown_goal"] += 1
                    continue
                if g in drop:
                    rep["dropped_goal_rows"] += 1
                    continue
                order = int(r["order"]) if r.get("order") is not None else -1
                rows.append((g, order, fi, _text(r.get("response"))))
        pairs: Dict[Tuple[str, int], int] = {}
        for g, o, _fi, _t in rows:
            pairs[(g, o)] = pairs.get((g, o), 0) + 1
        if any(v > 1 for v in pairs.values()):
            last: Dict[str, int] = {}
            for g, _o, fi, _t in rows:
                last[g] = max(last.get(g, -1), fi)
            before = len(rows)
            rows = [x for x in rows if x[2] == last[x[0]]]
            rep["last_shard_dropped"] += before - len(rows)
        if seen_goals:
            before = len(rows)
            rows = [x for x in rows if x[0] not in seen_goals]
            rep["shared_goal_dropped"] += before - len(rows)
        seen_goals |= {x[0] for x in rows}
        occurrence: Dict[Tuple[str, int], int] = {}
        position: Dict[str, int] = {}
        for g, o, _fi, t in rows:
            # a goal can hold several attack streams whose replies restart at order 0 (ActorAttack's actors, Crescendo's
            # backtracks) in ONE shard; the grader judges every reply
            k = occurrence.get((g, o), 0)
            occurrence[(g, o)] = k + 1
            n = position.get(g, 0)
            position[g] = n + 1
            yield g, 0, n, -1, t, f"order={o};stream={k}"


def bon_unit_rows(unit: dict, rep: dict):
    gs = GOAL_SETS["bon"]
    for fp in sorted(unit["files"]):
        if not os.path.exists(fp):
            rep["missing"].append(fp)
            continue
        for r in _read_jsonl(fp):
            if not isinstance(r, dict) or "goal_id" not in r:
                rep["bad_lines"] += 1
                continue
            if r.get("response") is None:
                rep["no_response"] += 1
                continue
            g = str(r["goal_id"])
            if g not in gs:
                rep["unknown_goal"] += 1
                continue
            s = int(r["sample_idx"]) if r.get("sample_idx") is not None else -1
            yield g, int(r.get("trial") or 0), s, -1, _text(r.get("response")), ""


def bon_rows(job: dict, rep: dict):
    if job.get("fold"):
        store = [x for u in job["fold"]["store"] for x in bon_unit_rows(u, rep)]
        climb = [x for u in job["fold"]["climb"] for x in bon_unit_rows(u, rep)]
        slots = {(x[0], x[2]) for x in climb}
        rep["climb_replaced"] = sum(1 for x in store if (x[0], x[2]) in slots)
        for x in store:
            if (x[0], x[2]) not in slots:
                yield x
        for x in climb:
            yield x
    else:
        for u in job["units"]:
            yield from bon_unit_rows(u, rep)


def process(job: dict):
    t0 = time.time()
    rep = {"run_id": job["run_id"], "category": job["category"], "rows": 0, "units": 0, "collisions": 0, "missing": [], "bad_lines": 0,
           "unknown_goal": 0, "nonstr_response": 0, "no_response": 0, "dropped_goal_rows": 0, "last_shard_dropped": 0,
           "shared_goal_dropped": 0}
    kind = job["kind"]
    if job["category"] == "static":
        it = (x for u in job["units"] for x in static_rows(u, rep))
    elif job["category"] == "mt":
        it = mt_rows(job["units"], rep)
    else:
        it = bon_rows(job, rep)
    units: List[Tuple] = []
    texts: Dict[str, Tuple[str, str, str, str, str]] = {}
    seen = set()
    for g, trial, sample, turn, resp, note in it:
        rep["rows"] += 1
        ik = (g, trial, sample, turn)
        if ik in seen:
            rep["collisions"] += 1
            continue
        seen.add(ik)
        meta = GOALS[g]
        k = cache_key(kind, meta["dataset"], meta["revision"], meta["text"], resp)
        units.append((job["run_id"], g, trial, sample, turn, k, kind, len(resp), note))
        if k not in texts:
            texts[k] = (kind, meta["dataset"], meta["revision"], g, resp)
    rep["units"] = len(units)
    rep["distinct_keys"] = len(texts)
    rep["goals"] = len({u[1] for u in units})
    rep["secs"] = round(time.time() - t0, 2)
    return rep, units, texts


def _init(goals_path: str):
    global GOALS, GOAL_SETS
    d = json.load(open(goals_path))
    GOALS = d["goals"]
    GOAL_SETS = {k: set(v) for k, v in d["goal_sets"].items()}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--goals", required=True)
    ap.add_argument("--site", required=True, help="the site name scope.py gave this machine's jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--only-runs", help="file of run ids to restrict to")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)
    jobs = [json.loads(l) for l in open(args.jobs)]
    jobs = [j for j in jobs if j["sites"] == [args.site]]
    if args.only_runs:
        keep = {l.strip() for l in open(args.only_runs) if l.strip()}
        jobs = [j for j in jobs if j["run_id"] in keep]
    # biggest first so the pool's tail is short (multi-turn and best-of-N stores are the large ones)
    jobs.sort(key=lambda j: {"mt": 0, "bon": 1, "static": 2}[j["category"]])
    print(f"[enumerate] site={args.site} jobs={len(jobs)} workers={args.workers}", flush=True)
    seen_keys = set()
    reps = []
    t0 = time.time()
    n_units = 0
    with gzip.open(os.path.join(args.out, "units.tsv.gz.tmp"), "wt") as uf, \
            gzip.open(os.path.join(args.out, "texts.jsonl.gz.tmp"), "wt") as tf, \
            Pool(args.workers, initializer=_init, initargs=(args.goals,)) as pool:
        uf.write("run_id\titem_id\ttrial\tsample\tturn\tcustom_id\tkind\tresponse_chars\tnote\n")
        for i, (rep, units, texts) in enumerate(pool.imap_unordered(process, jobs, chunksize=1)):
            for u in units:
                uf.write("\t".join(str(x) for x in u) + "\n")
            n_units += len(units)
            for k, (kind, ds, rev, g, resp) in texts.items():
                if k in seen_keys:
                    continue
                seen_keys.add(k)
                tf.write(json.dumps({"k": k, "kind": kind, "dataset": ds, "revision": rev, "goal_id": g, "response": resp}) + "\n")
            reps.append(rep)
            if (i + 1) % 50 == 0 or i + 1 == len(jobs):
                print(f"[enumerate] {i + 1}/{len(jobs)} jobs, {n_units} units, {len(seen_keys)} keys, {time.time() - t0:.0f}s", flush=True)
    os.replace(os.path.join(args.out, "units.tsv.gz.tmp"), os.path.join(args.out, "units.tsv.gz"))
    os.replace(os.path.join(args.out, "texts.jsonl.gz.tmp"), os.path.join(args.out, "texts.jsonl.gz"))
    summary = {"site": args.site, "jobs": len(jobs), "units": n_units, "distinct_keys": len(seen_keys),
               "jobs_with_missing": sum(1 for r in reps if r["missing"]), "collisions": sum(r["collisions"] for r in reps),
               "secs": round(time.time() - t0, 1)}
    json.dump({"summary": summary, "jobs": reps}, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
