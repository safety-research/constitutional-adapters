#!/usr/bin/env python3
"""Map every CURRENT item the Opus 5 Petri judge filed to the stored inspect transcript it judged: the input of ``scope.py --matched``.

    PYTHONPATH=src:scripts python scripts/rejudge/petri/match_items.py --logs LOG_ROOT [--logs LOG_ROOT ...] --out items_matched.json \\
        [--registry ~/registry/registry.sqlite] [--runs RUNS.txt] [--index INDEX.json] [--workers 4]

The registry keeps the Petri judge's scores per (behaviour/scenario, trial), not the transcript, so each item is matched to the stored
inspect sample whose stored judge scores are the registry's (a score FINGERPRINT), in three steps:

  index     every ``.eval`` log under the ``--logs`` roots (``inspect_ai.log``, header and sample summaries only): its Petri metadata
            (``petri_run_key``, ``behavior_dir_key``, ...) and per sample its id, epoch, completion and ``audit_judge`` scores. Written to
            ``--index`` (default ``<out>.index.json``) and read from there when it exists (``--reindex`` rebuilds it)
  items     every CURRENT item of the Opus 5 Petri judge (``17e480e5``, judgments ``complete`` / ``partial``) from the registry, read
            only: per (run, item, trial, sample, turn) its ``petri_score`` / ``eval_awareness`` / ``scenario_realism`` values (``--runs``
            keeps the listed runs)
  match     per run, the AUDIT ROOT whose samples match most of the run's checkable scores (a sample is a candidate of an item when its
            log's ``petri_run_key`` is the run's, its behaviour is the item's and its (id, epoch) the item's (scenario, trial + 1)); ties
            go to the root the run's ``store_uri`` names. Then per item the root's candidate with the most matching scores (ties: a log
            outside a ``before-rejudge`` directory, then the newest log). ``petri_score`` is compared with the behaviour's OWN dimension
            (the sample's one score key that is not a shared dimension: cyber is scored as cyber_bio_chem_operational_uplift)

The audit root of a log is the part of its path between the Petri harness's state directory and ``inspect_logs/``
(``.../.local/state/petri[_native]/<root>/.../inspect_logs/...``, plus ``#seed<N>`` for a ``target-seed-<N>`` directory; ``--root-regex``
names another layout). The physical transcript key is ``<root>|<petri_run_key>|<behaviour>|<scenario>|<epoch>``.

Output: ``{"items": [{run_id, item_id, trial_index, sample_index, campaign, eval_id, paper_status, phys, path, scenario, epoch, behavior,
dir_key, score_match, score_tot, judge_usage, n_candidates_same_root}], "unmatched": [[run_id, item_id]], "run_root": {run_id: {key,
root, matches, total, n_roots, ambiguous, alt}}}``. ``scope.py`` refuses an item this mapping does not cover or whose value moved.
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.petri.scope import PETRI_JUDGE  # noqa: E402

#: the audit root of a log path (module docstring)
ROOT_REGEX = r".*?/\.local/state/(petri(?:_native)?/.+?)/inspect_logs/"
#: the two dimensions every behaviour's answer tool shares; the third score key is the behaviour's own (``petri_score``)
STANDARD = ("eval_awareness", "scenario_realism")
#: the log metadata the index keeps
MD_KEYS = ("audit", "cell_id", "behavior_dir_key", "petri_run_key", "virtual_model_name", "scenario_bundle_hash", "prompt_files_hash",
           "judge_model", "target_seed_base")


# --------------------------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------------------------
def index_file(path: str) -> Dict[str, Any]:
    """One ``.eval`` log's index entry (header and sample summaries; never the transcripts), or ``{path, err}``."""
    try:
        from inspect_ai.log import read_eval_log, read_eval_log_sample_summaries
        h = read_eval_log(path, header_only=True)
        md = h.eval.metadata or {}
        try:
            roles = {k: v.model for k, v in (h.eval.model_roles or {}).items()}
        except Exception:  # noqa: BLE001 -- an older log without model roles
            roles = {}
        samples = []
        for s in read_eval_log_sample_summaries(path):
            ju = (s.role_usage or {}).get("judge")
            sc = None
            if s.scores and "audit_judge" in s.scores:
                v = s.scores["audit_judge"].value
                sc = v if isinstance(v, dict) else {"_": v}
            samples.append({"id": str(s.id), "epoch": s.epoch, "completed": s.completed, "error": bool(s.error),
                            "judge_usage": ju.model_dump() if ju else None, "scores": sc, "n_msgs": s.message_count,
                            "role_models": list((s.model_usage or {}).keys())})
        return {"path": path, "size": os.path.getsize(path), "mtime": os.path.getmtime(path), "status": h.status, "task": h.eval.task,
                "md": {k: md.get(k) for k in MD_KEYS}, "roles": roles, "samples": samples}
    except Exception as e:  # noqa: BLE001 -- an unreadable log is listed, never fatal
        return {"path": path, "err": repr(e)[:300]}


def log_files(roots: Iterable[str]) -> List[str]:
    files: List[str] = []
    for r in roots:
        files += glob.glob(os.path.join(os.path.expanduser(r), "**", "*.eval"), recursive=True)
    return sorted(set(files))


def build_index(roots: Sequence[str], workers: int = 4) -> List[Dict[str, Any]]:
    files = log_files(roots)
    if workers <= 1:
        return [index_file(f) for f in files]
    import multiprocessing as mp
    with mp.Pool(workers) as p:
        return p.map(index_file, files, chunksize=8)


# --------------------------------------------------------------------------------------------
# items
# --------------------------------------------------------------------------------------------
def current_items(con: sqlite3.Connection, run_ids: Optional[Iterable[str]] = None) -> Dict[str, Any]:
    """``{runs: {run_id: {...}}, items: [{run_id, item_id, trial_index, sample_index, turn_index, metrics}]}``: every CURRENT item of
    the Opus 5 Petri judge's complete / partial judgments (read only), optionally only the listed runs."""
    con.row_factory = sqlite3.Row
    jrows = con.execute("SELECT j.judgment_id, j.status, j.n_items_judged, r.run_id, r.campaign, r.eval_id, e.paper_status, r.model_id, "
                        "r.object_id, r.store_uri, r.extra_json, r.instrument_id FROM judgment j JOIN run r USING (run_id) "
                        "JOIN eval e ON e.eval_id = r.eval_id WHERE j.judge_id = ? AND j.status IN ('complete', 'partial')",
                        (PETRI_JUDGE,)).fetchall()
    want = set(run_ids) if run_ids is not None else None
    runs: Dict[str, Dict[str, Any]] = {}
    for r in jrows:
        if want is not None and r["run_id"] not in want:
            continue
        ex = json.loads(r["extra_json"] or "{}")
        runs[r["run_id"]] = {"judgment_id": r["judgment_id"], "status": r["status"], "n_items_judged": r["n_items_judged"],
                             "campaign": r["campaign"], "eval_id": r["eval_id"], "paper_status": r["paper_status"],
                             "model_id": r["model_id"], "object_id": r["object_id"], "store_uri": r["store_uri"],
                             "instrument_id": r["instrument_id"], "petri_run_key": ex.get("petri_run_key")}
    jids = {v["judgment_id"] for v in runs.values()}
    out: Dict[Tuple, Dict[str, Any]] = collections.defaultdict(dict)
    ids = sorted(runs)
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        for it in con.execute(f"SELECT run_id, item_id, metric_id, trial_index, sample_index, turn_index, judgment_id, value "
                              f"FROM v_item_result_current WHERE run_id IN ({','.join('?' * len(part))})", part):
            if it["judgment_id"] not in jids:
                continue
            key = (it["run_id"], it["item_id"], it["trial_index"], it["sample_index"], it["turn_index"])
            out[key][it["metric_id"]] = it["value"]
    return {"runs": runs, "items": [{"run_id": k[0], "item_id": k[1], "trial_index": k[2], "sample_index": k[3], "turn_index": k[4],
                                     "metrics": v} for k, v in out.items()]}


# --------------------------------------------------------------------------------------------
# match
# --------------------------------------------------------------------------------------------
def root_of(path: str, regex: str = ROOT_REGEX) -> str:
    """The audit root of a log path (module docstring); the path itself when the layout does not match."""
    m = re.match(regex, path)
    root = m.group(1) if m else path
    seed = re.search(r"target-seed-(\d+)", path)
    return root + (f"#seed{seed.group(1)}" if seed else "")


def run_key(r: Mapping[str, Any]) -> Optional[str]:
    """A run's ``petri_run_key`` (its extra), else the cell directory its ``store_uri`` names."""
    k = r.get("petri_run_key")
    su = r.get("store_uri") or ""
    if not k and su and "/cells/" in su:
        k = su.rstrip("/").rsplit("/", 1)[1]
    return k


def match_score(c: Mapping[str, Any], metrics: Mapping[str, Any], behavior: str) -> Tuple[int, int]:
    """``(matching scores, checkable scores)`` of a candidate sample against an item's registry values: ``petri_score`` is compared with
    the behaviour's OWN dimension (the one score key that is not a shared dimension), the shared dimensions by name."""
    sc = c["scores"]
    n = tot = 0
    own = [k for k in sc if k not in STANDARD]
    for m, v in metrics.items():
        key = (own[0] if len(own) == 1 else behavior) if m == "petri_score" else m
        if key in sc and v is not None and sc[key] is not None:
            tot += 1
            n += float(sc[key]) == float(v)
    return n, tot


def candidates(index: Iterable[Mapping[str, Any]], regex: str = ROOT_REGEX) -> Dict[Tuple, List[Dict[str, Any]]]:
    """``(petri_run_key, behaviour, sample id, epoch) -> [candidate]`` over every completed, scored sample of the index."""
    cand: Dict[Tuple, List[Dict[str, Any]]] = collections.defaultdict(list)
    for f in index:
        if "err" in f:
            continue
        md = f.get("md") or {}
        k, dk = md.get("petri_run_key"), md.get("behavior_dir_key") or ""
        b = dk.split("/")[0]
        mtime = f.get("mtime")
        if mtime is None:
            mtime = os.path.getmtime(f["path"]) if os.path.exists(f["path"]) else 0.0
        for s in f.get("samples") or []:
            if not s.get("scores") or not s.get("completed"):
                continue
            cand[(k, b, s["id"], s["epoch"])].append({"path": f["path"], "root": root_of(f["path"], regex), "dir_key": dk,
                                                      "scores": s["scores"], "judge_usage": s.get("judge_usage"),
                                                      "before_rejudge": "before-rejudge" in f["path"], "mtime": mtime})
    return cand


def match(index: Iterable[Mapping[str, Any]], items_doc: Mapping[str, Any], regex: str = ROOT_REGEX) -> Dict[str, Any]:
    """The mapping (module docstring) of ``items_doc`` (:func:`current_items`) onto the indexed samples."""
    cand = candidates(index, regex)
    runs = items_doc["runs"]
    by_run: Dict[str, List[Mapping[str, Any]]] = collections.defaultdict(list)
    for it in items_doc["items"]:
        by_run[it["run_id"]].append(it)
    out: List[Dict[str, Any]] = []
    unmatched: List[Tuple[str, str]] = []
    run_root: Dict[str, Dict[str, Any]] = {}
    for rid, r in runs.items():
        k = run_key(r)
        its = by_run.get(rid, [])
        rootscore: collections.Counter = collections.Counter()
        roottot: collections.Counter = collections.Counter()
        for it in its:
            b, s = it["item_id"].split("/", 1)
            best: Dict[str, Tuple[int, int]] = {}
            for c in cand.get((k, b, s, it["trial_index"] + 1), []):
                best[c["root"]] = max(best.get(c["root"], (0, 0)), match_score(c, it["metrics"], b))
            for root, (n, t) in best.items():
                rootscore[root] += n
                roottot[root] += t
        su = r.get("store_uri") or ""

        def pref(root: str) -> Tuple[int, int]:
            return rootscore[root], 1 if root.split("#")[0].split("/", 1)[-1] in su else 0

        roots = sorted(rootscore, key=pref, reverse=True)
        chosen = roots[0] if roots else None
        ambiguous = len(roots) > 1 and pref(roots[0]) == pref(roots[1])
        run_root[rid] = {"key": k, "root": chosen, "matches": rootscore.get(chosen, 0), "total": roottot.get(chosen, 0),
                         "n_roots": len(roots), "ambiguous": ambiguous, "alt": [(x, rootscore[x]) for x in roots[1:4]]}
        for it in its:
            b, s = it["item_id"].split("/", 1)
            cs = [c for c in cand.get((k, b, s, it["trial_index"] + 1), []) if c["root"] == chosen]
            if not cs:
                unmatched.append((rid, it["item_id"]))
                continue
            cs.sort(key=lambda c: (match_score(c, it["metrics"], b)[0], not c["before_rejudge"], c["mtime"]), reverse=True)
            c = cs[0]
            n, t = match_score(c, it["metrics"], b)
            out.append({"run_id": rid, "item_id": it["item_id"], "trial_index": it["trial_index"], "sample_index": it["sample_index"],
                        "campaign": r.get("campaign"), "eval_id": r.get("eval_id"), "paper_status": r.get("paper_status"),
                        "phys": f"{chosen}|{k}|{b}|{s}|{it['trial_index'] + 1}", "path": c["path"], "scenario": s,
                        "epoch": it["trial_index"] + 1, "behavior": b, "dir_key": c["dir_key"], "score_match": n, "score_tot": t,
                        "judge_usage": c["judge_usage"], "n_candidates_same_root": len(cs)})
    return {"items": out, "unmatched": unmatched, "run_root": run_root}


def summary(m: Mapping[str, Any]) -> Dict[str, Any]:
    items = m["items"]
    return {"matched": len(items), "unmatched": len(m["unmatched"]),
            "runs_ambiguous_root": sum(1 for v in m["run_root"].values() if v["ambiguous"]),
            "runs_without_root": sum(1 for v in m["run_root"].values() if not v["root"]),
            "items_without_checkable_scores": sum(1 for o in items if o["score_tot"] == 0),
            "items_with_score_mismatch": sum(1 for o in items if o["score_match"] < o["score_tot"]),
            "distinct_transcripts": len({o["phys"] for o in items})}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", action="append", default=[], help="a root holding Petri inspect .eval logs (repeatable)")
    ap.add_argument("--out", required=True, help="items_matched.json")
    ap.add_argument("--registry", default=os.path.join(os.path.expanduser("~"), "registry", "registry.sqlite"), help="read only")
    ap.add_argument("--runs", help="file of run ids (one per line, # lines skipped): only these runs' items")
    ap.add_argument("--index", help="the log index (default <out>.index.json): read when it exists, else built and written")
    ap.add_argument("--reindex", action="store_true", help="rebuild the index even when --index exists")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--root-regex", default=ROOT_REGEX, help="the audit root of a log path (group 1)")
    args = ap.parse_args(argv)
    index_path = args.index or args.out + ".index.json"
    if os.path.exists(index_path) and not args.reindex:
        index = json.load(open(index_path))
    else:
        if not args.logs:
            raise SystemExit("no index to read: name the log roots (--logs)")
        index = build_index(args.logs, args.workers)
        with open(index_path, "w") as fh:
            json.dump(index, fh)
    run_ids = None
    if args.runs:
        run_ids = sorted({l.strip() for l in open(args.runs) if l.strip() and not l.startswith("#")})
    con = sqlite3.connect(f"file:{os.path.expanduser(args.registry)}?mode=ro", uri=True, timeout=120)
    try:
        items_doc = current_items(con, run_ids)
    finally:
        con.close()
    m = match(index, items_doc, args.root_regex)
    with open(args.out, "w") as fh:
        json.dump(m, fh)
    s = dict(summary(m), index=index_path, logs_indexed=len(index), unreadable_logs=sum(1 for f in index if "err" in f))
    print(json.dumps(s, indent=1))
    return 0 if not m["unmatched"] else 2


if __name__ == "__main__":
    sys.exit(main())
