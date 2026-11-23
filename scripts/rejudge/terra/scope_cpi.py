#!/usr/bin/env python3
"""The paper-row scope of the OR-Bench hard / XSTest re-judge: every paper run's Sonnet 5 C/P/I judgment -> the judged cells it reads
-> one UNIT per judged item, each with the exact text the judge saw, its request key recomputed and checked against the store's.

    HF_HUB_OFFLINE=1 PYTHONPATH=src:scripts python scripts/rejudge/terra/scope_cpi.py --out SCOPE/cpi \
        [--registry ~/registry/registry.sqlite] [--xstest-subset safe|all] [--runs RUNS.txt]

SCOPE: runs with ``v_run_current.in_current_paper = 1`` on ``or_bench_hard`` / ``xstest`` and their
complete judgment by the Sonnet 5 C/P/I judge (``or_bench_cpi_v1_sonnet5`` / ``xstest_cpi_v1_sonnet5``). The registry is opened
read-only. ``--runs`` replaces the paper-row selection by exactly the listed run ids (a SUPPLEMENT: the contrast anchors of paper runs,
which are not paper rows themselves; or named paper runs rebuilt to diff against an earlier scope); every later step is unchanged, so
a run's units and texts are what the paper-row scope gives for it. ``report.json`` then carries ``selection``, and an id that is not an
or_bench_hard / xstest run is a problem.

WHICH CELLS. A run's judged items live in a cell store's ``judge_items.jsonl`` (one row per item: ``item_id``, ``request_key``,
``label``, ``status``, ``rubric_sha``, ``protocol_sha``, ``judge_model``) beside the generations it judged
(``private/gens_<task>.jsonl``). The judgment names its store three ways, resolved in this order:
  1. a cell: ``store_uri`` is ``<cell>/judge_items.jsonl`` or the cell directory -> that cell, trial 0 (the run's own items);
  2. a final-rung index entry (the pooled sampled-lane runs, the r2 export rows, the paired ReLU lane): ``final_rung.json`` ``targets.<T>
     .controls['<control>|<cell>']``, located from the store_uri's fragment, the run's ``extra.finality_record``, or the r2 row key
     ``r2:<T>|<cell>|1`` -> each trial's ``pass2_store`` (the judged store whose pass-2 labels the pooled reading counts), trial = the
     index's trial; the index's ``judge.judge_items_sha256`` must equal the file's sha256 today (a re-judged store refuses).
WHICH ITEMS. Every ``judge_items.jsonl`` row of the run's task (``judged/or_bench_hard_1k`` / ``judged/xstest_v2``) under the frozen rubric
and protocol and ``judge_model`` claude-sonnet-5, ABSTAINED rows included (the new judge may answer what Sonnet withheld). XSTest:
``--xstest-subset safe`` (default) keeps the 250 safe prompts, the only half any paper number reads (``xst_refused`` /
``over_refusal_label_mean`` over the safe subset); ``all`` adds the 200 contrast
prompts the Sonnet judge also labelled.
THE TEXT. The pinned rows (``tiers.judged.or_bench_rows`` / ``xstest_rows`` over ``corpora.load_pins()``, offline from the HF cache) give
the request; the cell's spool gives the response (first write wins; a duplicate item with different text refuses, as ``GenSpool`` does,
but the spool is read without ``GenSpool``'s torn-line repair, which writes). ``JudgeRequest(...).key()`` is recomputed for every item
and must equal the row's ``request_key``; a mismatch is listed in ``report.json`` and the item gets no unit.

OUTPUT (``--out``):
  units.tsv.gz   run_id, item_id, trial, sample, turn, custom_id, rubric, eval_id, orig_label, orig_status, note (one row per unit)
  texts.jsonl.gz one row per distinct custom_id: {k, rubric, item_id, prompt, response}
  runs.jsonl     one row per paper run: the judgment re-measured, the sources, the counts
  report.json    totals, per-lane counts, every problem by name
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import os
import re
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.terra import cpi as C  # noqa: E402

HOME = os.path.expanduser("~")
_FR_RE = re.compile(r"targets\.(?P<t>[A-Za-z0-9_.:-]+?)\.controls\[['\"](?P<k>[^'\"]+)['\"]\]")


def expand(p: Optional[str]) -> Optional[str]:
    if not p:
        return p
    return HOME + p[1:] if p.startswith("~/") else p


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sonnet_judges(con) -> Dict[str, str]:
    """eval_id -> the Sonnet 5 C/P/I judge id (one per rubric)."""
    out = {}
    for rubric, ev in C.EVAL_OF_RUBRIC.items():
        rows = con.execute("SELECT judge_id FROM judge WHERE rubric_name = ? AND judge_model = ?", (rubric, C.ORIGINAL_MODEL)).fetchall()
        if len(rows) != 1:
            raise SystemExit(f"expected one {C.ORIGINAL_MODEL} judge for {rubric}, found {len(rows)}")
        out[ev] = rows[0][0]
    return out


_RUN_COLS = ("r.run_id, r.campaign, r.eval_id, r.lane_source_legacy, r.store_uri, r.producer_run_key_legacy, r.producer_object_key_legacy, "
             "r.extra_json, r.trial_index, r.judge_id AS run_judge_id, r.model_id, r.object_id, vc.selection_state ")


def read_run_ids(path: str) -> List[str]:
    """A ``--runs`` file: one run id per line (blank lines and ``#`` comments skipped), each once."""
    out: List[str] = []
    with open(path) as fh:
        for line in fh:
            rid = line.split("#", 1)[0].strip()
            if rid and rid not in out:
                out.append(rid)
    return out


def paper_runs(db: str, run_ids: Optional[List[str]] = None):
    """The runs to scope: the paper rows (``v_run_current.in_current_paper = 1``) on the two evals, or exactly ``run_ids`` (a supplement,
    e.g. the contrast anchors of paper runs, which are not paper rows themselves); every later step is the same either way."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    judges = sonnet_judges(con)
    if run_ids is None:
        runs = [dict(r) for r in con.execute(
            "SELECT " + _RUN_COLS + "FROM v_run_current vc JOIN run r ON r.run_id = vc.run_id "
            "WHERE vc.in_current_paper = 1 AND r.eval_id IN ('or_bench_hard', 'xstest') ORDER BY r.run_id")]
    else:
        runs = []
        for i in range(0, len(run_ids), 500):
            part = run_ids[i:i + 500]
            runs += [dict(r) for r in con.execute(
                "SELECT " + _RUN_COLS + "FROM run r LEFT JOIN v_run_current vc ON vc.run_id = r.run_id "
                f"WHERE r.run_id IN ({','.join('?' * len(part))}) AND r.eval_id IN ('or_bench_hard', 'xstest')", part)]
        runs.sort(key=lambda r: r["run_id"])
    judg: Dict[str, List[dict]] = collections.defaultdict(list)
    ids = [r["run_id"] for r in runs]
    for i in range(0, len(ids), 500):
        part = ids[i:i + 500]
        for j in con.execute(f"SELECT judgment_id, run_id, judge_id, status, n_items_judged, n_judge_refusals, n_errors, store_uri, finished_utc "
                             f"FROM judgment WHERE run_id IN ({','.join('?' * len(part))})", part):
            judg[j["run_id"]].append(dict(j))
    con.close()
    return runs, judg, judges


# --------------------------------------------------------------------------------------------
# store readers (read-only; never a write into a producer's store)
# --------------------------------------------------------------------------------------------
def read_judge_items(path: str) -> List[dict]:
    out = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            out.append(json.loads(line))
    return out


def read_spool(path: str) -> Dict[str, str]:
    """``{item_id: text}`` of a generation spool: line 0 the header; first write wins; a duplicate with different text refuses; a torn
    FINAL line is skipped (never repaired: the store is a peer's)."""
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    lines = text.splitlines()
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if i == len(lines) - 1 and not text.endswith("\n"):
                break
            raise ValueError(f"{path}: corrupt row at line {i + 1}")
        if "__spool_header__" in row:
            continue
        iid, t = row["item_id"], row["text"]
        if iid in out:
            if out[iid] != t:
                raise ValueError(f"{path}: item {iid} appears twice with different text")
            continue
        out[iid] = t
    return out


def load_finals(paths) -> Dict[str, dict]:
    cache: Dict[str, dict] = {}
    for p in paths:
        if p not in cache:
            with open(p) as fh:
                cache[p] = json.load(fh)
    return cache


def locate_final_rung(run: dict, judgment: dict) -> Optional[Tuple[str, str, str, str]]:
    """``(final_rung.json path, target, control key, how)`` for a run read through a final-rung index, else None."""
    extra = json.loads(run.get("extra_json") or "{}")
    for how, uri in (("judgment.store_uri", judgment.get("store_uri")), ("run.store_uri", run.get("store_uri")),
                     ("run.producer_run_key_legacy", run.get("producer_run_key_legacy"))):
        if uri and "final_rung.json" in uri and "#" in uri:
            m = _FR_RE.search(uri.split("#", 1)[1])
            if m:
                return expand(uri.split("#", 1)[0]), m.group("t"), m.group("k"), how
    fr = extra.get("finality_record")
    if isinstance(fr, str) and "final_rung.json" in fr:
        m = _FR_RE.search(fr)
        path = fr.split(" ", 1)[0].strip()
        if m:
            return expand(path), m.group("t"), m.group("k"), "run.extra.finality_record"
    pk = run.get("producer_run_key_legacy") or ""
    if pk.startswith("r2:"):
        parts = pk[3:].split("|")
        target, cell = parts[0], parts[1]
        path = expand((run.get("store_uri") or "").split("#", 1)[0])
        if path and path.endswith("final_rung.json"):
            return path, target, f"{C.CONTROL_OF_EVAL[run['eval_id']]}|{cell}", "r2 row key + run.store_uri"
    return None


def resolve_sources(run: dict, judgment: dict, finals: Dict[str, dict]) -> Tuple[List[Tuple[int, str, Optional[str]]], str, List[str]]:
    """``([(trial, cell_dir, expected judge_items sha256 or None)], how, problems)``."""
    problems: List[str] = []
    su = expand((judgment.get("store_uri") or "").split("#", 1)[0])
    if su and su.endswith("/judge_items.jsonl"):
        return [(0, os.path.dirname(su), None)], "judgment.store_uri (cell judge_items)", problems
    if su and os.path.isdir(su) and os.path.exists(os.path.join(su, "judge_items.jsonl")):
        return [(0, su, None)], "judgment.store_uri (cell dir)", problems
    extra = json.loads(run.get("extra_json") or "{}")
    driver = driver_cells(extra)
    loc = locate_final_rung(run, judgment)
    if loc is None:
        if driver:
            # the paired lane's cross-model runs: no final-rung entry (the index lists the Qwen targets only), but the run names each
            # trial's judged store at its final cap (extra.driver_at_cap_by_trial[t].source: <pass-2 cell>/cap_ladder.json), and that cap
            # must be the run's own final cap (extra.final_caps[t])
            caps = extra.get("final_caps") or {}
            out = []
            for t, (cell, cap) in sorted(driver.items()):
                if caps and str(caps.get(str(t))) != str(cap):
                    problems.append(f"trial {t}: the driver's cap {cap} is not the run's final cap {caps.get(str(t))}")
                    continue
                out.append((t, cell, None))
            return out, "run.extra.driver_at_cap_by_trial (pass-2 cells at the final caps)", problems
        return [], "unresolved", [f"no cell and no final-rung entry for store_uri {judgment.get('store_uri')!r}"]
    path, target, key, how = loc
    if path not in finals:
        finals.update(load_finals([path]))
    try:
        entry = finals[path]["targets"][target]["controls"][key]
    except KeyError:
        return [], how, [f"final-rung entry {target}/{key} not in {path}"]
    out = []
    for t, tr in sorted((entry.get("trials") or {}).items(), key=lambda kv: int(kv[0])):
        cell = tr.get("pass2_store")
        if not cell:
            problems.append(f"trial {t} of {target}/{key} has no pass2_store (status {tr.get('status')})")
            continue
        cell = expand(cell)
        if driver and int(t) in driver and os.path.realpath(driver[int(t)][0]) != os.path.realpath(cell):
            problems.append(f"trial {t}: the index's pass2_store {cell} is not the run's driver cell {driver[int(t)][0]}")
            continue
        out.append((int(t), cell, (tr.get("judge") or {}).get("judge_items_sha256")))
    return out, f"final_rung {os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(path))))}:{target}/{key} via {how}", problems


def driver_cells(extra: Dict[str, Any]) -> Dict[int, Tuple[str, Any]]:
    """``{trial: (pass-2 cell dir, cap)}`` from a run's ``extra.driver_at_cap_by_trial`` (``source`` = ``<cell>/cap_ladder.json:controls.<c>``)."""
    out: Dict[int, Tuple[str, Any]] = {}
    for t, v in (extra.get("driver_at_cap_by_trial") or {}).items():
        src = (v or {}).get("source") if isinstance(v, dict) else None
        if not src:
            continue
        p = src.split(":", 1)[0]
        if p.endswith("/cap_ladder.json"):
            out[int(t)] = (expand(os.path.dirname(p)), v.get("cap"))
    return out


# --------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(HOME, "registry", "registry.sqlite"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--xstest-subset", choices=("safe", "all"), default="safe")
    ap.add_argument("--runs", help="scope exactly these run ids (one per line) instead of the paper rows: a supplement (the contrast "
                                   "anchors of paper runs), or a rebuild of named runs to diff against an earlier scope")
    args = ap.parse_args(argv)
    run_ids = read_run_ids(args.runs) if args.runs else None
    os.makedirs(args.out, exist_ok=True)
    os.chmod(args.out, 0o700)

    from sisteer.capeval import corpora
    from sisteer.capeval.tiers import judged
    pins = corpora.load_pins()
    pinned = {"or_bench_hard": {r["item_id"]: r for r in judged.or_bench_rows(pins)},
              "xstest": {r["item_id"]: r for r in judged.xstest_rows(pins)}}
    rubric_sha16 = {ev: C.RUBRICS[C.RUBRIC_OF_EVAL[ev]].sha for ev in pinned}
    psha16 = {ev: C.judging.protocol_sha(C.RUBRICS[C.RUBRIC_OF_EVAL[ev]]) for ev in pinned}

    runs, judg, judges = paper_runs(args.registry, run_ids)
    finals: Dict[str, dict] = {}
    cell_cache: Dict[Tuple[str, str], Tuple[List[dict], Dict[str, str], str]] = {}
    problems: List[dict] = []
    texts: Dict[str, dict] = {}
    n_units = 0
    by_lane: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    ufh = gzip.open(os.path.join(args.out, "units.tsv.gz"), "wt")
    ufh.write("run_id\titem_id\ttrial\tsample\tturn\tcustom_id\trubric\teval_id\torig_label\torig_status\tnote\n")
    rfh = open(os.path.join(args.out, "runs.jsonl"), "w")
    for run in runs:
        ev = run["eval_id"]
        lane = run["lane_source_legacy"] or "(full22 sampled cells)"
        sj = [j for j in judg[run["run_id"]] if j["judge_id"] == judges[ev] and j["status"] == "complete"]
        if len(sj) != 1:
            problems.append({"run_id": run["run_id"], "problem": f"{len(sj)} complete Sonnet judgments"})
            if not sj:
                continue
            sj.sort(key=lambda j: j["finished_utc"] or "")
        jm = sj[-1]
        sources, how, probs = resolve_sources(run, jm, finals)
        for p in probs:
            problems.append({"run_id": run["run_id"], "problem": p})
        task = C.TASK_OF_EVAL[ev]
        rubric = C.RUBRIC_OF_EVAL[ev]
        cnt: collections.Counter = collections.Counter()
        src_info = []
        for trial, cell, want_sha in sources:
            ji_path = os.path.join(cell, "judge_items.jsonl")
            spool_path = os.path.join(cell, "private", f"gens_{task.split('/', 1)[1]}.jsonl")
            if not (os.path.exists(ji_path) and os.path.exists(spool_path)):
                problems.append({"run_id": run["run_id"], "problem": f"cell {cell} lacks judge_items.jsonl or its spool"})
                continue
            ck = (cell, task)
            if ck not in cell_cache:
                cell_cache[ck] = (read_judge_items(ji_path), read_spool(spool_path), sha256_file(ji_path))
            items, gens, have_sha = cell_cache[ck]
            if want_sha and want_sha != have_sha:
                problems.append({"run_id": run["run_id"], "problem": f"judge_items.jsonl of {cell} has sha256 {have_sha[:12]}, the index recorded {want_sha[:12]}"})
                continue
            src_info.append({"trial": trial, "cell": cell, "judge_items_sha256": have_sha})
            for row in items:
                if row.get("task") != task:
                    continue
                if row.get("rubric_sha") != rubric_sha16[ev] or row.get("protocol_sha") != psha16[ev] or row.get("judge_model") != C.ORIGINAL_MODEL:
                    cnt["other_judge_rows"] += 1
                    continue
                iid = row["item_id"]
                pr = pinned[ev].get(iid)
                if pr is None:
                    problems.append({"run_id": run["run_id"], "problem": f"item {iid} is not a pinned {ev} row"})
                    continue
                if ev == "xstest" and args.xstest_subset == "safe" and pr.get("subset") != "safe":
                    cnt["xstest_contrast_skipped"] += 1
                    continue
                resp = gens.get(iid)
                if resp is None:
                    problems.append({"run_id": run["run_id"], "problem": f"item {iid} has no generation in {spool_path}"})
                    continue
                key = C.request_key(rubric, pr["prompt"], resp)
                if key != row.get("request_key"):
                    cnt["key_mismatch"] += 1
                    problems.append({"run_id": run["run_id"], "problem": f"item {iid} trial {trial}: recomputed key {key[:12]} != stored {str(row.get('request_key'))[:12]}"})
                    continue
                if key not in texts:
                    texts[key] = {"k": key, "rubric": rubric, "item_id": iid, "prompt": pr["prompt"], "response": resp}
                note = {"cell": len(src_info) - 1}
                if ev == "xstest":
                    note["subset"] = pr.get("subset")
                ufh.write("\t".join([run["run_id"], iid, str(trial), "0", "-1", key, rubric, ev, str(row.get("label") or ""),
                                     str(row.get("status") or ""), json.dumps(note, separators=(",", ":"))]) + "\n")
                cnt["units"] += 1
                cnt["status_" + str(row.get("status"))] += 1
        n_units += cnt["units"]
        rec = {"run_id": run["run_id"], "eval_id": ev, "campaign": run["campaign"], "lane": lane, "rubric": rubric,
               "sonnet_judge_id": jm["judge_id"], "sonnet_judgment_id": jm["judgment_id"], "sonnet_n_items_judged": jm["n_items_judged"],
               "sonnet_n_judge_refusals": jm["n_judge_refusals"], "sources_how": how, "sources": src_info, "counts": dict(cnt),
               "model_id": run["model_id"], "object_id": run["object_id"], "selection_state": run["selection_state"]}
        rfh.write(json.dumps(rec, sort_keys=True) + "\n")
        bl = by_lane[f"{ev}|{lane}"]
        bl["runs"] += 1
        bl["units"] += cnt["units"]
        bl["sonnet_n_items_judged"] += int(jm["n_items_judged"] or 0)
        bl["sources"] += len(src_info)
        bl["key_mismatch"] += cnt["key_mismatch"]
        bl["status_judged"] += cnt["status_judged"]
    ufh.close()
    rfh.close()
    with gzip.open(os.path.join(args.out, "texts.jsonl.gz"), "wt") as fh:
        for k in sorted(texts):
            fh.write(json.dumps(texts[k], ensure_ascii=True) + "\n")
    distinct = collections.Counter(t["rubric"] for t in texts.values())
    if run_ids is not None:
        found = {r["run_id"] for r in runs}
        problems += [{"run_id": rid, "problem": "a --runs id that is not an or_bench_hard / xstest run of the registry"}
                     for rid in run_ids if rid not in found]
    report = {"paper_runs": len(runs), "units": n_units, "distinct_requests": len(texts), "distinct_by_rubric": dict(distinct),
              "xstest_subset": args.xstest_subset, "cells_read": len(cell_cache), "by_lane": {k: dict(v) for k, v in sorted(by_lane.items())},
              "n_problems": len(problems), "problems": problems,
              "rubric_sha16": rubric_sha16, "protocol_sha16": psha16, "sonnet_judges": judges}
    if run_ids is not None:
        report["selection"] = {"runs_file": os.path.abspath(args.runs), "requested": len(run_ids),
                               "rule": "exactly the --runs ids, not the paper rows (paper_runs counts the runs scoped)"}
    with open(os.path.join(args.out, "report.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({k: v for k, v in report.items() if k != "problems"}, indent=1))
    for p in problems[:30]:
        print("  PROBLEM", p["run_id"], p["problem"])
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
