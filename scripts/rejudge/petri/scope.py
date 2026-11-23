#!/usr/bin/env python3
"""The paper-row scope of the Petri judge re-judge: every CURRENT item the Opus 5 Petri judge filed for a paper run, mapped to the
stored inspect transcript it judged.

    PYTHONPATH=src python scripts/rejudge/petri/scope.py --out SCOPE_DIR \
        --matched items_matched.json [--registry ~/registry/registry.sqlite] [--runs RUNS.txt]

SCOPE: paper rows only -- runs with ``v_run_current.in_current_paper = 1`` on the seven Petri evals
(``petri_pooled`` and the six per-behaviour evals), each with its CURRENT judgment by the Petri judge ``17e480e5`` (petri_bloom_judge,
claude-opus-5; status complete or partial). The Petri AUDITOR (Sonnet 5) is generation and is not re-run. The registry is read
READ-ONLY.

``--runs FILE`` (one run id per line, ``#`` lines skipped) scopes exactly those runs of the seven Petri evals instead of the paper
rows, whatever their ``in_current_paper``: e.g. a supplement for the contrast ANCHORS (the ``none`` arms
the paper runs' ``run_contrast`` rows are measured against, which are not paper rows themselves). Every other step is the paper
scope's, so a run scoped both ways gets the same job and unit rows. A listed id that names no run of a Petri eval is refused.

WHICH TRANSCRIPT. The registry keeps the judge's scores per (behaviour/scenario, trial), not the transcript, so each item is mapped to the
stored inspect sample whose stored judge scores equal the registry's -- ``match_items.py`` (per run, the audit root whose samples match
the run's checkable scores; for the paper's runs every current item matched with 0 score mismatches). This script joins that mapping
to the paper runs' CURRENT items as they stand now and refuses (lists) any item the mapping does not cover or whose value moved since.

Outputs (``SCOPE_DIR``):
  jobs.jsonl      one per paper run: run_id, campaign, eval_id, model_id, object_id, instrument_id, selection_state, the Opus
                  judgment (id, status, n_items_judged), n_items, n_units_mapped, the run's audit root
  units.tsv.gz    one per (run, item, trial, sample, turn): the physical transcript key (``phys`` = audit root | petri_run_key |
                  behaviour | scenario | epoch), its .eval path, scenario id, epoch, behaviour, and the Claude values (petri_score,
                  eval_awareness, scenario_realism, label) for agreement
  report.json     counts per campaign / eval, unmapped items, value mismatches, item-source kinds (native vs adapter revision)
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Optional, Sequence

PETRI_JUDGE = "17e480e55f265f19464b0a17abf52bb4e725cf18bdf24ff3b27203ad80a55727"
AUDITOR_JUDGE = "2d4470edbfabd456c3b01b6a4f57dd44d69cfdf112166cfe82c5e6b2257bf4fe"
PETRI_EVALS = ("petri_pooled", "petri_cyber_bio_chem_capabilities", "petri_defer_to_users", "petri_delusion_sycophancy",
               "petri_instruction_evasion", "petri_reasoning_unfaithfulness", "petri_self_preservation")
METRICS = ("petri_score", "eval_awareness", "scenario_realism")
UNIT_COLUMNS = ("run_id", "item_id", "trial_index", "sample_index", "turn_index", "phys", "path", "scenario", "epoch", "behavior",
                "claude_petri_score", "claude_eval_awareness", "claude_scenario_realism", "claude_label")
HOME = os.path.expanduser("~")


def ro(db: str) -> sqlite3.Connection:
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=120)
    con.row_factory = sqlite3.Row
    return con


def paper_runs(con: sqlite3.Connection, run_ids: Optional[Sequence[str]] = None) -> List[Dict[str, Any]]:
    """The paper runs of the Petri evals, or (``run_ids``) exactly the listed runs of those evals, paper rows or not."""
    marks = ",".join("?" * len(PETRI_EVALS))
    where, params = "vc.in_current_paper = 1", ()
    if run_ids is not None:
        where, params = f"r.run_id IN ({','.join('?' * len(run_ids))})", tuple(run_ids)
    runs = [dict(r) for r in con.execute(
        f"SELECT r.run_id, r.campaign, r.eval_id, r.model_id, r.object_id, r.instrument_id, r.judge_id, r.trial_index, r.store_uri, "
        f"r.extra_json, vc.selection_state, vc.in_current_paper FROM run r JOIN v_run_current vc ON vc.run_id = r.run_id "
        f"WHERE {where} AND r.eval_id IN ({marks}) ORDER BY r.run_id", params + PETRI_EVALS)]
    if run_ids is not None:
        missing = sorted(set(run_ids) - {r["run_id"] for r in runs})
        if missing:
            raise SystemExit(f"--runs: {len(missing)} listed ids name no run of a Petri eval (first {missing[:3]})")
    return runs


def fnum(v):
    return None if v is None else float(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(HOME, "registry", "registry.sqlite"))
    ap.add_argument("--matched", required=True, help="items_matched.json (match_items.py writes it)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", help="file of run ids to scope instead of the paper rows (module docstring)")
    args = ap.parse_args(argv)
    run_ids = None
    if args.runs:
        run_ids = sorted({l.strip() for l in open(args.runs) if l.strip() and not l.startswith("#")})
        if not run_ids:
            raise SystemExit(f"--runs {args.runs} lists no run id")
    os.makedirs(args.out, exist_ok=True)
    con = ro(args.registry)
    runs = paper_runs(con, run_ids)
    m = json.load(open(args.matched))
    by_key = {(o["run_id"], o["item_id"], int(o["trial_index"]), int(o.get("sample_index") or 0)): o for o in m["items"]}
    run_root = m.get("run_root", {})
    jobs, units, problems = [], [], []
    counts: collections.Counter = collections.Counter()
    source_kinds: collections.Counter = collections.Counter()
    for r in runs:
        js = [dict(j) for j in con.execute("SELECT judgment_id, status, n_items_judged, n_judge_refusals, finished_utc FROM judgment "
                                           "WHERE run_id = ? AND judge_id = ? AND status IN ('complete','partial')", (r["run_id"], PETRI_JUDGE))]
        if not js:
            problems.append({"run_id": r["run_id"], "problem": "no current Opus 5 Petri judgment"})
            counts["runs_without_judgment"] += 1
            continue
        if len(js) > 1:
            js.sort(key=lambda j: (j["status"] != "complete", j["finished_utc"] or ""))
            problems.append({"run_id": r["run_id"], "problem": f"{len(js)} current Petri judgments; the complete / latest one taken",
                             "judgments": [j["judgment_id"] for j in js]})
        j = js[0]
        rows = con.execute("SELECT item_id, metric_id, trial_index, sample_index, turn_index, value, label, source_revision_id "
                           "FROM v_item_result_current WHERE run_id = ? AND judgment_id = ?", (r["run_id"], j["judgment_id"])).fetchall()
        items: Dict[tuple, Dict[str, Any]] = collections.OrderedDict()
        for x in rows:
            k = (x["item_id"], int(x["trial_index"]), int(x["sample_index"]), int(x["turn_index"]))
            d = items.setdefault(k, {})
            if x["metric_id"] in METRICS:
                d[x["metric_id"]] = fnum(x["value"])
                if x["metric_id"] == "petri_score" and x["label"]:
                    d["label"] = x["label"]
            source_kinds["native" if x["source_revision_id"] == "" else "adapter_revision"] += 1
        n_mapped = 0
        for (item_id, trial, sample, turn), d in sorted(items.items()):
            if "petri_score" not in d:
                continue                                    # a key carrying only the shared dimensions (none seen); not a unit
            o = by_key.get((r["run_id"], item_id, trial, sample))
            if o is None:
                problems.append({"run_id": r["run_id"], "item_id": item_id, "trial_index": trial, "problem": "not in the transcript mapping"})
                counts["unmapped_items"] += 1
                continue
            units.append({"run_id": r["run_id"], "item_id": item_id, "trial_index": trial, "sample_index": sample, "turn_index": turn,
                          "phys": o["phys"], "path": o["path"], "scenario": o["scenario"], "epoch": int(o["epoch"]), "behavior": o["behavior"],
                          "claude_petri_score": d.get("petri_score"), "claude_eval_awareness": d.get("eval_awareness"),
                          "claude_scenario_realism": d.get("scenario_realism"), "claude_label": d.get("label") or ""})
            n_mapped += 1
        ex = json.loads(r["extra_json"] or "{}")
        jobs.append({"run_id": r["run_id"], "campaign": r["campaign"], "eval_id": r["eval_id"], "model_id": r["model_id"],
                     "object_id": r["object_id"], "instrument_id": r["instrument_id"], "primary_judge_id": r["judge_id"],
                     "selection_state": r["selection_state"], "petri_run_key": ex.get("petri_run_key"),
                     "opus_judgment_id": j["judgment_id"], "opus_judgment_status": j["status"], "opus_n_items_judged": j["n_items_judged"],
                     "n_items": sum(1 for d in items.values() if "petri_score" in d), "n_units_mapped": n_mapped,
                     "audit_root": (run_root.get(r["run_id"]) or {}).get("root")})
        counts["runs"] += 1
        counts[f"runs|{r['campaign']}|{r['eval_id']}"] += 1
        counts["units"] += n_mapped
    with open(os.path.join(args.out, "jobs.jsonl"), "w") as fh:
        for jb in jobs:
            fh.write(json.dumps(jb, sort_keys=True) + "\n")
    with gzip.open(os.path.join(args.out, "units.tsv.gz"), "wt") as fh:
        fh.write("\t".join(UNIT_COLUMNS) + "\n")
        for u in units:
            fh.write("\t".join("" if u[c] is None else str(u[c]) for c in UNIT_COLUMNS) + "\n")
    phys = collections.Counter(u["phys"] for u in units)
    rep = {"registry": args.registry, "matched": args.matched, "runs_file": os.path.abspath(args.runs) if args.runs else None,
           "counts": dict(counts), "distinct_transcripts": len(phys),
           "units_per_transcript": dict(collections.Counter(phys.values())), "item_row_sources": dict(source_kinds),
           "per_campaign_transcripts": dict(collections.Counter(u["phys"].split("|")[0].split("/")[0] + "|" + next(
               jb["campaign"] for jb in jobs if jb["run_id"] == u["run_id"]) for u in {x["phys"]: x for x in units}.values())),
           "problems": problems[:500], "n_problems": len(problems)}
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps({k: rep[k] for k in ("counts", "distinct_transcripts", "units_per_transcript", "item_row_sources", "n_problems")}, indent=1))
    return 0 if not any(p.get("problem") == "not in the transcript mapping" for p in problems) else 2


if __name__ == "__main__":
    sys.exit(main())
