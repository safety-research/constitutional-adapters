#!/usr/bin/env python3
"""A Petri re-judge scope for POOLED runs composed from their behaviour runs: for a ``petri_pooled`` run whose Opus 5 judgment filed no
item rows, the units the paper-row scope (``scope.py``) could not map, so ``file_records.py`` can file the re-judge's verdicts on it.

    PYTHONPATH=src:scripts python scripts/rejudge/petri/pooled_scope.py --scope MAIN_SCOPE --runs RUNS.txt --out SCOPE_DIR \
        [--registry ~/registry/registry.sqlite]

WHY. ``scope.py`` maps each CURRENT Opus item of a run to the transcript it judged, so a run whose Opus judgment carries no item rows
(an export that filed the pooled mean alone: ``sdfbase_full22:petri_pooled:20a6134100b7:e663923c9485c53a`` and its anchor
``...:0a6abac91af2bca5``, 60 items in each judgment header and 0 rows) gets a job with ``n_units_mapped`` 0 and no units, and the
filing skips it.

THE POOLED RULE. A pooled run is its six behaviour runs: the same Petri run's 60 transcripts (6 behaviours x 10 scenarios), each pooled
item ``<behaviour>/<scenario>`` the behaviour run's item of the same key. Measured on the registry and the main scope: in every
pooled run the main scope mapped (81 runs, 4,858 units), each pooled unit equals, column for column, the unit of
the same (item, trial, sample, turn) in the paper behaviour run of the same campaign and run-id object key (the last field of the
run id); every filed pooled Opus item equals that behaviour run's Opus item (82 runs); every filed pooled GLM-5.3 row equals the
behaviour run's GLM-5.3 row of the same key (81 runs: value, label, request key, batch, attempt, provider, transcript); and the
pooled ``petri_score`` run_metric is the mean of the six behaviour means (132 rows). So composing the units here, then filing them
with ``file_records.py`` unchanged, files exactly what the main filing files for a pooled run whose Opus items exist.

WHAT IS COMPOSED. For each listed run (``eval_id`` petri_pooled; refused otherwise): its registry row and CURRENT Opus 5 Petri judgment
(``scope.py``'s selection); its six behaviour runs, taken from MAIN_SCOPE's jobs (``scope.py``'s output for the paper rows): same
campaign and run-id object key, one per behaviour eval, each with the pooled run's model, object and ``petri_run_key`` and every one
of its items mapped. Its units are those six runs' units with ``run_id`` replaced (the transcript, path, scenario, epoch, behaviour
and the Opus values of that transcript are the behaviour unit's). Refused, by name: a missing or duplicated behaviour run, a
model / object / run-key mismatch, a behaviour run with an unmapped item, an item key not under its behaviour, a key two behaviours
both carry; for a pooled run whose Opus judgment carries no items, a composed count that differs from its ``n_items_judged``; and for
one whose Opus judgment DOES carry items, a composed key set that differs from them (that run is then rebuilt, not invented, which is
how the proof checks the rule against filed judgments).

Outputs (``SCOPE_DIR``), the formats ``scope.py`` writes, so ``file_records.py --scope SCOPE_DIR`` reads them unchanged:
  jobs.jsonl      one per listed run: ``scope.py``'s fields, plus ``units_composed_from`` ({behaviour eval: run id}) and
                  ``opus_item_keys`` (the Opus judgment's own petri_score item count; 0 for the runs this exists for)
  units.tsv.gz    ``scope.py``'s columns
  report.json     the inputs, per-run composition and counts
The registry is read READ-ONLY; nothing here calls a model.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import sqlite3
import sys
from typing import Any, Dict, List, Mapping, Sequence, Tuple

_SCRIPTS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from rejudge.petri.scope import HOME, PETRI_EVALS, PETRI_JUDGE, UNIT_COLUMNS, ro  # noqa: E402

POOLED = "petri_pooled"
BEHAVIOUR_EVALS = tuple(e for e in PETRI_EVALS if e != POOLED)


class Refused(SystemExit):
    pass


def run_parts(run_id: str) -> Tuple[str, str, str]:
    """(campaign, eval_id, object key) of ``campaign:eval:instrument12:objectkey``."""
    p = run_id.split(":")
    if len(p) != 4:
        raise Refused(f"{run_id!r} is not a campaign:eval:instrument:key run id")
    return p[0], p[1], p[3]


def read_scope(scope_dir: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, str]]]]:
    jobs = {j["run_id"]: j for j in (json.loads(l) for l in open(os.path.join(scope_dir, "jobs.jsonl")))}
    units: Dict[str, List[Dict[str, str]]] = collections.defaultdict(list)
    with gzip.open(os.path.join(scope_dir, "units.tsv.gz"), "rt") as fh:
        cols = next(fh).rstrip("\n").split("\t")
        if tuple(cols) != UNIT_COLUMNS:
            raise Refused(f"{scope_dir}/units.tsv.gz has columns {cols}, not scope.py's {list(UNIT_COLUMNS)}")
        for line in fh:
            d = dict(zip(cols, line.rstrip("\n").split("\t")))
            units[d["run_id"]].append(d)
    return jobs, units


def registry_runs(con: sqlite3.Connection, run_ids: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """The listed runs' rows, as ``scope.py`` reads a run (its selection state included)."""
    marks = ",".join("?" * len(run_ids))
    rows = [dict(r) for r in con.execute(
        f"SELECT r.run_id, r.campaign, r.eval_id, r.model_id, r.object_id, r.instrument_id, r.judge_id, r.extra_json, vc.selection_state "
        f"FROM run r JOIN v_run_current vc ON vc.run_id = r.run_id WHERE r.run_id IN ({marks})", tuple(run_ids))]
    return {r["run_id"]: r for r in rows}


def opus_judgment(con: sqlite3.Connection, run_id: str) -> Dict[str, Any]:
    """The run's CURRENT Opus 5 Petri judgment: ``scope.py``'s selection (complete or partial; the complete, then latest, one)."""
    js = [dict(j) for j in con.execute("SELECT judgment_id, status, n_items_judged, n_judge_refusals, finished_utc FROM judgment "
                                       "WHERE run_id = ? AND judge_id = ? AND status IN ('complete','partial')", (run_id, PETRI_JUDGE))]
    if not js:
        raise Refused(f"{run_id}: no current Opus 5 Petri judgment")
    js.sort(key=lambda j: (j["status"] != "complete", j["finished_utc"] or ""))
    return js[0]


def opus_item_keys(con: sqlite3.Connection, run_id: str, judgment_id: str) -> set:
    return {(x[0], str(int(x[1])), str(int(x[2])), str(int(x[3]))) for x in con.execute(
        "SELECT item_id, trial_index, sample_index, turn_index FROM v_item_result_current WHERE run_id = ? AND judgment_id = ? "
        "AND metric_id = 'petri_score'", (run_id, judgment_id))}


def behaviour_runs(pooled: Mapping[str, Any], jobs: Mapping[str, Mapping[str, Any]], units: Mapping[str, List[Mapping[str, str]]]) -> Dict[str, str]:
    """{behaviour eval: run id} of a pooled run's six behaviour runs in the main scope (module docstring, WHAT IS COMPOSED)."""
    camp, _, key = run_parts(pooled["run_id"])
    found: Dict[str, List[str]] = collections.defaultdict(list)
    for rid in jobs:
        c, ev, k = run_parts(rid)
        if c == camp and k == key and ev in BEHAVIOUR_EVALS:
            found[ev].append(rid)
    missing = [e for e in BEHAVIOUR_EVALS if not found.get(e)]
    dup = {e: v for e, v in found.items() if len(v) > 1}
    if missing or dup:
        raise Refused(f"{pooled['run_id']}: behaviour runs in the main scope missing for {missing} / duplicated {dup}")
    rkey = (json.loads(pooled.get("extra_json") or "{}")).get("petri_run_key")
    out = {}
    for ev in BEHAVIOUR_EVALS:
        rid = found[ev][0]
        j = jobs[rid]
        bad = {f: (j.get(f), want) for f, want in (("model_id", pooled["model_id"]), ("object_id", pooled["object_id"]),
                                                    ("petri_run_key", rkey)) if j.get(f) != want}
        if bad:
            raise Refused(f"{pooled['run_id']}: behaviour run {rid} differs from the pooled run in {bad}")
        n_units = len(units.get(rid, []))
        if not n_units or n_units != j.get("n_items") or n_units != j.get("n_units_mapped"):
            raise Refused(f"{pooled['run_id']}: behaviour run {rid} has {n_units} units for {j.get('n_items')} items "
                          f"({j.get('n_units_mapped')} mapped); every item must map")
        out[ev] = rid
    return out


def compose(pooled_id: str, beh: Mapping[str, str], units: Mapping[str, List[Mapping[str, str]]]) -> List[Dict[str, str]]:
    """The pooled run's units: the behaviour runs' units under the pooled run id, in behaviour-eval then unit order."""
    out: List[Dict[str, str]] = []
    seen: Dict[tuple, str] = {}
    for ev in BEHAVIOUR_EVALS:
        prefix = ev[len("petri_"):] + "/"
        for u in units[beh[ev]]:
            if not u["item_id"].startswith(prefix):
                raise Refused(f"{pooled_id}: item {u['item_id']!r} of {beh[ev]} is not a {prefix[:-1]} item")
            k = (u["item_id"], u["trial_index"], u["sample_index"], u["turn_index"])
            if k in seen:
                raise Refused(f"{pooled_id}: item key {k} is carried by both {seen[k]} and {beh[ev]}")
            seen[k] = beh[ev]
            out.append(dict(u, run_id=pooled_id))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--registry", default=os.path.join(HOME, "registry", "registry.sqlite"))
    ap.add_argument("--scope", required=True, help="the main paper-row scope (scope.py's output) holding the behaviour runs' units")
    ap.add_argument("--runs", required=True, help="file of petri_pooled run ids, one per line ('#' lines skipped)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    run_ids = sorted({l.strip() for l in open(args.runs) if l.strip() and not l.startswith("#")})
    if not run_ids:
        raise Refused(f"--runs {args.runs} lists no run id")
    if os.path.isdir(args.out) and os.listdir(args.out):
        raise Refused(f"--out {args.out} is not empty")
    jobs, units = read_scope(args.scope)
    con = ro(args.registry)
    rows = registry_runs(con, run_ids)
    missing = [r for r in run_ids if r not in rows]
    if missing:
        raise Refused(f"--runs: {len(missing)} ids name no run (first {missing[:3]})")
    not_pooled = [r for r in run_ids if rows[r]["eval_id"] != POOLED]
    if not_pooled:
        raise Refused(f"--runs: {not_pooled[:3]} are not {POOLED} runs; only a pooled run is composed")
    out_jobs, out_units, per_run = [], [], {}
    for rid in run_ids:
        r = rows[rid]
        j = opus_judgment(con, rid)
        beh = behaviour_runs(r, jobs, units)
        us = compose(rid, beh, units)
        own = opus_item_keys(con, rid, j["judgment_id"])
        mine = {(u["item_id"], u["trial_index"], u["sample_index"], u["turn_index"]) for u in us}
        if own and own != mine:
            raise Refused(f"{rid}: the Opus judgment's {len(own)} item keys differ from the {len(mine)} composed ones "
                          f"({len(own - mine)} only filed, {len(mine - own)} only composed)")
        if not own and len(us) != j["n_items_judged"]:
            raise Refused(f"{rid}: {len(us)} composed units, but the Opus judgment {j['judgment_id']} judged {j['n_items_judged']} items")
        roots = sorted({jobs[b].get("audit_root") for b in beh.values()}, key=str)
        ex = json.loads(r["extra_json"] or "{}")
        out_jobs.append({"run_id": rid, "campaign": r["campaign"], "eval_id": r["eval_id"], "model_id": r["model_id"],
                         "object_id": r["object_id"], "instrument_id": r["instrument_id"], "primary_judge_id": r["judge_id"],
                         "selection_state": r["selection_state"], "petri_run_key": ex.get("petri_run_key"),
                         "opus_judgment_id": j["judgment_id"], "opus_judgment_status": j["status"], "opus_n_items_judged": j["n_items_judged"],
                         "n_items": len(us), "n_units_mapped": len(us), "audit_root": roots[0] if len(roots) == 1 else None,
                         "units_composed_from": dict(beh), "opus_item_keys": len(own)})
        out_units.extend(us)
        per_run[rid] = {"opus_judgment_id": j["judgment_id"], "opus_n_items_judged": j["n_items_judged"], "opus_item_keys": len(own),
                        "units": len(us), "behaviour_runs": dict(beh)}
    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "jobs.jsonl"), "w") as fh:
        for jb in out_jobs:
            fh.write(json.dumps(jb, sort_keys=True) + "\n")
    with gzip.open(os.path.join(args.out, "units.tsv.gz"), "wt") as fh:
        fh.write("\t".join(UNIT_COLUMNS) + "\n")
        for u in out_units:
            fh.write("\t".join(u[c] for c in UNIT_COLUMNS) + "\n")
    rep = {"registry": args.registry, "main_scope": os.path.abspath(args.scope), "runs_file": os.path.abspath(args.runs),
           "rule": "pooled units = the six same-campaign, same-object-key paper behaviour runs' units (module docstring)",
           "counts": {"runs": len(out_jobs), "units": len(out_units)}, "runs": per_run}
    json.dump(rep, open(os.path.join(args.out, "report.json"), "w"), indent=1)
    print(json.dumps(rep["counts"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
