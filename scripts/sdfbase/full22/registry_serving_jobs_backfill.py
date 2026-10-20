#!/usr/bin/env python3
"""The SERVINGJOBS backfill: ONE run_patch per CURRENT run of a model
whose store was served by two or more jobs (the multi-turn shard groups), stating them on the run as ``extra.serving_jobs`` — a SORTED list
of the sacct-verified job ids from OUR funnel ledger (never a store's routing tables) with ``extra.serving_jobs_source`` — written after the
producer's last full run record (the spool's next sequence number). Dry by default; nothing is written without ``--write``.

The declared set is degeneracy_gap_fill.declared_jobs_for (the same rule degap-fix-6's walker holds every store to): {run.cluster_job_id} ∪
{the producer key's @job} ∪ the ledger's ``attempt`` job ids of the run's OWN unit — resolved (backfill-fix-1) from the
ledger's record of the run's job, else the producer key's shape (``…_t1`` -> static/t1, ``g8k-…-<object>`` -> gsm8k/<object>, the attack +
shard group for multi-turn), never from registry trial_index or a family prefix — each verified terminal through sacct on its cluster
(read-only). Refusals by name, nothing written for that run: a unit that is not this run's own (a mis-filed job, a key the ledger
contradicts, an unresolvable unit); a sacct-unverified ledger job; a serving_jobs entry neither the ledger nor the registry knows; with
``--census <results.jsonl>`` (the walker's cluster results), a store serving job the declared set does not cover AND a declared job no shard
names that is not an attempt of this run's unit (an addition); a run directory without its full ``run`` record. A run whose
extra.serving_jobs already states exactly the set is skipped (idempotent). The ``--sacct-cache`` keeps TERMINAL states only (non-terminal
entries are expunged on load and before save) and a dry run never writes it."""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[2] / "src"))

SOURCE = "funnel ledger D1915"
DECISION = "D1918"


def _walker():
    spec = importlib.util.spec_from_file_location("degeneracy_gap_fill_for_backfill", HERE / "degeneracy_gap_fill.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)   # type: ignore[union-attr]
    return mod


def current_runs(db: Path, model: str, run_ids: Optional[List[str]] = None) -> List[dict]:
    """The model's CURRENT runs (v_run_current.is_current = 1) with the identity the ledger lookup and the patch need."""
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True); conn.row_factory = sqlite3.Row
    q = ("SELECT r.run_id, r.model_id, r.eval_id, r.view_legacy, r.trial_index, r.cluster_job_id, r.producer_run_key_legacy, r.campaign, r.extra_json "
         "FROM v_run_current c JOIN run r ON r.run_id = c.run_id WHERE c.is_current = 1 AND r.model_id = ?")
    rows = []
    for r in conn.execute(q, (model,)):
        if run_ids and r["run_id"] not in run_ids:
            continue
        try:
            extra = json.loads(r["extra_json"]) if r["extra_json"] else {}
        except (TypeError, json.JSONDecodeError):
            extra = {}
        rows.append({"run_id": r["run_id"], "model_id": r["model_id"], "eval_id": r["eval_id"], "view": r["view_legacy"], "trial_index": r["trial_index"],
                     "cluster_job_id": r["cluster_job_id"], "producer_run_key": r["producer_run_key_legacy"], "campaign": r["campaign"], "extra": extra})
    conn.close()
    return rows


def census_store_jobs(census: Optional[Path]) -> Dict[str, List[str]]:
    """run_id -> the serving jobs the walker's cluster results attest for its store (degap-fix-5/6 identity facts); {} without a census."""
    out: Dict[str, List[str]] = {}
    if not census:
        return out
    for line in Path(census).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        jobs = set()
        for i in ((r.get("store") or {}).get("identity") or {}).values():
            if isinstance(i, dict):
                jobs.update(str(x) for x in (((i.get("checks") or {}).get("job") or {}).get("store") or []))
        if jobs:
            out[str(r.get("run_id"))] = sorted(jobs)
    return out


def has_full_run_record(spool: Path, campaign: str, run_id: str, identity) -> bool:
    """True when the run's spool directory holds a full `run` record (the producer's statement the patch must follow) — read from the records
    themselves (a producer may bundle several record types in one file, as synth does), never inferred from file names."""
    import gzip
    d = spool / campaign / identity.run_dir_name(run_id)
    for path in sorted(glob.glob(str(d / "*.jsonl")) + glob.glob(str(d / "*.jsonl.gz"))):
        opener = gzip.open if path.endswith(".gz") else open
        try:
            with opener(path, "rt", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line and json.loads(line).get("record_type") == "run":
                        return True
        except (OSError, json.JSONDecodeError):
            continue
    return False


LIVE_SACCT = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "REQUEUED", "REQUEUE_HOLD", "SUSPENDED", "RESIZING", "STAGE_OUT"}   # A job the scheduler still holds


def plan_runs(runs: List[dict], *, ledger_state: Optional[str], sacct: Callable, sacct_cache: dict, census: Dict[str, List[str]], spool: Path,
              walker, identity) -> List[dict]:
    """Per run: the declared set, its sources and the action — patch | skip (one job) | skip (already stated) | not_yet (a live sibling) |
    refuse (reason)."""
    out = []
    for run in runs:
        d = walker.declared_jobs_for(run, extra=run.get("extra"), ledger_state=ledger_state, sacct=sacct, sacct_cache=sacct_cache)
        ids = list(d["declared_jobs"]); unit = d.get("declared_jobs_unit")
        row = {"run_id": run["run_id"], "eval_id": run["eval_id"], "view": run["view"], "campaign": run["campaign"], "unit": unit, "unit_how": d.get("declared_jobs_unit_how"),
               "declared_jobs": ids, "sources": d["declared_jobs_sources"], "unverified": d["declared_jobs_unverified"]}
        stated = [str(x) for x in ((run.get("extra") or {}).get("serving_jobs") or [])]
        store_jobs = census.get(run["run_id"], [])
        # the run's own attempts = its registry job(s) + the ledger attempts filed under ITS unit (read from each
        # source's attempt id, not trusted from the resolver); a declared job no shard names that is not one of them is an addition
        member = walker.unit_members(unit) if unit else (lambda lwid: False)
        own_attempts = set(walker.own_jobs(run)) | {j for j, srcs in d["declared_jobs_sources"].items()
                                                     if any(s.startswith("ledger:") and member(s[len("ledger:"):].split("@", 1)[0]) for s in srcs)}
        additions = sorted(set(ids) - set(store_jobs) - own_attempts) if store_jobs else []
        if d["declared_jobs_problem"]:
            row.update(action="refuse", reason=d["declared_jobs_problem"])
        elif d["declared_jobs_unverified"]:
            # a sibling the scheduler still runs is NOT a defect — the set is final only when every attempt of the unit is terminal
            # ("one patch per run, written AFTER the producer's last full run record"); it reads as "not yet", named with its sacct state
            live = [u for u in d["declared_jobs_unverified"] if u.rsplit(":", 1)[-1].split()[0] in LIVE_SACCT]
            if live and len(live) == len(d["declared_jobs_unverified"]):
                row.update(action="not_yet", reason="; ".join(f"sibling {u.split('@', 1)[0]} {u.rsplit(':', 1)[-1]} — not yet; re-run after the unit is terminal" for u in live))
            else:
                row.update(action="refuse", reason=f"sacct-unverified ledger job(s) {d['declared_jobs_unverified']} — a job the scheduler does not know as ours is not declared")
        elif store_jobs and sorted(set(store_jobs) - set(ids)):
            row.update(action="refuse", reason=f"the store names serving job(s) {sorted(set(store_jobs) - set(ids))} the ledger + registry do not declare (census cross-check)")
        elif additions:
            row.update(action="refuse", reason=f"declared job(s) {additions} that no shard of this store names and that are not attempts of this run's unit {unit} (census cross-check: additions)")
        elif len(ids) < 2 and not store_jobs:
            row.update(action="skip", reason="one serving job: nothing to state (the registry's cluster_job_id)")
        elif stated == ids:
            row.update(action="skip", reason="extra.serving_jobs already states exactly this set")
        elif not has_full_run_record(spool, run["campaign"], run["run_id"], identity):
            row.update(action="refuse", reason=f"no full run record under {spool / run['campaign'] / identity.run_dir_name(run['run_id'])} — a patch must follow the producer's statement")
        else:
            row.update(action="patch", payload={"serving_jobs": ids, "serving_jobs_source": SOURCE})
        out.append(row)
    return out


def run(a, *, sacct: Optional[Callable] = None) -> int:
    from sisteer.registry import emit, identity
    walker = _walker()
    sacct = sacct or walker.sacct_states
    cache_p = Path(os.path.expanduser(a.sacct_cache)) if a.sacct_cache else None
    cache: dict = json.loads(cache_p.read_text()) if cache_p and cache_p.is_file() else {}
    expunged = walker.expunge_nonterminal_sacct(cache)                         # a live state is never trusted across passes
    spool = Path(os.path.expanduser(a.spool))
    runs = current_runs(Path(os.path.expanduser(a.registry)), a.model, [x for x in (a.runs or "").split(",") if x] or None)
    plan = plan_runs(runs, ledger_state=(a.ledger_state or None), sacct=sacct, sacct_cache=cache, census=census_store_jobs(Path(a.census) if a.census else None),
                     spool=spool, walker=walker, identity=identity)
    if cache_p and a.write:
        walker.expunge_nonterminal_sacct(cache)
        cache_p.parent.mkdir(parents=True, exist_ok=True); cache_p.write_text(json.dumps(cache, indent=1, sort_keys=True))
    n = {k: sum(1 for r in plan if r["action"] == k) for k in ("patch", "skip", "not_yet", "refuse")}
    mode = "WRITE" if a.write else "DRY-RUN"
    print(f"[serving-jobs backfill] {mode}: {len(plan)} current run(s) of {a.model}; {n['patch']} to patch, {n['skip']} skipped, {n['not_yet']} not yet (live siblings), {n['refuse']} refused")
    if cache_p:
        print(f"[serving-jobs backfill] sacct cache {cache_p}: {len(expunged)} non-terminal entr{'y' if len(expunged) == 1 else 'ies'} expunged on load; "
              + (f"{len(cache)} terminal state(s) written" if a.write else "NOT written (dry run)"))
    fams: dict = {}
    for r in plan:                                        # the four classes per family and per view
        for key in (("family", r["eval_id"]), ("view", str(r["view"]))):
            fams.setdefault(key, {"patch": 0, "skip": 0, "not_yet": 0, "refuse": 0})[r["action"]] += 1
    for (kind, name), c in sorted(fams.items()):
        print(f"  by {kind:6s} {name:34s} patch {c['patch']:3d}  skip {c['skip']:3d}  not_yet {c['not_yet']:3d}  refuse {c['refuse']:3d}")
    written = []
    for r in plan:                                        # the would-patch set per run first, then the not-yet runs apart from the refusals
        if r["action"] == "patch":
            print(f"  {'patch' if a.write else 'would patch'} {r['eval_id']:12s} {r['run_id'][:44]}… unit={r['unit']} serving_jobs={r['payload']['serving_jobs']}")
    for r in plan:
        if r["action"] == "not_yet":
            print(f"  NOT YET {r['eval_id']:12s} {r['run_id'][:44]}…: {r['reason'][:220]}")
    for r in plan:
        if r["action"] == "refuse":
            print(f"  REFUSED {r['eval_id']:12s} {r['run_id'][:44]}…: {r['reason'][:200]}")
        elif r["action"] == "patch":
            if a.write:
                producer = emit.producer_block(str(Path(__file__).resolve()), code_sha=a.code_sha)
                w = emit.RecordWriter(str(spool), r["campaign"], r["run_id"], producer, push_script=(a.push_script or None))
                p = w.emit_patch(fields={"extra_json": r["payload"]})
                w.push(final=True)
                written.append({"run_id": r["run_id"], "record": p}); r["record"] = p
    if a.report:
        Path(a.report).parent.mkdir(parents=True, exist_ok=True)
        Path(a.report).write_text(json.dumps({"mode": mode, "model": a.model, "registry": a.registry, "ledger_state": a.ledger_state, "census": a.census,
                                              "source": SOURCE, "decision": DECISION, "plan": plan, "written": written}, indent=1, sort_keys=True, default=str) + "\n")
    print(f"[serving-jobs backfill] {len(written)} record(s) written" + (f" -> {a.report}" if a.report else ""))
    return 0 if not n["refuse"] else 3


def main(argv=None, *, sacct: Optional[Callable] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--registry", required=True, help="the registry database (read-only)")
    ap.add_argument("--model", required=True, help="the registry model_id whose current runs are backfilled")
    ap.add_argument("--spool", required=True, help="the records spool the patches are written to (<root>/records; the harvester folds them)")
    ap.add_argument("--ledger-state", default="", help="the funnel state root holding <model>/attempts.jsonl (D1915); '' = no ledger (the registry's job only)")
    ap.add_argument("--sacct-cache", default="", help="a JSON file caching `cluster:job -> sacct state` across passes")
    ap.add_argument("--census", default="", help="the walker's cluster results.jsonl: the stores' attested serving jobs, cross-checked against the declared set")
    ap.add_argument("--runs", default="", help="comma list: only these run ids")
    ap.add_argument("--code-sha", default="", help="the tree this pass runs from (the records' producer block)")
    ap.add_argument("--push-script", default="", help="the spool push script (optional; the harvester pulls the spool either way)")
    ap.add_argument("--report", default="", help="write the plan + written records as JSON")
    ap.add_argument("--write", action="store_true", help="write the run_patch records (default: dry run, nothing written)")
    return run(ap.parse_args(argv), sacct=sacct)


if __name__ == "__main__":
    raise SystemExit(main())
