"""29-fix: the run-state reconciliation for a superseded ARRAY job proved only that ALL children were terminal — it never
bound a run to the child that produced it — so on a mixed array (one task FAILED, another COMPLETED with generated work awaiting harvest) the
completed child's run was emitted `failed` from the aggregate task states and the latest End; `failed` is lateral in the reducer (not held behind
`generated`), so that record overrode valid work and removed it from grading. Now each orphaned run is bound to ITS registered array task (the
attempt's retired registered binding, `runs[].array_task`, which 06 --gate wrote from the job's pending file), decided on that task alone: a
COMPLETED task is never `failed` from here (task_completed: the store is the closer's / the harvester's); an array run without a task binding is
not inferred (task_unbound). 29-fix-2: array-ness is READ from the retired binding (its `array` spec / per-run
`array_task`), never inferred from sacct's rows — when every child row is purged and only the aggregate parent answers, every run is refused, none
marked; a plain binding facing task rows, or no binding at all, is refused too. Secondary: a local receipt stands unless the registry REFUSED the
record it names — by the registry's own identity (records.record_sha of the canonical JSON, the root-relative `records/...` uri), proven against a
REAL refusal the harvester's own writer filed."""
from __future__ import annotations

import json
import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB  # noqa: E402
from test_reconcile_registrations import PRODUCER  # noqa: E402

sys.path.insert(0, str(LIB))
import grant_ledger as GL  # noqa: E402
import stage as ST  # noqa: E402

LW = "m/u0/static/t0"
RA = "sdfbase_full22:pasttense:aaaaaaaaaaaa:000000000000000a"
RB = "sdfbase_full22:pasttense:aaaaaaaaaaaa:000000000000000b"
AID1 = f"{LW}@cluster_a@1"


class TaskSacct:
    """(cluster, job id) -> the funnel reader's per-TASK rows."""
    def __init__(self, rows: dict):
        self.rows = rows
    def __call__(self, cluster, job_id):
        return list(self.rows.get((cluster, str(job_id)), []))


def _world(tmp: Path, *, retired_runs=True, registry_states=("running", "running"), array: object = "0-1", run_ids=(RA, RB), db: str | None = None) -> dict:
    """The ledger of a superseded attempt (job 7; an ARRAY of tasks 0/1 by default) whose registered binding was RETIRED, as the funnel leaves it.
    `array`: the binding's sbatch array spec (None = a plain job; "omit" = no `array` key at all — a binding from before the spec was recorded);
    `retired_runs`: whether the retired binding carries `runs` (the per-run facts, with `array_task` for an array); `db`: a REAL registry to read
    instead of the fake `run` table; `run_ids`: the two runs (the first is the one decided)."""
    ra, rb = run_ids
    state = tmp / "state"; (state / "gate").mkdir(parents=True); ledger = str(state / "submissions.jsonl")
    ST.write_atomic(str(state / "gate" / "expected_units.json"), {"schema": 2, "model": "m", "view": "u0", "code_sha": "a" * 40, "units": []})
    L = GL.Ledger(str(state / "attempts.jsonl"))
    L.append({"event": "attempt", "logical_work_id": LW, "attempt_id": AID1, "cluster": "cluster_a", "job_id": "7", "sha": "a" * 40, "utc": "2026-01-01T00:00:00Z"})
    # what 06 --gate wrote when the job registered: the submitted record's array spec + one run per (eval, task) from the pending file — then the
    # attempt was superseded and the binding retired
    rec = {"logical_work_id": LW, "state": "registered", "run_ids": [ra, rb], "job_id": "7", "cluster": "cluster_a", "attempt_id": AID1, "utc": "2026-01-01T00:30:00Z"}
    if array != "omit":
        rec["array"] = array
    if retired_runs:
        tasks = (0, 1) if array not in (None, "omit") else (None, None)
        rec["runs"] = [{"run_id": r, "eval_id": "pasttense", "object": "none", **({"array_task": t} if t is not None else {})} for r, t in zip((ra, rb), tasks)]
    ST.write_binding(ledger, rec)
    assert ST.retire_registered_binding(ledger, LW, AID1)
    L.append({"event": "superseded_attempt", "logical_work_id": LW, "attempt_id": AID1, "job_id": "7", "cluster": "cluster_a", "state": "COMPLETED,FAILED", "binding_state": "registered",
              "run_ids": [ra, rb], "why": "task 0 died", "utc": "2026-01-01T01:00:00Z"})
    ST.write_binding(ledger, {"logical_work_id": LW, "state": "registered", "run_ids": ["other-run"], "job_id": "8", "cluster": "cluster_a", "attempt_id": f"{LW}@cluster_a@2", "utc": "2026-01-01T01:10:00Z"})
    if db is None:
        dbp = tmp / "registry.sqlite"; c = sqlite3.connect(dbp)
        c.execute("CREATE TABLE run (run_id TEXT PRIMARY KEY, lifecycle_state TEXT, campaign TEXT, cluster_job_id TEXT)")
        c.executemany("INSERT INTO run VALUES (?, ?, ?, ?)", [(ra, registry_states[0], "sdfbase_full22", "7"), (rb, registry_states[1], "sdfbase_full22", "7")])
        c.commit(); c.close(); db = str(dbp)
    spool = tmp / "spool"; spool.mkdir()
    return {"state": state, "ledger": ledger, "db": db, "spool": str(spool)}


MIXED = TaskSacct({("cluster_a", "7"): [{"raw": "70", "task": 0, "state": "FAILED", "terminal": True, "end": "2026-01-01T00:50:00"},
                                        {"raw": "71", "task": 1, "state": "COMPLETED", "terminal": True, "end": "2026-01-01T00:55:00"}]})


def test_on_a_mixed_array_each_run_is_decided_on_its_own_task_and_the_completed_task_is_never_failed(tmp_path):
    w = _world(tmp_path)
    rep = ST.reconcile_registrations(w["ledger"], w["db"], MIXED, spool=w["spool"], write=False, producer=PRODUCER)
    assert rep["n_superseded_runs"] == 2
    assert [(x["run_id"], x["state"], x["event_utc"], x["array_task"]) for x in rep["plan"]] == [(RA, "failed", "2026-01-01T00:50:00.000Z", 0)]   # task 0's own End, not the latest
    assert rep["plan"][0]["sacct_states"] == ["FAILED"] and json.loads(rep["plan"][0]["note"])["array_task"] == 0
    assert [x["run_id"] for x in rep["skipped"]["task_completed"]] == [RB] and rep["skipped"]["task_completed"][0]["task"] == 1 \
        and "never `failed` from here" in rep["skipped"]["task_completed"][0]["why"]
    assert rep["skipped"]["task_unbound"] == [] and rep["skipped"]["job_alive"] == []
    # the write files exactly the one terminal record; the completed task's run is untouched
    rep = ST.reconcile_registrations(w["ledger"], w["db"], MIXED, spool=w["spool"], write=True, producer=PRODUCER)
    assert [x["run_id"] for x in rep["written"]] == [RA] and len(list(Path(w["spool"]).rglob("*run_state*.jsonl"))) == 1
    rec = json.loads(Path(rep["written"][0]["record_path"]).read_text().splitlines()[-1])
    assert rec["payload"]["state"] == "failed" and rec["payload"]["event_utc"] == "2026-01-01T00:50:00.000Z" if "payload" in rec else True


def test_an_array_run_without_a_task_binding_is_not_inferred_from_its_siblings(tmp_path):
    w = _world(tmp_path, retired_runs=False)
    rep = ST.reconcile_registrations(w["ledger"], w["db"], MIXED, spool=w["spool"], write=False, producer=PRODUCER)
    assert rep["plan"] == [] and sorted(x["run_id"] for x in rep["skipped"]["task_unbound"]) == sorted([RA, RB])
    assert all("not inferred from sibling tasks" in x["why"] for x in rep["skipped"]["task_unbound"])
    # a plain (non-array) job that COMPLETED is not `failed` either: its run's store is the closer's / harvester's to read — the binding declares the
    # plain job (array None; 29-fix-2: array-ness is read from the binding, never from the rows)
    wp = _world(tmp_path / "plain", retired_runs=False, array=None)
    plain = TaskSacct({("cluster_a", "7"): [{"raw": "7", "task": None, "state": "COMPLETED", "terminal": True, "end": "2026-01-01T00:55:00"}]})
    rep = ST.reconcile_registrations(wp["ledger"], wp["db"], plain, spool=wp["spool"], write=False, producer=PRODUCER)
    assert rep["plan"] == [] and sorted(x["run_id"] for x in rep["skipped"]["task_completed"]) == sorted([RA, RB])
    # … while a plain FAILED job still files both of its runs failed (no array: no per-task question)
    dead = TaskSacct({("cluster_a", "7"): [{"raw": "7", "task": None, "state": "TIMEOUT", "terminal": True, "end": "2026-01-01T00:59:00"}]})
    rep = ST.reconcile_registrations(wp["ledger"], wp["db"], dead, spool=wp["spool"], write=False, producer=PRODUCER)
    assert sorted((x["run_id"], x["state"], x["array_task"]) for x in rep["plan"]) == [(RA, "failed", None), (RB, "failed", None)]


PARENT_ONLY = TaskSacct({("cluster_a", "7"): [{"raw": "7", "task": None, "state": "FAILED", "terminal": True, "end": "2026-01-01T00:55:00"}]})


def test_with_every_child_row_purged_the_aggregate_parent_decides_nothing_and_every_run_is_refused_not_marked(tmp_path):
    """29-fix-2 HIGH (1): sacct returns only the aggregate parent row (task None, FAILED) — before, array-ness was inferred from the rows, the job
    looked plain, and the parent's FAILED marked EVERY orphan run, the completed child's included. Now the binding says array: each run needs its
    exact child row; absent, it is refused by name and nothing is written."""
    w = _world(tmp_path)
    rep = ST.reconcile_registrations(w["ledger"], w["db"], PARENT_ONLY, spool=w["spool"], write=True, producer=PRODUCER)
    assert rep["plan"] == [] and rep["written"] == [] and not list(Path(w["spool"]).rglob("*.jsonl"))
    assert sorted(x["run_id"] for x in rep["skipped"]["job_unproven"]) == sorted([RA, RB]) and all("never decided from the aggregate parent row" in x["why"] for x in rep["skipped"]["job_unproven"])
    assert [x["why"] for x in rep["skipped"]["job_unproven"] if x["run_id"] == RA][0].startswith("array task 0 of job 7 is not in sacct's answer")
    # the binding declares the array only by its spec (no per-run tasks): still an array — task_unbound, never the parent
    w = _world(tmp_path / "spec", retired_runs=False)
    rep = ST.reconcile_registrations(w["ledger"], w["db"], PARENT_ONLY, spool=w["spool"], write=True, producer=PRODUCER)
    assert rep["plan"] == [] and sorted(x["run_id"] for x in rep["skipped"]["task_unbound"]) == sorted([RA, RB])


def test_a_plain_binding_facing_task_rows_and_a_binding_that_says_nothing_are_refused_never_inferred(tmp_path):
    # the binding declares a plain job (array None, runs without tasks) but sacct lists array task rows: a contradiction — refused, nothing decided
    w = _world(tmp_path / "plain", array=None)
    rep = ST.reconcile_registrations(w["ledger"], w["db"], MIXED, spool=w["spool"], write=True, producer=PRODUCER)
    assert rep["plan"] == [] and sorted(x["run_id"] for x in rep["skipped"]["job_unproven"]) == sorted([RA, RB]) and all("a contradiction" in x["why"] for x in rep["skipped"]["job_unproven"])
    # the same plain binding with a plain row: decided from the job's own row
    rep = ST.reconcile_registrations(w["ledger"], w["db"], PARENT_ONLY, spool=w["spool"], write=False, producer=PRODUCER)
    assert sorted((x["run_id"], x["state"], x["array_task"]) for x in rep["plan"]) == [(RA, "failed", None), (RB, "failed", None)]   # both ride the one plain job
    # no retired binding carrying `runs` and no `array` key: array-ness unknowable — refused by name, whatever sacct shows
    for rows in (PARENT_ONLY, MIXED):
        w = _world(tmp_path / f"unknown{len(rows.rows[('cluster_a', '7')])}", retired_runs=False, array="omit")
        rep = ST.reconcile_registrations(w["ledger"], w["db"], rows, spool=w["spool"], write=True, producer=PRODUCER)
        assert rep["plan"] == [] and sorted(x["run_id"] for x in rep["skipped"]["job_unproven"]) == sorted([RA, RB]) and all(x["why"].startswith("array-ness unknowable") for x in rep["skipped"]["job_unproven"])


def test_a_local_receipt_stands_unless_the_registry_refused_its_record(tmp_path):
    """29-fix-2 HIGH (2): the receipt names its record by the REGISTRY'S identity — records.record_sha of the canonical JSON and the root-relative
    `records/...` uri — and a REAL refusal, filed by the harvester's own writer over a record the reconciliation itself wrote, voids it. Ordinary
    ingest lag (the record not yet folded) honours the receipt: the second `--write` is a no-op by contract."""
    import hashlib
    from sisteer.registry import db as DB, harvest as H, records as R, synth
    root = str(tmp_path / "reg"); H.ensure_layout(root)
    p = synth.run_payload("k1", i=0)                                                                           # a real registered run (lifecycle queued)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions() + [synth.rec("run", p, synth.t(5))])
    rep0 = H.ingest_once(root, workers=0, min_free=0); assert rep0.refused == 0, rep0
    db = DB.db_path(root); spool = os.path.join(root, "records"); rid = p["run_id"]
    def q(sql, *a):
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return c.execute(sql, a).fetchall()
        finally:
            c.close()
    assert q("SELECT lifecycle_state, campaign FROM run WHERE run_id = ?", rid) == [("queued", p["campaign"])]
    w = _world(tmp_path, run_ids=(rid, RB), db=db, array=None)                                                # a plain job whose run is the real one
    bad = {**PRODUCER, "code_sha": "not-a-sha"}                   # the one thing a well-formed terminal run_state can be refused for: producer.code_sha (validate.py)
    rep = ST.reconcile_registrations(w["ledger"], db, PARENT_ONLY, spool=spool, write=True, producer=bad)
    assert [x["run_id"] for x in rep["written"]] == [rid], rep
    receipt = ST.read_reconciled(w["ledger"])[rid]
    with open(receipt["record_path"], encoding="utf-8") as fh:
        obj = json.loads(fh.readline())
    assert receipt["record_sha256"] == R.record_sha(obj) != hashlib.sha256(open(receipt["record_path"], "rb").read()).hexdigest()           # the registry's identity, not the file's bytes
    assert receipt["record_uri"] == "records/" + os.path.relpath(receipt["record_path"], spool) and receipt["record_uri"].startswith(f"records/{p['campaign']}/")
    # not folded yet: lag — the receipt stands, nothing planned, nothing written twice
    rep = ST.reconcile_registrations(w["ledger"], db, PARENT_ONLY, spool=spool, write=True, producer=bad)
    assert rep["plan"] == [] and rep["written"] == [] and rep["skipped"]["already_reconciled"] == [rid] and rep["retried_receipts"] == []
    # the REAL harvester refuses the record — its own writer files record_refusal with exactly the identity the receipt carries
    rep1 = H.ingest_once(root, workers=0, min_free=0); assert rep1.refused == 1, rep1
    rows = q("SELECT record_sha256, record_uri, field, reason FROM record_refusal")
    assert len(rows) == 1 and rows[0][0] == receipt["record_sha256"] and rows[0][1].split("#", 1)[0] == receipt["record_uri"] and rows[0][2] == "producer.code_sha" and "not 40-hex" in rows[0][3], rows
    assert q("SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("queued",)]                                                      # the refused record changed nothing
    rep = ST.reconcile_registrations(w["ledger"], db, PARENT_ONLY, spool=spool, write=False, producer=PRODUCER)
    assert [x["run_id"] for x in rep["plan"]] == [rid] and rep["plan"][0]["retry_of_receipt"] is True and rep["skipped"]["already_reconciled"] == [] \
        and rep["retried_receipts"][0]["run_id"] == rid and rep["retried_receipts"][0]["refusal"]["field"] == "producer.code_sha" \
        and json.loads(rep["plan"][0]["note"]).get("retry_of_receipt") == receipt["written_utc"]
    # the retry under a sound producer is accepted: the run turns failed and the receipt stands from then on
    rep = ST.reconcile_registrations(w["ledger"], db, PARENT_ONLY, spool=spool, write=True, producer=PRODUCER)
    assert [x["run_id"] for x in rep["written"]] == [rid]
    rep2 = H.ingest_once(root, workers=0, min_free=0); assert rep2.refused == 0, rep2
    assert q("SELECT lifecycle_state FROM run WHERE run_id = ?", rid) == [("failed",)]
    rep = ST.reconcile_registrations(w["ledger"], db, PARENT_ONLY, spool=spool, write=False, producer=PRODUCER)
    assert rep["plan"] == [] and rep["skipped"]["already_reconciled"] == [rid] and rep["retried_receipts"] == []
