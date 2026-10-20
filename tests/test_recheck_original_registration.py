"""32c-b: the live `run.code_sha` is the fold's LAST writer, not provenance — the old proxy registers a run id at job
start under its WAVE_SHA; the amended drive imports the same identity and emits ANOTHER full `run` record for it with its own checkout sha,
and the reducer's fold overwrites `code_sha` while the old `cluster_job_id` remains. The recheck therefore classifies a run the receipt never
listed by its ORIGINAL registration — row_history's first `run` record (stamp, sha, job) — against the receipt's stamp. The fold here is the
REAL one: two producers' records in two spools, folded by the real harvester into a real registry."""
from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, make_registry  # noqa: E402

sys.path.insert(0, str(LIB))
import stage as ST  # noqa: E402

from sisteer.registry import db as DB  # noqa: E402
from sisteer.registry import harvest as H  # noqa: E402
from sisteer.registry import records as R  # noqa: E402
from sisteer.registry import synth  # noqa: E402

OLD, NEW = synth.code_sha("b"), synth.code_sha("c")   # two distinct derived shas: the run code_sha rule (ce4748e23) refuses a placeholder (forty identical digits)
OLD_JOB = "808428"
FRESH_JOB = "909090"          # a job the receipt does not name: the second pass's own (explicit fresh job provenance, 32c-c)


def _emit(root: str, sub: str, name: str, recs) -> None:
    R.write_records(os.path.join(root, "records", sub, name), recs)


def _state(rid: str, state: str, i: int, kind: str, prod, **extra):
    return synth.rec("run_state", {"run_id": rid, "state": state, "event_utc": synth.t(i), "evidence_kind": kind, **extra}, synth.t(i), prod)


def _row(db: str, rid: str):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = conn.execute("select code_sha, cluster_job_id, lifecycle_state from run where run_id=?", (rid,)).fetchone()
        hist = conn.execute("select written_utc, json_extract(payload_json, '$.code_sha') from row_history where table_name='run' and record_type='run' "
                            "and json_extract(natural_key_json, '$.run_id')=? order by written_utc, record_sha256", (rid,)).fetchall()
    finally:
        conn.close()
    return row, hist


def test_a_late_old_proxy_registration_reopened_by_the_amended_drive_reads_amended_on_the_row_but_is_refused_by_its_first_record(tmp_path):
    root = str(tmp_path / "reg"); H.ensure_layout(root)
    old_prod = synth.producer("scripts/sdfbase/full22/common.sh", "cluster_a"); old_prod["code_sha"] = OLD
    new_prod = synth.producer("scripts/sdfbase/full22/petri_drive_full22.sh", "grading_host"); new_prod["code_sha"] = NEW
    _emit(root, "dims", "000000-dims.jsonl", synth.seed_dimensions())
    # spool A (the cluster's): the old proxy registers run K at job start under its WAVE_SHA and the old job, t(0..1)
    pA = synth.run_payload("K", i=0, code_sha=OLD, cluster_job_id=OLD_JOB); rid = str(pA["run_id"])
    _emit(root, "spool_cluster", "000001-K.jsonl", [synth.rec("run", pA, synth.t(0), old_prod), _state(rid, "queued", 0, "submit_wrapper", old_prod),
                                                    _state(rid, "running", 1, "job_log", old_prod, scheduler_state="RUNNING")])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep           # the old registration lands first (as it did)
    receipt_utc = synth.t(5)[:19] + "Z"                   # the receipt, stamped between the two producers' records
    # spool B (the grading host's): the amended drive imports the SAME identity and emits another full `run` record with its own sha, later
    pB = synth.run_payload("K", i=10, code_sha=NEW, lifecycle_state="running", started_utc=synth.t(10)); pB.pop("cluster_job_id", None)
    _emit(root, "spool_grading_host", "000001-K.jsonl", [synth.rec("run", pB, synth.t(10), new_prod)])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep
    db = DB.db_path(root)
    row, hist = _row(db, rid)
    assert row == (NEW, OLD_JOB, "running"), row                                   # the premise from the REAL fold: amended sha over the old job
    assert [h[1] for h in hist] == [OLD, NEW] and hist[0][0] == synth.t(0)
    first = ST.registry_run_first_registration(db, [rid])[rid]
    assert first["code_sha"] == OLD and first["cluster_job_id"] == OLD_JOB and first["written_utc"] == synth.t(0) and first["record_uri"]
    rows = ST.registry_run_rows(db, [rid]); assert rows[rid]["code_sha"] == NEW and rows[rid]["cluster_job_id"] == OLD_JOB
    kw = dict(carried=True, carried_ids=set(), known_ids=set(), amended_shas={NEW}, receipt_utc=receipt_utc, old_job_ids={OLD_JOB})
    lines, probs = ST.petri_run_verdicts([rid], rows, first_regs={rid: first}, **kw)
    # the first record's JOB decides first (32c-c): registered under the receipt's old job — refused whatever its stamp and sha
    assert len(probs) == 1 and rid in probs[0] and "ABSENT" in probs[0] and "under the receipt's OLD job 808428" in probs[0] and "fold's last writer" in probs[0], probs
    assert lines[0]["carried"] is False and "amended_registration" not in lines[0] and "OLD job 808428" in lines[0]["why"]
    # with the receipt's old jobs handed in as another job, the stamp decides: at or before the receipt — refused too
    lines2, probs2 = ST.petri_run_verdicts([rid], rows, first_regs={rid: first}, **{**kw, "old_job_ids": {"1"}})
    assert probs2 and "at or before the receipt" in probs2[0]
    # the live row's sha alone (no history handed in) proves nothing either
    lines, probs = ST.petri_run_verdicts([rid], rows, first_regs={}, **kw)
    assert probs and "no registration history" in probs[0]
    # completed by the amended drive (`generated`, a completed-class state the fold reaches from running): still the OLD contract's
    # completion (its first record predates the receipt)
    _emit(root, "spool_grading_host", "000002-K.jsonl", [_state(rid, "generated", 12, "completion_artifact", new_prod, evidence_uri="synthetic:///store/K/results.json")])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0 and rep.blocked == 0, rep
    rows = ST.registry_run_rows(db, [rid]); assert rows[rid]["lifecycle_state"] == "generated", rows
    lines, probs = ST.petri_run_verdicts([rid], rows, first_regs={rid: first}, **kw)
    assert probs and "COMPLETED under the old contract" in probs[0], probs
    # a run the amended drive FIRST registered after the receipt (a new identity) under a FRESH job is its own: recorded as
    # amended_registration, not carried
    pC = synth.run_payload("L", i=20, code_sha=NEW, lifecycle_state="running", started_utc=synth.t(20), cluster_job_id=FRESH_JOB); ridC = str(pC["run_id"])
    _emit(root, "spool_grading_host", "000003-L.jsonl", [synth.rec("run", pC, synth.t(20), new_prod), _state(ridC, "running", 21, "job_log", new_prod, scheduler_state="RUNNING")])
    # 32c-c: the amended drive runs under the ORIGINAL proxy's job id — a new identity it registers after the mint under
    # the OLD job (amended sha, post-mint stamp) never belonged to the receipt: refused by name; and one with no job at all: refused
    pE = synth.run_payload("N", i=22, code_sha=NEW, lifecycle_state="running", started_utc=synth.t(22), cluster_job_id=OLD_JOB); ridE = str(pE["run_id"])
    _emit(root, "spool_grading_host", "000004-N.jsonl", [synth.rec("run", pE, synth.t(22), new_prod), _state(ridE, "running", 23, "job_log", new_prod, scheduler_state="RUNNING")])
    pF = synth.run_payload("P", i=24, code_sha=NEW, lifecycle_state="running", started_utc=synth.t(24)); pF.pop("cluster_job_id", None); ridF = str(pF["run_id"])
    _emit(root, "spool_grading_host", "000005-P.jsonl", [synth.rec("run", pF, synth.t(24), new_prod), _state(ridF, "running", 25, "job_log", new_prod, scheduler_state="RUNNING")])
    # and a run registered late (after the receipt) under a fresh job but by the OLD sha — the old code re-run elsewhere: refused as the
    # old proxy's (under the old job itself the job rule refuses first, as for K above)
    pD = synth.run_payload("M", i=40, code_sha=OLD, cluster_job_id="909091"); ridD = str(pD["run_id"])
    _emit(root, "spool_cluster", "000002-M.jsonl", [synth.rec("run", pD, synth.t(40), old_prod), _state(ridD, "running", 41, "job_log", old_prod, scheduler_state="RUNNING")])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep
    firsts = ST.registry_run_first_registration(db, [ridC, ridD, ridE, ridF, "11111111-2222-4333-8444-000000000000"])
    assert firsts[ridC]["code_sha"] == NEW and firsts[ridC]["written_utc"] == synth.t(20) and firsts[ridC]["cluster_job_id"] == FRESH_JOB
    assert firsts[ridD]["code_sha"] == OLD and firsts[ridE]["cluster_job_id"] == OLD_JOB and firsts[ridF]["cluster_job_id"] is None and firsts["11111111-2222-4333-8444-000000000000"] is None
    rows = ST.registry_run_rows(db, [ridC, ridD, ridE, ridF])
    lines, probs = ST.petri_run_verdicts([ridC, ridD, ridE, ridF], rows, first_regs=firsts, **kw)
    byid = {x["run_id"]: x for x in lines}
    assert byid[ridC]["amended_registration"] is True and byid[ridC]["carried"] is False and "after the receipt" in byid[ridC]["why"] and FRESH_JOB in byid[ridC]["why"]
    assert len(probs) == 3, probs
    pD_, pE_, pF_ = (next(p for p in probs if r in p) for r in (ridD, ridE, ridF))
    assert "not an amendment's commit" in pD_ and "registering late" in pD_
    assert "under the receipt's OLD job 808428" in pE_ and "never the amended drive's own registration" in pE_ and "amended_registration" not in byid[ridE]
    assert "NO cluster job recorded" in pF_ and "no fresh amended-job provenance" in pF_
    # 32c-d: a LATE old-job registration (after the mint, the old sha) re-emitted by the second pass under a FRESH job —
    # same deterministic id, changed cluster_job_id — leaves the live row under the fresh job: the live read by old job never finds it, the
    # immutable registration history does, and its first record (the old job) refuses it
    pQ = synth.run_payload("Q", i=50, code_sha=OLD, cluster_job_id=OLD_JOB); ridQ = str(pQ["run_id"])
    _emit(root, "spool_cluster", "000003-Q.jsonl", [synth.rec("run", pQ, synth.t(50), old_prod), _state(ridQ, "running", 51, "job_log", old_prod, scheduler_state="RUNNING")])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep
    pQ2 = synth.run_payload("Q", i=60, code_sha=NEW, lifecycle_state="running", started_utc=synth.t(60), cluster_job_id=FRESH_JOB)
    _emit(root, "spool_grading_host", "000006-Q.jsonl", [synth.rec("run", pQ2, synth.t(60), new_prod)])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep
    rowQ, histQ = _row(db, ridQ); assert rowQ == (NEW, FRESH_JOB, "running"), rowQ                 # the live row forgot the old job
    assert ridQ not in ST.registry_runs_of_jobs(db, [OLD_JOB])                                       # the live read cannot find it …
    hist_by_job = ST.registry_runs_of_jobs_history(db, [OLD_JOB])
    assert ridQ in hist_by_job and hist_by_job[ridQ]["cluster_job_id"] == OLD_JOB and hist_by_job[ridQ]["written_utc"] == synth.t(50) and hist_by_job[ridQ]["code_sha"] == OLD
    assert set(hist_by_job) >= {rid, ridQ, ridE} and ridC not in hist_by_job and ridF not in hist_by_job   # K, Q and N named the old job; L and P never did
    assert ST.registry_runs_of_jobs_history(db, []) == {} and ST.registry_runs_of_jobs_history(str(fake := make_registry(tmp_path / "fake2", [])), [OLD_JOB]) == {}
    firstQ = ST.registry_run_first_registration(db, [ridQ])[ridQ]; assert firstQ["cluster_job_id"] == OLD_JOB and firstQ["code_sha"] == OLD
    lines, probs = ST.petri_run_verdicts([ridQ], ST.registry_run_rows(db, [ridQ]), first_regs={ridQ: firstQ}, **kw)
    assert probs and "under the receipt's OLD job 808428" in probs[0], probs
    # a producer that files the job in a run_patch at job start: the first `run` record names no job, the earliest record that does is read
    pR = synth.run_payload("R", i=70, code_sha=OLD); pR.pop("cluster_job_id", None); ridR = str(pR["run_id"])
    _emit(root, "spool_cluster", "000004-R.jsonl", [synth.rec("run", pR, synth.t(70), old_prod),
                                                    synth.rec("run_patch", {"run_id": ridR, "fields": {"cluster_job_id": OLD_JOB}}, synth.t(71), old_prod)])
    rep = H.ingest_once(root, workers=0, min_free=0); assert rep.refused == 0, rep
    firstR = ST.registry_run_first_registration(db, [ridR])[ridR]; assert firstR["cluster_job_id"] == OLD_JOB and firstR["written_utc"] == synth.t(70)
    assert ridR in ST.registry_runs_of_jobs_history(db, [OLD_JOB])
    # the old jobs unknown to the check: nothing is admitted (fail closed) — even the fresh-job identity
    lines, probs = ST.petri_run_verdicts([ridC], rows, first_regs=firsts, **{**kw, "old_job_ids": None})
    assert probs and "old jobs are unknown to this check" in probs[0]
    # array-task spellings of the old job count as the old job
    lines, probs = ST.petri_run_verdicts([ridE], rows, first_regs=firsts, **{**kw, "old_job_ids": {"808428_3", OLD_JOB, "808429"}})
    assert probs and "OLD job 808428" in probs[0]
    # the same new identity against a receipt stamped AFTER its first record is an old registration: refused
    lines, probs = ST.petri_run_verdicts([ridC], rows, first_regs=firsts, **{**kw, "receipt_utc": synth.t(30)[:19] + "Z"})
    assert probs and "at or before the receipt" in probs[0]
    # the same second as the receipt counts as at-or-before (conservative)
    lines, probs = ST.petri_run_verdicts([ridC], rows, first_regs=firsts, **{**kw, "receipt_utc": synth.t(20)[:19] + "Z"})
    assert probs and "at or before the receipt" in probs[0]
    # no history at all, and a registry without row_history (the fixture one): None, refused by name at the caller
    assert ST._amended_registration(None, receipt_utc, {NEW})[0] is False and "no registration history" in ST._amended_registration(None, receipt_utc, {NEW})[1]
    fake = make_registry(tmp_path / "fake", [{"run_id": "x", "lifecycle_state": "running"}])
    assert ST.registry_run_first_registration(str(fake), ["x"]) == {"x": None}
