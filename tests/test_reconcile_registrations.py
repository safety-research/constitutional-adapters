"""`stage.py reconcile-registrations` (commit 29): the runs a superseded attempt registered that NO later generation continues become
terminal in the registry too — `run_state` records (cancelled | failed, sacct's state as the evidence) written into the producers' spool,
never a database write (the registry owner's ruling, 2026-10-01 02:34Z: spool = the live records dir, evidence_kind `scheduler_state`,
state `cancelled` for CANCELLED else `failed`, event_utc = sacct End; the fold is monotonic). Dry-run by default; `--write` writes once
(a local receipt beside the ledger keeps a second pass from writing again); 06 --gate calls it after each registered binding."""
from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, VENV_PY, set_job_state, write_fake_slurm, write_site  # noqa: E402

sys.path.insert(0, str(LIB))
import grant_ledger as GL  # noqa: E402
import stage as ST  # noqa: E402

KEY = "m"
LW1 = f"{KEY}/u0/mmlu/mmlu"; LW2 = f"{KEY}/u0/gsm8k/g"; LW3 = f"{KEY}/u0/static/t0"
R = {k: f"sdfbase_full22:{e}:aaaaaaaaaaaa:{k:0>16}" for k, e in (("r1", "mmlu"), ("r2", "mmlu"), ("r3", "gsm8k"), ("r4", "pasttense"), ("r5", "pasttense"))}


class FakeSacct:
    """(cluster, job id) -> the funnel reader's rows [{raw, task, state, terminal, end}]."""
    def __init__(self, states: dict):
        self.states = states; self.calls = []
    def __call__(self, cluster, job_id):
        self.calls.append((cluster, str(job_id)))
        v = self.states.get((cluster, str(job_id)))
        if v == "UNREACHABLE":
            raise RuntimeError("ssh: Connection timed out")
        if v is None:
            return []
        st, end = v
        return [{"raw": str(job_id), "task": None, "state": st, "terminal": st not in ST.SACCT_ACTIVE, "end": end}]


def _world(tmp: Path) -> dict:
    state = tmp / "state"; (state / "gate").mkdir(parents=True); ledger = str(state / "submissions.jsonl")
    ST.write_atomic(str(state / "gate" / "expected_units.json"), {"schema": 2, "model": KEY, "view": "u0", "code_sha": "a" * 40, "units": []})
    L = GL.Ledger(str(state / "attempts.jsonl"))
    # LW1: attempt 1 (job 1, FAILED) registered r1 + r2; attempt 2 (job 2) registered r2 only -> r1 is the orphan, r2 continued
    L.append({"event": "attempt", "logical_work_id": LW1, "attempt_id": f"{LW1}@cluster_a@1", "cluster": "cluster_a", "job_id": "1", "sha": "a" * 40, "utc": "2026-01-01T00:00:00Z"})
    L.append({"event": "superseded_attempt", "logical_work_id": LW1, "attempt_id": f"{LW1}@cluster_a@1", "job_id": "1", "cluster": "cluster_a", "state": "FAILED", "binding_state": "registered",
              "run_ids": [R["r1"], R["r2"]], "why": "registers=registry: the job died unregistered-complete", "utc": "2026-01-01T01:00:00Z"})
    # 29-fix-2: what the funnel leaves of a superseded attempt — its registered binding RETIRED, carrying the sbatch array spec (None: a plain job)
    # and the per-run facts from the job's pending file; array-ness is read from it, never from sacct's rows
    ST.write_binding(ledger, {"logical_work_id": LW1, "state": "registered", "run_ids": [R["r1"], R["r2"]], "job_id": "1", "cluster": "cluster_a", "attempt_id": f"{LW1}@cluster_a@1",
                              "array": None, "runs": [{"run_id": R["r1"], "eval_id": "mmlu", "object": "none"}, {"run_id": R["r2"], "eval_id": "mmlu", "object": "none"}], "utc": "2026-01-01T00:30:00Z"})
    assert ST.retire_registered_binding(ledger, LW1, f"{LW1}@cluster_a@1")
    L.append({"event": "attempt", "logical_work_id": LW1, "attempt_id": f"{LW1}@cluster_a@2", "cluster": "cluster_a", "job_id": "2", "sha": "a" * 40, "utc": "2026-01-01T01:00:01Z"})
    ST.write_binding(ledger, {"logical_work_id": LW1, "state": "registered", "run_ids": [R["r2"]], "job_id": "2", "cluster": "cluster_a", "attempt_id": f"{LW1}@cluster_a@2", "utc": "2026-01-01T01:10:00Z"})
    # LW2: attempt 1 (job 3, CANCELLED) registered r3; attempt 2 (job 4) is live and not registered -> pending, nothing decided
    L.append({"event": "superseded_attempt", "logical_work_id": LW2, "attempt_id": f"{LW2}@cluster_b@1", "job_id": "3", "cluster": "cluster_b", "state": "CANCELLED", "binding_state": "registered",
              "run_ids": [R["r3"]], "why": "cancelled by the STOP sentinel", "utc": "2026-01-02T00:00:00Z"})
    ST.write_binding(ledger, {"logical_work_id": LW2, "state": "registered", "run_ids": [R["r3"]], "job_id": "3", "cluster": "cluster_b", "attempt_id": f"{LW2}@cluster_b@1",
                              "array": None, "runs": [{"run_id": R["r3"], "eval_id": "gsm8k", "object": "none"}], "utc": "2026-01-01T23:30:00Z"})
    assert ST.retire_registered_binding(ledger, LW2, f"{LW2}@cluster_b@1")
    ST.write_binding(ledger, ST.submitted_record(LW2, "4", "cluster_b", f"{LW2}@cluster_b@2", "g8k", array=None))
    # LW3: attempt 1 (job 5, TIMEOUT) registered r4 (already ingested in the registry) + r5 (never reached the registry); attempt 2 registered nothing of them
    L.append({"event": "superseded_attempt", "logical_work_id": LW3, "attempt_id": f"{LW3}@cluster_a@1", "job_id": "5", "cluster": "cluster_a", "state": "TIMEOUT", "binding_state": "registered",
              "run_ids": [R["r4"], R["r5"]], "why": "walltime", "utc": "2026-01-03T00:00:00Z"})
    ST.write_binding(ledger, {"logical_work_id": LW3, "state": "registered", "run_ids": [R["r4"], R["r5"]], "job_id": "5", "cluster": "cluster_a", "attempt_id": f"{LW3}@cluster_a@1",
                              "array": None, "runs": [{"run_id": R["r4"], "eval_id": "pasttense", "object": "none"}, {"run_id": R["r5"], "eval_id": "pasttense", "object": "none"}], "utc": "2026-01-02T23:30:00Z"})
    assert ST.retire_registered_binding(ledger, LW3, f"{LW3}@cluster_a@1")
    ST.write_binding(ledger, {"logical_work_id": LW3, "state": "registered", "run_ids": ["other-run"], "job_id": "6", "cluster": "cluster_a", "attempt_id": f"{LW3}@cluster_a@2", "utc": "2026-01-03T01:00:00Z"})
    db = tmp / "registry.sqlite"; c = sqlite3.connect(db)
    c.execute("CREATE TABLE run (run_id TEXT PRIMARY KEY, lifecycle_state TEXT, campaign TEXT, cluster_job_id TEXT)")
    c.executemany("INSERT INTO run VALUES (?, ?, ?, ?)", [(R["r1"], "running", "sdfbase_full22", "1"), (R["r2"], "running", "sdfbase_full22", "2"), (R["r3"], "running", "sdfbase_full22", "3"),
                                                          (R["r4"], "ingested", "sdfbase_full22", "5")])
    c.commit(); c.close()
    spool = tmp / "spool"; spool.mkdir()
    return {"state": state, "ledger": ledger, "db": str(db), "spool": str(spool)}


PRODUCER = {"script": "reproduce/lib/stage.py reconcile-registrations", "code_sha": "b" * 40, "session": None, "host": "TESTHOST", "substrate": "grading_host"}


def test_only_the_orphaned_runs_of_dead_attempts_are_planned_and_a_write_emits_terminal_run_state_records_once(tmp_path):
    w = _world(tmp_path)
    sacct = FakeSacct({("cluster_a", "1"): ("FAILED", "2026-01-01T00:50:00"), ("cluster_b", "3"): ("CANCELLED by 1", "2026-01-02T00:00:30"), ("cluster_a", "5"): ("TIMEOUT", "2026-01-03T00:00:00")})
    rep = ST.reconcile_registrations(w["ledger"], w["db"], sacct, spool=w["spool"], write=False, producer=PRODUCER)
    assert rep["n_superseded_runs"] == 3 and [x["run_id"] for x in rep["plan"]] == [R["r1"]] and rep["written"] == []            # r2 continued; r3 pending; r4 past running; r5 absent
    assert rep["pending_units"] == {LW2: "its current attempt (job 4 on cluster_b) is submitted — not registered yet; its generation may re-register the same run ids"}
    assert rep["skipped"]["past_running"] == [{"run_id": R["r4"], "lifecycle_state": "ingested"}] and rep["skipped"]["not_in_registry"] == [R["r5"]]
    line = rep["plan"][0]
    assert line["state"] == "failed" and line["sacct_states"] == ["FAILED"] and line["event_utc"] == "2026-01-01T00:50:00.000Z" and line["campaign"] == "sdfbase_full22" \
        and line["evidence_uri"].endswith("/attempts.jsonl#2026-01-01T01:00:00Z") and json.loads(line["note"])["job_id"] == "1" and line["registry_state"] == "running"
    assert not list(Path(w["spool"]).rglob("*.jsonl")) and not (w["state"] / "gate" / ST.RECONCILED_FILE).exists()                  # the dry run wrote nothing
    # --write: one record file, the producers' layout, the registry owner's shape; the receipt makes a second pass write nothing
    rep2 = ST.reconcile_registrations(w["ledger"], w["db"], sacct, spool=w["spool"], write=True, producer=PRODUCER)
    assert [x["run_id"] for x in rep2["written"]] == [R["r1"]]
    files = list(Path(w["spool"]).rglob("*run_state*.jsonl")); assert len(files) == 1 and files[0].parent.parent.name == "sdfbase_full22", files
    rec = json.loads(files[0].read_text().splitlines()[0]); payload = rec["payload"] if "payload" in rec else rec
    assert rec.get("record_type", rec.get("type")) == "run_state"
    assert payload["run_id"] == R["r1"] and payload["state"] == "failed" and payload["event_utc"] == "2026-01-01T00:50:00.000Z" and payload["evidence_kind"] == "scheduler_state" \
        and payload["scheduler_state"] == "FAILED" and payload["evidence_uri"].endswith("#2026-01-01T01:00:00Z") and json.loads(payload["note"])["attempt_id"] == f"{LW1}@cluster_a@1"
    assert rec["producer"]["script"].endswith("reconcile-registrations") and rec["producer"]["substrate"] == "grading_host"
    receipt = [json.loads(l) for l in (w["state"] / "gate" / ST.RECONCILED_FILE).read_text().splitlines()]
    assert [r["run_id"] for r in receipt] == [R["r1"]] and receipt[0]["state"] == "failed" and receipt[0]["record_path"] == str(files[0])
    rep3 = ST.reconcile_registrations(w["ledger"], w["db"], sacct, spool=w["spool"], write=True, producer=PRODUCER)
    assert rep3["written"] == [] and rep3["skipped"]["already_reconciled"] == [R["r1"]] and len(list(Path(w["spool"]).rglob("*.jsonl"))) == 1
    # the pending unit's attempt registers and its dead attempt's run turns out NOT continued: now cancelled (CANCELLED on sacct, End as event_utc)
    ST.write_binding(w["ledger"], {"logical_work_id": LW2, "state": "registered", "run_ids": ["new-run"], "job_id": "4", "cluster": "cluster_b", "attempt_id": f"{LW2}@cluster_b@2", "utc": "2026-01-02T02:00:00Z"})
    rep4 = ST.reconcile_registrations(w["ledger"], w["db"], sacct, spool=w["spool"], write=True, producer=PRODUCER, units=[LW2])
    assert [x["run_id"] for x in rep4["written"]] == [R["r3"]] and rep4["written"][0]["state"] == "cancelled" and rep4["written"][0]["event_utc"] == "2026-01-02T00:00:30.000Z" \
        and rep4["written"][0]["sacct_states"] == ["CANCELLED by 1"]


def test_a_live_or_unproven_dead_job_decides_nothing_and_a_missing_registry_or_spool_refuses(tmp_path):
    w = _world(tmp_path)
    rep = ST.reconcile_registrations(w["ledger"], w["db"], FakeSacct({("cluster_a", "1"): ("RUNNING", None), ("cluster_a", "5"): "UNREACHABLE"}), spool=w["spool"], write=True, producer=PRODUCER)
    assert rep["written"] == [] and rep["skipped"]["job_alive"] == [{"run_id": R["r1"], "job_id": "1", "cluster": "cluster_a", "states": ["RUNNING"]}]
    rep = ST.reconcile_registrations(w["ledger"], w["db"], FakeSacct({}), spool=w["spool"], write=True, producer=PRODUCER, units=[LW1])
    assert rep["written"] == [] and rep["skipped"]["job_unproven"][0]["why"] == "job unknown to sacct"
    import pytest
    with pytest.raises(ValueError, match="does not exist"):
        ST.reconcile_registrations(w["ledger"], str(tmp_path / "absent.sqlite"), FakeSacct({}), spool=w["spool"])
    with pytest.raises(ValueError, match="no spool"):
        ST.reconcile_registrations(w["ledger"], w["db"], FakeSacct({("cluster_a", "1"): ("FAILED", "2026-01-01T00:50:00")}), spool=None, write=True, producer=PRODUCER)
    assert ST._utc_ms("2026-10-01T01:19:53") == "2026-10-01T01:19:53.000Z" and ST._utc_ms("2026-10-01T01:19:53.5Z") == "2026-10-01T01:19:53.500Z" and ST._utc_ms("Unknown") is None


def test_the_cli_reads_sacct_through_the_funnel_plans_by_default_and_writes_with_write(tmp_path):
    w = _world(tmp_path)
    fake = {lab: tmp_path / f"fake_{lab}" for lab in ("cluster_a", "cluster_b")}
    for d in fake.values():
        (d / "jobs").mkdir(parents=True)
    bin_dir = write_fake_slurm(tmp_path / "bin")
    site = write_site(tmp_path, ssh={lab: f"local:{d}" for lab, d in fake.items()})
    env = dict(os.environ, PATH=f"{bin_dir}:{os.environ['PATH']}", PYTHONPATH=str(REPO / "src"), SISTEER_REGISTRY_SUBSTRATE="grading_host")
    for k in ("SISTEER_SITE_DIR", "SISTEER_CONFIG", "SISTEER_REGISTRY_SPOOL"):
        env.pop(k, None)
    set_job_state(fake["cluster_a"], "1", "FAILED", end="2026-01-01T00:50:00"); set_job_state(fake["cluster_b"], "3", "CANCELLED", end="2026-01-02T00:00:30"); set_job_state(fake["cluster_a"], "5", "TIMEOUT", end="2026-01-03T00:00:00")
    base = [VENV_PY or sys.executable, str(LIB / "stage.py"), "reconcile-registrations", "--ledger", w["ledger"], "--registry", w["db"], "--config", str(site / "config.env"),
            "--spool", w["spool"], "--repo", str(REPO), "--out", str(tmp_path / "report.json")]
    r = subprocess.run(base, capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and f"would write run_state failed for {R['r1']}" in r.stdout and f"pending {LW2}:" in r.stdout and "1 to write" in r.stdout, r.stdout + r.stderr
    assert json.load(open(tmp_path / "report.json"))["plan"][0]["run_id"] == R["r1"] and not list(Path(w["spool"]).rglob("*.jsonl"))
    r = subprocess.run([*base, "--write"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and f"wrote run_state failed for {R['r1']}" in r.stdout and "1 written (1 record file(s)" in r.stdout, r.stdout + r.stderr
    assert len(list(Path(w["spool"]).rglob("*run_state*.jsonl"))) == 1
    r = subprocess.run([*base, "--write"], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0 and "0 written" in r.stdout and "already_reconciled 1" in r.stdout, r.stdout + r.stderr
    r = subprocess.run([*base[:5], "--registry", str(tmp_path / "absent.sqlite"), "--config", str(site / "config.env")], capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 2 and "REFUSED: reconcile-registrations:" in r.stderr and "does not exist" in r.stderr, r.stderr
