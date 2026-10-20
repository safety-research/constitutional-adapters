"""Submit-time registration: ``sbatch_registered.sh`` with a fake ``sbatch`` on PATH
(prints ``Submitted batch job 4242``) writes the queued records + the job binding + the pending file;
a plan whose instrument fields are not resolved REFUSES TO SUBMIT (sbatch never runs);
``registry register-run`` is strict the same way; the stdlib ``emit_cli`` ``state`` command attaches a
job's ``running`` / terminal states through the pending file, registers late from a plan when there is no
pending file, and never exits non-zero; ``push`` resolves the run directory from the pending file.
"""
from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

from sisteer.registry import cli, emit, emit_cli, identity
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry import validate as VAL

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "scripts" / "registry" / "sbatch_registered.sh"


def _plan(**over):
    p = {k: v for k, v in synth.run_payload("wrap_key", i=3).items() if k != "run_id"}
    p.update(over)
    return p


def _fake_sbatch(tmp_path, job="4242", fail=False):
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    log = tmp_path / "sbatch_calls.log"
    body = f"#!/bin/sh\necho \"$@\" >> {log}\n" + (f"echo 'sbatch: error: synthetic' >&2\nexit 1\n" if fail else f"echo 'Submitted batch job {job}'\n")
    (bindir / "sbatch").write_text(body)
    (bindir / "sbatch").chmod(0o755)
    return bindir, log


def _env(tmp_path, bindir):
    env = {k: v for k, v in os.environ.items() if k not in (emit.SPOOL_ENV, emit.PUSH_SCRIPT_ENV, emit.DISABLED_ENV, "REGISTRY_EMIT")}
    env["PATH"] = f"{bindir}:{env.get('PATH', '')}"
    env["HOME"] = str(tmp_path / "home")
    env["USER"] = "nobody-synthetic"
    return env


def _records(spool):
    return [(p, rec) for p in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(p)]


def test_sbatch_registered_writes_queued_records_binding_and_pending_file(tmp_path):
    bindir, log = _fake_sbatch(tmp_path)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(_plan()))
    spool = tmp_path / "spool"
    pending = tmp_path / "pending"
    out = subprocess.run([str(WRAPPER), "--planned", str(plan_file), "--spool", str(spool), "--pending-dir", str(pending), "--",
                          "sbatch", "--comment=sha:" + "a" * 40, "--job-name=synthjob", "job.sbatch", "1", "5"],
                         capture_output=True, text=True, env=_env(tmp_path, bindir), timeout=120)
    assert out.returncode == 0, out.stderr
    assert "Submitted batch job 4242" in out.stdout and "job 4242 registered" in out.stdout
    assert log.read_text().strip() == "--comment=sha:" + "a" * 40 + " --job-name=synthjob job.sbatch 1 5"
    rid = identity.run_id(synth.CAMPAIGN, "wrap_key", "none", synth.EVAL_ID, synth.INSTRUMENT_ID)
    recs = _records(spool)
    assert [r["record_type"] for _, r in recs] == ["run", "run_state", "run_patch"]
    run, state, patch = (r["payload"] for _, r in recs)
    assert run["run_id"] == rid and run["lifecycle_state"] == "queued" and run["submit_sha_pin"] == "a" * 40
    assert state["state"] == "queued" and state["evidence_kind"] == "submit_wrapper" and state["evidence_uri"] == "slurm://job/4242"
    assert patch["fields"] == {"cluster_job_id": "4242", "job_name": "synthjob"}
    for p, rec in recs:
        assert isinstance(VAL.validate_record(rec, str(p)), VAL.Validated)
    pend = json.loads((pending / "4242.json").read_text())
    assert pend["run_id"] == rid and pend["plan"]["run_id"] == rid and pend["plan"]["eval_id"] == synth.EVAL_ID and pend["job_name"] == "synthjob"


def test_sbatch_registered_refuses_to_submit_an_unresolved_plan(tmp_path):
    """gsm8k without a resolved item set / instrument does not launch, and the wrapper says what it needs."""
    bindir, log = _fake_sbatch(tmp_path)
    plan = _plan(eval_id="gsm8k")
    plan.pop("item_set_id")
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(plan))
    out = subprocess.run([str(WRAPPER), "--planned", str(plan_file), "--spool", str(tmp_path / "spool"), "--", "sbatch", "job.sbatch"],
                         capture_output=True, text=True, env=_env(tmp_path, bindir), timeout=120)
    assert out.returncode == 2
    assert "REFUSING TO SUBMIT" in out.stderr and "item_set_id" in out.stderr and "gsm8k" in out.stderr
    assert not log.exists()                       # sbatch never ran
    assert not (tmp_path / "spool").exists()      # nothing written
    # --check-only on a good plan: OK, nothing submitted
    plan_file.write_text(json.dumps(_plan()))
    out = subprocess.run([str(WRAPPER), "--planned", str(plan_file), "--check-only", "--", "sbatch", "job.sbatch"],
                         capture_output=True, text=True, env=_env(tmp_path, bindir), timeout=120)
    assert out.returncode == 0 and "check only" in out.stdout and not log.exists()


def test_sbatch_registered_passes_sbatch_failure_through_and_never_registers(tmp_path):
    bindir, log = _fake_sbatch(tmp_path, fail=True)
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(_plan()))
    out = subprocess.run([str(WRAPPER), "--planned", str(plan_file), "--spool", str(tmp_path / "spool"), "--", "sbatch", "job.sbatch"],
                         capture_output=True, text=True, env=_env(tmp_path, bindir), timeout=120)
    assert out.returncode == 1 and "nothing registered" in out.stderr and log.exists()
    assert not (tmp_path / "spool").exists()


def test_register_run_cli_is_strict_and_binds_the_job(tmp_path, capsys, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)  # the suite runs with emission off (tests/conftest.py); this test writes to its own tmp spool, so it opts in
    plan = _plan(eval_id="gsm8k")
    plan.pop("instrument_id")
    pf = tmp_path / "plan.json"
    pf.write_text(json.dumps(plan))
    assert cli.main(["register-run", str(pf), "--spool", str(tmp_path / "spool")]) == 2
    err = capsys.readouterr().err
    assert "REFUSED" in err and "instrument_id" in err and "gsm8k" in err
    assert not (tmp_path / "spool").exists()
    pf.write_text(json.dumps(_plan()))
    assert cli.main(["register-run", str(pf), "--spool", str(tmp_path / "spool"), "--job-id", "77", "--pending-dir", str(tmp_path / "pend"), "--no-push"]) == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 3 and out[0].endswith("000001-run.jsonl") and out[2].endswith("000003-run_patch.jsonl")
    assert (tmp_path / "pend" / "77.json").exists()


def test_emit_cli_state_attaches_through_the_pending_file_and_registers_late(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    spool = str(tmp_path / "spool")
    pend_dir = tmp_path / "pending"
    # no pending file, no plan: a notice, exit 0, nothing written
    assert emit_cli.main(["state", "--state", "running", "--pending-dir", str(pend_dir), "--job", "1", "--spool", spool]) == 0
    assert "no run to attach to" in capsys.readouterr().err and not os.path.exists(spool)
    # late registration from a plan (the job ran without a wrapper): run (running) + running state, registered_at=job_start
    pf = tmp_path / "plan.json"
    pf.write_text(json.dumps(_plan()))
    assert emit_cli.main(["state", "--state", "running", "--late-register", str(pf), "--node", "TESTHOST", "--gpu-count", "2", "--gpu-type", "H200",
                          "--scheduler-state", "RUNNING", "--spool", spool]) == 0
    recs = _records(spool)
    assert [r["record_type"] for _, r in recs] == ["run", "run_state", "run_patch"]
    run = recs[0][1]["payload"]
    assert run["lifecycle_state"] == "running" and run["extra"]["registered_at"] == "job_start" and run["queued_utc"] == run["started_utc"]
    assert recs[2][1]["payload"]["fields"] == {"node_hostname": "TESTHOST", "gpu_count": 2, "gpu_type": "H200"}
    for p, rec in recs:
        assert isinstance(VAL.validate_record(rec, str(p)), VAL.Validated), rec
    # through a pending file: the run id is the file's, a terminal state carries the scheduler state
    rid = run["run_id"]
    pend_dir.mkdir()
    (pend_dir / "9.json").write_text(json.dumps({"run_id": rid, "job_id": "9", "plan": dict(_plan(), run_id=rid)}))
    assert emit_cli.main(["state", "--state", "failed", "--pending-dir", str(pend_dir), "--job", "9", "--scheduler-state", "FAILED", "--final", "--spool", spool]) == 0
    last = _records(spool)[-1][1]["payload"]
    assert last["state"] == "failed" and last["scheduler_state"] == "FAILED" and last["evidence_kind"] == "scheduler_state" and last["run_id"] == rid
    # an unresolved late plan is refused, exit 0 (the job continues), nothing written
    bad = _plan(eval_id="gsm8k")
    bad.pop("item_set_id")
    pf.write_text(json.dumps(bad))
    n_before = len(_records(spool))
    assert emit_cli.main(["state", "--state", "running", "--late-register", str(pf), "--spool", spool]) == 0
    assert "late registration refused" in capsys.readouterr().err and len(_records(spool)) == n_before


def test_emit_cli_push_resolves_the_run_dir_from_the_pending_file(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    calls = tmp_path / "calls.log"
    script = tmp_path / "push.sh"
    script.write_text(f"#!/bin/sh\necho \"$@\" >> {calls}\nexit 0\n")
    script.chmod(0o755)
    monkeypatch.setenv(emit.PUSH_SCRIPT_ENV, str(script))
    spool = str(tmp_path / "spool")
    rid = emit.check_planned_run(_plan())
    paths = emit.write_planned_run(spool, _plan(), script="t")
    pend_dir = tmp_path / "pending"
    pend_dir.mkdir()
    (pend_dir / "5.json").write_text(json.dumps({"run_id": rid, "plan": dict(_plan(), run_id=rid)}))
    assert emit_cli.main(["push", "--pending-dir", str(pend_dir), "--job", "5", "--spool", spool, "--final"]) == 0
    assert calls.read_text().strip() == f"{os.path.dirname(paths[0])} --final"
    # no pending file → quiet exit 0, no call
    assert emit_cli.main(["push", "--pending-dir", str(pend_dir), "--job", "6", "--spool", spool]) == 0
    assert len(calls.read_text().strip().splitlines()) == 1


def test_wrapper_and_install_scripts_are_executable_and_parse():
    for name in ("sbatch_registered.sh", "install_cluster.sh", "inbox_to_log.sh", "push_records.sh", "registry_env.sh"):
        p = REPO / "scripts" / "registry" / name
        assert p.stat().st_mode & stat.S_IXUSR, name
        subprocess.run(["bash", "-n", str(p)], check=True)
    s = WRAPPER.read_text()
    assert "--export=" not in s.replace("Never `--export=`", "")
