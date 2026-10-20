"""The job shells' registry lines: ``stage2_common.sh`` and ``full22/common.sh`` carry
``running`` after the SHA gate / before ``uv sync`` and the terminal state + final push in their finish
paths, never via ``--export=``; sourced in a stub environment (a fake ``$SHARED`` with a pending file, a
tmp spool, no scheduler) the helper functions emit the ``running`` record from the job's own environment
and the terminal records from the trap outcome, exit 0 whatever happens, and stay silent when no spool is
provisioned. Text checks + ``bash -n``, the way ``tests/test_full22_cap4096_sbatch.py`` pins job text.
"""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from sisteer.registry import records as R
from sisteer.registry import synth

REPO = Path(__file__).resolve().parents[1]
S2 = REPO / "scripts" / "sbatch" / "stage2_common.sh"
F22 = REPO / "scripts" / "sdfbase" / "full22" / "common.sh"
ENV_SH = REPO / "scripts" / "registry" / "registry_env.sh"


def test_shell_syntax():
    for f in (S2, F22, ENV_SH):
        subprocess.run(["bash", "-n", str(f)], check=True)


def test_stage2_common_text():
    s = S2.read_text()
    assert "s2_registry_state() {" in s
    # running: after the SHA gate (SUBMIT_SHA parsed, HEAD compared) and BEFORE uv sync
    i_gate = s.index('SUBMIT_SHA="${pin_lines#sha:}"')
    i_run = s.index('s2_registry_state running "${SLURM_JOB_STATE:-RUNNING}" || true')
    i_sync = s.index("uv sync --frozen --extra cu128")
    assert i_gate < i_run < i_sync
    # finish: FAILED → failed, PREEMPTED → cancelled, else a final push; every line best-effort
    fin = s[s.index("s2_finish() {"):s.index("s2_escalate() {")]
    assert 's2_registry_state failed "$state" final' in fin and 's2_registry_state cancelled "$state" final' in fin
    assert "emit_cli push" in fin and "--final" in fin
    assert "python3 -S -m sisteer.registry.emit_cli" in s
    assert "--export=" not in s.replace("never sbatch --export", "").replace("sbatch --export", "").replace("--export on this cluster", "")
    helper = s[s.index("s2_registry_state() {"):s.index("s2_init() {")]
    assert "|| true" in helper and 'return 0' in helper and "SISTEER_REGISTRY_RUN_IDENTITY" in helper


def test_full22_common_text():
    s = F22.read_text()
    assert "f22_registry_state(){" in s and "f22_registry_result(){" in s
    init = s[s.index("f22_init(){"):s.index("f22_stop_requested(){")]
    assert 'f22_registry_state running "${SLURM_JOB_STATE:-RUNNING}" || true' in init
    assert 'f22_result(){' in s and 'f22_registry_result "$@"' in s
    res = s[s.index("f22_registry_result(){"):]
    assert "failed" in res and "cancelled" in res and "emit_cli push" in res


def _stub_env(tmp_path, with_spool=True):
    """A fake cluster: $SHARED with the pending file and a spool, the registry env script installed, no slurm."""
    shared = tmp_path / "shared"
    (shared / "scripts" / "registry").mkdir(parents=True)
    (shared / "registry" / "pending").mkdir(parents=True)
    if with_spool:
        (shared / "registry" / "records").mkdir(parents=True)
    # registry_env.sh as installed on a cluster; REGISTRY_SHARED_ROOT is not a knob it reads, so hand the spool over directly
    env_sh = shared / "scripts" / "registry" / "registry_env.sh"
    env_sh.write_text("#!/usr/bin/env bash\n" + (f"export SISTEER_REGISTRY_SPOOL={shared}/registry/records\n" if with_spool else ""))
    plan = {k: v for k, v in synth.run_payload("s2_key", i=1).items()}
    rid = plan["run_id"]
    (shared / "registry" / "pending" / "4242.json").write_text(json.dumps({"run_id": rid, "job_id": "4242", "plan": plan}))
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path / "home"), "USER": "nobody-synthetic", "SHARED": str(shared),
           "S2_REPO_DIR": str(REPO), "REPO_DIR": str(REPO), "SLURM_JOB_ID": "4242", "SLURMD_NODENAME": "TESTHOST",
           "SLURM_JOB_NAME": "synthjob", "CUDA_VISIBLE_DEVICES": "0,1"}
    return shared, rid, env


def _records(spool):
    return [(p, rec) for p in R.iter_record_files(str(spool)) for _, rec, _ in R.read_record_file(p)]


def test_s2_registry_state_emits_running_from_the_job_environment(tmp_path):
    shared, rid, env = _stub_env(tmp_path)
    script = f"""
set -u
S2_REPO_DIR="{REPO}"
source {S2}
s2_registry_state running RUNNING
echo "rc=$?"
echo "ident=${{SISTEER_REGISTRY_RUN_IDENTITY:-unset}}"
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "rc=0" in out.stdout and f"ident={shared}/registry/pending/4242.json" in out.stdout
    recs = _records(shared / "registry" / "records")
    assert [r["record_type"] for _, r in recs] == ["run_state", "run_patch"]
    st, patch = recs[0][1]["payload"], recs[1][1]["payload"]
    assert st["run_id"] == rid and st["state"] == "running" and st["scheduler_state"] == "RUNNING" and st["evidence_kind"] == "job_log"
    assert patch["fields"]["node_hostname"] == "TESTHOST" and patch["fields"]["cluster_job_id"] == "4242" and patch["fields"]["gpu_count"] == 2
    assert patch["fields"]["job_name"] == "synthjob"
    assert recs[0][1]["producer"]["script"] == "sisteer.registry.emit_cli state"


def test_s2_finish_paths_emit_terminal_states_and_never_fail(tmp_path):
    shared, rid, env = _stub_env(tmp_path)
    script = f"""
set -u
S2_REPO_DIR="{REPO}"
source {S2}
s2_registry_state failed FAILED final; echo "rc1=$?"
s2_registry_state cancelled PREEMPTED final; echo "rc2=$?"
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "rc1=0" in out.stdout and "rc2=0" in out.stdout
    states = [(r["payload"]["state"], r["payload"].get("scheduler_state")) for _, r in _records(shared / "registry" / "records") if r["record_type"] == "run_state"]
    assert states == [("failed", "FAILED"), ("cancelled", "PREEMPTED")]


def test_registry_lines_are_silent_without_a_spool_and_survive_a_missing_pending_file(tmp_path):
    shared, rid, env = _stub_env(tmp_path, with_spool=False)
    script = f"""
set -u
S2_REPO_DIR="{REPO}"
source {S2}
s2_registry_state running RUNNING; echo "rc=$?"
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0 and "rc=0" in out.stdout and not (shared / "registry" / "records").exists()
    # spool present, pending file for another job: exit 0, one notice, nothing written
    shared, rid, env = _stub_env(tmp_path / "b")
    env["SLURM_JOB_ID"] = "9999"
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0 and "rc=0" in out.stdout and "no run to attach to" in out.stdout + out.stderr
    assert not _records(shared / "registry" / "records")


def test_f22_registry_result_maps_the_result_line(tmp_path):
    shared, rid, env = _stub_env(tmp_path)
    script = f"""
set -u
REPO_DIR="{REPO}"
source {F22}
f22_registry_state running RUNNING; echo "rc0=$?"
f22_registry_result "FAILED exit=1"; echo "rc1=$?"
f22_registry_result "PREEMPTED"; echo "rc2=$?"
f22_registry_result "COMPLETED exit=0"; echo "rc3=$?"
"""
    out = subprocess.run(["bash", "-c", script], capture_output=True, text=True, env=env, timeout=180)
    assert out.returncode == 0, out.stderr
    # the clean branch runs the store closer, which REFUSES this synthetic pending file (it names no `runs`) — the
    # function returns the closer's rc (f22_result turns it into the job's exit 75). On the `f22_registry_closer_refused` path, the
    # post-refusal `running` statement goes ONLY to the runs the closer refused, read from its <pending>.refused_run_ids.json sidecar — never to
    # the job's whole run set (888525's exit path restated `running` over twins its closer had just superseded). A refusal that names no runs
    # writes no sidecar, so nothing is stated and the shell says so by name: THREE run_state records, not a fourth job-wide `running`.
    assert all(f"rc{i}=0" in out.stdout for i in range(3)) and "rc3=2" in out.stdout, out.stdout + out.stderr
    assert "[closer] REFUSED rc=2" in out.stderr and "names no `runs`" in out.stdout, out.stdout + out.stderr
    assert "no refused-run list at" in out.stdout and "stating nothing (D2100 (b)" in out.stdout, out.stdout + out.stderr
    assert not list((shared / "registry" / "pending").glob("*.refused_run_ids.json")), sorted(p.name for p in (shared / "registry" / "pending").glob("*"))
    states = [(r["payload"]["state"], r["payload"].get("scheduler_state")) for _, r in _records(shared / "registry" / "records") if r["record_type"] == "run_state"]
    assert states == [("running", "RUNNING"), ("failed", "FAILED"), ("cancelled", "PREEMPTED")], states
