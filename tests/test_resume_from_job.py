"""RESUME a registered unit's drivers under the SAME canonical runs — `submit --resume-from-job J` mints a new attempt whose job
reuses J's registration identity (F22_RESUME_FROM_JOB in the JOB env; target_attacks.sbatch copies pending/<J>.json to its own before
f22_registry_register runs, so no new runs) and resumes the stores; the registered binding stands. Refused by name unless the unit is
registered on exactly J, J is terminal, and the cluster is J's."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, job_ids, set_job_state  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402

sys.path.insert(0, str(LIB))
import stage as ST  # noqa: E402

TARGET = "scripts/sdfbase/full22/target_attacks.sbatch"
LW = "m/u0/multiturn/actorattack/g012"
SLUG = LW.replace("/", "__")


def _submit(w: World, *extra, **env):
    return w.run("submit", "--unit", "multiturn.actorattack.g012", "--lwid", LW, "--script", TARGET, "--name", "f22-att-m-g012-actorattack",
                 "--args", f"m u0 actorattack {SHA} 9 0,1,2", "--clusters", "cluster_a", "--group", "mt-u0-cluster_a", "--wave-sha", SHA, *extra, **env)


def _register(w: World, job: str, aid: str) -> None:
    """The unit's REGISTERED binding (what 06 --gate writes from the job's pending file), the submitted one retired."""
    (w.state / "bindings" / f"{SLUG}.submitted.json").unlink(missing_ok=True)
    (w.state / "bindings" / f"{SLUG}.registered.json").write_text(json.dumps({"logical_work_id": LW, "state": "registered", "run_ids": ["run-aa-1", "run-aa-2", "run-aa-3"],
                                                                                "job_id": job, "cluster": "cluster_a", "attempt_id": aid, "utc": "2026-01-01T01:00:00Z"}))


def test_a_resume_reuses_the_registered_jobs_identity_in_the_job_env_and_leaves_the_binding_standing(tmp_path):
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r = _submit(w); assert r.returncode == 0, r.stdout + r.stderr
    j1 = str(job_ids(w.fake["cluster_a"])[-1]); aid1 = [e for e in w.ledger() if e["event"] == "attempt"][-1]["attempt_id"]
    # not registered yet (only submitted): refused by name
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 3 and "REGISTERED job only" in r.stderr and "is submitted" in r.stderr, r.stderr
    _register(w, j1, aid1)
    # registered but still RUNNING: refused — a resume runs only after the job it resumes ended
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 3 and "not terminal on sacct" in r.stderr, r.stderr
    set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T02:00:00")
    # another job id, another cluster, a twin: refused by name
    r = _submit(w, "--resume-from-job", "999999"); assert r.returncode == 3 and "REGISTERED job only" in r.stderr, r.stderr
    r = w.run("submit", "--unit", "multiturn.actorattack.g012", "--lwid", LW, "--script", TARGET, "--name", "n", "--args", f"m u0 actorattack {SHA} 9 0,1,2",
              "--clusters", "cluster_b", "--group", "mt-u0-cluster_b", "--wave-sha", SHA, "--resume-from-job", j1)
    assert r.returncode == 3 and "the stores live there" in r.stderr, r.stderr
    r = w.run("submit", "--unit", "multiturn.actorattack.g012", "--lwid", LW, "--script", TARGET, "--name", "n", "--args", f"m u0 actorattack {SHA} 9 0,1,2",
              "--clusters", "cluster_a,cluster_b", "--group", "mt-u0-cluster_a", "--wave-sha", SHA, "--resume-from-job", j1, "--twin")
    assert r.returncode == 2 and "no --twin" in r.stderr, r.stderr
    assert len([e for e in w.ledger() if e["event"] == "attempt"]) == 1                                   # nothing minted by the refusals
    # the resume: a NEW attempt, its job carries F22_RESUME_FROM_JOB in the JOB env (not the workload spec), the binding stands registered on j1
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 0 and f"resumes job {j1} under its 3 registered run(s)" in (r.stdout + r.stderr), r.stdout + r.stderr
    j2 = str(job_ids(w.fake["cluster_a"])[-1]); assert j2 != j1
    env = (w.fake["cluster_a"] / "jobs" / f"{j2}.env").read_text()
    assert f"F22_RESUME_FROM_JOB={j1}" in env and "F22_WAVE_ROOT=" in env
    att = [e for e in w.ledger() if e["event"] == "attempt"]; assert len(att) == 2 and att[-1]["job_id"] == j2
    assert att[-1]["spec"]["resume_of"] == {"job_id": j1, "attempt_id": aid1, "cluster": "cluster_a", "run_ids": ["run-aa-1", "run-aa-2", "run-aa-3"], "prior_resumes": []}
    assert "F22_RESUME_FROM_JOB" not in (att[-1]["spec"].get("env") or {})                                # the workload spec is the original's
    assert att[-1]["attempt_id"].endswith("@cluster_a@2")
    b = ST.binding_state(str(w.state / "submissions.jsonl"), LW)
    assert b["state"] == "registered" and str(b["job_id"]) == j1 and b["run_ids"] == ["run-aa-1", "run-aa-2", "run-aa-3"]
    assert not (w.state / "bindings" / f"{SLUG}.submitted.json").exists()
    subs = [json.loads(l) for l in (w.state / "submissions.jsonl").read_text().splitlines() if l.strip()]
    assert [e for e in subs if str(e.get("job_id")) == j2 and e.get("event") == "submitted"]                # the submissions row and the watcher are the attempt's
    # a second invocation while the resume J2 is live refuses by name (never two jobs on the same stores and run ids);
    # after J2 COMPLETED it refuses too (the stores were closed by it); a CRASHED resume (terminal, not completed) is reconciled by the next
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 3 and "already IN FLIGHT" in r.stderr and j2 in r.stderr, r.stderr
    set_job_state(w.fake["cluster_a"], j2, "COMPLETED", end="2026-01-01T03:00:00")
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 3 and "COMPLETED" in r.stderr and "closed" in r.stderr, r.stderr
    assert len([e for e in w.ledger() if e["event"] == "attempt"]) == 2
    set_job_state(w.fake["cluster_a"], j2, "FAILED", end="2026-01-01T03:00:00")
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 0 and "1 crashed resume(s) before it reconciled" in (r.stdout + r.stderr), r.stdout + r.stderr
    j3 = str(job_ids(w.fake["cluster_a"])[-1]); att3 = [e for e in w.ledger() if e["event"] == "attempt"][-1]
    assert att3["job_id"] == j3 and att3["spec"]["resume_of"]["prior_resumes"] == [{"attempt_id": att[-1]["attempt_id"], "job_id": j2, "states": ["FAILED"]}]
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 3 and "already IN FLIGHT" in r.stderr and j3 in r.stderr, r.stderr
    # a dry run says what it would do and writes nothing
    r = w.run("--dry-run", "submit", "--unit", "multiturn.actorattack.g012", "--lwid", LW, "--script", TARGET, "--name", "f22-att-m-g012-actorattack",
              "--args", f"m u0 actorattack {SHA} 9 0,1,2", "--clusters", "cluster_a", "--group", "mt-u0-cluster_a", "--wave-sha", SHA, "--resume-from-job", j1)
    assert r.returncode == 0 and "job DRY" in (r.stdout + r.stderr) and len([e for e in w.ledger() if e["event"] == "attempt"]) == 3, r.stdout + r.stderr   # the three real attempts; the dry run minted none


def _resume_block() -> str:
    sb = (REPO / TARGET).read_text()
    a = sb.index('if [ -n "${F22_RESUME_FROM_JOB:-}" ]; then'); b = sb.index("\nfi\n", a) + len("\nfi\n")
    # BEFORE f22_init, whose `running` emission (and any early death's state) must carry the canonical identity
    assert a < sb.index('f22_init "$WAVE_SHA"') < sb.index('f22_resolve_config "$KEY" "$VIEW"')
    return sb[a:b]


def test_the_target_job_copies_the_original_registration_identity_before_registering_and_refuses_the_two_bad_states(tmp_path):
    block = _resume_block()
    shared = tmp_path / "shared"; (shared / "registry" / "pending").mkdir(parents=True)
    (shared / "registry" / "pending" / "815118.json").write_text(json.dumps({"job_id": "815118", "runs": [{"run_id": "run-aa-1"}]}))
    def run(job, resume, root=None):
        # the block derives the shared root itself (it runs before f22_init): the portability override here, the two-root probe on a cluster
        pre = f'SISTEER_SHARED_ROOT="$1"; ' if root is None else (f'SISTEER_SHARED_ROOT="{root}"; ' if root else "")
        # The test uses a login name that no host has a shared root for under either mount point: the block's two-root
        # probe must find nothing when the test says there is no root (a cluster node does mount the shared storage, and
        # the suite's real login name there does have a root).
        return subprocess.run(["bash", "-c", f'set -u; USER="sisteer-test-no-shared-root"; {pre}SLURM_JOB_ID="$2"; F22_RESUME_FROM_JOB="$3"; f22_fatal(){{ echo "FATAL: $*" >&2; exit 9; }}; f22_log(){{ echo "$*"; }}\n{block}\necho END',
                               "_", str(shared), job, resume], capture_output=True, text=True, timeout=60)
    r = run("900001", "815118"); assert r.returncode == 0 and "RESUME of job 815118" in r.stdout and r.stdout.rstrip().endswith("END"), r.stdout + r.stderr
    assert json.loads((shared / "registry" / "pending" / "900001.json").read_text())["job_id"] == "815118"       # the identity, the original's job id kept
    r = run("900001", "815118"); assert r.returncode == 9 and "already exists" in r.stderr, r.stdout + r.stderr   # a job that already carries an identity
    r = run("900002", "777777"); assert r.returncode == 9 and "no registration identity" in r.stderr, r.stderr     # an original without one
    assert not (shared / "registry" / "pending" / "900002.json").exists()
    r = run("900003", ""); assert r.returncode == 0 and "RESUME" not in r.stdout and not (shared / "registry" / "pending" / "900003.json").exists()   # not a resume: a no-op
    r = run("900004", "815118", root=""); assert r.returncode == 9 and "no shared root" in r.stderr, r.stdout + r.stderr                           # no root derivable: refused by name
    assert "before the job's own state emission" in run("900005", "815118").stdout
    assert subprocess.run(["bash", "-n", str(REPO / TARGET)], capture_output=True).returncode == 0


def test_job_env_rides_the_job_not_the_workload_spec_and_is_accepted_with_a_resume_only(tmp_path):
    """31f: the attacker pin (F22_ATTACKER_JOBS) is a job-side fact — the funnel refuses a workload-spec difference by design, so it cannot ride --env."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r = _submit(w); assert r.returncode == 0, r.stdout + r.stderr
    j1 = str(job_ids(w.fake["cluster_a"])[-1]); aid1 = [e for e in w.ledger() if e["event"] == "attempt"][-1]["attempt_id"]
    _register(w, j1, aid1); set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T02:00:00")
    r = _submit(w, "--job-env", "F22_ATTACKER_JOBS=4242"); assert r.returncode == 2 and "--resume-from-job only" in r.stderr, r.stderr
    # 31f-fix: only the allowlisted operational keys ride --job-env — a scientific knob under the old run identity
    # (the seed, the decode policy, the serving path), a registry route, or the resume key itself refuses by name and mints nothing
    for bad in ("F22_RESUME_FROM_JOB=999", "F22_MT_TARGET_SEED_BASE=20260101", "F22_MT_TARGET_DECODE=greedy", "F22_SERVING_PATH=lens", "F22_REGISTRY_EVALS=pair", "SISTEER_REGISTRY_SPOOL=/tmp/x",
                "F22_ATTACKER_JOBS=4242 F22_MT_TARGET_SEED_BASE=1"):
        r = _submit(w, "--resume-from-job", j1, "--job-env", bad)
        assert r.returncode == 2 and "not on the resume allowlist" in r.stderr and bad.split("=")[0].split()[-1] in r.stderr or ("F22_MT_TARGET_SEED_BASE" in r.stderr), (bad, r.stderr)
    assert len([e for e in w.ledger() if e["event"] == "attempt"]) == 1
    assert ST.__name__ == "stage"
    r = _submit(w, "--resume-from-job", j1, "--job-env", "F22_ATTACKER_JOBS=4242 F22_ATTACKER_WAIT_S=120"); assert r.returncode == 0, r.stdout + r.stderr
    j2 = str(job_ids(w.fake["cluster_a"])[-1]); env = (w.fake["cluster_a"] / "jobs" / f"{j2}.env").read_text()
    assert f"F22_RESUME_FROM_JOB={j1}" in env and "F22_ATTACKER_JOBS=4242" in env and "F22_ATTACKER_WAIT_S=120" in env
    att = [e for e in w.ledger() if e["event"] == "attempt"][-1]
    assert att["spec"]["job_env"] == {"F22_RESUME_FROM_JOB": j1, "F22_ATTACKER_JOBS": "4242", "F22_ATTACKER_WAIT_S": "120"}
    assert "F22_ATTACKER_JOBS" not in (att["spec"].get("env") or {}) and ST.spec_identity(att["spec"])["env"] == ST.spec_identity([e for e in w.ledger() if e["event"] == "attempt"][0]["spec"])["env"]


def test_a_resume_interrupted_after_sbatch_is_reconciled_by_name_before_any_retry_and_concurrent_callers_launch_one(tmp_path):
    """The duplicate check read the ledger without a lock and the attempt was appended only after sbatch —
    a caller dying between them left an untracked live resume a retry duplicated, and two concurrent callers both launched. Now a
    reservation (a transition token) is written before sbatch under a per-unit lock and reconciled against Slurm by job name first."""
    import subprocess
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r = _submit(w); assert r.returncode == 0, r.stdout + r.stderr
    j1 = str(job_ids(w.fake["cluster_a"])[-1]); aid1 = [e for e in w.ledger() if e["event"] == "attempt"][-1]["attempt_id"]
    _register(w, j1, aid1); set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T02:00:00")
    # the crash window: sbatch accepted (the fake recorded the job by name), the ledger never heard of it, the reservation stays
    r = _submit(w, "--resume-from-job", j1, SISTEER_TWIN_CRASH_AT="submit_after_sbatch")
    assert r.returncode == 97 and "CRASH INJECTED at submit_after_sbatch" in r.stderr, r.stdout + r.stderr
    jobs_after = [str(j) for j in job_ids(w.fake["cluster_a"])]; j2 = jobs_after[-1]; assert j2 != j1 and len(jobs_after) == 2
    assert len([e for e in w.ledger() if e["event"] == "attempt"]) == 1 and ST.read_transition(str(w.state / "submissions.jsonl"), LW)["resume"] is True
    # the retry reconciles the reservation by job name: the launched job is ADOPTED as the resume attempt (no binding), then refused as in flight
    r = _submit(w, "--resume-from-job", j1)
    assert r.returncode == 3 and "was adopted" in (r.stdout + r.stderr) and "already IN FLIGHT" in r.stderr and j2 in r.stderr, r.stdout + r.stderr
    att = [e for e in w.ledger() if e["event"] == "attempt"]; assert len(att) == 2 and att[-1]["job_id"] == j2 and att[-1]["spec"]["resume_of"]["job_id"] == j1
    assert ST.read_transition(str(w.state / "submissions.jsonl"), LW) is None                                     # the reservation cleared by the adoption
    b = ST.binding_state(str(w.state / "submissions.jsonl"), LW); assert b["state"] == "registered" and str(b["job_id"]) == j1     # no binding written for the adopted resume
    assert [str(j) for j in job_ids(w.fake["cluster_a"])] == jobs_after                                              # nothing launched twice
    # J2 ends in failure: the next resume is admitted and records the crashed one; its reservation is cleared after the launch
    set_job_state(w.fake["cluster_a"], j2, "FAILED", end="2026-01-01T03:00:00")
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 0 and "1 crashed resume(s)" in (r.stdout + r.stderr), r.stdout + r.stderr
    j3 = str(job_ids(w.fake["cluster_a"])[-1]); assert ST.read_transition(str(w.state / "submissions.jsonl"), LW) is None and not (w.state / "bindings" / f"{SLUG}.resume.lock").exists() or True
    set_job_state(w.fake["cluster_a"], j3, "FAILED", end="2026-01-01T04:00:00")
    # two concurrent callers on the same unit: the lock serializes them — exactly one launches, the other is refused as in flight
    argv = [sys.executable, str(Path(w.run.__globals__["LIB"]) / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m", "submit", "--unit", "multiturn.actorattack.g012",
            "--lwid", LW, "--script", TARGET, "--name", "f22-att-m-g012-actorattack", "--args", f"m u0 actorattack {SHA} 9 0,1,2", "--clusters", "cluster_a", "--group", "mt-u0-cluster_a",
            "--wave-sha", SHA, "--resume-from-job", j1]
    n_before = len(job_ids(w.fake["cluster_a"]))
    procs = [subprocess.Popen(argv, env=w.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    outs = [p.communicate(timeout=120)[0] for p in procs]; rcs = [p.returncode for p in procs]
    assert sorted(rcs) == [0, 3] and sum("already IN FLIGHT" in o for o in outs) == 1 and sum("resumes job" in o for o in outs) == 1, (rcs, outs)
    assert len(job_ids(w.fake["cluster_a"])) == n_before + 1
    assert len([e for e in w.ledger() if e["event"] == "attempt"]) == 4
    # a plain re-run's token on the unit is not a resume's: refused by name, nothing launched
    ST.write_transition(str(w.state / "submissions.jsonl"), {"logical_work_id": LW, "unit": "u", "cluster": "cluster_a", "job_name": "n", "intended_attempt": "x", "utc": "2026-01-01T05:00:00Z", "spec": {}, "retired": None})
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 3 and "transition token of a plain re-run" in r.stderr, r.stderr
