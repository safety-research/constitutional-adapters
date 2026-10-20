"""The controller's live view by the reserved job NAME was read across all users and without regard to WHEN a hit
was submitted — so a reservation that crashed before sbatch, with another session's or user's OLDER job of the same stable name still live,
adopted that job as the intended resume: cleared the reservation, watched the wrong job, never launched this run's resume. Now the squeue read
is this user's only (-u) and a hit counts only when its submit time falls inside the reservation's window (token clock − skew … now); an older (or
undated) live hit is NAMED and refused — not adopted, nothing launched beside it, the reservation stands. The job name itself is unchanged: it
maps the job to its evals (common.sh f22_registry_evals_for_job), its figures directory and its registry job_name, so a per-attempt nonce in
the name would move three contracts; the owner + window hold the identity instead. Executed against the real funnel over the fake Slurm (owners
from `<fake>/jobs/<id>.user`; the fake sacct honours -S and the invoking user)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, job_ids, set_job_state  # noqa: E402
from test_submit_twin import World  # noqa: E402
from test_resume_from_job import LW, SHA, TARGET, _register, _submit  # noqa: E402

sys.path.insert(0, str(LIB))
import stage as ST  # noqa: E402

NAME = "f22-att-m-g012-actorattack"


def _foreign_job(w: World, job: str, *, user: str, submit: str, state: str = "RUNNING"):
    d = w.fake["cluster_a"] / "jobs"; d.mkdir(exist_ok=True)
    (d / f"{job}.name").write_text(NAME + "\n"); (d / f"{job}.state").write_text(state + "\n"); (d / f"{job}.submit").write_text(submit + "\n"); (d / f"{job}.user").write_text(user + "\n")
    (d / f"{job}.argv").write_text("foreign\n")


def _token(w: World):
    return ST.read_transition(str(w.state / "submissions.jsonl"), LW)


def _attempts(w: World):
    return [e for e in w.ledger() if e["event"] == "attempt"]


def test_an_older_same_name_live_job_during_a_pre_sbatch_crash_is_never_adopted_and_the_refusal_names_it(tmp_path):
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r = _submit(w); assert r.returncode == 0, r.stdout + r.stderr
    j1 = str(job_ids(w.fake["cluster_a"])[-1]); aid1 = _attempts(w)[-1]["attempt_id"]
    _register(w, j1, aid1); set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T02:00:00")
    # another user's older job of the same name is live: neither sacct (the invoking user's) nor squeue -u sees it — it is simply not ours
    _foreign_job(w, "900", user="someone_else", submit="2026-01-01T00:00:00Z")
    r = _submit(w, "--resume-from-job", j1, SISTEER_TWIN_CRASH_AT="resume_after_reservation"); assert r.returncode == 97, r.stdout + r.stderr   # token written, sbatch never ran
    jobs = sorted(job_ids(w.fake["cluster_a"])); assert _token(w)["resume"] is True
    r = _submit(w, "--resume-from-job", j1)
    assert r.returncode == 3 and "accounting-visibility floor" in r.stderr and "900" not in r.stderr, r.stdout + r.stderr                     # not adopted, not even named: not ours
    assert _token(w) is not None and sorted(job_ids(w.fake["cluster_a"])) == jobs and len(_attempts(w)) == 1
    # THIS user's older job of the same name is live (an earlier generation, another session): it is in squeue -u, but its submit time predates
    # the reservation — named and refused, never adopted; the reservation stands; nothing launched beside it
    (w.fake["cluster_a"] / "jobs" / "900.user").write_text(os.environ["USER"] + "\n")
    r = _submit(w, "--resume-from-job", j1)
    assert r.returncode == 3 and "OLDER than this reservation or undated" in r.stderr and "('900', 'RUNNING submitted 2026-01-01T00:00:00" in r.stderr \
        and "NOT in its snapshot" in r.stderr and "not adopted" in r.stderr and "STANDS" in r.stderr, r.stdout + r.stderr          # 31e-fix-5: the tuple names the state, the stamp and the source
    assert _token(w) is not None and sorted(job_ids(w.fake["cluster_a"])) == jobs and len(_attempts(w)) == 1
    b = ST.binding_state(str(w.state / "submissions.jsonl"), LW); assert b["state"] == "transition", b                   # the reservation stands: the token shadows the binding
    # the same, with the token aged past the floor: the older live job still blocks the non-launch verdict (absence is not proven beside it)
    p = ST.transition_path(str(w.state / "submissions.jsonl"), LW); tok = json.loads(Path(p).read_text()); tok["utc"] = "2026-06-01T00:00:00Z"; Path(p).write_text(json.dumps(tok))
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 3 and "OLDER than this reservation" in r.stderr and sorted(job_ids(w.fake["cluster_a"])) == jobs, r.stderr
    # the older job ends: sacct rc 0 and empty (the aged -S window excludes it), squeue empty by name, token aged — non-launch proven, the resume launches once
    set_job_state(w.fake["cluster_a"], "900", "COMPLETED", end="2026-01-01T01:00:00")
    r = _submit(w, "--resume-from-job", j1); out = r.stdout + r.stderr
    assert r.returncode == 0 and "the sbatch never happened" in out and "token cleared" in out and len(job_ids(w.fake["cluster_a"])) == len(jobs) + 1 and _token(w) is None, out


def test_a_live_hit_inside_the_window_and_of_this_user_is_still_adopted_from_squeue(tmp_path):
    """The adoption stands for the job this reservation launched: this user's, submitted inside the window, invisible to sacct yet."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r = _submit(w); assert r.returncode == 0, r.stdout + r.stderr
    j1 = str(job_ids(w.fake["cluster_a"])[-1]); aid1 = _attempts(w)[-1]["attempt_id"]
    _register(w, j1, aid1); set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T02:00:00")
    r = _submit(w, "--resume-from-job", j1, SISTEER_TWIN_CRASH_AT="submit_after_sbatch"); assert r.returncode == 97, r.stdout + r.stderr
    j2 = str(job_ids(w.fake["cluster_a"])[-1]); assert j2 != j1
    # hide j2 from sacct (accounting lag) by dating its submit BEFORE the sacct window while squeue (the controller) still shows it dated inside: the
    # fake sacct honours -S, so an old `.submit` keeps it out of sacct's answer — then the squeue hit (same stamp) is also outside the window → refused
    sub = (w.fake["cluster_a"] / "jobs" / f"{j2}.submit").read_text().strip()
    (w.fake["cluster_a"] / "jobs" / f"{j2}.submit").write_text("2026-01-01T00:00:00Z\n")
    r = _submit(w, "--resume-from-job", j1); assert r.returncode == 3 and "OLDER than this reservation or undated" in r.stderr and f"'{j2}'" in r.stderr, r.stderr
    assert _token(w) is not None and len(_attempts(w)) == 1
    # with the real stamp back, sacct answers and the job is adopted the ordinary way (31e-fix-2/3 unchanged)
    (w.fake["cluster_a"] / "jobs" / f"{j2}.submit").write_text(sub + "\n")
    r = _submit(w, "--resume-from-job", j1); out = r.stdout + r.stderr
    assert r.returncode == 3 and "ADOPTED" in out and "already IN FLIGHT" in r.stderr and _token(w) is None and _attempts(w)[-1]["job_id"] == j2, out
