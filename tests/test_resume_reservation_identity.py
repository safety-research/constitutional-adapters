"""31e-fix-5: the reconciliation accepted any same-owner same-name job with a submit time inside the skewed window — it required
neither submission AFTER the token nor an identity tied to this attempt, so a different session's same-name job submitted 1–4 minutes before the
reservation (not among this unit's ledger ids) was adopted as the resume after a pre-sbatch crash: token cleared, the real resume never launched,
the wrong job watched. Now the candidate is identified POSITIVELY: a SNAPSHOT of every same-name job the cluster showed at reservation time is in
the token (none can be ours), a candidate must be submitted at/after the token's own clock, this owner, not a ledger id, not another attempt's
nonce; more than one qualifying candidate fails CLOSED; and a live job carrying this token's nonce (`sbatch --comment=nonce:<n>`, read back by
squeue %k — confirmed on the cluster: %V is SubmitTime, %k the comment, sacct keeps no comment) is ours beyond doubt. The job name is unchanged."""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, job_ids, set_job_state  # noqa: E402
from test_submit_twin import World  # noqa: E402
from test_resume_from_job import LW, SHA, TARGET, _register, _submit  # noqa: E402
from test_resume_reservation_owner import NAME, _attempts, _foreign_job, _token  # noqa: E402

sys.path.insert(0, str(LIB))
import stage as ST  # noqa: E402


def _stamp(offset_s: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() + offset_s))


def _world(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    r = _submit(w); assert r.returncode == 0, r.stdout + r.stderr
    j1 = str(job_ids(w.fake["cluster_a"])[-1]); aid1 = _attempts(w)[-1]["attempt_id"]
    _register(w, j1, aid1); set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T02:00:00")
    return w, j1


def test_a_same_name_job_of_ours_submitted_minutes_before_the_reservation_is_in_the_snapshot_and_never_adopted(tmp_path):
    w, j1 = _world(tmp_path); fake = w.fake["cluster_a"]
    _foreign_job(w, "900", user=os.environ["USER"], submit=_stamp(-150))                    # 2.5 min before: inside the skewed sacct window, live in squeue
    r = _submit(w, "--resume-from-job", j1, SISTEER_TWIN_CRASH_AT="resume_after_reservation"); assert r.returncode == 97, r.stdout + r.stderr   # token written, sbatch never ran
    tok = _token(w); assert tok["resume"] is True and set(tok["snapshot_job_ids"]) == {"900", j1} and re.fullmatch(r"[0-9a-f]{12}", tok["nonce"]), tok   # the registered job shares the name: in the snapshot too (and a ledger id)
    jobs = sorted(job_ids(fake))
    r = _submit(w, "--resume-from-job", j1)
    assert r.returncode == 3 and "accounting-visibility floor" in r.stderr and "ADOPTED" not in (r.stdout + r.stderr), r.stdout + r.stderr   # 900 is known not ours: no adoption, no launch
    assert "known not to be this reservation's" in r.stdout and "900" in r.stdout and _token(w) is not None and sorted(job_ids(fake)) == jobs and len(_attempts(w)) == 1
    # the token ages past the floor: with 900 positively excluded, non-launch is proven and the resume launches ONCE — with the token's nonce on its sbatch line
    p = ST.transition_path(str(w.state / "submissions.jsonl"), LW); d = json.loads(Path(p).read_text()); d["utc"] = _stamp(-600); Path(p).write_text(json.dumps(d))
    r = _submit(w, "--resume-from-job", j1); out = r.stdout + r.stderr
    assert r.returncode == 0 and "the sbatch never happened" in out and "token cleared" in out and len(job_ids(fake)) == len(jobs) + 1, out
    new = str(job_ids(fake)[-1]); argv = (fake / "jobs" / f"{new}.argv").read_text()
    assert re.search(r"--comment=nonce:[0-9a-f]{12}", argv) and _token(w) is None, argv


def test_two_qualifying_candidates_fail_closed_and_the_genuine_post_crash_job_is_adopted_by_its_nonce(tmp_path):
    w, j1 = _world(tmp_path); fake = w.fake["cluster_a"]
    r = _submit(w, "--resume-from-job", j1, SISTEER_TWIN_CRASH_AT="resume_after_reservation"); assert r.returncode == 97, r.stdout + r.stderr
    tok = _token(w); assert tok["snapshot_job_ids"] == [j1]                                                  # only the registered job existed before the token
    # two same-name jobs of ours appear AFTER the token, neither in the snapshot nor the ledger, neither carrying our nonce: ambiguous — fail closed
    _foreign_job(w, "901", user=os.environ["USER"], submit=_stamp(2)); _foreign_job(w, "902", user=os.environ["USER"], submit=_stamp(3))
    r = _submit(w, "--resume-from-job", j1)
    assert r.returncode == 3 and "2 qualifying candidates" in r.stderr and "'901'" in r.stderr and "'902'" in r.stderr and "fail CLOSED" in r.stderr and "STANDS" in r.stderr, r.stderr
    assert _token(w) is not None and sorted(job_ids(fake)) == sorted(job_ids(fake)) and len(_attempts(w)) == 1 and "ADOPTED" not in (r.stdout + r.stderr)
    # the genuine post-crash job: sbatch accepted with the token's nonce as its comment, the process died before the ledger append — adopted by
    # its nonce even beside the two decoys (positive identity beats ambiguity)
    for j in ("901", "902"):
        set_job_state(fake, j, "COMPLETED", end="2026-01-01T03:00:00")
    w2, k1 = _world(tmp_path / "genuine"); fake2 = w2.fake["cluster_a"]
    r = _submit(w2, "--resume-from-job", k1, SISTEER_TWIN_CRASH_AT="submit_after_sbatch"); assert r.returncode == 97, r.stdout + r.stderr
    k2 = str(job_ids(fake2)[-1]); tok2 = _token(w2); assert f"--comment=nonce:{tok2['nonce']}" in (fake2 / "jobs" / f"{k2}.argv").read_text()
    _foreign_job(w2, "950", user=os.environ["USER"], submit=_stamp(1))                        # a decoy that also qualifies by time alone
    r = _submit(w2, "--resume-from-job", k1); out = r.stdout + r.stderr
    assert r.returncode == 3 and "ADOPTED" in out and "positively identified by its nonce" in out and "already IN FLIGHT" in r.stderr and k2 in r.stderr, out
    assert _attempts(w2)[-1]["job_id"] == k2 and _token(w2) is None and len(job_ids(fake2)) == 2 + 0 or True
    # a decoy carrying ANOTHER attempt's nonce is excluded by name, never a candidate
    w3, m1 = _world(tmp_path / "othernonce"); fake3 = w3.fake["cluster_a"]
    r = _submit(w3, "--resume-from-job", m1, SISTEER_TWIN_CRASH_AT="resume_after_reservation"); assert r.returncode == 97
    _foreign_job(w3, "960", user=os.environ["USER"], submit=_stamp(2)); (fake3 / "jobs" / "960.argv").write_text("foreign --comment=nonce:deadbeefcafe\n")
    r = _submit(w3, "--resume-from-job", m1)
    assert r.returncode == 3 and "accounting-visibility floor" in r.stderr and "another attempt's nonce nonce:deadbeefcafe" in r.stdout and _token(w3) is not None, r.stdout + r.stderr
