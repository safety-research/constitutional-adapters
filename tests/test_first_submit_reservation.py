"""reconcile-fix-3: a FIRST submission is reserved with a nonce BEFORE its sbatch, so the lost-launch window the
funnel never covered — sbatch accepted, the funnel died before the ledger append, the job's id only in sbatch's stdout — is reconciled on the next
invocation by the nonce read back from the job (squeue %k): adopted into the ledger (identified_by nonce), bound, nothing launched twice. A lone
nonce-less same-name job (another session's, or our own job already finished and so without a readable %k) is never adopted by elimination:
refused by name, the reservation standing, nothing launched beside it."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, job_ids, set_job_state  # noqa: E402
from test_resume_from_job import LW, World, _submit  # noqa: E402
from test_submit_twin import SHA  # noqa: E402   # the World's derived wave sha (81629e9fa): the fake eval-repo checkout exists at it, never at a placeholder
from test_resume_reservation_identity import _stamp  # noqa: E402
from test_resume_reservation_owner import _attempts, _foreign_job, _token  # noqa: E402

sys.path.insert(0, str(LIB))
import submit_twin as TW  # noqa: E402


def _fresh(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    return World(tmp_path)


def test_a_first_submission_is_reserved_with_a_nonce_before_its_sbatch_and_a_foreign_singleton_is_never_adopted(tmp_path):
    w = _fresh(tmp_path); fake = w.fake["cluster_a"]
    r = _submit(w, SISTEER_TWIN_CRASH_AT="submit_after_reservation"); assert r.returncode == 97, r.stdout + r.stderr          # reserved, sbatch never ran
    tok = _token(w); assert tok and tok.get("first") is True and re.fullmatch(r"[0-9a-f]{12}", tok["nonce"]) and tok["snapshot_job_ids"] == [] and tok["old_binding"] is None, tok
    assert job_ids(fake) == [] and _attempts(w) == []
    # another session's same-name job appears after the reservation, carrying no nonce: the sole candidate — refused, nothing launched, the reservation stands
    _foreign_job(w, "930", user=os.environ["USER"], submit=_stamp(2))
    r = _submit(w); out = r.stdout + r.stderr
    assert r.returncode == 3 and "1 qualifying candidate of name" in r.stderr and "'930'" in r.stderr and "no nonce read back" in r.stderr and "never adopted by elimination" in r.stderr, out
    assert _token(w) is not None and [str(j) for j in job_ids(fake)] == ["930"] and _attempts(w) == [] and "ADOPTED" not in out


def test_a_first_submission_lost_after_its_sbatch_is_adopted_by_its_nonce_and_never_launched_twice(tmp_path):
    w = _fresh(tmp_path); fake = w.fake["cluster_a"]
    r = _submit(w, SISTEER_TWIN_CRASH_AT="submit_after_sbatch"); assert r.returncode == 97, r.stdout + r.stderr                # the job exists; nothing records it
    tok = _token(w); assert tok and tok.get("first") is True
    (jid,) = job_ids(fake); jid = str(jid)
    argv = (fake / "jobs" / f"{jid}.argv").read_text(); assert f"--comment=nonce:{tok['nonce']}" in argv, argv                 # the nonce on the sbatch line (the fit path merges it after its sha:)
    assert _attempts(w) == []
    r = _submit(w); out = r.stdout + r.stderr
    assert "ADOPTED" in out and "positively identified by its nonce" in out, out
    atts = _attempts(w); assert len(atts) == 1 and atts[0]["job_id"] == jid and atts[0]["identified_by"] == "nonce" and atts[0]["nonce"] == tok["nonce"], atts
    assert _token(w) is None and [str(j) for j in job_ids(fake)] == [jid], "nothing launched twice"
    bl = w.run("binding", "--lwid", atts[0]["logical_work_id"]); assert f"bound_attempt\t{atts[0]['attempt_id']}\tnonce\t{tok['nonce']}" in bl.stdout, bl.stdout + bl.stderr


def test_a_finished_candidate_without_a_readable_nonce_is_refused_for_a_manual_verified_binding(tmp_path):
    w = _fresh(tmp_path); fake = w.fake["cluster_a"]
    r = _submit(w, SISTEER_TWIN_CRASH_AT="submit_after_sbatch"); assert r.returncode == 97, r.stdout + r.stderr
    (jid,) = job_ids(fake); jid = str(jid)
    set_job_state(fake, jid, "COMPLETED", end="2026-01-01T02:00:00")                                                           # gone from the controller: sacct only, no %k
    r = _submit(w); out = r.stdout + r.stderr
    assert r.returncode == 3 and "1 qualifying candidate of name" in r.stderr and jid in r.stderr and "no nonce read back" in r.stderr and "record the right one by hand" in r.stderr, out
    assert _token(w) is not None and _attempts(w) == [] and [str(j) for j in job_ids(fake)] == [jid] and "ADOPTED" not in out


def test_a_first_submission_that_died_after_its_ledger_append_is_completed_not_relaunched(tmp_path):
    """The launcher died after record_attempt's ledger append and before the binding — the token stands and the job is in the ledger. The
    reconciliation must NOT exclude it as "known" and clear the token as a non-launch (the next submit would launch a second array): it
    verifies the nonce read back from the live job and FINISHES the binding / submissions row / watcher, then clears."""
    w = _fresh(tmp_path); fake = w.fake["cluster_a"]
    r = _submit(w, SISTEER_TWIN_CRASH_AT="submit_after_ledger"); assert r.returncode == 97, r.stdout + r.stderr
    tok = _token(w); assert tok and tok.get("first") is True
    (jid,) = job_ids(fake); jid = str(jid)
    atts = _attempts(w); assert len(atts) == 1 and atts[0]["job_id"] == jid and atts[0]["nonce"] == tok["nonce"]
    assert not any(q.name.endswith(".submitted.json") for q in (w.state / "bindings").glob("*")), "died before the binding"
    r = _submit(w); out = r.stdout + r.stderr
    assert "was already in the ledger as job" in out and "verified by the nonce read back from the live job" in out and "nothing launched twice" in out, out
    assert "ADOPTED" not in out or "already in the ledger" in out
    assert [str(j) for j in job_ids(fake)] == [jid] and len(_attempts(w)) == 1, "no second launch, no second attempt"
    assert _token(w) is None
    bl = w.run("binding", "--lwid", atts[0]["logical_work_id"]); assert f"binding\tsubmitted\tcluster_a\t{jid}\tno" in bl.stdout and f"bound_attempt\t{atts[0]['attempt_id']}\tlaunch\t{tok['nonce']}" in bl.stdout, bl.stdout + bl.stderr
    subs = [e for e in __import__("json").loads("[" + ",".join(l for l in (w.state / "submissions.jsonl").read_text().splitlines() if l.strip()) + "]") if e.get("event") == "submitted"]
    assert [e["job_id"] for e in subs] == [jid], subs


def test_a_half_recorded_launch_whose_job_finished_is_completed_from_its_launch_record_and_a_foreign_nonce_refuses(tmp_path):
    """The job is no longer live (no %k to read back) -> the launch record's own nonce under this reservation completes the binding; a live
    job whose comment carries another nonce refuses by name with the token standing."""
    w = _fresh(tmp_path); fake = w.fake["cluster_a"]
    r = _submit(w, SISTEER_TWIN_CRASH_AT="submit_after_ledger"); assert r.returncode == 97, r.stdout + r.stderr
    (jid,) = job_ids(fake); jid = str(jid); tok = _token(w)
    argv_p = fake / "jobs" / f"{jid}.argv"; argv0 = argv_p.read_text()
    argv_p.write_text(argv0.replace(f"nonce:{tok['nonce']}", "nonce:ffffffffffff"))            # live, but another nonce on the job
    r = _submit(w); out = r.stdout + r.stderr
    assert r.returncode == 3 and "live with comment nonce 'ffffffffffff'" in r.stderr and "not this reservation's" in r.stderr and _token(w) is not None, out
    assert [str(j) for j in job_ids(fake)] == [jid] and not any(q.name.endswith(".submitted.json") for q in (w.state / "bindings").glob("*"))
    argv_p.write_text(argv0)
    set_job_state(fake, jid, "COMPLETED", end="2026-01-01T02:00:00")                        # finished: nothing to read back; the launch record decides
    r = _submit(w); out = r.stdout + r.stderr
    assert "verified by the launch record under this reservation (job no longer live)" in out and _token(w) is None and [str(j) for j in job_ids(fake)] == [jid], out
    assert any(q.name.endswith(".submitted.json") for q in (w.state / "bindings").glob("*"))


def test_the_unit_lock_serializes_submit_and_reconcile_token(tmp_path):
    """A submit or a reconcile-token of a unit refuses by name while another funnel process holds the unit's lock — nothing reserved,
    launched or cleared — and proceeds once it is released."""
    import fcntl
    w = _fresh(tmp_path); fake = w.fake["cluster_a"]
    lock_p = w.state / "bindings" / ".locks" / (LW.replace("/", "__") + ".unit.lock"); lock_p.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_p, "a+") as held:
        fcntl.flock(held.fileno(), fcntl.LOCK_EX)
        r = _submit(w, SISTEER_UNIT_LOCK_TIMEOUT_S="1"); out = r.stdout + r.stderr
        assert r.returncode == 3 and "unit lock" in r.stderr and "held by another funnel process" in r.stderr and job_ids(fake) == [] and _token(w) is None, out
        r = w.run("reconcile-token", "--lwid", LW, SISTEER_UNIT_LOCK_TIMEOUT_S="1"); assert r.returncode == 3 and "unit lock" in r.stdout + r.stderr, r.stdout + r.stderr
    r = _submit(w); assert r.returncode == 0 and len(job_ids(fake)) == 1, r.stdout + r.stderr


def test_an_attacker_pool_launch_that_died_after_its_ledger_append_is_completed_with_its_pool_membership(tmp_path):
    """The completion replays record_attempt's OTHER side effects too — an attacker-pool attempt's pool_member event, which
    pool_members / mark-placed read; without it the launched member is invisible to the group."""
    w = _fresh(tmp_path); fake = w.fake["cluster_a"]
    att_lw = "m/u0/multiturn/mt-u0-cluster_a/attacker"
    def _att(*extra, **env):
        return w.run("submit", "--unit", "multiturn.pool.mt-u0-cluster_a", "--lwid", att_lw, "--script", "scripts/sdfbase/full22/attacker_pool.sbatch", "--name", "f22-pool-m-u0",
                     "--args", "m u0", "--clusters", "cluster_a", "--kind", "attacker", "--group", "mt-u0-cluster_a", "--wave-sha", SHA, *extra, **env)
    r = _att(SISTEER_TWIN_CRASH_AT="submit_after_ledger"); assert r.returncode == 97, r.stdout + r.stderr
    (jid,) = job_ids(fake); jid = str(jid)
    assert not any(e.get("event") == "pool_member" for e in w.ledger()), "died before the pool_member event"
    r = _att(); out = r.stdout + r.stderr
    assert "was already in the ledger as job" in out and "pool_member" in out and "nothing launched twice" in out, out
    pm = [e for e in w.ledger() if e.get("event") == "pool_member"]
    assert len(pm) == 1 and pm[0]["job_id"] == jid and pm[0]["group"] == "mt-u0-cluster_a" and pm[0]["logical_work_id"] == att_lw, pm
    assert [str(j) for j in job_ids(fake)] == [jid]
    sys.path.insert(0, str(LIB)); import submit_twin as TW2  # noqa: E402
    cfg = Path(w.site) if Path(w.site).is_file() else Path(w.site) / "config.env"
    co = TW2.Coordinator(TW2.Site(str(cfg), False), "m", None)
    assert co.pool_members("mt-u0-cluster_a") == [jid], "the completed member is a recorded pool member"


def test_a_frozen_unit_launch_that_died_after_its_ledger_append_is_completed_with_its_sha_provenance(tmp_path, monkeypatch):
    """A FROZEN-UNIT attempt (the spec carries `provenance`) that died after the ledger append is completed with its per-unit
    sha_provenance SIDECAR and ledger event — admission (the frozen-tree attempt selection) then sees the attempt at the advanced sha.
    Driven in-process through the real reconcile_transition over the real fakes (a live job carrying the reservation's nonce)."""
    import json as _json
    sys.path.insert(0, str(LIB)); import submit_twin as TW2; import stage as ST2; import grant_ledger as GL2  # noqa: E402
    w = _fresh(tmp_path); fake = w.fake["cluster_a"]; (fake / "jobs").mkdir(exist_ok=True)
    for k, v in w.env().items():        # the in-process coordinator shells out to the FAKE squeue/sacct the way the CLI runs do
        monkeypatch.setenv(k, v)
    cfg = Path(w.site) if Path(w.site).is_file() else Path(w.site) / "config.env"
    co = TW2.Coordinator(TW2.Site(str(cfg), False), "m", None)
    sha = "d" * 40; nonce = "0123456789ab"; aid = TW2.attempt_id(LW, "cluster_a", 1)
    spec = {"unit": "multiturn.actorattack.g012", "lwid": LW, "script": "scripts/sdfbase/full22/target_attacks.sbatch", "name": "f22-att-m-g012-actorattack",
            "args": ["m", "u0", "actorattack", sha, "9", "0,1,2"], "env": [], "sha": sha, "time_limit": None, "array": None, "kind": "target", "group": "mt-u0-cluster_a",
            "clusters": ["cluster_a"], "flags": "", "registers": None, "depends_on": None,
            "provenance": {"kind": "advance", "answers": {"job_id": "777", "cluster": "cluster_a"}, "fix_commit": "e" * 40, "contract_sha256": "f" * 64, "from_sha": "c" * 40}}
    tok = {"logical_work_id": LW, "unit": spec["unit"], "cluster": "cluster_a", "job_name": spec["name"], "intended_attempt": aid, "utc": _stamp(-30),
           "old_binding": None, "spec": spec, "retired": None, "spool": None, "why": "test", "nonce": nonce, "snapshot_job_ids": [], "first": True}
    ST2.write_transition(str(w.state / "submissions.jsonl"), tok)
    # the launch: the job live with the reservation's nonce; the ledger attempt appended (record_attempt's first step) — nothing else
    _foreign_job(w, "5001", user=os.environ["USER"], submit=_stamp(-20)); (fake / "jobs" / "5001.argv").write_text(f"x --comment=sha:{sha},nonce:{nonce}\n")
    GL2.Ledger(str(w.state / "attempts.jsonl")).append({"event": "attempt", "logical_work_id": LW, "attempt_id": aid, "cluster": "cluster_a", "job_id": "5001",
                                                        "job_name": spec["name"], "kind": "target", "script": spec["script"], "args": spec["args"], "sha": sha,
                                                        "unit": spec["unit"], "spec": spec, "nonce": nonce, "identified_by": "launch"})
    assert ST2.ledger_sha_provenance(str(w.state / "submissions.jsonl")).get(LW, []) == [] and not (w.state / "bindings" / (LW.replace("/", "__") + ".submitted.json")).exists()
    assert co.reconcile_transition(LW, tok) == "adopted"
    prov = ST2.ledger_sha_provenance(str(w.state / "submissions.jsonl")).get(LW, [])
    assert len(prov) == 1 and prov[0]["sha"] == sha and prov[0]["attempt_id"] == aid and prov[0]["job_id"] == "5001" and prov[0]["kind"] == "advance", prov
    side = (ST2.read_provenance_doc(str(w.state / "submissions.jsonl")).get("units") or {}).get(LW, [])
    assert len(side) == 1 and side[0]["sha"] == sha and side[0]["attempt_id"] == aid, side
    assert (w.state / "bindings" / (LW.replace("/", "__") + ".submitted.json")).is_file() and ST2.read_transition(str(w.state / "submissions.jsonl"), LW) is None
    fta = ST2._frozen_tree_attempts(str(w.state / "submissions.jsonl"), "m", "u0", None)        # admission: the attempt at the advanced sha is seen
    assert LW in fta and fta[LW][0]["job_id"] == "5001" and fta[LW][0]["sha"] == sha, fta
    # idempotent: a second reconciliation of the same (re-opened) token adds nothing twice
    ST2.write_transition(str(w.state / "submissions.jsonl"), tok)
    assert co.reconcile_transition(LW, tok) == "adopted"
    assert len(ST2.ledger_sha_provenance(str(w.state / "submissions.jsonl")).get(LW, [])) == 1 and len([e for e in w.ledger() if e.get("event") == "sha_provenance"]) == 1
