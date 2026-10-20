"""A committed placement can race a stale waiter into stopping its replacement: holding the backstop's destructive writes behind
attackers/PLACED.transition, read under the group's lock, was not enough — a WHOLE --resubmit pass (open ->
re-mint -> mark-placed --supersede) could land between a poll's read of PLACED v1 and that poll's teardown; the commit had cleared the marker,
so the lock found nothing to hold on and wrote STOP for every member the ledger listed, the ones v2 had just been assigned included. Now every
destructive write carries the identity (version + (unit, job) pairs) of the record its caller decided on — the waiter's poll, the by-hand
command's own read — and the program compares it with the record under the lock, after the marker check and before its one write: a record
that moved answers `stale` / `stale_placement` with nothing written; the waiter logs it, skips that poll's ratio, and polls on; the by-hand
`teardown-pool` and `retire-member` refuse by name. Over the fake clusters of test_submit_twin's World, with the REAL funnel subcommands as
06's pass runs them; the interleaving is made exact with the fake sacct's test-only gate (FAKE_SACCT_GATE, keyed on the frozen process's env),
never with timing."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, job_ids, set_job_state  # noqa: E402
from test_pool_dead_member_torn_down import _walk_target  # noqa: E402
from test_pool_member_lifecycle_funnel import GROUP  # noqa: E402
from test_pool_retire_funnel import _lease, _retiring, _wait_until  # noqa: E402
from test_pool_self_exit_skip import _walk  # noqa: E402
from test_wait_pool_holds_under_placement_transition import REASON, _open, _placed_world, _stops, _text, _torn, _wait_line  # noqa: E402


def _gated(w, args, gate: Path, log: Path | None = None):
    """A funnel subcommand whose sacct calls block while `gate` exists (the fake sacct's FAKE_SACCT_GATE, on THIS process's env alone); its
    output to `log` when given (a file the test reads while it runs), else piped."""
    argv = [sys.executable, str(LIB / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m", *args]
    if log is not None:
        return subprocess.Popen(argv, env=w.env(FAKE_SACCT_GATE=str(gate)), stdout=open(log, "w"), stderr=subprocess.STDOUT, text=True)
    return subprocess.Popen(argv, env=w.env(FAKE_SACCT_GATE=str(gate)), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _frozen_on(gate: Path, proc, *, log: Path | None = None, timeout_s=90.0) -> str:
    """The job ids the gated process is blocked on (its first sacct since the gate stood). Fails naming it when the deadline passes or the
    process exited first."""
    waiting = Path(str(gate) + ".waiting"); t0 = time.monotonic()
    while True:
        if waiting.exists() and waiting.read_text().strip():
            return waiting.read_text().strip()
        exited = proc.poll() is not None
        if exited or time.monotonic() - t0 > timeout_s:
            out = _text(log) if log is not None else ((proc.communicate(timeout=30)[0] or "") if exited else "")
            if not exited:
                proc.kill(); proc.wait(timeout=30)
            raise AssertionError((f"the gated process exited (rc {proc.returncode}) before" if exited else f"waited {timeout_s:g}s for") + f" its sacct to block on the gate\n{out}")
        time.sleep(0.05)


def _pass(w, lw1, lw2, *, targets):
    """06's --resubmit pass as its subcommands run it: open the marker -> re-mint member 1 (dead) -> re-mint the named dead targets ->
    mark-placed --supersede (v2; the commit clears the marker). Returns (the replacement member's job, [the new target jobs], v2)."""
    _open(w)
    r = _walk(w, 1); assert r.returncode == 0 and "RESUBMIT" in (r.stdout + r.stderr) and "recorded torn_down (dead)" in (r.stdout + r.stderr), r.stdout + r.stderr
    j3 = str(job_ids(w.fake["cluster_a"])[-1])
    units = ["multiturn.attacker1"]
    for unit, shards in targets:
        r = _walk_target(w, unit, shards); assert r.returncode == 0 and "submitted cluster_a" in r.stdout, r.stdout + r.stderr
        units.append(f"multiturn.{unit.replace('/', '.')}")
    new = [str(j) for j in job_ids(w.fake["cluster_a"])[-len(targets):]] if targets else []
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", "--supersede", "--reason", f"{REASON}: re-minted {','.join(units)}")
    assert r.returncode == 0 and "SUPERSEDED version 1 -> 2" in (r.stdout + r.stderr) and "(the pass's PLACED.transition marker cleared: the transition committed)" in (r.stdout + r.stderr), r.stdout + r.stderr
    pool = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers"
    v2 = json.loads((pool / "PLACED").read_text()); assert v2["version"] == 2 and not (pool / "PLACED.transition").exists()
    return j3, new, v2


def test_a_placement_committed_between_the_waiters_read_of_v1_and_its_teardown_is_stale_under_the_lock_and_stops_nothing(tmp_path):
    """The finding's interleaving, exactly: the waiter's poll reads the ledger's bindings and PLACED v1 and is frozen at its first sacct; the
    member and both old targets die and a WHOLE pass lands (v2 over [survivor, replacement], the marker cleared by the commit); the poll resumes,
    judges its OLD roster terminal, and tears down with the record it read — the transaction finds v2 under the lock: STALE, nothing written.
    Before the fix the lock saw no marker and wrote STOP for the survivor and the replacement, the members v2 had just been assigned. The next
    poll reads the current placement (two live targets, each member assigned one) and the pool is stopped only when THOSE end."""
    w, (j1, j2), (lw1, lw2), (t1, t2), root, pool, v1 = _placed_world(tmp_path)
    gate = tmp_path / "sacct.gate"; gate.write_text("")          # stands before the waiter starts: its FIRST poll freezes at its first sacct
    log = tmp_path / "wait-pool.log"; proc = _gated(w, ["wait-pool", "--group", GROUP, "--poll-s", "0.3", "--max-hours", "0.05"], gate, log)
    try:
        # the poll has read the ledger's bindings and PLACED v1 (both precede the roster's sacct loop) and judged nothing yet: it is blocked on the
        # roster's first target (the roster is sorted by unit; g012 < g34)
        assert _frozen_on(gate, proc, log=log) == t1 and "tearing the pool down" not in _text(log), _text(log)
        # the world moves under the frozen poll: member 1 and BOTH old targets die ...
        for j in (j1, t1, t2):
            set_job_state(w.fake["cluster_a"], j, "FAILED", end="2026-01-01T01:00:00")
        # ... and the whole --resubmit pass lands — open, the member re-minted, both targets re-minted, PLACED superseded, the marker cleared
        j3, (t3, t4), v2 = _pass(w, lw1, lw2, targets=(("crescendo/g012", "0,1,2"), ("crescendo/g34", "3,4")))
        assert v2["members"] == [j2, j3] and [t["job_id"] for t in v2["targets"]] == [t3, t4] and [t["assigned_member_job_id"] for t in v2["targets"]] == [j2, j3], v2
        assert not list(pool.glob("*.STOP")) and not _stops(w) and proc.poll() is None
        # the gate lifts: the frozen poll finds its old roster terminal (it is) and tears down with the record it read — v1 — which the transaction
        # finds superseded under the lock. Nothing written; the waiter says why and polls on; this poll's ratio is skipped (its counts are v1's)
        gate.unlink()
        txt = _wait_line(log, "teardown STALE this poll", 1, proc)
        assert "placement complete (2 placed target(s)) and every target terminal on sacct (2 target(s)) — tearing the pool down" in txt, txt
        assert (f"teardown {GROUP}: STALE — attackers/PLACED on cluster_a is version 2, not the version 1 this teardown decided on (a placement committed in between: "
                f"2 target pair(s) added, 2 removed); the current placement may assign targets to members this would have stopped — nothing written on cluster_a") in txt, txt
        assert f"wait-pool {GROUP}: teardown STALE this poll — attackers/PLACED moved from version 1 to 2 after this poll read it (a pass re-placed the group); nothing written; polling on" in txt, txt
        assert f"wait-pool {GROUP}: ratio check skipped this poll (the placement this poll read is superseded" in txt, txt
        assert not list(pool.glob("*.STOP")) and not _stops(w) and not list(pool.glob("*.RETIRE")) and not _retiring(w) and proc.poll() is None, sorted(p.name for p in pool.iterdir())
        # the next poll reads the current placement: the ledger binds t3 and t4 (live) and v2 assigns one to each member — nothing to retire, nothing to stop
        txt = _wait_line(log, "still serve live targets", 1, proc)
        assert not list(pool.glob("*.STOP")) and not _stops(w) and not list(pool.glob("*.RETIRE")) and proc.poll() is None
        # the NEW targets end: this poll's record is v2, the record under the lock -> STOP for all three (the dead member's moot and harmless), rc 0
        set_job_state(w.fake["cluster_a"], t3, "COMPLETED", end="2026-01-01T03:00:00"); set_job_state(w.fake["cluster_a"], t4, "COMPLETED", end="2026-01-01T03:10:00")
        proc.wait(timeout=90); out = _text(log)
        assert proc.returncode == 0, out
        assert sorted(p.name for p in pool.glob("*.STOP")) == sorted(f"{j}.STOP" for j in (j1, j2, j3)) and sorted(e["job_id"] for e in _stops(w)) == sorted([j1, j2, j3]), out
        assert out.count("teardown STALE this poll") == 1 and out.count("tearing the pool down") == 2 and "HELD" not in out and "was ignored" not in out, out
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=30)


def test_the_ratios_retirement_carries_the_polls_placement_and_a_commit_in_between_holds_it_until_the_next_poll_reads_the_current_one(tmp_path):
    """The sibling write: with a target still live the frozen poll never tears down — its RATIO decides, on consumer counts and a live-target
    count that are v1's. The pass commits v2 under it; the retirement it then writes carries v1 and is held under the lock (STALE, no RETIRE);
    the next poll reads v2, counts again, and retires the member that is idle under the CURRENT placement."""
    w, (j1, j2), (lw1, lw2), (t1, t2), root, pool, v1 = _placed_world(tmp_path)
    gate = tmp_path / "sacct.gate"; gate.write_text("")
    log = tmp_path / "wait-pool.log"; proc = _gated(w, ["wait-pool", "--group", GROUP, "--poll-s", "0.3", "--max-hours", "0.05"], gate, log)
    try:
        assert _frozen_on(gate, proc, log=log) == t1
        # member 1 dies with its target; target 2 runs on — so this poll never tears down. The pass re-mints the member and the dead target and
        # supersedes: v2 keeps (g34, t2) on j2 and assigns the new (g012, t3) round-robin from the surviving members — j2 again; j3 is idle under v2
        set_job_state(w.fake["cluster_a"], j1, "FAILED", end="2026-01-01T01:00:00"); set_job_state(w.fake["cluster_a"], t1, "FAILED", end="2026-01-01T01:05:00")
        j3, (t3,), v2 = _pass(w, lw1, lw2, targets=(("crescendo/g012", "0,1,2"),))
        assert v2["members"] == [j2, j3] and {t["job_id"]: t["assigned_member_job_id"] for t in v2["targets"]} == {t3: j2, t2: j2}, v2
        gate.unlink()
        # the frozen poll resumes: one live target (t2) -> one member allowed; two live members (j2 serving, j3 idle) -> the retirement of j3 carries
        # v1 and is HELD: the record under the lock is v2. No RETIRE, no `retiring`, the hold named with the versions and the retirement it would have made
        txt = _wait_line(log, "enforcement held this poll — attackers/PLACED moved since this poll read it", 1, proc)
        assert (f"retire {GROUP}: member job {j3} on cluster_a — STALE: attackers/PLACED is version 2, not the version 1 this retirement decided on (a placement committed in "
                f"between; the current one may assign it targets); nothing written — would have retired it: 2 live member(s) for 1 live target(s); 1 allowed") in txt, txt
        assert "(1 idle member(s) above one-per-three not retired)" in txt and "tearing the pool down" not in txt, txt
        assert not list(pool.glob("*.RETIRE")) and not _retiring(w) and not list(pool.glob("*.STOP")) and proc.poll() is None, sorted(p.name for p in pool.iterdir())
        # the next poll reads v2 and counts again: j3 is idle under the current placement too, so it retires now — the write carries v2, which stands
        ret = _wait_until(lambda: _retiring(w), "the next poll's `retiring` event", proc=proc)
        assert [e["job_id"] for e in ret] == [j3] and ret[0]["survivors"] == [j2] and (pool / f"{j3}.RETIRE").exists() and not (pool / f"{j2}.RETIRE").exists()
        out = _text(log); assert out.index("STALE: attackers/PLACED is version 2") < out.index(f"member job {j3} on cluster_a marked RETIRING"), out
        assert not list(pool.glob("*.STOP")) and not _stops(w)
        # the remaining targets end: the teardown carries v2 -> stopped, rc 0
        set_job_state(w.fake["cluster_a"], t2, "COMPLETED", end="2026-01-01T03:00:00"); set_job_state(w.fake["cluster_a"], t3, "COMPLETED", end="2026-01-01T03:10:00")
        proc.wait(timeout=90); out = _text(log)
        assert proc.returncode == 0 and sorted(p.name for p in pool.glob("*.STOP")) == sorted(f"{j}.STOP" for j in (j1, j2, j3)) and "STALE" not in out[out.index("marked RETIRING"):], out
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=30)


def test_retire_member_and_teardown_pool_by_hand_refuse_when_the_placement_moved_between_their_read_and_the_lock(tmp_path):
    """The by-hand commands read the record they decide on and carry it into the lock: `retire-member` (frozen at its live-target count after its
    read of v1, the pass commits v2) refuses rc 3 by name with nothing written, and proceeds when run again against the current placement;
    `teardown-pool` (blocked on the group's lock after its read of v1 while the record moves) refuses rc 3 by name with nothing written, and
    proceeds when run again."""
    w, (j1, j2), (lw1, lw2), (t1, t2), root, pool, v1 = _placed_world(tmp_path)
    gate = tmp_path / "sacct.gate"; gate.write_text("")
    proc = _gated(w, ["retire-member", "--group", GROUP, "--job", j2, "--why", "operator"], gate)
    assert _frozen_on(gate, proc) == t1                      # PLACED v1 read (it precedes the count); the count's first sacct is where it froze
    for j in (j1, t1, t2):
        set_job_state(w.fake["cluster_a"], j, "FAILED", end="2026-01-01T01:00:00")
    j3, (t3, t4), v2 = _pass(w, lw1, lw2, targets=(("crescendo/g012", "0,1,2"), ("crescendo/g34", "3,4")))
    gate.unlink(); out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 3, out
    assert f"retire {GROUP}: member job {j2} on cluster_a — STALE: attackers/PLACED is version 2, not the version 1 this retirement decided on" in out and "would have retired it: operator" in out, out
    assert f"REFUSED: retire-member {GROUP}: attackers/PLACED on cluster_a moved under this retirement (above) — refused, nothing written" in out and "run this again against the current placement" in out, out
    assert not list(pool.glob("*.RETIRE")) and not _retiring(w)
    # run again against the current placement: the replacement serves now (its record + lease), so the survivor it names is j3
    (pool / f"{j3}.json").write_text(json.dumps({"url": "http://n2:8709", "job_id": j3})); _lease(pool, j3, "http://n2:8709")
    r = w.run("retire-member", "--group", GROUP, "--job", j2, "--why", "operator")
    assert r.returncode == 0 and f"member job {j2} on cluster_a marked RETIRING (1 member(s) still serving)" in r.stdout and "STALE" not in r.stdout, r.stdout + r.stderr
    assert (pool / f"{j2}.RETIRE").exists() and [e["job_id"] for e in _retiring(w)] == [j2] and _retiring(w)[0]["survivors"] == [j3]
    # teardown-pool by hand (a second world): it reads v1, then its transaction waits on the group's lock, which the test holds; the record moves
    # meanwhile (v2's shape written by the fixture — the real commit needs the lock the test holds; the compare reads version + pairs alone)
    (tmp_path / "w2").mkdir(); w2, (k1, k2), (m1, m2), (u1, u2), root2, pool2, p1 = _placed_world(tmp_path / "w2")
    for t in (u1, u2):
        set_job_state(w2.fake["cluster_a"], t, "COMPLETED", end="2026-01-01T01:00:00")
    lock = pool2 / "PLACED.lock"
    # the holder is ONE process (a `flock <file> sleep` would hand the locked fd to its child, and terminating flock would release nothing)
    holder = subprocess.Popen([sys.executable, "-c", "import fcntl, sys, time; f = open(sys.argv[1], 'a+'); fcntl.flock(f.fileno(), fcntl.LOCK_EX); print('held', flush=True); time.sleep(300)",
                               str(lock)], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        proc = subprocess.Popen([sys.executable, str(LIB / "submit_twin.py"), "--config", str(w2.site / "config.env"), "--model", "m", "teardown-pool", "--group", GROUP],
                                env=w2.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        # its transaction (a child `bash -c 'mkdir -p … && flock -w 300 <lock> python3 -c …'`) is waiting on the lock: the record was read before it
        _wait_until(lambda: subprocess.run(["pgrep", "-P", str(proc.pid), "-f", "flock -w 300 "], capture_output=True).returncode == 0, "the teardown's flock on the group's lock", proc=proc)
        assert proc.poll() is None and not list(pool2.glob("*.STOP"))
        (pool2 / "PLACED.v1").write_text(json.dumps(p1))
        (pool2 / "PLACED").write_text(json.dumps(dict(p1, version=2, targets=[dict(t, job_id=str(int(t["job_id"]) + 100)) for t in p1["targets"]],
                                                       supersedes={"version": 1}, reason="fixture: the record moved while the teardown waited for the lock")))
    finally:
        holder.terminate(); holder.wait(timeout=30)
    out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 3, out
    assert f"teardown {GROUP}: STALE — attackers/PLACED on cluster_a is version 2, not the version 1 this teardown decided on (a placement committed in between: 2 target pair(s) added, 2 removed)" in out, out
    assert f"REFUSED: teardown-pool {GROUP}: attackers/PLACED on cluster_a moved under this teardown (version 1 when it was read, version 2 under the lock; target pairs added [[" in out, out
    assert "refused, nothing written" in out and "run this again against the current placement" in out and not list(pool2.glob("*.STOP")) and not _torn(w2), out
    # run again: it reads v2, which is the record under the lock -> STOP for both members, two torn_down
    r = w2.run("teardown-pool", "--group", GROUP)
    assert r.returncode == 0 and f"teardown {GROUP}: {pool2}/{k1}.STOP" in r.stdout and f"teardown {GROUP}: {pool2}/{k2}.STOP" in r.stdout and "STALE" not in r.stdout, r.stdout + r.stderr
    assert sorted(p.name for p in pool2.glob("*.STOP")) == sorted([f"{k1}.STOP", f"{k2}.STOP"]) and sorted(e["job_id"] for e in _torn(w2)) == sorted([k1, k2])
    # dry-run names the new condition and writes nothing
    r = w2.run("--dry-run", "teardown-pool", "--group", GROUP)
    assert r.returncode == 0 and r.stdout.count("or attackers/PLACED is no longer the record read)") == 2 and "DRY-RUN: ssh" not in r.stdout, r.stdout + r.stderr


def test_the_locked_programs_compare_the_placements_identity_after_the_marker_and_before_the_one_write(tmp_path):
    """The transactions themselves, over a bare pool root: the identity is the version AND the (unit, job) pairs; a moved, appeared or vanished
    record answers stale with nothing written and names the versions and the pair changes; a fresh marker is read first; a matching identity
    proceeds — the teardown to STOP, the retirement to its ordinary decision; no record expected against no record proceeds (the by-hand teardown
    of an unplaced group). Plus the grading host's side of the identity, Coordinator._placed_expect."""
    sys.path.insert(0, str(LIB))
    import submit_twin as st  # noqa: E402
    root = tmp_path / "attackers"; root.mkdir()
    def tx(prog, intent):
        r = subprocess.run([sys.executable, "-c", prog, str(root), json.dumps(intent)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        return json.loads(r.stdout.strip().splitlines()[-1])
    v1 = {"group": GROUP, "version": 1, "members": ["1", "2"], "targets": [{"logical_work_id": "u/a", "job_id": "11", "assigned_member_job_id": "1"},
                                                                          {"logical_work_id": "u/b", "job_id": "12", "assigned_member_job_id": "2"}]}
    v2 = {"group": GROUP, "version": 2, "members": ["2", "3"], "targets": [{"logical_work_id": "u/a", "job_id": "13", "assigned_member_job_id": "2"},
                                                                          {"logical_work_id": "u/b", "job_id": "14", "assigned_member_job_id": "3"}]}
    exp = st.Coordinator._placed_expect
    assert exp(None) == {"version": 0, "targets": []} and exp(v1) == {"version": 1, "targets": [["u/a", "11"], ["u/b", "12"]]}
    assert exp({"version": 3, "targets": [{"logical_work_id": "u/z"}, {"logical_work_id": "u/a", "job_id": 9}]}) == {"version": 3, "targets": [["u/a", "9"]]}   # a row without a job is no pair
    teardown = lambda expect, jobs=("1", "2", "3"): tx(st.TEARDOWN_TX_PY, {"group": GROUP, "jobs": list(jobs), "expect": expect, "utc": "2026-01-01T00:00:00Z"})
    retire = lambda expect, job="3", n_live=1: tx(st.RETIRE_TX_PY, {"group": GROUP, "job_id": job, "why": "test", "n_live_targets": n_live, "live_jobs": ["2", "3"], "expect": expect, "utc": "2026-01-01T00:00:00Z"})
    # moved (v1 expected, v2 standing): stale, the pair changes named, nothing written
    (root / "PLACED").write_text(json.dumps(v2))
    res = teardown(exp(v1))
    assert res == {"result": "stale", "expected_version": 1, "current_version": 2, "added": [["u/a", "13"], ["u/b", "14"]], "removed": [["u/a", "11"], ["u/b", "12"]], "marker": None, "left_s": None}, res
    res = retire(exp(v1))
    assert res["result"] == "stale_placement" and (res["expected_version"], res["current_version"]) == (1, 2) and res["added"] == [["u/a", "13"], ["u/b", "14"]], res
    assert not list(root.glob("*.STOP")) and not list(root.glob("*.RETIRE"))
    # the same version with other pairs is another record (the identity is version AND pairs); a record that appeared (none expected) and one that
    # vanished (v2 expected, none standing) are changes too
    res = teardown(exp(dict(v2, targets=v2["targets"][:1] + [dict(v2["targets"][1], job_id="15")])))
    assert res["result"] == "stale" and (res["expected_version"], res["current_version"]) == (2, 2) and res["added"] == [["u/b", "14"]] and res["removed"] == [["u/b", "15"]], res
    res = teardown(exp(None)); assert res["result"] == "stale" and (res["expected_version"], res["current_version"]) == (0, 2) and res["removed"] == [], res
    (root / "PLACED").unlink()
    res = teardown(exp(v2)); assert res["result"] == "stale" and (res["expected_version"], res["current_version"]) == (2, 0) and res["added"] == [], res
    res = retire(exp(v2)); assert res["result"] == "stale_placement" and res["current_version"] == 0, res
    assert not list(root.glob("*.STOP")) and not list(root.glob("*.RETIRE"))
    # an unreadable record is no record (what placed_record answers None for): none expected matches it; the by-hand teardown of an unplaced group proceeds
    (root / "PLACED").write_text("{not json"); res = teardown(exp(None), jobs=("1",)); assert res["result"] == "stopped" and res["stopped"] == ["1"], res
    (root / "1.STOP").unlink(); (root / "PLACED").unlink()
    res = teardown(exp(None), jobs=("1",)); assert res["result"] == "stopped" and res["stopped"] == ["1"] and res["marker_ignored"] is None, res
    (root / "1.STOP").unlink()
    # the marker is read FIRST: a fresh one holds a matching identity; cleared, the match proceeds — STOP for every job, the retirement to its
    # ordinary decision (no server record for job 3: not serving; survivors none; n_live > 0 -> refused_last, nothing written)
    (root / "PLACED").write_text(json.dumps(v2))
    (root / "PLACED.transition").write_text(json.dumps({"group": GROUP, "from_version": 2, "reason": "r", "lease_s": 7200, "epoch": int(time.time()), "utc": "u"}))
    assert teardown(exp(v2))["result"] == "held" and retire(exp(v2))["result"] == "held_transition" and not list(root.glob("*.STOP"))
    (root / "PLACED.transition").unlink()
    res = retire(exp(v2)); assert res["result"] == "refused_last" and not list(root.glob("*.RETIRE")), res
    res = retire(exp(v2), n_live=0); assert res["result"] == "retired" and (root / "3.RETIRE").exists(), res
    res = teardown(exp(v2)); assert res["result"] == "stopped" and res["stopped"] == ["1", "2", "3"] and sorted(p.name for p in root.glob("*.STOP")) == ["1.STOP", "2.STOP", "3.STOP"], res


def test_every_destructive_write_carries_the_record_its_caller_read_and_the_compare_sits_between_the_marker_and_the_write():
    """Static pins on the shape: in both programs the identity compare (`= stale_answer(`) follows the fresh-marker exit and precedes the one
    write, and exits before it; teardown_pool and retire_member put the caller's record into the intent; wait_pool passes its poll's record to
    both and skips the ratio on a stale teardown; enforce_pool_ratio stops on `stale_placement`; the by-hand commands read the record BEFORE
    their transaction (retire-member before its live-target count) and refuse on `stale`; no caller calls either without one."""
    sys.path.insert(0, str(LIB))
    import submit_twin as st  # noqa: E402
    src = (LIB / "submit_twin.py").read_text()
    for prog, write, result in ((st.TEARDOWN_TX_PY, 'job + ".STOP"', '"stale"'), (st.RETIRE_TX_PY, 'job + ".RETIRE"', '"stale_placement"')):
        i_fresh = prog.index('if tr_state == "fresh":'); i_stale = prog.index("= stale_answer("); i_write = prog.index(write)
        assert prog.index("def placed_identity(root):") < i_fresh < i_stale < i_write and prog.count(write) == 1, prog
        assert result in prog[i_stale:i_write] and "sys.exit(0)" in prog[i_stale:i_write] and 'intent["expect"]' in prog[i_stale:i_write], prog
    body = src[src.index("    def teardown_pool(self, group: str, *, placed: Optional[dict]) -> dict:"):src.index("    def _placed_expect(")]
    assert "expect = self._placed_expect(placed)" in body and '"expect": expect' in body and 'res.get("result") == "stale"' in body and '"result": "stale"' in body
    ret = src[src.index("    def retire_member(self, group: str"):src.index("    def enforce_pool_ratio(")]
    assert "placed: Optional[dict]" in ret.splitlines()[0] and '"expect": self._placed_expect(placed)' in ret and 'result == "stale_placement"' in ret
    ratio = src[src.index("    def enforce_pool_ratio("):src.index("    def mark_placed(")]
    assert "*, placed: Optional[dict]" in ratio.splitlines()[0] and "placed=placed, live_jobs=live_jobs" in ratio and 'res == "stale_placement":' in ratio
    assert "break" in ratio[ratio.index('res == "stale_placement":'):]
    wp = src[src.index("    def wait_pool(self, group: str"):src.index("#: the placement-transition marker as the destructive transactions read it")]
    assert "self.teardown_pool(group, placed=placed)" in wp and "self.enforce_pool_ratio(group, n_live_targets=len(live), placed=placed)" in wp
    assert "stale_poll = True" in wp and "elif stale_poll:" in wp and wp.index("elif stale_poll:") < wp.index("self.enforce_pool_ratio(")
    cli_td = src[src.index('    if a.cmd == "teardown-pool":'):src.index('    if a.cmd == "mark-placed":')]
    assert "placed = co.placed_record(label, a.group)" in cli_td and "co.teardown_pool(a.group, placed=placed)" in cli_td and 'res.get("result") == "stale":' in cli_td
    assert cli_td.index("placed = co.placed_record(") < cli_td.index("co.teardown_pool(") and "raise Refused" in cli_td[cli_td.index('res.get("result") == "stale":'):]
    cli_rm = src[src.index('    if a.cmd == "retire-member":'):src.index('    if a.cmd == "wait-pool":')]
    assert cli_rm.index("placed = co.placed_record(") < cli_rm.index("n_live = 0") < cli_rm.index("n_live_targets=n_live, placed=placed)") and 'res == "stale_placement":' in cli_rm
    assert "raise Refused" in cli_rm[cli_rm.index('res == "stale_placement":'):]
    assert "teardown_pool(group)" not in src and "teardown_pool(a.group)" not in src and ", n_live_targets=n_live)" not in src
