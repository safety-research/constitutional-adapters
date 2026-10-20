"""The funnel side of the attacker pool member's lifecycle (31b): `mark-placed` writes the group's attackers/PLACED once every target is
bound (and refuses by name while one is not); a member that wrote attackers/<job>.SELF_EXIT is a teardown to `--resubmit --walk` (its
COMPLETED is never a result) and `teardown-pool` records it without a second STOP. Over the fake clusters of test_submit_twin's World."""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import job_ids, set_job_state  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402

GROUP = "mt-u0-cluster_a"
POOL = "scripts/sdfbase/full22/attacker_pool.sbatch"


def _members(w: World, n=2):
    for i in range(1, n + 1):
        r = w.run("submit", "--unit", f"multiturn.attacker{i}", "--lwid", f"m/u0/multiturn/{GROUP}/attackers/{i}", "--script", POOL, "--name", f"f22-atk-m-{i}",
                  "--args", SHA, "--clusters", "cluster_a", "--kind", "attacker", "--group", GROUP, "--wave-sha", SHA)
        assert r.returncode == 0, r.stdout + r.stderr
    return [str(j) for j in job_ids(w.fake["cluster_a"])]          # the ledger records job ids as strings


def _target(w: World, unit: str, group_shards: str):
    lw = f"m/u0/multiturn/{unit}"
    r = w.run("submit", "--unit", f"multiturn.{unit.replace('/', '.')}", "--lwid", lw, "--script", "scripts/sdfbase/full22/target_attacks.sbatch", "--name", f"f22-att-m-{unit.replace('/', '-')}",
              "--args", f"m u0 crescendo {SHA} 9 {group_shards}", "--clusters", "cluster_a", "--group", GROUP, "--wave-sha", SHA)
    assert r.returncode == 0, r.stdout + r.stderr
    return lw


def test_mark_placed_writes_the_groups_placed_record_once_every_target_is_bound_and_refuses_while_one_is_not(tmp_path):
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    _members(w)
    lw1 = _target(w, "crescendo/g012", "0,1,2")
    lw2 = f"m/u0/multiturn/crescendo/g34"
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 3 and "is unbound — the group is not fully placed" in r.stderr, r.stdout + r.stderr
    placed = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers/PLACED"
    assert not placed.exists()
    _target(w, "crescendo/g34", "3,4")
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and f"placed {GROUP}: 2 target(s) bound on cluster_a" in r.stdout, r.stdout + r.stderr
    doc = json.loads(placed.read_text())
    jobs = [str(j) for j in job_ids(w.fake["cluster_a"])]
    assert doc["group"] == GROUP and doc["n_targets"] == 2 and [t["logical_work_id"] for t in doc["targets"]] == [lw1, lw2] and {t["job_id"] for t in doc["targets"]} == set(jobs[2:])
    # 31b-fix: the IMMUTABLE assignment at placement — target i -> member i mod n over the group's recorded members
    assert doc["members"] == jobs[:2] and [t["assigned_member_job_id"] for t in doc["targets"]] == jobs[:2]
    ev = [e for e in w.ledger() if e["event"] == "placement_complete"]
    assert len(ev) == 1 and ev[0]["group"] == GROUP and ev[0]["cluster"] == "cluster_a" and [t["job_id"] for t in ev[0]["targets"]] == jobs[2:] and ev[0]["members"] == jobs[:2]
    # the placement is IMMUTABLE — a repeat returns the record unchanged (nothing written, no second event)
    before = placed.read_bytes()
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}"); assert r.returncode == 0 and "already placed, unchanged" in (r.stdout + r.stderr), r.stdout + r.stderr
    assert placed.read_bytes() == before and len([e for e in w.ledger() if e["event"] == "placement_complete"]) == 1 and doc["version"] == 1
    # a changed roster (a third target) refuses by name without --supersede; with it, a versioned transition: the previous record archived,
    # the unchanged pairs keep their member, the new pair is assigned, the ledger records the change with its reason
    lw3 = _target(w, "pair/g012", "0,1,2"); j3 = str(job_ids(w.fake["cluster_a"])[-1])
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}")
    assert r.returncode == 3 and "a placement is immutable" in r.stderr and "--supersede" in r.stderr and placed.read_bytes() == before, r.stdout + r.stderr
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}", "--supersede"); assert r.returncode == 2 and "needs --reason" in r.stderr, r.stderr
    assert placed.read_bytes() == before
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}", "--supersede", "--reason", "pair/g012 submitted on the second pass")
    assert r.returncode == 0 and "SUPERSEDED version 1 -> 2" in (r.stdout + r.stderr), r.stdout + r.stderr
    doc2 = json.loads(placed.read_text())
    assert doc2["version"] == 2 and doc2["supersedes"]["version"] == 1 and doc2["reason"] == "pair/g012 submitted on the second pass" and doc2["n_targets"] == 3
    assert [t["assigned_member_job_id"] for t in doc2["targets"]] == jobs[:2] + [jobs[0]] and doc2["changes"] == {"added": [[lw3, j3]], "removed": [], "kept": 2, "members_before": jobs[:2]}
    assert json.loads((placed.parent / "PLACED.v1").read_text()) == json.loads(before.decode())
    sup = [e for e in w.ledger() if e["event"] == "placement_superseded"]
    assert len(sup) == 1 and sup[0]["from_version"] == 1 and sup[0]["to_version"] == 2 and sup[0]["changes"]["added"] == [[lw3, j3]] and sup[0].get("logical_work_id") is None
    # a group with no recorded member cannot be placed (nothing to assign): refused by name, nothing written
    (tmp_path / "w3").mkdir(); w3 = World(tmp_path / "w3"); (w3.fake["cluster_a"] / "policy").write_text("start_on_submit\n"); lw3 = _target(w3, "crescendo/g012", "0,1,2")
    r = w3.run("mark-placed", "--group", GROUP, "--targets", lw3); assert r.returncode == 3 and "no attacker members recorded" in r.stderr, r.stdout + r.stderr
    assert not (w3.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers/PLACED").exists()
    # dry-run writes nothing new
    r = w.run("--dry-run", "mark-placed", "--group", GROUP, "--targets", lw1); assert r.returncode == 0 and r.stdout.startswith("DRY-RUN: mark-placed"), r.stdout + r.stderr
    assert w.run("mark-placed", "--group", GROUP, "--targets", "").returncode != 0


def test_a_self_exited_member_is_a_teardown_to_the_family_pass_and_teardown_pool_records_it_without_a_stop(tmp_path):
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    j1, j2 = _members(w)
    pool = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers"; pool.mkdir(parents=True, exist_ok=True)
    # member 1 closed itself (its last consumer ended) and COMPLETED; member 2 COMPLETED with no sentinel of any kind
    (pool / f"{j1}.SELF_EXIT").write_text(json.dumps({"job_id": j1, "reason": "exit_last_consumer", "consumers": "9001", "utc": "2026-01-01T02:00:00Z"}))
    for j in (j1, j2):
        set_job_state(w.fake["cluster_a"], j, "COMPLETED", end="2026-01-01T02:00:00")
    w.spool("cluster_a")
    # the family pass (--resubmit --walk): the self-exited member is fenced like a torn-down one (re-minted), the plain COMPLETED one is REFUSED as a result
    r = w.run("submit", "--unit", "multiturn.attacker1", "--lwid", f"m/u0/multiturn/{GROUP}/attackers/1", "--script", POOL, "--name", "f22-atk-m-1", "--args", SHA, "--clusters", "cluster_a",
              "--kind", "attacker", "--group", GROUP, "--wave-sha", SHA, "--resubmit", "--walk")
    assert r.returncode == 0 and "SELF-EXITED" in (r.stdout + r.stderr) and "its COMPLETED is the teardown, not a result" in (r.stdout + r.stderr), r.stdout + r.stderr
    assert [e for e in w.ledger() if e["event"] == "torn_down" and e.get("self_exit") and e["job_id"] == j1]
    r = w.run("submit", "--unit", "multiturn.attacker2", "--lwid", f"m/u0/multiturn/{GROUP}/attackers/2", "--script", POOL, "--name", "f22-atk-m-2", "--args", SHA, "--clusters", "cluster_a",
              "--kind", "attacker", "--group", GROUP, "--wave-sha", SHA, "--resubmit", "--walk")
    assert r.returncode == 3 and "COMPLETED task(s)" in r.stderr and "their runs are the unit's results" in r.stderr, r.stdout + r.stderr
    # teardown: the self-exited member gets no STOP (recorded), the other does
    r = w.run("teardown-pool", "--group", GROUP)
    assert r.returncode == 0 and "SELF_EXIT already there" in (r.stdout + r.stderr), r.stdout + r.stderr
    names = sorted(p.name for p in pool.glob("*"))
    assert f"{j1}.SELF_EXIT" in names and f"{j2}.STOP" in names and f"{j1}.STOP" not in names and "STOP" not in names, names
    torn = [e for e in w.ledger() if e["event"] == "torn_down"]
    assert {e["job_id"] for e in torn} >= {j1, j2} and all(e.get("self_exit") for e in torn if e["job_id"] == j1)


def test_wait_pool_polls_sacct_and_tears_the_pool_down_the_poll_after_the_last_target_is_terminal(tmp_path):
    """The group-level backstop in the repo: sacct, never squeue absence; ≤ 60 s polls; one poll of idle at most."""
    import subprocess, time
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    j1, j2 = _members(w)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4")
    t1, t2 = job_ids(w.fake["cluster_a"])[2:]
    pool = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers"
    # dry-run says what it would do; a group with no bound target refuses
    r = w.run("--dry-run", "wait-pool", "--group", GROUP, "--poll-s", "0.2"); assert r.returncode == 0 and r.stdout.startswith("DRY-RUN: wait-pool"), r.stdout + r.stderr
    r = w.run("wait-pool", "--group", "mt-u0-nowhere", "--poll-s", "0.2"); assert r.returncode == 2 and "binds no target" in r.stderr, r.stderr
    # a PARTIAL group (no PLACED yet — a --units subset, a refused mark-placed) never tears the pool down, however
    # idle: every bound target terminal, the backstop says so and waits, and at its bound refuses by name; nothing stopped
    set_job_state(w.fake["cluster_a"], t1, "COMPLETED", end="2026-01-01T00:30:00"); set_job_state(w.fake["cluster_a"], t2, "COMPLETED", end="2026-01-01T00:40:00")
    r = w.run("wait-pool", "--group", GROUP, "--poll-s", "0.2", "--max-hours", "0.0003")
    assert r.returncode == 3 and "placement never completed" in r.stderr and "placement is NOT complete" in (r.stdout + r.stderr), r.stdout + r.stderr
    assert not list(pool.glob("*.STOP")) and not [e for e in w.ledger() if e["event"] == "torn_down"]
    set_job_state(w.fake["cluster_a"], t1, "RUNNING"); set_job_state(w.fake["cluster_a"], t2, "RUNNING")
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}"); assert r.returncode == 0, r.stdout + r.stderr
    # both targets live and the placement complete: the waiter waits; the first ends, the waiter still waits; the LAST ends -> teardown within one poll
    proc = subprocess.Popen([sys.executable, str(Path(w.run.__globals__["LIB"]) / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m",
                             "wait-pool", "--group", GROUP, "--poll-s", "0.3", "--max-hours", "0.05"], env=w.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    time.sleep(1.0); assert proc.poll() is None and not list(pool.glob("*.STOP"))
    set_job_state(w.fake["cluster_a"], t1, "COMPLETED", end="2026-01-01T01:00:00")
    time.sleep(1.0); assert proc.poll() is None and not list(pool.glob("*.STOP"))
    set_job_state(w.fake["cluster_a"], t2, "FAILED", end="2026-01-01T01:10:00")
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0 and "placement complete (2 placed target(s)) and every target terminal on sacct (2 target(s))" in out, out
    assert sorted(p.name for p in pool.glob("*.STOP")) == [f"{j1}.STOP", f"{j2}.STOP"]
    assert len([e for e in w.ledger() if e["event"] == "torn_down"]) == 2
    # the bound (max hours) refuses by name instead of waiting forever
    (tmp_path / "w2").mkdir(); w2 = World(tmp_path / "w2"); (w2.fake["cluster_a"] / "policy").write_text("start_on_submit\n"); _members(w2); _target(w2, "crescendo/g012", "0,1,2")
    r = w2.run("wait-pool", "--group", GROUP, "--poll-s", "0.2", "--max-hours", "0.0002")
    assert r.returncode == 3 and "still live after" in r.stderr and "not torn down" in r.stderr, r.stderr


def test_06_arms_the_backstop_only_after_mark_placed_succeeded():
    """The arm line printed on a partial group invited stopping the pool when the subset ended."""
    sh = (Path(__file__).resolve().parents[1] / "reproduce" / "06_eval_waves.sh").read_text()
    m_if = re.search(r'mark-placed --group "\$g" --targets "\$\{GROUP_TARGETS\[\$g\]\}"[^\n]*; then', sh)   # 6cf973aa4: the move-group knob appends ${MARK_EXTRA[@]+"${MARK_EXTRA[@]}"} before `; then`
    assert m_if, "06 has no `if ... mark-placed --group \"$g\" --targets \"${GROUP_TARGETS[$g]}\" ...; then` line"
    i_if = m_if.start()
    i_arm = sh.index('echo "followup: arm the group-level backstop now:')
    i_else = sh.index("do NOT arm wait-pool before it succeeds")
    assert i_if < i_arm < i_else and sh.count('echo "followup: arm the group-level backstop now:') == 1
    assert "arm wait-pool only after mark-placed has succeeded" in sh


def test_two_concurrent_mark_placed_serialize_on_the_groups_lock_and_never_publish_two_first_records(tmp_path):
    """The read/compare/version/archive/write is ONE program under flock on the group's shared root.
    Two invocations racing on a fresh group: exactly one places (one placement_complete), the other reads the published record — unchanged
    when it brought the same roster, refused when it brought another; never two first records, never the same version twice."""
    import subprocess
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    _members(w)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4"); lw3 = _target(w, "pair/g012", "0,1,2")
    pool = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers"
    argv = [sys.executable, str(Path(w.run.__globals__["LIB"]) / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m", "mark-placed", "--group", GROUP, "--targets"]
    procs = [subprocess.Popen([*argv, f"{lw1},{lw2}"], env=w.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for _ in range(2)]
    outs = [p.communicate(timeout=120)[0] for p in procs]; rcs = [p.returncode for p in procs]
    assert rcs == [0, 0], (rcs, outs)
    assert sum("already placed, unchanged" in o for o in outs) == 1 and sum("target(s) bound on cluster_a" in o for o in outs) == 1, outs
    doc = json.loads((pool / "PLACED").read_text()); assert doc["version"] == 1 and doc["n_targets"] == 2 and not (pool / "PLACED.v1").exists()
    assert len([e for e in w.ledger() if e["event"] == "placement_complete"]) == 1 and not [e for e in w.ledger() if e["event"] == "placement_superseded"]
    # a second fresh group, raced by two DIFFERENT rosters: one places, the other is refused against the published record; one version, no archive
    (tmp_path / "w2").mkdir(); w2 = World(tmp_path / "w2"); (w2.fake["cluster_a"] / "policy").write_text("start_on_submit\n"); _members(w2)
    m1 = _target(w2, "crescendo/g012", "0,1,2"); m2 = _target(w2, "crescendo/g34", "3,4"); m3 = _target(w2, "pair/g012", "0,1,2")
    argv2 = [sys.executable, str(Path(w2.run.__globals__["LIB"]) / "submit_twin.py"), "--config", str(w2.site / "config.env"), "--model", "m", "mark-placed", "--group", GROUP, "--targets"]
    procs = [subprocess.Popen([*argv2, f"{m1},{m2}"], env=w2.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True),
             subprocess.Popen([*argv2, f"{m1},{m2},{m3}"], env=w2.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)]
    outs = [p.communicate(timeout=120)[0] for p in procs]; rcs = [p.returncode for p in procs]
    assert sorted(rcs) == [0, 3], (rcs, outs)
    assert sum("a placement is immutable" in o for o in outs) == 1 and sum("target(s) bound on cluster_a" in o for o in outs) == 1, outs
    pool2 = w2.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers"
    doc = json.loads((pool2 / "PLACED").read_text()); assert doc["version"] == 1 and doc["n_targets"] in (2, 3) and not (pool2 / "PLACED.v1").exists()
    assert len([e for e in w2.ledger() if e["event"] == "placement_complete"]) == 1
    assert (pool2 / "PLACED.lock").exists()                                                          # the group's lock, on the shared root
