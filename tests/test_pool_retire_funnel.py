"""31d: the funnel RETIRES the surplus of a shared attacker pool — `wait-pool` enforces one member per three LIVE targets each poll (fewest bound
consumers first), `retire-member` marks one by hand; a retiring member withdraws, drains and self-exits, and its drivers re-bind."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import job_ids, set_job_state  # noqa: E402
from test_pool_member_lifecycle_funnel import GROUP, World, _members, _target  # noqa: E402

def test_wait_pool_retires_the_surplus_member_with_the_fewest_consumers_and_retire_member_marks_one_by_hand(tmp_path):
    """31d: one member per three LIVE targets; the surplus is marked RETIRING (attackers/<job>.RETIRE + ledger `retiring`), fewest bound
    consumers first; the member then withdraws, drains and self-exits, and its drivers re-bind (the client's failover)."""
    import os, shlex, subprocess
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    j1, j2 = _members(w)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); t1 = job_ids(w.fake["cluster_a"])[-1]
    root = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}"; pool = root / "attackers"; pool.mkdir(parents=True, exist_ok=True)
    for j, url in ((j1, "http://n0:8707"), (j2, "http://n1:8708")):
        (pool / f"{j}.json").write_text(json.dumps({"url": url, "job_id": j}))
        _lease(pool, j, url)                                   # 31d-fix-3: the fresh ok health lease every serving member carries
    bind = root / "serving" / f"attacks_m_u0_crescendo_{t1}"; bind.mkdir(parents=True)
    (bind / "attacker_binding.json").write_text(json.dumps({"declared": 2, "alive": 2, "serving": 2, "urls": ["http://n0:8707"]}))     # the one target binds member 1 only
    r = w.run("mark-placed", "--group", GROUP, "--targets", lw1); assert r.returncode == 0, r.stdout + r.stderr                        # placement complete (31b-fix): assigns t1 -> member 1
    # The waiter's poll clock: a poll that retires nothing prints nothing, so the test fronts the fake `sacct` with a shim (Remote.sh in
    # `local:` mode inherits this PATH) that logs every call — the roster's `sacct -j <t1>` opens EVERY poll, so its count is the number of
    # polls begun; waiting on it, not on wall time, is what makes "another poll ran" a fact rather than a 1 s guess on a loaded box
    shim = tmp_path / "shim"; shim.mkdir(); sacct_log = tmp_path / "sacct.log"
    (shim / "sacct").write_text(f'#!/bin/bash\necho "$*" >> {shlex.quote(str(sacct_log))}\nexec {shlex.quote(str(w.bin / "sacct"))} "$@"\n'); (shim / "sacct").chmod(0o755)
    ticks = lambda: sum(ln.split()[-2:] == ["-j", str(t1)] for ln in (sacct_log.read_text().splitlines() if sacct_log.exists() else []))
    proc = subprocess.Popen([sys.executable, str(Path(w.run.__globals__["LIB"]) / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m",
                             "wait-pool", "--group", GROUP, "--poll-s", "0.3", "--max-hours", "0.05"],
                            env=w.env(PATH=f"{shim}:{w.bin}:{os.environ['PATH']}"), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    # The mark is TWO writes — attackers/<job>.RETIRE by the transaction on the cluster, then the ledger `retiring` once it returns — so the
    # test waits for the LATER one (the file is its precondition); a wait on the file alone reads the ledger before the funnel appends it
    retiring = _wait_until(lambda: _retiring(w), "a ledger `retiring` event from wait-pool", proc=proc)
    assert (pool / f"{j2}.RETIRE").exists() and not (pool / f"{j1}.RETIRE").exists()           # two members for one live target: the unbound one retires
    assert len(retiring) == 1 and retiring[0]["job_id"] == j2 and "1 live target" in retiring[0]["why"] and "1 allowed" in retiring[0]["why"]
    # marked ONCE, not every poll: two roster ticks after the mark bracket one COMPLETE poll (roster -> ratio check -> sleep) that ran with
    # the member already RETIRING in the ledger and marked nothing
    c0 = ticks(); _wait_until(lambda: ticks() >= c0 + 2, f"two wait-pool polls after the mark (sacct -j {t1} calls {c0} -> {c0 + 2})", proc=proc)
    assert len(_retiring(w)) == 1 and not (pool / f"{j1}.RETIRE").exists()
    set_job_state(w.fake["cluster_a"], t1, "COMPLETED", end="2026-01-01T01:00:00")
    out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 0 and "every target terminal" in out, out
    assert sorted(p.name for p in pool.glob("*.STOP")) == [f"{j1}.STOP", f"{j2}.STOP"]       # the backstop still stops both at the end
    assert [e["job_id"] for e in _retiring(w)] == [j2]                                        # and still the one mark once the waiter has ended
    # retire-member by hand: refuses an unknown job, marks a known one
    (tmp_path / "w2").mkdir(); w2 = World(tmp_path / "w2"); (w2.fake["cluster_a"] / "policy").write_text("start_on_submit\n"); k1, k2 = _members(w2)
    r = w2.run("retire-member", "--group", GROUP, "--job", "999999"); assert r.returncode == 2 and "no member with job" in r.stderr, r.stderr
    r = w2.run("retire-member", "--group", GROUP, "--job", k2, "--why", "operator"); assert r.returncode == 0 and "marked RETIRING" in (r.stdout + r.stderr), r.stdout + r.stderr
    assert (w2.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers/{k2}.RETIRE").exists()
    assert [e["job_id"] for e in w2.ledger() if e["event"] == "retiring"] == [k2]


def _world2(tmp_path):
    """Two members, two targets placed (t1 -> member 1, t2 -> member 2 by the round-robin), member records on the shared root."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    j1, j2 = _members(w)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4"); t1, t2 = (str(j) for j in job_ids(w.fake["cluster_a"])[2:])
    root = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}"; pool = root / "attackers"; pool.mkdir(parents=True, exist_ok=True)
    for j, url in ((j1, "http://n0:8707"), (j2, "http://n1:8708")):
        (pool / f"{j}.json").write_text(json.dumps({"url": url, "job_id": j}))
        _lease(pool, j, url)                                   # 31d-fix-3: a serving member carries the fresh ok health lease its sbatch writes
    return w, (j1, j2), (lw1, lw2), (t1, t2), root, pool


def _lease(pool, job, url, *, ok=True, age_s=0, lease_s=180):
    """The member's own health lease (attacker_pool_lifecycle.sh f22_pool_health_write): fresh and ok unless the test says otherwise."""
    import time
    (pool / f"{job}.health").write_text(json.dumps({"job_id": job, "url": url, "ok": ok, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - age_s)), "lease_s": lease_s}))


def _bind(root, job, url):
    d = root / "serving" / f"attacks_m_u0_crescendo_{job}"; d.mkdir(parents=True, exist_ok=True)
    (d / "attacker_binding.json").write_text(json.dumps({"declared": 2, "alive": 2, "serving": 2, "urls": [url]}))


def _wait_until(pred, what: str, *, proc=None, timeout_s=60.0, poll_s=0.05):
    """Return `pred()` as soon as it is truthy. Fails naming `what` when the deadline passes — or at once when `proc` (the wait-pool under
    test) exited first — with the process's output in the message. The deadline bounds the FAILURE path only, so it is generous for a loaded
    box; the happy path returns on the first true poll, where a fixed sleep either races the event or idles past it."""
    import time
    t0 = time.monotonic()
    while True:
        got = pred()
        if got:
            return got
        exited = proc is not None and proc.poll() is not None
        if exited or time.monotonic() - t0 > timeout_s:
            out = ""
            if proc is not None:
                if not exited:
                    proc.kill()
                out = proc.communicate(timeout=30)[0] or ""
            raise AssertionError((f"wait-pool exited (rc {proc.returncode}) before" if exited else f"waited {timeout_s:g}s for") + f" {what}\n{out}")
        time.sleep(poll_s)


def _retiring(w):
    """The ledger's `retiring` events. A line still mid-append (Ledger.append is one O_APPEND write; a reader polling the file can land inside
    it) reads as none yet rather than an error, so the ledger is safe to poll while the waiter writes it."""
    try:
        return [e for e in w.ledger() if e["event"] == "retiring"]
    except ValueError:
        return []


def _wait_pool(w, poll="0.3", hours="0.05"):
    import subprocess
    return subprocess.Popen([sys.executable, str(Path(w.run.__globals__["LIB"]) / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m",
                             "wait-pool", "--group", GROUP, "--poll-s", poll, "--max-hours", hours], env=w.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def test_the_ratio_retires_only_a_member_with_no_live_consumer_and_the_one_member_survivor_keeps_serving(tmp_path):
    """Two members each assigned one target; the LOWER job's target finished, the HIGHER's still runs — both scored 1
    historical consumer and the tie retired the HIGHER (serving) member, whose own lifecycle then left: nothing served the live target.
    Now a historical assignment to a finished target is not a consumer: the idle member retires, the serving one never does."""
    w, (j1, j2), (lw1, lw2), (t1, t2), root, pool = _world2(tmp_path)
    assert w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}").returncode == 0
    doc = json.loads((pool / "PLACED").read_text()); assert [t["assigned_member_job_id"] for t in doc["targets"]] == [j1, j2]
    _bind(root, t1, "http://n0:8707"); _bind(root, t2, "http://n1:8708")
    set_job_state(w.fake["cluster_a"], t1, "COMPLETED", end="2026-01-01T01:00:00")          # the lower member's target finished; the higher's runs
    proc = _wait_pool(w)
    ret = _wait_until(lambda: _retiring(w), "a ledger `retiring` event from wait-pool", proc=proc)   # the ledger event, which the RETIRE file precedes
    assert (pool / f"{j1}.RETIRE").exists() and not (pool / f"{j2}.RETIRE").exists(), sorted(p.name for p in pool.iterdir())   # the IDLE member retires, never the serving one
    assert len(ret) == 1 and ret[0]["job_id"] == j1 and "serves no live target" in ret[0]["why"] and ret[0]["survivors"] == [j2]
    set_job_state(w.fake["cluster_a"], t2, "COMPLETED", end="2026-01-01T02:00:00")
    out, _ = proc.communicate(timeout=60); assert proc.returncode == 0 and "tearing the pool down" in out, out
    # both members serving live targets and the ratio exceeded (two members, one target each live): nothing retires, the log says so
    (tmp_path / "w2").mkdir(); w2, (k1, k2), (m1, m2), (u1, u2), root2, pool2 = _world2(tmp_path / "w2")
    assert w2.run("mark-placed", "--group", GROUP, "--targets", f"{m1},{m2}").returncode == 0
    _bind(root2, u1, "http://n0:8707"); _bind(root2, u2, "http://n1:8708")
    set_job_state(w2.fake["cluster_a"], u1, "COMPLETED", end="2026-01-01T01:00:00")        # one target left: 1 allowed, 2 members — but member 2 serves it …
    _bind(root2, u2, "http://n0:8707")                                                        # … and member 1 serves it too (a failover): both busy
    proc = _wait_pool(w2, hours="0.0006"); out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 3 and "still serve live targets" in out and not list(pool2.glob("*.RETIRE")), out


def test_retire_member_refuses_the_groups_last_serving_member_while_a_target_is_live_and_two_concurrent_retirers_leave_one(tmp_path):
    """`retire-member` accepted the sole member with live targets, and concurrent retirers deciding on stale rosters
    could withdraw every endpoint. Now the check and the write run under the group's lock, and the last serving member is never withdrawn
    while a target is live."""
    import subprocess
    w, (j1, j2), (lw1, lw2), (t1, t2), root, pool = _world2(tmp_path)
    _bind(root, t1, "http://n0:8707"); _bind(root, t2, "http://n1:8708")
    argv = [sys.executable, str(Path(w.run.__globals__["LIB"]) / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m", "retire-member", "--group", GROUP, "--why", "operator", "--job"]
    procs = [subprocess.Popen([*argv, j], env=w.env(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) for j in (j1, j2)]
    outs = [p.communicate(timeout=120)[0] for p in procs]; rcs = [p.returncode for p in procs]
    assert sorted(rcs) == [0, 3], (rcs, outs)
    assert sum("marked RETIRING" in o for o in outs) == 1 and sum("LAST serving member" in o for o in outs) == 1, outs
    assert len(list(pool.glob("*.RETIRE"))) == 1 and len([e for e in w.ledger() if e["event"] == "retiring"]) == 1
    survivor = j2 if (pool / f"{j1}.RETIRE").exists() else j1
    r = w.run("retire-member", "--group", GROUP, "--job", survivor); assert r.returncode == 3 and "LAST serving member" in r.stderr, r.stdout + r.stderr   # the sole member, targets live
    for t in (t1, t2):
        set_job_state(w.fake["cluster_a"], t, "COMPLETED", end="2026-01-01T03:00:00")
    r = w.run("retire-member", "--group", GROUP, "--job", survivor); assert r.returncode == 0 and "marked RETIRING" in (r.stdout + r.stderr), r.stdout + r.stderr   # no live target: allowed
    assert len(list(pool.glob("*.RETIRE"))) == 2
    r = w.run("retire-member", "--group", GROUP, "--job", survivor); assert r.returncode == 0 and "already RETIRING" in (r.stdout + r.stderr)      # idempotent
    assert len([e for e in w.ledger() if e["event"] == "retiring"]) == 2


def test_wait_pool_enforces_no_ratio_before_the_placement_is_complete(tmp_path):
    """Without PLACED a partial group's finished subset collapsed a pool to one member before the remaining targets
    were submitted. Ratio enforcement waits for the placement."""
    w, (j1, j2), (lw1, lw2), (t1, t2), root, pool = _world2(tmp_path)
    _bind(root, t1, "http://n0:8707")
    for t in (t1, t2):
        set_job_state(w.fake["cluster_a"], t, "COMPLETED", end="2026-01-01T01:00:00")       # every BOUND target finished; no PLACED (more targets may come)
    proc = _wait_pool(w, hours="0.0006"); out, _ = proc.communicate(timeout=60)
    assert proc.returncode == 3 and "ratio enforcement waits for the placement" in out and "placement never completed" in out, out
    assert not list(pool.glob("*.RETIRE")) and not [e for e in w.ledger() if e["event"] == "retiring"]


def test_a_pending_member_is_not_a_surviving_server_and_a_dead_record_is_not_either(tmp_path):
    """The last-server check counted `<job>.pending` as serving, but a pending member is still booting (~22 min)
    and only `<job>.json` exposes a usable URL; with A healthy and B pending, retiring A left no reachable attacker. A surviving server is a
    `.json` record whose job is live on sacct."""
    w, (j1, j2), (lw1, lw2), (t1, t2), root, pool = _world2(tmp_path)
    _bind(root, t1, "http://n0:8707"); _bind(root, t2, "http://n0:8707")
    (pool / f"{j2}.json").unlink(); (pool / f"{j2}.pending").write_text(json.dumps({"job_id": j2}))        # B: pending, not yet serving
    r = w.run("retire-member", "--group", GROUP, "--job", j1); assert r.returncode == 3 and "LAST serving member" in r.stderr and "pending" in (r.stdout + r.stderr), r.stdout + r.stderr
    assert not (pool / f"{j1}.RETIRE").exists()
    (pool / f"{j2}.json").write_text(json.dumps({"url": "http://n1:8708", "job_id": j2})); _lease(pool, j2, "http://n1:8708")   # B came up (record + fresh lease): A may retire
    r = w.run("retire-member", "--group", GROUP, "--job", j1); assert r.returncode == 0 and "marked RETIRING" in (r.stdout + r.stderr) and (pool / f"{j1}.RETIRE").exists(), r.stdout + r.stderr
    # a `.json` record whose job ENDED on sacct is not a server either: with B's job terminal, retiring A (the only live one) is refused
    (tmp_path / "w2").mkdir(); w2, (k1, k2), (m1, m2), (u1, u2), root2, pool2 = _world2(tmp_path / "w2")
    _bind(root2, u1, "http://n0:8707")
    set_job_state(w2.fake["cluster_a"], k2, "FAILED", end="2026-01-01T01:00:00")                              # B's record lingers, its job is gone
    r = w2.run("retire-member", "--group", GROUP, "--job", k1); assert r.returncode == 3 and "LAST serving member" in r.stderr and "not live on sacct" in (r.stdout + r.stderr), r.stdout + r.stderr
