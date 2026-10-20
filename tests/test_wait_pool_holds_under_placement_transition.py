"""The placement fence over the pool backstop: fencing the MEMBER's own
exit behind attackers/PLACED.transition left the group-level backstop never reading the marker. An armed `wait-pool` polling while a --resubmit
pass re-placed the group could (a) see every OLD target terminal after the pass had re-minted the member but before it re-minted the targets,
and `teardown_pool` wrote STOP for the replacement; or (b) with some targets still live, retire the replacement through the ratio — v1 assigned
it nothing, so it had no live consumer. Either way the supersession then assigned targets to a server that was gone. Now the STOP writes are one
program under the group's lock that reads the marker first (TEARDOWN_TX_PY), the retirement transaction reads it under the same lock
(RETIRE_TX_PY `held_transition`), a fresh marker HOLDS both with nothing written and the waiter polls on; the by-hand `teardown-pool` and
`retire-member` are refused by name; a lapsed or unreadable marker is ignored and said so (the lease is the bound), and at its bound the waiter
refuses by name instead of stopping a group whose placement is still changing. Over the fake clusters of test_submit_twin's World, with the
REAL funnel subcommands as 06's --resubmit pass runs them (open -> re-mint the member -> re-mint the targets -> mark-placed --supersede)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, job_ids, set_job_state  # noqa: E402
from test_pool_dead_member_torn_down import _walk_target  # noqa: E402
from test_pool_member_lifecycle_funnel import GROUP, _members, _target  # noqa: E402
from test_pool_retire_funnel import _bind, _lease, _retiring  # noqa: E402
from test_pool_self_exit_skip import _walk  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402

REASON = f"06 --resubmit pass on {SHA}"


def _placed_world(tmp_path):
    """Two serving members (records + fresh health leases), two targets placed on them (t1 -> member 1, t2 -> member 2) and bound to their
    URLs; the registry spool and inbox the targets' re-mint proof reads. The shape of test_pool_retire_funnel's _world2 after mark-placed."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    j1, j2 = _members(w)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4"); t1, t2 = (str(j) for j in job_ids(w.fake["cluster_a"])[2:])
    root = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}"; pool = root / "attackers"; pool.mkdir(parents=True, exist_ok=True)
    for j, url in ((j1, "http://n0:8707"), (j2, "http://n1:8708")):
        (pool / f"{j}.json").write_text(json.dumps({"url": url, "job_id": j})); _lease(pool, j, url)
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}"); assert r.returncode == 0, r.stdout + r.stderr
    v1 = json.loads((pool / "PLACED").read_text()); assert v1["version"] == 1 and [t["assigned_member_job_id"] for t in v1["targets"]] == [j1, j2]
    _bind(root, t1, "http://n0:8707"); _bind(root, t2, "http://n1:8708")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T02:00:00")
    return w, (j1, j2), (lw1, lw2), (t1, t2), root, pool, v1


def _open(w: World, *extra: str):
    r = w.run("placement-transition", "open", "--group", GROUP, "--cluster", "cluster_a", "--reason", REASON, *extra)
    assert r.returncode == 0 and "/PLACED.transition opened on cluster_a" in r.stdout, r.stdout + r.stderr
    return r


def _waiter(w: World, log: Path, poll="0.3", hours="0.05"):
    """`wait-pool` armed in the background, its output to `log` (a file, so the test can read it while the waiter runs)."""
    fh = open(log, "w")
    return subprocess.Popen([sys.executable, str(LIB / "submit_twin.py"), "--config", str(w.site / "config.env"), "--model", "m",
                             "wait-pool", "--group", GROUP, "--poll-s", poll, "--max-hours", hours], env=w.env(), stdout=fh, stderr=subprocess.STDOUT, text=True)


def _text(p: Path) -> str:
    return p.read_text() if p.exists() else ""


def _wait_line(log: Path, needle: str, n: int, proc, *, timeout_s=90.0):
    """Return once `needle` appears at least `n` times in the waiter's log; fail naming it — with the log — when the deadline passes or the
    waiter exited first. The deadline bounds the failure path only."""
    t0 = time.monotonic()
    while True:
        txt = _text(log)
        if txt.count(needle) >= n:
            return txt
        exited = proc.poll() is not None
        if exited or time.monotonic() - t0 > timeout_s:
            if not exited:
                proc.kill(); proc.wait(timeout=30)
            raise AssertionError((f"wait-pool exited (rc {proc.returncode}) before" if exited else f"waited {timeout_s:g}s for") + f" {n}x {needle!r}\n{_text(log)}")
        time.sleep(0.05)


def _torn(w: World):
    try:
        return [e for e in w.ledger() if e["event"] == "torn_down"]
    except ValueError:
        return []


def _stops(w: World):
    """The TEARDOWN's torn_down traces (a STOP written, or a SELF_EXIT recorded) — never the walk's own `torn_down {dead: true}` trace for the
    member it re-minted, which the pass records and the waiter never writes."""
    return [e for e in _torn(w) if not e.get("dead")]


def test_an_armed_waiter_holds_its_teardown_and_its_ratio_while_a_resubmit_pass_re_places_the_group_and_stops_the_replacement_only_with_the_rest(tmp_path):
    """The finding's two paths, concurrently with a REAL armed waiter: the pass opens the marker and re-mints the dead member (b: the ratio would
    retire the replacement — one live target, two live members, the new one assigned nothing by v1 — HELD); the last old target ends while the
    pass is still submitting (a: the teardown would STOP the replacement — HELD, the waiter polls on); the pass re-mints the targets and
    supersedes PLACED (the commit clears the marker); the waiter then reads the roster the pass re-placed — live targets, every member serving —
    and stops the pool only when THOSE end, the replacement with the rest. Never a STOP or RETIRE on the replacement before the supersession."""
    w, (j1, j2), (lw1, lw2), (t1, t2), root, pool, v1 = _placed_world(tmp_path)
    log = tmp_path / "wait-pool.log"; proc = _waiter(w, log)
    try:
        # ---- the waiter polls: two members for two live targets (one allowed) and both serve a live target -> nothing retired, the line says so
        _wait_line(log, "still serve live targets", 1, proc)
        assert not list(pool.glob("*.STOP")) and not list(pool.glob("*.RETIRE")) and not _retiring(w) and not _stops(w)
        # ---- member 1 dies with its target (an operator's scancel, a node): the waiter sees one live target and one live member — nothing to do
        set_job_state(w.fake["cluster_a"], j1, "FAILED", end="2026-01-01T01:00:00"); set_job_state(w.fake["cluster_a"], t1, "FAILED", end="2026-01-01T01:05:00")
        # ---- the --resubmit pass begins as 06 runs it: the marker FIRST, then the member's re-mint (its torn_down (dead) trace recorded; the
        #      replacement is live on sacct with no record yet — booting)
        _open(w); marker = pool / "PLACED.transition"; mdoc = json.loads(marker.read_text()); assert mdoc["from_version"] == 1 and mdoc["reason"] == REASON
        r = _walk(w, 1); assert r.returncode == 0 and "RESUBMIT" in (r.stdout + r.stderr) and "recorded torn_down (dead)" in (r.stdout + r.stderr), r.stdout + r.stderr
        j3 = str(job_ids(w.fake["cluster_a"])[-1]); assert j3 not in (j1, j2)
        # (b) the RATIO path: one live target (t2) -> one member allowed; two live members (j2 serving t2, j3 with no consumer — v1 assigns it
        #     nothing, nothing bound it) -> the surplus is j3, and the retirement is HELD under the lock: no RETIRE, no `retiring`, the hold named
        txt = _wait_line(log, "enforcement held this poll", 1, proc)
        assert f"retire {GROUP}: member job {j3} on cluster_a — HELD: the group's placement is in TRANSITION ({pool}/PLACED.transition since {mdoc['utc']} for 7200 s (" in txt, txt
        assert f"would have retired it: 2 live member(s) for 1 live target(s); 1 allowed" in txt and "s left): " + REASON in txt, txt
        assert not list(pool.glob("*.RETIRE")) and not _retiring(w) and not list(pool.glob("*.STOP")), sorted(p.name for p in pool.iterdir())
        # (a) the TEARDOWN path: the last OLD target ends while the pass is still submitting — every target the waiter knows is terminal and the
        #     placement is complete (v1), so before the fix it tore the pool down (STOP for j3 too); now the teardown is HELD under the lock and the
        #     waiter polls on. Two held polls, so the second read the same marker and still wrote nothing.
        set_job_state(w.fake["cluster_a"], t2, "FAILED", end="2026-01-01T01:10:00")
        txt = _wait_line(log, "teardown HELD this poll", 2, proc)
        assert f"teardown {GROUP}: HELD — the group's placement is in TRANSITION ({pool}/PLACED.transition since {mdoc['utc']} for 7200 s (" in txt and "tearing the pool down" in txt, txt
        assert f"placement-transition close --group {GROUP} --cluster cluster_a" in txt and proc.poll() is None
        assert not list(pool.glob("*.STOP")) and not _stops(w) and not list(pool.glob("*.RETIRE")), sorted(p.name for p in pool.iterdir())
        # ---- the pass re-mints the targets (dead, never registered — the registry contract's proof) ...
        for unit, shards in (("crescendo/g012", "0,1,2"), ("crescendo/g34", "3,4")):
            r = _walk_target(w, unit, shards); assert r.returncode == 0 and "submitted cluster_a" in r.stdout, r.stdout + r.stderr
        t3, t4 = (str(j) for j in job_ids(w.fake["cluster_a"])[-2:])
        # ... and SUPERSEDES the placement: v2 over the surviving member and the replacement, the new targets assigned round-robin, the marker
        #     cleared in the same locked step
        r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", "--supersede", "--reason", f"{REASON}: re-minted multiturn.attacker1,multiturn.crescendo.g012,multiturn.crescendo.g34")
        assert r.returncode == 0 and "SUPERSEDED version 1 -> 2" in (r.stdout + r.stderr) and "(the pass's PLACED.transition marker cleared: the transition committed)" in (r.stdout + r.stderr), r.stdout + r.stderr
        v2 = json.loads((pool / "PLACED").read_text())
        assert v2["version"] == 2 and v2["members"] == [j2, j3] and [t["job_id"] for t in v2["targets"]] == [t3, t4] and [t["assigned_member_job_id"] for t in v2["targets"]] == [j2, j3], v2
        assert not marker.exists()
        # ---- the waiter now reads the roster the pass re-placed: two live targets, each member assigned one -> nothing retired, nothing stopped
        n_busy = _text(log).count("still serve live targets")
        _wait_line(log, "still serve live targets", n_busy + 1, proc)
        assert not list(pool.glob("*.STOP")) and not list(pool.glob("*.RETIRE")) and not _retiring(w) and not _stops(w) and proc.poll() is None
        # ---- the NEW targets end: the backstop stops the pool — the replacement WITH the rest, now that nothing is in transition
        set_job_state(w.fake["cluster_a"], t3, "COMPLETED", end="2026-01-01T03:00:00"); set_job_state(w.fake["cluster_a"], t4, "COMPLETED", end="2026-01-01T03:10:00")
        proc.wait(timeout=90); out = _text(log)
        assert proc.returncode == 0, out
        assert sorted(p.name for p in pool.glob("*.STOP")) == sorted(f"{j}.STOP" for j in (j1, j2, j3)), out       # j1's STOP is moot (dead) and harmless, as before
        assert sorted(e["job_id"] for e in _stops(w)) == sorted([j1, j2, j3]) and "placement complete (2 placed target(s)) and every target terminal" in out
        assert out.count("teardown HELD this poll") >= 2 and "a placement-transition marker on cluster_a was ignored" not in out, out   # held while fresh; cleared by the commit, never lapsed
        assert not list(pool.glob("*.RETIRE")) and not _retiring(w)
    finally:
        if proc.poll() is None:
            proc.kill(); proc.wait(timeout=30)


def test_teardown_pool_and_retire_member_by_hand_are_refused_while_the_placement_is_in_transition_and_the_waiters_bound_names_it(tmp_path):
    """The same fence on the by-hand commands and at the waiter's bound: `teardown-pool` and `retire-member` refuse by name (rc 3, nothing written,
    the release named); a waiter whose every target is terminal under a fresh marker holds every poll and refuses by name at --max-hours; a LAPSED
    marker releases both and is named as ignored (the bound); `placement-transition close` releases them by hand; an unreadable marker is
    ignored and named."""
    w, (j1, j2), (lw1, lw2), (t1, t2), root, pool, v1 = _placed_world(tmp_path)
    _open(w); marker = pool / "PLACED.transition"; mdoc = json.loads(marker.read_text())
    r = w.run("teardown-pool", "--group", GROUP)
    assert r.returncode == 3 and f"teardown-pool {GROUP}: the group's placement is in TRANSITION ({pool}/PLACED.transition since {mdoc['utc']} for 7200 s (" in r.stderr, r.stdout + r.stderr
    assert "refused, nothing written" in r.stderr and f"placement-transition close --group {GROUP} --cluster cluster_a" in r.stderr and f"teardown {GROUP}: HELD" in r.stdout
    assert not list(pool.glob("*.STOP")) and not _torn(w)
    r = w.run("retire-member", "--group", GROUP, "--job", j2, "--why", "operator")
    assert r.returncode == 3 and "is in TRANSITION" in r.stderr and "refused, nothing written" in r.stderr and f"placement-transition close --group {GROUP} --cluster cluster_a" in r.stderr, r.stdout + r.stderr
    assert f"retire {GROUP}: member job {j2} on cluster_a — HELD: the group's placement is in TRANSITION" in r.stdout and "would have retired it: operator" in r.stdout, r.stdout
    assert not list(pool.glob("*.RETIRE")) and not _retiring(w)
    # the waiter's BOUND: every target terminal, the placement complete and in transition -> held every poll, refused by name at the bound, nothing stopped
    for t in (t1, t2):
        set_job_state(w.fake["cluster_a"], t, "COMPLETED", end="2026-01-01T01:00:00")
    r = w.run("wait-pool", "--group", GROUP, "--poll-s", "0.2", "--max-hours", "0.0004")
    assert r.returncode == 3 and f"wait-pool {GROUP}: every target terminal but the group's placement stayed in TRANSITION within 0.0004 h ({pool}/PLACED.transition since {mdoc['utc']}" in r.stderr, r.stdout + r.stderr
    assert "teardown HELD this poll" in r.stdout and "tearing the pool down" in r.stdout and f"placement-transition close --group {GROUP} --cluster cluster_a" in r.stderr
    assert not list(pool.glob("*.STOP")) and not _torn(w) and marker.exists()
    # LAPSED (the bound): the same marker past its lease releases the retirement, which names the lapse — the fence cannot pin a member for ever
    marker.write_text(json.dumps(dict(mdoc, epoch=mdoc["epoch"] - mdoc["lease_s"] - 1)))
    r = w.run("retire-member", "--group", GROUP, "--job", j2, "--why", "operator")
    assert r.returncode == 0 and f"member job {j2} on cluster_a marked RETIRING" in r.stdout and "(a placement-transition marker was ignored — lapsed: " in r.stdout and "lapsed " in r.stdout and "s ago): " + REASON in r.stdout, r.stdout + r.stderr
    assert (pool / f"{j2}.RETIRE").exists() and [e["job_id"] for e in _retiring(w)] == [j2]
    # fresh again: the teardown is refused once more; `close` releases it by hand and the teardown proceeds (STOP for both members, two torn_down)
    marker.write_text(json.dumps(mdoc))
    r = w.run("teardown-pool", "--group", GROUP); assert r.returncode == 3 and not list(pool.glob("*.STOP")), r.stdout + r.stderr
    r = w.run("placement-transition", "close", "--group", GROUP, "--cluster", "cluster_a"); assert r.returncode == 0 and "/PLACED.transition closed on cluster_a" in r.stdout, r.stdout + r.stderr
    r = w.run("teardown-pool", "--group", GROUP)
    assert r.returncode == 0 and f"teardown {GROUP}: {pool}/{j1}.STOP" in r.stdout and f"teardown {GROUP}: {pool}/{j2}.STOP" in r.stdout and "was ignored" not in r.stdout, r.stdout + r.stderr
    assert sorted(p.name for p in pool.glob("*.STOP")) == sorted([f"{j1}.STOP", f"{j2}.STOP"]) and sorted(e["job_id"] for e in _torn(w)) == sorted([j1, j2])
    # UNREADABLE: a marker that cannot be read is ignored and named — the teardown proceeds (a second world: STOP and torn_down are once-only here)
    (tmp_path / "w2").mkdir(); w2, (k1, k2), _, (u1, u2), root2, pool2, _ = _placed_world(tmp_path / "w2")
    (pool2 / "PLACED.transition").write_text("{not json")
    for t in (u1, u2):
        set_job_state(w2.fake["cluster_a"], t, "COMPLETED", end="2026-01-01T01:00:00")
    r = w2.run("teardown-pool", "--group", GROUP)
    assert r.returncode == 0 and f"teardown {GROUP}: a placement-transition marker on cluster_a was ignored (unreadable: {pool2}/PLACED.transition unreadable (" in r.stdout, r.stdout + r.stderr
    assert sorted(p.name for p in pool2.glob("*.STOP")) == sorted([f"{k1}.STOP", f"{k2}.STOP"]) and len(_torn(w2)) == 2
    # a self-exited member is still recorded, not stopped again, through the locked transaction (31b, unchanged)
    (tmp_path / "w3").mkdir(); w3, (m1, m2), _, _, root3, pool3, _ = _placed_world(tmp_path / "w3")
    (pool3 / f"{m1}.SELF_EXIT").write_text(json.dumps({"job_id": m1, "reason": "exit_last_consumer", "utc": "2026-01-01T02:00:00Z"}))
    r = w3.run("teardown-pool", "--group", GROUP)
    assert r.returncode == 0 and f"{pool3}/{m1}.SELF_EXIT already there (the member closed itself) — recorded, no STOP written" in r.stdout and f"{pool3}/{m2}.STOP" in r.stdout, r.stdout + r.stderr
    assert sorted(p.name for p in pool3.glob("*.STOP")) == [f"{m2}.STOP"] and {(e["job_id"], bool(e.get("self_exit"))) for e in _torn(w3)} == {(m1, True), (m2, False)}
    # dry-run writes nothing
    r = w3.run("--dry-run", "teardown-pool", "--group", GROUP); assert r.returncode == 0 and r.stdout.count("DRY-RUN: teardown-pool") == 2, r.stdout + r.stderr


def test_the_destructive_writes_read_the_marker_under_the_groups_lock_and_the_waiter_never_exits_on_a_held_teardown():
    """Static pins on the shape that makes the fence a fence: the STOP and RETIRE writes are programs run under `flock <pool_root>/PLACED.lock`
    — the lock the pass's `placement-transition open` takes — and each reads the marker BEFORE its first write; teardown_pool writes no
    sentinel outside that program; wait_pool returns only on a teardown that was not held; 06's hold note names the backstop."""
    sys.path.insert(0, str(LIB))
    import submit_twin as st  # noqa: E402
    src = (LIB / "submit_twin.py").read_text()
    for prog, write in ((st.TEARDOWN_TX_PY, 'job + ".STOP"'), (st.RETIRE_TX_PY, 'job + ".RETIRE"')):
        i_def = prog.index("def transition_state(root):"); i_call = prog.index("= transition_state(root)", i_def); i_write = prog.index(write)
        assert i_def < i_call < i_write and prog.count(write) == 1, prog                      # the marker is read before the one write
        i_fresh = prog.index('if tr_state == "fresh":'); assert i_call < i_fresh < i_write and "sys.exit(0)" in prog[i_fresh:i_write], prog   # and a fresh one exits before it
    assert '"held_transition"' in st.RETIRE_TX_PY and '"held"' in st.TEARDOWN_TX_PY
    body = src[src.index("    def teardown_pool(self, group: str, *, placed: Optional[dict]) -> dict:"):src.index("    def _placed_expect(")]
    assert "flock -w 300" in body and "PLACED.lock" in body and "TEARDOWN_TX_PY" in body and "write_sentinel" not in body and "member_self_exited" not in body
    ret = src[src.index("    def retire_member(self, group: str"):src.index("    def enforce_pool_ratio(")]
    assert "RETIRE_TX_PY" in ret and "flock -w 300" in ret and 'result == "held_transition"' in ret
    wp = src[src.index("    def wait_pool(self, group: str"):src.index("#: the placement-transition marker as the destructive transactions read it")]
    # round 5 tightened the return: the waiter leaves on a teardown that STOPPED (never on a held or a stale one, nor on any other answer)
    assert "self.teardown_pool(group); return 0" not in wp and 'if res.get("result") == "stopped":\n                    return 0' in wp and wp.count("return 0") == 1 and "stayed in TRANSITION" in wp
    sh = (REPO / "reproduce" / "06_eval_waves.sh").read_text()
    assert "hold every exit while this pass's attackers/PLACED.transition marker is fresh (its lease, 2 h by default), and an armed wait-pool holds its teardown and ratio under the same marker" in sh
