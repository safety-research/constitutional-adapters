"""A DEAD pool member (kind attacker, CANCELLED / FAILED on sacct, no SELF_EXIT, never torn down) was re-minted by the
`--resubmit --walk` with NO ledger trace, so `pool_members` kept the dead job forever and the next mark-placed carried it beside its
replacement — the demo smoke's PLACED v1 assigned every target to the cancelled member 981273, the re-minted 983283 found nothing
assigned to it and self-exited (exit_unbound), and a target failed on its refused endpoint. Now the walk appends
`torn_down {dead: true, job_id, state}` ONCE on that path (guarded like the self-exit branch), so the supersession that follows names
only the replacement. The decision is READ-ONLY — it derives the traces into its evidence and `submit --resubmit` alone records them,
so `prove-resubmit` moves nothing; the supersession's units come from the funnel's read-only `placement-changes`, the current bindings
against the record on the cluster. A member that LEFT beside a live one with nothing re-minted (the walk's skip plus its recorded
teardown) is a `dropped` member the derivation names, so the member-only change is superseded with a truthful reason instead of refused
on every pass. Over the fake clusters of test_submit_twin's World."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import job_ids, set_job_state  # noqa: E402
from test_pool_member_lifecycle_funnel import GROUP, POOL, _members, _target  # noqa: E402
from test_pool_self_exit_skip import _walk  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402


def _walk_target(w: World, unit: str, group_shards: str):
    return w.run("submit", "--unit", f"multiturn.{unit.replace('/', '.')}", "--lwid", f"m/u0/multiturn/{unit}", "--script", "scripts/sdfbase/full22/target_attacks.sbatch",
                 "--name", f"f22-att-m-{unit.replace('/', '-')}", "--args", f"m u0 crescendo {SHA} 9 {group_shards}", "--clusters", "cluster_a", "--group", GROUP,
                 "--wave-sha", SHA, "--resubmit", "--walk")


@pytest.mark.parametrize("state", ["CANCELLED", "FAILED"])
def test_a_dead_member_re_minted_by_the_walk_leaves_a_torn_down_trace_and_the_supersession_names_only_its_replacement(tmp_path, state):
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    (j1,) = _members(w, n=1)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4")
    t1, t2 = [str(j) for j in job_ids(w.fake["cluster_a"])[1:]]
    placed = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers/PLACED"
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}"); assert r.returncode == 0, r.stdout + r.stderr
    v1 = json.loads(placed.read_text())
    assert v1["version"] == 1 and v1["members"] == [j1] and [t["assigned_member_job_id"] for t in v1["targets"]] == [j1, j1]
    # the member dies without a sentinel of any kind — an operator's scancel (no SELF_EXIT, no <job>.STOP, no torn_down) or a crash — and its
    # targets die with it (their endpoint refused)
    set_job_state(w.fake["cluster_a"], j1, state, end="2026-01-01T01:00:00")
    for t in (t1, t2):
        set_job_state(w.fake["cluster_a"], t, "CANCELLED", end="2026-01-01T01:05:00")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T02:00:00")
    assert not [e for e in w.ledger() if e["event"] == "torn_down"]
    before = job_ids(w.fake["cluster_a"])
    # the family pass re-mints the member (dead, contract none) — and NOW records the teardown Slurm performed, once
    r = _walk(w, 1); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RESUBMIT" in out and f"job {j1} is DEAD on sacct ({state}; no SELF_EXIT, never torn down)" in out and "recorded torn_down (dead)" in out, out
    assert len(job_ids(w.fake["cluster_a"])) == len(before) + 1
    j2 = str(job_ids(w.fake["cluster_a"])[-1])
    torn = [e for e in w.ledger() if e["event"] == "torn_down" and e["job_id"] == j1]
    assert len(torn) == 1 and torn[0]["dead"] is True and torn[0]["state"] == state and torn[0]["cluster"] == "cluster_a" \
        and torn[0]["logical_work_id"] == f"m/u0/multiturn/{GROUP}/attackers/1" and not torn[0].get("self_exit"), torn
    # the walk is idempotent on it: the unit is bound to the live replacement -> skipped; no second trace (once)
    r = _walk(w, 1); assert r.returncode == 0 and f"skipped cluster_a {j2}" in r.stdout and "submitted " not in r.stdout, r.stdout + r.stderr
    assert len([e for e in w.ledger() if e["event"] == "torn_down" and e["job_id"] == j1]) == 1 and job_ids(w.fake["cluster_a"])[-1] == int(j2)
    # the targets are re-minted by the same pass (dead, never registered — the registry contract's proof)
    for unit, shards in (("crescendo/g012", "0,1,2"), ("crescendo/g34", "3,4")):
        r = _walk_target(w, unit, shards); assert r.returncode == 0 and "RESUBMIT" in (r.stdout + r.stderr) and "submitted cluster_a" in r.stdout, r.stdout + r.stderr
    n1, n2 = [str(j) for j in job_ids(w.fake["cluster_a"])[-2:]]
    # the fence still stands without --supersede — and the refusal already names the members as [j2]: the dead job is out of pool_members
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 3 and "a placement is immutable" in r.stderr and f"members now ['{j2}']" in r.stderr and json.loads(placed.read_text()) == v1, r.stdout + r.stderr
    # the sanctioned transition: members == [j2] alone (before the fix: [j1, j2], and the round-robin handed half the targets to the dead job)
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", "--supersede", "--reason", "06 --resubmit pass: the member was cancelled, member + targets re-minted")
    assert r.returncode == 0 and "SUPERSEDED version 1 -> 2" in (r.stdout + r.stderr), r.stdout + r.stderr
    v2 = json.loads(placed.read_text())
    assert v2["version"] == 2 and v2["members"] == [j2] and [t["job_id"] for t in v2["targets"]] == [n1, n2] and [t["assigned_member_job_id"] for t in v2["targets"]] == [j2, j2]
    assert v2["changes"]["kept"] == 0 and v2["changes"]["members_before"] == [j1] and sorted(v2["changes"]["removed"]) == sorted([[lw1, t1], [lw2, t2]])
    assert json.loads((placed.parent / "PLACED.v1").read_text()) == v1
    sup = [e for e in w.ledger() if e["event"] == "placement_superseded"]
    assert len(sup) == 1 and sup[0]["members"] == [j2] and sup[0]["from_version"] == 1 and sup[0]["to_version"] == 2


def test_the_walk_sweeps_an_earlier_dead_pool_job_that_was_re_minted_before_the_trace_existed(tmp_path):
    """The demo smoke's exact ledger shape on the fake cluster: member j1 (981273) CANCELLED and re-minted into j2 (983283) on the OLD tree — no
    trace — then j2 came up, found nothing assigned to it under PLACED v1 and SELF-EXITED. The next pass's walk decides the BOUND job (j2), so
    the trace-on-re-mint path alone never sees j1; the sweep records j1's teardown (dead) beside j2's (self_exit), the pool is gone, j3 is
    minted, and the supersession names [j3] alone — PLACED v2 carries only the new member + the new target ids, as the ruling expects."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    (j1,) = _members(w, n=1)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4")
    t1, t2 = [str(j) for j in job_ids(w.fake["cluster_a"])[1:]]
    placed = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers/PLACED"
    assert w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}").returncode == 0
    v1 = json.loads(placed.read_text()); assert v1["members"] == [j1]
    set_job_state(w.fake["cluster_a"], j1, "CANCELLED", end="2026-01-01T01:00:00")
    for t in (t1, t2):
        set_job_state(w.fake["cluster_a"], t, "CANCELLED", end="2026-01-01T01:05:00")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T02:00:00")
    r = _walk(w, 1); assert r.returncode == 0 and "RESUBMIT" in (r.stdout + r.stderr), r.stdout + r.stderr
    j2 = str(job_ids(w.fake["cluster_a"])[-1])
    # the OLD tree's walk left NO trace for j1: this synthetic ledger (never the cluster's) drops the line this tree wrote, to stand for it
    ledger = w.state / "attempts.jsonl"
    ledger.write_text("".join(ln + "\n" for ln in ledger.read_text().splitlines() if not (json.loads(ln).get("event") == "torn_down" and json.loads(ln).get("job_id") == j1)))
    assert not [e for e in w.ledger() if e["event"] == "torn_down"]
    # j2 came up, was assigned nothing under v1, and closed itself (exit_unbound); the targets died on their refused endpoint (re-minted below)
    pool = placed.parent
    (pool / f"{j2}.SELF_EXIT").write_text(json.dumps({"job_id": j2, "reason": "exit_unbound", "consumers": "", "why": "none bound me (none assigned me either)", "utc": "2026-01-01T03:00:00Z"}))
    set_job_state(w.fake["cluster_a"], j2, "COMPLETED", end="2026-01-01T03:00:00")
    for unit, shards in (("crescendo/g012", "0,1,2"), ("crescendo/g34", "3,4")):
        r = _walk_target(w, unit, shards); assert r.returncode == 0 and "submitted cluster_a" in r.stdout, r.stdout + r.stderr
    # the READ-ONLY proof over j2 names BOTH traces it derives (j2 self_exit, j1 dead and superseded) and writes neither
    before = ledger.read_bytes()
    r = w.run("prove-resubmit", "--unit", "multiturn.attacker1", "--lwid", f"m/u0/multiturn/{GROUP}/attackers/1", "--kind", "attacker", "--registers", "none")
    assert r.returncode == 0 and f"would record torn_down (self_exit) for job {j2} on cluster_a — READ-ONLY" in r.stdout \
        and f"would record torn_down (dead) for job {j1} on cluster_a (CANCELLED) — READ-ONLY" in r.stdout, r.stdout + r.stderr
    assert ledger.read_bytes() == before and not [e for e in w.ledger() if e["event"] == "torn_down"]
    # the member's walk: j2 self-exited (recorded), j1 swept as dead (recorded, superseded), no live member left -> re-minted as j3
    r = _walk(w, 1); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RESUBMIT" in out and "SELF-EXITED" in out and f"earlier pool member {j1} on cluster_a is DEAD on sacct (CANCELLED)" in out, out
    j3 = str(job_ids(w.fake["cluster_a"])[-1]); assert j3 not in (j1, j2)
    torn = {e["job_id"]: e for e in w.ledger() if e["event"] == "torn_down"}
    assert set(torn) == {j1, j2} and torn[j1]["dead"] is True and torn[j1]["state"] == "CANCELLED" and torn[j1]["superseded"] is True and not torn[j1].get("self_exit")
    assert torn[j2]["self_exit"] is True and not torn[j2].get("dead") and len([e for e in w.ledger() if e["event"] == "torn_down"]) == 2
    # idempotent: another pass skips the live j3 and sweeps nothing twice
    r = _walk(w, 1); assert r.returncode == 0 and f"skipped cluster_a {j3}" in r.stdout and len([e for e in w.ledger() if e["event"] == "torn_down"]) == 2, r.stdout + r.stderr
    # the supersession the pass writes: only the new member + the new target ids
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", "--supersede", "--reason", "06 --resubmit pass: j2 self-exited, j1 dead, member + targets re-minted")
    assert r.returncode == 0 and "SUPERSEDED version 1 -> 2" in (r.stdout + r.stderr), r.stdout + r.stderr
    v2 = json.loads(placed.read_text())
    assert v2["version"] == 2 and v2["members"] == [j3] and all(t["assigned_member_job_id"] == j3 for t in v2["targets"]) and v2["changes"]["kept"] == 0
    assert not {t1, t2} & {t["job_id"] for t in v2["targets"]} and v2["changes"]["members_before"] == [j1]


def test_the_read_only_proof_names_the_trace_it_derives_and_writes_nothing_the_submit_command_alone_records_it(tmp_path):
    """`prove-resubmit` — the documented READ-ONLY proof, which 03's design archive acts on — runs the same
    decision, and the decision appended the dead member's torn_down (and the sweep's) by merely running: a check of whether a cancelled
    attacker could be re-minted removed its job from pool_members and changed what the next placement assigns, even when nothing was then
    submitted. Now the decision DERIVES the traces into its evidence (`teardowns`) and only `submit --resubmit` records them."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    (j1,) = _members(w, n=1)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4")
    t1, t2 = [str(j) for j in job_ids(w.fake["cluster_a"])[1:]]
    assert w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}").returncode == 0
    set_job_state(w.fake["cluster_a"], j1, "CANCELLED", end="2026-01-01T01:00:00")
    for t in (t1, t2):
        set_job_state(w.fake["cluster_a"], t, "CANCELLED", end="2026-01-01T01:05:00")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T02:00:00")
    ledger = w.state / "attempts.jsonl"; before = ledger.read_bytes(); n_jobs = len(job_ids(w.fake["cluster_a"]))
    lwid = f"m/u0/multiturn/{GROUP}/attackers/1"
    # the proof, twice: AUTHORIZED, the trace it would record NAMED, the ledger byte-identical, nothing minted
    for _ in range(2):
        r = w.run("prove-resubmit", "--unit", "multiturn.attacker1", "--lwid", lwid, "--kind", "attacker", "--registers", "none"); out = r.stdout + r.stderr
        assert r.returncode == 0 and f"prove-resubmit: AUTHORIZED {lwid}" in r.stdout and f"bound_job\tcluster_a\t{j1}" in r.stdout, out
        assert f"would record torn_down (dead) for job {j1} on cluster_a (CANCELLED) — READ-ONLY: nothing written" in r.stdout and "recorded torn_down" not in out, out
        assert f"job {j1} is DEAD on sacct (CANCELLED; no SELF_EXIT, never torn down)" in r.stdout and ledger.read_bytes() == before and len(job_ids(w.fake["cluster_a"])) == n_jobs
    # pool_members still carries the dead job (the proof moved nothing): the same roster is `unchanged` with members [j1]
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}"); out = r.stdout + r.stderr
    assert r.returncode == 0 and "already placed, unchanged" in out and f"members ['{j1}']" in out and not [e for e in w.ledger() if e["event"] == "torn_down"], out
    # the authorized writer: the walk records exactly the trace the proof named, once, then re-mints
    r = _walk(w, 1); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RESUBMIT" in out and f"recorded torn_down (dead) for job {j1} on cluster_a (CANCELLED)" in out, out
    torn = [e for e in w.ledger() if e["event"] == "torn_down"]
    assert len(torn) == 1 and torn[0]["job_id"] == j1 and torn[0]["dead"] is True and torn[0]["state"] == "CANCELLED" and torn[0]["logical_work_id"] == lwid, torn
    j2 = str(job_ids(w.fake["cluster_a"])[-1]); assert j2 != j1 and len(job_ids(w.fake["cluster_a"])) == n_jobs + 1
    # the proof over the live replacement: not authorized, nothing named, nothing written
    before2 = ledger.read_bytes()
    r = w.run("prove-resubmit", "--unit", "multiturn.attacker1", "--lwid", lwid, "--kind", "attacker", "--registers", "none")
    assert r.returncode == 3 and f"NOT authorized {lwid}: skip" in r.stdout and "would record" not in r.stdout and ledger.read_bytes() == before2, r.stdout + r.stderr


def test_placement_changes_names_the_units_re_minted_since_the_record_and_nothing_the_ledger_cannot_account_for(tmp_path):
    """The --supersede reason's units come from the funnel's READ-ONLY placement-changes — the current bindings and pool
    members against attackers/PLACED on the cluster, each change accounted for by a ledger attempt or teardown — so a retry after a failed
    mark-placed names the same units; a difference the ledger cannot account for derives nothing (the fence decides); `none` is never said."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    (j1,) = _members(w, n=1)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4")
    t1, t2 = [str(j) for j in job_ids(w.fake["cluster_a"])[1:]]
    placed = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers/PLACED"
    # before any record: nothing to supersede
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and r.stdout.strip() == "" and "no attackers/PLACED record" in r.stderr, r.stdout + r.stderr
    assert w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}").returncode == 0
    v1 = json.loads(placed.read_text())
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and r.stdout.strip() == "" and "is the record's (version 1) — unchanged" in r.stderr, r.stdout + r.stderr
    set_job_state(w.fake["cluster_a"], j1, "CANCELLED", end="2026-01-01T01:00:00")
    for t in (t1, t2):
        set_job_state(w.fake["cluster_a"], t, "CANCELLED", end="2026-01-01T01:05:00")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T02:00:00")
    # the member alone re-minted: the member's unit, once; the targets (still the record's pairs) are no change
    r = _walk(w, 1); assert r.returncode == 0 and "RESUBMIT" in (r.stdout + r.stderr), r.stdout + r.stderr
    j2 = str(job_ids(w.fake["cluster_a"])[-1])
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and r.stdout.strip() == "re-minted multiturn.attacker1" and "re-minted since version 1: multiturn.attacker1" in r.stderr, r.stdout + r.stderr
    assert "dropped" not in r.stdout and "dropped since version" not in r.stderr        # j1's unit IS the re-minted one: its leaving is the re-mint's account, never a second clause
    for unit, shards in (("crescendo/g012", "0,1,2"), ("crescendo/g34", "3,4")):
        assert _walk_target(w, unit, shards).returncode == 0
    n1, n2 = [str(j) for j in job_ids(w.fake["cluster_a"])[-2:]]
    ledger = w.state / "attempts.jsonl"; before = ledger.read_bytes()
    # member first, then the targets in roster order — the same answer on every read (a retry's), the ledger untouched
    for _ in range(2):
        r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
        assert r.returncode == 0 and r.stdout.strip() == "re-minted multiturn.attacker1,multiturn.crescendo.g012,multiturn.crescendo.g34", r.stdout + r.stderr
        assert ledger.read_bytes() == before and json.loads(placed.read_text()) == v1
    # a roster that DROPS a target the record names is not a re-mint: nothing derived, the difference named, the fence decides
    r = w.run("placement-changes", "--group", GROUP, "--targets", lw1)
    assert r.returncode == 0 and r.stdout.strip() == "" and f"target {lw2} (job {t2}) named by version 1 is not in this roster" in r.stderr and "no supersession derived" in r.stderr, r.stdout + r.stderr
    assert "none" not in r.stdout
    # the supersession with the units it named; afterwards: unchanged (never a v3 reason)
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", "--supersede", "--reason", "06 --resubmit pass on 0000: re-minted multiturn.attacker1,multiturn.crescendo.g012,multiturn.crescendo.g34")
    assert r.returncode == 0 and "SUPERSEDED version 1 -> 2" in (r.stdout + r.stderr), r.stdout + r.stderr
    v2 = json.loads(placed.read_text()); assert v2["members"] == [j2] and sorted(t["job_id"] for t in v2["targets"]) == sorted([n1, n2])
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and r.stdout.strip() == "" and "is the record's (version 2) — unchanged" in r.stderr, r.stdout + r.stderr


def test_a_member_that_left_beside_a_live_one_with_nothing_re_minted_is_a_dropped_member_the_supersession_names(tmp_path):
    """Round 2 (on 8b9a7b5c7): a group of TWO placed members; one self-exits (its last consumer ended) while the other serves on. The walk skips
    the exited slot and records its teardown, so the placement's member list changes with NO new job — and placement-changes, which
    named added members and replaced target jobs alone, derived nothing for it: 06 passed no --supersede, and mark-placed refused the changed
    roster on every pass. Now a member the record names whose torn_down the ledger carries is a DROPPED member — named with its unit, its job
    and how it left — and the supersession's reason names it; a dropped member whose unit was itself re-minted is covered by `re-minted <unit>`
    alone (its old job leaving and its new one arriving are one change)."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    j1, j2 = _members(w, n=2)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4")
    t1, t2 = [str(j) for j in job_ids(w.fake["cluster_a"])[2:]]
    placed = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}/attackers/PLACED"
    assert w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}").returncode == 0
    v1 = json.loads(placed.read_text())
    assert v1["version"] == 1 and v1["members"] == [j1, j2] and [t["assigned_member_job_id"] for t in v1["targets"]] == [j1, j2]
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and r.stdout.strip() == "" and "is the record's (version 1) — unchanged" in r.stderr, r.stdout + r.stderr
    # member 1 closed itself (attackers/<j1>.SELF_EXIT) and COMPLETED; member 2 serves on; both targets run (the one assigned j1 bound j2 at its runtime)
    pool = placed.parent
    (pool / f"{j1}.SELF_EXIT").write_text(json.dumps({"job_id": j1, "reason": "exit_last_consumer", "consumers": t1, "why": "placement complete and every consumer terminal on sacct (bound or assigned)", "utc": "2026-01-01T02:00:00Z"}))
    set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T02:00:00"); set_job_state(w.fake["cluster_a"], j2, "RUNNING")
    for t in (t1, t2):
        set_job_state(w.fake["cluster_a"], t, "RUNNING")
    w.spool("cluster_a")
    before = job_ids(w.fake["cluster_a"])
    # the family pass: the exited slot is skipped beside the live member and its teardown recorded; every other unit is live and skipped
    r = _walk(w, 1); out = r.stdout + r.stderr
    assert r.returncode == 0 and "SELF-EXITED" in out and "never re-placed (D1713)" in out and f"recorded torn_down (self_exit) for job {j1} on cluster_a" in out and "RESUBMIT" not in r.stdout, out
    r = _walk(w, 2); assert r.returncode == 0 and f"skipped cluster_a {j2}" in r.stdout, r.stdout + r.stderr
    for unit, shards in (("crescendo/g012", "0,1,2"), ("crescendo/g34", "3,4")):
        r = _walk_target(w, unit, shards); assert r.returncode == 0 and "skipped cluster_a" in r.stdout, r.stdout + r.stderr
    assert job_ids(w.fake["cluster_a"]) == before                                                                 # nothing re-minted
    torn = [e for e in w.ledger() if e["event"] == "torn_down"]
    assert len(torn) == 1 and torn[0]["job_id"] == j1 and torn[0]["self_exit"] is True and torn[0]["logical_work_id"] == f"m/u0/multiturn/{GROUP}/attackers/1", torn
    # the READ-ONLY derivation names the dropped member — its unit, its job, how it left — and no re-mint; the same answer on every read, the ledger untouched
    ledger = w.state / "attempts.jsonl"; before_l = ledger.read_bytes()
    clause = f"dropped multiturn.attacker1 (job {j1}: self-exited)"
    for _ in range(2):
        r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
        assert r.returncode == 0 and r.stdout.strip() == clause, r.stdout + r.stderr
        assert f"placement-changes {GROUP}: dropped since version 1: multiturn.attacker1 (job {j1}: self-exited)" in r.stderr and "re-minted" not in r.stderr and "no supersession derived" not in r.stderr, r.stderr
        assert ledger.read_bytes() == before_l and json.loads(placed.read_text()) == v1
    # the fence still refuses the bare roster (members now [j2]); the supersession with the derived reason is the sanctioned transition: the live
    # member alone, the target the exited member was assigned re-assigned to it, the other target KEPT (its member stands), no pair added or removed
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 3 and "a placement is immutable" in r.stderr and f"members now ['{j2}']" in r.stderr and json.loads(placed.read_text()) == v1, r.stdout + r.stderr
    reason = f"06 --resubmit pass on 0000: {clause}"
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", "--supersede", "--reason", reason)
    assert r.returncode == 0 and "SUPERSEDED version 1 -> 2 (0 added, 0 removed, 1 kept their member)" in (r.stdout + r.stderr), r.stdout + r.stderr
    v2 = json.loads(placed.read_text())
    assert v2["version"] == 2 and v2["members"] == [j2] and [t["job_id"] for t in v2["targets"]] == [t1, t2] and [t["assigned_member_job_id"] for t in v2["targets"]] == [j2, j2], v2
    assert v2["changes"] == {"added": [], "removed": [], "kept": 1, "members_before": [j1, j2]} and v2["reason"] == reason and v2["supersedes"]["version"] == 1, v2
    assert json.loads((pool / "PLACED.v1").read_text()) == v1
    sup = [e for e in w.ledger() if e["event"] == "placement_superseded"]
    assert len(sup) == 1 and sup[0]["members"] == [j2] and sup[0]["from_version"] == 1 and sup[0]["to_version"] == 2 and sup[0]["reason"] == reason and sup[0]["changes"]["kept"] == 1, sup
    # afterwards: unchanged; another pass over the exited slot skips it again and records nothing twice
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and r.stdout.strip() == "" and "is the record's (version 2) — unchanged" in r.stderr, r.stdout + r.stderr
    r = _walk(w, 1); assert r.returncode == 0 and "never re-placed (D1713)" in (r.stdout + r.stderr) and len([e for e in w.ledger() if e["event"] == "torn_down"]) == 1
    # the rule's other half: the LIVE member dies and is re-minted (its targets with it) — its leaving is `re-minted multiturn.attacker2`, never a
    # second `dropped` clause for the same unit; the exited slot stays dropped-and-recorded (in the record's past, not in this change)
    set_job_state(w.fake["cluster_a"], j2, "CANCELLED", end="2026-01-01T03:00:00")
    for t in (t1, t2):
        set_job_state(w.fake["cluster_a"], t, "CANCELLED", end="2026-01-01T03:05:00")
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T04:00:00")
    r = _walk(w, 2); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RESUBMIT" in out and f"job {j2} is DEAD on sacct (CANCELLED" in out and "recorded torn_down (dead)" in out, out
    j3 = str(job_ids(w.fake["cluster_a"])[-1]); assert j3 not in (j1, j2)
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and r.stdout.strip() == "re-minted multiturn.attacker2" and "dropped" not in r.stdout and "dropped since version" not in r.stderr, r.stdout + r.stderr
    for unit, shards in (("crescendo/g012", "0,1,2"), ("crescendo/g34", "3,4")):
        assert _walk_target(w, unit, shards).returncode == 0
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and r.stdout.strip() == "re-minted multiturn.attacker2,multiturn.crescendo.g012,multiturn.crescendo.g34" and "dropped" not in r.stdout, r.stdout + r.stderr
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", "--supersede", "--reason", f"06 --resubmit pass on 0000: {r.stdout.strip()}")
    assert r.returncode == 0 and "SUPERSEDED version 2 -> 3" in (r.stdout + r.stderr), r.stdout + r.stderr
    v3 = json.loads(placed.read_text())
    assert v3["members"] == [j3] and all(t["assigned_member_job_id"] == j3 for t in v3["targets"]) and v3["changes"]["kept"] == 0 and v3["changes"]["members_before"] == [j2]


def test_the_decision_derives_the_traces_read_only_and_the_submit_command_alone_records_them():
    from front_door_testlib import LIB
    src = (LIB / "submit_twin.py").read_text()
    body = src[src.index("def _resubmit_decision("):]; body = body[:body.index("# the registry contract: fail CLOSED")]
    assert "self.ledger.append(" not in body                                   # the decision writes NOTHING: prove-resubmit runs it
    i_se = body.index('if str(job_id) not in torn_jobs:\n                    teardowns.append({"event": "torn_down", "logical_work_id": lwid, "job_id": str(job_id), "cluster": label, "self_exit": True})')
    i_dead = body.index('if (att.get("kind") or kind) == "attacker" and not self_exited and str(job_id) not in torn_jobs:')   # never a second trace beside the self-exit one
    i_app = body.index('teardowns.append({"event": "torn_down", "logical_work_id": lwid, "job_id": str(job_id), "cluster": label, "dead": True, "state": states})', i_dead)
    i_ret = body.index('return "resubmit", why, {"tasks": tasks, "evidence": ev, "contract": contract}', i_app)
    assert i_se < i_dead < i_app < i_ret                                   # the dead-member trace sits on the contract-`none` resubmit return, after the self-exit branch
    assert body.count("teardowns.append(") == 4 and "a torn_down (dead) trace is due" in body and "torn_down {dead: true, state}" in src
    wrapper = src[src.index("def resubmit_decision("):src.index("def record_teardowns(")]
    assert 'return action, why, {**evidence, "teardowns": teardowns}' in wrapper and "self.ledger.append(" not in wrapper   # every decided path carries them
    # the writer: `submit --resubmit` records them on each decided path (skip, reclose, re-mint); the proof never does
    submit = src[src.index('if a.cmd == "submit":'):]
    assert submit.count("co.record_teardowns(evidence)") == 3
    prove = src[src.index('if a.cmd == "prove-resubmit":'):src.index('if a.cmd == "submit":')]
    assert "co.record_teardowns(" not in prove and "ledger.append(" not in prove and "READ-ONLY: nothing written" in prove   # the proof calls no writer
    # on a re-mint the write follows the move preflight (a refusal leaves the ledger as it was) and precedes the transition token
    i_pre = submit.index("moved_from = placement_receipt_origin("); i_rec = submit.index("co.record_teardowns(evidence)", i_pre)
    i_tok = submit.index("ST.write_transition(co.subs.path, tok); transition_open = True", i_rec)
    assert i_pre < i_rec < i_tok
