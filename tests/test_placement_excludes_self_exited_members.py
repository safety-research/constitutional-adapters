"""The interleaving this closes: a member decides its old consumers are done; a `06 --family multiturn --resubmit` pass opens
attackers/PLACED.transition; the pass's --walk reads the member's job as active on Slurm (a job in its exit path is still
RUNNING/COMPLETING there) and keeps it; the member writes attackers/<member>.SELF_EXIT and its EXIT trap removes <member>.json /
.pending; mark-placed --supersede ran PLACEMENT_TX_PY with intent["members"] exactly as the walk handed them, and committed v2
round-robining the re-minted target onto the exited member. Now the transaction reads the pool's SELF_EXIT stems UNDER PLACED.lock
and drops those members BEFORE its unchanged / refused / superseded decision: a record that names an exited member is never answered
`unchanged`; an unchanged pair whose member was excluded is re-assigned over the members that remain and named in `changes`; new
pairs round-robin over the remaining members alone; with no member left the transaction answers `no_live_member` — nothing written,
the marker left as every refusal leaves it (its lease the bound) — and the caller prints the remedy (the same pass again, whose walk
then re-mints). A pass with no SELF_EXIT answers and writes byte for byte as before. The member-side half (serialising the exit
decision with the marker open) is deliberately NOT here: the window between the exit decision and the tombstone stays open, and the
named-residual test (tests/test_attacker_pool_lifecycle.py) keeps it loud. Over the fake clusters of test_submit_twin's World with the
REAL funnel subcommands as 06's pass runs them, the REAL member lifecycle for the exit verdict, and the fake sacct's test-only gate
(FAKE_SACCT_GATE, keyed on the frozen process's env) to park the walk at its sacct read while the tombstone lands — exact, never
timing; plus the real 06 over the wave-integration World for the surfacing and the remedy."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import LIB, REPO, job_ids, set_job_state  # noqa: E402
from test_pool_dead_member_torn_down import _walk_target  # noqa: E402
from test_pool_member_lifecycle_funnel import GROUP, POOL, _members, _target  # noqa: E402
from test_pool_retire_funnel import _bind, _lease  # noqa: E402
from test_pool_self_exit_skip import _walk  # noqa: E402
from test_submit_twin import SHA, World  # noqa: E402
from test_wait_pool_holds_under_placement_transition import REASON, _open  # noqa: E402
from test_wait_pool_stale_placement_fence import _frozen_on, _gated  # noqa: E402

LIFE = REPO / "scripts" / "sdfbase" / "full22" / "attacker_pool_lifecycle.sh"
SBATCH = REPO / "scripts" / "sdfbase" / "full22" / "attacker_pool.sbatch"
URL1, URL2 = "http://n0:8707", "http://n1:8708"


# ----------------------------------------------------------------------------------------------------------------- the member's side
def _verdict(w: World, root: Path, url: str, member: str):
    """The REAL member lifecycle (attacker_pool_lifecycle.sh f22_pool_member_verdict), as the member's tick runs it over the World's wave root
    and fake Slurm: (verdict, the reason the sbatch logs)."""
    env = dict(os.environ, PATH=f"{w.bin}:{os.environ['PATH']}", FAKE_SLURM_DIR=str(w.fake["cluster_a"]), PY_LIFECYCLE=sys.executable)
    r = subprocess.run(["bash", "-c", 'source "$1"; shift; "$@"', "_", str(LIFE), "f22_pool_member_verdict", str(root), url, member],
                       capture_output=True, text=True, env=env, timeout=120)
    assert r.returncode == 0 and len(r.stdout.split()) == 1, r.stdout + r.stderr
    return r.stdout.strip(), r.stderr.strip().splitlines()[-1]


def _member_exits(pool: Path, member: str, verdict: str, consumers: str, why: str) -> None:
    """The member's exit path as attacker_pool.sbatch runs it on an exit verdict: attackers/<job>.SELF_EXIT {job_id, reason, consumers, why, utc}
    written, `exit 0`, and the EXIT trap `cleanup` removes its record (<job>.json) and its pending marker (<job>.pending). Slurm still lists
    the job RUNNING for a moment after this — the test leaves the fake job's state exactly there."""
    (pool / f"{member}.SELF_EXIT").write_text(json.dumps({"job_id": member, "reason": verdict, "consumers": consumers, "why": why.replace('"', ""),
                                                          "utc": "2026-01-01T01:09:00Z"}) + "\n")
    for suffix in (".json", ".pending"):
        p = pool / f"{member}{suffix}"
        if p.exists():
            p.unlink()


# ------------------------------------------------------------------------------------------------------ the remedy 06 prints, as the operator runs it
REMEDY_HEAD = "run exactly this pass again, in this environment (its effective flags; a --units selection widened by the group's member unit(s)): "


def _remedy_cmd(line: str) -> str:
    """The command the `followup: REMEDY for group <g>` line prints, as printed (the text after the head, before the `  (` note)."""
    assert REMEDY_HEAD in line, line
    return line.split(REMEDY_HEAD, 1)[1].split("  (", 1)[0]


def _remedy_argv(line: str) -> list:
    """The printed command as the shell reads it back."""
    return shlex.split(_remedy_cmd(line))


def _incomplete(r, state: Path, view: str, group: str, lab: str) -> dict:
    """A pass that ended INCOMPLETE on `group`'s NO LIVE MEMBER: exit 4, the refusal naming the group and the remedy, no `06 done`, and
    WAVE.<view>.partial.json naming the group, its label and the printed remedy verbatim — returned parsed."""
    out = r.stdout + r.stderr
    assert r.returncode == 4 and "06 done" not in r.stdout, out[-5000:]
    assert (f"REFUSED: placement INCOMPLETE — group(s) {group} (on {lab}) have NO LIVE MEMBER (above)" in r.stderr
            and f"run the printed REMEDY (`followup: REMEDY for group {group}`)" in r.stderr and f"WAVE.{view}.json is NOT written" in r.stderr), r.stderr[-3000:]
    remedy = [ln for ln in r.stdout.splitlines() if f"followup: REMEDY for group {group}" in ln]
    assert len(remedy) == 1, r.stdout[-4000:]
    pdoc = json.loads((state / f"WAVE.{view}.partial.json").read_text())
    assert pdoc["refusal"] == "no_live_member" and pdoc["unplaced_groups"] == [{"group": group, "cluster": lab, "remedy": _remedy_cmd(remedy[0])}], pdoc
    return pdoc


def _run_printed(w, line: str, **env):
    """Run the printed remedy verbatim over the wave-integration World `w` (its PATH, state and registry; `env` adds this pass's knobs)."""
    argv = _remedy_argv(line)
    return subprocess.run(["bash", *argv], capture_output=True, text=True, env=w.env(**env), cwd=str(w.repo), timeout=900)


def _without(argv: list, *flags: str) -> list:
    """`argv` with the named value-taking flags and their values dropped — the pre-fix spelling of a remedy."""
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False; continue
        if a in flags:
            skip = True; continue
        out.append(a)
    return out


def _frozen_walk(w: World, i: int, gate: Path):
    """The pass's --walk of member `i`, its sacct calls blocked while `gate` exists (this process's env alone)."""
    return _gated(w, ["submit", "--unit", f"multiturn.attacker{i}", "--lwid", f"m/u0/multiturn/{GROUP}/attackers/{i}", "--script", POOL, "--name", f"f22-atk-m-{i}",
                      "--args", SHA, "--clusters", "cluster_a", "--kind", "attacker", "--group", GROUP, "--wave-sha", SHA, "--resubmit", "--walk"], gate)


def _events(w: World, kind: str):
    return [e for e in w.ledger() if e["event"] == kind]


def _three_target_world(tmp_path):
    """Two serving members (records + fresh health leases) and THREE targets placed round-robin (t1 -> j1, t2 -> j2, t3 -> j1), each bound to
    its member's URL; the registry spool and inbox the targets' re-mint proof reads."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    j1, j2 = _members(w)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4"); lw3 = _target(w, "pair/g012", "0,1,2")
    t1, t2, t3 = (str(j) for j in job_ids(w.fake["cluster_a"])[2:])
    root = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}"; pool = root / "attackers"; pool.mkdir(parents=True, exist_ok=True)
    for j, url in ((j1, URL1), (j2, URL2)):
        (pool / f"{j}.json").write_text(json.dumps({"url": url, "job_id": j})); (pool / f"{j}.pending").write_text(json.dumps({"job_id": j, "state": "starting"})); _lease(pool, j, url)
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}"); assert r.returncode == 0, r.stdout + r.stderr
    v1 = json.loads((pool / "PLACED").read_text())
    assert v1["version"] == 1 and v1["members"] == [j1, j2] and [t["assigned_member_job_id"] for t in v1["targets"]] == [j1, j2, j1] and "excluded_self_exited" not in v1
    _bind(root, t1, URL1); _bind(root, t2, URL2); _bind(root, t3, URL1)
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T02:00:00")
    return w, (j1, j2), (lw1, lw2, lw3), (t1, t2, t3), root, pool, v1


def _one_member_world(tmp_path):
    """ONE serving member and two targets placed on it, both bound to its URL."""
    w = World(tmp_path); (w.fake["cluster_a"] / "policy").write_text("start_on_submit\n")
    (j1,) = _members(w, n=1)
    lw1 = _target(w, "crescendo/g012", "0,1,2"); lw2 = _target(w, "crescendo/g34", "3,4")
    t1, t2 = (str(j) for j in job_ids(w.fake["cluster_a"])[1:])
    root = w.shared["cluster_a"] / f"exp/reproduce/m/waves/{GROUP}"; pool = root / "attackers"; pool.mkdir(parents=True, exist_ok=True)
    (pool / f"{j1}.json").write_text(json.dumps({"url": URL1, "job_id": j1})); (pool / f"{j1}.pending").write_text(json.dumps({"job_id": j1, "state": "starting"})); _lease(pool, j1, URL1)
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}"); assert r.returncode == 0, r.stdout + r.stderr
    v1 = json.loads((pool / "PLACED").read_text()); assert v1["version"] == 1 and v1["members"] == [j1] and [t["assigned_member_job_id"] for t in v1["targets"]] == [j1, j1]
    _bind(root, t1, URL1); _bind(root, t2, URL1)
    w.spool("cluster_a"); w.ingested("cluster_a", through="2026-01-01T02:00:00")
    return w, j1, (lw1, lw2), (t1, t2), root, pool, v1


# ------------------------------------------------------------------------------------------------------------- the exact interleaving
def test_the_interleaving_with_two_members_assigns_everything_to_the_survivor_and_names_the_exclusion(tmp_path):
    """The interleaving, step by step, with a second member live: (1) member 1's consumers end (t1 COMPLETED — its results stand — and
    t3 FAILED) and the REAL lifecycle, with no marker standing, answers exit_last_consumer; (2) the pass opens the marker; (3) the pass's --walk
    of member 1 is frozen at its sacct read of the job, which is RUNNING there; (4) the member writes SELF_EXIT and its EXIT trap removes its
    record and pending marker — the gate lifts and the walk, reading RUNNING, keeps the member (skip, nothing re-minted, no teardown recorded);
    (5) the pass re-mints the dead target and runs mark-placed --supersede with pool_members [j1, j2]. Before the fix v2 kept (t1 -> j1) and
    round-robined the new pair onto j1, the exited member; now the transaction excludes j1 under the lock: v2 names j2 alone, the unchanged
    pair t1 is RE-ASSIGNED to j2 and named in `changes`, the new pair goes to j2, t2 stays kept, and the exclusion is named in the answer, the
    record and the placement_superseded event. The next pass (the job COMPLETED on sacct) records the teardown, derives nothing and is
    `unchanged` with members [j2]."""
    w, (j1, j2), (lw1, lw2, lw3), (t1, t2, t3), root, pool, v1 = _three_target_world(tmp_path)
    # (1) member 1's consumers are done: t1 COMPLETED (a registering unit's results — the walk skips it), t3 FAILED (dead — the pass re-mints it)
    set_job_state(w.fake["cluster_a"], t1, "COMPLETED", end="2026-01-01T01:00:00"); set_job_state(w.fake["cluster_a"], t3, "FAILED", end="2026-01-01T01:00:00")
    v, why = _verdict(w, root, URL1, j1)
    assert v == "exit_last_consumer" and why.startswith("placement complete and every consumer terminal on sacct (bound or assigned): ") and t1 in why and t3 in why and "TRANSITION" not in why, (v, why)
    assert _verdict(w, root, URL2, j2)[0] == "serve"
    # (2) the pass opens the marker
    _open(w); marker = pool / "PLACED.transition"; mdoc = json.loads(marker.read_text()); assert mdoc["from_version"] == 1 and mdoc["reason"] == REASON
    # (3) the pass's --walk of member 1, frozen at its sacct read of j1 (the member's own job; the unit has no earlier pool job to sweep)
    gate = tmp_path / "sacct.gate"; gate.write_text("")
    proc = _frozen_walk(w, 1, gate)
    assert _frozen_on(gate, proc) == j1
    # (4) the member acts on its verdict while the walk is parked: SELF_EXIT written, record + pending marker removed; Slurm still says RUNNING
    _member_exits(pool, j1, v, consumers=f"{t1},{t3}", why=why)
    assert (pool / f"{j1}.SELF_EXIT").exists() and not (pool / f"{j1}.json").exists() and not (pool / f"{j1}.pending").exists()
    assert (w.fake["cluster_a"] / "jobs" / f"{j1}.state").read_text().strip() == "RUNNING"
    gate.unlink(); out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0 and f"skipped cluster_a {j1} multiturn.attacker1 (still RUNNING" in out and "RESUBMIT" not in out and "SELF-EXITED" not in out, out
    assert not _events(w, "torn_down") and job_ids(w.fake["cluster_a"]) == [int(x) for x in (j1, j2, t1, t2, t3)]
    # (5) the pass walks on: member 2 and the live / completed targets are skipped, the dead target is re-minted
    r = _walk(w, 2); assert r.returncode == 0 and f"skipped cluster_a {j2}" in r.stdout, r.stdout + r.stderr
    r = _walk_target(w, "crescendo/g012", "0,1,2"); assert r.returncode == 0 and f"skipped cluster_a {t1}" in r.stdout and "COMPLETED: results stand" in r.stdout, r.stdout + r.stderr
    r = _walk_target(w, "crescendo/g34", "3,4"); assert r.returncode == 0 and f"skipped cluster_a {t2}" in r.stdout, r.stdout + r.stderr
    r = _walk_target(w, "pair/g012", "0,1,2"); assert r.returncode == 0 and "RESUBMIT" in (r.stdout + r.stderr) and "submitted cluster_a" in r.stdout, r.stdout + r.stderr
    t4 = str(job_ids(w.fake["cluster_a"])[-1]); assert t4 not in (t1, t2, t3)
    # the derivation names the re-mint alone: j1 is still a pool member to the ledger (no teardown recorded) and named by v1 — nothing dropped
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}")
    assert r.returncode == 0 and r.stdout.strip() == "re-minted multiturn.pair.g012" and "dropped" not in r.stderr, r.stdout + r.stderr
    reason = f"{REASON}: re-minted multiturn.pair.g012"
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}", "--supersede", "--reason", reason); out = r.stdout + r.stderr
    assert r.returncode == 0, out
    assert (f"placed {GROUP}: SUPERSEDED version 1 -> 2 (1 added, 1 removed, 1 kept their member, 1 unchanged pair(s) re-assigned off the excluded member(s) to ['{j2}']) — {reason} "
            f"(the pass's PLACED.transition marker cleared: the transition committed) — 1 member(s) EXCLUDED under the group's lock: attackers/<job>.SELF_EXIT stood on cluster_a for "
            f"['{j1}'] (left, or leaving: the walk read their jobs as active on Slurm, and a job in its exit path is still RUNNING/COMPLETING there); the placement assigns them nothing") in out, out
    v2 = json.loads((pool / "PLACED").read_text())
    # v2 never names the member whose SELF_EXIT stood at commit time — not as a member, not as any target's assignment
    assert v2["version"] == 2 and v2["members"] == [j2] and all(t["assigned_member_job_id"] == j2 for t in v2["targets"]), v2
    assert [(t["logical_work_id"], t["job_id"]) for t in v2["targets"]] == [(lw1, t1), (lw2, t2), (lw3, t4)]
    assert v2["excluded_self_exited"] == [j1] and v2["supersedes"] == {"version": 1, "utc": v1["utc"], "n_targets": 3} and v2["reason"] == reason
    assert v2["changes"] == {"added": [[lw3, t4]], "removed": [[lw3, t3]], "kept": 1, "members_before": [j1, j2], "excluded_self_exited": [j1], "reassigned": [[lw1, t1, j1, j2]]}, v2["changes"]
    assert json.loads((pool / "PLACED.v1").read_text()) == v1 and not marker.exists()
    sup = _events(w, "placement_superseded")
    assert len(sup) == 1 and sup[0]["members"] == [j2] and sup[0]["excluded_self_exited"] == [j1] and sup[0]["changes"] == v2["changes"] and sup[0]["to_version"] == 2, sup
    assert all(t["assigned_member_job_id"] == j2 for t in sup[0]["targets"]) and "excluded_self_exited" not in _events(w, "placement_complete")[0]
    # the member the record was built around keeps serving its assignment: the REAL lifecycle over v2 answers serve for j2 (t2 and t4 live)
    v, why = _verdict(w, root, URL2, j2); assert v == "serve" and t4 in why, (v, why)
    # ---- the member's job ends on sacct; the NEXT pass: the walk reads the SELF_EXIT beside the live member -> skip and the teardown
    #      recorded; nothing re-minted; the derivation finds the record already free of j1 -> no --supersede; mark-placed is `unchanged` with
    #      members [j2] and no exclusion (j1 is no longer handed: pool_members drops the torn_down job)
    set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T01:10:00")
    _open(w)
    r = _walk(w, 1); out = r.stdout + r.stderr
    assert r.returncode == 0 and "SELF-EXITED" in out and "never re-placed (D1713)" in out and f"recorded torn_down (self_exit) for job {j1}" in out and "RESUBMIT" not in r.stdout, out
    torn = _events(w, "torn_down"); assert len(torn) == 1 and torn[0]["job_id"] == j1 and torn[0]["self_exit"] is True
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}")
    assert r.returncode == 0 and r.stdout.strip() == "" and "is the record's (version 2) — unchanged" in r.stderr, r.stdout + r.stderr
    before = (pool / "PLACED").read_bytes()
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2},{lw3}"); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"already placed, unchanged (3 target(s), members ['{j2}'], version 2) — nothing written (the pass's PLACED.transition marker cleared: the placement stands as recorded)" in out and "EXCLUDED" not in out, out
    assert (pool / "PLACED").read_bytes() == before and not marker.exists() and len(_events(w, "placement_superseded")) == 1


def test_a_pass_whose_only_member_self_exited_is_refused_no_live_member_with_nothing_written_and_the_re_run_re_mints_and_commits(tmp_path):
    """The same interleaving with ONE member: every target of the group is re-minted and the walk kept the member (RUNNING on sacct), so the
    transaction excludes the only member it was handed and answers `no_live_member` — rc 4, the excluded id named, the remedy printed (the
    same pass again once sacct reports the job terminal), NOTHING written: PLACED byte-identical, no archive, no event, the marker exactly as
    the open left it (its lease named as the bound). The exclusion precedes the refused decision, so the bare roster (no --supersede) is the
    same refusal, never the immutable-placement fence. Then the remedy: the job COMPLETED on sacct, the same pass's walk reads the SELF_EXIT
    with no live member left and re-mints; its mark-placed supersedes v1 -> v2 onto the replacement alone, with no exclusion to name."""
    w, j1, (lw1, lw2), (t1, t2), root, pool, v1 = _one_member_world(tmp_path)
    for t in (t1, t2):
        set_job_state(w.fake["cluster_a"], t, "FAILED", end="2026-01-01T01:00:00")
    v, why = _verdict(w, root, URL1, j1); assert v == "exit_last_consumer" and t1 in why and t2 in why, (v, why)
    _open(w); marker = pool / "PLACED.transition"; marker_bytes = marker.read_bytes()
    gate = tmp_path / "sacct.gate"; gate.write_text("")
    proc = _frozen_walk(w, 1, gate); assert _frozen_on(gate, proc) == j1
    _member_exits(pool, j1, v, consumers=f"{t1},{t2}", why=why)
    gate.unlink(); out, _ = proc.communicate(timeout=120)
    assert proc.returncode == 0 and f"skipped cluster_a {j1} multiturn.attacker1 (still RUNNING" in out and "RESUBMIT" not in out, out
    for unit, shards in (("crescendo/g012", "0,1,2"), ("crescendo/g34", "3,4")):
        r = _walk_target(w, unit, shards); assert r.returncode == 0 and "submitted cluster_a" in r.stdout, r.stdout + r.stderr
    t3, t4 = (str(j) for j in job_ids(w.fake["cluster_a"])[-2:])
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and r.stdout.strip() == "re-minted multiturn.crescendo.g012,multiturn.crescendo.g34", r.stdout + r.stderr
    before = (pool / "PLACED").read_bytes(); ledger_before = (w.state / "attempts.jsonl").read_bytes()
    reason = f"{REASON}: re-minted multiturn.crescendo.g012,multiturn.crescendo.g34"
    for extra in (("--supersede", "--reason", reason), ()):
        r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", *extra); out = r.stdout + r.stderr
        assert r.returncode == 4, out
        assert (f"REFUSED: mark-placed {GROUP}: NO LIVE MEMBER — every member the ledger records for the group (['{j1}']) has attackers/<job>.SELF_EXIT on cluster_a under the "
                f"group's lock (excluded ['{j1}']); this pass's --walk minted no replacement for any of them (it read their jobs as active on Slurm, and a job in its exit path is still "
                f"RUNNING/COMPLETING there), so a placement would assign 2 target(s) to servers that are gone — nothing written: attackers/PLACED stands at version 1 "
                f"(2 target(s), members ['{j1}']), no archive, no ledger event. REMEDY: once sacct reports job(s) ['{j1}'] terminal, run the same `06 --family multiturn "
                f"--resubmit` pass again, with the flags that pass ran with (06 prints it as `followup: REMEDY`, its effective flags shell-quoted) — its --walk then "
                f"reads the SELF_EXIT, records the teardown (torn_down self_exit) and re-mints the member, and that pass's "
                f"mark-placed commits the supersession onto the replacement; the pass's PLACED.transition marker stands as before (since ") in r.stderr, out
        mdoc = json.loads(marker_bytes)
        assert f"{mdoc['utc']}, 7200 s lease, " in r.stderr and f" s left: {REASON!r}) — it holds no exit of a member that has already left, and its lease bounds it; the re-run refreshes it and its commit clears it, or release it with `placement-transition close --group {GROUP} --cluster cluster_a`" in r.stderr, out
        assert "a placement is immutable" not in out and "SUPERSEDED" not in out
        # nothing written: the record, the archive slot, the ledger and the marker are exactly as they were
        assert (pool / "PLACED").read_bytes() == before and not (pool / "PLACED.v1").exists() and not list(pool.glob("PLACED.tmp.*")) and (w.state / "attempts.jsonl").read_bytes() == ledger_before
        assert marker.read_bytes() == marker_bytes and (pool / f"{j1}.SELF_EXIT").exists()
    assert not _events(w, "placement_superseded") and len(_events(w, "placement_complete")) == 1 and not _events(w, "torn_down")
    # dry-run writes nothing and runs no transaction
    r = w.run("--dry-run", "mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", "--supersede", "--reason", reason)
    assert r.returncode == 0 and r.stdout.startswith("DRY-RUN: mark-placed") and (pool / "PLACED").read_bytes() == before
    # ---- the REMEDY: the member's exit reaches sacct; the SAME pass again — open (refreshed), the walk reads the SELF_EXIT with no live member
    #      left and re-mints (the teardown recorded), the live targets are skipped, the derivation names the member and the targets, and
    #      mark-placed commits v2 onto the replacement alone: nothing to exclude, so no exclusion is named anywhere
    set_job_state(w.fake["cluster_a"], j1, "COMPLETED", end="2026-01-01T01:10:00")
    r = w.run("placement-transition", "open", "--group", GROUP, "--cluster", "cluster_a", "--reason", REASON)
    assert r.returncode == 0 and "/PLACED.transition refreshed on cluster_a" in r.stdout and "(an earlier marker since " in r.stdout, r.stdout + r.stderr
    r = _walk(w, 1); out = r.stdout + r.stderr
    assert r.returncode == 0 and "RESUBMIT" in r.stdout and "SELF-EXITED" in out and "never re-placed" not in out and f"recorded torn_down (self_exit) for job {j1}" in out, out
    j3 = str(job_ids(w.fake["cluster_a"])[-1]); assert j3 not in (j1, t1, t2, t3, t4)
    for unit, shards, t in (("crescendo/g012", "0,1,2", t3), ("crescendo/g34", "3,4", t4)):
        r = _walk_target(w, unit, shards); assert r.returncode == 0 and f"skipped cluster_a {t}" in r.stdout, r.stdout + r.stderr
    r = w.run("placement-changes", "--group", GROUP, "--targets", f"{lw1},{lw2}")
    assert r.returncode == 0 and r.stdout.strip() == "re-minted multiturn.attacker1,multiturn.crescendo.g012,multiturn.crescendo.g34", r.stdout + r.stderr
    reason2 = f"{REASON}: {r.stdout.strip()}"
    r = w.run("mark-placed", "--group", GROUP, "--targets", f"{lw1},{lw2}", "--supersede", "--reason", reason2); out = r.stdout + r.stderr
    assert r.returncode == 0 and f"SUPERSEDED version 1 -> 2 (2 added, 2 removed, 0 kept their member) — {reason2} (the pass's PLACED.transition marker cleared: the transition committed)" in out, out
    assert "EXCLUDED" not in out and "re-assigned" not in out and "NO LIVE MEMBER" not in out
    v2 = json.loads((pool / "PLACED").read_text())
    assert v2["version"] == 2 and v2["members"] == [j3] and [t["job_id"] for t in v2["targets"]] == [t3, t4] and all(t["assigned_member_job_id"] == j3 for t in v2["targets"]), v2
    assert v2["changes"] == {"added": [[lw1, t3], [lw2, t4]], "removed": [[lw1, t1], [lw2, t2]], "kept": 0, "members_before": [j1]} and "excluded_self_exited" not in v2
    assert json.loads((pool / "PLACED.v1").read_text()) == v1 and not marker.exists()
    sup = _events(w, "placement_superseded"); assert len(sup) == 1 and sup[0]["members"] == [j3] and "excluded_self_exited" not in sup[0], sup
    assert len(_events(w, "torn_down")) == 1
    # the replacement reads v2 and serves its assignment (t3, t4 live); the exited member's tombstone stays where the funnel reads it
    (pool / f"{j3}.json").write_text(json.dumps({"url": "http://n2:8709", "job_id": j3}))
    v, why = _verdict(w, root, "http://n2:8709", j3); assert v == "serve" and t3 in why and t4 in why, (v, why)


# ------------------------------------------------------------------------------------------------- the transaction over a bare pool root
def test_the_placement_transaction_excludes_self_exited_members_under_the_lock_on_every_decision_path(tmp_path):
    """PLACEMENT_TX_PY itself, over a bare pool root, each decision path with and without SELF_EXIT stems: without any, every answer and record is
    exactly the pre-round-6 shape (no new key anywhere); with one, the handed member is dropped before the decision — a first placement
    round-robins over the rest and names it; a record naming it is never `unchanged` (refused, needs_reason, or superseded with its unchanged
    pairs re-assigned and named); a record already free of it is `unchanged` with the exclusion named; none left answers `no_live_member`
    with nothing written and the marker left; a tombstone of a job NOT handed excludes nothing."""
    sys.path.insert(0, str(LIB))
    import submit_twin as st  # noqa: E402
    assert st.PLACEMENT_TX_PY.startswith(st.TRANSITION_READ_PY)
    root = tmp_path / "attackers"; root.mkdir()
    UTC = "2026-01-01T00:00:00Z"

    def rows(*pairs):
        return [{"logical_work_id": u, "job_id": j, "cluster": "c", "state": "submitted"} for u, j in pairs]

    def tx(members, rs, *, supersede=False, reason=None):
        intent = {"group": GROUP, "rows": rs, "members": list(members), "supersede": supersede, "reason": reason, "utc": UTC}
        r = subprocess.run([sys.executable, "-c", st.PLACEMENT_TX_PY, str(root), json.dumps(intent, sort_keys=True)], capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        lines = r.stdout.strip().splitlines(); assert len(lines) == 1, r.stdout
        return json.loads(lines[0])

    def placed_doc():
        return json.loads((root / "PLACED").read_text())

    def clear():
        for p in root.iterdir():
            p.unlink()

    def fresh_marker():
        (root / "PLACED.transition").write_text(json.dumps({"group": GROUP, "from_version": 1, "reason": "r", "lease_s": 7200, "epoch": int(time.time()), "utc": "u"}))
        return (root / "PLACED.transition").read_bytes()

    R2 = rows(("u/a", "11"), ("u/b", "12"))
    # ---- placed, no SELF_EXIT: the exact pre-round-6 answer and record
    res = tx(["1", "2"], R2)
    doc1 = {"group": GROUP, "targets": [dict(R2[0], assigned_member_job_id="1"), dict(R2[1], assigned_member_job_id="2")], "members": ["1", "2"], "n_targets": 2, "version": 1, "utc": UTC}
    assert res == {"result": "placed", "doc": doc1, "transition_cleared": False}, res
    assert placed_doc() == doc1 and (root / "PLACED").read_text() == json.dumps(doc1, sort_keys=True) + "\n"
    # ---- unchanged, no SELF_EXIT: exact; a tombstone of a job NOT handed changes nothing (excluded = handed ∩ exited)
    (root / "9.SELF_EXIT").write_text("{}")
    res = tx(["1", "2"], rows(("u/a", "11"), ("u/b", "12")))
    assert res == {"result": "unchanged", "doc": doc1, "prev_version": 1, "versions": [], "transition_cleared": False}, res
    (root / "9.SELF_EXIT").unlink()
    # ---- a record naming the exited member is NEVER unchanged: refused (the exclusion named, nothing written), needs_reason (named), superseded
    #      (members [2]; the pair on 1 re-assigned to 2 and named; the pair on 2 kept; the record and the answer name the exclusion)
    (root / "1.SELF_EXIT").write_text(json.dumps({"job_id": "1", "reason": "exit_last_consumer"}))
    mk = fresh_marker()
    res = tx(["1", "2"], rows(("u/a", "11"), ("u/b", "12")))
    assert res == {"result": "refused", "doc": doc1, "prev_version": 1, "prev_members": ["1", "2"], "versions": [], "added": [], "removed": [], "excluded_self_exited": ["1"]}, res
    assert placed_doc() == doc1 and not (root / "PLACED.v1").exists() and (root / "PLACED.transition").read_bytes() == mk
    res = tx(["1", "2"], rows(("u/a", "11"), ("u/b", "12")), supersede=True)
    assert res == {"result": "needs_reason", "doc": doc1, "prev_version": 1, "versions": [], "excluded_self_exited": ["1"]}, res
    assert placed_doc() == doc1 and (root / "PLACED.transition").read_bytes() == mk
    res = tx(["1", "2"], rows(("u/a", "11"), ("u/b", "12")), supersede=True, reason="the exited member's tick crossed the marker open")
    assert res["result"] == "superseded" and res["excluded_self_exited"] == ["1"] and res["prev_version"] == 1 and res["transition_cleared"] is True and not (root / "PLACED.transition").exists(), res
    doc2 = res["doc"]
    assert doc2["members"] == ["2"] and [t["assigned_member_job_id"] for t in doc2["targets"]] == ["2", "2"] and doc2["excluded_self_exited"] == ["1"] and doc2["version"] == 2
    assert doc2["changes"] == {"added": [], "removed": [], "kept": 1, "members_before": ["1", "2"], "excluded_self_exited": ["1"], "reassigned": [["u/a", "11", "1", "2"]]}, doc2["changes"]
    assert placed_doc() == doc2 and json.loads((root / "PLACED.v1").read_text()) == doc1
    # ---- a record already free of the exited member: unchanged, the exclusion named (truthful: the record assigns it nothing), nothing written
    res = tx(["1", "2"], rows(("u/a", "11"), ("u/b", "12")))
    assert res == {"result": "unchanged", "doc": doc2, "prev_version": 2, "versions": [doc1], "transition_cleared": False, "excluded_self_exited": ["1"]}, res
    assert placed_doc() == doc2
    # ---- new pairs round-robin over the remaining members only; an unchanged pair on a live member is kept; the one on the exited member moves
    (root / "3.SELF_EXIT").write_text("{}")
    res = tx(["1", "2", "3", "4"], rows(("u/a", "11"), ("u/b", "12"), ("u/c", "13"), ("u/d", "14"), ("u/e", "15")), supersede=True, reason="grown")
    d3 = res["doc"]
    assert d3["members"] == ["2", "4"] and [t["assigned_member_job_id"] for t in d3["targets"]] == ["2", "2", "2", "4", "2"] and d3["excluded_self_exited"] == ["1", "3"], d3   # two kept on 2; the three new ones 2, 4, 2
    assert d3["changes"] == {"added": [["u/c", "13"], ["u/d", "14"], ["u/e", "15"]], "removed": [], "kept": 2, "members_before": ["2"], "excluded_self_exited": ["1", "3"], "reassigned": []}, d3["changes"]
    # ---- superseded with NO SELF_EXIT: no new key on the answer, the record or its changes (byte-identical shape)
    clear(); (root / "PLACED").write_text(json.dumps(doc1, sort_keys=True) + "\n")
    res = tx(["1", "2", "3"], rows(("u/a", "11"), ("u/b", "12"), ("u/c", "13")), supersede=True, reason="a third target")
    assert set(res) == {"result", "doc", "prev_version", "versions", "transition_cleared"} and res["result"] == "superseded", res
    assert set(res["doc"]) == {"group", "targets", "members", "n_targets", "version", "supersedes", "reason", "changes", "utc"}
    assert res["doc"]["changes"] == {"added": [["u/c", "13"]], "removed": [], "kept": 2, "members_before": ["1", "2"]} and res["doc"]["members"] == ["1", "2", "3"]
    assert [t["assigned_member_job_id"] for t in res["doc"]["targets"]] == ["1", "2", "1"]
    # ---- no_live_member on a FIRST placement: nothing written, the marker left (fresh: named with what is left of its lease), the excluded named
    clear(); (root / "7.SELF_EXIT").write_text("{}"); mk = fresh_marker()
    res = tx(["7"], R2)
    assert res["result"] == "no_live_member" and res["excluded_self_exited"] == ["7"] and res["handed"] == ["7"] and res["doc"] is None and res["prev_version"] == 0 and res["versions"] == [], res
    assert res["marker_state"] == "fresh" and res["marker"]["lease_s"] == 7200 and 7000 < res["left_s"] <= 7200, res
    assert not (root / "PLACED").exists() and (root / "PLACED.transition").read_bytes() == mk and sorted(p.name for p in root.iterdir()) == ["7.SELF_EXIT", "PLACED.transition"]
    # a lapsed marker is named as such and left too; no marker -> absent
    (root / "PLACED.transition").write_text(json.dumps({"group": GROUP, "from_version": 0, "reason": "r", "lease_s": 10, "epoch": int(time.time()) - 100, "utc": "u"}))
    res = tx(["7"], R2); assert res["result"] == "no_live_member" and res["marker_state"] == "lapsed" and res["left_s"] < 0 and (root / "PLACED.transition").exists(), res
    (root / "PLACED.transition").unlink()
    res = tx(["7"], R2); assert res["result"] == "no_live_member" and res["marker_state"] == "absent" and res["marker"] is None and res["left_s"] is None, res
    # nothing handed at all is the same refusal (nothing to assign), with nothing excluded
    res = tx([], R2); assert res["result"] == "no_live_member" and res["excluded_self_exited"] == [] and res["handed"] == [] and not (root / "PLACED").exists(), res
    # ---- no_live_member against a STANDING record: the exclusion precedes every decision — with or without supersede, same rows or new ones —
    #      the record stands as it was (still naming the exited member: the re-run's supersession is what moves it), no archive, the marker left
    clear(); (root / "PLACED").write_text(json.dumps(doc1, sort_keys=True) + "\n"); (root / "1.SELF_EXIT").write_text("{}"); (root / "2.SELF_EXIT").write_text("{}"); mk = fresh_marker()
    for kw in ({}, {"supersede": True}, {"supersede": True, "reason": "both left"}):
        for rs in (rows(("u/a", "11"), ("u/b", "12")), rows(("u/a", "21"), ("u/b", "22"))):
            res = tx(["1", "2"], rs, **kw)
            assert res["result"] == "no_live_member" and res["excluded_self_exited"] == ["1", "2"] and res["doc"] == doc1 and res["prev_version"] == 1 and res["marker_state"] == "fresh", (kw, res)
            assert placed_doc() == doc1 and not (root / "PLACED.v1").exists() and (root / "PLACED.transition").read_bytes() == mk
    # ---- placed (first placement) with one of two members exited: the rest get everything, the marker cleared by the commit, the exclusion named
    clear(); (root / "2.SELF_EXIT").write_text("{}"); fresh_marker()
    res = tx(["1", "2"], rows(("u/a", "11"), ("u/b", "12"), ("u/c", "13")))
    d = res["doc"]
    assert res["result"] == "placed" and res["excluded_self_exited"] == ["2"] and res["transition_cleared"] is True and not (root / "PLACED.transition").exists(), res
    assert d["members"] == ["1"] and [t["assigned_member_job_id"] for t in d["targets"]] == ["1", "1", "1"] and d["excluded_self_exited"] == ["2"] and d["version"] == 1 and "changes" not in d, d
    assert placed_doc() == d
    # the grading host side: the ledger event built from a record carries the exclusion exactly when the record does
    ev = st.Coordinator._placement_event_of("cluster_a", GROUP, d)
    assert ev["event"] == "placement_complete" and ev["members"] == ["1"] and ev["excluded_self_exited"] == ["2"], ev
    ev = st.Coordinator._placement_event_of("cluster_a", GROUP, doc2)
    assert ev["event"] == "placement_superseded" and ev["excluded_self_exited"] == ["1"] and ev["changes"]["reassigned"] == [["u/a", "11", "1", "2"]] and ev["from_version"] == 1 and ev["to_version"] == 2, ev
    assert "excluded_self_exited" not in st.Coordinator._placement_event_of("cluster_a", GROUP, doc1)
    # the identity the ledger dedupes on is unchanged by the key (group, cluster, version, the (unit, job, member) set)
    assert st.Coordinator._placement_identity(ev) == st.Coordinator._placement_identity({k: v for k, v in ev.items() if k != "excluded_self_exited"})


# ---------------------------------------------------------------------------------------------------------------------- the shape
def test_the_exclusion_is_read_under_the_lock_before_every_decision_and_the_callers_surface_no_live_member_with_the_remedy():
    """Static pins: in PLACEMENT_TX_PY the SELF_EXIT read follows the record read and precedes the `no_live_member` exit, which precedes the
    first placement, the unchanged compare (against the LIVE list), the refusal and the supersession; the program indexes rows over the
    filtered `members` only; the coordinator handles `no_live_member` with rc 4 and the remedy (the same pass again), threads the exclusion
    into every other answer, reconciles the standing record's event on it, and the event builder carries the key; 06 reads mark-placed's
    exit code, prints the remedy (the same pass, verbatim) on 4 and the sanctioned supersession otherwise; the member side is untouched."""
    sys.path.insert(0, str(LIB))
    import submit_twin as st  # noqa: E402
    prog = st.PLACEMENT_TX_PY
    i_prev = prog.index('path = os.path.join(root, "PLACED")'); i_ex = prog.index('f.endswith(".SELF_EXIT")'); i_members = prog.index("members = [m for m in handed if m not in exited]")
    i_none = prog.index('"result": "no_live_member"'); i_first = prog.index("if not isinstance(prev, dict):"); i_cmp = prog.index("prev_members == members"); i_unch = prog.index('"result": "unchanged"')
    i_ref = prog.index('"result": "refused"'); i_sup = prog.index('"result": "superseded"'); i_kept = prog.index("kept = {k: v for k, v in prev_pairs.items() if k in new_pairs and v in members}")
    assert i_prev < i_ex < i_members < i_none < i_first < prog.index('"result": "placed"') < i_cmp < i_unch < i_ref < i_kept < i_sup
    assert "if not members:" in prog[i_members:i_none] and "sys.exit(0)" in prog[i_none:i_first] and "write(" not in prog[i_none:i_first]     # the refusal writes nothing
    assert prog.count("members[i % len(members)]") == 1 and prog.count("members[counter % len(members)]") == 1 and "handed[" not in prog     # rows index the LIVE list only
    assert 'if k in prev_pairs and prev_pairs[k] in excluded:' in prog and 'changes["excluded_self_exited"] = excluded; changes["reassigned"] = reassigned' in prog
    assert prog.count("**excl()") == 7 and 'return {"excluded_self_exited": excluded} if excluded else {}' in prog       # named only when it happened
    assert "transition_state(root)" in prog[i_members:i_none] and prog.index("def transition_state(root):") < i_prev     # the marker named on the refusal, read under the lock
    src = (LIB / "submit_twin.py").read_text()
    body = src[src.index("    def mark_placed(self, group: str"):src.index("    def _placement_event_of(")]
    i_res = body.index("res = self._placement_transaction("); i_excl = body.index('excluded = [str(m) for m in (res.get("excluded_self_exited") or [])]')
    i_nl = body.index('if result == "no_live_member":'); i_unch = body.index('if result == "unchanged":'); i_ref = body.index('if result == "refused":')
    assert i_res < i_excl < i_nl < i_unch < i_ref
    nl = body[i_nl:i_unch]
    assert 'raise Refused(' in nl and nl.rstrip().endswith(", 4)") and "REMEDY: once sacct reports job(s)" in nl and "run the same `06 --family " in nl and "multiturn --resubmit` pass again" in nl
    assert "nothing written: {standing}, no archive, no ledger event" in nl and "its lease bounds it" in nl and "placement-transition close --group {group} --cluster {label}" in nl
    assert "re-minted none" not in nl and "re-minted none" not in (REPO / "reproduce" / "06_eval_waves.sh").read_text()      # the 06 pin's forbidden provenance phrase, nowhere
    assert 'result in ("refused", "needs_reason", "no_live_member")' in body                       # the standing record's event reconciled on it too
    assert body.count("+ excl_note") == 3 and body.count("{excl_note}") == 2 and "members now {live_members}" in body and "members {live_members}" in body
    evb = src[src.index("    def _placement_event_of("):src.index("    def _placement_identity(")]
    assert 'if doc.get("excluded_self_exited"):' in evb and 'base["excluded_self_exited"]' in evb
    # 06: the exit code read first thing in the else branch; 4 -> the remedy (the same pass, verbatim), else the sanctioned supersession as before
    sh = (REPO / "reproduce" / "06_eval_waves.sh").read_text()
    assert subprocess.run(["bash", "-n", str(REPO / "reproduce" / "06_eval_waves.sh")], capture_output=True).returncode == 0
    i_mp = sh.index('mark-placed --group "$g" --targets "${GROUP_TARGETS[$g]}" ${MARK_EXTRA[@]+"${MARK_EXTRA[@]}"}; then')
    i_rc = sh.index("mp_rc=$?"); i_hold = sh.index('[ "$RESUBMIT" = 1 ] && hold_note='); i_if = sh.index('if [ "$mp_rc" != 4 ]; then')
    i_note = sh.index("do NOT arm wait-pool before it succeeds"); i_cmd = sh.index("the sanctioned transition is exactly:"); i_else = sh.index("NO LIVE MEMBER (above)"); i_rem = sh.index("followup: REMEDY for group $g")
    assert i_mp < i_rc < i_hold < i_if < i_note < i_cmd < i_else < i_rem and sh.count("mp_rc=$?") == 1
    assert sh[i_rc - 40:i_rc].rstrip().endswith("else") and sh.count('[ "$mp_rc" != 4 ]') == 1
    # the remedy is THIS pass's EFFECTIVE invocation (--nshards, --wave-sha, --cells and --units were dropped from a command the operator
    # is told to run verbatim): an array built BEFORE the echo with every value quoted by `printf '%q '`, carrying the effective shard
    # count and wave sha unconditionally, --cells / --units / --force-gate when given, the RESOLVED overrides document and config; a
    # --units selection is widened by the group's member units, recorded from the whole plan before the --units filter; no declared move
    # is re-passed; the old spelling is gone
    i_arr = sh.index('remedy=("$HERE/06_eval_waves.sh" --model "$K" --view "$VIEW" --family multiturn --clusters "$CLUSTERS" --nshards "$NSHARDS" --wave-sha "$WAVE_SHA" --resubmit)')
    assert i_else < i_arr < i_rem
    tail = sh[i_arr:i_rem]
    assert '[ -n "$CELLS_SEL" ] && remedy+=(--cells "$CELLS_SEL")' in tail and 'if [ -n "${UNITS:-}" ]; then' in tail and 'remedy+=(--units "$remedy_units")' in tail
    assert 'for mu in ${GROUP_MEMBERS[$g]:+${GROUP_MEMBERS[$g]//,/ }}; do case ",$remedy_units," in *",$mu,"*) ;; *) remedy_units="$remedy_units,$mu";; esac; done' in tail
    assert 'for f in ${FORCE[@]+"${FORCE[@]}"}; do remedy+=(--force-gate "$f"); done' in tail and 'remedy+=(--overrides "$SISTEER_RUN_OVERRIDES" --config "$SISTEER_CONFIG")' in tail
    assert 'remedy_cmd="$(shell_words "${remedy[@]}")"; remedy_cmd="${remedy_cmd% }"' in tail and "--move-group" not in tail and "--re-move" not in tail
    assert sh.index("shell_words(){ local v; for v in \"$@\"; do case \"$v\" in \"\"|*[!A-Za-z0-9_./:,=+@%-]*) printf '%q ' \"$v\";; *) printf '%s ' \"$v\";; esac; done; }") < i_mp
    assert "run exactly this pass again, in this environment (its effective flags; a --units selection widened by the group's member unit(s)): $remedy_cmd  (" in sh[i_rem:i_rem + 600]
    assert "--resubmit${REPRO_OVERRIDES:+ --overrides $REPRO_OVERRIDES} --config $SISTEER_CONFIG  (" not in sh          # the dropped-flags spelling, nowhere
    i_loop = sh.index("while IFS=$'\\t' read -r sf fam unit lwid cl kind grp script name array tkey twin_ok args env depends; do")
    i_gm = sh.index('[ "$grp" != - ] && [ "$kind" = attacker ] && GROUP_MEMBERS[$grp]="${GROUP_MEMBERS[$grp]:-}${GROUP_MEMBERS[$grp]:+,}$u"')
    i_units = sh.index('if [ -n "${UNITS:-}" ]; then case ",$UNITS," in *",$u,"*) ;; *) continue;; esac; fi')
    assert sh.index("declare -A GROUP_MEMBERS=()") < i_loop < sh.index('u="$fam.${unit//\\//.}"') < i_gm < i_units        # recorded for every row, before the selection
    assert "never hand-edit PLACED, a marker or a ledger" in sh[i_rem:i_rem + 1200] and "$hold_note" in sh[i_else:i_rem] and "do NOT arm wait-pool on it" in sh[i_else:i_rem]
    assert "re-minted none" not in sh and sh.count("placement-changes --group") == 1 and sh.count('echo "followup: arm the group-level backstop now:') == 1
    # the pass ends INCOMPLETE on a NO LIVE MEMBER refusal (it used to write WAVE.<view>.json, print `06 done` and exit 0): the rc-4 branch
    # records `<group>\t<label>\t<remedy>` right after the REMEDY echo; after the forced-admissions record and BEFORE the --units note and
    # the wave write, a non-empty record writes WAVE.<view>.partial.json and refuses with rc 4 — the one REPRO_REFUSE_RC override in the
    # stage — so neither the wave write nor `06 done` is reached; the rc-3 handling is untouched
    i_track = sh.index('UNPLACED+=("$g"$\'\\t\'"$l"$\'\\t\'"$remedy_cmd")')
    i_forced = sh.index('FORCED_JSON="[]"; FORCED_LINE="none"'); i_inc = sh.index('if [ ${#UNPLACED[@]} -gt 0 ]; then')
    i_pwrite = sh.index('repro_write "$STATE/WAVE.$VIEW.partial.json" "$partial"', i_inc); i_fatal = sh.index('REPRO_REFUSE_RC=4 repro_fatal "placement INCOMPLETE — group(s) $unplaced_line have NO LIVE MEMBER (above)', i_inc)
    i_units_note = sh.index('if [ -n "${UNITS:-}" ]; then\n  repro_say "units=$UNITS selected: the wave file is NOT written'); i_wave = sh.index('repro_write "$STATE/WAVE.$VIEW.json"'); i_done = sh.index('repro_say "06 done:')
    assert i_rem < i_track < i_forced < i_inc < i_pwrite < i_fatal < i_units_note < i_wave < i_done
    assert sh.count("UNPLACED=()") == 1 and sh.index("UNPLACED=()") < i_loop and sh.count("REPRO_REFUSE_RC=") == 1 and sh.count('"$remedy_cmd")') == 1
    blk = sh[i_inc:i_units_note]
    assert '"refusal": "no_live_member", "unplaced_groups": groups' in blk and '"failed_unit": None' in blk and 'r.split("\\t", 2)' in blk and '"${UNPLACED[@]}")"' in blk
    assert "WAVE.$VIEW.json is NOT written (an earlier pass's record stands as it was)" in blk and "run the printed REMEDY ($remedy_refs) then" in blk and "every other group's placement above stands" in blk
    assert "exit 0" not in sh[i_inc:] and "re-minted none" not in blk
    # the member side is untouched: the exit path writes SELF_EXIT and exits 0 with no lock, and the EXIT trap removes the record + pending marker
    sb = SBATCH.read_text()
    i_se = sb.index('> "$POOL_DIR/${SLURM_JOB_ID}.SELF_EXIT"\n      f22_log "SELF_EXIT ($verdict)'); i_exit = sb.index("exit 0;;", i_se)
    assert "flock" not in sb[i_se - 600:i_exit] and "PLACED.lock" not in sb and 'rm -f "$REC" "$PENDING"' in sb and "PLACED.lock" not in LIFE.read_text()
    assert "test_a_placement_committed_to_a_member_that_self_exited_across_the_marker_open_ends_the_target_fatal_and_binds_no_url" in (REPO / "tests" / "test_attacker_pool_lifecycle.py").read_text()


# ------------------------------------------------------------------------------------------------------------- 06, the real pass
def test_06_surfaces_no_live_member_with_the_remedy_and_the_same_pass_re_run_re_mints_the_member_and_commits(tmp_path):
    """The REAL 06 over the wave-integration World (the demo smoke's one member per label): the member's exit path crossed the marker open
    — its SELF_EXIT is on the shared root and its EXIT trap removed its records — while Slurm still lists the job RUNNING, and its targets
    died. The --resubmit pass keeps the member (its walk reads RUNNING), re-mints the targets, derives `re-minted <targets>` and runs
    mark-placed --supersede, which answers `no_live_member` (rc 4): 06 prints the NO LIVE MEMBER note and the REMEDY (this pass again,
    verbatim) instead of the sanctioned supersession, arms nothing on that label, and nothing is written on the cluster — PLACED v1 stands,
    no archive, no event, the marker fresh — and the pass ends INCOMPLETE (it used to write WAVE.<view>.json, print `06 done` and exit 0,
    a success an unattended caller had no reason to doubt, while the re-minted targets had no member to bind): the groups and their
    remedies are recorded in WAVE.<view>.partial.json, WAVE.<view>.json is NOT rewritten (the first pass's record stands byte for byte),
    no `06 done`, exit 4 with the refusal naming the group and the remedy. The remedy: the job COMPLETED on sacct; the same pass re-mints
    the member (SELF_EXIT, no live member left), skips the live targets, supersedes v1 -> v2 onto the replacement alone, no exclusion
    named, and completes (WAVE.<view>.json rewritten, `06 done`, exit 0); the other label is inert throughout."""
    import test_front_door_wave_integration as WI  # noqa: E402
    from test_06_resubmit_pass_supersedes_placement import DOC_REL, MODEL, VIEW, _placed, _stage06, _submitted  # noqa: E402
    w = WI.World(tmp_path)
    r = w.stage("05_stage_roster.sh", "--model", MODEL, "--step", "freeze", "--overrides", DOC_REL, SISTEER_RUN_OVERRIDES="")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    for lab in w.fake:
        (w.shared[lab] / "registry" / "records").mkdir(parents=True, exist_ok=True)
    r = _stage06(w); assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    members = {lab: j for lab, j, u in _submitted(r) if ".attacker" in u}; assert set(members) == {"cluster_a", "cluster_b"}
    v1 = {}; paths = {}
    for lab in ("cluster_a", "cluster_b"):
        paths[lab], v1[lab] = _placed(w, lab); assert v1[lab]["version"] == 1 and v1[lab]["members"] == [members[lab]]
    lab = max(("cluster_a", "cluster_b"), key=lambda l: v1[l]["n_targets"]); other = "cluster_b" if lab == "cluster_a" else "cluster_a"
    member = members[lab]; old_targets = sorted(t["job_id"] for t in v1[lab]["targets"]); group = f"mt-{VIEW}-{lab}"; pool = paths[lab].parent
    # the member's consumers ended (dead: the pass re-mints them); its exit path crossed the marker open of the pass below: SELF_EXIT written, its
    # record and pending marker gone, the job still RUNNING on Slurm
    for j in old_targets:
        set_job_state(w.fake[lab], j, "FAILED", end="2026-01-01T01:00:00")
    _member_exits(pool, member, "exit_last_consumer", consumers=",".join(old_targets), why="placement complete and every consumer terminal on sacct (bound or assigned)")
    assert (w.fake[lab] / "jobs" / f"{member}.state").read_text().strip() == "RUNNING" and not list(pool.glob(f"{member}.json")) and not list(pool.glob(f"{member}.pending"))
    before_other = paths[other].read_bytes(); before = paths[lab].read_bytes()
    wave_path = w.state / f"WAVE.{VIEW}.json"; partial_path = w.state / f"WAVE.{VIEW}.partial.json"
    wave_before = wave_path.read_bytes(); assert json.loads(wave_before)["view"] == VIEW and not partial_path.exists()      # the first pass's record
    # ---- the pass: the member kept (RUNNING), the targets re-minted, mark-placed refused NO LIVE MEMBER; nothing written on the cluster; the
    #      pass ends INCOMPLETE — exit 4, the refusal naming the group, WAVE.<view>.json NOT rewritten, the partial record naming group + remedy
    r = _stage06(w, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 4, out[-5000:]
    assert "06 done" not in r.stdout and wave_path.read_bytes() == wave_before, out[-3000:]
    assert (f"REFUSED: placement INCOMPLETE — group(s) {group} (on {lab}) have NO LIVE MEMBER (above): the targets this pass re-minted for them have no member to bind"
            in r.stderr and f"run the printed REMEDY (`followup: REMEDY for group {group}`)" in r.stderr and f"WAVE.{VIEW}.json is NOT written" in r.stderr
            and f"recorded in {w.state}/WAVE.{VIEW}.partial.json" in r.stderr), r.stderr[-3000:]
    subs = _submitted(r)
    assert len(subs) == len(old_targets) and all(l == lab and ".attacker" not in u for l, _, u in subs), subs
    assert f"skipped {lab} {member} multiturn.attacker1 (still RUNNING" in r.stdout and "SELF-EXITED" not in out and "recorded torn_down" not in out, out[-5000:]
    assert f"placement-changes {group}: re-minted since version 1: " in r.stderr and "dropped since version" not in r.stderr, r.stderr[-3000:]
    assert (f"REFUSED: mark-placed {group}: NO LIVE MEMBER — every member the ledger records for the group (['{member}']) has attackers/<job>.SELF_EXIT on {lab} under the group's "
            f"lock (excluded ['{member}'])") in r.stderr and "REMEDY: once sacct reports job(s) " in r.stderr and "the pass's PLACED.transition marker stands as before (since " in r.stderr, r.stderr[-4000:]
    note = [ln for ln in r.stdout.splitlines() if f"note: group {group}: NO LIVE MEMBER (above)" in ln]
    assert len(note) == 1 and f"has attackers/<job>.SELF_EXIT on {lab} (read under the group's lock)" in note[0] and "do NOT arm wait-pool on it" in note[0] \
        and "hold every exit while this pass's attackers/PLACED.transition marker is fresh" in note[0] and f"placement-transition close --group {group} --cluster {lab}" in note[0], note
    remedy = [ln for ln in r.stdout.splitlines() if f"followup: REMEDY for group {group}" in ln]
    cfg = w.site / "config.env"
    # the printed pass is this pass's EFFECTIVE invocation: the default shard count and the wave sha spelled out, the
    # overrides document RESOLVED (the pass was given a relative path), the config, no --units / --cells / --force-gate (none given)
    assert len(remedy) == 1 and "run exactly this pass again, in this environment (its effective flags; a --units selection widened by the group's member unit(s)): " in remedy[0], remedy
    argv = _remedy_argv(remedy[0]); i_o = argv.index("--overrides")
    assert Path(argv[0]).resolve() == (w.repo / "reproduce" / "06_eval_waves.sh").resolve(), argv
    assert argv[1:i_o] == ["--model", MODEL, "--view", VIEW, "--family", "multiturn", "--clusters", "cluster_a,cluster_b", "--nshards", "9", "--wave-sha", w.sha, "--resubmit"], argv
    assert argv[i_o + 2:] == ["--config", argv[-1]] and Path(argv[i_o + 1]).resolve() == (w.repo / DOC_REL).resolve() and Path(argv[-1]).resolve() == cfg.resolve(), argv
    assert f"--clusters cluster_a,cluster_b --nshards 9 --wave-sha {w.sha} --resubmit --overrides " in remedy[0]        # plain values are printed plain, not `%q`-escaped
    assert "re-mints the member" in remedy[0] and "never hand-edit PLACED, a marker or a ledger" in remedy[0]
    assert "the sanctioned transition is exactly:" not in r.stdout and "is not fully placed yet" not in r.stdout and r.stdout.count("followup: arm the group-level backstop now") == 1
    # the INCOMPLETE record: the finding-8 vehicle (ignored by the daemon's wave discovery and the sha-table refresh), naming the group, its
    # label and the remedy EXACTLY as printed, this pass's wave identity and every unit the pass walked (submitted or skipped)
    pdoc = json.loads(partial_path.read_text())
    assert pdoc["refusal"] == "no_live_member" and pdoc["unplaced_groups"] == [{"group": group, "cluster": lab, "remedy": _remedy_cmd(remedy[0])}], pdoc
    assert shlex.split(pdoc["unplaced_groups"][0]["remedy"]) == argv and pdoc["failed_unit"] is None
    assert {k: pdoc[k] for k in ("model", "view", "families", "clusters", "wave_sha", "forced")} == {"model": MODEL, "view": VIEW, "families": "multiturn", "clusters": "cluster_a,cluster_b", "wave_sha": w.sha, "forced": []}, pdoc
    walked = sorted(ln.split()[3] for ln in r.stdout.splitlines() if ln.startswith(("submitted ", "skipped ")))
    assert sorted(pdoc["submitted"]) == walked and len(walked) == json.loads(wave_before)["n_units"] and pdoc["utc"].endswith("Z"), (pdoc["submitted"], walked)
    assert paths[lab].read_bytes() == before and not (pool / "PLACED.v1").exists() and not list(pool.glob("PLACED.tmp.*")) and (pool / f"{member}.SELF_EXIT").exists()
    assert not [e for e in w.attempts() if e["event"] == "placement_superseded"] and not [e for e in w.attempts() if e["event"] == "torn_down"]
    assert len([e for e in w.attempts() if e["event"] == "placement_complete"]) == 2
    marker = pool / "PLACED.transition"; mdoc = json.loads(marker.read_text())
    assert mdoc["from_version"] == 1 and mdoc["reason"] == f"06 --resubmit pass on {w.sha}" and not (paths[other].parent / "PLACED.transition").exists()
    assert paths[other].read_bytes() == before_other and "already placed, unchanged" in out
    new_targets = sorted(j for _, j, _ in subs)
    # ---- the REMEDY, as printed: the member's exit reaches sacct and the same pass runs again — the walk reads the SELF_EXIT with no live member
    #      left and re-mints (the teardown recorded), the re-minted targets are live and skipped, and mark-placed supersedes onto the replacement
    set_job_state(w.fake[lab], member, "COMPLETED", end="2026-01-01T01:10:00")
    r = _run_printed(w, remedy[0], SISTEER_RUN_OVERRIDES=""); out = r.stdout + r.stderr          # the PRINTED command, as the operator runs it
    assert r.returncode == 0, out[-5000:]
    # the pass COMPLETES: the wave record rewritten by this pass (the first pass's record stood untouched through the refused one), `06 done`
    assert "06 done: " in r.stdout and "INCOMPLETE" not in out and wave_path.read_bytes() != wave_before, out[-3000:]
    wdoc = json.loads(wave_path.read_text()); assert wdoc["n_units"] == json.loads(wave_before)["n_units"] and wdoc["wave_sha"] == w.sha and wdoc["utc"] >= pdoc["utc"], wdoc
    subs2 = _submitted(r)
    assert len(subs2) == 1 and subs2[0][0] == lab and subs2[0][2] == "multiturn.attacker1", subs2
    new_member = subs2[0][1]; assert new_member != member
    assert "SELF-EXITED" in out and "never re-placed" not in out and f"recorded torn_down (self_exit) for job {member} on {lab}" in out, out[-5000:]
    assert sum(1 for ln in r.stdout.splitlines() if ln.startswith("skipped ") and f" {lab} " in ln) == len(old_targets), r.stdout[-4000:]
    torn = [e for e in w.attempts() if e["event"] == "torn_down"]
    assert len(torn) == 1 and torn[0]["job_id"] == member and torn[0]["self_exit"] is True and torn[0]["cluster"] == lab, torn
    path, v2 = _placed(w, lab)
    assert v2["version"] == 2 and v2["members"] == [new_member] and sorted(t["job_id"] for t in v2["targets"]) == new_targets, v2
    assert all(t["assigned_member_job_id"] == new_member for t in v2["targets"]) and v2["changes"]["kept"] == 0 and v2["changes"]["members_before"] == [member]
    assert "excluded_self_exited" not in v2 and "excluded_self_exited" not in v2["changes"] and "reassigned" not in v2["changes"]
    assert v2["reason"].startswith(f"06 --resubmit pass on {w.sha}: re-minted multiturn.attacker1,") and len(v2["reason"].split("re-minted ", 1)[1].split(",")) == 1 + len(old_targets), v2["reason"]
    assert json.loads((pool / "PLACED.v1").read_text()) == v1[lab] and not marker.exists()
    assert f"/PLACED.transition refreshed on {lab}" in r.stdout and "SUPERSEDED version 1 -> 2" in out and "(the pass's PLACED.transition marker cleared: the transition committed)" in out, out[-5000:]
    assert "NO LIVE MEMBER" not in out and "EXCLUDED" not in out and "REMEDY" not in out and "the sanctioned transition is exactly:" not in r.stdout
    assert r.stdout.count("followup: arm the group-level backstop now") == 2
    sup = [e for e in w.attempts() if e["event"] == "placement_superseded"]
    assert len(sup) == 1 and sup[0]["group"] == group and sup[0]["members"] == [new_member] and "excluded_self_exited" not in sup[0] and sup[0]["reason"] == v2["reason"], sup
    assert paths[other].read_bytes() == before_other and len([e for e in w.attempts() if e["event"] == "placement_complete"]) == 2
    # ---- a further pass: inert on both labels
    r = _stage06(w, "--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and not _submitted(r) and out.count("already placed, unchanged") == 2 and "EXCLUDED" not in out and "NO LIVE MEMBER" not in out, out[-3000:]
    assert json.loads(path.read_text()) == v2 and len([e for e in w.attempts() if e["event"] == "placement_superseded"]) == 1 and len([e for e in w.attempts() if e["event"] == "torn_down"]) == 1


def test_the_printed_remedy_is_the_effective_pass_and_runs_at_a_non_default_shard_count_with_a_units_selection_widened_by_the_member(tmp_path):
    """The remedy 06 printed on `no_live_member` dropped --nshards, --wave-sha, --cells and --units although the operator is told to run
    it verbatim — at a non-default shard count the plan refused it before the member could be re-minted, and without --units it would submit
    other units. The REAL 06 over the wave-integration World on the PAPER plan at a NON-default shard count: the
    inventory frozen at --nshards 3 under SISTEER_SHARD_GROUPS "0 1 2" through freeze-inventory's own call (05 freezes at the default and takes
    no --nshards), the member cap SISTEER_MT_ATTACKER_GPUS=1 sizing ONE member per label, nine targets (three attacks x three groups of one shard)
    over the two labels. The label with more targets: every target dies and the member's exit path crosses the marker open (SELF_EXIT, records
    gone, the job RUNNING); the --resubmit pass is a --units SELECTION naming the dead targets alone (the walk never visits the member), and
    mark-placed answers NO LIVE MEMBER. The printed remedy carries --nshards 3, --wave-sha <this tree>, the selection WIDENED by
    multiturn.attacker1, the RESOLVED overrides document (the paper defaults, given by no flag) and the config, plain values printed plain. Then:
    (a) the dropped-flags spelling (no --nshards: the default 9 under the pinned groups) is refused by the plan with nothing submitted and nothing
    written — the original failure; (b) with the job's exit on sacct, the selection AS GIVEN is refused NO LIVE MEMBER again — the member is not
    visited, so nothing re-mints it; (c) the printed command re-mints the member (its teardown recorded), skips the live targets and commits v2
    onto the replacement, the marker cleared; the other label is `unchanged` throughout."""
    import test_front_door_wave_integration as WI  # noqa: E402
    from test_06_resubmit_pass_supersedes_placement import MODEL, VIEW, _placed, _submitted  # noqa: E402
    knobs = dict(SISTEER_RUN_OVERRIDES="", SISTEER_SHARD_GROUPS="0 1 2", SISTEER_MT_ATTACKER_GPUS="1")
    w = WI.World(tmp_path)
    # the freeze at the non-default count: 05's own freeze call (05_stage_roster.sh `want freeze`) with --nshards 3, which 05 does not take
    (w.state / "gate").mkdir(parents=True, exist_ok=True)
    r = subprocess.run([sys.executable, str(w.repo / "reproduce" / "lib" / "stage.py"), "freeze-inventory", "--model", MODEL,
                        "--roster", str(w.repo / "configs" / "sdfbase_full22_roster.yaml"), "--models-table", str(w.repo / "reproduce" / "models.yaml"),
                        "--code-sha", w.sha, "--out", str(w.state / "gate" / "expected_units.json"), "--clusters", "cluster_a,cluster_b", "--nshards", "3"],
                       capture_output=True, text=True, env=w.env(**knobs), cwd=str(w.repo), timeout=900)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    inv = w.inventory()
    assert inv["nshards"] == 3 and inv["shard_groups"] == [[0], [1], [2]] and inv["plan_knobs"]["SISTEER_SHARD_GROUPS"] == "0 1 2" and inv["plan_knobs"]["SISTEER_MT_ATTACKER_GPUS"] == "1", inv["plan_knobs"]
    for lab in w.fake:
        (w.shared[lab] / "registry" / "records").mkdir(parents=True, exist_ok=True)

    def stage06(*args):
        return w.stage("06_eval_waves.sh", "--model", MODEL, "--view", VIEW, "--family", "multiturn", "--nshards", "3", *args, **knobs)

    # ---- the wave at the non-default count: nine targets in groups g0 / g1 / g2, one member per label
    r = stage06(); assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    subs = _submitted(r)
    members = {l: j for l, j, u in subs if u == "multiturn.attacker1"}
    assert set(members) == {"cluster_a", "cluster_b"} and not [u for _, _, u in subs if ".attacker" in u and u != "multiturn.attacker1"], subs
    targets = {l: [(j, u) for l2, j, u in subs if l2 == l and ".attacker" not in u] for l in members}
    assert sum(len(v) for v in targets.values()) == 9 and {u.rsplit(".", 1)[1] for v in targets.values() for _, u in v} == {"g0", "g1", "g2"}, subs
    assert r.stdout.count("followup: arm the group-level backstop now") == 2
    lab = max(members, key=lambda l: (len(targets[l]), l)); other = "cluster_b" if lab == "cluster_a" else "cluster_a"
    member = members[lab]; group = f"mt-{VIEW}-{lab}"
    paths, v1 = {}, {}
    for l in members:
        paths[l], v1[l] = _placed(w, l); assert v1[l]["version"] == 1 and v1[l]["members"] == [members[l]] and v1[l]["n_targets"] == len(targets[l]), v1[l]
    pool = paths[lab].parent
    dead = sorted(u for _, u in targets[lab]); dead_units = ",".join(dead)
    for j, _ in targets[lab]:
        set_job_state(w.fake[lab], j, "FAILED", end="2026-01-01T01:00:00")
    _member_exits(pool, member, "exit_last_consumer", consumers=",".join(j for j, _ in targets[lab]), why="placement complete and every consumer terminal on sacct (bound or assigned)")
    assert (w.fake[lab] / "jobs" / f"{member}.state").read_text().strip() == "RUNNING" and not (pool / f"{member}.json").exists()
    before, before_other = paths[lab].read_bytes(), paths[other].read_bytes()
    wave_path = w.state / f"WAVE.{VIEW}.json"; wave_before = wave_path.read_bytes()                  # the first (full) pass's record
    # ---- the no-live-member pass: a --units selection naming the dead targets alone — the member unit is never visited; the pass ends
    #      INCOMPLETE (exit 4, the partial record) exactly as a full pass does — a selected call writes no wave file either way
    r = stage06("--resubmit", "--units", dead_units); out = r.stdout + r.stderr
    pdoc = _incomplete(r, w.state, VIEW, group, lab)
    subs1 = _submitted(r)
    assert sorted(u for _, _, u in subs1) == dead and all(l == lab for l, _, _ in subs1), subs1
    assert not [ln for ln in r.stdout.splitlines() if ln.startswith("skipped ") and ".attacker" in ln] and "recorded torn_down" not in out, out[-5000:]
    assert f"REFUSED: mark-placed {group}: NO LIVE MEMBER" in r.stderr and f"(excluded ['{member}'])" in r.stderr and "06 prints it as `followup: REMEDY`" in r.stderr, r.stderr[-4000:]
    assert "selected: the wave file is NOT written" not in r.stdout and "already placed, unchanged" in out and wave_path.read_bytes() == wave_before
    assert sorted(pdoc["submitted"]) == dead and pdoc["clusters"] == "cluster_a,cluster_b" and pdoc["wave_sha"] == w.sha, pdoc
    remedy = [ln for ln in r.stdout.splitlines() if f"followup: REMEDY for group {group}" in ln]
    assert len(remedy) == 1 and "the sanctioned transition is exactly:" not in r.stdout, r.stdout[-4000:]
    argv = _remedy_argv(remedy[0]); i_o = argv.index("--overrides")
    assert Path(argv[0]).resolve() == (w.repo / "reproduce" / "06_eval_waves.sh").resolve(), argv
    assert argv[1:i_o] == ["--model", MODEL, "--view", VIEW, "--family", "multiturn", "--clusters", "cluster_a,cluster_b", "--nshards", "3", "--wave-sha", w.sha, "--resubmit",
                           "--units", f"{dead_units},multiturn.attacker1"], argv
    assert argv[i_o + 2:] == ["--config", argv[-1]] and Path(argv[i_o + 1]).resolve() == (w.repo / "reproduce" / "overrides" / "paper_defaults.env").resolve() \
        and Path(argv[-1]).resolve() == (w.site / "config.env").resolve(), argv
    assert not {"--cells", "--force-gate", "--move-group", "--move-reason", "--re-move", "--twin"} & set(argv)
    assert f"--nshards 3 --wave-sha {w.sha} --resubmit --units {dead_units},multiturn.attacker1 --overrides " in remedy[0], remedy[0]      # plain values plain
    assert paths[lab].read_bytes() == before and not (pool / "PLACED.v1").exists() and (pool / "PLACED.transition").exists() and paths[other].read_bytes() == before_other
    assert not [e for e in w.attempts() if e["event"] in ("placement_superseded", "torn_down")]
    new_targets = sorted(j for _, j, _ in subs1)
    # ---- (a) the finding's failure: the dropped-flags spelling of the same pass (no --nshards, --wave-sha, --units) — the default shard count
    #      under the pinned groups is refused by the plan before anything is submitted; nothing written
    pre_fix = _without(argv, "--nshards", "--wave-sha", "--units")
    assert "--nshards" not in pre_fix and "--units" not in pre_fix and len(pre_fix) == len(argv) - 6
    r = subprocess.run(["bash", *pre_fix], capture_output=True, text=True, env=w.env(**knobs), cwd=str(w.repo), timeout=900); out = r.stdout + r.stderr
    assert r.returncode != 0 and not _submitted(r) and "REFUSED" in r.stderr and "mark-placed" not in out, out[-4000:]
    assert paths[lab].read_bytes() == before and not (pool / "PLACED.v1").exists() and paths[other].read_bytes() == before_other
    assert not [e for e in w.attempts() if e["event"] in ("placement_superseded", "torn_down")]
    # ---- the member's exit reaches sacct
    set_job_state(w.fake[lab], member, "COMPLETED", end="2026-01-01T01:10:00")
    # ---- (b) the selection AS GIVEN: the walk never visits the member, so nothing re-mints it — NO LIVE MEMBER again, nothing written, the
    #      pass INCOMPLETE again (the partial record rewritten with the same group and remedy)
    r = stage06("--resubmit", "--units", dead_units); out = r.stdout + r.stderr
    pdoc_b = _incomplete(r, w.state, VIEW, group, lab)
    assert not _submitted(r) and f"REFUSED: mark-placed {group}: NO LIVE MEMBER" in r.stderr and "recorded torn_down" not in out, out[-5000:]
    assert pdoc_b["unplaced_groups"] == pdoc["unplaced_groups"] and sorted(pdoc_b["submitted"]) == dead and wave_path.read_bytes() == wave_before
    assert sum(1 for ln in r.stdout.splitlines() if ln.startswith("skipped ") and f" {lab} " in ln) == len(dead) and not [ln for ln in r.stdout.splitlines() if ln.startswith("skipped ") and ".attacker" in ln]
    assert paths[lab].read_bytes() == before and not (pool / "PLACED.v1").exists() and len([ln for ln in r.stdout.splitlines() if f"followup: REMEDY for group {group}" in ln]) == 1
    # ---- (c) the PRINTED command: the member visited and re-minted (SELF_EXIT + COMPLETED: torn_down recorded), the live targets skipped,
    #      v2 onto the replacement alone, the marker cleared; the other label `unchanged`; the pass completes (a selected call: no wave file)
    r = _run_printed(w, remedy[0], **knobs); out = r.stdout + r.stderr
    assert r.returncode == 0 and "06 done: " in r.stdout and "INCOMPLETE" not in out and wave_path.read_bytes() == wave_before, out[-5000:]
    subs2 = _submitted(r)
    assert len(subs2) == 1 and subs2[0][0] == lab and subs2[0][2] == "multiturn.attacker1", subs2
    new_member = subs2[0][1]; assert new_member != member
    assert "SELF-EXITED" in out and f"recorded torn_down (self_exit) for job {member} on {lab}" in out, out[-5000:]
    assert sum(1 for ln in r.stdout.splitlines() if ln.startswith("skipped ") and f" {lab} " in ln) == len(dead), r.stdout[-4000:]
    torn = [e for e in w.attempts() if e["event"] == "torn_down"]
    assert len(torn) == 1 and torn[0]["job_id"] == member and torn[0]["self_exit"] is True and torn[0]["cluster"] == lab, torn
    path, v2 = _placed(w, lab)
    assert v2["version"] == 2 and v2["members"] == [new_member] and sorted(t["job_id"] for t in v2["targets"]) == new_targets, v2
    assert all(t["assigned_member_job_id"] == new_member for t in v2["targets"]) and v2["changes"]["members_before"] == [member] and "excluded_self_exited" not in v2, v2
    assert v2["reason"].startswith(f"06 --resubmit pass on {w.sha}: re-minted multiturn.attacker1,"), v2["reason"]
    assert json.loads((pool / "PLACED.v1").read_text()) == v1[lab] and not (pool / "PLACED.transition").exists()
    assert "SUPERSEDED version 1 -> 2" in out and "NO LIVE MEMBER" not in out and "REMEDY" not in out and r.stdout.count("followup: arm the group-level backstop now") == 2, out[-5000:]
    sup = [e for e in w.attempts() if e["event"] == "placement_superseded"]
    assert len(sup) == 1 and sup[0]["group"] == group and sup[0]["members"] == [new_member], sup
    assert paths[other].read_bytes() == before_other and "already placed, unchanged" in out and f"units={dead_units},multiturn.attacker1 selected: the wave file is NOT written" in r.stdout
    # ---- a further plain --resubmit pass at the same count: inert on both labels, and it completes the wave record (the full pass rewrites it)
    r = stage06("--resubmit"); out = r.stdout + r.stderr
    assert r.returncode == 0 and not _submitted(r) and out.count("already placed, unchanged") == 2 and "NO LIVE MEMBER" not in out, out[-3000:]
    assert "06 done: " in r.stdout and "INCOMPLETE" not in out and wave_path.read_bytes() != wave_before and json.loads(wave_path.read_text())["n_units"] == json.loads(wave_before)["n_units"]
    assert json.loads(path.read_text()) == v2 and len([e for e in w.attempts() if e["event"] == "placement_superseded"]) == 1
