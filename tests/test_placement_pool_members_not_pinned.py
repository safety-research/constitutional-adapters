"""Pool members are RECORDED in the placement manifest, never pinned against today's pool sizing (commit 31c).

Commit 31 made the shared attacker pool's size a function of its targets (one member per three; the env knob a cap). The live u0
manifest had been pinned when the pool had four members per cluster, so after 31 landed `placement-manifest ensure` and `check` refused
the whole multi-turn family — "attackers/3: pinned but not planned (the plan lost a multi-turn unit of the wave)" — and the front door
closed on a sizing change (found by a live read, 2026-10-01 03:1xZ). Targets stay strict: a pinned target the plan no longer carries, or a
moved target, still refuses by name; a pool member's cluster must still not move."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"
SHA = "a" * 40
AB = ["cluster_a", "cluster_b"]
TARGET = f"{KEY}/battery/multiturn/crescendo/g34"


def _four_member_manifest() -> dict:
    """The live manifest's shape: today's plan (two members per cluster) plus the members the wave ran under the four-member sizing."""
    man = ST.placement_from_plan(KEY, "battery", AB, sha=SHA)
    members = [lw for lw, u in man["units"].items() if u.get("kind") == "attacker"]
    assert len(members) == 4 and all("/attackers/" in lw for lw in members), members          # commit 31: 6 targets per cluster -> 2 members each
    for lab in AB:
        for i in (3, 4):
            man["units"][f"{KEY}/battery/multiturn/mt-battery-{lab}/attackers/{i}"] = {"cluster": lab, "group": f"mt-battery-{lab}", "kind": "attacker"}
    return man


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "state"; (root / "gate").mkdir(parents=True)
    ST.write_atomic(str(root / "gate" / "expected_units.json"), {"schema": 2, "model": KEY, "view": "battery", "code_sha": SHA, "frozen_utc": "2026-01-02T00:00:00Z", "units": []})
    return root


def _ev(root: Path, *events):
    with open(root / "attempts.jsonl", "a") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def _attempt(lw, cluster, job, utc, group=None, kind="target"):
    return {"event": "attempt", "logical_work_id": lw, "attempt_id": f"{lw}@{cluster}@{job}", "cluster": cluster, "job_id": job, "kind": kind, "sha": SHA,
            "spec": {"group": group or f"mt-battery-{cluster}", "kind": kind, "sha": SHA}, "utc": utc}


def test_recorded_pool_members_beyond_todays_sizing_are_neither_lost_nor_grown_while_targets_stay_strict(tmp_path):
    man = _four_member_manifest(); rows = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    assert ST.is_pool_member(f"{KEY}/battery/multiturn/mt-battery-cluster_a/attackers/3", {"kind": "attacker"}) and not ST.is_pool_member(TARGET, {"kind": "target"})
    assert ST.check_placement(man, AB, rows) == []                                                  # the four-member record passes today's two-member plan
    # a manifest that lacks a member today's plan sizes is not "grown" either
    lean = json.loads(json.dumps(man)); lean["units"].pop(f"{KEY}/battery/multiturn/mt-battery-cluster_a/attackers/1")
    assert ST.check_placement(lean, AB, rows) == []
    # a pinned TARGET the plan lost, and a moved target, still refuse by name
    extra = json.loads(json.dumps(man)); extra["units"][f"{KEY}/battery/multiturn/crescendo/g99"] = {"cluster": "cluster_a", "group": "mt-battery-cluster_a", "kind": "target"}
    assert [p for p in ST.check_placement(extra, AB, rows) if "crescendo/g99" in p and "not planned" in p]
    moved = json.loads(json.dumps(man)); moved["units"][TARGET] = {"cluster": "cluster_a", "group": "mt-battery-cluster_a", "kind": "target"}
    assert [p for p in ST.check_placement(moved, AB, rows) if "crescendo/g34" in p and "the placement moved" in p]
    # a pool member whose pin MOVED cluster (a real placement change) still refuses
    moved_member = json.loads(json.dumps(man)); moved_member["units"][f"{KEY}/battery/multiturn/mt-battery-cluster_a/attackers/1"]["cluster"] = "cluster_b"
    assert [p for p in ST.check_placement(moved_member, AB, rows) if "attackers/1" in p and "the placement moved" in p]


def test_ensure_keeps_the_recorded_members_and_the_members_attempts_reconcile_against_their_group_not_the_count(tmp_path):
    root = _root(tmp_path); ledger = root / "attempts.jsonl"; man = _four_member_manifest()
    # the wave's attempts: four members per cluster (the old sizing) and the pinned target, each on its group's cluster
    _ev(root, *[_attempt(f"{KEY}/battery/multiturn/mt-battery-{lab}/attackers/{i}", lab, str(100 + 10 * j + i), f"2026-01-04T0{j}:0{i}:00Z", group=f"mt-battery-{lab}", kind="attacker")
                for j, lab in enumerate(AB) for i in (1, 2, 3, 4)],
        _attempt(TARGET, "cluster_b", "80", "2026-01-04T01:00:00Z"))
    out, added, probs = ST.expand_placement(man, str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs == [] and added == [] and all(f"{KEY}/battery/multiturn/mt-battery-{lab}/attackers/{i}" in out["units"] for lab in AB for i in (1, 2, 3, 4)), probs
    man2, probs2 = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs2 == [] and man2["units"][TARGET]["cluster"] == "cluster_b"
    # a RECORDED member (pinned in the manifest) that later ran on ANOTHER cluster: its pin moved — refused by name like any placed unit
    _ev(root, _attempt(f"{KEY}/battery/multiturn/mt-battery-cluster_a/attackers/4", "cluster_b", "999", "2026-01-05T00:00:00Z", group="mt-battery-cluster_a", kind="attacker"))
    _, _, probs3 = ST.expand_placement(man, str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs3 and "attackers/4" in probs3[0] and "cluster_b" in probs3[0] and "placed it elsewhere" in probs3[0], probs3
    # a member NEITHER the manifest nor the plan knows (a fifth, from an older sizing) that ran off its group's cluster: still a problem, by
    # its GROUP — never "the plan places no such unit"
    root2 = _root(tmp_path / "r2"); ledger2 = root2 / "attempts.jsonl"
    _ev(root2, _attempt(f"{KEY}/battery/multiturn/mt-battery-cluster_a/attackers/5", "cluster_b", "998", "2026-01-05T00:00:00Z", group="mt-battery-cluster_a", kind="attacker"),
        _attempt(TARGET, "cluster_b", "80", "2026-01-04T01:00:00Z"))
    _, _, probs4 = ST.expand_placement(_four_member_manifest(), str(ledger2), KEY, "battery", AB, sha=SHA)
    assert probs4 and "attackers/5" in probs4[0] and "a member serves its group's cluster" in probs4[0] and not any("places no such unit" in p for p in probs4), probs4
    # … and one that ran on its group's cluster is no problem at all
    root3 = _root(tmp_path / "r3"); ledger3 = root3 / "attempts.jsonl"
    _ev(root3, _attempt(f"{KEY}/battery/multiturn/mt-battery-cluster_a/attackers/5", "cluster_a", "997", "2026-01-05T00:00:00Z", group="mt-battery-cluster_a", kind="attacker"),
        _attempt(TARGET, "cluster_b", "80", "2026-01-04T01:00:00Z"))
    assert ST.expand_placement(_four_member_manifest(), str(ledger3), KEY, "battery", AB, sha=SHA)[2] == []


def test_the_cli_ensure_and_check_pass_over_the_live_manifests_shape(tmp_path):
    root = _root(tmp_path); path = root / "gate" / ST.placement_file("battery"); ST.write_atomic(str(path), _four_member_manifest())
    _ev(root, _attempt(TARGET, "cluster_b", "80", "2026-01-04T01:00:00Z"))
    before = path.read_text()
    def cli(*args):
        return subprocess.run([sys.executable, str(REPO / "reproduce" / "lib" / "stage.py"), "placement-manifest", *args, "--model", KEY, "--view", "battery", "--clusters", ",".join(AB),
                               "--ledger", str(root / "attempts.jsonl"), "--sha", SHA, "--nshards", "9"], capture_output=True, text=True, env={**os.environ, "PYTHONPATH": str(REPO / "src")}, timeout=600)
    r = cli("ensure"); assert r.returncode == 0 and r.stdout.startswith("placement manifest: ") and path.read_text() == before, r.stdout + r.stderr
    r = cli("check"); assert r.returncode == 0 and r.stdout.startswith("placement OK"), r.stdout + r.stderr


def test_a_pool_member_is_the_canonical_id_and_kind_attacker_and_a_disagreement_fails_closed(tmp_path):
    """Either-or let a target row mislabeled attacker, or a pinned target whose kind drifted, ride the membership
    skip. Now BOTH the canonical id (`…/multiturn/mt-<view>-<cluster>/attackers/<positive int>`) and kind `attacker` are required, and an
    id/kind disagreement is refused by name wherever the placement readers meet it."""
    import pytest
    member = f"{KEY}/battery/multiturn/mt-battery-cluster_a/attackers/3"
    assert ST.is_pool_member(member) and ST.is_pool_member(member, {"kind": "attacker"}) and ST.is_pool_member(member, {"cluster": "cluster_a"})   # no kind: the id decides
    for bad in ("x", TARGET, f"{KEY}/battery/multiturn/mt-battery-cluster_a/attackers/0", f"{KEY}/battery/multiturn/mt-battery-cluster_a/attackers/x",
                f"{KEY}/battery/multiturn/mt-s2-cluster_a/attackers/1", f"{KEY}/battery/static/attackers/1"):
        assert not ST.is_pool_member(bad), bad
    with pytest.raises(ValueError, match="is not a pool member's .* but the row's kind is 'attacker' .*fail closed"):
        ST.is_pool_member("x", {"kind": "attacker"})
    with pytest.raises(ValueError, match="is a pool member's .* but the row's kind is 'target'"):
        ST.is_pool_member(member, {"kind": "target"})
    man = _four_member_manifest(); rows = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    # a NEW target row deliberately mislabeled attacker in a supplied --plan-json: refused by name, never skipped as a grown member
    rows_bad = rows + [dict(next(r for r in rows if r["logical_work_id"] == TARGET), logical_work_id=f"{KEY}/battery/multiturn/crescendo/g99", kind="attacker")]
    probs = ST.check_placement(man, AB, rows_bad)
    assert probs and "crescendo/g99" in probs[0] and "is not a pool member's" in probs[0] and "fail closed" in probs[0], probs
    # a PINNED target whose kind drifted to attacker while the regenerated plan dropped it: refused, not a lost member
    drifted = json.loads(json.dumps(man)); drifted["units"][TARGET]["kind"] = "attacker"
    rows_without = [r for r in rows if r["logical_work_id"] != TARGET]
    probs = ST.check_placement(drifted, AB, rows_without)
    assert probs and "crescendo/g34" in probs[0] and "but the row's kind is 'attacker'" in probs[0], probs
    root = _root(tmp_path); ledger = root / "attempts.jsonl"; _ev(root, _attempt(TARGET, "cluster_b", "80", "2026-01-04T01:00:00Z"))
    _, _, probs = ST.expand_placement(drifted, str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs and any("crescendo/g34" in p and "fail closed" in p for p in probs), probs
    # a member row that calls itself a target (kind drift the other way) is refused too
    rows_member_as_target = [dict(r, kind="target") if r["logical_work_id"].endswith("/attackers/1") else r for r in rows]
    probs = ST.check_placement(man, AB, rows_member_as_target)
    assert probs and any("attackers/1" in p and "but the row's kind is 'target'" in p for p in probs), probs
    # an attempt whose ledger kind disagrees with its id is a problem in the results walk as well
    root2 = _root(tmp_path / "r2"); ledger2 = root2 / "attempts.jsonl"
    _ev(root2, _attempt(f"{KEY}/battery/multiturn/mt-battery-cluster_a/attackers/9", "cluster_a", "900", "2026-01-05T00:00:00Z", group="mt-battery-cluster_a", kind="target"),
        _attempt(TARGET, "cluster_b", "80", "2026-01-04T01:00:00Z"))
    _, _, probs = ST.expand_placement(_four_member_manifest(), str(ledger2), KEY, "battery", AB, sha=SHA)
    assert probs and any("attackers/9" in p and "but the row's kind is 'target'" in p for p in probs), probs


def test_the_frozen_twenty_unit_manifest_replays_against_the_sixteen_unit_plan(tmp_path):
    """The standing regression: the manifest the u0 wave froze under FOUR members per cluster (20 multi-turn units: 12 targets +
    8 members, `source: ledger`) replayed against the plan commit 31 sizes at two members per cluster (16 units) — `ensure` keeps it, `check`
    passes, and only the four extra members differ between the two."""
    frozen = _four_member_manifest(); frozen["source"] = "ledger"; frozen["pinned_utc"] = "2026-09-30T21:06:50Z"
    assert len(frozen["units"]) == 20 and sum(1 for u in frozen["units"].values() if u["kind"] == "attacker") == 8
    plan = ST.placement_from_plan(KEY, "battery", AB, sha=SHA)
    assert len(plan["units"]) == 16 and sum(1 for u in plan["units"].values() if u["kind"] == "attacker") == 4
    extra = set(frozen["units"]) - set(plan["units"])
    assert extra == {f"{KEY}/battery/multiturn/mt-battery-{lab}/attackers/{i}" for lab in AB for i in (3, 4)} and all(frozen["units"][lw] == plan["units"][lw] for lw in plan["units"])
    rows = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    assert ST.check_placement(frozen, AB, rows) == []
    root = _root(tmp_path); ledger = root / "attempts.jsonl"
    _ev(root, *[_attempt(lw, u["cluster"], str(500 + i), f"2026-01-04T00:{i:02d}:00Z", group=u["group"], kind=u["kind"]) for i, (lw, u) in enumerate(sorted(frozen["units"].items()))])
    out, added, probs = ST.expand_placement(frozen, str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs == [] and added == [] and set(out["units"]) == set(frozen["units"]), probs
    man2, probs2 = ST.placement_from_ledger(str(ledger), KEY, "battery", AB, sha=SHA)
    assert probs2 == [] and man2["reconciled_attempts"] == 20
