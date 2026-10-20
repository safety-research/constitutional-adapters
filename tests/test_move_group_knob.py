"""`--move-group <shard group>=<label>`: a multi-turn shard group is placed on another cluster by DECLARATION, with provenance: cluster B
cannot stand the gdn serving venv up (CUDA 13 only), so the s2r wave's g5678 must run on cluster A — and the
funnel had no path that moves a pinned group: the 24d cross-cluster replacement lands on the PLAN ROW's cluster (round-robin over the pinned
universe) and `placement-manifest check` refuses a planned cluster that differs from the pin without the receipt only that replacement writes.

Pinned here: the plan places a moved group at the label and sizes THAT label's pool from the moved count (a label every group left gets no
pool); the move is refused by name outside the universe or for an unknown group; `placement-manifest ensure --move-group … --move-reason`
records the declaration (to, from, reason, utc, units) and leaves the pins at the origin; `check` accepts a plan at the declared destination
and still refuses an UNDECLARED move; `plan --ledger` applies the manifest's declared moves (sticky); a receipt then advances the pin; a
re-pointed declaration refuses; 06 forwards the flags and supersedes the grown group's PLACED record with the reason. Synthetic state root."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"
SHA = "a" * 40
AB = ["cluster_a", "cluster_b"]
TWO_GROUPS = "0,1,2,3,4 5,6,7,8"      # a two-group shard split: g01234 on the first label, g5678 on the second


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("SISTEER_SHARD_GROUPS", TWO_GROUPS)
    monkeypatch.setenv("SISTEER_MT_ATTACKER_GPUS", "2")


def _mt(rows):
    return [r for r in rows if r["family"] == "multiturn"]


def _targets(rows):
    return [r for r in _mt(rows) if r.get("kind", "target") == "target"]


def _members(rows):
    return [r for r in _mt(rows) if r.get("kind") == "attacker"]


def _gn(r):
    return ST._group_of_target(r["logical_work_id"])


def test_a_declared_move_places_the_group_at_the_label_and_sizes_that_pool_from_the_moved_count():
    base = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB)
    assert {r["cluster"] for r in _targets(base) if _gn(r) == "g5678"} == {"cluster_b"}
    assert [m["cluster"] for m in _members(base)] == ["cluster_a", "cluster_b"]            # one member per label (3 targets each)
    moved = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB, move_groups={"g5678": "cluster_a"})
    tg = _targets(moved)
    assert len(tg) == 6 and all(r["cluster"] == "cluster_a" and r["group"] == "mt-battery-cluster_a" for r in tg)
    assert {r.get("moved_from") for r in tg if _gn(r) == "g5678"} == {"cluster_b"} and all("moved_from" not in r for r in tg if _gn(r) == "g01234")
    mem = _members(moved)
    assert [m["cluster"] for m in mem] == ["cluster_a", "cluster_a"], mem                    # 6 targets -> 2 members (cap 2), NONE on the emptied label
    assert sorted(m["unit"] for m in mem) == ["attacker1", "attacker2"]
    assert {r["logical_work_id"] for r in tg} == {r["logical_work_id"] for r in _targets(base)}    # the same units, re-placed, never re-minted


def test_a_move_is_refused_by_name_outside_the_universe_or_for_an_unknown_group():
    with pytest.raises(ValueError, match="not a shard group of this plan"):
        ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB, move_groups={"g99": "cluster_a"})
    with pytest.raises(ValueError, match="not a label of the placement universe"):
        ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB, move_groups={"g5678": "cluster_c"})
    with pytest.raises(ValueError, match="placement-free plan has no destination"):
        ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=None, move_groups={"g5678": "cluster_a"})
    assert ST.parse_move_groups(["g5678=cluster_a", " g01234 = cluster_b "]) == {"g5678": "cluster_a", "g01234": "cluster_b"}
    for bad in (["g5678"], ["=cluster_a"], ["x5678=cluster_a"]):
        with pytest.raises(ValueError, match="expected <shard group>=<label>"):
            ST.parse_move_groups(bad)
    with pytest.raises(ValueError, match="named twice with different labels"):
        ST.parse_move_groups(["g5678=cluster_a", "g5678=cluster_b"])
    with pytest.raises(ValueError, match="already declares g5678"):
        ST.merge_move_groups({"g5678": "cluster_a"}, {"g5678": "cluster_b"})
    assert ST.merge_move_groups({"g5678": "cluster_a"}, {"g5678": "cluster_a", "g01234": "cluster_b"}) == {"g5678": "cluster_a", "g01234": "cluster_b"}


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "state"; (root / "gate").mkdir(parents=True)
    ST.write_atomic(str(root / "gate" / "expected_units.json"), {"schema": 2, "model": KEY, "view": "battery", "code_sha": SHA, "frozen_utc": "2026-01-02T00:00:00Z", "units": []})
    (root / "attempts.jsonl").write_text("")
    return root


def _ev(root: Path, *events):
    with open(root / "attempts.jsonl", "a") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def _cli(*argv):
    return ST.main([str(a) for a in argv])


def test_the_manifest_declares_a_move_with_provenance_and_the_plan_and_check_honour_it_until_the_receipt(tmp_path, capsys):
    root = _root(tmp_path); ledger = str(root / "attempts.jsonl"); man_path = ST.placement_path(ledger, "battery")   # the single-phase baseline wave: its per-view manifest file
    common = ["--model", KEY, "--view", "battery", "--clusters", ",".join(AB), "--ledger", ledger, "--sha", SHA]
    assert _cli("placement-manifest", "ensure", *common) == 0
    man = ST.read_placement(man_path)
    lw5678 = sorted(lw for lw in man["units"] if ST._group_of_target(lw) == "g5678")
    assert len(lw5678) == 3 and all(man["units"][lw]["cluster"] == "cluster_b" for lw in lw5678)
    # a declaration needs its reason
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a") == 2
    assert "needs --move-reason" in capsys.readouterr().err
    assert ST.read_placement(man_path).get("declared_moves") is None
    # check cannot declare
    assert _cli("placement-manifest", "check", *common, "--move-group", "g5678=cluster_a", "--move-reason", "x") == 2
    assert "check only reads it" in capsys.readouterr().err
    reason = "cluster_b cannot build the serving venv (CUDA 13 toolkit only); the group runs on cluster_a"
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", reason) == 0
    out = capsys.readouterr().out
    # the declaration is persisted — and the moved count sizes a SECOND member on the destination, which the manifest records as an added unit
    assert ("MOVED" in out or "EXPANDED" in out) and "group move(s) DECLARED (g5678 -> cluster_a" in out, out
    man2 = ST.read_placement(man_path)
    d = man2["declared_moves"]["g5678"]
    assert d["to"] == "cluster_a" and d["from"] == "cluster_b" and d["reason"] == reason and d["units"] == lw5678 and d["declared_utc"]
    assert all(man2["units"][lw]["cluster"] == "cluster_b" and not man2["units"][lw].get("receipt") for lw in lw5678)   # the pins wait for the receipt
    assert man2.get("source", "").endswith("+moved") and man2.get("moves_declared_utc")
    # the plan applies the declared move from the ledger's manifest (sticky), sizing the destination's pool from the moved count
    assert _cli("plan", *common, "--families", "multiturn", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert all(r["cluster"] == "cluster_a" for r in _targets(rows)) and [m["cluster"] for m in _members(rows)] == ["cluster_a", "cluster_a"]
    plan_json = tmp_path / "plan.json"; plan_json.write_text(json.dumps(rows))
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(plan_json)) == 0
    # an UNDECLARED move still refuses, naming the declaration
    rows_bad = ST.unit_plan(KEY, "battery", families=["multiturn"], sha=SHA, clusters=AB, move_groups={"g01234": "cluster_b", "g5678": "cluster_a"})
    bad_json = tmp_path / "bad.json"; bad_json.write_text(json.dumps(rows_bad))
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(bad_json)) == 2
    err = capsys.readouterr().err
    assert "g01234" in err and "the placement moved" in err and "--move-group g01234=<label>" in err
    # re-pointing a declared move is a refusal, not a flag
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_b", "--move-reason", "oops") == 2
    assert "already declares g5678" in capsys.readouterr().err
    # a repeat declaration of the same move is idempotent (nothing new declared)
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", reason) == 0
    assert ST.read_placement(man_path)["declared_moves"]["g5678"]["declared_utc"] == d["declared_utc"]
    # the resubmit's RECEIPT (the dead job on cluster_b replaced on cluster_a) advances the pin; the plan with the move checks clean
    lw = lw5678[0]
    _ev(root,
        {"event": "attempt", "logical_work_id": lw, "attempt_id": f"{lw}@cluster_b@80", "cluster": "cluster_b", "job_id": "80", "kind": "target", "sha": SHA,
         "spec": {"group": "mt-battery-cluster_b", "kind": "target", "sha": SHA}, "utc": "2026-01-04T00:00:00Z"},
        {"event": "registered", "logical_work_id": lw, "attempt_id": f"{lw}@cluster_b@80", "cluster": "cluster_b", "job_id": "80", "utc": "2026-01-04T00:06:00Z"},
        {"event": "attempt", "logical_work_id": lw, "attempt_id": f"{lw}@cluster_a@90", "cluster": "cluster_a", "job_id": "90", "kind": "target", "sha": SHA,
         "spec": {"group": "mt-battery-cluster_a", "kind": "target", "sha": SHA}, "utc": "2026-01-06T00:00:00Z"},
        {"event": "registered", "logical_work_id": lw, "attempt_id": f"{lw}@cluster_a@90", "cluster": "cluster_a", "job_id": "90", "utc": "2026-01-06T00:06:00Z"},
        {"event": "placement_receipt", "logical_work_id": lw, "from_cluster": "cluster_b", "to_cluster": "cluster_a", "to_group": "mt-battery-cluster_a",
         "replaced_job_id": "80", "replaced_state": "FAILED", "attempt_id": f"{lw}@cluster_a@90", "job_id": "90", "utc": "2026-01-06T00:00:00Z",
         "reason": "fenced dead-job replacement on another cluster (submit_twin --resubmit)"})
    assert _cli("placement-manifest", "ensure", *common) == 0
    man3 = ST.read_placement(man_path)
    assert man3["units"][lw]["cluster"] == "cluster_a" and man3["units"][lw].get("receipt") == f"{lw}@cluster_a@90" and man3["declared_moves"]["g5678"]["to"] == "cluster_a"
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(plan_json)) == 0


def test_a_fresh_pin_with_a_declared_move_places_the_group_at_the_destination_and_records_the_declaration(tmp_path, capsys):
    root = _root(tmp_path); ledger = str(root / "attempts.jsonl")
    common = ["--model", KEY, "--view", "battery", "--clusters", ",".join(AB), "--ledger", ledger, "--sha", SHA]
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", "declared before the first job") == 0
    man = ST.read_placement(ST.placement_path(ledger, "battery"))
    assert all(u["cluster"] == "cluster_a" for lw, u in man["units"].items() if ST._group_of_target(lw) == "g5678")
    assert man["declared_moves"]["g5678"]["from"] == "cluster_b" and man["declared_moves"]["g5678"]["to"] == "cluster_a"


def test_06_forwards_the_move_flags_to_the_plan_the_manifest_and_the_grown_groups_placement():
    w = (REPO / "reproduce" / "06_eval_waves.sh").read_text()
    assert subprocess.run(["bash", "-n", str(REPO / "reproduce" / "06_eval_waves.sh")]).returncode == 0
    assert '--move-group) MOVE+=("$2"); shift 2;; --move-reason) MOVE_REASON="$2"; shift 2;;' in w
    assert 'repro_fatal "--move-group needs --move-reason' in w
    assert 'PLAN_ARGS+=(--ledger "$STATE/attempts.jsonl" ${MOVE_ARGS[@]+"${MOVE_ARGS[@]}"} ${MOVE_REMOVE:+--re-move} ${CELLS_SEL:+--cells "$CELLS_SEL"})' in w      # move-knob-3 + cells-fix: the plan carries --re-move and --cells too
    ensure = w[w.index("placement-manifest ensure"):w.index("placement-manifest check")]
    assert '${MOVE_ARGS[@]+"${MOVE_ARGS[@]}"} ${MOVE_REASON:+--move-reason "$MOVE_REASON"}' in ensure
    assert 'MARK_EXTRA=(--supersede --reason "declared move-group $mv_g=$l (from $mv_from): $mv_reason")' in w     # derived from the persisted declaration
    assert 'placement-manifest moves --model "$K" --view "$VIEW"' in w
    assert 'mark-placed --group "$g" --targets "${GROUP_TARGETS[$g]}" ${MARK_EXTRA[@]+"${MARK_EXTRA[@]}"}' in w
    assert "--move-group <shard group>=<label>" in w[:6000]      # the usage header documents it
