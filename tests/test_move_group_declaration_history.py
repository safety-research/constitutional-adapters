"""move-knob-2 (four findings): each view has its OWN placement namespace (u0 keeps the historic file; an s2r declaration
never lands in or reads from the u0 manifest, and a manifest of another view refuses by name); a declared group is re-pointed only through
the explicit `--re-move` with a new reason, the earlier declarations kept in a history and the FINAL destination sticky for every later
plan; `placement-manifest moves` prints the persisted declarations as TSV, which 06 reads to derive mark-placed's --supersede + --reason on
every pass (flag-free completions included); and 06 ensures + checks the manifest for EVERY view's multi-turn call, not only inside the u0
inventory gate. Synthetic state root, the s2r wave's two-group shape."""
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
TWO_GROUPS = "0,1,2,3,4 5,6,7,8"


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("SISTEER_SHARD_GROUPS", TWO_GROUPS)
    monkeypatch.setenv("SISTEER_MT_ATTACKER_GPUS", "2")


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


def _targets(rows):
    return [r for r in rows if r["family"] == "multiturn" and r.get("kind", "target") == "target"]


def _members(rows):
    return [r for r in rows if r["family"] == "multiturn" and r.get("kind") == "attacker"]


def _gn(r):
    return ST._group_of_target(r["logical_work_id"])


def test_each_view_has_its_own_placement_namespace(tmp_path):
    ledger = str(tmp_path / "state" / "attempts.jsonl")
    assert ST.placement_path(ledger) == str(tmp_path / "state" / "gate" / "expected_units.placement.json")
    assert ST.placement_path(ledger, "u0") == ST.placement_path(ledger)                      # a declared u0 keeps the historic file (the phased recipe)
    assert ST.placement_path(ledger, "battery") == str(tmp_path / "state" / "gate" / "expected_units.placement.battery.json")   # the single-phase baseline wave: ITS per-view file (05 writes it, 06 reads it)
    assert ST.placement_path(ledger, "s2r") == str(tmp_path / "state" / "gate" / "expected_units.placement.s2r.json")
    assert ST.placement_file(None) == ST.placement_file("u0") == ST.PLACEMENT_FILE and ST.placement_file("d0p75_relu") == "expected_units.placement.d0p75_relu.json"
    with pytest.raises(ValueError, match="not a plain token"):
        ST.placement_file("s2r/../x")


def test_an_s2r_declaration_never_touches_the_u0_manifest_and_a_flag_free_s2r_replan_keeps_the_move(tmp_path, capsys):
    root = _root(tmp_path); ledger = str(root / "attempts.jsonl")
    u0_path, s2r_path = ST.placement_path(ledger, "battery"), ST.placement_path(ledger, "s2r")
    # the freeze pins the u0 manifest (its historic file) from the u0 plan
    ST.write_atomic(u0_path, ST.placement_from_plan(KEY, "battery", AB, sha=SHA))
    u0_bytes = Path(u0_path).read_bytes()
    s2r = ["--model", KEY, "--view", "s2r", "--clusters", ",".join(AB), "--ledger", ledger, "--sha", SHA]
    u0 = ["--model", KEY, "--view", "battery", "--clusters", ",".join(AB), "--ledger", ledger, "--sha", SHA]
    assert _cli("placement-manifest", "ensure", *s2r) == 0            # pins the s2r manifest in ITS file; the u0 ids are not "extra"
    assert Path(s2r_path).is_file() and Path(u0_path).read_bytes() == u0_bytes
    man_s2r = ST.read_placement(s2r_path)
    assert man_s2r["view"] == "s2r" and all(lw.startswith(f"{KEY}/s2r/multiturn/") for lw in man_s2r["units"])
    reason = "cluster_b cannot build the serving venv (CUDA 13 toolkit only)"
    assert _cli("placement-manifest", "ensure", *s2r, "--move-group", "g5678=cluster_a", "--move-reason", reason) == 0
    assert Path(u0_path).read_bytes() == u0_bytes and ST.read_placement(u0_path).get("declared_moves") is None     # the u0 manifest never learns of it
    d = ST.read_placement(s2r_path)["declared_moves"]["g5678"]
    assert d["to"] == "cluster_a" and d["from"] == "cluster_b" and d["history"] == [] and "re_move_of" not in d
    # a flag-free s2r replan keeps the move (sticky) and sizes the destination's pool from the moved count; u0's plan is untouched
    capsys.readouterr()          # drain the ensure/plan prints above: the next read must be this call's stdout alone
    assert _cli("plan", *s2r, "--families", "multiturn", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert all(r["cluster"] == "cluster_a" for r in _targets(rows)) and [m["cluster"] for m in _members(rows)] == ["cluster_a", "cluster_a"]
    assert _cli("plan", *u0, "--families", "multiturn", "--json") == 0
    rows_s1 = json.loads(capsys.readouterr().out)
    assert {r["cluster"] for r in _targets(rows_s1) if _gn(r) == "g5678"} == {"cluster_b"} and [m["cluster"] for m in _members(rows_s1)] == ["cluster_a", "cluster_b"]
    plan_json = tmp_path / "plan.json"; plan_json.write_text(json.dumps(rows))
    assert _cli("placement-manifest", "check", *s2r, "--plan-json", str(plan_json)) == 0
    assert _cli("placement-manifest", "ensure", *u0) == 0 and Path(u0_path).read_bytes() == u0_bytes
    # a manifest of ANOTHER view read through this view's call refuses by name
    assert _cli("placement-manifest", "ensure", *s2r, "--manifest", u0_path) == 2
    assert "is view 'battery', not 's2r'" in capsys.readouterr().err
    # the moves TSV is per view
    capsys.readouterr()          # drain the ensure/plan prints above: the next read must be this call's stdout alone
    assert _cli("placement-manifest", "moves", *s2r) == 0
    tsv = capsys.readouterr().out.strip().split("\t")
    assert tsv[:4] == ["g5678", "cluster_a", "cluster_b", reason] and tsv[5] == "0"
    assert _cli("placement-manifest", "moves", *u0) == 0 and capsys.readouterr().out == ""


def test_a_declared_group_is_re_pointed_only_through_re_move_and_keeps_its_history(tmp_path, capsys):
    root = _root(tmp_path); ledger = str(root / "attempts.jsonl")
    common = ["--model", KEY, "--view", "s2r", "--clusters", ",".join(AB), "--ledger", ledger, "--sha", SHA]
    man_path = ST.placement_path(ledger, "s2r")
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", "first: b cannot build the venv") == 0
    lw5678 = sorted(lw for lw in ST.read_placement(man_path)["units"] if ST._group_of_target(lw) == "g5678")
    lw = lw5678[0]
    # the move happens: the dead job on b is replaced on a under a receipt; the pin advances
    _ev(root,
        {"event": "attempt", "logical_work_id": lw, "attempt_id": f"{lw}@cluster_b@80", "cluster": "cluster_b", "job_id": "80", "kind": "target", "sha": SHA,
         "spec": {"group": "mt-s2r-cluster_b", "kind": "target", "sha": SHA}, "utc": "2026-01-04T00:00:00Z"},
        {"event": "registered", "logical_work_id": lw, "attempt_id": f"{lw}@cluster_b@80", "cluster": "cluster_b", "job_id": "80", "utc": "2026-01-04T00:06:00Z"},
        {"event": "attempt", "logical_work_id": lw, "attempt_id": f"{lw}@cluster_a@90", "cluster": "cluster_a", "job_id": "90", "kind": "target", "sha": SHA,
         "spec": {"group": "mt-s2r-cluster_a", "kind": "target", "sha": SHA}, "utc": "2026-01-06T00:00:00Z"},
        {"event": "registered", "logical_work_id": lw, "attempt_id": f"{lw}@cluster_a@90", "cluster": "cluster_a", "job_id": "90", "utc": "2026-01-06T00:06:00Z"},
        {"event": "placement_receipt", "logical_work_id": lw, "from_cluster": "cluster_b", "to_cluster": "cluster_a", "to_group": "mt-s2r-cluster_a",
         "replaced_job_id": "80", "replaced_state": "FAILED", "attempt_id": f"{lw}@cluster_a@90", "job_id": "90", "utc": "2026-01-06T00:00:00Z",
         "reason": "fenced dead-job replacement on another cluster (submit_twin --resubmit)"})
    assert _cli("placement-manifest", "ensure", *common) == 0
    assert ST.read_placement(man_path)["units"][lw]["cluster"] == "cluster_a" and ST.read_placement(man_path)["units"][lw].get("receipt")
    # re-pointing without --re-move refuses, naming the flag; --re-move on an undeclared group refuses too
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_b", "--move-reason", "second: a lost its venv too") == 2
    err = capsys.readouterr().err
    assert "already declares g5678" in err and "--re-move" in err
    assert _cli("placement-manifest", "ensure", *common, "--re-move", "--move-group", "g01234=cluster_b", "--move-reason", "x") == 2
    assert "no declaration to re-point" in capsys.readouterr().err
    assert ST.read_placement(man_path)["declared_moves"]["g5678"]["to"] == "cluster_a"
    # the explicit re-move: a new reason, the earlier declaration kept in the history, `from` = the previous destination
    assert _cli("placement-manifest", "ensure", *common, "--re-move", "--move-group", "g5678=cluster_b", "--move-reason", "second: a lost its venv too") == 0
    d = ST.read_placement(man_path)["declared_moves"]["g5678"]
    assert d["to"] == "cluster_b" and d["from"] == "cluster_a" and d["re_move_of"] == "cluster_a" and d["reason"] == "second: a lost its venv too"
    assert len(d["history"]) == 1 and d["history"][0]["to"] == "cluster_a" and d["history"][0]["from"] == "cluster_b" and d["history"][0]["reason"].startswith("first:")
    assert "history" not in d["history"][0]
    # the FINAL destination is what every later plan follows; the moves TSV says so, with the history depth
    capsys.readouterr()          # drain the ensure/plan prints above: the next read must be this call's stdout alone
    assert _cli("plan", *common, "--families", "multiturn", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["cluster"] for r in _targets(rows) if _gn(r) == "g5678"} == {"cluster_b"} and {r["cluster"] for r in _targets(rows) if _gn(r) == "g01234"} == {"cluster_a"}
    assert sorted(m["cluster"] for m in _members(rows)) == ["cluster_a", "cluster_b"]
    plan_json = tmp_path / "plan.json"; plan_json.write_text(json.dumps(rows))
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(plan_json)) == 0
    capsys.readouterr()          # drain the ensure/plan prints above: the next read must be this call's stdout alone
    assert _cli("placement-manifest", "moves", *common) == 0
    tsv = capsys.readouterr().out.strip().split("\t")
    assert tsv[0:3] == ["g5678", "cluster_b", "cluster_a"] and tsv[3].startswith("second:") and tsv[5] == "1"
    # repeating the current declaration is idempotent; a third destination needs --re-move again
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_b", "--move-reason", "same") == 0
    assert len(ST.read_placement(man_path)["declared_moves"]["g5678"]["history"]) == 1


def test_06_keeps_the_manifest_for_every_view_and_derives_the_supersede_from_the_persisted_declaration():
    path = REPO / "reproduce" / "06_eval_waves.sh"; w = path.read_text()
    assert subprocess.run(["bash", "-n", str(path)]).returncode == 0
    # the baseline wave's inventory gate (u0, or the single-phase recipe's one wave); `[ "$PASS" = first ]` since petri-completion: the Petri
    # overflow pass is a follow-up unit the freeze never inventories, so the gate is the first pass's alone
    gate_start = w.index('if [ "$VIEW" = "$BASE_VIEW" ] && [ "$PASS" = first ]; then   # the baseline wave')
    gate_end = w.index("\nfi\n", gate_start)
    assert "placement-manifest" not in w[gate_start:gate_end], "the placement manifest must not be ensured/checked inside the baseline inventory gate alone"
    ensure_at = w.index("placement-manifest ensure")
    assert ensure_at > gate_end
    assert 'case ",$FAMILIES," in *,multiturn,*)\n  [ -n "${PLAN_JSON:-}" ] || PLAN_JSON=' in w
    assert 'DECLARED_MOVES_TSV="$("${STAGE_PY[@]}" placement-manifest moves --model "$K" --view "$VIEW"' in w
    assert 'MARK_EXTRA=(--supersede --reason "declared move-group $mv_g=$l (from $mv_from): $mv_reason")' in w
    assert '--re-move) MOVE_REMOVE=1; shift;;' in w and '${MOVE_REMOVE:+--re-move}' in w
    assert 'MARK_EXTRA=(--supersede --reason "declared move-group ${MOVE[*]}' not in w
