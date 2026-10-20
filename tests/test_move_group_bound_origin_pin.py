"""move-knob-3 (the g5678 go on 897106ffe, 2026-10-02 06:01Z): a DECLARED move's targets that were ever BOUND on the origin keep the ORIGIN
pin until the fenced resubmit's placement receipt moves them. The s2r view's first manifest was pinned from the plan at the move's DESTINATION
(`placement_from_ledger` is None for a view whose attempts sit at no frozen tree, so the fresh-pin branch read "nothing ran yet"), while
`check_placement`'s latest-binding rule found the dead origin `submitted` bindings -> `pinned on cluster_a but its latest binding … is bound
on cluster_b`, and the wave could not move. Pinned here, in the s2r two-group shape: a fresh pin with a declared move holds the
origin for a group whose units carry a binding and takes the destination for a group that was never bound; an existing manifest with the
destination pin (the refused call's artifact) RETREATS to the origin on the next flag-free `ensure`, persisted and named; `check` passes
against the sticky plan either way (the declared-destination clause); a torn binding is a refusal by name, never "no binding"."""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO, VENV_PY, run_stage  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"
SHA = "a" * 40
AB = ["cluster_a", "cluster_b"]
TWO_GROUPS = "0,1,2,3,4 5,6,7,8"
REASON = "cluster_b cannot build the serving venv (CUDA 13 toolkit only)"


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("SISTEER_SHARD_GROUPS", TWO_GROUPS)
    monkeypatch.setenv("SISTEER_MT_ATTACKER_GPUS", "2")


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "state"; (root / "gate").mkdir(parents=True)
    ST.write_atomic(str(root / "gate" / "expected_units.json"), {"schema": 2, "model": KEY, "view": "battery", "code_sha": SHA, "frozen_utc": "2026-01-02T00:00:00Z", "units": []})
    (root / "attempts.jsonl").write_text("")
    return root


def _cli(*argv):
    return ST.main([str(a) for a in argv])


def _g5678_targets(view="s2r"):
    rows = ST.unit_plan(KEY, view, families=["multiturn"], sha=SHA, clusters=AB)
    return sorted(r["logical_work_id"] for r in rows if r.get("kind", "target") == "target" and ST._group_of_target(r["logical_work_id"]) == "g5678")


def _bind(root: Path, lw: str, cluster: str, job: str):
    bdir = root / "bindings"; bdir.mkdir(exist_ok=True)
    (bdir / f"{ST.safe_id(lw)}.submitted.json").write_text(json.dumps(ST.submitted_record(lw, job, cluster, f"{lw}@{cluster}@{job}")))


def _ev(root: Path, *events):
    with open(root / "attempts.jsonl", "a") as fh:
        for e in events:
            fh.write(json.dumps(e) + "\n")


def _attempt(lw: str, cluster: str, job: str, utc: str):
    return {"event": "attempt", "logical_work_id": lw, "attempt_id": f"{lw}@{cluster}@{job}", "cluster": cluster, "job_id": job, "kind": "target", "sha": SHA,
            "spec": {"group": f"mt-s2r-{cluster}", "kind": "target", "sha": SHA}, "utc": utc}


def _registered(lw: str, cluster: str, job: str, utc: str):
    return {"event": "registered", "logical_work_id": lw, "attempt_id": f"{lw}@{cluster}@{job}", "cluster": cluster, "job_id": job, "utc": utc}


def _sticky_plan_json(tmp_path: Path, common, capsys) -> Path:
    capsys.readouterr()
    assert _cli("plan", *common, "--families", "multiturn", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    plan_json = tmp_path / "plan.json"; plan_json.write_text(json.dumps(rows))
    return plan_json


def _common(root: Path):
    return ["--model", KEY, "--view", "s2r", "--clusters", ",".join(AB), "--ledger", str(root / "attempts.jsonl"), "--sha", SHA]


def test_the_plain_plan_places_g5678_on_cluster_b(tmp_path):
    rows = ST.unit_plan(KEY, "s2r", families=["multiturn"], sha=SHA, clusters=AB)
    assert {r["cluster"] for r in rows if ST._group_of_target(r["logical_work_id"]) == "g5678"} == {"cluster_b"}      # the origin the move leaves


def test_a_fresh_pin_with_a_declared_move_holds_the_origin_for_a_bound_group(tmp_path, capsys):
    root = _root(tmp_path); common = _common(root); targets = _g5678_targets()
    assert len(targets) == 3
    for i, lw in enumerate(targets):
        _bind(root, lw, "cluster_b", str(100 + i))                      # the dead origin attempts: submitted, never registered, never folded
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", REASON) == 0
    out = capsys.readouterr().out
    assert "3 moved unit(s) hold their ORIGIN pin" in out
    man = ST.read_placement(ST.placement_path(str(root / "attempts.jsonl"), "s2r"))
    assert man["declared_moves"]["g5678"]["to"] == "cluster_a" and man["declared_moves"]["g5678"]["from"] == "cluster_b"
    assert all(man["units"][lw]["cluster"] == "cluster_b" and man["units"][lw]["group"] == "mt-s2r-cluster_b" for lw in targets)
    assert sorted(man["held_origin_units"]) == targets
    # the sticky plan places the group at the destination and the check admits it against the origin pin (declared) with the binding where the pin says
    capsys.readouterr()
    assert _cli("plan", *common, "--families", "multiturn", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["cluster"] for r in rows if ST._group_of_target(r["logical_work_id"]) == "g5678"} == {"cluster_a"}
    plan_json = tmp_path / "plan.json"; plan_json.write_text(json.dumps(rows))
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(plan_json)) == 0, capsys.readouterr().err
    # a flag-free ensure keeps the origin pins (no receipt yet) and rewrites nothing
    before = Path(ST.placement_path(str(root / "attempts.jsonl"), "s2r")).read_bytes()
    assert _cli("placement-manifest", "ensure", *common) == 0
    assert Path(ST.placement_path(str(root / "attempts.jsonl"), "s2r")).read_bytes() == before
    # the move happens for ONE unit the designed way: the --resubmit's fenced replacement on the destination (a new binding there) under a
    # placement receipt; the next ensure advances that pin and leaves the two still-bound-on-origin pins where their results are
    lw = targets[0]
    _ev(root, _attempt(lw, "cluster_b", "100", "2026-01-04T00:00:00Z"), _registered(lw, "cluster_b", "100", "2026-01-04T00:06:00Z"),
        _attempt(lw, "cluster_a", "190", "2026-01-06T00:00:00Z"), _registered(lw, "cluster_a", "190", "2026-01-06T00:06:00Z"),
        {"event": "placement_receipt", "logical_work_id": lw, "from_cluster": "cluster_b", "to_cluster": "cluster_a", "to_group": "mt-s2r-cluster_a",
         "replaced_job_id": "100", "replaced_state": "FAILED", "attempt_id": f"{lw}@cluster_a@190", "job_id": "190", "utc": "2026-01-06T00:00:00Z",
         "reason": "fenced dead-job replacement on another cluster (submit_twin --resubmit)"})
    _bind(root, lw, "cluster_a", "190")
    capsys.readouterr()
    assert _cli("placement-manifest", "ensure", *common) == 0
    assert "advanced under a placement receipt" in capsys.readouterr().out
    man = ST.read_placement(ST.placement_path(str(root / "attempts.jsonl"), "s2r"))
    assert man["units"][lw]["cluster"] == "cluster_a" and man["units"][lw].get("receipt")
    assert all(man["units"][o]["cluster"] == "cluster_b" for o in targets[1:])
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(_sticky_plan_json(tmp_path, common, capsys))) == 0, capsys.readouterr().err


def test_a_fresh_pin_with_a_declared_move_takes_the_destination_for_a_never_bound_group(tmp_path, capsys):
    root = _root(tmp_path); common = _common(root); targets = _g5678_targets()
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", REASON) == 0
    assert "hold their ORIGIN pin" not in capsys.readouterr().out
    man = ST.read_placement(ST.placement_path(str(root / "attempts.jsonl"), "s2r"))
    assert all(man["units"][lw]["cluster"] == "cluster_a" for lw in targets) and "held_origin_units" not in man


def test_an_existing_destination_pin_retreats_to_the_bound_origin_on_the_next_ensure_and_check_passes(tmp_path, capsys):
    root = _root(tmp_path); common = _common(root); targets = _g5678_targets(); path = ST.placement_path(str(root / "attempts.jsonl"), "s2r")
    # the artifact: pinned at the destination while nothing was bound yet (the refused 06:01Z call's manifest, in miniature)
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", REASON) == 0
    assert all(ST.read_placement(path)["units"][lw]["cluster"] == "cluster_a" for lw in targets)
    for i, lw in enumerate(targets):
        _bind(root, lw, "cluster_b", str(200 + i))
    capsys.readouterr()
    # before the fix: `check` refused "pinned on cluster_a but its latest binding … is bound on cluster_b"; now the flag-free ensure retreats
    assert _cli("placement-manifest", "ensure", *common) == 0
    out = capsys.readouterr().out
    assert "3 moved unit(s) RETREATED to their origin pin" in out
    man = ST.read_placement(path)
    assert all(man["units"][lw]["cluster"] == "cluster_b" and man["units"][lw]["group"] == "mt-s2r-cluster_b" for lw in targets)
    assert sorted(man["retreated_units"]) == targets and man["declared_moves"]["g5678"]["to"] == "cluster_a"      # the move stands; the pin says where the results are
    assert _cli("plan", *common, "--families", "multiturn", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    plan_json = tmp_path / "plan.json"; plan_json.write_text(json.dumps(rows))
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(plan_json)) == 0, capsys.readouterr().err
    # the g01234 group (not moved, bound nowhere) is untouched
    assert all(u["cluster"] == "cluster_a" for lw, u in man["units"].items() if ST._group_of_target(lw) == "g01234")


def test_a_torn_binding_under_a_declared_move_is_a_refusal_by_name(tmp_path, capsys):
    root = _root(tmp_path); common = _common(root); targets = _g5678_targets()
    bdir = root / "bindings"; bdir.mkdir(exist_ok=True)
    (bdir / f"{ST.safe_id(targets[0])}.submitted.json").write_text("{not json")
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", REASON) == 2
    err = capsys.readouterr().err
    assert "binding" in err and "unreadable" in err.lower()


def test_check_placement_tolerates_only_the_declared_origin_before_the_receipt(tmp_path):
    root = _root(tmp_path); ledger = str(root / "attempts.jsonl"); targets = _g5678_targets()
    man = ST.placement_from_plan(KEY, "s2r", AB, sha=SHA, move_groups={"g5678": "cluster_a"})          # the destination pins, no receipt
    man["declared_moves"] = ST.declare_moves(KEY, "s2r", AB, {"g5678": "cluster_a"}, REASON, sha=SHA)
    assert man["declared_moves"]["g5678"]["from"] == "cluster_b"
    rows = ST.unit_plan(KEY, "s2r", families=["multiturn"], sha=SHA, clusters=AB, move_groups={"g5678": "cluster_a"})
    for i, lw in enumerate(targets):
        _bind(root, lw, "cluster_b", str(300 + i))
    assert ST.check_placement(man, AB, rows, ledger=ledger) == []                                   # bound on the declared origin: the pre-receipt state
    _bind(root, targets[0], "cluster_c", "999")                                                      # bound somewhere the declaration never named: refused
    probs = ST.check_placement(man, AB, rows, ledger=ledger)
    assert len(probs) == 1 and "cluster_c" in probs[0] and "not where the pin says" in probs[0]
    # a unit the plan does not move keeps today's rule: any other cluster's binding refuses
    g01234 = sorted(lw for lw in man["units"] if ST._group_of_target(lw) == "g01234")
    _bind(root, targets[0], "cluster_b", "300")
    _bind(root, g01234[0], "cluster_b", "777")
    probs = ST.check_placement(man, AB, rows, ledger=ledger)
    assert len(probs) == 1 and g01234[0] in probs[0]


def test_a_first_manifest_seeded_from_effective_attempts_keeps_the_declaration(tmp_path, capsys):
    root = _root(tmp_path); common = _common(root); targets = _g5678_targets(); path = ST.placement_path(str(root / "attempts.jsonl"), "s2r")
    # old attempts that DO qualify as effective (at the frozen tree) on the origin: the manifest is seeded from the ledger …
    for i, lw in enumerate(targets):
        _ev(root, _attempt(lw, "cluster_b", str(400 + i), "2026-01-04T00:00:00Z"))
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", REASON) == 0
    out = capsys.readouterr().out
    assert "pinned from the ledger" in out
    man = ST.read_placement(path)
    # … and the declaration is recorded (before: dropped on this branch), the pins are the attempts' (the origin), the sticky plan moves the group
    assert man["declared_moves"]["g5678"]["to"] == "cluster_a" and all(man["units"][lw]["cluster"] == "cluster_b" for lw in targets)
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(_sticky_plan_json(tmp_path, common, capsys))) == 0, capsys.readouterr().err
    capsys.readouterr()
    assert _cli("plan", *common, "--families", "multiturn", "--json") == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["cluster"] for r in rows if ST._group_of_target(r["logical_work_id"]) == "g5678"} == {"cluster_a"}


def test_plan_re_move_merges_like_ensure_and_validates(tmp_path, capsys):
    root = _root(tmp_path); common = _common(root)
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", REASON) == 0
    capsys.readouterr()
    # the plan is what 06 calls FIRST: without --re-move a second destination refuses (as before); with it the plan follows the new destination
    assert _cli("plan", *common, "--families", "multiturn", "--json", "--move-group", "g5678=cluster_b") == 2
    assert "already declares g5678" in capsys.readouterr().err
    assert _cli("plan", *common, "--families", "multiturn", "--json", "--re-move", "--move-group", "g5678=cluster_b") == 0
    rows = json.loads(capsys.readouterr().out)
    assert {r["cluster"] for r in rows if ST._group_of_target(r["logical_work_id"]) == "g5678"} == {"cluster_b"}
    assert _cli("plan", *common, "--families", "multiturn", "--json", "--re-move", "--move-group", "g01234=cluster_b") == 2
    assert "no declaration to re-point" in capsys.readouterr().err


def test_06_runs_the_plan_ensure_check_sequence_with_re_move(tmp_path):
    """The ACTUAL 06 preflight: 06 calls `stage.py plan` FIRST with --move-group and --ledger, then `ensure --re-move`,
    then `check`; before, the plan merged the old and new destination without --re-move and refused the re-point before ensure could record it.
    06 plans with the knobs the site / frozen inventory pin (the DEFAULT shard groups here), so this test speaks that grouping, not the
    two-group fixture's; the view is qwen35_4b_demo's d0p75 (a dose rung, the one view beside the baseline wave it drives through the real 06)."""
    w = (REPO / "reproduce" / "06_eval_waves.sh").read_text()
    assert 'PLAN_ARGS+=(--ledger "$STATE/attempts.jsonl" ${MOVE_ARGS[@]+"${MOVE_ARGS[@]}"} ${MOVE_REMOVE:+--re-move} ${CELLS_SEL:+--cells "$CELLS_SEL"})' in w
    saved = {k: os.environ.pop(k, None) for k in ("SISTEER_SHARD_GROUPS", "SISTEER_MT_ATTACKER_GPUS")}
    try:
        site_root = tmp_path / "fd"; site = site_root / "site"
        shutil.copytree(REPO / "reproduce" / "site.example", site)
        state_dir = site_root / "state"; state = state_dir / KEY; (state / "gate").mkdir(parents=True)      # 06's state root is $SISTEER_STATE_DIR/<model>
        with open(site / "config.env", "a") as fh:
            fh.write(f"\nSISTEER_STATE_DIR={state_dir}\nSISTEER_PYTHON={VENV_PY}\nSISTEER_REGISTRY_ROOT={site_root / 'registry'}\n")
        (state / "attempts.jsonl").write_text("")
        ST.write_atomic(str(state / "gate" / "expected_units.json"), {"schema": 2, "model": KEY, "view": "battery", "code_sha": SHA, "frozen_utc": "2026-01-02T00:00:00Z", "units": []})
        ledger = str(state / "attempts.jsonl")
        rows = ST.unit_plan(KEY, "d0p75", families=["multiturn"], sha=SHA, clusters=AB)
        gn = sorted({ST._group_of_target(r["logical_work_id"]) for r in rows if r.get("kind", "target") == "target" and r["cluster"] == "cluster_b"})[0]
        common = ["--model", KEY, "--view", "d0p75", "--clusters", ",".join(AB), "--ledger", ledger, "--sha", SHA]
        assert _cli("placement-manifest", "ensure", *common, "--move-group", f"{gn}=cluster_a", "--move-reason", "first") == 0      # the first declaration, as 06's ensure records it
        base = ["--dry-run", "--view", "d0p75", "--family", "multiturn", "--model", KEY, "--clusters", "cluster_b", "--wave-sha", SHA,
                "--move-group", f"{gn}=cluster_b", "--move-reason", "second: a lost its venv too"]
        r = run_stage("06_eval_waves.sh", *base, site=site, timeout=600)
        out = r.stdout + r.stderr
        assert r.returncode != 0 and f"already declares {gn}" in out, out[-1500:]           # no --re-move: 06's FIRST call (the plan) refuses the re-point by name
        r = run_stage("06_eval_waves.sh", *base, "--re-move", site=site, timeout=600)
        out = r.stdout + r.stderr
        # with --re-move the plan passes and 06's ensure names the re-point; under --dry-run ensure persists nothing, so 06's check then reads
        # the manifest still declaring the FIRST destination against a plan at the second and refuses — the dry-run artifact of any NEW
        # declaration on an existing manifest (a first move's dry-run is coherent: check tolerates a missing manifest in dry-run). Pinned as
        # observed; the real run persists before it checks (replayed below with 06's own argv)
        assert "already declares" not in out, out[-1500:]
        assert f"group move(s) DECLARED ({gn} -> cluster_b; reason: second: a lost its venv too)" in out, out[-2500:]
        assert "the placement moved" in out and r.returncode != 0
        # 06's ensure --re-move and check, replayed for real with 06's flags: the re-point persists, then the check passes against the sticky plan
        assert _cli("placement-manifest", "ensure", "--model", KEY, "--view", "d0p75", "--clusters", ",".join(AB), "--ledger", ledger, "--nshards", "9", "--sha", SHA,
                    "--move-group", f"{gn}=cluster_b", "--move-reason", "second: a lost its venv too", "--re-move") == 0
        man = ST.read_placement(ST.placement_path(ledger, "d0p75"))
        assert man["declared_moves"][gn]["to"] == "cluster_b" and man["declared_moves"][gn]["re_move_of"] == "cluster_a"
        rows2 = ST.unit_plan(KEY, "d0p75", families=["multiturn"], sha=SHA, clusters=AB, move_groups=ST.declared_move_map(man))
        plan_json = tmp_path / "plan06.json"; plan_json.write_text(json.dumps(rows2))
        assert _cli("placement-manifest", "check", "--model", KEY, "--view", "d0p75", "--clusters", ",".join(AB), "--ledger", ledger, "--plan-json", str(plan_json)) == 0
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_a_group_declared_moved_before_anything_ran_and_first_placed_on_the_destination_is_tolerated_pre_receipt_and_the_resubmit_receipt_advances_it(tmp_path, capsys):
    """move-knob-4: the battery wave's g5678 shape. A plain `ensure` pins g5678 at its plan origin (cluster_b); the move to
    cluster_a is then DECLARED with nothing bound anywhere; the first placement is a plain submission on the destination (the declared plan)
    and that attempt DIES at the standup. The pin holds the origin, the latest binding is on the destination, no receipt exists: before the
    fix `check` (and the ensure mirror) refused "pinned on cluster_b but its latest binding … is bound on cluster_a" on every later multiturn
    call, and the resubmit could never write the receipt (it wrote one only for a replaced attempt on ANOTHER cluster) — the group was stuck.
    Now: check + ensure tolerate exactly this group's declared destination; the fenced resubmit's receipt (from_cluster = the PIN, the
    replaced attempt already on the destination) advances the pin; an UNDECLARED cross-cluster binding still refuses by name."""
    root = _root(tmp_path); common = _common(root); targets = _g5678_targets()
    assert _cli("placement-manifest", "ensure", *common) == 0                                   # the plain pin: g5678 at its origin cluster_b
    man = ST.read_placement(ST.placement_path(str(root / "attempts.jsonl"), "s2r"))
    assert all(man["units"][lw]["cluster"] == "cluster_b" for lw in targets) and not man.get("declared_moves")
    capsys.readouterr()
    assert _cli("placement-manifest", "ensure", *common, "--move-group", "g5678=cluster_a", "--move-reason", REASON) == 0   # declared, nothing bound
    man = ST.read_placement(ST.placement_path(str(root / "attempts.jsonl"), "s2r"))
    assert man["declared_moves"]["g5678"]["to"] == "cluster_a" and all(man["units"][lw]["cluster"] == "cluster_b" for lw in targets)
    # the first placement under the declaration: plain submissions on cluster_a that die at the standup (registered, then FAILED)
    for i, lw in enumerate(targets):
        _ev(root, _attempt(lw, "cluster_a", str(300 + i), "2026-01-05T00:00:00Z"), _registered(lw, "cluster_a", str(300 + i), "2026-01-05T00:06:00Z"))
        _bind(root, lw, "cluster_a", str(300 + i))
    for lw in targets:
        pin = man["units"][lw]
        assert ST.declared_destination_binding(man, lw, pin, "cluster_a") and not ST.declared_destination_binding(man, lw, pin, "cluster_c")
    plan_json = _sticky_plan_json(tmp_path, common, capsys)
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(plan_json)) == 0, capsys.readouterr().err     # the refusal before the fix
    before = Path(ST.placement_path(str(root / "attempts.jsonl"), "s2r")).read_bytes()
    assert _cli("placement-manifest", "ensure", *common) == 0, capsys.readouterr().err                                   # the mirror tolerates it too
    assert Path(ST.placement_path(str(root / "attempts.jsonl"), "s2r")).read_bytes() == before                          # and rewrites nothing (no receipt yet)
    # the fenced resubmit on the destination: submit_twin writes the receipt FROM THE PIN (the replaced attempt already sat on cluster_a)
    sys.path.insert(0, str(REPO / "reproduce" / "lib")); import submit_twin as TW  # noqa: E402
    lw = targets[0]; replaced = ({"cluster": "cluster_a", "job_id": "300", "attempt_id": f"{lw}@cluster_a@300"}, "dead at the standup", "FAILED")
    assert TW.placement_move_origin(replaced, "cluster_a", "cluster_b") == "cluster_b"                                  # via the pin (the bare decision)
    assert TW.placement_move_origin(({"cluster": "cluster_b", "job_id": "1"}, "x", "FAILED"), "cluster_a", "cluster_b") == "cluster_b"   # the classic cross-cluster case
    assert TW.placement_move_origin(replaced, "cluster_a", "cluster_a") is None and TW.placement_move_origin(None, "cluster_a", "cluster_b") is None
    assert TW.manifest_pin_cluster(str(root), lw) == "cluster_b" and TW.manifest_pin_cluster(str(tmp_path / "nowhere"), lw) is None
    # move-knob-5: the receipt's origin as the funnel DECIDES it before submitting — the DECLARED g5678 move admits the pin-based receipt;
    # the classic cross-cluster replacement is unchanged; pin == label moves nothing; an UNDECLARED same-cluster replacement with the pin
    # elsewhere (the g01234 target pinned on cluster_a, a dead attempt on cluster_b, --resubmit on cluster_b) is REFUSED by name, nothing submitted
    assert TW.placement_receipt_origin(str(root), lw, replaced, "cluster_a", "mt-s2r-cluster_a") == "cluster_b"
    assert TW.placement_receipt_origin(str(root), lw, ({"cluster": "cluster_b", "job_id": "1"}, "x", "FAILED"), "cluster_a", "mt-s2r-cluster_a") == "cluster_b"
    assert TW.placement_receipt_origin(str(root), lw, ({"cluster": "cluster_b", "job_id": "9", "group": "mt-s2r-cluster_b"}, "x", "FAILED"), "cluster_b", "mt-s2r-cluster_b") is None   # pin == label: no move
    assert TW.placement_receipt_origin(str(root), lw, None, "cluster_a", None) is None
    g01234_lw = sorted(r["logical_work_id"] for r in ST.unit_plan(KEY, "s2r", families=["multiturn"], sha=SHA, clusters=AB)
                       if r.get("kind", "target") == "target" and ST._group_of_target(r["logical_work_id"]) == "g01234")[0]
    assert man["units"][g01234_lw]["cluster"] == "cluster_a"
    capsys.readouterr()
    with pytest.raises(TW.Refused) as ei:        # Refused prints its message to stderr and carries the rc
        TW.placement_receipt_origin(str(root), g01234_lw, ({"cluster": "cluster_b", "job_id": "777", "attempt_id": f"{g01234_lw}@cluster_b@777"}, "dead", "FAILED"), "cluster_b", "mt-s2r-cluster_b")
    err = capsys.readouterr().err
    assert ei.value.code == 3 and "declares no move to cluster_b" in err and "NOTHING submitted" in err and "pins the unit on cluster_a" in err, err
    with pytest.raises(TW.Refused) as ei2:      # the declared destination but the replaced attempt in the WRONG group: refused too
        TW.placement_receipt_origin(str(root), lw, ({"cluster": "cluster_a", "job_id": "300", "group": "mt-s2r-cluster_b"}, "dead", "FAILED"), "cluster_a", "mt-s2r-cluster_a")
    assert ei2.value.code == 3 and "not at the declared destination's group" in capsys.readouterr().err
    _ev(root, _attempt(lw, "cluster_a", "390", "2026-01-06T00:00:00Z"), _registered(lw, "cluster_a", "390", "2026-01-06T00:06:00Z"),
        {"event": "placement_receipt", "logical_work_id": lw, "from_cluster": "cluster_b", "to_cluster": "cluster_a", "to_group": "mt-s2r-cluster_a",
         "replaced_job_id": "300", "replaced_state": "FAILED", "attempt_id": f"{lw}@cluster_a@390", "job_id": "390", "utc": "2026-01-06T00:00:00Z",
         "reason": "fenced dead-job replacement under a declared move (submit_twin --resubmit; move-knob-4)"})
    _bind(root, lw, "cluster_a", "390")
    assert TW.manifest_pin_cluster(str(root), lw) == "cluster_a"            # the EFFECTIVE pin: the receipt moved it (the manifest's own pin is stale until ensure)
    assert TW.placement_move_origin(({"cluster": "cluster_a", "job_id": "390"}, "x", "FAILED"), "cluster_a", TW.manifest_pin_cluster(str(root), lw)) is None   # a later same-cluster re-run owes no second receipt
    capsys.readouterr()
    assert _cli("placement-manifest", "ensure", *common) == 0, capsys.readouterr().err
    man = ST.read_placement(ST.placement_path(str(root / "attempts.jsonl"), "s2r"))
    assert man["units"][lw]["cluster"] == "cluster_a" and man["units"][lw].get("receipt") and all(man["units"][o]["cluster"] == "cluster_b" for o in targets[1:])
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(_sticky_plan_json(tmp_path, common, capsys))) == 0, capsys.readouterr().err
    # an UNDECLARED cross-cluster binding (a g01234 target bound on cluster_b while pinned on cluster_a) still refuses by name
    g01234 = sorted(r["logical_work_id"] for r in ST.unit_plan(KEY, "s2r", families=["multiturn"], sha=SHA, clusters=AB)
                    if r.get("kind", "target") == "target" and ST._group_of_target(r["logical_work_id"]) == "g01234")
    assert g01234 and man["units"][g01234[0]]["cluster"] == "cluster_a"
    _bind(root, g01234[0], "cluster_b", "777")
    assert _cli("placement-manifest", "check", *common, "--plan-json", str(_sticky_plan_json(tmp_path, common, capsys))) != 0
    assert f"{g01234[0]}: pinned on cluster_a but its latest binding (job 777, submitted) is bound on cluster_b" in capsys.readouterr().err
