"""A legacy inventory's UNPERSISTED freeze-time plan knobs are recovered from the first recorded frozen-tree attempt, or refused as
unknowable by name — the frozen tree's defaults are never asserted as the freeze's settings (commit 26c).

26b scrubbed the inherited `SISTEER_*` environment while re-planning a pre-commit-26 inventory's unit at its frozen tree and so planned with
the tree's DEFAULT multi-turn seed base (20260909). A wave frozen with SISTEER_MT_SEED_BASE=20261234 whose first frozen-tree attempt really ran
at that seed would, on retry, have been compared against the invented default and refused. Now the knob comes from the inventory's pinned
`plan_knobs` (every freeze since 26c pins them), else from the first frozen-tree attempt's recorded `F22_MT_TARGET_SEED_BASE`, else the
unit refuses by name; a unit whose row does not depend on the knob (`*` units) composes regardless."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_front_door_wave_integration as WI  # noqa: E402
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

VIEW = "battery"                              # the demo row's DECLARED wave view (demo-import M3): the inventory's units are <model>/battery/...
LW_MT = f"{WI.MODEL}/{VIEW}/multiturn/crescendo/g34"
LW_MMLU = f"{WI.MODEL}/{VIEW}/mmlu/mmlu"
AB = ["cluster_a", "cluster_b"]
SEED = "20261234"


def _frozen_with_seed(w, *, legacy: bool, drop_knobs: bool):
    r = w.stage("05_stage_roster.sh", "--model", WI.MODEL, "--step", "freeze", SISTEER_MT_SEED_BASE=SEED)
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    inv_p = w.state / "gate" / "expected_units.json"; inv = json.load(open(inv_p))
    assert inv["plan_knobs"]["SISTEER_MT_SEED_BASE"] == SEED                                     # 26c: the freeze pins the knobs it ran with
    mt = next(u for u in inv["units"] if u["logical_work_id"] == LW_MT)
    assert mt["workload"]["env"]["F22_MT_TARGET_SEED_BASE"] == SEED                               # commit 26: the composition carries the seed
    if legacy:
        for u in inv["units"]:
            u.pop("workload", None)                                                                # the shape of an inventory frozen before commit 26
    if drop_knobs:
        inv.pop("plan_knobs", None)                                                                 # … and before 26c
    ST.write_atomic(str(inv_p), inv)
    return str(w.state / "submissions.jsonl"), inv


def _record_frozen_attempt(w, inv, seed: str):
    """The unit's first attempt AT the frozen tree, as the funnel records it: its spec = the plan row at the frozen sha with the seed it ran."""
    row = next(r for r in ST.unit_plan(WI.MODEL, VIEW, table=str(w.repo / "reproduce" / "models.yaml"), families=["multiturn"], sha=inv["code_sha"], clusters=AB,
                                       mt_seed_base=seed) if r["logical_work_id"] == LW_MT)
    spec = {"script": row["script"], "args": [str(a) for a in row["args"]], "env": {**row["env"], "F22_ROSTER": "roster.yaml"}, "sha": inv["code_sha"],
            "array": row["array"], "group": row["group"], "kind": row["kind"]}
    ev = {"event": "attempt", "logical_work_id": LW_MT, "attempt_id": f"{LW_MT}@cluster_b@1", "cluster": "cluster_b", "job_id": "4242", "sha": inv["code_sha"],
          "kind": "target", "spec": spec, "utc": "2026-01-02T00:00:00Z"}
    with open(w.state / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps(ev) + "\n")


def test_a_pre_26c_legacy_inventory_recovers_the_seed_base_from_the_first_frozen_tree_attempt_and_refuses_without_one(tmp_path, monkeypatch):
    w = WI.World(tmp_path); ledger, inv = _frozen_with_seed(w, legacy=True, drop_knobs=True)
    monkeypatch.setenv("SISTEER_MT_SEED_BASE", "1")                                               # today's knob: never a source
    # no attempt yet: the multi-turn unit's seed base is unknowable — refused by name, never the tree's default
    with pytest.raises(ValueError, match=r"plan knob\(s\) \['SISTEER_MT_SEED_BASE'\] are unknowable .* no attempt at the frozen tree .* recorded \['F22_MT_TARGET_SEED_BASE'\]"):
        ST.frozen_composition(ledger, LW_MT, table=None, labels=AB, repo=str(w.repo))
    base, probs = ST.workload_baseline(ledger, LW_MMLU, table=None, labels=AB, repo=str(w.repo))     # a `*` unit does not depend on it: composes
    assert probs == [] and base["script"] == "scripts/sdfbase/full22/mmlu_full.sbatch" and base["baseline"]["kind"] == "composition"
    # the first frozen-tree attempt ran at the freeze's seed: recovered, the composition equals the attempt, the retry's baseline is sound
    _record_frozen_attempt(w, inv, SEED)
    comp = ST.frozen_composition(ledger, LW_MT, table=None, labels=AB, repo=str(w.repo))
    assert comp["env"]["F22_MT_TARGET_SEED_BASE"] == SEED and comp["baseline"]["plan_knobs"] == {"SISTEER_MT_SEED_BASE": SEED} \
        and comp["baseline"]["plan_knobs_source"].startswith("the unit's first frozen-tree attempt (job 4242 on cluster_b")
    base, probs = ST.workload_baseline(ledger, LW_MT, table=None, labels=AB, repo=str(w.repo))
    assert probs == [] and base["env"]["F22_MT_TARGET_SEED_BASE"] == SEED and base["baseline"]["kind"] == "composition+attempt" and base["baseline"]["job_id"] == "4242"


def test_a_26c_inventory_pins_its_knobs_so_the_composition_needs_no_attempt(tmp_path, monkeypatch):
    w = WI.World(tmp_path); ledger, inv = _frozen_with_seed(w, legacy=True, drop_knobs=False)
    monkeypatch.setenv("SISTEER_MT_SEED_BASE", "1")
    comp = ST.frozen_composition(ledger, LW_MT, table=None, labels=AB, repo=str(w.repo))
    assert comp["env"]["F22_MT_TARGET_SEED_BASE"] == SEED and comp["baseline"]["plan_knobs_source"] == "the inventory's pinned plan_knobs"
    # and a commit-26+ inventory reads the persisted workload, knobs included, without re-planning at all
    w2 = WI.World(tmp_path / "w2"); ledger2, _ = _frozen_with_seed(w2, legacy=False, drop_knobs=False)
    comp2 = ST.frozen_composition(ledger2, LW_MT, table=None, labels=AB, repo=str(w2.repo))
    assert comp2["env"]["F22_MT_TARGET_SEED_BASE"] == SEED and comp2["baseline"]["source"] == "inventory"


def test_an_attempt_at_another_seed_than_the_freeze_is_named_as_the_disagreement_it_is(tmp_path):
    """The attempt's seed IS the recovered knob; when a later retry composes something else the proof names both — never silently the default."""
    w = WI.World(tmp_path); ledger, inv = _frozen_with_seed(w, legacy=True, drop_knobs=True)
    _record_frozen_attempt(w, inv, "20269999")
    comp = ST.frozen_composition(ledger, LW_MT, table=None, labels=AB, repo=str(w.repo))
    assert comp["env"]["F22_MT_TARGET_SEED_BASE"] == "20269999"                                    # the evidence, not 20260909 and not today's
    base, probs = ST.workload_baseline(ledger, LW_MT, table=None, labels=AB, repo=str(w.repo))
    assert probs == [] and base["env"]["F22_MT_TARGET_SEED_BASE"] == "20269999"


def test_a_cap_only_refreeze_after_attempts_is_refused_and_the_live_plan_must_match_the_pinned_knobs(tmp_path, monkeypatch):
    """The attempted-wave refreeze refusal compared units and the tree only, so changing just
    SISTEER_MT_ATTACKER_GPUS after attempts passed the freeze-once gate and rewrote the header, and a later 06 pass planned extra pool
    members for the running wave. The knobs are part of the refusal, and `plan --inventory` refuses a live plan whose knobs differ."""
    import subprocess
    from test_admission import MODEL, freeze
    state = tmp_path / "state"; doc = freeze(state, families=["multiturn"])
    inv_p = state / "gate" / "expected_units.json"
    monkeypatch.setenv("SISTEER_MT_ATTACKER_GPUS", "1"); monkeypatch.delenv("SISTEER_MT_SEED_BASE", raising=False); monkeypatch.delenv("SISTEER_SHARD_GROUPS", raising=False)
    pinned = ST.freeze_plan_knobs(); assert pinned["SISTEER_MT_ATTACKER_GPUS"] == "1"
    doc["plan_knobs"] = pinned; ST.write_atomic(str(inv_p), doc)
    units, sha = doc["units"], doc["code_sha"]
    # no attempt yet: a changed cap may re-freeze; the same knobs are a no-op
    assert ST.refreeze_refusal(str(inv_p), units, sha, {**pinned, "SISTEER_MT_ATTACKER_GPUS": None}) is None
    with open(state / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps({"event": "attempt", "logical_work_id": units[0]["logical_work_id"], "attempt_id": f"{units[0]['logical_work_id']}@cluster_a@1", "cluster": "cluster_a",
                             "job_id": "4242", "sha": sha, "kind": "target", "spec": {}, "utc": "2026-01-02T00:00:00Z"}) + "\n")
    assert ST.refreeze_refusal(str(inv_p), units, sha, pinned) is None
    why = ST.refreeze_refusal(str(inv_p), units, sha, {**pinned, "SISTEER_MT_ATTACKER_GPUS": None})
    assert why and "1 plan knob(s) would change: SISTEER_MT_ATTACKER_GPUS '1' -> None" in why and "frozen ONCE" in why, why
    why = ST.refreeze_refusal(str(inv_p), units, sha, {**pinned, "SISTEER_MT_SEED_BASE": "1"})
    assert why and "SISTEER_MT_SEED_BASE" in why
    assert ST.plan_knob_diff(None, pinned) == {} and ST.plan_knob_diff(pinned, pinned) == {}                      # a pre-26c inventory pins nothing to compare
    # the live plan: `plan --inventory` refuses a differing knob, plans with the pinned ones, and ignores an absent inventory
    base = [sys.executable, str(REPO / "reproduce" / "lib" / "stage.py"), "plan", "--model", MODEL, "--view", VIEW, "--families", "multiturn", "--sha", sha,
            "--clusters", "cluster_a,cluster_b", "--models-table", str(REPO / "reproduce" / "models.yaml"), "--inventory", str(inv_p)]
    env = dict(__import__("os").environ); env["SISTEER_MT_ATTACKER_GPUS"] = "1"; env.pop("SISTEER_MT_SEED_BASE", None); env.pop("SISTEER_SHARD_GROUPS", None)
    r = subprocess.run(base, capture_output=True, text=True, env=env, timeout=300); assert r.returncode == 0 and r.stdout.strip(), r.stderr[-800:]
    env2 = dict(env); env2.pop("SISTEER_MT_ATTACKER_GPUS")
    r = subprocess.run(base, capture_output=True, text=True, env=env2, timeout=300)
    assert r.returncode == 2 and "live plan's knobs differ from the frozen inventory's" in r.stderr and "SISTEER_MT_ATTACKER_GPUS pinned '1', live None" in r.stderr and not r.stdout.strip(), r.stderr
    env3 = dict(env, SISTEER_MT_SEED_BASE="1")
    r = subprocess.run(base, capture_output=True, text=True, env=env3, timeout=300); assert r.returncode == 2 and "SISTEER_MT_SEED_BASE" in r.stderr, r.stderr
    r = subprocess.run(base[:-2] + ["--inventory", str(tmp_path / "absent.json")], capture_output=True, text=True, env=env2, timeout=300); assert r.returncode == 0, r.stderr[-800:]
    sh = (REPO / "reproduce" / "06_eval_waves.sh").read_text()
    assert 'PLAN_ARGS+=(--inventory "$STATE/gate/expected_units.json")' in sh and sh.index("PLAN_ARGS+=(--inventory") < sh.index('PLAN_TSV="$(')
def test_a_pool_only_attempt_counts_as_the_wave_started_so_a_cap_change_refreeze_is_refused(tmp_path, monkeypatch):
    """The freeze-once guard counted only attempts of inventory UNITS, but the attacker-pool rows register no
    unit and are absent from the inventory — a modular 06 pass that launched the capped pool first (no target attempt yet), then a
    SISTEER_MT_ATTACKER_GPUS change + refreeze, took `not attempted` and rewrote the pinned cap, and `plan --inventory` then submitted more
    members for the same started wave. ANY attempt of the wave, a pool member's included, pins the knobs."""
    from test_admission import MODEL, freeze
    state = tmp_path / "state"; doc = freeze(state, families=["multiturn"])
    inv_p = state / "gate" / "expected_units.json"
    monkeypatch.setenv("SISTEER_MT_ATTACKER_GPUS", "1"); monkeypatch.delenv("SISTEER_MT_SEED_BASE", raising=False); monkeypatch.delenv("SISTEER_SHARD_GROUPS", raising=False)
    pinned = ST.freeze_plan_knobs(); doc["plan_knobs"] = pinned; ST.write_atomic(str(inv_p), doc)
    units, sha = doc["units"], doc["code_sha"]
    pool_lw = f"{MODEL}/{VIEW}/multiturn/mt-{VIEW}-cluster_a/attackers/1"
    assert pool_lw not in {u["logical_work_id"] for u in units}                                            # the pool row is not an inventory unit
    with open(state / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps({"event": "attempt", "logical_work_id": pool_lw, "attempt_id": f"{pool_lw}@cluster_a@1", "cluster": "cluster_a", "job_id": "4300", "sha": sha,
                             "kind": "attacker", "spec": {"group": f"mt-{VIEW}-cluster_a"}, "utc": "2026-01-02T00:00:00Z"}) + "\n")
    assert ST.refreeze_refusal(str(inv_p), units, sha, pinned) is None                                        # the same knobs: a no-op
    why = ST.refreeze_refusal(str(inv_p), units, sha, {**pinned, "SISTEER_MT_ATTACKER_GPUS": None})
    assert why and "SISTEER_MT_ATTACKER_GPUS '1' -> None" in why and "attacker pool has already launched (1 member attempt(s)" in why and pool_lw in why \
        and "no consumers" in why, why
    # another view's pool attempt does not pin THIS inventory's knobs
    (state / "attempts.jsonl").write_text("")
    with open(state / "attempts.jsonl", "a") as fh:
        fh.write(json.dumps({"event": "attempt", "logical_work_id": f"{MODEL}/s2/multiturn/mt-s2-cluster_a/attackers/1", "attempt_id": "x@cluster_a@1", "cluster": "cluster_a",
                             "job_id": "4301", "sha": sha, "kind": "attacker", "spec": {}, "utc": "2026-01-02T00:00:00Z"}) + "\n")
    assert ST.refreeze_refusal(str(inv_p), units, sha, {**pinned, "SISTEER_MT_ATTACKER_GPUS": None}) is None
