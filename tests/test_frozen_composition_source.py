"""The freeze-time workload composition comes from the FROZEN wave, never from today's working tree (commit 26).

`frozen_composition` used to call today's `unit_plan` with today's models table and only the frozen SHA as an argument, so a plan
or site-table change made before a unit's first frozen-tree attempt became its alleged baseline and a workload knob changed without a
registry-contract change shipped unrecognised. Now (a) the freeze persists each unit's composed workload identity in the inventory
(`workload`: script, args with the sha token, the plan's env knobs, array, group, kind) and `frozen_composition` reads it; (b) a legacy
inventory without it is derived from the FROZEN git tree — `git archive <code_sha>` into the state's gate directory once, and THAT tree's
`stage.py plan --json` over the pinned cluster order — with the source recorded on the identity. Synthetic World (fake Slurm)."""
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
LW = f"{WI.MODEL}/{VIEW}/mmlu/mmlu"
AB = ["cluster_a", "cluster_b"]


def _frozen(w):
    r = w.stage("05_stage_roster.sh", "--model", WI.MODEL, "--step", "freeze")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    return str(w.state / "submissions.jsonl")


def test_the_freeze_persists_each_units_workload_and_the_composition_reads_it_not_todays_plan(tmp_path, monkeypatch):
    w = WI.World(tmp_path); ledger = _frozen(w)
    inv = w.inventory()
    (unit,) = [u for u in inv["units"] if u["logical_work_id"] == LW]
    wl = unit["workload"]
    assert wl["script"] == "scripts/sdfbase/full22/mmlu_full.sbatch" and "{sha}" in wl["args"] and w.sha not in " ".join(wl["args"])
    assert set(wl) >= {"script", "args", "env", "array", "group", "kind"} and isinstance(wl["env"], dict)
    # every unit of the inventory carries one, and the multi-turn targets' carry their group per the pinned order
    assert all(isinstance(u.get("workload"), dict) for u in inv["units"])
    mt = [u for u in inv["units"] if u["family"] == "multiturn"]
    assert mt and all(u["workload"]["group"] in (f"mt-{VIEW}-{lab}" for lab in AB) for u in mt)
    # the composition is the persisted identity — today's unit_plan is never consulted
    monkeypatch.setattr(ST, "unit_plan", lambda *a, **k: (_ for _ in ()).throw(AssertionError("today's plan must not be consulted")))
    comp = ST.frozen_composition(ledger, LW, table=None, labels=AB, repo=str(w.repo))
    assert comp is not None and comp["script"] == wl["script"] and comp["args"] == wl["args"] and comp["env"] == wl["env"]
    assert comp["baseline"] == {"kind": "composition", "sha": inv["code_sha"], "source": "inventory"}


def test_a_legacy_inventory_is_composed_from_the_frozen_git_tree_never_the_working_tree(tmp_path):
    w = WI.World(tmp_path); ledger = _frozen(w)
    inv_p = w.state / "gate" / "expected_units.json"; inv = json.load(open(inv_p))
    for u in inv["units"]:
        u.pop("workload", None)                                   # the shape of an inventory frozen before this commit
    ST.write_atomic(str(inv_p), inv)
    comp = ST.frozen_composition(ledger, LW, table=None, labels=AB, repo=str(w.repo))
    assert comp is not None and comp["baseline"]["source"] == f"frozen-tree {inv['code_sha'][:12]}"
    row = next(r for r in ST.unit_plan(WI.MODEL, VIEW, table=str(w.repo / "reproduce" / "models.yaml"), families=["mmlu"], sha=inv["code_sha"], clusters=AB) if r["logical_work_id"] == LW)
    assert comp["script"] == row["script"] and comp["args"] == ST.spec_identity({"args": [str(a) for a in row["args"]], "sha": inv["code_sha"]})["args"]
    # the frozen tree is materialised ONCE beside the inventory and reused
    tree = w.state / "gate" / f"frozen_tree_{inv['code_sha'][:12]}"
    assert (tree / "reproduce" / "lib" / "stage.py").is_file()
    stamp = (tree / "reproduce" / "lib" / "stage.py").stat().st_mtime
    ST.frozen_composition(ledger, LW, table=None, labels=AB, repo=str(w.repo))
    assert (tree / "reproduce" / "lib" / "stage.py").stat().st_mtime == stamp
    # a code_sha the repo does not hold refuses by name — the working tree is never the fallback
    inv["code_sha"] = "e" * 40; ST.write_atomic(str(inv_p), inv)
    with pytest.raises(ValueError, match="frozen tree .* is not a commit of"):
        ST.frozen_composition(ledger, LW, table=None, labels=AB, repo=str(w.repo))
