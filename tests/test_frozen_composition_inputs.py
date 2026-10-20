"""A legacy inventory's freeze-time composition takes EVERY input from the frozen wave — the archived tree's models table, the pinned
placement universe — never today's working tree or an unpinned plan knob (commit 26b).

`frozen_composition`'s legacy fallback executed the FROZEN stage.py but fed it TODAY's inputs: `--models-table` from the live site path,
TODAY's cluster labels, and the inherited `SISTEER_SHARD_GROUPS` / seed-base env — so a model-table change, a seed-base change or a cluster
reorder made after the freeze and before a unit's first frozen-tree attempt became its alleged frozen baseline; a model removed from
today's table read as "no baseline" (None). Now: the models table is the archived frozen tree's (`reproduce/models.yaml` inside
`git archive <code_sha>`); the cluster labels come ONLY from the pinned placement manifest or, failing that, from the unit's effective
attempt evidence — with neither, the composition REFUSES by name; the frozen tree's own defaults stand for the plan knobs (the inherited
env is scrubbed for the subprocess); a removed model is a refusal, never None.

The manifest read is the INVENTORY'S VIEW's — the file the freeze pins (`placement_path(ledger, view)`: u0's historic
`expected_units.placement.json`, or the single-phase recipe's one wave's per-view `expected_units.placement.battery.json`). The legacy
composition read the u0 name for every inventory, so an battery-baseline wave's pinned universe was never found and every placed unit refused as
unknowable (the demo key, 2026-10-09); the first test below pins the namespace and the second that the refusal names the view's file."""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_front_door_wave_integration as WI  # noqa: E402
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import stage as ST  # noqa: E402

LW_MT = f"{WI.MODEL}/battery/multiturn/crescendo/g34"
AB = ["cluster_a", "cluster_b"]


def _frozen_legacy(w):
    r = w.stage("05_stage_roster.sh", "--model", WI.MODEL, "--step", "freeze")
    assert r.returncode == 0, r.stdout[-1500:] + r.stderr[-1500:]
    inv_p = w.state / "gate" / "expected_units.json"; inv = json.load(open(inv_p))
    for u in inv["units"]:
        u.pop("workload", None)                                   # the shape of an inventory frozen before commit 26
    ST.write_atomic(str(inv_p), inv)
    return str(w.state / "submissions.jsonl"), inv


def test_the_models_table_and_the_labels_come_from_the_frozen_wave_not_today(tmp_path, monkeypatch):
    w = WI.World(tmp_path); ledger, inv = _frozen_legacy(w)
    before = ST.frozen_composition(ledger, LW_MT, table=None, labels=AB, repo=str(w.repo))
    assert before and before["baseline"]["source"].startswith("frozen-tree") and before["group"] == "mt-battery-cluster_b"
    # the universe is the manifest the FREEZE pinned — the inventory's view's file (the single-phase baseline wave battery: its per-view
    # `expected_units.placement.battery.json`; no freeze of an battery-baseline wave writes the u0 historic name, so reading that name finds nothing)
    pm = Path(ST.placement_path(ledger, inv["view"]))
    assert inv["view"] == "battery" and pm.name == "expected_units.placement.battery.json" and pm.is_file() and not (pm.parent / ST.PLACEMENT_FILE).exists()
    assert before["baseline"]["labels"] == AB and before["baseline"]["labels_source"] == f"the pinned placement manifest {pm}"
    # AFTER the freeze: today's models table changes the model's served id, today's env carries a foreign seed base, and the site's
    # clusters are reordered — none of it reaches the composition
    table = w.repo / "reproduce" / "models.yaml"; txt = table.read_text()
    # the demo's SERVED checkpoint line alone: the table names Qwen/Qwen3.5-4B in the 4B paper row too and the fit rows' `-Base` ids contain
    # it, so the edit is scoped to WI.MODEL's block (the World's own seam) and to its `serve:` checkpoint's hf_repo (the trailing comma
    # excludes the `-Base` fit line)
    i = txt.index(f"\n  {WI.MODEL}:\n"); j = txt.index("\n  gemma4e4b:\n", i); block = txt[i:j]
    served = "hf_repo: Qwen/Qwen3.5-4B,"; assert WI.MODEL == "qwen35_4b_demo" and block.count(served) == 1, (WI.MODEL, block.count(served))
    table.write_text(txt[:i] + block.replace(served, "hf_repo: Qwen/Qwen3.5-4B-MUTATED,", 1) + txt[j:])
    monkeypatch.setenv("SISTEER_MT_SEED_BASE", "1")
    monkeypatch.setenv("SISTEER_SHARD_GROUPS", "0,1,2,3,4,5,6,7,8")
    after = ST.frozen_composition(ledger, LW_MT, table=str(table), labels=list(reversed(AB)), repo=str(w.repo))
    assert {k: after[k] for k in ("script", "args", "env", "array", "group", "kind")} == {k: before[k] for k in ("script", "args", "env", "array", "group", "kind")}
    assert after["baseline"]["labels"] == AB                       # the pinned placement universe, not the reordered call
    # a model REMOVED from today's table is irrelevant to the frozen tree's own table — the composition still derives
    table.write_text(re.sub(r"(?m)^(\s*)" + re.escape(WI.MODEL) + r":", r"\1" + WI.MODEL + "_gone:", txt, count=1))
    gone = ST.frozen_composition(ledger, LW_MT, table=str(table), labels=AB, repo=str(w.repo))
    assert gone and gone["args"] == before["args"]


def test_without_a_pinned_placement_or_attempt_evidence_the_composition_refuses_by_name(tmp_path):
    w = WI.World(tmp_path); ledger, inv = _frozen_legacy(w)
    pm = Path(ST.placement_path(ledger, inv["view"])); pm.unlink()             # the pinned universe (the VIEW's file) gone, no attempts: the labels are unknowable
    with pytest.raises(ValueError, match=r"placement universe .* neither a pinned placement manifest nor attempt evidence") as ei:
        ST.frozen_composition(ledger, LW_MT, table=None, labels=list(reversed(AB)), repo=str(w.repo))
    assert f"(looked for {pm})" in str(ei.value) and pm.name == "expected_units.placement.battery.json", str(ei.value)   # it looked for the view's file, never the u0 name
    # a single-GPU unit (`*`) needs no placement: it still composes
    comp = ST.frozen_composition(ledger, f"{WI.MODEL}/battery/mmlu/mmlu", table=None, labels=list(reversed(AB)), repo=str(w.repo))
    assert comp and comp["script"] == "scripts/sdfbase/full22/mmlu_full.sbatch"


def test_a_frozen_tree_whose_table_lacks_the_model_refuses_never_none(tmp_path):
    w = WI.World(tmp_path); ledger, inv = _frozen_legacy(w)
    inv_p = w.state / "gate" / "expected_units.json"; inv["model"] = "no_such_model_key"
    for u in inv["units"]:
        u["logical_work_id"] = u["logical_work_id"].replace(WI.MODEL + "/", "no_such_model_key/", 1)
    ST.write_atomic(str(inv_p), inv)
    with pytest.raises(ValueError, match="no_such_model_key.*not in the frozen tree's models table"):
        ST.frozen_composition(ledger, "no_such_model_key/battery/mmlu/mmlu", table=None, labels=AB, repo=str(w.repo))
