"""scripts/sdfbase/full22/sampled/submit_sampled.py plan(): an ONBOARDED model's steered launch is sized through geometry.budget_for /
rows_for WITH the SELECTED LoRA module set — the models table's promoted `lora_architecture`, read
through padding_preflight.selected_module_set, the ONE source the padded pins use — so the site geometry overlay's certified row for
THAT set sizes the launch instead of the provisional ceiling. A steered view with no promoted pick refuses by name; an unsteered view
(a prompted-only wave: base + prompted only — the phased recipe's u0 shape, which the single-phase demo does not declare, so a prompted-only
block is pasted under its key by hand here) sizes from the built-in row and says so; a pick the overlay does not certify refuses by name.
Synthetic table + overlay under tmp; the front door's synthetic roster block (tests/test_full22_front_door_targets.front_door_roster) for its
one steered wave battery."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts/sdfbase/full22/sampled"))
sys.path.insert(0, str(REPO / "scripts/sdfbase/full22/static_padding"))
sys.path.insert(0, str(REPO / "tests"))
from sisteer.capeval.serving import geometry as g  # noqa: E402
import padding_preflight as pf  # noqa: E402
import submit_sampled as ss  # noqa: E402
from test_full22_front_door_targets import front_door_roster  # noqa: E402
from test_site_geometry_overlay import _record, _set, _table, _write, KEY, QKVO, WO  # noqa: E402

S2, U0 = f"{KEY}_battery", f"{KEY}_u0"       # the steered standup = the demo's one wave battery; the unsteered = a SYNTHETIC prompted-only block (below)
#: the front door's synthetic roster block, built ONCE from the real models table (before any test points SISTEER_MODELS_TABLE at a synthetic one)
ROSTER = front_door_roster(KEY, ("battery",))
ROSTER["targets"][KEY]["views"]["u0"] = {"config": f"configs/sdfbase_full22_{KEY}_u0.yaml", "manifest": f"configs/capeval/manifests/petri_full22_{KEY}_u0.yaml",
                                          "cells": {"prompted__safety_refusal": None, "prompted__si_constitution": None}}   # prompted-only: no object, no module set


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for v in (g.SITE_GEOMETRY_DIR_ENV, g.SITE_DIR_ENV, g.SHARED_ROOT_ENV, pf.MODELS_TABLE_ENV):
        monkeypatch.delenv(v, raising=False)
    orig = ss.split_target
    monkeypatch.setattr(ss, "split_target", lambda t: (KEY, t[len(KEY) + 1:]) if t.startswith(KEY + "_") else orig(t))


def _plan(tag: str):
    import copy
    return ss.plan(copy.deepcopy(ROSTER), [tag], {tag: "cluster_b"}, 9)


def test_steered_view_sizes_from_the_selected_sets_certified_overlay_row(monkeypatch, tmp_path):
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path / "t", WO)))                 # the promoted pick: o_proj -> wo
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(_write(tmp_path, _record(sets={"wo": _set("wo", WO, 4096, S=6, L=49152)}))))
    p = _plan(S2)
    served = {r["key"]: r for r in p["served"]}[S2]
    assert served["module_set"] == "wo" and served["budget"] == 6 * 49152 == g.budget_for(KEY, "wo").batched_token_budget   # the overlay's row, not 393,216
    assert served["budget"] != 393_216
    assert served["geom"]["static_base"][1] == g.budget_for(KEY, "wo").entry.seqs_at(served["geom"]["static_base"][0])
    # the planner's printed geometry line names the set beside the budget
    src = (REPO / "scripts/sdfbase/full22/sampled/submit_sampled.py").read_text()
    assert "' module set ' + s['module_set']" in src


def test_steered_view_with_a_pending_pick_refuses_by_name_and_u0_sizes_from_the_built_in_row(monkeypatch, tmp_path):
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path / "t", None, selected_by="pending_user_pick")))
    with pytest.raises(SystemExit, match=f"{S2}: {KEY}: no promoted LoRA architecture pick.*steered cells \\["):
        _plan(S2)
    p = _plan(U0)                                                                            # base + prompted only: no module set needed
    served = {r["key"]: r for r in p["served"]}[U0]
    assert served["module_set"] is None and served["budget"] == 393_216 and served["hybrid"] is True      # qwen35_4b_demo: qwen3_5_hybrid (g.ONBOARDED_HYBRID_MODEL_KEYS; the HYBRID_SEQS_CAP applies)
    assert KEY in g.ONBOARDED_HYBRID_MODEL_KEYS and served["geom"]["static_base"] == (4864, 393_216 // 4864)
    assert any("unsteered view, no promoted LoRA pick yet" in s for s in p["skipped"]), p["skipped"]


def test_a_pick_the_overlay_does_not_certify_refuses_by_name(monkeypatch, tmp_path):
    monkeypatch.setenv(pf.MODELS_TABLE_ENV, str(_table(tmp_path / "t", WO)))
    monkeypatch.setenv(g.SITE_GEOMETRY_DIR_ENV, str(_write(tmp_path, _record(sets={"qkvo": _set("qkvo", QKVO, 9216, S=2, L=98304)}))))
    with pytest.raises(SystemExit, match="selected module set 'wo' has NO certified geometry record.*certified module sets: \\['qkvo'\\]"):
        _plan(S2)


def test_a_paper_key_is_sized_as_before_without_a_module_set():
    import copy
    p = ss.plan(copy.deepcopy(ROSTER), ["qwen35_4b"], {"qwen35_4b": "cluster_a"}, 9)
    served = {r["key"]: r for r in p["served"]}["qwen35_4b"]
    assert served["module_set"] is None and served["budget"] == g.budget_for("qwen35_4b").batched_token_budget and g.budget_for("qwen35_4b").measured
    src = (REPO / "scripts/sdfbase/full22/sampled/submit_sampled.py").read_text()
    assert "BUDGET_BY_MODEL[" not in src and "g.budget_for(key, module_set)" in src and "g.rows_for(key, module_set)" in src
    assert "pf.selected_module_set(key)" in src                                            # the one source, the padded pins' too
