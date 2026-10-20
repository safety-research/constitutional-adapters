"""The planner used the target-aware predicate INSIDE refuse_unsteered_legs_on_steered_wave_view but still decided WHETHER to run it — and
whether view_cells_of checks explicit_cells, and whether the Petri proxy row gets its cells positional — with the old name-only
`M.steered_wave_view(view)`. A front-door s1 / s2 / dose view with a stale generated config or manifest carrying `none` was therefore planned
without inspecting its artefacts, `--cells none` passed the validator, and the proxy row named no cells (a whole-enumeration audit). All three
reads are target-aware now:

* unit_plan on a front-door non-u0 view whose generated config lists `none` (or whose Petri manifest lists `none__lora0`) is refused by name
  before any row; `--cells none` on such a view is refused as well;
* the Petri proxy row of a front-door non-u0 view (s1 included — not an `s<N>r` name) carries the view's cells on its positional and in
  F22_REGISTRY_CELLS, never `none`; u0's proxy keeps the whole baseline (no positional);
* a paper model's `main` view (no phase view) is untouched by the walk.

The tree's front-door model (`qwen35_4b_demo`) runs the SINGLE-PHASE recipe (one wave s1 carrying every arm beside its own unsteered legs, no
u0), so none of its views is a steered wave and a `none` in its s1 config is the recipe, not a stale artefact; the target-aware reads are
therefore exercised on a SYNTHETIC row that runs in phases (u0 + s1 + dose views) on a tmp copy of the models table (SISTEER_MODELS_TABLE),
and the demo's own waves are checked to keep their whole enumeration. Synthetic roster / configs under tmp (F22_ROSTER); the committed
artefacts are read, never written."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import models as MT  # noqa: E402
import stage as ST  # noqa: E402

KEY = "qwen35_4b_demo"                 # the tree's front-door model: the SINGLE-PHASE recipe (one wave s1 + two dose rungs, no u0)
PHASED = "phased_demo"                 # a SYNTHETIC row that RUNS IN PHASES (u0 + s1 + the dose views) on a tmp copy of the table: the steered-wave rule's subject
S1_CELLS = ["constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl"]


def _phased_table(d: Path) -> str:
    """A copy of the models table under tmp plus PHASED: the demo row's pins and objects under its own key and view_legacy_prefix, with the
    phased recipe's views — u0 (none + the prompted instructions), s1 (the LoRA wave), the two dose rungs. The ONE reader loads it."""
    import copy
    t = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text())
    row = copy.deepcopy(t["models"][KEY]); row["view_legacy_prefix"] = PHASED
    row["views"] = {"u0": {"cells": ["none", "prompted__safety_refusal", "prompted__si_constitution"]},
                    "s1": {"cells": ["control_lora", "constitution_lora", "difference_lora"]},
                    "d0p75": {"cells": ["difference_lora"], "dose": 0.75, "dose_unit": "lora_write_basis_multiplier"},
                    "d0p75_relu": {"cells": ["difference_relu"], "dose": 0.75, "dose_unit": "psr_direction_norm_multiplier", "object_family": "relu"}}
    t["models"][PHASED] = row
    d.mkdir(parents=True, exist_ok=True); p = d / "models.yaml"; p.write_text(yaml.safe_dump(t, sort_keys=False))
    return str(p)


def _stale_world(root: Path, view: str, *, config_cells, manifest_cells, key: str = PHASED) -> Path:
    (root / "configs" / "capeval" / "manifests").mkdir(parents=True, exist_ok=True)
    cfg = root / "configs" / f"sdfbase_full22_{key}_{view}.yaml"; cfg.write_text(yaml.safe_dump({"explicit_cells": list(config_cells)}))
    man = root / "configs" / "capeval" / "manifests" / f"petri_full22_{key}_{view}.yaml"; man.write_text(yaml.safe_dump({"petri": {"cells": list(manifest_cells)}}))
    roster = root / "configs" / "sdfbase_full22_roster.yaml"
    roster.write_text(yaml.safe_dump({"targets": {key: {"views": {view: {"config": str(cfg), "manifest": str(man),
                                                                            "cells": {c: f"base__{c}" for c in config_cells if c != "none"}}}}}}))
    return roster


@pytest.fixture
def phased_table(tmp_path, monkeypatch) -> str:
    """The tmp table with the phased row, bound as THE models table (stage.py's predicate reads reproduce/lib/models.py's default table:
    SISTEER_MODELS_TABLE) for the test."""
    p = _phased_table(tmp_path / "table")
    monkeypatch.setenv("SISTEER_MODELS_TABLE", p)
    return p


def test_the_three_predicate_reads_in_the_planner_are_target_aware(phased_table):
    src = (REPO / "reproduce" / "lib" / "stage.py").read_text()
    body_vc = src[src.index("def view_cells_of("):src.index("def refuse_unsteered_legs_on_steered_wave_view(")]
    # target-aware AND table-aware: the key beside the view, and the caller's table beside the key (2026-10-08)
    assert "M.steered_wave_view(view, model_key, table)" in body_vc and "M.steered_wave_view(view)" not in body_vc
    body_up = src[src.index("def unit_plan("):]
    assert 'if M.steered_wave_view(view, model_key, table) and not os.environ.get("SISTEER_PLAN_SKIP_ARTEFACT_CHECK")' in body_up
    assert "        if M.steered_wave_view(view, model_key, table):" in body_up and "        if M.steered_wave_view(view):\n" not in body_up
    assert "M.steered_wave_view(view, model_key)" not in body_up and "M.baseline_view(model_key)" not in body_up   # no read without the table
    # the predicate itself: s1 is a steered wave FOR a front-door target that runs in phases (and for a bare phase-view name), never for a paper
    # model's key; u0 never; and never for the SINGLE-PHASE demo, whose one wave s1 carries its own unsteered legs
    assert MT.steered_wave_view("s1", PHASED) and MT.steered_wave_view("s1") and not MT.steered_wave_view("s1", "gemma4e4b") and MT.steered_wave_view("s2r")
    assert not MT.steered_wave_view("u0", PHASED) and not MT.steered_wave_view("u0") and not MT.steered_wave_view("main", "gemma4e4b")
    assert MT.single_phase(KEY) and MT.baseline_view(KEY) == "battery" and not MT.single_phase(PHASED) and MT.baseline_view(PHASED) == "u0"
    for v in ("battery", "d0p75", "d0p75_relu"):
        assert not MT.steered_wave_view(v, KEY), v


def test_unit_plan_refuses_a_front_door_non_u0_view_whose_stale_artefacts_carry_an_unsteered_leg(tmp_path, monkeypatch, phased_table):
    monkeypatch.delenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", raising=False)
    # a stale s1 config listing `none` (the shape before the steered-wave rule) on the PHASED row: refused by name at view_cells_of, at the guard, and at unit_plan before any row
    roster = _stale_world(tmp_path / "a", "s1", config_cells=["none", *S1_CELLS], manifest_cells=[f"si_{c}" for c in S1_CELLS])
    with pytest.raises(ValueError, match=r"lists the unsteered leg\(s\) \['none'\] on a steered-wave view"):
        ST.view_cells_of(PHASED, "s1", roster_path=str(roster))
    with pytest.raises(ValueError, match="unsteered leg"):
        ST.refuse_unsteered_legs_on_steered_wave_view(PHASED, "s1", roster_path=str(roster))
    monkeypatch.setenv("F22_ROSTER", str(roster))
    with pytest.raises(ValueError, match=r"unsteered leg\(s\) \['none'\]"):
        ST.unit_plan(PHASED, "s1", families=["static"], sha="a" * 40, clusters=["cluster_a"])
    with pytest.raises(ValueError, match=r"unsteered leg\(s\) \['none'\]"):                       # `--cells none` cannot pass the validator either
        ST.unit_plan(PHASED, "s1", families=["static"], sha="a" * 40, clusters=["cluster_a"], cells=["none"])
    # the config clean, the manifest carrying the controls twin: refused by name too — and a dose view is a steered wave as well
    roster2 = _stale_world(tmp_path / "b", "d0p75", config_cells=["constv2-mix-full__lora_r4_ce_minus_ctl"], manifest_cells=["si_constv2-mix-full__lora_r4_ce_minus_ctl", "none__lora0"])
    monkeypatch.setenv("F22_ROSTER", str(roster2))
    with pytest.raises(ValueError, match=r"Petri manifest .* lists the unsteered leg\(s\) \['none__lora0'\]"):
        ST.unit_plan(PHASED, "d0p75", families=["static"], sha="a" * 40, clusters=["cluster_a"])
    # clean artefacts: planned; `--cells none` is still refused (not in the view's explicit_cells)
    roster3 = _stale_world(tmp_path / "c", "s1", config_cells=S1_CELLS, manifest_cells=[f"si_{c}" for c in S1_CELLS])
    monkeypatch.setenv("F22_ROSTER", str(roster3))
    rows = ST.unit_plan(PHASED, "s1", families=["static", "petri"], sha="a" * 40, clusters=["cluster_a"])
    assert rows and all(r["logical_work_id"].split("/")[1] == "s1" for r in rows)
    with pytest.raises(ValueError, match=r"--cells: \['none'\] are not in"):
        ST.unit_plan(PHASED, "s1", families=["static"], sha="a" * 40, clusters=["cluster_a"], cells=["none"])
    # the SINGLE-PHASE demo: `none` in its s1 config is the recipe (its own unsteered leg), not a stale artefact — nothing refuses, `--cells none` is a valid selection
    roster4 = _stale_world(tmp_path / "d", "battery", key=KEY, config_cells=["none", *S1_CELLS], manifest_cells=["none", *(f"si_{c}" for c in S1_CELLS)])
    monkeypatch.setenv("F22_ROSTER", str(roster4))
    assert ST.view_cells_of(KEY, "battery", roster_path=str(roster4)) == ["none", *S1_CELLS]
    ST.refuse_unsteered_legs_on_steered_wave_view(KEY, "battery", roster_path=str(roster4))
    rows = ST.unit_plan(KEY, "battery", families=["static"], sha="a" * 40, clusters=["cluster_a"])
    assert rows and all(r["logical_work_id"].split("/")[1] == "battery" for r in rows)
    assert all(r["cells"] == ["none"] for r in ST.unit_plan(KEY, "battery", families=["static"], sha="a" * 40, clusters=["cluster_a"], cells=["none"]) if r["registers"])


def test_the_petri_proxy_row_of_a_front_door_s1_view_declares_its_cells_and_u0s_keeps_the_whole_baseline(tmp_path, monkeypatch, phased_table):
    monkeypatch.delenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", raising=False)
    roster = _stale_world(tmp_path / "a", "s1", config_cells=S1_CELLS, manifest_cells=[f"si_{c}" for c in S1_CELLS])
    monkeypatch.setenv("F22_ROSTER", str(roster))
    petri = [r for r in ST.unit_plan(PHASED, "s1", families=["petri"], sha="a" * 40, clusters=["cluster_a"]) if r["family"] == "petri"]
    assert len(petri) == 1 and petri[0]["args"][:3] == [PHASED, "s1", "a" * 40]
    assert petri[0]["args"][-1] == ",".join(S1_CELLS) and petri[0].get("env", {}).get(ST.CELLS_ENV) == ",".join(S1_CELLS), petri[0]
    assert "none" not in str(petri[0]["args"]) and "none__lora0" not in str(petri[0]["args"])
    # u0: the whole baseline — no cells positional, no registry cells env
    monkeypatch.delenv("F22_ROSTER")
    petri_u0 = [r for r in ST.unit_plan(PHASED, "u0", families=["petri"], sha="a" * 40, clusters=["cluster_a"]) if r["family"] == "petri"]
    assert len(petri_u0) == 1 and petri_u0[0]["args"] == [PHASED, "u0", "a" * 40] and not (petri_u0[0].get("env") or {}).get(ST.CELLS_ENV)
    # the SINGLE-PHASE demo's s1 IS its baseline wave: the whole enumeration (none included), no cells positional, no registry cells env
    petri_s1 = [r for r in ST.unit_plan(KEY, "battery", families=["petri"], sha="a" * 40, clusters=["cluster_a"]) if r["family"] == "petri"]
    assert len(petri_s1) == 1 and petri_s1[0]["args"] == [KEY, "battery", "a" * 40] and not (petri_s1[0].get("env") or {}).get(ST.CELLS_ENV)
    # a paper model's main view: no phase view, never a steered wave — the walk does not run
    assert not MT.steered_wave_view("main", "gemma4e4b")
