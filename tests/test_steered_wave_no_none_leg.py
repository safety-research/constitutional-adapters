"""A STEERED-WAVE view (`s<N>r`) generates from its declared cell set with no
unconditional `none` — the wave config, the six sampled rung configs, the Petri manifest, the misalign pair, the GSM8K matrix and overlays —
as a GENERATOR invariant (f22_roster.view_cell_universe), and default planning refuses by name when a steered-wave view's generated artefacts
carry `none` / `none__lora0`. SERVED != MEASURED: the standup keeps serving the base route implicitly (load_check.py probes `none`,
the native attestation reads vmodels["none"]) while nothing generated, planned, registered or scored names it — the sampled drivers iterate the
config's cells, the misalign config's `measure_none: false` leaves the served base leg unmeasured, the Petri frozen set is the explicit
declaration (g32). u0's generation is byte-identical to before (pinned). Before: the s3r artefacts carried `none` everywhere and a
default `go_s3r.sh static` would have drawn, registered and scored an unsteered run, which is refused.
The tree's one front-door model (`qwen35_4b_demo`) runs the SINGLE-PHASE recipe (one wave s1 beside its own unsteered legs,
two dose rungs, no u0), so none of ITS views is a steered wave; the rule (every phase view but u0 of a model that runs in phases) is
exercised on a SYNTHETIC phased row added to a tmp copy of the table. Synthetic fixtures only (placeholder shas, TESTHOST); the committed
artefacts are read, never written."""
from __future__ import annotations

import hashlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "misalign"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "gsm8k"))
sys.path.insert(0, str(REPO / "src"))
import f22_roster as FR  # noqa: E402
import models as MT  # noqa: E402
import stage as ST  # noqa: E402
import g8k_views as GV  # noqa: E402
import misalign_cells as MC  # noqa: E402

KEY = "qwen35_4b_demo"                                               # the front-door model of reproduce/models.yaml (configured, not run): the SINGLE-PHASE
DEMO_VIEWS = ("battery", "d0p75", "d0p75_relu")                      # recipe — ONE wave `battery` (the row's declared wave) carrying every arm beside its own unsteered legs,
UNSTEERED = ("none", "none__lora0", "none__lens0")                   # plus the two dose rungs; no u0, no s2; none of its views is a steered wave
PHASED = "phased_demo"                                               # a SYNTHETIC row that RUNS IN PHASES (u0 + s1 + s2 + the dose views) on a tmp copy of the table:
PHASED_STEERED_VIEWS = ("s1", "s2", "d0p75", "d0p75_relu")           # the phased rule's subject (every phase view but u0 a steered wave) — no model of the tree declares u0
DIFF = "constv2-mix-full__psr_ce_all_minus_ctl"
DIFF_LORA = "constv2-mix-full__lora_r4_ce_minus_ctl"
DEMO_S1_ROSTER_CELLS = ["prompted__safety_refusal", "prompted__si_constitution", "arditi_allpos", "constv2ctl2-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce",
                        DIFF_LORA, "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all", DIFF]   # roster_block's order (none omitted: the generator adds the anchor); the ReLU difference is the COMPOSED paired write (demo-import M2)


def _front_door_roster() -> dict:
    """The production roster + the front-door model's block exactly as 05 --step roster prints it (reproduce/lib/models.py roster_block over the
    table's views; the production roster commits no front-door block)."""
    roster = FR.load_roster(REPO / "configs" / "sdfbase_full22_roster.yaml")
    roster["targets"][KEY] = yaml.load("targets:\n" + MT.roster_block(KEY, list(DEMO_VIEWS)), Loader=FR._StrictRosterLoader)["targets"][KEY]
    return roster


ROSTER = _front_door_roster()


def _phased_table(d: Path) -> str:
    """A copy of the models table under tmp plus PHASED: the demo row's pins and objects under its own key and view_legacy_prefix, with the PHASED
    recipe's views — u0 (none + the prompted instructions), s1 (the LoRA wave), s2 (the ReLU wave), the two dose rungs. The ONE reader loads it."""
    import copy
    t = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text())
    row = copy.deepcopy(t["models"][KEY]); row["view_legacy_prefix"] = PHASED
    row["views"] = {"u0": {"cells": ["none", "prompted__safety_refusal", "prompted__si_constitution"]},
                    "s1": {"cells": ["refusal_direction", "control_lora", "constitution_lora", "difference_lora"]},
                    "s2": {"cells": ["control_relu", "constitution_relu", "difference_relu"]},
                    "d0p75": {"cells": ["difference_lora"], "dose": 0.75, "dose_unit": "lora_write_basis_multiplier"},
                    "d0p75_relu": {"cells": ["difference_relu"], "dose": 0.75, "dose_unit": "psr_direction_norm_multiplier", "object_family": "relu"}}
    t["models"][PHASED] = row
    d.mkdir(parents=True, exist_ok=True); p = d / "models.yaml"; p.write_text(yaml.safe_dump(t, sort_keys=False))
    return str(p)


@pytest.fixture(scope="module")
def phased_table(tmp_path_factory) -> str:
    return _phased_table(tmp_path_factory.mktemp("phased"))


def _misalign_manifest_module(table: str):
    """A FRESH copy of make_misalign_manifest.py imported with SISTEER_MODELS_TABLE = `table` (its FRONT_DOOR / SINGLE_PHASE_KEYS are read at
    import), so the shared module object the other tests import is never mutated."""
    import importlib.util
    prev = os.environ.get("SISTEER_MODELS_TABLE")
    os.environ["SISTEER_MODELS_TABLE"] = table
    try:
        spec = importlib.util.spec_from_file_location("_no_none_leg_mm_" + hashlib.sha1(table.encode()).hexdigest()[:8], REPO / "scripts/sdfbase/full22/misalign/make_misalign_manifest.py")
        mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    finally:
        if prev is None:
            os.environ.pop("SISTEER_MODELS_TABLE", None)
        else:
            os.environ["SISTEER_MODELS_TABLE"] = prev
    return mod


def test_the_one_predicate_and_its_mirrors_agree(phased_table, monkeypatch):
    import make_misalign_manifest as MM
    assert FR.STEERED_WAVE_VIEW_RE.pattern == MT.STEERED_WAVE_VIEW_RE.pattern == MM.STEERED_WAVE_VIEW_RE.pattern == r"^s[0-9]+rl?$"   # S2RL: + s<N>rl
    for v in ("s2r", "s3r", "s9r", "s2rl"):
        assert FR.steered_wave_view(v) and MT.steered_wave_view(v)
    for v in ("u0", "s2p", "s2rp", "main", "it", "sr", "s2r_x"):
        assert not FR.steered_wave_view(v) and not MT.steered_wave_view(v, "qwen35_4b"), v
    # the SINGLE-PHASE recipe: the demo declares no u0, so NONE of its views is a steered wave — each carries its own
    # unsteered legs; its baseline wave is s1 (the one the gate's inventory freezes from); the `s<N>r` name still rules by itself
    assert MT.single_phase(KEY) and MT.baseline_view(KEY) == "battery" and KEY in FR.single_phase_targets() and KEY in MM.SINGLE_PHASE_KEYS
    assert MT.views(KEY) == list(DEMO_VIEWS)
    for v in DEMO_VIEWS:
        assert not FR.steered_wave_view(v, KEY) and not MT.steered_wave_view(v, KEY) and not MM.is_steered_wave_target(f"{KEY}_{v}"), v
    assert FR.steered_wave_view("s2r", KEY) and MT.steered_wave_view("s2r", KEY) and not MM.is_front_door(f"{KEY}_s2r")   # by name; the demo declares no s2r (test_s2r_phase_view_grammar adds one)
    # For a front-door target that RUNS IN PHASES (its table declares u0) every view except u0 is a steered wave (the LoRA, residual and
    # dose waves) — the SYNTHETIC phased row, through every copy of the predicate (the misalign generator reads the table at import: a fresh copy)
    monkeypatch.setenv("SISTEER_MODELS_TABLE", phased_table)
    MMP = _misalign_manifest_module(phased_table)
    assert not MT.single_phase(PHASED) and MT.baseline_view(PHASED) == "u0" and PHASED in FR.front_door_targets(phased_table) and PHASED not in FR.single_phase_targets(phased_table)
    assert PHASED not in MMP.SINGLE_PHASE_KEYS and KEY in MMP.SINGLE_PHASE_KEYS and f"{PHASED}_u0" in MMP.FRONT_DOOR and f"{PHASED}_s2" in MMP.FRONT_DOOR
    for v in (*PHASED_STEERED_VIEWS, "s2r", "s3r", "d0p75_relul"):
        assert FR.steered_wave_view(v, PHASED, table_path=phased_table) and MT.steered_wave_view(v, PHASED) and MT.steered_wave_view(v), v
    for v in PHASED_STEERED_VIEWS:
        assert MMP.is_steered_wave_target(f"{PHASED}_{v}"), v
    assert not FR.steered_wave_view("u0", PHASED, table_path=phased_table) and not MT.steered_wave_view("u0", PHASED) and not MT.steered_wave_view("u0") and not MMP.is_steered_wave_target(f"{PHASED}_u0")
    for v in DEMO_VIEWS:                                              # the demo row of the same table is still single-phase beside it
        assert not FR.steered_wave_view(v, KEY, table_path=phased_table) and not MT.steered_wave_view(v, KEY) and not MMP.is_steered_wave_target(f"{KEY}_{v}"), v
    monkeypatch.delenv("SISTEER_MODELS_TABLE")
    # a PAPER target's roster views keep their anchor whatever their names (s1 / d0p1 collide with the phase grammar: the rule is target-aware)
    for v in ("s1", "s2", "d0p1", "d1", "main", "it", "s1p"):
        assert not FR.steered_wave_view(v, "qwen35_4b") and not FR.steered_wave_view(v) and not MT.steered_wave_view(v, "qwen35_4b"), v
    assert KEY in FR.front_door_targets() and "qwen35_4b" not in FR.front_door_targets() and FR.front_door_targets(REPO / "nonexistent.yaml") == frozenset()
    assert MT.UNSTEERED_LEGS == UNSTEERED


def test_the_universe_is_none_plus_the_served_cells_for_an_ordinary_view_and_exactly_the_served_cells_for_a_steered_wave_view(phased_table, monkeypatch):
    # the SINGLE-PHASE demo: every view keeps the same-wave `none` anchor — the one wave s1 (every arm beside its own unsteered legs) and both dose rungs
    assert FR.view_cell_universe(ROSTER, KEY, "battery") == ["none", *DEMO_S1_ROSTER_CELLS]
    assert FR.view_cell_universe(ROSTER, KEY, "d0p75") == ["none", DIFF_LORA]
    assert FR.view_cell_universe(ROSTER, KEY, "d0p75_relu") == ["none", DIFF]                            # the one composed cell (the paired write's 0.75 copy) beside its anchor
    assert FR.view_cell_universe(ROSTER, "qwen35_4b", "main")[0] == "none"                              # a paper target: as before
    # The steered-wave rule on the SYNTHETIC phased row (the block exactly as 05 --step roster prints it over the tmp table): u0 = none + the prompted
    # instructions; every steered view EXACTLY the block's cells — the table's `none` is the u0 anchor and is printed into no steered block
    monkeypatch.setenv("SISTEER_MODELS_TABLE", phased_table)
    phased = yaml.load("targets:\n" + MT.roster_block(PHASED, ["u0", *PHASED_STEERED_VIEWS], phased_table), Loader=FR._StrictRosterLoader)
    assert FR.view_cell_universe(phased, PHASED, "u0") == ["none", "prompted__safety_refusal", "prompted__si_constitution"]
    for view in PHASED_STEERED_VIEWS:
        cells = list(phased["targets"][PHASED]["views"][view]["cells"])
        assert cells and "none" not in cells and FR.view_cell_universe(phased, PHASED, view) == cells, view
    assert FR.view_cell_universe(phased, PHASED, "d0p75_relu") == [DIFF]                     # the composed paired write (demo-import M2), no anchor on a phased dose view
    # a phased front-door LoRA / dose view enumerates exactly its cells (synthetic roster blocks); the demo's and a paper target's s1 keep their anchor
    fd = {"targets": {PHASED: {"views": {"s1": {"cells": {"control_lora": "base__ctl__lora_r4_ce", "constitution_lora": "base__mix__lora_r4_ce", "difference_lora": "derived__mix__lora_r4_ce_minus_ctl"}},
                                           "d0p75": {"cells": {"difference_lora": "d0p75__mix__lora_r4_ce_minus_ctl"}}, "u0": {"cells": {"prompted__safety_refusal": None}}}},
                      KEY: {"views": {"battery": {"cells": {"control_lora": "base__ctl__lora_r4_ce", "prompted__safety_refusal": None}}}},
                      "qwen35_4b": {"views": {"s1": {"cells": {"control_lora": "base__ctl__lora_r4_ce"}}}}}}
    assert FR.view_cell_universe(fd, PHASED, "s1") == ["control_lora", "constitution_lora", "difference_lora"]
    assert FR.view_cell_universe(fd, PHASED, "d0p75") == ["difference_lora"] and FR.view_cell_universe(fd, PHASED, "u0") == ["none", "prompted__safety_refusal"]
    assert FR.view_cell_universe(fd, KEY, "battery") == ["none", "control_lora", "prompted__safety_refusal"]      # the single-phase demo's battery: its own anchor
    assert FR.view_cell_universe(fd, "qwen35_4b", "s1") == ["none", "control_lora"]                       # the paper target's s1 keeps its anchor
    monkeypatch.delenv("SISTEER_MODELS_TABLE")
    bad = {"targets": {"m": {"views": {"s2r": {"cells": {"none": "base__none", "x__psr_ce_all": "base__x__psr_ce_all"}}}}}}   # a listed none (with a staged id: the roster grammar's other refusal would fire first on null)
    with pytest.raises(ValueError, match="a steered-wave view lists `none` in the roster"):
        FR.view_cell_universe(bad, "m", "s2r")


def test_default_planning_refuses_a_steered_wave_view_whose_artefacts_carry_an_unsteered_leg(tmp_path, monkeypatch):
    """The refusal reads the artefacts the launchers read: the wave config (06's view_cells_of), the Petri manifest, the GSM8K matrix. A
    synthetic roster + config under tmp (F22_ROSTER); the front door's views, which the production roster does not carry, have no artefact
    to witness and pass the walk."""
    for view in DEMO_VIEWS:
        ST.refuse_unsteered_legs_on_steered_wave_view(KEY, view)                                      # the single-phase demo: never a steered wave, the walk is a no-op
    monkeypatch.setenv("SISTEER_MODELS_TABLE", _phased_table(tmp_path / "phased"))
    for view in PHASED_STEERED_VIEWS:
        ST.refuse_unsteered_legs_on_steered_wave_view(PHASED, view)                                   # a phased row with no block in the roster: nothing to witness
    monkeypatch.delenv("SISTEER_MODELS_TABLE")
    root = tmp_path / "repo"; (root / "configs" / "capeval" / "manifests").mkdir(parents=True)
    cfg = root / "configs" / "sdfbase_full22_m_s2r.yaml"; cfg.write_text(yaml.safe_dump({"explicit_cells": ["none", "x__psr_ce_all"]}))
    roster = root / "configs" / "sdfbase_full22_roster.yaml"
    roster.write_text(yaml.safe_dump({"targets": {"m": {"views": {"s2r": {"config": str(cfg), "manifest": str(root / "configs/capeval/manifests/petri_full22_m_s2r.yaml"),
                                                                           "cells": {"x__psr_ce_all": "base__x__psr_ce_all"}}}}}}))
    with pytest.raises(ValueError, match=r"lists the unsteered leg\(s\) \['none'\] on a steered-wave view"):
        ST.view_cells_of("m", "s2r", roster_path=str(roster))
    with pytest.raises(ValueError, match="unsteered leg"):
        ST.refuse_unsteered_legs_on_steered_wave_view("m", "s2r", roster_path=str(roster))
    # the config clean, the manifest carrying the controls twin: refused by name too
    cfg.write_text(yaml.safe_dump({"explicit_cells": ["x__psr_ce_all"]}))
    (root / "configs/capeval/manifests/petri_full22_m_s2r.yaml").write_text(yaml.safe_dump({"petri": {"cells": ["si_x__psr_ce_all", "none__lora0"]}}))
    with pytest.raises(ValueError, match=r"Petri manifest .* lists the unsteered leg\(s\) \['none__lora0'\]"):
        ST.refuse_unsteered_legs_on_steered_wave_view("m", "s2r", roster_path=str(roster))
    # an ordinary view is untouched by the walk (the single-phase demo's battery lists none by design — its own unsteered legs)
    ST.refuse_unsteered_legs_on_steered_wave_view(KEY, "battery")
    # unit_plan runs the walk for a steered-wave view before any row (the knob is the tests' escape for synthetic trees, never set in production)
    monkeypatch.setenv("F22_ROSTER", str(roster)); monkeypatch.delenv("SISTEER_PLAN_SKIP_ARTEFACT_CHECK", raising=False)
    tbl = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text())
    models = tbl["models"] if "models" in tbl else tbl
    entry = models.get(KEY) or next(v[KEY] for v in models.values() if isinstance(v, dict) and KEY in v)
    import copy; t2 = copy.deepcopy(tbl)
    m2 = (t2["models"] if "models" in t2 else t2)
    (m2.get("m") if m2.get("m") else m2.setdefault("m", copy.deepcopy(entry)))
    bad_table = tmp_path / "models.yaml"; bad_table.write_text(yaml.safe_dump(t2, sort_keys=False))
    with pytest.raises(ValueError, match="unsteered leg"):
        ST.unit_plan("m", "s2r", table=str(bad_table), families=["static"], sha="a" * 40, clusters=["cluster_a"])
    monkeypatch.delenv("F22_ROSTER")
    src = (REPO / "reproduce" / "lib" / "stage.py").read_text()
    i_m, i_call, i_fams = src.index('    m = M.model(model_key, table)\n    model_id = m["serve"]["checkpoint"]["model_id"]'), src.index("refuse_unsteered_legs_on_steered_wave_view(model_key, view, table=table)"), src.index("    fams = list(families) if families else list(FAMILIES)")
    assert i_m < i_call < i_fams                                                                        # before any row is planned


def test_served_but_unlisted_none_cannot_be_scored_registered_or_counted(tmp_path):
    """The base route is SERVED implicitly and nothing downstream picks it up from the served list."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase"))
    import mt_serving_up as MS
    # (1) the standup serves the base route first when the config lacks it — and lists it nowhere else
    assert MS.select_cells([DIFF], None) == ["none", DIFF] and MS.select_cells(["constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all"], []) == ["none", "constv2ctl2-full__psr_ce_all", "constv2-mix-full__psr_ce_all"]
    assert MS.select_cells(["none", "a"], None) == ["none", "a"]                                        # an ordinary config: byte-identical
    # (2) the sampled drivers iterate the CONFIG's cells (CELLS from explicit_cells), so none is never a driver cell, a registered pair or a store
    sb = (REPO / "scripts/sdfbase/full22/sampled/target_sampled.sbatch").read_text()
    assert 'if [ -n "$CELLS_ARG" ]; then IFS=\',\' read -ra CELLS <<< "$CELLS_ARG"; else CELLS=("${ALL_CELLS[@]}"); fi' in sb
    assert 'for c in "${CELLS[@]}"; do case " ${ALL_CELLS[*]} " in *" $c "*) ;; *) f22_fatal "cell $c is not in $CFG\'s explicit_cells' in sb
    assert 'for pc in "${!NBASE[@]}"; do if [ "${NBASE[$pc]}" -gt 0 ]; then' in sb or 'f22s_register_gated_pairs' in sb or 'for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do' in sb
    common = (REPO / "scripts/sdfbase/full22/common.sh").read_text()
    assert 'mapfile -t ALL_CELLS < <("$py" - "$CFG" <<\'PY\'\nimport sys, yaml; print("\\n".join(yaml.safe_load(open(sys.argv[1]))["explicit_cells"]))' in common
    assert '[ "${#ALL_CELLS[@]}" -ge 1 ] || f22_fatal "$CFG lists no cell' in common                   # one cell is a complete steered-wave view
    # (3) the misalign runner's cells.json: the served none leg (and its controls twin) is UNMEASURED under the config's policy
    # the misalign config's policy block the generator renders for a PHASED front-door steered view (make_misalign_manifest: lane
    # front_door_steered_wave, measure_none false, the view's measured cells, the u0 wave's bare run as the pairing anchor) — the synthetic phased
    # row's s2 under tmp; the single-phase demo renders no such lane (its views measure their own none)
    mis = tmp_path / f"misalign_full22_{PHASED}_s2.yaml"
    mis.write_text(yaml.safe_dump({"explicit_cells": [DIFF], "misalign_policy": {"lane": "front_door_steered_wave", "view": "s2", "measure_none": False,
                                                                                 "measured_cells": [DIFF], "pairing_anchor": f"misalign_full22_{PHASED}_u0/agentic/none"}}))
    pol = MC.read_policy(mis)
    assert pol["measure_none"] is False and pol["measured_cells"] == [DIFF] and pol["lane"] == "front_door_steered_wave"
    leg = lambda cid, vm, kind: {"cell_id": cid, "virtual_model_name": vm, "serving_identity": f"identity-{cid}", "intervention_hash": "0" * 16, "kind": kind}  # noqa: E731
    served = {"canonical": [leg("none", "petri__none", "none"), leg("si_constv2_mix_full__psr_ce_all_minus_ctl", "petri__diff", "psr")],
              "controls": [leg("none__lora0", "petri__none__lora0", "none")]}
    ep = {"served_models": ["petri__none", "petri__diff", "petri__none__lora0"]}
    cells_doc, full = MC.build_cells(served, ep, target=f"{PHASED}_s2", wave_sha="a" * 40, model_key=PHASED, wave_view="s2", policy=pol)
    measured = [c["cell_id"] for c in (cells_doc["cells"] if isinstance(cells_doc, dict) else cells_doc)]
    assert measured == ["si_constv2_mix_full__psr_ce_all_minus_ctl"], measured
    legs = {l["cell_id"]: l for l in full["legs"]}
    assert legs["none"]["measured"] is False and legs["none__lora0"]["measured"] is False and legs["si_constv2_mix_full__psr_ce_all_minus_ctl"]["measured"] is True
    # (4) the Petri frozen set is the explicit declaration, with the controls twin only when `none` is among the declared
    pr = (REPO / "scripts" / "petri_run.py").read_text()
    assert re.compile(r"_s[0-9]+rl?$").pattern in pr and "a steered-only view drives no none and no none__lora0" in pr   # S2RLFIX5-PETRI: the pinned text moves with the grammar
    mp = (REPO / "scripts/sdfbase/full22/misalign/misalign_proxy.sbatch").read_text()
    assert '[ "${#ALL_CELLS[@]}" -ge 1 ] || f22_fatal "$CFG lists no cell"' in mp
