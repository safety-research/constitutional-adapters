"""ctl-neg1 (2026-09-25): the batch-64 seed-replicate CONTROL parents at strength -1, served through a LANE-PRIVATE roster.

The production-roster tests never open `configs/sdfbase_full22_roster_ctl_neg1_eb64.yaml` — invisibility to the
campaign's tests is not the same as being tested — so the lane's own contract is pinned here:

* six views `dm1_ctllora64s<k>` / `dm1_ctlrelu64s<k>` (k in 1..3), dose -1, the family's unit, `object_family:` spelling the
  name's segment (seed included) while the generator reads the SEEDLESS family (`family_of_segment`);
* every view binds exactly ONE cell -- the constv2ctl2 control of its class -- in the pin form against
  `configs/ctl_neg1_eb64_pins.json`, whose entry carries the builder's facts (bank / source / kind / served_layer / sidecar) in
  the shape `make_bank_view.plan_pin_entry` resolves;
* the rendered wave configs are the generator's (byte-identical), with `explicit_cells = [none, <cell>]` and the body's caps;
* the sampled leg renders the six views under the lane roster ONLY, at seed_base = k with ONE trial (the s<k> views' rule),
  and the production scope is untouched;
* the two new object families admit the control cells ONLY, and the production roster names none of the six views (common.sh's
  non-production-roster collision guard).
"""
from __future__ import annotations

import copy
import json
import re
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22", REPO / "scripts" / "sdfbase" / "full22" / "sampled",
           REPO / "scripts" / "sdfbase" / "seedrep", REPO / "src", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import make_full22_configs as gen  # noqa: E402
import make_sampled_configs as msc  # noqa: E402
import eval_seed as es  # noqa: E402
import f22_roster as fr  # noqa: E402
from full22_wave_configs import EXCLUDED_TABLES  # noqa: E402

LANE_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster_ctl_neg1_eb64.yaml"
LANE_PINS_PATH = REPO / "configs/ctl_neg1_eb64_pins.json"
PROD_ROSTER = yaml.safe_load((REPO / "configs/sdfbase_full22_roster.yaml").read_text())
LANE_ROSTER = yaml.safe_load(LANE_ROSTER_PATH.read_text())
LANE_PINS = json.loads(LANE_PINS_PATH.read_text())
TARGET = "qwen35_4b"
CLASSES = {"ctllora64": ("constv2ctl2-full__lora_r4_ce", "lora_write_basis_multiplier", "lora_all", 3),
           "ctlrelu64": ("constv2ctl2-full__psr_ce_all", "psr_direction_norm_multiplier", "psr", 0)}
VIEWS = tuple(f"dm1_{cls}s{k}" for k in (1, 2, 3) for cls in CLASSES)


def _view_parts(view: str):
    m = re.fullmatch(r"dm1_(ctllora64|ctlrelu64)s([123])", view)
    assert m, view
    return m.group(1), int(m.group(2))


def test_the_lane_roster_carries_the_production_contract_and_exactly_the_six_views():
    assert {k: v for k, v in LANE_ROSTER.items() if k != "targets"} == {k: v for k, v in PROD_ROSTER.items() if k != "targets"}
    assert list(LANE_ROSTER["targets"]) == [TARGET]
    t, pt = LANE_ROSTER["targets"][TARGET], PROD_ROSTER["targets"][TARGET]
    assert (t["model"], t["model_revision"], t["body_config"]) == (pt["model"], pt["model_revision"], pt["body_config"])
    assert tuple(t["views"]) == VIEWS
    # none of the six is a production view: common.sh f22_bank_view refuses a non-production roster that names one
    assert not set(VIEWS) & set(pt["views"])
    assert LANE_ROSTER_PATH.name in EXCLUDED_TABLES          # a roster under the wave-config glob, named as a table


@pytest.mark.parametrize("view", VIEWS)
def test_each_view_is_a_dose_minus_one_view_of_its_seedless_family_binding_one_pinned_control_cell(view):
    cls, k = _view_parts(view)
    cell, unit, kind, served_layer = CLASSES[cls]
    v = LANE_ROSTER["targets"][TARGET]["views"][view]
    assert v["dose"] == -1 and v["dose_unit"] == unit and v["object_family"] == f"{cls}s{k}"
    assert gen.view_dose(LANE_ROSTER, TARGET, view) == -1.0
    assert gen.view_object_family(LANE_ROSTER, TARGET, view) == cls == gen.family_of_segment(f"{cls}s{k}")
    assert gen.DOSE_UNITS[cls] == unit
    assert fr.view_legs(LANE_ROSTER, TARGET, view) == ("static", "controls", "attacks")
    assert set(fr.view_legs_omitted(LANE_ROSTER, TARGET, view)) == {"bon", "capability", "petri"}
    assert fr.view_controls(LANE_ROSTER, TARGET, view) == ("xstest", "or_bench_hard_1k")
    # ONE cell, the pin form, against the lane pins file
    assert list(v["cells"]) == [cell]
    mapping = v["cells"][cell]
    assert fr.is_pin_form(mapping) and mapping["pins"] == "configs/ctl_neg1_eb64_pins.json"
    sid = f"dm1_eb64s{k}__{cell}"
    assert mapping["pin"] == f"{TARGET}.{sid}"
    pin = LANE_PINS[TARGET][sid]
    assert set(pin) == {"object_sha256", "kind", "source", "identity_sidecar", "bank", "staged_id", "served_layer"}
    assert re.fullmatch(r"[0-9a-f]{64}", pin["object_sha256"])
    assert pin["staged_id"] == sid and pin["kind"] == kind and pin["served_layer"] == served_layer
    assert pin["identity_sidecar"] == "train_budget_meta.json" and pin["bank"] == "ctl_neg1_eb64"
    # the builder's shape: `<bank>/<target>/cells_view/<view>/<cell>`, relative to exp/objects, bank = the first component
    assert pin["source"] == f"ctl_neg1_eb64/{TARGET}/cells_view/{view}/{cell}"
    parts = Path(pin["source"]).parts
    assert parts[0] == pin["bank"] and parts[-3] == "cells_view" and parts[-2] == view and parts[-1] == cell
    # the generator's own pin reading agrees
    assert gen.pin_form_facts(TARGET, cell, mapping) == (sid, "ctl_neg1_eb64", pin["object_sha256"], mapping["pin"])


def test_the_six_pins_are_six_distinct_objects():
    shas = [p["object_sha256"] for p in LANE_PINS[TARGET].values()]
    assert len(LANE_PINS[TARGET]) == 6 == len(set(shas))


@pytest.mark.parametrize("view", VIEWS)
def test_the_rendered_wave_config_and_manifest_are_the_generators_and_serve_none_plus_the_one_cell(view):
    cls, k = _view_parts(view)
    cell = CLASSES[cls][0]
    v = LANE_ROSTER["targets"][TARGET]["views"][view]
    cfg_path, man_path = REPO / v["config"], REPO / v["manifest"]
    assert cfg_path.name == f"sdfbase_full22_{TARGET}_{view}.yaml" and man_path.name == f"petri_full22_{TARGET}_{view}.yaml"
    pins = gen.load_pins()
    assert cfg_path.read_text() == gen.render_config(LANE_ROSTER, pins, TARGET, view, roster_path=LANE_ROSTER_PATH)
    assert man_path.read_text() == gen.render_manifest(LANE_ROSTER, TARGET, view, roster_path=LANE_ROSTER_PATH)
    cfg = yaml.safe_load(cfg_path.read_text())
    assert cfg["explicit_cells"] == ["none", cell]
    assert cfg["bon"]["max_new_tokens"] == gen.BON_MAX_NEW_TOKENS == 512 and cfg["bon"]["climb_cap"] == LANE_ROSTER["bon_climb_cap"]
    assert cfg["attacks"]["target_max_new_tokens"] == LANE_ROSTER["target_max_new_tokens"] == 4096
    head = cfg_path.read_text()
    assert f"from\n# configs/{LANE_ROSTER_PATH.name} (target {TARGET}, view {view})" in head
    assert f"<- dm1_eb64s{k}__{cell}  (bank ctl_neg1_eb64, THROUGH the pin configs/ctl_neg1_eb64_pins.json" in head
    assert LANE_PINS[TARGET][f"dm1_eb64s{k}__{cell}"]["object_sha256"] in head


def test_the_sampled_leg_renders_the_six_views_under_the_lane_roster_only_at_their_training_seed_one_trial(monkeypatch):
    presets = msc.load_presets()
    # the production scope is untouched: not in SAMPLED_VIEWS, not in scope, not among the production TAGs
    assert not set(VIEWS) & set(msc.SAMPLED_VIEWS[TARGET])
    assert all(not msc.in_scope(TARGET, v) for v in VIEWS) and not set(msc.lane_replicate_tags()) & set(msc.sampled_tags())
    # the seed rule rows: literal data in the rule table AND the data file, at the training seed, one trial
    assert msc.SEEDED_LANE_VIEWS == {TARGET: VIEWS}
    for view in VIEWS:
        _cls, k = _view_parts(view)
        tag = msc.tag_of(TARGET, view)
        assert msc.seed_of_lane_view(view) == k
        assert msc.SEED_BASE_RULE_BY_TAG[tag] == presets["seed_base_by_view"][tag] == es.seed_base(k) == k != presets["seed_base"]
        assert msc.seed_base_of_tag(presets, tag) == k and msc.n_trials_of_tag(presets, tag) == presets["n_trials_by_view"][tag] == 1
    # under the LANE roster the scope is exactly the six views and the committed sampled configs are the generator's
    monkeypatch.setattr(msc, "_ROSTER_PATH", LANE_ROSTER_PATH.resolve())
    assert msc._private() and msc.scope_views(TARGET) == VIEWS and all(msc.scope_views(k) == () for k in msc.SAMPLED_VIEWS if k != TARGET)
    assert all(msc.in_scope(TARGET, v) for v in VIEWS) and not msc.in_scope(TARGET, "s1") and not msc.in_scope(TARGET, "main")
    msc.check_pins(presets, msc.load_roster())
    outs = msc.outputs(presets)
    assert len(outs) == 18 == 6 * 1 * 3 and {t for _p, t, *_ in outs} == set(msc.lane_replicate_tags())
    for path, tag, trial, esc in outs:
        assert trial == 0 and path.is_file(), path.name
        assert path.read_text() == msc.render(presets, tag, trial, esc)
        sd = yaml.safe_load(path.read_text())["sampled_decode"]
        _cls, k = _view_parts(tag.split(f"{TARGET}_", 1)[1])
        assert sd["seed_base"] == k and sd["n_trials"] == 1 and sd["trial"] == 0 and sd["arm"] == "vendor_recommended"
    assert msc.stale_outputs(presets) == []
    # the lane roster's seeded views need their rows: a data file without them refuses by name
    stripped = copy.deepcopy(presets)
    stripped["n_trials_by_view"] = {t: n for t, n in presets["n_trials_by_view"].items() if t not in msc.lane_replicate_tags()}
    with pytest.raises(SystemExit, match="seeded lane view"):
        msc.check_pins(stripped, msc.load_roster())


def test_the_control_families_admit_the_control_cells_only():
    ctl_lora, ctl_relu = CLASSES["ctllora64"][0], CLASSES["ctlrelu64"][0]
    for bad in ("constv2-mix-full__lora_r4_ce", "constv2-mix-full__lora_r4_ce_minus_ctl", ctl_relu):
        assert not gen.OBJECT_FAMILIES["ctllora64"][0](bad), bad
    for bad in ("constv2-mix-full__psr_ce_all", "constv2-mix-full__psr_ce_all_minus_ctl_distilled", ctl_lora):
        assert not gen.OBJECT_FAMILIES["ctlrelu64"][0](bad), bad
    assert gen.OBJECT_FAMILIES["ctllora64"][0](ctl_lora) and gen.OBJECT_FAMILIES["ctlrelu64"][0](ctl_relu)
    # a constitution cell smuggled onto a control-family view refuses by name
    r = copy.deepcopy(LANE_ROSTER)
    r["targets"][TARGET]["views"]["dm1_ctllora64s1"]["cells"] = {"constv2-mix-full__lora_r4_ce": {"pins": "configs/ctl_neg1_eb64_pins.json",
                                                                                                     "pin": f"{TARGET}.x"}}
    with pytest.raises(SystemExit, match="object family `ctllora64` but serves `constv2-mix-full__lora_r4_ce`"):
        gen.view_object_family(r, TARGET, "dm1_ctllora64s1")
    # the unit is the family's: the wrong axis on a control view refuses
    r = copy.deepcopy(LANE_ROSTER)
    r["targets"][TARGET]["views"]["dm1_ctlrelu64s2"]["dose_unit"] = "lora_write_basis_multiplier"
    with pytest.raises(SystemExit, match="a dose view declares `dose_unit: psr_direction_norm_multiplier`"):
        gen.view_dose(r, TARGET, "dm1_ctlrelu64s2")
    # the family segment must spell the seed the object_family key declares
    r = copy.deepcopy(LANE_ROSTER)
    r["targets"][TARGET]["views"]["dm1_ctlrelu64s2"]["object_family"] = "ctlrelu64"
    with pytest.raises(SystemExit, match="the two must agree"):
        gen.view_object_family(r, TARGET, "dm1_ctlrelu64s2")
