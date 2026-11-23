"""The non-production roster ``configs/sdfbase_full22_roster_ctlstop_eb64.yaml``: the single-turn families, best-of-N, the capability and
over-refusal controls and the multi-turn attacks on the batch-64 control objects fit with early stopping on the control pool and on their
constitution-minus-control differences.

The production-roster tests never open this file, so its contract is pinned here (no pins needed: the pins file and the rendered
configs land with the objects, and their own test pins them):

* every top-level key but ``targets`` is the production roster's as of the roster's landing, pinned by the sha256 of its canonical JSON
  (so a later change to the production roster does not redden this file; a change here is deliberate), and the target's model, revision
  and body config are pinned likewise;
* exactly three views ``ctlstop64s<k>`` (k in 1..3), none a production view (common.sh refuses a non-production roster that names one);
* each view runs every leg but Petri (static, bon, capability, controls, attacks), omits Petri with a reason, declares the per-view
  best-of-N cap ``caps: {bon: 4096}`` (the seed views' native best-of-75 pass) and no other cap, and carries no dose key (the operating
  point);
* each view binds the same four cells, the production spellings, every one in the pin form against ``configs/ctlstop_eb64_pins.json``
  with the pin ``qwen35_4b.ctlstop64s<k>__<cell>``; its config / manifest paths follow the generator's naming;
* ``roster_view.py ... has_leg`` admits each view's five legs under this roster and refuses Petri;
* the model key a job registers under (``qwen35_4b_ctlstop64s<k>``) resolves through ``legacy_spellings.yaml``;
* the wave-config enumerator names this file as a table."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22", REPO / "src", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import f22_roster as fr  # noqa: E402
import make_full22_configs as gen  # noqa: E402
from full22_wave_configs import EXCLUDED_TABLES  # noqa: E402
from sisteer.registry import jobplan as JP  # noqa: E402

LANE_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster_ctlstop_eb64.yaml"
LANE_PINS_REL = "configs/ctlstop_eb64_pins.json"
PROD_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster.yaml"
PROD_ROSTER = yaml.safe_load(PROD_ROSTER_PATH.read_text())
LANE_ROSTER = yaml.safe_load(LANE_ROSTER_PATH.read_text())
TARGET = "qwen35_4b"
VIEWS = ("ctlstop64s1", "ctlstop64s2", "ctlstop64s3")
#: the control LoRA, the control ReLU, the LoRA difference, the composed ReLU difference (paired write)
CELLS = ("constv2ctl2-full__lora_r4_ce", "constv2ctl2-full__psr_ce_all",
         "constv2-mix-full__lora_r4_ce_minus_ctl", "constv2-mix-full__psr_ce_all_minus_ctl")
#: the legs each view runs, in the roster grammar's launch order (f22_roster.LEGS minus Petri), and the one it omits
LEGS = ("static", "bon", "capability", "controls", "attacks")
OMITTED = ("petri",)
#: the per-view best-of-N cap: the native best-of-75 pass at 4,096, as on the seed views s<k> (make_full22_configs.view_declared_bon_cap)
VIEW_CAPS = {"bon": 4096}
#: sha256 of the canonical JSON (sorted keys, compact separators) of the roster's top level minus ``targets``: the production roster's
#: serving contract when this roster was written (serving_impl_commit 94e294c4…, caps, controls window, bon climb, attacker flags, lane flags)
TOP_LEVEL_SHA256 = "12de193b1412a428271407d856c0fc18dffb9356adbd63a2839e9e0e1d37ee15"


def _top_level_sha256(roster: dict) -> str:
    top = {k: v for k, v in roster.items() if k != "targets"}
    return hashlib.sha256(json.dumps(top, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def test_the_roster_carries_the_production_contract_and_exactly_the_three_views():
    assert _top_level_sha256(LANE_ROSTER) == TOP_LEVEL_SHA256
    assert LANE_ROSTER["serving_impl_commit"] == "94e294c445dc5eec6ebe0ce3a3e25553004cd4b0" and LANE_ROSTER["target_max_new_tokens"] == 4096
    assert list(LANE_ROSTER["targets"]) == [TARGET]
    t, pt = LANE_ROSTER["targets"][TARGET], PROD_ROSTER["targets"][TARGET]
    assert set(t) == {"model", "model_revision", "body_config", "views"}
    assert (t["model"], t["model_revision"], t["body_config"]) == ("Qwen/Qwen3.5-4B", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
                                                                   "configs/sdfbase_transfer_qwen35_4b_r3.yaml")
    assert tuple(t["views"]) == VIEWS
    assert not set(VIEWS) & set(pt["views"])                         # none is a production view (common.sh f22_bank_view refuses one)
    assert "caps" not in t                                            # no TARGET caps table: a view's own caps.bon is admitted only then
    assert LANE_ROSTER_PATH.name in EXCLUDED_TABLES                   # a roster under the wave-config glob, named as a table


def test_the_cells_are_production_spellings():
    production_cells = {c for v in PROD_ROSTER["targets"][TARGET]["views"].values() for c in (v.get("cells") or {})}
    assert set(CELLS) <= production_cells, sorted(set(CELLS) - production_cells)


@pytest.mark.parametrize("view", VIEWS)
def test_each_view_runs_every_leg_but_petri_at_the_operating_point_and_binds_four_pinned_cells(view):
    k = int(view[-1])
    v = LANE_ROSTER["targets"][TARGET]["views"][view]
    assert set(v) == {"legs", "legs_omitted", "caps", "config", "manifest", "cells"}   # no dose / dose_unit / object_family, no Petri knobs
    assert fr.view_legs(LANE_ROSTER, TARGET, view) == LEGS
    assert v["caps"] == VIEW_CAPS and gen.view_declared_bon_cap(LANE_ROSTER, TARGET, view) == 4096
    omitted = fr.view_legs_omitted(LANE_ROSTER, TARGET, view)
    assert set(omitted) == set(OMITTED) and all(isinstance(why, str) and why.strip() for why in omitted.values())
    assert gen.view_dose(LANE_ROSTER, TARGET, view) is None
    assert not fr.steered_wave_view(view, TARGET)
    assert fr.view_cell_universe(LANE_ROSTER, TARGET, view) == ["none", *CELLS]
    assert v["config"] == f"configs/sdfbase_full22_{TARGET}_{view}.yaml"
    assert v["manifest"] == f"configs/capeval/manifests/petri_full22_{TARGET}_{view}.yaml"
    assert tuple(v["cells"]) == CELLS
    for cell, mapping in v["cells"].items():
        assert fr.is_pin_form(mapping) and mapping == {"pins": LANE_PINS_REL, "pin": f"{TARGET}.ctlstop64s{k}__{cell}"}, (view, cell, mapping)
    assert fr.staged_cells(LANE_ROSTER, TARGET, view) == dict(v["cells"])


def test_the_twelve_pins_are_distinct_and_carry_their_view_and_cell():
    pins = [m["pin"] for v in VIEWS for m in LANE_ROSTER["targets"][TARGET]["views"][v]["cells"].values()]
    assert len(pins) == 12 == len(set(pins))
    assert all(p.startswith(f"{TARGET}.ctlstop64s") for p in pins)


@pytest.mark.parametrize("view", VIEWS)
def test_roster_view_admits_every_leg_but_petri(view):
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    tool = REPO / "scripts/sdfbase/full22/roster_view.py"
    for leg in LEGS:
        ok = subprocess.run([sys.executable, str(tool), TARGET, view, "has_leg", leg, "--roster", str(LANE_ROSTER_PATH)],
                            capture_output=True, text=True, env=env, cwd=REPO, timeout=120)
        assert ok.returncode == 0, (leg, ok.stderr)
    for leg in OMITTED:
        r = subprocess.run([sys.executable, str(tool), TARGET, view, "has_leg", leg, "--roster", str(LANE_ROSTER_PATH)],
                           capture_output=True, text=True, env=env, cwd=REPO, timeout=120)
        assert r.returncode != 0, (leg, r.stdout, r.stderr)
    count = REPO / "scripts/sdfbase/full22/roster_cell_count.py"
    assert subprocess.run([sys.executable, str(count), TARGET, view, "--roster", str(LANE_ROSTER_PATH)], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=120).stdout.strip() == "4"
    # the production roster names no such view: the collision guard of common.sh f22_bank_view stays quiet
    assert subprocess.run([sys.executable, str(count), TARGET, view, "--roster", str(PROD_ROSTER_PATH)], capture_output=True, text=True,
                          env=env, cwd=REPO, timeout=120).returncode != 0


@pytest.mark.parametrize("view", VIEWS)
def test_the_registration_model_key_of_each_view_is_a_legacy_spelling(view):
    rows = yaml.safe_load((REPO / "configs/registry/legacy_spellings.yaml").read_text())["model_keys"]
    assert JP.model_for_key(rows, f"{TARGET}_{view}") == ("qwen35_4b_it", "base_to_instruct")
    row = next(r for r in rows if r["legacy"] == f"{TARGET}_{view}")
    assert row["fit_model_id"] == "qwen35_4b_base" and "fit_replicate_label" not in row
