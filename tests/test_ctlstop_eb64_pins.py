"""The pins of the non-production roster ``configs/sdfbase_full22_roster_ctlstop_eb64.yaml``, and the configs rendered from it.

``configs/ctlstop_eb64_pins.json`` names the twelve objects the roster's three views serve: per seed k in 1..3 the control LoRA, the
control ReLU, the LoRA difference and the composed ReLU difference, each a staged copy ``ctlstop64s<k>__<cell>`` in the bank
``ctlstop_eb64``. Pinned here:

* every roster pin resolves in the pins file to an entry with exactly the bank-view builder's seven keys
  (``make_bank_view.PIN_ENTRY_KEYS``), plus ``derived_id`` on the two derived kinds, and the file names no other entry;
* per cell, the kind is the harness's (``registry_adapter.ARM_OBJECT_KINDS``), the served layer is the first stored block of that
  kind (LoRA kinds store layers 3, 7, ..., 31; PSR kinds all 32), and the identity sidecar is the fit's budget record for the two
  controls and the derived record for the two differences;
* every entry lives in the bank ``ctlstop_eb64`` at the view path ``ctlstop_eb64/qwen35_4b/cells_view/ctlstop64s<k>/<cell>``;
* the twelve shas are distinct 64-hex digests and none is pinned in the production pins file, so no published object is served
  again under a new name;
* the generator's reading of each pin agrees, and so does the job-start registration's (``jobplan.object_id_for`` with the checkout
  as the pins root): the object a run registers is the object its view serves;
* the rendered configs and manifests are the generator's (``make_full22_configs.py --check --roster ...`` exits 0, and each config
  is ``render_config``'s text), each config serving ``none`` and the view's four cells."""
from __future__ import annotations

import json
import os
import re
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
import make_bank_view as mbv  # noqa: E402
import make_full22_configs as gen  # noqa: E402
from sisteer.capeval.petri.registry_adapter import ARM_OBJECT_KINDS  # noqa: E402
from sisteer.eval.defense_cell import arm_of  # noqa: E402
from sisteer.registry import jobplan as JP  # noqa: E402

LANE_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster_ctlstop_eb64.yaml"
LANE_PINS_REL = "configs/ctlstop_eb64_pins.json"
LANE_ROSTER = yaml.safe_load(LANE_ROSTER_PATH.read_text())
LANE_PINS = json.loads((REPO / LANE_PINS_REL).read_text())
PROD_PINS = json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text())
TARGET = "qwen35_4b"
BANK = "ctlstop_eb64"
VIEWS = ("ctlstop64s1", "ctlstop64s2", "ctlstop64s3")
#: cell -> (kind, served_layer, identity sidecar): the control LoRA, the control ReLU, the LoRA difference, the composed ReLU difference
CELLS = {"constv2ctl2-full__lora_r4_ce": ("lora_all", 3, "train_budget_meta.json"),
         "constv2ctl2-full__psr_ce_all": ("psr", 0, "train_budget_meta.json"),
         "constv2-mix-full__lora_r4_ce_minus_ctl": ("lora_subtraction", 3, "derived_meta.json"),
         "constv2-mix-full__psr_ce_all_minus_ctl": ("psr_paired_write", 0, "derived_meta.json")}
DERIVED_KINDS = ("lora_subtraction", "psr_paired_write")
SEVEN = {"bank", "source", "staged_id", "object_sha256", "kind", "served_layer", "identity_sidecar"}
CASES = [(view, cell) for view in VIEWS for cell in CELLS]


def staged_id(view: str, cell: str) -> str:
    return f"{view}__{cell}"


def production_shas() -> set:
    out = set()

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "object_sha256" and isinstance(v, str):
                    out.add(v)
                else:
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(PROD_PINS)
    return out


def test_the_pins_file_names_exactly_the_roster_pins():
    assert set(LANE_PINS) == {"_comment", TARGET}
    assert isinstance(LANE_PINS["_comment"], list) and all(isinstance(line, str) for line in LANE_PINS["_comment"])
    views = LANE_ROSTER["targets"][TARGET]["views"]
    assert tuple(views) == VIEWS
    refs = []
    for view in VIEWS:
        assert tuple(views[view]["cells"]) == tuple(CELLS)
        for cell, mapping in views[view]["cells"].items():
            assert fr.is_pin_form(mapping) and mapping["pins"] == LANE_PINS_REL
            refs.append(mapping["pin"])
    assert sorted(refs) == sorted(f"{TARGET}.{sid}" for sid in LANE_PINS[TARGET])
    assert len(LANE_PINS[TARGET]) == 12


def test_the_entry_keys_are_the_bank_view_builders():
    assert set(mbv.PIN_ENTRY_KEYS) == SEVEN


@pytest.mark.parametrize("view,cell", CASES)
def test_each_pin_carries_its_kind_layer_sidecar_bank_and_view_path(view, cell):
    kind, served_layer, sidecar = CELLS[cell]
    sid = staged_id(view, cell)
    pin = LANE_PINS[TARGET][sid]
    assert set(pin) == SEVEN | ({"derived_id"} if kind in DERIVED_KINDS else set()), sorted(pin)
    assert re.fullmatch(r"[0-9a-f]{64}", pin["object_sha256"])
    assert pin["kind"] == kind == ARM_OBJECT_KINDS[arm_of(cell)]
    assert pin["served_layer"] == served_layer and pin["identity_sidecar"] == sidecar
    assert pin["bank"] == BANK and pin["staged_id"] == sid
    assert pin["source"] == f"{BANK}/{TARGET}/cells_view/{view}/{cell}"
    parts = Path(pin["source"]).parts
    assert parts[0] == pin["bank"] and parts[-3] == "cells_view" and parts[-2] == view and parts[-1] == cell
    if kind in DERIVED_KINDS:
        assert pin["derived_id"] == cell
    # the generator reads the same pin
    mapping = LANE_ROSTER["targets"][TARGET]["views"][view]["cells"][cell]
    assert gen.pin_form_facts(TARGET, cell, mapping) == (sid, BANK, pin["object_sha256"], f"{TARGET}.{sid}")


def test_the_twelve_shas_are_distinct_and_none_is_a_production_pin():
    shas = [p["object_sha256"] for p in LANE_PINS[TARGET].values()]
    assert len(shas) == 12 == len(set(shas))
    assert not set(shas) & production_shas()


@pytest.mark.parametrize("view,cell", CASES)
def test_registration_names_the_object_the_view_serves(view, cell):
    sha = LANE_PINS[TARGET][staged_id(view, cell)]["object_sha256"]
    assert JP.object_id_for(cell, model_key=TARGET, roster_selection=view, roster=LANE_ROSTER, pins=None, si_sha={},
                            pins_root=str(REPO)) == sha


def test_the_generator_check_passes_on_the_committed_configs():
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    r = subprocess.run([sys.executable, str(REPO / "scripts/sdfbase/make_full22_configs.py"), "--roster", str(LANE_ROSTER_PATH),
                        "--check"], capture_output=True, text=True, cwd=REPO, env=env, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    assert r.stdout.count("ok: ") == 2 * len(VIEWS) and "DRIFT" not in r.stdout


@pytest.mark.parametrize("view", VIEWS)
def test_the_rendered_config_and_manifest_are_the_generators_and_serve_none_plus_the_four_cells(view):
    v = LANE_ROSTER["targets"][TARGET]["views"][view]
    cfg_path, man_path = REPO / v["config"], REPO / v["manifest"]
    assert cfg_path.name == f"sdfbase_full22_{TARGET}_{view}.yaml" and man_path.name == f"petri_full22_{TARGET}_{view}.yaml"
    text = cfg_path.read_text()
    assert text == gen.render_config(LANE_ROSTER, gen.load_pins(), TARGET, view, roster_path=LANE_ROSTER_PATH)
    assert man_path.read_text() == gen.render_manifest(LANE_ROSTER, TARGET, view, roster_path=LANE_ROSTER_PATH)
    cfg = yaml.safe_load(text)
    assert cfg["explicit_cells"] == ["none", *CELLS]
    assert cfg["private_run"] == f"sdfbase_full22_{TARGET}_{view}"
    assert cfg["attacks"]["target_max_new_tokens"] == LANE_ROSTER["target_max_new_tokens"] == 4096
    assert f"from\n# configs/{LANE_ROSTER_PATH.name} (target {TARGET}, view {view})" in text
    for cell in CELLS:
        sid = staged_id(view, cell)
        assert f"<- {sid}  (bank {BANK}, THROUGH the pin {LANE_PINS_REL} {TARGET}.{sid})" in text
        assert LANE_PINS[TARGET][sid]["object_sha256"] in text
