"""The pins of the non-production roster ``configs/sdfbase_full22_roster_sikl4.yaml``, and the configs rendered from it.

``configs/sikl4_pins.json`` names the eighteen objects the roster's three views serve: per fit seed k in 1..3 the SI-distilled LoRA and
ReLU of the safety SI, the constitution SI and the short irrelevant control SI C16, each a staged copy ``sikl4s<k>__<cell>`` in the
bank ``sikl4``. Pinned here:

* the bank-view builder's seven entry keys (``make_bank_view.PIN_ENTRY_KEYS``), and per cell the kind the harness gives its arm
  (``registry_adapter.ARM_OBJECT_KINDS``), the served layer (the first stored block of that kind: the LoRA stores o_proj layers
  3, 7, ..., 31, the ReLU all 32) and the identity sidecar (the fit's budget record);
* the roster names one pin per (view, cell), ``qwen35_4b.sikl4s<k>__<cell>``;
* the pins file holds exactly the eighteen roster pins and nothing else: every roster pin resolves to an entry with exactly the seven
  keys, its kind, layer, sidecar, bank and view path; the eighteen shas are distinct 64-hex digests, none a production pin, a
  published recipe object or a constitutional adapter;
* the generator's and the job-start registration's reading of each pin agree;
* the committed configs and manifests are the generator's, each serving ``none`` and the view's six cells with the attacker, attacks
  and grader blocks of the early-stopped control views of the same seed."""
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

LANE_ROSTER_PATH = REPO / "configs/sdfbase_full22_roster_sikl4.yaml"
LANE_PINS_REL = "configs/sikl4_pins.json"
LANE_PINS = json.loads((REPO / LANE_PINS_REL).read_text())
LANE_ROSTER = yaml.safe_load(LANE_ROSTER_PATH.read_text())
TARGET = "qwen35_4b"
BANK = "sikl4"
VIEWS = ("sikl4s1", "sikl4s2", "sikl4s3")
#: cell -> (kind, served_layer, identity sidecar): per SI the rank-4 o_proj LoRA and the all-layer ReLU steer, both fit by the trainer
CELLS = {"safety_refusal__lora_r4": ("lora_all", 3, "train_budget_meta.json"),
         "safety_refusal__psr_kl_all": ("psr", 0, "train_budget_meta.json"),
         "si_constitution__lora_r4": ("lora_all", 3, "train_budget_meta.json"),
         "si_constitution__psr_kl_all": ("psr", 0, "train_budget_meta.json"),
         "C16__lora_r4": ("lora_all", 3, "train_budget_meta.json"),
         "C16__psr_kl_all": ("psr", 0, "train_budget_meta.json")}
SEVEN = {"bank", "source", "staged_id", "object_sha256", "kind", "served_layer", "identity_sidecar"}
CASES = [(view, cell) for view in VIEWS for cell in CELLS]
#: the objects these eighteen must never be: the six published single fits whose recipes they reproduce, and the constitutional
#: adapters (corpus fits) the figure draws as its reference group
PUBLISHED_RECIPE_OBJECTS = {
    "0dbf406f94a14ab9f9b960155bb81ddb685d2b1c916f9a04dda2f49eda206341", "75625f59809498899db9efcd2b19e9ce8c2f95593e7c7c1a109a03e29fe5b5c8",
    "88523971a71fbefe8b1d70f03fe9ab83821a796d8621ab882dc810596731cb9f", "0a69aab85e1ead1e3b38d377e197fb03520c5ae7ca1930185ce0990d46bb3482",
    "859d4d9a1eb60698790d92485148f28c5b6ad7a258a6e033cdf60c3f846233ef", "1284d158128ab0d8672eeb9e5d2a7b101f53b75b8789fbce9cc2b05ca3c868cf"}
CONSTITUTIONAL_ADAPTERS = {
    "8be1e1a7ab36f3f65557f7835a30c6b8f18143c781fbf7fb312a5cc7a6e08871", "724fe6ddb52cb679656442b2537057900f3c523ddc3c45b3edde50bdc4a487d2",
    "a9934f6a796c238d58f4da10615c9f1c211d8b067afcf1330241cdcff967ce42", "7b87767d91bd9787d22a23a80a5e78c53d16b506cc25e4d9483f74e88238fda1",
    "2d43a58e55e674d3e33c6cbbaf7bb0ad9b06021d308b9e924213acf7c0609fd8", "cc1ec068380ed8fe51de1e91e8b8fe6b32c95fa9a00bcf381a536007c075bf09"}


def staged_id(view: str, cell: str) -> str:
    return f"{view}__{cell}"


def _production_shas() -> set:
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
    walk(json.loads((REPO / "configs/sdfbase_full22_pins.json").read_text()))
    return out


# ------------------------------------------------------------------------------------------------- the table and the roster

def test_the_entry_keys_are_the_bank_view_builders():
    assert set(mbv.PIN_ENTRY_KEYS) == SEVEN


@pytest.mark.parametrize("cell", list(CELLS))
def test_each_cells_expected_kind_is_the_harness_kind_of_its_arm(cell):
    kind, served_layer, sidecar = CELLS[cell]
    assert ARM_OBJECT_KINDS[arm_of(cell)] == kind
    assert served_layer == (3 if kind == "lora_all" else 0) and sidecar == "train_budget_meta.json"


def test_the_roster_names_one_pin_per_view_and_cell():
    views = LANE_ROSTER["targets"][TARGET]["views"]
    assert tuple(views) == VIEWS
    refs = []
    for view in VIEWS:
        assert tuple(views[view]["cells"]) == tuple(CELLS)
        for cell, mapping in views[view]["cells"].items():
            assert fr.is_pin_form(mapping) and mapping["pins"] == LANE_PINS_REL
            refs.append(mapping["pin"])
    assert sorted(refs) == sorted(f"{TARGET}.{staged_id(v, c)}" for v, c in CASES)


# ------------------------------------------------------------------------------------------------- the pins and the rendered configs

def test_the_pins_file_names_exactly_the_roster_pins():
    assert set(LANE_PINS) == {"_comment", TARGET}
    assert isinstance(LANE_PINS["_comment"], list) and all(isinstance(line, str) for line in LANE_PINS["_comment"])
    assert sorted(LANE_PINS[TARGET]) == sorted(staged_id(v, c) for v, c in CASES)
    assert len(LANE_PINS[TARGET]) == 18


@pytest.mark.parametrize("view,cell", CASES)
def test_each_pin_carries_its_kind_layer_sidecar_bank_and_view_path(view, cell):
    kind, served_layer, sidecar = CELLS[cell]
    sid = staged_id(view, cell)
    pin = LANE_PINS[TARGET][sid]
    assert set(pin) == SEVEN, sorted(pin)
    assert re.fullmatch(r"[0-9a-f]{64}", pin["object_sha256"])
    assert pin["kind"] == kind == ARM_OBJECT_KINDS[arm_of(cell)]
    assert pin["served_layer"] == served_layer and pin["identity_sidecar"] == sidecar
    assert pin["bank"] == BANK and pin["staged_id"] == sid
    assert pin["source"] == f"{BANK}/{TARGET}/cells_view/{view}/{cell}"
    # the generator reads the same pin
    mapping = LANE_ROSTER["targets"][TARGET]["views"][view]["cells"][cell]
    assert gen.pin_form_facts(TARGET, cell, mapping) == (sid, BANK, pin["object_sha256"], f"{TARGET}.{sid}")


def test_the_eighteen_shas_are_new_objects():
    shas = [p["object_sha256"] for p in LANE_PINS[TARGET].values()]
    assert len(shas) == 18 == len(set(shas))
    assert not set(shas) & _production_shas()
    assert not set(shas) & PUBLISHED_RECIPE_OBJECTS and not set(shas) & CONSTITUTIONAL_ADAPTERS


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
def test_the_rendered_config_and_manifest_are_the_generators_and_serve_none_plus_the_six_cells(view):
    k = int(view[-1])
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
    # the evaluation setup of the early-stopped control views of the same seed: the attacker, the attacks and the grader, as data
    ctl = yaml.safe_load((REPO / f"configs/sdfbase_full22_{TARGET}_ctlstop64s{k}.yaml").read_text())
    for block in ("attacker", "attacks", "grader"):
        assert cfg[block] == ctl[block], (view, block)
