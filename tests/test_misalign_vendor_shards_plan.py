"""The Modal vendor-shards launcher's plan / path contract (one app, several qwen stems, an explicit root).

Loaded with ``modal`` stubbed: the helpers under test are pure, and the grading host test venv carries no Modal client."""
from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "scripts" / "modal" / "misalign_vendor_shards.py"


@pytest.fixture(scope="module")
def L():
    stub = types.ModuleType("modal")
    m = mock.MagicMock()
    for name in ("App", "Volume", "Image"):
        setattr(stub, name, getattr(m, name))
    with mock.patch.dict(sys.modules, {"modal": stub}):
        spec = importlib.util.spec_from_file_location("misalign_vendor_shards_under_test", LAUNCHER)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        yield mod


def test_parse_plan_expands_ranges_in_order_and_refuses_bad_shards(L):
    items = L.parse_plan("qwen35_4b:13:0,2-4;qwen35_4b_it:8:7;qwen35_4b_bl:4:0-3")
    assert [(i["target"], i["k"], i["n_shards"]) for i in items] == [
        ("qwen35_4b", 0, 13), ("qwen35_4b", 2, 13), ("qwen35_4b", 3, 13), ("qwen35_4b", 4, 13),
        ("qwen35_4b_it", 7, 8),
        ("qwen35_4b_bl", 0, 4), ("qwen35_4b_bl", 1, 4), ("qwen35_4b_bl", 2, 4), ("qwen35_4b_bl", 3, 4)]
    assert L.parse_shards("1") == [1] and L.parse_shards("0,2-12") == [0] + list(range(2, 13))
    with pytest.raises(ValueError, match="shard 8 >= n_shards 8"):
        L.parse_plan("qwen35_4b_it:8:8")
    with pytest.raises(ValueError, match="unknown target"):
        L.parse_plan("gemma4e4b:13:0")
    with pytest.raises(ValueError, match="bad shard list"):
        L.parse_shards("1,1")
    with pytest.raises(ValueError, match="empty plan"):
        L.parse_plan(";")


def test_target_paths_are_per_stem_and_the_root_stays_under_misalign_vendor(L):
    main = L.target_paths("qwen35_4b")
    assert main["out_root"] == "/vol/misalign_vendor/qwen35_4b" and main["view"] == "main" and not main["baselines"]
    assert main["objects"] == "/vol/misalign/objects/qwen35_4b/cells_view"
    it = L.target_paths("qwen35_4b_it", "misalign_vendor_leak")
    assert it["out_root"] == "/vol/misalign_vendor_leak/qwen35_4b_it" and it["view"] == "it"
    assert it["cfg"] == "configs/misalign_full22_qwen35_4b_it.yaml"
    assert it["manifest"] == "configs/capeval/manifests/misalign_full22_qwen35_4b_it.yaml"
    bl = L.target_paths("qwen35_4b_bl", "misalign_vendor_leak")
    assert bl["baselines"] and bl["view"] == "main" and bl["model_key"] == bl["family"] == "qwen35_4b"
    # every stem's config pair exists in this tree, so a plan cannot name a target the standup cannot serve
    for t in L.TARGETS:
        tp = L.target_paths(t)
        assert (ROOT / tp["cfg"]).is_file() and (ROOT / tp["manifest"]).is_file(), t
    # the murder record's root is the default and a second pass needs its own root — never another prefix
    with pytest.raises(ValueError, match="root must be misalign_vendor"):
        L.target_paths("qwen35_4b", "misalign_16k")
    with pytest.raises(ValueError, match="root must be misalign_vendor"):
        L.target_paths("qwen35_4b", "misalign_vendorleak")


def test_baseline_pins_mirror_the_sbatch_record_from_the_real_pins_file(L):
    pins = L.baseline_pins(str(ROOT), "qwen35_4b")
    real = json.loads((ROOT / "configs" / "misalign_baseline_pins.json").read_text())["qwen35_4b"]
    assert set(pins) == {"arditi", "alphasteer"}
    for cell in pins:
        assert set(pins[cell]) <= set(L.PIN_KEYS) and pins[cell]["object_sha256"] == real[cell]["object_sha256"]
        assert pins[cell]["selection_outcome"] == real[cell]["selection_outcome"]
    # the same key set misalign_proxy.sbatch writes into cells.json: since the reseed lane split SERVED from MEASURED the sbatch
    # exports F22_BASELINE_PINS and delegates the record to misalign_cells.py, whose BASELINE_PIN_KEYS is the one writer of the keys
    sbatch = (ROOT / "scripts" / "sdfbase" / "full22" / "misalign" / "misalign_proxy.sbatch").read_text()
    assert 'F22_BASELINE_PINS="$REPO_DIR/configs/misalign_baseline_pins.json"' in sbatch
    assert "scripts/sdfbase/full22/misalign/misalign_cells.py" in sbatch
    spec = importlib.util.spec_from_file_location("misalign_cells_under_test",
                                                  ROOT / "scripts" / "sdfbase" / "full22" / "misalign" / "misalign_cells.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    assert tuple(mc.BASELINE_PIN_KEYS) == tuple(L.PIN_KEYS)


def test_the_standup_argv_still_pins_the_lens_path_once(L):
    text = LAUNCHER.read_text()
    assert text.count('"--serving-path", "lens"') == 1 and '"--reduction-mode", "stacked"' in text


def test_the_psr_panel_lands_in_the_qwen35_4b_root_with_tagged_files_and_the_native_path_is_known(L):
    """The three PSR / ReLU cells join the qwen35_4b leaking vendor panel (same root, paired against its lens `none`),
    their shard files tagged so they never collide with the lens shards', served on the native path."""
    tp = L.target_paths("qwen35_4b_psr", "misalign_vendor_leak")
    assert tp["out_root"] == "/vol/misalign_vendor_leak/qwen35_4b"
    assert tp["objects"] == "/vol/misalign/objects/qwen35_4b/cells_view"
    assert tp["cfg"] == "configs/misalign_full22_qwen35_4b_psr.yaml" and tp["view"] == "main" and not tp["baselines"]
    assert tp["file_tag"] == "psr_" and L.target_paths("qwen35_4b")["file_tag"] == ""
    assert L.SERVING_PATHS == ("lens", "lora_native")
    assert L.parse_plan("qwen35_4b_psr:4:1-3") == [{"target": "qwen35_4b_psr", "k": k, "n_shards": 4} for k in (1, 2, 3)]
    text = LAUNCHER.read_text()
    assert '["--serving-path", "lora_native", "--adapter-bank"' in text and "serving_engine" in text
    assert L.ADAPTER_BANK_ROOT == "/vol/tinycap/multilora_adapters"   # the tinycap launcher's bank, shared across native standups


def test_the_allpos_panel_pins_its_object_digest_explicitly_and_lives_in_its_own_root(L):
    """The Arditi all-positions panel (view it, single-object native standup) holds its object digest at the standup."""
    tp = L.target_paths("qwen35_4b_allpos", "misalign_vendor")
    assert tp["out_root"] == "/vol/misalign_vendor/qwen35_4b_allpos" and tp["view"] == "it" and tp["file_tag"] == ""
    assert tp["objects"] == "/vol/misalign/objects/qwen35_4b_allpos/cells_view"
    assert tp["expect_digests"] == {"arditi_allpos": "ba2212f0c2f9ffef2997bdd24831182052560cf980054bcee9cf95ea0db085b2"}
    assert L.target_paths("qwen35_4b")["expect_digests"] == {}
    assert '"--expect-object-sha256"' in LAUNCHER.read_text()
