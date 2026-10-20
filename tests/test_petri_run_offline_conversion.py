"""petri_run --offline-models-json: the proxy-free CONVERSION engine handle (2026-09-21).

Re-converting a cell whose every sample already sits in the Inspect logs needs no GPU, yet the
runner's `run` path built its engine by loading the LIVE proxy (`/v1/models`) and read every
identity fact off that handle. The flag builds the same handle from the /v1/models response the
grading host drive SAVED from the live proxy, validated exactly as `ProxyClientEngine.load()` validates
a live listing, and points it at an unreachable endpoint so nothing can be audited by accident.
"""
import importlib.util, json, sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if "petri_run" in sys.modules:
    petri_run = sys.modules["petri_run"]
else:
    _l = importlib.util.spec_from_file_location("petri_run", ROOT / "scripts" / "petri_run.py")
    petri_run = importlib.util.module_from_spec(_l); sys.modules["petri_run"] = petri_run; _l.loader.exec_module(petri_run)

CELL = SimpleNamespace(virtual_model_name="petri__si_x__lora_r4_ce",
                       serving_identity="proxy_vllm+lora+interleaved")
IDENT = {"engine": {"kind": "lora_native"}, "petri_agent_loop_entry": True, "serving_mode": "interleaved"}


def _listing(tmp_path, ids=("petri__none__lora0", "petri__si_x__lora_r4_ce"), ident=IDENT, owned="petri"):
    p = tmp_path / "models.json"
    p.write_text(json.dumps({"data": [{"id": i, "owned_by": owned} for i in ids], "petri_proxy": ident}))
    return p


def test_offline_handle_mirrors_a_loaded_engine(tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("PETRI_PROXY_BASE_URL", raising=False)
    eng = petri_run.offline_engine_from_models_json(_listing(tmp_path), CELL)
    assert eng.proxy_identity == IDENT
    assert eng.served_models == ("petri__none__lora0", "petri__si_x__lora_r4_ce")
    assert eng.identity == CELL.serving_identity and eng.name == CELL.serving_identity
    assert eng.base_url == petri_run.OFFLINE_BASE_URL
    assert "OFFLINE CONVERSION" in capsys.readouterr().out


def test_offline_handle_refuses_a_listing_that_did_not_serve_the_cell(tmp_path):
    with pytest.raises(petri_run.DriverError, match="does not serve"):
        petri_run.offline_engine_from_models_json(_listing(tmp_path, ids=("petri__none__lora0",)), CELL)


def test_offline_handle_refuses_a_foreign_listing_and_a_missing_identity(tmp_path):
    with pytest.raises(petri_run.DriverError, match="outside"):
        petri_run.offline_engine_from_models_json(_listing(tmp_path, ids=("gpt-4o", "petri__si_x__lora_r4_ce")), CELL)
    with pytest.raises(petri_run.DriverError, match="owned by"):
        petri_run.offline_engine_from_models_json(_listing(tmp_path, owned="someone"), CELL)
    p = tmp_path / "m.json"; p.write_text(json.dumps({"data": [{"id": "petri__si_x__lora_r4_ce"}]}))
    with pytest.raises(petri_run.DriverError, match="petri_proxy"):
        petri_run.offline_engine_from_models_json(p, CELL)
    with pytest.raises(petri_run.DriverError, match="unreadable"):
        petri_run.offline_engine_from_models_json(tmp_path / "absent.json", CELL)


def test_flag_is_a_top_level_option_and_defaults_off():
    ns = petri_run.build_parser().parse_args(["--manifest", "m.yaml", "print-cells"])
    assert ns.offline_models_json is None
    ns = petri_run.build_parser().parse_args(["--manifest", "m.yaml", "--offline-models-json", "x.json", "run", "--cell-index", "0"])
    assert str(ns.offline_models_json) == "x.json"
