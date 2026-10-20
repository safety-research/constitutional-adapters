"""The s3r static standups 858118 / 858119 refused "enumerate_cells derived no none__lora0 leg" — the LoRA path's cell selection
(lora_serving_up.select_cells_lora) served EXACTLY the config's explicit_cells when nothing was requested, and a steered-wave config lists no
`none`, so the native core enumerated no base route, derived no none__lora0 control twin and refused. The served ≠ measured rule covered
mt_serving_up.select_cells (the lens path) only. Now select_cells_lora serves the base route implicitly too: the core derives the twin from it
(the positive controls compare against it, load_check probes it, the attestation reads vmodels["none"]), while neither none nor the twin is a
measured, registered or scored leg — the drivers iterate the CONFIG's cells and the declaration (cells_requested_explicit) stays what --cell
named. Through the REAL mt_serving_up main over lora_serving_up.bring_up_lora and the real native core, the vllm child the scripted upstream
(the serving-path tests' harness); a config that lists `none` first is served byte-identically."""
from __future__ import annotations

import importlib.util
import json
import socket
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "sdfbase", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import test_lora_serving_up as E  # noqa: E402
from sisteer.capeval.serving import geometry as sg  # noqa: E402,F401
from sisteer.capeval.serving import vllm_server as vs  # noqa: E402

KEY = "gemma4e4b"


@pytest.fixture(scope="module")
def mt():
    spec = importlib.util.spec_from_file_location("_mt_steered_wave_under_test", REPO / "scripts/sdfbase/mt_serving_up.py")
    mod = importlib.util.module_from_spec(spec); sys.modules["_mt_steered_wave_under_test"] = mod; spec.loader.exec_module(mod)
    return mod


def _run_without_cell(mt, tmp_path, monkeypatch, *, drop_none: bool):
    """The REAL ``mt.main`` under the native default with NO --cell (the static / controls / misalign standups' call) over the gemma campaign
    pair, its config's explicit_cells rewritten to the one object cell when ``drop_none`` (a steered-wave config's shape), the vllm child the
    scripted upstream. Returns the captured records while the standup is READY."""
    cfg_path, man_path, cfg = E._campaign(KEY)
    E.cached_or_skip(cfg)
    view, shas = E.fake_bank(tmp_path, cfg)
    cfg2 = dict(cfg)
    cfg2["explicit_cells"] = [E.GEMMA_CELL] if drop_none else ["none", E.GEMMA_CELL]           # one object cell (its digest given); the ordinary shape lists none first
    cfg2_path = tmp_path / ("steered_wave.yaml" if drop_none else "ordinary.yaml"); cfg2_path.write_text(yaml.safe_dump(cfg2, sort_keys=False))
    # the manifest enumerates exactly the config's cells (view_cell_universe): narrowed to this config's harness ids, else mt's manifest/config
    # cell check refuses before any selection — the generated s3r pair is exactly the steered-wave shape
    man = yaml.safe_load(Path(man_path).read_text()); man["petri"]["cells"] = [E.GEMMA_HCELL] if drop_none else ["none", E.GEMMA_HCELL]
    man_path = tmp_path / ("steered_wave_manifest.yaml" if drop_none else "ordinary_manifest.yaml"); man_path.write_text(yaml.safe_dump(man, sort_keys=False))
    out = tmp_path / ("out_sw" if drop_none else "out_ord"); out.mkdir()
    stop = out / "STOP"
    monkeypatch.setenv("HF_HUB_OFFLINE", "1"); monkeypatch.delenv("TT_TWO_SI_DEFENSES_DIR", raising=False)
    E.snapshot_seam(monkeypatch, tmp_path, cfg)
    monkeypatch.setattr(socket, "gethostname", lambda: "TESTHOST")
    captured: dict = {"shas": shas, "out": out, "config_cells": list(cfg2["explicit_cells"])}
    monkeypatch.setattr(vs, "launch_server", E.fake_upstream_launch_server(tmp_path / ("fu_sw" if drop_none else "fu_ord"), captured=captured,
                                                                            chat=E.default_chat(cfg["model"]), genuine_name=E.GEMMA_HCELL, chat_delay_s=0.05))

    def snapshot(stop_file, server, **kw):
        captured["endpoint"] = json.loads((out / "endpoint.json").read_text())
        captured["cells_served"] = json.loads((out / "cells_served.json").read_text())
        return 0
    monkeypatch.setattr(mt, "wait_until_stop", snapshot)
    argv = ["--repo-dir", str(REPO), "--config", str(cfg2_path), "--manifest", str(man_path), "--defenses-dir", str(view), "--out", str(out),
            "--stop-file", str(stop), "--mode", "attacks", "--eval", "families_served:short", "--model-key", KEY, "--adapter-bank", str(tmp_path / "bank"),
            "--expect-object-sha256", f"{E.GEMMA_CELL}={shas[E.GEMMA_CELL]}"]
    captured["rc"] = mt.main(argv)
    return captured


def test_a_steered_wave_config_stands_up_with_the_base_route_served_implicitly_and_the_twin_derived_from_it(mt, tmp_path, monkeypatch):
    cap = _run_without_cell(mt, tmp_path, monkeypatch, drop_none=True)
    assert cap["config_cells"] == [E.GEMMA_CELL], cap["config_cells"]                                   # the config lists NO none: a steered-wave shape
    assert cap["rc"] == 0, "the standup must stand up on a steered-only config (858118 / 858119 refused here)"
    ep, cs = cap["endpoint"], cap["cells_served"]
    assert set(ep["served_models"]) == {"petri__base", f"petri__{E.GEMMA_HCELL}", f"petri__{E.ZERO}"}         # the base route + the cell + the zero twin, SERVED
    assert "none" in cs["cells_requested"] and E.GEMMA_CELL in cs["cells_requested"]                        # the served set carries the implicit base route
    assert cs["cells_requested_explicit"] == [] and cs["cells_requested_explicit_harness"] == []             # nothing DECLARED: no --cell was named
    assert ep["cells_skipped"] == []


def test_a_config_that_lists_none_first_is_served_byte_identically(mt, tmp_path, monkeypatch):
    cap = _run_without_cell(mt, tmp_path, monkeypatch, drop_none=False)
    assert cap["rc"] == 0 and cap["config_cells"][0] == "none"
    ep, cs = cap["endpoint"], cap["cells_served"]
    assert set(ep["served_models"]) >= {"petri__base", f"petri__{E.GEMMA_HCELL}", f"petri__{E.ZERO}"}
    assert cs["cells_requested"][0] == "none" and cs["cells_requested_explicit"] == []
    src = (REPO / "scripts/sdfbase/lora_serving_up.py").read_text()
    assert 'return (want if "none" in want else ["none", *want]), []' in src                             # the lens-path rule, mirrored
    assert 'return want if "none" in want else ["none", *want]' in (REPO / "scripts/sdfbase/mt_serving_up.py").read_text()
