"""The misalign view builder accepts a PROMPTED-ONLY front-door view (paper-companion-repo stage E, 2026-09-30).

The unsteered u0 view serves `none` + the two prompted SIs and stages no object, so the roster names no staged cells for it.
`misalign_bank_view.sh` read that as "still pending_fit" and refused, and the u0 agentic proxy died at 46 s. Now a config that lists
no object cell names a prompted-only view: the misalign view is built with NO entries (+ its manifest) and verifies; a config that
lists object cells with an empty staged set keeps the pending-fit refusal. The tree's front-door model (`qwen35_4b_demo`) runs the
SINGLE-PHASE recipe (2026-10-08: one wave s1 carrying every arm, no prompted-only u0), so the u0 view rides a SYNTHETIC row that
runs in phases (`phased_demo`: the demo's pins under its own key, views u0 + s1) added to a copy of the models table inside a SYNTHETIC
checkout (the row's u0 block pasted into the production roster as `05 --step roster` prints it, its u0 wave pair and misalign pair rendered
by the real generators) and a fake shared root (MISALIGN_SHARED) that holds the wave's u0 view exactly as `05 --step bank-views` leaves it
for u0: an empty VIEW_MANIFEST.json."""
from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
for p in (REPO / "scripts" / "sdfbase", REPO / "scripts" / "sdfbase" / "full22", REPO / "reproduce" / "lib", REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import f22_roster as fr  # noqa: E402
import make_full22_configs as wave  # noqa: E402
import models as MT  # noqa: E402

SCRIPT_REL = "scripts/sdfbase/full22/misalign/misalign_bank_view.sh"
DEMO = "qwen35_4b_demo"                                             # the tree's front-door model: single-phase, no u0
KEY, TARGET, VIEW = "phased_demo_u0", "phased_demo", "u0"         # the SYNTHETIC phased row's prompted-only wave


def _phased_table(path: Path) -> Path:
    """The models table plus the synthetic phased row (the demo's pins and objects under its own key and view_legacy_prefix; views u0 + s1)."""
    import copy
    t = yaml.safe_load((REPO / "reproduce" / "models.yaml").read_text(encoding="utf-8"))
    row = copy.deepcopy(t["models"][DEMO]); row["view_legacy_prefix"] = TARGET
    row["views"] = {"u0": {"cells": ["none", "prompted__safety_refusal", "prompted__si_constitution"]},
                    "s1": {"cells": ["control_lora", "constitution_lora", "difference_lora"]}}
    t["models"][TARGET] = row
    path.write_text(yaml.safe_dump(t, sort_keys=False), encoding="utf-8")
    return path


def _checkout(tmp_path: Path) -> Path:
    """A checkout the script can run in: the real full22 scripts + reproduce tree (copied, so the misalign generator's REPO is this tree), the
    repo's configs (linked) except the roster, which carries the front-door model's u0 block, plus the u0 wave pair and the misalign pair the
    generators render from it."""
    co = tmp_path / "checkout"
    shutil.copytree(REPO / "scripts" / "sdfbase" / "full22", co / "scripts" / "sdfbase" / "full22", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(REPO / "reproduce", co / "reproduce", ignore=shutil.ignore_patterns("__pycache__"))
    (co / "configs" / "capeval" / "manifests").mkdir(parents=True)
    for p in (REPO / "configs").iterdir():
        if p.name not in ("sdfbase_full22_roster.yaml", "capeval"):
            (co / "configs" / p.name).symlink_to(p)
    for p in (REPO / "configs" / "capeval").iterdir():
        if p.name != "manifests":
            (co / "configs" / "capeval" / p.name).symlink_to(p)
    for p in (REPO / "configs" / "capeval" / "manifests").iterdir():
        (co / "configs" / "capeval" / "manifests" / p.name).symlink_to(p)
    table = _phased_table(co / "reproduce" / "models.yaml")                     # the checkout's table carries the phased row (the generators read THIS tree's)
    roster = fr.load_roster(REPO / "configs" / "sdfbase_full22_roster.yaml")
    roster["targets"][TARGET] = yaml.load("targets:\n" + MT.roster_block(TARGET, [VIEW], str(table)), Loader=fr._StrictRosterLoader)["targets"][TARGET]
    pins = json.loads((REPO / "configs" / "sdfbase_full22_pins.json").read_text(encoding="utf-8")); pins[TARGET] = {}     # u0 stages no object
    (co / "configs" / "sdfbase_full22_roster.yaml").write_text(yaml.safe_dump(roster, sort_keys=False), encoding="utf-8")
    fd = {**wave.FRONT_DOOR_MODELS, **wave.front_door_models(table)}            # the wave generator's import-time table + the phased row
    orig = wave.FRONT_DOOR_MODELS
    wave.FRONT_DOOR_MODELS = fd
    try:
        (co / "configs" / f"sdfbase_full22_{KEY}.yaml").write_text(wave.render_config(roster, pins, TARGET, VIEW), encoding="utf-8")
        (co / "configs" / "capeval" / "manifests" / f"petri_full22_{KEY}.yaml").write_text(wave.render_manifest(roster, TARGET, VIEW), encoding="utf-8")
    finally:
        wave.FRONT_DOOR_MODELS = orig
    spec = importlib.util.spec_from_file_location("_mm_prompted_only", co / "scripts" / "sdfbase" / "full22" / "misalign" / "make_misalign_manifest.py")
    mm = importlib.util.module_from_spec(spec); spec.loader.exec_module(mm)
    assert mm.REPO == co and mm.wave_sources_present(KEY)
    m_text, cells = mm.render_manifest(KEY)
    c_text, _tt = mm.render_config(KEY, cells)
    (co / "configs" / f"misalign_full22_{KEY}.yaml").write_text(c_text, encoding="utf-8")
    (co / "configs" / "capeval" / "manifests" / f"misalign_full22_{KEY}.yaml").write_text(m_text, encoding="utf-8")
    return co


def _run(co: Path, shared: Path, *args):
    env = {k: v for k, v in os.environ.items() if k not in ("MISALIGN_SHARED", "MISALIGN_ORG_PINS", "MISALIGN_PY")}
    env["MISALIGN_SHARED"] = str(shared); env["PYTHONPATH"] = str(REPO / "src"); env["SISTEER_MODELS_TABLE"] = str(co / "reproduce" / "models.yaml")
    r = subprocess.run(["bash", str(co / SCRIPT_REL), KEY, "--python", sys.executable, *args], env=env, capture_output=True, text=True,
                       cwd=str(co), timeout=300)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="GNU find -printf / flock")
def test_a_prompted_only_front_door_view_builds_an_empty_misalign_view_and_verifies(tmp_path):
    co = _checkout(tmp_path)
    shared = tmp_path / "shared"
    wave_view = shared / "exp/objects/sdfbase/full22" / TARGET / "cells_view" / VIEW
    wave_view.mkdir(parents=True)
    (wave_view / "VIEW_MANIFEST.json").write_text(json.dumps({"wave": "full22", "target": TARGET, "view": VIEW, "entries": []}))
    rc, out = _run(co, shared, "--check")                 # --check before a build: no misalign view yet
    assert rc == 2 and "no view at" in out and "pending_fit" not in out, out
    rc, out = _run(co, shared)
    assert rc == 0, out
    assert f"prompted-only view {TARGET}/{VIEW}" in out and "no staged cells" in out
    view = shared / "exp/objects/misalign" / KEY / "cells_view"
    assert view.is_dir() and sorted(p.name for p in view.iterdir()) == ["VIEW_MANIFEST.json"]
    man = json.loads((view / "VIEW_MANIFEST.json").read_text())
    assert man.get("entries") == [] or man.get("cells") == [], man
    rc, out = _run(co, shared, "--check")
    assert rc == 0, out
