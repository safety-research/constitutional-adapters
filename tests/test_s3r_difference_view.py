"""The composed ReLU difference's own steered-only view `s3r` — the mechanisms it landed, on the
front-door model of reproduce/models.yaml (qwen35_4b_demo, configured-not-run: its `difference_relu` is the composed paired write under the
psr_paired_write kind directory) and a SYNTHETIC copy of the table that declares the view: 05_stage_roster.sh resolves a derived object
through its kind directory (the old <fit_id>/derived/ root was never a real layout) and by its REGISTRY id (the pin; s3r-fix P1-2), refusing a
table without the directory and a derived object without a pin; unit_plan honours `exclude_families: [bon, padding]` and plans
the view steered-only (no `none`), the Petri proxy declaring its one measured cell. No model of the tree declares `s3r` now, so
the view rides a tmp copy of the table; the pins, the wave config and the roster it reads are synthetic too (the front-door model commits none:
05 --step configs renders them at run time)."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402

sys.path.insert(0, str(REPO / "reproduce" / "lib"))
import models as MT  # noqa: E402
import stage as ST  # noqa: E402

KEY, FIT = "qwen35_4b_demo", "qwen35_4b_demo_base"
CELL = "constv2-mix-full__psr_ce_all_minus_ctl"                   # the table's difference_relu cell id: the composed paired write (demo-import M2)
STAGED = f"derived__{CELL}"
LORA_CELL = "constv2-mix-full__lora_r4_ce_minus_ctl"
LORA_STAGED = f"derived__{LORA_CELL}"
ARDITI_STAGED = "arditi__ar_c0p3__arditi_allpos"
OBJ_SHA, LORA_SHA, ARDITI_SHA = "1" * 64, "2" * 64, "3" * 64       # synthetic registry ids: the pins a 05 run would carry
TABLE = REPO / "reproduce" / "models.yaml"
SHA = "a" * 40
EXCLUDE_REASON = "USER RULING D913 / D889 (paper-companion-repo): BoN and the padding ladder never run on the composed difference"


@pytest.fixture(autouse=True)
def _knobs(monkeypatch):
    monkeypatch.setenv("SISTEER_SHARD_GROUPS", "0,1,2,3,4 5,6,7,8")
    monkeypatch.setenv("SISTEER_MT_ATTACKER_GPUS", "2")


def _s3r_table(tmp_path: Path, name: str = "table") -> Path:
    """A copy of the models table with `s3r` (the composed difference alone) declared on the front-door model, bon + padding excluded by ruling."""
    t = yaml.safe_load(TABLE.read_text())
    t["models"][KEY]["views"]["s3r"] = {"cells": ["difference_relu"], "exclude_families": ["bon", "padding"], "exclude_reason": EXCLUDE_REASON}
    d = tmp_path / name; d.mkdir(exist_ok=True); p = d / "models.yaml"; p.write_text(yaml.safe_dump(t, sort_keys=False))
    return p


def _pins(tmp_path: Path, name: str, entries: dict) -> Path:
    p = tmp_path / name; p.write_text(json.dumps({KEY: {sid: {"object_sha256": sha} for sid, sha in entries.items()}}))
    return p


def test_the_table_names_the_composed_object_under_its_kind_directory_and_the_view_fits_the_grammar(tmp_path):
    m = MT.model(KEY, str(TABLE))
    o = m["objects"]["difference_relu"]
    assert o["cell_id"] == CELL and o["source"] == "derived_bank" and o["derived_dir"] == "psr_paired_write" and o["object_kind"] == "composed"   # the composed paired write (demo-import M2)
    assert m["objects"]["difference_lora"]["derived_dir"] == "lora_subtraction" and m["objects"]["difference_lora"]["cell_id"] == LORA_CELL
    assert MT.roster_cell_id(m, "difference_relu") == CELL and MT.staged_id(m, "difference_relu") == STAGED
    assert re.fullmatch(MT.PHASE_VIEW_RE.pattern, "s3r") and "s3r" not in m["views"]                      # the grammar admits the name (no widening); no model declares it
    s = MT.model(KEY, str(_s3r_table(tmp_path)))
    assert s["views"]["s3r"]["cells"] == ["difference_relu"] and sorted(s["views"]["s3r"]["exclude_families"]) == ["bon", "padding"] and "D913" in s["views"]["s3r"]["exclude_reason"]
    assert MT.roster_view_cells(s, "s3r") == {CELL: STAGED} and MT.steered_wave_view("s3r", KEY) and MT.excluded_families(KEY, "s3r", str(tmp_path / "table" / "models.yaml")) == {"bon": EXCLUDE_REASON, "padding": EXCLUDE_REASON}


def _cells_json(table: Path, views: str, pins: Path):
    """05_stage_roster.sh's cells_json heredoc, run as the script runs it (argv: lib, table, key, views, shared, fit_id, fit_cluster, pins)."""
    src = (REPO / "reproduce" / "05_stage_roster.sh").read_text()
    body = src[src.index("<<'PYEOF'\n", src.index("cells_json()")) + len("<<'PYEOF'\n"):]
    body = body[:body.index("\nPYEOF\n")]
    r = subprocess.run([sys.executable, "-", str(REPO / "reproduce" / "lib"), str(table), KEY, views, "/shared", FIT, "cluster_a", str(pins)],
                       input=body, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr


def test_05_resolves_a_derived_object_through_its_kind_directory_and_by_its_pin_and_refuses_without_either(tmp_path):
    table = _s3r_table(tmp_path)
    pins = _pins(tmp_path, "pins.json", {STAGED: OBJ_SHA, LORA_STAGED: LORA_SHA, ARDITI_STAGED: ARDITI_SHA})
    rc, out, err = _cells_json(table, "s3r", pins)
    assert rc == 0, err
    assert json.loads(out) == [{"cell_id": CELL, "lane": "derived", "kind": "derived",
                                "source": f"cluster_a:/shared/exp/objects/sdfbase/full22/derived/psr_paired_write/{FIT}/{CELL}", "view_cell": "difference_relu",
                                "object_sha256_expected": OBJ_SHA, "object_sha256_expected_source": f"{pins}:{KEY}.{STAGED}"}]   # s3r-fix P1-2: the registry id rides the row
    rc, out, err = _cells_json(table, "battery", pins)                                                     # the LoRA difference: pinned from its registry row
    assert rc == 0, err
    lora = [r for r in json.loads(out) if r["view_cell"] == "difference_lora"]
    assert len(lora) == 1 and lora[0]["object_sha256_expected"] == LORA_SHA and lora[0]["object_sha256_expected_source"] == f"{pins}:{KEY}.{LORA_STAGED}"
    assert lora[0]["source"] == f"cluster_a:/shared/exp/objects/sdfbase/full22/derived/lora_subtraction/{FIT}/{LORA_CELL}"
    rc, out, err = _cells_json(table, "battery", pins)                                                     # the fitted parents: the fit bank; the Arditi object its own bank (the demo's one wave)
    assert rc == 0, err
    srcs = {r["view_cell"]: r["source"] for r in json.loads(out)}
    # the refusal direction's source is 04's publication dir (<$SHARED/exp/objects/refusal_direction>/<SERVED registry model id>/<cell_id>; gap-refusal-pin —
    # the demo serves `qwen35_4b_demo`, an id spelled like its key, so the path reads KEY here), never the paper's ladder bank
    assert MT.model(KEY, str(TABLE))["serve"]["checkpoint"]["model_id"] == KEY
    assert all(f"/full22/{FIT}/cells_all/" in srcs[c] for c in ("control_relu", "constitution_relu")) and srcs["refusal_direction"] == f"cluster_a:/shared/exp/objects/refusal_direction/{KEY}/ar_c0p3__arditi_allpos"
    assert "none" not in srcs and not any(c.startswith("prompted__") for c in srcs)                   # no object, no row
    # without a pin a derived object is refused by name — never staged by directory discovery (s3r-fix P1-2)
    nopins = _pins(tmp_path, "nopins.json", {})
    rc, out, err = _cells_json(table, "s3r", nopins)
    assert rc != 0 and f"derived object difference_relu ({STAGED}) has no pin" in err and "staged by its REGISTRY id" in err, err
    # a derived object whose table row names no kind directory refuses by name (never a guessed layout)
    t = yaml.safe_load(table.read_text()); t["models"][KEY]["objects"]["difference_relu"].pop("derived_dir")
    bad = tmp_path / "bad"; bad.mkdir(); (bad / "models.yaml").write_text(yaml.safe_dump(t, sort_keys=False))
    rc, out, err = _cells_json(bad / "models.yaml", "s3r", pins)
    assert rc != 0 and "has source derived_bank but no derived_dir" in err and "refusing to guess" in err, err


def test_the_plan_for_s3r_is_steered_only_without_bon_and_the_excluded_families_are_refused_by_name(tmp_path, monkeypatch):
    table = str(_s3r_table(tmp_path))
    # the view's wave config and a roster naming it, under tmp (the front door commits none; 05 --step configs renders them): what the plan's
    # steered-wave artefact walk and the Petri proxy's cells positional read (view_cells_of)
    cfg = tmp_path / f"sdfbase_full22_{KEY}_s3r.yaml"; cfg.write_text(yaml.safe_dump({"explicit_cells": [CELL]}))
    roster = tmp_path / "roster.yaml"; roster.write_text(yaml.safe_dump({"targets": {KEY: {"views": {"s3r": {"config": str(cfg), "cells": {CELL: STAGED}}}}}}))
    monkeypatch.setenv("F22_ROSTER", str(roster))
    rows = ST.unit_plan(KEY, "s3r", table=table, families=["multiturn", "static", "controls", "agentic", "gsm8k", "petri"], sha=SHA, clusters=["cluster_a", "cluster_b"])
    fams = sorted({r["family"] for r in rows})
    assert fams == ["agentic", "controls", "gsm8k", "multiturn", "petri", "static"] and not [r for r in rows if r["family"] == "bon"], fams
    targets = [r for r in rows if r["kind"] == "target"]
    assert all(any(str(a) in ("s3r", f"{KEY}_s3r") for a in r["args"]) for r in targets) and len([r for r in rows if r["kind"] == "attacker"]) == 2   # every target job is the view's
    assert sorted(r["unit"] for r in rows if r["family"] == "multiturn" and r["kind"] == "target") == ["actorattack/g01234", "actorattack/g5678", "crescendo/g01234", "crescendo/g5678", "pair/g01234", "pair/g5678"]
    assert sorted(r["unit"] for r in rows if r["family"] == "static") == ["t0", "t1"] and not any(r["script"].endswith("target_static.sbatch") for r in rows)
    g8 = [r for r in rows if r["family"] == "gsm8k"]
    assert len(g8) == 1 and g8[0]["args"][:3] == [KEY, "s3r", CELL]                                     # the ONE fitted cell, no none leg
    assert "none" not in json.dumps([r["args"] for r in targets])
    petri = next(r for r in rows if r["family"] == "petri")
    assert petri["args"] == [KEY, "s3r", SHA, "", CELL] and petri["env"]["F22_REGISTRY_CELLS"] == CELL   # the proxy declares the one measured cell
    with pytest.raises(ValueError, match="the bon family is EXCLUDED for this view by ruling"):
        ST.unit_plan(KEY, "s3r", table=table, families=["bon"], sha=SHA, clusters=["cluster_a", "cluster_b"])
    with pytest.raises(ValueError, match="the padding family is EXCLUDED"):
        ST.unit_plan(KEY, "s3r", table=table, families=["padding"], sha=SHA, clusters=["cluster_a", "cluster_b"])
    battery = ST.unit_plan(KEY, "battery", table=table, families=["static"], sha=SHA, clusters=["cluster_a", "cluster_b"])
    assert any(r["family"] == "bon" for r in battery)                                                      # battery's plan is as it was (bon not excluded there)
    assert ST.view_cells_of(KEY, "s3r") == [CELL]                                                       # s3r-fix: the config enumerates EXACTLY the cell — no base leg
