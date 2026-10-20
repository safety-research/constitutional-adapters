"""`petri_run.py legs-status`: the drive's OWN done-ness reading of every leg of a frozen enumeration (petri-completion, 2026-10-13).

The funnel's `--resubmit` proof (reproduce/lib/submit_twin.py petri_resume_proof), 06's overflow pass (reproduce/lib/petri_overflow.py)
and the drive's `ledgered` selector all need ONE answer per leg — "would a resumed drive regenerate this leg?" — read without a proxy, a
standup record, a GPU or a fresh enumeration. The subcommand answers with the SAME predicate `run` applies (`leg_already_done`: results.json
bound to its manifest, the directory named by the run key, the footing reconciled with the store over a settled ledger), the SAME record
reader and record<->manifest check, plus the ledger's unresolved and context-overflow counts (what `_arm_overflow_rerun` requires).
Synthetic world (test_petri_store_shape); nothing here is recorded data."""
from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "tests"))

import petri_run as prun  # noqa: E402
import test_petri_store_shape as shape  # noqa: E402
from test_petri_already_done import _complete_cell, _epochs  # noqa: E402
from test_petri_store_shape import world  # noqa: E402,F401

from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval.tiers import petri_runner as pr  # noqa: E402


def _record(world, cells, controls=(), **over):
    """A frozen enumeration record naming `cells` (PetriCell dicts) as canonical and `controls` as the controls legs, for the world's spec."""
    rec = {"audit": world.spec.audit, "expected_cells": list(world.spec.cells), "canonical": [dataclasses.asdict(c) for c in cells],
           "controls": [dataclasses.asdict(c) for c in controls], "serving_path": "lens"}
    rec.update(over)
    return rec


def _freeze(out_root: Path, rec: dict) -> Path:
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / prun.CELLS_JSON).write_text(json.dumps(rec), encoding="utf-8")
    return out_root


def _overflow_entry(sid):
    return {"behavior": "self_preservation", "scenario_id": sid, "epoch": 1, "dimension": "self_preservation",
            "reason": pr.overflow_reason("maximum context length"), "retries": 0}


def test_a_genuinely_complete_cell_reads_done(world, tmp_path, capsys):
    cell, store = _complete_cell(world, tmp_path)
    out = _freeze(tmp_path / "sweep", _record(world, [cell]))
    doc = prun.legs_status(out, world.spec)
    assert doc["record_refused"] is None and doc["audit"] == world.spec.audit and doc["serving_path"] == "lens"
    (leg,) = doc["legs"]
    assert leg["idx"] == 0 and leg["cell_id"] == cell.cell_id and leg["petri_run_key"] == cell.petri_run_key
    assert leg["done"] is True and leg["refused"] is False and leg["results_present"] is True
    assert leg["unresolved"] == 0 and leg["overflow_unresolved"] == 0 and leg["cell_dir"] == str(store.cell_dir)
    assert "already done" not in capsys.readouterr().out                       # the already-done print is suppressed (one JSON document on stdout)


def test_a_results_json_over_an_unresolved_ledger_is_refused_as_run_would(world, tmp_path):
    """The H9 refusal `run` makes: a results.json whose ledger still holds unresolved entries is no done marker — `refused`, never "not done"."""
    cell, store = _complete_cell(world, tmp_path)
    ledger = json.loads(pr.ledger_path(store).read_text(encoding="utf-8"))
    ledger["unresolved"] = [{"behavior": "self_preservation", "scenario_id": shape.SIDS[0], "epoch": 1, "dimension": "scenario_realism",
                             "reason": "synthetic", "retries": 0}]
    pr.ledger_path(store).write_text(json.dumps(ledger), encoding="utf-8")
    out = _freeze(tmp_path / "sweep", _record(world, [cell]))
    (leg,) = prun.legs_status(out, world.spec)["legs"]
    assert leg["done"] is False and leg["refused"] is True and "unresolved failure-ledger" in leg["reason"]
    assert leg["results_present"] is True and leg["unresolved"] == 1 and leg["overflow_unresolved"] == 0


def test_an_overflow_ledger_without_results_reads_ledgered(world, tmp_path):
    cell = world.cells["si_safety__psr_kl_all"]
    out = _freeze(tmp_path / "sweep", _record(world, [cell]))
    store = cs.CellStore(out, cell.petri_run_key, recorder=cs.NO_RECORDER)
    store.cell_dir.mkdir(parents=True)
    (store.cell_dir / "manifest.json").write_text(json.dumps({"cell_id": cell.cell_id, "petri_run_key": cell.petri_run_key}), encoding="utf-8")
    (store.cell_dir / prun.FAILURE_LEDGER).write_text(json.dumps({"version": 1, "cell_id": cell.cell_id, "petri_run_key": cell.petri_run_key, "written_at": 0.0,
                                                                  "unresolved": [_overflow_entry("scenario_a"), _overflow_entry("scenario_b"),
                                                                                 {**_overflow_entry("scenario_c"), "reason": "sample error (judge)"}]}), encoding="utf-8")
    (leg,) = prun.legs_status(out, world.spec)["legs"]
    assert leg["done"] is False and leg["refused"] is False and leg["reason"] is None and leg["results_present"] is False
    assert leg["unresolved"] == 3 and leg["overflow_unresolved"] == 2                # the converter's prefix classifies, as _arm_overflow_rerun does
    # a leg with no store at all: not done, nothing unresolved
    other = world.cells["none"]
    out2 = _freeze(tmp_path / "sweep2", _record(world, [other]))
    (leg2,) = prun.legs_status(out2, world.spec)["legs"]
    assert leg2["done"] is False and leg2["refused"] is False and leg2["unresolved"] == 0 and leg2["overflow_unresolved"] == 0 and leg2["results_present"] is False


def test_a_copied_directory_under_another_legs_key_is_refused(world, tmp_path):
    cell, store = _complete_cell(world, tmp_path)
    other = dataclasses.replace(cell, petri_run_key="0" * 16)
    import shutil
    shutil.copytree(store.cell_dir, store.cell_dir.parent / ("0" * 16))
    out = _freeze(tmp_path / "sweep", _record(world, [other]))
    (leg,) = prun.legs_status(out, world.spec)["legs"]
    assert leg["done"] is False and leg["refused"] is True and "copied or renamed" in leg["reason"] and leg["results_present"] is True


def test_no_frozen_record_refuses_by_name_and_the_cli_exits_2(world, tmp_path, capsys):
    with pytest.raises(prun.DriverError, match="no frozen enumeration .* nothing is provably done"):
        prun.legs_status(tmp_path / "nowhere", world.spec)
    args = SimpleNamespace(out_root=str(tmp_path / "nowhere"))
    assert prun.cmd_legs_status(args, world.spec) == prun.LEGS_STATUS_RECORD_MISSING_RC == 2
    cap = capsys.readouterr()
    assert cap.out == "" and "REFUSED: legs-status" in cap.err
    # the real CLI, with a real manifest of the tree and an empty out root: exit 2, the reason on stderr, nothing on stdout
    manifest = REPO / "configs/capeval/manifests/petri_full22_qwen35_4b_demo_battery.yaml"
    r = subprocess.run([sys.executable, str(REPO / "scripts/petri_run.py"), "--manifest", str(manifest), "legs-status", "--out-root", str(tmp_path / "empty")],
                       capture_output=True, text=True, timeout=600, env={**dict(__import__("os").environ), "PYTHONPATH": str(REPO / "src")})
    assert r.returncode == 2 and "nothing is provably done" in r.stderr and r.stdout.strip() == "", r.stdout[-500:] + r.stderr[-800:]


def test_stdout_is_one_json_document(world, tmp_path, capsys):
    cell, _store = _complete_cell(world, tmp_path)
    capsys.readouterr()                                                   # the runner's own lines from building the cell, not legs-status's
    out = _freeze(tmp_path / "sweep", _record(world, [cell]))
    assert prun.cmd_legs_status(SimpleNamespace(out_root=str(out)), world.spec) == 0
    cap = capsys.readouterr()
    doc = json.loads(cap.out)
    assert doc["legs"][0]["done"] is True and "already done" not in cap.out
    assert "legs-status" in prun.COMMANDS and prun.COMMANDS["legs-status"] is prun.cmd_legs_status
    ns = prun.build_parser().parse_args(["--manifest", "m.yaml", "legs-status", "--out-root", "x"])
    assert ns.cmd == "legs-status" and ns.out_root == "x"


def test_a_record_the_resumed_drive_would_refuse_is_reported_and_every_leg_is_refused(world, tmp_path):
    cell, _store = _complete_cell(world, tmp_path)
    out = _freeze(tmp_path / "sweep", _record(world, [cell], audit="another-audit"))
    doc = prun.legs_status(out, world.spec)
    assert doc["record_refused"] and "foreign enumeration record" in doc["record_refused"]
    (leg,) = doc["legs"]
    assert leg["done"] is False and leg["refused"] is True and "the frozen record is refused" in leg["reason"]
    assert leg["results_present"] is True and leg["unresolved"] == 0        # the store facts are still read


def test_legs_are_reported_in_record_order_canonical_then_controls(world, tmp_path):
    """The index legs-status reports is the drive's and `run --cell-index`'s: canonical legs in record order, then the controls."""
    cells = [world.cells["none"], world.cells["prompted__safety"]]
    control = dataclasses.replace(world.cells["none"], cell_id="none__lens0", control_of="none", petri_run_key="c" * 16)
    out = _freeze(tmp_path / "sweep", _record(world, cells, controls=[control]))
    doc = prun.legs_status(out, world.spec)
    assert [(l["idx"], l["cell_id"], l["control_of"]) for l in doc["legs"]] == [(0, "none", None), (1, "prompted__safety", None), (2, "none__lens0", "none")]
    assert all(l["done"] is False and l["refused"] is False for l in doc["legs"])


def test_already_done_keeps_its_print_and_leg_already_done_can_stay_quiet(world, tmp_path, capsys):
    cell, store = _complete_cell(world, tmp_path)
    assert prun.leg_already_done(store, cell, _epochs(world), say=False) is True
    assert "already done" not in capsys.readouterr().out            # the reconcile's own footing line ([petri_runner] …) is not this print
    assert prun._already_done(store, cell, _epochs(world)) is True
    assert f"cell {cell.cell_id} already done" in capsys.readouterr().out
