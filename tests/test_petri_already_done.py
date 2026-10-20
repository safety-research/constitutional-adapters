"""The driver's already-done path reconciles a done marker the way every
collector does (merge review 2026-09-07).

``run`` used to accept any results.json whose failure ledger merely lacked
unresolved entries: no manifest, no cell/run identity, no footing, no store
evidence -- a bare ``{"unscored": 0}`` passed and the leg reported "already
done", while ``bind_cell_dirs`` / ``_reconcile_done_marker`` refused the very
same cell at consolidate. One routine now: ``_bind_cell_dir`` +
``_reconcile_cell_dir``, run by ``_already_done`` and by every collector's
walk. Synthetic world (test_petri_store_shape); nothing here is recorded data.
"""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import petri_run as prun  # noqa: E402
import test_petri_store_shape as shape  # noqa: E402
from test_petri_overflow_rerun import _wire_cmd_run  # noqa: E402
from test_petri_store_shape import world  # noqa: E402,F401

from sisteer.capeval import store as cs  # noqa: E402
from sisteer.capeval.tiers import petri_runner as pr  # noqa: E402

LEGACY_BEHAVIOR_KEYS = ("mean", "n_scenarios", "rows_filed", "unresolved",
                        "incomplete")  # the pre-footing per-behavior shape


def _epochs(world):
    return world.spec.bloom_options.epochs


def _complete_cell(world, tmp_path):
    """A genuinely COMPLETE cell written by the runner itself, under the
    driver's layout (``cells/<run_key>/``, the key ``cmd_run`` opens the
    store with): manifest, rows, an identity-bearing settled ledger and a
    footed results.json."""
    cell = world.cells["si_safety__psr_kl_all"]
    store = cs.CellStore(tmp_path / "sweep", cell.petri_run_key)
    result = pr._run(
        object(), world.spec, cell, store,
        seed=world.spec.bloom_options.seed,
        driver=shape._driver(world, cell),
        config_root=world.config_root, bundle_dir=world.bundle.bundle_dir,
        work_root=tmp_path / "work", log_root=tmp_path / "logs",
        prompt_files=shape.PROMPT_FILES)
    assert store.is_done() and result["incomplete"] is False
    return cell, store


def _record_of(cell):
    """The frozen enumeration record bind_cell_dirs binds against."""
    return {"canonical": [{"cell_id": cell.cell_id,
                           "petri_run_key": cell.petri_run_key}],
            "controls": []}


def test_already_done_accepts_a_genuinely_complete_cell(world, tmp_path, capsys):
    cell, store = _complete_cell(world, tmp_path)
    assert prun._already_done(store, cell, _epochs(world)) is True
    assert f"cell {cell.cell_id} already done" in capsys.readouterr().out
    # the same cell, the same verdict, through the collectors' walk
    bound = prun.bind_cell_dirs(tmp_path / "sweep", _record_of(cell),
                                _epochs(world))
    assert bound[cell.cell_id].results is not None


def test_already_done_is_false_without_a_done_marker(world, tmp_path):
    cell = world.cells["si_safety__psr_kl_all"]
    store = cs.CellStore(tmp_path / "sweep", cell.petri_run_key)
    assert prun._already_done(store, cell, _epochs(world)) is False


def test_already_done_refuses_a_bare_unscored_marker(tmp_path):
    """The fixture the old path accepted: ``{"unscored": 0}`` alone -- no
    manifest, no ledger, no rows, no footing."""
    store = cs.CellStore(tmp_path / "sweep", "ab" * 8)
    store.write_results({"unscored": 0})
    cell = SimpleNamespace(cell_id="cell", petri_run_key="ab" * 8)
    with pytest.raises(prun.DriverError, match="no manifest.json"):
        prun._already_done(store, cell, 1)


def test_already_done_refuses_a_missing_ledger(world, tmp_path):
    cell, store = _complete_cell(world, tmp_path)
    pr.ledger_path(store).unlink()
    with pytest.raises(prun.DriverError, match=f"no {prun.FAILURE_LEDGER}"):
        prun._already_done(store, cell, _epochs(world))


def test_already_done_refuses_an_identity_free_ledger(world, tmp_path):
    cell, store = _complete_cell(world, tmp_path)
    pr.ledger_path(store).write_text(
        json.dumps({"version": 1, "unresolved": []}), encoding="utf-8")
    with pytest.raises(prun.DriverError, match="names its cell and run"):
        prun._already_done(store, cell, _epochs(world))


def test_already_done_refuses_a_stale_ledger_under_the_marker(world, tmp_path):
    """The one refusal the old path DID make (H9), still made -- now by the
    shared reconcile rather than a private ledger read."""
    cell, store = _complete_cell(world, tmp_path)
    ledger = json.loads(pr.ledger_path(store).read_text(encoding="utf-8"))
    ledger["unresolved"] = [{"behavior": "self_preservation",
                             "scenario_id": shape.SIDS[0], "epoch": 1,
                             "dimension": "scenario_realism",
                             "reason": "synthetic", "retries": 0}]
    pr.ledger_path(store).write_text(json.dumps(ledger), encoding="utf-8")
    with pytest.raises(prun.DriverError, match="unresolved failure-ledger"):
        prun._already_done(store, cell, _epochs(world))


def test_already_done_refuses_a_legacy_pre_footing_results(world, tmp_path):
    cell, store = _complete_cell(world, tmp_path)
    results = json.loads(store.results_path.read_text(encoding="utf-8"))
    assert all(k in results for k in prun.FOOTING_KEYS)
    legacy = {k: v for k, v in results.items() if k not in prun.FOOTING_KEYS}
    legacy["behaviors"] = {
        b: {k: v for k, v in e.items() if k in LEGACY_BEHAVIOR_KEYS}
        for b, e in results["behaviors"].items()}
    store.write_results(legacy)
    with pytest.raises(prun.DriverError, match="no 'audited' footing"):
        prun._already_done(store, cell, _epochs(world))
    # the same cell, the same refusal, through the collectors' walk: ONE routine
    with pytest.raises(prun.DriverError, match="no 'audited' footing"):
        prun.bind_cell_dirs(tmp_path / "sweep", _record_of(cell),
                            _epochs(world))


def test_already_done_refuses_another_legs_directory(world, tmp_path):
    cell, store = _complete_cell(world, tmp_path)
    other = SimpleNamespace(cell_id=cell.cell_id, petri_run_key="0" * 16)
    with pytest.raises(prun.DriverError, match="another leg's"):
        prun._already_done(store, other, _epochs(world))


def test_cmd_run_refuses_a_bare_done_marker_before_launching(tmp_path,
                                                             monkeypatch):
    """End to end through cmd_run: a results.json with nothing behind it is
    no longer "already done" -- the leg refuses before any engine is made
    (the old contract printed "already done" and returned 0)."""
    launched = []

    def never_run_petri(*_a, **_k):
        launched.append(True)
        return {"unscored": 0}

    args, spec, cell_dir = _wire_cmd_run(tmp_path, monkeypatch, never_run_petri)
    args.overflow_rerun = False
    cell_dir.mkdir(parents=True)
    (cell_dir / "results.json").write_text(json.dumps({"unscored": 0}),
                                           encoding="utf-8")
    monkeypatch.setattr(prun.cs.CellStore, "is_done", lambda self: True)
    with pytest.raises(prun.DriverError, match="no manifest.json"):
        prun.cmd_run(args, spec)
    assert launched == []


# --- the RE-KEY guard: a harness change must not silently re-audit finished cells -------
#
# petri_run_key is content-addressed over the whole harness config, so any harness change
# (2026-09-15: the abliterated-Qwen auditor leg left the chain) gives every cell a fresh,
# empty namespace. Correct for provenance, dangerous operationally: a roster-wide launch
# under the new config sees every finished cell as absent. These pin the detector's four
# cases, including the three where it must stay QUIET -- a guard that fires on everything
# is as bad as one that never fires.

def _rekey_store(tmp_path, *, prior_done=True, prior_harness="OLDHASH000000000",
                 prior_cell="prompted__constitution", prior_iv="aaaaaaaaaaaaaaaa"):
    """A cells/ root holding one PRIOR cell dir plus the empty dir this launch would use."""
    cells = tmp_path / "cells"
    old = cells / "oldrunkey00000000"
    old.mkdir(parents=True)
    if prior_done:
        (old / "results.json").write_text("{}")
    (old / "manifest.json").write_text(json.dumps({
        "cell_id": prior_cell, "intervention_hash": prior_iv,
        "harness_config_hash": prior_harness, "petri_run_key": "oldrunkey00000000"}))
    new = cells / "newrunkey0000000"
    new.mkdir(parents=True)
    return SimpleNamespace(cell_dir=new, is_done=lambda: (new / "results.json").exists())


def _rekey_cell(**kw):
    base = dict(cell_id="prompted__constitution", intervention_hash="aaaaaaaaaaaaaaaa",
                harness_config_hash="NEWHASH000000000", petri_run_key="newrunkey0000000")
    base.update(kw)
    return SimpleNamespace(**base)


def test_rekey_guard_detects_a_completed_audit_under_a_different_harness(tmp_path):
    store = _rekey_store(tmp_path)
    found, manifest = pr.prior_audit_under_other_harness(store, _rekey_cell())
    assert found is not None
    assert manifest["petri_run_key"] == "oldrunkey00000000"
    assert manifest["harness_config_hash"] == "OLDHASH000000000"


def test_rekey_guard_is_quiet_when_the_harness_is_unchanged(tmp_path):
    """A genuine resume: same harness, same key -- the guard must not block it."""
    store = _rekey_store(tmp_path)
    found, _ = pr.prior_audit_under_other_harness(
        store, _rekey_cell(harness_config_hash="OLDHASH000000000"))
    assert found is None


def test_rekey_guard_is_quiet_when_the_prior_audit_is_incomplete(tmp_path):
    """results.json is the ONLY done marker: an unfinished prior is not a finished audit."""
    store = _rekey_store(tmp_path, prior_done=False)
    found, _ = pr.prior_audit_under_other_harness(store, _rekey_cell())
    assert found is None


def test_rekey_guard_is_quiet_for_a_different_intervention(tmp_path):
    """A different object is a different cell, however the harness moved."""
    store = _rekey_store(tmp_path)
    found, _ = pr.prior_audit_under_other_harness(
        store, _rekey_cell(intervention_hash="bbbbbbbbbbbbbbbb"))
    assert found is None
