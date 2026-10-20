"""S2RLFIX6-PETRI: an overflow rerun drives a SUBSET of the first pass's frozen enumeration.

The drive reuses the first pass's out-root; its immutable cells.json froze the first proxy's legs and declaration. A rerun proxy
declares only the overflowed legs, so `_frozen_enumeration` derives a smaller payload — `cmd_run` used to compare the WHOLE payload
with the snapshot and refuse every leg as `enumeration drift` (never seen on s2r / s1, whose reruns declared the complete frozen set).
Through the ACTUAL rerun path (the real native enumeration, the real standup writer's records, the real record / standup checks; only
the serving runtime stubbed): three legs frozen under standup A; the rerun against standup B declaring two resolves exactly those two
by their FROZEN indices, the completed third leg and the record are untouched; without --overflow-rerun the same declaration is still
drift; a declaration naming a leg the first pass never froze, or a frozen leg whose identity moved, refuses by name.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tests"))

import petri_run as prun  # noqa: E402
import test_petri_store_shape as shape  # noqa: E402
from test_petri_run_serving_path import (  # noqa: E402,F401  (the stage-1 world fixture rides along)
    A32, B32, CELLS, LORA, NATIVE_IDENTITY, _Ran, _args, _bf, _real_standup_records, _source, _stage1_world, _wire)
from sisteer.capeval.tiers import petri_runner as pr  # noqa: E402

KEY = "synth_fd"          # a synthetic front-door key: the audit id's VIEW suffix is all the steered-wave grammar reads
NP = prun.SERVING_PATH_LORA_NATIVE
THREE = ["none", "prompted__safety", LORA]          # the first pass's declaration (three canonical legs; `none` brings the controls twin)
TWO = ["prompted__safety", LORA]                    # the rerun's: the two overflowed legs, a strict subset


def _world(tmp_path, monkeypatch, calls):
    source = _source(tmp_path / "bank")
    spec = shape._spec("ab" * 8, audit=f"petri_full22_{KEY}_s2rl", cells=list(CELLS))
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=calls)
    assert prun.steered_wave_audit(spec)
    return source, spec, native


def _rerun_args(out, standup, table, index):
    a = _args(out, serving_path=NP, standup=standup, table=table, cell_index=index)
    a.overflow_rerun = True
    return a


def _arm_ledger(out: Path, key: str) -> Path:
    """The cell's failure ledger with one context-overflow casualty — what --overflow-rerun arms on."""
    d = out / "cells" / key; d.mkdir(parents=True, exist_ok=True)
    (d / prun.FAILURE_LEDGER).write_text(json.dumps({
        "version": 1, "cell_id": "x", "petri_run_key": key, "written_at": 0.0,
        "unresolved": [{"behavior": "self_preservation", "scenario_id": "scenario_a", "epoch": 1, "dimension": "self_preservation",
                        "reason": pr.overflow_reason("maximum context length"), "retries": 0}]}), encoding="utf-8")
    return d


def _snapshot(root: Path, skip=()):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not any(str(p.relative_to(root)).startswith(s) for s in skip)}


def test_a_rerun_declaring_a_subset_of_the_frozen_legs_resolves_them_by_the_frozen_indices_and_leaves_the_rest_untouched(tmp_path, monkeypatch, capsys):
    calls = []
    source, spec, native = _world(tmp_path, monkeypatch, calls)
    out = tmp_path / "out"
    # the first pass: standup A declares three legs, print-cells freezes them (and the controls twin `none` brings)
    rec_a, table_a, _ = _real_standup_records(tmp_path / "standup_a", native, source, spec, requested=THREE, attempt=A32, monkeypatch=monkeypatch)
    assert prun.cmd_print_cells(_args(out, serving_path=NP, standup=rec_a, table=table_a), spec) == 0
    frozen = json.loads((out / prun.CELLS_JSON).read_text())
    ids = [c["cell_id"] for c in frozen["canonical"]]
    assert ids == THREE and frozen[prun.REQUESTED_HARNESS_KEY] == THREE and len(frozen["controls"]) == 1
    keys = {c["cell_id"]: c["petri_run_key"] for c in frozen["canonical"] + frozen["controls"]}
    # the first pass ran the two legs that will overflow (their manifests are the first pass's; the runtime seam raises after the
    # manifest is written, as it does in every serving-path test) and completed `none`, which filed its results on the first proxy
    for cid in TWO:
        with pytest.raises(_Ran):
            prun.cmd_run(_args(out, serving_path=NP, standup=rec_a, table=table_a, cell_index=ids.index(cid)), spec)
    assert [c["cell"] for c in calls] == TWO; calls.clear()
    done = out / "cells" / keys["none"]; done.mkdir(parents=True); (done / "results.json").write_text('{"done": true}', encoding="utf-8")
    for cid in TWO:
        _arm_ledger(out, keys[cid])                                  # the overflow casualties the first pass left in each ledger
    skip = tuple(f"cells/{keys[c]}" for c in TWO)
    frozen_bytes = (out / prun.CELLS_JSON).read_bytes(); before = _snapshot(out, skip=skip)
    # the rerun: standup B (a second attempt) declares the two overflowed legs — a strict subset of the frozen set
    rec_b, table_b, _ = _real_standup_records(tmp_path / "standup_b", native, source, spec, requested=TWO, attempt=B32, monkeypatch=monkeypatch)
    for cid in TWO:
        with pytest.raises(_Ran):                                   # the runtime seam: the leg RESOLVED and reached run_petri
            prun.cmd_run(_rerun_args(out, rec_b, table_b, ids.index(cid)), spec)
    assert [c["cell"] for c in calls] == TWO
    assert "is a subset of the frozen enumeration" in capsys.readouterr().out
    # the record is byte-identical, the completed leg and everything outside the two rerun cells untouched
    assert (out / prun.CELLS_JSON).read_bytes() == frozen_bytes and _snapshot(out, skip=skip) == before
    assert (done / "results.json").read_text(encoding="utf-8") == '{"done": true}'
    # the manifests the rerun wrote name the frozen identities (the index resolved against the record, not a remap)
    for cid in TWO:
        man = json.loads((out / "cells" / keys[cid] / "manifest.json").read_text())
        assert man["cell_id"] == cid and man["petri_run_key"] == keys[cid]
    # without --overflow-rerun the same two-cell declaration is still enumeration drift: the first-pass contract is unchanged
    with pytest.raises(prun.DriverError, match="enumeration drift"):
        prun.cmd_run(_args(out, serving_path=NP, standup=rec_b, table=table_b, cell_index=1), spec)
    assert [c["cell"] for c in calls] == TWO


def test_a_rerun_declaring_an_unfrozen_leg_or_a_frozen_leg_whose_identity_moved_refuses_by_name(tmp_path, monkeypatch):
    calls = []
    source, spec, native = _world(tmp_path, monkeypatch, calls)
    out = tmp_path / "out"
    rec_a, table_a, _ = _real_standup_records(tmp_path / "standup_a", native, source, spec, requested=TWO, attempt=A32, monkeypatch=monkeypatch)
    assert prun.cmd_print_cells(_args(out, serving_path=NP, standup=rec_a, table=table_a), spec) == 0     # the first pass froze TWO
    frozen = json.loads((out / prun.CELLS_JSON).read_text()); keys = {c["cell_id"]: c["petri_run_key"] for c in frozen["canonical"]}
    for cid in TWO:
        with pytest.raises(_Ran):
            prun.cmd_run(_args(out, serving_path=NP, standup=rec_a, table=table_a, cell_index=TWO.index(cid)), spec)
        _arm_ledger(out, keys[cid])
    calls.clear()
    # (1) a rerun declaring a leg the first pass never froze (`none`, with its controls twin): a rerun re-serves frozen legs, never a new one
    rec_c, table_c, _ = _real_standup_records(tmp_path / "standup_c", native, source, spec, requested=THREE, attempt=B32, monkeypatch=monkeypatch)
    with pytest.raises(prun.DriverError, match="is not in the frozen first-pass enumeration"):
        prun.cmd_run(_rerun_args(out, rec_c, table_c, 0), spec)
    # (2) a frozen leg whose identity moved: the record's copy of prompted__safety carries another harness hash → refused by name
    rec = json.loads((out / prun.CELLS_JSON).read_text()); rec["canonical"][0]["harness_config_hash"] = "f" * 16
    (out / prun.CELLS_JSON).write_text(json.dumps(rec), encoding="utf-8")
    rec_b, table_b, _ = _real_standup_records(tmp_path / "standup_b", native, source, spec, requested=TWO, attempt=B32, monkeypatch=monkeypatch)
    with pytest.raises(prun.DriverError, match="differs from its frozen identity in \\['harness_config_hash'\\]"):
        prun.cmd_run(_rerun_args(out, rec_b, table_b, 0), spec)
    assert calls == []                                                    # nothing ran, nothing remapped
    # the pure function, on records: an equal payload is no problem; a payload differing outside the leg lists is
    assert prun.overflow_rerun_subset_problem(frozen, {k: v for k, v in frozen.items() if k != "written_at"}, TWO, "p") is None
    assert "differs from the frozen first-pass record in 'expected_cells'" in prun.overflow_rerun_subset_problem(
        frozen, {**{k: v for k, v in frozen.items() if k != "written_at"}, "expected_cells": ["x"]}, TWO, "p")
