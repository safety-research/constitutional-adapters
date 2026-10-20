"""The overflow re-run pass of a BASELINE view's audit declares a SUBSET of the frozen complete baseline (petri-completion, 2026-10-13).

A rerun proxy serves only the overflowed cells, so its standup declares a subset — but a baseline
view's audit is non-steered, and the completeness check (`complete_baseline_problem`, written so a FREEZE can never narrow the
baseline) ran on ANY declaration before `cmd_run` ever reached the subset branch: every leg of a declared-subset overflow drive on the
demo's `battery` view would have refused. `_frozen_enumeration(..., rerun_subset_of=record)` — passed by `cmd_run` ONLY under
`--overflow-rerun` — skips that check iff the frozen record is itself a complete-baseline record (declares nothing, or exactly its
canonical ids), requires the live declaration to be a non-empty subset of the frozen canonical ids, and leaves `restrict_to_requested`
(the controls twin only when `none` is declared) and the existing subset check untouched. `print-cells` and a plain `run` keep the
refusal byte for byte. Through the REAL path (the real native enumeration, the real standup writer's records, the real record / standup
checks; only the serving runtime stubbed) on a NON-steered synthetic front-door audit."""
from __future__ import annotations

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
from test_petri_overflow_rerun_subset_of_frozen import _arm_ledger, _rerun_args  # noqa: E402
from test_petri_run_serving_path import (  # noqa: E402,F401  (the stage-1 world fixture rides along)
    A32, B32, CELLS, LORA, PSR, NATIVE_IDENTITY, _Ran, _args, _bf, _real_standup_records, _source, _stage1_world, _wire)
from sisteer.capeval.petri import defense_cells as dc  # noqa: E402

AUDIT = "petri_full22_synth_fd_battery"     # a synthetic front-door key's single-phase wave: NOT a steered audit (no _s<N>r suffix, no dose tail)
NP = prun.SERVING_PATH_LORA_NATIVE
CONTROL = dc.NOOP_LORA_CELL_ID              # the native controls twin (none__lora0), control_of none


def _world(tmp_path, monkeypatch, calls):
    source = _source(tmp_path / "bank")
    spec = shape._spec("ab" * 8, audit=AUDIT, cells=list(CELLS))
    native = prun.enumerate_petri_native(spec, source, behavior_files=_bf(spec), prompt_files=shape.PROMPT_FILES)
    _wire(monkeypatch, source, identity=NATIVE_IDENTITY, calls=calls)
    assert not prun.steered_wave_audit(spec)
    return source, spec, native


def _first_pass(tmp_path, monkeypatch, calls, source, spec, native):
    """The first pass: a standup declaring NOTHING, the complete baseline frozen (none, prompted, the LoRA leg + the controls twin); the LoRA
    leg and the twin were started (their manifests are the first pass's) and left on overflow ledgers; `none` completed."""
    out = tmp_path / "out"
    rec_a, table_a, _ = _real_standup_records(tmp_path / "standup_a", native, source, spec, requested=[], attempt=A32, write_endpoint=False, monkeypatch=monkeypatch)
    assert prun.cmd_print_cells(_args(out, serving_path=NP, standup=rec_a, table=table_a), spec) == 0
    frozen = json.loads((out / prun.CELLS_JSON).read_text())
    ids = [c["cell_id"] for c in frozen["canonical"]] + [c["cell_id"] for c in frozen["controls"]]
    assert ids == ["none", "prompted__safety", LORA, CONTROL] and prun.REQUESTED_HARNESS_KEY not in frozen
    assert prun.frozen_record_is_complete_baseline(frozen)
    keys = {c["cell_id"]: c["petri_run_key"] for c in frozen["canonical"] + frozen["controls"]}
    for cid in (LORA, CONTROL):
        with pytest.raises(_Ran):
            prun.cmd_run(_args(out, serving_path=NP, standup=rec_a, table=table_a, cell_index=ids.index(cid)), spec)
        _arm_ledger(out, keys[cid])
    assert [c["cell"] for c in calls] == [LORA, CONTROL]; calls.clear()
    done = out / "cells" / keys["none"]; done.mkdir(parents=True); (done / "results.json").write_text('{"done": true}', encoding="utf-8")
    return out, frozen, ids, keys


def test_a_rerun_declaring_a_subset_plus_none_of_a_frozen_complete_baseline_drives_the_ledgered_leg_and_the_twin(tmp_path, monkeypatch, capsys):
    calls = []
    source, spec, native = _world(tmp_path, monkeypatch, calls)
    out, frozen, ids, keys = _first_pass(tmp_path, monkeypatch, calls, source, spec, native)
    frozen_bytes = (out / prun.CELLS_JSON).read_bytes()
    # (a) the rerun proxy declares the overflowed LoRA cell + `none` (the twin's parent): the completeness check is skipped for a
    # rerun of a complete-baseline record, the subset is held to the frozen identities, the legs resolve by their FROZEN indices
    rec_b, table_b, _ = _real_standup_records(tmp_path / "standup_b", native, source, spec, requested=["none", LORA], attempt=B32, monkeypatch=monkeypatch)
    with pytest.raises(_Ran):
        prun.cmd_run(_rerun_args(out, rec_b, table_b, ids.index(LORA)), spec)
    with pytest.raises(_Ran):                                   # the controls twin is INSIDE the attested set because `none` is declared
        prun.cmd_run(_rerun_args(out, rec_b, table_b, ids.index(CONTROL)), spec)
    assert [c["cell"] for c in calls] == [LORA, CONTROL]
    cap = capsys.readouterr().out
    assert "the completeness check is for a freeze, and a rerun freezes nothing" in cap and "is a subset of the frozen enumeration" in cap
    assert (out / prun.CELLS_JSON).read_bytes() == frozen_bytes                      # nothing re-frozen
    for cid in (LORA, CONTROL):
        man = json.loads((out / "cells" / keys[cid] / "manifest.json").read_text())
        assert man["cell_id"] == cid and man["petri_run_key"] == keys[cid]
    assert (out / "cells" / keys["none"] / "results.json").read_text(encoding="utf-8") == '{"done": true}'
    calls.clear()
    # (b) the same records WITHOUT --overflow-rerun: the first-pass contract, byte for byte — a partial declaration on a baseline audit refuses
    with pytest.raises(prun.DriverError, match="u0 / legacy audit whose legs are the whole baseline.*declare the complete baseline exactly"):
        prun.cmd_run(_args(out, serving_path=NP, standup=rec_b, table=table_b, cell_index=ids.index(LORA)), spec)
    # (c) print-cells with the partial declaration: the FREEZE is untouched by the exception
    with pytest.raises(prun.DriverError, match="declare the complete baseline exactly"):
        prun.cmd_print_cells(_args(tmp_path / "out_c", serving_path=NP, standup=rec_b, table=table_b), spec)
    assert not (tmp_path / "out_c" / prun.CELLS_JSON).exists() and calls == []


def test_a_declaration_outside_the_frozen_legs_refuses_by_name_before_any_leg(tmp_path, monkeypatch):
    calls = []
    source, spec, native = _world(tmp_path, monkeypatch, calls)
    out, frozen, ids, keys = _first_pass(tmp_path, monkeypatch, calls, source, spec, native)
    # (d) the PSR cell is in the manifest's universe but was never a frozen leg (skipped by name under the native path): a rerun may only
    # re-serve frozen legs — refused by the baseline-subset rule, before restrict_to_requested and before any leg
    rec_d, table_d, _ = _real_standup_records(tmp_path / "standup_d", native, source, spec, requested=["none", PSR], attempt=B32, monkeypatch=monkeypatch)
    with pytest.raises(prun.DriverError, match="names \\['si_safety__psr_kl_all'\\], not among the frozen first-pass legs"):
        prun.cmd_run(_rerun_args(out, rec_d, table_d, ids.index(LORA)), spec)
    assert calls == []


def test_a_declaration_without_none_keeps_no_controls_and_the_predicate_pins_the_complete_baseline_shape(tmp_path, monkeypatch):
    calls = []
    source, spec, native = _world(tmp_path, monkeypatch, calls)
    out, frozen, ids, keys = _first_pass(tmp_path, monkeypatch, calls, source, spec, native)
    # (e) declaring the LoRA cell alone: the live set has NO controls (restrict_to_requested keeps the twin only when `none` is declared) — why
    # 06's overflow pass declares `none` for a ledgered control leg; the LoRA leg itself still runs under the subset rule
    rec_e, table_e, _ = _real_standup_records(tmp_path / "standup_e", native, source, spec, requested=[LORA], attempt=B32, monkeypatch=monkeypatch)
    a = _rerun_args(out, rec_e, table_e, ids.index(LORA))
    cell_set, payload, _native, requested = prun._frozen_enumeration(a, spec, NP, rerun_subset_of=frozen)
    assert requested == [LORA] and [c.cell_id for c in cell_set.canonical] == [LORA] and cell_set.controls == ()
    assert payload[prun.REQUESTED_HARNESS_KEY] == [LORA]
    with pytest.raises(_Ran):
        prun.cmd_run(a, spec)
    assert [c["cell"] for c in calls] == [LORA]
    # (f) a frozen record that itself declares a strict subset (a steered shape) is NOT a complete-baseline record: the exception is not
    # consulted and the non-steered audit's completeness check decides as before
    subset_rec = {**frozen, prun.REQUESTED_HARNESS_KEY: [LORA]}
    assert not prun.frozen_record_is_complete_baseline(subset_rec)
    with pytest.raises(prun.DriverError, match="declare the complete baseline exactly"):
        prun._frozen_enumeration(a, spec, NP, rerun_subset_of=subset_rec)
    # the predicate on records: nothing declared, exactly the canonical ids (any order), [] -> complete; a subset, a superset, a non-list -> not
    canon = [c["cell_id"] for c in frozen["canonical"]]
    assert prun.frozen_record_is_complete_baseline({**frozen, prun.REQUESTED_HARNESS_KEY: list(reversed(canon))})
    assert prun.frozen_record_is_complete_baseline({**frozen, prun.REQUESTED_HARNESS_KEY: []})
    assert not prun.frozen_record_is_complete_baseline({**frozen, prun.REQUESTED_HARNESS_KEY: canon + ["extra"]})
    assert not prun.frozen_record_is_complete_baseline({**frozen, prun.REQUESTED_HARNESS_KEY: "none"})
    # and with no rerun record at all the plain path is unchanged: the partial declaration refuses
    with pytest.raises(prun.DriverError, match="declare the complete baseline exactly"):
        prun._frozen_enumeration(a, spec, NP)
