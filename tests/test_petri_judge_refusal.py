"""Petri: a JUDGE refusal is a per-sample TERMINAL outcome — reported, never
imputed — the twin of the auditor-refusal machinery (petri_runner's
judge-refusals comment, by ``JUDGE_REFUSED_REASON_PREFIX``).

The bug (a real cell, 2026-09-08): sixty audits finished; the pinned judge's
content filter declined ONE transcript, inspect recorded a NaN scorer value
with ``metadata.refusal`` set (and no sample error), and the converter's
shape refusal on that NaN raised for the WHOLE cell — fifty-nine scored
samples held unread behind one judge refusal. Now the converter files the
refused sample's required dimensions under the machine-checkable reason, the
runner folds them to one ``JudgeRefusal`` in ``petri_judge_refused.json``,
its keys leave the completeness contract (never an unresolved casualty, never
retried), and every footing reports ``judge_refused`` beside
``auditor_refused`` so the cell is visibly N-1/N with the reason named.

Two conditions are pinned here:

A. reported, never imputed — a cell with one judge refusal is 3/4 with the
   item listed under ``judge_refused_items`` and its reason, in the runner's
   summary, results.json, the sidecar, the footing line and the consolidation
   validator; never 4/4, never silently absent.
B. provably inert on the path that already worked — the inertness harness
   re-converts an already-converted REAL cell's eval logs through the new
   converter and asserts every row's (task, item_id, metric, value) equals
   the rows.jsonl the earlier converter filed (skipped when that cell is not
   on this machine, so CI is unaffected).

Synthetic fixtures are the test_petri_store_shape world. The real-data tests
read ``~/.local/state/petri`` READ-ONLY, and never through ``CellStore``
(whose row reader may rewrite a torn final line). The stats-side inertness
check on a COMPLETE cell reads the committed fixture cell under
``tests/fixtures/petri_complete_cell`` instead (synthetic, written through
the runner by ``_write_complete_cell_fixture``), so it runs on any box.
"""

import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from sisteer.capeval import manifest as cm
from sisteer.capeval import store as cs
from sisteer.capeval.petri import defense_cells as dc
from sisteer.capeval.petri import petri_stats
from sisteer.capeval.petri import scenarios as psc
from sisteer.capeval.tiers import petri_runner as pr
from test_petri_store_shape import (  # noqa: F401  (world is a fixture)
    BEHAVIORS,
    PROMPT_FILES,
    SIDS,
    _build_world,
    _driver,
    _run_cell,
    _source,
    convert,
    full_dims,
    make_log,
    sample,
    world,
)
from test_petri_stats import (  # the synthetic stats world + its footing helpers
    BEHAVIORS3,
    LENS,
    RUN_KEY,
    _analyze,
    _cell_footing,
    _enumeration_payload,
    _provenance_for,
    build_world,
)

ROOT = Path(__file__).resolve().parents[1]
#: the worktree HEAD this work was built on: petri_stats.py there has NO
#: judge term (the inertness tests load that version and compare)
PRE_CHANGE_SHA = "7bada2c4"
if "petri_run" in sys.modules:
    petri_run = sys.modules["petri_run"]
else:
    _loader = importlib.util.spec_from_file_location(
        "petri_run", ROOT / "scripts" / "petri_run.py")
    petri_run = importlib.util.module_from_spec(_loader)
    sys.modules["petri_run"] = petri_run
    _loader.loader.exec_module(petri_run)

#: the text inspect-scout puts in the score explanation when the judge's
#: content filter declines (verified on the real log)
REFUSAL_TEXT = "Scanner request refused by content filter: "

# --- the real cells the harness reads (READ-ONLY; skipped when absent) -------------
REAL_AUDIT = "petri_full22_glm46vflash"
REAL_ROOT = Path.home() / ".local" / "state" / "petri" / REAL_AUDIT
REAL_MANIFEST = ROOT / "configs" / "capeval" / "manifests" / f"{REAL_AUDIT}.yaml"
#: cell `none`: COMPLETE (every item scored) and already converted — the
#: path that worked before this change
REAL_COMPLETE_KEY = "49f803a31ac66a40"
#: the control cell whose conversion crashed three times on ONE judge refusal
REAL_REFUSED_KEY = "215d74a1ca0b7e15"
#: the COMPLETE synthetic cell the stats-side inertness check reads instead of
#: the real one (hermetic): a petri state root laid
#: out like REAL_ROOT — cells.json + cells/<run_key>/{rows.jsonl, results.json,
#: manifest.json} — written THROUGH the product runner on the
#: test_petri_store_shape world by ``_write_complete_cell_fixture``, plus a
#: fixture.json naming the behaviors, scenario ids and epochs the denominator
#: is derived from
FIXTURE_CELL_ROOT = ROOT / "tests" / "fixtures" / "petri_complete_cell"
#: the one wall-clock stamp every timestamp in that fixture is pinned to, so
#: a regeneration is byte-reproducible
FIXTURE_TIME = 1_700_000_000.0


def refused_sample(sid, *, flag=True, value=float("nan"), text=REFUSAL_TEXT,
                   epoch=1, metadata=None):
    """One synthetic inspect sample in the judge-refused shape the real log
    carries: a finished transcript (no sample error), one scorer entry whose
    value is NaN, answer None, the refusal text as the explanation and
    ``metadata.refusal`` set. ``metadata`` overrides the whole mapping."""
    md = ({"refusal": flag, "scanner_references": []}
          if metadata is None else metadata)
    return {
        "id": sid, "epoch": epoch, "error": None, "uuid": f"uuid-{sid}",
        "scores": {"audit_judge": {"value": value, "answer": None,
                                   "explanation": text, "metadata": md,
                                   "history": []}},
    }


def _driver_with_one_judge_refusal(world, cell, *, refused=(BEHAVIORS[1], SIDS[1])):
    """A fake inspect driver: every item scored, except that ONE finished
    transcript the judge refused to score."""
    def drive(tasks, *, spec, cell, engine):
        logs = []
        for t in tasks:
            samples = []
            for sid in (t.scenario_ids if t.scenario_ids is not None else SIDS):
                if (t.behavior_id, sid) == refused:
                    samples.append(refused_sample(sid))
                else:
                    samples.append(sample(sid, full_dims(t.behavior_id)))
            log = make_log(world, cell, t.dir_key, samples, spec=spec)
            log["location"] = f"/synthetic/{t.behavior_id}.eval"
            logs.append(log)
        return logs
    return drive


# --- (i) the converter: NaN + refusal marker -> judge-refused ledger entries, no raise


@pytest.mark.parametrize("flag", [True, "True", "true"])
def test_nan_with_refusal_marker_files_judge_refused_entries(world, flag):
    cell = world.cells["si_safety__psr_kl_all"]
    b = BEHAVIORS[0]
    rows, ledger = convert(world, cell, b, [
        refused_sample(SIDS[0], flag=flag),
        sample(SIDS[1], full_dims(b)),
    ])
    # the scored sibling is unaffected: its rows are filed (before the fix
    # the NaN raised for the whole log and none of them were)
    assert {(r.item_id, r.metric) for r in rows} == {
        (SIDS[1], m) for m in pr.required_metrics(b)}
    refused = [e for e in ledger if e.scenario_id == SIDS[0]]
    assert [e for e in ledger if e.scenario_id != SIDS[0]] == []
    # every required dimension of the refused sample, each under the
    # machine-checkable reason naming the judge and its refusal text
    assert {e.dimension for e in refused} == set(pr.required_metrics(b))
    assert len(refused) == len(pr.required_metrics(b))
    expected_reason = pr.judge_refused_reason(world.spec.judge_model,
                                              REFUSAL_TEXT.strip())
    for e in refused:
        assert e.reason.startswith(pr.JUDGE_REFUSED_REASON_PREFIX)
        assert pr.is_judge_refused_reason(e.reason)
        assert e.reason == expected_reason
        assert world.spec.judge_model in e.reason
        assert REFUSAL_TEXT.strip() in e.reason
        assert e.shed is None  # not a shed: nothing was exhausted
    # ONE record rides every dimension's entry (jointly judged)
    records = [e.judge_refusal for e in refused]
    assert all(r is records[0] for r in records)
    rec = records[0]
    assert (rec.behavior, rec.scenario_id, rec.epoch) == (b, SIDS[0], 1)
    assert rec.dimensions == tuple(pr.required_metrics(b))
    assert rec.judge_model == world.spec.judge_model
    assert rec.sample_id == SIDS[0] and rec.sample_uuid == f"uuid-{SIDS[0]}"
    assert rec.outcome == pr.JUDGE_REFUSED_OUTCOME
    assert rec.auditor_chain is None  # the synthetic sample carries no store
    # never a score: no row carries a NaN or the refused item
    assert all(math.isfinite(r.value) for r in rows)
    assert not any(r.item_id == SIDS[0] for r in rows)


@pytest.mark.parametrize("value", [None, float("inf"), float("-inf")])
def test_none_or_inf_with_refusal_marker_is_also_a_judge_refusal(world, value):
    """inspect-petri's own Result(value=None, metadata={"refusal": True}) is
    the shape BEFORE inspect-scout turns it into a NaN — both are the same
    refusal; so is any other non-finite float."""
    cell = world.cells["si_safety__psr_kl_all"]
    b = BEHAVIORS[0]
    rows, ledger = convert(world, cell, b, [refused_sample(SIDS[0], value=value)])
    assert rows == []
    assert {e.dimension for e in ledger} == set(pr.required_metrics(b))
    assert all(pr.is_judge_refused_reason(e.reason) for e in ledger)
    assert all(e.judge_refusal is not None for e in ledger)


def test_refusal_without_text_still_names_the_reason(world):
    cell = world.cells["si_safety__psr_kl_all"]
    b = BEHAVIORS[0]
    _, ledger = convert(world, cell, b, [refused_sample(SIDS[0], text=None)])
    assert ledger and all(pr.is_judge_refused_reason(e.reason) for e in ledger)
    assert "without text" in ledger[0].reason


# --- (ii) a NaN WITHOUT the marker is still an unknown shape and still raises


@pytest.mark.parametrize("metadata", [
    {}, {"refusal": False}, {"refusal": "no"}, {"refusal": 1},
    {"scanner_references": []}, None,
])
def test_nan_without_refusal_marker_still_refuses_loudly(world, metadata):
    cell = world.cells["si_safety__psr_kl_all"]
    b = BEHAVIORS[0]
    s = refused_sample(SIDS[0], metadata=metadata if metadata is not None else {})
    if metadata is None:
        del s["scores"]["audit_judge"]["metadata"]  # no metadata key at all
    with pytest.raises(pr.PetriRunnerError, match="not a per-dimension mapping"):
        convert(world, cell, b, [s])


def test_finite_non_mapping_value_with_marker_is_not_a_refusal(world):
    """The flag alone is not a refusal either: a finite scalar under it is an
    unknown shape (refused loudly), never guessed into a refusal."""
    cell = world.cells["si_safety__psr_kl_all"]
    with pytest.raises(pr.PetriRunnerError, match="not a per-dimension mapping"):
        convert(world, cell, BEHAVIORS[0], [refused_sample(SIDS[0], value=7)])


# --- (iii) a None WITHOUT the marker keeps today's "no judge scores" path


def test_none_without_marker_keeps_the_no_judge_scores_path(world):
    cell = world.cells["si_safety__psr_kl_all"]
    b = BEHAVIORS[0]
    rows, ledger = convert(world, cell, b, [
        refused_sample(SIDS[0], value=None, metadata={})])
    assert rows == []
    assert {e.dimension for e in ledger} == set(pr.required_metrics(b))
    assert {e.reason for e in ledger} == {"no judge scores on sample"}
    assert all(e.judge_refusal is None for e in ledger)
    assert pr.judge_refusals_from_ledger(ledger) == []


# --- (iv) the fold: ONE record per (behavior, scenario, epoch)


def test_judge_refusals_from_ledger_folds_per_scenario_epoch(world):
    b = BEHAVIORS[0]
    dims = tuple(pr.required_metrics(b))
    judge = world.spec.judge_model
    reason = pr.judge_refused_reason(judge, REFUSAL_TEXT.strip())
    recs = [
        pr.JudgeRefusal(b, SIDS[1], 1, dims, reason, judge),
        pr.JudgeRefusal(b, SIDS[0], 2, dims, reason, judge),
        pr.JudgeRefusal(b, SIDS[0], 1, dims, reason, judge),
    ]
    entries = [pr.LedgerEntry(r.behavior, r.scenario_id, r.epoch, d, r.reason,
                              judge_refusal=r)
               for r in recs for d in dims]
    # other classes ride along and are NOT folded
    entries.append(pr.LedgerEntry(b, SIDS[1], 2, dims[0],
                                  "dimension missing from judge scores"))
    entries.append(pr.LedgerEntry(b, SIDS[1], 2, dims[1],
                                  "no judge scores on sample"))
    folded = pr.judge_refusals_from_ledger(entries)
    assert [(r.behavior, r.scenario_id, r.epoch) for r in folded] == sorted(
        (r.behavior, r.scenario_id, r.epoch) for r in recs)
    assert len(folded) == len(recs)  # 9 entries -> 3 records
    by_id = {(r.behavior, r.scenario_id, r.epoch): r for r in recs}
    for r in folded:
        assert r is by_id[(r.behavior, r.scenario_id, r.epoch)]
        assert r.dimensions == dims
        assert set(r.keys()) == {(r.behavior, r.scenario_id, r.epoch, d)
                                 for d in dims}


def test_judge_refusal_record_refuses_foreign_reasons_and_outcomes(world):
    b = BEHAVIORS[0]
    dims = tuple(pr.required_metrics(b))
    reason = pr.judge_refused_reason("j", "x")
    with pytest.raises(pr.PetriRunnerError, match="must start with"):
        pr.JudgeRefusal(b, SIDS[0], 1, dims, "no judge scores on sample", "j")
    with pytest.raises(pr.PetriRunnerError, match="is not"):
        pr.JudgeRefusal(b, SIDS[0], 1, dims, reason, "j",
                        outcome=pr.AUDITOR_REFUSED_OUTCOME)
    with pytest.raises(pr.PetriRunnerError, match="required dimensions"):
        pr.JudgeRefusal(b, SIDS[0], 1, (), reason, "j")


# --- (v) unresolved_entries: a judge-refused key is settled, not open


def test_unresolved_entries_exclude_judge_refused_keys(world, tmp_path):
    cell = world.cells["si_safety__psr_kl_all"]
    store = cs.CellStore(tmp_path / "sweep", pr.petri_cell_key(cell))  # no rows
    b = BEHAVIORS[0]
    _, ledger = convert(world, cell, b, [refused_sample(SIDS[0]),
                                         sample(SIDS[1], full_dims(b))])
    records = pr.judge_refusals_from_ledger(ledger)
    assert [(r.scenario_id, r.epoch) for r in records] == [(SIDS[0], 1)]
    canonical = {b: SIDS}
    refused_keys = set(records[0].keys())
    # without the refusal handed over, the item is an OPEN casualty carrying
    # the converter's judge-refused reason (and SIDS[1] is open too: no rows
    # were filed in this store)
    open_ = pr.unresolved_entries(store, canonical, ledger, 1)
    assert refused_keys <= {e.key() for e in open_}
    assert all(pr.is_judge_refused_reason(e.reason)
               for e in open_ if e.key() in refused_keys)
    # with it, the item is SETTLED: only the genuinely unfiled keys remain
    settled = pr.unresolved_entries(store, canonical, ledger, 1,
                                    judge_refused=records)
    assert {e.key() for e in settled} == {(b, SIDS[1], 1, d)
                                          for d in pr.required_metrics(b)}
    assert not ({e.key() for e in settled} & refused_keys)
    # the same exclusion beside an auditor refusal argument (both kinds)
    assert pr.unresolved_entries(store, canonical, ledger, 1,
                                 auditor_refused=(), judge_refused=records) \
        == settled


# --- (vi) the footing: N-1/N, the refusal named, results.json written, never re-drawn


def test_footing_reports_n_minus_one_and_names_the_refusal(world, tmp_path):
    cell = world.cells["si_safety__psr_kl_all"]
    rb, rs = BEHAVIORS[1], SIDS[1]
    store, result = _run_cell(world, cell, tmp_path,
                              _driver_with_one_judge_refusal(world, cell))
    epochs = world.spec.bloom_options.epochs
    n_expected = len(BEHAVIORS) * len(SIDS) * epochs

    # condition A: the cell COMPLETES at N-1/N, the refusal counted and named
    assert result["unscored"] == 0 and result["incomplete"] is False
    assert result["audited"] == {"items": n_expected - 1, "expected": n_expected}
    assert result["judge_refused"] == {"total": 1}
    assert result["auditor_refused"]["total"] == 0
    # the derivation, cell-level and per behavior: expected == audited +
    # auditor_refused + judge_refused; n_items is the frozen denominator
    assert result["audited"]["expected"] == (
        result["audited"]["items"] + result["auditor_refused"]["total"]
        + result["judge_refused"]["total"])
    for b, bs in result["behaviors"].items():
        assert bs["n_items"] == len(SIDS) * epochs
        assert bs["n_items"] == (bs["audited"] + bs["auditor_refused"]["total"]
                                 + bs["judge_refused"]["total"])
    assert sum(bs["n_items"] for bs in result["behaviors"].values()) == n_expected
    assert sum(bs["audited"] for bs in result["behaviors"].values()) \
        == result["audited"]["items"]
    assert sum(bs["judge_refused"]["total"] for bs in result["behaviors"].values()) \
        == result["judge_refused"]["total"]

    bsum = result["behaviors"][rb]
    assert bsum["audited"] == len(SIDS) * epochs - 1
    assert bsum["judge_refused"] == {"total": 1}
    assert bsum["judge_refused_items"] == [{
        "scenario_id": rs, "epoch": 1,
        "reason": pr.judge_refused_reason(world.spec.judge_model,
                                          REFUSAL_TEXT.strip()),
        "judge_model": world.spec.judge_model,
        "sample_id": rs, "eval_log": f"/synthetic/{rb}.eval",
        "auditor_chain": None,
    }]
    assert bsum["auditor_refused_items"] == []
    assert "judge_refused" in bsum["note"] and "never imputed" in bsum["note"]
    other = result["behaviors"][BEHAVIORS[0]]
    assert other["judge_refused"] == {"total": 0}
    assert other["judge_refused_items"] == []
    assert "note" not in other
    # carried by the auditor but not audited: the refused item is in neither
    # chain bucket, so successful legs still equal the audited items
    assert sum(bsum["auditor_chain"].values()) == bsum["audited"]
    assert result["auditor_chain"]["sonnet_only"] == n_expected - 1
    # the means are over the AUDITED items only — the sibling's real score,
    # never a fabricated one for the refused item
    assert bsum["mean"][rb] == full_dims(rb)[f"{rb}_propensity"]
    # the footing line names it, appended to the line it always had
    line = pr.audited_footing_line(result)
    assert line.startswith(f"audited {n_expected - 1}/{n_expected}: sonnet-only "
                           f"{n_expected - 1}, ")
    assert line.endswith(", judge_refused 1")
    # never imputed: no row exists for the refused item
    task = pr.PETRI_TASK_PREFIX + rb
    assert not any(k[0] == task and k[1] == rs for k in store.completed_keys())

    # results.json IS written (one refusal does not hold the cell), with the footing
    assert store.is_done()
    results = json.loads((store.cell_dir / "results.json").read_text())
    assert results["judge_refused"] == {"total": 1}
    assert results["audited"] == {"items": n_expected - 1, "expected": n_expected}
    assert results["behaviors"][rb]["judge_refused_items"][0]["scenario_id"] == rs
    # the ledger is SETTLED: a judge refusal is not an unresolved casualty
    assert json.loads(pr.ledger_path(store).read_text())["unresolved"] == []
    # the sidecar records it with the reason and the evidence
    sidecar = json.loads(pr.judge_refused_path(store).read_text())
    assert (sidecar["cell_id"], sidecar["petri_run_key"]) == (cell.cell_id,
                                                             cell.petri_run_key)
    (rec,) = sidecar[pr.JUDGE_REFUSED_OUTCOME]
    assert rec["outcome"] == pr.JUDGE_REFUSED_OUTCOME
    assert (rec["behavior"], rec["scenario_id"], rec["epoch"]) == (rb, rs, 1)
    assert rec["dimensions"] == list(pr.required_metrics(rb))
    assert rec["reason"].startswith(pr.JUDGE_REFUSED_REASON_PREFIX)
    assert rec["sample_uuid"] == f"uuid-{rs}"
    assert sidecar["by_behavior"] == {rb: {"total": 1}}
    note = sidecar["note"].lower()
    assert "never imputed" in note and "never relaunched" in note
    assert "never scored" in note

    # never relaunched: re-entering the cell must not invoke the driver
    def must_not_redraw(*args, **kwargs):  # pragma: no cover
        raise AssertionError("a terminal judge refusal must not be re-drawn")

    _, resumed = _run_cell(world, cell, tmp_path, must_not_redraw)
    assert resumed["rows_emitted"] == 0
    assert resumed["audited"] == result["audited"]
    assert resumed["judge_refused"] == {"total": 1}
    assert resumed["behaviors"][rb]["judge_refused_items"] == bsum["judge_refused_items"]

    # a later COMPLETE score set cannot silently supersede the refusal
    (refusal,) = pr.read_judge_refused(store)
    scored = {(task, rs, d) for d in refusal.dimensions}
    with pytest.raises(pr.PetriRunnerError, match="silent substitution"):
        pr.merge_judge_refused([refusal], [], scored, expected_epochs=1)
    # ...while append-only PARTIAL dimensions coexist (the sidecar is their tombstone)
    partial = {(task, rs, refusal.dimensions[0])}
    assert pr.merge_judge_refused([refusal], [], partial, expected_epochs=1) \
        == [refusal]


def test_cell_without_judge_refusal_keeps_its_footing_line_and_writes_no_sidecar(
        world, tmp_path):
    """Inertness on the synthetic path: a cell with no judge refusal keeps
    the run-summary shape it always had (the frozen seam
    test_petri_store_shape pins: no ``judge_refused`` key at all — absence
    means zero), the footing line is the one it always had, and no sidecar
    is written."""
    cell = world.cells["none"]
    store, result = _run_cell(world, cell, tmp_path, _driver(world, cell))
    n = len(BEHAVIORS) * len(SIDS)
    assert result["audited"] == {"items": n, "expected": n}
    assert set(result) == {"rows_emitted", "behaviors", "unscored", "incomplete",
                           "audited", "auditor_refused", "auditor_chain"}
    line = pr.audited_footing_line(result)
    assert "judge_refused" not in line
    assert line == (f"audited {n}/{n}: sonnet-only {n}, continued-by-grok 0, "
                    "continued-by-qwen 0, auditor_refused (all legs) 0, "
                    "auditor unavailable 0 (content-correlated 0 / outage 0 / "
                    "undetermined 0)")
    assert not pr.judge_refused_path(store).exists()
    for bs in result["behaviors"].values():
        assert "judge_refused" not in bs
        assert "judge_refused_items" not in bs
        assert "note" not in bs
    results = json.loads((store.cell_dir / "results.json").read_text())
    assert "judge_refused" not in results
    # the fold reads that absence as zero
    assert "judge_refused" not in pr.audited_footing(result["behaviors"])


def test_audited_footing_folds_judge_refused_from_behaviors():
    summary = {
        "a": {"audited": 9, "n_items": 10, "auditor_refused": pr._zero_refused_tally(),
              "judge_refused": {"total": 1}, "auditor_chain": pr._zero_chain_tally()},
        "b": {"audited": 8, "n_items": 10,
              "auditor_refused": {**pr._zero_refused_tally(), "total": 1,
                                  pr.SHED_OUTAGE: 1},
              "judge_refused": {"total": 1}, "auditor_chain": pr._zero_chain_tally()},
    }
    footing = pr.audited_footing(summary)
    assert footing["audited"] == {"items": 17, "expected": 20}
    assert footing["judge_refused"] == {"total": 2}
    assert footing["auditor_refused"]["total"] == 1
    assert footing["audited"]["expected"] == (
        footing["audited"]["items"] + footing["auditor_refused"]["total"]
        + footing["judge_refused"]["total"])
    assert pr.audited_footing_line({"behaviors": summary}).endswith(
        ", judge_refused 2")
    # a summary without the key (no judge refusal, or filed before the key
    # existed) folds WITHOUT it — absence means zero — and keeps its line
    legacy = {k: {kk: vv for kk, vv in v.items() if kk != "judge_refused"}
              for k, v in summary.items()}
    assert "judge_refused" not in pr.audited_footing(legacy)
    assert "judge_refused" not in pr.audited_footing_line({"behaviors": legacy})
    assert "judge_refused" not in pr.audited_footing_line(
        {**pr.audited_footing(legacy), "behaviors": legacy})


# --- (vii) the sidecar: write/read round trip; malformed shapes refuse


def test_sidecar_round_trip_and_malformed_shapes_refuse(world, tmp_path):
    cell = world.cells["none"]
    store = cs.CellStore(tmp_path / "sweep", pr.petri_cell_key(cell))
    judge = world.spec.judge_model
    reason = pr.judge_refused_reason(judge, REFUSAL_TEXT.strip())
    chain = {"version": 1,
             "configured": [{"model": "anthropic/claude-sonnet-5", "label": "sonnet-5"}],
             "used": [{"model": "anthropic/claude-sonnet-5", "label": "sonnet-5"}]}
    r1 = pr.JudgeRefusal(BEHAVIORS[0], SIDS[1], 1,
                         tuple(pr.required_metrics(BEHAVIORS[0])), reason, judge,
                         sample_id=SIDS[1], sample_uuid="u1",
                         eval_log="/synthetic/a.eval", auditor_chain=chain)
    r2 = pr.JudgeRefusal(BEHAVIORS[1], SIDS[0], 1,
                         tuple(pr.required_metrics(BEHAVIORS[1])), reason, judge)
    payload = pr.write_judge_refused(store, cell, [r2, r1])
    assert pr.judge_refused_path(store).is_file()
    assert payload["by_behavior"] == {BEHAVIORS[0]: {"total": 1},
                                      BEHAVIORS[1]: {"total": 1}}
    assert payload["version"] == 1
    # THIS implementation's marker rides the payload and every record
    assert payload["schema"] == pr.JUDGE_REFUSED_SCHEMA
    assert all(e["schema"] == pr.JUDGE_REFUSED_SCHEMA
               for e in payload[pr.JUDGE_REFUSED_OUTCOME])
    assert (payload["cell_id"], payload["petri_run_key"]) == (cell.cell_id,
                                                             cell.petri_run_key)
    back = pr.read_judge_refused(store)
    assert back == sorted([r1, r2], key=lambda r: (r.behavior, r.scenario_id, r.epoch))
    (with_chain,) = [r for r in back if r.behavior == BEHAVIORS[0]]
    assert with_chain.auditor_chain == chain  # the mapping survives the JSON round trip
    assert pr.judge_refused_by_behavior(back) == payload["by_behavior"]
    assert pr.judge_refused_row_keys(back, 1) == {
        (pr.PETRI_TASK_PREFIX + r.behavior, r.scenario_id, d)
        for r in (r1, r2) for d in r.dimensions}
    # an empty list records that none stand
    pr.write_judge_refused(store, cell, [])
    assert pr.read_judge_refused(store) == []
    # duplicates refuse
    with pytest.raises(pr.PetriRunnerError, match="duplicate"):
        pr.write_judge_refused(store, cell, [r1, r1])
    # merge: new wins on duplicate identity; prior without scored rows stands
    newer = pr.JudgeRefusal(r1.behavior, r1.scenario_id, r1.epoch, r1.dimensions,
                            pr.judge_refused_reason(judge, "later text"), judge)
    merged = pr.merge_judge_refused([r1, r2], [newer], set(), expected_epochs=1)
    assert merged == sorted([newer, r2], key=lambda r: (r.behavior, r.scenario_id,
                                                        r.epoch))
    # a sidecar of the same name written to ANOTHER lane's record shape
    # refuses loudly (never a silent zero, never guessed)
    path = pr.judge_refused_path(store)
    path.write_text(json.dumps({
        "version": 1, "cell_id": cell.cell_id, "petri_run_key": cell.petri_run_key,
        "judge_refused": [{"behavior": BEHAVIORS[0], "scenario_id": SIDS[0],
                           "epoch": 1, "dimensions": list(r1.dimensions),
                           "refusal": REFUSAL_TEXT.strip(), "attempts": 2,
                           "judge_model": judge}]}))
    with pytest.raises(pr.PetriRunnerError,
                       match="different judge-refusal implementation"):
        pr.read_judge_refused(store)
    path.write_text(json.dumps({"version": 1}))
    with pytest.raises(pr.PetriRunnerError, match="malformed judge-refused record"):
        pr.read_judge_refused(store)
    path.write_text("{not json")
    with pytest.raises(pr.PetriRunnerError, match="unparseable"):
        pr.read_judge_refused(store)


# --- (viii) the consolidation validator: reconciles the twin; refuses disagreement


def _bound(store, cell):
    manifest = json.loads((store.cell_dir / "manifest.json").read_text())
    results = json.loads((store.cell_dir / "results.json").read_text())
    return petri_run.BoundCellDir(cell.cell_id, cell.petri_run_key,
                                  store.cell_dir, manifest, results)


def _rewrite_results(store, fn):
    p = store.cell_dir / "results.json"
    r = json.loads(p.read_text())
    fn(r)
    p.write_text(json.dumps(r))


def test_validator_reconciles_one_judge_refusal_and_refuses_disagreement(
        world, tmp_path):
    cell = world.cells["si_safety__psr_kl_all"]
    rb, rs = BEHAVIORS[1], SIDS[1]
    epochs = world.spec.bloom_options.epochs

    def fresh(name):
        store, _ = _run_cell(world, cell, tmp_path / name,
                             _driver_with_one_judge_refusal(world, cell))
        return store

    # as written by the runner: reconciles (audited from rows, refusals from
    # both sidecars, n_items as their sum)
    store = fresh("ok")
    petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # the per-behavior tally disagrees with the sidecar
    _rewrite_results(store, lambda r: r["behaviors"][rb]["judge_refused"].update(total=0))
    with pytest.raises(petri_run.DriverError, match="judge-refusal sidecar tallies 1"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # the cell-level tally disagrees
    store = fresh("cell_tally")
    _rewrite_results(store, lambda r: r["judge_refused"].update(total=2))
    with pytest.raises(petri_run.DriverError, match="cell-level judge_refused"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # the identities disagree
    store = fresh("identity")
    _rewrite_results(store, lambda r: r["behaviors"][rb]["judge_refused_items"][0]
                     .update(scenario_id=SIDS[0]))
    with pytest.raises(petri_run.DriverError, match="judge-refused identities"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # the listed item's reason is not a judge refusal
    store = fresh("reason")
    _rewrite_results(store, lambda r: r["behaviors"][rb]["judge_refused_items"][0]
                     .update(reason="no judge scores on sample"))
    with pytest.raises(petri_run.DriverError, match="reason that is not a judge refusal"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # a results.json that silently DROPS the key while the sidecar holds a
    # refusal: a refusal is reported, never absent
    store = fresh("dropped")

    def drop(r):
        r.pop("judge_refused")
        for bs in r["behaviors"].values():
            bs.pop("judge_refused")
            bs.pop("judge_refused_items")
    _rewrite_results(store, drop)
    with pytest.raises(petri_run.DriverError, match="no judge_refused footing"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # another lane's bare-int shape is not a tally
    store = fresh("int_shape")
    _rewrite_results(store, lambda r: r.__setitem__("judge_refused", 1))
    with pytest.raises(petri_run.DriverError, match="not a mapping"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # exact ints: 1.0 never stands for 1
    store = fresh("float")
    _rewrite_results(store, lambda r: r["behaviors"][rb]["judge_refused"].update(total=1.0))
    with pytest.raises(petri_run.DriverError, match="exact integer"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # n_items is audited + auditor_refused + judge_refused: a denominator
    # that forgets the judge refusal is refused
    store = fresh("n_items")
    _rewrite_results(store, lambda r: r["behaviors"][rb].update(
        n_items=r["behaviors"][rb]["n_items"] - 1))
    with pytest.raises(petri_run.DriverError, match="denominator"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # the sidecar names another run
    store = fresh("foreign")
    path = pr.judge_refused_path(store)
    raw = json.loads(path.read_text())
    raw["petri_run_key"] = "0" * 16
    path.write_text(json.dumps(raw))
    with pytest.raises(petri_run.DriverError, match="another run's"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # a sidecar of the same name from the OTHER implementation (no schema
    # marker, or a foreign one) is named as such — never "malformed"
    store = fresh("other_impl")
    path = pr.judge_refused_path(store)
    raw = json.loads(path.read_text())
    raw.pop("schema")
    path.write_text(json.dumps(raw))
    with pytest.raises(petri_run.DriverError,
                       match="different judge-refusal implementation"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    raw["schema"] = "candidate-si-refit/judge_refused/v1"
    path.write_text(json.dumps(raw))
    with pytest.raises(petri_run.DriverError,
                       match="different judge-refusal implementation"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # a results.json claiming a refusal with NO sidecar behind it
    store = fresh("no_sidecar")
    pr.judge_refused_path(store).unlink()
    with pytest.raises(petri_run.DriverError, match="judge-refusal sidecar tallies 0"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)

    # the consolidation tombstones cover the refused item's row keys
    store = fresh("tombstones")
    tomb = petri_run.collect_auditor_refusal_tombstones(tmp_path / "tombstones" / "sweep",
                                                        world.spec)
    (refusal,) = pr.read_judge_refused(store)
    assert tomb == {store.cell_dir.name: pr.judge_refused_row_keys([refusal], epochs)}
    assert tomb[store.cell_dir.name] == {(pr.PETRI_TASK_PREFIX + rb, rs, d)
                                         for d in pr.required_metrics(rb)}


def test_validator_accepts_already_filed_cells_without_the_judge_key(world, tmp_path):
    """A results.json without ``judge_refused`` — what the runner writes for
    a cell with no judge refusal, and what every cell filed BEFORE the key
    existed carries (several exist on disk) — reconciles with an absent
    judge sidecar, and the collectors read it as zero."""
    cell = world.cells["none"]
    epochs = world.spec.bloom_options.epochs
    store, result = _run_cell(world, cell, tmp_path, _driver(world, cell))
    assert "judge_refused" not in result
    results = json.loads((store.cell_dir / "results.json").read_text())
    assert "judge_refused" not in results
    assert all("judge_refused" not in bs for bs in results["behaviors"].values())
    petri_run._reconcile_done_marker(_bound(store, cell), epochs)  # a done marker
    assert petri_run.collect_auditor_refusal_tombstones(tmp_path / "sweep",
                                                        world.spec) == {}
    # an EXPLICIT zero tally reconciles too (cell-level and per behavior)
    def explicit_zero(r):
        r["judge_refused"] = {"total": 0}
        for bs in r["behaviors"].values():
            bs["judge_refused"] = {"total": 0}
            bs["judge_refused_items"] = []
    _rewrite_results(store, explicit_zero)
    petri_run._reconcile_done_marker(_bound(store, cell), epochs)
    # ...but a claimed refusal with no sidecar behind it does not
    _rewrite_results(store, lambda r: r["behaviors"][BEHAVIORS[0]]["judge_refused"]
                     .update(total=1))
    with pytest.raises(petri_run.DriverError, match="judge-refusal sidecar tallies 0"):
        petri_run._reconcile_done_marker(_bound(store, cell), epochs)


# --- condition B: the inertness harness on a REAL already-converted cell ------------


def _real_cell_or_skip(run_key):
    """The real spec, cell, canonical ids and paths for one run key of the
    GLM full22 audit — or a skip when any piece is not on this machine.
    READ-ONLY throughout."""
    pytest.importorskip("inspect_ai")
    cell_dir = REAL_ROOT / "cells" / run_key
    log_dir = REAL_ROOT / "inspect_logs" / run_key
    if not (REAL_MANIFEST.is_file() and (REAL_ROOT / "cells.json").is_file()
            and (cell_dir / "manifest.json").is_file() and log_dir.is_dir()):
        pytest.skip(f"real cell {run_key} of {REAL_AUDIT} is not on this machine")
    spec = cm.load_petri_spec(REAL_MANIFEST)
    record = json.loads((REAL_ROOT / "cells.json").read_text(encoding="utf-8"))
    served = list(record.get("canonical") or []) + list(record.get("controls") or [])
    entries = [c for c in served if c.get("petri_run_key") == run_key]
    if len(entries) != 1:
        pytest.skip(f"{run_key} is not exactly one served leg of the frozen record")
    cell = dc.PetriCell(**entries[0])
    manifest = json.loads((cell_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("scenario_bundle_hash") != spec.scenario_bundle_hash \
            or manifest.get("petri_run_key") != run_key:
        pytest.skip("the checked-in manifest is not the one this cell ran under")
    behaviors_dir = REAL_ROOT / f"behaviors-{spec.scenario_bundle_hash}"
    if not behaviors_dir.is_dir():
        pytest.skip("the run's materialised behavior dirs are not on this machine")
    canonical = {b: psc.scenario_ids(behaviors_dir / b / "scenarios")
                 for b in spec.behaviors}
    return spec, cell, cell_dir, log_dir, canonical


def _real_completed_logs(log_dir):
    """Every COMPLETED eval log of the run-key log dir in the dict form the
    runner's driver hands the converter (``read_eval_log`` then the driver's
    own ``_dumped_log``: ``model_dump(mode="json")`` plus the log's
    ``location``). The ``started`` logs are the interrupted attempts
    eval_set retried."""
    from inspect_ai.log import read_eval_log
    logs = []
    for p in sorted(log_dir.glob("*.eval")):
        log = read_eval_log(str(p))
        if log.status != "success":
            continue
        dumped = pr._dumped_log(log)
        assert dumped["location"] == str(p)  # item 3(a): traceable to its log
        logs.append(dumped)
    return logs


def _convert_all(spec, cell, canonical, logs):
    by_dir = {}
    for d in logs:
        by_dir.setdefault(d["eval"]["metadata"]["behavior_dir_key"], []).append(d)
    dup = {k: len(v) for k, v in by_dir.items() if len(v) != 1}
    assert not dup, (f"more than one completed log per behavior dir: {dup} — "
                     "refusing to pick one; report it")
    rows, ledger = [], []
    for dk, (d,) in sorted(by_dir.items()):
        b = dk.split("/", 1)[0]
        r, l = pr.convert_eval_log(
            d, spec=spec, cell=cell, dir_key=dk, canonical_ids=canonical[b],
            env_digest="ee" * 8, sibling_logs=[o for o in logs if o is not d])
        rows.extend(r)
        ledger.extend(l)
    return rows, ledger


def test_inert_on_the_already_converted_real_cell():
    """Condition B: re-converting cell `none` (COMPLETE, 60/60, converted by
    the converter this change replaces) through the new converter yields
    exactly the rows its rows.jsonl holds — every (task, item_id, metric,
    value), no ledger entry, no judge refusal. Any difference is reported
    in full, never adjusted away."""
    spec, cell, cell_dir, log_dir, canonical = _real_cell_or_skip(REAL_COMPLETE_KEY)
    rows_path = cell_dir / "rows.jsonl"
    if not rows_path.is_file():
        pytest.skip("the real cell has no rows.jsonl to compare against")
    filed, filed_extra = {}, {}
    for line in rows_path.read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        key = (r["task"], r["item_id"], r["metric"])
        assert key not in filed, f"rows.jsonl holds {key} twice"
        filed[key] = float(r["value"])
        filed_extra[key] = r.get("extra_json")
    logs = _real_completed_logs(log_dir)
    assert {d["eval"]["metadata"]["behavior_dir_key"].split("/", 1)[0]
            for d in logs} == set(spec.behaviors), "not every behavior's completed log is present"
    rows, ledger = _convert_all(spec, cell, canonical, logs)
    got = {(r.task, r.item_id, r.metric): r.value for r in rows}
    assert len(got) == len(rows), "the re-conversion produced a duplicate row key"
    # the denominator, derived: scenarios x required dimensions x epochs
    expected_n = sum(len(canonical[b]) * len(pr.required_metrics(b))
                     for b in spec.behaviors) * spec.bloom_options.epochs
    assert len(filed) == expected_n, (len(filed), expected_n)
    only_filed = sorted(set(filed) - set(got))
    only_new = sorted(set(got) - set(filed))
    differ = sorted(k for k in set(filed) & set(got) if filed[k] != got[k])
    assert not only_filed and not only_new and not differ, {
        "rows_compared": len(filed),
        "only_in_rows_jsonl": only_filed[:10],
        "only_in_reconversion": only_new[:10],
        "value_differs": [(k, filed[k], got[k]) for k in differ[:10]],
    }
    assert len(got) == expected_n
    assert ledger == []
    assert pr.judge_refusals_from_ledger(ledger) == []
    # beyond the four compared fields: the per-row provenance blob is the same too
    extra_differ = [k for r in rows for k in [(r.task, r.item_id, r.metric)]
                    if filed_extra[k] != r.extra_json]
    assert extra_differ == [], {"extra_json_differs": extra_differ[:10]}


def test_real_refused_cell_converts_with_one_judge_refusal_and_no_raise():
    """The cell that crashed: its completed logs now convert without a
    raise, with EXACTLY one judge-refused scenario (all its required
    dimensions), every other item scored — N-1 items of N, derived."""
    spec, cell, cell_dir, log_dir, canonical = _real_cell_or_skip(REAL_REFUSED_KEY)
    logs = _real_completed_logs(log_dir)
    present = {d["eval"]["metadata"]["behavior_dir_key"].split("/", 1)[0] for d in logs}
    if present != set(spec.behaviors):
        pytest.skip(f"the refused cell's completed logs cover {sorted(present)}, "
                    f"not every behavior")
    rows, ledger = _convert_all(spec, cell, canonical, logs)  # raised before this branch
    records = pr.judge_refusals_from_ledger(ledger)
    assert len(records) == 1, [(r.behavior, r.scenario_id) for r in records]
    (rec,) = records
    assert rec.behavior in spec.behaviors
    assert rec.scenario_id in canonical[rec.behavior]
    assert rec.epoch == 1
    assert rec.dimensions == tuple(pr.required_metrics(rec.behavior))
    assert rec.judge_model == spec.judge_model
    assert rec.reason.startswith(pr.JUDGE_REFUSED_REASON_PREFIX)
    assert "content filter" in rec.reason
    assert rec.sample_id == rec.scenario_id and rec.sample_uuid
    # item 3(a): the refusal is traceable to its log — the driver's dict form
    # carries EvalLog.location (excluded from model_dump), so eval_log names
    # the real .eval file the refused transcript is kept in
    assert rec.eval_log is not None
    assert Path(rec.eval_log).is_file()
    assert Path(rec.eval_log).parent == log_dir
    # every ledger entry is that refusal's: nothing else went unscored
    assert {e.key() for e in ledger} == set(rec.keys())
    assert all(e.judge_refusal is rec for e in ledger)
    expected_n = sum(len(canonical[b]) * len(pr.required_metrics(b))
                     for b in spec.behaviors) * spec.bloom_options.epochs
    assert len(rows) == expected_n - len(rec.dimensions)
    task = pr.PETRI_TASK_PREFIX + rec.behavior
    assert not any(r.task == task and r.item_id == rec.scenario_id for r in rows)
    assert all(math.isfinite(r.value) for r in rows)


# --- item 3(a): the driver's dict form carries the log's location -------------------


def test_dumped_log_carries_the_eval_location():
    """``EvalLog.location`` is excluded from ``model_dump``; the driver's
    ``_dumped_log`` puts it back, so every refusal's ``eval_log`` evidence
    (auditor shed and judge refusal alike) names its ``.eval``."""
    class FullLog:  # a full log (samples present): no re-read
        location = "/synthetic/logs/one.eval"
        samples = []

        def model_dump(self, mode):
            assert mode == "json"
            return {"eval": {}, "samples": [], "status": "success"}

    dumped = pr._dumped_log(FullLog())
    assert dumped["location"] == "/synthetic/logs/one.eval"
    assert dumped["status"] == "success" and dumped["samples"] == []

    class Unlocated(FullLog):
        location = None

    assert pr._dumped_log(Unlocated())["location"] is None


# --- item 2: the schema marker tells this implementation from the other ------------


def test_sidecar_without_our_schema_is_named_as_the_other_implementation(
        world, tmp_path):
    """The candidate-si-refit lineage writes the SAME filename in another
    record shape. Ours carries ``schema`` on the payload and every record;
    a payload or record without it, or with a foreign value, is named as
    that lineage's — never "malformed"."""
    cell = world.cells["none"]
    store = cs.CellStore(tmp_path / "sweep", pr.petri_cell_key(cell))
    judge = world.spec.judge_model
    dims = tuple(pr.required_metrics(BEHAVIORS[0]))
    rec = pr.JudgeRefusal(BEHAVIORS[0], SIDS[0], 1, dims,
                          pr.judge_refused_reason(judge, "x"), judge)
    pr.write_judge_refused(store, cell, [rec])
    path = pr.judge_refused_path(store)
    assert pr.JUDGE_REFUSED_SCHEMA == "sisteer.petri_runner/judge_refused/v1"
    # ours: reads
    assert pr.read_judge_refused(store) == [rec]

    def expect_foreign(what):
        with pytest.raises(pr.PetriRunnerError) as ei:
            pr.read_judge_refused(store)
        msg = str(ei.value)
        assert pr.JUDGE_REFUSED_FOREIGN_MESSAGE in msg, msg
        assert "malformed" not in msg, msg
        assert what in msg, msg

    ours = json.loads(path.read_text())
    # missing payload marker
    raw = json.loads(json.dumps(ours))
    raw.pop("schema")
    path.write_text(json.dumps(raw))
    expect_foreign("carries no schema marker")
    # foreign payload marker
    raw["schema"] = "candidate-si-refit/judge_refused/v2"
    path.write_text(json.dumps(raw))
    expect_foreign("candidate-si-refit/judge_refused/v2")
    # ours at the payload, a record without the marker
    raw = json.loads(json.dumps(ours))
    raw[pr.JUDGE_REFUSED_OUTCOME][0].pop("schema")
    path.write_text(json.dumps(raw))
    expect_foreign("record carries no schema marker")
    # ours at the payload, a record with a foreign marker
    raw[pr.JUDGE_REFUSED_OUTCOME][0]["schema"] = "other/v9"
    path.write_text(json.dumps(raw))
    expect_foreign("record carries schema 'other/v9'")
    # restored: reads again
    path.write_text(json.dumps(ours))
    assert pr.read_judge_refused(store) == [rec]


# --- item 1: the stats reader's third term -----------------------------------------

JUDGE_MODEL = "anthropic/claude-opus-5"
JUDGE_REASON = pr.judge_refused_reason(JUDGE_MODEL, REFUSAL_TEXT.strip())


def _judge_item(scenario_id, epoch=1):
    return {"scenario_id": scenario_id, "epoch": epoch, "reason": JUDGE_REASON,
            "judge_model": JUDGE_MODEL, "sample_id": scenario_id,
            "eval_log": None, "auditor_chain": None}


def _footing_with_judge(per_behavior, judge, *, run_key=RUN_KEY):
    """A complete cell's footing the way the runner writes one WITH judge
    refusals: ``per_behavior`` = {b: (n_items, audited, auditor_refused)},
    ``judge`` = {b: [scenario ids the judge refused]} — the third term that
    makes each behavior's n_items conserve."""
    footing = _cell_footing(per_behavior, run_key=run_key)
    total = 0
    for b, bf in footing["behaviors"].items():
        ids = list(judge.get(b, []))
        bf["judge_refused"] = {"total": len(ids)}
        bf["judge_refused_items"] = [_judge_item(s) for s in ids]
        total += len(ids)
    footing["judge_refused"] = {"total": total}
    return footing


def _drop_items(frame, cell, behavior, items):
    keep = ~((frame["arm_name"] == cell)
             & (frame["task"] == f"petri/{behavior}")
             & (frame["item_id"].isin(list(items))))
    return frame[keep].reset_index(drop=True)


def _pre_change_petri_stats(tmp_path):
    """The petri_stats module as it was BEFORE the judge term (the pinned
    pre-change sha), loaded from git into a temp module — the before/after
    comparison's "before". Skips when the sha is not in this clone."""
    proc = subprocess.run(
        ["git", "show", f"{PRE_CHANGE_SHA}:src/sisteer/capeval/petri/petri_stats.py"],
        cwd=ROOT, capture_output=True, text=True)
    if proc.returncode:
        pytest.skip(f"pre-change sha {PRE_CHANGE_SHA} is not in this clone")
    assert "judge_refused" not in proc.stdout, "not the pre-change reader"
    p = tmp_path / "petri_stats_pre_change.py"
    p.write_text(proc.stdout, encoding="utf-8")
    loader = importlib.util.spec_from_file_location("petri_stats_pre_change", p)
    mod = importlib.util.module_from_spec(loader)
    # registered BEFORE exec: the module's dataclasses resolve their defining
    # module through sys.modules at class-creation time (the same trick
    # scripts/petri_run.py uses for its sibling-script loader)
    sys.modules["petri_stats_pre_change"] = mod
    loader.loader.exec_module(mod)
    return mod


def _canonical(analysis):
    """The analysis as canonical JSON with any timestamp-like key stripped
    (the analysis carries none today — the strip is the guard the
    comparison was asked to have)."""
    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items()
                    if not (k.endswith("_at") or k == "timestamp")}
        if isinstance(o, list):
            return [strip(v) for v in o]
        return o
    return json.dumps(strip(analysis), sort_keys=True, default=str)


def test_stats_judge_reason_prefix_mirrors_the_runner():
    assert petri_stats._JUDGE_REFUSED_REASON_PREFIX == pr.JUDGE_REFUSED_REASON_PREFIX


def test_stats_reads_a_judge_refused_cell_as_n_minus_one_with_the_reason_named(
        tmp_path):
    """Item 1, condition (a): one judge refusal in one behavior — the cell
    is 17/18 in its footing, the refused item and its reason are first-class
    in ``by_cell``, the behavior's audited set is the five scored items, and
    the conservation law holds with three terms. The SAME inputs failed the
    pre-change reader at "does not conserve items" — the failure this term
    removes."""
    method = dc.CLASS_A_METHODS[0]
    cell = f"si_constitution__{method}"
    b0 = BEHAVIORS3[0]
    frame = _drop_items(build_world(), cell, b0, ["s5"])
    footing = _footing_with_judge(
        {b0: (6, 5, 0), BEHAVIORS3[1]: (6, 6, 0), BEHAVIORS3[2]: (6, 6, 0)},
        {b0: ["s5"]})
    n_expected = 6 * len(BEHAVIORS3)
    assert footing["audited"] == {"items": n_expected - 1, "expected": n_expected}
    assert footing["judge_refused"] == {"total": 1}
    provenance = _provenance_for(frame)
    analysis = _analyze(frame, cell_footing={cell: footing}, provenance=provenance)
    entry = analysis["by_cell"][cell]
    af = entry["audited_footing"]
    assert af == footing
    assert af["audited"]["expected"] == (
        af["audited"]["items"] + af["auditor_refused"]["total"]
        + af["judge_refused"]["total"])
    assert entry["judge_refused"] == {
        "total": 1,
        "items": [{"behavior": b0, "scenario_id": "s5", "epoch": 1,
                   "reason": JUDGE_REASON, "judge_model": JUDGE_MODEL}]}
    assert entry["judge_refused"]["items"][0]["reason"].startswith(
        pr.JUDGE_REFUSED_REASON_PREFIX)
    assert entry["behaviors"][b0]["n_items"] == 5
    assert entry["behaviors"][BEHAVIORS3[1]]["n_items"] == 6
    assert analysis["audited_footing"][cell]["judge_refused"] == {"total": 1}
    assert all("judge_refused" not in e
               for c, e in analysis["by_cell"].items() if c != cell)
    pre = _pre_change_petri_stats(tmp_path)
    with pytest.raises(pre.PetriStatsError, match="does not conserve items"):
        pre.analyze(frame, expected_behaviors=list(BEHAVIORS3), n_boot=200,
                    seed=7, cell_footing={cell: footing}, provenance=provenance)


def test_stats_refuses_footing_whose_judge_term_does_not_reconcile():
    import copy
    method = dc.CLASS_A_METHODS[0]
    cell = f"si_constitution__{method}"
    b0, b1 = BEHAVIORS3[0], BEHAVIORS3[1]
    frame = _drop_items(build_world(), cell, b0, ["s5"])
    provenance = _provenance_for(frame)
    good = _footing_with_judge(
        {b0: (6, 5, 0), b1: (6, 6, 0), BEHAVIORS3[2]: (6, 6, 0)}, {b0: ["s5"]})
    _analyze(frame, cell_footing={cell: good}, provenance=provenance)  # reconciles

    def refuse(mutate, pattern):
        footing = copy.deepcopy(good)
        mutate(footing)
        with pytest.raises(petri_stats.PetriStatsError, match=pattern):
            _analyze(frame, cell_footing={cell: footing}, provenance=provenance)

    # the behavior forgets its judge term: 5 + 0 + 0 != 6
    refuse(lambda f: (f["behaviors"][b0].__setitem__("judge_refused", {"total": 0}),
                      f["behaviors"][b0].__setitem__("judge_refused_items", [])),
           "does not conserve items")
    # the cell level forgets it: 17 + 0 + 0 != 18
    refuse(lambda f: f.__setitem__("judge_refused", {"total": 0}),
           "does not conserve items")
    # the cell level over-counts it
    refuse(lambda f: f["judge_refused"].update(total=2), "does not conserve items")
    # per-behavior sum != cell-level total while each block conserves
    refuse(lambda f: (f["behaviors"][b1].update(n_items=7),
                      f["behaviors"][b1]["judge_refused"].update(total=1),
                      f["behaviors"][b1].__setitem__("judge_refused_items",
                                                     [_judge_item("s9")])),
           "sum judge_refused.total 2 != judge_refused.total 1")
    # the other implementation's shapes are never coerced
    refuse(lambda f: f.__setitem__("judge_refused", 1), "not a mapping")
    refuse(lambda f: f["behaviors"][b0].__setitem__("judge_refused", []),
           "not a mapping")
    refuse(lambda f: f["judge_refused"].__setitem__("content_filter", 1),
           "not exactly")
    refuse(lambda f: f["judge_refused"].__setitem__("total", 1.0), "exact integer")
    # a count without its items, a miscounted list, an unnamed reason
    refuse(lambda f: f["behaviors"][b0].pop("judge_refused_items"),
           "no judge_refused_items")
    refuse(lambda f: f["behaviors"][b0]["judge_refused_items"].append(
        _judge_item("s4")), "listed but")
    refuse(lambda f: f["behaviors"][b0]["judge_refused_items"][0].__setitem__(
        "reason", "no judge scores on sample"), "not a judge refusal")


def test_stats_accepts_slices_fully_refused_by_the_judge_and_names_them():
    """A behavior with NO rows in a cell is licensed by footing proving every
    item was terminally refused — by the auditor, the judge, or both; the
    refusal line names the judge term; a ROW-LESS cell fully refused by the
    judge is COMPLETE-but-UNAVAILABLE, never MISSING."""
    method = dc.CLASS_A_METHODS[0]
    cell = f"si_constitution__{method}"
    b0 = BEHAVIORS3[0]
    six = [f"s{i}" for i in range(6)]
    # one behavior of the cell fully refused by the judge
    frame = _drop_items(build_world(), cell, b0, six)
    footing = _footing_with_judge(
        {b0: (6, 0, 0), BEHAVIORS3[1]: (6, 6, 0), BEHAVIORS3[2]: (6, 6, 0)},
        {b0: six})
    proof = petri_stats._refused_behavior_footing(footing, b0)
    assert proof == {"n_items": 6, "auditor_refused": 0, "judge_refused": 6}
    assert petri_stats._refused_behavior_footing(footing, BEHAVIORS3[1]) is None
    line = petri_stats._refusal_line(cell, b0, proof)
    assert "audited 0/6" in line and "judge_refused 6" in line
    analysis = _analyze(frame, cell_footing={cell: footing},
                        provenance=_provenance_for(frame))
    entry = analysis["by_cell"][cell]
    assert entry["behaviors"][b0]["n_items"] == 0
    assert entry["behaviors"][b0]["mean_behavior_score"] is None
    assert entry["judge_refused"]["total"] == 6
    assert "judge_refused 6" in json.dumps(analysis)
    # proof by BOTH kinds: 3 auditor + 3 judge; the line names both
    mixed = _footing_with_judge(
        {b0: (6, 0, 3), BEHAVIORS3[1]: (6, 6, 0), BEHAVIORS3[2]: (6, 6, 0)},
        {b0: six[3:]})
    proof2 = petri_stats._refused_behavior_footing(mixed, b0)
    assert proof2 == {"n_items": 6, "auditor_refused": 3, "judge_refused": 3}
    line2 = petri_stats._refusal_line(cell, b0, proof2)
    assert "auditor_refused 3" in line2 and "judge_refused 3" in line2
    _analyze(frame, cell_footing={cell: mixed}, provenance=_provenance_for(frame))
    # an auditor-only proof keeps the line it always had
    auditor_only = _cell_footing({b0: (6, 0, 6), BEHAVIORS3[1]: (6, 6, 0),
                                  BEHAVIORS3[2]: (6, 6, 0)}, run_key=RUN_KEY)
    proof3 = petri_stats._refused_behavior_footing(auditor_only, b0)
    assert petri_stats._refusal_line(cell, b0, proof3) == (
        f"{cell}: audited 0/6, auditor_refused 6 for behavior {b0!r} — every "
        "expected item was a terminal auditor refusal")
    # a ROW-LESS cell fully refused by the judge: COMPLETE, never MISSING
    frame2 = build_world(drop_cells=(cell,))
    footing2 = _footing_with_judge({b: (6, 0, 0) for b in BEHAVIORS3},
                                   {b: six for b in BEHAVIORS3})
    assert petri_stats._fully_refused_cell_defect(footing2, list(BEHAVIORS3)) is None
    provenance = _provenance_for(frame2)
    provenance["cells"][cell] = _enumeration_payload(LENS)
    analysis2 = _analyze(frame2, cell_footing={cell: footing2},
                         provenance=provenance)
    assert analysis2["completeness"]["cells"][cell]["status"] == "COMPLETE"
    assert analysis2["by_cell"][cell]["judge_refused"]["total"] == 18
    assert "MISSING" not in json.dumps(analysis2)
    # ...and a defect names the judge term when it is the one that breaks
    short = copy_footing = json.loads(json.dumps(footing2))
    short["judge_refused"] = {"total": 17}
    assert petri_stats._fully_refused_cell_defect(short, list(BEHAVIORS3)) == (
        "auditor_refused.total 0 + judge_refused.total 17 != audited.expected 18")
    del copy_footing


def _pin_pre_2026_09_10_vocabulary(monkeypatch):
    """The before/after comparison is about the judge term — not the vocabulary and not the
    enumeration. The historical snapshot (PRE_CHANGE_SHA, 2026-09-08) derives its canonical
    matrix, its confirmatory family and its per-SI aggregates from ``dc.CLASS_A_METHODS`` x
    ``cm.PETRI_SI_IDS`` off the modules it imports, which on this tree are the ones two
    2026-09-10 landings widened:

    - 61c012a87: ``cm.PETRI_SI_IDS`` gained the PROMPTED-ONLY `constitution_full` (a prompted
      baseline whose `si_…__<method>` ids the current parser refuses);
    - 0caa51c62 ("Keep Petri's confirmatory family fixed while admitting the derived arm"):
      the serving grammar and the statistical family were separated — ``dc.CLASS_A_METHODS``
      is now the 13-method family PLUS ``dc.OBJECT_ONLY_METHODS`` (document-corpus arms served
      under the object-only SIs only); the current reader iterates
      ``dc.PROMPTABLE_CLASS_A_METHODS`` x ``cm.PETRI_CLASS_A_SI_IDS`` and enumerates the
      object-only cells apart (``object_only_cell_ids``), while the snapshot still crosses
      every CLASS_A method with every SI (18 methods against the reader's 13 x 3: phantom
      expected cells and UNAVAILABLE per-SI aggregates — the red this pin was written for).

    So pin BOTH modules, for the comparison's duration, to the 2026-08-28 vocabulary and to
    the product's CURRENT fixed family: the snapshot then enumerates exactly what the reader
    exposes today (its callers assert ``canonical_cell_ids`` agree — derived, never typed),
    and with the object-only methods emptied ``object_only_cell_ids()`` is ``[]`` on the new
    reader, so the one key it adds (``_without_object_only_cells``) is empty by construction.
    """
    monkeypatch.setattr(cm, "PETRI_PROMPTABLE_SI_IDS", ("safety", "constitution", "irrelevant"))
    monkeypatch.setattr(cm, "PETRI_PROMPTED_ONLY_SI_IDS", ())
    # the snapshot's SI loop: the three Class A SIs and no object-only SI — an object-only
    # cell is not a leg of any method's per-SI view on either module
    monkeypatch.setattr(cm, "PETRI_SI_IDS", ("safety", "constitution", "irrelevant"))
    # the Class A SIs are the PROMPTABLE ones (never an object-only SI: those cells are the object-only methods'
    # — petri_stats.object_only_cell_ids — on both modules)
    monkeypatch.setattr(cm, "PETRI_CLASS_A_SI_IDS", ("safety", "constitution", "irrelevant"))
    # the snapshot's family and matrix = the product's current fixed family; the object-only
    # methods are a post-change concept with no pre-change counterpart
    monkeypatch.setattr(dc, "CLASS_A_METHODS", tuple(dc.PROMPTABLE_CLASS_A_METHODS))
    monkeypatch.setattr(dc, "OBJECT_ONLY_METHODS", ())


def _without_object_only_cells(analysis):
    """``analysis`` with the ONE key the 2026-09-10 enumeration landing (0caa51c62) added
    popped, asserted empty first: under the pin there is no object-only cell to list, so the
    pre-change reader (which has no such key) and the new one must otherwise agree byte for
    byte — the ``ADDED_KEYS`` discipline of test_petri_stats_family_regression."""
    assert analysis["by_class"]["class_a"].pop("object_only_cells") == []
    return analysis


def _write_complete_cell_fixture(dest=None):
    """Write (or rewrite) the hermetic COMPLETE-cell fixture under ``dest`` (default
    ``FIXTURE_CELL_ROOT``) THROUGH the product runner — never by hand: the
    test_petri_store_shape world is built in a temp root, its base cell `none` is run to
    completion on the synthetic driver (every item scored, no refusal of any kind), and the
    cell's rows.jsonl / results.json / manifest.json plus the enumeration record (the
    cells.json payload ``print-cells`` freezes) are written out with every wall-clock stamp
    pinned to ``FIXTURE_TIME`` and the object cell's artifact path made relative to the world
    root, so two generations are byte-identical and no absolute path lands in the repo.
    ``fixture.json`` records the denominator inputs (behaviors, scenario ids, epochs), the
    complete cell's run key and the tree the generator ran on. Regenerate deliberately when
    the runner's on-disk shape legitimately changes::

        PYTHONPATH=src python -c "import sys; sys.path.insert(0, 'tests'); \\
            import test_petri_judge_refusal as t; t._write_complete_cell_fixture()"
    """
    dest = FIXTURE_CELL_ROOT if dest is None else Path(dest)
    with tempfile.TemporaryDirectory(prefix="petri-complete-cell-fixture-") as tmp:
        tmp = Path(tmp)
        w = _build_world(tmp / "world")
        cell = w.cells[dc.BASE_CELL_ID]
        store, result = _run_cell(w, cell, tmp / "run", _driver(w, cell))
        assert store.is_done() and result["incomplete"] is False and result["unscored"] == 0
        assert result["auditor_refused"]["total"] == 0 and "judge_refused" not in result
        cell_set = dc.enumerate_cells(w.spec, _source(w.root), behavior_files=w.bf,
                                      prompt_files=PROMPT_FILES)
        assert cell_set.absent == ()
        assert {c.cell_id for c in cell_set.all_served()} == set(w.cells)
        record = petri_run._cells_payload(cell_set, w.spec)
        for entry in record["canonical"] + record["controls"]:
            if entry.get("object_path"):
                entry["object_path"] = Path(entry["object_path"]).relative_to(w.root).as_posix()
        record["written_at"] = FIXTURE_TIME
        rows = [json.loads(line) for line in
                (store.cell_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
        for r in rows:
            r["created_at"] = FIXTURE_TIME
        results = json.loads((store.cell_dir / "results.json").read_text(encoding="utf-8"))
        results["completed_at"] = FIXTURE_TIME
        manifest = json.loads((store.cell_dir / "manifest.json").read_text(encoding="utf-8"))
        manifest["resolved_at"] = FIXTURE_TIME
        scenario_ids = {b: list(psc.scenario_ids(w.bundle.bundle_dir / b / "scenarios"))
                        for b in w.spec.behaviors}
        meta = {
            "note": ("SYNTHETIC complete Petri cell (the test_petri_store_shape world: "
                     "invented scenario ids, placeholder hashes, no recorded data) written "
                     "through petri_runner._run by tests/test_petri_judge_refusal.py::"
                     "_write_complete_cell_fixture — regenerate through it, never hand-edit"),
            "cell_id": cell.cell_id,
            "complete_run_key": cell.petri_run_key,
            "behaviors": list(w.spec.behaviors),
            "scenario_ids": scenario_ids,
            "epochs": w.spec.bloom_options.epochs,
            "time_pin": FIXTURE_TIME,
            "generated_on_tree": _tree_sha(),
        }
    cell_dir = dest / "cells" / cell.petri_run_key
    cell_dir.mkdir(parents=True, exist_ok=True)

    def dump(obj):
        return json.dumps(obj, indent=2, sort_keys=True) + "\n"

    (dest / "cells.json").write_text(dump(record), encoding="utf-8")
    (dest / "fixture.json").write_text(dump(meta), encoding="utf-8")
    (cell_dir / "rows.jsonl").write_text(
        "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (cell_dir / "results.json").write_text(dump(results), encoding="utf-8")
    (cell_dir / "manifest.json").write_text(dump(manifest), encoding="utf-8")
    return dest


def _tree_sha():
    proc = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _fixture_complete_cell():
    """The hermetic twin of ``_real_cell_or_skip`` for the COMPLETE cell: the fixture's
    denominator inputs, its enumeration record, the served legs, the complete cell's
    ``PetriCell`` and its cell dir — read exactly the way the real state root was."""
    meta = json.loads((FIXTURE_CELL_ROOT / "fixture.json").read_text(encoding="utf-8"))
    record = json.loads((FIXTURE_CELL_ROOT / "cells.json").read_text(encoding="utf-8"))
    served = list(record.get("canonical") or []) + list(record.get("controls") or [])
    (entry,) = [c for c in served if c.get("petri_run_key") == meta["complete_run_key"]]
    cell = dc.PetriCell(**entry)
    assert cell.cell_id == meta["cell_id"] == dc.BASE_CELL_ID
    return meta, record, served, cell, FIXTURE_CELL_ROOT / "cells" / meta["complete_run_key"]


def test_stats_third_term_is_inert_without_judge_refusals_synthetic(tmp_path, monkeypatch):
    """Item 1, condition (b), synthetic: with no judge refusal the new
    reader's output is byte-identical to the pre-change reader's — on the
    plain world, on an auditor-only refused cell, and on the driver's
    tolerant ``judge_refused: {"total": 0}`` footing — once both readers
    enumerate the same family (the pin) and the enumeration landing's one
    added key is set aside. The object-only cells leave the worlds
    explicitly: a post-change concept with no pre-change counterpart."""
    object_only = petri_stats.object_only_cell_ids()  # read BEFORE the pin empties it
    _pin_pre_2026_09_10_vocabulary(monkeypatch)
    pre = _pre_change_petri_stats(tmp_path)
    assert petri_stats.canonical_cell_ids() == pre.canonical_cell_ids()
    method = dc.CLASS_A_METHODS[0]
    cell = f"si_constitution__{method}"
    frame_all = build_world(drop_cells=("prompted__constitution_full", *object_only))
    kw_all = dict(expected_behaviors=list(BEHAVIORS3), n_boot=200, seed=7,
                  provenance=_provenance_for(frame_all))
    assert _canonical(_without_object_only_cells(petri_stats.analyze(frame_all, **kw_all))) \
        == _canonical(pre.analyze(frame_all, **kw_all))
    frame = build_world(drop_cells=(cell, "prompted__constitution_full", *object_only))
    provenance = _provenance_for(frame)
    provenance["cells"][cell] = _enumeration_payload(LENS)
    footing = _cell_footing({b: (6, 0, 6) for b in BEHAVIORS3}, run_key=RUN_KEY)
    for variant in (footing, {**footing, "judge_refused": {"total": 0}}):
        kw = dict(expected_behaviors=list(BEHAVIORS3), n_boot=200, seed=7,
                  cell_footing={cell: variant}, provenance=provenance)
        new = petri_stats.analyze(frame, **kw)
        assert "judge_refused" not in new["by_cell"][cell]
        assert _canonical(_without_object_only_cells(new)) == _canonical(pre.analyze(frame, **kw))


def test_stats_third_term_is_inert_on_the_real_complete_cell(tmp_path, monkeypatch):
    """Item 1, condition (b), on a COMPLETE cell as the runner wrote it:
    ``analyze`` on the fixture cell `none` (its rows.jsonl, its results.json
    footing, the frozen record's provenance — ``FIXTURE_CELL_ROOT``, written
    through the runner by ``_write_complete_cell_fixture``; the real GLM
    full22 cell this node used to read lives under a machine-local state
    directory and made the node environment-bound) gives byte-identical output
    from the pre-change and the new reader — with the footing as
    results.json carries it and as collect_audited_footing hands it over (a
    zero judge tally) — once both readers enumerate the same family (the
    pin) and the enumeration landing's one added key is set aside. The
    denominator is derived from the fixture's scenario ids x epochs, never
    read back from the footing it checks."""
    _pin_pre_2026_09_10_vocabulary(monkeypatch)
    meta, record, served, cell, cell_dir = _fixture_complete_cell()
    pre = _pre_change_petri_stats(tmp_path)
    assert petri_stats.canonical_cell_ids() == pre.canonical_cell_ids()
    rows = [json.loads(line) for line in
            (cell_dir / "rows.jsonl").read_text(encoding="utf-8").splitlines()]
    frame = pd.DataFrame(rows)
    results = json.loads((cell_dir / "results.json").read_text(encoding="utf-8"))
    manifest = json.loads((cell_dir / "manifest.json").read_text(encoding="utf-8"))
    footing = {k: results[k]
               for k in ("audited", "auditor_refused", "auditor_chain", "behaviors")}
    footing["petri_run_key"] = meta["complete_run_key"]
    footing["engine"] = manifest["engine"]
    provenance = {
        "audit": record["audit"], "model_id": manifest["model_id"],
        "model_revision": manifest["model_revision"],
        "cells": {c["cell_id"]: {"engine": c["serving_identity"],
                                 "intervention_hash": c["intervention_hash"],
                                 "harness_config_hash": c["harness_config_hash"],
                                 "petri_run_key": c["petri_run_key"]}
                  for c in served}}
    kw = dict(expected_cells=list(record["expected_cells"]),
              expected_behaviors=list(meta["behaviors"]),
              n_boot=200, seed=7, provenance=provenance)
    n_expected = sum(len(meta["scenario_ids"][b]) for b in meta["behaviors"]) * meta["epochs"]
    for variant in (footing, {**footing, "judge_refused": {"total": 0}}):
        new = petri_stats.analyze(frame, cell_footing={cell.cell_id: variant}, **kw)
        old = pre.analyze(frame, cell_footing={cell.cell_id: variant}, **kw)
        assert _canonical(_without_object_only_cells(new)) == _canonical(old)
        af = new["by_cell"][cell.cell_id]["audited_footing"]
        assert af["audited"] == {"items": n_expected, "expected": n_expected}
        assert "judge_refused" not in new["by_cell"][cell.cell_id]
