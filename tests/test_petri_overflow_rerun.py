"""The context-overflow re-run pass, driver side (2026-08-27;
petri_runner module docstring "context-overflow protocol").

scripts/petri_run.py `run --overflow-rerun` is an ASSERTION of intent,
not a scenario selector: it refuses unless the cell's failure ledger
actually holds overflow-classified unresolved entries (the converter's
OVERFLOW_REASON_PREFIX contract), arms PETRI_OVERFLOW_RERUN so the
runner caps its eval width to rerun_serving_geometry(), and leaves
WHICH scenarios re-run entirely to the ordinary resume machinery.
CPU-only; no inspect_ai, no vLLM."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import petri_run as prun  # noqa: E402

from sisteer.capeval.tiers import petri_runner as pr  # noqa: E402


def _ledger_entry(reason, sid="scenario_a", dim="self_preservation"):
    return {"behavior": "self_preservation", "scenario_id": sid,
            "epoch": 1, "dimension": dim, "reason": reason, "retries": 0}


def _write_ledger(cell_dir: Path, entries) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / prun.FAILURE_LEDGER).write_text(json.dumps({
        "version": 1, "cell_id": "cell", "petri_run_key": "ab" * 8,
        "written_at": 0.0, "unresolved": list(entries)}), encoding="utf-8")


def test_overflow_ledger_entries_filters_on_the_converter_prefix():
    casualty = _ledger_entry(pr.overflow_reason("maximum context length"))
    other = _ledger_entry("no judge scores on sample", sid="scenario_b")
    assert prun.overflow_ledger_entries([casualty, other]) == [casualty]
    assert prun.overflow_ledger_entries([other]) == []
    assert prun.overflow_ledger_entries([]) == []


def test_arm_refuses_without_overflow_casualties(tmp_path, monkeypatch):
    monkeypatch.delenv(pr.PETRI_OVERFLOW_RERUN_ENV, raising=False)
    # no ledger at all
    with pytest.raises(prun.DriverError, match="overflow"):
        prun._arm_overflow_rerun(tmp_path / "cell", "cell")
    # a ledger with only non-overflow unresolved entries
    cell_dir = tmp_path / "cell2"
    _write_ledger(cell_dir, [_ledger_entry("no judge scores on sample")])
    with pytest.raises(prun.DriverError, match="overflow"):
        prun._arm_overflow_rerun(cell_dir, "cell2")
    # a refusal must never leave the re-run armed
    import os
    assert os.environ.get(pr.PETRI_OVERFLOW_RERUN_ENV) is None


def test_arm_sets_the_env_and_names_the_geometry(tmp_path, monkeypatch,
                                                 capsys):
    # setenv first so monkeypatch restores the pre-test state even
    # though the function under test overwrites the value itself
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    cell_dir = tmp_path / "cell"
    _write_ledger(cell_dir, [
        _ledger_entry(pr.overflow_reason("maximum context length is")),
        _ledger_entry("no judge scores on sample", sid="scenario_b"),
    ])
    prun._arm_overflow_rerun(cell_dir, "cell")
    import os
    assert os.environ[pr.PETRI_OVERFLOW_RERUN_ENV] == "1"
    out = capsys.readouterr().out
    geo = pr.rerun_serving_geometry()
    # the operator contract is printed loudly: the serving stack's
    # required geometry and the from-turn-1 semantics
    assert f"max_model_len={geo['max_model_len']}" in out
    assert f"max_num_seqs={geo['max_num_seqs']}" in out
    assert "FROM TURN 1" in out
    assert "scenario_a" in out and "scenario_b" not in out


def test_rerun_product_never_exceeds_the_certified_one(monkeypatch):
    """The re-run widens the window by halving the sequence count, so the
    batched-token product — which is what memory is actually paid on,
    twice over (KV at 128 KiB/token plus the whole-budget warmup
    intermediate) — must never rise above the certified product.

    The hazard is the `max(1, seqs // scale)` clamp: at seqs=1 it cannot
    halve, so 1 x 2L is DOUBLE the certified 1 x L. Jobs 229194/229390
    measured that an over-certified product does not fit one H200 and
    cannot be rescued by raising utilization (it is zero-sum between KV
    and warmup headroom), so this refuses rather than emitting a
    geometry the hardware has already rejected.
    """
    from sisteer.capeval.serving import vllm_server as vsrv

    window = pr._default_serving_window()

    # the ordinary case: product held exactly, not merely bounded
    monkeypatch.setattr(vsrv, "DEFAULT_MAX_NUM_SEQS", 24)
    geom = pr.rerun_serving_geometry()
    assert geom == {"max_model_len": window * 2, "max_num_seqs": 12}
    assert geom["max_num_seqs"] * geom["max_model_len"] == 24 * window

    # truncation may only LOWER the product — safe, so still allowed
    monkeypatch.setattr(vsrv, "DEFAULT_MAX_NUM_SEQS", 25)
    geom = pr.rerun_serving_geometry()
    assert geom["max_num_seqs"] * geom["max_model_len"] <= 25 * window

    # the clamp case must refuse, naming the measured reason
    monkeypatch.setattr(vsrv, "DEFAULT_MAX_NUM_SEQS", 1)
    with pytest.raises(pr.PetriRunnerError) as excinfo:
        pr.rerun_serving_geometry()
    assert "exceeds the certified product" in str(excinfo.value)
    assert "229194" in str(excinfo.value)


# --- the zero-progress guard ---------------------------
#
# A --overflow-rerun pass that resolves NOTHING must refuse loudly, never
# return rc=3 with an unchanged count that reads like progress. The
# failure it kills: a casualty's sample COMPLETED at the inspect level
# (the judge scored the truncated transcript; only the converter
# withholds the rows), so a resume sub-namespace log dir already holding
# those completed samples makes eval_set replay them verbatim on every
# pass — observed live on the gemma4e4b audit: three passes, 27 -> 27 ->
# 27, byte-identical model-usage totals, 1-3s each.


def test_arm_returns_the_armed_casualties(tmp_path, monkeypatch):
    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    casualty = _ledger_entry(pr.overflow_reason("maximum context length is"))
    other = _ledger_entry("no judge scores on sample", sid="scenario_b")
    cell_dir = tmp_path / "cell"
    _write_ledger(cell_dir, [casualty, other])
    armed = prun._arm_overflow_rerun(cell_dir, "cell")
    assert armed == [casualty]


def test_zero_progress_with_identical_evidence_is_the_replay_refusal(
        tmp_path):
    """Every armed casualty survives with a byte-identical reason ->
    nothing re-executed; the refusal names the replay mechanism and the
    quarantine remedy."""
    casualties = [
        _ledger_entry(pr.overflow_reason("maximum context length is"),
                      sid="scenario_a"),
        _ledger_entry(pr.overflow_reason("maximum context length is"),
                      sid="scenario_b"),
    ]
    cell_dir = tmp_path / "cell"
    _write_ledger(cell_dir, casualties)  # the "after" ledger: unchanged
    with pytest.raises(prun.DriverError) as excinfo:
        prun._assert_overflow_rerun_progress(cell_dir, "cell", casualties)
    msg = str(excinfo.value)
    assert "resolved NONE" in msg
    assert "resume" in msg
    assert "never delete" in msg


def test_zero_progress_with_changed_evidence_names_the_window(tmp_path):
    """Same unresolved keys but FRESH evidence -> the scenarios genuinely
    re-ran and still overflowed at the re-run window; the refusal must
    say a retry cannot converge, not blame the replay."""
    armed = [_ledger_entry(pr.overflow_reason("maximum context length is "
                                              "16384 tokens"))]
    after = [_ledger_entry(pr.overflow_reason("maximum context length is "
                                              "32768 tokens"))]
    cell_dir = tmp_path / "cell"
    _write_ledger(cell_dir, after)
    with pytest.raises(prun.DriverError) as excinfo:
        prun._assert_overflow_rerun_progress(cell_dir, "cell", armed)
    msg = str(excinfo.value)
    assert "STILL overflowed" in msg
    assert "cannot converge" in msg


def test_partial_progress_does_not_refuse(tmp_path):
    """One armed casualty resolved (left the ledger) -> real progress;
    the ordinary INCOMPLETE path stands even though another survives
    unchanged."""
    kept = _ledger_entry(pr.overflow_reason("maximum context length is"),
                         sid="scenario_a")
    gone = _ledger_entry(pr.overflow_reason("maximum context length is"),
                         sid="scenario_b")
    cell_dir = tmp_path / "cell"
    _write_ledger(cell_dir, [kept])
    prun._assert_overflow_rerun_progress(cell_dir, "cell", [kept, gone])


def test_full_resolution_does_not_refuse(tmp_path):
    casualty = _ledger_entry(pr.overflow_reason("maximum context length is"))
    cell_dir = tmp_path / "cell"
    _write_ledger(cell_dir, [])
    prun._assert_overflow_rerun_progress(cell_dir, "cell", [casualty])


def _wire_cmd_run(tmp_path, monkeypatch, run_petri, evidence="standup"):
    """Stub cmd_run's heavy seams so the overflow-rerun wiring runs on
    CPU: a fake store, a fake runtime module, and pass-through record/
    enumeration checks. Returns (args, spec, cell_dir). ``evidence`` is the
    attestation the drive hands in: ``standup`` (--verify-standup, the
    native path), ``endpoint`` (--served-endpoint, the lens path) or
    ``none``."""
    from types import SimpleNamespace

    monkeypatch.setenv(pr.PETRI_OVERFLOW_RERUN_ENV, "0")
    out_root = tmp_path / "out"
    cell_dir = out_root / "cells" / ("ab" * 8)

    record = {"audit": "a", "expected_cells": [],
              "canonical": [], "controls": []}
    monkeypatch.setattr(prun, "read_cells_record", lambda _root: dict(record))
    monkeypatch.setattr(prun, "_check_record_matches_spec",
                        lambda *_a, **_k: None)
    monkeypatch.setattr(prun, "enumerate_petri", lambda _spec: None)
    monkeypatch.setattr(prun, "_cells_payload",
                        lambda _cs, _spec: dict(record))
    # the standup declaration reading refuses a hand record without a `canonical` list before the seams below are reached:
    # the frozen enumeration is a pass-through here too -- the record payload, no declaration (a bare-tip red since that landing)
    monkeypatch.setattr(prun, "_frozen_enumeration",       # (`rerun_subset_of=` rides along under --overflow-rerun, petri-completion)
                        lambda _a, _s, _sp, **_k: (None, dict(record), None, None))
    # the standup evidence is a pass-through here like the record checks (the real verify_standup_or_refuse wants the routing
    # table beside the cells_served.json; this world stubs the heavy seams, not the standup contract)
    monkeypatch.setattr(prun, "verify_standup_or_refuse", lambda *_a, **_k: None)

    cell = SimpleNamespace(
        cell_id="cell", virtual_model_name="petri__cell", control_of=None,
        si_prefix_hash=None, petri_run_key="ab" * 8,
        intervention_hash="ih", harness_config_hash="hh",
        serving_identity="proxy_vllm+passthrough+interleaved",
        kind="prompted")
    monkeypatch.setattr(prun, "_resolve_cell", lambda _served, _args: cell)
    monkeypatch.setattr(prun, "_row_constants",
                        lambda _spec, _cell, _seed: {})

    class FakeStore:
        def __init__(self, _root, _key, row_constants=None):
            self.cell_dir = cell_dir
            self.results_path = cell_dir / "results.json"

        def validate_resume(self, _resolved):
            pass

        def write_cell_manifest(self, _resolved):
            pass

        def is_done(self):
            return False

    monkeypatch.setattr(prun.cs, "CellStore", FakeStore)

    import sisteer.capeval as _cap
    rt_stub = SimpleNamespace(make_engine=lambda _ns: object(),
                              run_petri=run_petri)
    monkeypatch.setitem(sys.modules, "sisteer.capeval.runtime", rt_stub)
    monkeypatch.setattr(_cap, "runtime", rt_stub, raising=False)

    # X5: an armed pass needs the standup's ATTESTED window (--verify-standup cells_served.json,
    # attested_max_model_len) equal to the requested re-run window; the stub attests exactly what this process requests
    window = pr.rerun_serving_geometry()["max_model_len"]
    standup = tmp_path / "cells_served.json"
    standup.write_text(json.dumps({"engine": "lora_native", "stage": "final", "attested_max_model_len": window}), encoding="utf-8")
    endpoint = tmp_path / "endpoint.json"      # the lens drive's only record (X6): the pulled endpoint.json attests the window too
    endpoint.write_text(json.dumps({"node": "n", "slurm_job_id": "1", "mode": "petri", "lens_reduction_mode": "stacked",
                                    "attested_max_model_len": window}), encoding="utf-8")
    args = SimpleNamespace(cell_index=0, cell_id=None, seed=None, out_root=str(out_root), overflow_rerun=True,
                           verify_standup=str(standup) if evidence == "standup" else None,
                           served_endpoint=str(endpoint) if evidence == "endpoint" else None)
    spec = SimpleNamespace(audit="a", cells=[],
                           bloom_options=SimpleNamespace(seed=0, epochs=1),
                           target_model="m", target_revision="r")
    return args, spec, cell_dir


def test_cmd_run_noop_overflow_rerun_raises_instead_of_rc3(tmp_path,
                                                           monkeypatch):
    """The reported bug end-to-end: run_petri comes back with the ledger
    untouched (the eval_set replay) — cmd_run must RAISE, not return
    EXIT_INCOMPLETE as if the pass were ordinary progress."""
    def replaying_run_petri(_engine, _spec, _cell, _store, _seed):
        return {"unscored": 2, "rows_emitted": 0}  # ledger untouched

    args, spec, cell_dir = _wire_cmd_run(tmp_path, monkeypatch,
                                         replaying_run_petri)
    _write_ledger(cell_dir, [
        _ledger_entry(pr.overflow_reason("maximum context length is"),
                      sid="scenario_a"),
        _ledger_entry(pr.overflow_reason("maximum context length is"),
                      sid="scenario_b"),
    ])
    with pytest.raises(prun.DriverError, match="resolved NONE"):
        prun.cmd_run(args, spec)


def test_cmd_run_partial_overflow_progress_keeps_rc3(tmp_path, monkeypatch):
    """A pass that resolves SOME casualties is real progress: the
    ordinary INCOMPLETE contract (rc=3, no results.json) stands."""
    kept = _ledger_entry(pr.overflow_reason("maximum context length is"),
                         sid="scenario_a")
    gone = _ledger_entry(pr.overflow_reason("maximum context length is"),
                         sid="scenario_b")

    def resolving_run_petri(_engine, _spec, _cell, store, _seed):
        _write_ledger(store.cell_dir, [kept])  # scenario_b resolved
        return {"unscored": 1, "rows_emitted": 3}

    args, spec, cell_dir = _wire_cmd_run(tmp_path, monkeypatch,
                                         resolving_run_petri)
    _write_ledger(cell_dir, [kept, gone])
    assert prun.cmd_run(args, spec) == prun.EXIT_INCOMPLETE


def test_cmd_run_arms_from_the_endpoint_record_on_the_lens_path_and_refuses_with_no_attestation(tmp_path, monkeypatch):
    """X6: a lens drive passes no --verify-standup; the
    armed pass reads the proxy's attested window from --served-endpoint and
    runs (here: partial progress, the ordinary rc=3). With neither record it
    refuses before any sample, naming both flags."""
    kept = _ledger_entry(pr.overflow_reason("maximum context length is"), sid="scenario_a")
    gone = _ledger_entry(pr.overflow_reason("maximum context length is"), sid="scenario_b")

    def resolving_run_petri(_engine, _spec, _cell, store, _seed):
        _write_ledger(store.cell_dir, [kept])
        return {"unscored": 1, "rows_emitted": 3}

    args, spec, cell_dir = _wire_cmd_run(tmp_path, monkeypatch, resolving_run_petri, evidence="endpoint")
    _write_ledger(cell_dir, [kept, gone])
    assert prun.cmd_run(args, spec) == prun.EXIT_INCOMPLETE
    import os
    assert os.environ[pr.PETRI_SERVED_MAX_MODEL_LEN_ENV] == str(pr.rerun_serving_geometry()["max_model_len"])
    args, spec, cell_dir = _wire_cmd_run(tmp_path, monkeypatch, resolving_run_petri, evidence="none")
    _write_ledger(cell_dir, [kept, gone])
    with pytest.raises(prun.DriverError, match="needs the proxy's attestation"):
        prun.cmd_run(args, spec)


# --- the driver refuses a rejected pass by name, and an armed COMPLETE at zero rows ---------------------------------------------


def _write_ledger_payload(cell_dir: Path, entries, **extra) -> None:
    cell_dir.mkdir(parents=True, exist_ok=True)
    (cell_dir / prun.FAILURE_LEDGER).write_text(json.dumps({
        "version": 1, "cell_id": "cell", "petri_run_key": "ab" * 8,
        "written_at": 0.0, "unresolved": list(entries), **extra}), encoding="utf-8")


def _hard_error_record(sid="scenario_a", status=400, etype="invalid_request_error"):
    return {"behavior": "self_preservation", "scenario_id": sid, "epoch": 1,
            "dimensions": ["eval_awareness", "scenario_realism", "self_preservation"],
            "hard_error": {"sample_id": sid, "sample_uuid": None, "eval_log": None, "model": "anthropic/claude-sonnet-5",
                           "role": "auditor", "status": status, "error_type": etype, "attempts": 1,
                           "error": "BadRequestError('Error code: 400 - ...does not support the `fallbacks` parameter.')",
                           "why": "final status HTTP 400 is not one a ladder retries"}}


def test_a_ledger_recording_rejected_requests_refuses_before_the_replay_diagnosis(tmp_path):
    """Byte-identical casualties PLUS hard_errors is the rejected pass, not the eval_set replay: the refusal names the HTTP error and
    the request fix, never the quarantine remedy."""
    casualty = _ledger_entry(pr.overflow_reason("maximum context length is"))
    cell_dir = tmp_path / "cell"
    _write_ledger_payload(cell_dir, [casualty], **{pr.HARD_ERRORS_KEY: [_hard_error_record()]})
    with pytest.raises(prun.DriverError) as excinfo:
        prun._assert_overflow_rerun_progress(cell_dir, "cell", [casualty])
    msg = str(excinfo.value)
    assert "REJECTED by the API" in msg and "HTTP 400 invalid_request_error" in msg and "never delete" not in msg and "resolved NONE" not in msg
    # a withheld-terminal record is the other named refusal
    _write_ledger_payload(cell_dir, [casualty], **{pr.TERMINAL_WITHHELD_KEY: {"why": "zero resolutions", "auditor_refused": [{}]}})
    with pytest.raises(prun.DriverError, match="resolved NOTHING"):
        prun._assert_overflow_rerun_progress(cell_dir, "cell", [casualty])
    # neither key: today's diagnoses stand
    _write_ledger_payload(cell_dir, [casualty])
    with pytest.raises(prun.DriverError, match="resolved NONE"):
        prun._assert_overflow_rerun_progress(cell_dir, "cell", [casualty])


def test_cmd_run_refuses_a_rejected_pass_even_when_the_runner_reports_zero_unscored(tmp_path, monkeypatch):
    """The incident's exact surface: run_petri comes back unscored 0 / rows 0 (every casualty 'terminal'). With hard_errors in the
    ledger cmd_run refuses by the HTTP error; without them, an armed pass at zero rows whose auditor sidecar GREW is refused by the
    shared zero-resolution rule — in neither case is results.json written or COMPLETE printed; a zero-row pass with no new auditor
    record (judge re-rolls only) and a scoring pass both complete."""
    casualty = _ledger_entry(pr.overflow_reason("maximum context length is"))

    def rejected_run_petri(_engine, _spec, _cell, store, _seed):
        _write_ledger_payload(store.cell_dir, [casualty], **{pr.HARD_ERRORS_KEY: [_hard_error_record()]})
        return {"unscored": 0, "rows_emitted": 0}

    args, spec, cell_dir = _wire_cmd_run(tmp_path, monkeypatch, rejected_run_petri)
    _write_ledger(cell_dir, [casualty])
    with pytest.raises(prun.DriverError, match="REJECTED by the API"):
        prun.cmd_run(args, spec)
    assert not (cell_dir / "results.json").exists()

    # the SAME rule as the runner's, re-read from the evidence: a runner that settled the ledger at zero
    # rows by writing NEW auditor-terminal records is refused by the sidecar's growth across the pass
    shed = pr.SampleShed(kind=pr.SHED_UNDETERMINED, sample_id="scenario_a", sample_uuid=None, eval_log=None,
                         model="anthropic/claude-sonnet-5", role=pr.AUDITOR_ROLE, error="Error code: 529", excerpt=None,
                         excerpt_source="none", turns_reached=0, n_messages=1, last_target_output=None,
                         last_target_output_source="none", window_start=None, window_end=None, siblings_succeeded=0,
                         siblings_failed=0, siblings_total=0, sibling_success_timestamps=(), sibling_failure_timestamps=(),
                         auditor_chain=None, basis="synthetic", final_status=529)
    record = pr.AuditorRefusal(behavior="self_preservation", scenario_id="scenario_a", epoch=1,
                               dimensions=tuple(pr.required_metrics("self_preservation")), shed=shed)

    def terminal_only_run_petri(_engine, _spec, _cell, store, _seed):
        _write_ledger(store.cell_dir, [])          # the ledger settled by terminal records alone, zero rows
        (store.cell_dir / pr.AUDITOR_REFUSED_LEDGER_NAME).write_text(json.dumps(
            {"version": 1, "cell_id": "cell", "petri_run_key": "ab" * 8, pr.AUDITOR_REFUSED_OUTCOME: [
                pr.asdict(record) | {"dimensions": list(record.dimensions)}]}), encoding="utf-8")
        return {"unscored": 0, "rows_emitted": 0}

    args, spec, cell_dir = _wire_cmd_run(tmp_path, monkeypatch, terminal_only_run_petri)
    _write_ledger(cell_dir, [casualty])
    with pytest.raises(prun.DriverError, match="1 NEW auditor-terminal record"):
        prun.cmd_run(args, spec)
    assert not (cell_dir / "results.json").exists()

    # the wiring's fake store never reached the done marker before; each wiring mints a fresh class, so the writer is given
    # AFTER each one, letting the ordinary completions land
    def _with_results_writer():
        monkeypatch.setattr(prun.cs.CellStore, "write_results",
                            lambda self, d: (self.cell_dir / "results.json").write_text(json.dumps(d), encoding="utf-8"), raising=False)

    def judge_only_run_petri(_engine, _spec, _cell, store, _seed):
        _write_ledger(store.cell_dir, [])          # settled at zero rows with NO new auditor-terminal record (judge re-rolls only)
        return {"unscored": 0, "rows_emitted": 0}

    args, spec, cell_dir = _wire_cmd_run(tmp_path, monkeypatch, judge_only_run_petri); _with_results_writer()
    _write_ledger(cell_dir, [casualty])
    assert prun.cmd_run(args, spec) == 0      # outside the floor: a pass whose only work was judge re-rolls completes
    assert (cell_dir / "results.json").exists()

    def scoring_run_petri(_engine, _spec, _cell, store, _seed):
        _write_ledger(store.cell_dir, [])
        return {"unscored": 0, "rows_emitted": 3}

    args, spec, cell_dir = _wire_cmd_run(tmp_path, monkeypatch, scoring_run_petri); _with_results_writer()
    _write_ledger(cell_dir, [casualty])
    assert prun.cmd_run(args, spec) == 0      # rows filed: the ordinary completion stands
    assert (cell_dir / "results.json").exists()
