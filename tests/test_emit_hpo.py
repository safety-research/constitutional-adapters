"""Native `fit_search` / `fit_trial` emission (`sisteer.registry.emit_hpo`): the opening record, trials at
their TERMINAL outcome only, a superseding progress record, the closing record with the winner and the fit diagnostics -- all through
`emit.RecordWriter`, folded by the harvester into rows that validate; the `hpo-native:` id prefix never collides with the post-hoc
reader's `hpo:` keys; the named params ride `search_space_json` / `params_json`; a `running` status, an unknown method and a complete
trial without an objective are refused by name before anything is written."""
from __future__ import annotations

import itertools
import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import emit, emit_hpo, identity, synth
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import validate as VAL
from sisteer.registry.adapters import hpo_logs

SPACE = {"lr": {"kind": "log_uniform", "low": 1e-5, "high": 1e-3}, "lora_rank": {"kind": "choice", "values": [4, 16, 64]},
         "lora_modules": {"kind": "choice", "values": [["o_proj"], ["q_proj", "k_proj", "v_proj", "o_proj"]]},
         "warmup_frac": {"kind": "choice", "values": [0.15]}, "max_grad_norm": {"kind": "choice", "values": [0.37]}}


def _root(tmp_path):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    R.write_records(os.path.join(root, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    assert H.ingest_once(root, workers=0, min_free=0).refused == 0
    return root


def test_search_id_prefix_is_distinct_from_the_post_hoc_reader():
    sid = emit_hpo.search_id_for("c", "m", "lora", synth.t(0))
    assert sid.startswith("hpo-native:") and len(sid) == len("hpo-native:") + 16
    assert not sid.startswith("hpo:") and hpo_logs.search_id_for("l", "m", "a", "p", "s", "k").startswith("hpo:")
    assert emit_hpo.per_expert_rank(4) == {"shared_expert": 4, "routed_expert": 1} and emit_hpo.per_expert_rank(64) == {"shared_expert": 64, "routed_expert": 8}


def test_open_trials_progress_close_fold_into_one_search_and_its_trials(tmp_path, monkeypatch):
    root = _root(tmp_path)
    spool = os.path.join(root, "records", "hpo")
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    kw = dict(campaign=synth.CAMPAIGN, spool=spool, code_sha="a" * 40, substrate="cluster_a")
    sid = emit_hpo.open_search(model_id=synth.FIT_MODEL_ID, object_kind="lora", method="sobol", search_space=SPACE, n_trials=3,
                               extra={"clip_percentile": 95, "design": "sobol16_coverage"}, started_utc=synth.t(0),
                               grad_norm_reconnaissance_uri="sha256:" + "b" * 64, **kw)
    opening = None
    for p in R.iter_record_files(spool):
        for _i, rec, _sha in R.read_record_file(p, compute_sha=False):
            if rec["record_type"] == "fit_search":
                opening = rec["payload"]
    assert opening and opening["search_id"] == sid and opening["n_trials_completed"] == 0 and opening["extra"]["phase"] == "open"
    assert opening["search_space_json"]["lora_modules"]["values"][1] == ["q_proj", "k_proj", "v_proj", "o_proj"]   # a space's VALUES are not re-sorted
    # a trial may not be recorded while running, without a terminal status, or complete without its objective
    with pytest.raises(emit_hpo.HpoEmitRefused):
        emit_hpo.record_trial(sid, "r1_t00", {"lr": 1e-4}, status="running", **kw)
    with pytest.raises(emit_hpo.HpoEmitRefused):
        emit_hpo.record_trial(sid, "r1_t00", {"lr": 1e-4}, status="complete", **kw)
    with pytest.raises(emit_hpo.HpoEmitRefused):
        emit_hpo.open_search(model_id="m", object_kind="lora", method="lucky_guess", search_space=SPACE, n_trials=1, **kw)
    with pytest.raises(emit_hpo.HpoEmitRefused):
        emit_hpo.open_search(model_id="m", object_kind="lora", method="sobol", search_space=SPACE, n_trials=1, substrate="cluster_c",
                             campaign=synth.CAMPAIGN, spool=spool, code_sha="a" * 40)
    opening = emit_hpo.update_search(sid, opening=opening, n_trials_completed=0, trials_in_flight=["r1_t00", "r1_t01"], job_ids=["9001"], **kw)
    assert opening["extra"]["trials_in_flight"] == ["r1_t00", "r1_t01"] and opening["job_ids_json"] == ["9001"]
    params = {"lr": 1e-4, "lora_rank": 16, "lora_modules": ["v_proj", "q_proj", "k_proj", "o_proj"], "per_expert_rank": emit_hpo.per_expert_rank(16),
              "warmup_frac": 0.15, "max_grad_norm": 0.37, "init_scheme": "random_lecun_1e-3", "trainable_params": 860160, "lora_set_id": "qkvo"}
    emit_hpo.record_trial(sid, "r1_t00", params, status="complete", objective_value=2.31, clip_fraction=0.02, epochs_seen=3, wandb_run_id="w1",
                          started_utc=synth.t(1), finished_utc=synth.t(2), **kw)
    emit_hpo.record_trial(sid, "r1_t01", dict(params, lr=3e-4), status="failed", extra={"reason": "walltime; abandoned"}, **kw)
    emit_hpo.record_trial(sid, "r1_t02", dict(params, lr=5e-5, lora_rank=4), status="pruned", **kw)
    emit_hpo.close_search(sid, opening=opening, winner_params=params, winner_trial_id="r1_t00", winner_objective=2.31, n_trials_completed=1,
                          argmin_interior=True, quadratic_fit_positive_definite=True, quadratic_fit_params={"a": 1.0}, recentre_rounds=1,
                          confirmation_trial_id="r2_t00", confirmation_within_noise=True, noise_estimate=0.01, job_ids=["9002"],
                          trials_uri="sha256:" + "c" * 64, **kw)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        fs = conn.execute("SELECT n_trials, n_trials_completed, winner_trial_id, winner_objective, argmin_interior, quadratic_fit_positive_definite, "
                          "recentre_rounds, confirmation_trial_id, confirmation_within_noise, noise_estimate, substrate, job_ids_json, winner_params_json, "
                          "search_space_json, extra_json, method, grad_norm_reconnaissance_uri, trials_uri FROM fit_search WHERE search_id = ?", (sid,)).fetchone()
        assert fs[0] == 3 and fs[1] == 1 and fs[2] == "r1_t00" and fs[3] == 2.31 and fs[4] == 1 and fs[5] == 1 and fs[6] == 1
        assert fs[7] == "r2_t00" and fs[8] == 1 and fs[9] == 0.01 and fs[10] == "cluster_a" and json.loads(fs[11]) == ["9001", "9002"]
        assert json.loads(fs[12])["lora_modules"] == ["k_proj", "o_proj", "q_proj", "v_proj"]           # the winner's modules are a SORTED list
        assert json.loads(fs[12])["per_expert_rank"] == {"shared_expert": 16, "routed_expert": 2}
        assert json.loads(fs[13]) == emit_hpo._sorted_modules(SPACE) and fs[15] == "sobol"
        ex = json.loads(fs[14])
        assert ex["phase"] == "closed" and ex["trials_in_flight"] == [] and ex["clip_percentile"] == 95 and ex["model_id"] == synth.FIT_MODEL_ID
        assert fs[16] == "sha256:" + "b" * 64 and fs[17] == "sha256:" + "c" * 64
        assert conn.execute("SELECT COUNT(*) FROM fit_search").fetchone()[0] == 1                            # three records, ONE row (superseding)
        trials = {r[0]: r for r in conn.execute("SELECT trial_id, status, objective_value, clip_fraction, epochs_seen, params_json, extra_json FROM fit_trial WHERE search_id = ?", (sid,)).fetchall()}
        assert set(trials) == {"r1_t00", "r1_t01", "r1_t02"}
        assert trials["r1_t00"][1:5] == ("complete", 2.31, 0.02, 3) and json.loads(trials["r1_t00"][5])["lora_modules"] == ["k_proj", "o_proj", "q_proj", "v_proj"]
        assert trials["r1_t01"][1] == "failed" and json.loads(trials["r1_t01"][6])["reason"].startswith("walltime")
        assert trials["r1_t02"][1] == "pruned" and trials["r1_t02"][2] is None
        assert conn.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 0
    finally:
        conn.close()
    for p in R.iter_record_files(spool):
        for _i, rec, _sha in R.read_record_file(p, compute_sha=False):
            assert isinstance(VAL.validate_record(rec, p), VAL.Validated), (p, VAL.validate_record(rec, p))
            assert rec["producer"]["script"] == "scripts/tune_arm.py"


def test_hpo_records_carry_the_central_stamps(tmp_path, monkeypatch):
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    f = tmp_path / "o.env"
    f.write_text("OVERRIDE_ID=synthetic_pipeline_test_v1\nOVERRIDE_LORA_ARCH_SEARCH=1\n")
    monkeypatch.setenv(emit.RUN_OVERRIDES_ENV, str(f))
    spool = str(tmp_path / "spool")
    sid = emit_hpo.open_search(campaign="c", model_id="m", object_kind="relu_steer", method="sobol", search_space={"lr": {"kind": "log_uniform", "low": 1e-5, "high": 1e-3}},
                               n_trials=1, substrate="grading_host", code_sha="a" * 40, spool=spool)
    emit_hpo.record_trial(sid, "r1_t00", {"lr": 1e-4}, status="complete", objective_value=1.0, campaign="c", spool=spool, substrate="grading_host")
    seen = []
    for p in R.iter_record_files(spool):
        for _i, rec, _sha in R.read_record_file(p, compute_sha=False):
            seen.append(rec["record_type"])
            assert rec["payload"]["extra"]["run_overrides"]["id"] == "synthetic_pipeline_test_v1", rec
            assert rec["payload"]["extra"]["run_overrides"]["keys"]["OVERRIDE_LORA_ARCH_SEARCH"] == "1"
    assert sorted(seen) == ["fit_search", "fit_trial"]


def test_emission_off_writes_nothing_and_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv(emit.DISABLED_ENV, "1")
    spool = str(tmp_path / "spool")
    sid = emit_hpo.open_search(campaign="c", model_id="m", object_kind="lora", method="grid", search_space={}, n_trials=1, substrate="grading_host",
                               code_sha="a" * 40, spool=spool)
    assert sid.startswith("hpo-native:") and not os.path.exists(spool)
    assert emit_hpo.record_trial(sid, "t", {}, status="pruned", campaign="c", spool=spool, substrate="grading_host") is None


def _slow_clock(base="2026-01-01T00:00:00.000Z", reads_per_ms=3):
    """A UTC-ms clock that advances one millisecond every ``reads_per_ms`` reads: fast enough that a writer waiting for the next
    millisecond returns, slow enough that two writers reading it once each see the same instant."""
    from datetime import datetime, timedelta, timezone
    t0 = datetime.strptime(base, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
    reads = itertools.count()

    def now() -> str:
        return (t0 + timedelta(milliseconds=next(reads) // reads_per_ms)).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    return now


def test_a_searchs_records_are_stamped_strictly_increasing_across_its_writers_so_the_closing_record_folds_last(tmp_path, monkeypatch):
    """The flake's mechanism, pinned: every open / progress / close call builds its own writer, a writer's stamp is monotone only within
    itself, and the fold keeps a search's LATEST-stamped fit_search record -- so two records stamped in one millisecond folded in sha
    order and the closing record sometimes lost. Under a clock that ties two fresh writers, the search's records still carry strictly
    increasing stamps (the per-search watermark each writer starts above), the folded row is the close, and the mechanism itself is shown
    on two bare writers under the same clock (they tie)."""
    monkeypatch.delenv(emit.DISABLED_ENV, raising=False)
    monkeypatch.setattr(emit, "utc_now", _slow_clock())
    # the mechanism: two writers of one key built back to back stamp the same millisecond
    a = emit.RecordWriter(str(tmp_path / "bare"), "c", "k", emit.producer_block("t.py", substrate="grading_host"))
    b = emit.RecordWriter(str(tmp_path / "bare"), "c", "k", emit.producer_block("t.py", substrate="grading_host"))
    a.write("si_text", {"si_id": "x", "si_text_sha256": "1" * 64, "si_kind": "constitution_short", "source_uri": "synthetic://x"})
    b.write("si_text", {"si_id": "y", "si_text_sha256": "2" * 64, "si_kind": "constitution_short", "source_uri": "synthetic://y"})
    assert a.last_written_utc == b.last_written_utc, (a.last_written_utc, b.last_written_utc)
    # the search: open -> progress -> three trials -> close through fresh writers under the same clock
    root = _root(tmp_path)
    spool = os.path.join(root, "records", "hpo")
    kw = dict(campaign=synth.CAMPAIGN, spool=spool, code_sha="a" * 40, substrate="cluster_a")
    sid = emit_hpo.open_search(model_id=synth.FIT_MODEL_ID, object_kind="lora", method="sobol", search_space=SPACE, n_trials=1, started_utc=synth.t(0), **kw)
    opening = next(rec["payload"] for p in R.iter_record_files(spool) for _i, rec, _s in R.read_record_file(p, compute_sha=False) if rec["record_type"] == "fit_search")
    opening = emit_hpo.update_search(sid, opening=opening, n_trials_completed=0, trials_in_flight=["r1_t00"], **kw)
    emit_hpo.record_trial(sid, "r1_t00", {"lr": 1e-4}, status="complete", objective_value=2.0, **kw)
    emit_hpo.close_search(sid, opening=opening, winner_params={"lr": 1e-4}, winner_trial_id="r1_t00", winner_objective=2.0, n_trials_completed=1, **kw)
    stamps = [rec["written_utc"] for p in sorted(R.iter_record_files(spool)) for _i, rec, _s in R.read_record_file(p, compute_sha=False)]
    assert len(stamps) == 4 and stamps == sorted(stamps) and len(set(stamps)) == 4, stamps
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0, rep.refusals
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        row = conn.execute("SELECT n_trials_completed, winner_trial_id, json_extract(extra_json, '$.phase') FROM fit_search WHERE search_id = ?", (sid,)).fetchone()
        assert tuple(row) == (1, "r1_t00", "closed"), tuple(row)
    finally:
        conn.close()
