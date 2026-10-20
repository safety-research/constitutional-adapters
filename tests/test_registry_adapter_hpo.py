"""hpo_logs: a synthetic reseeded sweep → `fit_search` + `fit_trial` rows (`test_reseed_search_records`): the
quadratic fit's interiority and positive-definiteness, the confirmation verdict, the folded winner (fitted argmin
vs raw-best-adopted after an alarm), the clip reconnaissance; the object ↔ search link; the tune_arm layout with
and without substrate evidence (no evidence files `substrate = unverified` and ingests, never a refusal); a sweep
of a kind the scope table rules out of scope (token_faithful / bidpo) PARKED by name -- an accepted empty read with the structured
`parked` fact the status page lists -- while an unruled spelling (one that merely begins with a parked spelling included)
still refuses by name; the kind decided off the records' own claims (a claimless first summary never hands the decision to the
directory's name); validation + ingest.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import status
from sisteer.registry import synth
from sisteer.registry import validate as VAL
from sisteer.registry import vocab as V
from sisteer.registry.adapters import fit_sidecars as FS
from sisteer.registry.adapters import hpo_logs as HPO
from sisteer.registry.adapters import objects_bank as OB
from sisteer.registry.adapters import runner as RUN
from registry_fit_fixtures import OUT_OF_SCOPE_TABLE, _write, seeds

T = synth.t
KEY = "synthmodel_4b_base_eb64_s1"


def _trial(idx, lr, warm, clip, loss, rnd=1, status="ok", wandb="wb%02d" % 0, seed=1, extra=None):
    d = {"trial": {"index": idx, "lr": lr, "warmup_frac": warm, "max_grad_norm": clip}, "round": rnd, "status": status, "arm": "psr_ce_all",
         "model": "synthetic/base-4b", "model_revision": "f" * 40, "si": "synth-corpus-10m", "position": "all", "box": {"lr": [5e-4, 5e-3], "max_grad_norm": [clip, clip]},
         "warmup_levels": [0.05, 0.15], "epochs": 8, "early_stop_patience": 2, "seed": seed, "best_heldout_loss": loss, "best_epoch": 8,
         "heldout_curve": [2.3] + [loss + 0.01 * (8 - i) for i in range(8)], "stopped_early": False, "wandb_run": wandb, "fit_fingerprint": "%064x" % idx,
         "steps_run": 520, "total_steps": 520, "wall_s": 8000.0, "writer": "tune_arm.py", "batch_size": 2, "accum_steps": 32, "effective_batch": 64, "max_len": 8192}
    d.update(extra or {})
    return d


def _reseed(tmp_path, alarm=False):
    """`hpo/results/<key>/psr_ce_all/{r1_t00..r1_t03, r9_t00, r1_fit.json, confirm_check.json}`, `hpo/trials`, `hpo/manifests`,
    `hpo/hpo_fold.json`, `hpo/search_result.json`, `recon/recon_clip_*.json`."""
    base = str(tmp_path / "reseed_eb64")
    hpo = os.path.join(base, "hpo")
    sweep = os.path.join(hpo, "results", KEY, "psr_ce_all")
    lrs = [5.5e-4, 5.2e-3, 1.7e-3, 2.9e-3]
    losses = [2.0957, 2.0694, 2.0750, 2.0710]
    rows = []
    for i, (lr, loss) in enumerate(zip(lrs, losses)):
        _write(os.path.join(sweep, f"r1_t0{i}"), "summary.json", _trial(i, lr, 0.05 if i % 2 == 0 else 0.15, 0.34, loss, wandb=f"wb0{i}"))
        _write(os.path.join(sweep, f"r1_t0{i}"), "metrics.jsonl", '{"step": 0}\n')
        rows.append({"index": i, "lr": lr, "warmup_frac": 0.05 if i % 2 == 0 else 0.15, "max_grad_norm": 0.34, "best_heldout_loss": loss, "round": 1,
                     "clip_activation_rate": 0.02 + 0.01 * i, "clip_n_steps": 520})
    winner_lr = 4.578836147882968e-3
    sel = {"lr": winner_lr, "warmup_frac": 0.15, "max_grad_norm": 0.34, "fitted_min_log_y": 0.7273, "on_edge": False, "edge_pos": {"lr": 0.93, "max_grad_norm": None},
           "clamped_axes": [], "selected_from": "fitted_argmin", "alarmed": alarm, "alarms": (["sampled_extreme:lr"] if alarm else [])}
    _write(sweep, "r1_fit.json", {"fit": {"names": ["1", "lr", "lr*lr", "warmup", "warmup*lr"], "n_rows": 4, "design_rank": 5, "design_cond": 648.1, "levels": [0.05, 0.15],
                                          "cont_axes": ["lr"], "fixed_axes": {"max_grad_norm": 0.34}, "coef": [0.81, 0.03, 0.003, -0.002, -0.0003], "hessian_eigs": [0.0057],
                                          "positive_definite": True, "sigma_log_y": 0.00019, "argmin": {"lr": winner_lr, "warmup_frac": 0.15, "max_grad_norm": 0.34}},
                                  "rows": rows, "box": {"lr": [5e-4, 5e-3], "max_grad_norm": [0.34, 0.34]}, "rounds": [1], "y_key": "best_heldout_loss",
                                  "arm": "psr_ce_all", "model": "synthetic/base-4b", "position": "all", "si": "synth-corpus-10m", "selection": sel,
                                  "selection_guard": {"selected_from": "raw_best_trial" if alarm else "fitted_argmin", "selected_trial": "r1_t01" if alarm else None,
                                                      "alarms": ([{"code": "sampled_extreme", "dim": "lr"}] if alarm else [])},
                                  "winner": {"arm": "psr_ce_all", "model": "synthetic/base-4b", "position": "all", "si": "synth-corpus-10m", "lr": winner_lr,
                                             "max_grad_norm": 0.34, "warmup_frac": 0.15, "lr_schedule": "warmup_cosine", "selected_from": "fitted_argmin"}})
    _write(sweep, "confirm_check.json", {"arm": "psr_ce_all", "pass": True, "selection_action": "adopt", "confirmation_verdict": "in_interval", "fit_reliability": "ok",
                                         "criterion_3_met": True, "d_over_sigma": 0.52, "seed": 1})
    _write(os.path.join(sweep, "r9_t00"), "summary.json", _trial(0, winner_lr, 0.15, 0.34, 2.0698, rnd=9, wandb="wb90",
                                                                 extra={"confirmation_of": {"fit": os.path.join(sweep, "r1_fit.json"), "argmin": sel}}))
    fold_lr = 6.681234631156978e-3 if alarm else winner_lr
    _write(hpo, "hpo_fold.json", {KEY: {"psr_ce_all": {"lr": fold_lr, "max_grad_norm": 0.34, "warmup_frac": 0.15,
                                                       "selection_guard": {"selected_from": "raw_best_trial" if alarm else "fitted_argmin",
                                                                           "selected_trial": "r1_t01" if alarm else None,
                                                                           "alarms": (["sampled_extreme"] if alarm else []), "fall_through": None}}}})
    _write(hpo, "search_result.json", {"sha": synth.code_sha("3"), "app": "sdfbase-reseed-hpo", "seed": 1})
    _write(os.path.join(hpo, "trials"), f"r1_{KEY}_psr_ce_all_all.json", {"round": 1, "box": {"lr": [5e-4, 5e-3], "max_grad_norm": [0.34, 0.34]},
                                                                            "warmup_levels": [0.05, 0.15], "sobol_m": 2, "sobol_d": 1, "axes": ["lr"], "arm": "psr_ce_all",
                                                                            "model": "synthetic/base-4b", "config_sha": "b3277c2995c65d64", "position": "all",
                                                                            "trials": [{"index": i, "lr": lr} for i, lr in enumerate(lrs)],
                                                                            "constant_axes": {"max_grad_norm": 0.34}, "si": "synth-corpus-10m"})
    _write(os.path.join(hpo, "manifests"), f"hpo_calls_r1_{KEY}_psr_ce_all.json", {"app": "sdfbase-reseed-hpo", "round": 1, "written_at": "2026-09-16T06:52:28+00:00",
                                                                                     "trials": [{"index": i, "call_id": f"fc-SYNTH{i:02d}"} for i in range(4)]})
    _write(os.path.join(base, "recon"), f"recon_clip_{KEY}_psr_ce_all.json", {"arm": "psr_ce_all", "clip": 0.34, "clip_raw": 0.331, "rule": "ceil2sf(p95 …)", "percentile": 0.95})
    return base, hpo, sweep, winner_lr, fold_lr


def _seeds():
    s = seeds()
    s.corpus_ids["synth-corpus-10m"] = "synth-corpus-10m"
    return s


def test_reseed_search_records(tmp_path):
    base, hpo, sweep, winner_lr, _ = _reseed(tmp_path)
    sweeps = HPO.enumerate_sweeps([("reseed_eb64", hpo, "reseed")])
    assert len(sweeps) == 1 and sweeps[0].key == KEY and sweeps[0].substrate_evidence[0] == "modal"
    s, trials = HPO.build_search(sweeps[0], _seeds(), T(9000))
    assert s["campaign"] == "reseed" and s["method"] == "sobol" and s["n_trials"] == 5 and s["n_trials_completed"] == 5
    assert s["target_object_class_json"]["object_kind"] == "relu_steer" and s["target_object_class_json"]["model_id"] == synth.FIT_MODEL_ID
    assert s["search_space_json"]["lr"] == {"kind": "log_uniform", "low": 5e-4, "high": 5e-3}
    assert s["search_space_json"]["warmup_frac"] == {"kind": "choice", "values": [0.05, 0.15]}
    assert s["search_space_json"]["max_grad_norm"] == {"kind": "choice", "values": [0.34]}
    assert s["winner_params_json"] == {"lr": winner_lr, "max_grad_norm": 0.34, "warmup_frac": 0.15}
    assert "winner_trial_id" not in s and s["field_confidence_json"]["winner_trial_id"]["confidence"] == "unverified"
    assert s["argmin_interior"] == 1 and s["quadratic_fit_positive_definite"] == 1 and s["noise_estimate"] == 0.00019
    assert s["quadratic_fit_params_json"]["hessian_eigs"] == [0.0057]
    assert s["confirmation_trial_id"] == "r9_t00" and s["confirmation_within_noise"] == 1
    assert s["grad_norm_reconnaissance_uri"].endswith(f"recon_clip_{KEY}_psr_ce_all.json") and s["extra"]["recon_clip"]["clip"] == 0.34
    assert s["substrate"] == "modal" and s["code_sha"] == synth.code_sha("3") and s["job_ids_json"] == [f"fc-SYNTH{i:02d}" for i in range(4)]
    assert s["started_utc"] == "2026-09-16T06:52:28.000Z" and s["winner_objective"] == 2.0694 and s["extra"]["best_raw_trial"] == "r1_t01"
    assert s["recentre_rounds"] == 0 and s["objective"] == "best_heldout_loss" and s["objective_direction"] == "minimize"
    assert len(trials) == 5
    t1 = next(t for t in trials if t["trial_id"] == "r1_t01")
    assert t1["params_json"]["lr"] == 5.2e-3 and t1["objective_value"] == 2.0694 and t1["status"] == "complete"
    assert t1["clip_fraction"] == pytest.approx(0.03) and t1["epochs_seen"] == 8 and t1["wandb_run_id"] == "wb01"
    assert t1["extra"]["field_confidence"]["clip_fraction"]["source"] == "r1_fit.json:rows[].clip_activation_rate"
    t9 = next(t for t in trials if t["trial_id"] == "r9_t00")
    assert t9["extra"]["field_confidence"]["clip_fraction"]["confidence"] == "unverified"
    # every record validates
    for rec in [synth.rec("fit_search", s, T(9001))] + [synth.rec("fit_trial", t, T(9002)) for t in trials]:
        v = VAL.validate_record(rec, "<memory>")
        assert isinstance(v, VAL.Validated), v


def test_alarmed_lane_adopts_the_raw_best_trial(tmp_path):
    base, hpo, sweep, winner_lr, fold_lr = _reseed(tmp_path, alarm=True)
    sd = HPO.enumerate_sweeps([("reseed_eb64", hpo, "reseed")])[0]
    s, _ = HPO.build_search(sd, _seeds(), T(9000))
    assert s["winner_params_json"]["lr"] == fold_lr and s["winner_trial_id"] == "r1_t01"
    assert s["argmin_interior"] == 1                       # the fitted argmin was interior; the ALARM (not the edge) moved the selection
    assert s["extra"]["selection_guard"] == {"selected_from": "raw_best_trial", "alarmed": None, "alarms": ["sampled_extreme"], "fall_through": None} or \
        s["extra"]["selection_guard"]["selected_from"] == "raw_best_trial"
    assert s["extra"]["selection_alarms"] == ["sampled_extreme:lr"] and s["extra"]["selection_action"] == "adopt"


def test_link_objects_by_winner_lr_and_by_unique_class(tmp_path):
    base, hpo, sweep, winner_lr, _ = _reseed(tmp_path)
    s, _ = HPO.build_search(HPO.enumerate_sweeps([("reseed_eb64", hpo, "reseed")])[0], _seeds(), T(9000))
    exact = {"fit_target_kind": "corpus_document_fit", "fit_model_id": synth.FIT_MODEL_ID, "object_kind": "relu_steer", "fit_corpus_id": "synth-corpus-10m",
             "fit_positions": "all", "fit_lr": winner_lr, "fit_lr_selected_by": "unverified", "field_confidence_json": {}}
    import copy
    other_lr = copy.deepcopy(dict(exact, fit_lr=1e-3))
    no_lr = copy.deepcopy({k: v for k, v in exact.items() if k != "fit_lr"})
    wrong_kind = copy.deepcopy(dict(exact, object_kind="lora"))
    n = HPO.link_objects([exact, other_lr, no_lr, wrong_kind], [s])
    assert n == 1
    assert exact["fit_search_id"] == s["search_id"] and exact["fit_lr_selected_by"] == "search" and exact["fit_argmin_interior"] == 1
    assert exact["field_confidence_json"]["fit_search_id"]["confidence"] == "recorded"
    assert "fit_search_id" not in other_lr and "fit_search_id" not in wrong_kind
    assert "fit_search_id" not in no_lr and "fit_lr" not in no_lr          # no lr of its own: never linked by class alone
    n2 = HPO.link_objects([copy.deepcopy(no_lr)], [s], allow_class_link=True)
    assert n2 == 1
    # a search whose fit record names NO adopted winner (argmin on the box edge) links nothing
    edge = copy.deepcopy(s)
    edge.pop("winner_params_json")
    fresh = copy.deepcopy(exact)
    fresh.pop("fit_search_id")
    assert HPO.link_objects([fresh], [edge]) == 0


def _flat_sweep_without_substrate_evidence(root: str) -> str:
    sweep = os.path.join(root, "psr_kl_all_response")
    for i in range(2):
        _write(os.path.join(sweep, f"r1_t0{i}"), "summary.json", _trial(i, 1e-3 * (i + 1), 0.05, 0.2, 0.06 + i / 100, seed=None,
                                                                          extra={"si": "synthetic_si", "position": "response", "arm": "psr_kl_all", "model": "synthetic/instruct-4b"}))
    return sweep


def test_tune_arm_sweep_without_substrate_evidence_files_unverified(tmp_path):
    root = str(tmp_path / "hpo-x")
    sweep = _flat_sweep_without_substrate_evidence(root)
    sd = HPO.enumerate_sweeps([("hpo-x", root, "flat")])[0]
    assert sd.substrate_evidence is None
    # no substrate record is not a refusal — the row files `unverified` naming the evidence that is missing, and validates
    s0, _ = HPO.build_search(sd, _seeds(), T(9000), mtime_utc=T(100))
    assert s0["substrate"] == "unverified"
    sub = s0["field_confidence_json"]["substrate"]
    assert sub["confidence"] == "unverified" and "PODS.md" in sub["reason"] and "profiler trace" in sub["reason"]
    assert isinstance(VAL.validate_record(synth.rec("fit_search", s0, T(1)), "<m>"), VAL.Validated)
    # the value is admitted for fit_search ALONE: the run / object substrate vocabularies do not widen
    assert "unverified" in V.FIT_SEARCH_SUBSTRATES and "unverified" not in V.SUBSTRATES
    # a Modal profiler trace beside a trial is the worker's own stamp
    _write(os.path.join(sweep, "r1_t00", "tb"), "modal_5.178.pt.trace.json", "{}")
    sd = HPO.enumerate_sweeps([("hpo-x", root, "flat")])[0]
    assert sd.substrate_evidence[0] == "modal"
    s, trials = HPO.build_search(sd, _seeds(), T(9000), mtime_utc=T(100))
    assert s["substrate"] == "modal" and s["method"] == "sobol" and s["field_confidence_json"]["method"]["confidence"] == "derived"
    assert s["started_utc"] == T(100) and s["field_confidence_json"]["started_utc"]["confidence"] == "derived"
    assert "winner_params_json" not in s and s["field_confidence_json"]["winner_params_json"]["confidence"] == "unverified"
    # a fit record whose selection sits on the box edge with winner null is NOT a winner
    _write(sweep, "r1_fit.json", {"fit": {"positive_definite": True}, "rows": [], "box": {"lr": [1e-4, 1e-2]}, "rounds": [1], "y_key": "best_heldout_loss",
                                  "arm": "psr_kl_all", "model": "synthetic/instruct-4b", "position": "response", "si": "synthetic_si",
                                  "selection": {"lr": 7.5e-4, "on_edge": True, "selected_from": "fitted_argmin"}, "winner": None})
    s2, _ = HPO.build_search(sd, _seeds(), T(9000), mtime_utc=T(100))
    assert "winner_params_json" not in s2 and "winner is null" in s2["field_confidence_json"]["winner_params_json"]["reason"]
    assert s2["argmin_interior"] == 0 and s2["extra"]["fitted_argmin"]["lr"] == 7.5e-4
    assert len(trials) == 2 and all(isinstance(VAL.validate_record(synth.rec("fit_trial", t, T(1)), "<m>"), VAL.Validated) for t in trials)
    assert isinstance(VAL.validate_record(synth.rec("fit_search", s, T(1)), "<m>"), VAL.Validated)
    # a PODS.md beside the root is RunPod evidence
    _write(root, "PODS.md", "# pods\n")
    assert HPO.enumerate_sweeps([("hpo-x", root, "flat")])[0].substrate_evidence[0] == "runpod"


def test_flat_layout_winner_trial_comes_from_the_fit_records_selection_guard(tmp_path):
    """The tune_arm layouts have no hpo_fold.json: when the fit record's own selection_guard adopted a raw trial after an
    alarm, that trial is the winner_trial_id (recorded) — not an `unverified` fitted-argmin note written without looking."""
    root = str(tmp_path / "hpo-x")
    sweep = _flat_sweep_without_substrate_evidence(root)
    _write(root, "PODS.md", "# pods\n")
    fit = {"fit": {"positive_definite": True}, "rows": [], "box": {"lr": [1e-4, 1e-2]}, "rounds": [1], "y_key": "best_heldout_loss",
           "arm": "psr_kl_all", "model": "synthetic/instruct-4b", "position": "response", "si": "synthetic_si",
           "selection": {"lr": 1.5e-3, "on_edge": False, "selected_from": "raw_best_trial", "alarmed": True, "alarms": ["sampled_extreme:lr"]},
           "selection_guard": {"selected_from": "raw_best_trial", "selected_trial": "r1_t01", "alarms": [{"code": "sampled_extreme", "dim": "lr"}]},
           "winner": {"lr": 2e-3, "max_grad_norm": 0.2, "warmup_frac": 0.05, "selected_from": "raw_best_trial"}}
    _write(sweep, "r1_fit.json", fit)
    s, _ = HPO.build_search(HPO.enumerate_sweeps([("hpo-x", root, "flat")])[0], _seeds(), T(9000), mtime_utc=T(100))
    assert s["winner_trial_id"] == "r1_t01" and s["field_confidence_json"]["winner_trial_id"]["confidence"] == "recorded"
    assert s["extra"]["selection_guard"]["selected_from"] == "raw_best_trial" and s["winner_params_json"]["lr"] == 2e-3
    assert isinstance(VAL.validate_record(synth.rec("fit_search", s, T(1)), "<m>"), VAL.Validated)
    # a fitted-argmin winner (selected_trial null) is a point, not a trial — and the reason says which case it is
    fit["selection_guard"] = {"selected_from": "fitted_argmin", "selected_trial": None, "alarms": []}
    _write(sweep, "r1_fit.json", fit)
    s2, _ = HPO.build_search(HPO.enumerate_sweeps([("hpo-x", root, "flat")])[0], _seeds(), T(9000), mtime_utc=T(100))
    assert "winner_trial_id" not in s2 and "selected_trial is null" in s2["field_confidence_json"]["winner_trial_id"]["reason"]
    del fit["selection_guard"]
    _write(sweep, "r1_fit.json", fit)
    s3, _ = HPO.build_search(HPO.enumerate_sweeps([("hpo-x", root, "flat")])[0], _seeds(), T(9000), mtime_utc=T(100))
    assert "winner_trial_id" not in s3 and "no selection_guard" in s3["field_confidence_json"]["winner_trial_id"]["reason"] and "selection_guard" not in s3["extra"]


def _flat_sweep_of_spelling(root: str, spelling: str, n: int = 1) -> str:
    """`<root>/<spelling>_all/r1_t0<i>/summary.json`, each trial summary spelling its intervention `spelling`."""
    sweep = os.path.join(root, f"{spelling}_all")
    for i in range(n):
        _write(os.path.join(sweep, f"r1_t0{i}"), "summary.json", _trial(i, 1e-3 * (i + 1), 0.05, 0.2, 0.06 + i / 100,
                                                                          extra={"arm": spelling, "model": "synthetic/instruct-4b"}))
    return sweep


def _registry(tmp_path) -> str:
    reg = str(tmp_path / "reg")
    H.ensure_layout(reg)
    R.write_records(os.path.join(reg, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    H.ingest_once(reg, workers=0, min_free=0)
    return reg


def _parked_sources(conn):
    """The status page's "Parked by name" rows, through the renderer's own query (status.QUERIES['parked_sources'])."""
    cur = conn.execute(status.QUERIES["parked_sources"])
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def test_out_of_scope_sweep_is_parked_by_name_not_refused(tmp_path):
    """Park-by-name: a sweep directory whose intervention is a kind ruled out of scope (`token_faithful`) is
    PARKED -- `build_search` raises :class:`HpoParked` (an :class:`HpoRefusal`, so a caller that refuses on the base class still does),
    `scan` counts it apart from refusals, the pass ACCEPTS the group with no `fit_search` / `fit_trial` record and files its `source_status`
    row `ok = 0` with the ruling under `extra.refuse` (the explanation) and `extra.parked` (the entry the status page lists, in the objects
    bank's shape) -- while the in-scope sweep beside it reads as before, nothing is refused, and an unchanged second pass writes nothing."""
    root = str(tmp_path / "hpo-x")
    in_scope = _flat_sweep_without_substrate_evidence(root)
    parked_dir = _flat_sweep_of_spelling(root, "token_faithful")
    _write(root, "PODS.md", "# pods\n")
    sweeps = {sd.path: sd for sd in HPO.enumerate_sweeps([("hpo-x", root, "flat")])}
    assert set(sweeps) == {in_scope, parked_dir}
    with pytest.raises(HPO.HpoParked) as ei:
        HPO.build_search(sweeps[parked_dir], _seeds(), T(9000))
    exc = ei.value
    assert isinstance(exc, HPO.HpoRefusal) and exc.where == parked_dir and exc.kind_legacy == "token_faithful" and exc.prefix == "token_faithful"
    assert exc.entry["status"] == "withdrawn" and "out of scope" in exc.why and "D37" in exc.why and "parked by name" in exc.why
    facts = HPO.parked_status_facts(exc)
    assert set(facts) == {"refuse", "parked"} and len(facts["refuse"]) == 1 and len(facts["parked"]) == 1
    ent = facts["parked"][0]
    # the status page's SQL reads these keys off every parked entry (status.QUERIES['parked_sources']), and the objects bank's entry
    # (the shape the page was written for) has nothing this one lacks
    assert {"dir", "kind_legacy", "status_ruled", "object_role_ruled", "reason", "ruling"} <= set(ent)
    bank_entry = OB.parked_status_facts(FS.OutOfScopeObject("/synthetic/d", "why", "lofit", OUT_OF_SCOPE_TABLE[0]))["parked"][0]
    assert set(bank_entry) <= set(ent)
    assert ent["dir"] == parked_dir and ent["kind_legacy"] == "token_faithful" and ent["kind_legacy_source"] == "r1_t00/summary.json:arm"
    assert ent["legacy"] == "token_faithful | token_faithful_all | token_appended" and ent["status_ruled"] == "withdrawn" and ent["object_role_ruled"] == "other"
    assert ent["reason"].startswith("synthetic:") and ent["ruling"] == HPO.PARKED_RULING == OB.PARKED_RULING and ent["ruling_source"] == "decisions.md D37"
    assert ent["withheld"] == ["fit_search", "fit_trial"] and ent["n_trial_summaries"] == 1 and ent["root_label"] == "hpo-x"
    assert ent["table_row_read"] is True and ent["seeds_source"] == "test"
    assert "parked by name" in facts["refuse"][0] and "D37" in facts["refuse"][0] and parked_dir in facts["refuse"][0]
    # scan: counted apart from refusals
    hs = HPO.scan([("hpo-x", root, "flat")], seeds=_seeds())
    assert hs.by_root == {"hpo-x": {"sweeps": 2, "searches": 1, "trials": 2, "refused": 0, "parked": 1}} and hs.refusals == []
    assert len(hs.parked) == 1 and hs.parked[0]["dir"] == parked_dir and hs.parked[0]["kind_legacy"] == "token_faithful" and hs.parked[0]["parked"] == ent
    # the pass: both groups accepted; the in-scope one's rows filed, the parked one's status row explained
    reg = _registry(tmp_path)
    ad = HPO.HpoLogsAdapter([("hpo-x", root, "flat")], seeds=_seeds())
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed == 2, [r.error for r in rep.results]
    by_group = {r.group: r for r in rep.results}
    assert by_group[in_scope].accepted and by_group[parked_dir].accepted
    assert ad.status_extra(in_scope) == {} and ad.status_extra(parked_dir) == facts
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM fit_search").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM fit_trial").fetchone()[0] == 2
        assert conn.execute("SELECT trials_uri FROM fit_search").fetchone()[0] == in_scope
        rows = {r[0]: r for r in conn.execute("SELECT source_id, ok, detail, extra_json FROM source_status WHERE adapter = 'hpo_logs'")}
        assert len(rows) == 2
        parked_rows = {sid: json.loads(r[3]) for sid, r in rows.items() if json.loads(r[3]).get("parked")}
        assert len(parked_rows) == 1
        (sid, extra), = parked_rows.items()
        assert rows[sid][1] == 0 and all(rows[o][1] == 1 for o in set(rows) - {sid})
        assert extra["parked"] == [ent] and extra["refuse"] == facts["refuse"] and extra.get("emitted_nothing_unexplained") is None
        assert "parked by name" in rows[sid][2] and "EMITTED NOTHING" not in rows[sid][2]      # explained, never a silent read
        assert conn.execute("SELECT COUNT(*) FROM v_source_status_current WHERE adapter = 'hpo_logs' AND ok = 0 "
                            "AND json_extract(extra_json, '$.parked') IS NOT NULL").fetchone()[0] == 1
        # the parked group's revision is ACCEPTED: no standing refusal for a window or tick to re-read
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'hpo_logs' AND accepted = 0").fetchone()[0] == 0
        # the status page's "Parked by name" section lists the sweep off the same facts, under the columns the template renders
        listed = [r for r in _parked_sources(conn) if r["adapter"] == "hpo_logs"]
    finally:
        conn.close()
    assert [(r["kind_legacy"], r["dir"], r["status_ruled"], r["object_role_ruled"]) for r in listed] == [("token_faithful", parked_dir, "withdrawn", "other")]
    assert listed[0]["ruling"] == HPO.PARKED_RULING and listed[0]["reason"].startswith("synthetic:")
    # an unchanged second pass writes nothing: the park is an accepted head, not a refusal to retry
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep2 = RUN.run_pass(reg, [ad], conn=conn, now=T(9500))
    conn.close()
    assert rep2.inputs_changed == 0 and rep2.records_written == 0 and rep2.groups_refused == 0


def test_an_unmapped_intervention_spelling_still_refuses_by_name(tmp_path):
    """Parking is for the declared out-of-scope set only (`token_faithful` / `bidpo`: the `None` entries of hpo_logs's kind table), never
    inferred: a sweep whose intervention spelling no prefix maps -- a kind nobody has ruled on -- still REFUSES by name, unchanged. `build_search`
    raises a plain :class:`HpoRefusal` naming the spelling, `scan` lists it under refusals (not parked), the pass refuses the group and files
    no `parked` fact for it, while the `bidpo` sweep beside it parks and the in-scope one reads."""
    root = str(tmp_path / "hpo-x")
    in_scope = _flat_sweep_without_substrate_evidence(root)
    parked_dir = _flat_sweep_of_spelling(root, "bidpo")
    novel = _flat_sweep_of_spelling(root, "novel_method")
    _write(root, "PODS.md", "# pods\n")
    sweeps = {sd.path: sd for sd in HPO.enumerate_sweeps([("hpo-x", root, "flat")])}
    with pytest.raises(HPO.HpoRefusal) as ei:
        HPO.build_search(sweeps[novel], _seeds(), T(9000))
    assert not isinstance(ei.value, HPO.HpoParked) and ei.value.where == novel
    assert "'novel_method'" in ei.value.why and "maps to no object_kind" in ei.value.why
    with pytest.raises(HPO.HpoParked) as pi:
        HPO.build_search(sweeps[parked_dir], _seeds(), T(9000))
    assert pi.value.kind_legacy == "bidpo" and pi.value.entry["legacy"] == "bidpo | did | cast | circuit_breaker"
    hs = HPO.scan([("hpo-x", root, "flat")], seeds=_seeds())
    assert hs.by_root == {"hpo-x": {"sweeps": 3, "searches": 1, "trials": 2, "refused": 1, "parked": 1}}
    assert [r["dir"] for r in hs.refusals] == [novel] and "novel_method" in hs.refusals[0]["why"] and [p["dir"] for p in hs.parked] == [parked_dir]
    reg = _registry(tmp_path)
    ad = HPO.HpoLogsAdapter([("hpo-x", root, "flat")], seeds=_seeds())
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 1 and rep.groups_refreshed == 2, [r.error for r in rep.results]      # refreshed counts the accepted reads
    by_group = {r.group: r for r in rep.results}
    assert not by_group[novel].accepted and "'novel_method'" in by_group[novel].error and "maps to no object_kind" in by_group[novel].error
    assert by_group[in_scope].accepted and by_group[parked_dir].accepted
    assert ad.status_extra(novel) == {} and set(ad.status_extra(parked_dir)) == {"refuse", "parked"}
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM fit_search").fetchone()[0] == 1
        # the unruled sweep's revision is a REFUSAL (no accepted empty revision, no parked fact); the bidpo one is the only park
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE source_id = ? AND accepted = 1", (RUN.source_id_for(ad.name, novel),)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'hpo_logs' AND accepted = 0").fetchone()[0] == 1
        listed = [r for r in _parked_sources(conn) if r["adapter"] == "hpo_logs"]
    finally:
        conn.close()
    assert [(r["kind_legacy"], r["dir"]) for r in listed] == [("bidpo", parked_dir)]


def test_a_declared_out_of_scope_sweep_with_an_unregistered_model_still_parks(tmp_path):
    """The park must sit ABOVE the model-closure checks. A token_faithful sweep whose trial summaries name a
    model with no `model` row (an archived sweep on a model the registry never registered), and a bidpo sweep whose summaries name no
    model at all, each park on the declared kind alone -- the scope rule needs no model to apply, and the parked path files no fit_search that
    would need the model's key -- where the first commit refused them at the model check and never reached the park, leaving the
    adapter not read whole on exactly the condition the park replaces."""
    root = str(tmp_path / "hpo-x")
    unregistered = os.path.join(root, "token_faithful_all")
    _write(os.path.join(unregistered, "r1_t00"), "summary.json", _trial(0, 1e-3, 0.05, 0.2, 0.06, extra={"arm": "token_faithful", "model": "synthetic/unregistered-9b"}))
    nameless = os.path.join(root, "bidpo_all")
    _write(os.path.join(nameless, "r1_t00"), "summary.json", _trial(0, 1e-3, 0.05, 0.2, 0.06, extra={"arm": "bidpo", "model": None}))
    _write(root, "PODS.md", "# pods\n")
    sweeps = {sd.path: sd for sd in HPO.enumerate_sweeps([("hpo-x", root, "flat")])}
    assert "synthetic/unregistered-9b" not in _seeds().model_id_by_hf_repo
    for d, spelling in ((unregistered, "token_faithful"), (nameless, "bidpo")):
        with pytest.raises(HPO.HpoParked) as ei:
            HPO.build_search(sweeps[d], _seeds(), T(9000))
        assert ei.value.kind_legacy == spelling and "model" not in ei.value.why.split("parked by name")[0].lower().replace("model_kind", "")
    hs = HPO.scan([("hpo-x", root, "flat")], seeds=_seeds())
    assert hs.by_root == {"hpo-x": {"sweeps": 2, "searches": 0, "trials": 0, "refused": 0, "parked": 2}} and hs.refusals == []
    reg = _registry(tmp_path)
    ad = HPO.HpoLogsAdapter([("hpo-x", root, "flat")], seeds=_seeds())
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed == 2, [r.error for r in rep.results]
    assert all(r.accepted for r in rep.results) and set(ad.status_extra(unregistered)) == {"refuse", "parked"} == set(ad.status_extra(nameless))
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM fit_search").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'hpo_logs' AND accepted = 0").fetchone()[0] == 0
        listed = sorted((r["kind_legacy"], r["dir"]) for r in _parked_sources(conn) if r["adapter"] == "hpo_logs")
    finally:
        conn.close()
    assert listed == [("bidpo", nameless), ("token_faithful", unregistered)]


def test_an_in_scope_sweep_with_an_unregistered_model_still_refuses_on_the_model(tmp_path):
    """The model-closure checks are unchanged for in-scope searches: a psr sweep whose summaries name a model with no `model` row refuses
    by name on the model (plain HpoRefusal, never a park), `scan` lists it under refusals, the pass refuses the group and files no
    `parked` fact."""
    root = str(tmp_path / "hpo-x")
    sweep = os.path.join(root, "psr_kl_all_response")
    _write(os.path.join(sweep, "r1_t00"), "summary.json", _trial(0, 1e-3, 0.05, 0.2, 0.06, extra={"arm": "psr_kl_all", "position": "response",
                                                                                                   "si": "synthetic_si", "model": "synthetic/unregistered-9b"}))
    _write(root, "PODS.md", "# pods\n")
    with pytest.raises(HPO.HpoRefusal) as ei:
        HPO.build_search(HPO.enumerate_sweeps([("hpo-x", root, "flat")])[0], _seeds(), T(9000))
    assert not isinstance(ei.value, HPO.HpoParked) and "has no `model` row" in ei.value.why and "synthetic/unregistered-9b" in ei.value.why
    hs = HPO.scan([("hpo-x", root, "flat")], seeds=_seeds())
    assert hs.by_root == {"hpo-x": {"sweeps": 1, "searches": 0, "trials": 0, "refused": 1, "parked": 0}} and [r["dir"] for r in hs.refusals] == [sweep]
    reg = _registry(tmp_path)
    ad = HPO.HpoLogsAdapter([("hpo-x", root, "flat")], seeds=_seeds())
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 1 and rep.groups_refreshed == 0 and "has no `model` row" in rep.results[0].error
    assert ad.status_extra(sweep) == {}


def test_a_park_without_the_table_in_the_seeds_still_parks_and_says_the_row_was_not_read(tmp_path):
    """The registered factory loads its seeds from the registry connection alone (`load_seeds(None, conn)`: the model closure, no
    `legacy_spellings.yaml`), so on the cluster the scope table row is not in hand. The park stands on hpo_logs's own kind table (its `None`
    entries ARE the spellings the yaml declares out of scope), the entry's row fields read None and `table_row_read` says so -- never a
    refusal, never an invented reason; the ruling and the spelling are recorded either way."""
    root = str(tmp_path / "hpo-x")
    parked_dir = _flat_sweep_of_spelling(root, "token_faithful", n=2)
    _write(root, "PODS.md", "# pods\n")
    bare = FS.Seeds(source="conn")
    bare.model_id_by_hf_repo = {"synthetic/instruct-4b": synth.MODEL_ID}
    assert not bare.out_of_scope_by_spelling and not bare.out_of_scope_prefixes
    with pytest.raises(HPO.HpoParked) as ei:
        HPO.build_search(HPO.enumerate_sweeps([("hpo-x", root, "flat")])[0], bare, T(9000))
    assert ei.value.entry == {}
    ent = HPO.parked_status_facts(ei.value)["parked"][0]
    assert ent["dir"] == parked_dir and ent["kind_legacy"] == "token_faithful" and ent["ruling"] == HPO.PARKED_RULING and ent["n_trial_summaries"] == 2
    assert ent["legacy"] is None and ent["status_ruled"] is None and ent["object_role_ruled"] is None and ent["reason"] is None and ent["ruling_source"] is None
    assert ent["table_row_read"] is False and ent["seeds_source"] == "conn"
    refuse = HPO.parked_status_facts(ei.value)["refuse"][0]
    assert "parked by name" in refuse and "not in the seeds" in refuse


def _sweep_of_claims(root: str, name: str, claims, fit_claim=None) -> str:
    """`<root>/<name>/r1_t0<i>/summary.json` per entry of `claims` -- the spelling the trial summary records, or None for a summary that
    records no spelling at all (several live sweeps' failed trials carry none) -- plus an `r1_fit.json` naming `fit_claim` when given."""
    sweep = os.path.join(root, name)
    for i, spelling in enumerate(claims):
        t = _trial(i, 1e-3 * (i + 1), 0.05, 0.2, 0.06 + i / 100, extra={"model": "synthetic/instruct-4b"})
        if spelling is None:
            del t["arm"]
        else:
            t["arm"] = spelling
        _write(os.path.join(sweep, f"r1_t0{i}"), "summary.json", t)
    if fit_claim is not None:
        _write(sweep, "r1_fit.json", {"fit": {"positive_definite": True}, "rows": [], "box": {"lr": [1e-4, 1e-2]}, "rounds": [1], "y_key": "best_heldout_loss",
                                      "arm": fit_claim, "model": "synthetic/instruct-4b", "position": "all", "si": "synth-corpus-10m",
                                      "selection": {"lr": 1.5e-3, "on_edge": False, "selected_from": "fitted_argmin"}, "winner": None})
    return sweep


def test_a_sweep_whose_records_disagree_on_the_kind_refuses_by_name_instead_of_parking(tmp_path):
    """Review round 1 of aafa76c51 (2026-10-08): the park was decided off ONE record -- the fit record, else the first trial summary -- so a
    sweep holding a `token_faithful` summary beside a `psr_ce_all` one PARKED whole (an accepted empty revision, the in-scope trials filed
    as nothing, the fact calling the whole sweep out of scope), and the same sweep with the summaries in the other order read as one
    relu_steer search with the out-of-scope trial among its trials. Every record that claims a spelling is now compared before the kind is
    decided: records resolving to different entries of the kind table REFUSE by name, citing each other (a plain HpoRefusal, never a park:
    `scan` counts it refused, the pass refuses the group, no accepted revision, no parked fact), whichever record carries the out-of-scope
    spelling. Spellings that differ within ONE table entry agree, so that sweep still parks; a summary with no spelling claims nothing."""
    root = str(tmp_path / "hpo-x")
    first_oos = _sweep_of_claims(root, "mixed_a_all", ["token_faithful", "psr_ce_all"])
    first_in_scope = _sweep_of_claims(root, "mixed_b_all", ["psr_ce_all", "token_faithful"])
    fit_oos = _sweep_of_claims(root, "mixed_c_all", ["psr_ce_all", "psr_ce_all"], fit_claim="token_faithful")
    one_entry = _sweep_of_claims(root, "token_faithful_all", ["token_faithful", "token_faithful_all"])
    claimless = _sweep_of_claims(root, "psr_ce_all_all", ["psr_ce_all", None])
    _write(root, "PODS.md", "# pods\n")
    sweeps = {sd.path: sd for sd in HPO.enumerate_sweeps([("hpo-x", root, "flat")])}
    assert set(sweeps) == {first_oos, first_in_scope, fit_oos, one_entry, claimless}
    expected_cites = {first_oos: ("r1_t00/summary.json", "r1_t01/summary.json"), first_in_scope: ("r1_t00/summary.json", "r1_t01/summary.json"),
                      fit_oos: ("r1_fit.json", "r1_t00/summary.json", "r1_t01/summary.json")}
    for d, cites in expected_cites.items():
        with pytest.raises(HPO.HpoRefusal) as ei:
            HPO.build_search(sweeps[d], _seeds(), T(9000))
        exc = ei.value
        assert not isinstance(exc, HPO.HpoParked) and exc.where == d, d
        assert "disagree" in exc.why and "'token_faithful'" in exc.why and "'psr_ce_all'" in exc.why, exc.why
        assert all(c in exc.why for c in cites), (d, exc.why)
        assert "OUT OF SCOPE" in exc.why and "relu_steer" in exc.why                 # each side says what its spelling resolves to
    # two spellings of ONE table entry agree: the sweep parks on the kind, the fact recording the decisive record's spelling
    with pytest.raises(HPO.HpoParked) as pi:
        HPO.build_search(sweeps[one_entry], _seeds(), T(9000))
    assert pi.value.kind_legacy == "token_faithful" and pi.value.prefix == "token_faithful" and pi.value.n_trial_summaries == 2
    # a summary that records no spelling claims nothing: the sweep beside it reads as one search over both trials
    s, trials = HPO.build_search(sweeps[claimless], _seeds(), T(9000), mtime_utc=T(100))
    assert s["target_object_class_json"]["object_kind"] == "relu_steer" and s["n_trials"] == 2 and len(trials) == 2
    hs = HPO.scan([("hpo-x", root, "flat")], seeds=_seeds())
    assert hs.by_root == {"hpo-x": {"sweeps": 5, "searches": 1, "trials": 2, "refused": 3, "parked": 1}}
    assert sorted(r["dir"] for r in hs.refusals) == sorted(expected_cites) and all("disagree" in r["why"] for r in hs.refusals)
    assert [p["dir"] for p in hs.parked] == [one_entry]
    # the pass: the three mixed sweeps REFUSED (no accepted empty revision, no parked fact), the consistent park and the search accepted
    reg = _registry(tmp_path)
    ad = HPO.HpoLogsAdapter([("hpo-x", root, "flat")], seeds=_seeds())
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 3 and rep.groups_refreshed == 2, [r.error for r in rep.results]
    by_group = {r.group: r for r in rep.results}
    for d in expected_cites:
        assert not by_group[d].accepted and "disagree" in by_group[d].error and ad.status_extra(d) == {}, d
    assert by_group[one_entry].accepted and by_group[claimless].accepted and set(ad.status_extra(one_entry)) == {"refuse", "parked"}
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        # the connection returns sqlite3.Row objects, which never compare equal to a tuple: compare the column values
        assert [r[0] for r in conn.execute("SELECT trials_uri FROM fit_search").fetchall()] == [claimless]
        assert conn.execute("SELECT COUNT(*) FROM fit_trial").fetchone()[0] == 2
        for d in expected_cites:
            assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE source_id = ? AND accepted = 1", (RUN.source_id_for(ad.name, d),)).fetchone()[0] == 0, d
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'hpo_logs' AND accepted = 0").fetchone()[0] == 3
        listed = [(r["kind_legacy"], r["dir"]) for r in _parked_sources(conn) if r["adapter"] == "hpo_logs"]
    finally:
        conn.close()
    assert listed == [("token_faithful", one_entry)]


def test_a_spelling_that_only_begins_with_a_parked_one_refuses_instead_of_parking(tmp_path):
    """The kind table matched the out-of-scope spellings by PREFIX, so a sweep whose summaries
    spell `token_faithful_v2` or `bidpo_control` -- spellings `legacy_spellings.yaml` `out_of_scope_object_kinds` never names (the scope table
    declares exact spellings; only one ending in `_` is a prefix there) -- PARKED whole: an accepted empty revision nothing retries, a fact citing
    the scope table for a kind it never ruled on, every trial of the sweep absent from fit_search / fit_trial. The out-of-scope half of the resolver is now
    a table of EXACT spellings beside the in-scope prefixes: each such sweep REFUSES by name (a plain HpoRefusal naming the spelling and the
    spellings that do park; `scan` counts it refused, the pass refuses the group, no accepted revision, no parked fact) whether its model is
    registered (the unmapped-spelling refusal) or not (the model refusal), while the declared `token_faithful`, `token_faithful_all` and
    `bidpo` sweeps beside them still park and the in-scope prefixes still resolve."""
    # the resolver itself: exact out-of-scope spellings, in-scope prefixes, nothing in between
    assert HPO._kind_entry("token_faithful") == ("token_faithful", None) and HPO._kind_entry("token_faithful_all") == ("token_faithful", None)
    assert HPO._kind_entry("bidpo") == ("bidpo", None)
    assert HPO._kind_entry("token_faithful_v2") == (None, None) == HPO._kind_entry("bidpo_control") == HPO._kind_entry("token_faithful_")
    assert HPO._kind_entry("psr_ce_all") == ("psr", "relu_steer") == HPO._kind_entry("psr_kl_all_response") and HPO._kind_entry("lora_r1_all") == ("lora", "lora")
    assert set(HPO._parked_spellings()) == {"token_faithful", "token_faithful_all", "bidpo"}
    root = str(tmp_path / "hpo-x")
    v2 = _flat_sweep_of_spelling(root, "token_faithful_v2", n=2)
    control = _flat_sweep_of_spelling(root, "bidpo_control")
    v3_unregistered = os.path.join(root, "token_faithful_v3_all")
    _write(os.path.join(v3_unregistered, "r1_t00"), "summary.json", _trial(0, 1e-3, 0.05, 0.2, 0.06, extra={"arm": "token_faithful_v3", "model": "synthetic/unregistered-9b"}))
    declared = {"token_faithful": _flat_sweep_of_spelling(root, "token_faithful"), "token_faithful_all": _flat_sweep_of_spelling(root, "token_faithful_all"),
                "bidpo": _flat_sweep_of_spelling(root, "bidpo")}
    _write(root, "PODS.md", "# pods\n")
    sweeps = {sd.path: sd for sd in HPO.enumerate_sweeps([("hpo-x", root, "flat")])}
    assert set(sweeps) == {v2, control, v3_unregistered, *declared.values()}
    for d, spelling in ((v2, "token_faithful_v2"), (control, "bidpo_control")):
        with pytest.raises(HPO.HpoRefusal) as ei:
            HPO.build_search(sweeps[d], _seeds(), T(9000))
        exc = ei.value
        assert not isinstance(exc, HPO.HpoParked) and exc.where == d, d
        # the refusal names the spelling and the exact spellings that DO park, so the reader sees the difference
        assert f"'{spelling}'" in exc.why and "maps to no object_kind" in exc.why and "'token_faithful'" in exc.why and "'bidpo'" in exc.why, exc.why
    with pytest.raises(HPO.HpoRefusal) as ei:
        HPO.build_search(sweeps[v3_unregistered], _seeds(), T(9000))
    assert not isinstance(ei.value, HPO.HpoParked) and "has no `model` row" in ei.value.why       # refuses either way, on the model here
    for spelling, d in declared.items():
        with pytest.raises(HPO.HpoParked) as pi:
            HPO.build_search(sweeps[d], _seeds(), T(9000))
        assert pi.value.kind_legacy == spelling and pi.value.prefix == ("bidpo" if spelling == "bidpo" else "token_faithful"), spelling
        assert pi.value.entry["legacy"] == ("bidpo | did | cast | circuit_breaker" if spelling == "bidpo" else "token_faithful | token_faithful_all | token_appended")
    hs = HPO.scan([("hpo-x", root, "flat")], seeds=_seeds())
    assert hs.by_root == {"hpo-x": {"sweeps": 6, "searches": 0, "trials": 0, "refused": 3, "parked": 3}}
    assert sorted(r["dir"] for r in hs.refusals) == sorted([v2, control, v3_unregistered])
    assert sorted((p["kind_legacy"], p["dir"]) for p in hs.parked) == sorted((k, d) for k, d in declared.items())
    # the pass: the three undeclared sweeps REFUSED (no accepted empty revision, no parked fact), the three declared ones parked
    reg = _registry(tmp_path)
    ad = HPO.HpoLogsAdapter([("hpo-x", root, "flat")], seeds=_seeds())
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 3 and rep.groups_refreshed == 3, [r.error for r in rep.results]
    by_group = {r.group: r for r in rep.results}
    for d in (v2, control, v3_unregistered):
        assert not by_group[d].accepted and ad.status_extra(d) == {}, d
    assert "'token_faithful_v2'" in by_group[v2].error and "'bidpo_control'" in by_group[control].error and "has no `model` row" in by_group[v3_unregistered].error
    for d in declared.values():
        assert by_group[d].accepted and set(ad.status_extra(d)) == {"refuse", "parked"}, d
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM fit_search").fetchone()[0] == 0 and conn.execute("SELECT COUNT(*) FROM fit_trial").fetchone()[0] == 0
        for d in (v2, control, v3_unregistered):
            assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE source_id = ? AND accepted = 1", (RUN.source_id_for(ad.name, d),)).fetchone()[0] == 0, d
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'hpo_logs' AND accepted = 0").fetchone()[0] == 3
        assert conn.execute("SELECT COUNT(*) FROM source_status WHERE adapter = 'hpo_logs' AND json_extract(extra_json, '$.parked') IS NOT NULL").fetchone()[0] == 3
        listed = sorted((r["kind_legacy"], r["dir"]) for r in _parked_sources(conn) if r["adapter"] == "hpo_logs")
    finally:
        conn.close()
    assert listed == sorted((k, d) for k, d in declared.items())


def test_the_parked_spellings_are_exact_entries_of_the_d37_table():
    """Every spelling hpo_logs parks by name is one the repository's own `legacy_spellings.yaml` `out_of_scope_object_kinds` names EXACTLY,
    and none is a prefix entry of it -- the adapter's table is a copy of the ruling (the registered factory's seeds carry no yaml), so this
    pins the copy to the ruling: a spelling added here that the yaml does not declare, or a yaml row the adapter has drifted from, fails."""
    seeds_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "registry")
    real = FS.load_seeds(seeds_dir)
    assert real.out_of_scope_by_spelling, seeds_dir
    for sp in HPO._parked_spellings():
        assert sp in real.out_of_scope_by_spelling and sp not in real.out_of_scope_prefixes and not sp.endswith("_"), sp
    for kind, spellings in HPO._OUT_OF_SCOPE_SPELLINGS_BY_KIND.items():
        assert kind in spellings, (kind, spellings)                                       # the entry is the kind's own spelling
        assert len({real.out_of_scope_by_spelling[sp]["legacy"] for sp in spellings}) == 1, kind      # one table row per entry
    for sp in ("token_faithful_v2", "bidpo_control", "token_faithful_"):
        assert real.out_of_scope_for(sp) is None and HPO._kind_entry(sp) == (None, None), sp


def test_a_claimless_first_summary_never_hands_the_kind_to_the_directory_name(tmp_path):
    """Review round 3 of 125f4194a (2026-10-08): the decisive spelling was the fit record's, else the FIRST summary's, else the directory's
    name -- so a sweep whose first summary recorded no spelling (a failed trial) was classified by its DIRECTORY NAME while
    `_check_one_kind` read only the later summaries' claims and found them consistent. Three faces of it: an in-scope `psr_ce_all` claim
    under a `token_faithful_all` directory PARKED (an accepted empty revision, the in-scope trial filed as nothing); a declared
    `token_faithful` claim under an unmapped directory name REFUSED as unmapped; and a `token_faithful` claim under a `psr_ce_all_all`
    directory read as a relu_steer SEARCH with the out-of-scope trial among its trials. The decisive spelling is now the first of the
    SAME claims the consistency check reads -- the fit record's, else the first summary that names one -- and the directory's name stands
    in only when no record claims a spelling at all; the check still refuses a disagreement behind a claimless first summary."""
    root = str(tmp_path / "hpo-x")
    # the three in-scope sweeps spell three DISTINCT relu_steer interventions: a search's identity is (root label, model, spelling, position,
    # si, key) -- the directory is not part of it -- so two directories spelling the same intervention would be one fit_search row
    in_scope_under_parked_name = _sweep_of_claims(root, "token_faithful_all", [None, "psr_ce_all"])
    in_scope_under_unmapped_name = _sweep_of_claims(root, "novel_all", [None, "psr_mse_all"])
    parked_under_unmapped_name = _sweep_of_claims(root, "novel_b_all", [None, "token_faithful"])
    parked_under_in_scope_name = _sweep_of_claims(root, "psr_ce_all_all", [None, "token_faithful"])
    no_record_claims = _sweep_of_claims(root, "token_faithful", [None, None])
    fit_record_decides = _sweep_of_claims(root, "fit_first_all", [None, "psr_ce_all"], fit_claim="psr_kl_all")
    mixed_behind_claimless = _sweep_of_claims(root, "mixed_e_all", [None, "psr_ce_all", "token_faithful"])
    _write(root, "PODS.md", "# pods\n")
    sweeps = {sd.path: sd for sd in HPO.enumerate_sweeps([("hpo-x", root, "flat")])}
    searches = {in_scope_under_parked_name: "psr_ce_all", in_scope_under_unmapped_name: "psr_mse_all", fit_record_decides: "psr_kl_all"}
    parks = {parked_under_unmapped_name: "r1_t01/summary.json:arm", parked_under_in_scope_name: "r1_t01/summary.json:arm",
             no_record_claims: "the sweep directory's name"}
    assert set(sweeps) == {*searches, *parks, mixed_behind_claimless}
    # the in-scope claim reads as a relu_steer search over both trials whatever the directory is called; the fit record's spelling wins
    for d, spelling in searches.items():
        s, trials = HPO.build_search(sweeps[d], _seeds(), T(9000), mtime_utc=T(100))
        cls = s["target_object_class_json"]
        assert cls["object_kind"] == "relu_steer" and cls["arm_legacy"] == spelling and s["n_trials"] == 2 and len(trials) == 2, d
        assert s["extra"]["arm_dir"] == os.path.basename(d) and s["trials_uri"] == d                 # the directory is still recorded, as a fact
        assert isinstance(VAL.validate_record(synth.rec("fit_search", s, T(1)), "<m>"), VAL.Validated)
    # the declared out-of-scope claim parks whatever the directory is called, the fact naming the record it was read off; the directory's
    # name decides only when NO record claims a spelling
    for d, source in parks.items():
        with pytest.raises(HPO.HpoParked) as pi:
            HPO.build_search(sweeps[d], _seeds(), T(9000))
        exc = pi.value
        assert exc.where == d and exc.kind_legacy == "token_faithful" and exc.prefix == "token_faithful" and exc.kind_legacy_source == source, d
        assert exc.n_trial_summaries == 2 and exc.entry["status"] == "withdrawn"
        assert HPO.parked_status_facts(exc)["parked"][0]["kind_legacy_source"] == source
    # a disagreement behind a claimless first summary still refuses by name, citing the two records that disagree
    with pytest.raises(HPO.HpoRefusal) as ei:
        HPO.build_search(sweeps[mixed_behind_claimless], _seeds(), T(9000))
    assert not isinstance(ei.value, HPO.HpoParked) and ei.value.where == mixed_behind_claimless
    assert "disagree" in ei.value.why and "r1_t01/summary.json" in ei.value.why and "r1_t02/summary.json" in ei.value.why and "r1_t00" not in ei.value.why
    hs = HPO.scan([("hpo-x", root, "flat")], seeds=_seeds())
    assert hs.by_root == {"hpo-x": {"sweeps": 7, "searches": 3, "trials": 6, "refused": 1, "parked": 3}}
    assert [r["dir"] for r in hs.refusals] == [mixed_behind_claimless] and sorted(p["dir"] for p in hs.parked) == sorted(parks)
    assert {p["dir"]: p["parked"]["kind_legacy_source"] for p in hs.parked} == parks
    # the pass: the three searches and the three parks ACCEPTED, the mixed sweep REFUSED; no park and no refusal contrary to the records
    reg = _registry(tmp_path)
    ad = HPO.HpoLogsAdapter([("hpo-x", root, "flat")], seeds=_seeds())
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 1 and rep.groups_refreshed == 6, [r.error for r in rep.results]
    by_group = {r.group: r for r in rep.results}
    assert not by_group[mixed_behind_claimless].accepted and "disagree" in by_group[mixed_behind_claimless].error and ad.status_extra(mixed_behind_claimless) == {}
    for d in searches:
        assert by_group[d].accepted and ad.status_extra(d) == {}, d
    for d in parks:
        assert by_group[d].accepted and set(ad.status_extra(d)) == {"refuse", "parked"}, d
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        rows = conn.execute("SELECT trials_uri, json_extract(target_object_class_json, '$.arm_legacy'), json_extract(target_object_class_json, '$.object_kind') "
                            "FROM fit_search").fetchall()
        assert {r[0]: (r[1], r[2]) for r in rows} == {d: (sp, "relu_steer") for d, sp in searches.items()}
        assert conn.execute("SELECT COUNT(*) FROM fit_trial").fetchone()[0] == 6
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE source_id = ? AND accepted = 1",
                            (RUN.source_id_for(ad.name, mixed_behind_claimless),)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM source_revision WHERE adapter = 'hpo_logs' AND accepted = 0").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM source_status WHERE adapter = 'hpo_logs' AND json_extract(extra_json, '$.parked') IS NOT NULL").fetchone()[0] == 3
        listed = sorted((r["kind_legacy"], r["dir"]) for r in _parked_sources(conn) if r["adapter"] == "hpo_logs")
    finally:
        conn.close()
    assert listed == sorted(("token_faithful", d) for d in parks)


def test_pass_and_ingest_fit_search_rows(tmp_path):
    base, hpo, sweep, winner_lr, _ = _reseed(tmp_path)
    reg = str(tmp_path / "reg")
    H.ensure_layout(reg)
    R.write_records(os.path.join(reg, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    H.ingest_once(reg, workers=0, min_free=0)
    ad = HPO.HpoLogsAdapter([("reseed_eb64", hpo, "reseed")], seeds=_seeds())
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed >= 1, [r.error for r in rep.results]
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        assert conn.execute("SELECT COUNT(*) FROM fit_search").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM fit_trial").fetchone()[0] == 5
        row = conn.execute("SELECT argmin_interior, quadratic_fit_positive_definite, confirmation_within_noise, method, substrate FROM fit_search").fetchone()
        assert tuple(row) == (1, 1, 1, "sobol", "modal")
    finally:
        conn.close()
    hs = HPO.scan([("reseed_eb64", hpo, "reseed")], seeds=_seeds())
    assert hs.by_root == {"reseed_eb64": {"sweeps": 1, "searches": 1, "trials": 5, "refused": 0, "parked": 0}}


def test_unverified_substrate_passes_the_ddl_check_and_ingests(tmp_path):
    """Known-bad-turned-admitted: before the vocabulary change this row failed `fit_search.substrate`'s CHECK; it must
    now reach the table through the incremental pass and the harvester, still as `unverified`."""
    root = str(tmp_path / "hpo-x")
    _flat_sweep_without_substrate_evidence(root)
    reg = str(tmp_path / "reg")
    H.ensure_layout(reg)
    R.write_records(os.path.join(reg, "records", "dims", "000001-dims.jsonl"), synth.seed_dimensions())
    H.ingest_once(reg, workers=0, min_free=0)
    ad = HPO.HpoLogsAdapter([("hpo-x", root, "flat")], seeds=_seeds())
    conn = DB.connect(DB.db_path(reg), readonly=True)
    rep = RUN.run_pass(reg, [ad], conn=conn, now=T(9000))
    conn.close()
    assert rep.groups_refused == 0 and rep.groups_refreshed == 1, [r.error for r in rep.results]
    ing = H.ingest_once(reg, workers=0, min_free=0)
    assert ing.refused == 0, ing.refusals
    conn = DB.connect(DB.db_path(reg), readonly=True)
    try:
        rows = conn.execute("SELECT substrate, json_extract(field_confidence_json, '$.substrate.confidence') FROM fit_search").fetchall()
        assert [tuple(r) for r in rows] == [("unverified", "unverified")]
        assert conn.execute("SELECT COUNT(*) FROM fit_trial").fetchone()[0] == 2
    finally:
        conn.close()
    assert HPO.scan([("hpo-x", root, "flat")], seeds=_seeds()).by_root == {"hpo-x": {"sweeps": 1, "searches": 1, "trials": 2, "refused": 0, "parked": 0}}
