"""The refine / confirm split: after the design fit, `emit-trials
--refine-from` emits THREE trials at the selected pair on +-0.5 decade around
u* (round 2, 1-D scrambled Sobol seed 3); `analyze --rounds 1,2 --pair-only`
refits that pair; then the FOURTH trial differs by path — PD + interior refit
-> the exact CONFIRMATION (round 9, build_confirmation_trial's contract, never
pooled); non-PD refit, or an argmin outside the refined box -> the EXPLORATION
REPLICATE (round 3) at the promoted measured trial's exact coordinates with
`trial_role = "exploration_replicate"`, the confirmation fields NULL, and
analyze refusing to pool it. Both paths spend exactly 3 + 1.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from test_tune_arm import (  # noqa: E402
    _identified_payload, _trial_args, _write_cfg, _write_pool, tune_arm)
from test_hpo_design_v2 import (  # noqa: E402
    A0, ARM, BETA, BOX, C0, CLIP, GAMMA, MODEL, U0, WF, analyze_v2, build, write_summaries)

guard = tune_arm.hpo_selection_guard


def _cfg(tmp_path):
    p = _write_cfg(tmp_path, {"model": MODEL, "model_revision": "0f0813", "data_mode": "documents",
                              "train_kwargs": {"default": {}, ARM: {}}})
    return str(p)


def _emit(tmp_path, cfg, **over):
    ns = tune_arm.emit_args(out=str(tmp_path / over.pop("out")), arm=ARM, config=cfg, si="constv2-mix-10m", **over)
    tune_arm.emit_trials(ns)
    return json.loads(Path(ns.out).read_text())


def _design_round(tmp_path, cfg):
    p = build(tmp_path)
    p["config_sha"] = tune_arm._config_file_sha(cfg)
    p["search_id"] = "hpo-native:feedfacefeedface"
    (tmp_path / "r1_trials.json").write_text(json.dumps(p))
    write_summaries(tmp_path, p)
    art, rc = analyze_v2(tmp_path, allowed=tmp_path / "allowed_pairs.json")
    assert rc is None and art["selection"]["selected_from"] == "fitted_argmin"
    return p, art


def _refine_summaries(tmp_path, r2: dict, log_y):
    for t in r2["trials"]:
        d = tmp_path / f"r2_t{t['index']:02d}"; d.mkdir()
        (d / "summary.json").write_text(json.dumps({
            "trial": t, "round": 2, "status": "ok", "arm": ARM, "model": MODEL, "position": "all",
            "si": "constv2-mix-10m", "box": r2["box"], "warmup_levels": r2["warmup_levels"],
            "best_heldout_loss": math.exp(log_y(t)), "floor_ok": True, "heldout_curve": [2.4, 2.0, 1.95],
            "trainable_params": 58_900_480 * 4, "refine_of": r2["refine_of"], "lr_log10_box": r2["lr_log10_box"],
            "search_id": r2.get("search_id")}))


def planted(t):
    u = math.log10(t["lr"])
    return A0 + C0 * (u - U0) ** 2 + BETA[t["lora_rank"]] + GAMMA[t["lora_set_id"]]


def test_refine_emits_three_trials_at_the_winning_pair_on_half_a_decade(tmp_path):
    cfg = _cfg(tmp_path)
    p, art = _design_round(tmp_path, cfg)
    r2 = _emit(tmp_path, cfg, out="r2_trials.json", refine_from=str(tmp_path / "r1_fit.json"))
    assert r2["round"] == 2 and len(r2["trials"]) == 3 and r2["design"] == "refine_sobol4_first3"
    ro = r2["refine_of"]
    assert ro["pair"]["lora_rank"] == 16 and ro["pair"]["lora_set_id"] == "mlp"
    assert math.isclose(ro["center_log10"], U0, abs_tol=1e-6)
    assert math.isclose(ro["box_log10"][0], U0 - 0.5, abs_tol=1e-6) and math.isclose(ro["box_log10"][1], U0 + 0.5, abs_tol=1e-6)
    assert ro["seed"] == 3 and ro["selected_from"] == "fitted_argmin"
    for t in r2["trials"]:
        assert t["lora_rank"] == 16 and t["lora_set_id"] == "mlp" and t["max_grad_norm"] == CLIP
        assert t["warmup_frac"] == WF and t["init_scheme"] == "random_lecun_1e-3"
        assert t["lora_modules"] == ["down_proj", "gate_proj", "up_proj"]
        assert U0 - 0.5 <= math.log10(t["lr"]) <= U0 + 0.5
    assert r2["search_id"] == "hpo-native:feedfacefeedface" and r2["constant_axes"] == {"max_grad_norm": CLIP}
    # deterministic
    again = _emit(tmp_path, cfg, out="r2b.json", refine_from=str(tmp_path / "r1_fit.json"))
    assert again["trials"] == r2["trials"]
    # a design flag alongside refuses; a pooled round refuses
    with pytest.raises(SystemExit, match="takes no design flags"):
        _emit(tmp_path, cfg, out="x.json", refine_from=str(tmp_path / "r1_fit.json"), lr_min=1e-4)
    with pytest.raises(SystemExit, match="already pooled"):
        _emit(tmp_path, cfg, out="x.json", refine_from=str(tmp_path / "r1_fit.json"), round=1)


def test_pd_interior_refit_takes_the_exact_confirmation_never_pooled(tmp_path):
    cfg = _cfg(tmp_path)
    p, art = _design_round(tmp_path, cfg)
    r2 = _emit(tmp_path, cfg, out="r2_trials.json", refine_from=str(tmp_path / "r1_fit.json"))
    _refine_summaries(tmp_path, r2, planted)
    refit, rc = analyze_v2(tmp_path, rounds="1,2", pair_only=True)
    assert rc is None
    assert refit["pair_only"] and refit["refit_pair"] == [16, "mlp"]
    # the refit pools the pair's design points + the three refine trials, nothing else
    assert {r["round"] for r in refit["rows"]} == {1, 2}
    assert all((r["lora_rank"], r["lora_set_id"]) == (16, "mlp") for r in refit["rows"])
    assert len([r for r in refit["rows"] if r["round"] == 2]) == 3
    sel = refit["selection"]
    assert sel["selected_from"] == "fitted_argmin" and not sel["on_edge"]
    assert math.isclose(math.log10(sel["lr"]), U0, abs_tol=1e-6)
    assert refit["next"]["action"] == "confirm" and refit["next"]["round"] == 9
    assert refit["followup"]["confirmation_trial_id"] is None
    # the FOURTH trial: the exact confirmation, coordinates verbatim, round 9
    r9 = _emit(tmp_path, cfg, out="r9_trials.json", confirm_from=str(tmp_path / "r1+2_fit_pair.json"))
    assert r9["round"] == 9 and len(r9["trials"]) == 1 and tune_arm.is_confirmation_design(r9)
    t = r9["trials"][0]
    assert t["lr"] == sel["lr"] and t["max_grad_norm"] == CLIP and t["warmup_frac"] == WF
    assert t["lora_rank"] == 16 and t["lora_modules"] == ["down_proj", "gate_proj", "up_proj"]
    assert t["init_scheme"] == "random_lecun_1e-3" and t["per_expert_rank"] == {"shared_expert": 16, "routed_expert": 2}
    assert r9["confirmation_of"]["selected_from"] == "fitted_argmin"
    assert r9["search_id"] == "hpo-native:feedfacefeedface"
    # never pooled: a round-9 summary in the pool refuses
    d = tmp_path / "r9_t00"; d.mkdir()
    (d / "summary.json").write_text(json.dumps({"trial": t, "round": 9, "status": "ok", "arm": ARM, "model": MODEL,
                                               "position": "all", "box": r9["box"], "warmup_levels": r9["warmup_levels"],
                                               "best_heldout_loss": math.exp(planted(t) + 0.002), "floor_ok": True,
                                               "confirmation_of": r9["confirmation_of"]}))
    with pytest.raises(SystemExit, match="CONFIRMATION"):
        analyze_v2(tmp_path, rounds="1,2,9", pair_only=True)
    # the followup evidence reads the confirmation and rules within-noise against the refit's sigma
    refit2, _ = analyze_v2(tmp_path, rounds="1,2", pair_only=True)
    fu = refit2["followup"]
    assert fu["confirmation_trial_id"] == "r9_t00" and fu["confirmation_within_noise"] is not None
    assert fu["exploration_replicate"] is None
    # the exploration path is REFUSED on the PD-interior artefact
    with pytest.raises(SystemExit, match="INTERIOR fitted argmin"):
        _emit(tmp_path, cfg, out="r3.json", explore_from=str(tmp_path / "r1+2_fit_pair.json"))


def _fallback_case(tmp_path, cfg, log_y):
    p, art = _design_round(tmp_path, cfg)
    r2 = _emit(tmp_path, cfg, out="r2_trials.json", refine_from=str(tmp_path / "r1_fit.json"))
    # the pair's OWN surface differs from the design fit's additive picture: rewrite the pair's design-round
    # summaries onto the same surface the refine trials measure, so the pooled pair rows are one world
    for t in p["trials"]:
        if (t["lora_rank"], t["lora_set_id"]) == (16, "mlp"):
            sp = tmp_path / f"r1_t{t['index']:02d}" / "summary.json"
            s = json.loads(sp.read_text()); s["best_heldout_loss"] = math.exp(log_y(t)); sp.write_text(json.dumps(s))
    _refine_summaries(tmp_path, r2, log_y)
    refit, rc = analyze_v2(tmp_path, rounds="1,2", pair_only=True)
    return r2, refit, rc


def _assert_exploration_replicate(tmp_path, cfg, refit):
    sel = refit["selection"]
    assert sel["selected_from"] == "best_raw_trial"
    assert refit["winner"] is None
    assert refit["next"]["action"] == "explore" and refit["next"]["round"] == 3
    assert refit["followup"]["confirmation_trial_id"] is None
    assert refit["followup"]["confirmation_within_noise"] is None
    with pytest.raises(SystemExit, match="refit gave no|no fitted-argmin|trustworthy interior|box edge"):
        _emit(tmp_path, cfg, out="r9.json", confirm_from=str(tmp_path / "r1+2_fit_pair.json"))
    r3 = _emit(tmp_path, cfg, out="r3_trials.json", explore_from=str(tmp_path / "r1+2_fit_pair.json"))
    assert r3["round"] == 3 and len(r3["trials"]) == 1
    assert r3["trial_role"] == "exploration_replicate" and tune_arm.is_replicate_design(r3)
    assert not tune_arm.is_confirmation_design(r3) and "confirmation_of" not in r3
    tid = refit["selection_guard"]["selected_trial"]
    orig = next(r for r in refit["rows"] if guard.trial_id(r) == tid)
    t = r3["trials"][0]
    for k in ("lr", "warmup_frac", "max_grad_norm", "lora_rank", "lora_modules", "per_expert_rank", "init_scheme"):
        assert t[k] == orig[k], k
    ro = r3["exploration_replicate_of"]
    assert ro["trial"] == tid and ro["original_y"] == orig["best_heldout_loss"]
    assert "NOT a confirmation" in ro["note"]
    # its summary: role stamped, never pooled, CE difference = replicate noise; confirmation fields stay NULL
    d = tmp_path / "r3_t00"; d.mkdir()
    (d / "summary.json").write_text(json.dumps({"trial": t, "round": 3, "status": "ok", "arm": ARM, "model": MODEL,
                                               "position": "all", "box": r3["box"], "warmup_levels": r3["warmup_levels"],
                                               "best_heldout_loss": orig["best_heldout_loss"] + 0.004, "floor_ok": True,
                                               "trial_role": "exploration_replicate",
                                               "exploration_replicate_of": ro}))
    with pytest.raises(SystemExit, match="EXPLORATION REPLICATE"):
        analyze_v2(tmp_path, rounds="1,2,3", pair_only=True)
    refit2, _ = analyze_v2(tmp_path, rounds="1,2", pair_only=True)
    fu = refit2["followup"]
    assert fu["confirmation_trial_id"] is None and fu["confirmation_within_noise"] is None
    assert fu["exploration_replicate"]["trial_id"] == "r3_t00"
    assert math.isclose(fu["exploration_replicate"]["replicate_noise"], 0.004, abs_tol=1e-9)
    assert fu["exploration_replicate"]["replicates"] == tid
    return r3


def test_non_pd_refit_takes_the_exploration_replicate(tmp_path):
    cfg = _cfg(tmp_path)
    # a CONCAVE refine surface at the pair: the refit's quadratic in u has c < 0
    def log_y(t):
        u = math.log10(t["lr"])
        return A0 - 0.8 * (u - U0) ** 2 + BETA[16] + GAMMA["mlp"]
    r2, refit, rc = _fallback_case(tmp_path, cfg, log_y)
    assert rc is not None and not refit["fit"]["positive_definite"]
    assert refit["selection"]["fall_through"]["reason"] == "fit_not_positive_definite"
    _assert_exploration_replicate(tmp_path, cfg, refit)


def test_argmin_outside_the_refined_box_takes_the_exploration_replicate(tmp_path):
    cfg = _cfg(tmp_path)
    # a convex refine surface whose minimum sits a full decade above the refined box's top
    def log_y(t):
        u = math.log10(t["lr"])
        return A0 + 0.6 * (u - (U0 + 1.6)) ** 2 + BETA[16] + GAMMA["mlp"]
    r2, refit, rc = _fallback_case(tmp_path, cfg, log_y)
    assert rc is None and refit["fit"]["positive_definite"]
    assert refit["selection"]["fall_through"]["reason"] == "fitted_argmin_on_edge"
    assert refit["selection"]["on_edge"] is True
    _assert_exploration_replicate(tmp_path, cfg, refit)


def test_run_trial_summary_carries_the_role_and_the_registry_record_is_emitted_once(tmp_path, monkeypatch):
    """The trial runner stamps trial_role / search_id / trainable_params onto its summary and calls
    record_trial exactly once with status complete|failed (through the lazy emitter seam)."""
    calls = []

    class FakeEmit:
        @staticmethod
        def record_trial(sid, tid, params, **kw):
            calls.append((sid, tid, params, kw))

    monkeypatch.setattr(tune_arm, "_emit_hpo", lambda: FakeEmit)
    spec = {"round": 3, "search_id": "hpo-native:abc", "campaign": "two_si", "trial_role": "exploration_replicate",
            "exploration_replicate_of": {"trial": "r1_t02"}}
    trial = {"index": 0, "lr": 1e-3, "warmup_frac": WF, "max_grad_norm": CLIP, "lora_rank": 16,
             "lora_modules": ["down_proj", "gate_proj", "up_proj"], "lora_set_id": "mlp",
             "per_expert_rank": {"shared_expert": 16, "routed_expert": 2}, "init_scheme": "zero"}
    summary = {"status": "ok", "best_heldout_loss": 1.91, "trainable_params": 235_601_920, "epochs_seen": 2,
               "started_utc": "2026-09-29T00:00:00Z", "finished_utc": "2026-09-29T01:00:00Z", "wandb_run": None,
               "arm": ARM, "trial": trial, "heldout_curve": [2.4, 2.0, 1.91], "best_epoch": 2}
    out = tmp_path / "r3_t00"; out.mkdir()
    tune_arm._record_trial(spec, trial, summary, out)
    assert len(calls) == 1
    sid, tid, params, kw = calls[0]
    assert sid == "hpo-native:abc" and tid == "r3_t00" and kw["status"] == "complete" and kw["campaign"] == "two_si"
    assert params["lora_rank"] == 16 and params["lora_set_id"] == "mlp" and params["trainable_params"] == 235_601_920
    assert params["lora_modules"] == ["down_proj", "gate_proj", "up_proj"] and params["init_scheme"] == "zero"
    assert kw["extra"]["trial_role"] == "exploration_replicate" and kw["epochs_seen"] == 2
    assert kw["clip_fraction"] is None                     # no metrics.jsonl beside the summary
    tune_arm._record_trial(spec, trial, {**summary, "status": "nan_diverged", "error": "boom", "best_heldout_loss": None}, out)
    assert calls[1][3]["status"] == "failed" and calls[1][3]["extra"]["reason"] == "boom"
    assert calls[1][3]["objective_value"] is None
    # no search id / campaign -> nothing recorded, said
    tune_arm._record_trial({"round": 1}, trial, summary, out)
    tune_arm._record_trial({"round": 1, "search_id": "hpo-native:abc"}, trial, summary, out)
    assert len(calls) == 2


# --------------------------------------------------------------------------
# the fit_trial row's emission EVIDENCE beside the summary, and the matching-summary resume that verifies /
# replays it (never retrains); the completed count is every terminal trial of the search across every round
# (16 + 3 + 1), reconciled against those rows.
# --------------------------------------------------------------------------

SID = "hpo-native:feedfacefeedface"
OPENING = {"search_id": SID, "campaign": "two_si", "model_id": "toy_moe_base", "object_kind": "lora",
           "method": "sobol", "n_trials": 16, "n_trials_completed": 0, "objective": "heldout_ce",
           "objective_direction": "min", "substrate": "cluster_a", "started_utc": "2026-09-29T00:00:00.000Z",
           "search_space_json": {"lr": {"log10_box": list(BOX)}}, "extra": {"phase": "open"}}


def _fake_emit(calls, path="/spool/two_si/fit_search__feedface/000001-fit_trial.jsonl"):
    """The lazy emitter seam: record_trial returns the record path a real writer returns (None = disabled)."""
    class FakeEmit:
        @staticmethod
        def record_trial(sid, tid, params, **kw):
            calls.append((sid, tid, params, kw))
            return path

        @staticmethod
        def close_search(sid, **kw):
            calls.append(("close", sid, kw))
            return {}
    return FakeEmit


def _record_all(root: Path, sid=SID):
    """The evidence _record_trial leaves for every terminal trial under root (what a clean run produces)."""
    for p in sorted(root.glob("r*_t*/summary.json")):
        s = json.loads(p.read_text())
        (p.parent / tune_arm.TRIAL_RECORD_NAME).write_text(json.dumps({
            "search_id": sid, "trial_id": p.parent.name, "campaign": "two_si",
            "status": "complete" if s["status"] == "ok" else "failed",
            "fit_fingerprint": s.get("fit_fingerprint"), "record_path": f"/spool/{p.parent.name}.jsonl"}))


def _campaign_design_round(tmp_path, cfg):
    """_design_round for a CAMPAIGN-opened search: the payload (hence every summary) names the search, the
    campaign and the opening fit_search payload the closers chain from."""
    p = build(tmp_path)
    p["config_sha"] = tune_arm._config_file_sha(cfg)
    p.update(search_id=SID, campaign="two_si", search_opening=OPENING)
    (tmp_path / "r1_trials.json").write_text(json.dumps(p))
    write_summaries(tmp_path, p)
    art, rc = analyze_v2(tmp_path, allowed=tmp_path / "allowed_pairs.json")
    assert rc is None and art["selection"]["selected_from"] == "fitted_argmin"
    return p, art


def _confirmed_world(tmp_path, cfg):
    """16 design + 3 refine + the round-9 CONFIRMATION (PD interior path), every summary stamped with the
    search id the trial runner copies from its design. Returns (design payload, refit artefact, r9 design)."""
    p, _art = _campaign_design_round(tmp_path, cfg)
    r2 = _emit(tmp_path, cfg, out="r2_trials.json", refine_from=str(tmp_path / "r1_fit.json"))
    _refine_summaries(tmp_path, r2, planted)
    refit, rc = analyze_v2(tmp_path, rounds="1,2", pair_only=True)
    assert rc is None and refit["next"]["action"] == "confirm"
    r9 = _emit(tmp_path, cfg, out="r9_trials.json", confirm_from=str(tmp_path / "r1+2_fit_pair.json"))
    t = r9["trials"][0]
    d = tmp_path / "r9_t00"; d.mkdir()
    (d / "summary.json").write_text(json.dumps({
        "trial": t, "round": 9, "status": "ok", "arm": ARM, "model": MODEL, "position": "all",
        "box": r9["box"], "warmup_levels": r9["warmup_levels"], "best_heldout_loss": math.exp(planted(t) + 0.002),
        "floor_ok": True, "confirmation_of": r9["confirmation_of"], "search_id": SID, "campaign": "two_si"}))
    refit, rc = analyze_v2(tmp_path, rounds="1,2", pair_only=True)       # the followup now reads the confirmation
    assert rc is None and refit["followup"]["confirmation_trial_id"] == "r9_t00"
    return p, refit, r9


def test_record_trial_writes_evidence_only_when_the_writer_returned_a_path(tmp_path, monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(tune_arm, "_emit_hpo", lambda: _fake_emit(calls))
    spec = {"round": 1, "search_id": SID, "campaign": "two_si"}
    trial = {"index": 3, "lr": 1e-3, "warmup_frac": WF, "max_grad_norm": CLIP, "lora_rank": 16,
             "lora_modules": ["down_proj", "gate_proj", "up_proj"], "lora_set_id": "mlp", "init_scheme": "zero"}
    summary = {"status": "ok", "best_heldout_loss": 1.9, "trial": trial, "arm": ARM, "fit_fingerprint": "f" * 64}
    out = tmp_path / "r1_t03"; out.mkdir()
    ev = tune_arm._record_trial(spec, trial, summary, out)
    assert ev and ev["search_id"] == SID and ev["trial_id"] == "r1_t03" and ev["status"] == "complete"
    assert ev["fit_fingerprint"] == "f" * 64 and ev["record_path"].endswith("fit_trial.jsonl")
    assert json.loads((out / tune_arm.TRIAL_RECORD_NAME).read_text()) == ev
    assert tune_arm.trial_record_matches(tune_arm.trial_record_evidence(out), search_id=SID, trial_id="r1_t03",
                                         fit_fingerprint="f" * 64)
    assert not tune_arm.trial_record_matches(tune_arm.trial_record_evidence(out), search_id=SID, trial_id="r1_t03",
                                             fit_fingerprint="0" * 64)
    assert not tune_arm.trial_record_matches(tune_arm.trial_record_evidence(out), search_id="hpo-native:other",
                                             trial_id="r1_t03", fit_fingerprint="f" * 64)
    # a DISABLED writer (no spool) returns no path: no row was written, so no evidence is written either — said
    out2 = tmp_path / "r1_t04"; out2.mkdir()
    monkeypatch.setattr(tune_arm, "_emit_hpo", lambda: _fake_emit(calls, path=None))
    assert tune_arm._record_trial(spec, {**trial, "index": 4}, summary, out2) is None
    assert not (out2 / tune_arm.TRIAL_RECORD_NAME).exists()
    assert "NOT written" in capsys.readouterr().out


def _resume_world(tmp_path, monkeypatch, *, campaign=True):
    """test_tune_arm.TestTrialResume._world, with a CAMPAIGN-opened design (search_id + campaign on the
    payload) unless campaign=False; a sentinel replaces _setup so any path past the skip raises."""
    _write_cfg(tmp_path, {"model": "NONEXISTENT/never-loaded", "max_len": 64})
    _write_pool(tmp_path)
    payload = _identified_payload(tmp_path, arm="lora_r1_all")
    if campaign:
        payload["search_id"] = SID
        payload["campaign"] = "two_si"
    (tmp_path / "trials.json").write_text(json.dumps(payload))
    args = _trial_args(tmp_path)
    cfg = json.loads((tmp_path / "cfg.yaml").read_text())
    spec = json.loads((tmp_path / "trials.json").read_text())
    fp = tune_arm._trial_fingerprint(args, cfg, spec, spec["trials"][0], "lora_r1_all", "all",
                                     tune_arm.make_adapter("lora_r1_all"))

    def boom(*a, **k):
        raise RuntimeError("SETUP-REACHED")

    monkeypatch.setattr(tune_arm, "_setup", boom)
    d = tmp_path / "r1_t00"; d.mkdir()
    return args, spec, fp, d


def test_matching_summary_resume_replays_a_missing_record_once_and_never_retrains(tmp_path, monkeypatch, capsys):
    """Preemption between the durable summary and its fit_trial record: the resume replays the record from
    the STORED summary — exactly once, keyed by search / trial / fingerprint — and never reaches the model
    load; a summary that names a saved object refuses when that file is gone."""
    args, spec, fp, d = _resume_world(tmp_path, monkeypatch)
    calls = []
    monkeypatch.setattr(tune_arm, "_emit_hpo", lambda: _fake_emit(calls))
    trial = spec["trials"][0]
    (d / "summary.json").write_text(json.dumps({
        "status": "ok", "fit_fingerprint": fp, "best_heldout_loss": 1.93, "trial": trial, "arm": "lora_r1_all",
        "round": 1, "artifact": None, "epochs_seen": 1, "heldout_curve": [2.4, 1.93]}))
    tune_arm.run_trial(args)                                  # the skip holds: SETUP is never reached
    out = capsys.readouterr().out
    assert "skipping" in out and "replaying the record" in out
    assert len(calls) == 1
    sid, tid, params, kw = calls[0]
    assert sid == SID and tid == "r1_t00" and kw["status"] == "complete" and kw["objective_value"] == 1.93
    ev = tune_arm.trial_record_evidence(d)
    assert ev["fit_fingerprint"] == fp and ev["record_path"] and ev["trial_id"] == "r1_t00"
    # idempotent: a second resume verifies the evidence and emits nothing
    tune_arm.run_trial(args)
    assert len(calls) == 1 and "already recorded" in capsys.readouterr().out
    # evidence of OTHER work (another fingerprint) is not this trial's proof: replayed once more, then stable
    ev["fit_fingerprint"] = "0" * 64
    (d / tune_arm.TRIAL_RECORD_NAME).write_text(json.dumps(ev))
    tune_arm.run_trial(args)
    assert len(calls) == 2 and tune_arm.trial_record_evidence(d)["fit_fingerprint"] == fp
    # a summary that names its saved object is not the artefact: the file gone -> refuse (never a silent re-train)
    s = json.loads((d / "summary.json").read_text()); s["artifact"] = "object.pt"
    (d / "summary.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="names its saved object"):
        tune_arm.run_trial(args)
    (d / "object.pt").write_bytes(b"synthetic trial object")
    tune_arm.run_trial(args)
    assert len(calls) == 2


def test_resume_of_a_design_outside_any_search_records_nothing(tmp_path, monkeypatch, capsys):
    args, spec, fp, d = _resume_world(tmp_path, monkeypatch, campaign=False)
    calls = []
    monkeypatch.setattr(tune_arm, "_emit_hpo", lambda: _fake_emit(calls))
    (d / "summary.json").write_text(json.dumps({"status": "nan_diverged", "fit_fingerprint": fp}))
    tune_arm.run_trial(args)
    assert calls == [] and not (d / tune_arm.TRIAL_RECORD_NAME).exists()
    assert "skipping" in capsys.readouterr().out


def test_run_trial_saves_the_object_before_the_summary_and_names_it():
    """The summary is the terminal marker a resume skips on, so the artefact is saved BEFORE it and named in it."""
    src = Path(tune_arm.__file__).read_text()
    body = src.split("def run_trial(args)", 1)[1].split("\n# ---", 1)[0]
    i_save, i_summary, i_write, i_record = (body.index('saved_artifact = adapter.save(out) if status == "ok" else None'),
                                            body.index("summary = {"), body.index('tmp.replace(out / "summary.json")'),
                                            body.index("_record_trial(spec, trial, summary, out)"))
    assert i_save < i_summary < i_write < i_record
    assert '"artifact": saved_artifact' in body and body.count("adapter.save(out)") == 1


def test_completed_trials_counts_every_round_and_reconciles_the_emitted_rows(tmp_path, monkeypatch, capsys):
    """16 design + 3 refine + 1 confirmation = 20 completed trials, reconciled against their fit_trial evidence;
    the closing record carries that count (the pooled fit rows are the FIT's inputs, fewer than 20); a terminal
    trial without its emitted row, or a directory mixing searches, holds the close by name."""
    cfg = _cfg(tmp_path)
    _p, refit, _r9 = _confirmed_world(tmp_path, cfg)
    _record_all(tmp_path)
    done = tune_arm.completed_trials(tmp_path, SID)
    ids = done["trial_ids"]
    assert done["n"] == 20 and len(set(ids)) == 20 and done["by_status"] == {"ok": 20}
    assert sum(t.startswith("r1_") for t in ids) == 16 and sum(t.startswith("r2_") for t in ids) == 3 and "r9_t00" in ids
    assert done["unrecorded"] == [] and done["foreign"] == []
    calls = []
    monkeypatch.setattr(tune_arm, "_emit_hpo", lambda: _fake_emit(calls))
    tune_arm._close_search_from(refit, tmp_path)
    close = [c for c in calls if c[0] == "close"]
    assert len(close) == 1 and close[0][1] == SID
    kw = close[0][2]
    assert kw["n_trials_completed"] == 20 and sorted(kw["extra"]["trials_completed"]) == sorted(ids)
    assert kw["extra"]["n_trials_by_status"] == {"ok": 20}
    assert kw["extra"]["n_fit_rows"] == len(refit["rows"]) < 20        # the refit pools the pair's points + 3, never the confirmation
    assert kw["confirmation_trial_id"] == "r9_t00" and kw["opening"] == OPENING
    # a legacy summary (no search id) is listed, not counted; a summary naming ANOTHER search refuses the count
    legacy = tmp_path / "r0_t00"; legacy.mkdir()
    (legacy / "summary.json").write_text(json.dumps({"status": "ok", "best_heldout_loss": 2.0, "trial": {"lr": 1e-3}}))
    done2 = tune_arm.completed_trials(tmp_path, SID)
    assert done2["n"] == 20 and done2["foreign"] == [("r0_t00", None)]
    tune_arm.reconcile_completed_trials(tmp_path, SID)
    assert "name no search" in capsys.readouterr().out
    other = tmp_path / "r5_t00"; other.mkdir()
    (other / "summary.json").write_text(json.dumps({"status": "ok", "search_id": "hpo-native:0000000000000000"}))
    with pytest.raises(SystemExit, match="mixes searches.*r5_t00"):
        tune_arm.reconcile_completed_trials(tmp_path, SID)
    import shutil
    shutil.rmtree(other)
    # a terminal trial whose row has no emission evidence holds the close and names the replay
    (tmp_path / "r2_t01" / tune_arm.TRIAL_RECORD_NAME).unlink()
    assert tune_arm.completed_trials(tmp_path, SID)["unrecorded"] == ["r2_t01"]
    with pytest.raises(SystemExit, match="no fit_trial emission evidence.*r2_t01.*replays the record without retraining"):
        tune_arm._close_search_from(refit, tmp_path)
    assert len([c for c in calls if c[0] == "close"]) == 1
    # a torn summary is not terminal; a failed trial IS a completed one (its row says failed)
    (tmp_path / "r2_t01" / tune_arm.TRIAL_RECORD_NAME).write_text(json.dumps({
        "search_id": SID, "trial_id": "r2_t01", "fit_fingerprint": None, "record_path": "/spool/x.jsonl"}))
    s = json.loads((tmp_path / "r2_t02" / "summary.json").read_text()); s["status"] = "nan_diverged"
    (tmp_path / "r2_t02" / "summary.json").write_text(json.dumps(s))
    (tmp_path / "r1_t00" / "summary.json").write_text('{"status": "ok", "search_id": "' + SID)
    done3 = tune_arm.completed_trials(tmp_path, SID)
    assert done3["n"] == 19 and done3["by_status"] == {"ok": 18, "nan_diverged": 1} and done3["unrecorded"] == []


def test_record_trial_stamps_the_confirm_pick_role_on_its_row(tmp_path, monkeypatch):
    """The confirm-pick trial's fit_trial row (emitted once, at its terminal outcome, like every trial) carries the
    role and the request it answers, so the registry tells a confirm-pick from a search row."""
    calls = []
    monkeypatch.setattr(tune_arm, "_emit_hpo", lambda: _fake_emit(calls))
    cpo = {"pair": [16, "down"], "lr": 7.9e-4, "ruling_id": "r2-operating_point-toy_moe-lora-architecture-selection",
           "request": "hpo_pick_confirm_request.json", "lr_source": "design_fit:fitted_argmin_at_pair"}
    spec = {"round": tune_arm.CONFIRM_PICK_ROUND, "search_id": SID, "campaign": "two_si",
            "trial_role": tune_arm.CONFIRM_PICK_ROLE, "confirm_pick_of": cpo}
    trial = {"index": 0, "lr": 7.9e-4, "warmup_frac": WF, "max_grad_norm": CLIP, "lora_rank": 16,
             "lora_modules": tune_arm.modules_of_set("down"), "lora_set_id": "down", "init_scheme": "zero"}
    summary = {"status": "ok", "best_heldout_loss": 1.95, "trial": trial, "arm": ARM, "fit_fingerprint": "a" * 64,
               "trial_role": tune_arm.CONFIRM_PICK_ROLE, "confirm_pick_of": cpo}
    out = tmp_path / "r8_t00"; out.mkdir()
    ev = tune_arm._record_trial(spec, trial, summary, out)
    assert ev["trial_id"] == "r8_t00" and len(calls) == 1
    kw = calls[0][3]
    assert kw["status"] == "complete" and kw["extra"]["trial_role"] == "confirm_pick" and kw["extra"]["confirm_pick_of"] == cpo
    assert tune_arm.is_confirm_pick_design(spec) and not tune_arm.is_replicate_design(spec) and not tune_arm.is_confirmation_design(spec)


def test_both_closers_refuse_a_trial_whose_evidence_names_another_campaign_until_it_is_replayed(tmp_path, monkeypatch):
    """A campaign-less follow-up trial once replayed with the wrong --campaign has its sidecar (and record) under
    that other campaign; the search's campaign has no emitted row for it. BOTH closers — tune_arm's analyze --close-search and hpo_pick's
    close — reconcile on the search's campaign: the trial is named with its evidence's campaign and the replay is required BEFORE the close;
    after the replay (the sidecar under the search's campaign again) the close proceeds and counts it."""
    sys.path.insert(0, str(REPO / "scripts" / "sdfbase")); import hpo_pick  # noqa: E402
    cfg = _cfg(tmp_path)
    _p, refit, _r9 = _confirmed_world(tmp_path, cfg)
    _record_all(tmp_path)
    side = tmp_path / "r2_t01" / tune_arm.TRIAL_RECORD_NAME
    ev = json.loads(side.read_text()); side.write_text(json.dumps({**ev, "campaign": "other_campaign"}))
    done = tune_arm.completed_trials(tmp_path, SID, campaign="two_si")
    assert done["mis_campaigned"] == [("r2_t01", "other_campaign")] and done["unrecorded"] == [] and done["n"] == 20
    assert tune_arm.completed_trials(tmp_path, SID)["mis_campaigned"] == [], "no campaign asked: the pre-fix count (callers that only count)"
    with pytest.raises(SystemExit, match="under ANOTHER campaign than the search's 'two_si'.*r2_t01 \\(evidence under 'other_campaign'\\).*replayed under it"):
        tune_arm.reconcile_completed_trials(tmp_path, SID, campaign="two_si")
    calls = []
    monkeypatch.setattr(tune_arm, "_emit_hpo", lambda: _fake_emit(calls))
    with pytest.raises(SystemExit, match="under ANOTHER campaign.*r2_t01"):                    # the ReLU closer (analyze --close-search)
        tune_arm._close_search_from(refit, tmp_path)
    search = {"search_id": SID, "refit": None, "menu_fit": None, "design": refit, "trials": {}, "dir": str(tmp_path), "allowed": {}}
    pick = {"winner_params": {"lr": 1e-3}, "winner_trial_id": "r2_t01", "winner_objective": 1.9, "ruling_id": "ruling-1", "fit_key": "k",
            "lr_source": {"source": "refit:fitted_argmin", "trial_id": None, "alarms": [], "clip_evidence": {}}, "confirmation": None, "menu": {},
            "fit_search_config_sha256": {"value": None, "note": None}, "table_url": None}
    with pytest.raises(SystemExit, match="under ANOTHER campaign.*r2_t01"):                    # the LoRA closer (hpo_pick's close at the pick)
        hpo_pick._close_search(search, pick, execute=True)
    assert [c for c in calls if c[0] == "close"] == [], "nothing closed over the mis-campaigned evidence"
    side.write_text(json.dumps(ev))                                                              # the replay: the sidecar under the search's campaign again
    tune_arm._close_search_from(refit, tmp_path)
    close = [c for c in calls if c[0] == "close"]; assert len(close) == 1 and close[0][2]["n_trials_completed"] == 20 and "r2_t01" in close[0][2]["extra"]["trials_completed"]
