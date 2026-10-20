"""scripts/sdfbase/hpo_next_round.py — recenter / confirm / stop off one
analyze artefact, through the argmin guard.

CPU round trip: a synthetic LR-only round-1 design + summaries -> the real
`analyze` -> r1_fit.json -> the driver -> the emitted r2 / r9 design (or a
stop), plus the guard-reproduction refusal on a tampered artefact.
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from test_tune_arm import _analyze, _write_summaries, tune_arm  # noqa: E402
from test_tune_arm_lr_only_design import (  # noqa: E402
    CLIP, LLAMA_BASE_CFG, _lr_only_payload)
from test_tune_arm_confirmation import _write_metrics  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "hpo_next_round", REPO / "scripts" / "sdfbase" / "hpo_next_round.py")
hnr = importlib.util.module_from_spec(_spec)
sys.modules["hpo_next_round"] = hnr
_spec.loader.exec_module(hnr)

ARM = "psr_ce_all"
KEY = "llama8b_base"
SI = "constv2-mix-10m"


def test_driver_binds_the_same_tune_arm_and_guard_instance():
    assert hnr.tune_arm is tune_arm
    assert hnr.guard_mod is tune_arm.hpo_selection_guard
    assert hnr.CONFIRM_ROUND == 9 and hnr.RECENTER_ROUND == 2


def _summaries(out: Path, payload, *, mutate=None, si=SI, rates=None):
    def mut(idx, s):
        s["si"] = si
        if mutate:
            mutate(idx, s)
    _write_summaries(out, payload, mutate=mut)
    for t in payload["trials"]:
        if rates is not None:
            _write_metrics(out / f"r{payload['round']}_t{t['index']:02d}", ARM,
                           t["max_grad_norm"], rates.get(t["index"], 0.02))


def _edge(hi, decades_beyond=1.0):
    """Plant the lr optimum ``decades_beyond`` decades ABOVE the box top:
    an edge argmin. One recenter round (the seed box's own log-width — 2.5
    decades — centred on the argmin CLAMPED into the old box) reaches at
    most half that width above the old top, so anything beyond 1.25
    decades keeps the pooled argmin on the edge too."""
    target = math.log(hi) + decades_beyond * math.log(10.0)

    def mutate(idx, s):
        t = s["trial"]
        w = 0.0 if t["warmup_frac"] == 0.05 else 1.0
        s["best_heldout_loss"] = math.exp(
            2.0 * (math.log(t["lr"]) - target) ** 2 - 0.5 * w)
    return mutate


def _argv(fit: Path, out_dir: Path, **extra):
    argv = ["--fit", str(fit), "--config", str(LLAMA_BASE_CFG),
            "--model-key", KEY, "--arm", ARM, "--out-dir", str(out_dir)]
    for k, v in extra.items():
        argv += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return argv


# --------------------------------------------------------------------------
# edge -> recenter (one round, standing n, same width)
# --------------------------------------------------------------------------

def test_edge_argmin_recenters_with_the_standing_eight(tmp_path, capsys):
    payload = _lr_only_payload()
    lo, hi = tune_arm.LLAMA8B_CE_LR_BOX[ARM]
    _summaries(tmp_path, payload, mutate=_edge(hi))
    art = _analyze(tmp_path)
    assert art["selection"]["on_edge"] and art["next"]["action"] == "recenter"
    out_dir = tmp_path / "designs"
    code = hnr.main(_argv(tmp_path / "r1_fit.json", out_dir))
    assert code == 0
    design = out_dir / f"r2_{KEY}_{ARM}_all.json"
    p = json.loads(design.read_text())
    assert p["round"] == 2 and len(p["trials"]) == 8
    assert p["axes"] == ["lr"] and p["constant_axes"] == {"max_grad_norm": CLIP}
    assert p["si"] == SI
    nb = art["next"]["box"]["lr"]
    assert all(nb[0] <= t["lr"] <= nb[1] for t in p["trials"])
    assert math.isclose(math.log(nb[1]) - math.log(nb[0]),
                        math.log(hi) - math.log(lo), rel_tol=1e-9)
    rec = json.loads((tmp_path / "r1_next_round.json").read_text())
    assert rec["decision"] == "recenter"
    assert rec["reason"] == "fitted_argmin_on_edge"
    assert rec["emitted"] == str(design) and rec["emitted_round"] == 2
    assert rec["evidence"]["recenter"] == {
        "n": 8, "lr_only_design": True, "artefact_next_n": 8,
        "artefact_next_action": "recenter"}
    g = rec["evidence"]["guard"]
    assert g["reproduced"] and g["alarmed"]
    assert g["selected_from"] == "best_raw_trial"
    assert "fitted_argmin_on_edge" in g["recorded_codes"]
    out = capsys.readouterr().out
    assert "DECISION RECENTER" in out and "reproduced=True" in out


def test_two_axis_design_recenters_with_sixteen(tmp_path):
    box = {"lr": (3e-5, 3e-3), "max_grad_norm": (0.03, 3.0)}
    payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                    position="all", round_no=1, n=16, box=box)
    # plant the lr optimum a decade above the box top: an edge argmin
    def mutate(idx, s):
        t = s["trial"]
        s["best_heldout_loss"] = math.exp(
            2.0 * (math.log(t["lr"]) - math.log(3e-2)) ** 2
            + 1.5 * (math.log(t["max_grad_norm"]) - math.log(0.3)) ** 2)
    _write_summaries(tmp_path, payload, mutate=mutate)
    art = _analyze(tmp_path)
    assert art["selection"]["on_edge"]
    d = hnr.decide(art)
    assert d["decision"] == "recenter"
    assert d["evidence"]["recenter"]["n"] == 16
    assert d["evidence"]["recenter"]["lr_only_design"] is False


# --------------------------------------------------------------------------
# interior -> confirm (the one-row round-9 design), and nothing else
# --------------------------------------------------------------------------

def test_interior_argmin_confirms_with_the_one_row_design(tmp_path, capsys):
    payload = _lr_only_payload()
    _summaries(tmp_path, payload, rates={i: 0.03 for i in range(8)})
    art = _analyze(tmp_path)
    assert not art["selection"]["on_edge"] and art["next"]["action"] == "refine"
    out_dir = tmp_path / "designs"
    code = hnr.main(_argv(tmp_path / "r1_fit.json", out_dir))
    assert code == 0
    files = sorted(p.name for p in out_dir.iterdir())
    assert files == [f"r9_{KEY}_{ARM}_all.json"]          # nothing else
    p = json.loads((out_dir / files[0]).read_text())
    assert p["round"] == 9 and len(p["trials"]) == 1
    w = art["winner"]
    t = p["trials"][0]
    assert (t["lr"], t["max_grad_norm"], t["warmup_frac"]) == \
        (w["lr"], w["max_grad_norm"], w["warmup_frac"])
    assert p["si"] == SI and p["confirmation_of"]["fit"] == str(tmp_path / "r1_fit.json")
    rec = json.loads((tmp_path / "r1_next_round.json").read_text())
    assert rec["decision"] == "confirm" and rec["reason"] == "interior_fitted_argmin"
    assert rec["emitted_round"] == 9
    assert rec["evidence"]["guard"]["reproduced"] is True
    assert rec["evidence"]["guard"]["alarmed"] is False
    assert rec["evidence"]["winner"]["clip_activation"]["promotable"] is True
    out = capsys.readouterr().out
    assert "DECISION CONFIRM" in out and "FLAGGED" not in out


def test_flagged_clip_neighbourhood_still_confirms_but_says_so(tmp_path,
                                                                capsys):
    payload = _lr_only_payload()
    # the winner's level is 0.15 (the planted objective); clip every 0.15
    # trial on 40% of steps, the 0.05 ones on 2%
    rates = {t["index"]: (0.40 if t["warmup_frac"] == 0.15 else 0.02)
             for t in payload["trials"]}
    _summaries(tmp_path, payload, rates=rates)
    art = _analyze(tmp_path)
    assert art["winner"]["clip_activation"]["flagged"] is True
    assert not art["selection_guard"]["alarmed"]      # half the trials are fine
    d = hnr.decide(art)
    assert d["decision"] == "confirm"
    assert "clip activation FLAGGED" in d["text"]


def test_dry_run_decides_without_emitting(tmp_path):
    payload = _lr_only_payload()
    _summaries(tmp_path, payload)
    _analyze(tmp_path)
    out_dir = tmp_path / "designs"
    code = hnr.main(_argv(tmp_path / "r1_fit.json", out_dir, dry_run=True))
    assert code == 0 and not out_dir.exists()
    rec = json.loads((tmp_path / "r1_next_round.json").read_text())
    assert rec["decision"] == "confirm" and rec["emitted"] is None
    assert rec["dry_run"] is True


# --------------------------------------------------------------------------
# stop: non-PD, still on edge after pooling, alarmed interior
# --------------------------------------------------------------------------

def test_non_pd_fit_stops_and_names_the_fall_through(tmp_path, capsys):
    payload = _lr_only_payload()

    def saddle(idx, s):
        t = s["trial"]
        s["best_heldout_loss"] = math.exp(
            -2.0 * (math.log(t["lr"]) - math.log(3e-4)) ** 2)

    _summaries(tmp_path, payload, mutate=saddle,
               rates={i: 0.02 for i in range(8)})
    with pytest.raises(SystemExit, match="not positive definite"):
        _analyze(tmp_path)
    out_dir = tmp_path / "designs"
    code = hnr.main(_argv(tmp_path / "r1_fit.json", out_dir))
    assert code == hnr.EXIT_STOP == 3
    assert not out_dir.exists()
    rec = json.loads((tmp_path / "r1_next_round.json").read_text())
    assert rec["decision"] == "stop"
    assert rec["reason"] == "fit_not_positive_definite"
    assert rec["evidence"]["guard"]["reproduced"] is True
    assert "fit_not_positive_definite" in rec["evidence"]["guard"]["recorded_codes"]
    assert rec["evidence"]["clip_activation"]["best_raw_promotable"] in rec["text"]
    assert "DECISION STOP" in capsys.readouterr().out


def test_still_on_edge_after_pooling_two_rounds_stops(tmp_path):
    lo, hi = tune_arm.LLAMA8B_CE_LR_BOX[ARM]
    p1 = _lr_only_payload(round_no=1)
    far = _edge(hi, decades_beyond=4.0)     # beyond what one recenter reaches
    _summaries(tmp_path, p1, mutate=far)
    art1 = _analyze(tmp_path)
    assert art1["selection"]["on_edge"]
    nb = art1["next"]["box"]
    assert nb["lr"][1] < hi * 10 ** 4       # the recentred box stops short
    p2 = tune_arm.build_trials(arm=ARM, model=p1["model"], position="all",
                               round_no=2, n=8,
                               box={k: tuple(v) for k, v in nb.items()})
    _summaries(tmp_path, p2, mutate=far)    # the optimum is still beyond
    art = _analyze(tmp_path, rounds="1,2")
    assert art["rounds"] == [1, 2] and art["selection"]["on_edge"]
    assert art["fit"]["n_rows"] == 16
    d = hnr.decide(art)
    assert d["decision"] == "stop"
    assert d["reason"] == "still_on_edge_after_pooling"
    out_dir = tmp_path / "designs"
    code = hnr.main(_argv(tmp_path / "r1+2_fit.json", out_dir))
    assert code == hnr.EXIT_STOP and not out_dir.exists()
    assert (tmp_path / "r1+2_next_round.json").is_file()


def test_alarmed_interior_winner_stops_unless_allowed(tmp_path):
    payload = _lr_only_payload()
    _summaries(tmp_path, payload, rates={i: 0.9 for i in range(8)})
    art = _analyze(tmp_path)
    assert art["winner"] is not None and art["selection_guard"]["alarmed"]
    d = hnr.decide(art)
    assert d["decision"] == "stop" and d["reason"] == "alarmed_interior_winner"
    assert "clip_activation_over_threshold" in d["text"]
    out_dir = tmp_path / "designs"
    code = hnr.main(_argv(tmp_path / "r1_fit.json", out_dir))
    assert code == hnr.EXIT_STOP and not out_dir.exists()
    # deliberate: --allow-alarmed proceeds to the confirmation, on record
    code = hnr.main(_argv(tmp_path / "r1_fit.json", out_dir, allow_alarmed=True))
    assert code == 0
    p = json.loads((out_dir / f"r9_{KEY}_{ARM}_all.json").read_text())
    assert p["confirmation_of"]["guard_alarmed"] is True
    assert "clip_activation_over_threshold" in p["confirmation_of"]["guard_alarms"]
    rec = json.loads((tmp_path / "r1_next_round.json").read_text())
    assert rec["allow_alarmed"] is True and "acknowledged" in rec["text"]


# --------------------------------------------------------------------------
# the guard is called, never bypassed
# --------------------------------------------------------------------------

def test_a_tampered_guard_record_is_refused(tmp_path):
    payload = _lr_only_payload()
    _summaries(tmp_path, payload)
    _analyze(tmp_path)
    fit = tmp_path / "r1_fit.json"
    art = json.loads(fit.read_text())
    # (a) an alarm someone deleted
    lo, hi = tune_arm.LLAMA8B_CE_LR_BOX[ARM]
    _summaries(tmp_path / "edge", payload, mutate=_edge(hi))
    _analyze(tmp_path / "edge")
    edge = json.loads((tmp_path / "edge/r1_fit.json").read_text())
    edge["selection_guard"]["alarms"] = []
    edge["selection_guard"]["alarmed"] = False
    (tmp_path / "edge/r1_fit.json").write_text(json.dumps(edge))
    with pytest.raises(SystemExit, match="does NOT reproduce"):
        hnr.main(_argv(tmp_path / "edge/r1_fit.json", tmp_path / "d"))
    # (b) a selected point moved by hand
    art["selection"]["lr"] = art["selection"]["lr"] * 1.5
    fit.write_text(json.dumps(art))
    with pytest.raises(SystemExit, match="does NOT reproduce"):
        hnr.main(_argv(fit, tmp_path / "d"))
    # (c) no guard record at all
    del art["selection_guard"]
    fit.write_text(json.dumps(art))
    with pytest.raises(SystemExit, match="no selection_guard record"):
        hnr.main(_argv(fit, tmp_path / "d"))
    assert not (tmp_path / "d").exists()


def test_arm_and_si_contradictions_refuse(tmp_path):
    payload = _lr_only_payload()
    _summaries(tmp_path, payload)
    _analyze(tmp_path)
    fit = tmp_path / "r1_fit.json"
    argv = _argv(fit, tmp_path / "d")
    argv[argv.index("--arm") + 1] = "lora_r4_ce"
    with pytest.raises(SystemExit, match="not --arm"):
        hnr.main(argv)
    with pytest.raises(SystemExit, match="contradicts the artefact's data cell"):
        hnr.main(_argv(fit, tmp_path / "d", si="constv2-mix-1m"))
    with pytest.raises(SystemExit, match="no such artefact"):
        hnr.main(_argv(tmp_path / "nope.json", tmp_path / "d"))


def test_rank_deficient_artefact_stops_before_the_guard(tmp_path):
    payload = _lr_only_payload()
    for t in payload["trials"]:
        t["lr"] = 3e-4
    _summaries(tmp_path, payload)
    with pytest.raises(SystemExit, match="RANK-DEFICIENT"):
        _analyze(tmp_path)
    art = json.loads((tmp_path / "r1_fit.json").read_text())
    assert "selection_guard" not in art
    d = hnr.decide(art)
    assert d["decision"] == "stop" and d["reason"] == "fit_rank_deficient"


# --------------------------------------------------------------------------
# the emitted point is the checked point; alarms are recomputed
# --------------------------------------------------------------------------

def test_a_winner_moved_off_the_checked_point_is_refused(tmp_path):
    """Guard reproduction validates ``selection``; emission reads
    ``winner``. Editing only winner.lr to 1.0 (the checked point is inside
    the box, whose top is 4.5e-3) must not "reproduce" and emit a
    confirmation at an unchecked point."""
    payload = _lr_only_payload()
    _summaries(tmp_path, payload, rates={i: 0.03 for i in range(8)})
    art = _analyze(tmp_path)
    assert art["winner"]["lr"] == art["selection"]["lr"] == art["fit"]["argmin"]["lr"]
    fit = tmp_path / "r1_fit.json"
    for k, bad in (("lr", 1.0), ("max_grad_norm", 0.5), ("warmup_frac", 0.05)):
        art = json.loads(fit.read_text())
        assert art["winner"][k] != bad
        art["winner"][k] = bad
        fit.write_text(json.dumps(art))
        with pytest.raises(SystemExit, match="winner"):
            hnr.main(_argv(fit, tmp_path / "d"))
    # the fitted argmin moved under a consistent selection/winner pair
    art = json.loads(fit.read_text())
    art["winner"]["lr"] = art["selection"]["lr"]
    art["winner"]["max_grad_norm"] = art["selection"]["max_grad_norm"]
    art["winner"]["warmup_frac"] = art["selection"]["warmup_frac"]
    art["fit"]["argmin"]["lr"] = art["fit"]["argmin"]["lr"] * 2.0
    fit.write_text(json.dumps(art))
    with pytest.raises(SystemExit, match="argmin"):
        hnr.main(_argv(fit, tmp_path / "d"))
    assert not (tmp_path / "d").exists()


def test_a_flipped_alarm_flag_cannot_bypass_allow_alarmed(tmp_path):
    """The driver trusted the stored ``selection_guard.alarmed`` boolean:
    flipping it let an artefact still carrying the clip-activation alarm
    proceed to a confirmation without --allow-alarmed. Alarms are
    recomputed from the artefact's own rows — the clip-activation alarm
    included — and a recorded flag that disagrees is a tampered artefact."""
    payload = _lr_only_payload()
    _summaries(tmp_path, payload, rates={i: 0.9 for i in range(8)})
    art = _analyze(tmp_path)
    assert art["selection_guard"]["alarmed"]
    fit = tmp_path / "r1_fit.json"
    # (a) the flag alone
    art = json.loads(fit.read_text())
    art["selection_guard"]["alarmed"] = False
    fit.write_text(json.dumps(art))
    with pytest.raises(SystemExit, match="does NOT reproduce"):
        hnr.main(_argv(fit, tmp_path / "d"))
    # (b) the flag AND the clip-activation alarm deleted from the list
    art["selection_guard"]["alarms"] = [
        a for a in art["selection_guard"]["alarms"]
        if a["code"] != tune_arm.CLIP_ACTIVATION_ALARM]
    fit.write_text(json.dumps(art))
    with pytest.raises(SystemExit, match="does NOT reproduce"):
        hnr.main(_argv(fit, tmp_path / "d"))
    assert not (tmp_path / "d").exists()
    # the untampered artefact still reproduces, with the clip alarm in
    # BOTH the recorded and the recomputed sets
    _summaries(tmp_path / "ok", payload, rates={i: 0.9 for i in range(8)})
    art_ok = _analyze(tmp_path / "ok")
    g = hnr.reproduce_guard(art_ok)
    assert g["reproduced"] and g["alarmed"]
    assert tune_arm.CLIP_ACTIVATION_ALARM in g["recomputed_codes"]
    assert tune_arm.CLIP_ACTIVATION_ALARM in g["recorded_codes"]
