"""The HPO selection guard (scripts/hpo_selection_guard.py) — the standing
rule that a fold's selected point must be INTERIOR to its search, and that a
flat objective is written down next to the winner.

Why: the GLM-4.6V-Flash gated arm's pooled r1+2 quadratic was not
positive-definite, the fold fell through to the raw argmin trial
(max_grad_norm 89.93 — effectively unclipped) which "won" a flat objective
by a margin inside noise, nothing in the fold record said so, and the
full-scale fit diverged. These tests pin:

- a FLAT synthetic objective whose raw argmin sits on the box edge trips the
  guard on every criterion at once: non-PD fit, declared-box edge (a),
  sampled extreme (b) — and the record says "objective flat" in words;
- the GLM shape — a raw winner one step inside the sampled range whose only
  neighbour beyond is within noise — trips (c) no_evidence_beyond, while the
  SAME shape under a fitted argmin is filed, not alarmed;
- a normal interior positive-definite case does NOT trip;
- the discrete warmup axis: a 2-level factor never alarms, a >=3-level end
  does;
- tune_arm.analyze writes the guard into its artefact (PD and non-PD) and
  prints the ALARM lines; its _edge_check IS the guard's criterion (a);
- make_sdfbase_configs refuses an alarmed, unacknowledged guard at
  --fold-hpo time, accepts the acknowledged one, keeps the block out of the
  rendered train_kwargs, and the committed folds store is untouched.

CPU-only; no model, no network.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Registered under its real name FIRST, so tune_arm's ``import
# hpo_selection_guard`` binds the same module object the tests hold.
guard = _load_module("hpo_selection_guard",
                     REPO / "scripts/hpo_selection_guard.py")
tune_arm = _load_module("_guard_tune_arm", REPO / "scripts/tune_arm.py")
msc = _load_module("_guard_make_sdfbase_configs",
                   REPO / "scripts/sdfbase/make_sdfbase_configs.py")

Y = "best_heldout_loss"
LEVELS = (0.05, 0.15)
BOX = {"lr": [1e-5, 1e-1], "max_grad_norm": [0.02, 100.0]}


def _row(i, lr, clip, warm, y, rnd=1):
    return {"index": i, "round": rnd, "lr": lr, "max_grad_norm": clip,
            "warmup_frac": warm, Y: y}


def _codes(g):
    return sorted(guard.alarm_code(a) for a in g["alarms"])


def test_alarm_code_renders_the_dim_suffix_once_for_every_caller():
    assert guard.alarm_code({"code": "box_edge", "dim": "lr"}) == "box_edge:lr"
    assert guard.alarm_code({"code": "fit_not_positive_definite",
                             "dim": None}) == "fit_not_positive_definite"
    assert guard.alarm_code({"code": "x"}) == "x"


PD_FIT = {"positive_definite": True, "hessian_eigs": [0.001, 0.004]}
NONPD_FIT = {"positive_definite": False, "hessian_eigs": [-0.0013, 0.0068]}


# --------------------------------------------------------------------------
# Pure helpers
# --------------------------------------------------------------------------

def test_box_position_and_edge_rule_match_tune_arm():
    assert guard.box_position(1e-5, 1e-5, 1e-1) == pytest.approx(0.0)
    assert guard.box_position(1e-1, 1e-5, 1e-1) == pytest.approx(1.0)
    assert guard.box_position(1e-3, 1e-5, 1e-1) == pytest.approx(0.5)
    # tune_arm's historical _edge_check is the guard's criterion (a), with
    # ONE tolerance constant between them.
    assert tune_arm.EDGE_TOL == guard.EDGE_TOL == 0.02
    pt = {"lr": 1e-3, "max_grad_norm": 99.0}
    assert tune_arm._edge_check(pt, BOX) == guard.box_edge_positions(pt, BOX)
    on_edge, pos = guard.box_edge_positions(pt, BOX)
    assert on_edge and pos["max_grad_norm"] > 1 - guard.EDGE_TOL
    assert not guard.box_edge_positions({"lr": 1e-3, "max_grad_norm": 1.0},
                                        BOX)[0]


def test_residual_scale_is_dof_corrected_and_none_when_underdetermined():
    res = [0.01, -0.01, 0.02, -0.02]
    sc = guard.residual_scale(res, n_params=2)
    ssr = sum(x * x for x in res)
    assert sc["rms_log_y"] == pytest.approx(math.sqrt(ssr / 4))
    assert sc["sigma_log_y"] == pytest.approx(math.sqrt(ssr / (4 - 2)))
    assert sc["resid_n"] == 4 and sc["resid_p"] == 2
    assert sc["mad_log_y"] > 0
    assert guard.residual_scale(res, n_params=4)["sigma_log_y"] is None
    assert guard.residual_scale(res, n_params=9)["sigma_log_y"] is None


# --------------------------------------------------------------------------
# The flat, edge-argmin case: every criterion fires and the record says flat
# --------------------------------------------------------------------------

def _flat_rows():
    """16 trials over the box; the objective is FLAT (2.30 +- 0.002 of
    deterministic jitter) and its raw argmin is the trial sitting at the
    box's clip maximum — the GLM failure in caricature."""
    rows = []
    lrs = [1e-5 * 10 ** (4 * k / 15) for k in range(16)]
    clips = [0.02 * (100 / 0.02) ** (k / 15) for k in range(16)]
    for i in range(16):
        y = 2.30 + 0.002 * math.sin(3.0 * i + 1.0)
        rows.append(_row(i, lrs[(i * 7) % 16], clips[i],
                         LEVELS[i % 2], y))
    # Plant the argmin ON the clip edge (the last trial has clip == 100).
    rows[15][Y] = 2.2960
    return rows


def test_flat_objective_with_edge_argmin_trips_every_criterion():
    rows = _flat_rows()
    braw = min(rows, key=lambda r: r[Y])
    assert braw["max_grad_norm"] == pytest.approx(100.0)
    g = guard.guard_selection(
        selected=braw, selected_from="best_raw_trial", rows=rows, box=BOX,
        levels=LEVELS, fit=NONPD_FIT, y_key=Y, sigma_log_y=0.005,
        baseline=2.5, fall_through={"reason": "fit_not_positive_definite"})
    assert g["alarmed"]
    codes = _codes(g)
    assert "fit_not_positive_definite" in codes           # requirement 2
    assert "box_edge:max_grad_norm" in codes              # criterion (a)
    assert "sampled_extreme:max_grad_norm" in codes       # criterion (b)
    # (c) is silent on the side where NO trial lies beyond — (b) carries
    # that fact — and fires on every other side of a raw pick on a flat
    # objective (nothing anywhere is worse by more than noise).
    c_alarms = [(a["dim"], a["side"]) for a in g["alarms"]
                if a["code"] == "no_evidence_beyond"]
    assert ("max_grad_norm", "high") not in c_alarms
    assert sorted(c_alarms) == [("lr", "high"), ("lr", "low"),
                                ("max_grad_norm", "low")]
    assert len(g["alarms"]) == 6
    d = g["dims"]["max_grad_norm"]
    assert d["on_box_edge"] and d["box_pos"] == pytest.approx(1.0)
    assert d["sampled_extreme"] == "max"
    assert d["rank_ascending"] == d["n_sampled"] == 16
    assert d["evidence_beyond"]["high"]["n_beyond"] == 0
    assert d["evidence_beyond"]["high"]["has_evidence"] is False
    # lr is interior on both criteria.
    assert not g["dims"]["lr"]["on_box_edge"]
    assert g["dims"]["lr"]["sampled_extreme"] is None
    # Requirement 3: the spread is filed, and the FULL range is within noise
    # -> the record says so in words.
    sp = g["spread"]
    assert sp["n_ok"] == 16 and sp["min"] == pytest.approx(2.2960)
    assert sp["full_spread"] <= sp["noise"]["noise_y"]
    assert sp["flat"] and sp["flat_kind"] == "full"
    assert sp["text"].startswith("objective flat: full spread")
    assert "is within noise" in sp["text"]
    assert sp["ratio_min"] == pytest.approx(2.2960 / 2.5)
    assert sp["runner_up"]["margin_over_selected"] > 0
    assert sp["noise"]["sigma_log_y"] == 0.005
    assert sp["noise"]["noise_y"] == pytest.approx(0.005 * 2.2960)
    assert sp["within_noise"]["n"] == 16
    # The printable verdict names every alarm and the flatness.
    text = "\n".join(g["text"])
    assert "ALARM fit_not_positive_definite" in text
    assert "ALARM box_edge [max_grad_norm]" in text
    assert "ALARM sampled_extreme [max_grad_norm]" in text
    assert "FLAT: objective flat" in text
    assert "SELECTION GUARD: 6 alarm(s)" in text
    # Requirement 1: a discrete 2-level factor at its end is recorded, never
    # alarmed (no interior exists by design).
    w = g["dims"]["warmup_frac"]
    assert w["at_extreme"] and not w["alarmed"] and "2-level" in w["note"]
    # Nothing acknowledged yet -> every consumer must refuse it.
    assert guard.needs_acknowledgement(g)


def test_fitted_argmin_on_edge_fall_through_is_its_own_alarm():
    rows = _flat_rows()
    braw = min(rows, key=lambda r: r[Y])
    g = guard.guard_selection(
        selected=braw, selected_from="best_raw_trial", rows=rows, box=BOX,
        levels=LEVELS, fit=PD_FIT, y_key=Y, sigma_log_y=0.005,
        fall_through={"reason": "fitted_argmin_on_edge",
                      "fitted_argmin": {"lr": 3.0, "max_grad_norm": 500.0,
                                        "warmup_frac": 0.05},
                      "edge_pos": {"lr": 1.4, "max_grad_norm": 1.2}})
    assert "fitted_argmin_on_edge" in _codes(g)
    assert "fit_not_positive_definite" not in _codes(g)
    assert g["fall_through"]["reason"] == "fitted_argmin_on_edge"


# --------------------------------------------------------------------------
# The GLM shape: (c) no evidence beyond, raw vs fitted
# --------------------------------------------------------------------------

def _glm_shape_rows():
    """Seven clip values half a decade apart (lr jittered around 1e-3 so it
    is a real axis); the raw winner is the SECOND-highest clip and the only
    trial beyond it (the sampled max) is within noise of it; everything
    below is clearly worse."""
    clips = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    lrs = [8e-4, 9e-4, 1e-3, 1.1e-3, 1.2e-3, 1e-3, 9.5e-4]
    ys = [2.40, 2.39, 2.41, 2.40, 2.38, 2.300, 2.301]
    return [_row(i, lr, c, 0.05, y)
            for i, (lr, c, y) in enumerate(zip(lrs, clips, ys))]


GLM_BOX = {"lr": [1e-5, 1e-1], "max_grad_norm": [0.02, 1000.0]}


def test_raw_winner_one_step_inside_the_sampled_range_trips_no_evidence_beyond():
    rows = _glm_shape_rows()
    braw = min(rows, key=lambda r: r[Y])
    assert braw["max_grad_norm"] == 30.0
    g = guard.guard_selection(
        selected=braw, selected_from="best_raw_trial", rows=rows,
        box=GLM_BOX, levels=LEVELS, fit=NONPD_FIT, y_key=Y,
        sigma_log_y=0.004,  # noise ~0.0092: the 100-clip trial is within it
        fall_through={"reason": "fit_not_positive_definite"})
    d = g["dims"]["max_grad_norm"]
    # NOT the sampled max, NOT on the declared box edge...
    assert d["sampled_extreme"] is None and not d["on_box_edge"]
    assert d["rank_ascending"] == 6 and d["n_sampled"] == 7
    # ...but the one trial beyond it is within noise -> no evidence beyond.
    hi = d["evidence_beyond"]["high"]
    assert hi["n_beyond"] == 1 and hi["has_evidence"] is False
    assert hi["nearest_beyond"]["max_grad_norm"] == 100.0
    assert d["flat_to_sampled_edge"] == ["high"]
    assert "no_evidence_beyond:max_grad_norm" in _codes(g)
    # Below it there IS evidence (trials worse by > noise).
    assert d["evidence_beyond"]["low"]["has_evidence"] is True
    # lr is interior on every criterion — the only alarms are the fit and
    # the clip axis.
    assert g["dims"]["lr"]["sampled_extreme"] is None
    assert g["dims"]["lr"]["flat_to_sampled_edge"] == []
    assert _codes(g) == ["fit_not_positive_definite",
                         "no_evidence_beyond:max_grad_norm"]
    sp = g["spread"]
    assert sp["flat"] and sp["flat_kind"] == "near_optimum"
    assert sp["within_noise"]["n"] == 2
    assert "raw pick is arbitrary" in sp["text"]
    assert sp["runner_up"]["trial"] == "r1_t06"
    assert sp["runner_up"]["margin_over_selected"] == pytest.approx(0.001)
    # The best OTHER interior trial excludes the sampled-max runner-up (and
    # the sampled-extreme lr / clip rows): r1_t01, clip 0.3, y 2.39.
    assert sp["best_other_interior_trial"]["trial"] == "r1_t01"
    assert sp["best_other_interior_trial"]["margin_over_selected"] == \
        pytest.approx(0.09)


def test_same_shape_under_a_fitted_argmin_is_filed_not_alarmed():
    rows = _glm_shape_rows()
    sel = {"lr": 1e-3, "max_grad_norm": 30.0, "warmup_frac": 0.05}
    g = guard.guard_selection(
        selected=sel, selected_from="fitted_argmin", rows=rows, box=GLM_BOX,
        levels=LEVELS, fit=PD_FIT, y_key=Y, sigma_log_y=0.004,
        fitted_min_log_y=math.log(2.300))
    assert _codes(g) == [] and not g["alarmed"]
    assert g["dims"]["max_grad_norm"]["flat_to_sampled_edge"] == ["high"]
    assert g["spread"]["flat_to_sampled_edge"] == ["max_grad_norm:high"]
    assert "Flat out to the sampled edge on: max_grad_norm (high side)" \
        in g["spread"]["text"]
    assert "recorded, not alarmed" in g["spread"]["text"]
    assert g["selected_trial"] is None
    assert g["spread"]["fitted_min_below_best_trial"] == pytest.approx(0.0)


# --------------------------------------------------------------------------
# A normal interior positive-definite case does NOT trip
# --------------------------------------------------------------------------

def _parabola_rows():
    """A clean bowl centred at lr=1e-3, clip=1.0 (box centre), sampled on a
    4x4 log grid strictly inside the box, both warmup levels."""
    rows = []
    L0, C0 = math.log(1e-3), math.log(1.0)
    i = 0
    for a in (-1.2, -0.4, 0.4, 1.2):
        for b in (-1.2, -0.4, 0.4, 1.2):
            lr, clip = math.exp(L0 + a * 1.5), math.exp(C0 + b * 2.0)
            y = 2.0 * math.exp(0.05 * a * a + 0.05 * b * b)
            rows.append(_row(i, lr, clip, LEVELS[i % 2], y))
            i += 1
    return rows


def test_interior_pd_winner_is_clear():
    rows = _parabola_rows()
    sel = {"lr": 1e-3, "max_grad_norm": 1.0, "warmup_frac": 0.05}
    g = guard.guard_selection(
        selected=sel, selected_from="fitted_argmin", rows=rows, box=BOX,
        levels=LEVELS, fit=PD_FIT, y_key=Y, sigma_log_y=1e-4,
        fitted_min_log_y=math.log(2.0), baseline=2.5)
    assert not g["alarmed"] and g["alarms"] == []
    assert not guard.needs_acknowledgement(g)
    for k in ("lr", "max_grad_norm"):
        d = g["dims"][k]
        assert not d["on_box_edge"] and d["sampled_extreme"] is None
        assert d["flat_to_sampled_edge"] == []
        assert d["evidence_beyond"]["low"]["has_evidence"] is True
        assert d["evidence_beyond"]["high"]["has_evidence"] is True
        assert 0.3 < d["box_pos"] < 0.7
    sp = g["spread"]
    assert not sp["flat"] and sp["flat_kind"] is None
    assert sp["text"].startswith("objective resolved at the noise level")
    assert sp["within_noise"]["n"] == 0
    assert sp["ratio_min"] == pytest.approx(sp["min"] / 2.5)
    assert g["text"][-1].startswith("SELECTION GUARD: clear")
    # acknowledge() is a no-op on a clear record.
    assert guard.acknowledge(g, by="test")["acknowledged"] is None


def test_raw_selection_must_name_a_usable_row():
    rows = _parabola_rows()
    with pytest.raises(ValueError, match="usable rows"):
        guard.guard_selection(
            selected={"lr": 5e-3, "max_grad_norm": 5.0, "warmup_frac": 0.05},
            selected_from="best_raw_trial", rows=rows, box=BOX,
            levels=LEVELS, fit=NONPD_FIT, y_key=Y, sigma_log_y=0.01)
    with pytest.raises(ValueError, match="fitted_min_log_y"):
        guard.guard_selection(
            selected={"lr": 1e-3, "max_grad_norm": 1.0, "warmup_frac": 0.05},
            selected_from="fitted_argmin", rows=rows, box=BOX,
            levels=LEVELS, fit=PD_FIT, y_key=Y, sigma_log_y=0.01)


def test_noise_unavailable_still_files_the_spread():
    rows = _parabola_rows()
    sel = {"lr": 1e-3, "max_grad_norm": 1.0, "warmup_frac": 0.05}
    g = guard.guard_selection(
        selected=sel, selected_from="fitted_argmin", rows=rows, box=BOX,
        levels=LEVELS, fit=PD_FIT, y_key=Y, sigma_log_y=None,
        fitted_min_log_y=math.log(2.0))
    assert not g["alarmed"]
    assert g["spread"]["noise"]["noise_y"] is None
    assert g["spread"]["within_noise"] is None
    assert g["spread"]["text"].startswith("noise estimate unavailable")
    assert g["dims"]["lr"]["evidence_beyond"]["low"]["has_evidence"] is None


# --------------------------------------------------------------------------
# The discrete axis
# --------------------------------------------------------------------------

@pytest.mark.parametrize("levels, warm, expect_alarm, note_has", [
    ((0.05, 0.15), 0.05, False, "2-level"),
    ((0.05, 0.15), 0.15, False, "2-level"),
    ((0.05, 0.10, 0.15), 0.05, True, None),
    ((0.05, 0.10, 0.15), 0.15, True, None),
    ((0.05, 0.10, 0.15), 0.10, False, None),
])
def test_discrete_warmup_axis(levels, warm, expect_alarm, note_has):
    rows = _parabola_rows()
    for r in rows:
        r["warmup_frac"] = levels[0]
    rows[0]["warmup_frac"] = warm
    sel = {"lr": 1e-3, "max_grad_norm": 1.0, "warmup_frac": warm}
    g = guard.guard_selection(
        selected=sel, selected_from="fitted_argmin", rows=rows, box=BOX,
        levels=levels, fit=PD_FIT, y_key=Y, sigma_log_y=1e-4,
        fitted_min_log_y=math.log(2.0))
    w = g["dims"]["warmup_frac"]
    assert w["n_levels"] == len(levels)
    assert w["level_index"] == list(levels).index(warm)
    assert w["alarmed"] is expect_alarm
    assert ("discrete_extreme:warmup_frac" in _codes(g)) is expect_alarm
    if note_has:
        assert note_has in w["note"]


# --------------------------------------------------------------------------
# Acknowledgement
# --------------------------------------------------------------------------

def test_acknowledge_keeps_the_alarms_and_releases_the_refusal():
    rows = _flat_rows()
    braw = min(rows, key=lambda r: r[Y])
    g = guard.guard_selection(
        selected=braw, selected_from="best_raw_trial", rows=rows, box=BOX,
        levels=LEVELS, fit=NONPD_FIT, y_key=Y, sigma_log_y=0.005,
        fall_through={"reason": "fit_not_positive_definite"})
    before = list(g["alarms"])
    assert guard.needs_acknowledgement(g)
    guard.acknowledge(g, by="ws3_make_fold.py --allow-alarmed")
    assert g["alarms"] == before and g["alarmed"]   # the alarms survive
    assert g["acknowledged"]["by"] == "ws3_make_fold.py --allow-alarmed"
    assert g["acknowledged"]["at"]
    assert guard.is_acknowledged(g)
    assert not guard.needs_acknowledgement(g)
    assert not guard.needs_acknowledgement(None)
    json.dumps(g)  # the whole record is JSON-able


# --------------------------------------------------------------------------
# tune_arm.analyze: the producer files the guard and prints the ALARM lines
# --------------------------------------------------------------------------

A_BOX = {"lr": [3e-5, 3e-3], "max_grad_norm": [0.03, 3.0]}
_L0, _C0 = math.log(3e-4), math.log(0.3)


def _write_summaries(out: Path, payload: dict, log_y) -> None:
    for t in payload["trials"]:
        y = math.exp(log_y(t["lr"], t["max_grad_norm"], t["warmup_frac"]))
        s = {"trial": t, "round": payload["round"], "status": "ok",
             "arm": payload["arm"], "model": payload["model"],
             "position": payload["position"], "si": "si_constitution",
             "box": payload["box"], "warmup_levels": payload["warmup_levels"],
             Y: y, "heldout_curve": [2.5, y], "floor_ok": True}
        d = out / f"r{payload['round']}_t{t['index']:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(s))


def _analyze(out: Path):
    tune_arm.analyze(argparse.Namespace(out=str(out), rounds="1",
                                        y_key=Y))
    return json.loads((out / "r1_fit.json").read_text())


def test_analyze_files_a_clear_guard_for_an_interior_pd_optimum(tmp_path,
                                                                capsys):
    payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                    position="all", round_no=1, n=16,
                                    box=A_BOX)

    def bowl(lr, clip, warm):
        w = 0.0 if warm == 0.05 else 1.0
        return (2.0 * (math.log(lr) - _L0) ** 2
                + 1.5 * (math.log(clip) - _C0) ** 2 - 0.5 * w)

    _write_summaries(tmp_path, payload, bowl)
    art = _analyze(tmp_path)
    assert art["fit"]["positive_definite"]
    # The fit carries its residual statistics (the guard's noise).
    assert art["fit"]["sigma_log_y"] is not None
    assert len(art["fit"]["resid_log_y"]) == art["fit"]["n_rows"]
    assert art["fit"]["resid_p"] == len(art["fit"]["names"])
    g = art["selection_guard"]
    assert g["selected_from"] == "fitted_argmin"
    assert not g["alarmed"] and g["alarms"] == []
    assert art["selection"]["alarmed"] is False
    assert art["selection"]["alarms"] == []
    assert g["spread"]["baseline"] == pytest.approx(2.5)
    assert g["spread"]["ratio_min"] == pytest.approx(g["spread"]["min"] / 2.5)
    assert art["winner"] is not None
    out = capsys.readouterr().out
    assert "[analyze] SELECTION GUARD: clear" in out
    assert "ALARM" not in out


def test_analyze_files_an_alarmed_guard_and_refuses_on_a_non_pd_fit(tmp_path,
                                                                    capsys):
    payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                    position="all", round_no=1, n=16,
                                    box=A_BOX)

    def saddle(lr, clip, warm):
        # Positive curvature in lr, NEGATIVE in clip: a saddle, so the
        # continuous Hessian is not positive-definite.
        return (2.0 * (math.log(lr) - _L0) ** 2
                - 1.5 * (math.log(clip) - _C0) ** 2)

    _write_summaries(tmp_path, payload, saddle)
    with pytest.raises(SystemExit, match="not positive definite"):
        _analyze(tmp_path)
    art = json.loads((tmp_path / "r1_fit.json").read_text())
    assert not art["fit"]["positive_definite"]
    assert "selection" not in art and art.get("winner") is None
    g = art["selection_guard"]
    assert g["alarmed"]
    assert g["selected_from"] == "best_raw_trial"
    assert g["fall_through"]["reason"] == "fit_not_positive_definite"
    assert "fit_not_positive_definite" in _codes(g)
    # The raw argmin of a saddle in clip sits at a clip extreme: a boundary
    # hit the guard must also name.
    braw = art["best_raw_trial"]
    assert g["selected_trial"] == f"r1_t{braw['index']:02d}"
    assert g["dims"]["max_grad_norm"]["sampled_extreme"] in ("min", "max")
    assert any(c.startswith("sampled_extreme:max_grad_norm")
               for c in _codes(g))
    assert g["spread"]["noise"]["sigma_log_y"] == art["fit"]["sigma_log_y"]
    out = capsys.readouterr().out
    assert "[analyze] ALARM fit_not_positive_definite" in out
    assert "[analyze] SELECTION GUARD:" in out


# --------------------------------------------------------------------------
# make_sdfbase_configs --fold-hpo: refuse unacknowledged, accept acknowledged,
# keep the block out of the rendered config, committed store untouched
# --------------------------------------------------------------------------

def _guarded_fold(acknowledged: bool) -> dict:
    rows = _flat_rows()
    braw = min(rows, key=lambda r: r[Y])
    g = guard.guard_selection(
        selected=braw, selected_from="best_raw_trial", rows=rows, box=BOX,
        levels=LEVELS, fit=NONPD_FIT, y_key=Y, sigma_log_y=0.005,
        fall_through={"reason": "fit_not_positive_definite"})
    if acknowledged:
        guard.acknowledge(g, by="test --allow-alarmed")
    return {"glm46vflash_doc": {"psr_ce_all": {
        "lr": float(braw["lr"]), "warmup_frac": float(braw["warmup_frac"]),
        "max_grad_norm": float(braw["max_grad_norm"]),
        "selection_guard": g}}}


def test_fold_hpo_refuses_an_unacknowledged_alarmed_guard(tmp_path):
    fold = _guarded_fold(acknowledged=False)
    with pytest.raises(SystemExit, match="UNACKNOWLEDGED selection-guard"):
        msc.build_all(hpo=fold, folds_path=None)
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / msc.FOLDS_FILENAME
    with pytest.raises(SystemExit, match="UNACKNOWLEDGED selection-guard"):
        msc.update_folds("hpo", fold, path=path, repo=tmp_path)
    assert not path.exists()  # nothing written by the refused fold


def test_fold_hpo_accepts_an_acknowledged_guard_and_keeps_it_out_of_train_kwargs(
        tmp_path):
    fold = _guarded_fold(acknowledged=True)
    built = msc.build_all(hpo=fold, folds_path=None)
    cfg = msc.yaml.safe_load(built["sdfbase_glm46vflash_doc_fits.yaml"])
    tk = cfg["train_kwargs"]["psr_ce_all"]
    win = fold["glm46vflash_doc"]["psr_ce_all"]
    assert tk["lr"] == pytest.approx(win["lr"])
    assert tk["max_grad_norm"] == pytest.approx(win["max_grad_norm"])
    assert tk["warmup_frac"] == win["warmup_frac"]
    assert "selection_guard" not in tk          # inert for the rendered config
    assert "selection_guard" not in built["sdfbase_glm46vflash_doc_fits.yaml"]
    # ...and the block travels INTO the folds store with the numbers.
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / msc.FOLDS_FILENAME
    msc.update_folds("hpo", fold, path=path, repo=tmp_path)
    stored = msc.load_folds(path)["hpo"]["glm46vflash_doc"]["psr_ce_all"]
    assert stored["selection_guard"]["alarmed"]
    assert stored["selection_guard"]["acknowledged"]["by"] == \
        "test --allow-alarmed"
    assert [a["code"] for a in stored["selection_guard"]["alarms"]] == \
        [a["code"] for a in win["selection_guard"]["alarms"]]
    assert msc.build_all(folds_path=path)[
        "sdfbase_glm46vflash_doc_fits.yaml"] == \
        built["sdfbase_glm46vflash_doc_fits.yaml"]


def test_clear_guard_folds_without_ceremony():
    rows = _parabola_rows()
    g = guard.guard_selection(
        selected={"lr": 1e-3, "max_grad_norm": 1.0, "warmup_frac": 0.05},
        selected_from="fitted_argmin", rows=rows, box=BOX, levels=LEVELS,
        fit=PD_FIT, y_key=Y, sigma_log_y=1e-4, fitted_min_log_y=math.log(2.0))
    fold = {"qwen35_4b_base": {"psr_ce_all": {
        "lr": 1e-3, "warmup_frac": 0.05, "max_grad_norm": 1.0,
        "selection_guard": g}}}
    built = msc.build_all(hpo=fold, folds_path=None)
    tk = msc.yaml.safe_load(built["sdfbase_qwen35_4b_base_fits.yaml"])[
        "train_kwargs"]["psr_ce_all"]
    assert tk["lr"] == pytest.approx(1e-3) and "selection_guard" not in tk


def test_committed_folds_store_guard_blocks_are_absent_or_well_formed():
    """State-aware (the same shape one more time — the previous version
    pinned "no committed entry carries a selection_guard", true while the
    store predated the guard and red by construction the moment the first
    guarded winner folded, which made the fold launcher refuse at its own
    test gate). A committed hpo entry is in one of two states:
      * pre-guard: no selection_guard (the keys folded before the guard
        existed; test_sdfbase_configs.py pins their bytes), or
      * guarded: it carries the record the fold writer stores beside a
        CONFIRMED fitted argmin (the two Llama keys since the 2026-09-10
        fold). Then the load-time contract holds — alarmed is a bool,
        acknowledged is None or {by, at}, never alarmed-and-unacknowledged
        (load_folds refuses that) — and, because the store only receives a
        winner a round-9 trial confirmed, a `confirmation` block is present
        whose checks all passed and whose design / live fit shas are
        recorded; the folded lr / clip / warmup are the TRIAL's values
        verbatim and agree with the guard's `selected` (the live artefact's
        argmin) within relative 1e-9 (a rewritten-but-equivalent artefact
        differs at ~1e-15; a different fit would not pass confirm-check).
    The Llama keys are folded together or not at all (the reuse licence)."""
    folds = msc.load_folds()
    guarded = set()
    for key, arms in folds["hpo"].items():
        for arm, val in arms.items():
            g = val.get("selection_guard")
            if g is None:
                assert "selection_guard" not in val, (key, arm)
                continue
            if str(val.get("fit_search_id") or "").startswith("hpo-native:"):
                # the NATIVE search's rows (a fit lane whose fit_search_id is hpo-native:…): the guard is the closer's record of the selected
                # trial (the artefact that selected it, hashed; the fitted object's fingerprint) — the user's pick carries its own
                # provenance on top (basis user_pick, the ruling, the closed search, the menu identity the authorization read); the
                # searched row carries the closer's close. Pinned per row by tests/test_models_demo_front_door.py; here the contract.
                assert g.get("selected_trial") and g.get("artefact_sha256") and g.get("fit_fingerprint") and g.get("selected_from"), (key, arm)
                if val.get("fit_lr_selected_by") == "manual":
                    assert g["basis"] == "user_pick" and g["ruling_id"] == val["architecture_selection_ruling_id"] \
                        and g["search_id"] == val["fit_search_id"] and g["search_closed"] is True and g.get("menu_sha256"), (key, arm)
                else:
                    assert val.get("fit_lr_selected_by") == "search" and g.get("search_closed_utc"), (key, arm)
                continue
            guarded.add((key, arm))
            assert isinstance(g.get("alarmed"), bool), (key, arm)
            ack = g.get("acknowledged")
            assert ack is None or (isinstance(ack, dict) and {"by", "at"} <= set(ack)), (key, arm)
            assert not (g["alarmed"] and not ack), (key, arm)
            conf = g.get("confirmation")
            assert isinstance(conf, dict) and conf.get("checks"), (key, arm)
            assert all(c["pass"] for c in conf["checks"]), (key, arm)
            assert len(conf["fit_sha256"]) == 64 and len(conf["design_fit_sha256"]) == 64, (key, arm)
            for k in ("lr", "max_grad_norm", "warmup_frac"):
                assert isinstance(val[k], float) and val[k] > 0, (key, arm, k)
                assert math.isclose(val[k], g["selected"][k], rel_tol=1e-9, abs_tol=0.0), \
                    (key, arm, k, val[k], g["selected"][k])
    llama = {key for key, _ in guarded if key.startswith("llama8b")}
    assert llama in (set(), {"llama8b_base", "llama8b_it_doc"}), llama
    if llama:
        assert {arm for key, arm in guarded if key in llama} == set(msc.CE_ARMS)
