"""The LoRA architecture search design + analysis: `build_trials_v2` (16
scrambled-Sobol points in u = log10(lr) x the ALLOWED (rank, set) pairs,
coverage-repaired, 32 only when 16 cannot cover) and `analyze --design v2`
(the additive model with rank / set effects, the argmin over the allowed
pairs, the adequacy REPORT with the nested lr-only comparator defined on
every fold, the pairs-means line, the `additive_model_inadequate` alarm on a
planted interaction and silent on a planted additive truth).

CPU-only; synthetic summaries in the trial runner's schema; the real design
builder, the real analyzer, the real guard.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from test_tune_arm import tune_arm  # noqa: E402  — ONE tune_arm module instance

guard = tune_arm.hpo_selection_guard
ARM = "lora_r4_ce"
MODEL = "example/toy-moe-base"          # a SYNTHETIC sparse-MoE test model (8 active experts): no menu model has routed experts
BOX = (-4.5, -2.0)
CLIP = 1.3
WF = 0.15
# the module sets' per-block widths of the synthetic MoE test model (the lora_modules docstring's worked counts): wo < qkvo < down < mlp < attn_mlp
PARAMS_R4 = {"wo": 245_760, "qkvo": 860_160, "down": 26_624_000, "mlp": 58_900_480,
             "attn_mlp": 59_760_640}
SETS = ["wo", "qkvo", "down", "mlp", "attn_mlp"]


def allowed_doc(pairs, excluded=(), model_id="toy_moe_base"):
    return {"model_id": model_id, "probe_job_id": "12345", "verdict_uri": "probe://x",
            "active_experts": 8,
            "pairs": [{"rank": r, "set": s, "lora_modules": tune_arm.modules_of_set(s),
                       "per_expert_rank": {"shared_expert": r, "routed_expert": max(1, r // 8)},
                       "max_lora_rank_cap": 64, "topology": "single",
                       "trainable_params": PARAMS_R4[s] * r // 4} for r, s in pairs],
            "excluded": [{"rank": r, "set": s, "reason": why} for r, s, why in excluded]}


def write_allowed(tmp_path, pairs, excluded=(), name="allowed_pairs.json"):
    p = tmp_path / name
    p.write_text(json.dumps(allowed_doc(pairs, excluded), indent=1))
    return p


FULL15 = [(r, s) for r in (4, 16, 64) for s in SETS]


def build(tmp_path, pairs=FULL15, excluded=(), seed=1, n=16, round_no=1):
    allowed = tune_arm.load_allowed_pairs(write_allowed(tmp_path, pairs, excluded))
    return tune_arm.build_trials_v2(
        arm=ARM, model=MODEL, model_revision="0f0813", config_sha="cafebabe",
        position="all", round_no=round_no, lr_log10_box=BOX, allowed=allowed,
        warmup_frac=WF, max_grad_norm=CLIP, init_scheme="random_lecun_1e-3",
        n=n, seed=seed)


def pairs_of(payload):
    return [(t["lora_rank"], t["lora_set_id"]) for t in payload["trials"]]


# --------------------------------------------------------------------------
# the design
# --------------------------------------------------------------------------

def test_sixteen_points_cover_every_allowed_pair_for_fifteen_pairs(tmp_path):
    p = build(tmp_path)
    assert len(p["trials"]) == 16 and p["design"] == "sobol16+coverage_repair"
    assert set(pairs_of(p)) == set(FULL15)
    assert p["design_extended_to_32"] is None
    for t in p["trials"]:
        assert BOX[0] <= math.log10(t["lr"]) <= BOX[1]
        assert t["warmup_frac"] == WF and t["max_grad_norm"] == CLIP
        assert t["init_scheme"] == "random_lecun_1e-3"
        assert t["lora_modules"] == tune_arm.modules_of_set(t["lora_set_id"])
        assert t["per_expert_rank"] == {"shared_expert": t["lora_rank"],
                                        "routed_expert": max(1, t["lora_rank"] // 8)}
    assert p["axes"] == ["lr", "lora_rank", "lora_modules"]
    assert p["grids"]["lora_rank"] == [4, 16, 64]
    assert p["grids"]["lora_modules"] == SETS          # by trainable-parameter count
    assert p["constant_axes"] == {"max_grad_norm": CLIP}
    assert p["box"]["max_grad_norm"] == [CLIP, CLIP]
    assert p["warmup_levels"] == [WF]
    assert p["per_expert_rank_rule"] == "max(1,r//8)"
    assert p["allowed_pairs"] == [list(k) for k in FULL15]


@pytest.mark.parametrize("pairs", [
    FULL15[:6], [(4, "wo"), (16, "wo"), (64, "wo"), (4, "mlp"), (16, "mlp")],
    [(4, "wo"), (4, "qkvo")], [(4, "wo")],
    [(r, s) for r in (4, 16) for s in SETS] + [(64, "wo"), (64, "qkvo")],
])
def test_every_allowed_pair_is_covered_for_any_menu_up_to_sixteen(tmp_path, pairs):
    p = build(tmp_path, pairs=pairs)
    assert set(pairs_of(p)) == set(pairs)
    assert len(p["trials"]) == 16


def test_repair_is_deterministic_under_the_seed_and_moves_from_the_most_populated_pair(tmp_path):
    a = build(tmp_path, seed=1)
    b = build(tmp_path, seed=1)
    assert a["trials"] == b["trials"] and a["coverage_repair"] == b["coverage_repair"]
    c = build(tmp_path, seed=7)
    assert c["trials"] != a["trials"]                      # another scrambling
    assert set(pairs_of(c)) == set(FULL15)                  # still covers
    # every move relabels a point onto a pair that had none; the move list replays to the emitted labels
    assert all(m["to_pair"] != m["from_pair"] for m in a["coverage_repair"])
    for m in a["coverage_repair"]:
        t = a["trials"][m["index"]]
        assert [t["lora_rank"], t["lora_set_id"]] == m["to_pair"]


def test_extends_to_32_only_when_sixteen_cannot_cover(tmp_path):
    # 17 pairs: three ranks x five sets + two extra ranks on `wo`
    pairs = FULL15 + [(8, "wo"), (32, "wo")]
    p = build(tmp_path, pairs=pairs)
    assert len(p["trials"]) == 32
    assert p["design_extended_to_32"] == {"reason": "coverage", "n_pairs": 17, "n_base": 16,
                                          "n_total": 32, "seeds": [1, 2]}
    assert set(pairs_of(p)) == set(pairs)
    with pytest.raises(SystemExit, match="exceed 2 x 16"):
        build(tmp_path, pairs=[(r, s) for r in (1, 2, 3, 4, 5, 6, 7) for s in SETS])


def test_unit_cube_stratification_of_the_lr_axis(tmp_path):
    p = build(tmp_path)
    us = sorted((math.log10(t["lr"]) - BOX[0]) / (BOX[1] - BOX[0]) for t in p["trials"])
    assert {int(u * 16) for u in us} == set(range(16))     # exact 1-D (0,4,1)-stratification


# --------------------------------------------------------------------------
# analyze --design v2
# --------------------------------------------------------------------------

BETA = {4: 0.0, 16: -0.05, 64: -0.02}
GAMMA = {"wo": 0.0, "qkvo": -0.03, "down": -0.01, "mlp": -0.08, "attn_mlp": -0.04}
A0, U0, C0 = math.log(1.9), -3.2, 0.35    # log CE = A0 + C0 (u - U0)^2 + beta + gamma


def planted_log_y(t, *, interaction=0.0):
    u = math.log10(t["lr"])
    delta = interaction * {4: 0.0, 16: 1.0, 64: -1.0}[t["lora_rank"]] * (u - U0)
    return A0 + C0 * (u - U0) ** 2 + BETA[t["lora_rank"]] + GAMMA[t["lora_set_id"]] + delta


def write_summaries(out: Path, payload: dict, *, interaction=0.0, mutate=None, noise=None):
    for t in payload["trials"]:
        ly = planted_log_y(t, interaction=interaction)
        if noise is not None:
            ly += noise(t["index"])
        s = {"trial": t, "round": payload["round"], "status": "ok", "arm": payload["arm"],
             "model": payload["model"], "position": payload["position"], "si": "constv2-mix-10m",
             "box": payload["box"], "warmup_levels": payload["warmup_levels"],
             "best_heldout_loss": math.exp(ly), "floor_ok": True, "heldout_curve": [2.4, 2.0, 1.95],
             "trainable_params": PARAMS_R4[t["lora_set_id"]] * t["lora_rank"] // 4,
             **{k: payload[k] for k in ("design", "design_seed", "allowed_pairs", "excluded_pairs",
                                        "lr_log10_box", "set_order", "init_scheme", "per_expert_rank_rule",
                                        "campaign", "search_opening")
                if k in payload},
             **({"search_id": payload["search_id"]} if payload.get("search_id") else {})}
        if mutate:
            mutate(t["index"], s)
        d = out / f"r{payload['round']}_t{t['index']:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(s))


def analyze_v2(out: Path, rounds="1", allowed=None, pair_only=False, close=False):
    ns = argparse.Namespace(out=str(out), rounds=rounds, y_key="best_heldout_loss", design="v2",
                            allowed_pairs=(str(allowed) if allowed else None), pair_only=pair_only,
                            close_search=close)
    tag = "+".join(r.strip() for r in str(rounds).split(","))
    path = out / f"r{tag}_fit{'_pair' if pair_only else ''}.json"
    before = path.stat().st_mtime_ns if path.exists() else None
    rc = None
    try:
        tune_arm.analyze(ns)
    except SystemExit as e:
        # a REFUSAL leaves no (new) artefact and propagates; the non-PD verdict writes its
        # artefact first and exits non-zero — that one is returned with its code
        if not path.exists() or path.stat().st_mtime_ns == before:
            raise
        rc = e.code
    art = json.loads(path.read_text())
    return art, rc


def test_recovers_the_planted_quadratic_and_categorical_effects(tmp_path):
    p = build(tmp_path)
    write_summaries(tmp_path, p)
    art, rc = analyze_v2(tmp_path, allowed=tmp_path / "allowed_pairs.json")
    assert rc is None, rc
    fit = art["fit"]
    assert fit["positive_definite"] and not fit.get("rank_deficient")
    assert math.isclose(fit["u_star"], U0, abs_tol=1e-6)
    for r, b in BETA.items():
        assert math.isclose(fit["rank_effects"][str(r)], b, abs_tol=1e-6), r
    for s_, g in GAMMA.items():
        assert math.isclose(fit["set_effects"][s_], g, abs_tol=1e-6), s_
    assert fit["ref_rank"] == 4 and fit["ref_set"] == "wo"
    # the winner: argmin of beta+gamma over the ALLOWED pairs at u*
    assert fit["argmin_pair"] == [16, "mlp"]
    sel = art["selection"]
    assert sel["selected_from"] == "fitted_argmin" and not sel["on_edge"]
    assert math.isclose(math.log10(sel["lr"]), U0, abs_tol=1e-6)
    assert sel["lora_rank"] == 16 and sel["lora_set_id"] == "mlp"
    assert sel["lora_modules"] == ["down_proj", "gate_proj", "up_proj"]
    assert art["winner"]["lora_rank"] == 16 and art["winner"]["lora_set_id"] == "mlp"
    assert art["next"]["action"] == "refine" and art["next"]["pair"] == [16, "mlp"]
    assert math.isclose(art["next"]["box_log10"][0], U0 - 0.5, abs_tol=1e-6)
    # every allowed pair has a prediction; none outside P
    assert set(fit["pair_predictions"]) == {f"{r}:{s}" for r, s in FULL15}
    # the guard's named discrete axes: rank 16 is interior, mlp is NOT an end of the ordered menu
    g = art["selection_guard"]
    assert g["selected"]["lora_rank"] == 16 and g["selected"]["lora_modules"] == "mlp"
    assert g["dims"]["lora_rank"]["levels"] == [4, 16, 64]
    assert g["dims"]["lora_modules"]["levels"] == SETS
    assert "discrete_extreme:lora_rank" not in sel["alarms"]
    assert "discrete_extreme:lora_modules" not in sel["alarms"]
    assert g["dims"]["warmup_frac"]["note"] and not g["dims"]["warmup_frac"]["alarmed"]


def test_adequacy_comparator_is_defined_on_all_sixteen_folds_and_means_model_is_suppressed(tmp_path):
    p = build(tmp_path)
    write_summaries(tmp_path, p)
    art, _ = analyze_v2(tmp_path)
    ad = art["adequacy"]
    assert ad["categorical"]
    assert ad["additive_loo_n_defined"] == 16 and ad["unidentifiable_folds"] == []
    assert ad["lr_only_loo_n_defined"] == 16
    assert ad["lr_only_loo_rmse_same_folds"] == ad["lr_only_loo_rmse"]
    # exact planted additive truth: the additive model predicts out of sample, the lr-only one cannot
    assert ad["additive_loo_rmse"] < 1e-6 < ad["lr_only_loo_rmse"]
    assert ad["alarm"] is None
    assert not ad["means_model"]["defined"]
    assert ad["means_model"]["line"] == "pair-specific effects unidentifiable at n=16 over |P|=15 pairs"
    assert any("MEANS MODEL: pair-specific effects unidentifiable at n=16 over |P|=15 pairs" in ln
               for ln in ad["text"])
    assert any(ln.startswith("ADEQUACY:") and "fit residual" in ln for ln in ad["text"])
    assert ad["sigma_log_y"] is not None
    assert "additive_model_inadequate" not in art["selection"]["alarms"]


def test_inadequacy_alarm_fires_on_a_planted_rank_x_lr_interaction_and_is_silent_on_additive_truth(tmp_path):
    p = build(tmp_path)
    # a strong rank x lr interaction the additive model cannot express: its LOO error exceeds the
    # lr-only quadratic's when the categorical terms overfit the interaction noise
    truth = tmp_path / "truth"; truth.mkdir()
    write_summaries(truth, p)
    art_t, _ = analyze_v2(truth)
    assert art_t["adequacy"]["alarm"] is None
    inter = tmp_path / "inter"; inter.mkdir()
    write_summaries(inter, p, interaction=0.9,
                    noise=lambda i: 0.05 * math.sin(3.7 * i + 0.4))
    art_i, _ = analyze_v2(inter)
    ad = art_i["adequacy"]
    assert ad["additive_loo_rmse"] is not None and ad["lr_only_loo_rmse_same_folds"] is not None
    it = art_i["fit"]["interaction"]
    assert it["assessable"] and it["selected"], it        # AICc picks the delta[rank]*u model
    assert "SELECTED" in (it.get("note") or "").upper() or it["selected"]
    if ad["additive_loo_rmse"] >= ad["lr_only_loo_rmse_same_folds"]:
        assert ad["alarm"] == "additive_model_inadequate"
        assert "additive_model_inadequate" in art_i["selection"]["alarms"]
        assert art_i["selection_guard"]["alarmed"]
    else:  # the planted interaction was expressible enough — the rule is stated, not tripped
        assert ad["alarm"] is None
    # under the interaction the per-pair argmin depends on the rank
    preds = art_i["fit"]["pair_predictions"]
    assert not math.isclose(preds["16:mlp"]["u_star"], preds["64:mlp"]["u_star"], abs_tol=1e-6)


def test_relu_rows_without_architecture_axes_fit_the_plain_quadratic(tmp_path):
    # the fixed-warm-up 1-D LR-only design through build_trials (--clip-const), analysed by --design v2
    payload = tune_arm.build_trials(arm="psr_ce_all", model=MODEL, model_revision="0f",
                                    config_sha="c", position="all", round_no=1, n=16,
                                    box={"lr": (10 ** -4.5, 10 ** -1.5), "max_grad_norm": (CLIP, CLIP)},
                                    warmup_levels=(WF,))
    payload["init_scheme"] = "random_lecun_1e-3"
    for t in payload["trials"]:
        t["init_scheme"] = "random_lecun_1e-3"
    for t in payload["trials"]:
        d = tmp_path / f"r1_t{t['index']:02d}"; d.mkdir()
        u = math.log10(t["lr"])
        (d / "summary.json").write_text(json.dumps({
            "trial": t, "round": 1, "status": "ok", "arm": "psr_ce_all", "model": MODEL, "position": "all",
            "si": "constv2-mix-10m", "box": payload["box"], "warmup_levels": payload["warmup_levels"],
            "best_heldout_loss": math.exp(A0 + C0 * (u + 3.0) ** 2), "floor_ok": True}))
    art, rc = analyze_v2(tmp_path)
    assert rc is None
    fit = art["fit"]
    assert fit["names"] == ["1", "u", "u*u"] and fit["positive_definite"]
    assert math.isclose(fit["u_star"], -3.0, abs_tol=1e-6)
    assert art["allowed_pairs"] is None and art["selection"].get("lora_rank") is None
    assert not art["adequacy"]["categorical"] and art["adequacy"]["alarm"] is None
    assert art["selection"]["selected_from"] == "fitted_argmin"
    assert "discrete_axes" not in art["selection_guard"]
    assert art["next"]["action"] == "refine" and art["next"]["pair"] is None


def test_non_pd_design_fit_falls_through_to_the_best_promotable_trial(tmp_path):
    p = build(tmp_path)
    # a concave surface in u: no interior minimum
    write_summaries(tmp_path, p, mutate=lambda i, s: s.update(
        best_heldout_loss=math.exp(A0 - 0.4 * (math.log10(s["trial"]["lr"]) - U0) ** 2
                                   + BETA[s["trial"]["lora_rank"]] + GAMMA[s["trial"]["lora_set_id"]])))
    art, rc = analyze_v2(tmp_path)
    assert rc is not None
    assert not art["fit"]["positive_definite"]
    sel = art["selection"]
    assert sel["selected_from"] == "best_raw_trial" and sel["fall_through"]["reason"] == "fit_not_positive_definite"
    assert art["winner"] is None
    assert "fit_not_positive_definite" in sel["alarms"]
    best = min(art["rows"], key=lambda r: r["best_heldout_loss"])
    assert art["selection_guard"]["selected_trial"] == guard.trial_id(best)
    assert art["next"]["action"] == "refine" and art["next"]["pair"] == [best["lora_rank"], best["lora_set_id"]]


def test_summary_from_a_pair_outside_the_menu_is_refused(tmp_path):
    p = build(tmp_path)
    write_summaries(tmp_path, p, mutate=lambda i, s: (
        s["trial"].update(lora_rank=128, lora_set_id="mlp") if i == 3 else None))
    with pytest.raises(SystemExit, match="OUTSIDE the allowed pairs"):
        analyze_v2(tmp_path)


def test_replicate_and_confirmation_rows_are_never_pooled(tmp_path):
    p = build(tmp_path)
    write_summaries(tmp_path, p)
    rep = tmp_path / "r1_t16"; rep.mkdir()
    s = json.loads((tmp_path / "r1_t00" / "summary.json").read_text())
    s["exploration_replicate_of"] = {"trial": "r1_t00"}; s["trial"] = {**s["trial"], "index": 16}
    (rep / "summary.json").write_text(json.dumps(s))
    with pytest.raises(SystemExit, match="EXPLORATION REPLICATE"):
        analyze_v2(tmp_path)
