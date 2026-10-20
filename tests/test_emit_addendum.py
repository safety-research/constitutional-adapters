"""ONE trial per (rank, set) pair a re-verdict certifies AFTER the round-1 design — `analyze --design v2
--extends-menu` refits the round-1 rows over the WIDENED menu and predicts the new pairs; `emit-trials --addendum-from <that refit>
--allowed-pairs <widened menu> --pairs rank:set,…` writes a round-4 design (trial_role addendum) at each pair's predicted lr, else the
set's best round-1 trial lr, with provenance; `03_hpo.sh --phase addendum --pairs …` is the explicit front door (never part of --phase all).
Synthetic menus and planted summaries (tests/test_hpo_design_v2.py's factories); no model, no network."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPRO, run_stage  # noqa: E402
from test_front_door_recon_cell import _site  # noqa: E402
from test_hpo_design_v2 import (A0, ARM, BETA, GAMMA, MODEL, U0, analyze_v2, build, write_allowed,  # noqa: E402
                                write_summaries)
from test_tune_arm import tune_arm  # noqa: E402

DRAWN9 = [(4, "wo"), (16, "wo"), (64, "wo"), (4, "qkvo"), (16, "qkvo"), (64, "qkvo"), (4, "attn_mlp"), (4, "mlp"), (4, "down")]
NEW6 = [(16, "attn_mlp"), (64, "attn_mlp"), (16, "mlp"), (64, "mlp"), (16, "down"), (64, "down")]
WIDE15 = DRAWN9 + NEW6
PAIRS_ARG = ",".join(f"{r}:{s}" for r, s in NEW6)


def _round1(tmp_path, mutate=None):
    """A round-1 design over the 9-pair menu with planted summaries, analyzed over its own menu (the artefact the pick is made from)."""
    p = build(tmp_path, pairs=DRAWN9)
    write_summaries(tmp_path, p, mutate=mutate)
    (tmp_path / "menu9.json").write_bytes((tmp_path / "allowed_pairs.json").read_bytes())
    art, rc = analyze_v2(tmp_path, allowed=tmp_path / "menu9.json")
    return p, art, rc


def _widen(tmp_path):
    return write_allowed(tmp_path, WIDE15, name="allowed_pairs_wide.json")


def _emit(tmp_path, fit: Path, menu: Path, pairs: str, out: Path, **over):
    ns = argparse.Namespace(**{**tune_arm.EMIT_TRIALS_DEFAULTS, "out": str(out), "arm": ARM, "config": str(tmp_path / "fits.yaml"),
                               "position": "all", "si": "constv2-mix-10m", "warmup_levels": "0.15",
                               "addendum_from": str(fit), "allowed_pairs": str(menu), "pairs": pairs, **over})
    (tmp_path / "fits.yaml").write_text(f"model: {MODEL}\nmodel_revision: '0f0813'\n")
    tune_arm.emit_trials(ns)
    return json.loads(out.read_text())


def test_extends_menu_refits_over_the_widened_menu_and_predicts_the_new_pairs(tmp_path):
    _, art9, rc = _round1(tmp_path)
    assert rc is None and art9["fit"]["positive_definite"]
    wide = _widen(tmp_path)
    with pytest.raises(SystemExit, match="one search, one design space.*--extends-menu"):
        analyze_v2(tmp_path, allowed=wide)                                 # the standing guard, now with the hint
    ns = argparse.Namespace(out=str(tmp_path), rounds="1", y_key="best_heldout_loss", design="v2", allowed_pairs=str(wide),
                            pair_only=False, close_search=False, extends_menu=True)
    tune_arm.analyze(ns)
    art = json.loads((tmp_path / "r1_fit.json").read_text())
    assert sorted(map(tuple, art["menu_extended_from"])) == sorted(DRAWN9) and sorted(map(tuple, art["menu_extension"])) == sorted(NEW6)
    assert sorted(map(tuple, art["allowed_pairs"])) == sorted(WIDE15)
    preds = art["fit"]["pair_predictions"]
    for r, s in NEW6:
        pr = preds[f"{r}:{s}"]; assert pr["lr"] and math.isfinite(pr["lr"]) and pr["n_rows"] == 0, pr
    # the fit itself is the same rows, so the coefficients match the 9-pair artefact
    assert art["fit"]["coef_by_name"] == art9["fit"]["coef_by_name"]
    # a menu that is NOT a superset still refuses, flag or not
    narrow = write_allowed(tmp_path, [(4, "wo"), (16, "wo")], name="narrow.json")
    with pytest.raises(SystemExit, match="not a superset either"):
        tune_arm.analyze(argparse.Namespace(**{**vars(ns), "allowed_pairs": str(narrow)}))


def _refit(tmp_path):
    wide = _widen(tmp_path)
    tune_arm.analyze(argparse.Namespace(out=str(tmp_path), rounds="1", y_key="best_heldout_loss", design="v2", allowed_pairs=str(wide),
                                        pair_only=False, close_search=False, extends_menu=True))
    fit = tmp_path / "r1_fit_menu.json"; shutil.move(tmp_path / "r1_fit.json", fit)
    return wide, fit


def test_addendum_emits_one_trial_per_new_pair_at_the_predicted_lr_with_provenance(tmp_path):
    _round1(tmp_path)
    wide, fit = _refit(tmp_path)
    art = json.loads(fit.read_text())
    payload = _emit(tmp_path, fit, wide, PAIRS_ARG, tmp_path / "r4_trials.json")
    assert payload["round"] == tune_arm.ADDENDUM_ROUND == 4 and payload["design"] == tune_arm.ADDENDUM_DESIGN
    assert [(t["lora_rank"], t["lora_set_id"]) for t in payload["trials"]] == NEW6 and [t["index"] for t in payload["trials"]] == list(range(6))
    for t in payload["trials"]:
        pred = art["fit"]["pair_predictions"][f"{t['lora_rank']}:{t['lora_set_id']}"]
        assert t["lr"] == pytest.approx(pred["lr"]) and t["lr_source"]["kind"] == "pair_prediction"
        assert t["trial_role"] == "addendum" and t["warmup_frac"] == 0.15 and t["max_grad_norm"] == art["fixed_axes"]["max_grad_norm"]
        assert t["per_expert_rank"] == {"shared_expert": t["lora_rank"], "routed_expert": max(1, t["lora_rank"] // 8)}
        assert t["init_scheme"] == "random_lecun_1e-3" and sorted(t["lora_modules"]) == tune_arm.modules_of_set(t["lora_set_id"])
    assert payload.get("search_id") == art.get("search_id") and payload["addendum_of"]["search_id"] == art.get("search_id")
    assert payload["addendum_of"]["fit_sha256"] and payload["addendum_of"]["fit_positive_definite"] is True
    assert sorted(map(tuple, payload["menu_extends"])) == sorted(DRAWN9) and sorted(map(tuple, payload["allowed_pairs"])) == sorted(WIDE15)
    assert payload["allowed_pairs_source"]["sha256"] and payload["allowed_pairs_source"]["probe_job_id"] == "12345"
    assert payload["constant_axes"] == {"max_grad_norm": art["fixed_axes"]["max_grad_norm"]} and payload["si"] == "constv2-mix-10m"


def test_addendum_refuses_a_covered_pair_an_outside_pair_and_an_unrefit_artefact(tmp_path):
    _round1(tmp_path)
    wide, fit = _refit(tmp_path)
    with pytest.raises(SystemExit, match="already has round-1 trial"):
        _emit(tmp_path, fit, wide, "4:wo", tmp_path / "x.json")
    with pytest.raises(SystemExit, match="EXCLUDED by the serving probe|is not one of the"):
        _emit(tmp_path, fit, wide, "128:wo", tmp_path / "x.json")
    # an artefact fit over the 9-pair menu (not the widened one) is refused by name
    _, art9, _ = _round1(tmp_path)   # re-creates r1_fit.json over menu9
    with pytest.raises(SystemExit, match="not the widened menu"):
        _emit(tmp_path, tmp_path / "r1_fit.json", wide, PAIRS_ARG, tmp_path / "x.json")
    with pytest.raises(SystemExit, match="lists .* twice"):
        tune_arm.parse_pairs_arg("16:mlp,16:mlp")
    with pytest.raises(SystemExit, match="not rank:set"):
        tune_arm.parse_pairs_arg("16mlp")


def test_addendum_falls_back_to_the_sets_best_round1_trial_when_the_fit_is_not_pd(tmp_path):
    concave = lambda i, s: s.update(best_heldout_loss=math.exp(A0 - 0.4 * (math.log10(s["trial"]["lr"]) - U0) ** 2
                                                              + BETA[s["trial"]["lora_rank"]] + GAMMA[s["trial"]["lora_set_id"]]))
    _, art9, rc = _round1(tmp_path, mutate=concave)
    assert rc is not None and not art9["fit"]["positive_definite"]
    wide = _widen(tmp_path)
    try:
        tune_arm.analyze(argparse.Namespace(out=str(tmp_path), rounds="1", y_key="best_heldout_loss", design="v2", allowed_pairs=str(wide),
                                            pair_only=False, close_search=False, extends_menu=True))
    except SystemExit:
        pass                                                               # the non-PD verdict exits non-zero after writing its artefact
    fit = tmp_path / "r1_fit_menu.json"; shutil.move(tmp_path / "r1_fit.json", fit)
    payload = _emit(tmp_path, fit, wide, "16:attn_mlp,64:down", tmp_path / "r4_trials.json")
    best = {}
    for d in tmp_path.glob("r1_t*/summary.json"):
        s = json.loads(d.read_text()); sid = s["trial"]["lora_set_id"]
        if sid not in best or s["best_heldout_loss"] < best[sid]["best_heldout_loss"]:
            best[sid] = {**s, "_id": d.parent.name}
    for t in payload["trials"]:
        b = best[t["lora_set_id"]]
        assert t["lr"] == pytest.approx(b["trial"]["lr"]) and t["lr_source"]["kind"] == "best_round1_trial" \
            and t["lr_source"]["trial_id"] == b["_id"] and "not positive-definite" in t["lr_source"]["reason"]
    assert payload["addendum_of"]["fit_positive_definite"] is False


def test_addendum_refuses_when_neither_a_prediction_nor_a_round1_trial_of_the_set_exists(tmp_path):
    """A set the design never trained (no rank-4 row either) has no fallback: refused by name, nothing written."""
    p = build(tmp_path, pairs=[(4, "wo"), (16, "wo"), (64, "wo"), (4, "qkvo"), (16, "qkvo"), (64, "qkvo")])
    write_summaries(tmp_path, p, mutate=lambda i, s: s.update(best_heldout_loss=math.exp(A0 - 0.4 * (math.log10(s["trial"]["lr"]) - U0) ** 2
                                                                                        + BETA[s["trial"]["lora_rank"]] + GAMMA[s["trial"]["lora_set_id"]])))
    (tmp_path / "menu6.json").write_bytes((tmp_path / "allowed_pairs.json").read_bytes())
    try:
        analyze_v2(tmp_path, allowed=tmp_path / "menu6.json")
    except SystemExit:
        pass
    wide = write_allowed(tmp_path, [(4, "wo"), (16, "wo"), (64, "wo"), (4, "qkvo"), (16, "qkvo"), (64, "qkvo"), (16, "down")], name="w.json")
    try:
        tune_arm.analyze(argparse.Namespace(out=str(tmp_path), rounds="1", y_key="best_heldout_loss", design="v2", allowed_pairs=str(wide),
                                            pair_only=False, close_search=False, extends_menu=True))
    except SystemExit:
        pass
    fit = tmp_path / "r1_fit_menu.json"; shutil.move(tmp_path / "r1_fit.json", fit)
    with pytest.raises(SystemExit, match="no usable round-1 trial of set 'down'"):
        _emit(tmp_path, fit, wide, "16:down", tmp_path / "x.json")
    assert not (tmp_path / "x.json").exists()


# --- the front door ------------------------------------------------------------------------------------------------------------------

FD6_OVERRIDES = """OVERRIDE_ID=fd6_synthetic
OVERRIDE_INIT_SCHEME=random_lecun_1e-3
OVERRIDE_LORA_ARCH_SEARCH=1
OVERRIDE_LORA_RANK=''
OVERRIDE_LORA_MODULE_SET=''
OVERRIDE_HPO_WARMUP_FRAC=0.15
OVERRIDE_HPO_CLIP_MODE=reconnaissance
OVERRIDE_HPO_CLIP_PERCENTILE=95
OVERRIDE_HPO_TRIALS=16
OVERRIDE_HPO_RECENTRE_TRIALS=4
OVERRIDE_REFUSAL_STEER_ALPHA=0.3
OVERRIDE_EFFECTIVE_BATCH=64
"""   #: the FD 6 protocol as a run-overrides file (03's header: reconnaissance clip + one warm-up level + the rank x module-set search);
      #: the addendum phase belongs to it, and no committed overrides file of this tree selects it


def test_front_door_addendum_phase_is_explicit_requires_pairs_and_composes_round_4(tmp_path):
    site = _site(tmp_path)
    fd6 = tmp_path / "fd6.env"; fd6.write_text(FD6_OVERRIDES)
    ovr = ("--overrides", str(fd6))
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "lora", "--phase", "addendum", "--model", "qwen35_4b_demo", *ovr, site=site, timeout=600)
    assert r.returncode == 2 and "--phase addendum needs --pairs" in r.stderr, r.stdout + r.stderr
    r = run_stage("03_hpo.sh", "--dry-run", "--object-class", "lora", "--phase", "addendum", "--pairs", "16:attn_mlp,64:mlp", "--model", "qwen35_4b_demo",
                  *ovr, site=site, timeout=600)
    out = r.stdout + r.stderr
    assert r.returncode == 0, out[-3000:]
    assert "analyze --design v2" in out and "--extends-menu" in out, out
    assert "emit-trials --addendum-from" in out and "--pairs 16:attn_mlp\\,64:mlp" in out, out      # local_run prints argv with %q (the comma escaped)
    assert "r4_trials.json" in out and "-J hpo-qwen35_4b_demo-lora-r4" in out and "--array=0-1" in out, out
    assert "the refit writes" in out and "r1_fit_menu.json" in out and "restored from" in out and "recognized non-PD artefact" in out, out
    # never part of --phase all; relu has no addendum
    r_all = run_stage("03_hpo.sh", "--dry-run", "--object-class", "lora", "--phase", "all", "--model", "qwen35_4b_demo", *ovr, site=site, timeout=600)
    assert "addendum" not in (r_all.stdout + r_all.stderr).replace("addendum excluded", ""), r_all.stdout[-2000:]
    r_relu = run_stage("03_hpo.sh", "--dry-run", "--object-class", "relu", "--phase", "addendum", "--pairs", "16:wo", "--model", "qwen35_4b_demo", *ovr, site=site, timeout=600)
    assert r_relu.returncode == 2 and "belongs to the FD 6 LoRA search" in r_relu.stderr, r_relu.stdout + r_relu.stderr
