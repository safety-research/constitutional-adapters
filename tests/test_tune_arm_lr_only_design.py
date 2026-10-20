"""The LR-ONLY hyperparameter search design and the fits' effective batch
on the search side.

Only the learning rate is swept; the clip is a
per-arm CONSTANT derived from the gradient-norm reconnaissance, so eight
trials per arm fit the 5-term quadratic (1, lr, lr*lr, warmup, warmup*lr)
with three dof. Pinned here, CPU-only, no model:

- the seed rows for the CE arms under the EXISTING slug ``llama`` carry an
  lr range and NO clip range — the LR-only declaration.
- ``build_trials`` on a zero-width clip box draws a 1-D Sobol design in
  log(lr), stamps the constant on every row, and reports it as
  ``constant_axes``; every lr x clip payload is byte-identical to before.
- ``emit-trials`` refuses the llama rows without ``--clip-const``, defaults
  ``--n`` to 8 for them, refuses a non-power-of-two split, and stamps
  ``--si`` on the payload (the cell pin ``run_trial`` enforces).
- ``analyze``: a constant clip is held FIXED (5 terms, ``fixed_axes``,
  the winner echoes it), a varying clip is the unchanged 9-term model with
  byte-identical term names, an edge argmin recentres with n = 8, and the
  selection guard records the fixed axis instead of dividing by its zero
  width or alarming on it.
- ``CESoloAdapter.effective_batch_target`` is the CONFIG's own
  ``batch_size x accum_steps`` once folded (64 on the Llama config, 8 on
  every frozen key), CE_EFFECTIVE_BATCH when unbound/legacy, a refusal on
  an unfolded config for a trial, and the sentinel's own EFF<n> for
  probe-mem.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from test_tune_arm import (  # noqa: E402  — ONE tune_arm module instance
    _L0, _analyze, _planted_log_y, _write_summaries, tune_arm)
from test_sdfbase_configs import msc  # noqa: E402  — the config generator

guard = tune_arm.hpo_selection_guard

LLAMA_BASE_CFG = REPO / "configs/sdfbase_llama8b_base_fits.yaml"
GEMMA_BASE_CFG = REPO / "configs/sdfbase_gemma4e4b_base_fits.yaml"
CLIP = 0.9   # a stand-in for the recon-derived per-arm constant
FIVE_TERMS = ["1", "lr", "lr*lr", "warmup", "warmup*lr"]


def _lr_only_payload(arm="psr_ce_all", n=8, round_no=1, clip=CLIP):
    box = {"lr": tune_arm.LLAMA8B_CE_LR_BOX[arm], "max_grad_norm": (clip, clip)}
    return tune_arm.build_trials(arm=arm, model="NousResearch/Meta-Llama-3.1-8B",
                                 position="all", round_no=round_no, n=n,
                                 box=box)


def _emit_ns(tmp_path, **over):
    ns = dict(arm="psr_ce_all", position="all", out=str(tmp_path / "t.json"),
              n=None, round=1, box_from=None, lr_min=None, lr_max=None,
              clip_min=None, clip_max=None, clip_const=None, si=None,
              warmup_levels="0.05,0.15", config=str(LLAMA_BASE_CFG),
              design_salt=0, pin_axis=[])
    ns.update(over)
    return argparse.Namespace(**ns)


# --------------------------------------------------------------------------
# the seed rows
# --------------------------------------------------------------------------

def test_llama_ce_rows_are_lr_only_under_the_existing_slug():
    boxes = tune_arm.SEED_BOXES_BY_MODEL
    assert tune_arm.LLAMA8B_CE_TRIALS_PER_ARM == 8
    # the tops registered at lr_c x 10^1.5 (PSR 1.4e-2, LoRA
    # 1.1e-2) were measured UNSTABLE by the reconnaissance at effective
    # batch 64 (`recon_5` heavy_tail, p95 7,391; `recon_6` diverged, p95
    # 27.3) and the next rung down was certified stable (`recon_5b` 4.5e-3,
    # `recon_6b` 3.5e-3), so each top is the certified rung; the bottoms
    # (lr_c / 10^1.5) are unchanged.
    assert tune_arm.LLAMA8B_CE_LR_BOX == {"psr_ce_all": (1.4e-5, 4.5e-3),
                                          "lora_r4_ce": (1.1e-5, 3.5e-3)}
    for arm, (lo, hi) in tune_arm.LLAMA8B_CE_LR_BOX.items():
        row = boxes[(arm, "llama")]
        assert set(row) == {"lr"}                      # NO clip range
        assert row["lr"] == (lo, hi)
        # the spec's minimum width: at least 2.5 decades (these are exactly
        # 2.5 — one decade came off the top, none off the bottom)
        assert math.log10(hi / lo) >= 2.5 - 1e-9, (arm, lo, hi)
        assert tune_arm.lr_only_box(row)
    # bottom = the SI winner scaled 32 -> 64 by sqrt(2) (4.5e-4 / 3.5e-4)
    # over 10^1.5; top = that anchor x 10, the reconnaissance-certified rung
    for arm, anchor in (("psr_ce_all", 4.5e-4), ("lora_r4_ce", 3.5e-4)):
        lo, hi = boxes[(arm, "llama")]["lr"]
        assert math.isclose(lo, anchor / 10 ** 1.5, rel_tol=0.03)
        assert math.isclose(hi, anchor * 10, rel_tol=1e-9)
    # the existing first-match slug, byte-unchanged, both Nous ids
    assert tune_arm.MODEL_SLUGS[0] == ("llama", "llama")
    assert tune_arm.model_slug("NousResearch/Meta-Llama-3.1-8B") == "llama"
    assert tune_arm.model_slug("NousResearch/Meta-Llama-3.1-8B-Instruct") \
        == "llama"
    # the SI-distillation llama rows are untouched two-axis boxes
    for arm in ("psr_kl_all", "psr_mse_all", "lora_r1_all", "lora_r4"):
        assert boxes[(arm, "llama")]["max_grad_norm"] == tune_arm.CLIP_BOX_WIDE
        assert not tune_arm.lr_only_box(boxes[(arm, "llama")])
    # every OTHER CE row still carries the wide clip range — except the
    # instruct-direct batch-64 rows, LR-only like these
    for (arm, slug), row in boxes.items():
        if arm in ("psr_ce_all", "lora_r4_ce") and slug != "llama" \
                and slug not in tune_arm.IT_EB64_CE_LR_BOX:
            assert row["max_grad_norm"] == tune_arm.CLIP_BOX_WIDE
    assert tune_arm.lr_only_box({"lr": (1, 2), "max_grad_norm": (0.9, 0.9)})
    assert not tune_arm.lr_only_box({"lr": (1, 2), "max_grad_norm": (0.5, 0.9)})


# --------------------------------------------------------------------------
# build_trials: the 1-D design
# --------------------------------------------------------------------------

def test_build_trials_lr_only_design_is_one_sobol_axis_plus_a_constant():
    for arm in ("psr_ce_all", "lora_r4_ce"):
        p = _lr_only_payload(arm)
        assert len(p["trials"]) == 8
        assert p["axes"] == ["lr"] and p["sobol_d"] == 1 and p["sobol_m"] == 2
        assert p["constant_axes"] == {"max_grad_norm": CLIP}
        assert p["box"]["max_grad_norm"] == [CLIP, CLIP]
        assert p["box"]["lr"] == list(tune_arm.LLAMA8B_CE_LR_BOX[arm])
        assert p["pinned_axes"] == {} and p["grids"] == {}
        lo, hi = (math.log(v) for v in tune_arm.LLAMA8B_CE_LR_BOX[arm])
        for lv in (0.05, 0.15):
            rows = [t for t in p["trials"] if t["warmup_frac"] == lv]
            assert len(rows) == 4
            assert all(t["max_grad_norm"] == CLIP for t in rows)
            # exact 1-D stratification: 4 points, 4 dyadic strata of log lr
            strata = {int((math.log(t["lr"]) - lo) / (hi - lo) * 4)
                      for t in rows}
            assert strata == {0, 1, 2, 3}, (arm, lv, rows)
        assert all(set(t) == {"index", "lr", "warmup_frac", "max_grad_norm"}
                   for t in p["trials"])
        # interleaved levels, one independent shift per level, d = 1
        assert [t["warmup_frac"] for t in p["trials"]] == [0.05, 0.15] * 4
        assert all(len(v) == 1 for v in p["design_shift"].values())
        assert p["design_shift"]["0.05"] != p["design_shift"]["0.15"]


def test_build_trials_refusals_for_the_lr_only_design():
    # no clip side at all: build_trials never invents the constant
    with pytest.raises(SystemExit, match="--clip-const"):
        tune_arm.build_trials(arm="psr_ce_all", model="m", position="all",
                              round_no=1, n=8,
                              box={"lr": tune_arm.LLAMA8B_CE_LR_BOX["psr_ce_all"]})
    # a non-positive / non-finite constant
    for bad in (0.0, -1.0, float("inf"), float("nan")):
        with pytest.raises(SystemExit, match="finite positive"):
            tune_arm.build_trials(arm="psr_ce_all", model="m", position="all",
                                  round_no=1, n=8,
                                  box={"lr": (1e-5, 1e-2),
                                       "max_grad_norm": (bad, bad)})
    # the LR-only design has no room for an arm's discrete axes
    with pytest.raises(SystemExit, match="discrete axes"):
        tune_arm.build_trials(arm="psr_kl_all", model="m", position="response",
                              round_no=1, n=8,
                              box={"lr": (1e-5, 1e-2),
                                   "max_grad_norm": (CLIP, CLIP)})
    # 8 trials = 4 per level = 2^2; 12 would be 6 per level
    with pytest.raises(SystemExit, match="power of 2"):
        _lr_only_payload(n=12)


def test_lr_x_clip_payloads_are_byte_identical_to_before():
    box = {"lr": (3e-5, 3e-3), "max_grad_norm": (0.03, 3.0)}
    p = tune_arm.build_trials(arm="token_faithful", model="m", position="all",
                              round_no=1, n=16, box=box)
    assert p["axes"] == ["lr", "max_grad_norm"] and p["sobol_d"] == 2
    assert "constant_axes" not in p          # emitted for LR-only only
    assert p["design_shift"]["0.05"] == list(
        tune_arm.design_shift(1, "token_faithful", "all", 0.05))
    clips = {t["max_grad_norm"] for t in p["trials"]}
    assert len(clips) == 16                  # the clip really is swept


# --------------------------------------------------------------------------
# emit-trials: the CLI on the committed llama config
# --------------------------------------------------------------------------

def test_emit_trials_llama_rows_need_the_clip_constant_and_default_to_eight(
        tmp_path):
    with pytest.raises(SystemExit, match="LR-only design"):
        tune_arm.emit_trials(_emit_ns(tmp_path))
    # one side alone cannot fill a range that was never declared
    with pytest.raises(SystemExit, match="LR-only design"):
        tune_arm.emit_trials(_emit_ns(tmp_path, clip_min=0.1))
    # the constant and a range contradict each other
    with pytest.raises(SystemExit, match="contradicts"):
        tune_arm.emit_trials(_emit_ns(tmp_path, clip_const=CLIP, clip_max=2.0))
    for arm in ("psr_ce_all", "lora_r4_ce"):
        tune_arm.emit_trials(_emit_ns(tmp_path, arm=arm, clip_const=CLIP,
                                      si="constv2-mix-10m"))
        p = json.loads((tmp_path / "t.json").read_text())
        assert len(p["trials"]) == 8            # the default, not 16
        assert p["round"] == 1 and p["arm"] == arm
        assert p["si"] == "constv2-mix-10m"     # the cell pin
        assert p["axes"] == ["lr"]
        assert p["constant_axes"] == {"max_grad_norm": CLIP}
        assert p["box"] == {"lr": list(tune_arm.LLAMA8B_CE_LR_BOX[arm]),
                            "max_grad_norm": [CLIP, CLIP]}
        cfg = yaml.safe_load(LLAMA_BASE_CFG.read_text())
        assert p["model"] == cfg["model"] == "NousResearch/Meta-Llama-3.1-8B"
        assert p["model_revision"] == cfg["model_revision"]
        assert p["config_sha"] == tune_arm._config_file_sha(str(LLAMA_BASE_CFG))
        assert all(t["max_grad_norm"] == CLIP for t in p["trials"])
    # --n must still split into a power of two per level
    with pytest.raises(SystemExit, match="power of 2"):
        tune_arm.emit_trials(_emit_ns(tmp_path, clip_const=CLIP, n=24))
    # --si is optional: absent, the payload carries no pin (HPO_SI applies)
    tune_arm.emit_trials(_emit_ns(tmp_path, clip_const=CLIP))
    assert "si" not in json.loads((tmp_path / "t.json").read_text())


def test_emit_trials_can_deliberately_reopen_the_clip_axis(tmp_path, capsys):
    tune_arm.emit_trials(_emit_ns(tmp_path, clip_min=0.1, clip_max=2.0))
    p = json.loads((tmp_path / "t.json").read_text())
    assert p["axes"] == ["lr", "max_grad_norm"]
    assert len(p["trials"]) == 16              # the 2-axis default
    assert "constant_axes" not in p
    assert "re-open the clip" in capsys.readouterr().out


def test_emit_trials_gemma_ce_rows_are_unchanged(tmp_path):
    tune_arm.emit_trials(_emit_ns(tmp_path, config=str(GEMMA_BASE_CFG)))
    p = json.loads((tmp_path / "t.json").read_text())
    assert p["axes"] == ["lr", "max_grad_norm"] and len(p["trials"]) == 16
    assert p["box"]["max_grad_norm"] == list(tune_arm.CLIP_BOX_WIDE)


# --------------------------------------------------------------------------
# analyze: constant clip -> 5 terms, fixed; varying clip -> the 9 terms
# --------------------------------------------------------------------------

def test_analyze_constant_clip_is_held_fixed_in_a_five_term_fit(tmp_path):
    payload = _lr_only_payload()
    _write_summaries(tmp_path, payload)   # planted lr optimum at _L0
    art = _analyze(tmp_path)
    fit = art["fit"]
    assert fit["names"] == FIVE_TERMS
    assert fit["cont_axes"] == ["lr"]
    assert fit["fixed_axes"] == {"max_grad_norm": CLIP}
    assert art["fixed_axes"] == {"max_grad_norm": CLIP}
    assert fit["design_rank"] == 5 and fit["n_rows"] == 8
    assert fit["positive_definite"] and len(fit["hessian_eigs"]) == 1
    assert fit["sigma_log_y"] is not None     # 3 dof -> a real noise estimate
    sel = art["selection"]
    assert math.isclose(math.log(sel["lr"]), _L0, abs_tol=1e-6)
    assert sel["warmup_frac"] == 0.15
    assert sel["max_grad_norm"] == CLIP        # echoed, not fitted
    assert not sel["on_edge"]
    assert sel["edge_pos"]["max_grad_norm"] is None   # no edge on a fixed axis
    assert 0.0 < sel["edge_pos"]["lr"] < 1.0
    w = art["winner"]
    assert w["max_grad_norm"] == CLIP and w["selected_from"] == "fitted_argmin"
    assert math.isclose(math.log(w["lr"]), _L0, abs_tol=1e-6)
    assert art["next"]["action"] == "refine"
    assert art["next"]["box"]["max_grad_norm"] == [CLIP, CLIP]
    g = art["selection_guard"]
    assert g["dims"]["max_grad_norm"]["fixed"] is True
    assert g["dims"]["max_grad_norm"]["box_pos"] is None
    assert not [a for a in g["alarms"] if a["dim"] == "max_grad_norm"]
    assert not g["alarmed"]
    # the planted objective is exactly quadratic, so sigma_log_y ~ 0 and
    # nothing else lies within noise; the fixed axis still spans nothing
    assert "max_grad_norm" not in g["spread"]["within_noise"]["span"]
    assert set(g["spread"]["within_noise"]["span"]) <= {"lr", "warmup_frac"}


def test_analyze_varying_clip_keeps_the_nine_term_model_byte_for_byte(
        tmp_path):
    box = {"lr": (3e-5, 3e-3), "max_grad_norm": (0.03, 3.0)}
    payload = tune_arm.build_trials(arm="token_faithful", model="m",
                                    position="all", round_no=1, n=16, box=box)
    _write_summaries(tmp_path, payload)
    art = _analyze(tmp_path)
    assert art["fit"]["names"] == list(tune_arm.QUAD9_NAMES)
    assert art["fit"]["cont_axes"] == ["lr", "max_grad_norm"]
    assert art["fit"]["fixed_axes"] == {} and art["fixed_axes"] == {}
    assert art["fit"]["design_rank"] == 9
    assert art["next"]["n"] == 8 and art["next"]["action"] == "refine"


def test_analyze_lr_only_edge_argmin_recentres_with_eight(tmp_path):
    payload = _lr_only_payload()
    lo, hi = tune_arm.LLAMA8B_CE_LR_BOX["psr_ce_all"]

    def mutate(idx, s):
        t = s["trial"]
        w = 0.0 if t["warmup_frac"] == 0.05 else 1.0
        # the lr optimum planted a decade ABOVE the box top
        s["best_heldout_loss"] = math.exp(
            2.0 * (math.log(t["lr"]) - math.log(hi * 10)) ** 2 - 0.5 * w)

    _write_summaries(tmp_path, payload, mutate=mutate)
    art = _analyze(tmp_path)
    assert art["fit"]["names"] == FIVE_TERMS
    assert art["selection"]["on_edge"]
    assert art["winner"] is None
    nxt = art["next"]
    assert nxt["action"] == "recenter"
    assert nxt["n"] == 8                                  # not 16
    assert nxt["box"]["max_grad_norm"] == [CLIP, CLIP]    # the constant rides
    old = math.log(hi) - math.log(lo)
    nb = nxt["box"]["lr"]
    assert math.isclose(math.log(nb[1]) - math.log(nb[0]), old, rel_tol=1e-9)
    assert nb[1] > hi
    g = art["selection_guard"]
    assert g["dims"]["max_grad_norm"]["fixed"] is True
    assert {a["code"] for a in g["alarms"]} >= {"fitted_argmin_on_edge"}
    assert not [a for a in g["alarms"] if a["dim"] == "max_grad_norm"]
    # a --box-from round inherits the constant: no --clip-const needed
    fit_path = tmp_path / "r1_fit.json"
    out2 = tmp_path / "r2.json"
    tune_arm.emit_trials(_emit_ns(tmp_path, box_from=str(fit_path),
                                  out=str(out2), round=2, n=8,
                                  si="constv2-mix-10m"))
    p2 = json.loads(out2.read_text())
    assert len(p2["trials"]) == 8 and p2["round"] == 2
    assert p2["axes"] == ["lr"] and p2["constant_axes"] == {"max_grad_norm": CLIP}
    assert all(nb[0] <= t["lr"] <= nb[1] for t in p2["trials"])


def test_a_constant_lr_is_still_a_degenerate_pool(tmp_path):
    payload = _lr_only_payload()
    for t in payload["trials"]:
        t["lr"] = 3e-4
    _write_summaries(tmp_path, payload)
    with pytest.raises(SystemExit, match="RANK-DEFICIENT"):
        _analyze(tmp_path)


# --------------------------------------------------------------------------
# the selection guard on a fixed axis
# --------------------------------------------------------------------------

def test_guard_treats_a_zero_width_axis_as_fixed():
    box = {"lr": [1e-5, 1e-2], "max_grad_norm": [CLIP, CLIP]}
    assert guard.is_fixed_axis(CLIP, CLIP)
    assert not guard.is_fixed_axis(0.5, CLIP)
    on_edge, pos = guard.box_edge_positions(
        {"lr": 3e-4, "max_grad_norm": CLIP}, box)
    assert not on_edge and pos["max_grad_norm"] is None
    assert math.isclose(pos["lr"], math.log10(3e-4 / 1e-5) / 3.0)
    on_edge, pos = guard.box_edge_positions(
        {"lr": 1e-2, "max_grad_norm": CLIP}, box)
    assert on_edge and pos["lr"] == 1.0
    rows = [{"lr": lr, "max_grad_norm": CLIP, "warmup_frac": wf,
             "best_heldout_loss": 1.0 + (math.log(lr / 3e-4)) ** 2,
             "round": 1, "index": i}
            for i, (lr, wf) in enumerate(
                [(3e-5, 0.05), (1e-4, 0.15), (3e-4, 0.05), (1e-3, 0.15),
                 (3e-3, 0.05), (5e-5, 0.15), (5e-4, 0.05), (2e-3, 0.15)])]
    fit = {"positive_definite": True, "hessian_eigs": [4.0]}
    g = guard.guard_selection(selected={"lr": 3e-4, "max_grad_norm": CLIP,
                                        "warmup_frac": 0.05},
                              selected_from="fitted_argmin", rows=rows,
                              box=box, levels=(0.05, 0.15), fit=fit,
                              sigma_log_y=0.05, fitted_min_log_y=0.0)
    d = g["dims"]["max_grad_norm"]
    assert d["fixed"] is True and d["box_pos"] is None
    assert d["sampled_min"] == d["sampled_max"] == CLIP
    assert not [a for a in g["alarms"] if a["dim"] == "max_grad_norm"]
    assert g["selected"]["max_grad_norm"] == CLIP
    span = g["spread"]["within_noise"]["span"]
    assert "lr" in span and "max_grad_norm" not in span   # fixed spans nothing
    assert "fixed" not in g["dims"]["lr"] and g["dims"]["lr"]["box_pos"] is not None
    # a selection off the declared constant is the one thing that alarms
    g2 = guard.guard_selection(selected={"lr": 3e-4, "max_grad_norm": 0.5,
                                         "warmup_frac": 0.05},
                               selected_from="fitted_argmin", rows=rows,
                               box=box, levels=(0.05, 0.15), fit=fit,
                               sigma_log_y=0.05, fitted_min_log_y=0.0)
    assert [a["code"] for a in g2["alarms"] if a["dim"] == "max_grad_norm"] \
        == ["fixed_axis_mismatch"]
    # a genuine two-axis box still divides normally
    with pytest.raises(ValueError, match="degenerate box"):
        guard.box_position(1.0, CLIP, CLIP)


# --------------------------------------------------------------------------
# the search-side effective batch is the fits'
# --------------------------------------------------------------------------

def _folded(key: str, bs: dict) -> dict:
    return msc.build_config(key, memprobe={key: bs})


def test_effective_batch_target_is_the_folded_configs_own_product():
    for arm in ("psr_ce_all", "lora_r4_ce"):
        ad = tune_arm.make_adapter(arm)
        # unbound: the historical rule (test_ce_adapters_registered pins 8)
        assert ad.cfg is None and ad.effective_batch_target == 8
        # the Llama rows, folded at bs 1 / bs 2: 64 either way, and
        # accum_for(bs) IS the config's own accum_steps
        for bs in (1, 2, 4):
            cfg = _folded("llama8b_base", {"psr_ce_all": bs, "lora_r4_ce": bs})
            ad.bind_config(cfg)
            assert cfg["train_kwargs"][arm]["accum_steps"] == 64 // bs
            assert cfg["batch_size_by_arm"][arm] == bs
            assert ad.effective_batch_target == 64
            assert ad.accum_for(bs) == cfg["train_kwargs"][arm]["accum_steps"]
        cfg = _folded("llama8b_it_doc", {"psr_ce_all": 1, "lora_r4_ce": 2})
        ad.bind_config(cfg)
        assert ad.effective_batch_target == 64
        # every frozen key reproduces 8 from its committed (folded) bytes
        for name in ("sdfbase_gemma4e4b_base_fits.yaml",
                     "sdfbase_qwen35_4b_base_fits.yaml",
                     "sdfbase_gemma4e4b_it_doc_fits.yaml",
                     "sdfbase_qwen35_4b_it_doc_fits.yaml"):
            cfg = yaml.safe_load((REPO / "configs" / name).read_text())
            ad.bind_config(cfg)
            assert ad.effective_batch_target == 8, name
            bs = int(cfg["batch_size_by_arm"][arm])
            assert ad.accum_for(bs) == int(cfg["train_kwargs"][arm]["accum_steps"])
        # a config without the generator's keys: the legacy rule
        ad.bind_config({"model": "x", "train_kwargs": {arm: {"lr": 1e-4}}})
        assert ad.effective_batch_target == 8
        # --accum still overrides (bench / probe-mem)
        ad.bind_config(_folded("llama8b_base", {"psr_ce_all": 1, "lora_r4_ce": 1}))
        ad.accum_override = 3
        assert ad.accum_for(1) == 3
        ad.accum_override = 0
        assert ad.accum_for(1) == 64


def _unfolded_copy_of_committed(key: str, eff: int) -> dict:
    """The committed config with its memprobe fold REVERTED — the two
    sentinels re-inserted on both arms — so the unfolded fixture is
    synthesised here and never read off the committed file."""
    cfg = yaml.safe_load(
        (REPO / "configs" / f"sdfbase_{key}_fits.yaml").read_text())
    for arm in msc.CE_ARMS:
        cfg["batch_size_by_arm"][arm] = msc.MEMPROBE_SENTINEL
        cfg["train_kwargs"][arm]["accum_steps"] = \
            msc.ACCUM_SENTINEL_FMT.format(eff=eff)
    return cfg


def test_an_unfolded_config_refuses_for_a_trial_but_probe_mem_reads_eff():
    # the unfolded fixture is SYNTHESISED — the pure derivation with
    # NO memprobe fold for every key, plus the committed llama files with
    # their fold reverted — never the committed file as-is, which pinned the
    # pre-fold state and went red the moment the fold landed.
    for key, eff in (("llama8b_base", 64), ("llama8b_it_doc", 64),
                     ("glm46vflash_doc", 8), ("nemotron3nano4b_doc_full", 8),
                     ("gemma4e4b_base", 8)):
        forms = [msc.build_config(key)]
        if key.startswith("llama8b"):
            reverted = _unfolded_copy_of_committed(key, eff)
            # fold-independent: the committed file differs from the pure
            # derivation ONLY by its folds (memprobe reverted here, HPO
            # taken from the store — sentinels until the winners fold)
            assert reverted == msc.build_config(
                key, hpo=msc.load_folds()["hpo"])
            forms.append(reverted)
        for cfg in forms:
            for arm in ("psr_ce_all", "lora_r4_ce"):
                assert cfg["batch_size_by_arm"][arm] == "MEMPROBE"
                assert cfg["train_kwargs"][arm]["accum_steps"] == \
                    f"PENDING_MEMPROBE_EFF{eff}_ACCUM"
                ad = tune_arm.make_adapter(arm)
                ad.bind_config(cfg)
                with pytest.raises(SystemExit, match="not memprobe-folded"):
                    ad.effective_batch_target
                with pytest.raises(SystemExit, match="not memprobe-folded"):
                    ad.accum_for(1)
                ad.allow_unfolded_config = True       # what probe_mem sets
                assert ad.effective_batch_target == eff
                assert ad.accum_for(1) == eff and ad.accum_for(2) == eff // 2
    # a half-folded config (int width, sentinel accum) is broken, not legacy
    ad = tune_arm.make_adapter("psr_ce_all")
    ad.bind_config({"batch_size_by_arm": {"psr_ce_all": 1},
                    "train_kwargs": {"psr_ce_all": {
                        "accum_steps": "PENDING_MEMPROBE_EFF64_ACCUM"}}})
    with pytest.raises(SystemExit, match="not memprobe-folded"):
        ad.effective_batch_target
    # bools are not widths
    ad.bind_config({"batch_size_by_arm": {"psr_ce_all": True},
                    "train_kwargs": {"psr_ce_all": {"accum_steps": 8}}})
    with pytest.raises(SystemExit, match="not memprobe-folded"):
        ad.effective_batch_target


def test_probe_mem_and_run_trial_bind_the_config():
    src = (REPO / "scripts/tune_arm.py").read_text()
    i_probe = src.index("def probe_mem(args)")
    assert "adapter.allow_unfolded_config = True" in src[i_probe:i_probe + 800]
    i_trial = src.index("def run_trial(args)")
    j = src.index("expected_fp = _trial_fingerprint(", i_trial)
    assert "adapter.bind_config(cfg)" in src[i_trial:j]
    # prepare_data binds setup.cfg for probe-mem / bench
    i_prep = src.index("def prepare_data(self, setup, fit_pairs, *, length_sort=True,\n"
                       "                     take_longest=None, batch_size=None) -> int:\n"
                       "        from sisteer.data.pairs import docs_to_ce_examples")
    assert "self.bind_config(setup.cfg)" in src[i_prep:i_prep + 400]


# --------------------------------------------------------------------------
# the recenter round keeps the clip constant BITWISE
# --------------------------------------------------------------------------

def test_boxed_preserves_a_zero_width_axis_verbatim():
    """exp(log(c)) != c for constants like 0.34 / 0.096, so a box helper that
    round-trips a FIXED axis through log space stamps a drifted constant on
    the next round. A zero-width side rides verbatim; the searched side is
    re-centred as before."""
    for clip in (0.34, 0.096):
        assert math.exp(math.log(clip)) != clip           # the premise
        box = {"lr": [1e-5, 1e-2], "max_grad_norm": [clip, clip]}
        for zoom in (1.0, tune_arm.REFINE_ZOOM):
            nb = tune_arm._boxed({"lr": 3e-4, "max_grad_norm": clip}, box, zoom)
            assert nb["max_grad_norm"] == [clip, clip]
            assert math.isclose(math.log(nb["lr"][1]) - math.log(nb["lr"][0]),
                                zoom * math.log(1e3), rel_tol=1e-9)
    # a searched clip axis is unchanged: same log-width, re-centred
    box = {"lr": [1e-5, 1e-2], "max_grad_norm": [0.03, 3.0]}
    nb = tune_arm._boxed({"lr": 3e-4, "max_grad_norm": 0.3}, box, 1.0)
    assert math.isclose(math.log(nb["max_grad_norm"][1])
                        - math.log(nb["max_grad_norm"][0]), math.log(100.0),
                        rel_tol=1e-9)
    assert math.isclose(nb["max_grad_norm"][0], 0.03, rel_tol=1e-9)


@pytest.mark.parametrize("clip", [0.34, 0.096])
def test_two_rounds_pool_into_one_five_term_design_at_a_drifting_constant(
        tmp_path, clip):
    """Round 1 on the edge -> recenter -> round 2 -> pool 1+2: the clip
    constant the recon derived (a value exp(log(x)) does NOT round-trip)
    must reach round 2 bitwise and the pooled fit must still be the 5-term
    LR-only model with the clip FIXED. Before the fix the recenter box
    carried 0.3400000000000001, round 2 was stamped with it, and pooling
    built a 9-term design that refused as rank-deficient — after the second
    eight trials had been paid for."""
    assert math.exp(math.log(clip)) != clip
    payload = _lr_only_payload(clip=clip)
    _lo, hi = tune_arm.LLAMA8B_CE_LR_BOX["psr_ce_all"]

    def mutate(idx, s):
        t = s["trial"]
        w = 0.0 if t["warmup_frac"] == 0.05 else 1.0
        s["best_heldout_loss"] = math.exp(
            2.0 * (math.log(t["lr"]) - math.log(hi * 10)) ** 2 - 0.5 * w)

    _write_summaries(tmp_path, payload, mutate=mutate)
    art1 = _analyze(tmp_path)
    assert art1["fit"]["names"] == FIVE_TERMS
    assert art1["fixed_axes"] == {"max_grad_norm": clip}
    assert art1["next"]["action"] == "recenter"
    assert art1["next"]["box"]["max_grad_norm"] == [clip, clip]   # verbatim
    out2 = tmp_path / "r2.json"
    tune_arm.emit_trials(_emit_ns(tmp_path, box_from=str(tmp_path / "r1_fit.json"),
                                  out=str(out2), round=2, n=8,
                                  si="constv2-mix-10m"))
    p2 = json.loads(out2.read_text())
    assert p2["constant_axes"] == {"max_grad_norm": clip}
    assert p2["box"]["max_grad_norm"] == [clip, clip]
    assert all(t["max_grad_norm"] == clip for t in p2["trials"])
    _write_summaries(tmp_path, p2, mutate=mutate)
    art = _analyze(tmp_path, rounds="1,2")
    assert art["rounds"] == [1, 2]
    assert art["fit"]["names"] == FIVE_TERMS                    # not 9 terms
    assert art["fit"]["cont_axes"] == ["lr"]
    assert not art["fit"].get("rank_deficient")
    assert art["fit"]["design_rank"] == 5 and art["fit"]["n_rows"] == 16
    assert art["fit"]["fixed_axes"] == {"max_grad_norm": clip}
    assert art["fixed_axes"] == {"max_grad_norm": clip}
    assert art["box"]["max_grad_norm"] == [clip, clip]
    assert art["selection"]["max_grad_norm"] == clip
    assert art["selection_guard"]["dims"]["max_grad_norm"]["fixed"] is True


def test_analyze_validates_rows_against_the_declared_clip_constant(tmp_path):
    """The design DECLARES the fixed axis (its box side is [c, c]); rows are
    validated against that declaration instead of the pool's dimensionality
    being inferred from float uniqueness. A row whose clip differs from its
    own design's constant is a corrupt summary, refused by name; a pool that
    mixes an LR-only round with a clip-sweeping round is two designs."""
    payload = _lr_only_payload(clip=0.34)

    def drift(idx, s):
        if idx == 3:
            s["trial"]["max_grad_norm"] = 0.3400000000000001

    _write_summaries(tmp_path / "drift", payload, mutate=drift)
    with pytest.raises(SystemExit, match="declared"):
        _analyze(tmp_path / "drift")
    # a clip-sweeping round pooled with the LR-only round: two designs
    # (a fresh payload: the mutator above edited the trial dicts in place)
    _write_summaries(tmp_path / "mixed", _lr_only_payload(clip=0.34))
    swept = tune_arm.build_trials(
        arm="psr_ce_all", model=payload["model"], position="all", round_no=2,
        n=8, box={"lr": tune_arm.LLAMA8B_CE_LR_BOX["psr_ce_all"],
                  "max_grad_norm": (0.1, 1.0)})
    _write_summaries(tmp_path / "mixed", swept)
    with pytest.raises(SystemExit, match="ONE design"):
        _analyze(tmp_path / "mixed", rounds="1,2")


# --------------------------------------------------------------------------
# the instruct-direct BATCH-64 rows (2026-09-11)
# --------------------------------------------------------------------------

IT_EB64_CFG = {"gemma4e4b": REPO / "configs/sdfbase_gemma4e4b_it_doc_eb64_fits.yaml",
               "qwen35_4b": REPO / "configs/sdfbase_qwen35_4b_it_doc_eb64_fits.yaml"}


def test_it_eb64_ce_rows_are_lr_only_under_the_instruct_slugs():
    boxes = tune_arm.SEED_BOXES_BY_MODEL
    # tops = the reconnaissance-certified stable rungs, bottoms the preset lows
    assert tune_arm.IT_EB64_CE_LR_BOX == {
        "gemma4e4b": {"psr_ce_all": (2e-4, 2e-3), "lora_r4_ce": (3e-5, 1e-2)},
        "qwen35_4b": {"psr_ce_all": (1e-4, 3.162e-3), "lora_r4_ce": (5e-4, 6.324e-3)}}
    # the instruct repo ids resolve to these slugs; the base ids keep theirs
    assert tune_arm.model_slug("google/gemma-4-E4B-it") == "gemma4e4b"
    assert tune_arm.model_slug("Qwen/Qwen3.5-4B") == "qwen35_4b"
    assert tune_arm.model_slug("google/gemma-4-E4B") == "gemma4e4b_base"
    assert tune_arm.model_slug("Qwen/Qwen3.5-4B-Base") == "qwen35_4b_base"
    for slug, arms in tune_arm.IT_EB64_CE_LR_BOX.items():
        for arm, (lo, hi) in arms.items():
            row = boxes[(arm, slug)]
            assert row == {"lr": (lo, hi)} and "max_grad_norm" not in row
            assert tune_arm.lr_only_box(row)
            # widths after the reconnaissance certified the tops:
            # gemma psr 1.0 / gemma lora 2.5 / qwen psr 1.5 / qwen lora 1.1
            # decades — every top is the highest rung measured STABLE at
            # clip 1e6, the bottoms are the preset lower bounds; 8 Sobol
            # points over that is 0.13-0.3 decades apart
            assert 0 < lo < hi and math.log10(hi / lo) >= 1.0 - 1e-9
    # the qwen needle activated nothing else: every (arm, qwen35_4b) row is ours
    assert {a for a, s in boxes if s == "qwen35_4b"} == {"psr_ce_all", "lora_r4_ce"}
    # the two_si gemma instruct rows are untouched by the CE additions
    assert {a for a, s in boxes if s == "gemma4e4b"} == {
        "lora_r1_all", "lora_r4", "psr_kl_all", "psr_mse_all", "psr_ce_all", "lora_r4_ce"}


@pytest.mark.parametrize("slug", sorted(IT_EB64_CFG))
@pytest.mark.parametrize("arm", ["psr_ce_all", "lora_r4_ce"])
def test_emit_trials_it_eb64_rows_take_the_registered_box(tmp_path, slug, arm):
    """emit-trials on an `_eb64` config resolves the registered box for the
    instruct slug, needs the clip constant (LR-only design), defaults to
    eight rows inside the box, and pins the search cell."""
    cfg = IT_EB64_CFG[slug]
    with pytest.raises(SystemExit, match="clip-const"):
        tune_arm.emit_trials(_emit_ns(tmp_path, config=str(cfg), arm=arm,
                                      si="constv2-mix-10m"))
    ns = _emit_ns(tmp_path, config=str(cfg), arm=arm, clip_const=CLIP,
                  si="constv2-mix-10m")
    tune_arm.emit_trials(ns)
    p = json.loads((tmp_path / "t.json").read_text())
    lo, hi = tune_arm.IT_EB64_CE_LR_BOX[slug][arm]
    assert p["box"] == {"lr": [lo, hi], "max_grad_norm": [CLIP, CLIP]}
    assert p["si"] == "constv2-mix-10m" and len(p["trials"]) == 8
    assert p["model"] == yaml.safe_load(cfg.read_text())["model"]
    for t in p["trials"]:
        assert lo <= t["lr"] <= hi and t["max_grad_norm"] == CLIP
        assert t["warmup_frac"] in (0.05, 0.15)
