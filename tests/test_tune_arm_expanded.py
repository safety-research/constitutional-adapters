"""The re-tune@32 expansion of the tune_arm harness (twosi-model-sweeps).

Covers: the per-arm expanded Sobol designs (joint log-lr x log-clip x
temperature x weight_decay [x lora_rank], 32 trials/cell, grid snap
balance + non-aliasing, the exact lr x clip net property preserved at any
d, legacy arms byte-unchanged); the generalized quadratic analyze (planted
4-axis recovery, wd=0 on the log axis — sub-eps snap AND grid-edge clamp —
rank as a main-effects factor with per-rank raw bests, the winner carrying
temperature/weight_decay/lora_rank + the realized (batch_size,
accum_steps)); the analyze consistency refusals (mixed ranking objectives,
mixed teacher legs, mixed accumulation, own-T ranking of a T-swept cell);
the rank-parametric MultiLayerLoRAObject (trains at rank 4/16, rank-1
bit-compat, save/load incl. legacy blobs); and the tunable-key filter
plumbing (swept keys stripped from config extras, per-arm refusals, the
fingerprint distinguishing the new axes and the teacher leg).

CPU-only: tiny_llama fixture + synthetic French pairs, never real data.
"""

from __future__ import annotations

import argparse
import collections
import copy
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch

TUNE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "tune_arm.py"
_spec = importlib.util.spec_from_file_location("_tune_arm_exp", TUNE_PATH)
tune_arm = importlib.util.module_from_spec(_spec)
sys.modules["_tune_arm_exp"] = tune_arm
_spec.loader.exec_module(tune_arm)

BOX = {"lr": (3e-5, 3e-3), "max_grad_norm": (0.02, 10.0)}
RETUNE_ARMS = ("psr_kl_all", "psr_mse_all", "lora_r1_all")


# --------------------------------------------------------------------------
# The expanded Sobol design
# --------------------------------------------------------------------------

class TestExpandedSobol:
    def test_per_arm_axes_and_32_trial_designs(self):
        want_axes = {
            "psr_kl_all": ["lr", "max_grad_norm", "temperature",
                           "weight_decay"],
            "psr_mse_all": ["lr", "max_grad_norm", "weight_decay"],
            "lora_r1_all": ["lr", "max_grad_norm", "temperature",
                            "weight_decay", "lora_rank"],
        }
        for arm in RETUNE_ARMS:
            p = tune_arm.build_trials(arm=arm, model="m",
                                      position="response", round_no=1,
                                      n=32, box=BOX)
            assert p["axes"] == want_axes[arm], arm
            assert p["sobol_d"] == len(want_axes[arm])
            assert len(p["trials"]) == 32
            counts = collections.Counter(t["warmup_frac"]
                                         for t in p["trials"])
            assert counts == {0.05: 16, 0.15: 16}
            # every trial carries every axis, snapped exactly onto its grid
            for ax in p["axes"][2:]:
                grid = set(tune_arm.AXIS_GRIDS[ax])
                lv = collections.Counter(t[ax] for t in p["trials"])
                assert set(lv) <= grid, (arm, ax, lv)
                # near-exact balance: >= floor(32/3) - slack per level
                assert min(lv.values()) >= 9, (arm, ax, lv)
                assert set(lv) == grid, (arm, ax, "level missing")

    def test_lr_clip_projection_keeps_the_exact_net_property(self):
        # The 2-D projection of the d-D design IS the 2-D design (Sobol
        # coordinates depend only on their own direction numbers), so the
        # exact dyadic-net assertion holds at every d.
        p = tune_arm.build_trials(arm="lora_r1_all", model="m",
                                  position="response", round_no=1, n=32,
                                  box=BOX)
        for lv in (0.05, 0.15):
            pts = []
            for t in p["trials"]:
                if t["warmup_frac"] != lv:
                    continue
                u = []
                for k in ("lr", "max_grad_norm"):
                    lo, hi = BOX[k]
                    est = ((math.log(t[k]) - math.log(lo))
                           / (math.log(hi) - math.log(lo)))
                    u.append(round(est * (1 << 30)) / (1 << 30))
                pts.append(tuple(u))
            tune_arm.assert_net_coverage(pts, 4, tag=f"w{lv}")

    def test_no_discrete_level_aliases_the_continuous_axes(self):
        # The parity-aliasing bug class, on the snapped axes: every grid
        # level's points must span the lr AND clip ranges.
        for arm in RETUNE_ARMS:
            p = tune_arm.build_trials(arm=arm, model="m",
                                      position="response", round_no=1,
                                      n=32, box=BOX)
            lo, hi = (math.log(v) for v in BOX["lr"])
            for ax in p["axes"][2:]:
                for level in tune_arm.AXIS_GRIDS[ax]:
                    lrs = [math.log(t["lr"]) for t in p["trials"]
                           if t[ax] == level]
                    quarters = {int((v - lo) / (hi - lo) * 4) for v in lrs}
                    assert len(quarters) >= 2, (arm, ax, level)

    def test_legacy_arms_keep_the_2d_design_bit_for_bit(self):
        # token_faithful/bidpo: same shifts, same trial keys, same values
        # as the pre-expansion harness (salt 0 keeps historical shifts).
        p = tune_arm.build_trials(
            arm="token_faithful", model="m", position="all", round_no=1,
            n=16, box={"lr": (3e-5, 3e-3), "max_grad_norm": (0.03, 3.0)})
        assert p["axes"] == ["lr", "max_grad_norm"]
        assert p["sobol_d"] == 2 and p["design_salt"] == 0
        assert all(set(t) == {"index", "lr", "warmup_frac",
                              "max_grad_norm"} for t in p["trials"])
        shift = tune_arm.design_shift(1, "token_faithful", "all", 0.05)
        assert p["design_shift"]["0.05"] == list(shift)

    def test_design_salt_redraws_deterministically(self):
        a = tune_arm.build_trials(arm="psr_kl_all", model="m",
                                  position="response", round_no=1, n=32,
                                  box=BOX, design_salt=1)
        b = tune_arm.build_trials(arm="psr_kl_all", model="m",
                                  position="response", round_no=1, n=32,
                                  box=BOX, design_salt=1)
        c = tune_arm.build_trials(arm="psr_kl_all", model="m",
                                  position="response", round_no=1, n=32,
                                  box=BOX)
        assert a["design_shift"] == b["design_shift"]
        assert a["design_shift"] != c["design_shift"]
        assert a["design_salt"] == 1 and c["design_salt"] == 0

    def test_seed_boxes_exist_per_arm_model_and_reach_the_deconf_region(self):
        for arm in RETUNE_ARMS:
            for slug in ("llama", "qwen38_27b", "gemma4e4b"):
                box = tune_arm.SEED_BOXES_BY_MODEL[(arm, slug)]
                lr_lo, lr_hi = box["lr"]
                c_lo, c_hi = box["max_grad_norm"]
                # clip widened UP into the non-binding region (median
                # preclips 0.17-0.31 -> clip >= 1 non-binding; a decade of
                # it sits inside the box), lr widened DOWN (>= 2 decades).
                assert c_hi >= 10.0 and c_lo <= 0.03
                assert lr_hi / lr_lo >= 100.0
        # the emit path resolves the model slug from the config model id;
        # the BASE Llama mirror resolves to the same existing slug (its
        # CE boxes register under "llama", no needle is inserted ahead of
        # the first-match entry)
        assert tune_arm.model_slug("NousResearch/Meta-Llama-3.1-8B-Instruct") \
            == "llama"
        assert tune_arm.model_slug("NousResearch/Meta-Llama-3.1-8B") == "llama"
        assert tune_arm.model_slug("Qwen/Qwen3.8-27B") == "qwen38_27b"
        assert tune_arm.model_slug("google/gemma-4-E4B-it") == "gemma4e4b"
        assert tune_arm.model_slug("Unknown/Model") is None

    def test_ws3_chat_lanes_route_to_inherited_ce_boxes(self):
        """The two chat checkpoints on the doc-CE lane resolve to their
        `_doc` slugs and INHERIT a tuned lane's CE boxes — GLM-4.6V-Flash
        (dense) the gemma boxes, Nemotron-3-Nano-4B (NemotronH hybrid) the
        qwen3_5 boxes — as the SAME dict objects (a parent box edit moves
        the child). The historical slugs and the first-match order are
        untouched."""
        assert tune_arm.model_slug("zai-org/GLM-4.6V-Flash") == \
            "glm46vflash_doc"
        assert tune_arm.model_slug("nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16") \
            == "nemotron3nano4b_doc"
        assert tune_arm.SEED_BOX_PARENT == {
            "glm46vflash_doc": "gemma4e4b_base",
            "nemotron3nano4b_doc": "qwen35_4b_base"}
        for arm in ("psr_ce_all", "lora_r4_ce"):
            glm = tune_arm.SEED_BOXES_BY_MODEL[(arm, "glm46vflash_doc")]
            nemo = tune_arm.SEED_BOXES_BY_MODEL[(arm, "nemotron3nano4b_doc")]
            assert glm is tune_arm.SEED_BOXES_BY_MODEL[(arm, "gemma4e4b_base")]
            assert nemo is tune_arm.SEED_BOXES_BY_MODEL[(arm, "qwen35_4b_base")]
            # the inherited boxes are the wide CE boxes
            assert glm["max_grad_norm"] == nemo["max_grad_norm"] == \
                tune_arm.CLIP_BOX_WIDE
            assert glm["lr"][1] / glm["lr"][0] >= 100.0
            assert nemo["lr"][1] / nemo["lr"][0] >= 100.0
        # no CE box for the two_si siblings' slugs, and the historical
        # first-match resolution is byte-unchanged
        assert not [k for k in tune_arm.SEED_BOXES_BY_MODEL
                    if k[1] in ("glm46vflash", "nemotron3nano4b")]
        assert tune_arm.model_slug("google/gemma-4-E4B") == "gemma4e4b_base"
        assert tune_arm.model_slug("google/gemma-4-E4B-it") == "gemma4e4b"


# --------------------------------------------------------------------------
# --pin-axis: a follow-up round holds one arm axis at a single grid value
# (the empirical driver: under the offline top-K teacher head, T=2/4
# training diverges — 27/27 completed T!=1 re-tune trials floor-failed —
# so the psr_kl_all/lora_r1_all follow-up rounds pin temperature=1.0).
# --------------------------------------------------------------------------

class TestPinAxis:
    def test_pin_shrinks_the_design_and_stamps_the_constant(self):
        p = tune_arm.build_trials(arm="psr_kl_all", model="m",
                                  position="response", round_no=2, n=32,
                                  box=BOX, pin_axes={"temperature": 1.0})
        # the axis leaves the Sobol dims; the remainder keeps ARM_AXES order
        assert p["axes"] == ["lr", "max_grad_norm", "weight_decay"]
        assert p["sobol_d"] == 3
        assert p["pinned_axes"] == {"temperature": 1.0}
        assert "temperature" not in p["grids"]
        # ... and the pinned VALUE is stamped on every row, so TrialHP (and
        # the fit fingerprint) carry it explicitly — never a registry
        # default temperature, which is not 1.0 for every arm.
        assert all(t["temperature"] == 1.0 for t in p["trials"])
        # the swept remainder keeps its snap balance
        lv = collections.Counter(t["weight_decay"] for t in p["trials"])
        assert set(lv) == set(tune_arm.WEIGHT_DECAY_GRID)
        assert min(lv.values()) >= 9
        # a middle-axis pin keeps the SURROUNDING axes in ARM_AXES order
        q = tune_arm.build_trials(arm="lora_r1_all", model="m",
                                  position="response", round_no=2, n=32,
                                  box=BOX, pin_axes={"weight_decay": 0.0})
        assert q["axes"] == ["lr", "max_grad_norm", "temperature",
                             "lora_rank"]
        assert q["sobol_d"] == 4
        # pinning every extra axis reduces to the legacy 2-D design, rows
        # still stamped
        r = tune_arm.build_trials(
            arm="psr_kl_all", model="m", position="response", round_no=2,
            n=32, box=BOX,
            pin_axes={"temperature": 1.0, "weight_decay": 0.0})
        assert r["axes"] == ["lr", "max_grad_norm"] and r["sobol_d"] == 2
        assert r["grids"] == {}
        assert all(t["temperature"] == 1.0 and t["weight_decay"] == 0.0
                   for t in r["trials"])

    def test_pinned_design_keeps_the_exact_lr_clip_net(self):
        # Sobol coordinates depend only on their own direction numbers, so
        # the (lr, clip) projection of the SHRUNKEN design is still the
        # exact 2-D net — reconstructed from the emitted values, the same
        # way the unpinned expanded test does.
        p = tune_arm.build_trials(arm="psr_kl_all", model="m",
                                  position="response", round_no=2, n=32,
                                  box=BOX, pin_axes={"temperature": 1.0})
        for lv in (0.05, 0.15):
            pts = []
            for t in p["trials"]:
                if t["warmup_frac"] != lv:
                    continue
                u = []
                for k in ("lr", "max_grad_norm"):
                    lo, hi = BOX[k]
                    est = ((math.log(t[k]) - math.log(lo))
                           / (math.log(hi) - math.log(lo)))
                    u.append(round(est * (1 << 30)) / (1 << 30))
                pts.append(tuple(u))
            tune_arm.assert_net_coverage(pts, 4, tag=f"pin-w{lv}")

    def test_pin_lora_rank_stamps_the_int_grid_level(self):
        # the pin stamps the GRID element, so lora_rank stays an int even
        # when the CLI parsed "4" as a float
        p = tune_arm.build_trials(arm="lora_r1_all", model="m",
                                  position="response", round_no=2, n=32,
                                  box=BOX, pin_axes={"lora_rank": 4.0})
        assert p["axes"] == ["lr", "max_grad_norm", "temperature",
                             "weight_decay"]
        assert p["pinned_axes"] == {"lora_rank": 4}
        assert all(t["lora_rank"] == 4 and isinstance(t["lora_rank"], int)
                   for t in p["trials"])

    def test_pin_refusals(self):
        kw = dict(model="m", position="response", round_no=2, n=32, box=BOX)
        # lr/max_grad_norm are the core continuous axes — never pinnable
        for core in ("lr", "max_grad_norm"):
            with pytest.raises(SystemExit, match="core continuous"):
                tune_arm.build_trials(arm="psr_kl_all",
                                      pin_axes={core: 1.0}, **kw)
        # an axis the arm does not sweep refuses (per-arm, not global)
        with pytest.raises(SystemExit, match="does not sweep"):
            tune_arm.build_trials(arm="psr_mse_all",
                                  pin_axes={"temperature": 1.0}, **kw)
        with pytest.raises(SystemExit, match="does not sweep"):
            tune_arm.build_trials(arm="psr_kl_all",
                                  pin_axes={"lora_rank": 4}, **kw)
        with pytest.raises(SystemExit, match="does not sweep"):
            tune_arm.build_trials(arm="token_faithful",
                                  pin_axes={"temperature": 1.0}, **kw)
        # off-grid values refuse — the audited grid is the design
        with pytest.raises(SystemExit, match="not on the temperature grid"):
            tune_arm.build_trials(arm="psr_kl_all",
                                  pin_axes={"temperature": 3.0}, **kw)
        with pytest.raises(SystemExit, match="not on the lora_rank grid"):
            tune_arm.build_trials(arm="lora_r1_all",
                                  pin_axes={"lora_rank": 2}, **kw)

    def _emit_ns(self, tmp_path, pin_axis):
        cfg = {"model": "google/gemma-4-E4B-it", "model_revision": "abc123"}
        cfgp = tmp_path / "cfg.yaml"
        cfgp.write_text(json.dumps(cfg))
        return argparse.Namespace(
            arm="psr_kl_all", position="response",
            out=str(tmp_path / "t.json"), n=None, round=2, box_from=None,
            lr_min=None, lr_max=None, clip_min=None, clip_max=None,
            warmup_levels="0.05,0.15", config=str(cfgp), design_salt=0,
            pin_axis=pin_axis)

    def test_emit_cli_repeatable_pin_round_trip(self, tmp_path, capsys):
        ns = self._emit_ns(tmp_path, ["temperature=1.0"])
        tune_arm.emit_trials(ns)
        payload = json.loads((tmp_path / "t.json").read_text())
        assert payload["pinned_axes"] == {"temperature": 1.0}
        assert payload["axes"] == ["lr", "max_grad_norm", "weight_decay"]
        assert all(t["temperature"] == 1.0 for t in payload["trials"])
        out = capsys.readouterr().out
        assert "pinned={'temperature': 1.0}" in out
        assert "temperature=1(pinned)" in out
        # repeatable: two pins take two axes out
        ns2 = self._emit_ns(tmp_path, ["temperature=1.0",
                                       "weight_decay=0"])
        tune_arm.emit_trials(ns2)
        payload = json.loads((tmp_path / "t.json").read_text())
        assert payload["pinned_axes"] == {"temperature": 1.0,
                                          "weight_decay": 0.0}
        assert payload["axes"] == ["lr", "max_grad_norm"]

    def test_emit_cli_pin_format_refusals(self, tmp_path):
        for bad, needle in ((["temperature"], "AXIS=VALUE"),
                            (["temperature="], "AXIS=VALUE"),
                            (["=1.0"], "AXIS=VALUE"),
                            (["temperature=abc"], "is not a number"),
                            (["temperature=1", "temperature=2"],
                             "given twice")):
            with pytest.raises(SystemExit, match=needle):
                tune_arm.emit_trials(self._emit_ns(tmp_path, bad))


# --------------------------------------------------------------------------
# The generalized quadratic analyze
# --------------------------------------------------------------------------

L0, C0 = math.log(3e-4), math.log(0.5)
T0, D0 = math.log(2.0), math.log(1.1e-3)  # interior T and wd optima


def _kl_loss(t):
    w = 0.0 if t["warmup_frac"] == 0.05 else 1.0
    v = (2.0 * (math.log(t["lr"]) - L0) ** 2
         + 1.5 * (math.log(t["max_grad_norm"]) - C0) ** 2
         + 1.0 * (math.log(t["temperature"]) - T0) ** 2
         + 0.8 * (math.log(t["weight_decay"] + tune_arm.WD_LOG_EPS)
                  - D0) ** 2
         + 0.3 * (math.log(t["lr"]) - L0)
         * (math.log(t["max_grad_norm"]) - C0)
         - 0.5 * w)
    if "lora_rank" in t:
        v += {1: 0.4, 4: 0.0, 16: 0.9}[t["lora_rank"]]  # rank 4 planted
    return math.exp(v)


def _write_cell(out: Path, arm: str, loss_fn, *, mutate=None, n=32,
                round_no=1, box=BOX, pin_axes=None):
    p = tune_arm.build_trials(arm=arm, model="m", position="response",
                              round_no=round_no, n=n, box=box,
                              pin_axes=pin_axes)
    for t in p["trials"]:
        s = {"trial": t, "round": p["round"], "status": "ok", "arm": arm,
             "model": "m", "position": "response", "si": "si_constitution",
             "box": p["box"], "warmup_levels": p["warmup_levels"],
             "heldout_objective": "final_residual_kl",
             "batch_size": 2, "accum_steps": 16,
             "teacher_forward": "precomputed_topk",
             "best_heldout_loss": loss_fn(t), "floor_ok": True}
        if mutate is not None:
            mutate(t, s)
        d = out / f"r{p['round']}_t{t['index']:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "summary.json").write_text(json.dumps(s))
    return p


def _analyze(out: Path, rounds="1"):
    tune_arm.analyze(argparse.Namespace(out=str(out), rounds=rounds,
                                        y_key="best_heldout_loss"))
    tag = "+".join(r.strip() for r in str(rounds).split(","))
    return json.loads((out / f"r{tag}_fit.json").read_text())


class TestExpandedAnalyze:
    def test_recovers_planted_4_axis_optimum_exactly(self, tmp_path):
        _write_cell(tmp_path, "psr_kl_all", _kl_loss)
        art = _analyze(tmp_path)
        fit, sel = art["fit"], art["selection"]
        assert fit["cont_axes"] == ["lr", "max_grad_norm", "temperature",
                                    "weight_decay"]
        assert len(fit["names"]) == 20  # 1 + 4 + 4 + 6 + (1 + 4)
        assert fit["positive_definite"] and fit["design_rank"] == 20
        assert math.isclose(math.log(sel["lr"]), L0, abs_tol=1e-5)
        assert math.isclose(math.log(sel["max_grad_norm"]), C0,
                            abs_tol=1e-5)
        assert math.isclose(math.log(sel["temperature"]), T0, abs_tol=1e-5)
        assert math.isclose(sel["weight_decay"], 1e-3, rel_tol=1e-3)
        assert sel["warmup_frac"] == 0.15 and not sel["on_edge"]
        w = art["winner"]
        for k in ("temperature", "weight_decay", "lr", "max_grad_norm",
                  "warmup_frac"):
            assert k in w
        # the realized effective batch rides the winner (consistency
        # enforced) so the production fold reproduces effective 32.
        assert (w["batch_size"], w["accum_steps"]) == (2, 16)
        assert w["effective_batch"] == 32

    def test_rank_is_a_factor_with_best_level_and_raw_visibility(
            self, tmp_path):
        _write_cell(tmp_path, "lora_r1_all", _kl_loss)
        art = _analyze(tmp_path)
        fit = art["fit"]
        assert fit["rank_levels"] == [1, 4, 16]
        assert "rank=4" in fit["names"] and "rank=16" in fit["names"]
        assert len(fit["names"]) == 22  # the 20-term surface + 2 dummies
        assert fit["best_rank"] == 4
        assert art["winner"]["lora_rank"] == 4
        # the additive-rank assumption's visibility valve
        assert set(art["per_rank_best_raw"]) == {"1", "4", "16"}

    def test_wd_zero_sub_eps_snap_and_grid_edge_clamp(self, tmp_path):
        # (a) vertex just inside the range but sub-eps -> exact 0.0
        def snap_loss(t):
            w = 0.0 if t["warmup_frac"] == 0.05 else 1.0
            return math.exp(
                2.0 * (math.log(t["lr"]) - L0) ** 2
                + 1.5 * (math.log(t["max_grad_norm"]) - C0) ** 2
                + 0.8 * (math.log(t["weight_decay"] + tune_arm.WD_LOG_EPS)
                         - math.log(1e-5 + tune_arm.WD_LOG_EPS)) ** 2
                - 0.5 * w)

        d1 = tmp_path / "snap"
        _write_cell(d1, "psr_mse_all", snap_loss)
        sel = _analyze(d1)["selection"]
        assert sel["weight_decay"] == 0.0
        assert sel["clamped_axes"] == []
        # (b) vertex a full nat BELOW the grid range -> active-set clamp to
        # the wd=0 edge, with the OTHER axes re-solved exactly.
        below = math.log(tune_arm.WD_LOG_EPS) - 1.0

        def clamp_loss(t):
            w = 0.0 if t["warmup_frac"] == 0.05 else 1.0
            return math.exp(
                2.0 * (math.log(t["lr"]) - L0) ** 2
                + 1.5 * (math.log(t["max_grad_norm"]) - C0) ** 2
                + 0.8 * (math.log(t["weight_decay"] + tune_arm.WD_LOG_EPS)
                         - below) ** 2 - 0.5 * w)

        d2 = tmp_path / "clamp"
        _write_cell(d2, "psr_mse_all", clamp_loss)
        sel = _analyze(d2)["selection"]
        assert sel["weight_decay"] == 0.0
        assert sel["clamped_axes"] == ["weight_decay"]
        assert math.isclose(math.log(sel["lr"]), L0, abs_tol=1e-5)

    def test_legacy_2_axis_pool_reproduces_the_9_term_model(self, tmp_path):
        def loss2(t):
            w = 0.0 if t["warmup_frac"] == 0.05 else 1.0
            return math.exp(2.0 * (math.log(t["lr"]) - L0) ** 2
                            + 1.5 * (math.log(t["max_grad_norm"])
                                     - C0) ** 2 - 0.5 * w)

        _write_cell(tmp_path, "token_faithful", loss2,
                    box={"lr": (3e-5, 3e-3), "max_grad_norm": (0.03, 3.0)},
                    n=16,
                    mutate=lambda t, s: (
                        s.pop("accum_steps"), s.pop("batch_size"),
                        s.pop("teacher_forward"),
                        s.__setitem__("heldout_objective",
                                      "lockstep_heldout_loss"),
                        s.__setitem__("position", "all")))
        art = _analyze(tmp_path)
        assert art["fit"]["names"] == list(tune_arm.QUAD9_NAMES)
        assert math.isclose(math.log(art["selection"]["lr"]), L0,
                            abs_tol=1e-6)

    @pytest.mark.parametrize("name,mutate,needle", [
        ("mixed_labels",
         lambda t, s: (s.__setitem__("heldout_objective",
                                     "lockstep_heldout_loss")
                       if t["index"] == 3 else None),
         "mixed ranking objectives"),
        ("mixed_teacher",
         lambda t, s: (s.__setitem__("teacher_forward", "online")
                       if t["index"] == 5 else None),
         "mixed teacher legs"),
        ("mixed_accum",
         lambda t, s: (s.__setitem__("accum_steps", 8)
                       if t["index"] == 7 else None),
         "consistent"),
        ("accum_some_absent",
         lambda t, s: (s.pop("accum_steps")
                       if t["index"] == 9 else None),
         "present on some summaries"),
    ])
    def test_cell_consistency_refusals(self, tmp_path, name, mutate,
                                       needle):
        _write_cell(tmp_path, "psr_kl_all", _kl_loss, mutate=mutate)
        with pytest.raises(SystemExit, match=needle):
            _analyze(tmp_path)

    def test_temperature_swept_cell_must_rank_by_the_probe(self, tmp_path):
        # Every summary claiming the own-T heldout loss (a consistent but
        # WRONG ranking for a T sweep) refuses with the T guard.
        _write_cell(tmp_path, "psr_kl_all", _kl_loss,
                    mutate=lambda t, s: s.__setitem__(
                        "heldout_objective", "lockstep_heldout_loss"))
        with pytest.raises(SystemExit, match="not comparable across T"):
            _analyze(tmp_path)

    def test_refine_round_pools_with_r1_on_the_expanded_axes(self, tmp_path):
        _write_cell(tmp_path, "psr_kl_all", _kl_loss)
        zoom = {"lr": (1e-4, 9e-4), "max_grad_norm": (0.2, 1.4)}
        _write_cell(tmp_path, "psr_kl_all", _kl_loss, n=8, round_no=2,
                    box=zoom)
        art = _analyze(tmp_path, rounds="1,2")
        assert art["fit"]["n_rows"] == 40
        assert art["winner"]["rounds_pooled"] == [1, 2]
        assert math.isclose(math.log(art["selection"]["temperature"]), T0,
                            abs_tol=1e-4)


# --------------------------------------------------------------------------
# Constant-axis analyze: an extra axis identical across every pooled
# usable row (a --pin-axis round, or T=1 survivors after the T!=1
# top-K-head floor exclusions) leaves the quadratic design and is echoed
# as a FIXED value on the fit artifact and the winner.
# --------------------------------------------------------------------------

class TestConstantAxisAnalyze:
    def test_pinned_cell_fits_over_the_remaining_axes(self, tmp_path):
        # planted-optimum recovery with T constant: the T term of the
        # planted loss is a constant offset, absorbed by the intercept —
        # the remaining (lr, clip, wd, warmup) surface recovers exactly.
        _write_cell(tmp_path, "psr_kl_all", _kl_loss,
                    pin_axes={"temperature": 1.0})
        art = _analyze(tmp_path)
        fit, sel = art["fit"], art["selection"]
        assert fit["cont_axes"] == ["lr", "max_grad_norm", "weight_decay"]
        assert fit["fixed_axes"] == {"temperature": 1.0}
        assert art["fixed_axes"] == {"temperature": 1.0}
        assert len(fit["names"]) == 14  # 1 + 3 + 3 + 3 + (1 + 3)
        assert not any("temp" in nm for nm in fit["names"])
        assert fit["positive_definite"] and fit["design_rank"] == 14
        assert math.isclose(math.log(sel["lr"]), L0, abs_tol=1e-5)
        assert math.isclose(math.log(sel["max_grad_norm"]), C0,
                            abs_tol=1e-5)
        assert math.isclose(sel["weight_decay"], 1e-3, rel_tol=1e-3)
        assert sel["warmup_frac"] == 0.15 and not sel["on_edge"]
        # the fixed value is echoed on selection AND winner — the
        # production fold must state temperature=1.0 explicitly, never
        # fall back to a registry default
        assert sel["temperature"] == 1.0
        assert art["winner"]["temperature"] == 1.0

    def test_r1_t1_survivors_pool_with_a_pinned_r2(self, tmp_path):
        # THE motivating pooling: round 1 swept T {1,2,4} but every T!=1
        # trial floor-failed under the offline top-K head (renormalized
        # mass collapse), so its usable rows carry T=1.0 only; round 2 is
        # emitted with --pin-axis temperature=1.0. Pooled, temperature is
        # constant -> one fit over the remaining axes.
        def floor_fail_t_ne_1(t, s):
            if t["temperature"] != 1.0:
                s["floor_ok"] = False

        r1 = _write_cell(tmp_path, "psr_kl_all", _kl_loss,
                         mutate=floor_fail_t_ne_1)
        zoom = {"lr": (1e-4, 9e-4), "max_grad_norm": (0.2, 1.4)}
        _write_cell(tmp_path, "psr_kl_all", _kl_loss, n=16, round_no=2,
                    box=zoom, pin_axes={"temperature": 1.0})
        art = _analyze(tmp_path, rounds="1,2")
        n_t1 = sum(1 for t in r1["trials"] if t["temperature"] == 1.0)
        assert art["fit"]["n_rows"] == n_t1 + 16
        assert len(art["excluded"]) == 32 - n_t1
        assert all(reason == "floor_failed"
                   for _name, reason in art["excluded"])
        assert art["fixed_axes"] == {"temperature": 1.0}
        assert art["fit"]["cont_axes"] == ["lr", "max_grad_norm",
                                           "weight_decay"]
        assert art["winner"]["temperature"] == 1.0
        assert art["winner"]["rounds_pooled"] == [1, 2]
        sel = art["selection"]
        assert math.isclose(math.log(sel["lr"]), L0, abs_tol=1e-5)
        assert math.isclose(sel["weight_decay"], 1e-3, rel_tol=1e-3)

    def test_constant_rank_is_dropped_and_echoed_fixed(self, tmp_path):
        # rank enters as FACTOR dummies; one shared level carries no
        # factor dof -> no dummies, echoed fixed like a constant axis.
        _write_cell(tmp_path, "lora_r1_all", _kl_loss,
                    pin_axes={"lora_rank": 4})
        art = _analyze(tmp_path)
        fit = art["fit"]
        assert fit["rank_levels"] == []
        assert not any(nm.startswith("rank=") for nm in fit["names"])
        assert len(fit["names"]) == 20  # the 4-axis surface, no dummies
        assert fit["fixed_axes"] == {"lora_rank": 4}
        assert art["winner"]["lora_rank"] == 4
        assert art["per_rank_best_raw"] is None
        # the still-varying axes fit as usual
        assert math.isclose(math.log(art["selection"]["temperature"]), T0,
                            abs_tol=1e-4)

    def test_pinned_wd_pool_reduces_to_the_exact_9_term_model(self,
                                                              tmp_path):
        # with the one extra axis pinned, the design IS the legacy 9-term
        # model — names and order included — plus the fixed echo.
        def loss2(t):
            w = 0.0 if t["warmup_frac"] == 0.05 else 1.0
            return math.exp(2.0 * (math.log(t["lr"]) - L0) ** 2
                            + 1.5 * (math.log(t["max_grad_norm"])
                                     - C0) ** 2 - 0.5 * w)

        _write_cell(tmp_path, "psr_mse_all", loss2,
                    pin_axes={"weight_decay": 0.0})
        art = _analyze(tmp_path)
        assert art["fit"]["names"] == list(tune_arm.QUAD9_NAMES)
        assert art["fixed_axes"] == {"weight_decay": 0.0}
        assert art["winner"]["weight_decay"] == 0.0
        assert math.isclose(math.log(art["selection"]["lr"]), L0,
                            abs_tol=1e-6)

    def test_t_guard_binds_only_when_temperature_varies(self, tmp_path):
        # constant-T + own-T label: ONE T is one objective — no across-T
        # comparison to protect, so the guard stands down...
        d1 = tmp_path / "const"
        _write_cell(d1, "psr_kl_all", _kl_loss,
                    pin_axes={"temperature": 1.0},
                    mutate=lambda t, s: s.__setitem__(
                        "heldout_objective", "lockstep_heldout_loss"))
        art = _analyze(d1)
        assert art["fixed_axes"] == {"temperature": 1.0}
        # ... while a genuinely T-swept pool with the own-T label refuses
        d2 = tmp_path / "vary"
        _write_cell(d2, "psr_kl_all", _kl_loss,
                    mutate=lambda t, s: s.__setitem__(
                        "heldout_objective", "lockstep_heldout_loss"))
        with pytest.raises(SystemExit, match="not comparable across T"):
            _analyze(d2)

    def test_varying_pools_carry_empty_fixed_axes(self, tmp_path):
        # the regression face of the feature: a full varying-T pool fits
        # exactly as before, with nothing echoed fixed
        _write_cell(tmp_path, "psr_kl_all", _kl_loss)
        art = _analyze(tmp_path)
        assert art["fixed_axes"] == {}
        assert art["fit"]["fixed_axes"] == {}
        assert art["fit"]["cont_axes"] == ["lr", "max_grad_norm",
                                           "temperature", "weight_decay"]
        assert len(art["fit"]["names"]) == 20


# --------------------------------------------------------------------------
# The rank-parametric MultiLayerLoRAObject
# --------------------------------------------------------------------------

from sisteer.data.instructions import BY_ID  # noqa: E402
from sisteer.data.pairs import Pair, pairs_to_distill_examples  # noqa: E402
from sisteer.extraction.multilayer import MultiLayerLoRAObject  # noqa: E402

INSTRUCTION = BY_ID["french"]


def _pairs(n=4):
    data = [
        ("Say something nice.", "Bonjour le monde", "Hello world"),
        ("Describe a tree.", "C'est un grand arbre", "It is a tall tree"),
        ("Greet the reader.", "Salut a tous", "Hi everyone"),
        ("Name a color.", "Le bleu est joli", "Blue is nice"),
    ]
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data[:n]]


class TestRankParametricLoRA:
    @pytest.mark.parametrize("rank", [4, 16])
    def test_trains_at_higher_rank(self, tiny_llama, rank):
        from sisteer.extraction import tokcache
        from sisteer.extraction.objects import train_object_kl

        model, tok = tiny_llama
        tokcache.set_active(tokcache.TokCache())
        obj = MultiLayerLoRAObject(model, rank=rank)
        n_layers = len(obj.layers)
        d_in = obj.A[obj.layers[0]].shape[0]
        d_out = obj.B[obj.layers[0]].shape[0]
        assert obj.A[obj.layers[0]].shape == (d_in, rank)
        assert obj.B[obj.layers[0]].shape == (d_out, rank)
        assert obj.n_params() == n_layers * rank * (d_in + d_out)
        # B zero-init: the untrained object is exactly the unsteered model
        ids = torch.tensor([[1, 2, 3]])
        base = model(input_ids=ids).logits
        with obj.apply(model):
            steered0 = model(input_ids=ids).logits
        assert torch.equal(base, steered0)
        ex = pairs_to_distill_examples(tok, _pairs(4), INSTRUCTION)
        history = train_object_kl(model, tok, obj, ex, lr=5e-3, epochs=1,
                                  batch_size=2, max_len=96, seed=0)
        assert all(math.isfinite(h) for h in history)
        assert any(obj.B[l].abs().sum() > 0 for l in obj.layers), \
            "training left every B at zero"
        # and a trained rank-r object CHANGES the forward
        with obj.apply(model):
            steered = model(input_ids=ids).logits
        assert not torch.equal(base, steered)

    def test_rank1_is_bit_identical_to_the_legacy_object(self, tiny_llama):
        model, _tok = tiny_llama
        obj = MultiLayerLoRAObject(model)  # default rank=1
        l0 = obj.layers[0]
        assert obj.rank == 1
        # LEGACY shapes and the LEGACY RNG draw: randn(d_in) under the
        # per-layer generator, so pre-rank artifacts/trajectories match.
        d_in = obj.A[l0].shape[0]
        assert obj.A[l0].dim() == 1 and obj.B[l0].dim() == 1
        g = torch.Generator().manual_seed(0 + l0)
        want = torch.randn(d_in, generator=g) / d_in ** 0.5
        assert torch.equal(obj.A[l0].data, want)

    def test_save_load_roundtrip_and_legacy_blob(self, tiny_llama,
                                                 tmp_path):
        model, tok = tiny_llama
        obj = MultiLayerLoRAObject(model, rank=4)
        for l in obj.layers:
            obj.B[l].data.normal_()
        p = tmp_path / "obj.pt"
        obj.save(p)
        blob = torch.load(p, weights_only=True)
        assert blob["rank"] == 4
        back = MultiLayerLoRAObject.load_blob(blob, model, tok)
        assert back.rank == 4
        for l in obj.layers:
            assert torch.equal(back.A[l].data.float(),
                               obj.A[l].data.float())
        # a LEGACY (pre-rank) blob: 1-D tensors, no "rank" key
        legacy = MultiLayerLoRAObject(model)
        legacy.save(tmp_path / "legacy.pt")
        lb = torch.load(tmp_path / "legacy.pt", weights_only=True)
        del lb["rank"]  # what an old writer produced
        old = MultiLayerLoRAObject.load_blob(lb, model, tok)
        assert old.rank == 1
        assert torch.equal(old.A[old.layers[0]].data.float(),
                           legacy.A[legacy.layers[0]].data.float())
        # an inconsistent rank key refuses
        bad = copy.deepcopy(blob)
        bad["rank"] = 2
        with pytest.raises(ValueError, match="internally inconsistent"):
            MultiLayerLoRAObject.load_blob(bad, model, tok)

    def test_registry_solo_path_takes_lora_rank(self):
        import inspect

        from sisteer import methods

        assert "lora_rank" in inspect.signature(
            methods._train_lora_all).parameters


# --------------------------------------------------------------------------
# Tunable-key filter plumbing + per-arm refusals
# --------------------------------------------------------------------------

class TestKeyPlumbing:
    def test_swept_keys_are_stripped_from_config_extras(self):
        for k in ("temperature", "weight_decay", "lora_rank",
                  "accum_steps"):
            assert k in tune_arm._STRIPPED_CFG_KEYS, k
        cfg = {"train_kwargs": {"psr_kl_all": {
            "temperature": 4.0, "weight_decay": 0.1, "lora_rank": 16,
            "reg_coeff": 0.2}}}
        extras = tune_arm._cfg_extras(cfg, "psr_kl_all")
        assert extras == {"reg_coeff": 0.2}

    def test_lora_rank_refused_on_non_lora_arms(self, tiny_llama):
        model, tok = tiny_llama
        from sisteer.data.pairs import pairs_to_distill_examples as p2d

        setup = tune_arm.TuneSetup(
            cfg={"model": "TESTMODEL", "train_kwargs": {}}, driver=None,
            model=model, tok=tok, instruction=INSTRUCTION,
            si="si_constitution", fit_pairs=_pairs(4),
            held_pairs=_pairs(4), held_ex=p2d(tok, _pairs(4), INSTRUCTION),
            max_len=96, positions="all", response_header_len=None, layer=1)
        for arm in ("psr_kl_all", "psr_mse_all", "bidpo"):
            adapter = tune_arm.make_adapter(arm)
            with pytest.raises(SystemExit, match="lora arm's search axis"):
                adapter.make_object(setup, lora_rank=4)

    def test_lora_adapter_builds_a_rank_r_object(self, tiny_llama):
        model, tok = tiny_llama
        from sisteer.data.pairs import pairs_to_distill_examples as p2d

        setup = tune_arm.TuneSetup(
            cfg={"model": "TESTMODEL", "train_kwargs": {}}, driver=None,
            model=model, tok=tok, instruction=INSTRUCTION,
            si="si_constitution", fit_pairs=_pairs(4),
            held_pairs=_pairs(4), held_ex=p2d(tok, _pairs(4), INSTRUCTION),
            max_len=96, positions="response", response_header_len=3,
            layer=1)
        adapter = tune_arm.make_adapter("lora_r1_all")
        obj = adapter.make_object(setup, lora_rank=16)
        assert isinstance(obj, MultiLayerLoRAObject) and obj.rank == 16
        # the fit-mask leg rides exactly like the builder's own lines
        assert obj.fit_positions == "response"
        assert obj.fit_response_header_len == 3

    def test_psr_mse_refuses_a_temperature(self, tiny_llama):
        model, tok = tiny_llama
        from sisteer.data.pairs import pairs_to_distill_examples as p2d

        setup = tune_arm.TuneSetup(
            cfg={"model": "TESTMODEL", "train_kwargs": {}}, driver=None,
            model=model, tok=tok, instruction=INSTRUCTION,
            si="si_constitution", fit_pairs=_pairs(4),
            held_pairs=_pairs(4), held_ex=p2d(tok, _pairs(4), INSTRUCTION),
            max_len=96, positions="all", response_header_len=None, layer=1)
        adapter = tune_arm.make_adapter("psr_mse_all")
        adapter.prepare_data(setup, setup.fit_pairs, length_sort=False)
        adapter.make_object(setup)
        hp = tune_arm.TrialHP(lr=1e-4, warmup_frac=0.05, max_grad_norm=1.0,
                              temperature=2.0)
        with pytest.raises(SystemExit, match="no temperature axis"):
            adapter.train(setup, hp, epochs=1, batch_size=2, seed=0,
                          log_fn=None, probe_cb=None, probe_every=25,
                          diagnostics={})

    @pytest.mark.xfail(
        reason="driver-side registration pends the trainer lane's commit: "
               "temperature/weight_decay must join train_two_si_objects."
               "_LOCKSTEP_OPTIMIZER_KEYS (merged into LockstepArmParams by "
               "lockstep_arm_params) and lora_rank must be driver-consumed "
               "at build_lockstep_object + accepted by "
               "lockstep_dropped_keys, or a re-tune winner folded into "
               "train_kwargs.<arm> is refused/dropped at fit launch",
        strict=False)
    def test_driver_allowlists_accept_the_retune_winner_keys(self):
        import importlib.util as ilu

        p = TUNE_PATH.parent / "train_two_si_objects.py"
        spec2 = ilu.spec_from_file_location("_t2_retune_keys", p)
        t2 = ilu.module_from_spec(spec2)
        sys.modules["_t2_retune_keys"] = t2
        spec2.loader.exec_module(t2)
        assert "temperature" in t2._LOCKSTEP_OPTIMIZER_KEYS
        assert "weight_decay" in t2._LOCKSTEP_OPTIMIZER_KEYS
        assert "accum_steps" in t2._LOCKSTEP_OPTIMIZER_KEYS
        cfg = {"train_kwargs": {"lora_r1_all": {"lora_rank": 4}}}
        assert t2.lockstep_dropped_keys(cfg, "lora_r1_all") == []


# --------------------------------------------------------------------------
# The fingerprint distinguishes the new axes and the teacher leg
# --------------------------------------------------------------------------

class TestExpandedFingerprint:
    def _world(self, tmp_path):
        cfg = {"model": "TESTMODEL", "model_revision": "r1"}
        (tmp_path / "cfg.yaml").write_text(json.dumps(cfg))
        pool = tmp_path / "pool.jsonl"
        pool.write_text(json.dumps({"prompt": "p", "pos": "a", "neg": "b",
                                    "instruction_id": "french"}) + "\n")
        payload = tune_arm.build_trials(
            arm="psr_kl_all", model="TESTMODEL", model_revision="r1",
            config_sha=tune_arm._config_file_sha(str(tmp_path / "cfg.yaml")),
            position="response", round_no=1, n=32, box=BOX)
        args = argparse.Namespace(
            pairs=str(pool), si="si_constitution", batch_size=2, epochs=3,
            fit_subsample=0, bidpo_harmful_pairs=None,
            bidpo_benign_pairs=None, teacher_topk="off")
        return cfg, payload, args

    def test_axes_and_teacher_leg_change_the_fingerprint(self, tmp_path):
        cfg, payload, args = self._world(tmp_path)
        adapter = tune_arm.make_adapter("psr_kl_all")
        t0 = payload["trials"][0]
        fp = tune_arm._trial_fingerprint(args, cfg, payload, t0,
                                         "psr_kl_all", "response", adapter)
        # a different temperature at identical lr/clip/warmup is different
        # work
        t_other = dict(t0, temperature=[x for x in (1.0, 2.0, 4.0)
                                        if x != t0["temperature"]][0])
        fp2 = tune_arm._trial_fingerprint(args, cfg, payload, t_other,
                                          "psr_kl_all", "response", adapter)
        assert fp != fp2
        # ... and so is the same hp against a different teacher leg
        art = tmp_path / "store.safetensors"
        art.write_bytes(b"tensors")
        Path(str(art) + ".json").write_text("{}")
        args.teacher_topk = str(art)
        fp3 = tune_arm._trial_fingerprint(args, cfg, payload, t0,
                                          "psr_kl_all", "response", adapter)
        assert fp3 != fp
        # accumulation is part of the optimizer identity: a different
        # batch size realizes a different accum at effective 32
        args.teacher_topk = "off"
        args.batch_size = 1
        fp4 = tune_arm._trial_fingerprint(args, cfg, payload, t0,
                                          "psr_kl_all", "response", adapter)
        assert fp4 != fp

    def test_auto_topk_refuses_when_the_sidecar_is_missing(self, tmp_path):
        _cfg, _payload, args = self._world(tmp_path)
        args.teacher_topk = "auto"
        with pytest.raises(SystemExit, match="no artifact"):
            tune_arm._teacher_topk_path(args)


# --------------------------------------------------------------------------
# teacher_topk plumbing at the adapter seam
# --------------------------------------------------------------------------

class TestTeacherTopkPlumbing:
    def test_lockstep_adapter_passes_an_attached_store(self, tiny_llama,
                                                       monkeypatch):
        import dataclasses

        import sisteer.extraction.lockstep as lockstep_mod
        from sisteer.extraction.lockstep import LockstepArmParams

        if "accum_steps" not in {f.name for f in
                                 dataclasses.fields(LockstepArmParams)}:
            pytest.skip("LockstepArmParams.accum_steps not landed yet "
                        "(trainer lane)")
        model, tok = tiny_llama
        from sisteer.data.pairs import pairs_to_distill_examples as p2d

        setup = tune_arm.TuneSetup(
            cfg={"model": "TESTMODEL", "train_kwargs": {}}, driver=None,
            model=model, tok=tok, instruction=INSTRUCTION,
            si="si_constitution", fit_pairs=_pairs(4),
            held_pairs=_pairs(4), held_ex=p2d(tok, _pairs(4), INSTRUCTION),
            max_len=96, positions="all", response_header_len=None, layer=1)
        adapter = tune_arm.make_adapter("psr_kl_all")
        adapter.prepare_data(setup, setup.fit_pairs, length_sort=False)
        adapter.make_object(setup)
        marker = object()
        adapter.teacher_topk = marker
        calls = {}

        def spy(model_, tok_, arms, examples, **kw):
            calls["kw"] = kw
            return {a.name: [0.0] for a in arms}

        monkeypatch.setattr(lockstep_mod, "train_lockstep", spy)
        hp = tune_arm.TrialHP(lr=1e-4, warmup_frac=0.05, max_grad_norm=1.0)
        adapter.train(setup, hp, epochs=1, batch_size=2, seed=0,
                      log_fn=None, probe_cb=None, probe_every=25,
                      diagnostics={})
        assert calls["kw"]["teacher_topk"] is marker

    def test_explicit_store_on_a_non_consuming_arm_refuses(self, tmp_path,
                                                           tiny_llama):
        model, tok = tiny_llama
        from sisteer.data.pairs import pairs_to_distill_examples as p2d

        art = tmp_path / "store.safetensors"
        art.write_bytes(b"tensors")
        Path(str(art) + ".json").write_text("{}")
        args = argparse.Namespace(teacher_topk=str(art), si="french",
                                  pairs=str(tmp_path / "pool.jsonl"))
        setup = tune_arm.TuneSetup(
            cfg={"model": "TESTMODEL", "train_kwargs": {}}, driver=None,
            model=model, tok=tok, instruction=INSTRUCTION,
            si="si_constitution", fit_pairs=_pairs(4),
            held_pairs=_pairs(4), held_ex=p2d(tok, _pairs(4), INSTRUCTION),
            max_len=96, positions="all", response_header_len=None, layer=1)
        adapter = tune_arm.make_adapter("psr_mse_all")
        with pytest.raises(SystemExit, match="no teacher-logit leg"):
            tune_arm._load_teacher_topk(args, setup, adapter)
        # "auto" on the same arm stands down with a note instead
        args.teacher_topk = "auto"
        pool = tmp_path / "pool.jsonl"
        pool.write_text("{}\n")
        sidecar = Path(str(pool) + ".teachertopk.safetensors")
        sidecar.write_bytes(b"tensors")
        Path(str(sidecar) + ".json").write_text("{}")
        assert tune_arm._load_teacher_topk(args, setup, adapter) is None
        assert adapter.teacher_topk is None


def test_single_level_warmup_scoped_sweep():
    """A scoped sweep may FIX warmup at one level (the E2B safety lane,
    2026-08-27): the design collapses to ONE d-D Sobol net, every row
    carries the fixed level, and pinned axes still stamp their constants."""
    import tune_arm

    pay = tune_arm.build_trials(
        arm="psr_kl_all", model="m", model_revision="r",
        config_sha="c" * 16, position="response", round_no=1, n=8,
        box={"lr": (1e-4, 3e-3), "max_grad_norm": (0.05, 2.0)},
        warmup_levels=(0.15,),
        pin_axes={"temperature": 1.0, "weight_decay": 0.01})
    assert len(pay["trials"]) == 8
    assert pay["warmup_levels"] == [0.15]
    assert pay["axes"] == ["lr", "max_grad_norm"]
    assert all(t["warmup_frac"] == 0.15 for t in pay["trials"])
    assert all(t["temperature"] == 1.0 and t["weight_decay"] == 0.01
               for t in pay["trials"])
    # two-level emission is unchanged
    pay2 = tune_arm.build_trials(
        arm="psr_kl_all", model="m", model_revision="r",
        config_sha="c" * 16, position="response", round_no=1, n=8,
        box={"lr": (1e-4, 3e-3), "max_grad_norm": (0.05, 2.0)})
    assert sorted({t["warmup_frac"] for t in pay2["trials"]}) == [0.05, 0.15]


def test_lora_r4_is_the_same_arm_under_its_new_name():
    """lora_r4 = lora_r1_all renamed post-re-tune (rank-4 winner): same
    adapter, same axes/grids/boxes, and ARM_CANON pools legacy-labeled
    round-1 summaries with lora_r4 follow-up rounds."""
    import tune_arm

    assert tune_arm.ARM_CANON["lora_r1_all"] == "lora_r4"
    assert tune_arm.ARM_AXES["lora_r4"] == tune_arm.ARM_AXES["lora_r1_all"]
    a_new, a_old = (tune_arm.make_adapter("lora_r4"),
                    tune_arm.make_adapter("lora_r1_all"))
    assert type(a_new) is type(a_old)
    for m in ("llama", "gemma4e4b", "qwen38_27b"):
        assert (tune_arm.SEED_BOXES_BY_MODEL[("lora_r4", m)]
                == tune_arm.SEED_BOXES_BY_MODEL[("lora_r1_all", m)])
    pay = tune_arm.build_trials(
        arm="lora_r4", model="m", model_revision="r", config_sha="c" * 16,
        position="response", round_no=2, n=8,
        box={"lr": (1e-5, 3e-3), "max_grad_norm": (0.02, 10.0)},
        pin_axes={"temperature": 1.0})
    assert pay["arm"] == "lora_r4"
    assert all("lora_rank" in t for t in pay["trials"])
