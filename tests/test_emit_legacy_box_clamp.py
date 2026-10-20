"""The legacy (1-D LR-only) design honours --lr-log10-min / --lr-log10-max as CLAMPS on its lr box (paper-companion-repo, the front
door's rule: the reconnaissance's certified top lowers the box top so an unstable rung never enters the ReLU search) — never
widening, refusing an inverted box, and naming the width (a NOTE below the spec's 2.5 decades). Until now the flag was refused outside
--design sobol16_coverage, so 03's ReLU design phase died on the flag it passes by design."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_tune_arm import tune_arm  # noqa: E402

ARM = "psr_ce_all"


def _emit(tmp_path, monkeypatch, capsys, **over):
    cfg = tmp_path / "fits.yaml"; cfg.write_text("model: Qwen/Qwen3.5-4B-Base\nmodel_revision: r\nbatch_size_by_arm:\n  psr_ce_all: 1\n")
    monkeypatch.setattr(tune_arm, "_load_cfg", lambda p: {"model": "Qwen/Qwen3.5-4B-Base", "model_revision": "r", "batch_size_by_arm": {ARM: 1}})
    monkeypatch.setattr(tune_arm, "model_slug", lambda m: "qwen35_4b_base", raising=False)   # the demo fits the paper's Qwen3.5-4B-Base: its boxes are qwen35_4b_base's (tune_arm.model_slug)
    recon = tmp_path / "recon.json"; recon.write_text(json.dumps({"clip": 1.5, "percentile": 95.0, "steps": 150}))
    out = tmp_path / "r1_trials.json"
    ns = dict(design="legacy", out=str(out), arm=ARM, config=str(cfg), position="all", si="constv2-mix-10m", warmup_levels="0.15",
              init_scheme="random_lecun_1e-3", trial_epochs=3, n=16, recon=str(recon), clip_const=None, clip_min=None, clip_max=None,
              lr_min=None, lr_max=None, lr_log10_min=None, lr_log10_max=None, allowed_pairs=None, box_from=None, refine_from=None,
              confirm_from=None, explore_from=None, confirm_pick_from=None, pin_axis=None, design_salt=0, campaign="two_si",
              model_id="qwen35_4b_demo_base", round=1)
    ns.update(over)
    tune_arm.emit_trials(argparse.Namespace(**ns))
    d = json.loads(out.read_text()); lrs = sorted(float(t["lr"]) for t in d["trials"])
    return d, lrs, capsys.readouterr().out


def test_the_certified_top_lowers_the_box_top_and_the_bottom_stays(tmp_path, monkeypatch, capsys):
    d, lrs, out = _emit(tmp_path, monkeypatch, capsys, lr_log10_max=-2.25)
    assert len(d["trials"]) == 16 and lrs[-1] <= 10 ** -2.25 + 1e-12 and lrs[0] >= 5e-05 - 1e-12, (lrs[0], lrs[-1])   # qwen35_4b_base's seed box: [5e-05, 5e-02]
    assert "lr box clamped to" in out and "-2.250" in out and "2.05 decades" in out and "NOTE: below the spec's 2.5-decade minimum" in out, out   # -4.301 .. -2.25
    assert all(t["max_grad_norm"] == 1.5 for t in d["trials"])                        # the recon's clip constant rides every trial


def test_a_clamp_above_the_box_top_is_a_no_op_and_never_widens(tmp_path, monkeypatch, capsys):
    d0, lrs0, _ = _emit(tmp_path, monkeypatch, capsys)
    d1, lrs1, out = _emit(tmp_path, monkeypatch, capsys, lr_log10_max=-1.0, lr_log10_min=-6.0)
    assert lrs1 == lrs0 and "3.00 decades" in out, out                                 # the seed box stands: qwen35_4b_base's [5e-05, 5e-02] (log10 -4.301 .. -1.301)


def test_a_clamp_below_the_bottom_refuses(tmp_path, monkeypatch, capsys):
    with pytest.raises(SystemExit) as e:
        _emit(tmp_path, monkeypatch, capsys, lr_log10_max=-5.0)
    assert "leaves no box" in str(e.value)
