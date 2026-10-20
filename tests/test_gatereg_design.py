"""The round-2 gate_reg mini-sweep design.

Pins: the bespoke emitter's payload (6-point grid over gate_reg with the
winning lr/clip/warmup FIXED, read verbatim from the committed folded
config; identity fields run_trial validates; the pinned data cell
constv2-mix-1m); the gate_reg seam through tune_arm (TrialHP carries it,
the trial fingerprint distinguishes it, CESoloAdapter threads it into the
PSR trainer's reg_coeff and REFUSES it on the lora arm, run_trial refuses
it on non-CE arms and refuses an --si contradicting the design's pin).

CPU-only; no model, no network (the trainer is monkeypatched where the
threading is exercised).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "scripts", REPO / "scripts" / "sdfbase"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import emit_gatereg_trials as egt  # noqa: E402
import tune_arm  # noqa: E402

GEMMA_CFG = REPO / "configs/sdfbase_gemma4e4b_base_fits.yaml"


# --------------------------------------------------------------------------
# The emitter payload
# --------------------------------------------------------------------------

def test_r3_design_header_states_the_analyze_side_gap():
    """The round-3 design header declares (rather than papers over) that
    tune_arm's analyze path does not consume gate_reg: the analyze-side
    tuples stay 3-wide, so the §4.6 read comes from the trial summaries,
    never from an analyze winner. Pinned so the gap cannot silently become
    an undocumented surprise; TRIAL_EXTRA_AXES stays trial-side."""
    doc = " ".join(egt.__doc__.split())  # unwrap the docstring's lines
    assert "analyze-side gap" in doc and "3-wide" in doc
    assert 'must NOT be used to pick a gate_reg "winner"' in doc
    assert "gate_reg" in tune_arm.TRIAL_EXTRA_AXES
    analyze_src = (REPO / "scripts/tune_arm.py").read_text().split(
        "def analyze(", 1)[1].split("\ndef ", 1)[0]
    assert "gate_reg" not in analyze_src  # the gap is real, as stated


def test_payload_grid_and_fixed_winners():
    payload = egt.build_gatereg_trials(GEMMA_CFG)
    cfg = yaml.safe_load(GEMMA_CFG.read_text())
    block = cfg["train_kwargs"]["psr_ce_all"]
    assert payload["grid"] == [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
    assert payload["axes"] == ["gate_reg"]
    assert payload["si"] == "constv2-mix-1m"
    assert payload["arm"] == "psr_ce_all"
    assert payload["position"] == "all"
    assert payload["round"] == 3
    # the round-2 caps ride along for the operator's cost math
    assert payload["epochs_cap"] == {"epochs": 24, "early_stop_patience": 3}
    trials = payload["trials"]
    assert [t["gate_reg"] for t in trials] == payload["grid"]
    assert [t["index"] for t in trials] == list(range(6))
    for t in trials:
        # the winners, verbatim from the committed folded config — FIXED
        assert t["lr"] == float(block["lr"])
        assert t["max_grad_norm"] == float(block["max_grad_norm"])
        assert t["warmup_frac"] == float(block["warmup_frac"])
    # 0.1 (the campaign default) is in the grid: the in-design control
    assert 0.1 in payload["grid"]


def test_payload_identity_validates_against_config():
    """run_trial's _validate_trials_identity accepts the payload against
    the exact config it was emitted from, and refuses a drifted one."""
    payload = egt.build_gatereg_trials(GEMMA_CFG)
    cfg = yaml.safe_load(GEMMA_CFG.read_text())
    assert payload["model"] == cfg["model"]
    assert payload["model_revision"] == cfg["model_revision"]
    # the emitter re-states run_trial's own config-sha convention
    assert payload["config_sha"] == tune_arm._config_file_sha(str(GEMMA_CFG))
    tune_arm._validate_trials_identity(payload, cfg, str(GEMMA_CFG))
    with pytest.raises(SystemExit, match="does not match"):
        tune_arm._validate_trials_identity(
            {**payload, "config_sha": "0" * 16}, cfg, str(GEMMA_CFG))


def test_emitter_refusals(tmp_path):
    with pytest.raises(SystemExit, match="psr_ce_all"):
        egt.build_gatereg_trials(GEMMA_CFG, arm="lora_r4_ce")
    with pytest.raises(SystemExit, match="train_kwargs_by_si"):
        egt.build_gatereg_trials(GEMMA_CFG, si="constv2-mix-999k")
    # a config whose winner slots still carry HPO sentinels cannot emit
    cfg = yaml.safe_load(GEMMA_CFG.read_text())
    cfg["train_kwargs"]["psr_ce_all"]["lr"] = "PENDING_SDFBASE_HPO_WINNER"
    p = tmp_path / "sdfbase_unfolded.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError):
        egt.build_gatereg_trials(p)


def test_emitter_cli_writes_hpo_spawn_shaped_file(tmp_path):
    out = tmp_path / "r3_gemma4e4b_base_psr_ce_all_all.json"
    assert egt.main(["--out", str(out)]) == 0
    payload = json.loads(out.read_text())
    assert len(payload["trials"]) == 6
    # hpo-spawn reads len(payload["trials"]) and payload.get("si")
    assert payload["si"] == "constv2-mix-1m"


# --------------------------------------------------------------------------
# The gate_reg seam through tune_arm
# --------------------------------------------------------------------------

def _fp_args(tmp_path):
    pool = tmp_path / "pairs_constv2-mix-1m.jsonl"
    pool.write_text('{"prompt": "", "pos": "x", "neg": ""}\n')
    held = tmp_path / "pairs_constv2-heldout.jsonl"
    held.write_text('{"prompt": "", "pos": "y", "neg": ""}\n')
    args = SimpleNamespace(pairs=str(pool), batch_size=1, epochs=24,
                           fit_subsample=0, si="constv2-mix-1m",
                           teacher_topk="off")
    cfg = {"model": "google/gemma-4-E4B", "model_revision": "r" * 40,
           "train_kwargs": {"default": {"heldout_pool": str(held)}}}
    return args, cfg


def test_trial_fingerprint_distinguishes_gate_reg(tmp_path):
    """Two designs differing only in gate_reg at the same index must never
    alias to the same fit identity (the axis is part of the optimizer
    mapping, like the re-tune axes)."""
    args, cfg = _fp_args(tmp_path)
    adapter = tune_arm.make_adapter("psr_ce_all")
    spec = {"config_sha": "a" * 16, "round": 3,
            "box": {"lr": [1e-4, 1e-4]}, "warmup_levels": [0.05]}
    base = {"index": 0, "lr": 1e-4, "warmup_frac": 0.05,
            "max_grad_norm": 0.1}
    fp = {v: tune_arm._trial_fingerprint(
              args, cfg, spec, {**base, "gate_reg": v}, "psr_ce_all",
              "all", adapter)
          for v in (0.01, 3.0)}
    assert fp[0.01] != fp[3.0]
    no_axis = tune_arm._trial_fingerprint(args, cfg, spec, base,
                                          "psr_ce_all", "all", adapter)
    assert no_axis not in fp.values()


def test_run_trial_refuses_gate_reg_on_non_ce_arm(tmp_path):
    tf = tmp_path / "trials.json"
    tf.write_text(json.dumps({
        "round": 3, "arm": "psr_kl_all", "position": "all",
        "trials": [{"index": 0, "lr": 1e-4, "warmup_frac": 0.05,
                    "max_grad_norm": 0.1, "gate_reg": 0.3}]}))
    args = SimpleNamespace(trials=str(tf), index=0, arm=None,
                           positions=None, si="constv2-mix-1m",
                           config="/nonexistent.yaml", out=str(tmp_path))
    with pytest.raises(SystemExit, match="gate_reg"):
        tune_arm.run_trial(args)


def test_run_trial_refuses_si_contradicting_design_pin(tmp_path):
    tf = tmp_path / "trials.json"
    tf.write_text(json.dumps({
        "round": 3, "arm": "psr_ce_all", "position": "all",
        "si": "constv2-mix-1m",
        "trials": [{"index": 0, "lr": 1e-4, "warmup_frac": 0.05,
                    "max_grad_norm": 0.1, "gate_reg": 0.3}]}))
    args = SimpleNamespace(trials=str(tf), index=0, arm=None,
                           positions=None, si="constv2-mix-100k",
                           config="/nonexistent.yaml", out=str(tmp_path))
    with pytest.raises(SystemExit, match="pinned data cell"):
        tune_arm.run_trial(args)


def test_ce_adapter_refuses_gate_reg_on_lora_arm():
    ad = tune_arm.make_adapter("lora_r4_ce")
    hp = tune_arm.TrialHP(lr=1e-4, warmup_frac=0.05, max_grad_norm=0.1,
                          gate_reg=0.3)
    with pytest.raises(SystemExit, match="gate_reg"):
        ad.train(SimpleNamespace(), hp, epochs=1, batch_size=1, seed=0,
                 log_fn=None, probe_cb=None, probe_every=0,
                 diagnostics=None)


@pytest.mark.parametrize("swept,expected", [(0.3, 0.3), (None, 0.1)])
def test_ce_adapter_threads_gate_reg_into_reg_coeff(monkeypatch, swept,
                                                    expected):
    """hp.gate_reg overrides the config's train_kwargs.psr_ce_all.gate_reg
    (0.1) as the PSR trainer's reg_coeff; an unswept trial keeps the
    config value. The trainer is monkeypatched to a recorder — the seam
    under test is the kwarg threading, not the optimization."""
    import sisteer.extraction.objects as objects_mod

    seen = {}

    def recorder(model, tok, obj, examples, **kw):
        seen.update(kw)
        return []

    monkeypatch.setattr(objects_mod, "train_object_ce", recorder)
    ad = tune_arm.make_adapter("psr_ce_all")
    cfg = yaml.safe_load(GEMMA_CFG.read_text())
    setup = SimpleNamespace(cfg=cfg, model=None, tok=None,
                            si="constv2-mix-1m", max_len=8192,
                            positions="all", response_header_len=None)
    ad._fit_ex = ["ex"] * 8
    ad._n_fit = 8
    ad._held_ex = []
    ad._obj = SimpleNamespace(trainable=[])
    hp = tune_arm.TrialHP(lr=1e-4, warmup_frac=0.05, max_grad_norm=0.1,
                          gate_reg=swept)
    ad.train(setup, hp, epochs=2, batch_size=2, seed=0, log_fn=None,
             probe_cb=None, probe_every=0, diagnostics={})
    assert seen["reg_coeff"] == pytest.approx(expected)
    assert seen["lr"] == pytest.approx(1e-4)
    assert seen["max_grad_norm"] == pytest.approx(0.1)
