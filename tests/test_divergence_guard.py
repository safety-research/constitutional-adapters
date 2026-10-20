"""Step-level divergence guard (fit-wave hardening, 2026-08-27): a
clearly-diverging fit must self-terminate in STEPS, not at the per-epoch
early-stop boundary (a doomed qwen psr trial burned ~15h at ~8h/epoch
while its loss visibly exploded inside epoch 1 — 0.002 -> 0.772, 386x).

Pinned here, in the parity-test style (CPU fp32 tiny model, synthetic
French pairs):

- GUARD MATH (no model): a descending/plateau-noisy series never
  triggers; a transient spike shorter than the patience never triggers; a
  sustained explosion triggers via clause A at exactly baseline-crossing
  + patience; a fast strictly-monotone blow-up triggers via clause B; a
  non-finite loss triggers immediately.
- DIVERGING CONFIG DIES IN STEPS: an absurd-lr fit under a small spec
  aborts early — train_lockstep RETURNS with the arm marked
  ``step-divergence`` (partners-train-on contract) and a short history;
  train_psr(mse) RAISES fitmon.TrainingDiverged (the probe_cb abort
  contract: no history returns, the object is never treated as trained).
- HEALTHY CONFIG UNTOUCHED: guard-on (the default) equals guard-off
  BITWISE — final params and full-length history — for both trainers.
- The production fit driver records a step-diverged solo arm FAILED with
  the reason, and ships no artifact.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.fitmon import (
    StepDivergenceGuard,
    StepDivergenceSpec,
    TrainingDiverged,
)
from sisteer.extraction.lockstep import (
    LockstepArm,
    build_lockstep_object,
    default_lockstep_params,
    train_lockstep,
)
from sisteer.extraction.psr import PSRObject, train_psr

INSTRUCTION = BY_ID["french"]
MAX_LEN = 192

_DATA = [
    ("Say something nice.", "Bonjour", "Hello"),
    ("Describe a tree.",
     "C'est un tres grand arbre vert qui pousse lentement dans le jardin",
     "A tall tree"),
    ("Greet the reader.", "Salut a tous mes amis", "Hi everyone"),
    ("Name a color.", "Le bleu", "Blue"),
    ("Count to three.", "Un deux trois et puis encore un peu plus loin",
     "One two three"),
]


def _pairs():
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in _DATA]


def _examples(tok):
    return pairs_to_distill_examples(tok, _pairs(), INSTRUCTION)


#: A spec small enough that a tiny CPU fit can reach a verdict, but with
#: every clause still active (median smoothing, warmup, patience).
_FAST_SPEC = StepDivergenceSpec(factor=5.0, patience=3, smooth=2,
                                rise_window=3, rise_factor=10.0,
                                warmup_steps=3)


# -- guard math (no model) -----------------------------------------------------


def _feed(guard, series):
    for i, v in enumerate(series, 1):
        guard.step(i, v)


def test_guard_healthy_series_never_triggers():
    """A noisy descent to a plateau (the healthy shape, with the live
    canary's ~10x per-step spread simulated by a x0.3..x3 cycle around the
    trend) runs 500 steps without a verdict at the PRODUCTION spec."""
    g = StepDivergenceGuard(StepDivergenceSpec(), tag="healthy")
    jitter = [0.4, 1.0, 2.8, 0.7, 1.6]
    series = [(1.0 / (1 + 0.05 * i) + 0.05) * jitter[i % 5]
              for i in range(500)]
    _feed(g, series)  # must not raise


def test_guard_transient_spike_survives_sustained_explosion_dies():
    """At smooth=1 (so the arithmetic is exact): a spike of patience-1
    steps over the bar resets and survives; the same spike one step longer
    is the verdict — K x baseline for M CONSECUTIVE steps, no fewer."""
    spec = StepDivergenceSpec(factor=10.0, patience=5, smooth=1,
                              rise_window=8, rise_factor=1e9,
                              warmup_steps=1)
    base = [1.0, 0.9, 0.8, 0.9, 0.85]  # establishes best=0.8
    g = StepDivergenceGuard(spec)
    _feed(g, base + [9.0] * 4 + [0.9] + [9.0] * 4 + [0.9])  # spikes < M: ok
    g2 = StepDivergenceGuard(spec)
    with pytest.raises(TrainingDiverged, match="step-divergence"):
        _feed(g2, base + [9.0] * 5)  # the 5th consecutive step rules
    with pytest.raises(TrainingDiverged, match="consecutive"):
        _feed(StepDivergenceGuard(spec), base + [9.0] * 5)


def test_guard_monotone_blowup_clause_b():
    spec = StepDivergenceSpec(factor=1e9, patience=10**6, smooth=1,
                              rise_window=4, rise_factor=8.0,
                              warmup_steps=1)
    g = StepDivergenceGuard(spec)
    with pytest.raises(TrainingDiverged, match="monotonically"):
        _feed(g, [0.1, 0.3, 0.9, 2.7])  # 27x, strictly rising over 4


def test_guard_nonfinite_is_immediate():
    g = StepDivergenceGuard(StepDivergenceSpec(), tag="nan")
    with pytest.raises(TrainingDiverged, match="non-finite"):
        g.step(1, float("nan"))
    g2 = StepDivergenceGuard(StepDivergenceSpec())
    with pytest.raises(TrainingDiverged, match="non-finite"):
        g2.step(1, math.inf)


def test_guard_no_verdicts_during_warmup():
    """The first warmup steps of a fit that STARTS high and descends can
    never self-condemn: the baseline is its own early values."""
    g = StepDivergenceGuard(StepDivergenceSpec())
    _feed(g, [50.0, 40.0, 30.0, 20.0, 10.0, 5.0, 2.0, 1.0, 0.5, 0.4])


def test_flag_resolver_vocabulary():
    from sisteer.extraction.fitmon import resolve_divergence_guard

    assert resolve_divergence_guard(False, tag="t") is None
    assert resolve_divergence_guard(None, tag="t") is not None  # default ON
    assert resolve_divergence_guard(True, tag="t") is not None
    assert resolve_divergence_guard(_FAST_SPEC, tag="t").spec is _FAST_SPEC
    with pytest.raises(ValueError, match="divergence_guard"):
        resolve_divergence_guard("yes", tag="t")


# -- train_lockstep: diverging arm dies in steps, marked, isolated -------------


def _run_lockstep(model, tok, *, lr, guard, epochs=6, corrupt_at=None):
    """``corrupt_at=k``: after optimizer step k, blow the object's params
    up x1e4 through the probe seam — the REALISTIC divergence shape
    (healthy descent establishing a baseline, THEN explosion; the qwen
    psr failure's 0.002 -> 0.772 profile). A from-the-start bounded
    garbage plateau is deliberately NOT a step-divergence (its own values
    are the baseline; the epoch-level early stop + floor guard own it)."""
    obj = build_lockstep_object("lora_r1_all", model, tok, INSTRUCTION,
                                _pairs(), layer=1)
    params = replace(default_lockstep_params("lora_r1_all"), lr=lr)
    diag: dict = {}

    def cb(step_i, name, obj_, loss):
        cb.n = getattr(cb, "n", 0) + 1
        if corrupt_at is not None and cb.n == corrupt_at:
            with torch.no_grad():
                for p in obj_.trainable:
                    p.data.mul_(1e4)

    hist = train_lockstep(model, tok,
                          [LockstepArm("lora_r1_all", obj, params)],
                          _examples(tok), epochs=epochs, batch_size=2,
                          max_len=MAX_LEN, seed=0, n_probe=2,
                          probe_cb=cb, diagnostics=diag,
                          divergence_guard=guard)
    return (hist["lora_r1_all"], diag["lora_r1_all"],
            [p.detach().clone() for p in obj.trainable])


def test_lockstep_diverging_config_dies_in_steps(tiny_llama):
    """A healthy fit whose object explodes after step 6 (params x1e4 —
    loss leaves its established baseline for garbage) is marked
    ``step-divergence`` within the fast spec's ~3-step patience and STOPS
    well short of its 18 scheduled optimizer steps — train_lockstep
    RETURNS (the partners-train-on contract), the diagnostics carry the
    reason, and no artifact ships (diverged => failed, the pre-existing
    driver path)."""
    model, tok = tiny_llama
    hist, d, _ = _run_lockstep(model, tok, lr=5e-3, guard=_FAST_SPEC,
                               corrupt_at=6)
    assert d.get("diverged"), "the guard never ruled on the blow-up"
    assert d["diverged"].startswith("step-divergence")
    assert d["stopped_early"] is True
    assert d["final_probe_loss"] is None  # never probed as trained
    assert 6 < len(hist) < 14, (
        f"arm took {len(hist)} of 18 steps — expected an abort within a "
        "few steps of the step-6 explosion")


def test_lockstep_healthy_run_bitwise_untouched(tiny_llama):
    """Default-ON guard vs guard OFF on a healthy fit: identical history
    (float-exact) and bitwise-identical trained params — the guard is
    observational."""
    model, tok = tiny_llama
    hist_off, d_off, p_off = _run_lockstep(model, tok, lr=5e-3,
                                           guard=False, epochs=3)
    hist_on, d_on, p_on = _run_lockstep(model, tok, lr=5e-3,
                                        guard=None, epochs=3)
    assert not d_on.get("diverged")
    assert hist_on == hist_off and len(hist_on) == 9  # full length
    for a, b in zip(p_on, p_off):
        assert torch.equal(a, b)
    assert sum(float(p.abs().sum()) for p in p_on) > 0.0


# -- train_psr (mse): raises the documented abort -------------------------------


def _run_psr(model, tok, *, lr, guard, epochs=6):
    obj = PSRObject(model, [1], train_direction=True)
    hist = train_psr(model, tok, obj, _examples(tok), loss="mse",
                     epochs=epochs, batch_size=2, max_len=MAX_LEN, seed=0,
                     lr=lr, lr_schedule="constant", max_grad_norm=math.inf,
                     divergence_guard=guard)
    return hist, [p.detach().clone() for p in obj.trainable]


def test_psr_mse_diverging_config_raises_in_steps(tiny_llama):
    model, tok = tiny_llama
    with pytest.raises(TrainingDiverged, match="step-divergence"):
        _run_psr(model, tok, lr=500.0, guard=_FAST_SPEC)


def test_psr_mse_healthy_run_bitwise_untouched(tiny_llama):
    model, tok = tiny_llama
    hist_off, p_off = _run_psr(model, tok, lr=5e-3, guard=False, epochs=3)
    hist_on, p_on = _run_psr(model, tok, lr=5e-3, guard=None, epochs=3)
    assert hist_on == hist_off and len(hist_on) == 9
    for a, b in zip(p_on, p_off):
        assert torch.equal(a, b)


def test_psr_kl_path_refuses_explicit_guard_setting():
    """The KL path's optimizer loop lives in train_object_kl; an explicit
    divergence_guard there would be silently inert — refuse loudly. The
    None default passes through (the production KL arms are guarded via
    train_lockstep)."""
    with pytest.raises(ValueError, match="divergence_guard"):
        train_psr(None, None, None, [], loss="kl", divergence_guard=False)


# -- driver surfacing -----------------------------------------------------------


import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_t2_divguard", REPO / "scripts/train_two_si_objects.py")
t2 = importlib.util.module_from_spec(_spec)
sys.modules["_t2_divguard"] = t2
_spec.loader.exec_module(t2)

SI = "safety_refusal"


def test_driver_records_step_diverged_solo_arm_failed(tiny_llama, tmp_path):
    """A step-diverged solo mse arm is recorded FAILED with the
    step-divergence reason and publishes NO cell — the floor-fail-
    equivalent terminal state. The spec rides train_kwargs.<arm> (a
    PROTECTED trainer kwarg), which is also the per-arm escape hatch
    (divergence_guard: false)."""
    model, tok = tiny_llama
    cfg = {
        "layer": 1, "completion_tokens": 16, "max_len": 256,
        "max_len_by_si": {SI: 256},
        "teacher_prefix_cache": False,
        "batch_size_by_arm": {"psr_mse_all": 2},
        "train_kwargs": {
            "default": {"epochs": 6, "seed": 0, "holdout_frac": 0.0,
                        "floor_ratio_max": None, "probe_n": 2,
                        "monitors": False},
            "psr_mse_all": {"lr": 500.0, "lr_schedule": "constant",
                            "divergence_guard": _FAST_SPEC},
        },
        "si_ids": [SI],
    }
    pairs = [Pair(prompt=p, pos=pos, neg=neg, instruction_id=SI)
             for p, pos, neg in _DATA]
    outcome = t2.train_si_objects(model, tok, SI, pairs, ["psr_mse_all"],
                                  cfg=cfg, save_dir=tmp_path)
    res = outcome.arms["psr_mse_all"]
    assert res.status == "failed"
    assert res.error_type == "TrainingDiverged"
    assert "step-divergence" in (res.error or "")
    assert t2.validated_complete(tmp_path, f"{SI}__psr_mse_all", "x") is None
    assert not (tmp_path / f"{SI}__psr_mse_all").exists()
