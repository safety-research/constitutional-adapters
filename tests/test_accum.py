"""Gradient accumulation for the non-bidpo trainers (re-tune@32,
2026-08-27): train_lockstep (the KL arms) and train_psr's mse loop.

The load-bearing contract, pinned here with the two traps it must not
fall into:

- EQUIVALENCE: accum_steps=N at micro-batch M produces the same gradients
  — and therefore the same post-step object — as ONE step at batch M x N,
  to fp32 tolerance. The parity tests use RAGGED groups (micro-batches
  with different row AND completion-token counts) on purpose: with
  equal-sized micro-batches a flat 1/N average is indistinguishable from
  the correct per-denominator share weighting (completion-token share for
  the divergence/mse, row share for the gate regularizer), so only a
  ragged group can catch the mis-weighting.
- CLIP ONCE, ON THE ACCUMULATED GRADIENT: the parity runs use a
  max_grad_norm BELOW the measured unclipped norm (asserted active),
  because with clipping off, per-micro-batch clipping and global clipping
  are indistinguishable — an inactive clip would let the
  bs=2-granularity-clipping regression slip through.
- accum_steps=1 is byte-identical to the historical loop (the 30 bitwise
  solo==lockstep tests in test_lockstep.py run against this same code and
  stay the primary pin; here the structural knobs are validated).

CPU-only: tiny_llama fixture, synthetic French pairs.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction.lockstep import (
    LockstepArm,
    _encode_arm_batches,
    build_lockstep_object,
    default_lockstep_params,
    train_lockstep,
)
from sisteer.extraction.psr import PSRObject, train_psr

INSTRUCTION = BY_ID["french"]
MAX_LEN = 192

# Deliberately RAGGED completions (2 to ~14 tokens) so micro-batches carry
# different token counts — the weighting trap's precondition.
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


def _grad_norm(grads) -> float:
    return float(torch.norm(torch.stack([g.float().norm() for g in grads])))


# -- train_lockstep ------------------------------------------------------------


def _run_lockstep(model, tok, arm, *, batch_size, accum, clip,
                  diagnostics=None):
    examples = _examples(tok)
    obj = build_lockstep_object(arm, model, tok, INSTRUCTION, _pairs(),
                                layer=1)
    params = replace(default_lockstep_params(arm), accum_steps=accum,
                     max_grad_norm=clip)
    captured = {}

    def cb(step, name, obj_, loss):
        captured["grads"] = [p.grad.detach().clone()
                             for p in obj_.trainable]
        captured["loss"] = float(loss)

    train_lockstep(model, tok, [LockstepArm(arm, obj, params)], examples,
                   epochs=1, batch_size=batch_size, max_len=MAX_LEN,
                   seed=0, probe_cb=cb, n_probe=2,
                   diagnostics=diagnostics)
    return captured, [p.detach().clone() for p in obj.trainable]


@pytest.mark.parametrize("arm", ["lora_r1_all", "psr_kl_all"])
def test_lockstep_accum_matches_one_true_batch(tiny_llama, arm):
    """accum(bs=2 x 3 micro-batches: rows 2/2/1, ragged token counts) ==
    one bs=5 step — gradients, post-step params, and the logged loss —
    under an ACTIVE clip. psr_kl_all additionally carries the gate
    regularizer (reg_coeff=0.1), whose ROW-share weighting a ragged group
    exercises against the divergence's TOKEN-share weighting."""
    model, tok = tiny_llama

    # Ragged premise, asserted not assumed: the three micro-batches carry
    # different completion-token counts (and the last a different row
    # count), so flat-1/N averaging or per-micro clipping cannot pass.
    class _Id:
        def inject(self, s):
            return s

    _t, s_batches = _encode_arm_batches(tok, _Id(), _examples(tok),
                                        max_len=MAX_LEN, batch_size=2)
    counts = [int((sb.labels[:, 1:] != -100).sum()) for sb in s_batches]
    rows = [int(sb.input_ids.shape[0]) for sb in s_batches]
    assert len(counts) == 3 and len(set(counts)) > 1 and rows == [2, 2, 1]

    # Measure the unclipped accumulated norm, then clip BELOW it so the
    # clip is provably active in both parity runs.
    unclipped, _ = _run_lockstep(model, tok, arm, batch_size=2, accum=3,
                                 clip=None)
    n0 = _grad_norm(unclipped["grads"])
    assert n0 > 0.0
    clip = n0 / 2.0

    big, big_params = _run_lockstep(model, tok, arm, batch_size=5, accum=1,
                                    clip=clip)
    acc, acc_params = _run_lockstep(model, tok, arm, batch_size=2, accum=3,
                                    clip=clip)
    # The clip really fired: the post-clip norm sits at the bound, well
    # under the unclipped norm.
    assert _grad_norm(acc["grads"]) < 0.75 * n0
    assert len(big["grads"]) == len(acc["grads"]) > 0
    for g_b, g_a in zip(big["grads"], acc["grads"]):
        assert torch.allclose(g_b, g_a, atol=1e-6, rtol=1e-5), (
            f"{arm}: accumulated gradient differs from the true batch "
            f"(max abs diff {float((g_b - g_a).abs().max()):.3e})")
    for p_b, p_a in zip(big_params, acc_params):
        assert torch.allclose(p_b, p_a, atol=1e-6, rtol=1e-5)
    assert abs(big["loss"] - acc["loss"]) < 1e-5
    assert sum(float(p.abs().sum()) for p in big_params) > 0.0


def test_lockstep_accum_ragged_tail_and_schedule_budget(tiny_llama):
    """5 pairs at bs=2 -> 3 micro-batches; accum=2 -> groups of [2, 1] ->
    2 optimizer steps per epoch (the ragged tail is its own smaller
    step), and a warmup_frac resolves against the OPTIMIZER-step budget
    (epochs x ceil(batches/accum)), never the micro-batch count."""
    model, tok = tiny_llama
    examples = _examples(tok)
    obj = build_lockstep_object("lora_r1", model, tok, INSTRUCTION,
                                _pairs(), layer=1)
    params = replace(default_lockstep_params("lora_r1"), accum_steps=2,
                     lr_schedule="warmup_cosine", warmup_frac=0.5)
    diag = {}
    hist = train_lockstep(model, tok, [LockstepArm("lora_r1", obj, params)],
                          examples, epochs=2, batch_size=2, max_len=MAX_LEN,
                          seed=0, n_probe=2, diagnostics=diag)
    assert len(hist["lora_r1"]) == 4  # 2 optimizer steps x 2 epochs
    d = diag["lora_r1"]
    assert d["accum_steps"] == 2
    assert d["optimizer_steps"] == 4
    # 0.5 of the 4-optimizer-step budget, not 0.5 of 6 micro-batches.
    assert d["resolved_warmup_steps"] == 2


def test_lockstep_accum_validates_loudly(tiny_llama):
    model, tok = tiny_llama
    examples = _examples(tok)
    obj = build_lockstep_object("lora_r1", model, tok, INSTRUCTION,
                                _pairs(), layer=1)
    params = replace(default_lockstep_params("lora_r1"), accum_steps=0)
    with pytest.raises(ValueError, match="accum_steps"):
        train_lockstep(model, tok, [LockstepArm("lora_r1", obj, params)],
                       examples, epochs=1, batch_size=2, max_len=MAX_LEN,
                       seed=0)


# -- train_psr (mse loop) ------------------------------------------------------


def _run_psr_mse(model, tok, *, batch_size, accum, clip):
    examples = _examples(tok)
    obj = PSRObject(model, [1], train_direction=True)
    captured = {}

    def cb(step, o):
        captured["grads"] = [p.grad.detach().clone() for p in o.trainable]

    hist = train_psr(model, tok, obj, examples, loss="mse", epochs=1,
                     batch_size=batch_size, max_len=MAX_LEN, seed=0,
                     lr_schedule="constant", max_grad_norm=clip,
                     accum_steps=accum, probe_cb=cb, probe_every=1)
    return captured, [p.detach().clone() for p in obj.trainable], hist


def test_psr_mse_accum_matches_one_true_batch(tiny_llama):
    """Same contract for train_psr's mse loop: ragged accumulation groups
    (mse token-share + regularizer row-share weighting) under an ACTIVE
    clip equal one true batch's step."""
    model, tok = tiny_llama
    unclipped, _, _ = _run_psr_mse(model, tok, batch_size=2, accum=3,
                                   clip=math.inf)
    n0 = _grad_norm(unclipped["grads"])
    assert n0 > 0.0
    clip = n0 / 2.0

    big, big_params, big_hist = _run_psr_mse(model, tok, batch_size=5,
                                             accum=1, clip=clip)
    acc, acc_params, acc_hist = _run_psr_mse(model, tok, batch_size=2,
                                             accum=3, clip=clip)
    assert _grad_norm(acc["grads"]) < 0.75 * n0  # the clip really fired
    for g_b, g_a in zip(big["grads"], acc["grads"]):
        assert torch.allclose(g_b, g_a, atol=1e-6, rtol=1e-5), (
            "psr mse: accumulated gradient differs from the true batch "
            f"(max abs diff {float((g_b - g_a).abs().max()):.3e})")
    for p_b, p_a in zip(big_params, acc_params):
        assert torch.allclose(p_b, p_a, atol=1e-6, rtol=1e-5)
    assert len(big_hist) == len(acc_hist) == 1
    assert abs(big_hist[0] - acc_hist[0]) < 1e-5


def test_psr_mse_accum_step_counts_and_validation(tiny_llama):
    model, tok = tiny_llama
    examples = _examples(tok)
    # 3 micro-batches at bs=2, accum=2 -> 2 optimizer steps per epoch
    # (ragged tail steps alone).
    obj = PSRObject(model, [1], train_direction=True)
    hist = train_psr(model, tok, obj, examples, loss="mse", epochs=1,
                     batch_size=2, max_len=MAX_LEN, seed=0,
                     lr_schedule="constant", max_grad_norm=math.inf,
                     accum_steps=2)
    assert len(hist) == 2
    with pytest.raises(ValueError, match="accum_steps"):
        train_psr(model, tok, PSRObject(model, [1], train_direction=True),
                  examples, loss="mse", epochs=1, batch_size=2,
                  max_len=MAX_LEN, accum_steps=0)


def test_psr_kl_path_refuses_accum():
    """The KL path delegates its optimizer loop to train_object_kl (and
    production KL arms accumulate through lockstep) — a silently dropped
    accum_steps is the 3dd9e36 bug class, so it must refuse loudly. The
    refusal fires before any model work, so dummies suffice."""
    with pytest.raises(ValueError, match="accum_steps"):
        train_psr(None, None, None, [], loss="kl", accum_steps=2)


# -- driver wiring (train_two_si_objects.py) -----------------------------------


import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_t2_accum", REPO / "scripts/train_two_si_objects.py")
t2 = importlib.util.module_from_spec(_spec)
sys.modules["_t2_accum"] = t2
_spec.loader.exec_module(t2)

SI = "safety_refusal"


def _si_pairs():
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id=SI)
            for p, pos, neg in _DATA]


def _driver_cfg(arm: str, arm_tk: dict) -> dict:
    return {
        "layer": 1, "completion_tokens": 16, "max_len": 256,
        "max_len_by_si": {SI: 256},
        "teacher_prefix_cache": False,
        "batch_size_by_arm": {arm: 2},
        "train_kwargs": {
            "default": {"epochs": 1, "seed": 0, "holdout_frac": 0.0,
                        "floor_ratio_max": None, "probe_n": 2,
                        "monitors": False},
            arm: dict(arm_tk),
        },
        "si_ids": [SI],
    }


def test_allowlist_and_dropped_keys_guard_take_the_new_keys():
    """accum_steps + weight_decay are LOCKSTEP optimizer keys now: the
    params merge picks them up (typed) and the dropped-keys guard accepts
    them — while a genuinely unknown key still refuses."""
    assert "accum_steps" in t2._LOCKSTEP_OPTIMIZER_KEYS
    assert "weight_decay" in t2._LOCKSTEP_OPTIMIZER_KEYS
    cfg = {"train_kwargs": {"psr_kl_all": {"accum_steps": "16",
                                           "weight_decay": "0.05"}}}
    params = t2.lockstep_arm_params(cfg, "psr_kl_all")
    assert params.accum_steps == 16 and isinstance(params.accum_steps, int)
    assert params.weight_decay == 0.05
    t2.refuse_silently_dropped_kwargs(cfg, ["psr_kl_all"])  # no raise
    bad = {"train_kwargs": {"psr_kl_all": {"accum_steppes": 16}}}
    with pytest.raises(SystemExit, match="accum_steppes"):
        t2.refuse_silently_dropped_kwargs(bad, ["psr_kl_all"])


def test_fingerprint_distinguishes_accum_and_weight_decay():
    """A cell trained at effective batch 32 (or another weight decay) is a
    DIFFERENT identity: its fingerprint must not match a marker written at
    the old settings, in BOTH training modes."""
    base = {"model": "m", "model_revision": "r"}
    for arm in ("psr_kl_all",   # lockstep mode
                "psr_mse_all"):  # solo mode
        fps = set()
        for tk in ({}, {"accum_steps": 16}, {"weight_decay": 0.0}):
            cfg = dict(base, train_kwargs={arm: dict(tk)})
            fps.add(t2.cell_fit_fingerprint(
                cfg, SI, arm, positions="response",
                pool_sha256="p", config_sha="c"))
        assert len(fps) == 3, arm


def test_effective_settings_report_accum_for_lockstep_arms():
    eff = t2.effective_optimizer_settings(
        {"train_kwargs": {"lora_r1_all": {"accum_steps": 8}}},
        "lora_r1_all", lockstep=True)
    assert eff["accum_steps"] == (8, "config")
    assert eff["weight_decay"][0] == 0.01  # the exposed AdamW default


def test_driver_end_to_end_lockstep_accum(tiny_llama):
    """train_kwargs.<lockstep arm>.accum_steps flows config -> params ->
    train_lockstep and the fit record says what really ran."""
    model, tok = tiny_llama
    cfg = _driver_cfg("lora_r1_all", {"accum_steps": 2})
    outcome = t2.train_si_objects(model, tok, SI, _si_pairs(),
                                  ["lora_r1_all"], cfg=cfg, save_dir=None)
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
    detail = outcome.arms["lora_r1_all"].detail
    assert detail["accum_steps"] == 2
    # 3 micro-batches (5 pairs at bs=2) in groups of 2 -> 2 optimizer
    # steps for the single epoch (ragged tail steps alone).
    assert detail["optimizer_steps"] == 2


def test_driver_end_to_end_solo_psr_accum_and_weight_decay(tiny_llama):
    """The solo path forwards accum_steps + weight_decay to train_psr as
    PROTECTED kwargs — a trainer that could not take them would fail the
    arm loudly, so outcome.ok proves the wiring."""
    model, tok = tiny_llama
    cfg = _driver_cfg("psr_mse_all", {"accum_steps": 2,
                                      "weight_decay": 0.0,
                                      "lr_schedule": "constant"})
    outcome = t2.train_si_objects(model, tok, SI, _si_pairs(),
                                  ["psr_mse_all"], cfg=cfg, save_dir=None)
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
