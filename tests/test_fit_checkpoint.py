"""Mid-fit checkpoint + exact auto-resume (fit-wave durability,
2026-08-27): train_lockstep and train_psr's mse loop write a full-state
snapshot every checkpoint_every optimizer steps and resume from it
EXACTLY — the trainers build batches deterministically, so resume =
restore all state, recompute batch composition, skip to the checkpointed
batch, continue.

The contract pinned here, per trainer family (test_ckpt_grad_parity /
test_accum style — CPU fp32 tiny model, ragged batches):

- INTERRUPT/RESUME PARITY: train uninterrupted to completion; separately
  train to step k, kill (a raise out of the probe seam), rebuild fresh
  trainer state, resume from the checkpoint, run to completion — the
  final object params and the full loss history must match the
  uninterrupted run BITWISE (fp32/CPU). The parity runs cross an epoch
  boundary, carry an ACTIVE max_grad_norm (clip set below the measured
  unclipped norm, the test_accum trick), and use ragged batches (unequal
  row and completion-token counts), so a resume that mis-restores the
  schedule position, optimizer state, RNG streams, or accumulation-group
  arithmetic cannot pass.
- EARLY-STOP SURVIVAL (lockstep): a huge min_delta pins the best epoch to
  the FIRST held-out eval, before the kill point — so the final restored
  weights are the epoch-1 snapshot, which only the CHECKPOINTED stopper
  state can supply. A resume that rebuilt the stopper fresh would restore
  a post-resume snapshot and fail the bitwise compare.
- KEY MISMATCH = FRESH START: a checkpoint written under a different
  checkpoint_key is IGNORED (loudly), never loaded, never a crash — the
  run trains from scratch and matches an uncheckpointed fresh run.
- DEFAULT OFF = BYTE-IDENTITY: checkpoint kwargs at their None defaults
  leave both loops byte-identical to before the seam existed, and a
  checkpointing-ON uninterrupted run equals the OFF run bitwise (the
  write serializes copies; it must not perturb the trajectory). The
  existing solo==lockstep bitwise suite (test_lockstep.py) and the accum
  parity suite (test_accum.py) run against this same code and stay the
  primary pins.

CPU-only: tiny_llama fixture, synthetic French pairs.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest
import torch

from sisteer.data.instructions import BY_ID
from sisteer.data.pairs import Pair, pairs_to_distill_examples
from sisteer.extraction import fitckpt
from sisteer.extraction.distill import EarlyStopSpec
from sisteer.extraction.lockstep import (
    LockstepArm,
    build_lockstep_object,
    default_lockstep_params,
    train_lockstep,
)
from sisteer.extraction.psr import PSRObject, train_psr

INSTRUCTION = BY_ID["french"]
MAX_LEN = 192

# Deliberately RAGGED completions (test_accum's pool): micro-batches carry
# different row AND completion-token counts.
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

_HELDOUT_DATA = [
    ("Say goodbye.", "Au revoir mes amis", "Goodbye"),
    ("Name an animal.", "Le chat noir", "The cat"),
]


def _pairs(data=_DATA):
    return [Pair(prompt=p, pos=pos, neg=neg, instruction_id="french")
            for p, pos, neg in data]


def _examples(tok, data=_DATA):
    return pairs_to_distill_examples(tok, _pairs(data), INSTRUCTION)


class _Killed(RuntimeError):
    """The simulated mid-fit kill (raised out of the probe seam)."""


def _assert_bitwise(got, want, what: str):
    assert len(got) == len(want) and len(got) > 0
    for i, (a, b) in enumerate(zip(got, want)):
        assert torch.equal(a, b), (
            f"{what}: trainable[{i}] differs from the uninterrupted run "
            f"(max abs diff {float((a - b).abs().max()):.3e}) — resume is "
            "not exact")
    assert sum(float(p.abs().sum()) for p in want) > 0.0  # something trained


# -- train_lockstep -----------------------------------------------------------


def _run_lockstep(model, tok, arm, *, epochs=3, accum=1, clip=None,
                  ckpt_dir=None, ckpt_key="fp-test", every=1,
                  kill_after=None, heldout=False, early=None,
                  diagnostics=None):
    """One single-arm lockstep fit; returns (history, final trainables,
    captured). ``kill_after=N`` raises _Killed out of the probe seam right
    after the N-th optimizer step — an arbitrary point relative to the
    checkpoint fire points, like a real kill."""
    examples = _examples(tok)
    obj = build_lockstep_object(arm, model, tok, INSTRUCTION, _pairs(),
                                layer=1)
    params = replace(default_lockstep_params(arm), accum_steps=accum,
                     max_grad_norm=clip)
    calls = {"n": 0}
    captured = {}

    def cb(step, name, obj_, loss):
        calls["n"] += 1
        captured["grads"] = [p.grad.detach().clone()
                             for p in obj_.trainable]
        if kill_after is not None and calls["n"] >= kill_after:
            raise _Killed(f"killed after optimizer step {calls['n']}")

    kw = {}
    if ckpt_dir is not None:
        kw = dict(checkpoint_every=every, checkpoint_dir=str(ckpt_dir),
                  checkpoint_key=ckpt_key)
    if heldout:
        kw["heldout"] = _examples(tok, _HELDOUT_DATA)
        kw["early_stop"] = early
    hist = train_lockstep(model, tok, [LockstepArm(arm, obj, params)],
                          examples, epochs=epochs, batch_size=2,
                          max_len=MAX_LEN, seed=0, probe_cb=cb, n_probe=2,
                          diagnostics=diagnostics, **kw)
    return (hist[arm], [p.detach().clone() for p in obj.trainable],
            captured)


def _unclipped_norm_lockstep(model, tok, arm, accum=1) -> float:
    _h, _p, cap = _run_lockstep(model, tok, arm, epochs=1, accum=accum,
                                clip=None)
    n = float(torch.norm(torch.stack([g.float().norm()
                                      for g in cap["grads"]])))
    assert n > 0.0
    return n


@pytest.mark.parametrize("arm", ["lora_r1_all", "psr_kl_all"])
def test_lockstep_interrupt_resume_bitwise(tiny_llama, tmp_path, arm):
    """Kill after optimizer step 5 of 9 (mid-epoch 1 of 3 — the resume
    crosses an epoch boundary), checkpoint_every=1, ACTIVE clip, ragged
    batches (rows 2/2/1, unequal token counts). psr_kl_all additionally
    carries the gate regularizer. Final params AND full history bitwise
    == the uninterrupted run's; a checkpointing-ON uninterrupted run is
    also bitwise == the OFF run (write-does-not-perturb + default-off
    identity)."""
    model, tok = tiny_llama
    clip = _unclipped_norm_lockstep(model, tok, arm) / 4.0

    ref_hist, ref_params, _ = _run_lockstep(model, tok, arm, clip=clip)
    assert len(ref_hist) == 9  # 3 batches x 3 epochs, accum=1

    on_hist, on_params, _ = _run_lockstep(
        model, tok, arm, clip=clip, ckpt_dir=tmp_path / "on")
    _assert_bitwise(on_params, ref_params,
                    f"{arm} (checkpointing on, uninterrupted)")
    assert on_hist == ref_hist

    ck = tmp_path / "killed"
    with pytest.raises(_Killed):
        _run_lockstep(model, tok, arm, clip=clip, ckpt_dir=ck,
                      kill_after=5)
    assert (ck / fitckpt.LATEST_NAME).exists()
    res_hist, res_params, _ = _run_lockstep(model, tok, arm, clip=clip,
                                            ckpt_dir=ck)
    _assert_bitwise(res_params, ref_params, f"{arm} (resumed)")
    assert res_hist == ref_hist


def test_lockstep_resume_with_accumulation_boundaries(tiny_llama, tmp_path):
    """accum=3 over 3 ragged micro-batches -> ONE optimizer step per
    epoch: checkpoints may only fire at group ends, and a resume restarts
    on a fresh group with the restored optimizer/RNG state. Kill after
    step 2 (epoch 1) of 4 epochs; resumed final params + history bitwise
    == uninterrupted."""
    model, tok = tiny_llama
    arm = "lora_r1_all"
    clip = _unclipped_norm_lockstep(model, tok, arm, accum=3) / 4.0

    ref_hist, ref_params, _ = _run_lockstep(model, tok, arm, epochs=4,
                                            accum=3, clip=clip)
    assert len(ref_hist) == 4  # one ragged-tail-closing step per epoch

    ck = tmp_path / "killed"
    with pytest.raises(_Killed):
        _run_lockstep(model, tok, arm, epochs=4, accum=3, clip=clip,
                      ckpt_dir=ck, kill_after=2)
    res_hist, res_params, _ = _run_lockstep(model, tok, arm, epochs=4,
                                            accum=3, clip=clip,
                                            ckpt_dir=ck)
    _assert_bitwise(res_params, ref_params, f"{arm} (accum resumed)")
    assert res_hist == ref_hist


def test_lockstep_early_stop_state_survives_resume(tiny_llama, tmp_path):
    """min_delta=1e9 makes epoch 1's held-out eval the PERMANENT best
    (patience 2 then stops after epoch 3 of 6), so the final weights are
    the epoch-1 stopper snapshot. Killing in epoch 2 and resuming must
    reproduce them bitwise — only the CHECKPOINTED stopper state (best,
    patience counter, snapshot) can supply that; a stopper rebuilt fresh
    at resume would restore a post-resume snapshot instead."""
    model, tok = tiny_llama
    arm = "lora_r1_all"
    early = EarlyStopSpec(patience=2, min_epochs=1, min_delta=1e9)

    ref_diag: dict = {}
    ref_hist, ref_params, _ = _run_lockstep(
        model, tok, arm, epochs=6, heldout=True, early=early,
        diagnostics=ref_diag)
    d = ref_diag[arm]
    assert d["stopped_early"] is True
    assert d["best_epoch"] == 1 and d["epochs_run"] == 3

    ck = tmp_path / "killed"
    # 3 optimizer steps per epoch; kill after step 5 = mid-epoch 2, after
    # the epoch-1 eval installed the best snapshot.
    with pytest.raises(_Killed):
        _run_lockstep(model, tok, arm, epochs=6, heldout=True, early=early,
                      ckpt_dir=ck, kill_after=5)
    res_diag: dict = {}
    res_hist, res_params, _ = _run_lockstep(
        model, tok, arm, epochs=6, heldout=True, early=early,
        ckpt_dir=ck, diagnostics=res_diag)
    _assert_bitwise(res_params, ref_params, f"{arm} (early-stop resumed)")
    assert res_hist == ref_hist
    rd = res_diag[arm]
    assert rd["best_epoch"] == d["best_epoch"]
    assert rd["epochs_run"] == d["epochs_run"]
    assert rd["stopped_early"] is True
    assert rd["heldout_curve"] == d["heldout_curve"]
    assert rd["best_heldout_loss"] == d["best_heldout_loss"]


def test_lockstep_key_mismatch_starts_fresh(tiny_llama, tmp_path, capsys):
    """A checkpoint written under fit identity A must be IGNORED (loudly,
    no crash, no partial load) by a fit with identity B: the B run trains
    from scratch and matches an uncheckpointed fresh run bitwise."""
    model, tok = tiny_llama
    arm = "lora_r1_all"
    ck = tmp_path / "ck"
    with pytest.raises(_Killed):
        _run_lockstep(model, tok, arm, ckpt_dir=ck, ckpt_key="fp-A",
                      kill_after=2)
    assert (ck / fitckpt.LATEST_NAME).exists()
    capsys.readouterr()

    ref_hist, ref_params, _ = _run_lockstep(model, tok, arm)  # no ckpt
    got_hist, got_params, _ = _run_lockstep(model, tok, arm, ckpt_dir=ck,
                                            ckpt_key="fp-B")
    out = capsys.readouterr().out
    assert "IGNORING" in out and "DIFFERENT fit identity" in out
    _assert_bitwise(got_params, ref_params, f"{arm} (key-mismatch fresh)")
    assert got_hist == ref_hist


# -- train_psr (mse loop) ------------------------------------------------------


def _run_psr_mse(model, tok, *, epochs=3, clip=None, ckpt_dir=None,
                 ckpt_key="fp-test", every=1, kill_after=None,
                 heldout=False, early=None, diagnostics=None):
    examples = _examples(tok)
    obj = PSRObject(model, [1], train_direction=True)
    calls = {"n": 0}
    captured = {}

    def cb(step, o):
        calls["n"] += 1
        captured["grads"] = [p.grad.detach().clone() for p in o.trainable]
        if kill_after is not None and calls["n"] >= kill_after:
            raise _Killed(f"killed after optimizer step {calls['n']}")

    kw = {}
    if ckpt_dir is not None:
        kw = dict(checkpoint_every=every, checkpoint_dir=str(ckpt_dir),
                  checkpoint_key=ckpt_key)
    if heldout:
        kw["heldout"] = _examples(tok, _HELDOUT_DATA)
        kw["early_stop"] = early
    hist = train_psr(model, tok, obj, examples, loss="mse", epochs=epochs,
                     batch_size=2, max_len=MAX_LEN, seed=0,
                     max_grad_norm=clip, probe_cb=cb, probe_every=1,
                     diagnostics=diagnostics, **kw)
    return hist, [p.detach().clone() for p in obj.trainable], captured


def test_psr_mse_interrupt_resume_bitwise(tiny_llama, tmp_path):
    """Same parity bar for train_psr's mse loop, under its OWN default
    warmup_cosine schedule (the restored optimizer-step index IS the
    schedule position) and an ACTIVE clip, on ragged batches. Kill after
    step 4 of 9 (mid-epoch 1 of 3, so the resume crosses an epoch
    boundary); resumed params + history bitwise == uninterrupted, and a
    checkpointing-ON uninterrupted run == the OFF run."""
    model, tok = tiny_llama
    _, _, un = _run_psr_mse(model, tok, epochs=1, clip=math.inf)
    n0 = float(torch.norm(torch.stack([g.float().norm()
                                       for g in un["grads"]])))
    assert n0 > 0.0
    clip = n0 / 4.0

    ref_hist, ref_params, _ = _run_psr_mse(model, tok, clip=clip)
    assert len(ref_hist) == 9  # 3 batches x 3 epochs

    on_hist, on_params, _ = _run_psr_mse(model, tok, clip=clip,
                                         ckpt_dir=tmp_path / "on")
    _assert_bitwise(on_params, ref_params,
                    "psr mse (checkpointing on, uninterrupted)")
    assert on_hist == ref_hist

    ck = tmp_path / "killed"
    with pytest.raises(_Killed):
        _run_psr_mse(model, tok, clip=clip, ckpt_dir=ck, kill_after=4)
    assert (ck / fitckpt.LATEST_NAME).exists()
    res_hist, res_params, _ = _run_psr_mse(model, tok, clip=clip,
                                           ckpt_dir=ck)
    _assert_bitwise(res_params, ref_params, "psr mse (resumed)")
    assert res_hist == ref_hist


def test_psr_mse_key_mismatch_starts_fresh(tiny_llama, tmp_path, capsys):
    model, tok = tiny_llama
    ck = tmp_path / "ck"
    with pytest.raises(_Killed):
        _run_psr_mse(model, tok, ckpt_dir=ck, ckpt_key="fp-A", kill_after=2)
    capsys.readouterr()
    ref_hist, ref_params, _ = _run_psr_mse(model, tok)
    got_hist, got_params, _ = _run_psr_mse(model, tok, ckpt_dir=ck,
                                           ckpt_key="fp-B")
    out = capsys.readouterr().out
    assert "IGNORING" in out and "DIFFERENT fit identity" in out
    _assert_bitwise(got_params, ref_params, "psr mse (key-mismatch fresh)")
    assert got_hist == ref_hist


def test_psr_mse_early_stop_state_survives_resume(tiny_llama, tmp_path):
    """train_psr's mse loop, same bar as the lockstep early-stop resume
    pin: min_delta=1e9 makes epoch 1's held-out eval the PERMANENT best
    (patience 2 then stops after epoch 3 of 6), so the final weights are
    the epoch-1 stopper snapshot. Killing in epoch 2 and resuming must
    reproduce them bitwise — only the CHECKPOINTED stopper state (best,
    patience counter, snapshot, curve) can supply that; a stopper rebuilt
    fresh at resume would restore a post-resume snapshot and fail the
    bitwise compare."""
    model, tok = tiny_llama
    early = EarlyStopSpec(patience=2, min_epochs=1, min_delta=1e9)

    ref_diag: dict = {}
    ref_hist, ref_params, _ = _run_psr_mse(
        model, tok, epochs=6, heldout=True, early=early,
        diagnostics=ref_diag)
    assert ref_diag["stopped_early"] is True
    assert ref_diag["best_epoch"] == 1 and ref_diag["epochs_run"] == 3
    assert len(ref_hist) == 9  # 3 batches x 3 epochs run of the 6 cap
    # + the pre-training baseline point
    assert len(ref_diag["heldout_curve"]) == ref_diag["epochs_run"] + 1

    ck = tmp_path / "killed"
    # 3 optimizer steps per epoch; kill after step 5 = mid-epoch 2, after
    # the epoch-1 eval installed the best snapshot.
    with pytest.raises(_Killed):
        _run_psr_mse(model, tok, epochs=6, heldout=True, early=early,
                     ckpt_dir=ck, kill_after=5)
    assert (ck / fitckpt.LATEST_NAME).exists()
    res_diag: dict = {}
    res_hist, res_params, _ = _run_psr_mse(
        model, tok, epochs=6, heldout=True, early=early, ckpt_dir=ck,
        diagnostics=res_diag)
    _assert_bitwise(res_params, ref_params, "psr mse (early-stop resumed)")
    assert res_hist == ref_hist
    # the full stopping record — curve, best epoch, stop point — is exact
    assert res_diag == ref_diag


def test_psr_kl_path_refuses_checkpoint_kwargs():
    """The KL path's optimizer loop lives in train_object_kl; a silently
    dropped durability request is the 3dd9e36 bug class. The refusal
    fires before any model work, so dummies suffice."""
    with pytest.raises(ValueError, match="checkpoint"):
        train_psr(None, None, None, [], loss="kl",
                  checkpoint_every=100, checkpoint_dir="/tmp/x",
                  checkpoint_key="k")


def test_psr_kl_path_refuses_heldout_early_stop():
    """Same refusal class for the early-stop seam: the KL psr arms
    early-stop through train_lockstep, so an early-stop request on
    train_psr's KL path must fail loudly, never silently run the fixed
    epoch budget."""
    with pytest.raises(ValueError, match="heldout/early_stop"):
        train_psr(None, None, None, [], loss="kl",
                  heldout=[], early_stop=EarlyStopSpec())


# -- the fitckpt store itself --------------------------------------------------


def test_fitckpt_trio_validation():
    from sisteer.extraction.fitckpt import resolve_checkpoint_kwargs

    assert resolve_checkpoint_kwargs(None, None, None) is None
    assert resolve_checkpoint_kwargs("25", "/tmp/x", "k") == 25
    with pytest.raises(ValueError, match="without checkpoint_every"):
        resolve_checkpoint_kwargs(None, "/tmp/x", None)
    with pytest.raises(ValueError, match="checkpoint_dir"):
        resolve_checkpoint_kwargs(10, None, "k")
    with pytest.raises(ValueError, match="checkpoint_key"):
        resolve_checkpoint_kwargs(10, "/tmp/x", None)
    with pytest.raises(ValueError, match=">= 1"):
        resolve_checkpoint_kwargs(0, "/tmp/x", "k")


def test_fitckpt_rotation_and_corrupt_fallback(tmp_path, capsys):
    """Two writes leave latest + previous; a corrupt latest falls back to
    the previous checkpoint LOUDLY, never a crash."""
    d = tmp_path / "ck"
    fitckpt.save_checkpoint(d, {"checkpoint_key": "k", "step": 1})
    fitckpt.save_checkpoint(d, {"checkpoint_key": "k", "step": 2})
    assert (d / fitckpt.LATEST_NAME).exists()
    assert (d / fitckpt.PREVIOUS_NAME).exists()
    assert fitckpt.load_checkpoint(d, "k")["step"] == 2
    (d / fitckpt.LATEST_NAME).write_bytes(b"torn write, not a checkpoint")
    capsys.readouterr()
    state = fitckpt.load_checkpoint(d, "k")
    assert state is not None and state["step"] == 1
    assert "unreadable" in capsys.readouterr().out
    # and with BOTH slots bad: fresh start, still no crash
    (d / fitckpt.PREVIOUS_NAME).write_bytes(b"also torn")
    assert fitckpt.load_checkpoint(d, "k") is None


def test_fitckpt_expect_mismatch_ignored(tmp_path, capsys):
    d = tmp_path / "ck"
    fitckpt.save_checkpoint(d, {"checkpoint_key": "k", "n_batches": 3})
    capsys.readouterr()
    got = fitckpt.load_checkpoint(d, "k", expect={"n_batches": 7})
    assert got is None
    assert "structure" in capsys.readouterr().out


# -- driver wiring (train_two_si_objects.py) -----------------------------------


import importlib.util
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "_t2_fitckpt", REPO / "scripts/train_two_si_objects.py")
t2 = importlib.util.module_from_spec(_spec)
sys.modules["_t2_fitckpt"] = t2
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


@pytest.mark.parametrize("arm,mode", [("lora_r1_all", "lockstep"),
                                      ("psr_mse_all", "solo")])
def test_driver_checkpoint_dir_flows_to_trainers(tiny_llama, tmp_path,
                                                 arm, mode):
    """checkpoint_dir/checkpoint_every flow from train_si_objects into
    BOTH trainer paths; the checkpoint lands per cell under the root,
    OUTSIDE the objects store, keyed by the cell's fit fingerprint — the
    SAME identity the COMPLETE-marker resume validates."""
    model, tok = tiny_llama
    tk = {"lr_schedule": "constant"} if arm == "psr_mse_all" else {}
    cfg = _driver_cfg(arm, tk)
    ck_root = tmp_path / "ckpt"
    outcome = t2.train_si_objects(
        model, tok, SI, _si_pairs(), [arm], cfg=cfg, save_dir=None,
        checkpoint_dir=str(ck_root), checkpoint_every=1)
    assert outcome.ok, {a: (r.error_type, r.error)
                        for a, r in outcome.failed.items()}
    ck_file = ck_root / f"{SI}__{arm}" / fitckpt.LATEST_NAME
    assert ck_file.exists(), f"no checkpoint written for the {mode} path"
    state = torch.load(str(ck_file), map_location="cpu",
                       weights_only=False)
    expected_fp = t2.cell_fit_fingerprint(
        cfg, SI, arm, positions="all",
        pool_sha256=t2.pool_content_sha256(_si_pairs()),
        config_sha=t2._cfg_dict_sha(cfg),
        teacher=t2._teacher_leg_tag(None, cfg, arm))
    assert state["checkpoint_key"] == expected_fp


def test_driver_default_off_writes_nothing(tiny_llama, tmp_path):
    """No checkpoint_dir -> no checkpoint plumbing anywhere (the trainers'
    kwargs stay at their None defaults and no files appear)."""
    model, tok = tiny_llama
    cfg = _driver_cfg("lora_r1_all", {})
    outcome = t2.train_si_objects(model, tok, SI, _si_pairs(),
                                  ["lora_r1_all"], cfg=cfg, save_dir=None)
    assert outcome.ok
    assert not list(tmp_path.glob("**/ckpt*.pt"))
