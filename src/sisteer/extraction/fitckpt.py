"""Mid-fit checkpointing for the long fit trainers (fit-wave durability,
twosi-model-sweeps 2026-08-27): ``train_lockstep`` and ``train_psr``'s mse
loop write a resumable snapshot every ``checkpoint_every`` OPTIMIZER steps,
so a killed container loses at most one checkpoint interval instead of the
whole fit (~$1.9k of GPU time went that way when an ephemeral Modal app
died with its reaped client mid-wave).

Design points, all load-bearing:

- **Exact resume, not approximate.** A checkpoint carries EVERYTHING the
  step loop's forward trajectory depends on: trainable params, full
  ``optimizer.state_dict()``, lr-schedule position (via the restored
  optimizer-step count), epoch index + batch position within the epoch +
  the epoch's already-shuffled batch order + the order-RNG state,
  early-stop state (best held-out loss, patience counter, best-object
  snapshot), and torch CPU & CUDA RNG states. The trainers build batches
  deterministically from their inputs, so resume = restore all state,
  recompute batch composition, skip to the checkpointed batch, continue —
  bitwise on CPU fp32 (pinned by tests/test_fit_checkpoint.py). This
  resolves the earlier review decline of PARTIAL-state checkpointing:
  completeness is the requirement, a checkpoint that restores "most" of
  the state silently trains a different trajectory.

- **Optimizer-step boundaries only.** Micro-batch position within a
  gradient-accumulation group is deliberately NOT checkpointed — the
  trainers fire the seam only where every live arm has just completed an
  optimizer step, so there is never partial-group state to carry.

- **Atomic write, latest + previous.** ``torch.save`` to a tmp file, then
  ``os.replace`` into the latest slot, rotating the old latest to the
  previous slot first. A kill mid-write can never corrupt an existing
  checkpoint; a kill between the two replaces leaves the latest slot
  briefly empty, which is exactly why :func:`load_checkpoint` falls back
  to the previous slot.

- **Identity-bound, never a crash, never a silent partial load.** Every
  checkpoint carries the caller's ``checkpoint_key`` (the fit driver
  passes the cell's fit fingerprint). On load, a key mismatch — or a
  missing/corrupt/wrong-shape file — prints LOUDLY and is ignored (fresh
  start). A checkpoint for a different fit identity must never leak state
  into this one, and a torn file must never abort a wave.

- **No new per-step syncs.** State is serialized (GPU→CPU copies
  included) only at checkpoint moments; the step loop's one-sync-per-
  cadence monitoring discipline is untouched between them.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch

#: The production cadence (the fit driver's default; the trainers
#: themselves default to OFF so every existing caller is byte-identical).
DEFAULT_CHECKPOINT_EVERY = 100

LATEST_NAME = "ckpt.pt"
PREVIOUS_NAME = "ckpt.prev.pt"


def resolve_checkpoint_kwargs(checkpoint_every, checkpoint_dir,
                              checkpoint_key) -> int | None:
    """Validate a trainer's checkpoint kwarg trio; returns the interval as
    an int, or None for OFF. The three come together: an interval without
    a dir (or a dir/key without an interval) is a configured intent that
    would silently do nothing — the 3dd9e36 bug class — so it refuses
    loudly instead."""
    if checkpoint_every is None:
        if checkpoint_dir is not None or checkpoint_key is not None:
            raise ValueError(
                "checkpoint_dir/checkpoint_key without checkpoint_every — "
                "refusing to silently ignore a checkpoint request (set "
                "checkpoint_every, or drop all three)")
        return None
    every = int(checkpoint_every)
    if every < 1:
        raise ValueError(
            f"checkpoint_every must be >= 1, got {checkpoint_every!r}")
    if checkpoint_dir is None:
        raise ValueError(
            f"checkpoint_every={every} without a checkpoint_dir — nowhere "
            "to write the checkpoints")
    if not checkpoint_key:
        raise ValueError(
            f"checkpoint_every={every} without a checkpoint_key — a "
            "checkpoint must be bound to its fit's identity (the driver "
            "passes the cell fit fingerprint) so a stale file can never "
            "resume a different fit")
    return every


def to_cpu(obj):
    """Deep-copy a (possibly nested) state structure with every tensor
    detached to CPU — what makes a saved ``optimizer.state_dict()``
    device-independent and the checkpoint file loadable anywhere."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(to_cpu(v) for v in obj)
    return obj


def cuda_rng_states() -> list:
    """Every CUDA device's RNG state (empty on a CPU-only host)."""
    return (torch.cuda.get_rng_state_all()
            if torch.cuda.is_available() else [])


def restore_cuda_rng(states) -> None:
    """Restore CUDA RNG states saved by :func:`cuda_rng_states`. A
    checkpoint written on a different device layout (or read on a
    CPU-only host, e.g. the parity tests) skips the restore LOUDLY rather
    than crashing — the CPU state restore still happened, and on the
    production single-H200 fits the layouts always match."""
    if not states:
        return
    if (not torch.cuda.is_available()
            or torch.cuda.device_count() != len(states)):
        print(f"[fitckpt] checkpoint carries {len(states)} CUDA RNG "
              "state(s) but this host has "
              f"{torch.cuda.device_count() if torch.cuda.is_available() else 0}"
              " CUDA device(s) — skipping the CUDA RNG restore",
              flush=True)
        return
    torch.cuda.set_rng_state_all(states)


def save_checkpoint(checkpoint_dir, state: dict) -> Path:
    """Atomically publish ``state`` as the latest checkpoint, keeping the
    previous one. ``state`` must already carry ``checkpoint_key`` (the
    trainers build it). Write order: tmp file -> rotate latest to
    previous -> tmp into the latest slot; each step is an ``os.replace``
    (atomic on one filesystem), so no kill point leaves a corrupt or
    ambiguous pair."""
    d = Path(checkpoint_dir)
    d.mkdir(parents=True, exist_ok=True)
    if "checkpoint_key" not in state:
        raise ValueError("checkpoint state must carry its checkpoint_key")
    tmp = d / (LATEST_NAME + ".tmp")
    torch.save(state, str(tmp))
    latest = d / LATEST_NAME
    if latest.exists():
        os.replace(latest, d / PREVIOUS_NAME)
    os.replace(tmp, latest)
    return latest


def load_checkpoint(checkpoint_dir, checkpoint_key: str, *,
                    expect: dict | None = None) -> dict | None:
    """The newest loadable checkpoint bound to ``checkpoint_key``, or
    ``None`` (start fresh). Tries the latest slot, then the previous one
    (the rotation window / a torn latest). EVERY refusal — missing file,
    unreadable file, absent or MISMATCHED key, an ``expect`` field that
    does not match — prints loudly and falls through; this function never
    raises for a bad file and never returns a partially-valid state.

    ``expect`` is the trainer's structural identity (trainer name, arm
    list, batch count): the key already binds the fit's identity, but a
    shape mismatch would crash the restore mid-way, and a half-restored
    trainer is exactly the silent-partial-load this loader exists to
    forbid — so it is checked BEFORE any state is handed back."""
    d = Path(checkpoint_dir)
    for name in (LATEST_NAME, PREVIOUS_NAME):
        p = d / name
        if not p.exists():
            continue
        try:
            state = torch.load(str(p), map_location="cpu",
                               weights_only=False)
        except Exception as e:  # noqa: BLE001 — a torn file must not crash
            print(f"[fitckpt] {p} is unreadable ({type(e).__name__}: {e}) "
                  "— ignoring it", flush=True)
            continue
        if not isinstance(state, dict) or "checkpoint_key" not in state:
            print(f"[fitckpt] {p} carries no checkpoint_key — ignoring it",
                  flush=True)
            continue
        if state["checkpoint_key"] != checkpoint_key:
            print(f"[fitckpt] {p} was written for a DIFFERENT fit identity "
                  f"(checkpoint_key {str(state['checkpoint_key'])[:40]!r} "
                  f"!= expected {str(checkpoint_key)[:40]!r}) — IGNORING it "
                  "and starting this fit fresh", flush=True)
            continue
        bad = [k for k, v in (expect or {}).items() if state.get(k) != v]
        if bad:
            print(f"[fitckpt] {p} matches the fit key but not this "
                  f"trainer's structure ({', '.join(bad)} differ) — "
                  "IGNORING it and starting fresh", flush=True)
            continue
        print(f"[fitckpt] loaded checkpoint {p}", flush=True)
        return state
    return None
