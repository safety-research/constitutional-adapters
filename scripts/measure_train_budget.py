"""The two-SI training-budget GPU pass.

Measures, for every ``(si, arm)`` in ``configs/two_si.yaml`` (``si_ids`` x the
REGISTERED ``class_a_arms``), the real peak memory and throughput of the arm's
OWN production trainer at ascending batch widths, and commits the result as
the measured table ``train_budget.measure_si_budget`` consumes
(``configs/two_si_budget_measured.json``) — so the NEEDS_GPU placeholders in
the budget table become measured numbers with ZERO changes to the plan
command.

Measurement protocol:

- **Synthetic worst-case pairs**, never the real training pool: a
  >=1,024-token attack-allowance filler prompt per example (distinct per
  example so closed-form fits don't degenerate) + a completion long enough
  that ``_encode_aligned``'s truncation pads EVERY example's teacher sequence
  to exactly the per-SI ``max_len`` — rendered through the EXACT production
  path (``pairs_to_distill_examples`` + ``resolve_instruction(si)`` + the
  arm's own registry trainer, which runs its own encoder/collator). Realized
  tensor shapes are RECORDED into the row fingerprint and must reach the
  per-SI maximum; a prompt that cannot fit the config maximum fails loudly
  (the trainer's own ``preflight_max_len``).
- **Batch ladders** ``[1, 2, 4, 8, 16]``; ``alphakl_fisher`` (any
  ``FISHER_ARMS`` member) is hard-capped at ``FISHER_MAX_BATCH`` — the ladder
  never proposes above it. Ascent stops at the first OOM or at
  ``peak_reserved > 0.85 x device total``. bs=1 failing to complete for ANY
  (si, arm) fails acceptance loudly — no NEEDS_GPU row survives.
- **One FRESH subprocess per (si, arm, batch-candidate)** (``trial``): a clean
  process and a fresh model load per trial, so ascending trials never reuse a
  mutated model and an OOM cannot poison siblings. Per trial: ``epochs=1``,
  ``bs x (1 warmup + 3 timed)`` examples, the timed region delimited by the
  trainer's own ``log_fn`` step boundaries with ``torch.cuda.synchronize()``
  at each boundary (trainers that never step ``log_fn`` — closed-form /
  probe fits — fall back to a synchronized whole-call timing, recorded as
  such); 3 repeats, MEDIAN tokens/s; memory = ``max_memory_allocated`` AND
  ``max_memory_reserved`` (the 0.85 guard reads reserved).
  ``tokens_per_s = timed_steps x bs x total_seq x 2 / elapsed`` — the same
  forward-token convention as ``measure_si_budget``'s ``est_seconds``.
- **Two DISTINCT identity checks**: each row carries an exact-match
  run FINGERPRINT (model+revision, full config sha, inputs/code sha, GPU,
  torch/CUDA, realized shapes) — the pass's OWN resume skips a trial only on
  an exact fingerprint match and REFUSES on any mismatch. Consumption by
  ``train_budget`` is gated on ``measurement_inputs_sha`` ONLY (computed
  there; derived ``batch_size_by_*`` overrides and output paths excluded).

Subcommands:
  run    — the GPU driver: iterates the ladders, one fresh ``trial``
           subprocess per candidate, appends fingerprinted rows to the
           trials JSONL (fsync per row — preemption-tolerant resume), then
           emits the measured JSON + figures.
  trial  — ONE (si, arm, batch_size) measurement in THIS process (run's
           subprocess entry; usable standalone for debugging).
  emit   — trials JSONL -> configs/two_si_budget_measured.json: every trial
           row kept, per-(si, arm) ``safe_batch_size``, top-level
           ``measurement_inputs_sha``; refuses stale rows and any (si, arm)
           without a completed bs=1 trial.
  table  — CPU: render the measured per-(si, arm) table and ASSERT
           ``resolve_batch_size(si, arm) <= measured-safe`` for every pair
           (the §5.3 mechanical guarantee, checked at run time too).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

REPO = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()

# The worst-case attack-text allowance the config arithmetic reserves
# (configs/two_si.yaml max_len derivation) — the filler prompt carries at
# least this many tokens.
ATTACK_ALLOWANCE_TOKENS = 1024
# Per trial: bs x (1 warmup + 3 timed) examples, epochs=1.
WARMUP_STEPS = 1
TIMED_STEPS = 3
REPEATS = 3
# Ladder candidates; FISHER_ARMS are capped at FISHER_MAX_BATCH.
LADDER = (1, 2, 4, 8, 16)
# Ascent stops when peak RESERVED exceeds this fraction of device memory.
GUARD_FRAC = 0.85

DEFAULT_TRIALS = "out/two_si/budget/trials.jsonl"
# The W&B purpose of the budget figures (_log_wandb_figures): the project is
# sisteer.wandb_entity.wandb_project(WANDB_PURPOSE), so SISTEER_WANDB_PROJECT_BUDGET
# first. A compute node has no site config: the cluster's hf_site.env carries the
# site's value, and hf_env.sh (which stage2_common.sh's s2_init sources) exports it.
# Then WANDB_PROJECT / SISTEER_WANDB_PROJECT, else the neutral sisteer-budget.
WANDB_PURPOSE = "budget"


def _load_cfg(path: str) -> dict:
    import yaml

    p = Path(path)
    if not p.is_absolute():
        p = REPO / path
    return yaml.safe_load(p.read_text())


def _canonical_sha(payload) -> str:
    import hashlib

    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"),
                   default=str).encode("utf-8")).hexdigest()


def registered_class_a_arms(cfg: dict) -> list[str]:
    """The (si x arm) matrix's arm axis: config class_a_arms filtered to
    methods.ARMS — identical derivation to the registry's (never a
    literal)."""
    from sisteer.eval.defense_cell import CLASS_A_ARMS
    from sisteer.methods import ARMS

    arms = list(cfg.get("class_a_arms") or CLASS_A_ARMS)
    return [a for a in arms if a in ARMS]


def ladder_for(arm: str) -> list[int]:
    """The ascending batch ladder for one arm: LADDER, hard-capped at
    FISHER_MAX_BATCH for the fisher rung (a memory fact — O(T^2) forward-AD
    attention; the pass never proposes above it, whatever the config asks)."""
    from sisteer.eval.train_budget import FISHER_ARMS, FISHER_MAX_BATCH

    cap = FISHER_MAX_BATCH if arm in FISHER_ARMS else max(LADDER)
    return [b for b in LADDER if b <= cap]


# --------------------------------------------------------------------------- #
# Synthetic worst-case pairs through the EXACT production renderer            #
# --------------------------------------------------------------------------- #


def _filler_text(tok, n_tokens: int, word: str = "budget") -> str:
    """A deterministic string of AT LEAST ``n_tokens`` tokens (minimal
    per-word overshoot) under ``tok`` — the worst-case attack-allowance /
    completion filler. Grow-only: the worst case must UPPER-bound, so we
    never trim below the target."""

    def count(s: str) -> int:
        return len(tok(s, add_special_tokens=False)["input_ids"])

    text = word
    while count(text) < n_tokens:
        text += (" " + word) * (n_tokens - count(text))
    return text


def synthetic_pairs(tok, cfg: dict, si_id: str, n: int):
    """``n`` worst-case Pairs for one SI + the resolved Instruction. Prompts
    are the >=1,024-token filler with a distinct per-example marker (so
    closed-form/probe fits see non-degenerate data); completions are a filler
    LONGER than any per-SI ``avail``, so ``_encode_aligned`` truncates every
    example's teacher sequence to exactly ``max_len`` — the realized padded T
    reaches the per-SI maximum, provably upper-bounding every allowed
    production shape."""
    from sisteer.data.pairs import Pair
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.eval.train_budget import resolve_max_len

    instruction = resolve_instruction(si_id)
    max_len = resolve_max_len(cfg, si_id)
    prompt_filler = _filler_text(tok, ATTACK_ALLOWANCE_TOKENS)
    n_prompt = len(tok(prompt_filler, add_special_tokens=False)["input_ids"])
    if n_prompt < ATTACK_ALLOWANCE_TOKENS:
        raise SystemExit(
            f"[budget] filler construction bug: {n_prompt} < "
            f"{ATTACK_ALLOWANCE_TOKENS} attack-allowance tokens")
    completion_filler = _filler_text(tok, max_len, word="reply")
    pairs = [
        Pair(prompt=f"{prompt_filler} v{i}",
             pos=f"v{i} {completion_filler}", neg="",
             instruction_id=si_id)
        for i in range(n)
    ]
    return pairs, instruction


def expected_shapes(tok, cfg: dict, si_id: str, batch_size: int) -> dict:
    """Render the trial's examples through the EXACT production renderer +
    encoder + collator (pairs_to_distill_examples -> _encode_aligned ->
    collate) and RECORD the realized shapes (r1 #14). Loud failures:
    a prompt exceeding the config maximum (production's preflight), and a
    realized teacher width that does NOT reach the per-SI ``max_len`` (the
    worst-case construction must hit the maximum, or it upper-bounds
    nothing)."""
    from sisteer.data.pairs import pairs_to_distill_examples
    from sisteer.extraction.distill import _encode_aligned, preflight_max_len
    from sisteer.extraction.traindata import collate
    from sisteer.eval.train_budget import resolve_max_len

    max_len = resolve_max_len(cfg, si_id)
    n = batch_size * (WARMUP_STEPS + TIMED_STEPS)
    pairs, instruction = synthetic_pairs(tok, cfg, si_id, n)
    examples = pairs_to_distill_examples(tok, pairs, instruction)
    # Production's own refusal: a prompt the config maximum cannot hold.
    preflight_max_len(tok, examples, max_len)
    encoded = [_encode_aligned(tok, ex, max_len) for ex in examples]
    teacher = collate([t for t, _ in encoded], tok.pad_token_id or 0)
    student = collate([s for _, s in encoded], tok.pad_token_id or 0)
    t_width = int(teacher.input_ids.shape[1])
    if t_width != max_len:
        raise SystemExit(
            f"[budget] worst-case construction failed to reach the per-SI "
            f"maximum for {si_id!r}: realized teacher width {t_width} != "
            f"max_len {max_len} — refusing to measure a non-worst-case shape")
    return {
        "max_len": max_len,
        "total_seq_tokens": t_width,
        "student_seq_tokens": int(student.input_ids.shape[1]),
        "teacher_prompt_tokens": int(teacher.prompt_lens.max()),
        "student_prompt_tokens": int(student.prompt_lens.max()),
        "attack_allowance_tokens": ATTACK_ALLOWANCE_TOKENS,
        "completion_cap": int(cfg.get("completion_tokens", 512)),
        "n_examples": n,
        "batch_size": batch_size,
    }


def _trial_kwargs(cfg: dict, arm: str, bs: int,
                  max_len: int) -> tuple[dict, dict]:
    """Production-parity trial kwargs: ``train_kwargs.default`` merged UNDER
    ``train_kwargs.<arm>`` (the production driver's convention — e.g.
    ``train_kwargs: {alphakl_soft: {beta: 1.0}}`` reaches the stage-2
    trainer), then the measurement-forced knobs (epochs=1, the ladder
    batch, the per-SI max_len, seed 0) override. Returns
    ``(kwargs, recorded_extras)``. (cast used to get a RECORDED pairs
    fallback here; it left methods.ARMS entirely 2026-08-20, so no
    measurable arm needs one.)"""
    tk_all = cfg.get("train_kwargs") or {}
    kw = {**dict(tk_all.get("default") or {}), **dict(tk_all.get(arm) or {})}
    # Driver-level POLICY keys — pop them here, because _call_trainer's
    # TypeError-drop structurally cannot catch them:
    #   floor_ratio_max   a guard bar, never a trainer kwarg (the production
    #                     driver pops it for the same reason).
    #   lr_schedule / warmup_steps / max_grad_norm
    #                     train_psr ACCEPTS these in its signature (its mse
    #                     loop implements them) and raises ValueError on the
    #                     KL path — a ValueError, not a TypeError, so the
    #                     drop loop re-raises and a psr_kl_all re-measure
    #                     aborts instead of degrading. Live as of the
    #                     warmup_cosine pin added for psr_kl_all's runaway.
    # None of them changes peak memory or step time beyond the noise floor,
    # which is the whole of what this script measures; clipping's extra norm
    # reduction is second-order. Recorded so the drop is auditable per row.
    policy = {k: kw.pop(k) for k in
              ("floor_ratio_max", "lr_schedule", "warmup_steps",
               "max_grad_norm") if k in kw}
    kw.update({"epochs": 1, "batch_size": bs, "max_len": max_len, "seed": 0})
    return kw, ({"policy_keys_dropped": policy} if policy else {})


def run_fingerprint(cfg: dict, shapes: dict, *, si_id: str, arm: str,
                    gpu_name: str) -> dict:
    """The exact-match resume identity of one trial row (r1 #17 / r2 #1),
    ARM-SCOPED (train_budget.arm_scoped_inputs_sha): model+revision, the
    per-(si, arm) inputs sha — renderer/collator/registry sources, THIS
    arm's trainer sources + effective training spec, THIS SI's sequence
    budget, the manual harness-semantics version — plus GPU, torch/CUDA and
    the realized shapes. Resume skips ONLY on an exact match; a mismatch
    SUPERSEDES the stale row and re-runs it (never silently read, never
    blocking the other arms' finished rows — a one-arm config fix like
    train_kwargs.alphakl_soft.beta re-runs only that arm's trials)."""
    import torch
    from sisteer.eval.train_budget import (
        SEQ_CONSTRUCTION_VERSION,
        arm_scoped_inputs_sha,
    )

    return {
        "model": cfg.get("model"),
        "model_revision": cfg.get("model_revision"),
        "arm_inputs_sha": arm_scoped_inputs_sha(cfg, si_id, arm),
        "seq_construction_version": SEQ_CONSTRUCTION_VERSION,
        "gpu_name": gpu_name,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shapes": shapes,
    }


# --------------------------------------------------------------------------- #
# trial — one (si, arm, batch_size) in THIS (fresh) process                    #
# --------------------------------------------------------------------------- #


class _StepTimer:
    """log_fn seam: a CUDA-synchronized timestamp at every trainer step
    boundary. Accepts any (metrics, step) shape the trainers use."""

    def __init__(self):
        self.times: list[float] = []

    def __call__(self, *_args, **_kwargs):
        import torch

        torch.cuda.synchronize()
        self.times.append(time.perf_counter())


def _call_trainer(train_fn, model, tok, instruction, pairs, *, layer,
                  candidate_kw, log_fn):
    """Call a registry trainer with only the kwargs it accepts — the SAME
    call-time TypeError-drop discipline the production driver
    (train_two_si_objects._safe_registry_train) uses, reimplemented here so
    this script never imports that in-flight file. Only keys WE
    supplied are ever dropped; the dropped set is RECORDED in the row."""
    kw = dict(candidate_kw)
    dropped: list[str] = []
    while True:
        try:
            return train_fn(model, tok, instruction, pairs, layer=layer,
                            log_fn=log_fn, **kw), dropped
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if m and m.group(1) in kw:
                dropped.append(m.group(1))
                kw.pop(m.group(1))
                continue
            raise


def _is_oom(e: BaseException) -> bool:
    import torch

    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def cmd_trial(args, cfg: dict) -> None:
    import gc

    import torch

    from sisteer.eval.jailbreak import resolve_instruction  # noqa: F401 (production path)
    from sisteer.eval.train_budget import FISHER_ARMS, FISHER_MAX_BATCH
    from sisteer.methods import ARMS
    from sisteer.models import load_model_and_tokenizer

    if not torch.cuda.is_available():
        raise SystemExit("[budget] trial needs a CUDA device — this is the "
                         "GPU measurement pass (CPU numbers would be fiction)")
    si, arm, bs = args.si, args.arm, args.batch_size
    if arm not in ARMS:
        raise SystemExit(f"[budget] unknown arm {arm!r} (not in methods.ARMS)")
    if arm in FISHER_ARMS and bs > FISHER_MAX_BATCH:
        raise SystemExit(
            f"[budget] {arm} is hard-capped at batch {FISHER_MAX_BATCH} "
            "(O(T^2) forward-AD attention) — the ladder never proposes above "
            f"it; got bs={bs}")

    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"))
    shapes = expected_shapes(tok, cfg, si, bs)
    fingerprint = run_fingerprint(cfg, shapes, si_id=si, arm=arm,
                                  gpu_name=torch.cuda.get_device_name(0))
    n = shapes["n_examples"]
    pairs, instruction = synthetic_pairs(tok, cfg, si, n)
    candidate_kw, extra_kwargs = _trial_kwargs(cfg, arm, bs,
                                               shapes["max_len"])

    device_total_gb = (
        torch.cuda.get_device_properties(0).total_memory / 2**30)
    T = shapes["total_seq_tokens"]

    repeats = int(args.repeats)
    tokens_per_s_all: list[float] = []
    elapsed_all: list[float] = []
    n_log_calls: list[int] = []
    peak_alloc = 0.0
    peak_reserved = 0.0
    timing_mode = None
    dropped: list[str] = []
    oom = False
    error = None
    try:
        for _rep in range(repeats):
            timer = _StepTimer()
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            obj, dropped = _call_trainer(
                ARMS[arm].train, model, tok, instruction, pairs,
                layer=int(cfg.get("layer", 15)), candidate_kw=candidate_kw,
                log_fn=timer)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            peak_alloc = max(peak_alloc,
                             torch.cuda.max_memory_allocated() / 2**30)
            peak_reserved = max(peak_reserved,
                                torch.cuda.max_memory_reserved() / 2**30)
            del obj
            n_log_calls.append(len(timer.times))
            if len(timer.times) >= 2:
                # Timed region between the trainer's OWN step boundaries;
                # the interval before the first boundary is the warmup step.
                steps = len(timer.times) - 1
                elapsed = timer.times[-1] - timer.times[0]
                mode = "log_fn_steps"
                fwd_tokens = steps * bs * T * 2
            else:
                # Closed-form / probe fits never step log_fn: synchronized
                # whole-call timing over all n examples (recorded as such).
                steps = None
                elapsed = t1 - t0
                mode = "whole_call"
                fwd_tokens = n * T * 2
            if timing_mode is None:
                timing_mode = mode
            if elapsed <= 0:
                raise SystemExit(
                    f"[budget] degenerate timing (elapsed={elapsed}) for "
                    f"({si}, {arm}, bs={bs}) — refusing to record")
            elapsed_all.append(elapsed)
            tokens_per_s_all.append(fwd_tokens / elapsed)
    except Exception as e:  # noqa: BLE001 — classified + recorded, never lost
        if _is_oom(e):
            oom = True
            peak_alloc = max(peak_alloc,
                             torch.cuda.max_memory_allocated() / 2**30)
            peak_reserved = max(peak_reserved,
                                torch.cuda.max_memory_reserved() / 2**30)
        else:
            error = f"{type(e).__name__}: {e}"

    ok = not oom and error is None and len(tokens_per_s_all) == repeats
    # Measurement-validity annotation: the
    # closed-form arms' scale calibration can legitimately find no coherent
    # range on SYNTHETIC pairs (the degenerate filler drives the pooled
    # distinct-3 guard under threshold as n_examples = bs*4 grows) — a
    # data artifact, NOT a memory bound. Annotated so the table reader
    # never reads such a ladder stop as memory-bound.
    calib_incoherent = bool(error and "found NO coherent scale" in error)
    row = {
        "si_id": si,
        "arm": arm,
        "batch_size": bs,
        "ok": ok,
        "oom": oom,
        "error": error,
        "calibration_incoherent_on_synthetic": calib_incoherent,
        "repeats": repeats,
        "timing_mode": timing_mode,
        "n_log_calls_per_repeat": n_log_calls,
        "tokens_per_s_all": tokens_per_s_all,
        "tokens_per_s_median": (statistics.median(tokens_per_s_all)
                                if tokens_per_s_all else None),
        "elapsed_s_all": elapsed_all,
        "peak_gb_alloc": peak_alloc,
        "peak_gb_reserved": peak_reserved,
        "device_total_gb": device_total_gb,
        "guard_exceeded": peak_reserved > GUARD_FRAC * device_total_gb,
        "guard_frac": GUARD_FRAC,
        "max_len": shapes["max_len"],
        "total_seq_tokens": T,
        "teacher_prompt_tokens": shapes["teacher_prompt_tokens"],
        "student_prompt_tokens": shapes["student_prompt_tokens"],
        "n_examples": n,
        "epochs": 1,
        "dropped_kwargs": dropped,
        "extra_kwargs": extra_kwargs,
        "fingerprint": fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    out = Path(args.row_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(row, sort_keys=True))
    status = "OK" if ok else ("OOM" if oom else f"ERROR ({error})")
    print(f"[budget] trial ({si}, {arm}, bs={bs}): {status} "
          f"peak_reserved={peak_reserved:.1f}GB "
          f"tok/s={row['tokens_per_s_median']}")


# --------------------------------------------------------------------------- #
# run — the ladder driver: fresh subprocess per candidate, fingerprint resume  #
# --------------------------------------------------------------------------- #


def _read_trials(path: Path) -> dict[tuple[str, str, int], dict]:
    rows: dict[tuple[str, str, int], dict] = {}
    if not path.exists():
        return rows
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = (row["si_id"], row["arm"], int(row["batch_size"]))
        if key in rows:
            raise SystemExit(
                f"[budget] duplicate trial row for {key} in {path} — the "
                "trials file is corrupt; refusing to resume over it")
        rows[key] = row
    return rows


def _append_trial(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def check_resume(existing_row, fingerprint: dict, key, trials_path) -> str:
    """The exact-match resume rule (r1 #17): ``"run"`` when no row exists;
    ``"skip"`` on an IDENTICAL fingerprint; ``"supersede"`` on ANY
    difference — the stale row is moved to the superseded sidecar and the
    trial re-runs under the current world. A stale measurement is never
    silently READ (the r1 #17 hazard), and — since the fingerprint is
    arm-scoped — a one-arm input change never invalidates the other arms'
    finished rows."""
    if existing_row is None:
        return "run"
    if existing_row["fingerprint"] == fingerprint:
        return "skip"
    return "supersede"


def _supersede_row(trials_path: Path, key: tuple[str, str, int],
                   stale_row: dict) -> None:
    """Move one stale row out of the live trials file: append it (with a
    supersede timestamp) to ``<trials>.superseded.jsonl`` for the audit
    trail, then atomically rewrite the live file without it."""
    sidecar = trials_path.with_name(trials_path.name + ".superseded.jsonl")
    _append_trial(sidecar, {**stale_row,
                            "superseded_at":
                                datetime.now(timezone.utc).isoformat()})
    kept = []
    for line in trials_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if (row["si_id"], row["arm"], int(row["batch_size"])) != key:
            kept.append(line)
    tmp = trials_path.with_name(trials_path.name + ".tmp")
    with tmp.open("w") as f:
        for line in kept:
            f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, trials_path)
    print(f"[budget] SUPERSEDED stale trial {key} (inputs changed since it "
          f"was measured) -> {sidecar.name}; re-running under the current "
          "world")


def cmd_run(args, cfg: dict) -> None:
    import torch
    from transformers import AutoTokenizer

    if not torch.cuda.is_available():
        raise SystemExit("[budget] run needs a CUDA device — submit via "
                         "scripts/sbatch/evals_budget.sbatch")
    gpu_name = torch.cuda.get_device_name(0)
    # Tokenizer only in the parent (shape fingerprints); the MODEL loads in
    # each trial subprocess, never here.
    tok = AutoTokenizer.from_pretrained(cfg["model"],
                                        revision=cfg.get("model_revision"))
    trials_path = Path(args.trials)
    if not trials_path.is_absolute():
        trials_path = REPO / trials_path
    existing = _read_trials(trials_path)

    sis = list(cfg["si_ids"])
    arms = registered_class_a_arms(cfg)
    if args.si:
        sis = [s for s in sis if s in set(args.si.split(","))]
    if args.arms:
        arms = [a for a in arms if a in set(args.arms.split(","))]
    print(f"[budget] measuring {len(sis)} SIs x {len(arms)} arms "
          f"(ladders {list(LADDER)}, fisher capped) on {gpu_name}; "
          f"trials -> {trials_path}")

    shape_cache: dict[tuple[str, int], dict] = {}
    for si in sis:
        for arm in arms:
            for bs in ladder_for(arm):
                key = (si, arm, bs)
                if (si, bs) not in shape_cache:
                    shape_cache[(si, bs)] = expected_shapes(tok, cfg, si, bs)
                fp = run_fingerprint(cfg, shape_cache[(si, bs)], si_id=si,
                                     arm=arm, gpu_name=gpu_name)
                row = existing.get(key)
                verdict = check_resume(row, fp, key, trials_path)
                if verdict == "skip":
                    print(f"[budget] resume: {key} already measured — skip")
                else:
                    if verdict == "supersede":
                        _supersede_row(trials_path, key, row)
                        existing.pop(key, None)
                    row = _run_one_trial(args, key, trials_path)
                    if row["fingerprint"] != fp:
                        raise SystemExit(
                            f"[budget] trial {key}: subprocess fingerprint "
                            "disagrees with the driver's — tokenizer/config "
                            "drift between processes; refusing to record")
                    existing[key] = row
                if row["oom"]:
                    print(f"[budget] {si}/{arm}: OOM at bs={bs} — ladder stops")
                    break
                if row["error"]:
                    # A non-OOM trainer error stops the LADDER, not the run:
                    # acceptance is bs=1-must-complete, enforced at emit
                    # (_safe_batch_sizes). Calibration-incoherence on the
                    # synthetic pairs is the known benign case (annotated in
                    # the row; a data artifact, not a memory bound).
                    tag = (" [calibration incoherent on synthetic pairs]"
                           if row.get("calibration_incoherent_on_synthetic")
                           else "")
                    print(f"[budget] {si}/{arm}: ERROR at bs={bs}{tag} — "
                          f"ladder stops ({row['error']})")
                    break
                if row["guard_exceeded"]:
                    print(f"[budget] {si}/{arm}: bs={bs} reserved "
                          f"{row['peak_gb_reserved']:.1f}GB > "
                          f"{GUARD_FRAC:.0%} of device — ladder stops")
                    break

    # Acceptance (bs=1 for every pair) + the committed table live in emit.
    _emit(cfg, trials_path, Path(args.emit) if args.emit else None,
          figs_dir=args.figs)


def _run_one_trial(args, key: tuple[str, str, int], trials_path: Path) -> dict:
    si, arm, bs = key
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
        row_out = Path(tf.name)
    try:
        cmd = [sys.executable, str(SCRIPT), "trial",
               "--config", args.config, "--si", si, "--arm", arm,
               "--batch-size", str(bs), "--repeats", str(args.repeats),
               "--row-out", str(row_out)]
        print(f"[budget] trial subprocess: {si} {arm} bs={bs}")
        rc = subprocess.run(cmd, check=False).returncode
        if not row_out.exists() or not row_out.read_text().strip():
            raise SystemExit(
                f"[budget] trial subprocess for {key} died without writing a "
                f"row (rc={rc}) — an infrastructure failure, not an OOM; "
                "fix it before rerunning (no silent skip)")
        row = json.loads(row_out.read_text())
    finally:
        row_out.unlink(missing_ok=True)
    _append_trial(trials_path, row)
    return row


# --------------------------------------------------------------------------- #
# emit — trials JSONL -> the committed measured table (+ figures)              #
# --------------------------------------------------------------------------- #


def _safe_batch_sizes(cfg: dict, rows: dict) -> tuple[dict, list[str], list[str]]:
    """Per-(si, arm) safe width = the widest OK trial within the memory
    guard. bs=1 not completing is an acceptance failure (loud). bs=1
    completing but exceeding the guard still yields safe=1, WITH a recorded
    warning — the arm demonstrably runs, but with almost no headroom."""
    sis = list(cfg["si_ids"])
    arms = registered_class_a_arms(cfg)
    safe: dict[str, dict[str, int]] = {si: {} for si in sis}
    failures: list[str] = []
    warnings: list[str] = []
    for si in sis:
        for arm in arms:
            mine = [r for (s, a, _b), r in rows.items()
                    if s == si and a == arm]
            ok1 = any(r["batch_size"] == 1 and r["ok"] for r in mine)
            if not ok1:
                failures.append(f"({si}, {arm})")
                continue
            within = [r["batch_size"] for r in mine
                      if r["ok"] and not r["guard_exceeded"]]
            if within:
                safe[si][arm] = max(within)
            else:
                safe[si][arm] = 1
                warnings.append(
                    f"({si}, {arm}): bs=1 completed but exceeded the "
                    f"{GUARD_FRAC:.0%} reserved-memory guard — safe width 1 "
                    "with almost no headroom")
    return safe, failures, warnings


def _annotations(cfg: dict, rows: dict) -> tuple[dict, list[str]]:
    """Per-(si, arm) measurement-validity annotations: ladders that ended on
    a non-OOM trainer error (the calibration-incoherence-on-synthetic case
    named explicitly) — the safe width there is capped at the largest
    successful rung, a measurement-validity note, NOT a memory bound."""
    notes: dict[str, dict[str, dict]] = {}
    warnings: list[str] = []
    for (si, arm, bs), r in sorted(rows.items()):
        if r.get("error") is None:
            continue
        note = {"error_at_batch_size": bs,
                "calibration_incoherent_on_synthetic":
                    bool(r.get("calibration_incoherent_on_synthetic"))}
        notes.setdefault(si, {})[arm] = note
        if note["calibration_incoherent_on_synthetic"]:
            warnings.append(
                f"({si}, {arm}): scale calibration found no coherent range "
                f"on the SYNTHETIC pairs at bs={bs} (pooled distinct-3 falls "
                "with n_examples = bs*4 on degenerate filler) — safe width "
                "capped at the largest successful rung; a data artifact, "
                "not a memory bound")
        else:
            warnings.append(
                f"({si}, {arm}): non-OOM trainer error at bs={bs} capped the "
                f"ladder ({r['error']})")
    return notes, warnings


def _emit(cfg: dict, trials_path: Path, emit_path: Path | None,
          *, figs_dir=None) -> dict:
    from sisteer.eval.train_budget import (
        SEQ_CONSTRUCTION_VERSION,
        arm_scoped_inputs_sha,
        measurement_inputs_sha,
    )

    rows = _read_trials(trials_path)
    if not rows:
        raise SystemExit(f"[budget] no trial rows at {trials_path} — nothing "
                         "to emit")
    sha = measurement_inputs_sha(cfg)
    scoped: dict[tuple[str, str], str] = {}
    stale = []
    for (si, arm, _bs), r in rows.items():
        if (si, arm) not in scoped:
            scoped[(si, arm)] = arm_scoped_inputs_sha(cfg, si, arm)
        if r["fingerprint"].get("arm_inputs_sha") != scoped[(si, arm)]:
            stale.append((si, arm, _bs))
    if stale:
        raise SystemExit(
            f"[budget] {len(stale)} trial row(s) were measured under "
            f"DIFFERENT per-(si, arm) inputs (e.g. {stale[0]}) — stale rows "
            "are never emitted. Resume the run (it supersedes + re-measures "
            "them), or `migrate` a pre-arm-scoping trials file")
    safe, failures, warnings = _safe_batch_sizes(cfg, rows)
    if failures:
        raise SystemExit(
            "[budget] ACCEPTANCE FAILED — bs=1 did not complete for: "
            + ", ".join(failures)
            + " (plan §5.1: no NEEDS_GPU row survives)")
    notes, note_warnings = _annotations(cfg, rows)
    warnings = warnings + note_warnings
    any_fp = next(iter(rows.values()))["fingerprint"]
    measured = {
        "measurement_inputs_sha": sha,
        "seq_construction_version": SEQ_CONSTRUCTION_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        # Informational only — consumption gates on the sha above (r2 #1).
        "gpu_name": any_fp["gpu_name"],
        "torch": any_fp["torch"],
        "cuda": any_fp["cuda"],
        "guard_frac": GUARD_FRAC,
        "safe_batch_size": safe,
        "safe_warnings": warnings,
        "annotations": notes,
        "trials": [rows[k] for k in sorted(rows)],
    }
    if emit_path is not None:
        if not emit_path.is_absolute():
            emit_path = REPO / emit_path
        emit_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = emit_path.with_name(emit_path.name + ".tmp")
        tmp.write_text(json.dumps(measured, indent=1, sort_keys=True))
        os.replace(tmp, emit_path)
        print(f"[budget] wrote {emit_path} ({len(measured['trials'])} trial "
              f"rows, inputs sha {sha[:12]}…)")
    for w in warnings:
        print(f"[budget] WARNING: {w}")
    if figs_dir:
        try:
            _write_figures(measured, cfg, Path(figs_dir))
        except Exception as e:  # noqa: BLE001 — figures never eat the data
            print(f"[budget] WARNING: figure generation failed: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)
    return measured


def cmd_emit(args, cfg: dict) -> None:
    trials_path = Path(args.trials)
    if not trials_path.is_absolute():
        trials_path = REPO / trials_path
    _emit(cfg, trials_path, Path(args.emit) if args.emit else None,
          figs_dir=args.figs)


# --------------------------------------------------------------------------- #
# migrate — one-time re-key of pre-arm-scoping trial rows                      #
# --------------------------------------------------------------------------- #


def cmd_migrate(args, cfg: dict) -> None:
    """Soundly carry finished trial rows across the fingerprint re-scoping
    (global full-config sha -> per-(si, arm) sha) and/or a config/code
    change, without re-measuring what did not change.

    Inputs: ``--old-repo`` (a checkout of the EXACT commit the rows were
    measured under — e.g. the frozen cluster checkout) and ``--old-config``
    (that world's two_si.yaml). Per row:

    1. the row's recorded identity must be attested: an old-schema row's
       global ``inputs_sha`` must equal ``measurement_inputs_sha(old_cfg,
       repo_root=old_repo)`` recomputed NOW over the old tree (a row of
       unknown provenance is never migrated); a new-schema row's
       ``arm_inputs_sha`` must equal the old world's scoped sha;
    2. the old and NEW per-(si, arm) scoped shas are compared: EQUAL ->
       the row is re-keyed to the new fingerprint (recorded GPU/torch/
       shapes carried over — scoped-equal inputs imply identical
       construction); DIFFERENT -> the row is superseded (its inputs
       really changed; the resumed run re-measures it).
    """
    from sisteer.eval.train_budget import (
        SEQ_CONSTRUCTION_VERSION,
        arm_scoped_inputs_sha,
        measurement_inputs_sha,
    )

    old_cfg = _load_cfg(args.old_config)
    old_root = Path(args.old_repo)
    new_root = Path(args.repo_root) if args.repo_root else None
    trials_path = Path(args.trials)
    if not trials_path.is_absolute():
        trials_path = REPO / trials_path
    rows = _read_trials(trials_path)
    if not rows:
        raise SystemExit(f"[budget] nothing to migrate at {trials_path}")
    old_global = measurement_inputs_sha(old_cfg, repo_root=old_root)

    kept: list[dict] = []
    superseded: list[tuple] = []
    old_scoped_cache: dict[tuple[str, str], str] = {}
    new_scoped_cache: dict[tuple[str, str], str] = {}
    for key in sorted(rows):
        si, arm, _bs = key
        row = rows[key]
        fp = row["fingerprint"]
        if (si, arm) not in old_scoped_cache:
            old_scoped_cache[(si, arm)] = arm_scoped_inputs_sha(
                old_cfg, si, arm, repo_root=old_root)
            kwargs = {"repo_root": new_root} if new_root else {}
            new_scoped_cache[(si, arm)] = arm_scoped_inputs_sha(
                cfg, si, arm, **kwargs)
        # 1. attest the row against the OLD world
        if "arm_inputs_sha" in fp:
            attested = fp["arm_inputs_sha"] == old_scoped_cache[(si, arm)]
        else:
            attested = fp.get("inputs_sha") == old_global
        if not attested:
            raise SystemExit(
                f"[budget] migrate: row {key} matches NEITHER the old world "
                f"at {old_root} nor the new one — unknown provenance; wrong "
                "--old-repo/--old-config, or a mixed trials file. Refusing.")
        # 2. keep iff this (si, arm)'s inputs are unchanged old -> new
        if old_scoped_cache[(si, arm)] == new_scoped_cache[(si, arm)]:
            new_fp = {
                "model": cfg.get("model"),
                "model_revision": cfg.get("model_revision"),
                "arm_inputs_sha": new_scoped_cache[(si, arm)],
                "seq_construction_version": SEQ_CONSTRUCTION_VERSION,
                "gpu_name": fp["gpu_name"],
                "torch": fp["torch"],
                "cuda": fp["cuda"],
                "shapes": fp["shapes"],
            }
            kept.append({**row, "fingerprint": new_fp})
        else:
            superseded.append(key)

    sidecar = trials_path.with_name(trials_path.name + ".superseded.jsonl")
    now = datetime.now(timezone.utc).isoformat()
    for key in superseded:
        _append_trial(sidecar, {**rows[key], "superseded_at": now,
                                "superseded_by": "migrate"})
    tmp = trials_path.with_name(trials_path.name + ".tmp")
    with tmp.open("w") as f:
        for row in kept:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, trials_path)
    print(f"[budget] migrate: kept {len(kept)} row(s) (inputs unchanged), "
          f"superseded {len(superseded)}: "
          f"{sorted({(s, a) for s, a, _ in superseded})} — the resumed run "
          "re-measures only those")


def _write_figures(measured: dict, cfg: dict, figs_dir: Path) -> None:
    """Peak-GB (alloc + reserved) and tokens/s heatmaps over (SI, arm) at the
    safe width — to $FIGS_DIR (figure mirror) and W&B when live."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sis = list(cfg["si_ids"])
    arms = registered_class_a_arms(cfg)
    safe = measured["safe_batch_size"]
    by_key = {(t["si_id"], t["arm"], t["batch_size"]): t
              for t in measured["trials"]}

    def grid(field):
        g = np.full((len(sis), len(arms)), np.nan)
        for i, si in enumerate(sis):
            for j, arm in enumerate(arms):
                t = by_key.get((si, arm, safe[si][arm]))
                if t is not None and t.get(field) is not None:
                    g[i, j] = t[field]
        return g

    figs_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for field, label, fname in (
            ("peak_gb_reserved", "peak reserved GB @ safe bs",
             "final_budget_peak_gb_reserved.png"),
            ("peak_gb_alloc", "peak allocated GB @ safe bs",
             "final_budget_peak_gb_alloc.png"),
            ("tokens_per_s_median", "median tokens/s @ safe bs",
             "final_budget_tokens_per_s.png")):
        g = grid(field)
        fig, ax = plt.subplots(figsize=(1.1 * len(arms) + 2, 2.4))
        im = ax.imshow(g, aspect="auto")
        ax.set_xticks(range(len(arms)), arms, rotation=45, ha="right")
        ax.set_yticks(range(len(sis)), sis)
        for i in range(len(sis)):
            for j in range(len(arms)):
                if not np.isnan(g[i, j]):
                    ax.text(j, i, f"{g[i, j]:.3g}\nbs{safe[sis[i]][arms[j]]}",
                            ha="center", va="center", fontsize=7)
        ax.set_title(label)
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        out = figs_dir / fname
        fig.savefig(out, dpi=160)
        plt.close(fig)
        paths[label] = out
        print(f"[budget] figure: {out}")
    _log_wandb_figures(paths)


def _log_wandb_figures(paths: dict) -> None:
    """Log the budget figures to W&B (the project: WANDB_PURPOSE's) unless
    WANDB_DISABLED; a failure warns and never fails the emit."""
    try:
        import wandb

        if os.environ.get("WANDB_DISABLED", "").lower() not in ("1", "true"):
            from sisteer.wandb_entity import wandb_project

            run = wandb.init(project=wandb_project(WANDB_PURPOSE),
                             job_type="budget",
                             name=os.environ.get("SLURM_JOB_NAME",
                                                 "evals_budget"))
            run.log({f"budget/{p.stem}": wandb.Image(str(p))
                     for p in paths.values()})
            run.finish()
    except Exception as e:  # noqa: BLE001
        print(f"[budget] WARNING: wandb figure logging failed: {e}",
              file=sys.stderr)


# --------------------------------------------------------------------------- #
# table — CPU render + the §5.3 mechanical guarantee                           #
# --------------------------------------------------------------------------- #


def cmd_table(args, cfg: dict) -> None:
    from sisteer.eval.train_budget import (
        load_measured,
        measurement_inputs_sha,
        resolve_batch_size,
    )

    path = Path(args.measured or cfg.get("measured_path")
                or "configs/two_si_budget_measured.json")
    if not path.is_absolute():
        path = REPO / path
    measured = load_measured(path)
    sha = measurement_inputs_sha(cfg)
    if measured["measurement_inputs_sha"] != sha:
        raise SystemExit(
            f"[budget] {path} is measurement_inputs_sha-incompatible with "
            f"this config/code (table {measured['measurement_inputs_sha'][:12]}… "
            f"!= current {sha[:12]}…) — re-run the GPU pass")
    trials = {(t["si_id"], t["arm"], t["batch_size"]): t
              for t in measured["trials"] if t.get("ok")}
    safe = measured["safe_batch_size"]
    sis = list(cfg["si_ids"])
    arms = registered_class_a_arms(cfg)
    header = (f"{'SI':<16} {'arm':<15} {'bs':>3} {'safe':>4} "
              f"{'peak_alloc':>10} {'peak_resv':>10} {'tok/s':>10}")
    print(header)
    print("-" * len(header))
    violations: list[str] = []
    n_pairs = 0
    for si in sis:
        for arm in arms:
            n_pairs += 1
            resolved = resolve_batch_size(cfg, si, arm)
            s = safe.get(si, {}).get(arm)
            if s is None:
                violations.append(f"({si}, {arm}): no measured safe width")
                continue
            if resolved > s:
                violations.append(
                    f"({si}, {arm}): resolve_batch_size={resolved} > "
                    f"measured-safe {s}")
            t = trials.get((si, arm, resolved))
            if t is None:
                violations.append(
                    f"({si}, {arm}): no successful trial at the resolved "
                    f"bs={resolved} (r3 #2)")
                continue
            print(f"{si:<16} {arm:<15} {resolved:>3} {s:>4} "
                  f"{t['peak_gb_alloc']:>10.2f} {t['peak_gb_reserved']:>10.2f} "
                  f"{t['tokens_per_s_median']:>10.3g}")
    if violations:
        raise SystemExit(
            "[budget] MECHANICAL GUARANTEE VIOLATED (r2 #2 / r3 #2):\n  "
            + "\n  ".join(violations))
    print(f"[budget] OK: resolve_batch_size(si, arm) <= measured-safe for "
          f"all {n_pairs} (si, arm) pairs.")


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--config", default="configs/two_si.yaml")
    r.add_argument("--trials", default=DEFAULT_TRIALS,
                   help="fingerprinted trials JSONL (resume checkpoint)")
    r.add_argument("--emit", default="configs/two_si_budget_measured.json",
                   help="measured-table output path ('' to skip emit)")
    r.add_argument("--figs", default=os.environ.get("FIGS_DIR") or None)
    r.add_argument("--repeats", type=int, default=REPEATS)
    r.add_argument("--si", default="", help="optional si_id filter (csv)")
    r.add_argument("--arms", default="", help="optional arm filter (csv)")

    t = sub.add_parser("trial")
    t.add_argument("--config", default="configs/two_si.yaml")
    t.add_argument("--si", required=True)
    t.add_argument("--arm", required=True)
    t.add_argument("--batch-size", type=int, required=True)
    t.add_argument("--repeats", type=int, default=REPEATS)
    t.add_argument("--row-out", required=True)

    e = sub.add_parser("emit")
    e.add_argument("--config", default="configs/two_si.yaml")
    e.add_argument("--trials", default=DEFAULT_TRIALS)
    e.add_argument("--emit", default="configs/two_si_budget_measured.json")
    e.add_argument("--figs", default=None)

    b = sub.add_parser("table")
    b.add_argument("--config", default="configs/two_si.yaml")
    b.add_argument("--measured", default=None)

    m = sub.add_parser("migrate")
    m.add_argument("--config", default="configs/two_si.yaml")
    m.add_argument("--trials", default=DEFAULT_TRIALS)
    m.add_argument("--old-config", required=True,
                   help="the two_si.yaml the rows were measured under")
    m.add_argument("--old-repo", required=True,
                   help="a checkout of the exact commit the rows were "
                        "measured under (e.g. the frozen cluster checkout)")
    m.add_argument("--repo-root", default=None,
                   help="override the NEW world's repo root (tests)")

    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    {"run": cmd_run, "trial": cmd_trial, "emit": cmd_emit,
     "table": cmd_table, "migrate": cmd_migrate}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
