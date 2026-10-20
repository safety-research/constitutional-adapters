"""Pod acceptance probe — gate a GPU wave on MEASURED training throughput.

Runs ON the pod being tested, BEFORE any wave commits to it: ~50 REAL
optimizer steps of a real arm on the real model (a truncated slice of the
real pair pool through ``scripts/train_two_si_objects.py train`` itself —
never a synthetic matmul, which passes on hosts whose data path is the
actual bottleneck), then asserts the measured steps/s and GPU utilization
and REJECTS the pod otherwise.

Why this exists: the 37-GPU RunPod wave of 2026-08-22 lost most of its
budget to hosts that ran identical H100s at ~1/6 speed (I/O-starved
datacenters), discovered only hours in. Reference throughputs from that
wave, width-1 arms:

    healthy hosts:      2–6   steps/s,  GPU utilization 90%+
    pathological hosts: 0.3–0.8 steps/s, GPU utilization near 0

The default thresholds (``--min-steps-per-sec 1.0``, ``--min-gpu-util 40``)
sit between those bands FOR THE DEFAULT ARM. Thresholds are PER-ARM-
DEPENDENT — a heavier arm steps slower on a healthy host — so changing
``--arm`` means re-deriving them. ``lora_r1`` is the default probe arm on
purpose: fast, gradient-based (a real optimizer loop with a real backward,
not a closed-form capture), single GPU, batch size 8 per
``configs/two_si.yaml`` ``batch_size_by_arm``.

Invocation (on the pod, from the repo root)::

    uv run --no-sync python scripts/pod_acceptance_probe.py \
        --config configs/two_si.yaml --si <si> --pairs <pool.jsonl> \
        [--arm lora_r1] [--steps 50] [--min-steps-per-sec 1.0] \
        [--min-gpu-util 40] [--out verdict.json] [--timeout 900]

What it does:

1. Sizes a head-truncated copy of ``--pairs`` so one epoch of the arm is
   ~``--steps`` optimizer steps (``steps × batch_size`` fit rows, grossed
   up for the holdout carve), using the repo's own authoritative
   ``sisteer.eval.train_budget.resolve_batch_size``.
2. Derives a probe config from ``--config``: ``data_parallel.gpus: 1``
   (the single-process width-1 path — the reference numbers above are
   width-1), ``train_kwargs.default.epochs: 1``, ``floor_ratio_max: null``
   (a 1-epoch truncated fit legitimately plateaus above the 3× no-op
   floor; the probe judges THROUGHPUT, never fit quality), and
   ``wandb_fits: false``. Everything else — model pin, dtype, max_len,
   batch widths — is byte-preserved from the wave config.
3. Runs ``train_two_si_objects.py precompute-tokens`` then ``... train``
   as subprocesses (the true wave path, token-sidecar warm included),
   never modifying the trainer.
4. MEASURES, never infers: steps/s over the LAST ``--tail-steps`` steps
   only (wall-clock between line-count samples of the run's metrics
   jsonl — the per-step lines the trainer already writes — so the
   encode/tokenization phase is excluded), and mean GPU utilization
   sampled via ``nvidia-smi`` every ``--poll`` seconds during the
   stepping window only (max across visible GPUs per sample: the trainer
   uses one GPU, and idle siblings must not dilute the mean).
5. Verdict: prints exactly one machine-readable final line —
   ``ACCEPT steps_per_sec=X gpu_util=Y`` or
   ``REJECT reason=<throughput|gpu_util|no_steps|crash> steps_per_sec=X
   gpu_util=Y`` — writes every measurement to ``--out`` json, and exits
   0 on accept / 2 on reject / 1 on probe error. A pod that produces no
   steps within ``--timeout`` (default 900 s = 15 min, the whole-probe
   wall) is ``REJECT reason=no_steps``, never a hang.

Scope limits (all deliberate):
- DPO preference arms (``bidpo``, ``bidpo_lora``) are REFUSED as probe
  arms: they train on their own sha-pinned pool files, which a truncated
  copy cannot satisfy.
- Closed-form arms (e.g. ``diffmean``) produce no optimizer-step lines
  and would read as ``no_steps``; probe with a gradient arm.
- Stage the model in the pod's HF cache before probing: a cache miss
  spends the timeout on a ~16 GB download and rejects a healthy pod.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

REPO = Path(__file__).resolve().parents[1]
TRAINER = REPO / "scripts" / "train_two_si_objects.py"

ACCEPT = "ACCEPT"
REJECT = "REJECT"
REASON_THROUGHPUT = "throughput"
REASON_GPU_UTIL = "gpu_util"
REASON_NO_STEPS = "no_steps"
REASON_CRASH = "crash"

#: Below this many observed optimizer steps the tail-rate estimate is too
#: thin to certify a pod either way — the verdict is ``no_steps``. Clamped
#: to ``--steps`` so a deliberately tiny probe stays judgeable.
MIN_MEASURED_STEPS = 10

#: Gross-up on the pair count for the hash-based holdout carve's variance
#: (split_pairs is per-prompt sha, so the realised holdout fraction wobbles
#: around holdout_frac; 15% margin is >> the binomial spread at ~500 pairs).
PAIR_MARGIN = 1.15

#: Per-arm summary keys that end in ``_loss`` but are NOT per-step lines:
#: ``heldout_loss`` (once per epoch, plus a pre-training baseline),
#: ``step0_loss`` (startup_loss_check, once per arm) and
#: ``final_probe_loss`` (floor_ratio_check, once per arm).
_NON_STEP_SUFFIXES = ("heldout_loss", "step0_loss", "final_probe_loss")


class ProbeError(RuntimeError):
    """The PROBE itself failed (bad invocation, no nvidia-smi, measurement
    infrastructure broke) — exit 1, distinct from a pod verdict."""


# --------------------------------------------------------------------------- #
# Pure decision + measurement logic (unit-tested; no GPU, no subprocesses)     #
# --------------------------------------------------------------------------- #


def decide(*, steps_total: int, steps_per_sec: float | None,
           gpu_util: float | None, crashed: bool,
           min_steps_per_sec: float, min_gpu_util: float,
           min_measured_steps: int = MIN_MEASURED_STEPS
           ) -> tuple[str, str | None]:
    """Thresholds → verdict. Returns ``(ACCEPT, None)`` or
    ``(REJECT, reason)``. Priority order, first match wins:

    1. ``crash``      — the trainer exited nonzero ON ITS OWN (a timeout
                        kill by the probe is NOT a crash; the caller must
                        not set ``crashed`` for it).
    2. ``no_steps``   — fewer than ``min_measured_steps`` optimizer steps
                        observed, or no computable tail rate: covers a
                        clean-but-stepless exit, a timeout with nothing
                        produced, and a measurement window too thin to
                        certify either way.
    3. ``throughput`` — measured tail steps/s below ``min_steps_per_sec``.
    4. ``gpu_util``   — mean stepping-window utilization below
                        ``min_gpu_util``.

    Thresholds are inclusive: a measurement exactly AT the threshold
    passes (the thresholds are chosen between the healthy and
    pathological bands, so the boundary is healthy territory).

    ``gpu_util is None`` once throughput has passed is a MEASUREMENT
    failure, not pod evidence — raises ``ValueError`` (the caller turns
    it into a probe error, exit 1) rather than guessing a verdict.
    """
    if crashed:
        return REJECT, REASON_CRASH
    if steps_total < min_measured_steps or steps_per_sec is None:
        return REJECT, REASON_NO_STEPS
    if steps_per_sec < min_steps_per_sec:
        return REJECT, REASON_THROUGHPUT
    if gpu_util is None:
        raise ValueError(
            "gpu_util is unmeasured but the throughput gate passed — a "
            "verdict would be a guess; fix the utilization sampling "
            "(nvidia-smi) instead")
    if gpu_util < min_gpu_util:
        return REJECT, REASON_GPU_UTIL
    return ACCEPT, None


def is_step_line(rec) -> bool:
    """Whether one parsed metrics-jsonl record is a PER-STEP optimizer line.

    The trainer's per-step lines (lockstep: ``lockstep/<arm>/<loss>_loss``;
    solo distill: ``distill/<loss>_loss``) carry a non-null ``step`` and a
    numeric ``*_loss`` value. Per-arm summaries that also end in ``_loss``
    (:data:`_NON_STEP_SUFFIXES`) are excluded — ``heldout_loss`` lines
    carry a non-null step too, so the suffix filter is load-bearing."""
    if not isinstance(rec, dict) or rec.get("step") is None:
        return False
    for k, v in rec.items():
        if (isinstance(k, str) and isinstance(v, (int, float))
                and not isinstance(v, bool)
                and k.endswith("_loss") and not k.endswith(_NON_STEP_SUFFIXES)):
            return True
    return False


def count_step_lines(path: Path) -> int:
    """Count optimizer-step lines in a metrics jsonl, tolerating a missing
    file (0 — the trainer hasn't reached its first step) and an unparseable
    tail line (the trainer flushes per line, but a read can race the append
    mid-write; a torn line must skip, never crash the monitor)."""
    path = Path(path)
    if not path.exists():
        return 0
    n = 0
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    for line in text.splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if is_step_line(rec):
            n += 1
    return n


@dataclass
class TailRate:
    """Steps/s over the last-K window: ``steps`` counted between the chosen
    start sample and the FIRST sample observing the final count, over
    ``seconds`` of wall clock between those two samples."""
    steps: int
    seconds: float
    rate: float


def steps_per_sec_tail(samples: list[tuple[float, int]],
                       tail_steps: int) -> TailRate | None:
    """Steps/s over the LAST ``tail_steps`` steps only, from monotonic
    ``(wall_time, cumulative_step_count)`` samples.

    Window start: the LATEST sample whose count is <= (final − tail_steps)
    — for a short run that is the last all-zero sample, so the encode/
    tokenization phase (during which the count sits at 0) is excluded by
    construction. Window end: the FIRST sample observing the final count,
    so post-training work (held-out pass, save) never dilutes the rate.
    Both bounds are conservative at poll granularity (they can only
    UNDERestimate the true rate, never inflate a slow pod).

    Returns ``None`` when no rate is computable (no samples, zero steps,
    or every observed step predates the first usable window)."""
    if not samples:
        return None
    final = samples[-1][1]
    if final <= 0:
        return None
    end_idx = next(i for i, (_, c) in enumerate(samples) if c >= final)
    target = max(final - tail_steps, 0)
    start_idx = None
    for i, (_, c) in enumerate(samples):
        if c <= target:
            start_idx = i
        else:
            break  # counts are non-decreasing
    if start_idx is None:
        start_idx = 0  # steps already underway at the first sample
    if start_idx >= end_idx:
        return None
    t0, c0 = samples[start_idx]
    t1, c1 = samples[end_idx]
    if c1 <= c0 or t1 <= t0:
        return None
    return TailRate(steps=c1 - c0, seconds=t1 - t0,
                    rate=(c1 - c0) / (t1 - t0))


def stepping_window(counts: list[int]) -> tuple[int, int] | None:
    """Index range ``(first, last)`` (inclusive) of samples taken DURING
    the stepping phase: from the first sample with a nonzero count to the
    first sample observing the final count. ``None`` when no step was ever
    observed."""
    if not counts or counts[-1] <= 0:
        return None
    final = counts[-1]
    first = next(i for i, c in enumerate(counts) if c > 0)
    last = next(i for i, c in enumerate(counts) if c >= final)
    return first, last


def mean_gpu_util(samples: list[tuple[float, int, float | None]]
                  ) -> float | None:
    """Mean GPU utilization over samples inside the stepping window only —
    encode/model-load samples (count still 0) and post-training samples
    (after the final count was first observed) are excluded, exactly as
    required for the utilization number to describe the OPTIMIZER phase.
    Samples whose utilization read failed (``None``) are skipped; returns
    ``None`` when the window holds no usable sample."""
    w = stepping_window([c for _, c, _ in samples])
    if w is None:
        return None
    utils = [u for _, _, u in samples[w[0]:w[1] + 1] if u is not None]
    if not utils:
        return None
    return sum(utils) / len(utils)


# --------------------------------------------------------------------------- #
# Pilot sustained-duty helpers (pod_pilot.sh step 7; review fix F11)            #
#                                                                               #
# These live HERE so the pilot and the probe share ONE optimizer-step-line      #
# definition (is_step_line/count_step_lines above): the pilot's duty window     #
# opens on the first true optimizer step, never on "any metrics line" — the     #
# trainer emits step0_loss and an initial held-out eval (high-utilization       #
# forwards) BEFORE the optimizer loop, so a window keyed on file-nonempty can   #
# measure startup and false-GO a ragged optimizer phase.                        #
# --------------------------------------------------------------------------- #

DUTY_GO = "GO"
DUTY_NO_GO = "NO-GO"
DUTY_TRAIN_EXITED = "train_exited"
DUTY_THIN_WINDOW = "thin_window"
DUTY_NO_STEP_ADVANCE = "no_step_advance"
DUTY_LOW = "low_duty"

#: A sustained window needs at least this many usable 1 s SM% samples —
#: shorter windows lie for bursty loads.
MIN_DUTY_SAMPLES = 20


def parse_dmon_sm_max(text: str) -> list[float]:
    """Per-sample MAX SM% across GPUs from ``nvidia-smi dmon -s u`` output.

    Rows are ``<gpu> <sm> <mem> <enc> <dec> [...]``; ``#`` lines are headers
    and ``-`` means unsupported (the row is skipped). One row per GPU per
    sample; the pilot's run is width-1 and multi-GPU pods can rotate per-GPU
    load, so each sample takes the max across GPUs. Truncated output
    (unequal per-GPU row counts) keeps only the complete samples."""
    per_gpu: dict[int, list[float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        toks = line.split()
        try:
            gpu, sm = int(toks[0]), float(toks[1])
        except (ValueError, IndexError):
            continue
        per_gpu.setdefault(gpu, []).append(sm)
    if not per_gpu:
        return []
    n = min(len(v) for v in per_gpu.values())
    return [max(per_gpu[g][i] for g in per_gpu) for i in range(n)]


def duty_verdict(*, sample_max: list[float], min_duty: float,
                 want_samples: int, train_alive_through_window: bool,
                 steps_before: int, steps_after: int) -> dict:
    """The pilot's step-7 sustained-duty GO/NO-GO record.

    GO requires ALL of: the trainer ran through the whole dmon window, the
    window is wide enough to be sustained, the optimizer step count ADVANCED
    across the window (``count_step_lines`` before vs after — a hot window
    with no step advance measured startup work, not the optimizer phase),
    and mean SM% >= ``min_duty`` (inclusive). NO-GO reasons, priority order:

    1. ``train_exited``    — the trainer stopped running inside the window,
                             so part of it sampled an idle GPU.
    2. ``thin_window``     — fewer than max(:data:`MIN_DUTY_SAMPLES`,
                             ``want_samples`` − 5) usable samples.
    3. ``no_step_advance`` — zero optimizer steps completed across the
                             window (the F11 false-GO: pre-training forwards
                             run at high SM% too).
    4. ``low_duty``        — mean SM% below ``min_duty``.
    """
    n = len(sample_max)
    mean = (sum(sample_max) / n) if n else None
    if not train_alive_through_window:
        reason = DUTY_TRAIN_EXITED
    elif n < max(MIN_DUTY_SAMPLES, int(want_samples) - 5):
        reason = DUTY_THIN_WINDOW
    elif steps_after <= steps_before:
        reason = DUTY_NO_STEP_ADVANCE
    elif mean < float(min_duty):
        reason = DUTY_LOW
    else:
        reason = None
    return {
        "mean_sm_pct": None if mean is None else round(mean, 1),
        "n_samples": n,
        "sample_interval_s": 1,
        "per_sample_max_sm": list(sample_max),
        "threshold_pct": float(min_duty),
        "steps_before": int(steps_before),
        "steps_after": int(steps_after),
        "train_alive_through_window": bool(train_alive_through_window),
        "verdict": DUTY_GO if reason is None else DUTY_NO_GO,
        "no_go": reason,
    }


def pairs_needed(steps: int, batch_size: int, holdout_frac: float,
                 margin: float = PAIR_MARGIN) -> int:
    """How many pool rows to head-truncate so ONE epoch is ~``steps``
    optimizer steps: ``steps × batch_size`` fit rows, grossed up for the
    holdout carve (the trainer fits on ``(1 − holdout_frac)`` of the pool)
    plus :data:`PAIR_MARGIN` for the hash split's variance."""
    if steps < 1 or batch_size < 1:
        raise ValueError(f"steps and batch_size must be >= 1, got "
                         f"steps={steps} batch_size={batch_size}")
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
    return max(1, math.ceil(steps * batch_size / (1.0 - holdout_frac)
                            * margin))


def truncate_pool(src: Path, dst: Path, n: int) -> int:
    """Write the first ``n`` non-blank lines of ``src`` to ``dst``
    byte-preserved (head truncation — valid because every non-DPO arm
    trains on the full pool, so a prefix is a real sub-pool). Returns the
    number written, which is < ``n`` when the pool is smaller.

    ``dst`` is chmod'd 0600 unconditionally (review fix F13): the copy is
    raw attack prompts/completions landing on ``/workspace``, which other
    users can read on shared mounts — and a rewrite of a pre-existing file
    keeps its old (possibly 0644) mode, so umask alone is not enough."""
    taken = 0
    with open(src, encoding="utf-8") as fin, \
            open(dst, "w", encoding="utf-8") as fout:
        for line in fin:
            if taken >= n:
                break
            if not line.strip():
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            taken += 1
    os.chmod(dst, 0o600)
    return taken


def derive_probe_config(cfg: dict) -> dict:
    """The probe's config: the wave config with exactly four overrides —
    ``data_parallel.gpus: 1`` (single-process width-1 path, matching the
    reference numbers), ``train_kwargs.default.epochs: 1`` (one truncated
    epoch is the probe), ``train_kwargs.default.floor_ratio_max: null``
    (the no-op-floor guard judges fit QUALITY, which a 1-epoch truncated
    fit legitimately fails; the probe judges throughput) and
    ``wandb_fits: false`` (no W&B init on a throwaway pod). The
    floor-bar null covers PER-ARM ``train_kwargs.<arm>.floor_ratio_max``
    overrides too — arm_floor_ratio_max lets a per-arm bar BEAT the
    default, so nulling only the default would leave e.g. the DPO arms'
    own bar judging fit quality on the probe's truncated fit. Everything
    else — model pin, dtype, max_len, batch widths, per-arm schedule
    kwargs — is preserved, and the input dict is never mutated."""
    out = deepcopy(cfg)
    out["data_parallel"] = {"gpus": 1}
    tk_all = dict(out.get("train_kwargs") or {})
    tk = dict(tk_all.get("default") or {})
    tk["epochs"] = 1
    tk["floor_ratio_max"] = None
    tk_all["default"] = tk
    for arm, kw in tk_all.items():
        if (arm != "default" and isinstance(kw, dict)
                and "floor_ratio_max" in kw):
            kw = dict(kw)
            kw["floor_ratio_max"] = None
            tk_all[arm] = kw
    out["train_kwargs"] = tk_all
    out["wandb_fits"] = False
    return out


def format_final_line(verdict: str, reason: str | None,
                      steps_per_sec: float | None,
                      gpu_util: float | None) -> str:
    """The one machine-readable final line. Unmeasured values print as
    ``nan`` (parseable by ``float()``), never silently as 0."""
    x = "nan" if steps_per_sec is None else f"{steps_per_sec:.3f}"
    y = "nan" if gpu_util is None else f"{gpu_util:.1f}"
    if verdict == ACCEPT:
        return f"ACCEPT steps_per_sec={x} gpu_util={y}"
    return f"REJECT reason={reason} steps_per_sec={x} gpu_util={y}"


# --------------------------------------------------------------------------- #
# The monitor loop (impure edges injected, so the loop itself is testable)     #
# --------------------------------------------------------------------------- #


@dataclass
class MonitorResult:
    samples: list = field(default_factory=list)  # (t, steps, util | None)
    timed_out: bool = False
    returncode: int | None = None


def monitor_trainer(proc, *, count_steps, sample_util, deadline: float,
                    poll_s: float, now=time.monotonic, sleep=time.sleep,
                    kill=None) -> MonitorResult:
    """Poll the trainer: every ``poll_s`` record ``(now, step_count,
    gpu_util)``. Exits at the trainer's exit or at ``deadline`` (kill the
    process group — the probe NEVER hangs). The post-exit/post-kill sample
    carries the authoritative final count with ``util=None``, so an idle
    GPU after the last step can never dilute the stepping-window mean."""
    kill = kill or _kill_tree
    res = MonitorResult()
    while True:
        rc = proc.poll()
        if rc is not None:
            res.returncode = rc
            res.samples.append((now(), count_steps(), None))
            return res
        if now() >= deadline:
            res.timed_out = True
            kill(proc)
            res.returncode = proc.poll()
            res.samples.append((now(), count_steps(), None))
            return res
        res.samples.append((now(), count_steps(), sample_util()))
        sleep(poll_s)


def _kill_tree(proc) -> None:
    """SIGTERM the trainer's process group, escalate to SIGKILL after 10 s
    (the trainer is spawned with ``start_new_session=True``, so children —
    tokenizer workers, CUDA compile helpers — die with it)."""
    for sig, patience in ((signal.SIGTERM, 10), (signal.SIGKILL, None)):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=patience)
            return
        except subprocess.TimeoutExpired:
            continue


# --------------------------------------------------------------------------- #
# Impure edges: nvidia-smi + trainer subprocess plumbing                       #
# --------------------------------------------------------------------------- #


def _gpu_preflight() -> list[str]:
    """The GPUs nvidia-smi can see, by name — BEFORE any trainer spawn, so
    a pod where utilization cannot be measured is a probe ERROR in
    milliseconds, never a verdict built on inferred numbers."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        raise ProbeError(
            f"nvidia-smi unavailable ({type(e).__name__}: {e}) — GPU "
            "utilization cannot be measured, so no verdict is possible")
    if out.returncode != 0:
        raise ProbeError(
            f"nvidia-smi failed (exit {out.returncode}): "
            f"{out.stderr.strip()[:500]}")
    names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
    if not names:
        raise ProbeError("nvidia-smi reports no GPUs on this host")
    return names


def _sample_gpu_util() -> float | None:
    """One utilization sample: MAX across visible GPUs (the trainer uses a
    single GPU; idle siblings on a multi-GPU pod must not dilute the mean).
    Honors an integer-list ``CUDA_VISIBLE_DEVICES``. Returns ``None`` on a
    failed read — the sample is skipped, and an all-``None`` window
    surfaces as a probe error downstream, never a silent 0."""
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu",
           "--format=csv,noheader,nounits"]
    ids = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if ids and all(tok.strip().isdigit() for tok in ids.split(",")):
        cmd += ["-i", ids]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if out.returncode != 0:
        return None
    vals = []
    for ln in out.stdout.splitlines():
        try:
            vals.append(float(ln.strip()))
        except ValueError:
            continue
    return max(vals) if vals else None


def _build_precompute_cmd(cfg_path: Path, si: str, pairs_path: Path
                          ) -> list[str]:
    return [sys.executable, str(TRAINER), "precompute-tokens",
            "--config", str(cfg_path), "--si", si, "--pairs", str(pairs_path)]


def _build_train_cmd(cfg_path: Path, si: str, arm: str, pairs_path: Path,
                     out_dir: Path) -> list[str]:
    return [sys.executable, str(TRAINER), "train",
            "--config", str(cfg_path), "--si", si, "--arms", arm,
            "--pairs", str(pairs_path), "--out", str(out_dir), "--overwrite"]


def _subprocess_env() -> dict:
    env = dict(os.environ)
    env["WANDB_DISABLED"] = "true"       # belt on top of wandb_fits: false
    env.setdefault("WANDB_MODE", "disabled")
    env["PYTHONUNBUFFERED"] = "1"        # live train.log tails
    return env


def _log_tail(path: Path, max_bytes: int = 8192) -> str:
    try:
        data = Path(path).read_bytes()
    except OSError:
        return ""
    return data[-max_bytes:].decode("utf-8", errors="replace")


# --------------------------------------------------------------------------- #
# The probe                                                                    #
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/two_si.yaml",
                    help="the WAVE config; the probe derives its own copy "
                         "(gpus=1, epochs=1, floor guard off, no W&B) and "
                         "never modifies this file")
    ap.add_argument("--si", required=True,
                    help="SI id, as for train_two_si_objects.py")
    ap.add_argument("--pairs", required=True,
                    help="the REAL pair pool jsonl; the probe trains on a "
                         "head-truncated copy sized to ~--steps steps")
    ap.add_argument("--arm", default="lora_r1",
                    help="probe arm (default lora_r1: fast, gradient-based, "
                         "single GPU, bs=8). Must be a gradient arm that "
                         "trains on the full pool — DPO preference arms are "
                         "refused, closed-form arms read as no_steps")
    ap.add_argument("--steps", type=int, default=50,
                    help="target optimizer steps (~50 ≈ a 10-minute probe)")
    ap.add_argument("--min-steps-per-sec", type=float, default=1.0,
                    help="ACCEPT floor on tail steps/s (healthy width-1 "
                         "lora_r1: 2–6; pathological: 0.3–0.8)")
    ap.add_argument("--min-gpu-util", type=float, default=40.0,
                    help="ACCEPT floor on mean stepping-window GPU util %% "
                         "(healthy: 90+; I/O-starved: near 0)")
    ap.add_argument("--out", default="verdict.json",
                    help="verdict + all measurements land here as json")
    ap.add_argument("--timeout", type=float, default=900.0,
                    help="whole-probe wall clock in SECONDS (default 900 = "
                         "15 min); zero steps by then => REJECT no_steps")
    ap.add_argument("--tail-steps", type=int, default=30,
                    help="K: steps/s is measured over the LAST K steps only")
    ap.add_argument("--poll", type=float, default=2.0,
                    help="sampling cadence (metrics line count + nvidia-smi)")
    ap.add_argument("--scratch", default="",
                    help="scratch dir (default: a fresh temp dir; kept after "
                         "the run and named in the verdict json)")
    return ap


def run_probe(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    # Everything the probe (and the trainer it spawns) writes to scratch is
    # raw pair data or derived from it, on a mount other users can read —
    # owner-only by construction (review fix F13).
    os.umask(0o077)
    started_unix = time.time()
    t0 = time.monotonic()
    deadline = t0 + float(args.timeout)

    verdict_out = Path(args.out)
    record: dict = {
        "probe": "pod_acceptance_probe",
        "hostname": socket.gethostname(),
        "started_unix": started_unix,
        "config": str(args.config),
        "si": args.si,
        "arm": args.arm,
        "pairs": str(args.pairs),
        "target_steps": args.steps,
        "tail_steps": args.tail_steps,
        "poll_s": args.poll,
        "timeout_s": args.timeout,
        "thresholds": {"min_steps_per_sec": args.min_steps_per_sec,
                       "min_gpu_util": args.min_gpu_util},
    }

    def _fail(msg: str) -> int:
        record.update({"verdict": "error", "error": msg})
        _write_json(verdict_out, record)
        print(f"PROBE-ERROR: {msg}", file=sys.stderr)
        return 1

    try:
        import yaml

        import train_two_si_objects as t2  # sibling script (sys.path above)
        from sisteer.eval.train_budget import resolve_batch_size

        if args.arm in t2.DPO_PREF_ARMS:
            raise ProbeError(
                f"--arm {args.arm} is a DPO preference arm: it trains on "
                "its own sha-pinned pool files, which a truncated pool "
                "cannot satisfy — probe with a full-pool gradient arm "
                "(default lora_r1)")

        record["gpus"] = _gpu_preflight()

        cfg_path = Path(args.config)
        if not cfg_path.is_absolute():
            cfg_path = REPO / args.config
        cfg = yaml.safe_load(cfg_path.read_text())
        probe_cfg = derive_probe_config(cfg)

        batch_size = resolve_batch_size(cfg, args.si, args.arm)
        holdout_frac = float(
            ((cfg.get("train_kwargs") or {}).get("default") or {})
            .get("holdout_frac", 0.1))
        n_pairs = pairs_needed(args.steps, batch_size, holdout_frac)

        scratch = Path(args.scratch) if args.scratch else Path(
            tempfile.mkdtemp(prefix="pod-accept-probe-"))
        scratch.mkdir(parents=True, exist_ok=True)
        # mkdir -p leaves a pre-existing dir's mode alone; enforce 0700 (F13).
        os.chmod(scratch, 0o700)
        out_dir = scratch / "objects"
        out_dir.mkdir(exist_ok=True)
        os.chmod(out_dir, 0o700)
        trunc = scratch / "probe_pairs.jsonl"
        n_written = truncate_pool(Path(args.pairs), trunc, n_pairs)
        if n_written < n_pairs:
            print(f"[probe] pool has only {n_written} pairs "
                  f"(wanted {n_pairs}) — fewer steps than targeted",
                  file=sys.stderr)
        probe_cfg_path = scratch / "probe_config.yaml"
        probe_cfg_path.write_text(yaml.safe_dump(probe_cfg, sort_keys=False))
        # The trainer owns this layout — a rename there moves this poll too.
        metrics_path = t2.metrics_path_for(out_dir, args.si)
        train_log = scratch / "train.log"
        record.update({
            "scratch": str(scratch),
            "batch_size": batch_size,
            "holdout_frac": holdout_frac,
            "pairs_requested": n_pairs,
            "pairs_used": n_written,
            "expected_steps": math.ceil(
                n_written * (1.0 - holdout_frac) / batch_size),
            "metrics_jsonl": str(metrics_path),
            "train_log": str(train_log),
        })
        env = _subprocess_env()
        print(f"[probe] scratch={scratch}", file=sys.stderr)
    except ProbeError as e:
        return _fail(str(e))
    except Exception as e:  # noqa: BLE001 — a probe must never die silently
        return _fail(f"{type(e).__name__}: {e}")

    # ---- phase 1: token sidecar (the true wave path warms from one) --------
    crashed = False
    timed_out = False
    samples: list[tuple[float, int, float | None]] = []
    with open(train_log, "w", encoding="utf-8") as log_fh:
        pre_cmd = _build_precompute_cmd(probe_cfg_path, args.si, trunc)
        record["precompute_cmd"] = pre_cmd
        pre = subprocess.Popen(pre_cmd, stdout=log_fh,
                               stderr=subprocess.STDOUT, env=env,
                               start_new_session=True)
        try:
            pre_rc = pre.wait(timeout=max(1.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            _kill_tree(pre)
            pre_rc = pre.returncode
            timed_out = True
        record["precompute_returncode"] = pre_rc
        if timed_out:
            samples.append((time.monotonic(), 0, None))
        elif pre_rc != 0:
            crashed = True
            record["crash_phase"] = "precompute-tokens"

        # ---- phase 2: the real trainer, monitored -------------------------
        if not crashed and not timed_out:
            train_cmd = _build_train_cmd(probe_cfg_path, args.si, args.arm,
                                         trunc, out_dir)
            record["train_cmd"] = train_cmd
            proc = subprocess.Popen(train_cmd, stdout=log_fh,
                                    stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)
            mon = monitor_trainer(
                proc,
                count_steps=lambda: count_step_lines(metrics_path),
                sample_util=_sample_gpu_util,
                deadline=deadline, poll_s=args.poll)
            samples = mon.samples
            timed_out = mon.timed_out
            record["train_returncode"] = mon.returncode
            crashed = (not mon.timed_out and mon.returncode is not None
                       and mon.returncode != 0)
            if crashed:
                record["crash_phase"] = "train"

    # ---- measurements + verdict --------------------------------------------
    steps_total = samples[-1][1] if samples else 0
    tail = steps_per_sec_tail([(t, c) for t, c, _ in samples],
                              args.tail_steps)
    util = mean_gpu_util(samples)
    record.update({
        "timed_out": timed_out,
        "crashed": crashed,
        "steps_total": steps_total,
        "steps_per_sec": None if tail is None else tail.rate,
        "tail_window": None if tail is None else
            {"steps": tail.steps, "seconds": tail.seconds},
        "gpu_util_mean": util,
        "elapsed_s": time.monotonic() - t0,
        "samples": [[round(t - t0, 3), c, u] for t, c, u in samples],
    })
    if crashed or timed_out:
        record["log_tail"] = _log_tail(train_log)

    try:
        verdict, reason = decide(
            steps_total=steps_total,
            steps_per_sec=None if tail is None else tail.rate,
            gpu_util=util, crashed=crashed,
            min_steps_per_sec=args.min_steps_per_sec,
            min_gpu_util=args.min_gpu_util,
            min_measured_steps=min(MIN_MEASURED_STEPS, args.steps))
    except ValueError as e:
        return _fail(str(e))

    record.update({"verdict": verdict, "reason": reason})
    _write_json(verdict_out, record)
    print(format_final_line(verdict, reason,
                            None if tail is None else tail.rate, util))
    return 0 if verdict == ACCEPT else 2


def _write_json(path: Path, record: dict) -> None:
    try:
        Path(path).write_text(json.dumps(record, indent=2) + "\n",
                              encoding="utf-8")
    except OSError as e:
        print(f"[probe] could not write {path}: {e}", file=sys.stderr)


def main() -> None:
    sys.exit(run_probe())


if __name__ == "__main__":
    main()
