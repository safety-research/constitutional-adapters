#!/usr/bin/env python
"""Hyperparameter search + divergence-monitored fits for the two-model
steering-defense sweep's tunable arms.

Generalisation of scripts/tune_lora_all.py (ported at tip
8ded0ea — including the cuDNN-SDPA disable, 8468792, and length-bucketed
trial batches + the worst-case probe-mem mode, 8ded0ea). One harness, five
arms, every trial the PRODUCTION training path for its arm:

  lockstep singletons (psr_kl_all, lora_r1_all, token_faithful) — a
    singleton-group ``train_lockstep`` with the trial's
    (lr, warmup_frac, max_grad_norm) on ``LockstepArmParams`` — exactly the
    seam a winner ships on (``train_kwargs.<arm>``); per-arm registry
    defaults ride ``default_lockstep_params`` (token_faithful keeps
    T=10/loss_scale=100, psr_kl_all keeps reg_coeff=0.1).
  solo psr_mse_all — ``train_psr(loss="mse", ...)`` with EXPLICIT
    lr_schedule/warmup_steps/max_grad_norm always passed (that loop resolves
    ``None`` to its OWN warmup_cosine + clip-1.0 defaults, so "None" is not
    "no schedule" there, unlike the lockstep contract).
  solo bidpo — ``train_bidpo`` on preference pairs routed through the
    PRODUCTION resolver (``pairs_for_arm``: the per-SI own-pool files
    bidpo_harmful_<si>.jsonl + REVERSED-polarity bidpo_benign_<si>.jsonl
    when pool.bidpo_benign is on, holdout-carved and capped exactly as the
    driver does — a matched-pool conversion has the OPPOSITE benign
    polarity and its winners do not transfer). ``accum_steps`` is fixed so
    the effective batch stays >= 128 while lr x clip sweep, and the fit
    rows are trimmed to a MULTIPLE of batch_size x accum_steps so no
    optimizer group is ragged (< 128 rows); its mid-fit probe object is
    the TRANSIENT VectorObject the trainer hands to the
    ``probe_cb(step, obj)`` seam (the live vec is internal to the trainer).

Sobol design (spec Fixed 7; EXPANDED for the re-tune@32,
2026-08-27): warmup_frac in {0.05, 0.15} is a 2-LEVEL FACTOR, never a
Sobol axis. Each level gets its OWN independent d-dimensional Sobol design,
digitally shifted (XOR of the base sequence's integer lattice with a
per-level sha-derived 30-bit vector — an Owen-style digital shift, which
preserves the net properties). The first two axes are always
log(lr) x log(max_grad_norm); the re-tune arms add per-arm DISCRETE axes
snapped to their grids by equal thirds of the unit interval (ARM_AXES):

  psr_kl_all    temperature {1,2,4} x weight_decay {0,1e-3,1e-2}   (d=4)
  psr_mse_all   weight_decay {0,1e-3,1e-2}                          (d=3)
                (NO temperature axis: the mse loss matches teacher
                ACTIVATIONS — train_psr's temperature is a KL-path knob)
  lora_r1_all   temperature x weight_decay x lora_rank {1,4,16}     (d=5)

``--pin-axis name=value`` (repeatable) PINS an arm axis for a follow-up
round: the axis leaves the Sobol dimensions (d shrinks; the remaining axes
keep their ARM_AXES order — Sobol coordinates depend only on their own
direction numbers, so the projection onto the remaining dims is a valid
design and the lr x clip net assertion still binds) and the pinned value is
stamped on every emitted row, so run_trial's TrialHP carries it EXPLICITLY
(the trainer must never fall back to a registry default, which is not 1.0
for every arm). Motivation (empirical, re-HPO wave 2026-08-27): under the
offline top-K teacher head, training psr_kl_all/lora_r1_all at T=2/4
diverges — the head's renormalized mass Z collapses at high T and the
gradient loses the tail pull with weight 1-Z — 27/27 completed T!=1 trials
floor-failed while T=1 trained fine, so those cells' follow-up rounds pin
temperature=1.0. Pins must sit ON the axis grid; lr/max_grad_norm (the
core continuous axes) refuse.

NEVER assign warmup by Sobol-index parity: splitting one 16-point sequence
that way puts the even indices' second coordinate in [0, 0.4375] and the
odd in [0.5, 0.9375] — 5% warmup would only ever see low clip and 15% only
high (warmup aliased with clip). emit-trials asserts, per warmup level:
the exact (0,m,2)-net property of the lr x clip projection on EVERY dyadic
grid (Sobol coordinates depend only on their own direction numbers, so the
2-D projection of the d-D design IS the 2-D design); exact 1-D
stratification of every axis; near-exact per-level balance of every
discrete grid; and a non-aliasing check that every discrete level's points
span the lr and clip ranges. So the aliasing class of bug cannot ship
silently at any d.

Analysis (the re-tune form; the legacy 2-axis pools fit the original
9-term model bit-for-bit through the same code): least squares on log(y)
over the FULL quadratic in the K continuous log-axes
  L = log lr, C = log clip [, tau = log T] [, D = log(weight_decay + 1e-4)]
— 1, the K linears, the K squares, the K(K-1)/2 cross terms — plus the
warmup-level indicator W and its K interactions W*axis, plus (lora only)
LoRA-rank main-effect dummies against the rank-1 baseline (rank is a
FACTOR: best level per model, additive by assumption; the artifact carries
per-rank raw bests so a rank x lr interaction is visible if it matters).
weight_decay=0 rides the log axis as log(0 + 1e-4) — one decade below the
smallest nonzero grid level — and a fitted argmin at that grid edge maps
back to exactly 0.0. An EXTRA axis (temperature / weight_decay, and
likewise a single-level lora_rank) whose value is IDENTICAL across every
pooled usable row is EXCLUDED from the design — a constant column is
rank-deficient by construction, and the pool carries no information along
it — and echoed as a FIXED value instead (fit.fixed_axes + the artifact's
top-level fixed_axes + the winner, e.g. winner.temperature = 1.0). That is
what lets round-1 T=1 survivors (the T!=1 rows floor-excluded upstream)
pool with a ``--pin-axis temperature=1.0`` round 2 into one fit over the
remaining axes. lr is never dropped: a constant there is a degenerate
pool, kept behind the existing rank-deficiency refusal. A constant
max_grad_norm is the sdfbase-llama8b LR-ONLY design (the clip is a
recon-derived per-arm constant, `emit-trials --clip-const`, not a
search axis) and is held FIXED the same way — decided BEFORE the extra
axes, so every lr x clip design's term names stay byte-identical.
All behind the design rank/conditioning check and the
K x K continuous-Hessian PD check. The fitted argmin solves the
unconstrained stationary point per warmup level, then CLAMPS tau/D into
their grid ranges with a reduced re-solve (active-set on a PD quadratic);
lr/clip stay edge-checked against the box (edge -> recenter). Because
temperature CHANGES the KL objective's scale, trials in a
temperature-swept cell are ranked by the T=1 arm-agnostic probe residual
KL (final_residual_kl), never by the trainer's own-T held-out loss;
analyze refuses a pool that mixes ranking objectives. That own-T guard
binds when temperature actually VARIES across the pooled usable rows — a
constant-T pool trains ONE objective, so there is no across-T comparison
to protect.

Floor-failing (floor_ok=false) and non-finite rows are EXCLUDED before the
fit (a floor-failing trial is a failed fit, not a data point; the exclusion
count lands in the fit artifact). Selection is the FITTED ARGMIN, never the
best raw trial; a fitted argmin on the lr/clip box edge recommends
re-center + re-run at the same width; otherwise the refine round is 8
trials keeping BOTH warmup levels (4+4) on a box zoomed to ~1/3 log-width
(discrete grids ride unchanged); the winner refit pools the ok-trials of
both rounds (``analyze --rounds 1,2``). Winners are recorded as
warmup_frac — never a copied step count — plus temperature/weight_decay/
lora_rank and the realized (batch_size, accum_steps) where swept.

Effective batch (re-tune@32): the three re-tune arms train at effective
batch 32 via gradient accumulation — accum_steps = ceil(32 / batch_size),
threaded to LockstepArmParams.accum_steps (KL arms) / train_psr's
accum_steps (mse) — and the offline top-K teacher artifact
(--teacher-topk, sisteer.extraction.teacher_topk) replaces the online
teacher leg for the KL arms. NOTE the trainer-side gate currently refuses
top-K consumption at temperature != 1: the re-tune needs that gate relaxed
to the audited grid {1,2,4} (the renormalized-head KL is well-defined at
any T; the re-tuned objective IS the head objective) — a T!=1 trial
launched before that lands refuses loudly at setup, by design.

Monitoring hygiene (the gpu-util footgun class): the per-step monitoring
this harness ADDS never syncs CPU<->GPU per step — grad-norm scalars are
accumulated as 0-dim GPU tensors and drained in ONE sync at the log/probe
cadence. cuDNN SDPA is disabled on every CUDA run (per-ragged-shape CPU
plan compilation halved GPU duty on ragged batches); trial batches are
length-bucketed by default; probe-mem probes the WORST-CASE (longest-rows)
batch by default, because memory peaks on the densest batch's first
optimizer step.

Data: --pairs points at the blessed pool copy WITH its .tokcache.npz
sidecar next to it (warmed with the pool-content guard, so a stale sidecar
refuses); the run refuses a silent full tokenize unless --allow-tokenize is
given. bidpo cells additionally read the emit-pairs own-pool files —
default siblings beside --pairs, or --bidpo-harmful-pairs /
--bidpo-benign-pairs — through the driver's own loader (sidecar sha
verification included). All tuning runs on si_constitution.

Identity: emit-trials stamps the payload with the config's canonical model
repo id + revision + config-file sha; `trial` refuses on any mismatch with
its --config BEFORE the idempotent skip or any model load, and the skip
itself is gated on a stored fit_fingerprint (model+revision, SI, arm,
positions, pool content incl. bidpo's own files, trial hp, and the search
coordinates/budget) matching the requested work — a torn or stale
summary.json re-runs instead of being trusted.

Subcommands: emit-trials / probe-mem / trial / analyze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

import sys

import yaml

# torch is imported INSIDE the five GPU-side functions that use it
# (GradNormBuffer._materialize, ArmAdapter.norm_groups, probe_mem, bench,
# run_trial): emit-trials, analyze, hpo_table, hpo_pick and the stage
# scripts' constant reads load this module on the grading host and must not pay
# the ~3 s torch import for pure-python work (paper-companion-repo, 2026-09-29).

# The HPO selection guard (scripts/hpo_selection_guard.py) — the SAME code
# the fold writers run, so analyze's artefact and the fold record file one
# verdict. ``python scripts/tune_arm.py`` has scripts/ on sys.path; a
# by-path load (the tests, qwen35_select-style drivers) does not, so fall
# back to the sibling file.
try:
    import hpo_selection_guard
except ImportError:
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location(
        "hpo_selection_guard",
        Path(__file__).resolve().parent / "hpo_selection_guard.py")
    hpo_selection_guard = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(hpo_selection_guard)

# The reconnaissance clip rule + clip-activation rate
# (scripts/sdfbase/recon_clip.py) — pure python, shared with the Modal app
# and the fold writer so the bench's series, the emit-time clip constant
# and analyze's activation column are ONE derivation. scripts/sdfbase is
# not on sys.path under `python scripts/tune_arm.py`, so load the sibling
# file by path.
try:
    import recon_clip
except ImportError:
    import importlib.util as _ilu2
    _spec2 = _ilu2.spec_from_file_location(
        "recon_clip",
        Path(__file__).resolve().parent / "sdfbase" / "recon_clip.py")
    recon_clip = _ilu2.module_from_spec(_spec2)
    _spec2.loader.exec_module(recon_clip)

REPO = Path(__file__).resolve().parent.parent

#: The two warmup levels of the 2-level factor (spec Fixed 7).
WARMUP_LEVELS = (0.05, 0.15)

#: bidpo's effective-batch floor: batch_size x accum_steps >= this while the
#: lr x clip sweep runs.
EFFECTIVE_BATCH_FLOOR = 128

#: The re-tune arms' effective batch (twosi-model-sweeps re-HPO, 2026-08-27):
#: every psr_kl_all / psr_mse_all / lora_r1_all trial trains at effective
#: batch 32 — accum_steps = ceil(32 / batch_size), one clip per OPTIMIZER
#: step on the accumulated gradient. The old sweeps tuned at effective
#: 2 (Llama/Qwen) / 7 (E4B), which is exactly why their LR anchors are
#: re-tuned rather than carried.
DISTILL_EFFECTIVE_BATCH = 32

#: Early stopping for the lockstep re-tune trials. Patience 3: per-epoch
#: held-out is NOISY and patience-1 false-triggered on single wobbles —
#: 5 full-horizon psr_mse runs kept improving to epoch ~13, and an
#: epoch-2.5 stop left a mean 23.5% of the achievable residual reduction
#: on the table. Three consecutive non-improving epochs cannot be one
#: wobble; best-epoch restore caps the downside.
EARLY_STOP_PATIENCE = 3

#: The sdf-base-steering CE arms' effective batch: every
#: psr_ce_all / lora_r4_ce trial trains at effective batch 8 via
#: accumulation — the FITS' rule, so HPO winners transfer without an
#: lr x batch confound. Module constant = the documented monkeypatch seam.
CE_EFFECTIVE_BATCH = 8

#: make_sdfbase_configs.ACCUM_SENTINEL_FMT ("PENDING_MEMPROBE_EFF{eff}_ACCUM"),
#: re-stated as a pattern rather than imported (the generator is yaml-only
#: and imports nothing from this tree): the sentinel NAMES the effective
#: batch it stands in for, which is how probe-mem knows a key's batch
#: before its memprobe fold exists.
_ACCUM_SENTINEL_RE = re.compile(r"PENDING_MEMPROBE_EFF(\d+)_ACCUM")

#: The CE-native floor bar: held-out CE(active)/CE(noop) —
#: a degeneration/pathology guard, not a "did it help" gate; cells that
#: merely fail to help land at ~1.0 and publish. Config
#: train_kwargs.<ce_arm>.ce_floor_ratio_max overrides per lane.
CE_FLOOR_RATIO_MAX = 1.1

#: Discrete search grids (re-tune@32). Sobol unit coordinates snap to a
#: grid by equal subdivision (index = floor(u * len(grid))), so exact 1-D
#: Sobol stratification gives near-exact per-level balance.
TEMPERATURE_GRID = (1.0, 2.0, 4.0)
WEIGHT_DECAY_GRID = (0.0, 1e-3, 1e-2)   # 1e-2 = AdamW's silent incumbent
LORA_RANK_GRID = (1, 4, 16)

#: log-axis epsilon for weight_decay in the analyze fit: D = log(wd + eps).
#: One decade below the smallest nonzero grid level, so the three levels
#: sit near-evenly on the log axis and D at the lower grid edge maps back
#: to exactly wd = 0.0.
WD_LOG_EPS = 1e-4

#: Per-arm EXTRA Sobol axes beyond log-lr x log-clip (order matters: it is
#: the Sobol dimension order). psr_mse_all has NO temperature axis — its
#: loss matches teacher ACTIVATIONS (train_psr's temperature is a KL-path
#: knob, inert for mse); token_faithful/bidpo keep the legacy 2-D design.
ARM_AXES: dict[str, tuple[str, ...]] = {
    "psr_kl_all": ("temperature", "weight_decay"),
    "psr_mse_all": ("weight_decay",),
    "lora_r1_all": ("temperature", "weight_decay", "lora_rank"),
    "lora_r4": ("temperature", "weight_decay", "lora_rank"),
    "token_faithful": (),
    "bidpo": (),
    # sdf-base-steering CE arms: a 2-D lr x max_grad_norm Sobol
    # + the standing 2-level warmup factor. Smallest honest box — the
    # objective changed, so re-tuning lr/clip is the point; weight_decay
    # pinned 0.01, lora_rank pinned 4, temperature N/A (refused).
    "psr_ce_all": (),
    "lora_r4_ce": (),
}

#: Canonical arm names for cross-round pooling: a rename maps the legacy
#: label to its current name (same arm, same machinery).
ARM_CANON = {"lora_r1_all": "lora_r4"}

AXIS_GRIDS: dict[str, tuple] = {
    "temperature": TEMPERATURE_GRID,
    "weight_decay": WEIGHT_DECAY_GRID,
    "lora_rank": LORA_RANK_GRID,
}

#: Legacy per-arm seed boxes (kept for token_faithful/bidpo and as the
#: fallback when no per-(arm, model) box exists).
SEED_BOXES: dict[str, dict[str, tuple[float, float]]] = {
    "lora_r1_all": {"lr": (3e-5, 3e-3), "max_grad_norm": (0.03, 3.0)},
    "lora_r4": {"lr": (3e-5, 3e-3), "max_grad_norm": (0.03, 3.0)},
    "token_faithful": {"lr": (3e-5, 3e-3), "max_grad_norm": (0.03, 3.0)},
    "psr_kl_all": {"lr": (5e-5, 5e-3), "max_grad_norm": (0.03, 3.0)},
    "psr_mse_all": {"lr": (5e-5, 5e-3), "max_grad_norm": (0.03, 3.0)},
    "bidpo": {"lr": (5e-5, 5e-3), "max_grad_norm": (0.1, 10.0)},
}

#: Per-(arm, model) seed boxes for the re-tune@32 (the clip
#: confound): every old bs2/bs1 anchor was CLIP-INFLATED — measured on the
#: old cells (Modal-48 clip analysis, 2026-08-27): the pre-clip grad norm
#: exceeded the trial's clip on 83.5% of psr_kl steps (median preclip
#: 0.310), 100% of psr_mse steps (0.227 — the LR coordinate was
#: UNIDENTIFIED; clip was the effective-step knob), 93.1% of lora steps
#: (0.167; that fit ran to clip -> 0). So the realized step was ~lr*clip,
#: the nominal lr biased HIGH, and the joint Sobol at effective-32 must
#: REACH the non-binding region to de-confound them: lr widened DOWNWARD
#: (~1.5-2.5 decades below each old anchor, top kept at/above it) and clip
#: widened UPWARD to 10.0 (median preclips 0.17-0.31 mean clip >= ~1 is
#: essentially non-binding — the box holds a full decade of it).
#: Old anchors (positions=response cells): Llama ~/hpo-llama (psr_kl
#: winner lr 1.25e-3/clip 0.083; psr_mse 1.02e-2/0.027; lora argmin edge-
#: degenerate at clip -> 0, lr 7.5e-4); Qwen ~/hpo-r1/qwen_interim (psr_kl
#: 3.72e-4/0.080; psr_mse 5.79e-4/0.163; lora best-raw 9.3e-4/0.039 — its
#: quadratic was not PD); E4B e4b_hpo_winners_v4.json (psr_kl 1.06e-3/
#: 0.180; psr_mse 1.42e-3/0.671; lora 2.57e-3/0.065, near its old box top
#: 3e-3, so its top is raised to 1e-2).
CLIP_BOX_WIDE = (0.02, 10.0)
SEED_BOXES_BY_MODEL: dict[tuple[str, str], dict[str, tuple[float, float]]] = {
    ("psr_kl_all", "llama"): {"lr": (3e-5, 5e-3),
                              "max_grad_norm": CLIP_BOX_WIDE},
    ("psr_mse_all", "llama"): {"lr": (3e-5, 3e-2),
                               "max_grad_norm": CLIP_BOX_WIDE},
    ("lora_r1_all", "llama"): {"lr": (1e-5, 3e-3),
                               "max_grad_norm": CLIP_BOX_WIDE},
    ("lora_r4", "llama"): {"lr": (1e-5, 3e-3),
                           "max_grad_norm": CLIP_BOX_WIDE},
    ("psr_kl_all", "qwen38_27b"): {"lr": (1e-5, 2e-3),
                                   "max_grad_norm": CLIP_BOX_WIDE},
    ("psr_mse_all", "qwen38_27b"): {"lr": (2e-5, 5e-3),
                                    "max_grad_norm": CLIP_BOX_WIDE},
    ("lora_r1_all", "qwen38_27b"): {"lr": (1e-5, 3e-3),
                                    "max_grad_norm": CLIP_BOX_WIDE},
    ("lora_r4", "qwen38_27b"): {"lr": (1e-5, 3e-3),
                                "max_grad_norm": CLIP_BOX_WIDE},
    ("psr_kl_all", "gemma4e4b"): {"lr": (3e-5, 5e-3),
                                  "max_grad_norm": CLIP_BOX_WIDE},
    ("psr_mse_all", "gemma4e4b"): {"lr": (5e-5, 1e-2),
                                   "max_grad_norm": CLIP_BOX_WIDE},
    ("lora_r1_all", "gemma4e4b"): {"lr": (3e-5, 1e-2),
                                   "max_grad_norm": CLIP_BOX_WIDE},
    ("lora_r4", "gemma4e4b"): {"lr": (3e-5, 1e-2),
                               "max_grad_norm": CLIP_BOX_WIDE},
    # E2B safety lane (2026-08-27): a scoped 8-point lr x clip
    # sweep; wf fixed 0.15, wd pinned 0.01, T pinned 1.0.
    ("psr_kl_all", "gemma4e2b"): {"lr": (1e-4, 3e-3),
                                  "max_grad_norm": (0.05, 2.0)},
    # sdf-base-steering CE lanes: log-uniform boxes centered on
    # the instruct campaign's winners (gemma psr_kl 4.387e-4 / lora
    # 9.11e-5; qwen35_4b psr_kl 2.191e-3 / lora 2.193e-3), deliberately
    # WIDE on lr (the objective changed from soft-KL to document CE);
    # clip = the wide box so the non-binding region is reachable.
    ("psr_ce_all", "gemma4e4b_base"): {"lr": (1e-5, 2e-2),
                                       "max_grad_norm": CLIP_BOX_WIDE},
    ("lora_r4_ce", "gemma4e4b_base"): {"lr": (3e-6, 5e-3),
                                       "max_grad_norm": CLIP_BOX_WIDE},
    ("psr_ce_all", "qwen35_4b_base"): {"lr": (5e-5, 5e-2),
                                       "max_grad_norm": CLIP_BOX_WIDE},
    ("lora_r4_ce", "qwen35_4b_base"): {"lr": (5e-5, 5e-2),
                                       "max_grad_norm": CLIP_BOX_WIDE},
}

#: sdf-base-steering doc-CE lane (2026-09-02): the two CHAT checkpoints
#: joining it have no instruct-campaign winner of their own to center a box
#: on, so each INHERITS the box of the architecturally closest lane already
#: tuned — the sdfbase boxes are deliberately wide (2-3 decades of lr), so
#: the inheritance is a prior on the search RANGE, not a transferred
#: optimum:
#:   glm46vflash_doc (GLM-4.6V-Flash: dense, full attention on all 40
#:     layers) <- gemma4e4b_base, the dense-model precedent.
#:   nemotron3nano4b_doc (Nemotron-3-Nano-4B: NemotronH hybrid Mamba2/MLP/
#:     attention) <- qwen35_4b_base, the hybrid-stack precedent.
#: The child rows alias the parent's dict objects ON PURPOSE (not copies):
#: a box edit to the parent lane moves the child with it, and the routing
#: test pins the identity, so the numbers live in exactly one place.
SEED_BOX_PARENT: dict[str, str] = {"glm46vflash_doc": "gemma4e4b_base",
                                   "nemotron3nano4b_doc": "qwen35_4b_base"}
for _child, _parent in SEED_BOX_PARENT.items():
    for _arm in ("psr_ce_all", "lora_r4_ce"):
        SEED_BOXES_BY_MODEL[(_arm, _child)] = SEED_BOXES_BY_MODEL[(_arm, _parent)]
del _child, _parent, _arm

#: sdfbase-llama8b lane (2026-09-09): ONE continuous axis. The two-axis
#: lr x clip design is not used on this lane — the clip is NOT searched.
#: It is a per-arm CONSTANT derived from the gradient-norm reconnaissance
#: (clip_lo = ceil2sf(max over the STABLE recon runs of that
#: arm of p95(pre-clip grad norm, steps > 5)) — every trial then meets a
#: clip that binds on <~5% of steps, a spike guard by construction rather
#: than a normaliser), supplied at emit time (`emit-trials --clip-const`)
#: and stamped verbatim on every row. Only log(lr) is Sobol-swept;
#: warmup_frac stays the standing 2-level factor;
#: LLAMA8B_CE_TRIALS_PER_ARM = 8 (4 per level) fits the 5-term quadratic
#: (1, lr, lr*lr, warmup, warmup*lr) with 3 dof — analyze holds the
#: constant clip FIXED (fit.fixed_axes) instead of refusing it as
#: rank-deficient. These rows therefore carry NO max_grad_norm range: a
#: box without one IS the LR-only design's declaration, and emit-trials
#: refuses to draw it without the constant.
#:
#: LR box. The BOTTOM is lr_c / 10^1.5, where lr_c is this
#: model's SI-distillation winner scaled for the batch change 32 -> 64 by
#: sqrt(2) — PSR 3.19e-4 x 1.41 = 4.5e-4, LoRA 2.45e-4 x 1.41 = 3.5e-4.
#: The box was registered as (lr_c / 10^1.5, lr_c x 10^1.5), 3 decades,
#: with the TOP conditional on the gradient-norm reconnaissance certifying
#: lr_c x 10^1.5 STABLE at clip 1e6. It did not (effective batch 64, 150
#: recon steps, the stability classifier): PSR at 1.4e-2 (`recon_5`) was
#: heavy_tail — certified-interval p95 7,391 at 2,134x its own median, an
#: early-window explosion followed by a collapse, late loss above
#: pre-training — and LoRA at 1.1e-2 (`recon_6`) diverged (late loss 7.09
#: over early 2.11; p95 27.3 at 184x the median). The ladder's next rung,
#: one decade above lr_c, was certified STABLE for both arms: PSR 4.5e-3
#: (`recon_5b`, p95 0.21) and LoRA 3.5e-3 (`recon_6b`, p95 0.127). So each
#: TOP is lowered to its certified rung and the bottoms stay:
#: (lr_c / 10^1.5, lr_c x 10) — exactly 2.5 decades, the minimum width,
#: with the batch-scaled anchor one decade below the top. Sobol points in
#: the dropped decade would have clipped constantly,
#: failed the 25% activation ceiling and been excluded, leaving too few of
#: the eight rows for the five-term fit. Registered under the EXISTING
#: slug "llama": both Nous ids resolve to it first-match, the
#: SI-distillation lane's rows above are untouched, and ONE box serves
#: both fit keys — the search runs once on llama8b_base and llama8b_it_doc
#: reuses the winners verbatim.
LLAMA8B_CE_TRIALS_PER_ARM = 8
LLAMA8B_CE_LR_BOX: dict[str, tuple[float, float]] = {
    "psr_ce_all": (1.4e-5, 4.5e-3),
    "lora_r4_ce": (1.1e-5, 3.5e-3),
}
for _arm, _lr_box in LLAMA8B_CE_LR_BOX.items():
    SEED_BOXES_BY_MODEL[(_arm, "llama")] = {"lr": _lr_box}
del _arm, _lr_box

#: sdf-base-steering instruct-direct BATCH-64 lanes (2026-09-11): the two
#: `_it_doc` checkpoints (gemma-4-E4B-it, Qwen3.5-4B) re-fit at
#: effective batch 64 under the `_eb64` keys. The SAME LR-only design as the
#: Llama rows above (clip = the recon-derived per-arm constant via
#: `emit-trials --clip-const`, log(lr) the single Sobol axis, 8 per arm),
#: registered under the slugs the instruct repo ids resolve to — "gemma4e4b"
#: (the two_si instruct lane's slug; its rows above are other arms, so no
#: box moves) and "qwen35_4b" (a NEW needle: the bare "qwen3.5-4b" id had no
#: slug; it sits AFTER "qwen3.5-4b-base" so the base key keeps its slug, and
#: no other (arm, "qwen35_4b") row exists, so nothing else changes). Boxes
#: are opened well ABOVE the batch-8 anchors (5.09e-4 / 3.31e-5 / 1.58e-4 /
#: 1.26e-3) because the optimum moves up 3-8x for an 8x batch increase;
#: gemma-lora's box still reaches its batch-8 anchor at the bottom because
#: that value is a 23-39x outlier below its three siblings and "Gemma really
#: is low" must stay answerable. Tops are provisional until the gradient-norm
#: reconnaissance certifies them STABLE at clip 1e6 (the Llama rule: an
#: unstable top comes down to the certified rung).
#: Tops CERTIFIED by the gradient-norm reconnaissance: gemma4e4b_it_doc_eb64/psr_ce_all top 0.02 -> 0.002 (1.00 decades); qwen35_4b_it_doc_eb64/psr_ce_all top 0.01 -> 0.003162 (1.50 decades); qwen35_4b_it_doc_eb64/lora_r4_ce top 0.02 -> 0.006324 (1.10 decades). Bottoms unchanged.
IT_EB64_CE_LR_BOX: dict[str, dict[str, tuple[float, float]]] = {
    "gemma4e4b": {"psr_ce_all": (0.0002, 0.002), "lora_r4_ce": (3e-05, 0.01)},
    "qwen35_4b": {"psr_ce_all": (0.0001, 0.003162), "lora_r4_ce": (0.0005, 0.006324)},
}
for _slug, _boxes in IT_EB64_CE_LR_BOX.items():
    for _arm, _lr_box in _boxes.items():
        assert (_arm, _slug) not in SEED_BOXES_BY_MODEL, (_arm, _slug)
        SEED_BOXES_BY_MODEL[(_arm, _slug)] = {"lr": _lr_box}
del _slug, _boxes, _arm, _lr_box


#: The front-door test-model lane's CE boxes. The LoRA architecture search
#: sweeps u = log10(lr) on [-4.5, -2.0] (rank x module set as NAMED discrete
#: axes, build_trials_v2); the ReLU search sweeps [-4.5, -1.5] alone. Both
#: rows keep the wide clip RANGE like every other non-Llama CE row (the
#: routing test pins it); the recon-derived constant enters at emit time
#: (`--recon` / `--clip-const`, a zero-width side [c, c]) exactly as on the
#: LR-only lanes.
LORA_ARCH_LR_LOG10_BOX: tuple[float, float] = (-4.5, -2.0)
RELU_LR_LOG10_BOX: tuple[float, float] = (-4.5, -1.5)
#: EMPTY since 2026-10-08: no lane runs the FD 6 architecture search on its own boxes (the worked example's demo lane shares the
#: paper's Qwen3.5-4B-Base checkpoint, so its config resolves to qwen35_4b_base's boxes through model_slug; its LoRA architecture
#: is the paper's pick). A lane that declares one registers its arch-search boxes through this tuple.
ARCH_SEARCH_MODEL_SLUGS: tuple[str, ...] = ()


def _register_arch_search_boxes(slugs: tuple[str, ...]) -> None:
    """The FD 6 architecture-search boxes of every slug in ``slugs`` (lr: the arm's wide log10 box; clip: CLIP_BOX_WIDE), registered
    once at import; a slug that already has a seed box refuses (one box per (arm, slug))."""
    for slug in slugs:
        for arm, lg in (("psr_ce_all", RELU_LR_LOG10_BOX),
                        ("lora_r4_ce", LORA_ARCH_LR_LOG10_BOX)):
            assert (arm, slug) not in SEED_BOXES_BY_MODEL, (arm, slug)
            SEED_BOXES_BY_MODEL[(arm, slug)] = {
                "lr": (10.0 ** lg[0], 10.0 ** lg[1]),
                "max_grad_norm": CLIP_BOX_WIDE}


_register_arch_search_boxes(ARCH_SEARCH_MODEL_SLUGS)


#: The round reserved for CONFIRMATION trials: ONE trial at the fitted
#: argmin's exact coordinates, on the reference cell and schedule, through
#: the real trial path, so the promoted point is one that actually ran.
#: analyze pools search rounds by explicit --rounds and refuses a
#: confirmation summary, so a round-9 directory is never pooled into a fit.
CONFIRMATION_ROUND = 9


def is_confirmation_design(spec: dict) -> bool:
    """True for a design `emit-trials --confirm-from` wrote (its
    ``confirmation_of`` block names the fit it confirms)."""
    return bool(isinstance(spec, dict) and spec.get("confirmation_of"))


#: The refine round (three trials at the winning pair on +-0.5 decade around
#: u*) and the round reserved for the EXPLORATION REPLICATE — the fourth
#: trial on the FALLBACK path (non-PD refit, argmin outside the refined box,
#: or a best_raw_trial promotion), a fresh trial at the promoted measured
#: trial's exact coordinates. It is never a confirmation
#: (confirmation_trial_id / confirmation_within_noise stay NULL), never
#: pooled into any fit, and its CE difference from the original is the
#: observed replicate noise.
REFINE_ROUND = 2
EXPLORATION_ROUND = 3
EXPLORATION_ROLE = "exploration_replicate"


def is_replicate_design(spec: dict) -> bool:
    """True for a design `emit-trials --explore-from` wrote (its
    ``exploration_replicate_of`` block names the promoted trial it
    replicates). analyze refuses to pool it, like a confirmation."""
    return bool(isinstance(spec, dict) and spec.get("exploration_replicate_of"))


#: The CONFIRM-PICK trial: the user's pick is theirs by FD 6a and
#: has no default, so an OFF-WINNER pick — a (rank, set) with no measured
#: clip-activation evidence at its inferred learning rate — is never a dead
#: end: hpo_pick records the ruling and requests ONE trial at the picked
#: pair at that learning rate (``emit-trials --confirm-pick-from``, the
#: exploration replicate's machinery: coordinates copied verbatim, round
#: ``r8``, ``trial_role = "confirm_pick"``, never pooled into any fit, its
#: fit_trial row emitted once at its terminal outcome); the pick becomes
#: promotable only when THAT trial's measured rate is below the 25 % limit,
#: otherwise hpo_pick refuses by name with the measured rate.
CONFIRM_PICK_ROUND = 8
#: paper-companion-repo: ONE trial per (rank, set) pair the serving probe certifies AFTER the
#: round-1 design was drawn (a re-verdict widening the menu) — round 4, ``trial_role = "addendum"``, at the learning rate the
#: design's OWN v2 analyze predicts for that pair when refit over the widened menu (`pair_predictions[rank:set]`; the
#: interaction model only when its AICc warrants it), else the set's best round-1 trial lr, stated in provenance. Published as
#: an addendum to the rank x module-set table; never pooled into the round-1 fit the pick was made from.
ADDENDUM_ROUND = 4
ADDENDUM_ROLE = "addendum"
ADDENDUM_DESIGN = "addendum_pairs"
CONFIRM_PICK_ROLE = "confirm_pick"


def is_confirm_pick_design(spec: dict) -> bool:
    """True for a design `emit-trials --confirm-pick-from` wrote (its
    ``confirm_pick_of`` block names the request it answers)."""
    return bool(isinstance(spec, dict) and spec.get("confirm_pick_of"))


def lr_only_box(box: dict) -> bool:
    """True when ``box`` declares the LR-only design: no
    max_grad_norm range at all (a SEED_BOXES_BY_MODEL row of the Llama CE
    shape), or a zero-width one ``[c, c]`` (an emitted payload's box after
    --clip-const, which a --box-from recenter round inherits)."""
    rng = box.get("max_grad_norm")
    return rng is None or float(rng[0]) == float(rng[1])

#: Model repo id -> the seed-box slug (make_fit_configs' model slugs).
#: Order is FIRST-MATCH: the instruct "gemma-4-e4b-it" needle must sit
#: before the base "gemma-4-e4b" (a substring of it), so every existing
#: instruct config keeps its historical slug byte-for-byte and only the
#: bare base repo id resolves to the sdf-base-steering slug.
MODEL_SLUGS = (("llama", "llama"), ("qwen3.8", "qwen38_27b"),
               ("qwen3.5-4b-base", "qwen35_4b_base"),
               # the bare instruct id (Qwen/Qwen3.5-4B): AFTER the base
               # needle (a substring of it), for the `_eb64` CE rows above
               ("qwen3.5-4b", "qwen35_4b"),
               ("gemma-4-e4b-it", "gemma4e4b"),
               ("gemma-4-e4b", "gemma4e4b_base"),
               ("gemma-4-e2b", "gemma4e2b"),
               # Chat checkpoints on the doc-CE lane: unique
               # substrings of the repo ids, and the `_doc` slug is the ONLY
               # tuned lane for each (their two_si siblings have no boxes,
               # so nothing else could claim the needle).
               ("glm-4.6v-flash", "glm46vflash_doc"),
               ("nemotron-3-nano-4b", "nemotron3nano4b_doc"))


def model_slug(repo_id: str) -> str | None:
    low = str(repo_id).lower()
    for needle, slug in MODEL_SLUGS:
        if needle in low:
            return slug
    return None


#: The 9-term quadratic's feature names (logs; W = warmup-level indicator).
#: The legacy 2-axis design's exact term set and order — the expanded
#: designs generalize it (see quad_terms).
QUAD9_NAMES = ("1", "lr", "clip", "lr*lr", "clip*clip", "lr*clip",
               "warmup", "warmup*lr", "warmup*clip")

#: Refuse the fit beyond this normalized design condition number.
MAX_DESIGN_COND = 1e8

#: "On the edge" when the fitted argmin sits within this fraction of the
#: box's log-width from a boundary (or outside the box entirely). ONE
#: definition, owned by the selection guard the fold writers share.
EDGE_TOL = hpo_selection_guard.EDGE_TOL  # 0.02

#: Refine round: 8 trials (4 per warmup level) on a box zoomed to ~1/3 the
#: log-width around the fitted argmin.
REFINE_N = 8
REFINE_ZOOM = 1.0 / 3.0


# --------------------------------------------------------------------------
# Sobol (base-2, dims 1-5, Joe-Kuo direction numbers) — no scipy dependency.
# Ported from tune_lora_all.py; dims 4-5 (new-joe-kuo-6 rows d=4: s=3 a=1
# m=(1,3,1); d=5: s=3 a=2 m=(1,1,1)) joined for the re-tune@32's per-arm
# discrete axes (temperature / weight_decay / lora_rank).
# --------------------------------------------------------------------------

_SOBOL_POLY = {
    2: (1, 0, (1,)),
    3: (2, 1, (1, 3)),
    4: (3, 1, (1, 3, 1)),
    5: (3, 2, (1, 1, 1)),
}
_SOBOL_BITS = 30


def _direction_ints(dim: int) -> list[int]:
    if dim == 1:
        m = [1] * _SOBOL_BITS
    else:
        s, a, m_init = _SOBOL_POLY[dim]
        m = list(m_init)
        while len(m) < _SOBOL_BITS:
            k = len(m)  # 0-based index of the value being computed
            new = m[k - s] ^ (m[k - s] << s)
            for j in range(1, s):
                if (a >> (s - 1 - j)) & 1:
                    new ^= m[k - j] << j
            m.append(new)
    return [m[k] << (_SOBOL_BITS - 1 - k) for k in range(_SOBOL_BITS)]


def sobol_ints(m_log2: int, d: int = 2) -> list[tuple[int, ...]]:
    """First 2^m_log2 points of the d-dimensional Sobol sequence, as
    ``_SOBOL_BITS``-bit integer lattice points (gray-code order, point 0 at
    the origin). Coverage guarantees hold exactly at 2^m points, which is
    why trial counts are powers of two."""
    if d > 5:
        raise ValueError("direction numbers included for d <= 5 only")
    n = 1 << m_log2
    vs = [_direction_ints(dim + 1) for dim in range(d)]
    x = [0] * d
    out = [tuple(x)]
    for i in range(1, n):
        k = i - 1
        c = ((k + 1) & ~k).bit_length() - 1  # lowest zero bit of k
        for dim in range(d):
            x[dim] ^= vs[dim][c]
        out.append(tuple(x))
    return out


def design_shift(round_no: int, arm: str, position: str, level: float,
                 d: int = 2, salt: int = 0) -> tuple[int, ...]:
    """Deterministic per-(round, arm, position, warmup-level) digital-shift
    vector: d integers in [0, 2^_SOBOL_BITS). Distinct levels get distinct
    shifts, which is what makes the two per-level designs INDEPENDENT
    randomized-QMC designs rather than two halves of one sequence.
    ``salt`` re-draws a design deterministically (e.g. if a draw fails the
    emit-time balance assertions); salt 0 keeps every historical shift
    byte-identical."""
    out = []
    for dim in range(d):
        tail = f"|s{salt}" if salt else ""
        key = (f"twosweep-sobol|r{round_no}|{arm}|{position}"
               f"|w{level:.6g}|dim{dim}{tail}")
        h = hashlib.sha256(key.encode()).digest()
        out.append(int.from_bytes(h[:8], "big") % (1 << _SOBOL_BITS))
    return tuple(out)


def shifted_unit_points(m_log2: int, shift: tuple[int, ...],
                        d: int = 2) -> list[tuple[float, ...]]:
    """The 2^m base Sobol points, digitally shifted (XOR on the integer
    lattice) and mapped to [0, 1)^d. A digital shift maps (0,m,d)-nets to
    (0,m,d)-nets, so per-axis and dyadic-grid stratum coverage is exact."""
    return [
        tuple(((xi ^ s) / (1 << _SOBOL_BITS)) for xi, s in zip(pt, shift))
        for pt in sobol_ints(m_log2, d)
    ]


def assert_net_coverage(points: list[tuple[float, float]], m_log2: int,
                        tag: str) -> None:
    """Assert the (0,m,2)-net property: for EVERY dyadic grid 2^a x 2^b with
    a+b = m, each of the 2^m cells holds exactly one point. The a=m and b=m
    cases are the per-axis stratifications — every stratum of BOTH
    continuous axes is spanned; the mixed cases kill the parity-aliasing
    failure mode (a design can cover each axis marginally yet still
    correlate them)."""
    n = 1 << m_log2
    if len(points) != n:
        raise SystemExit(f"[emit-trials] {tag}: {len(points)} points, "
                         f"expected {n}")
    for a in range(m_log2 + 1):
        b = m_log2 - a
        cells = {(int(u * (1 << a)), int(v * (1 << b))) for u, v in points}
        if len(cells) != n:
            raise SystemExit(
                f"[emit-trials] {tag}: stratum coverage FAILED on the "
                f"2^{a} x 2^{b} dyadic grid ({len(cells)}/{n} occupied "
                "cells) — the design does not span every stratum; refusing "
                "to emit an aliased round")


def map_log(u: float, lo: float, hi: float) -> float:
    return math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo)))


def snap_to_grid(u: float, grid: tuple):
    """Unit coordinate -> grid level by equal subdivision. With exact 1-D
    Sobol stratification (each dyadic stratum holds exactly one point) the
    per-level counts are balanced to within one boundary stratum."""
    return grid[min(int(u * len(grid)), len(grid) - 1)]


def assert_axis_stratified(points, dim: int, m_log2: int, tag: str) -> None:
    """Every Sobol coordinate is 1-D (0,m,1)-stratified: at 2^m points each
    dyadic stratum of width 2^-m holds exactly one point. Guards the snap's
    balance guarantee (a broken direction-number table would fail here)."""
    n = 1 << m_log2
    strata = {int(p[dim] * n) for p in points}
    if len(strata) != n:
        raise SystemExit(
            f"[emit-trials] {tag}: axis {dim} is not 1-D stratified "
            f"({len(strata)}/{n} strata occupied) — broken Sobol dimension")


def assert_grid_balance_and_span(points, dim: int, grid: tuple,
                                 tag: str) -> None:
    """The snapped discrete axis must (a) occupy every grid level with a
    near-equal share (>= floor(n/len(grid)) - 1 points, boundary-stratum
    slack), and (b) NOT alias the continuous axes: each level's points must
    span the lr and clip unit ranges (>= min(2, level count) distinct
    quarters on axes 0 and 1). Kills the parity-aliasing bug class for the
    snapped axes the exact 2-D net assertion cannot see."""
    n = len(points)
    g = len(grid)
    by_level: dict[int, list] = {i: [] for i in range(g)}
    for p in points:
        by_level[min(int(p[dim] * g), g - 1)].append(p)
    if n < 2 * g:
        # A small design (the 8-trial refine round's 4/level, test/smoke
        # sizes) cannot guarantee per-level coverage — 1-D stratification
        # only bounds counts to within one boundary stratum, which at
        # n/g < 2 can empty a level. Refine rounds pool with round 1,
        # whose full 16-per-level design IS enforced; the span check
        # below still applies to whatever levels are present.
        by_level = {lv: pts for lv, pts in by_level.items() if pts}
        floor_share = 0
    else:
        floor_share = max(1, n // g - 1)
    for lv, pts in by_level.items():
        if len(pts) < floor_share:
            raise SystemExit(
                f"[emit-trials] {tag}: grid axis {dim} level {grid[lv]!r} "
                f"holds {len(pts)}/{n} points (< {floor_share}) — the snap "
                "is unbalanced; re-emit with a different --design-salt")
        if len(pts) < 3:
            # Two points sharing a quarter is honest chance, not aliasing;
            # the check binds where a clump is diagnostic (>= 3 points —
            # every production level holds 5-6 at 16/level).
            continue
        for cont in (0, 1):
            quarters = {int(p[cont] * 4) for p in pts}
            if len(quarters) < 2:
                raise SystemExit(
                    f"[emit-trials] {tag}: grid axis {dim} level "
                    f"{grid[lv]!r} is ALIASED with continuous axis {cont} "
                    f"(its {len(pts)} points sit in {len(quarters)} "
                    "quarter(s)) — re-emit with a different --design-salt")


def build_trials(*, arm: str, model: str, position: str, round_no: int,
                 n: int, box: dict[str, tuple[float, float]],
                 warmup_levels: tuple[float, ...] = WARMUP_LEVELS,
                 model_revision: str | None = None,
                 config_sha: str | None = None,
                 design_salt: int = 0,
                 pin_axes: dict[str, float] | None = None) -> dict:
    """The full trial payload for one round: two independent digitally-
    shifted per-warmup-level d-D Sobol designs — log(lr) x log(clip) plus
    the arm's discrete axes (ARM_AXES, snapped to their grids) —
    interleaved (index i -> level i%2) so any contiguous START/COUNT slice
    a pod takes spans both levels. Schema keeps the ported shape
    ({round, box, sobol_m, trials:[{index, lr, warmup_frac,
    max_grad_norm, ...extras}]}) plus top-level arm/model/position/
    design_shift/axes/grids and the canonical model identity (model = the
    config's repo id, model_revision, config_sha = the config FILE's sha)
    that `trial` validates against its --config before training (a Gemma
    trials JSON on a Qwen config must refuse, never mislabel a sweep).

    ``pin_axes`` holds an ARM_AXES axis at one grid value for a follow-up
    round (the T!=1-diverges-under-the-top-K-head finding): the axis
    leaves the Sobol dimensions — the remaining axes keep their ARM_AXES
    order, and the (lr, clip) projection keeps the exact net property, so
    the shrunken design is a valid design, not a filtered one — and the
    pinned value is stamped on EVERY emitted row (payload.pinned_axes
    records it), so run_trial's TrialHP carries it explicitly instead of
    the trainer falling back to a registry default."""
    levels = tuple(sorted(set(float(w) for w in warmup_levels)))
    if len(levels) not in (1, 2):
        raise SystemExit(
            f"[emit-trials] warmup is a TWO-level factor (or a single "
            f"FIXED level for a scoped sweep), got {levels}")
    if n % len(levels) != 0:
        raise SystemExit(f"[emit-trials] --n {n} does not split evenly "
                         f"across {len(levels)} warmup levels")
    per = n // len(levels)
    m_log2 = int(round(math.log2(per)))
    if (1 << m_log2) != per:
        raise SystemExit(
            f"[emit-trials] per-level count must be a power of 2, got {per}")
    pins: dict[str, object] = {}
    for ax, val in (pin_axes or {}).items():
        if ax in ("lr", "max_grad_norm"):
            raise SystemExit(
                f"[emit-trials] --pin-axis {ax}: lr and max_grad_norm are "
                "the core continuous axes — every round sweeps them; a "
                "constant there is a degenerate design, not a follow-up "
                "round")
        if ax not in ARM_AXES.get(arm, ()):
            raise SystemExit(
                f"[emit-trials] --pin-axis {ax}: {arm} does not sweep that "
                f"axis (its extra axes: "
                f"{list(ARM_AXES.get(arm, ())) or None})")
        grid = AXIS_GRIDS[ax]
        hit = [g for g in grid if float(g) == float(val)]
        if not hit:
            raise SystemExit(
                f"[emit-trials] --pin-axis {ax}={val:g} is not on the {ax} "
                f"grid {list(grid)} — a pinned follow-up round must stay "
                "on the audited grid")
        # The GRID element, not the parsed float — lora_rank stays an int.
        pins[ax] = hit[0]
    extra_axes = tuple(ax for ax in ARM_AXES.get(arm, ()) if ax not in pins)
    # sdfbase-llama8b: a zero-width clip box [c, c] is the LR-ONLY
    # design — log(lr) is the single Sobol axis and the clip is a constant
    # stamped on every row (payload.constant_axes). build_trials never
    # invents the constant: emit_trials puts it in the box from
    # --clip-const, and a box with NO clip side at all is refused here.
    if "max_grad_norm" not in box:
        raise SystemExit(
            "[emit-trials] the box carries no max_grad_norm range — an "
            "LR-only (D26) seed box needs the recon-derived clip constant: "
            "pass --clip-const <value>")
    clip_const: float | None = None
    c_lo, c_hi = (float(v) for v in box["max_grad_norm"])
    if not all(math.isfinite(v) and v > 0 for v in (c_lo, c_hi)):
        # a nan never equals itself, so check finiteness BEFORE deciding
        # whether the side is a constant — a nan/inf/non-positive clip
        # bound is garbage on either design
        raise SystemExit(f"[emit-trials] clip bounds {box['max_grad_norm']!r} "
                         "must be finite positive numbers (a constant clip "
                         "is a finite positive number, a range two of them)")
    if c_lo == c_hi:
        clip_const = c_lo
        if extra_axes:
            raise SystemExit(
                f"[emit-trials] the LR-only design holds the clip constant "
                f"and sweeps log(lr) alone; {arm} also sweeps the discrete "
                f"axes {list(extra_axes)} — that combination is not a "
                "design this harness defines (the CE arms have no extra "
                "axes)")
    n_cont = 1 if clip_const is not None else 2
    d = n_cont + len(extra_axes)
    designs: dict[float, list[tuple[float, ...]]] = {}
    shifts: dict[float, tuple[int, ...]] = {}
    for lv in levels:
        shifts[lv] = design_shift(round_no, arm, position, lv, d=d,
                                  salt=design_salt)
        pts = shifted_unit_points(m_log2, shifts[lv], d=d)
        # The lr x clip projection of a d-D Sobol design IS the 2-D design
        # (each coordinate depends only on its own direction numbers), so
        # the exact dyadic-net assertion still applies to dims (0, 1). The
        # LR-only design has one continuous axis: its guarantee is the
        # exact 1-D stratification below (nothing to alias against).
        if clip_const is None:
            assert_net_coverage([p[:2] for p in pts], m_log2,
                                tag=f"warmup={lv}")
        for dim in range(d):
            assert_axis_stratified(pts, dim, m_log2, tag=f"warmup={lv}")
        for k, ax in enumerate(extra_axes):
            assert_grid_balance_and_span(pts, n_cont + k, AXIS_GRIDS[ax],
                                         tag=f"warmup={lv}|{ax}")
        designs[lv] = pts
    if len(levels) == 2 and (
            shifts[levels[0]] == shifts[levels[1]]
            or set(designs[levels[0]]) == set(designs[levels[1]])):
        raise SystemExit("[emit-trials] the two warmup levels produced the "
                         "SAME design — per-level independence is broken")
    trials = []
    for i in range(n):
        lv = levels[i % len(levels)]
        u = designs[lv][i // len(levels)]
        row = {
            "index": i,
            "lr": map_log(u[0], *box["lr"]),
            "warmup_frac": lv,
            # the LR-only design stamps the constant on EVERY row, so
            # TrialHP and the fit fingerprint carry it explicitly
            "max_grad_norm": (clip_const if clip_const is not None
                              else map_log(u[1], *box["max_grad_norm"])),
        }
        # ARM_AXES order on the row, swept and pinned alike — the pinned
        # constant is stamped EXPLICITLY so TrialHP (and the fingerprint)
        # carry it, never a trainer registry default.
        k = 0
        for ax in ARM_AXES.get(arm, ()):
            if ax in pins:
                row[ax] = pins[ax]
            else:
                row[ax] = snap_to_grid(u[n_cont + k], AXIS_GRIDS[ax])
                k += 1
        trials.append(row)
    payload = {
        "round": round_no,
        "box": {k: [float(v[0]), float(v[1])] for k, v in box.items()},
        "warmup_levels": list(levels),
        "sobol_m": m_log2,
        "sobol_d": d,
        "axes": (["lr"] if clip_const is not None
                 else ["lr", "max_grad_norm"]) + list(extra_axes),
        "grids": {ax: list(AXIS_GRIDS[ax]) for ax in extra_axes},
        "pinned_axes": dict(pins),
        "design_salt": design_salt,
        "arm": arm,
        "model": model,
        "model_revision": model_revision,
        "config_sha": config_sha,
        "position": position,
        "design_shift": {f"{lv:.6g}": list(shifts[lv]) for lv in levels},
        "trials": trials,
    }
    if clip_const is not None:
        # Emitted ONLY for the LR-only design, so every lr x clip payload
        # stays byte-identical to what this harness always wrote.
        payload["constant_axes"] = {"max_grad_norm": clip_const}
    return payload


# --------------------------------------------------------------------------
# The LoRA architecture search design: u = log10(lr) x the ALLOWED
# (rank, module-set) pairs P.
# --------------------------------------------------------------------------

#: The design's name on the payload and the fit_search record.
V2_DESIGN = "sobol16+coverage_repair"
V2_TRIALS = 16
V2_SEED = 1
V2_EXTENSION_SEED = 2
REFINE_SEED = 3
REFINE_HALF_DECADES = 0.5
REFINE_TRIALS = 3

#: The module-set vocabulary the design, the table and the pick agree on —
#: MIRRORS sisteer.extraction.lora_modules.MODULE_SETS (that module imports
#: torch at import; the design path is torch-free). `_module_sets()` returns
#: the library's table when the library is ALREADY loaded (the GPU paths load
#: it through object_factory) and asserts the two agree; it never imports it.
#: tests/test_hpo_guard_discrete_axes.py pins the two tables equal.
_MODULE_SETS_FALLBACK: dict[str, tuple[str, ...]] = {
    "wo": ("o_proj",),
    "qkvo": ("k_proj", "o_proj", "q_proj", "v_proj"),
    "attn_mlp": ("down_proj", "gate_proj", "k_proj", "o_proj", "q_proj",
                 "up_proj", "v_proj"),
    "mlp": ("down_proj", "gate_proj", "up_proj"),
    "down": ("down_proj",),
}


#: lora_modules.INIT_SCHEMES, mirrored for the same reason (pinned equal by
#: the same test).
_INIT_SCHEMES_FALLBACK: tuple[str, ...] = ("zero", "random_lecun_1e-3")


def _lora_modules_lib():
    """``sisteer.extraction.lora_modules`` when some GPU-side path (or a test)
    has already imported it, else None — never imported from here (torch)."""
    return sys.modules.get("sisteer.extraction.lora_modules")


def _module_sets() -> dict[str, tuple[str, ...]]:
    lib = _lora_modules_lib()
    if lib is None:
        return dict(_MODULE_SETS_FALLBACK)
    if {k: tuple(v) for k, v in lib.MODULE_SETS.items()} != _MODULE_SETS_FALLBACK:
        raise SystemExit("[emit-trials] tune_arm's module-set table has "
                         "drifted from sisteer.extraction.lora_modules."
                         "MODULE_SETS — update _MODULE_SETS_FALLBACK")
    return {k: tuple(v) for k, v in lib.MODULE_SETS.items()}


def set_id_of(modules) -> str:
    """The set id of a sorted module list (``wo`` ... ``down``), or the
    ``+``-joined names for an ad-hoc set — lora_modules.set_id_for's rule,
    torch-free."""
    mods = tuple(sorted({str(m) for m in modules}))
    for sid, members in _module_sets().items():
        if members == mods:
            return sid
    return "+".join(mods)


def modules_of_set(set_id: str) -> list[str]:
    sets = _module_sets()
    if set_id in sets:
        return list(sets[set_id])
    parts = sorted({m for m in str(set_id).split("+") if m})
    if not parts:
        raise SystemExit(f"[emit-trials] empty module set {set_id!r}")
    return parts


def set_order_from_counts(counts: dict[str, int]) -> list[str]:
    """lora_modules.set_order_by_parameter_count's rule (ties by name),
    torch-free: the ORDINAL order of module sets by trainable-parameter
    count — the guard's narrowest ... widest menu."""
    return sorted(counts, key=lambda sid: (int(counts[sid]), sid))


def per_rank_counts(items) -> dict[str, float]:
    """``{set id: trainable_params / rank}`` (the minimum over the items
    naming that set) from ``(rank, set_id, trainable_params)`` triples; an
    item without a count contributes nothing. A set's count grows ~linearly
    in the rank, so the per-rank count orders the sets whatever ranks were
    admitted for each."""
    counts: dict[str, float] = {}
    for rank, sid, tp in items:
        if tp is None or rank is None or sid is None:
            continue
        per = float(tp) / float(rank)
        counts[sid] = min(counts.get(sid, per), per)
    return counts


def sets_in_width_order(sets_present, counts: dict[str, float], *,
                        who: str) -> list[str]:
    """The module sets present in the guard's ordinal order (narrowest ->
    widest), DERIVED from ``counts`` (per_rank_counts). The order comes
    from trainable-parameter counts, never from a hand-written list — a
    set with no count refuses by name (the serving probe records a count
    on every allowed pair; every v2 summary echoes the order)."""
    present = sorted(set(sets_present))
    missing = sorted(set(present) - set(counts))
    if missing:
        raise SystemExit(f"[{who}] no trainable_params recorded for module set(s) "
                         f"{missing} — the module-set order is derived from the "
                         "counts (plan §3.10), never hand-written; refusing to "
                         "order the menu")
    return set_order_from_counts({k: int(round(v)) for k, v in counts.items()
                                  if k in present})


def pair_key(rank, set_id) -> tuple[int, str]:
    return int(rank), str(set_id)


def load_allowed_pairs(path) -> dict:
    """The serving probe's ``allowed_pairs.json`` — the design space P,
    a FIRST-CLASS input:

        {"model_id": ..., "pairs": [{"rank": R, "set": "mlp",
           "lora_modules": [...], "per_expert_rank": {...} | null,
           "max_lora_rank_cap": C, "topology": "S1S2" | "single",
           "trainable_params": N}, ...],
         "excluded": [{"rank": R, "set": S, "reason": "..."}, ...],
         "probe_job_id": ..., "verdict_uri": ...}

    Returns the normalised document: every allowed pair with its sorted
    ``lora_modules`` (derived from the set id when absent) and its set id
    (derived from the modules when absent — the two must agree), the
    excluded list with reasons, and ``sha256`` of the file. Refuses an
    empty menu, a duplicate pair, a pair that is both allowed and excluded,
    a rank below 1, and an unknown module name."""
    path = Path(path)
    raw = path.read_bytes()
    doc = json.loads(raw.decode())
    pairs_in = doc.get("pairs")
    if not isinstance(pairs_in, list) or not pairs_in:
        raise SystemExit(f"[allowed-pairs] {path}: 'pairs' must be a non-empty "
                         "list — the serving probe admitted no (rank, set) "
                         "pair, so there is no design space")
    known = set().union(*_module_sets().values())
    pairs: list[dict] = []
    seen: set[tuple[int, str]] = set()
    for i, pr in enumerate(pairs_in):
        if not isinstance(pr, dict) or "rank" not in pr:
            raise SystemExit(f"[allowed-pairs] {path}: pairs[{i}] lacks 'rank'")
        rank = int(pr["rank"])
        if rank < 1:
            raise SystemExit(f"[allowed-pairs] {path}: pairs[{i}] rank {rank} < 1")
        mods = pr.get("lora_modules")
        sid = pr.get("set")
        if mods is None and sid is None:
            raise SystemExit(f"[allowed-pairs] {path}: pairs[{i}] names neither "
                             "'set' nor 'lora_modules'")
        if mods is None:
            mods = modules_of_set(str(sid))
        mods = sorted({str(m) for m in mods})
        unknown = sorted(set(mods) - known)
        if unknown:
            raise SystemExit(f"[allowed-pairs] {path}: pairs[{i}] names unknown "
                             f"module(s) {unknown} (vocabulary {sorted(known)})")
        derived = set_id_of(mods)
        if sid is None:
            sid = derived
        elif str(sid) != derived and modules_of_set(str(sid)) != mods:
            raise SystemExit(f"[allowed-pairs] {path}: pairs[{i}] set {sid!r} "
                             f"disagrees with its lora_modules {mods} "
                             f"(= {derived!r})")
        key = pair_key(rank, sid)
        if key in seen:
            raise SystemExit(f"[allowed-pairs] {path}: pair {key} listed twice")
        seen.add(key)
        per = pr.get("per_expert_rank")
        if per is not None and not isinstance(per, (dict, int)):
            raise SystemExit(f"[allowed-pairs] {path}: pairs[{i}] per_expert_rank "
                             f"must be a map or an int, got {per!r}")
        pairs.append({**pr, "rank": rank, "set": str(sid), "lora_modules": mods,
                      "per_expert_rank": per})
    excluded: list[dict] = []
    for i, ex in enumerate(doc.get("excluded") or []):
        if not isinstance(ex, dict) or "rank" not in ex or "set" not in ex:
            raise SystemExit(f"[allowed-pairs] {path}: excluded[{i}] needs rank "
                             "and set")
        key = pair_key(ex["rank"], ex["set"])
        if key in seen:
            raise SystemExit(f"[allowed-pairs] {path}: pair {key} is BOTH allowed "
                             "and excluded")
        excluded.append({"rank": key[0], "set": key[1],
                         "reason": str(ex.get("reason") or "unstated")})
    return {"model_id": doc.get("model_id"), "pairs": pairs,
            "excluded": excluded, "probe_job_id": doc.get("probe_job_id"),
            "verdict_uri": doc.get("verdict_uri"),
            "active_experts": doc.get("active_experts"),
            "result_generation": doc.get("result_generation"),
            "menu_override": doc.get("menu_override"),          # An operator override of the menu gate (ruling, pairs, utc) — carried verbatim
            "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def allowed_pair_keys(allowed: dict) -> list[tuple[int, str]]:
    return [pair_key(p["rank"], p["set"]) for p in allowed["pairs"]]


def refuse_pair_outside(allowed: dict, rank, set_id, *, who: str) -> dict:
    """The one refusal every consumer of P shares (design, analyze, table,
    pick): a (rank, set) outside the allowed pairs — excluded by the probe
    with its reason, or never probed — is never generated, selected,
    tabled as a candidate or offered."""
    key = pair_key(rank, set_id)
    for pr in allowed["pairs"]:
        if pair_key(pr["rank"], pr["set"]) == key:
            return pr
    reason = next((ex["reason"] for ex in allowed["excluded"]
                   if pair_key(ex["rank"], ex["set"]) == key), None)
    if reason is not None:
        raise SystemExit(f"[{who}] pair (rank {key[0]}, set {key[1]!r}) was "
                         f"EXCLUDED by the serving probe: {reason} — not a "
                         "candidate")
    raise SystemExit(f"[{who}] pair (rank {key[0]}, set {key[1]!r}) is not one "
                     f"of the {len(allowed['pairs'])} allowed pairs "
                     f"{allowed_pair_keys(allowed)} — refusing")


def set_order_for_pairs(allowed: dict) -> list[str]:
    """The module sets of P in the guard's ordinal order (narrowest ->
    widest), from the pairs' recorded ``trainable_params`` normalised by
    rank (a set's count grows ~linearly in the rank, so the per-rank count
    orders the sets whatever ranks the probe admitted for each); a pair
    without a count refuses by name."""
    return sets_in_width_order(
        {pr["set"] for pr in allowed["pairs"]},
        per_rank_counts((pr["rank"], pr["set"], pr.get("trainable_params"))
                        for pr in allowed["pairs"]),
        who="allowed-pairs")


def per_expert_rank_rule(allowed: dict) -> str:
    """The rule string the fit_search records: ``max(1,r//A)``
    with the model's active-expert count when the probe recorded it
    (``active_experts``), else the generic spelling — never inferred from a
    pair's map (``rank // routed_expert`` is not the inverse of
    ``max(1, rank // A)``); the per-pair ``per_expert_rank`` maps on
    ``allowed_pairs_detail`` are the data."""
    active = allowed.get("active_experts")
    return f"max(1,r//{int(active)})" if active else "max(1,r//active_experts)"


def _pair_centres(pairs: list[dict], ranks: list[int], sets: list[str]
                  ) -> list[tuple[float, float]]:
    """Each allowed pair's centre in the (rank, set) unit square: the
    rank axis in ascending rank order, the set axis in the guard's ordinal
    order — the geometry the Sobol coordinates are quantised onto."""
    out = []
    for pr in pairs:
        ri = ranks.index(int(pr["rank"]))
        si = sets.index(pr["set"])
        out.append(((ri + 0.5) / len(ranks), (si + 0.5) / len(sets)))
    return out


def _nearest_pair(u: tuple[float, float], centres) -> int:
    """The enumeration index of the allowed pair whose centre is nearest
    to ``u`` (squared distance; ties -> the lower index)."""
    best, best_d = 0, float("inf")
    for i, (cx, cy) in enumerate(centres):
        d = (u[0] - cx) ** 2 + (u[1] - cy) ** 2
        if d < best_d - 1e-15:
            best, best_d = i, d
    return best


def coverage_repair(assign: list[int], units: list[tuple[float, float]],
                    centres, n_pairs: int) -> list[dict]:
    """Relabel points until every allowed pair holds >= 1:
    while some pair is uncovered, take the MOST-populated pair (ties -> the
    lowest enumeration index) and move its point NEAREST to the uncovered
    pair's centre (ties -> the lowest point index) onto that pair. In place
    on ``assign``; returns the moves. Deterministic by construction: no
    randomness, total tie-breaks. With n >= |P| it terminates with every
    pair covered (an uncovered pair implies some pair holds >= 2)."""
    moves: list[dict] = []
    n = len(assign)
    if n < n_pairs:
        raise SystemExit(f"[emit-trials] {n} points cannot cover {n_pairs} "
                         "allowed pairs — extend the design first")
    while True:
        counts = [0] * n_pairs
        for a in assign:
            counts[a] += 1
        uncovered = [i for i, c in enumerate(counts) if c == 0]
        if not uncovered:
            return moves
        target = uncovered[0]
        donor = max(range(n_pairs), key=lambda i: (counts[i], -i))
        if counts[donor] < 2:
            raise SystemExit("[emit-trials] coverage repair stalled — a "
                             "pair is uncovered but no pair holds two "
                             "points (n < |P|?)")
        cx, cy = centres[target]
        members = [k for k in range(n) if assign[k] == donor]
        pick = min(members, key=lambda k: ((units[k][0] - cx) ** 2
                                           + (units[k][1] - cy) ** 2, k))
        moves.append({"index": pick, "from_pair": donor, "to_pair": target,
                      "distance": math.sqrt((units[pick][0] - cx) ** 2
                                            + (units[pick][1] - cy) ** 2)})
        assign[pick] = target


def build_trials_v2(*, arm: str, model: str, position: str, round_no: int,
                    lr_log10_box: tuple[float, float], allowed: dict,
                    warmup_frac: float, max_grad_norm: float,
                    init_scheme: str, n: int = V2_TRIALS, seed: int = V2_SEED,
                    model_revision: str | None = None,
                    config_sha: str | None = None) -> dict:
    """The LoRA architecture search round: ``n`` (16)
    points of a scrambled 3-D Sobol design (digital shift keyed on the
    seed) — u0 -> u = log10(lr) on ``lr_log10_box``; (u1, u2) -> the
    ALLOWED pair whose centre in the (rank, set) unit square is nearest
    (quantisation onto the enumeration of P, never onto the two axes
    separately, so a point can never land on a pair the probe excluded)
    — then coverage repair until every pair of P holds a point. Warm-up
    is ONE fixed level and the clip the recon-derived constant (both
    stamped on every row); ``init_scheme`` rides every row.

    With |P| <= n the ``n`` points always cover P. **32 only when 16 cannot
    cover** (the literal rule; v0.2's fit-quality trigger was dropped):
    when |P| > n a second scrambled Sobol-16 (seed + 1) is appended before
    the repair and ``design_extended_to_32 = {"reason": "coverage", ...}``
    is recorded; |P| > 2n refuses.

    Payload: build_trials' keys (``round, box, warmup_levels, sobol_m,
    sobol_d, axes, grids, pinned_axes, design_salt, arm, model,
    model_revision, config_sha, position, design_shift, trials,
    constant_axes``) plus ``design``, ``design_seed``, ``coverage_repair``,
    ``allowed_pairs`` ([[rank, set], ...]), ``allowed_pairs_detail``,
    ``excluded_pairs`` ([[rank, set, reason], ...]), ``per_expert_rank_rule``,
    ``init_scheme``, ``lr_log10_box``, ``set_order``,
    ``design_extended_to_32``. Rows: ``index, lr, warmup_frac,
    max_grad_norm, lora_rank, lora_modules (sorted list), lora_set_id,
    per_expert_rank, init_scheme``."""
    lo, hi = (float(v) for v in lr_log10_box)
    if not (math.isfinite(lo) and math.isfinite(hi) and hi > lo):
        raise SystemExit(f"[emit-trials] lr log10 box {lr_log10_box!r} must "
                         "be finite with hi > lo")
    if not (math.isfinite(max_grad_norm) and max_grad_norm > 0):
        raise SystemExit(f"[emit-trials] max_grad_norm {max_grad_norm!r} must "
                         "be a finite positive constant (the recon clip)")
    if not (0.0 <= float(warmup_frac) < 1.0):
        raise SystemExit(f"[emit-trials] warmup_frac {warmup_frac!r} must lie "
                         "in [0, 1)")
    wf = float(warmup_frac)
    pairs = allowed["pairs"]
    n_pairs = len(pairs)
    m_log2 = int(round(math.log2(n)))
    if (1 << m_log2) != n:
        raise SystemExit(f"[emit-trials] --n {n} must be a power of two")
    extended = None
    if n_pairs > n:
        if n_pairs > 2 * n:
            raise SystemExit(f"[emit-trials] {n_pairs} allowed pairs exceed "
                             f"2 x {n} — even the extended design cannot "
                             "cover every pair; reduce the menu")
        extended = {"reason": "coverage", "n_pairs": n_pairs, "n_base": n,
                    "n_total": 2 * n, "seeds": [seed, seed + 1]}
    ranks = sorted({int(pr["rank"]) for pr in pairs})
    sets = set_order_for_pairs(allowed)
    centres = _pair_centres(pairs, ranks, sets)
    pts: list[tuple[float, ...]] = []
    shifts: dict[str, list[int]] = {}
    for k, sd in enumerate([seed] + ([seed + 1] if extended else [])):
        sh = design_shift(round_no, arm, position, wf, d=3, salt=sd)
        shifts[f"seed{sd}"] = list(sh)
        block = shifted_unit_points(m_log2, sh, d=3)
        for dim in range(3):
            assert_axis_stratified(block, dim, m_log2, tag=f"v2 seed{sd}")
        pts.extend(block)
    units = [(pt[1], pt[2]) for pt in pts]
    assign = [_nearest_pair(u, centres) for u in units]
    moves = coverage_repair(assign, units, centres, n_pairs)
    rule = per_expert_rank_rule(allowed)
    trials = []
    for i, pt in enumerate(pts):
        pr = pairs[assign[i]]
        trials.append({
            "index": i,
            "lr": 10.0 ** (lo + pt[0] * (hi - lo)),
            "warmup_frac": wf,
            "max_grad_norm": float(max_grad_norm),
            "lora_rank": int(pr["rank"]),
            "lora_modules": list(pr["lora_modules"]),
            "lora_set_id": pr["set"],
            "per_expert_rank": pr.get("per_expert_rank"),
            "init_scheme": str(init_scheme),
        })
    covered = {(t["lora_rank"], t["lora_set_id"]) for t in trials}
    missing = [k for k in allowed_pair_keys(allowed) if k not in covered]
    if missing:
        raise SystemExit(f"[emit-trials] coverage repair left {missing} "
                         "uncovered — refusing to emit")
    payload = {
        "round": int(round_no),
        "box": {"lr": [10.0 ** lo, 10.0 ** hi],
                "max_grad_norm": [float(max_grad_norm), float(max_grad_norm)]},
        "warmup_levels": [wf],
        "sobol_m": m_log2,
        "sobol_d": 3,
        "axes": ["lr", "lora_rank", "lora_modules"],
        "grids": {"lora_rank": ranks, "lora_modules": sets},
        "pinned_axes": {},
        "design_salt": int(seed),
        "arm": arm,
        "model": model,
        "model_revision": model_revision,
        "config_sha": config_sha,
        "position": position,
        "design_shift": shifts,
        "trials": trials,
        "constant_axes": {"max_grad_norm": float(max_grad_norm)},
        "design": V2_DESIGN,
        "design_seed": int(seed),
        "coverage_repair": [{**mv,
                             "from_pair": list(pair_key(pairs[mv["from_pair"]]["rank"],
                                                        pairs[mv["from_pair"]]["set"])),
                             "to_pair": list(pair_key(pairs[mv["to_pair"]]["rank"],
                                                      pairs[mv["to_pair"]]["set"]))}
                            for mv in moves],
        "allowed_pairs": [list(k) for k in allowed_pair_keys(allowed)],
        "allowed_pairs_detail": [dict(pr) for pr in pairs],
        "allowed_pairs_source": {"path": allowed.get("path"),
                                 "sha256": allowed.get("sha256"),
                                 "model_id": allowed.get("model_id"),
                                 "probe_job_id": allowed.get("probe_job_id"),
                                 "verdict_uri": allowed.get("verdict_uri")},
        "excluded_pairs": [[ex["rank"], ex["set"], ex["reason"]]
                           for ex in allowed["excluded"]],
        "per_expert_rank_rule": rule,
        "init_scheme": str(init_scheme),
        "lr_log10_box": [lo, hi],
        "set_order": sets,
        "design_extended_to_32": extended,
    }
    return payload


def _pair_of(row: dict) -> tuple[int, str] | None:
    """(rank, set id) of a trial row / selection carrying the LoRA
    architecture axes, else None (a ReLU row)."""
    if row.get("lora_rank") is None:
        return None
    sid = row.get("lora_set_id") or (set_id_of(row["lora_modules"])
                                     if row.get("lora_modules") else "wo")
    return pair_key(row["lora_rank"], sid)


#: The architecture axes a row / winner carries beyond (lr, clip, warmup).
ARCH_AXES = ("lora_rank", "lora_modules", "lora_set_id", "per_expert_rank",
             "init_scheme")


def _art_base_payload(art: dict, *, round_no: int, arm: str, model: str,
                      model_revision: str | None, config_sha: str,
                      box: dict, trials: list[dict], si: str | None) -> dict:
    """The build_trials-shaped payload skeleton a follow-up design (refine /
    exploration replicate) shares: the artefact's fixed clip and warm-up,
    its position and search id."""
    fit = art.get("fit") or {}
    fixed = dict(art.get("fixed_axes") or fit.get("fixed_axes") or {})
    levels = [float(x) for x in (fit.get("levels") or art.get("warmup_levels")
                                 or WARMUP_LEVELS)]
    payload: dict = {
        "round": int(round_no), "box": box, "warmup_levels": sorted(set(levels)),
        "sobol_m": 0, "sobol_d": 0, "axes": ["lr"], "grids": {},
        "pinned_axes": {}, "design_salt": 0, "arm": arm, "model": model,
        "model_revision": model_revision, "config_sha": config_sha,
        "position": str(art.get("position") or "all"), "design_shift": {},
        "trials": trials,
    }
    if "max_grad_norm" in fixed:
        payload["constant_axes"] = {"max_grad_norm": fixed["max_grad_norm"]}
    if si:
        payload["si"] = si
    for k in ("search_id", "campaign", "init_scheme"):
        if art.get(k) is not None:
            payload[k] = art[k]
    return payload


def _art_si(art: dict, si: str | None, who: str) -> str | None:
    art_si = art.get("si")
    if si and art_si and str(si) != str(art_si):
        raise SystemExit(f"[emit-trials] {who}: --si {si!r} contradicts the "
                         f"artefact's data cell {art_si!r}")
    return str(si or art_si) if (si or art_si) else None


def _art_arm_model_check(art: dict, arm: str, model: str, who: str) -> None:
    art_arm = ARM_CANON.get(art.get("arm"), art.get("arm"))
    if art_arm is not None and art_arm != ARM_CANON.get(arm, arm):
        raise SystemExit(f"[emit-trials] {who}: the artefact is for arm "
                         f"{art.get('arm')!r}, not --arm {arm!r}")
    if art.get("model") and art["model"] != model:
        raise SystemExit(f"[emit-trials] {who}: the artefact's trials trained "
                         f"{art['model']!r} but --config names {model!r}")


def _refuse_pooled_round(art: dict, round_no: int, who: str) -> list[int]:
    """The artefact's pooled rounds; refuses a follow-up round the artefact
    already pooled (its trials would be fed back into the fit they extend)."""
    rounds = [int(r) for r in (art.get("rounds") or [])]
    if int(round_no) in rounds:
        raise SystemExit(f"[emit-trials] {who}: round {round_no} is already "
                         f"pooled in the artefact ({rounds})")
    return rounds


def build_refine_trials(art: dict, *, fit_path: str, fit_sha256: str,
                        model: str, model_revision: str | None,
                        config_sha: str, arm: str,
                        round_no: int = REFINE_ROUND, si: str | None = None,
                        n: int = REFINE_TRIALS,
                        half_decades: float = REFINE_HALF_DECADES) -> dict:
    """Recentre + 4, split 3 + 1: THREE refinement
    trials at the selected (rank, set) — the artefact's ``selection``
    (the fitted argmin's pair when the design fit was PD and interior,
    else the promoted measured trial's) — on a +-``half_decades`` box
    around u* = log10(selection.lr), a 1-D scrambled Sobol (seed 3; the
    first ``n`` of its 4 points, in sequence order), round ``r2``. The
    clip, warm-up and init scheme are the artefact's, verbatim. The same
    builder serves the ReLU search (no pair: the rows carry no
    architecture axes). Refuses a round the artefact already pooled."""
    who = "--refine-from"
    sel = art.get("selection")
    if not isinstance(sel, dict) or "lr" not in sel:
        raise SystemExit(f"[emit-trials] {who}: the artefact carries no "
                         "selection block (run `analyze --design v2` first)")
    _art_arm_model_check(art, arm, model, who)
    rounds = _refuse_pooled_round(art, round_no, who)
    si = _art_si(art, si, who)
    fit = art.get("fit") or {}
    fixed = dict(art.get("fixed_axes") or fit.get("fixed_axes") or {})
    if "max_grad_norm" not in fixed:
        raise SystemExit(f"[emit-trials] {who}: the artefact declares no fixed "
                         "clip — the refine round rides the recon constant")
    clip = float(fixed["max_grad_norm"])
    wf = float(sel.get("warmup_frac", (fit.get("levels") or [0.15])[0]))
    u_c = math.log10(float(sel["lr"]))
    lo, hi = u_c - float(half_decades), u_c + float(half_decades)
    position = str(art.get("position") or "all")
    shift = design_shift(round_no, arm, position, wf, d=1, salt=REFINE_SEED)
    pts = shifted_unit_points(2, shift, d=1)   # 4 points, 1-D stratified
    assert_axis_stratified(pts, 0, 2, tag="refine")
    arch = {k: sel[k] for k in ARCH_AXES if sel.get(k) is not None}
    if "lora_rank" in arch and "lora_set_id" not in arch:
        arch["lora_set_id"] = _pair_of(arch)[1]
    if art.get("init_scheme") is not None and "init_scheme" not in arch:
        arch["init_scheme"] = art["init_scheme"]
    trials = []
    for i, pt in enumerate(pts[:int(n)]):
        trials.append({"index": i, "lr": 10.0 ** (lo + pt[0] * (hi - lo)),
                       "warmup_frac": wf, "max_grad_norm": clip, **arch})
    box = {"lr": [10.0 ** lo, 10.0 ** hi], "max_grad_norm": [clip, clip]}
    payload = _art_base_payload(art, round_no=round_no, arm=arm, model=model,
                                model_revision=model_revision,
                                config_sha=config_sha, box=box, trials=trials,
                                si=si)
    payload.update({
        "sobol_m": 2, "sobol_d": 1, "design_salt": REFINE_SEED,
        "design_shift": {f"{wf:.6g}": list(shift)},
        "design": f"refine_sobol4_first{int(n)}",
        "constant_axes": {"max_grad_norm": clip},
        "lr_log10_box": [lo, hi],
        "refine_of": {
            "fit": str(fit_path), "fit_sha256": fit_sha256, "rounds": rounds,
            "y_key": art.get("y_key"),
            "selected_from": sel.get("selected_from"),
            "pair": ({k: arch.get(k) for k in ("lora_rank", "lora_modules",
                                                "lora_set_id", "per_expert_rank")}
                     if "lora_rank" in arch else None),
            "center_log10": u_c, "box_log10": [lo, hi],
            "half_decades": float(half_decades), "seed": REFINE_SEED,
            "n_points_drawn": len(pts), "n_used": int(n),
        },
    })
    if "init_scheme" in arch:
        payload["init_scheme"] = arch["init_scheme"]
    return payload


def build_exploration_replicate(art: dict, *, fit_path: str, fit_sha256: str,
                                model: str, model_revision: str | None,
                                config_sha: str, arm: str,
                                round_no: int = EXPLORATION_ROUND,
                                si: str | None = None) -> dict:
    """The FOURTH trial on the FALLBACK path: the refit was not PD, or
    its argmin left the refined box, or the best measured trial was
    promoted — build_confirmation_trial REFUSES all three (it confirms
    an interior fitted argmin only), so the fourth trial is a DISTINCT
    exploration replicate: a fresh trial at the promoted measured
    trial's EXACT coordinates (every axis copied verbatim), round
    ``r3``, ``trial_role = "exploration_replicate"`` on the payload (->
    summary -> fit_trial.extra), never labelled confirmation, never
    pooled (analyze refuses ``exploration_replicate_of``), its CE
    difference from the original reported as the observed replicate
    noise. Refuses an artefact whose selection IS an interior fitted
    argmin (that path takes --confirm-from)."""
    who = "--explore-from"
    sel = art.get("selection") or {}
    guard = art.get("selection_guard") or {}
    if sel.get("selected_from") == "fitted_argmin" and not sel.get("on_edge"):
        raise SystemExit(f"[emit-trials] {who}: the artefact's selection is an "
                         "INTERIOR fitted argmin — the fourth trial on that "
                         "path is the exact confirmation (--confirm-from), "
                         "not an exploration replicate")
    tid = guard.get("selected_trial")
    if guard.get("selected_from") != "best_raw_trial" or not tid:
        raise SystemExit(f"[emit-trials] {who}: the artefact's guard names no "
                         "promoted measured trial (selected_from="
                         f"{guard.get('selected_from')!r}) — nothing to "
                         "replicate")
    _art_arm_model_check(art, arm, model, who)
    rounds = _refuse_pooled_round(art, round_no, who)
    si = _art_si(art, si, who)
    row = next((r for r in (art.get("rows") or [])
                if hpo_selection_guard.trial_id(r) == tid), None)
    if row is None:
        raise SystemExit(f"[emit-trials] {who}: promoted trial {tid!r} is not "
                         "among the artefact's usable rows")
    y_key = art.get("y_key") or "best_heldout_loss"
    trial = {"index": 0, "lr": row["lr"], "warmup_frac": row["warmup_frac"],
             "max_grad_norm": row["max_grad_norm"]}
    for ax in (*TRIAL_EXTRA_AXES, "lora_set_id"):
        if ax in row:
            trial[ax] = row[ax]
    box = {k: [float(v[0]), float(v[1])] for k, v in (art.get("box") or {}).items()}
    if "lr" not in box or "max_grad_norm" not in box:
        raise SystemExit(f"[emit-trials] {who}: the artefact's box lacks lr / "
                         "max_grad_norm")
    payload = _art_base_payload(art, round_no=round_no, arm=arm, model=model,
                                model_revision=model_revision,
                                config_sha=config_sha, box=box, trials=[trial],
                                si=si)
    payload.update({
        "design": "exploration_replicate",
        "trial_role": EXPLORATION_ROLE,
        "exploration_replicate_of": {
            "fit": str(fit_path), "fit_sha256": fit_sha256, "rounds": rounds,
            "y_key": y_key, "trial": tid, "original_y": float(row[y_key]),
            "selected_from": guard.get("selected_from"),
            "fall_through": guard.get("fall_through"),
            "coordinates": {k: trial[k] for k in trial if k != "index"},
            "note": ("a fresh trial at the promoted measured trial's exact "
                     "coordinates; its CE difference from the original is the "
                     "observed replicate noise — NOT a confirmation "
                     "(confirmation_trial_id / confirmation_within_noise stay "
                     "NULL)"),
        },
    })
    return payload


def build_confirm_pick_trial(request: dict, art: dict, *, request_path: str,
                             request_sha256: str, fit_path: str, fit_sha256: str,
                             model: str, model_revision: str | None,
                             config_sha: str, arm: str,
                             round_no: int = CONFIRM_PICK_ROUND,
                             si: str | None = None) -> dict:
    """ONE trial at the REQUESTED (rank, set) at the learning rate hpo_pick
    would ship there (its ``hpo_pick_confirm_request.json``): the exploration
    replicate's machinery — every coordinate copied VERBATIM from the
    request, round ``r8`` (index = the request's ``index``, so a later pick's
    confirm trial never reuses a trial id), ``trial_role = "confirm_pick"``,
    never pooled (analyze refuses ``confirm_pick_of``). Its measured
    clip-activation rate is the promotion evidence hpo_pick re-reads at those
    exact coordinates. Refuses a request whose pair the design does not allow
    or whose search differs from the artefact's."""
    who = "--confirm-pick-from"
    pick = request.get("pick") or {}
    for k in ("lr", "warmup_frac", "max_grad_norm", "lora_rank", "lora_modules",
              "lora_set_id"):
        if pick.get(k) is None:
            raise SystemExit(f"[emit-trials] {who}: the request's pick names no "
                             f"{k!r} — not a confirm-pick request")
    _art_arm_model_check(art, arm, model, who)
    rounds = _refuse_pooled_round(art, round_no, who)
    si = _art_si(art, si, who)
    if request.get("search_id") and art.get("search_id") and \
            str(request["search_id"]) != str(art["search_id"]):
        raise SystemExit(f"[emit-trials] {who}: the request names search "
                         f"{request['search_id']!r} but the design fit is "
                         f"{art['search_id']!r}")
    key = pair_key(pick["lora_rank"], pick["lora_set_id"])
    allowed = [tuple(pair_key(a, b)) for a, b in (art.get("allowed_pairs") or [])]
    if allowed and key not in allowed:
        raise SystemExit(f"[emit-trials] {who}: the picked pair {key} is not one "
                         f"of the design's allowed pairs {allowed}")
    # The index is the request's RESERVED identity — it
    # decides the trial dir (run_trial and hpo_trial.sbatch both derive r8_tNN from this row's index), so a
    # request without one is refused, never defaulted to 0 (two picks on index 0 shared one directory)
    idx = request.get("index")
    if isinstance(idx, bool) or not isinstance(idx, int) or idx < 0:
        raise SystemExit(f"[emit-trials] {who}: the request reserves no integer index (got {idx!r}) — "
                         "hpo_pick.py writes one per request; refusing to default it")
    trial = {"index": idx, "lr": float(pick["lr"]),
             "warmup_frac": float(pick["warmup_frac"]),
             "max_grad_norm": float(pick["max_grad_norm"])}
    for ax in (*TRIAL_EXTRA_AXES, "lora_set_id"):
        if pick.get(ax) is not None:
            trial[ax] = pick[ax]
    box = {k: [float(v[0]), float(v[1])] for k, v in (art.get("box") or {}).items()}
    if "lr" not in box or "max_grad_norm" not in box:
        raise SystemExit(f"[emit-trials] {who}: the artefact's box lacks lr / "
                         "max_grad_norm")
    payload = _art_base_payload(art, round_no=round_no, arm=arm, model=model,
                                model_revision=model_revision,
                                config_sha=config_sha, box=box, trials=[trial],
                                si=si)
    for k in ("campaign", "search_opening", "allowed_pairs", "excluded_pairs",
              "set_order", "lr_log10_box", "recon", "per_expert_rank_rule"):
        if art.get(k) is not None and k not in payload:
            payload[k] = art[k]
    payload.update({
        "design": "confirm_pick",
        "trial_role": CONFIRM_PICK_ROLE,
        "confirm_pick_of": {
            "request": str(request_path), "request_sha256": request_sha256,
            "fit": str(fit_path), "fit_sha256": fit_sha256, "rounds": rounds,
            "pair": [key[0], key[1]], "lr": float(pick["lr"]),
            "lr_source": request.get("lr_source"),
            "ruling_id": request.get("ruling_id"),
            # the identity the request reserved: the per-request design path and job suffix
            "index": idx, "trial_id": request.get("trial_id") or f"r{round_no}_t{idx:02d}",
            "design": request.get("design"), "job_suffix": request.get("job_suffix"),
            "predecessor": request.get("predecessor"),
            "coordinates": {k: trial[k] for k in trial if k != "index"},
            "note": ("the user's pick at a pair with no measured clip evidence "
                     "at its learning rate: ONE trial there; its clip-activation "
                     "rate decides the promotion (FD 6: > 25 % is never "
                     "promoted) — never pooled, NOT a confirmation of a fit"),
        },
    })
    return payload


def build_confirmation_trial(art: dict, *, fit_path: str, fit_sha256: str,
                             model: str, model_revision: str | None,
                             config_sha: str, arm: str,
                             round_no: int = CONFIRMATION_ROUND,
                             si: str | None = None) -> dict:
    """The ONE-row confirmation design for an analyze artefact whose fitted
    argmin is interior: a trial at EXACTLY
    the winner's coordinates — lr / max_grad_norm / warmup_frac (+ any
    fixed extra axis the winner echoes) copied VERBATIM, no rounding — in
    build_trials' payload shape (round, box = the artefact's union box,
    warmup_levels, arm/model/model_revision/config_sha/position, axes,
    constant_axes for the LR-only design, sobol_m/sobol_d 0, empty
    design_shift), stamped with ``confirmation_of`` so run_trial's summary,
    the fold writer and analyze all see it as EVIDENCE FOR AN ALREADY
    SELECTED POINT and never as a new search row.

    Refuses: a rank-deficient / non-PD fit (no fitted minimum — the
    conservative fall-through applies, not a confirmation); an argmin on
    the box edge (recenter or fall-through); an artefact of another arm or
    model than the config's; a round already pooled in the artefact; a
    ``si`` contradicting the artefact's data cell."""
    fit = art.get("fit") or {}
    sel = art.get("selection")
    win = art.get("winner")
    rounds = [int(r) for r in (art.get("rounds") or [])]
    if fit.get("rank_deficient") or not fit.get("positive_definite"):
        raise SystemExit(
            "[emit-trials] --confirm-from: the artefact's quadratic has no "
            "trustworthy interior minimum (rank_deficient="
            f"{bool(fit.get('rank_deficient'))}, positive_definite="
            f"{fit.get('positive_definite')}) — nothing to confirm; the "
            "spec's conservative fall-through (best trial that RAN) applies")
    if not isinstance(sel, dict) or sel.get("selected_from") != "fitted_argmin":
        raise SystemExit("[emit-trials] --confirm-from: the artefact carries "
                         "no fitted-argmin selection block")
    if sel.get("on_edge") or win is None:
        raise SystemExit(
            "[emit-trials] --confirm-from: the fitted argmin sits on the box "
            f"edge (edge_pos {sel.get('edge_pos')}; winner is None) — a "
            "confirmation confirms an INTERIOR point; the edge policy is one "
            "recenter round (emit-trials --box-from) or the conservative "
            "fall-through, never a confirmation")
    art_arm = ARM_CANON.get(art.get("arm"), art.get("arm"))
    if art_arm is not None and art_arm != ARM_CANON.get(arm, arm):
        raise SystemExit(f"[emit-trials] --confirm-from: the artefact is for "
                         f"arm {art.get('arm')!r}, not --arm {arm!r}")
    if art.get("model") and art["model"] != model:
        raise SystemExit(
            f"[emit-trials] --confirm-from: the artefact's trials trained "
            f"{art['model']!r} but --config names {model!r} — a confirmation "
            "must run the model the search ran")
    if int(round_no) in rounds:
        raise SystemExit(
            f"[emit-trials] --confirm-from: round {round_no} is one of the "
            f"artefact's pooled search rounds {rounds} — a confirmation "
            f"needs its own round (default {CONFIRMATION_ROUND}) so analyze "
            "never pools it")
    art_si = art.get("si")
    if si and art_si and str(si) != str(art_si):
        raise SystemExit(
            f"[emit-trials] --confirm-from: --si {si!r} contradicts the "
            f"artefact's data cell {art_si!r} — the confirmation runs on the "
            "reference cell the search ran on")
    si = str(si or art_si) if (si or art_si) else None
    for k in ("lr", "max_grad_norm", "warmup_frac"):
        if k not in win:
            raise SystemExit(f"[emit-trials] --confirm-from: winner lacks {k}")
    row: dict = {"index": 0, "lr": win["lr"], "warmup_frac": win["warmup_frac"],
                 "max_grad_norm": win["max_grad_norm"]}
    # the echoed extra axes ride verbatim: the re-tune@32 axes as before,
    # plus the LoRA architecture axes of a v2 winner (a confirmation at the
    # winning (rank, set) with its per-expert rank and init scheme)
    for ax in ("temperature", "weight_decay", *ARCH_AXES):
        if ax in win and win[ax] is not None:
            row[ax] = win[ax]
    box = {k: [float(v[0]), float(v[1])] for k, v in (art.get("box") or {}).items()}
    if "lr" not in box or "max_grad_norm" not in box:
        raise SystemExit("[emit-trials] --confirm-from: the artefact's box "
                         "lacks lr / max_grad_norm")
    levels = [float(x) for x in (fit.get("levels") or WARMUP_LEVELS)]
    fixed = dict(art.get("fixed_axes") or fit.get("fixed_axes") or {})
    cont = list(fit.get("cont_axes") or ["lr", "max_grad_norm"])
    axes = cont + (["lora_rank"] if fit.get("rank_levels") else [])
    payload: dict = {
        "round": int(round_no),
        "box": box,
        "warmup_levels": sorted(levels),
        "sobol_m": 0,
        "sobol_d": 0,
        "axes": axes,
        "grids": {},
        "pinned_axes": {k: v for k, v in fixed.items()
                        if k not in ("max_grad_norm", "lr")},
        "design_salt": 0,
        "arm": arm,
        "model": model,
        "model_revision": model_revision,
        "config_sha": config_sha,
        "position": str(art.get("position") or "all"),
        "design_shift": {},
        "trials": [row],
    }
    if "max_grad_norm" in fixed:
        payload["constant_axes"] = {"max_grad_norm": fixed["max_grad_norm"]}
    if si:
        payload["si"] = si
    for k in ("search_id", "init_scheme"):
        if art.get(k) is not None:
            payload[k] = art[k]
    guard = art.get("selection_guard") or {}
    payload["confirmation_of"] = {
        "fit": str(fit_path),
        "fit_sha256": fit_sha256,
        "rounds": rounds,
        "y_key": art.get("y_key"),
        "selected_from": "fitted_argmin",
        "argmin": {k: win[k] for k in ("lr", "max_grad_norm", "warmup_frac")},
        "fitted_min_log_y": sel.get("fitted_min_log_y"),
        "sigma_log_y": fit.get("sigma_log_y"),
        "hessian_eigs": fit.get("hessian_eigs"),
        "edge_pos": sel.get("edge_pos"),
        "n_rows": len(art.get("rows") or []),
        "guard_alarmed": guard.get("alarmed"),
        "guard_alarms": [hpo_selection_guard.alarm_code(a)
                         for a in (guard.get("alarms") or [])],
        "clip_activation": win.get("clip_activation"),
    }
    return payload



def config_file_sha256_full(path: str) -> str:
    """The FULL 64-hex sha256 of the config file bytes — what the registry's fit_search.config_sha256 column records
    (schema SHA256_OR_UNVERIFIED). The design keeps the 16-hex prefix (:func:`_config_file_sha`) as its identity key;
    the two agree by construction (same bytes)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def close_config_sha256(opening: dict, config_path: str | None) -> str | None:
    """The config_sha256 a CLOSE records: with ``--config F`` the file's full sha256, REQUIRED to agree with the
    opening's recorded value on the 16-hex prefix (the opening may carry only the design's 16-hex, the pre-fix spelling;
    a full value must match exactly) — a different file refuses by name, never a silently re-keyed search. Without
    ``--config`` the opening's value is passed through unchanged (the writer records a non-64-hex as 'unverified'); an
    opening with no value stays without one (the column is nullable: a manual search has no config file)."""
    have = str(opening.get("config_sha256") or "")
    if not config_path:
        return have or None
    full = config_file_sha256_full(config_path)
    if have and have != "unverified" and not full.startswith(have.lower()[:16] if len(have) == 16 else have.lower()):
        raise SystemExit(f"[registry] --config {config_path} hashes to {full[:16]}…, but the search was opened on config "
                         f"{have[:16]}… — not the same config bytes; refusing to close under another config")
    return full

def _config_file_sha(path: str) -> str:
    """16-hex sha256 prefix of the config FILE BYTES — the driver's
    ``_config_sha`` convention (byte identity of the file the trial
    parsed, so a launch tarball shipping a pre-fix config refuses)."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def emit_trials(args) -> None:
    if args.arm == "token_faithful" and args.position != "all":
        raise SystemExit(
            "[emit-trials] token_faithful cannot honor a decode-only "
            "(response) fit — its soft PROMPT token has no generated-"
            "position notion (objects.py FIT_MASK_REFUSAL); this tuning "
            "cell is a permanent hole, not a round to emit")
    # The canonical model identity comes from the per-model config; it also
    # selects the per-(arm, model) seed box (loaded early — the box needs
    # the model before the payload does).
    cfg = _load_cfg(args.config)
    confirm_from = getattr(args, "confirm_from", None)
    # --round: 1 for a seed design, the artefact's next round for
    # --box-from (the operator passes it), CONFIRMATION_ROUND for a
    # confirmation unless given explicitly.
    refine_from = getattr(args, "refine_from", None)
    explore_from = getattr(args, "explore_from", None)
    confirm_pick_from = getattr(args, "confirm_pick_from", None)
    addendum_from = getattr(args, "addendum_from", None)
    modes = [f for f, v in (("--confirm-from", confirm_from),
                            ("--refine-from", refine_from),
                            ("--explore-from", explore_from),
                            ("--confirm-pick-from", confirm_pick_from),
                            ("--addendum-from", addendum_from)) if v]
    if len(modes) > 1:
        raise SystemExit(f"[emit-trials] {modes} are mutually exclusive — one "
                         "follow-up design per emit")
    default_round = (CONFIRMATION_ROUND if confirm_from
                     else REFINE_ROUND if refine_from
                     else EXPLORATION_ROUND if explore_from
                     else CONFIRM_PICK_ROUND if confirm_pick_from
                     else ADDENDUM_ROUND if addendum_from else 1)
    round_no = int(args.round) if args.round is not None else default_round
    if addendum_from:
        return _emit_addendum(args, cfg, addendum_from, round_no)
    if confirm_from:
        return _emit_confirmation(args, cfg, confirm_from, round_no)
    if confirm_pick_from:
        return _emit_confirm_pick(args, cfg, confirm_pick_from, round_no)
    if refine_from or explore_from:
        return _emit_followup(args, cfg, refine_from or explore_from, round_no,
                              refine=bool(refine_from))
    design = str(getattr(args, "design", "legacy") or "legacy")
    if design == "sobol16_coverage":
        return _emit_v2(args, cfg, round_no)
    if design != "legacy":
        raise SystemExit(f"[emit-trials] unknown --design {design!r} "
                         "(legacy | sobol16_coverage)")
    if getattr(args, "allowed_pairs", None) is not None:
        raise SystemExit("[emit-trials] --allowed-pairs belongs to --design sobol16_coverage")
    n = args.n
    if args.box_from:
        prev = json.loads(Path(args.box_from).read_text())
        nxt = prev.get("next")
        if not nxt:
            raise SystemExit(f"[emit-trials] {args.box_from} carries no "
                             "'next' recommendation")
        box = {k: (float(v[0]), float(v[1])) for k, v in nxt["box"].items()}
        if n is None:
            n = int(nxt["n"])
        print(f"[emit-trials] box from {args.box_from}: "
              f"action={nxt['action']} n={n}")
    else:
        slug = model_slug(cfg["model"])
        per_model = SEED_BOXES_BY_MODEL.get((args.arm, slug or ""))
        if per_model is not None:
            box = {k: tuple(v) for k, v in per_model.items()}
            print(f"[emit-trials] seed box for ({args.arm}, {slug}) — the "
                  "re-tune@32 clip-deconfounded box (D144)")
        else:
            if args.arm in {a for a, _m in SEED_BOXES_BY_MODEL}:
                raise SystemExit(
                    f"[emit-trials] {args.arm} has per-model seed boxes but "
                    f"config model {cfg['model']!r} maps to no known slug "
                    f"({[s for _, s in MODEL_SLUGS]}) — refusing to fall "
                    "back to a legacy box for a re-tune arm")
            box = {k: tuple(v) for k, v in SEED_BOXES[args.arm].items()}
    # Explicit bounds override whichever side is given.
    if args.lr_min is not None or args.lr_max is not None:
        box["lr"] = (args.lr_min if args.lr_min is not None else box["lr"][0],
                     args.lr_max if args.lr_max is not None else box["lr"][1])
    # --lr-log10-min / --lr-log10-max CLAMP the lr box (the front door's rule: the reconnaissance's certified top lowers the
    # box top so an unstable rung never enters the search; the bottom stays) — never widening, refusing an inverted box, and saying
    # the width that results (the spec's minimum is 2.5 decades: a narrower box is a NOTE for the operator, not a silent fact).
    lo, hi = float(box["lr"][0]), float(box["lr"][1])
    g_lo, g_hi = getattr(args, "lr_log10_min", None), getattr(args, "lr_log10_max", None)
    if g_lo is not None or g_hi is not None:
        new_lo = max(lo, 10.0 ** float(g_lo)) if g_lo is not None else lo
        new_hi = min(hi, 10.0 ** float(g_hi)) if g_hi is not None else hi
        if not new_hi > new_lo:
            raise SystemExit(f"[emit-trials] the lr box clamp (--lr-log10-min {g_lo}, --lr-log10-max {g_hi}) leaves no box: "
                             f"[{new_lo:.3e}, {new_hi:.3e}] inside the seed box [{lo:.3e}, {hi:.3e}] — the certified top is below the box bottom")
        width = math.log10(new_hi) - math.log10(new_lo)
        print(f"[emit-trials] lr box clamped to [{new_lo:.3e}, {new_hi:.3e}] (log10 [{math.log10(new_lo):.3f}, {math.log10(new_hi):.3f}], "
              f"{width:.2f} decades; seed box was [{lo:.3e}, {hi:.3e}])"
              + ("" if width >= 2.5 else f" — NOTE: below the spec's 2.5-decade minimum width; the bottom stays (D81) — the coordinator's call"),
              flush=True)
        box["lr"] = (new_lo, new_hi)
    # The clip side: a CONSTANT (--clip-const, the LR-only design), a
    # range override on a box that has one, or — for a seed box that
    # declares NO clip range — either the constant or BOTH bounds (the
    # operator deliberately re-opening the axis); never a silent default.
    clip_const, recon_rec = _clip_const_from_args(args)
    if recon_rec is not None:
        print(f"[emit-trials] clip constant {clip_const!r} from the "
              f"reconnaissance {args.recon} (p{recon_rec['percentile']:g} "
              "of the pooled pre-clip grad norm)")
    if clip_const is not None:
        if args.clip_min is not None or args.clip_max is not None:
            raise SystemExit("[emit-trials] --clip-const contradicts "
                             "--clip-min/--clip-max: the clip is either a "
                             "held constant or a searched range")
        clip_const = float(clip_const)
        if not (math.isfinite(clip_const) and clip_const > 0):
            raise SystemExit(f"[emit-trials] --clip-const {clip_const!r} "
                             "must be a finite positive number")
        box["max_grad_norm"] = (clip_const, clip_const)
    elif "max_grad_norm" in box:
        if args.clip_min is not None or args.clip_max is not None:
            box["max_grad_norm"] = (
                args.clip_min if args.clip_min is not None
                else box["max_grad_norm"][0],
                args.clip_max if args.clip_max is not None
                else box["max_grad_norm"][1])
    elif args.clip_min is not None and args.clip_max is not None:
        box["max_grad_norm"] = (float(args.clip_min), float(args.clip_max))
        print(f"[emit-trials] NOTE: the ({args.arm}, {model_slug(cfg['model'])}) "
              "seed box is the LR-only design (D26); --clip-min/--clip-max "
              "re-open the clip as a searched axis — a deliberate design "
              "change, recorded on the payload's axes")
    else:
        raise SystemExit(
            f"[emit-trials] the ({args.arm}, {model_slug(cfg['model'])}) seed "
            "box is the LR-only design (D26): the clip is the recon-derived "
            "per-arm CONSTANT, not a search axis — pass --clip-const <value> "
            "(plan §4.2: ceil2sf of the max p95 pre-clip grad norm over the "
            "STABLE recon runs of this arm), or BOTH --clip-min and "
            "--clip-max to deliberately re-open the axis")
    if n is None:
        # The LR-only design is 8 per arm; re-tune arms
        # default to the 32-trial joint design; the legacy 2-axis arms keep
        # their original 16.
        n = (LLAMA8B_CE_TRIALS_PER_ARM if lr_only_box(box)
             else 32 if ARM_AXES.get(args.arm) else 16)
    levels = tuple(float(x) for x in args.warmup_levels.split(","))
    # --pin-axis name=value (repeatable): hold an arm axis at one grid
    # value for a follow-up round (build_trials validates axis + grid
    # membership; this parses the CLI form only).
    pin_axes: dict[str, float] = {}
    for spec_str in (getattr(args, "pin_axis", None) or []):
        name, sep, val = str(spec_str).partition("=")
        name = name.strip()
        if not sep or not name or not val.strip():
            raise SystemExit(f"[emit-trials] --pin-axis {spec_str!r}: "
                             "expected AXIS=VALUE (e.g. temperature=1.0)")
        try:
            fval = float(val)
        except ValueError:
            raise SystemExit(f"[emit-trials] --pin-axis {spec_str!r}: "
                             f"{val.strip()!r} is not a number")
        if name in pin_axes:
            raise SystemExit(f"[emit-trials] --pin-axis {name} given twice")
        pin_axes[name] = fval
    # The trials file names the model+revision+config it was designed for,
    # and `trial` refuses any other --config (a Gemma trials JSON on a Qwen
    # config mislabels a whole sweep otherwise).
    payload = build_trials(arm=args.arm, model=cfg["model"],
                           model_revision=cfg.get("model_revision"),
                           config_sha=_config_file_sha(args.config),
                           position=args.position, round_no=round_no,
                           n=n, box=box, warmup_levels=levels,
                           design_salt=int(args.design_salt),
                           pin_axes=pin_axes)
    # --si pins the design to ONE data cell: run_trial refuses a
    # contradicting --si and the Modal spawner's _design_si reads it (the
    # gate_reg grid's precedent, emit_gatereg_trials.py) — the
    # sdfbase-llama8b search rides constv2-mix-10m, not the standing HPO_SI.
    si = getattr(args, "si", None)
    if si:
        payload["si"] = str(si)
    init_scheme = getattr(args, "init_scheme", None)
    if init_scheme:
        # FD 7: the run override's init scheme rides EVERY row explicitly
        # (TrialHP.init_scheme -> object_factory), and the payload names it;
        # absent flag -> absent key -> byte-identical legacy payloads.
        _check_init_scheme(init_scheme)
        payload["init_scheme"] = str(init_scheme)
        for t in payload["trials"]:
            t["init_scheme"] = str(init_scheme)
    if recon_rec is not None:
        payload["recon"] = {k: recon_rec.get(k) for k in _RECON_PAYLOAD_KEYS}
    _open_search_for(args, cfg, payload, recon_rec)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    pinned = payload.get("pinned_axes") or {}
    pin_note = f", pinned={pinned}" if pinned else ""
    const = payload.get("constant_axes") or {}
    const_note = f", constant={const}" if const else ""
    si_note = f", si={payload['si']}" if payload.get("si") else ""
    print(f"[emit-trials] wrote {len(payload['trials'])} trials "
          f"({args.arm}, {payload['model']}@{payload['model_revision']}, "
          f"positions={args.position}, round {round_no}, "
          f"axes={payload['axes']}{pin_note}{const_note}{si_note}) -> "
          f"{args.out}")
    extra_axes = payload["axes"][2:]
    for t in payload["trials"]:
        extras = "".join(f" {ax}={t[ax]:g}" for ax in extra_axes)
        extras += "".join(f" {ax}={t[ax]:g}(pinned)" for ax in pinned)
        print(f"  t{t['index']:02d}: lr={t['lr']:.3e} "
              f"warmup_frac={t['warmup_frac']:.3f} "
              f"clip={t['max_grad_norm']:.3f}{extras}")


#: emit-trials' design flags by spelling and Namespace attribute — the
#: legacy lr [x clip] box flags and the v2 (architecture search) inputs. A
#: follow-up design (--confirm-from / --refine-from / --explore-from) takes
#: its point, box and levels from the artefact and refuses them.
_LEGACY_DESIGN_FLAGS = (("--box-from", "box_from"), ("--lr-min", "lr_min"),
                        ("--lr-max", "lr_max"), ("--clip-min", "clip_min"),
                        ("--clip-max", "clip_max"), ("--clip-const", "clip_const"))
_V2_DESIGN_FLAGS = (("--allowed-pairs", "allowed_pairs"), ("--recon", "recon"),
                    ("--lr-log10-min", "lr_log10_min"),
                    ("--lr-log10-max", "lr_log10_max"))


def _design_flag_clashes(args, flags) -> list[str]:
    """The flags of ``flags`` ((spelling, attribute) pairs) set on ``args``
    (value not None), by spelling, plus ``--pin-axis`` when any axis is
    pinned — what a follow-up design refuses."""
    clashes = [flag for flag, attr in flags
               if getattr(args, attr, None) is not None]
    if getattr(args, "pin_axis", None):
        clashes.append("--pin-axis")
    return clashes


def _emit_confirmation(args, cfg: dict, confirm_from: str,
                       round_no: int) -> None:
    """`emit-trials --confirm-from <rN_fit.json>`: the one-row confirmation
    design at the fitted argmin (build_confirmation_trial). The other
    design flags are refused — the point, the box and the levels come from
    the artefact, the model identity from --config, and nothing else may
    move it."""
    clashes = _design_flag_clashes(args, (*_LEGACY_DESIGN_FLAGS, ("--n", "n")))
    if int(getattr(args, "design_salt", 0) or 0):
        clashes.append("--design-salt")
    if clashes:
        raise SystemExit(
            f"[emit-trials] --confirm-from takes no design flags ({clashes}): "
            "the confirmation's point, box and warmup levels are the "
            "artefact's, verbatim")
    fit_path = Path(confirm_from)
    fit_bytes = fit_path.read_bytes()
    art = json.loads(fit_bytes.decode())
    payload = build_confirmation_trial(
        art, fit_path=str(fit_path),
        fit_sha256=hashlib.sha256(fit_bytes).hexdigest(),
        model=cfg["model"], model_revision=cfg.get("model_revision"),
        config_sha=_config_file_sha(args.config), arm=args.arm,
        round_no=round_no, si=getattr(args, "si", None))
    if args.position and payload["position"] != args.position:
        raise SystemExit(
            f"[emit-trials] --confirm-from: --position {args.position!r} "
            f"contradicts the artefact's {payload['position']!r}")
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    t = payload["trials"][0]
    c = payload["confirmation_of"]
    print(f"[emit-trials] wrote 1 CONFIRMATION trial ({args.arm}, "
          f"{payload['model']}@{payload['model_revision']}, "
          f"positions={payload['position']}, round {round_no}, "
          f"si={payload.get('si')}) confirming the fitted argmin of "
          f"{c['fit']} (rounds {c['rounds']}, fitted min log y "
          f"{c['fitted_min_log_y']}, sigma_log_y {c['sigma_log_y']}) -> "
          f"{args.out}")
    print(f"  t00: lr={t['lr']!r} warmup_frac={t['warmup_frac']!r} "
          f"clip={t['max_grad_norm']!r} (verbatim from the winner)")


def _check_init_scheme(name: str) -> str:
    lib = _lora_modules_lib()
    schemes = tuple(lib.INIT_SCHEMES) if lib is not None else _INIT_SCHEMES_FALLBACK
    if str(name) not in schemes:
        raise SystemExit(f"[emit-trials] --init-scheme {name!r} is not one of "
                         f"{list(schemes)}")
    return str(name)


def _load_recon(path) -> dict:
    """The reconnaissance record `tune_arm.py recon` wrote: ``clip`` (the
    pooled-percentile constant), ``percentile``, ``steps``, ``ladder``,
    plus its sha for the search record."""
    path = Path(path)
    raw = path.read_bytes()
    rec = json.loads(raw.decode())
    clip = rec.get("clip")
    if clip is None or not (math.isfinite(float(clip)) and float(clip) > 0):
        raise SystemExit(f"[emit-trials] {path} carries no finite positive "
                         f"'clip' (got {clip!r}) — not a recon record")
    if rec.get("alarms"):
        print(f"[emit-trials] NOTE: the reconnaissance filed alarms "
              f"{rec['alarms']} — recorded on the search")
    return {**rec, "clip": float(clip),
            "percentile": float(rec.get("percentile", 95.0)),
            "path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


#: The recon record's facts a design payload carries (``payload["recon"]``).
_RECON_PAYLOAD_KEYS = ("path", "sha256", "clip", "percentile", "steps")


def _clip_const_from_args(args) -> tuple[float | None, dict | None]:
    """The held clip constant of a fixed-clip design and the recon record
    it came from: ``--clip-const``, or the ``clip`` of ``--recon``'s record
    (the two together refuse — they name the same constant); ``(None,
    None)`` when neither is given."""
    clip_const = getattr(args, "clip_const", None)
    recon_rec = None
    if getattr(args, "recon", None):
        if clip_const is not None:
            raise SystemExit("[emit-trials] --recon and --clip-const both name "
                             "the clip constant — pass one")
        recon_rec = _load_recon(args.recon)
        clip_const = recon_rec["clip"]
    return clip_const, recon_rec


def _emit_hpo():
    """The registry's native HPO emitter, ``sisteer.registry.emit_hpo``.
    Imported lazily; absent -> a printed notice and None, never a failure
    of the science (contract §5)."""
    try:
        from sisteer.registry import emit_hpo
    except ImportError:
        print("[registry] emit_hpo unavailable — search not recorded")
        return None
    return emit_hpo


def _code_sha() -> str | None:
    sha = os.environ.get("SISTEER_CODE_SHA")
    if sha:
        return sha
    try:
        import subprocess
        return subprocess.check_output(["git", "-C", str(REPO), "rev-parse",
                                        "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:  # noqa: BLE001 — a checkout without git is allowed
        return None


def _object_kind_of(arm: str) -> str:
    """The registry object kind of a tuned arm — the hooks' ONE legacy-kind
    table (lora_r4_ce -> lora, psr_ce_all -> relu_steer), the arm's own
    spelling when the table does not name it. Called only once the registry
    imported (after _emit_hpo())."""
    from sisteer.registry import hooks
    return hooks.object_kind_for_legacy_kind(arm) or arm


def _open_search_for(args, cfg: dict, payload: dict, recon_rec: dict | None,
                     ) -> str | None:
    """`emit-trials` -> ``open_search`` (contract §5) when ``--campaign`` is
    given: the design becomes the ``fit_search.search_space_json``,
    the search id is stamped on the payload so every trial's
    terminal ``fit_trial`` names it, and the id is printed as
    ``registry: fit_search <id>``. No campaign -> nothing recorded, said."""
    campaign = getattr(args, "campaign", None)
    if not campaign:
        if payload.get("design") == V2_DESIGN or payload.get("init_scheme"):
            print("[registry] no --campaign: the search is not recorded (pass "
                  "--campaign <name> to open a fit_search)")
        return None
    eh = _emit_hpo()
    if eh is None:
        return None
    arm = payload["arm"]
    model_id = getattr(args, "model_id", None) or model_slug(cfg["model"]) \
        or cfg["model"]
    lr_box = payload.get("lr_log10_box") or [math.log10(payload["box"]["lr"][0]),
                                             math.log10(payload["box"]["lr"][1])]
    const = payload.get("constant_axes") or {}
    space = {
        "lr": {"log10_box": [float(lr_box[0]), float(lr_box[1])]},
        "allowed_pairs": payload.get("allowed_pairs"),
        "excluded_pairs": payload.get("excluded_pairs"),
        "per_expert_rank_rule": payload.get("per_expert_rank_rule"),
        "warmup_frac": (payload["warmup_levels"][0]
                        if len(payload.get("warmup_levels") or []) == 1
                        else payload.get("warmup_levels")),
        "max_grad_norm": (const.get("max_grad_norm")
                          if const else payload["box"].get("max_grad_norm")),
        "init_scheme": payload.get("init_scheme"),
        "design": payload.get("design") or "sobol",
        "seed": payload.get("design_seed", payload.get("design_salt", 0)),
        "trial_budget": {"pool": payload.get("si"),
                         "epochs": int(getattr(args, "trial_epochs", 3) or 3),
                         "effective_batch": _config_effective_batch(cfg, arm)},
        "axes": payload.get("axes"),
    }
    extra = {
        "design_extended_to_32": payload.get("design_extended_to_32"),
        "coverage_repair": payload.get("coverage_repair"),
        "allowed_pairs_source": payload.get("allowed_pairs_source"),
        "clip_percentile": recon_rec.get("percentile") if recon_rec else None,
        "recon_alarms": (recon_rec or {}).get("alarms"),
        "round": payload["round"],
        "trials_in_flight": [f"r{payload['round']}_t{t['index']:02d}"
                             for t in payload["trials"]],
    }
    sid = eh.open_search(
        campaign=str(campaign), model_id=str(model_id),
        object_kind=_object_kind_of(arm), method="sobol", search_space=space,
        n_trials=len(payload["trials"]), objective="best_heldout_loss",
        objective_direction="minimize",
        # None = the producer's own substrate (emit.producer_block); a third
        # cluster names itself through SISTEER_REGISTRY_SUBSTRATE
        substrate=os.environ.get("SISTEER_REGISTRY_SUBSTRATE") or None,
        code_sha=_code_sha(), config_sha256=config_file_sha256_full(args.config),
        grad_norm_reconnaissance_uri=(recon_rec or {}).get("path"),
        extra=extra)
    payload["search_id"] = sid
    payload["campaign"] = str(campaign)
    # close_search / update_search chain from the OPENING payload (emit_hpo's
    # contract); open_search returns only the id, so read the record back
    # from the spool it just wrote and carry it on the design -> every
    # summary -> the analyze artefact -> the pick.
    payload["search_opening"] = _read_opening(eh, sid, str(campaign))
    print(f"registry: fit_search {sid}")
    return sid


def _read_opening(eh, sid: str, campaign: str) -> dict | None:
    """The opening ``fit_search`` payload emit_hpo just wrote for ``sid``
    (the record dir ``<spool>/<campaign>/fit_search__<sha16>/``), or None
    with a printed notice when the spool cannot be read (emission off)."""
    try:
        from sisteer.registry import emit as _emit, records as _records
        spool = _emit.spool_root()
        if not spool:
            return None
        d = os.path.join(spool, campaign, eh.run_dir_for(sid))
        if not os.path.isdir(d):
            print(f"[registry] no record dir at {d} — the opening payload is "
                  "not carried (emission off?)")
            return None
        opening = None
        for pth in sorted(_records.iter_record_files(d)):
            for _i, rec, _sha in _records.read_record_file(pth, compute_sha=False):
                if rec.get("record_type") == "fit_search" and \
                        (rec.get("payload") or {}).get("search_id") == sid:
                    opening = rec["payload"]
        return opening
    except Exception as e:  # noqa: BLE001 — never fail the design on a read-back
        print(f"[registry] could not read the opening fit_search back: {e}")
        return None


def _config_effective_batch(cfg: dict, arm: str) -> int | None:
    """The config's effective batch for ``trial_budget``;
    None — said, never silent — when the adapter cannot state it."""
    try:
        ad = make_adapter(arm)
        if isinstance(ad, CESoloAdapter):
            ad.bind_config(cfg)
            ad.allow_unfolded_config = True
            return int(ad.config_effective_batch())
    except SystemExit as e:
        print(f"[registry] trial_budget.effective_batch not recorded — the "
              f"{arm} adapter refused the config: {e}")
        return None
    return None


def _v2_lr_box(args, cfg: dict) -> tuple[float, float]:
    """The v2 design's log10 lr box: --lr-log10-min/max, else the per-model
    seed box's lr side (SEED_BOXES_BY_MODEL via the config's model slug),
    refusing a model with no registered box."""
    slug = model_slug(cfg["model"])
    per_model = SEED_BOXES_BY_MODEL.get((args.arm, slug or ""))
    defaults = ([math.log10(float(v)) for v in per_model["lr"]]
                if per_model is not None else [None, None])
    sides = []
    for flag, given, default in (
            ("--lr-log10-min", getattr(args, "lr_log10_min", None), defaults[0]),
            ("--lr-log10-max", getattr(args, "lr_log10_max", None), defaults[1])):
        if given is None:
            if default is None:
                raise SystemExit(
                    f"[emit-trials] no seed box for ({args.arm}, {slug}) and no "
                    f"{flag} — the design needs its lr box")
            given = default
        sides.append(float(given))
    return sides[0], sides[1]


def _emit_v2(args, cfg: dict, round_no: int) -> None:
    """`emit-trials --design sobol16_coverage`: the LoRA architecture
    search round (build_trials_v2) over the probe's allowed pairs."""
    if args.arm != "lora_r4_ce":
        raise SystemExit("[emit-trials] --design sobol16_coverage is the LoRA "
                         f"architecture search (arm lora_r4_ce), not {args.arm}")
    ap = getattr(args, "allowed_pairs", None)
    if not ap:
        raise SystemExit("[emit-trials] --design sobol16_coverage needs "
                         "--allowed-pairs <allowed_pairs.json> (the serving "
                         "probe's verdict, plan §5 step 4) — the design space "
                         "is a first-class input")
    allowed = load_allowed_pairs(ap)
    for flag in ("box_from", "clip_min", "clip_max", "pin_axis"):
        if getattr(args, flag, None):
            raise SystemExit(f"[emit-trials] --{flag.replace('_', '-')} does "
                             "not apply to --design sobol16_coverage")
    clip_const, recon_rec = _clip_const_from_args(args)
    if clip_const is None:
        raise SystemExit("[emit-trials] the v2 design holds the clip at the "
                         "reconnaissance constant — pass --recon <recon.json> "
                         "or --clip-const <value>")
    init_scheme = _check_init_scheme(getattr(args, "init_scheme", None)
                                     or "zero")
    levels = tuple(float(x) for x in str(args.warmup_levels).split(",")
                   if x.strip())
    if len(set(levels)) != 1:
        raise SystemExit("[emit-trials] the v2 design pins warm-up to ONE "
                         f"level (OVERRIDE_HPO_WARMUP_FRAC); got {levels} — "
                         "pass --warmup-levels 0.15")
    lo, hi = _v2_lr_box(args, cfg)
    n = int(args.n) if args.n is not None else V2_TRIALS
    payload = build_trials_v2(
        arm=args.arm, model=cfg["model"], model_revision=cfg.get("model_revision"),
        config_sha=_config_file_sha(args.config), position=args.position,
        round_no=round_no, lr_log10_box=(lo, hi), allowed=allowed,
        warmup_frac=levels[0], max_grad_norm=float(clip_const),
        init_scheme=init_scheme, n=n, seed=int(args.design_salt) or V2_SEED)
    si = getattr(args, "si", None)
    if si:
        payload["si"] = str(si)
    if recon_rec is not None:
        payload["recon"] = {k: recon_rec.get(k) for k in _RECON_PAYLOAD_KEYS}
    _open_search_for(args, cfg, payload, recon_rec)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    ext = payload.get("design_extended_to_32")
    print(f"[emit-trials] wrote {len(payload['trials'])} trials "
          f"({args.arm}, {payload['model']}@{payload['model_revision']}, "
          f"round {round_no}, design {payload['design']} seed "
          f"{payload['design_seed']}, u=log10(lr) on [{lo:g}, {hi:g}], "
          f"{len(payload['allowed_pairs'])} allowed pairs, "
          f"{len(payload['excluded_pairs'])} excluded, clip={clip_const!r}, "
          f"warmup={levels[0]:g}, init={init_scheme}, "
          f"coverage repair moved {len(payload['coverage_repair'])} point(s)"
          f"{', EXTENDED to 32 for coverage' if ext else ''}) -> {args.out}")
    for t in payload["trials"]:
        print(f"  t{t['index']:02d}: lr={t['lr']:.3e} rank={t['lora_rank']} "
              f"set={t['lora_set_id']} per_expert={t['per_expert_rank']}")


def parse_pairs_arg(text: str) -> list[tuple[int, str]]:
    """``"16:attn_mlp,64:attn_mlp,…"`` -> [(16, "attn_mlp"), …]; refuses a malformed or duplicated item."""
    out: list[tuple[int, str]] = []
    for item in str(text or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise SystemExit(f"[emit-trials] --pairs item {item!r} is not rank:set")
        r, sid = item.split(":", 1)
        try:
            key = pair_key(int(r), sid.strip())
        except ValueError:
            raise SystemExit(f"[emit-trials] --pairs item {item!r}: rank must be an integer")
        if key in out:
            raise SystemExit(f"[emit-trials] --pairs lists {key} twice")
        out.append(key)
    if not out:
        raise SystemExit("[emit-trials] --pairs names no (rank:set) pair")
    return out


def _round1_best_by_set(search_dir: Path, y_key: str = "best_heldout_loss") -> dict[str, dict]:
    """The best (lowest y) usable round-1 trial per module set, through the v2 pool's OWN reader (`_pool_rows_v2`, the rows the
    analyze fits) — the fallback lr source when the refit predicts nothing for a pair. Rows: lr, y, trial_id, rank."""
    best: dict[str, dict] = {}
    try:
        pool = _pool_rows_v2(Path(search_dir), [1], y_key)
    except SystemExit:
        return best
    for r in pool["rows"]:
        key = _pair_of(r)
        if key is None or r.get("lr") is None or r.get(y_key) is None:
            continue
        sid = key[1]
        if sid not in best or float(r[y_key]) < float(best[sid]["y"]):
            best[sid] = {"lr": float(r["lr"]), "y": float(r[y_key]), "trial_id": hpo_selection_guard.trial_id(r), "rank": key[0]}
    return best


def build_addendum_trials(art: dict, allowed: dict, pairs: list[tuple[int, str]], *, round_no: int, arm: str, model: str,
                          model_revision: str | None, config_sha: str, si: str | None, fit_path: str, fit_sha256: str,
                          round1_best: dict[str, dict]) -> dict:
    """One trial per requested (rank, set) pair the design never covered, at the lr the refit over the widened menu
    predicts for that pair (`fit.pair_predictions[rank:set].lr`; the interaction model's u*_rank when selected), else the
    set's best round-1 trial lr — the source is recorded per trial. Refuses a pair outside the menu, a pair the pooled rows
    already cover (an addendum covers NEW pairs), a pair with neither a prediction nor a round-1 trial of its set, and a
    non-finite lr. The payload is build_trials-shaped (round ``round_no``, ``trial_role`` addendum, the artefact's clip,
    warm-up, init and search id; ``allowed_pairs`` = the WIDENED menu with ``menu_extends`` = the design's) plus
    ``addendum_of`` / ``allowed_pairs_source`` provenance."""
    fit = art.get("fit") or {}
    preds = fit.get("pair_predictions") or {}
    pd = bool(fit.get("positive_definite"))
    interaction = bool(fit.get("interaction_selected") or fit.get("interaction"))
    covered = {tuple(_pair_of(r)) for r in (art.get("rows") or []) if _pair_of(r) is not None}
    menu_drawn = [tuple(pair_key(a, b)) for a, b in (art.get("menu_extended_from") or art.get("allowed_pairs") or [])]
    levels = [float(x) for x in (fit.get("levels") or art.get("warmup_levels") or [])]
    if len(set(levels)) != 1:
        raise SystemExit(f"[emit-trials] --addendum-from: the artefact pins no single warm-up level ({levels})")
    fixed = dict(art.get("fixed_axes") or fit.get("fixed_axes") or {})
    if "max_grad_norm" not in fixed:
        raise SystemExit("[emit-trials] --addendum-from: the artefact carries no fixed clip (fixed_axes.max_grad_norm)")
    init_scheme = str(art.get("init_scheme") or "zero")
    lo, hi = (float(v) for v in (art.get("lr_log10_box") or fit.get("lr_log10_box") or [None, None]))
    trials: list[dict] = []
    for i, key in enumerate(pairs):
        pr = refuse_pair_outside(allowed, key[0], key[1], who="emit-trials --addendum-from")
        if key in covered:
            raise SystemExit(f"[emit-trials] --addendum-from: pair {key} already has round-1 trial(s) — an addendum covers "
                             "pairs the design never trained; refusing")
        label = f"{key[0]}:{key[1]}"
        pred = preds.get(label) or {}
        lr = pred.get("lr") if pd else None
        if lr is not None and math.isfinite(float(lr)) and float(lr) > 0:
            src = {"kind": "pair_prediction", "u_star": pred.get("u_star"), "interaction_model": interaction,
                   "box_pos": pred.get("box_pos"), "interior": pred.get("interior")}
        else:
            fb = round1_best.get(key[1])
            if fb is None:
                raise SystemExit(f"[emit-trials] --addendum-from: no prediction for {label} (fit PD={pd}) and no usable "
                                 f"round-1 trial of set {key[1]!r} to fall back on — refusing")
            lr = float(fb["lr"])
            src = {"kind": "best_round1_trial", "trial_id": fb["trial_id"], "trial_rank": fb.get("rank"),
                   "trial_lr": lr, "trial_y": fb.get("y"), "reason": ("fit not positive-definite" if not pd
                                                                      else f"no prediction for {label}")}
        lr = float(lr)
        if not math.isfinite(lr) or lr <= 0:
            raise SystemExit(f"[emit-trials] --addendum-from: non-finite lr {lr!r} for {label}")
        trials.append({"index": i, "lr": lr, "warmup_frac": levels[0], "max_grad_norm": float(fixed["max_grad_norm"]),
                       "lora_rank": int(pr["rank"]), "lora_modules": list(pr["lora_modules"]), "lora_set_id": pr["set"],
                       "per_expert_rank": pr.get("per_expert_rank"), "init_scheme": init_scheme,
                       "trial_role": ADDENDUM_ROLE, "lr_source": src,
                       **({"menu_override": (allowed.get("menu_override") or {}).get("ruling")} if allowed.get("menu_override") else {})})
    box = {"lr": [10.0 ** lo, 10.0 ** hi] if lo is not None else None,
           "max_grad_norm": [float(fixed["max_grad_norm"])] * 2}
    payload = _art_base_payload(art, round_no=round_no, arm=arm, model=model, model_revision=model_revision,
                                config_sha=config_sha, box=box, trials=trials, si=si)
    payload.update({
        "design": ADDENDUM_DESIGN, "axes": ["lr", "lora_rank", "lora_modules"],
        "trial_role": ADDENDUM_ROLE, "init_scheme": init_scheme,
        "lr_log10_box": [lo, hi] if lo is not None else None,
        "allowed_pairs": [list(k) for k in allowed_pair_keys(allowed)],
        "allowed_pairs_detail": [dict(p) for p in allowed["pairs"]],
        "menu_extends": [list(k) for k in menu_drawn],
        "allowed_pairs_source": {"path": allowed.get("path"), "sha256": allowed.get("sha256"), "model_id": allowed.get("model_id"),
                                 "probe_job_id": allowed.get("probe_job_id"), "verdict_uri": allowed.get("verdict_uri"),
                                 "result_generation": allowed.get("result_generation")},
        "excluded_pairs": [[e["rank"], e["set"], e["reason"]] for e in allowed["excluded"]],
        "per_expert_rank_rule": per_expert_rank_rule(allowed),
        "set_order": set_order_for_pairs(allowed),
        "campaign": art.get("campaign"),                        # the trials' records need it (the ReLU lane's a4278b045 for follow-ups; the same here)
        "menu_override": allowed.get("menu_override"),          # the menu-gate override, in the design's provenance
        "addendum_of": {"round": 1, "search_id": art.get("search_id"), "fit_path": fit_path, "fit_sha256": fit_sha256,
                        "fit_design": art.get("design"), "fit_positive_definite": pd, "interaction_model": interaction,
                        "menu_extended_from": [list(k) for k in menu_drawn],
                        "menu_extension": art.get("menu_extension")},
    })
    return payload


def _emit_addendum(args, cfg: dict, addendum_from: str, round_no: int) -> None:
    """`emit-trials --addendum-from <r1 refit over the widened menu> --allowed-pairs <widened menu> --pairs rank:set,…`:
    the design flags are refused — the clip, warm-up, init and search id are the artefact's, the pairs come from
    --pairs and must be in the menu and new to the design; one row per pair at the predicted (else fallback) lr."""
    who = "--addendum-from"
    clashes = _design_flag_clashes(args, (*_LEGACY_DESIGN_FLAGS, *[f for f in _V2_DESIGN_FLAGS if f[0] != "--allowed-pairs"],
                                          ("--n", "n")))
    if clashes:
        raise SystemExit(f"[emit-trials] {who} takes no design flags ({clashes}): the clip, warm-up, init and search id are "
                         "the artefact's; the pairs come from --pairs")
    ap = getattr(args, "allowed_pairs", None)
    if not ap:
        raise SystemExit(f"[emit-trials] {who} needs --allowed-pairs <the WIDENED allowed_pairs.json> (the re-verdict's menu)")
    pairs_arg = getattr(args, "pairs", None)
    if not pairs_arg:
        raise SystemExit(f"[emit-trials] {who} needs --pairs rank:set,… (the pairs the re-verdict certified)")
    allowed = load_allowed_pairs(ap)
    fit_path = Path(addendum_from)
    fit_bytes = fit_path.read_bytes()
    art = json.loads(fit_bytes.decode())
    if art.get("design") != V2_ANALYZE_DESIGN:
        raise SystemExit(f"[emit-trials] {who}: {fit_path} is not a v2 analyze artefact (design={art.get('design')!r})")
    art_menu = [tuple(pair_key(a, b)) for a, b in (art.get("allowed_pairs") or [])]
    if sorted(art_menu) != sorted(allowed_pair_keys(allowed)):
        raise SystemExit(f"[emit-trials] {who}: the artefact was fit over {sorted(art_menu)}, not the widened menu "
                         f"{sorted(allowed_pair_keys(allowed))} — re-run `analyze --design v2 --allowed-pairs <menu> --extends-menu` first")
    payload = build_addendum_trials(
        art, allowed, parse_pairs_arg(pairs_arg), round_no=round_no, arm=args.arm, model=cfg["model"],
        model_revision=cfg.get("model_revision"), config_sha=_config_file_sha(args.config), si=_art_si(art, getattr(args, "si", None), who),
        fit_path=str(fit_path), fit_sha256=hashlib.sha256(fit_bytes).hexdigest(), round1_best=_round1_best_by_set(fit_path.parent))
    if args.position and payload["position"] != args.position:
        raise SystemExit(f"[emit-trials] {who}: --position {args.position!r} contradicts the artefact's {payload['position']!r}")
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"[emit-trials] wrote {len(payload['trials'])} ADDENDUM trial(s) ({args.arm}, round {round_no}, search "
          f"{payload.get('search_id')}) over the widened menu ({len(payload['allowed_pairs'])} pairs; design drawn over "
          f"{len(payload['menu_extends'])}) -> {args.out}")
    for t in payload["trials"]:
        print(f"  t{t['index']:02d}: lr={t['lr']:.3e} rank={t['lora_rank']} set={t['lora_set_id']} "
              f"per_expert={t['per_expert_rank']} lr_source={t['lr_source']['kind']}")


def _emit_followup(args, cfg: dict, fit_from: str, round_no: int, *,
                   refine: bool) -> None:
    """`--refine-from` / `--explore-from`: a follow-up design off an analyze
    artefact. Like --confirm-from, the design flags are refused — the
    point(s), box and levels are the artefact's."""
    who = "--refine-from" if refine else "--explore-from"
    clashes = _design_flag_clashes(
        args, (*_LEGACY_DESIGN_FLAGS, *_V2_DESIGN_FLAGS,
               *(() if refine else (("--n", "n"),))))
    if clashes:
        raise SystemExit(f"[emit-trials] {who} takes no design flags "
                         f"({clashes}): everything comes from the artefact")
    fit_path = Path(fit_from)
    fit_bytes = fit_path.read_bytes()
    art = json.loads(fit_bytes.decode())
    common = dict(fit_path=str(fit_path),
                  fit_sha256=hashlib.sha256(fit_bytes).hexdigest(),
                  model=cfg["model"], model_revision=cfg.get("model_revision"),
                  config_sha=_config_file_sha(args.config), arm=args.arm,
                  round_no=round_no, si=getattr(args, "si", None))
    if refine:
        payload = build_refine_trials(
            art, n=int(args.n) if args.n is not None else REFINE_TRIALS,
            **common)
    else:
        payload = build_exploration_replicate(art, **common)
    if args.position and payload["position"] != args.position:
        raise SystemExit(f"[emit-trials] {who}: --position {args.position!r} "
                         f"contradicts the artefact's {payload['position']!r}")
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    if refine:
        ro = payload["refine_of"]
        print(f"[emit-trials] wrote {len(payload['trials'])} REFINE trials "
              f"({args.arm}, round {round_no}) at pair {ro['pair']} on "
              f"log10(lr) in [{ro['box_log10'][0]:.3f}, {ro['box_log10'][1]:.3f}] "
              f"around {ro['center_log10']:.3f} (from {ro['fit']}, selected_from "
              f"{ro['selected_from']}) -> {args.out}")
    else:
        ro = payload["exploration_replicate_of"]
        print(f"[emit-trials] wrote 1 EXPLORATION REPLICATE ({args.arm}, round "
              f"{round_no}) of trial {ro['trial']} (original {ro['y_key']} "
              f"{ro['original_y']:.5f}; fall-through "
              f"{(ro.get('fall_through') or {}).get('reason')}) -> {args.out}")
    for t in payload["trials"]:
        arch = "".join(f" {k}={t[k]!r}" for k in ("lora_rank", "lora_set_id")
                       if k in t)
        print(f"  t{t['index']:02d}: lr={t['lr']!r} warmup_frac="
              f"{t['warmup_frac']!r} clip={t['max_grad_norm']!r}{arch}")


def _emit_confirm_pick(args, cfg: dict, request_from: str, round_no: int) -> None:
    """`emit-trials --confirm-pick-from <hpo_pick_confirm_request.json>`: the
    one-row design at the requested pair (build_confirm_pick_trial); the
    design fit it rides on is the one the request names beside itself. The
    design flags are refused — everything comes from the request and the
    artefact."""
    who = "--confirm-pick-from"
    clashes = _design_flag_clashes(
        args, (*_LEGACY_DESIGN_FLAGS, *_V2_DESIGN_FLAGS, ("--n", "n")))
    if clashes:
        raise SystemExit(f"[emit-trials] {who} takes no design flags "
                         f"({clashes}): everything comes from the request")
    req_path = Path(request_from)
    req_bytes = req_path.read_bytes()
    request = json.loads(req_bytes.decode())
    if request.get("kind") != "confirm-pick":
        raise SystemExit(f"[emit-trials] {who}: {req_path} is not a confirm-pick "
                         f"request (kind={request.get('kind')!r})")
    fit_path = req_path.parent / str(request.get("design_fit") or "r1_fit.json")
    if not fit_path.is_file():
        raise SystemExit(f"[emit-trials] {who}: the request's design fit "
                         f"{fit_path} is missing")
    fit_bytes = fit_path.read_bytes()
    art = json.loads(fit_bytes.decode())
    payload = build_confirm_pick_trial(
        request, art, request_path=str(req_path),
        request_sha256=hashlib.sha256(req_bytes).hexdigest(),
        fit_path=str(fit_path), fit_sha256=hashlib.sha256(fit_bytes).hexdigest(),
        model=cfg["model"], model_revision=cfg.get("model_revision"),
        config_sha=_config_file_sha(args.config), arm=args.arm,
        round_no=round_no, si=getattr(args, "si", None))
    if args.position and payload["position"] != args.position:
        raise SystemExit(f"[emit-trials] {who}: --position {args.position!r} "
                         f"contradicts the artefact's {payload['position']!r}")
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n")
    t = payload["trials"][0]
    c = payload["confirm_pick_of"]
    print(f"[emit-trials] wrote 1 CONFIRM-PICK trial ({args.arm}, round {round_no}, "
          f"t{t['index']:02d}) at the user's pick {c['pair']} lr={t['lr']!r} "
          f"({c['lr_source']}; ruling {c['ruling_id']}) -> {args.out}")


#: The emit-trials CLI's argument defaults — ONE table for the parser
#: (main), for in-process callers (scripts/sdfbase/hpo_next_round.py) and
#: for the tests, so a Namespace built outside argparse cannot drift from
#: the CLI. ``round`` None = resolved by emit_trials (1 / CONFIRMATION_ROUND).
EMIT_TRIALS_DEFAULTS: dict = {
    "out": None, "arm": None, "config": None, "position": "all", "n": None,
    "design_salt": 0, "pin_axis": [], "round": None, "box_from": None,
    "lr_min": None, "lr_max": None, "clip_min": None, "clip_max": None,
    "clip_const": None, "si": None, "warmup_levels": "0.05,0.15",
    "confirm_from": None,
    # The LoRA architecture search design, its inputs, the refine /
    # exploration follow-ups, the registry open
    "design": "legacy", "allowed_pairs": None, "recon": None,
    "init_scheme": None, "lr_log10_min": None, "lr_log10_max": None,
    "refine_from": None, "explore_from": None, "confirm_pick_from": None,
    "campaign": None, "model_id": None, "trial_epochs": 3,
    # the addendum round over pairs certified after the design was drawn
    "addendum_from": None, "pairs": None,
}


def emit_args(**over) -> argparse.Namespace:
    """An emit_trials Namespace with the CLI's defaults, ``over`` applied;
    an unknown key is a typo, not a new flag."""
    unknown = sorted(set(over) - set(EMIT_TRIALS_DEFAULTS))
    if unknown:
        raise TypeError(f"emit_args: unknown emit-trials argument(s) {unknown}")
    ns = {**EMIT_TRIALS_DEFAULTS, "pin_axis": list(EMIT_TRIALS_DEFAULTS["pin_axis"])}
    ns.update(over)
    return argparse.Namespace(**ns)


# --------------------------------------------------------------------------
# Quadratic response surface in the logs (house heuristic 3;
# generalized for the re-tune@32's extra axes — the legacy 2-axis pools
# reproduce the original 9-term fit exactly, names and order included).
# --------------------------------------------------------------------------

#: Display names for the fit's term labels.
_DISP = {"lr": "lr", "max_grad_norm": "clip", "temperature": "temp",
         "weight_decay": "wd"}


def _cont_coord(axis: str, value) -> float:
    """A row value -> its fit coordinate. weight_decay rides
    log(wd + WD_LOG_EPS) so the 0 grid level has a finite coordinate; every
    other axis is a plain log."""
    v = float(value)
    if axis == "weight_decay":
        return math.log(v + WD_LOG_EPS)
    return math.log(v)


def _cont_value(axis: str, coord: float) -> float:
    if axis == "weight_decay":
        wd = math.exp(coord) - WD_LOG_EPS
        # Sub-eps residue is the log-eps mapping's floor, not a real
        # recommendation: the lower grid edge means exactly "no decay".
        return wd if wd > WD_LOG_EPS * 0.5 else 0.0
    return math.exp(coord)


def _grid_log_bounds(axis: str) -> tuple[float, float]:
    grid = AXIS_GRIDS[axis]
    return (_cont_coord(axis, min(grid)), _cont_coord(axis, max(grid)))


# --------------------------------------------------------------------------
# Deterministic dense linear algebra for the response-surface fit.
#
# WHY NOT torch / a BLAS call: the fit artefact's BYTES are an identity. The
# confirmation design (emit-trials --confirm-from) pins ``fit_sha256`` and the
# argmin it was spawned from, and the fold's confirm-check anchors on them. On
# 2026-09-10 the sdfbase-llama8b search found that ``torch.linalg.lstsq`` on CPU
# (MKL LAPACK) is NOT run-to-run reproducible for identical float64 inputs:
# eight fresh interpreters on the same 14 rows gave three or four distinct
# argmins (relative spread ~5e-15; coefficients and residuals differed too);
# ``torch.set_num_threads(1)``, ``OMP/MKL_NUM_THREADS=1`` and
# ``MKL_CBWR=COMPATIBLE`` each still produced several distinct results, so the
# source is MKL's input-address-dependent kernel dispatch, not threading. A
# --reuse replay of the launcher re-ran analyze and the rewritten artefact's
# sha and 1e-17 lr difference failed the identity checks of a confirmation
# that had passed every substantive check.
#
# Plain CPython float arithmetic is IEEE-754 double evaluated strictly in
# program order — no FMA contraction, no SIMD reassociation, no threads — so
# the same rows give the same bytes on every run. The problems are tiny (a
# 14 x 5 design, a K <= 4 Hessian), so the textbook algorithms below cost
# microseconds: Householder QR for the least squares, cyclic Jacobi for the
# symmetric eigenvalues, one-sided (Hestenes) Jacobi for the design's
# singular values (no squaring, so a rank-deficient column still reads as
# ~1e-16, far under the 1e-10 rank threshold), partial-pivot elimination for
# the boxed argmin. ``math.fsum`` is the exactly-rounded sum, so no dot
# product depends on summation order. tests/test_tune_arm_fit_determinism.py
# pins the byte identity across fresh interpreters and the agreement with an
# independent solver.
# --------------------------------------------------------------------------

def _dot(u: list[float], v: list[float]) -> float:
    return math.fsum(a * b for a, b in zip(u, v))


def _lstsq_householder(A: list[list[float]], y: list[float]) -> list[float]:
    """The least-squares solution of A x ~ y (A: n x p, n >= p, full column
    rank — the caller's ``check_design`` refuses otherwise) by Householder
    QR with the standard sign choice and back-substitution."""
    n, p = len(A), len(A[0])
    R = [list(map(float, row)) for row in A]
    b = [float(v) for v in y]
    for k in range(p):
        x = [R[i][k] for i in range(k, n)]
        norm = math.sqrt(math.fsum(v * v for v in x))
        if norm == 0.0:
            continue
        alpha = -norm if x[0] >= 0.0 else norm
        v = list(x)
        v[0] -= alpha
        vv = math.fsum(t * t for t in v)
        if vv == 0.0:
            continue
        for j in range(k, p):
            col = [R[i][j] for i in range(k, n)]
            f = 2.0 * _dot(v, col) / vv
            for t in range(len(v)):
                R[k + t][j] = col[t] - f * v[t]
        seg = b[k:]
        f = 2.0 * _dot(v, seg) / vv
        for t in range(len(v)):
            b[k + t] = seg[t] - f * v[t]
    x = [0.0] * p
    for i in range(p - 1, -1, -1):
        s = math.fsum(R[i][j] * x[j] for j in range(i + 1, p))
        x[i] = (b[i] - s) / R[i][i]
    return x


def _eigvalsh_jacobi(M: list[list[float]], max_sweeps: int = 100
                     ) -> list[float]:
    """Eigenvalues of a small real symmetric matrix, ascending (the order
    ``torch.linalg.eigvalsh`` returned), by cyclic Jacobi rotations."""
    K = len(M)
    A = [list(map(float, row)) for row in M]
    if K == 1:
        return [A[0][0]]
    for _ in range(max_sweeps):
        off = math.fsum(A[i][j] * A[i][j]
                        for i in range(K) for j in range(K) if i != j)
        diag = math.fsum(A[i][i] * A[i][i] for i in range(K))
        if off == 0.0 or off <= 1e-32 * diag:
            break
        for pi in range(K - 1):
            for q in range(pi + 1, K):
                apq = A[pi][q]
                if apq == 0.0:
                    continue
                app, aqq = A[pi][pi], A[q][q]
                theta = (aqq - app) / (2.0 * apq)
                t = ((1.0 if theta >= 0.0 else -1.0)
                     / (abs(theta) + math.sqrt(theta * theta + 1.0)))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                A[pi][pi] = app - t * apq
                A[q][q] = aqq + t * apq
                A[pi][q] = A[q][pi] = 0.0
                for k in range(K):
                    if k == pi or k == q:
                        continue
                    akp, akq = A[k][pi], A[k][q]
                    A[k][pi] = A[pi][k] = c * akp - s * akq
                    A[k][q] = A[q][k] = s * akp + c * akq
    return sorted(A[i][i] for i in range(K))


def _svdvals_jacobi(A: list[list[float]], max_sweeps: int = 100
                    ) -> list[float]:
    """Singular values of a small n x p matrix (n >= p), descending (the
    order ``torch.linalg.svdvals`` returned), by one-sided Jacobi: rotate
    column pairs until every pair is orthogonal; the singular values are
    then the column norms. Each is accurate to ~eps * ||A|| absolutely, so
    an exactly collinear column reads as ~1e-16 and the caller's relative
    1e-10 rank threshold still catches it (a Gram-matrix route would square
    the gap away)."""
    n, p = len(A), len(A[0])
    cols = [[float(A[i][j]) for i in range(n)] for j in range(p)]
    for _ in range(max_sweeps):
        rotated = False
        for i in range(p - 1):
            for j in range(i + 1, p):
                ci, cj = cols[i], cols[j]
                alpha = math.fsum(x * x for x in ci)
                beta = math.fsum(x * x for x in cj)
                gamma = _dot(ci, cj)
                if gamma == 0.0 or abs(gamma) <= 1e-15 * math.sqrt(alpha * beta):
                    continue
                zeta = (beta - alpha) / (2.0 * gamma)
                t = ((1.0 if zeta >= 0.0 else -1.0)
                     / (abs(zeta) + math.sqrt(1.0 + zeta * zeta)))
                c = 1.0 / math.sqrt(1.0 + t * t)
                s = c * t
                for k in range(n):
                    x, yv = ci[k], cj[k]
                    ci[k] = c * x - s * yv
                    cj[k] = s * x + c * yv
                rotated = True
        if not rotated:
            break
    return sorted((math.sqrt(math.fsum(x * x for x in col)) for col in cols),
                  reverse=True)


def _solve_dense(M: list[list[float]], rhs: list[float]) -> list[float]:
    """x with M x = rhs by Gaussian elimination with partial pivoting. M is
    a principal submatrix of a positive-definite Hessian here, so a zero
    pivot cannot occur; it raises rather than returning garbage if it ever
    does."""
    n = len(M)
    a = [list(map(float, M[i])) + [float(rhs[i])] for i in range(n)]
    for c in range(n):
        piv = max(range(c, n), key=lambda r: abs(a[r][c]))
        if a[piv][c] == 0.0:
            raise ZeroDivisionError("_solve_dense: singular system")
        if piv != c:
            a[c], a[piv] = a[piv], a[c]
        for r in range(c + 1, n):
            f = a[r][c] / a[c][c]
            if f != 0.0:
                for k in range(c, n + 1):
                    a[r][k] -= f * a[c][k]
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = math.fsum(a[i][j] * x[j] for j in range(i + 1, n))
        x[i] = (a[i][n] - s) / a[i][i]
    return x


def _solve_boxed(H: list[list[float]], b: list[float],
                 bounds: dict[int, tuple[float, float]]
                 ) -> tuple[list[float], list[int]]:
    """argmin of b^T z + 1/2 z^T H z (H positive definite), with z[i]
    clamped into bounds[i] for the GRID axes (lr/clip carry no bounds here
    — their edge semantics are the recenter check downstream). Active-set
    on a PD quadratic: solve unconstrained, clamp violated grid coords to
    their nearest bound, re-solve the free coordinates given the clamped
    ones, repeat (the active set only grows, so <= len(bounds)+1 passes).
    Returns (z, clamped dim indices). Plain floats throughout — see the
    determinism note above."""
    K = len(H)
    fixed: dict[int, float] = {}
    z = [0.0] * K
    for _ in range(len(bounds) + 1):
        free = [i for i in range(K) if i not in fixed]
        z = [0.0] * K
        for i, v in fixed.items():
            z[i] = v
        if free:
            rhs = [-float(b[i]) for i in free]
            if fixed:
                idx = list(fixed)
                rhs = [rhs[k] - math.fsum(H[i][j] * fixed[j] for j in idx)
                       for k, i in enumerate(free)]
            sub = [[float(H[i][j]) for j in free] for i in free]
            zf = _solve_dense(sub, rhs)
            for k, i in enumerate(free):
                z[i] = zf[k]
        grew = False
        for i, (lo, hi) in bounds.items():
            if i not in fixed and not (lo <= z[i] <= hi):
                fixed[i] = lo if z[i] < lo else hi
                grew = True
        if not grew:
            break
    return z, sorted(fixed)


def check_design(A) -> tuple[int, float]:
    """(rank, condition number) of the COLUMN-NORMALIZED design matrix —
    normalization strips the trivial scale disparity between the linear and
    quadratic columns so the condition number measures genuine collinearity
    (e.g. every row sharing one clip value). ``A`` is a list of rows (a
    tensor is accepted and converted); the singular values come from the
    deterministic one-sided Jacobi above."""
    rows = A.tolist() if hasattr(A, "tolist") else [list(r) for r in A]
    n, p = len(rows), len(rows[0])
    norms = [math.sqrt(math.fsum(rows[i][j] * rows[i][j] for i in range(n)))
             for j in range(p)]
    norms = [nm if nm > 0.0 else 1.0 for nm in norms]
    An = [[rows[i][j] / norms[j] for j in range(p)] for i in range(n)]
    s = _svdvals_jacobi(An)
    rank = sum(1 for v in s if v > s[0] * 1e-10) if s[0] > 0 else 0
    cond = float(s[0] / s[-1]) if s[-1] > 0 else float("inf")
    return rank, cond


def fit_quadratic(rows: list[dict], *, levels: tuple[float, float],
                  y_key: str = "best_heldout_loss",
                  fixed_axes: dict | None = None) -> dict:
    """Least-squares quadratic response surface of log(y).

    THE EXACT FORM. Continuous log-axes, discovered from the rows —
    always L=log(lr), C=log(clip), plus tau=log(temperature) and/or
    D=log(weight_decay + WD_LOG_EPS) when every row carries the key. Terms:

        1;  the K linears;  the K squares;  the K(K-1)/2 cross terms;
        W (the warmup-level indicator);  the K W*axis interactions;
        (lora) one dummy per non-baseline LoRA-rank level — rank is a
        FACTOR with main effects only, baseline = the smallest level.

    For K=2 with no rank levels this is the original 9-term model, names
    and order included. Returns the design rank/cond, the continuous
    K x K Hessian's PD verdict (shared across warmup/rank levels — there
    are no W or rank interactions with the quadratic terms), per-warmup-
    level stationary points with tau/D CLAMPED into their grid ranges
    (active-set re-solve; lr/clip stay unclamped for the downstream box
    edge -> recenter check), the best rank level (additive main effect:
    the continuous argmin is rank-independent by construction), and the
    overall argmin (the lower fitted minimum of the two warmup levels).
    Diverged / floor-failing / non-finite rows must be excluded by the
    caller — ``analyze`` does exactly that.

    ``fixed_axes`` is the DECLARED fixed-axis identity of the pooled design
    (analyze reads it off every summary's own box: a zero-width clip side
    ``[c, c]`` declares the LR-only design's constant). A declared axis is
    held fixed at the declared value and every row is VALIDATED against it
    — never inferred from float uniqueness across rows, which a drifted
    constant (0.34 vs 0.3400000000000001) would split into two levels and
    a rank-deficient 9-term design. With no declaration
    (every lr x clip design, and synthetic pools) the axis discovery below
    is unchanged."""
    lv_lo, lv_hi = sorted(levels)
    declared = dict(fixed_axes or {})

    def _w(row) -> float:
        wf = float(row["warmup_frac"])
        if abs(wf - lv_lo) < 1e-9:
            return 0.0
        if abs(wf - lv_hi) < 1e-9:
            return 1.0
        raise SystemExit(f"[analyze] trial warmup_frac={wf} matches neither "
                         f"design level {lv_lo}/{lv_hi}")

    # Axis discovery + all-or-none consistency (one analyze run covers ONE
    # design; a pool mixing designs has no single model). An extra axis
    # whose value is IDENTICAL across every row (a --pin-axis round pooled
    # with same-valued survivors, e.g. temperature=1.0 after the T!=1
    # top-K-head divergence) is EXCLUDED from the design — a constant
    # column is rank-deficient by construction — and echoed as a FIXED
    # value instead. The clip takes the same route, decided FIRST: it is a
    # continuous axis only when it VARIES across the pooled usable rows
    # (every lr x clip design — their term names and order stay byte-
    # identical, QUAD9_NAMES at K=2); a constant clip is the LR-only design
    # (sdfbase-llama8b: the recon-derived per-arm constant, held on
    # purpose — not a degenerate pool) and is echoed fixed, leaving the
    # 5-term model 1, lr, lr*lr, warmup, warmup*lr. lr never takes it: a
    # constant lr IS a degenerate pool, kept behind the rank-deficiency
    # refusal.
    cont = ["lr"]
    fixed_axes = {}
    if "max_grad_norm" in declared:
        # The design DECLARED the clip constant: hold it at that value and
        # refuse any row off it (a corrupt or hand-edited summary), rather
        # than letting float noise split the pool into two clip levels.
        c_decl = float(declared["max_grad_norm"])
        off = [(hpo_selection_guard.trial_id(r), r["max_grad_norm"])
               for r in rows if float(r["max_grad_norm"]) != c_decl]
        if off:
            raise SystemExit(
                f"[analyze] {len(off)} row(s) carry a clip other than the "
                f"design's declared constant {c_decl!r}: {off[:5]} — the "
                "LR-only design stamps ONE constant on every row; a row off "
                "it is a corrupt summary, not a second clip level")
        fixed_axes["max_grad_norm"] = c_decl
    else:
        clip_vals = sorted({float(r["max_grad_norm"]) for r in rows})
        if len(clip_vals) == 1:
            fixed_axes["max_grad_norm"] = clip_vals[0]
        else:
            cont.append("max_grad_norm")
    for ax in ("temperature", "weight_decay"):
        have = [ax in r for r in rows]
        if any(have):
            if not all(have):
                raise SystemExit(
                    f"[analyze] {ax} present on some rows and absent on "
                    "others — one analyze run covers ONE design")
            vals = sorted({float(r[ax]) for r in rows})
            if len(vals) == 1:
                fixed_axes[ax] = vals[0]
            else:
                cont.append(ax)
    have_rank = ["lora_rank" in r for r in rows]
    rank_levels: list[int] = []
    if any(have_rank):
        if not all(have_rank):
            raise SystemExit(
                "[analyze] lora_rank present on some rows and absent on "
                "others — one analyze run covers ONE design")
        rank_levels = sorted({int(r["lora_rank"]) for r in rows})
        if len(rank_levels) == 1:
            # A single level carries no factor dof — it is the whole
            # pool's rank, echoed fixed like a constant continuous axis.
            fixed_axes["lora_rank"] = int(rank_levels[0])
            rank_levels = []

    K = len(cont)
    d = [_DISP[a] for a in cont]
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    names = (["1"] + d + [f"{a}*{a}" for a in d]
             + [f"{d[i]}*{d[j]}" for i, j in pairs]
             + ["warmup"] + [f"warmup*{a}" for a in d]
             + [f"rank={rl}" for rl in rank_levels[1:]])

    if len(rows) < len(names):
        raise SystemExit(
            f"[analyze] too few usable trials ({len(rows)}) for the "
            f"{len(names)}-term quadratic — investigate the exclusions")

    def _feats(row) -> list[float]:
        x = [_cont_coord(a, row[a]) for a in cont]
        w = _w(row)
        f = ([1.0] + x + [xi * xi for xi in x]
             + [x[i] * x[j] for i, j in pairs]
             + [w] + [w * xi for xi in x])
        for rl in rank_levels[1:]:
            f.append(1.0 if int(row["lora_rank"]) == rl else 0.0)
        return f

    A = [_feats(r) for r in rows]
    y = [math.log(max(float(r[y_key]), 1e-12)) for r in rows]
    rank, cond = check_design(A)
    out: dict = {"names": list(names), "n_rows": len(rows),
                 "design_rank": rank, "design_cond": cond,
                 "levels": [lv_lo, lv_hi], "cont_axes": list(cont),
                 "rank_levels": list(rank_levels),
                 "fixed_axes": dict(fixed_axes)}
    if rank < len(names):
        out["rank_deficient"] = True
        return out
    # Deterministic plain-float least squares (see the note above
    # _lstsq_householder): the artefact's bytes are an identity the
    # confirmation design pins, and torch's MKL lstsq is not run-to-run
    # reproducible on CPU.
    coef = _lstsq_householder(A, y)
    out["coef"] = list(coef)
    # Residuals in log y: the selection guard's noise estimate is their
    # dof-corrected RMS (sigma_log_y = sqrt(SSR/(n-p)), plus the plain RMS
    # and robust MAD scale as context) — written BEFORE the PD verdict so a
    # non-PD artefact carries it too.
    resid = [yi - _dot(ai, coef) for ai, yi in zip(A, y)]
    out["resid_log_y"] = resid
    out.update(hpo_selection_guard.residual_scale(resid, n_params=len(names)))
    c = dict(zip(names, coef))
    # Continuous K x K Hessian (shared across warmup/rank levels — no W or
    # rank interactions with the quadratic terms).
    H = [[0.0] * K for _ in range(K)]
    for i in range(K):
        H[i][i] = 2.0 * c[f"{d[i]}*{d[i]}"]
    for i, j in pairs:
        H[i][j] = H[j][i] = c[f"{d[i]}*{d[j]}"]
    eigs = _eigvalsh_jacobi(H)
    pd = all(e > 0 for e in eigs)
    out["hessian_eigs"] = list(eigs)
    out["positive_definite"] = pd
    if not pd:
        return out

    # Rank factor: additive main effects against the baseline level.
    best_rank = None
    if rank_levels:
        effects = {rank_levels[0]: 0.0}
        for rl in rank_levels[1:]:
            effects[rl] = c[f"rank={rl}"]
        out["rank_effects"] = {str(k): float(v) for k, v in effects.items()}
        best_rank = min(effects, key=effects.get)
        out["best_rank"] = int(best_rank)

    bounds = {i: _grid_log_bounds(cont[i]) for i in range(K)
              if cont[i] in AXIS_GRIDS}
    per_level = {}
    best = None
    for w, lv in ((0.0, lv_lo), (1.0, lv_hi)):
        b = [c[d[i]] + c[f"warmup*{d[i]}"] * w for i in range(K)]
        z, clamped = _solve_boxed(H, b, bounds)
        val = float(c["1"] + c["warmup"] * w + _dot(b, z)
                    + 0.5 * math.fsum(z[i] * H[i][j] * z[j]
                                      for i in range(K) for j in range(K)))
        if best_rank is not None and best_rank != rank_levels[0]:
            val += c[f"rank={best_rank}"]
        rec = {"argmin": {cont[i]: _cont_value(cont[i], z[i])
                          for i in range(K)},
               "fitted_min_log_y": val,
               "clamped_axes": [cont[i] for i in clamped]}
        per_level[f"{lv:.6g}"] = rec
        if best is None or rec["fitted_min_log_y"] < best[1]:
            best = (lv, rec["fitted_min_log_y"], rec)
    out["per_level"] = per_level
    out["argmin"] = {**best[2]["argmin"], "warmup_frac": best[0]}
    if best_rank is not None:
        out["argmin"]["lora_rank"] = int(best_rank)
    # Fixed axes ride the argmin (and hence the selection/winner blocks) as
    # the constant the pool was trained at — the winner must state e.g.
    # temperature=1.0 explicitly, or the production fold falls back to a
    # registry default that is not 1.0 for every arm.
    out["argmin"].update(fixed_axes)
    out["clamped_axes"] = best[2]["clamped_axes"]
    out["fitted_min_log_y"] = best[1]
    return out


def _edge_check(argmin: dict, box: dict) -> tuple[bool, dict]:
    """(on_edge, {axis: log-position in the box}) — the guard's criterion
    (a), kept under its historical name for the drivers that import it."""
    return hpo_selection_guard.box_edge_positions(argmin, box,
                                                  edge_tol=EDGE_TOL)


def _boxed(center: dict, box: dict, zoom: float) -> dict:
    """A new log-space box of ``zoom`` x the old width, centered at
    ``center`` (clamped into the old box first, so a far-out argmin shifts
    the box at most half a width toward it). A FIXED axis — a zero-width
    side ``[c, c]``, the LR-only design's clip constant — rides
    VERBATIM: ``exp(log(c))`` does not round-trip every float (0.34 comes
    back 0.3400000000000001, 0.096 as 0.09600000000000002), and a drifted
    constant stamped on the recenter round is a SECOND clip value at
    pooling time — a 9-term design that refuses as rank-deficient after
    the round has been paid for."""
    out = {}
    for k in ("lr", "max_grad_norm"):
        lo_v, hi_v = float(box[k][0]), float(box[k][1])
        if hpo_selection_guard.is_fixed_axis(lo_v, hi_v):
            out[k] = [lo_v, hi_v]
            continue
        lo, hi = math.log(lo_v), math.log(hi_v)
        c = min(max(math.log(center[k]), lo), hi)
        half = (hi - lo) * zoom / 2.0
        out[k] = [math.exp(c - half), math.exp(c + half)]
    return out


def _trial_clip_activation(trial_dir: Path, summary: dict
                           ) -> tuple[float | None, int]:
    """(rate, n_steps) of ONE trial from its own metrics.jsonl beside
    summary.json: the share of its logged optimizer steps whose pre-clip
    grad norm (``<arm>/grad_norm``) exceeded the trial's OWN clip
    (``summary["trial"]["max_grad_norm"]``) — verified-facts §3, computed
    from the record the trainer already wrote, never re-measured, over the
    rows stamped with the summary's own attempt id (a summary without one
    predates the stamp and describes the whole file — which must then carry
    no stamp; an unstamped summary over a file holding a stamped re-run's
    rows is refused as ambiguous). (None, 0) when the trial directory
    carries no metrics.jsonl (synthetic artefacts, or a design analysed
    before the column existed)."""
    mpath = trial_dir / "metrics.jsonl"
    arm = summary.get("arm")
    clip = (summary.get("trial") or {}).get("max_grad_norm")
    if not mpath.is_file() or arm is None or clip is None:
        return None, 0
    # The file is append-mode across re-runs (run_trial re-runs
    # fingerprint-mismatched or torn work in the same directory), so the
    # rate is read over the segment THIS summary stamped — its attempt —
    # never the whole file (an earlier attempt's rows halved a
    # 40% reading to 20% and passed the promotion constraint).
    return recon_clip.clip_activation_from_metrics(
        mpath, arm=str(arm), clip=float(clip),
        attempt=summary.get(recon_clip.METRICS_ATTEMPT_KEY))


def _log_box_distance(row: dict, point: dict, box: dict,
                      fixed_axes: dict) -> float:
    """Distance between a trial row and a point in the DESIGN's own
    coordinates: per searched continuous axis the log-gap as a fraction of
    the box's log-width, squared and summed, plus 1.0 (a full box width)
    for a warmup-level mismatch — so the nearest trials of a 2-level design
    are its same-level neighbours. A fixed axis spans nothing."""
    d = 0.0
    for k in ("lr", "max_grad_norm"):
        if k in fixed_axes or k not in box:
            continue
        lo, hi = float(box[k][0]), float(box[k][1])
        if not (lo > 0 and hi > lo):
            continue
        width = math.log(hi) - math.log(lo)
        d += ((math.log(float(row[k])) - math.log(float(point[k]))) / width) ** 2
    if not math.isclose(float(row["warmup_frac"]), float(point["warmup_frac"]),
                        rel_tol=1e-9, abs_tol=0.0):
        d += 1.0
    return math.sqrt(d)


#: Alarm code analyze adds to the selection guard when EVERY trial with
#: metrics clipped on >= CLIP_ACTIVATION_MAX of its steps: the clip constant
#: is a normaliser on this design, not a spike guard, and nothing in it is
#: promotable. Filed through the guard so the
#: fold refuses it like any other alarm until acknowledged.
CLIP_ACTIVATION_ALARM = "clip_activation_over_threshold"

CLIP_ACTIVATION_RULE = (
    "share of a trial's logged optimizer steps whose PRE-clip grad norm "
    "(metrics.jsonl <arm>/grad_norm) exceeded that trial's own "
    "max_grad_norm (summary.trial); a trial at or above `threshold` is not "
    "promotable (ineligible as the raw-argmin fall-through; flagged as a "
    "fitted argmin's nearest neighbour); `target` is the regime the promoted "
    "point should sit in — verified-facts §3: healthy 0-4%, borderline "
    "24-65%, defective 100%")


def clip_activation_summary(rows: list[dict], *, y_key: str,
                            threshold: float = recon_clip.CLIP_ACTIVATION_MAX,
                            target: float = recon_clip.CLIP_ACTIVATION_TARGET
                            ) -> dict:
    """The fold table's clip-activation block over the usable rows (each
    carrying ``clip_activation_rate`` / ``clip_n_steps`` from
    _trial_clip_activation): per-trial rates, the trials at/over the
    threshold, and the best raw trial that IS promotable (rate below the
    threshold, or unknown — a trial without metrics is not excluded on
    evidence it does not have). ``best_raw_promotable_row`` is the row
    object (None when every trial with metrics is over); callers pop it
    before writing the artefact."""
    tid = hpo_selection_guard.trial_id
    known = [r for r in rows if r.get("clip_activation_rate") is not None]
    over = sorted(tid(r) for r in known
                  if float(r["clip_activation_rate"]) >= threshold)
    promotable = [r for r in rows
                  if r.get("clip_activation_rate") is None
                  or float(r["clip_activation_rate"]) < threshold]
    best = (min(promotable, key=lambda r: float(r[y_key]))
            if promotable else None)
    return {
        "threshold": threshold, "target": target,
        "rule": CLIP_ACTIVATION_RULE,
        "n_trials": len(rows), "n_with_metrics": len(known),
        "per_trial": {tid(r): {"rate": r.get("clip_activation_rate"),
                               "n_steps": int(r.get("clip_n_steps") or 0)}
                      for r in rows},
        "over_threshold": over,
        "all_known_over_threshold": bool(known) and len(over) == len(known),
        "best_raw_promotable": tid(best) if best is not None else None,
        "best_raw_promotable_row": best,
    }


def trial_clip_evidence_at(out_dir: Path, pair, lr: float, *,
                           rel_tol: float = 1e-9) -> list[dict]:
    """Every TERMINAL trial under ``out_dir`` — any round: the design, the
    refine, the round-9 confirmation, the round-3 exploration replicate — at
    EXACTLY the ``(pair, lr)`` coordinates (``pair`` None = an lr-only ReLU
    search), each with the clip-activation rate measured from its own
    metrics.jsonl (``_trial_clip_activation``; None without metrics). This is
    the measured evidence a PROMOTION of that regime rests on: FD 6's "clip
    activation > 25 % is not promotable" is decided by :func:`clip_gate` over
    exactly these trials, not over the pick's best-measured fallback path
    alone."""
    ev: list[dict] = []
    want = tuple(pair) if pair is not None else None
    for p in sorted(Path(out_dir).glob("r*_t*/summary.json")):
        try:
            s = json.loads(p.read_text())
        except (ValueError, OSError):
            continue                          # torn: not a terminal trial
        if not isinstance(s, dict) or s.get("status") not in _TERMINAL_STATUSES:
            continue
        t = s.get("trial") or {}
        if t.get("lr") is None:
            continue
        if not math.isclose(float(t["lr"]), float(lr), rel_tol=rel_tol):
            continue
        if want is not None and _pair_of(t) != want:
            continue
        rate, n_steps = _trial_clip_activation(p.parent, s)
        ev.append({"trial_id": p.parent.name, "round": s.get("round"),
                   "status": s.get("status"), "lr": float(t["lr"]),
                   "clip_activation_rate": rate, "clip_n_steps": n_steps,
                   "trial_role": s.get("trial_role"),
                   "confirmation": is_confirmation_design(s),
                   "confirm_pick": is_confirm_pick_design(s),
                   "y": s.get("best_heldout_loss")})
    return ev


def clip_gate(evidence: list[dict], *,
              threshold: float = recon_clip.CLIP_ACTIVATION_MAX
              ) -> tuple[bool, str]:
    """``(promotable, reason)`` for a (pair, lr) regime from
    :func:`trial_clip_evidence_at`'s trials. Promotable iff at least one
    COMPLETED trial at those coordinates MEASURED its clip activation below
    the threshold and none measured at/over it. Not promotable, by name:
    no trial at the coordinates (an inferred learning rate without a
    confirmation at that architecture), no measurement (metrics absent),
    any trial at/over the limit, or only failed trials (FD 6)."""
    ids = [e["trial_id"] for e in evidence]
    if not evidence:
        return False, ("no trial ran at these coordinates — an inferred "
                       "learning rate with no confirmation trial at this "
                       "architecture")
    known = [e for e in evidence if e.get("clip_activation_rate") is not None]
    if not known:
        return False, (f"no clip-activation measurement at these coordinates "
                       f"(trial(s) {ids} carry no metrics.jsonl)")
    over = [e for e in known if float(e["clip_activation_rate"]) >= threshold]
    if over:
        return False, ("clip activation " + ", ".join(
            f"{float(e['clip_activation_rate']):.1%} at {e['trial_id']}"
            for e in over) + f" is at/over the {threshold:.0%} promotion "
            "limit (FD 6: not promotable)")
    ok = [e for e in known if e.get("status") == "ok"]
    if not ok:
        return False, (f"no completed trial at these coordinates measured "
                       f"below the limit (trial(s) {ids} ended "
                       + ", ".join(str(e.get("status")) for e in known) + ")")
    return True, ("measured below the "
                  f"{threshold:.0%} limit at {[e['trial_id'] for e in ok]} "
                  f"(max {max(float(e['clip_activation_rate']) for e in known):.1%})")


def winner_clip_activation(point: dict, rows: list[dict], box: dict,
                           fixed_axes: dict, *,
                           threshold: float = recon_clip.CLIP_ACTIVATION_MAX,
                           target: float = recon_clip.CLIP_ACTIVATION_TARGET,
                           k: int = 3) -> dict:
    """A fitted argmin has no metrics of its own (no trial ran there), so
    its clip regime is read off its ``k`` nearest trials in log-box
    distance: FLAGGED when any known neighbour is at/over the
    threshold — the confirmation trial then has to MEASURE below it at the
    point; promotable when every known neighbour is below; None when no
    neighbour carries metrics."""
    tid = hpo_selection_guard.trial_id
    ranked = sorted(rows, key=lambda r: _log_box_distance(r, point, box,
                                                          fixed_axes))[:k]
    nearest = [{"trial": tid(r), "rate": r.get("clip_activation_rate"),
                "n_steps": int(r.get("clip_n_steps") or 0),
                "log_box_distance": _log_box_distance(r, point, box, fixed_axes),
                "lr": float(r["lr"]), "max_grad_norm": float(r["max_grad_norm"]),
                "warmup_frac": float(r["warmup_frac"])} for r in ranked]
    known = [float(n["rate"]) for n in nearest if n["rate"] is not None]
    flagged = any(rt >= threshold for rt in known)
    return {
        "threshold": threshold, "target": target, "k": k,
        "nearest": nearest, "rates_known": len(known),
        "max_nearest_rate": max(known) if known else None,
        "flagged": flagged,
        "promotable": (None if not known else not flagged),
        "note": ("no metrics.jsonl beside the nearest trials — the regime at "
                 "the argmin is unknown until its confirmation trial runs"
                 if not known else
                 "a nearest trial clipped on >= threshold of its steps: the "
                 "fitted argmin is FLAGGED — the confirmation trial at the "
                 "point must measure below the threshold before the point "
                 "is promoted" if flagged else
                 "every nearest trial with metrics is below the threshold"),
    }


def _fmt_rate(v) -> str:
    return "n/a" if v is None else f"{float(v):.3f}"


def _stamp_clip_activation(guard: dict, clip_act: dict,
                           winner_ca: dict | None) -> None:
    """Write the clip-activation block into the selection-guard record (the
    summary the argmin guard stamps; _validate_fold reads only alarmed /
    acknowledged / alarms, so the block is inert for the rendered config)
    and, when EVERY trial with metrics is at/over the threshold, file
    CLIP_ACTIVATION_ALARM — nothing on such a design is promotable, and the
    fold must refuse until someone acknowledges it."""
    block = {k: v for k, v in clip_act.items()}
    block["winner"] = winner_ca
    guard["clip_activation"] = block
    if clip_act.get("all_known_over_threshold"):
        guard["alarms"].append({
            "code": CLIP_ACTIVATION_ALARM, "dim": None,
            "text": (f"every trial with metrics ({clip_act['n_with_metrics']}) "
                     f"clipped on >= {clip_act['threshold']:.2f} of its "
                     "optimizer steps — the clip constant is a normaliser on "
                     "this design, not a spike guard; nothing in it is "
                     "promotable (re-derive the clip from the recon)")})
        guard["alarmed"] = True
        guard["text"] = hpo_selection_guard.alarm_lines(guard)
    line = (f"CLIP ACTIVATION: threshold {clip_act['threshold']:.2f} "
            f"(target {clip_act['target']:.2f}); "
            f"{clip_act['n_with_metrics']}/{clip_act['n_trials']} trials "
            f"carry metrics; over threshold: {clip_act['over_threshold'] or 'none'}; "
            f"raw fall-through pick: {clip_act['best_raw_promotable']}")
    if winner_ca is not None:
        line += (f"; fitted argmin's nearest {winner_ca['k']}: "
                 + ", ".join(f"{n['trial']}={_fmt_rate(n['rate'])}"
                             for n in winner_ca["nearest"])
                 + f" -> {'FLAGGED' if winner_ca['flagged'] else 'promotable' if winner_ca['promotable'] else 'unknown'}")
    guard["text"] = list(guard["text"]) + [line]


def analyze(args) -> None:
    if str(getattr(args, "design", "legacy") or "legacy") == "v2":
        return analyze_v2(args)
    rounds = [int(r) for r in str(args.rounds).split(",") if r.strip()]
    rows, excluded = [], []
    meta: dict = {}
    boxes: list[dict] = []
    levels_seen: set[tuple[float, ...]] = set()
    batch_accum: set[tuple] = set()  # realized (batch_size, accum_steps)
    accum_rows = 0
    labels_seen: set = set()   # heldout_objective — the ranking metric
    teacher_seen: set = set()  # teacher_forward — online vs precomputed_topk
    base_vals: list[float] = []  # untrained objective (heldout_curve[0])
    declared_fixed: dict = {}   # the design's DECLARED fixed axes (box [c, c])
    clip_declared: set[bool] = set()   # per usable summary: clip fixed?
    for r in rounds:
        for p in sorted(Path(args.out).glob(f"r{r}_t*/summary.json")):
            s = json.loads(p.read_text())
            if is_confirmation_design(s):
                # A confirmation trial (emit-trials --confirm-from) is
                # evidence FOR an already-selected point; pooling it would
                # re-fit the surface around the very point it confirms.
                raise SystemExit(
                    f"[analyze] {p.parent.name} is a CONFIRMATION trial "
                    f"(round {s.get('round')}, confirming "
                    f"{(s.get('confirmation_of') or {}).get('fit')!r}) — "
                    "evidence for the promoted point, never a search row; "
                    "drop its round from --rounds (the fold writer reads it "
                    "directly)")
            _refuse_replicate_row(p, s)
            y = s.get(args.y_key)
            reason = None
            if s.get("status") != "ok":
                reason = s.get("status", "missing_status")
            elif y is None or not math.isfinite(float(y)):
                # A non-finite objective is a failed fit, not a data point.
                reason = "nonfinite_objective"
            elif s.get("floor_ok") is False:
                # The ported analyzer fitted every status=="ok" row; a
                # floor-failing trial is a FAILED fit.
                reason = "floor_failed"
            if reason is not None:
                excluded.append((p.parent.name, reason))
                continue
            row = {**s["trial"], args.y_key: float(y), "round": r}
            # The clip-activation column: from the trial's own
            # metrics.jsonl against its own clip; None without metrics.
            row["clip_activation_rate"], row["clip_n_steps"] = \
                _trial_clip_activation(p.parent, s)
            rows.append(row)
            if s.get("heldout_curve"):
                # The untrained objective — the guard's ratio baseline. One
                # pool, one model, so every usable trial agrees to bf16
                # jitter (checked below; disagreement drops the ratios).
                base_vals.append(float(s["heldout_curve"][0]))
            if s.get("accum_steps") is not None:
                batch_accum.add((s.get("batch_size"), s.get("accum_steps")))
                accum_rows += 1
            if "heldout_objective" in s:
                labels_seen.add(s["heldout_objective"])
            if "teacher_forward" in s:
                teacher_seen.add(s["teacher_forward"])
            for k in ("arm", "model", "position", "si"):
                if k in s:
                    v = s[k]
                    if k == "arm":
                        # lora_r4 is lora_r1_all's post-re-tune NAME for
                        # the same arm (rank axis; rank-4 winner) — round-1
                        # summaries carry the legacy label and must pool
                        # with lora_r4 follow-up rounds.
                        v = ARM_CANON.get(v, v)
                    prev = meta.setdefault(k, v)
                    if prev != v:
                        raise SystemExit(
                            f"[analyze] mixed {k} across summaries "
                            f"({prev!r} vs {s[k]!r}) — one analyze run "
                            "covers ONE tuning cell")
            if s.get("box"):
                boxes.append(s["box"])
                # The design's DECLARED fixed axis: a zero-width clip side
                # [c, c] is the LR-only design's constant. It is carried
                # to the fit as a DECLARATION and every row is validated
                # against it — never inferred from float uniqueness across
                # the pool, which a drifted constant (0.34 vs
                # 0.3400000000000001) would split into two clip levels and
                # a rank-deficient 9-term design.
                rng = s["box"].get("max_grad_norm")
                fixed_here = (rng is not None
                              and hpo_selection_guard.is_fixed_axis(rng[0], rng[1]))
                clip_declared.add(bool(fixed_here))
                if fixed_here:
                    c_decl = float(rng[0])
                    prev = declared_fixed.setdefault("max_grad_norm", c_decl)
                    if prev != c_decl:
                        raise SystemExit(
                            f"[analyze] {p.parent.name} declares the clip "
                            f"constant {c_decl!r} but an earlier pooled round "
                            f"declared {prev!r} — two constants are two "
                            "designs; one analyze run covers ONE design")
                    if float(row["max_grad_norm"]) != c_decl:
                        raise SystemExit(
                            f"[analyze] {p.parent.name}: trial clip "
                            f"{row['max_grad_norm']!r} differs from its own "
                            f"design's declared constant {c_decl!r} — the "
                            "LR-only design stamps ONE constant on every "
                            "row; a row off it is a corrupt summary, not a "
                            "second clip level")
            if s.get("warmup_levels"):
                levels_seen.add(tuple(float(x) for x in s["warmup_levels"]))
    tag = "+".join(str(r) for r in rounds)
    print(f"[analyze] rounds {tag}: {len(rows)} usable trials, "
          f"{len(excluded)} excluded: {excluded}")
    baseline = None
    if base_vals and min(base_vals) > 0:
        lo_b, hi_b = min(base_vals), max(base_vals)
        if (hi_b - lo_b) / lo_b < 1e-4:
            baseline = (lo_b + hi_b) / 2
        else:
            print("[analyze] NOTE: the untrained held-out objective disagrees "
                  f"across trials ([{lo_b:.6f}, {hi_b:.6f}]) — the selection "
                  "guard files no ratio-to-baseline columns")
    if len(levels_seen) > 1:
        raise SystemExit(f"[analyze] mixed warmup level sets across "
                         f"summaries: {sorted(levels_seen)}")
    if len(clip_declared) > 1:
        raise SystemExit(
            "[analyze] some pooled summaries declare the clip as a held "
            "CONSTANT (a zero-width box side, the LR-only design) and others "
            "sweep it — one analyze run covers ONE design")
    levels = (tuple(sorted(next(iter(levels_seen))))
              if levels_seen else WARMUP_LEVELS)
    if not boxes:
        raise SystemExit("[analyze] no summary carried its design box — "
                         "cannot run the edge check; re-emit with this "
                         "harness's trial schema")
    if len(labels_seen) > 1:
        raise SystemExit(
            f"[analyze] mixed ranking objectives across summaries "
            f"({sorted(labels_seen)}) — a pool ranked by two different "
            "metrics has no single surface; re-run the odd trials")
    if len(teacher_seen) > 1:
        raise SystemExit(
            f"[analyze] mixed teacher legs across summaries "
            f"({sorted(map(str, teacher_seen))}) — a row-binding fallback "
            "trained a DIFFERENT objective (full-vocab online vs "
            "renormalized top-K head); re-run the fallback trials against "
            "a matching artifact")
    # The guard binds when temperature actually VARIES across the usable
    # rows: a constant-T pool (a pinned round, or T=1 survivors after the
    # T!=1 floor exclusions) trains ONE objective, so there is no across-T
    # comparison to protect.
    temps_seen = {float(r["temperature"]) for r in rows
                  if "temperature" in r}
    if len(temps_seen) > 1 and \
            labels_seen and labels_seen != {"final_residual_kl"}:
        raise SystemExit(
            "[analyze] a temperature-swept cell must be ranked by the T=1 "
            "probe residual (final_residual_kl) — the trainer's own-T "
            f"held-out loss is not comparable across T (got {labels_seen})")
    if rows and accum_rows not in (0, len(rows)):
        raise SystemExit(
            "[analyze] accum_steps present on some summaries and absent on "
            "others — mixed effective batches; one analyze run covers ONE "
            "tuning cell")
    fit_ba = None
    if meta.get("arm") == "bidpo" and not batch_accum:
        # derive_model_config --hpo-results REQUIRES accum_steps for bidpo.
        raise SystemExit(
            "[analyze] bidpo needs ONE consistent (batch_size, "
            "accum_steps) across the pooled ok-summaries — got none; "
            "re-run (derive_model_config --hpo-results requires the "
            "realized accum_steps)")
    if batch_accum:
        # The fitted winner only transfers verbatim if every pooled trial
        # ran the SAME realized (batch_size, accum_steps) — a mixed pool
        # has no single winner setting, so refuse rather than emitting one
        # trial's accumulation as everyone's. Applies to bidpo (the 128
        # floor) and the re-tune@32 arms alike.
        if len(batch_accum) != 1 or any(v is None
                                        for v in next(iter(batch_accum))):
            raise SystemExit(
                f"[analyze] {meta.get('arm')} needs ONE consistent "
                f"(batch_size, accum_steps) across the pooled ok-summaries "
                f"— got {sorted(batch_accum, key=repr)}; re-run the "
                "inconsistent trials")
        fit_ba = next(iter(batch_accum))
    # Union box across rounds (refine rounds shrink it; edge semantics are
    # against everything the pooled trials actually explored).
    box = {k: [min(float(b[k][0]) for b in boxes),
               max(float(b[k][1]) for b in boxes)]
           for k in ("lr", "max_grad_norm")}

    fit = fit_quadratic(rows, levels=levels, y_key=args.y_key,
                        fixed_axes=declared_fixed)
    if fit.get("fixed_axes"):
        print("[analyze] NOTE: constant axes held FIXED (excluded from "
              f"the quadratic design): {fit['fixed_axes']} — every pooled "
              "usable row carries that one value (a pinned round, upstream "
              "exclusions, or the LR-only design's clip constant), so the "
              "surface has no information along them; the winner echoes "
              "them verbatim")
    best_raw = min(rows, key=lambda r: r[args.y_key])
    raw_keys = [k for k in ("lr", "max_grad_norm", "warmup_frac",
                            "temperature", "weight_decay", "lora_rank")
                if k in best_raw]
    print("[analyze] best raw trial (REFERENCE ONLY — selection is the "
          "fitted argmin): "
          + " ".join(f"{k}={best_raw[k]:.4g}" for k in raw_keys)
          + f" {args.y_key}={best_raw[args.y_key]:.5f}")
    # Clip activation as a COLUMN and a CONSTRAINT: the fold table below
    # carries every trial's rate; a trial at or
    # over the threshold is not promotable, so the raw-argmin fall-through
    # (non-PD / edge) is drawn from the promotable trials — best_raw_trial
    # stays the unconstrained reference it always was.
    clip_act = clip_activation_summary(rows, y_key=args.y_key)
    best_raw_promotable = clip_act.pop("best_raw_promotable_row")
    fall_through_pick = (best_raw_promotable if best_raw_promotable is not None
                         else best_raw)
    _tid = hpo_selection_guard.trial_id
    print(f"[analyze] trials by {args.y_key} (clip_act = share of logged "
          "optimizer steps whose pre-clip grad norm exceeded the trial's own "
          f"clip; threshold {clip_act['threshold']:.2f}, target "
          f"{clip_act['target']:.2f}; n/a = no metrics.jsonl):")
    for r in sorted(rows, key=lambda r: float(r[args.y_key])):
        print(f"  {_tid(r):8s} lr={float(r['lr']):.3e} "
              f"clip={float(r['max_grad_norm']):.4g} "
              f"warmup={float(r['warmup_frac']):.2f} "
              f"{args.y_key}={float(r[args.y_key]):.5f} "
              f"clip_act={_fmt_rate(r.get('clip_activation_rate'))} "
              f"n_steps={int(r.get('clip_n_steps') or 0)}")
    if clip_act["over_threshold"]:
        print(f"[analyze] clip activation >= {clip_act['threshold']:.2f} on "
              f"{clip_act['over_threshold']} — not promotable; the raw "
              f"fall-through pick is {clip_act['best_raw_promotable']} "
              f"(best_raw_trial {_tid(best_raw)} stays the unconstrained "
              "reference)")
    # Per-rank raw bests: the additive-rank assumption's visibility valve —
    # if the per-rank best lr moves by decades, the main-effects-only rank
    # factor is hiding an interaction and the human should see it.
    per_rank_best = None
    if fit.get("rank_levels"):
        per_rank_best = {}
        for rl in fit["rank_levels"]:
            sub = [r for r in rows if int(r["lora_rank"]) == rl]
            if sub:
                per_rank_best[str(rl)] = min(sub,
                                             key=lambda r: r[args.y_key])
    artifact: dict = {"fit": fit, "rows": rows, "excluded": excluded,
                      "n_excluded": len(excluded), "box": box,
                      "fixed_axes": fit.get("fixed_axes", {}),
                      "clip_activation": clip_act,
                      "rounds": rounds, "y_key": args.y_key,
                      "ranking_objective": (sorted(labels_seen)[0]
                                            if labels_seen else None),
                      "teacher_forward": (sorted(map(str, teacher_seen))[0]
                                          if teacher_seen else None),
                      "best_raw_trial": best_raw,
                      "per_rank_best_raw": per_rank_best, **meta}
    out_path = Path(args.out) / f"r{tag}_fit.json"
    if fit.get("rank_deficient"):
        out_path.write_text(json.dumps(artifact, indent=2) + "\n")
        raise SystemExit(
            f"[analyze] design matrix is RANK-DEFICIENT "
            f"(rank {fit['design_rank']}/{len(fit['names'])}) — the fit "
            f"cannot be trusted; artifact at {out_path}")
    if fit["design_cond"] > MAX_DESIGN_COND:
        out_path.write_text(json.dumps(artifact, indent=2) + "\n")
        raise SystemExit(
            f"[analyze] design conditioning {fit['design_cond']:.3g} exceeds "
            f"{MAX_DESIGN_COND:.0e} — near-collinear design, the fit cannot "
            f"be trusted; artifact at {out_path}")
    if not fit["positive_definite"]:
        # No fitted minimum: a downstream fold writer falls through to the
        # RAW argmin trial. File the guard for THAT point (non-PD is itself
        # an alarm) so the artefact says so in words, then refuse as before.
        fall = {"reason": "fit_not_positive_definite"}
        if clip_act["over_threshold"]:
            fall["clip_activation_excluded"] = clip_act["over_threshold"]
        guard = hpo_selection_guard.guard_selection(
            selected=fall_through_pick, selected_from="best_raw_trial",
            rows=rows, box=box, levels=levels, fit=fit, y_key=args.y_key,
            sigma_log_y=fit.get("sigma_log_y"), baseline=baseline,
            fall_through=fall)
        _stamp_clip_activation(guard, clip_act, None)
        artifact["selection_guard"] = guard
        out_path.write_text(json.dumps(artifact, indent=2) + "\n")
        for line in guard["text"]:
            print(f"[analyze] {line}")
        raise SystemExit(
            "[analyze] the continuous Hessian is not positive definite — no "
            f"trustworthy interior minimum (eigs {fit['hessian_eigs']}); "
            f"widen or re-center the box and re-run; artifact at {out_path}")

    on_edge, edges = _edge_check(fit["argmin"], box)
    selection = {**fit["argmin"], "fitted_min_log_y": fit["fitted_min_log_y"],
                 "on_edge": on_edge, "edge_pos": edges,
                 "clamped_axes": fit.get("clamped_axes", []),
                 "selected_from": "fitted_argmin"}
    if selection["clamped_axes"]:
        # tau/D at a grid edge is informational, never a recenter: the grid
        # is the design (widening it is a deliberate scope change).
        print(f"[analyze] NOTE: fitted argmin sits at the grid edge on "
              f"{selection['clamped_axes']} (clamped)")
    if on_edge:
        # Same log-width, re-centred, at least one full round of trials:
        # 16 for the lr x clip designs; 8 for the LR-only design — its
        # round IS 8: the 5-term fit needs no more, and 16 would double
        # the configured trial count.
        floor_n = (LLAMA8B_CE_TRIALS_PER_ARM
                   if "max_grad_norm" in (fit.get("fixed_axes") or {})
                   else 16)
        nxt = {"action": "recenter", "box": _boxed(fit["argmin"], box, 1.0),
               "n": max(len(rows) // len(rounds) // 8 * 8, floor_n)}
    else:
        nxt = {"action": "refine", "box": _boxed(fit["argmin"], box,
                                                 REFINE_ZOOM),
               "n": REFINE_N}
    winner = None
    if not on_edge:
        winner = {**meta,
                  "lr": selection["lr"],
                  "max_grad_norm": selection["max_grad_norm"],
                  # The winner ships as a FRACTION — the production fit
                  # re-derives steps from its own budget, never a copied
                  # step count from a subsampled trial.
                  "warmup_frac": selection["warmup_frac"],
                  "lr_schedule": "warmup_cosine",
                  "selected_from": "fitted_argmin",
                  "rounds_pooled": rounds}
        # The expanded axes ship on the winner where swept (make_fit_configs
        # folds them into train_kwargs.<arm> next to lr/clip/warmup_frac).
        for ax in ("temperature", "weight_decay", "lora_rank"):
            if ax in selection:
                winner[ax] = selection[ax]
        if meta.get("arm") == "bidpo":
            winner["effective_batch_floor"] = EFFECTIVE_BATCH_FLOOR
        if fit_ba is not None:
            # The REALIZED accumulation (consistency enforced above) —
            # derive_model_config's translate_hpo_results refuses a bidpo
            # winner without accum_steps, a manual copy without it silently
            # defaults to 1 microbatch, and a re-tune@32 winner without it
            # would refit at a different effective batch than it was tuned.
            winner["batch_size"] = int(fit_ba[0])
            winner["accum_steps"] = int(fit_ba[1])
            winner["effective_batch"] = int(fit_ba[0]) * int(fit_ba[1])
    # The selection guard files the facts for the point a fold SHIPS — the
    # fitted argmin here, or the best-raw fall-through when the argmin sits
    # on the edge (winner None) — so the fold writer and a future reader see
    # the same verdict (boundary hits, fit health, objective spread /
    # noise / flatness) without re-deriving it.
    winner_ca = None
    if on_edge:
        fall = {"reason": "fitted_argmin_on_edge",
                "fitted_argmin": {k: selection[k] for k in
                                  ("lr", "max_grad_norm", "warmup_frac")},
                "edge_pos": edges}
        if clip_act["over_threshold"]:
            fall["clip_activation_excluded"] = clip_act["over_threshold"]
        guard = hpo_selection_guard.guard_selection(
            selected=fall_through_pick, selected_from="best_raw_trial",
            rows=rows, box=box, levels=levels, fit=fit, y_key=args.y_key,
            sigma_log_y=fit.get("sigma_log_y"), baseline=baseline,
            fall_through=fall)
    else:
        guard = hpo_selection_guard.guard_selection(
            selected=selection, selected_from="fitted_argmin", rows=rows,
            box=box, levels=levels, fit=fit, y_key=args.y_key,
            sigma_log_y=fit.get("sigma_log_y"), baseline=baseline,
            fitted_min_log_y=fit["fitted_min_log_y"])
        # The fitted argmin has no metrics of its own: its clip regime is
        # its 3 nearest trials', printed beside the winner.
        winner_ca = winner_clip_activation(selection, rows, box,
                                           fit.get("fixed_axes") or {})
        winner["clip_activation"] = winner_ca
        near = ", ".join(f"{n['trial']} rate={_fmt_rate(n['rate'])}"
                         for n in winner_ca["nearest"])
        verdict = ("FLAGGED — a nearest trial is at/over the threshold; the "
                   "confirmation trial must measure below it at the point"
                   if winner_ca["flagged"] else
                   "promotable (every nearest trial with metrics is below)"
                   if winner_ca["promotable"] else
                   "unknown (no metrics beside the nearest trials)")
        print(f"[analyze] winner clip activation — threshold "
              f"{winner_ca['threshold']:.2f} (target "
              f"{winner_ca['target']:.2f}); nearest trials: {near} -> "
              f"{verdict}")
    _stamp_clip_activation(guard, clip_act, winner_ca)
    selection["alarmed"] = guard["alarmed"]
    selection["alarms"] = [hpo_selection_guard.alarm_code(a)
                           for a in guard["alarms"]]
    artifact["selection_guard"] = guard
    for line in guard["text"]:
        print(f"[analyze] {line}")
    artifact.update({"selection": selection, "next": nxt, "winner": winner})
    out_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"selection": selection, "next": nxt}, indent=2))
    print(f"[analyze] artifact -> {out_path}")


# --------------------------------------------------------------------------
# analyze --design v2: the additive
# model  log CE = a + b u + c u^2 + beta[rank] + gamma[set]  in u = log10(lr)
# (reference rank 4 / set `wo`), an optional delta[rank] u interaction when it
# lowers AICc by > 2, the argmin over the ALLOWED pairs only, the adequacy
# REPORT (never a trial trigger), and the guard with NAMED discrete axes.
# With no architecture axes (the ReLU search) the same code is the plain
# quadratic in u.
# --------------------------------------------------------------------------

ADDITIVE_INADEQUATE_ALARM = "additive_model_inadequate"
AICC_INTERACTION_DROP = 2.0
V2_ANALYZE_DESIGN = "v2"


def _refuse_replicate_row(p: Path, s: dict) -> None:
    if is_replicate_design(s):
        raise SystemExit(
            f"[analyze] {p.parent.name} is an EXPLORATION REPLICATE (round "
            f"{s.get('round')}, replicating "
            f"{(s.get('exploration_replicate_of') or {}).get('trial')!r}) — "
            "replicate noise evidence, never a search row; drop its round "
            "from --rounds")
    if is_confirm_pick_design(s):
        raise SystemExit(
            f"[analyze] {p.parent.name} is a CONFIRM-PICK trial (round "
            f"{s.get('round')}, the user's pick at "
            f"{(s.get('confirm_pick_of') or {}).get('pair')!r}) — promotion "
            "evidence for hpo_pick, never a search row; drop its round from "
            "--rounds")


def _pool_rows_v2(out_dir: Path, rounds: list[int], y_key: str) -> dict:
    """The v2 pool: usable rows + exclusions + the design facts every
    summary carries (box, warm-up level, declared clip constant, allowed /
    excluded pairs, set order, search id, refine block). One analyze run
    covers ONE design: mixed arms / models / clip constants / warm-up
    levels refuse."""
    rows, excluded = [], []
    meta: dict = {}
    boxes: list[dict] = []
    levels: set = set()
    clip_decl: set = set()
    allowed_pairs: list | None = None
    excluded_pairs: list | None = None
    set_order: list | None = None
    lr_box: list | None = None
    search_ids: set = set()
    refine_blocks: list = []
    base_vals: list[float] = []
    designs: set = set()
    recon: dict | None = None
    campaign: str | None = None
    opening: dict | None = None
    n_seen = 0
    for r in rounds:
        for p in sorted(out_dir.glob(f"r{r}_t*/summary.json")):
            s = json.loads(p.read_text())
            n_seen += 1
            if is_confirmation_design(s):
                raise SystemExit(
                    f"[analyze] {p.parent.name} is a CONFIRMATION trial — "
                    "never a search row; drop its round from --rounds")
            _refuse_replicate_row(p, s)
            y = s.get(y_key)
            reason = None
            if s.get("status") != "ok":
                reason = s.get("status", "missing_status")
            elif y is None or not math.isfinite(float(y)):
                reason = "nonfinite_objective"
            elif s.get("floor_ok") is False:
                reason = "floor_failed"
            if reason is not None:
                excluded.append((p.parent.name, reason))
                continue
            row = {**s["trial"], y_key: float(y), "round": r}
            row["clip_activation_rate"], row["clip_n_steps"] = \
                _trial_clip_activation(p.parent, s)
            row["trainable_params"] = s.get("trainable_params")
            if row.get("lora_rank") is not None and not row.get("lora_set_id"):
                row["lora_set_id"] = _pair_of(row)[1]
            rows.append(row)
            if s.get("heldout_curve"):
                base_vals.append(float(s["heldout_curve"][0]))
            for k in ("arm", "model", "position", "si"):
                if k in s:
                    v = ARM_CANON.get(s[k], s[k]) if k == "arm" else s[k]
                    prev = meta.setdefault(k, v)
                    if prev != v:
                        raise SystemExit(f"[analyze] mixed {k} across summaries "
                                         f"({prev!r} vs {v!r}) — one analyze "
                                         "run covers ONE tuning cell")
            if s.get("box"):
                boxes.append(s["box"])
                rng = s["box"].get("max_grad_norm")
                if rng is None or not hpo_selection_guard.is_fixed_axis(rng[0], rng[1]):
                    raise SystemExit(
                        f"[analyze] {p.parent.name} does not declare a FIXED "
                        f"clip (box max_grad_norm {rng!r}) — the v2 designs "
                        "hold the recon constant; this is not a v2 summary")
                clip_decl.add(float(rng[0]))
                if float(row["max_grad_norm"]) != float(rng[0]):
                    raise SystemExit(
                        f"[analyze] {p.parent.name}: trial clip "
                        f"{row['max_grad_norm']!r} differs from its design's "
                        f"declared constant {rng[0]!r}")
            if s.get("warmup_levels"):
                levels.update(float(x) for x in s["warmup_levels"])
            if s.get("allowed_pairs") is not None:
                ap = [list(pair_key(a, b)) for a, b in s["allowed_pairs"]]
                if allowed_pairs is not None and ap != allowed_pairs:
                    raise SystemExit("[analyze] the pooled summaries declare "
                                     "DIFFERENT allowed-pair menus — one "
                                     "analyze run covers ONE design space")
                allowed_pairs = ap
                excluded_pairs = [[int(a), str(b), str(c)]
                                  for a, b, c in (s.get("excluded_pairs") or [])]
            if s.get("set_order"):
                set_order = list(s["set_order"])
            if s.get("lr_log10_box") and not s.get("refine_of"):
                lr_box = [float(v) for v in s["lr_log10_box"]]
            if s.get("search_id"):
                search_ids.add(str(s["search_id"]))
            if s.get("refine_of"):
                refine_blocks.append({**s["refine_of"], "box": s.get("box"),
                                      "lr_log10_box": s.get("lr_log10_box")})
            if s.get("design"):
                designs.add(str(s["design"]))
            if s.get("recon"):
                recon = s["recon"]
            if s.get("campaign"):
                campaign = str(s["campaign"])
            if s.get("search_opening"):
                opening = s["search_opening"]
    if not boxes:
        raise SystemExit(f"[analyze] no usable summary under {out_dir} for "
                         f"rounds {rounds} ({n_seen} seen, {len(excluded)} "
                         "excluded) — nothing to fit")
    if len(clip_decl) > 1:
        raise SystemExit(f"[analyze] two clip constants {sorted(clip_decl)} "
                         "across the pooled summaries — two designs")
    if len(levels) != 1:
        raise SystemExit(f"[analyze] the v2 designs pin warm-up to ONE level; "
                         f"the pool declares {sorted(levels)}")
    if len(search_ids) > 1:
        raise SystemExit(f"[analyze] the pooled summaries name {len(search_ids)} "
                         f"search ids {sorted(search_ids)} — one analyze run "
                         "covers ONE search")
    baseline = None
    if base_vals and min(base_vals) > 0:
        lo_b, hi_b = min(base_vals), max(base_vals)
        baseline = (lo_b + hi_b) / 2 if (hi_b - lo_b) / lo_b < 1e-4 else None
    union_lr = [min(float(b["lr"][0]) for b in boxes),
                max(float(b["lr"][1]) for b in boxes)]
    return {"rows": rows, "excluded": excluded, "meta": meta,
            "clip": next(iter(clip_decl)), "warmup_frac": next(iter(levels)),
            "allowed_pairs": allowed_pairs, "excluded_pairs": excluded_pairs,
            "set_order": set_order, "lr_log10_box": lr_box,
            "union_lr_box": union_lr, "search_id": (next(iter(search_ids))
                                                     if search_ids else None),
            "refine_blocks": refine_blocks, "baseline": baseline,
            "designs": sorted(designs), "recon": recon,
            "campaign": campaign, "search_opening": opening}


def _v2_levels(rows: list[dict], keys: list,
               set_order: list | None) -> tuple[list[int], list[str], str, str]:
    """(ranks ascending, sets in ordinal order, reference rank, reference
    set) for the additive model — the levels of the design (``keys`` =
    _design_keys). Reference rank 4 / set ``wo`` when present, else the
    lowest level."""
    ranks = sorted({int(k[0]) for k in keys})
    sets_present = sorted({str(k[1]) for k in keys})
    if set_order:
        sets = [sid for sid in set_order if sid in sets_present]
        sets += [sid for sid in sets_present if sid not in sets]
    else:
        sets = sets_in_width_order(
            sets_present, per_rank_counts((k[0], k[1], r.get("trainable_params"))
                                          for r in rows for k in [_pair_of(r)]
                                          if k is not None),
            who="analyze")
    ref_rank = 4 if 4 in ranks else (ranks[0] if ranks else None)
    ref_set = "wo" if "wo" in sets else (sets[0] if sets else None)
    return ranks, sets, ref_rank, ref_set


def _v2_names(ranks, sets, ref_rank, ref_set, interaction: bool) -> list[str]:
    names = ["1", "u", "u*u"]
    names += [f"rank={r}" for r in ranks if r != ref_rank]
    names += [f"set={sid}" for sid in sets if sid != ref_set]
    if interaction:
        names += [f"rank={r}*u" for r in ranks if r != ref_rank]
    return names


def _v2_feats(u: float, rank, sid, ranks, sets, ref_rank, ref_set,
              interaction: bool) -> list[float]:
    f = [1.0, u, u * u]
    f += [1.0 if (rank is not None and int(rank) == r) else 0.0
          for r in ranks if r != ref_rank]
    f += [1.0 if sid == s_ else 0.0 for s_ in sets if s_ != ref_set]
    if interaction:
        f += [u if (rank is not None and int(rank) == r) else 0.0
              for r in ranks if r != ref_rank]
    return f


def _u_of(row: dict) -> float:
    return math.log10(float(row["lr"]))


def _design_keys(rows: list[dict], allowed_keys: list | None) -> list:
    """The (rank, set) levels of the design: the allowed pairs when known,
    else the pairs the rows carry."""
    return (list(allowed_keys) if allowed_keys
            else sorted({k for k in (_pair_of(r) for r in rows) if k is not None}))


def _ls(A: list[list[float]], y: list[float],
        checked: tuple[int, float] | None = None) -> dict | None:
    """Least squares with the rank / conditioning preflight (``checked`` =
    a preflight the caller already ran); None when the design is
    rank-deficient (the caller decides what that means)."""
    if len(A) < len(A[0]):
        return None
    rank, cond = checked if checked is not None else check_design(A)
    if rank < len(A[0]):
        return None
    coef = _lstsq_householder(A, y)
    resid = [yi - _dot(ai, coef) for ai, yi in zip(A, y)]
    rss = math.fsum(r * r for r in resid)
    return {"coef": coef, "resid": resid, "rss": rss, "rank": rank, "cond": cond}


def _aicc(rss: float, n: int, k: int) -> float | None:
    if n - k - 1 <= 0 or n <= 0:
        return None
    rss = max(rss, 1e-300)
    return n * math.log(rss / n) + 2 * k + 2 * k * (k + 1) / (n - k - 1)


def fit_additive_v2(rows: list[dict], *, y_key: str, lr_log10_box, allowed_keys,
                    set_order=None) -> dict:
    """The v2 response model. Returns the fit dict the
    artefact carries: names / coef / design rank + cond / residual scale
    (sigma_log_y ...) / positive_definite (c > 0) / hessian_eigs ([2c]) /
    u_star / the level sets and reference levels / rank_effects (beta) /
    set_effects (gamma) / the interaction assessment (AICc of the additive
    vs the delta[rank]*u model; selected when the drop exceeds
    AICC_INTERACTION_DROP) / pair_predictions at u* (u*_rank under the
    interaction) over the ALLOWED pairs / the argmin pair and its
    fitted_min_log_y / interior. Rank-deficient -> ``rank_deficient`` and
    nothing else."""
    lo, hi = (float(v) for v in lr_log10_box)
    keys = _design_keys(rows, allowed_keys)
    ranks, sets, ref_rank, ref_set = _v2_levels(rows, keys, set_order)
    y = [math.log(max(float(r[y_key]), 1e-12)) for r in rows]

    def design(interaction: bool):
        return [_v2_feats(_u_of(r), r.get("lora_rank"), r.get("lora_set_id"),
                          ranks, sets, ref_rank, ref_set, interaction)
                for r in rows]

    names = _v2_names(ranks, sets, ref_rank, ref_set, False)
    out: dict = {"design": V2_ANALYZE_DESIGN, "names": names, "n_rows": len(rows),
                 "levels": [], "cont_axes": ["lr"], "u_axis": "log10(lr)",
                 "lr_log10_box": [lo, hi], "ranks": ranks, "sets": sets,
                 "ref_rank": ref_rank, "ref_set": ref_set,
                 "rank_levels": ranks if len(ranks) > 1 else [],
                 "set_levels": sets if len(sets) > 1 else []}
    A = design(False)
    if len(rows) < len(names):
        raise SystemExit(f"[analyze] too few usable trials ({len(rows)}) for "
                         f"the {len(names)}-term additive model {names} — "
                         "investigate the exclusions")
    rank, cond = check_design(A)
    out["design_rank"], out["design_cond"] = rank, cond
    if rank < len(names):
        out["rank_deficient"] = True
        return out
    ls = _ls(A, y, checked=(rank, cond))
    coef = ls["coef"]
    c = dict(zip(names, coef))
    out["coef"] = list(coef)
    out["coef_by_name"] = {k: float(v) for k, v in c.items()}
    out["resid_log_y"] = ls["resid"]
    out.update(hpo_selection_guard.residual_scale(ls["resid"], n_params=len(names)))
    out["rank_effects"] = {str(r): (0.0 if r == ref_rank else float(c[f"rank={r}"]))
                           for r in ranks}
    out["set_effects"] = {sid: (0.0 if sid == ref_set else float(c[f"set={sid}"]))
                          for sid in sets}
    # -- the optional delta[rank] * u interaction (reported; selected by AICc)
    inter: dict = {"assessable": False, "selected": False, "delta": None,
                   "aicc_additive": _aicc(ls["rss"], len(rows), len(names)),
                   "aicc_interaction": None, "drop": None,
                   "threshold": AICC_INTERACTION_DROP}
    names_i = _v2_names(ranks, sets, ref_rank, ref_set, True)
    if len(ranks) > 1 and len(rows) > len(names_i) + 1:
        ls_i = _ls(design(True), y)
        if ls_i is not None:
            inter["assessable"] = True
            inter["aicc_interaction"] = _aicc(ls_i["rss"], len(rows), len(names_i))
            if inter["aicc_additive"] is not None and inter["aicc_interaction"] is not None:
                inter["drop"] = inter["aicc_additive"] - inter["aicc_interaction"]
                inter["selected"] = inter["drop"] > AICC_INTERACTION_DROP
            ci = dict(zip(names_i, ls_i["coef"]))
            inter["delta"] = {str(r): (0.0 if r == ref_rank else float(ci[f"rank={r}*u"]))
                              for r in ranks}
            inter["coef_by_name"] = {k: float(v) for k, v in ci.items()}
            inter["c_u2"] = float(ci["u*u"])
            if inter["selected"]:
                inter["note"] = ("the delta[rank]*u interaction lowers AICc by "
                                 f"{inter['drop']:.2f} > {AICC_INTERACTION_DROP}: "
                                 "the lr optimum depends on the rank; the "
                                 "per-pair u* below is rank-specific")
    out["interaction"] = inter
    # -- curvature, argmin per pair, the argmin over the allowed pairs -------
    use_i = inter["selected"]
    coefs = inter["coef_by_name"] if use_i else out["coef_by_name"]
    b, cq = float(coefs["u"]), float(coefs["u*u"])
    pd = cq > 0
    out["hessian_eigs"] = [2.0 * cq]
    out["positive_definite"] = pd
    out["interaction_used"] = use_i
    preds: dict = {}
    if not keys:
        keys = [None]  # the ReLU search: one "pair"
    for key in keys:
        r_, s_ = (key if key is not None else (None, None))
        off = 0.0
        if r_ is not None:
            off += 0.0 if r_ == ref_rank else float(coefs[f"rank={r_}"])
            off += 0.0 if s_ == ref_set else float(coefs[f"set={s_}"])
        b_r = b + (float(coefs[f"rank={r_}*u"]) if (use_i and r_ is not None
                                                     and r_ != ref_rank) else 0.0)
        u_star = (-b_r / (2.0 * cq)) if pd else None
        pred = (float(coefs["1"]) + b_r * u_star + cq * u_star * u_star + off
                if u_star is not None else None)
        t = ((u_star - lo) / (hi - lo)) if u_star is not None else None
        interior = (t is not None and EDGE_TOL < t < 1.0 - EDGE_TOL)
        label = f"{r_}:{s_}" if key is not None else "all"
        preds[label] = {"rank": r_, "set": s_, "u_star": u_star,
                        "lr": (10.0 ** u_star if u_star is not None else None),
                        "b": b_r, "offset": off, "pred_log_y": pred,
                        "pred_y": (math.exp(pred) if pred is not None else None),
                        "box_pos": t, "interior": interior,
                        "n_rows": sum(1 for r in rows if _pair_of(r) == key)}
    out["pair_predictions"] = preds
    if pd:
        win_label = min((k for k, v in preds.items() if v["pred_log_y"] is not None),
                        key=lambda k: (preds[k]["pred_log_y"], k))
        w = preds[win_label]
        out["argmin_pair"] = ([w["rank"], w["set"]] if w["rank"] is not None
                              else None)
        out["u_star"] = w["u_star"]
        out["argmin"] = {"lr": w["lr"]}
        if w["rank"] is not None:
            out["argmin"].update(lora_rank=int(w["rank"]), lora_set_id=w["set"])
        out["fitted_min_log_y"] = w["pred_log_y"]
        out["argmin_box_pos"] = w["box_pos"]
        out["argmin_interior"] = w["interior"]
    return out


def _loo_rmse_v2(rows: list[dict], *, y_key: str, ranks, sets, ref_rank,
                 ref_set, categorical: bool) -> dict:
    """Leave-one-out RMSE in OBJECTIVE units (held-out CE) of the additive
    model (``categorical=True``) or the nested lr-only quadratic. A fold
    whose held-out point is the ONLY observation of a rank or set level is
    unpredictable by the additive model and is reported as
    ``unidentifiable_fold`` (the lr-only model is defined on every fold)."""
    ys = [float(r[y_key]) for r in rows]
    errs: list[float] = []
    folds: list[dict] = []
    unident: list[str] = []
    rk, st = (ranks, sets) if categorical else ([], [])
    pairs = [_pair_of(r) for r in rows]
    for i, held in enumerate(rows):
        train = [r for j, r in enumerate(rows) if j != i]
        tid = hpo_selection_guard.trial_id(held)
        if categorical and pairs[i] is not None:
            hr, hs = pairs[i]
            others = [p for j, p in enumerate(pairs) if j != i and p is not None]
            if not any(p[0] == hr for p in others) or \
                    not any(p[1] == hs for p in others):
                unident.append(tid)
                folds.append({"trial": tid, "unidentifiable_fold": True})
                continue
        A = [_v2_feats(_u_of(r), r.get("lora_rank"), r.get("lora_set_id"),
                       rk, st, ref_rank, ref_set, False) for r in train]
        ytr = [math.log(max(float(r[y_key]), 1e-12)) for r in train]
        ls = _ls(A, ytr)
        if ls is None:
            unident.append(tid)
            folds.append({"trial": tid, "unidentifiable_fold": True,
                          "reason": "rank_deficient_without_the_point"})
            continue
        f = _v2_feats(_u_of(held), held.get("lora_rank"), held.get("lora_set_id"),
                      rk, st, ref_rank, ref_set, False)
        pred = math.exp(_dot(f, ls["coef"]))
        errs.append(pred - ys[i])
        folds.append({"trial": tid, "y": ys[i], "pred": pred, "err": pred - ys[i]})
    rmse = math.sqrt(math.fsum(e * e for e in errs) / len(errs)) if errs else None
    return {"rmse": rmse, "n_folds": len(rows), "n_defined": len(errs),
            "unidentifiable_folds": unident, "folds": folds}


def _means_model_loo(rows: list[dict], *, y_key: str) -> dict:
    """The pairs-only means model's LOO RMSE — printed ONLY when every pair
    holds >= 2 points (else the stated line; review D P1: with 16 trials
    over <= 15 pairs, all but one pair hold ONE point and its LOO prediction
    is undefined on 14-15 of 16 folds)."""
    by: dict = {}
    for r in rows:
        by.setdefault(_pair_of(r), []).append(float(r[y_key]))
    n_pairs = len(by)
    if any(len(v) < 2 for v in by.values()):
        return {"defined": False, "rmse": None, "n_pairs": n_pairs,
                "line": (f"pair-specific effects unidentifiable at n={len(rows)} "
                         f"over |P|={n_pairs} pairs")}
    errs = []
    for r in rows:
        others = list(by[_pair_of(r)])
        others.remove(float(r[y_key]))
        errs.append(sum(others) / len(others) - float(r[y_key]))
    return {"defined": True, "n_pairs": n_pairs,
            "rmse": math.sqrt(math.fsum(e * e for e in errs) / len(errs)),
            "line": None}


def _adequacy_v2(rows: list[dict], fit: dict, *, y_key: str) -> dict:
    """The model-adequacy REPORT: the
    additive model's LOO RMSE vs the nested lr-only quadratic's LOO RMSE
    (both in held-out-CE units), the fit's own sigma_log_y (labelled as the
    fit residual it is), the design's rank / condition preflight, the
    means-model line, and the ``additive_model_inadequate`` verdict
    (additive LOO RMSE >= lr-only LOO RMSE, over the folds where the
    additive prediction is defined). Never a trial trigger."""
    ranks, sets = fit.get("ranks") or [], fit.get("sets") or []
    categorical = bool(fit.get("rank_levels") or fit.get("set_levels"))
    add = _loo_rmse_v2(rows, y_key=y_key, ranks=ranks, sets=sets,
                       ref_rank=fit.get("ref_rank"), ref_set=fit.get("ref_set"),
                       categorical=categorical)
    lr_only = _loo_rmse_v2(rows, y_key=y_key, ranks=[], sets=[],
                           ref_rank=None, ref_set=None, categorical=False)
    # the comparator on the SAME folds (identical when every fold is
    # identifiable, which the coverage design guarantees)
    defined = {f["trial"] for f in add["folds"] if "err" in f}
    lr_same = [f["err"] for f in lr_only["folds"] if f["trial"] in defined]
    lr_same_rmse = (math.sqrt(math.fsum(e * e for e in lr_same) / len(lr_same))
                    if lr_same else None)
    means = _means_model_loo(rows, y_key=y_key) if categorical else \
        {"defined": False, "rmse": None, "n_pairs": 0, "line": "no pairs (lr-only design)"}
    inadequate = (categorical and add["rmse"] is not None
                  and lr_same_rmse is not None and add["rmse"] >= lr_same_rmse)
    rep = {
        "objective_units": y_key,
        "additive_loo_rmse": add["rmse"],
        "additive_loo_n_defined": add["n_defined"],
        "unidentifiable_folds": add["unidentifiable_folds"],
        "lr_only_loo_rmse": lr_only["rmse"],
        "lr_only_loo_rmse_same_folds": lr_same_rmse,
        "lr_only_loo_n_defined": lr_only["n_defined"],
        "sigma_log_y": fit.get("sigma_log_y"),
        "sigma_note": ("sigma_log_y is the additive fit's own dof-corrected "
                       "residual scale in log y — the fit residual, not an "
                       "out-of-sample error"),
        "design_rank": fit.get("design_rank"), "design_cond": fit.get("design_cond"),
        "means_model": means,
        "categorical": categorical,
        "alarm": ADDITIVE_INADEQUATE_ALARM if inadequate else None,
        "rule": ("additive_model_inadequate fires when the additive model's "
                 "leave-one-out RMSE (held-out CE units) is >= the nested "
                 "lr-only quadratic's LOO RMSE on the same folds: the rank / "
                 "set effects do not generalise out of sample; every fitted "
                 "beta+gamma prediction is then model-dependent and the "
                 "user's pick, not more trials, resolves it"),
    }
    lines = []
    if categorical:
        lines.append(f"ADEQUACY: additive LOO RMSE {_fmt_num(add['rmse'])} "
                     f"({add['n_defined']}/{len(rows)} folds"
                     + (f"; unidentifiable {add['unidentifiable_folds']}"
                        if add["unidentifiable_folds"] else "")
                     + f") vs lr-only quadratic LOO RMSE {_fmt_num(lr_same_rmse)} "
                     f"(same folds; {_fmt_num(lr_only['rmse'])} on all "
                     f"{lr_only['n_defined']}); fit sigma_log_y "
                     f"{_fmt_num(fit.get('sigma_log_y'))} (the fit residual); "
                     f"design rank {fit.get('design_rank')} cond "
                     f"{_fmt_num(fit.get('design_cond'))}")
        lines.append(f"MEANS MODEL: {means['line']}" if not means["defined"]
                     else f"MEANS MODEL: pair-means LOO RMSE {_fmt_num(means['rmse'])} "
                          f"over {means['n_pairs']} pairs (every pair >= 2 points)")
        if inadequate:
            lines.append(f"ALARM {ADDITIVE_INADEQUATE_ALARM}: the additive LOO "
                         f"RMSE {_fmt_num(add['rmse'])} >= the lr-only LOO RMSE "
                         f"{_fmt_num(lr_same_rmse)} — the categorical terms do "
                         "not generalise; every fitted beta+gamma is "
                         "model-dependent")
    else:
        lines.append(f"ADEQUACY: quadratic-in-u LOO RMSE {_fmt_num(lr_only['rmse'])} "
                     f"({lr_only['n_defined']}/{len(rows)} folds); fit sigma_log_y "
                     f"{_fmt_num(fit.get('sigma_log_y'))} (the fit residual); "
                     f"design rank {fit.get('design_rank')} cond "
                     f"{_fmt_num(fit.get('design_cond'))}")
    rep["text"] = lines
    return rep


def _fmt_num(v, spec=".4g") -> str:
    return "n/a" if v is None else format(float(v), spec)


def _followup_evidence(out_dir: Path, fit: dict, y_key: str) -> dict:
    """What the fourth trial says, when it has run: the round-9 CONFIRMATION
    (confirmation_within_noise against the refit's sigma_log_y: |log y_conf
    - fitted_min_log_y| <= sigma_log_y) or the round-3 EXPLORATION
    REPLICATE (its CE difference from the original = the observed replicate
    noise). Both NULL / absent until the trial exists."""
    ev: dict = {"confirmation_trial_id": None, "confirmation_within_noise": None,
                "confirmation": None, "exploration_replicate": None}
    conf = out_dir / f"r{CONFIRMATION_ROUND}_t00" / "summary.json"
    if conf.is_file():
        s = json.loads(conf.read_text())
        y = s.get(y_key)
        rec = {"trial_id": conf.parent.name, "status": s.get("status"), "y": y,
               "confirmation_of": s.get("confirmation_of")}
        if s.get("status") == "ok" and y is not None and math.isfinite(float(y)) \
                and fit.get("fitted_min_log_y") is not None:
            gap = math.log(max(float(y), 1e-12)) - float(fit["fitted_min_log_y"])
            sig = fit.get("sigma_log_y")
            rec["gap_log_y"] = gap
            rec["within_noise"] = (abs(gap) <= float(sig)) if sig is not None else None
            ev["confirmation_trial_id"] = conf.parent.name
            ev["confirmation_within_noise"] = rec["within_noise"]
        ev["confirmation"] = rec
    rep = out_dir / f"r{EXPLORATION_ROUND}_t00" / "summary.json"
    if rep.is_file():
        s = json.loads(rep.read_text())
        y = s.get(y_key)
        ro = s.get("exploration_replicate_of") or {}
        rec = {"trial_id": rep.parent.name, "status": s.get("status"), "y": y,
               "replicates": ro.get("trial"), "original_y": ro.get("original_y"),
               "trial_role": s.get("trial_role"),
               "replicate_noise": (float(y) - float(ro["original_y"])
                                   if (y is not None and ro.get("original_y") is not None
                                       and s.get("status") == "ok") else None),
               "note": "replicate noise (CE difference), NOT a confirmation"}
        ev["exploration_replicate"] = rec
    return ev


def _refine_next(u_c: float, pair, note: str | None = None) -> dict:
    """The ``next`` block of a design-round artefact: the three refine
    trials at ``pair`` on +-REFINE_HALF_DECADES around ``u_c``."""
    nxt = {"action": "refine", "round": REFINE_ROUND,
           "pair": list(pair) if pair else None, "center_log10": u_c,
           "box_log10": [u_c - REFINE_HALF_DECADES, u_c + REFINE_HALF_DECADES],
           "n": REFINE_TRIALS}
    if note is not None:
        nxt["note"] = note
    return nxt


def analyze_v2(args) -> None:
    rounds = [int(r) for r in str(args.rounds).split(",") if r.strip()]
    out_dir = Path(args.out)
    y_key = args.y_key
    pool = _pool_rows_v2(out_dir, rounds, y_key)
    rows, excluded, meta = pool["rows"], pool["excluded"], pool["meta"]
    tag = "+".join(str(r) for r in rounds)
    print(f"[analyze v2] rounds {tag}: {len(rows)} usable trials, "
          f"{len(excluded)} excluded: {excluded}")
    # -- the design space P: --allowed-pairs, else the summaries' own menu --
    allowed = None
    if getattr(args, "allowed_pairs", None):
        allowed = load_allowed_pairs(args.allowed_pairs)
        keys = allowed_pair_keys(allowed)
        if pool["allowed_pairs"] is not None and \
                sorted(map(tuple, pool["allowed_pairs"])) != sorted(keys):
            drawn = {tuple(k) for k in pool["allowed_pairs"]}
            if getattr(args, "extends_menu", False) and drawn <= set(keys):
                # the re-verdict WIDENED the menu after the design was drawn — the refit over the superset is the
                # addendum's lr source (pair_predictions for the new pairs); the extension is recorded, never silent
                meta["menu_extended_from"] = [list(k) for k in pool["allowed_pairs"]]
                meta["menu_extension"] = [list(k) for k in keys if tuple(k) not in drawn]
                print(f"[analyze v2] --extends-menu: the design was drawn over {len(drawn)} pairs; the refit's menu adds "
                      f"{meta['menu_extension']} (pair predictions over the widened menu)")
            else:
                raise SystemExit(
                    "[analyze] --allowed-pairs disagrees with the menu the design "
                    f"was drawn from ({sorted(keys)} vs "
                    f"{sorted(map(tuple, pool['allowed_pairs']))}) — one search, "
                    "one design space" + ("" if drawn <= set(keys) else "; not a superset either")
                    + (" (pass --extends-menu for a re-verdict that widened the menu)" if drawn <= set(keys) else ""))
        excluded_pairs = [[e["rank"], e["set"], e["reason"]] for e in allowed["excluded"]]
        set_order = set_order_for_pairs(allowed)
    elif pool["allowed_pairs"] is not None:
        keys = [tuple(k) for k in pool["allowed_pairs"]]
        excluded_pairs = pool["excluded_pairs"] or []
        set_order = pool["set_order"]
    else:
        keys, excluded_pairs, set_order = None, [], None
    lora_rows = [r for r in rows if _pair_of(r) is not None]
    if keys is None and lora_rows:
        raise SystemExit("[analyze] the rows carry LoRA architecture axes but "
                         "no allowed-pair menu is known (neither the summaries "
                         "nor --allowed-pairs) — the argmin must be taken over "
                         "P, refusing to infer P from the trials that ran")
    if keys is not None:
        allowed_set = set(keys)
        for r in rows:
            k = _pair_of(r)
            if k is None:
                raise SystemExit(f"[analyze] {hpo_selection_guard.trial_id(r)} "
                                 "carries no (rank, set) on a design with an "
                                 "allowed-pair menu — a corrupt row")
            if k not in allowed_set:
                reason = next((e[2] for e in excluded_pairs
                               if pair_key(e[0], e[1]) == k), "never probed")
                raise SystemExit(
                    f"[analyze] {hpo_selection_guard.trial_id(r)} trained pair "
                    f"{k}, which is OUTSIDE the allowed pairs ({reason}) — a "
                    "trial outside P is a defect, never a data point; refusing")
    # -- the refine refit: pair-only rows, the REFINED box -----------------------
    pair_only = bool(getattr(args, "pair_only", False))
    refit_pair = None
    lr_box = pool["lr_log10_box"] or [math.log10(pool["union_lr_box"][0]),
                                      math.log10(pool["union_lr_box"][1])]
    if pair_only:
        if not pool["refine_blocks"]:
            raise SystemExit("[analyze] --pair-only needs the refine round's "
                             "summaries (their refine_of block names the pair "
                             "and the refined box) among --rounds")
        rb = pool["refine_blocks"][0]
        if rb.get("pair"):
            refit_pair = pair_key(rb["pair"]["lora_rank"],
                                  rb["pair"].get("lora_set_id")
                                  or set_id_of(rb["pair"]["lora_modules"]))
            if keys is not None and refit_pair not in set(keys):
                raise SystemExit(f"[analyze] the refine round's pair {refit_pair} "
                                 "is outside the allowed pairs — refusing")
            rows = [r for r in rows if _pair_of(r) == refit_pair]
            excluded = excluded + [(hpo_selection_guard.trial_id(r), "other_pair")
                                   for r in pool["rows"] if _pair_of(r) != refit_pair]
        lr_box = [float(v) for v in (rb.get("lr_log10_box") or rb["box_log10"])]
        print(f"[analyze v2] --pair-only: {len(rows)} rows at pair {refit_pair} "
              f"(design + refine); judging the argmin against the REFINED box "
              f"log10(lr) in [{lr_box[0]:.3f}, {lr_box[1]:.3f}]")
    if not rows:
        raise SystemExit("[analyze] no usable rows after the pair filter")
    fit_keys = (None if pair_only else keys)
    fit = fit_additive_v2(rows, y_key=y_key, lr_log10_box=lr_box,
                          allowed_keys=fit_keys, set_order=set_order)
    clip = pool["clip"]
    wf = pool["warmup_frac"]
    box = {"lr": [10.0 ** lr_box[0], 10.0 ** lr_box[1]],
           "max_grad_norm": [clip, clip]}
    fit["levels"] = [wf]
    fit["fixed_axes"] = {"max_grad_norm": clip}
    best_raw = min(rows, key=lambda r: float(r[y_key]))
    clip_act = clip_activation_summary(rows, y_key=y_key)
    best_raw_promotable = clip_act.pop("best_raw_promotable_row")
    fall_pick = best_raw_promotable if best_raw_promotable is not None else best_raw
    per_pair_best = {}
    for r in rows:
        k = _pair_of(r)
        lab = f"{k[0]}:{k[1]}" if k else "all"
        if lab not in per_pair_best or float(r[y_key]) < float(per_pair_best[lab][y_key]):
            per_pair_best[lab] = r
    _tid = hpo_selection_guard.trial_id
    print(f"[analyze v2] trials by {y_key} (clip_act threshold "
          f"{clip_act['threshold']:.2f}):")
    for r in sorted(rows, key=lambda r: float(r[y_key])):
        k = _pair_of(r)
        arch = f" rank={k[0]} set={k[1]}" if k else ""
        print(f"  {_tid(r):8s} lr={float(r['lr']):.3e}{arch} "
              f"{y_key}={float(r[y_key]):.5f} "
              f"clip_act={_fmt_rate(r.get('clip_activation_rate'))} "
              f"params={r.get('trainable_params')}")
    artifact: dict = {
        "design": V2_ANALYZE_DESIGN, "fit": fit, "rows": rows,
        "excluded": excluded, "n_excluded": len(excluded), "box": box,
        "lr_log10_box": lr_box, "fixed_axes": fit["fixed_axes"],
        "warmup_frac": wf, "clip_activation": clip_act,
        "rounds": rounds, "y_key": y_key, "pair_only": pair_only,
        "refit_pair": list(refit_pair) if refit_pair else None,
        "allowed_pairs": [list(k) for k in keys] if keys is not None else None,
        "excluded_pairs": excluded_pairs, "set_order": set_order,
        "best_raw_trial": best_raw, "per_pair_best_raw": per_pair_best,
        "search_id": pool["search_id"], "designs_pooled": pool["designs"],
        "campaign": pool["campaign"], "search_opening": pool["search_opening"],
        "recon": pool["recon"], "init_scheme": (rows[0].get("init_scheme")
                                                if rows else None),
        **meta,
    }
    # --fit-name: a DISTINCT artefact path — the widened-menu refit never touches the canonical r1_fit.json
    # the round-1 selection and the pick were made from, on ANY exit (the non-PD verdict writes its artefact and exits nonzero)
    fit_name = str(getattr(args, "fit_name", None) or "").strip()
    out_path = out_dir / (fit_name if fit_name else f"r{tag}_fit{'_pair' if pair_only else ''}.json")
    if fit_name:
        artifact["fit_name"] = fit_name
    # A derived refit is BOUND to what it was derived from — the menu file's identity (sha + the probe's result generation
    # + job) and the source record (cluster:path, --source-record) — so a residual refit from another cluster / menu generation on a shared
    # mirror can never be taken for this search's (hpo_table refuses the mismatch by name)
    if allowed is not None:
        artifact["menu_source"] = {"path": allowed.get("path"), "sha256": allowed.get("sha256"), "model_id": allowed.get("model_id"),
                                   "probe_job_id": allowed.get("probe_job_id"), "result_generation": allowed.get("result_generation")}
    src_rec = str(getattr(args, "source_record", None) or "").strip()
    if src_rec:
        artifact["source_record"] = src_rec
    artifact["row_identity"] = sorted((hpo_selection_guard.trial_id(r), float(r[y_key])) for r in rows)
    if fit.get("rank_deficient"):
        out_path.write_text(json.dumps(artifact, indent=2) + "\n")
        raise SystemExit(f"[analyze] the additive design is RANK-DEFICIENT "
                         f"(rank {fit['design_rank']}/{len(fit['names'])}) — "
                         f"artifact at {out_path}")
    if fit["design_cond"] > MAX_DESIGN_COND:
        out_path.write_text(json.dumps(artifact, indent=2) + "\n")
        raise SystemExit(f"[analyze] design conditioning {fit['design_cond']:.3g} "
                         f"exceeds {MAX_DESIGN_COND:.0e}; artifact at {out_path}")
    adequacy = _adequacy_v2(rows, fit, y_key=y_key)
    artifact["adequacy"] = adequacy
    for line in adequacy["text"]:
        print(f"[analyze v2] {line}")
    if fit["interaction"].get("assessable"):
        it = fit["interaction"]
        print(f"[analyze v2] INTERACTION delta[rank]*u: AICc additive "
              f"{_fmt_num(it['aicc_additive'])} vs interaction "
              f"{_fmt_num(it['aicc_interaction'])} (drop {_fmt_num(it['drop'])}, "
              f"threshold {it['threshold']}) -> "
              f"{'SELECTED — the argmin depends on rank' if it['selected'] else 'not selected'}")
    # -- named discrete axes for the guard (the FULL design menu) --------------
    discrete_axes = None
    if keys is not None and (fit.get("rank_levels") or fit.get("set_levels")
                             or pair_only):
        ranks_menu = sorted({int(k[0]) for k in keys})
        sets_menu = list(set_order or fit["sets"])
        sets_menu = [sid for sid in sets_menu if sid in {k[1] for k in keys}]
        discrete_axes = {"lora_rank": {"levels": ranks_menu, "selected": None},
                         "lora_modules": {"levels": sets_menu, "selected": None,
                                          "order": "trainable_params"}}
    # -- selection ----------------------------------------------------------------
    pd = bool(fit.get("positive_definite"))
    interior = bool(fit.get("argmin_interior"))
    winner = None
    if pd and interior:
        am = fit["argmin"]
        pair = _pair_of(am) or (refit_pair if pair_only else None)
        selection = {"lr": float(am["lr"]), "max_grad_norm": clip, "warmup_frac": wf,
                     "fitted_min_log_y": fit["fitted_min_log_y"],
                     "on_edge": False, "edge_pos": {"lr": fit["argmin_box_pos"],
                                                    "max_grad_norm": None},
                     "clamped_axes": [], "selected_from": "fitted_argmin"}
        if pair is not None:
            detail = next((r for r in rows if _pair_of(r) == pair), None)
            selection.update(lora_rank=pair[0], lora_set_id=pair[1],
                             lora_modules=(detail or {}).get("lora_modules")
                             or modules_of_set(pair[1]),
                             per_expert_rank=(detail or {}).get("per_expert_rank"),
                             init_scheme=(detail or {}).get("init_scheme"))
        winner = {**meta, "lr": selection["lr"], "max_grad_norm": clip,
                  "warmup_frac": wf, "lr_schedule": "warmup_cosine",
                  "selected_from": "fitted_argmin", "rounds_pooled": rounds,
                  **{k: selection[k] for k in ARCH_AXES if selection.get(k) is not None}}
        winner_ca = winner_clip_activation(selection, rows, box, fit["fixed_axes"])
        winner["clip_activation"] = winner_ca
        # the guard judges the fitted point against the fitted minimum
        guard_selected, guard_kw = selection, {"fitted_min_log_y": fit["fitted_min_log_y"]}
        nxt = ({"action": "confirm", "round": CONFIRMATION_ROUND} if pair_only
               else _refine_next(fit["u_star"], pair))
    else:
        reason = ("fit_not_positive_definite" if not pd else "fitted_argmin_on_edge")
        fall: dict = {"reason": reason}
        if pd:
            fall["fitted_argmin"] = {"lr": fit["argmin"]["lr"], "max_grad_norm": clip,
                                     "warmup_frac": wf, **{k: fit["argmin"][k] for k in
                                                           ("lora_rank", "lora_set_id")
                                                           if k in fit["argmin"]}}
            fall["edge_pos"] = {"lr": fit.get("argmin_box_pos"), "max_grad_norm": None}
        if clip_act["over_threshold"]:
            fall["clip_activation_excluded"] = clip_act["over_threshold"]
        sel_pair = _pair_of(fall_pick)
        selection = {"lr": float(fall_pick["lr"]), "max_grad_norm": clip,
                     "warmup_frac": wf, "index": fall_pick["index"],
                     "round": fall_pick["round"], "on_edge": (not interior) if pd else None,
                     "selected_from": "best_raw_trial",
                     "selected_trial": _tid(fall_pick), "fall_through": fall,
                     **{k: fall_pick[k] for k in ARCH_AXES if fall_pick.get(k) is not None}}
        # the guard judges the promoted ROW itself, with the fall-through filed
        guard_selected, guard_kw = fall_pick, {"fall_through": fall}
        winner_ca = None
        nxt = ({"action": "explore", "round": EXPLORATION_ROUND,
                "trial": _tid(fall_pick)} if pair_only
               else _refine_next(math.log10(float(fall_pick["lr"])), sel_pair,
                                 note=reason))
    if discrete_axes is not None:
        # the named menus carry the selected level on either path (None when
        # the selection has no pair — the ReLU search's rows)
        sel_pair = _pair_of(selection)
        discrete_axes["lora_rank"]["selected"] = sel_pair[0] if sel_pair else None
        discrete_axes["lora_modules"]["selected"] = sel_pair[1] if sel_pair else None
    guard = hpo_selection_guard.guard_selection(
        selected=guard_selected, selected_from=selection["selected_from"], rows=rows,
        box=box, levels=[wf], fit=fit, y_key=y_key, sigma_log_y=fit.get("sigma_log_y"),
        baseline=pool["baseline"], discrete_axes=discrete_axes, **guard_kw)
    _stamp_clip_activation(guard, clip_act, winner_ca)
    if adequacy.get("alarm"):
        guard["alarms"].append({
            "code": ADDITIVE_INADEQUATE_ALARM, "dim": None,
            "text": (f"additive LOO RMSE {_fmt_num(adequacy['additive_loo_rmse'])} "
                     f">= lr-only LOO RMSE "
                     f"{_fmt_num(adequacy['lr_only_loo_rmse_same_folds'])} "
                     "(held-out CE units): the rank / module-set effects do "
                     "not generalise out of sample; every fitted beta+gamma "
                     "prediction is model-dependent — the user's pick "
                     "resolves it, not more trials")})
        guard["alarmed"] = True
        guard["text"] = hpo_selection_guard.alarm_lines(guard)
    selection["alarmed"] = guard["alarmed"]
    selection["alarms"] = [hpo_selection_guard.alarm_code(a) for a in guard["alarms"]]
    artifact.update({"selection_guard": guard, "selection": selection,
                     "winner": winner, "next": nxt,
                     "followup": _followup_evidence(out_dir, fit, y_key)})
    for line in guard["text"]:
        print(f"[analyze v2] {line}")
    out_path.write_text(json.dumps(artifact, indent=2) + "\n")
    print(json.dumps({"selection": {k: v for k, v in selection.items()
                                    if k != "fall_through"}, "next": nxt},
                     indent=2, default=str))
    print(f"[analyze v2] artifact -> {out_path}")
    if getattr(args, "close_search", False):
        _close_search_from(artifact, out_dir, config_path=getattr(args, "config", None))
    if not pd:
        raise SystemExit(
            "[analyze] the quadratic in u is not positive definite (c = "
            f"{fit['coef_by_name']['u*u']:.4g}) — no fitted interior minimum; "
            f"the fall-through (best measured promotable trial {_tid(fall_pick)}) "
            f"is filed; artifact at {out_path}")


def completed_trials(out_dir: Path, search_id: str, campaign: str | None = None) -> dict:
    """The distinct TERMINAL trial ids of ONE search under ``out_dir``, across
    EVERY round — the design round, the refine round, and the fourth trial
    (the round-9 confirmation or the round-3 exploration replicate): 16 + 3
    + 1 under FD 6 — read from the summaries the trials wrote, never from an
    analyze artefact (whose rows are the FIT's pooled inputs: the confirmation
    and the replicate are refused from every pool by design, and a diverged
    trial is excluded from it, yet each is a completed trial with its own
    fit_trial row). Each counted trial is reconciled against its emission
    evidence (:data:`TRIAL_RECORD_NAME`, same search / trial / fingerprint):
    ``unrecorded`` names the terminal trials whose row has no evidence;
    with ``campaign`` (both closers pass the search's),
    ``mis_campaigned`` names the trials whose evidence was written under
    ANOTHER campaign — a replay invoked with the wrong --campaign — which
    the search's campaign has no emitted row for: a close over them would
    count rows that do not exist under it. A terminal summary naming ANOTHER search is ``foreign`` (a mixed
    directory); one naming no search predates the campaign and is not this
    search's trial (listed, not counted)."""
    ids: list[str] = []
    by_status: dict[str, int] = {}
    unrecorded: list[str] = []
    mis_campaigned: list[tuple[str, str | None]] = []
    foreign: list[tuple[str, str | None]] = []
    for p in sorted(Path(out_dir).glob("r*_t*/summary.json")):
        try:
            s = json.loads(p.read_text())
        except (ValueError, OSError):
            continue
        if not isinstance(s, dict) or s.get("status") not in _TERMINAL_STATUSES:
            continue
        tid = p.parent.name
        sid = s.get("search_id")
        if sid is None or str(sid) != str(search_id):
            foreign.append((tid, None if sid is None else str(sid)))
            continue
        ids.append(tid)
        by_status[str(s["status"])] = by_status.get(str(s["status"]), 0) + 1
        ev = trial_record_evidence(p.parent)
        if not trial_record_matches(ev, search_id=search_id, trial_id=tid,
                                    fit_fingerprint=s.get("fit_fingerprint")):
            unrecorded.append(tid)
        elif campaign is not None and str(ev.get("campaign")) != str(campaign):
            mis_campaigned.append((tid, ev.get("campaign")))
    return {"n": len(ids), "trial_ids": ids, "by_status": by_status,
            "unrecorded": unrecorded, "mis_campaigned": mis_campaigned,
            "foreign": foreign}


def reconcile_completed_trials(out_dir: Path, search_id: str, campaign: str | None = None) -> dict:
    """:func:`completed_trials`, refusing to close over a mixed directory,
    over a terminal trial whose fit_trial row has no emission evidence, or
 over a trial whose evidence names ANOTHER campaign
    than the search's — in every case the remedy is a replay, never a
    retrain, named by trial id."""
    done = completed_trials(out_dir, search_id, campaign)
    mixed = [f"{tid} names search {sid}" for tid, sid in done["foreign"]
             if sid is not None]
    if mixed:
        raise SystemExit(f"[registry] {out_dir} mixes searches — refusing to "
                         f"count trials for {search_id}: " + "; ".join(mixed))
    if done["unrecorded"]:
        raise SystemExit(
            f"[registry] {len(done['unrecorded'])} terminal trial(s) of search "
            f"{search_id} have no fit_trial emission evidence "
            f"({TRIAL_RECORD_NAME}): {done['unrecorded']} — re-run "
            "`tune_arm.py trial --index N` for each (it validates the stored "
            "summary and replays the record without retraining), then close")
    if done.get("mis_campaigned"):
        bad = ", ".join(f"{tid} (evidence under {c!r})" for tid, c in done["mis_campaigned"])
        raise SystemExit(
            f"[registry] {len(done['mis_campaigned'])} terminal trial(s) of search {search_id} have fit_trial "
            f"emission evidence under ANOTHER campaign than the search's {campaign!r}: {bad} — the search's campaign "
            "has no emitted row for them (a replay invoked with the wrong --campaign); re-run `tune_arm.py trial "
            "--trials <design> --index N --out <dir>` for each (the design's own campaign proves the search; the "
            "stored summary is validated and the record replayed under it, no retraining), then close")
    if done["n"] == 0:
        raise SystemExit(f"[registry] no terminal trial of search {search_id} "
                         f"under {out_dir} — nothing to close over")
    legacy = [tid for tid, sid in done["foreign"] if sid is None]
    if legacy:
        print(f"[registry] {len(legacy)} terminal summary(ies) under {out_dir} "
              f"name no search ({legacy}) — a pre-campaign design's trials, "
              f"not counted for {search_id}")
    return done


def _close_search_from(art: dict, out_dir: Path, *, config_path: str | None = None) -> None:
    """``close_search`` (contract §5) from an analyze artefact: the ReLU
    search's final refit closes here; the LoRA search closes at hpo_pick
    with the recorded pick. ``n_trials_completed`` is every terminal trial of
    the search across every round, reconciled against the emitted rows
    (:func:`reconcile_completed_trials`)."""
    sid = art.get("search_id")
    campaign = art.get("campaign")
    if not sid or not campaign:
        print("[registry] the artefact names no search_id / campaign — nothing "
              "to close")
        return
    eh = _emit_hpo()
    if eh is None:
        return
    opening = art.get("search_opening")
    if not opening:
        # the closing record SUPERSEDES the opening one (the fold keeps the
        # latest): closing from an empty opening would drop method, n_trials,
        # search_space_json, objective and substrate — refuse instead
        raise SystemExit(f"[registry] the artefact carries no search_opening "
                         f"payload for {sid} (emit-trials could not read the "
                         "opening fit_search back) — refusing to close the "
                         "search with a hollow superseding record")
    sel = art["selection"]
    fit = art["fit"]
    guard = art["selection_guard"]
    fu = art.get("followup") or {}
    win_params, win_tid, win_obj, evidence = winner_evidence(art)
    done = reconcile_completed_trials(out_dir, str(sid), campaign=str(campaign))
    opening = dict(opening)
    _cfg_sha = close_config_sha256(opening, config_path)
    if _cfg_sha is not None:
        opening["config_sha256"] = _cfg_sha
    eh.close_search(
        sid, campaign=str(campaign), opening=dict(opening),
        winner_params=win_params, winner_trial_id=win_tid,
        winner_objective=win_obj, argmin_interior=bool(sel.get("selected_from") == "fitted_argmin"),
        quadratic_fit_positive_definite=bool(fit.get("positive_definite")),
        quadratic_fit_params=fit.get("coef_by_name"),
        recentre_rounds=max(len(art["rounds"]) - 1, 0),
        confirmation_trial_id=fu.get("confirmation_trial_id"),
        confirmation_within_noise=fu.get("confirmation_within_noise"),
        noise_estimate=fit.get("sigma_log_y"),
        grad_norm_reconnaissance_uri=(art.get("recon") or {}).get("path"),
        trials_uri=None,
        n_trials_completed=done["n"],
        extra={"alarms": [hpo_selection_guard.alarm_code(a) for a in guard["alarms"]],
               "trials_completed": done["trial_ids"],
               "n_trials_by_status": done["by_status"],
               "n_fit_rows": len(art["rows"]), "n_fit_excluded": len(art["excluded"]),
               "alarmed": guard["alarmed"], "adequacy": art.get("adequacy"),
               "interaction": fit.get("interaction"), "pair_only": art.get("pair_only"),
               "refit_pair": art.get("refit_pair"),
               "exploration_replicate": fu.get("exploration_replicate"),
               "selected_from": sel["selected_from"],
               "winner_trial_evidence": evidence,
               "fit_artifact": str(out_dir)})
    print(f"registry: fit_search {sid} (closed; winner {win_params})")


def winner_evidence(art: dict) -> tuple[dict, str, float, str]:
    """(winner_params, winner_trial_id, winner_objective, how) for a
    close_search from an analyze artefact — the emitter needs a trial id
    and an objective, never None: a raw promotion names its trial and
    observed CE; a fitted argmin names its round-9 confirmation when it
    ran (the fitted minimum as the objective), else the best measured
    trial at the winning pair as the nearest evidence (said so)."""
    sel = art["selection"]
    fu = art.get("followup") or {}
    y_key = art.get("y_key") or "best_heldout_loss"
    tid_of = hpo_selection_guard.trial_id
    params = {k: sel[k] for k in ("lr", "max_grad_norm", "warmup_frac", *ARCH_AXES)
              if sel.get(k) is not None}
    if params.get("lora_modules") is not None:
        params["lora_modules"] = sorted(params["lora_modules"])
    if sel["selected_from"] == "best_raw_trial":
        tid = sel.get("selected_trial") or (art.get("selection_guard") or {}).get("selected_trial")
        row = next(r for r in art["rows"] if tid_of(r) == tid)
        return params, str(tid), float(row[y_key]), "promoted_measured_trial"
    obj = math.exp(float(sel["fitted_min_log_y"]))
    if fu.get("confirmation_trial_id"):
        return params, str(fu["confirmation_trial_id"]), obj, "fitted_argmin+confirmation"
    pair = _pair_of(sel)
    cands = [r for r in art["rows"] if pair is None or _pair_of(r) == pair] or art["rows"]
    best = min(cands, key=lambda r: float(r[y_key]))
    return params, tid_of(best), obj, "fitted_argmin;nearest_evidence=best_measured_trial_at_pair"


# --------------------------------------------------------------------------
# Shared setup for probe-mem / trial (ported, generalized).
# --------------------------------------------------------------------------

def _load_driver():
    """Import scripts/train_two_si_objects.py as a module so the holdout
    split, pool loader and sidecar conventions are the driver's OWN code,
    never a reimplementation."""
    import importlib.util
    import sys

    p = Path(__file__).resolve().parent / "train_two_si_objects.py"
    # ONE driver instance per process: every consumer (the adapters' held-out
    # pool path, the setup, the stamps) reads the same module object, so a
    # process-wide fact (which volume root resolves /vol/) is one fact — and a
    # test that patches the driver's seam patches what the adapters call.
    # Re-executing the file per call (the previous behaviour) handed each
    # caller a fresh module and re-ran the driver's import work every time.
    cached = sys.modules.get("_two_si_driver")
    if cached is not None and getattr(cached, "__file__", None) == str(p):
        return cached
    spec = importlib.util.spec_from_file_location("_two_si_driver", p)
    mod = importlib.util.module_from_spec(spec)
    # Registered BEFORE exec: the driver defines @dataclass classes, and
    # dataclasses resolves cls.__module__ through sys.modules.
    sys.modules["_two_si_driver"] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_cfg(path: str) -> dict:
    return yaml.safe_load(Path(path).read_text())


def _pool_builder():
    """Import scripts/build_two_si_pool.py under its canonical module name
    (the driver's own ``import build_two_si_pool`` — pairs_for_arm's
    own-pool branch, ``_load_bidpo_pool`` — then resolves to the SAME
    module). Its sibling imports (jailbreak_run, pool_sample) need the
    scripts dir on sys.path, so this inserts it — the test suite's own
    convention. Stdlib-cheap: no torch at its module level."""
    import importlib
    import sys

    scripts = str(Path(__file__).resolve().parent)
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    return importlib.import_module("build_two_si_pool")


def _warm_tokcache(driver, cfg: dict, pairs_path: str, tok,
                   allow_tokenize: bool):
    from sisteer.extraction import tokcache

    sidecar = driver._token_sidecar_path(pairs_path)
    if sidecar.exists():
        # expect_pool_path engages the sidecar's POOL-CONTENT guard (the
        # same guard the production driver passes): a sidecar whose
        # recorded pool sha does not match --pairs refuses instead of
        # silently tuning on stale tokens.
        cache = tokcache.warm_from_sidecar(sidecar, tok,
                                           expect_model_id=cfg["model"],
                                           expect_pool_path=pairs_path)
        tokcache.set_active(cache)
        print(f"[tokcache] warmed {len(cache)} entries from {sidecar}")
    elif allow_tokenize:
        tokcache.set_active(tokcache.TokCache())
        print(f"[tokcache] NO sidecar at {sidecar}; tokenizing in-process")
    else:
        raise SystemExit(
            f"[tune-arm] no token sidecar at {sidecar} — the sweep is "
            "meant to run pre-tokenized (point --pairs at the pool copy "
            "whose .tokcache.npz sits next to it, or pass --allow-tokenize)")


@dataclass
class TuneSetup:
    """Everything a trial needs, resolved once. ``positions`` and the
    derived ``response_header_len`` thread through the SAME protected-kwarg
    path the production driver uses: lockstep arms through
    ``build_lockstep_object``'s positions leg, solo trainers as explicit
    keyword arguments their signatures declare — never droppable."""

    cfg: dict
    driver: object
    model: object
    tok: object
    instruction: object
    si: str
    fit_pairs: list
    held_pairs: list
    held_ex: list
    max_len: int
    positions: str
    response_header_len: int | None
    layer: int | None
    # bidpo's own-pool fit sides: the holdout-carved bidpo_harmful_<si> /
    # bidpo_benign_<si> pairs, the benign cap, and the anti-neg flag —
    # everything ``driver.pairs_for_arm`` needs to route bidpo EXACTLY as
    # production does. None sides = the feature is off (legacy routes
    # apply).
    bidpo_harmful_fit: list | None = None
    bidpo_benign_fit: list | None = None
    bidpo_benign_cap: float | None = 1.0  # None = uncapped (config semantics)
    bidpo_anti_neg: bool = False
    bidpo_pool_meta: dict | None = None
    #: the FULL loaded pool in file order (fit + held before the carve) —
    #: what the teacher top-K artifact's pool_content_sha256 hashes.
    all_pairs: list | None = None


def _derive_header_len(cfg: dict, tok, positions: str) -> int | None:
    """The driver's exact H-derivation semantics (train_two_si_objects.py
    positions block): config override wins; else derive from the tokenizer;
    H == 0 refuses loudly — a zero H silently collapses header-steered to
    plain decode-only."""
    if positions == "all":
        return None
    tk = dict((cfg.get("train_kwargs") or {}).get("default") or {})
    rhl_cfg = tk.get("response_header_len")
    if rhl_cfg is not None:
        return int(rhl_cfg)
    from sisteer.hooks import response_header_len as _derive_h

    h = _derive_h(tok)
    if h == 0:
        raise SystemExit(
            f"[tune-arm] positions={positions!r} but the assistant-header "
            "length could not be derived from this tokenizer (H == 0). Set "
            "train_kwargs.default.response_header_len explicitly.")
    return h


def _setup(args, positions: str, need_model: bool = True,
           arm: str | None = None) -> TuneSetup:
    # The shared backend pin: disables cuDNN SDPA, whose per-ragged-shape
    # CPU plan compilation halved GPU duty on ragged batches. Trials ARE
    # the production train paths, so the pin applies here too.
    from sisteer.data.pairs import pairs_to_distill_examples
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.eval.train_budget import resolve_max_len
    from sisteer.extraction.fitmon import pin_attention_backends
    from sisteer.models import load_model_and_tokenizer, load_tokenizer

    pin_attention_backends()
    cfg = _load_cfg(args.config)
    driver = _load_driver()
    # The config's leading-special-token policy (the driver's ONE spelling)
    # rides every tokenizer construction here too: a trial or a memory
    # probe must tokenize under exactly the post-processor the production
    # fit will run — otherwise HPO ranks hyperparameters under a different
    # token geometry (gemma-it: BOS-less) than the fit trains on, and a
    # policy-built sidecar refuses to warm.
    leading = driver._leading_policy(cfg)
    if need_model:
        model, tok = load_model_and_tokenizer(
            cfg["model"], dtype=cfg.get("dtype", "bfloat16"),
            revision=cfg.get("model_revision"),
            attn_implementation=cfg.get("attn_implementation", "sdpa"),
            leading_special_tokens=leading)
        # Trials/probes ARE the production train paths, so the config's
        # gradient_checkpointing flag applies here identically (per-arm) —
        # probe-mem must MEASURE the footprint the fits will run under.
        # probe-mem / bench ONLY: --gradient-checkpointing on|off measures
        # the OTHER setting without editing a committed config (whose bytes
        # are every cell's fingerprint); it rewrites this process's
        # in-memory per-arm block, so the adapter reads the same value the
        # model was set up with.
        gc_mode = getattr(args, "gradient_checkpointing", "config") or "config"
        if gc_mode != "config":
            if not arm:
                raise SystemExit("[tune-arm] --gradient-checkpointing "
                                 "override needs a single --arm")
            (cfg.setdefault("train_kwargs", {}).setdefault(arm, {})
             )["gradient_checkpointing"] = (gc_mode == "on")
            print(f"[setup] gradient_checkpointing OVERRIDE for {arm}: "
                  f"{gc_mode} (probe/bench only; config untouched)")
        driver.maybe_enable_gradient_checkpointing(
            model, cfg, arms=[arm] if arm else None)
    else:
        model, tok = None, load_tokenizer(cfg["model"],
                                          revision=cfg.get("model_revision"),
                                          leading_special_tokens=leading)
    _warm_tokcache(driver, cfg, args.pairs, tok, args.allow_tokenize)
    pairs = driver._load_pairs(args.pairs)
    tkd = driver._train_kwargs_default(cfg)
    holdout_frac = float(tkd.get("holdout_frac", 0.1))
    holdout_salt = str(tkd.get("holdout_salt", "two_si_holdout"))
    fit_pairs, held_pairs = driver.split_pairs(pairs, holdout_frac,
                                               holdout_salt)
    # bidpo's own pool, loaded and holdout-carved EXACTLY as the production
    # driver does it: the driver's own loader (default sibling files beside
    # --pairs, sha-verified against the emit sidecar) + the same
    # deterministic split. (None, None, None) when pool.bidpo_benign is off
    # — pairs_for_arm then takes its legacy routes over the matched pool.
    bh_fit = bb_fit = None
    bidpo_meta = None
    bidpo_cap: float | None = 1.0
    anti_neg = bool(
        ((cfg.get("pool") or {}).get("anti_neg") or {}).get("enabled", False))
    if arm is not None and arm in getattr(driver, "DPO_PREF_ARMS", ()):
        bp = _pool_builder()
        bidpo_cap = bp._bidpo_benign_cfg(cfg)["max_benign_per_harmful"]
        bh, bb, bidpo_meta = driver._load_bidpo_pool(args, cfg, [arm])
        if bh is not None:
            bh_fit, _bh_held = driver.split_pairs(bh, holdout_frac,
                                                  holdout_salt)
            bb_fit, _bb_held = driver.split_pairs(bb, holdout_frac,
                                                  holdout_salt)
            print(f"[setup] bidpo own pool (production loader + split): "
                  f"harmful {len(bh)}->{len(bh_fit)} fit, "
                  f"benign {len(bb)}->{len(bb_fit)} fit, "
                  f"cap={bidpo_cap}")
    instr = resolve_instruction(args.si)
    held_ex = pairs_to_distill_examples(tok, held_pairs, instr)
    max_len = resolve_max_len(cfg, args.si)
    rhl = _derive_header_len(cfg, tok, positions)
    layer = int(cfg["layer"]) if "layer" in cfg else None
    print(f"[setup] si={args.si} pairs={len(pairs)} fit={len(fit_pairs)} "
          f"heldout={len(held_pairs)} max_len={max_len} "
          f"positions={positions} H={rhl}")
    return TuneSetup(cfg=cfg, driver=driver, model=model, tok=tok,
                     instruction=instr, si=args.si, fit_pairs=fit_pairs,
                     held_pairs=held_pairs, held_ex=held_ex, max_len=max_len,
                     positions=positions, response_header_len=rhl,
                     layer=layer, bidpo_harmful_fit=bh_fit,
                     bidpo_benign_fit=bb_fit, bidpo_benign_cap=bidpo_cap,
                     bidpo_anti_neg=anti_neg, bidpo_pool_meta=bidpo_meta,
                     all_pairs=pairs)


# --------------------------------------------------------------------------
# Length bucketing (ported 8ded0ea) + supervised-token accounting.
# --------------------------------------------------------------------------

def length_sorted(tok, examples):
    """Length-bucket batching (the shared fitmon implementation — ported
    from lora tip 8ded0ea): sorting by rendered token length makes every
    batch pad to a near-uniform width (ragged batches wasted ~30-40% of
    token compute on padding, and every distinct padded shape made cuDNN
    SDPA re-plan on CPU). Batch COMPOSITION changes vs the unsorted
    production driver — same objective, different gradient grouping;
    recorded in the summary as ``length_sorted``."""
    from sisteer.extraction.fitmon import length_sorted as _ls

    return _ls(tok, examples)


def prefs_length_sorted(tok, prefs):
    """bidpo's analogue: sort preference examples by prompt + the LONGER of
    the two completions (both sides of a pref pair collate into their own
    batches, so the longer side sets the padded width)."""
    from sisteer.extraction.tokcache import cached_ids

    def keylen(p):
        return (len(cached_ids(tok, p.prompt, True))
                + max(len(cached_ids(tok, p.chosen, False)),
                      len(cached_ids(tok, p.rejected, False))))

    return sorted(prefs, key=keylen)


# --------------------------------------------------------------------------
# Trial-side monitoring plumbing.
# --------------------------------------------------------------------------

class TrialDiverged(RuntimeError):
    pass


class GradNormBuffer:
    """GPU-side buffered grad-norm series (the gpu-util rule: monitoring
    must not introduce per-step CPU<->GPU syncs). Per step the pre-clip
    norm (fitmon.GradNormTap.flush_tensor) and the post-clip norm
    (fitmon.postclip_grad_norm) are appended as 0-dim GPU tensors;
    ``drain()`` materializes the whole backlog in ONE sync at the log/probe
    cadence and returns (step, preclip, postclip) float triples, raising
    TrialDiverged if any drained value is infinite (bounded-delay
    divergence detection — the per-step loss NaN check still fires
    immediately; None = "not captured this step", recorded as nan)."""

    def __init__(self):
        self._steps: list[int] = []
        self._pre: list = []
        self._post: list = []

    def append(self, step: int, pre, post) -> None:
        self._steps.append(step)
        self._pre.append(pre)
        self._post.append(post)

    @staticmethod
    def _materialize(vals) -> list[float]:
        import torch
        tensors = [v for v in vals if torch.is_tensor(v)]
        if tensors:
            flat = torch.stack([v.detach().reshape(()).float()
                                for v in tensors]).tolist()  # ONE sync
            it = iter(flat)
            return [next(it) if torch.is_tensor(v)
                    else (float("nan") if v is None else float(v))
                    for v in vals]
        return [float("nan") if v is None else float(v) for v in vals]

    def drain(self, *, check: bool = True) -> list[tuple[int, float, float]]:
        if not self._steps:
            return []
        pre = self._materialize(self._pre)
        post = self._materialize(self._post)
        out = list(zip(self._steps, pre, post))
        self._steps, self._pre, self._post = [], [], []
        if check:
            for s, p, q in out:
                for v in (p, q):
                    if math.isinf(v):
                        raise TrialDiverged(
                            f"step {s}: non-finite grad norm {v}")
        return out


# --------------------------------------------------------------------------
# The arm adapter registry.
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class TrialHP:
    """The swept hyperparameters. warmup rides as a FRACTION end-to-end;
    each adapter resolves it against its trial's OWN total optimizer steps
    (and the winner ships the fraction, never a step count). The re-tune@32
    axes are None when the arm does not sweep them (ARM_AXES): None means
    "not swept — the trainer keeps its registry/config default", never
    "zero"."""

    lr: float
    warmup_frac: float
    max_grad_norm: float
    temperature: float | None = None
    weight_decay: float | None = None
    lora_rank: int | None = None
    # sdf-base-steering round-2 gate_reg mini-sweep (2026-09-02): the PSR
    # anti-dead-gate regularizer coefficient, swept via a bespoke
    # fixed-(lr, clip, warmup) grid design (scripts/sdfbase/
    # emit_gatereg_trials.py). psr_ce_all only — every other arm refuses a
    # trial row carrying it (run_trial guard) rather than silently
    # training at the config default.
    gate_reg: float | None = None
    # The LoRA architecture axes and the init scheme — the v2 design stamps
    # them on every row and CESoloAdapter.make_object builds through
    # object_factory.build_object with them. None = not declared by the
    # design (the paper's rank-4 o_proj zero-init object, via the config).
    lora_modules: tuple[str, ...] | None = None
    per_expert_rank: dict | int | None = None
    init_scheme: str | None = None


def _cast_modules(v) -> tuple[str, ...]:
    if isinstance(v, str):
        parts = [m.strip() for m in v.split(",") if m.strip()]
    else:
        parts = [str(m) for m in v]
    if not parts:
        raise SystemExit("[trial] lora_modules on the row is empty")
    return tuple(sorted(set(parts)))


def _cast_per_expert_rank(v):
    if v is None:
        return None
    if isinstance(v, dict):
        return {str(k): int(x) for k, x in v.items()}
    return int(v)


def _fmt_axis(v) -> str:
    """A trial axis value for the launch banner: numbers as :g, a module
    tuple as its set id, anything else via repr."""
    if isinstance(v, bool):
        return repr(v)
    if isinstance(v, (int, float)):
        return f"{v:g}"
    if isinstance(v, (list, tuple)):
        return set_id_of(v)
    return v if isinstance(v, str) else repr(v)


#: The OPTIONAL per-trial axes (TrialHP's None-able fields) and the cast each
#: takes off a trial row — the re-tune@32 ARM_AXES plus the round-2 gate_reg
#: grid, plus the LoRA architecture axes + init scheme of the v2 design. One
#: owner for the five trial-side sites that walk them (the trial
#: fingerprint, TrialHP construction, the launch banner, the summary echo,
#: the fit_trial params);
#: the analyze-side tuples deliberately stay 3-wide (the quadratic-winner
#: path does not consume the grid design).
TRIAL_EXTRA_AXES: dict[str, type] = {
    "temperature": float, "weight_decay": float, "lora_rank": int,
    "gate_reg": float,
    "lora_modules": _cast_modules, "per_expert_rank": _cast_per_expert_rank,
    "init_scheme": str,
}


#: Config train_kwargs.<arm> keys an adapter must NOT forward into a trial:
#: the swept keys belong to the trial's hp; the driver-level knobs
#: (floor_ratio_max & friends) are not trainer kwargs; epochs/batch_size/
#: max_len/seed/accum_steps are the harness's own CLI-owned budget.
_STRIPPED_CFG_KEYS = frozenset({
    "lr", "lr_schedule", "warmup_steps", "warmup_frac", "max_grad_norm",
    "floor_ratio_max", "epochs", "batch_size", "max_len", "seed",
    "accum_steps", "positions", "response_header_len",
    "teacher_prefix_cache",
    # re-tune@32 SWEPT keys: the trial's hp owns them (ARM_AXES), so a
    # config value could only double-set what the design chose per trial.
    "temperature", "weight_decay", "lora_rank",
    # the LoRA architecture axes + init scheme (v2 design): owned by the
    # trial row / object_factory, never trainer kwargs
    "lora_modules", "per_expert_rank", "init_scheme",
    # driver knob, consumed at model load (maybe_enable_gradient_
    # checkpointing) — never a trainer kwarg
    "gradient_checkpointing",
})


def _cfg_extras(cfg: dict, arm: str) -> dict:
    """Per-arm NON-swept config kwargs (beta, weight_decay, stop_pref_prob,
    reg_coeff, ...). Forwarded verbatim to the trainer, whose signature
    refuses unknown keys loudly — the protected-kwarg discipline, held here
    by never using a TypeError-drop retry at all."""
    raw = dict(((cfg.get("train_kwargs") or {}).get(arm)) or {})
    return {k: v for k, v in raw.items() if k not in _STRIPPED_CFG_KEYS}


class ArmAdapter:
    """One tunable arm behind a fixed interface:

      name / kind ("lockstep" | "solo")
      loss_key / lr_key       — the trainer's per-step metric names
      floor_ratio_max         — the objective's own floor bar (distillation
                                arms 1.0, bidpo 3.0 — config convention)
      route_pairs(setup)      — the pair subset this arm trains on, via the
                                PRODUCTION resolver (bidpo overrides with
                                driver.pairs_for_arm; every other arm gets
                                the full fit pool back)
      prepare_data(...)       — build this arm's training rows from pairs
      make_object(setup)      — production-exact object construction
      train(setup, hp, ...)   — the arm's production trainer, hp threaded
      current_object()        — the live object for mid-fit KL probes
      grad_params()           — params for fitmon.GradNormTap (None = the
                                trainer logs its own grad telemetry)
      norm_groups()           — per-step object norms, ONE sync per call
      save(out_dir)           — trial artifact
    """

    name: str = ""
    kind: str = ""
    floor_ratio_max: float = 1.0
    probe_via_trainer: bool = False
    probe_needs_inject: bool = False
    needs_layer: bool = False
    supports_response: bool = True
    uses_pref_pool: bool = False  # bidpo: own-pool routing + accum floor
    #: whether this arm's trainer can consume the offline top-K teacher
    #: store (train_lockstep's teacher_topk seam). False for the solo arms:
    #: psr_mse's teacher is ACTIVATIONS, bidpo has no teacher forward.
    consumes_teacher_topk: bool = False
    #: whether run_trial arms the SI-teacher KL probe (KLRatioProbe) at
    #: all. False for the CE arms: their objective has no SI teacher, so a
    #: KL(prompted||unsteered) probe is meaningless (the lockstep.py:722
    #: convention) — the held-out CE curve is their whole verdict.
    uses_kl_probe: bool = True

    def __init__(self):
        self._obj = None
        self._n_fit = 0
        self._fit_ex: list = []
        #: the loaded TeacherTopK store (run_trial/probe_mem attach it);
        #: consumed by the lockstep adapters, refused elsewhere.
        self.teacher_topk = None

    @property
    def effective_batch_target(self) -> int | None:
        """The effective batch this arm's trials must realize via gradient
        accumulation (None = no accumulation — the historical loop). Read
        as a property from the MODULE constants so tests can monkeypatch
        them. The re-tune arms pin DISTILL_EFFECTIVE_BATCH; bidpo pins its
        preference-objective floor."""
        return None

    def accum_for(self, batch_size: int) -> int:
        target = self.effective_batch_target
        if not target:
            return 1
        return max(1, math.ceil(target / int(batch_size)))

    # -- data ---------------------------------------------------------------
    def route_pairs(self, setup: TuneSetup) -> list:
        """The pair subset this arm trains on. Default: the full fit pool
        (production: every arm but the DPO preference arms trains on the
        full pool — pairs_for_arm's contract). bidpo overrides with the
        production resolver itself."""
        return setup.fit_pairs

    def data_summary(self) -> dict:
        """Per-arm data-provenance keys merged into the trial summary
        (bidpo records its routed pool's realized counts + polarity)."""
        return {}

    def fingerprint_pool_extra(self, args, cfg: dict) -> str:
        """Extra pool-identity material for the fit fingerprint, appended
        to the pool composite ("" = nothing extra — every legacy arm).
        The CE arms append the shared held-out pool's content sha: a trial
        judged on a different held-out pool is different work, and a stale
        summary must re-run, not be trusted."""
        return ""

    def prepare_data(self, setup: TuneSetup, fit_pairs: list, *,
                     length_sort: bool = True,
                     take_longest: int | None = None,
                     batch_size: int | None = None) -> int:
        """Default: distill examples into ``self._fit_ex`` (every arm but
        bidpo, whose preference-pair analogue overrides this).
        ``batch_size`` is bidpo's full-optimizer-group trim hook; the
        distill arms ignore it (their trainers have no accumulation)."""
        from sisteer.data.pairs import pairs_to_distill_examples

        ex = pairs_to_distill_examples(setup.tok, fit_pairs,
                                       setup.instruction)
        if take_longest is not None:
            ex = length_sorted(setup.tok, ex)[-take_longest:]
        elif length_sort:
            ex = length_sorted(setup.tok, ex)
        self._fit_ex = ex
        self._n_fit = len(ex)
        return self._n_fit

    def sup_tokens_per_epoch(self, tok) -> int:
        from sisteer.extraction.tokcache import cached_ids

        return sum(len(cached_ids(tok, ex.completion, False))
                   for ex in self._fit_ex)

    def optimizer_steps(self, batch_size: int, epochs: int) -> int:
        # Accumulation makes an optimizer step span accum_for(bs)
        # consecutive micro-batches (the ragged tail is its own smaller
        # group) — at accum 1 this is exactly the historical formula.
        micro = math.ceil(self._n_fit / batch_size)
        return math.ceil(micro / self.accum_for(batch_size)) * epochs

    def progress_steps(self, batch_size: int, epochs: int) -> int:
        """The horizon of the trainer's OWN step counter — what log_fn /
        probe_cb step values run up to. The solo trainers count optimizer
        steps; train_lockstep's counter ticks per MICRO-batch (it logs once
        per accumulation group, carrying the micro counter), so the
        lockstep adapter overrides this for heartbeat totals."""
        return self.optimizer_steps(batch_size, epochs)

    # -- object -------------------------------------------------------------
    def make_object(self, setup: TuneSetup, sentinel=None,
                    lora_rank: int | None = None, seed: int = 0):
        """Build this arm's object. ``seed`` (2026-09-15) is the trial's
        --seed: the CE LoRA arm seeds its A-factor init with it (the
        production driver's methods._train_lora_ce does the same); every
        other adapter's construction is the registry builder's own and
        ignores it — its seed governs the trainer alone. Default 0 keeps
        every object bit-identical to before."""
        raise NotImplementedError

    def current_object(self):
        return self._obj

    def grad_params(self):
        return None

    def norm_groups(self) -> dict:
        """Object norms via the shared fitmon reporter (attribute-
        dispatched), materialized in ONE device sync for the whole group —
        called at the log cadence only, never per step."""
        import torch
        from sisteer.extraction.fitmon import norm_groups as _ng

        obj = self.current_object()
        if obj is None:
            return {}
        vals = _ng(obj)
        if not vals:
            return {}
        keys = list(vals)
        floats = torch.stack(
            [vals[k].detach().float().reshape(()) for k in keys]
        ).tolist()  # ONE sync
        return dict(zip(keys, floats))

    def diverged_exceptions(self) -> tuple:
        return ()

    def post_train_check(self, setup: TuneSetup) -> None:
        return None

    # -- train --------------------------------------------------------------
    def train(self, setup: TuneSetup, hp: TrialHP, *, epochs: int,
              batch_size: int, seed: int, log_fn, probe_cb, probe_every: int,
              diagnostics: dict) -> None:
        raise NotImplementedError

    # -- verdicts -----------------------------------------------------------
    def heldout_objective(self, diagnostics: dict,
                          final_probe: dict) -> tuple[float | None, str]:
        """(value, label) of the scalar the quadratic ranks trials by.
        Solo arms use the FINAL held-out residual KL from the arm-agnostic
        probe (their trainers have no held-out loop); lockstep arms use the
        production held-out loss."""
        raise NotImplementedError

    def floor_verdict(self, diagnostics: dict, final_probe: dict
                      ) -> tuple[bool | None, float | None, float | None]:
        """(floor_ok, floor_ratio, floor_ratio_sem) against
        ``self.floor_ratio_max``."""
        raise NotImplementedError

    def save(self, out: Path) -> str | None:
        """Save the trained object under ``out``; returns the file NAME
        saved (the trial summary records it), None when there is nothing
        to save."""
        from sisteer.methods import object_artifact_name

        obj = self.current_object()
        if obj is not None and hasattr(obj, "save"):
            name = object_artifact_name(self.name)
            obj.save(out / name)
            return name
        return None


class LockstepAdapter(ArmAdapter):
    """psr_kl_all / lora_r1_all / token_faithful: a singleton group through
    the production ``train_lockstep`` — probes, early stopping, best-epoch
    restore, the mask->loss wiring — with the trial hp on
    ``LockstepArmParams`` (the exact seam the winner ships on).

    Re-tune@32: trials run at effective batch DISTILL_EFFECTIVE_BATCH via
    ``LockstepArmParams.accum_steps``; warmup ships as ``warmup_frac`` on
    the params (the trainer resolves steps against its OWN accum-aware
    budget); swept temperature/weight_decay ride the same params; and a
    temperature-swept trial is RANKED by the T=1 arm-agnostic probe
    residual (final_residual_kl), because the trainer's own-T held-out
    loss is not comparable across temperature levels (KL at higher T is
    smaller by construction). Early stopping inside the trainer still
    selects the best epoch by its own-T held-out loss — a within-trial,
    fixed-T comparison — and the probe ranks the RESTORED object."""

    kind = "lockstep"
    floor_ratio_max = 1.0
    consumes_teacher_topk = True

    def __init__(self, name: str):
        super().__init__()
        self.name = name
        self.loss_key = f"lockstep/{name}/soft_loss"
        self.lr_key = f"lockstep/{name}/lr"
        self._held_ex: list = []
        self._ranked_by_probe = False

    @property
    def effective_batch_target(self) -> int | None:
        return DISTILL_EFFECTIVE_BATCH

    def progress_steps(self, batch_size: int, epochs: int) -> int:
        # train_lockstep's step counter ticks per MICRO-batch (one log/
        # probe event per accumulation group, carrying the micro counter).
        return math.ceil(self._n_fit / batch_size) * epochs

    def prepare_data(self, setup, fit_pairs, *, length_sort=True,
                     take_longest=None, batch_size=None) -> int:
        n = super().prepare_data(setup, fit_pairs, length_sort=length_sort,
                                 take_longest=take_longest,
                                 batch_size=batch_size)
        # Eval-only sort: speed, not semantics.
        self._held_ex = (length_sorted(setup.tok, setup.held_ex)
                         if length_sort else list(setup.held_ex))
        return n

    def make_object(self, setup, sentinel=None, lora_rank=None, seed=0):
        import inspect

        from sisteer.extraction.lockstep import build_lockstep_object

        del seed  # legacy lockstep arms: the builder's own init (trainer-only seed)

        if lora_rank is not None and self.name not in ("lora_r1_all",
                                                       "lora_r4"):
            raise SystemExit(
                f"[tune-arm] lora_rank={lora_rank} for arm {self.name!r} — "
                "rank is the lora arm's search axis only")
        kw = {}
        if lora_rank is not None and int(lora_rank) != 1:
            if "lora_rank" in inspect.signature(
                    build_lockstep_object).parameters:
                kw["lora_rank"] = int(lora_rank)
            else:
                # The builder has not grown its lora_rank leg yet (the
                # trainer lane owns lockstep.py). Mirror the builder's OWN
                # lora_r1_all lines — construction + fit-mask leg — so the
                # trial is production-exact but rank-parametric; fold into
                # build_lockstep_object once that lane lands and this
                # branch dies (the signature check flips by itself).
                from sisteer.extraction.multilayer import MultiLayerLoRAObject

                obj = MultiLayerLoRAObject(setup.model,
                                           rank=int(lora_rank))
                if setup.positions != "all":
                    obj.set_fit_positions(setup.positions,
                                          setup.response_header_len)
                self._obj = obj
                return self._obj
        self._obj = build_lockstep_object(
            self.name, setup.model, setup.tok, setup.instruction,
            setup.fit_pairs, layer=setup.layer or 0, sentinel=sentinel,
            positions=setup.positions,
            response_header_len=setup.response_header_len, **kw)
        return self._obj

    def grad_params(self):
        return list(self._obj.trainable)

    def train(self, setup, hp, *, epochs, batch_size, seed, log_fn,
              probe_cb, probe_every, diagnostics) -> None:
        import dataclasses

        from sisteer.extraction.distill import EarlyStopSpec
        from sisteer.extraction.lockstep import (LockstepArm,
                                                 LockstepArmParams,
                                                 default_lockstep_params,
                                                 train_lockstep)

        accum = self.accum_for(batch_size)
        total = self.optimizer_steps(batch_size, epochs)
        fields = {f.name for f in dataclasses.fields(LockstepArmParams)}
        # warmup rides as the FRACTION on the params — the trainer resolves
        # steps against its own accum-aware optimizer-step budget.
        overrides: dict = {"lr": float(hp.lr),
                           "lr_schedule": "warmup_cosine",
                           "warmup_frac": float(hp.warmup_frac),
                           "max_grad_norm": float(hp.max_grad_norm)}
        if accum != 1:
            overrides["accum_steps"] = accum
        if hp.temperature is not None:
            overrides["temperature"] = float(hp.temperature)
            self._ranked_by_probe = True
        if hp.weight_decay is not None:
            overrides["weight_decay"] = float(hp.weight_decay)
        missing = sorted(set(overrides) - fields)
        if missing:
            raise SystemExit(
                f"[tune-arm] this trial sweeps {missing} but "
                "LockstepArmParams does not carry the field(s) yet — the "
                "trainer-lane half of the re-tune@32 (accum_steps/"
                "weight_decay on LockstepArmParams) has not landed; "
                "refusing to silently drop a swept hyperparameter")
        params = replace(default_lockstep_params(self.name), **overrides)
        extras = _cfg_extras(setup.cfg, self.name)
        allowed = {"loss", "loss_scale", "reg_coeff"}
        unknown = set(extras) - allowed
        if unknown:
            raise SystemExit(
                f"[tune-arm] train_kwargs.{self.name} carries keys "
                f"{sorted(unknown)} that LockstepArmParams does not take — "
                "refusing to silently drop them")
        if extras:
            params = replace(params, **extras)
        arm = LockstepArm(self.name, self._obj, params)
        self._resolved = {"total_steps": total, "accum_steps": accum,
                          "effective_batch": accum * int(batch_size)}
        kw = {}
        if self.teacher_topk is not None:
            # Offline top-K teacher (re-tune@32): the store replaces the
            # online teacher leg. Passed ONLY when set, so the harness
            # stays call-compatible with a pre-topk train_lockstep.
            kw["teacher_topk"] = self.teacher_topk
        train_lockstep(
            setup.model, setup.tok, [arm], self._fit_ex,
            epochs=epochs, batch_size=batch_size, max_len=setup.max_len,
            seed=seed, log_fn=log_fn, heldout=self._held_ex,
            early_stop=EarlyStopSpec(patience=EARLY_STOP_PATIENCE,
                                     min_epochs=1, min_delta=0.0),
            diagnostics=diagnostics, n_probe=256, teacher_prefix_cache=False,
            **kw)

    def heldout_objective(self, diagnostics, final_probe):
        if self._ranked_by_probe:
            # Temperature-swept cell: the trainer's held-out loss is the
            # arm's OWN-T KL — smaller at higher T by construction, so
            # incomparable across trials. Rank by the T=1 arm-agnostic
            # probe residual on the restored best-epoch object instead
            # (the solo arms' metric, so the label greps the same).
            return final_probe.get("probe/residual_kl"), "final_residual_kl"
        d = diagnostics.get(self.name, {})
        return d.get("best_heldout_loss"), "lockstep_heldout_loss"

    def floor_verdict(self, diagnostics, final_probe):
        from sisteer.extraction.distill import (FloorRatioError,
                                                floor_ratio_check)

        d = diagnostics.get(self.name, {})
        ratio = d.get("floor_ratio")
        sem = d.get("floor_ratio_sem")
        if d.get("final_probe_loss") is None or d.get("noop_floor") is None:
            return None, ratio, sem
        try:
            floor_ratio_check(d["final_probe_loss"], d["noop_floor"],
                              ratio_max=self.floor_ratio_max, tag=self.name)
            return True, ratio, sem
        except FloorRatioError:
            return False, ratio, sem


class PsrKlAllAdapter(LockstepAdapter):
    def post_train_check(self, setup) -> None:
        # Driver parity (train_two_si_objects lockstep leg): a PSR whose
        # gates trained shut is a FAILED trial, recorded, never a data
        # point. check_psr_gates raises DeadGateError.
        from sisteer.extraction.psr import GUARD_PROBE_FLOOR, check_psr_gates

        check_psr_gates(setup.model, setup.tok, obj=self._obj,
                        examples=self._fit_ex, max_len=setup.max_len,
                        tag=f"tune_{self.name}",
                        min_probe=min(GUARD_PROBE_FLOOR, len(self._fit_ex)))


class TokenFaithfulAdapter(LockstepAdapter):
    #: A soft PROMPT token has no generated-position notion: the
    #: response-positions tuning cell is a PERMANENT hole (FIT_MASK_REFUSAL).
    supports_response = False
    #: The sentinel lives in the PROMPT IDS — the probe must encode the
    #: injected student rendering or obj.apply() is a silent no-op there
    #: (fitmon.KLRatioProbe's ``inject`` seam).
    probe_needs_inject = True

    #: token_faithful is OUT of the re-tune@32 (dropped from the campaign,
    #: 2026-08-27) — it keeps the historical per-batch loop, no
    #: accumulation, so legacy invocations stay byte-identical.
    @property
    def effective_batch_target(self) -> int | None:
        return None


class _ProbeVerdictMixin:
    """The solo arms' shared verdicts: their trainers have no held-out
    loop, so the quadratic ranks trials by the FINAL residual KL from the
    arm-agnostic probe, and the floor verdict is the probe's residual
    ratio against ``floor_ratio_max``."""

    def heldout_objective(self, diagnostics, final_probe):
        return final_probe.get("probe/residual_kl"), "final_residual_kl"

    def floor_verdict(self, diagnostics, final_probe):
        ratio = final_probe.get("probe/residual_ratio")
        if ratio is None:
            return None, None, None
        return bool(ratio <= self.floor_ratio_max), float(ratio), None


class PsrMseAllAdapter(_ProbeVerdictMixin, ArmAdapter):
    """Solo arm over ``train_psr(loss="mse", ...)``. Interface nuance the
    plan calls out: the mse loop resolves ``lr_schedule=None`` to its OWN
    warmup_cosine + clip-1.0 defaults, so this adapter ALWAYS passes
    explicit values — "None" is not "no schedule" for this arm, unlike the
    lockstep contract. Mid-fit probes ride the trainer's own
    ``probe_cb(step, live PSRObject)`` seam — forwarded paired with
    probe_every (the trainer refuses a half-set pair) — so psr_mse trials
    get the same mid-fit KL monitoring as the lockstep arms.

    Trials early-stop per epoch on the trainer's OWN held-out masked MSE
    (train_psr's heldout/early_stop seam — the production driver's exact
    mechanism, at the HPO patience the lockstep adapters use), and the
    trainer restores the BEST epoch before returning. Ranking is
    UNCHANGED: the held-out MSE picks the best epoch WITHIN a trial (a
    within-trial, fixed-objective comparison), while trials are still
    ranked by the T=1 arm-agnostic probe residual on the restored object
    (_ProbeVerdictMixin) — the lockstep adapters' temperature-swept
    convention."""

    name = "psr_mse_all"
    kind = "solo"
    floor_ratio_max = 1.0
    probe_via_trainer = True
    loss_key = "psr/mse_loss"
    lr_key = "psr/lr"

    def __init__(self, name: str = "psr_mse_all"):
        super().__init__()
        self._held_ex: list = []

    def prepare_data(self, setup, fit_pairs, *, length_sort=True,
                     take_longest=None, batch_size=None) -> int:
        n = super().prepare_data(setup, fit_pairs, length_sort=length_sort,
                                 take_longest=take_longest,
                                 batch_size=batch_size)
        # Eval-only sort: speed, not semantics (LockstepAdapter's rule).
        self._held_ex = (length_sorted(setup.tok, setup.held_ex)
                         if length_sort else list(setup.held_ex))
        return n

    @property
    def effective_batch_target(self) -> int | None:
        return DISTILL_EFFECTIVE_BATCH

    def make_object(self, setup, sentinel=None, lora_rank=None, seed=0):
        del seed  # zero-init by construction: no seed reaches this object
        # The registry's exact construction (methods._make_train_psr,
        # all_layers=True): every all-layer variant ZERO-inits — the
        # all-layer diffmean init gate-died on the first AdamW step.
        from sisteer.extraction.psr import PSRObject
        from sisteer.models import num_layers

        if lora_rank is not None:
            raise SystemExit(
                f"[tune-arm] lora_rank={lora_rank} for arm {self.name!r} — "
                "rank is the lora arm's search axis only")
        self._obj = PSRObject(setup.model,
                              list(range(num_layers(setup.model))),
                              train_direction=True)
        return self._obj

    def grad_params(self):
        return list(self._obj.trainable)

    def diverged_exceptions(self) -> tuple:
        from sisteer.extraction.psr import DeadGateError

        return (DeadGateError,)

    def train(self, setup, hp, *, epochs, batch_size, seed, log_fn,
              probe_cb, probe_every, diagnostics) -> None:
        import inspect

        from sisteer.extraction.distill import EarlyStopSpec
        from sisteer.extraction.lockstep import resolve_warmup_steps
        from sisteer.extraction.psr import train_psr

        if hp.temperature is not None:
            raise SystemExit(
                "[tune-arm] psr_mse_all has no temperature axis — its loss "
                "matches teacher ACTIVATIONS (train_psr's temperature is a "
                "KL-path knob); a temperature on this trial is a design "
                "bug, not a knob to silently drop")
        accum = self.accum_for(batch_size)
        total = self.optimizer_steps(batch_size, epochs)
        # warmup resolves against the ACCUM-AWARE optimizer-step budget
        # (train_psr's mse loop takes steps, not a fraction — its own total
        # equals ours by the same ceil(ceil(n/bs)/accum)*epochs formula).
        warmup_steps = resolve_warmup_steps(hp.warmup_frac, total)
        sig = inspect.signature(train_psr)
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                             for p in sig.parameters.values())
        extra: dict = {}
        if accum != 1:
            extra["accum_steps"] = accum
        if hp.weight_decay is not None:
            extra["weight_decay"] = float(hp.weight_decay)
        missing = ([] if accepts_var_kw
                   else sorted(set(extra) - set(sig.parameters)))
        if missing:
            raise SystemExit(
                f"[tune-arm] this trial sweeps {missing} but train_psr "
                "does not take the parameter(s) yet — the trainer-lane "
                "half of the re-tune@32 has not landed; refusing to "
                "silently drop a swept hyperparameter")
        self._resolved = {"warmup_steps": warmup_steps, "total_steps": total,
                          "accum_steps": accum,
                          "effective_batch": accum * int(batch_size)}
        es_diag: dict = {}
        train_psr(
            setup.model, setup.tok, self._obj, self._fit_ex, loss="mse",
            lr=float(hp.lr), epochs=epochs, batch_size=batch_size,
            max_len=setup.max_len, seed=seed, log_fn=log_fn,
            positions=setup.positions,
            response_header_len=setup.response_header_len,
            lr_schedule="warmup_cosine", warmup_steps=warmup_steps,
            max_grad_norm=float(hp.max_grad_norm),
            teacher_prefix_cache=False,
            # Per-epoch early stopping on the trainer's own held-out
            # masked MSE — the production seam, at the SAME spec the
            # lockstep adapters run (patience EARLY_STOP_PATIENCE); the
            # trainer restores the best epoch before returning. An empty
            # held-out pool degrades to the fixed budget exactly like
            # train_lockstep's contract (spec withheld, since train_psr
            # refuses a stopper with no curve to rule on).
            heldout=self._held_ex or None,
            early_stop=(EarlyStopSpec(patience=EARLY_STOP_PATIENCE,
                                      min_epochs=1, min_delta=0.0)
                        if self._held_ex else None),
            diagnostics=es_diag,
            # The trainer-side probe seam: probe_cb(step, live PSRObject)
            # right after the optimizer step. Paired Nones when the caller
            # asked for no probes (train_psr refuses a half-set pair).
            probe_cb=(probe_cb if probe_cb is not None else None),
            probe_every=(probe_every if probe_cb is not None else None),
            **extra,
            **_cfg_extras(setup.cfg, self.name))
        if diagnostics is not None:
            # Nested under the arm name — the trial summary reads
            # diag.get(arm) for best_epoch / heldout_curve / stopped_early
            # (the lockstep adapters' shape), so the stopping record lands
            # in summary.json without new plumbing.
            diagnostics.setdefault(self.name, {}).update(es_diag)


class BidpoAdapter(_ProbeVerdictMixin, ArmAdapter):
    """Solo arm over ``train_bidpo`` on preference pairs routed through the
    PRODUCTION resolver (``route_pairs`` -> driver.pairs_for_arm — the
    per-SI own-pool files when pool.bidpo_benign is on, else the anti-neg /
    strict-subset legacy routes; a raw matched-pool conversion has the
    OPPOSITE benign preference polarity, so its winners do not transfer).
    accum_steps is FIXED per trial at ceil(128 / batch_size) so the
    effective batch stays >= 128 while lr x clip sweep,
    and ``prepare_data`` trims the fit rows to a MULTIPLE of
    batch_size x accum_steps so EVERY optimizer group is full — a ragged
    final group (< 128 rows) at 2-3 epochs biases the fitted lr. The probe
    object is the TRANSIENT VectorObject the trainer wraps around the live
    vec and hands to ``probe_cb(step, obj)`` — the vec is internal to the
    trainer, so the mid-fit KL rides that seam, never a
    reimplementation."""

    name = "bidpo"
    kind = "solo"
    floor_ratio_max = 3.0  # DPO preference objective: degeneracy backstop
    probe_via_trainer = True
    needs_layer = True
    uses_pref_pool = True
    loss_key = "bidpo/loss"
    lr_key = "bidpo/lr"

    def __init__(self, name: str = "bidpo"):
        super().__init__()
        self._prefs: list = []
        self._live = None
        self._sv = None
        self.accum_steps: int | None = None
        self._data_record: dict = {}

    @property
    def effective_batch_target(self) -> int | None:
        return EFFECTIVE_BATCH_FLOOR

    @staticmethod
    def _polarity(pairs) -> tuple[int, int]:
        """(n_harmful, n_benign_reversed) by the emit-pairs requirement_a
        stamp — the polarity record every bidpo summary must carry."""
        req = _pool_builder().BIDPO_BENIGN_REQ_A
        nb = sum(1 for p in pairs
                 if getattr(p, "requirement_a", None) == req)
        return len(pairs) - nb, nb

    def route_pairs(self, setup: TuneSetup) -> list:
        """F1: the SAME resolver production uses — driver.pairs_for_arm
        over the holdout-carved own-pool sides (setup.bidpo_*_fit) with
        the section-13 cap, or its legacy routes when the feature is off.
        Raises loudly (the resolver's own errors) rather than training a
        different objective than the final fit."""
        if setup.driver is None:
            raise SystemExit(
                "[tune-arm] bidpo routing needs the production driver on "
                "TuneSetup.driver (pairs_for_arm is the ONE resolver)")
        routed = setup.driver.pairs_for_arm(
            self.name, setup.fit_pairs,
            anti_neg_enabled=setup.bidpo_anti_neg,
            bidpo_harmful_pairs=setup.bidpo_harmful_fit,
            bidpo_benign_pairs=setup.bidpo_benign_fit,
            bidpo_benign_cap=setup.bidpo_benign_cap)
        own = setup.bidpo_harmful_fit is not None
        n_h, n_b = self._polarity(routed)
        self._data_record = {
            # Production's negative_source vocabulary (train_two_si_objects
            # neg_meta), so records grep the same across both harnesses.
            "bidpo_negative_source": (
                "anti_safety_compliance_per_si_thwart" if own
                else "anti_safety_compliance" if setup.bidpo_anti_neg
                else "undefended_strict_subset"),
            "bidpo_own_pool": own,
            "n_routed_harmful": n_h,
            "n_routed_benign_reversed": n_b,
        }
        print(f"[trial] bidpo pairs routed via the production resolver "
              f"(pairs_for_arm): {n_h} harmful + {n_b} benign_reversed "
              f"(source={self._data_record['bidpo_negative_source']})")
        return routed

    def data_summary(self) -> dict:
        return dict(self._data_record)

    def prepare_data(self, setup, fit_pairs, *, length_sort=True,
                     take_longest=None, batch_size=None) -> int:
        from sisteer.data.pairs import pairs_to_pref_examples

        pairs = list(fit_pairs)
        trimmed = 0
        if take_longest is None and batch_size is not None:
            # F5: ceil(128/bs) accumulation leaves a RAGGED final optimizer
            # group unless the row count is a multiple of bs x accum (Qwen
            # 3000/bs8 -> a 56-row group every epoch). Trim to full groups
            # with the repo's deterministic sha-rank convention (no length
            # bias) so every optimizer step sees the full effective batch.
            from sisteer.extraction.subsample import sha_rank_indices

            bs = int(batch_size)
            group = bs * self.accum_for(bs)
            keep = (len(pairs) // group) * group
            if keep == 0:
                raise SystemExit(
                    f"[tune-arm] bidpo has {len(pairs)} preference rows — "
                    f"not even ONE full optimizer group (bs {bs} x accum "
                    f"{group // bs} = {group} rows); the effective-batch "
                    "floor cannot be honored on this pool")
            if keep < len(pairs):
                idx = sha_rank_indices([p.prompt for p in pairs], keep)
                pairs = [pairs[i] for i in idx]
                trimmed = len(fit_pairs) - keep
        n_h, n_b = self._polarity(pairs)
        self._data_record.update({
            "n_fit_harmful": n_h,
            "n_fit_benign_reversed": n_b,
            "n_trimmed_for_full_groups": trimmed,
        })
        prefs = pairs_to_pref_examples(setup.tok, pairs)
        if take_longest is not None:
            prefs = prefs_length_sorted(setup.tok, prefs)[-take_longest:]
        elif length_sort:
            prefs = prefs_length_sorted(setup.tok, prefs)
        self._prefs = prefs
        self._n_fit = len(prefs)
        return self._n_fit

    def sup_tokens_per_epoch(self, tok) -> int:
        from sisteer.extraction.tokcache import cached_ids

        return sum(len(cached_ids(tok, p.chosen, False))
                   + len(cached_ids(tok, p.rejected, False))
                   for p in self._prefs)

    def make_object(self, setup, sentinel=None, lora_rank=None, seed=0):
        del seed  # the vector arm's init is the builder's own (trainer-only seed)
        if lora_rank is not None:
            raise SystemExit(
                f"[tune-arm] lora_rank={lora_rank} for arm {self.name!r} — "
                "rank is the lora arm's search axis only")
        if setup.layer is None:
            raise SystemExit("[tune-arm] bidpo needs the config's `layer` "
                             "(the single intervention layer) — none set")
        return None  # the trainer owns/creates the vector

    def current_object(self):
        return self._obj if self._obj is not None else self._live

    def norm_groups(self) -> dict:
        # The trainer already logs bidpo/vec_norm per optimizer step —
        # a second reporter here would duplicate the series.
        return {}

    def diverged_exceptions(self) -> tuple:
        try:
            from sisteer.extraction.bidpo import BidpoNonFiniteError

            return (BidpoNonFiniteError,)
        except ImportError:
            return ()

    def train(self, setup, hp, *, epochs, batch_size, seed, log_fn,
              probe_cb, probe_every, diagnostics) -> None:
        from sisteer.extraction.bidpo import train_bidpo
        from sisteer.methods import VectorObject

        self.accum_steps = self.accum_for(batch_size)
        total = self.optimizer_steps(batch_size, epochs)
        # The REALIZED optimizer-batch distribution (F5): with the
        # full-group trim every group is exactly accum x bs rows; the
        # min below is the worst case over epoch shuffles (the ragged
        # micro-batch and the short final group coinciding), so it EQUALS
        # effective_batch iff no group can ever be ragged.
        micro = math.ceil(self._n_fit / batch_size)
        last_group_batches = micro % self.accum_steps or self.accum_steps
        ragged_batch_rows = self._n_fit % batch_size or batch_size
        min_group_rows = ((last_group_batches - 1) * batch_size
                          + ragged_batch_rows)
        eff = self.accum_steps * batch_size
        self._resolved = {"total_steps": total,
                          "accum_steps": self.accum_steps,
                          "effective_batch": eff,
                          "min_optimizer_group_rows": min_group_rows,
                          "optimizer_groups_full": min_group_rows == eff}

        def _cb(step: int, obj) -> None:
            self._live = obj  # the trainer's transient VectorObject
            if probe_cb is not None:
                probe_cb(step, obj)

        sv = train_bidpo(
            setup.model, setup.tok, self._prefs, setup.layer,
            lr=float(hp.lr), lr_schedule="warmup_cosine",
            warmup_frac=float(hp.warmup_frac),
            max_grad_norm=float(hp.max_grad_norm),
            accum_steps=self.accum_steps, epochs=epochs,
            batch_size=batch_size, max_len=setup.max_len, seed=seed,
            positions=setup.positions,
            response_header_len=setup.response_header_len,
            instruction_id=setup.instruction.id, log_fn=log_fn,
            probe_cb=(_cb if probe_cb is not None else None),
            probe_every=(probe_every if probe_cb is not None else None),
            **_cfg_extras(setup.cfg, self.name))
        self._sv = sv
        self._obj = VectorObject(sv)

    def save(self, out: Path) -> str | None:
        if self._sv is not None:
            self._sv.save(out / "object.safetensors")
            return "object.safetensors"
        return None


class CESoloAdapter(ArmAdapter):
    """psr_ce_all / lora_r4_ce: solo arms over
    the document-CE trainer ``train_object_ce`` on DOCUMENT pools
    (loss-masked prefix in the prompt slot, trained text in the completion
    slot — ``docs_to_ce_examples``, never ``pairs_to_distill_examples``,
    which would chat-render).

    No SI teacher exists for these arms, so the KL(prompted||unsteered)
    probe machinery is OFF entirely (``uses_kl_probe = False``): the
    trial's whole verdict is the trainer's per-epoch held-out CE on the
    SHARED held-out pool (config ``train_kwargs.default.heldout_pool`` —
    the same 128-doc pool every fit of every cell evaluates), with
    early stopping + best-epoch restore inside every trial. summary.json
    keeps the existing schema: ``heldout_objective: "heldout_ce"``,
    ``best_heldout_loss`` = best held-out CE, ``floor_ratio`` = the CE
    ratio CE(active)/CE(noop) against ``ce_floor_ratio_max`` (1.1) — the
    label rides heldout_objective, so no field is silently reinterpreted.
    Ranking by held-out CE is ranking by CE ratio (the noop denominator is
    constant per model x pool) — the HPO objective is exactly the spec's.

    Positions: all, only — the campaign trains full-document steering
    (``supports_response = False`` makes a response-leg trial refuse).
    Effective batch = the FITS' rule, read from the bound per-model config
    (``config_effective_batch``: batch_size_by_arm x accum_steps once the
    memprobe is folded — 8 for every frozen key, 64 for the Llama configs),
    else CE_EFFECTIVE_BATCH (8), so winners transfer without an lr x batch
    confound."""

    kind = "solo"
    floor_ratio_max = CE_FLOOR_RATIO_MAX
    #: no probe seam in train_object_ce and no probe wanted: the trainer's
    #: own held-out loop is the verdict (probe_via_trainer keeps run_trial
    #: from probing at the log cadence; uses_kl_probe removes the probe).
    probe_via_trainer = True
    uses_kl_probe = False
    supports_response = False
    #: Trainer-variant knobs, set by probe-mem / bench from their CLI
    #: (never by a trial — a trial reads the config's own block):
    #: loss_impl_override beats train_kwargs.<arm>.loss_impl;
    #: chunk_tokens_override (0 = trainer default); max_steps bounds the
    #: run (bench); skip_heldout drops the held-out pass (bench: a bounded
    #: run rules on nothing); accum_override replaces ceil(8 / bs).
    loss_impl_override: str | None = None
    chunk_tokens_override: int = 0
    max_steps: int | None = None
    skip_heldout: bool = False
    accum_override: int = 0
    #: The per-model config this adapter serves (bind_config: run_trial
    #: binds it before the fingerprint; prepare_data binds setup.cfg), from
    #: which the effective batch is READ (config_effective_batch). None
    #: until bound -> CE_EFFECTIVE_BATCH, the historical rule.
    cfg: dict | None = None
    #: probe-mem only: it runs on the UNFOLDED config by definition (it
    #: PRODUCES the memprobe fold), so a sentinel accum_steps is expected
    #: there and the effective batch is read off the sentinel's own EFF<n>
    #: (make_sdfbase_configs.ACCUM_SENTINEL_FMT). Every other mode refuses
    #: an unfolded config: a trial or bench must train at the fits' batch.
    allow_unfolded_config: bool = False
    #: Checkpointing: run_trial sets the trio
    #: from --checkpoint-dir / --checkpoint-every with the trial's
    #: fit_fingerprint as the key; train() threads them into
    #: train_object_ce, which resumes from the newest matching checkpoint.
    #: None = off (bench / probe-mem / every historical trial).
    checkpoint_dir: str | None = None
    checkpoint_every: int | None = None
    checkpoint_key: str | None = None
    #: The built object's trainable-parameter count (lora_modules.
    #: parameter_count) and its describe_object record, set by make_object.
    trainable_params: int | None = None
    object_desc: dict | None = None

    def __init__(self, name: str):
        if name not in ("psr_ce_all", "lora_r4_ce"):
            raise SystemExit(f"[tune-arm] CESoloAdapter serves the CE arms "
                             f"only, not {name!r}")
        super().__init__()
        self.name = name
        # train_object_ce's log_fn metric names under log_tag=<arm> — the
        # registry trainers' convention (methods._train_psr_ce /
        # _train_lora_ce pass log_tag="psr_ce_all"/"lora_r4_ce").
        self.loss_key = f"{name}/ce_loss"
        self.lr_key = f"{name}/lr"
        self._held_ex: list = []
        self._held_meta: dict = {}

    def bind_config(self, cfg: dict | None) -> None:
        """Attach the per-model config the effective batch is read from."""
        self.cfg = cfg

    def config_effective_batch(self) -> int:
        """The effective batch THIS config's fits realize:
        ``batch_size_by_arm[arm] x train_kwargs.
        <arm>.accum_steps`` when both are ints (a memprobe-folded config;
        every frozen key reproduces 8 = 1x8 / 2x4 / 4x2, the Llama configs
        read 64), so ``accum_for(bs)`` = ceil(eff / bs) = the config's own
        accum_steps and a trial trains at EXACTLY the fit's accumulation.
        A config without the generator's two keys (legacy two_si configs,
        test configs) keeps CE_EFFECTIVE_BATCH. An UNFOLDED config (the
        MEMPROBE / PENDING_MEMPROBE_EFF<n>_ACCUM sentinels) is refused —
        a trial would otherwise silently train at the historical 8 on a
        config that declared 64 — except under ``allow_unfolded_config``
        (probe-mem), where the sentinel's own EFF<n> is the answer."""
        cfg = self.cfg or {}
        bs = (cfg.get("batch_size_by_arm") or {}).get(self.name)
        acc = ((cfg.get("train_kwargs") or {}).get(self.name) or {}) \
            .get("accum_steps")
        if bs is None and acc is None:
            return CE_EFFECTIVE_BATCH

        def _is_int(v) -> bool:
            return isinstance(v, int) and not isinstance(v, bool)

        if _is_int(bs) and _is_int(acc):
            eff = int(bs) * int(acc)
            if eff < 1:
                raise SystemExit(
                    f"[tune-arm] {self.name}: batch_size_by_arm={bs} x "
                    f"accum_steps={acc} = {eff} < 1 — a broken fold")
            return eff
        m = (_ACCUM_SENTINEL_RE.fullmatch(str(acc))
             if isinstance(acc, str) else None)
        if self.allow_unfolded_config:
            return int(m.group(1)) if m else CE_EFFECTIVE_BATCH
        raise SystemExit(
            f"[tune-arm] {self.name}: the config is not memprobe-folded "
            f"(batch_size_by_arm.{self.name}={bs!r}, train_kwargs."
            f"{self.name}.accum_steps={acc!r}) — a trial / bench trains at "
            "the FITS' effective batch, known only once --fold-memprobe has "
            "landed in the config; fold first (probe-mem is the one mode "
            "that runs unfolded, and it reads the sentinel's EFF<n>)")

    @property
    def effective_batch_target(self) -> int | None:
        return self.config_effective_batch()

    def accum_for(self, batch_size: int) -> int:
        if self.accum_override:
            return int(self.accum_override)
        return super().accum_for(batch_size)

    # -- config plumbing ------------------------------------------------------
    @classmethod
    def _heldout_missing_message(cls, cfg: dict, path: str | None) -> str:
        """The refusal, by name: the config's spelling, the path as resolved on this host, and how `/vol/` resolves here."""
        tkd = dict((cfg.get("train_kwargs") or {}).get("default") or {})
        drv = _load_driver()
        return (f"[tune-arm] {cls.__name__ if isinstance(cls, type) else cls}: the shared held-out pool (config "
                f"train_kwargs.default.heldout_pool = {tkd.get('heldout_pool')!r}) is absent or missing on disk at {path!r} "
                f"(resolved here with {drv.vol_path_state()}). The held-out CE IS the HPO objective; refusing a trial with nothing to rank on")

    @staticmethod
    def _heldout_pool_path(cfg: dict) -> str | None:
        """The shared held-out pool as THIS host reads it: the config names it under the volume root (`/vol/pool/<key>/…`, the fits
        config's spelling) and a cluster node maps `/vol/` to $SISTEER_VOL_ROOT — the driver's own resolve_vol_path, the ONE mapping the
        fits use, so a trial and the fit it tunes for read the same file (stage E, 2026-09-30: every ReLU trial refused with "missing on
        disk: '/vol/pool/…'" while the file sat under $SHARED/vol/pool/…; the recon and the memprobe skip the held-out pool, so
        the trials were the first path to read it)."""
        tkd = dict((cfg.get("train_kwargs") or {}).get("default") or {})
        p = tkd.get("heldout_pool")
        return _load_driver().resolve_vol_path(str(p)) if p else None

    def _refuse_si_shaped_config(self, cfg: dict) -> dict:
        """The §5.3 config-shape guard, held at the trial level too: a CE
        arm whose config carries SI/teacher-shaped keys reads as set and
        behaves as unset — refuse instead. Returns the arm block."""
        block = dict(((cfg.get("train_kwargs") or {}).get(self.name)) or {})
        for bad in ("temperature", "teacher_prefix_cache"):
            if bad in block:
                raise SystemExit(
                    f"[tune-arm] train_kwargs.{self.name}.{bad} is set, but "
                    "the CE arms have no teacher leg — configuration that "
                    "reads as set and behaves as unset is refused")
        if block.get("floor_ratio_max") is not None:
            raise SystemExit(
                f"[tune-arm] train_kwargs.{self.name}.floor_ratio_max = "
                f"{block['floor_ratio_max']!r}: the SI-teacher floor is "
                "meaningless for CE — set it explicitly to null and use "
                "ce_floor_ratio_max (the CE-native bar)")
        if cfg.get("data_mode") not in (None, "documents"):
            raise SystemExit(
                f"[tune-arm] {self.name} requires data_mode: documents, "
                f"got {cfg.get('data_mode')!r}")
        return block

    def data_summary(self) -> dict:
        return dict(self._held_meta)

    def fingerprint_pool_extra(self, args, cfg: dict) -> str:
        path = self._heldout_pool_path(cfg)
        if path is None:
            return "|heldout_pool:MISSING"
        sha = (_config_file_sha(path) if Path(path).is_file()
               else "MISSING")
        return f"|heldout_pool:{sha}"

    # -- data -----------------------------------------------------------------
    def prepare_data(self, setup, fit_pairs, *, length_sort=True,
                     take_longest=None, batch_size=None) -> int:
        from sisteer.data.pairs import docs_to_ce_examples

        self.bind_config(setup.cfg)
        block = self._refuse_si_shaped_config(setup.cfg)
        bar = block.get("ce_floor_ratio_max", CE_FLOOR_RATIO_MAX)
        self.floor_ratio_max = float(bar)

        ex = docs_to_ce_examples(setup.tok, fit_pairs)
        if take_longest is not None:
            ex = length_sorted(setup.tok, ex)[-take_longest:]
        elif length_sort:
            ex = length_sorted(setup.tok, ex)
        self._fit_ex = ex
        self._n_fit = len(ex)

        # The SHARED held-out pool (never the per-fit hash carve — the
        # campaign configs run holdout_frac 0). probe-mem's take_longest
        # path skips it: a 4-step memory probe needs no held-out eval; so
        # does the bounded throughput bench (skip_heldout).
        if take_longest is None and not self.skip_heldout:
            tkd = dict((setup.cfg.get("train_kwargs") or {})
                       .get("default") or {})
            if float(tkd.get("holdout_frac", 0.0) or 0.0) > 0:
                raise SystemExit(
                    f"[tune-arm] {self.name}: heldout_pool and "
                    "holdout_frac > 0 together — the CE arms evaluate on "
                    "the ONE shared pool, never a per-fit carve")
            path = self._heldout_pool_path(setup.cfg)
            if path is None or not Path(path).is_file():
                raise SystemExit(self._heldout_missing_message(setup.cfg, path))
            held_pairs = setup.driver._load_pairs(path)
            held = docs_to_ce_examples(setup.tok, held_pairs)
            self._held_ex = (length_sorted(setup.tok, held)
                             if length_sort else held)
            self._held_meta = {
                "heldout_pool": path,                                    # as resolved on this host (the file that was read)
                "heldout_pool_config": dict((setup.cfg.get("train_kwargs") or {}).get("default") or {}).get("heldout_pool"),
                "vol_root_source": _load_driver().vol_root_source(),  # which root /vol/ resolved through on this host (e1abcb3b1)
                "heldout_pool_sha": _config_file_sha(path),
                "n_heldout_docs": len(held),
            }
        # Leading special tokens: this adapter runs its own optimizer
        # path (train_object_ce), so the production driver's pre-step
        # assertion does not cover it — assert the config's policy on the
        # ENCODED fit (+ held-out) examples here, exactly as
        # train_si_objects does, before any trial / probe step; an
        # undeclared policy only observes the first example.
        from sisteer.extraction.distill import _encode_aligned
        from sisteer.extraction.leading_tokens import (
            LEADING_KEY, check_leading_special_tokens)

        policy = (setup.cfg or {}).get(LEADING_KEY) or None
        # exactly what THIS call prepared: the fit examples, plus the
        # held-out ones only when this call built them (probe-mem's
        # take_longest path skips the held-out pool)
        held_ex = list(self._held_ex) if take_longest is None else []
        exs = list(self._fit_ex) + held_ex if policy else \
            list(self._fit_ex)[:1]
        try:
            lt = check_leading_special_tokens(
                setup.tok, (_encode_aligned(setup.tok, e, setup.max_len)[1]
                            for e in exs),
                policy, label=f"{self.name}: ")
            print(lt["line"], flush=True)
        except SystemExit:
            raise
        except Exception as e:  # noqa: BLE001 — observe-only path
            if policy:
                raise
            print(f"[leading-tokens] {self.name}: observation skipped "
                  f"({type(e).__name__}: {e})", flush=True)
        return self._n_fit

    # -- object ---------------------------------------------------------------
    def make_object(self, setup, sentinel=None, lora_rank=None, seed=0,
                    lora_modules=None, per_expert_rank=None, init_scheme=None):
        """The trial's object, built through the SHARED
        ``extraction.object_factory.build_object`` — the
        same builder methods._train_lora_ce / _train_psr_ce use for the
        production fits, so the search trains what the fit trains
        (tests/test_object_factory_parity.py pins the default path
        byte-identical to the historical direct construction). The
        architecture axes come from the trial row (the v2 design declares
        ``lora_rank`` / ``lora_modules`` / ``per_expert_rank`` /
        ``init_scheme`` on every row); a value the design did NOT declare
        (None) falls back to the config's ``train_kwargs.lora_r4_ce.
        lora_rank`` (4) and the paper's o_proj / zero init. The ReLU arm
        refuses any lora_* value (it has no rank or module set)."""
        from sisteer.extraction import object_factory
        from sisteer.extraction.lora_modules import parameter_count

        is_lora = self.name != "psr_ce_all"
        if not is_lora and (lora_rank is not None or lora_modules is not None
                            or per_expert_rank is not None):
            raise SystemExit(
                f"[tune-arm] lora_rank/lora_modules/per_expert_rank on a "
                f"{self.name!r} trial — the ReLU steer has no LoRA "
                "architecture; a value here is a design bug, not a knob to "
                "silently drop")
        kw: dict = {"object_kind": "lora" if is_lora else "relu_steer",
                    "init_scheme": str(init_scheme) if init_scheme is not None else "zero",
                    "seed": int(seed or 0)}
        if is_lora:
            block = dict(((setup.cfg.get("train_kwargs") or {})
                          .get(self.name)) or {})
            kw.update(
                lora_rank=(int(lora_rank) if lora_rank is not None
                           else int(block.get("lora_rank", 4))),
                lora_modules=((sorted({str(m) for m in lora_modules}) or None)
                              if lora_modules is not None else None),
                per_expert_rank=per_expert_rank)
        self._obj = object_factory.build_object(setup.model, **kw)
        self.object_desc = object_factory.describe_object(self._obj)
        # lora_modules.parameter_count: the trainable scalars of the built
        # object (the table's / fit_trial's trainable_params)
        self.trainable_params = int(parameter_count(self._obj))
        return self._obj

    def grad_params(self):
        return list(self._obj.trainable)

    def diverged_exceptions(self) -> tuple:
        from sisteer.extraction.psr import DeadGateError

        return (DeadGateError,)

    def post_train_check(self, setup) -> None:
        if self.name != "psr_ce_all":
            return
        # The dead-gate guard (load-bearing for the science): a
        # gate-dead PSR is a near-no-op, scores ce_ratio ~ 1.0 and PASSES
        # the 1.1 floor by design — it must refuse, not become a data
        # point. Runs on the RESTORED best-epoch object.
        from sisteer.extraction.psr import GUARD_PROBE_FLOOR, check_psr_gates

        check_psr_gates(setup.model, setup.tok, obj=self._obj,
                        examples=self._fit_ex, max_len=setup.max_len,
                        tag=f"tune_{self.name}",
                        min_probe=min(GUARD_PROBE_FLOOR, len(self._fit_ex)))

    # -- train ------------------------------------------------------------------
    def train(self, setup, hp, *, epochs, batch_size, seed, log_fn,
              probe_cb, probe_every, diagnostics) -> None:
        import inspect

        from sisteer.extraction.distill import EarlyStopSpec
        from sisteer.extraction.lockstep import resolve_warmup_steps

        from sisteer.extraction.objects import train_object_ce

        if hp.temperature is not None:
            raise SystemExit(
                f"[tune-arm] {self.name} has no temperature axis — CE on "
                "document tokens has no teacher distribution to soften; a "
                "temperature on this trial is a design bug, not a knob to "
                "silently drop")
        if self.name == "psr_ce_all" and (
                hp.lora_rank is not None or hp.lora_modules is not None
                or hp.per_expert_rank is not None):
            raise SystemExit(
                f"[tune-arm] {self.name} has no LoRA architecture — a "
                "lora_rank / lora_modules / per_expert_rank on this trial is "
                "a design bug")
        if hp.gate_reg is not None and self.name != "psr_ce_all":
            raise SystemExit(
                f"[tune-arm] gate_reg={hp.gate_reg:g} for arm "
                f"{self.name!r} — only psr_ce_all has a gate to "
                "regularize; a gate_reg on this trial is a design bug")

        block = self._refuse_si_shaped_config(setup.cfg)
        tkd = dict((setup.cfg.get("train_kwargs") or {}).get("default") or {})
        by_si = dict((setup.cfg.get("train_kwargs_by_si") or {})
                     .get(setup.si) or {})
        self._patience = int(by_si.get(
            "early_stop_patience",
            tkd.get("early_stop_patience", EARLY_STOP_PATIENCE)))
        self._min_epochs = int(tkd.get("early_stop_min_epochs", 1))
        self._min_delta = float(tkd.get("early_stop_min_delta", 0.0))

        accum = self.accum_for(batch_size)
        total = self.optimizer_steps(batch_size, epochs)
        warmup_steps = resolve_warmup_steps(hp.warmup_frac, total)
        wd = (float(hp.weight_decay) if hp.weight_decay is not None
              else float(block.get("weight_decay", 0.01)))
        extra: dict = {}
        if self.name == "psr_ce_all":
            # The anti-dead-gate regularizer both PSR objectives carry;
            # config train_kwargs.psr_ce_all.gate_reg overrides the 0.1,
            # and a SWEPT trial value (hp.gate_reg — the round-2 mini-sweep
            # axis) overrides the config: the design chose it per trial.
            # (A gate_reg on any OTHER arm was refused at the top of train.)
            extra["reg_coeff"] = float(block.get("gate_reg", 0.1)
                                       if hp.gate_reg is None else hp.gate_reg)
        # Trainer variant: the config's per-arm loss_impl (the production
        # fits' source) unless probe-mem / bench overrode it; chunk size /
        # step cap only when set (trainer defaults otherwise).
        impl = self.loss_impl_override or block.get("loss_impl")
        if impl:
            extra["loss_impl"] = str(impl)
        if self.chunk_tokens_override:
            extra["chunk_tokens"] = int(self.chunk_tokens_override)
        if self.max_steps:
            extra["max_steps"] = int(self.max_steps)
        if self.checkpoint_every is not None:
            # The trial checkpoints on a step interval under its fit
            # fingerprint and RESUMES from the newest matching checkpoint
            # on a restart (the trainer's own logic) — an unfinished trial
            # no longer restarts from scratch.
            if not self.checkpoint_dir or not self.checkpoint_key:
                raise SystemExit("[tune-arm] checkpoint_every without a "
                                 "checkpoint_dir + checkpoint_key")
            extra["checkpoint_every"] = int(self.checkpoint_every)
            extra["checkpoint_dir"] = self.checkpoint_dir
            extra["checkpoint_key"] = self.checkpoint_key
        sig = inspect.signature(train_object_ce)
        accepts_var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD
                             for p in sig.parameters.values())
        want = {"lr", "weight_decay", "epochs", "batch_size", "max_len",
                "seed", "log_fn", "log_tag", "accum_steps", "lr_schedule",
                "warmup_steps", "max_grad_norm", "heldout", "early_stop",
                "diagnostics", "positions", "response_header_len",
                *extra}
        missing = ([] if accepts_var_kw
                   else sorted(want - set(sig.parameters)))
        if missing:
            raise SystemExit(
                f"[tune-arm] this trial needs {missing} but "
                "train_object_ce does not take the parameter(s) — the "
                "plan-§5.1 trainer signature has drifted; refusing to "
                "silently drop a seam")
        self._resolved = {"warmup_steps": warmup_steps,
                          "total_steps": total, "accum_steps": accum,
                          "effective_batch": accum * int(batch_size),
                          "early_stop_patience": self._patience}
        es_diag: dict = {}
        train_object_ce(
            setup.model, setup.tok, self._obj, self._fit_ex,
            lr=float(hp.lr), weight_decay=wd, epochs=epochs,
            batch_size=batch_size, max_len=setup.max_len, seed=seed,
            log_fn=log_fn, log_tag=self.name, accum_steps=accum,
            lr_schedule="warmup_cosine", warmup_steps=warmup_steps,
            max_grad_norm=float(hp.max_grad_norm),
            # Per-epoch held-out CE + early stopping on the SHARED pool,
            # mandatory (standing rule); best-epoch restore inside the
            # trainer. probe-mem's heldout-less short probe degrades to
            # the fixed budget (spec withheld with no curve to rule on).
            heldout=self._held_ex or None,
            early_stop=(EarlyStopSpec(patience=self._patience,
                                      min_epochs=self._min_epochs,
                                      min_delta=self._min_delta)
                        if self._held_ex else None),
            diagnostics=es_diag,
            positions=setup.positions,
            response_header_len=setup.response_header_len,
            **extra)
        if diagnostics is not None:
            # Nested under the arm name — the trial summary reads
            # diag.get(arm) for best_epoch / heldout_curve / stopped_early
            # (the psr_mse/lockstep shape).
            diagnostics.setdefault(self.name, {}).update(es_diag)

    # -- verdicts ---------------------------------------------------------------
    def heldout_objective(self, diagnostics, final_probe):
        d = (diagnostics or {}).get(self.name) or {}
        v = d.get("best_heldout_ce")
        return (float(v) if v is not None else None), "heldout_ce"

    def floor_verdict(self, diagnostics, final_probe):
        d = (diagnostics or {}).get(self.name) or {}
        active = d.get("best_heldout_ce")
        noop = d.get("heldout_noop_ce")
        if active is None or noop is None or not noop:
            return None, None, None
        ratio = float(active) / float(noop)
        return bool(ratio <= self.floor_ratio_max), ratio, None


#: The registry — the five instruct-campaign tunable arms plus the
#: sdf-base-steering CE arms.
ADAPTERS: dict[str, type] = {
    "psr_kl_all": PsrKlAllAdapter,
    "psr_mse_all": PsrMseAllAdapter,
    "lora_r1_all": LockstepAdapter,
    "lora_r4": LockstepAdapter,
    "bidpo": BidpoAdapter,
    "token_faithful": TokenFaithfulAdapter,
    "psr_ce_all": CESoloAdapter,
    "lora_r4_ce": CESoloAdapter,
}


def make_adapter(arm: str) -> ArmAdapter:
    if arm not in ADAPTERS:
        raise SystemExit(f"[tune-arm] unknown arm {arm!r} — tunable arms: "
                         f"{sorted(ADAPTERS)}")
    return ADAPTERS[arm](arm)


# --------------------------------------------------------------------------
# probe-mem
# --------------------------------------------------------------------------

def _apply_ce_variant(adapter, args) -> None:
    """probe-mem / bench: the trainer-variant CLI knobs (--loss-impl,
    --chunk-tokens) onto the CE adapter; refused for any other arm (the
    knobs would read as set and do nothing)."""
    li = getattr(args, "loss_impl", "config") or "config"
    ct = int(getattr(args, "chunk_tokens", 0) or 0)
    if not isinstance(adapter, CESoloAdapter):
        if li != "config" or ct:
            raise SystemExit("[tune-arm] --loss-impl / --chunk-tokens are "
                             f"CE-arm knobs, not {adapter.name}'s")
        return
    if li != "config":
        adapter.loss_impl_override = li
    if ct:
        adapter.chunk_tokens_override = ct


def probe_mem(args) -> None:
    import torch
    adapter = make_adapter(args.arm)
    if isinstance(adapter, CESoloAdapter):
        # The memprobe RUNS on the unfolded config (it produces the fold):
        # read the effective batch off the accum sentinel's EFF<n>.
        adapter.allow_unfolded_config = True
    if args.positions != "all" and not adapter.supports_response:
        raise SystemExit(f"[tune-arm] {args.arm} cannot honor "
                         f"positions={args.positions!r}")
    setup = _setup(args, args.positions, arm=args.arm)
    _apply_ce_variant(adapter, args)
    # The pool THIS arm actually trains on (bidpo: the production-routed
    # preference pool) — memory must be probed on the real rows.
    pool = adapter.route_pairs(setup)
    n = max(args.batch_size * args.steps, args.batch_size)
    if args.no_worst_case:
        from sisteer.extraction.subsample import sha_rank_indices

        idx = sha_rank_indices([p.prompt for p in pool], n)
        sub = [pool[i] for i in idx]
        adapter.prepare_data(setup, sub, length_sort=True)
    else:
        # DEFAULT: the LONGEST n rows — with length-bucketed batches the
        # densest batch is all-long rows, which is the true peak-memory
        # shape; memory peaks on that batch's first optimizer step.
        adapter.prepare_data(setup, pool, take_longest=n)
    # lora probes the WORST-CASE rank by default (the largest grid level:
    # more adapter params + a wider coef activation) so one probe bounds
    # the whole rank-swept cell.
    lora_rank = None
    if args.arm in ("lora_r1_all", "lora_r4"):
        lora_rank = int(args.lora_rank or max(LORA_RANK_GRID))
        print(f"[probe-mem] lora rank={lora_rank} "
              f"({'--lora-rank' if args.lora_rank else 'worst-case grid'})")
    elif args.lora_rank:
        raise SystemExit("[probe-mem] --lora-rank is the lora arm's axis "
                         f"only, not {args.arm}'s")
    adapter.make_object(setup, lora_rank=lora_rank)
    _load_teacher_topk(args, setup, adapter)
    hp = TrialHP(lr=1e-5, warmup_frac=0.05, max_grad_norm=1.0)
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    try:
        adapter.train(setup, hp, epochs=1, batch_size=args.batch_size,
                      seed=0, log_fn=None, probe_cb=None, probe_every=0,
                      diagnostics={})
    except adapter.diverged_exceptions() as e:
        print(f"[probe-mem] short probe fit tripped its own guard "
              f"({type(e).__name__}: {e}) — memory numbers below still "
              "stand")
    dt = time.time() - t0
    steps = adapter.optimizer_steps(args.batch_size, 1)
    alloc = torch.cuda.max_memory_allocated() / 2**30 if cuda else float("nan")
    reserv = (torch.cuda.max_memory_reserved() / 2**30 if cuda
              else float("nan"))
    variant = ""
    if isinstance(adapter, CESoloAdapter):
        variant = (f" loss_impl={adapter.loss_impl_override or 'config'}"
                   f" grad_ckpt={getattr(args, 'gradient_checkpointing', 'config')}")
    print(f"[probe-mem] arm={args.arm} bs={args.batch_size} "
          f"accum={adapter.accum_for(args.batch_size)} "
          f"max_len={setup.max_len} steps={steps} "
          f"mode={'sha-sample' if args.no_worst_case else 'WORST-CASE'}"
          f"{variant} "
          f"peak_alloc={alloc:.1f}GiB peak_reserved={reserv:.1f}GiB "
          f"wall={dt:.0f}s ({dt / max(steps, 1):.2f}s/step)")


# --------------------------------------------------------------------------
# bench: throughput of the production CE train path, bounded
# --------------------------------------------------------------------------

#: Optimizer steps dropped from the steady-state window (the
#: _median_sec_per_step convention: step >= 5): the first steps carry
#: allocator warm-up and the one-off kernel selection. ONE constant with
#: the recon clip rule's "steps > 5" (recon_clip.BENCH_WARMUP_STEPS).
BENCH_WARMUP_STEPS = recon_clip.BENCH_WARMUP_STEPS

#: bench's exit codes for an explicit non-ok OUTCOME: the record is
#: written FIRST (status diverged | failed, error,
#: steps_completed, the partial grad-norm / loss series), then the process
#: exits non-zero — a driver sees both, and a MISSING record is never the
#: only sign of a failed run.
BENCH_EXIT_DIVERGED = 3
BENCH_EXIT_FAILED = 4


def _bench_hp(block: dict, args) -> tuple[TrialHP, dict]:
    """The bench's (lr, warmup_frac, max_grad_norm): a per-run CLI override
    when given (--lr / --warmup-frac / --max-grad-norm — the gradient-norm
    reconnaissance runs a handful of short fits at CHOSEN settings at the
    production effective batch), else the config block's own
    value when numeric (an HPO-folded config), else the historical defaults
    (1e-5 / 0.05 / 1.0). Returns the hp and where each value came from
    ("flag" | "config" | "default"), recorded on the bench output.

    The clip must be FINITE and positive: the trainer logs the pre-clip
    norm only under a finite clip (objects.py: ``if math.isfinite(
    max_grad_norm)``), so an ``inf`` recon run would record no norms at
    all — use a large finite value (1e6) for an effectively unclipped
    run."""
    def _cfg_num(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool)

    vals: dict = {}
    src: dict = {}
    for name, default in (("lr", 1e-5), ("warmup_frac", 0.05),
                          ("max_grad_norm", 1.0)):
        flag = getattr(args, name, None)
        if flag is not None:
            vals[name], src[name] = float(flag), "flag"
        elif _cfg_num(block.get(name)):
            vals[name], src[name] = float(block[name]), "config"
        else:
            vals[name], src[name] = default, "default"
    if not (math.isfinite(vals["max_grad_norm"]) and vals["max_grad_norm"] > 0):
        raise SystemExit(
            f"[bench] max_grad_norm={vals['max_grad_norm']!r} must be a "
            "finite positive number: the trainer logs the pre-clip grad norm "
            "ONLY under a finite clip, so an inf/nan clip leaves the "
            "reconnaissance with no norms — use a large finite value (1e6)")
    if not (math.isfinite(vals["lr"]) and vals["lr"] > 0):
        raise SystemExit(f"[bench] lr={vals['lr']!r} must be a finite "
                         "positive number")
    if not (0.0 <= vals["warmup_frac"] < 1.0):
        raise SystemExit(f"[bench] warmup_frac={vals['warmup_frac']!r} must "
                         "lie in [0, 1)")
    return TrialHP(lr=vals["lr"], warmup_frac=vals["warmup_frac"],
                   max_grad_norm=vals["max_grad_norm"]), src


def _bench_rows(recs) -> list[dict]:
    """bench's ``recs`` ([(step, payload)]) as metrics-shaped rows (the
    ``step`` key on each payload) — the shape recon_clip reads and a trial's
    metrics.jsonl writes."""
    return [{**d, "step": s_} for s_, d in recs]


def bench(args, *, setup=None) -> None:
    """Measure tok/s, s/step, GPU duty and peak VRAM of ONE (config, arm,
    batch size, loss_impl, gradient-checkpointing) setting on a STAGED
    pool — the CE adapter's production train path (train_object_ce,
    length-bucketed, pre-tokenized sidecar), bounded at --steps optimizer
    steps with NO held-out pass, NO early stopping, NO checkpoint and NO
    publication (max_steps refuses those by construction). Rates are read
    from the trainer's own per-step record (<arm>/loss_tokens,
    real_tokens, padded_tokens, sec_per_step) over steps >
    BENCH_WARMUP_STEPS; GPU SM% from the heartbeat's continuous
    nvidia-smi dmon sampler over the same window. Writes one JSON record
    to --out.

    Reconnaissance use: --lr / --max-grad-norm /
    --warmup-frac override the config block per run; the record carries
    ``grad_norm_preclip`` — the trainer's own per-step ``<arm>/grad_norm``
    (the PRE-clip total norm clip_grad_norm_ returns; no second
    measurement) with its quantiles over all steps and over steps >
    BENCH_WARMUP_STEPS — the ``loss`` series, and an explicit ``status``
    (ok | diverged | failed). A divergence or any other exception inside
    the trainer still writes the record (partial series, ``error``,
    ``steps_completed``) and then exits BENCH_EXIT_DIVERGED /
    BENCH_EXIT_FAILED — never a crash with no file, never a silently
    truncated "ok"."""
    import torch
    import statistics

    from sisteer.extraction.chunked_ce import DEFAULT_CHUNK_TOKENS
    from sisteer.extraction.fitmon import Heartbeat, TrainingDiverged
    from sisteer.extraction.objects import ce_loss_impl_parity
    from sisteer.extraction.tokcache import pool_sha256

    adapter = make_adapter(args.arm)
    if not isinstance(adapter, CESoloAdapter):
        raise SystemExit("[bench] the throughput bench serves the CE arms "
                         f"only, not {args.arm!r}")
    # ``setup`` lets the reconnaissance ladder (recon) share ONE model load
    # across its rungs; a bench call loads its own.
    setup = setup if setup is not None else _setup(args, "all", arm=args.arm)
    pool = adapter.route_pairs(setup)
    adapter.skip_heldout = True
    adapter.prepare_data(setup, pool, length_sort=True)
    # the architecture / init axes (recon at a chosen (rank, set)): passed
    # only when set, so a legacy bench call is byte-identical
    mk = {ax: TRIAL_EXTRA_AXES[ax](getattr(args, ax))
          for ax in ("lora_rank", "lora_modules", "init_scheme")
          if getattr(args, ax, None) is not None}
    adapter.make_object(setup, **mk)
    _apply_ce_variant(adapter, args)
    adapter.max_steps = int(args.steps)
    if args.accum:
        adapter.accum_override = int(args.accum)
    bs = int(args.batch_size)
    accum = adapter.accum_for(bs)
    # --steps is a budget in OPTIMIZER steps at the arm's effective batch; the
    # trainer walks `epochs` passes and max_steps stops the run. ONE pass is
    # the paper's recipe (its reconnaissance cells cover 150 steps in a pass).
    # A cell shorter than the budget (4,102 rows / effective batch 64 = 65
    # steps) stops at one epoch and the recon reader files the rung
    # `incomplete` — the front door refuses such a cell by name BEFORE
    # submitting (03_hpo.sh recon_passes_for_cell) unless the operator opts
    # into cycling (SISTEER_HPO_RECON_ALLOW_CYCLING=1), which reaches here as
    # --passes N (repeated data lowers late-step norms and biases the p95
    # clip, so it never happens silently). Default 1: byte-identical.
    per_epoch = max(1, int(adapter.optimizer_steps(bs, 1)))
    passes_needed = max(1, math.ceil(int(args.steps) / per_epoch))
    epochs = max(1, int(getattr(args, "passes", 1) or 1))
    if epochs < passes_needed:
        print(f"[bench] NOTE one pass over this cell is {per_epoch} optimizer step(s) at "
              f"effective batch {bs * accum}; the {int(args.steps)}-step budget needs "
              f"{passes_needed} pass(es) and --passes is {epochs} — the run will stop "
              f"short of the budget (the recon reader files it incomplete)", flush=True)
    block = dict(((setup.cfg.get("train_kwargs") or {}).get(args.arm)) or {})
    tkd = dict((setup.cfg.get("train_kwargs") or {}).get("default") or {})

    # A per-run override (--lr / --max-grad-norm / --warmup-frac: the
    # reconnaissance), else the config's own (HPO-folded) rate/clip/warmup
    # when numeric, else the defaults — the source of each is recorded.
    hp, hp_source = _bench_hp(block, args)
    impl = adapter.loss_impl_override or block.get("loss_impl") or "full"
    grad_ckpt = bool(block.get("gradient_checkpointing",
                               tkd.get("gradient_checkpointing", False)))
    parity = None
    if args.check_grads:
        # Two median-length rows: chunked vs full loss AND dL/dparams on
        # the real model in its real dtype — the wiring proof at scale
        # (the CPU tests prove exactness in fp32).
        ex = adapter._fit_ex
        mid = len(ex) // 2
        parity = ce_loss_impl_parity(
            setup.model, setup.tok, adapter.current_object(),
            ex[mid:mid + 2], max_len=setup.max_len,
            reg_coeff=(float(block.get("gate_reg", 0.1))
                       if args.arm == "psr_ce_all" else 0.0),
            chunk_tokens=adapter.chunk_tokens_override or DEFAULT_CHUNK_TOKENS,
            perturb=1e-3)
        print(f"[bench] loss_impl parity (2 median rows): "
              f"{json.dumps(parity)}", flush=True)

    recs: list = []
    state = {"t_first": None, "t_warm": None}
    loss_key, lr_key = adapter.loss_key, adapter.lr_key
    total = min(adapter.optimizer_steps(bs, epochs), int(args.steps))
    hb = Heartbeat(total_steps=total, interval=30.0)

    def log_fn(payload, step=None):
        if not isinstance(payload, dict):
            return
        if loss_key in payload:
            now = time.time()
            if state["t_first"] is None:
                state["t_first"] = now
            if step == BENCH_WARMUP_STEPS:
                state["t_warm"] = now
            hb.step(step, float(payload[loss_key]), payload.get(lr_key),
                    tokens=payload.get(f"{args.arm}/loss_tokens"))
        recs.append((step, dict(payload)))

    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    hb.phase(f"bench {args.arm} bs={bs} accum={accum} loss_impl={impl} "
             f"grad_ckpt={grad_ckpt} steps={args.steps} epochs={epochs} "
             f"(one pass = {per_epoch} optimizer step(s)) "
             f"docs={adapter._n_fit} max_len={setup.max_len}")
    t0 = time.time()
    diag: dict = {}
    status, err, exc_type = "ok", "", None
    try:
        adapter.train(setup, hp, epochs=epochs, batch_size=bs, seed=int(args.seed),
                      log_fn=log_fn, probe_cb=None, probe_every=0,
                      diagnostics=diag)
    except (TrainingDiverged, *adapter.diverged_exceptions()) as e:
        # The trainer's own abort (non-finite loss / grad norm / held-out
        # CE): an OUTCOME the reconnaissance must see, not a crash.
        status, err, exc_type = "diverged", str(e), type(e).__name__
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # noqa: BLE001 — an explicit FAILED outcome
        status, err, exc_type = "failed", f"{type(e).__name__}: {e}", \
            type(e).__name__
    if cuda:
        try:
            torch.cuda.synchronize()
        except Exception as e:  # noqa: BLE001 — a dead context after a CUDA fault
            if status == "ok":
                status, err, exc_type = "failed", f"{type(e).__name__}: {e}", \
                    type(e).__name__
    wall = time.time() - t0
    duty_all = hb.duty_summary()
    duty_steady = (hb.duty.window(state["t_warm"])
                   if state["t_warm"] is not None else None)
    hb.close()
    rows = _bench_rows(recs)
    steps = [(s_, d) for s_, d in recs if loss_key in d]
    steady = [d for s_, d in steps if s_ > BENCH_WARMUP_STEPS]
    if status == "ok" and not steady:
        status = "failed"
        err = (f"only {len(steps)} optimizer steps ran — need > "
               f"{BENCH_WARMUP_STEPS} for a steady window; raise --steps")
    # The pre-clip grad-norm series: the trainer's <arm>/grad_norm record
    # (objects.py logs float(clip_grad_norm_(...)) once per optimizer step
    # under a finite clip) — reused, never re-measured.
    gn_key = recon_clip.gradnorm_key(args.arm)
    gn_steps, gn_vals = recon_clip.series_from_rows(rows, gn_key)
    gn_post = [v for s_, v in zip(gn_steps, gn_vals) if s_ > BENCH_WARMUP_STEPS]
    loss_steps, loss_vals = recon_clip.series_from_rows(rows, loss_key)

    def tot(k):
        return sum(int(d[f"{args.arm}/{k}"]) for d in steady)

    if steady:
        spst = [float(d[f"{args.arm}/sec_per_step"]) for d in steady]
        secs = sum(spst)
        tl, tr, tp = tot("loss_tokens"), tot("real_tokens"), tot("padded_tokens")
        throughput = {
            "steady_train_s": round(secs, 1),
            "tokens_loss": tl, "tokens_real": tr, "tokens_padded": tp,
            "tok_per_s_loss": round(tl / secs, 1),
            "tok_per_s_real": round(tr / secs, 1),
            "tok_per_s_padded": round(tp / secs, 1),
            "pad_fraction": round(1.0 - tr / max(tp, 1), 4),
            "sec_per_step_mean": round(secs / len(steady), 3),
            "sec_per_step_median": round(statistics.median(spst), 3),
            "sec_per_step_p90": round(sorted(spst)[int(0.9 * (len(spst) - 1))],
                                      3),
        }
    else:
        throughput = {k: None for k in (
            "steady_train_s", "tokens_loss", "tokens_real", "tokens_padded",
            "tok_per_s_loss", "tok_per_s_real", "tok_per_s_padded",
            "pad_fraction", "sec_per_step_mean", "sec_per_step_median",
            "sec_per_step_p90")}
    d_arm = diag.get(args.arm) or {}
    resolved = getattr(adapter, "_resolved", None) or {}
    out = {
        "status": status, "error": err, "exception_type": exc_type,
        "model": setup.cfg.get("model"),
        "model_revision": setup.cfg.get("model_revision"),
        "device": torch.cuda.get_device_name(0) if cuda else "cpu",
        "arm": args.arm, "si_label": args.si, "pool": args.pairs,
        "pool_sha256": pool_sha256(args.pairs)[0],
        "pool_bytes": pool_sha256(args.pairs)[1], "n_docs": adapter._n_fit,
        "max_len": setup.max_len,
        "loss_impl": d_arm.get("loss_impl", impl),
        "chunk_tokens": d_arm.get("chunk_tokens"),
        "gradient_checkpointing": grad_ckpt,
        "batch_size": bs, "accum_steps": accum, "effective_batch": bs * accum,
        "steps_requested": int(args.steps), "steps_run": len(steps),
        "steps_completed": len(steps),
        "epochs_planned": int(epochs), "optimizer_steps_per_epoch": int(per_epoch),
        "passes_needed_for_budget": int(passes_needed),
        "warmup_steps_dropped": BENCH_WARMUP_STEPS,
        "steps_measured": len(steady),
        "startup_s": (round(state["t_first"] - t0, 1)
                      if state["t_first"] else None),
        "wall_s": round(wall, 1),
        **throughput,
        "gpu_duty_all": duty_all, "gpu_duty_steady": duty_steady,
        "peak_alloc_gib": (round(torch.cuda.max_memory_allocated() / 2**30, 1)
                           if cuda else None),
        "peak_reserved_gib": (round(torch.cuda.max_memory_reserved() / 2**30, 1)
                              if cuda else None),
        "chunk_check": d_arm.get("chunk_check"),
        "grad_parity": parity,
        "stopped_by_max_steps": d_arm.get("stopped_by_max_steps"),
        "lr": hp.lr, "max_grad_norm": hp.max_grad_norm,
        "warmup_frac": hp.warmup_frac,
        "warmup_steps": resolved.get("warmup_steps"),
        "total_steps_planned": resolved.get("total_steps"),
        "hp_source": hp_source,
        "grad_norm_preclip": {
            "key": gn_key, "steps": gn_steps, "values": gn_vals,
            "n_steps": len(gn_vals),
            "warmup_steps_dropped": BENCH_WARMUP_STEPS,
            "quantiles": recon_clip.quantile_summary(gn_vals),
            "quantiles_after_warmup": recon_clip.quantile_summary(gn_post),
            "note": ("the PRE-clip total norm clip_grad_norm_ returns, one "
                     "per optimizer step — the trainer's own <arm>/grad_norm "
                     "record, not a second measurement"),
        },
        "loss": {"key": loss_key, "steps": loss_steps, "values": loss_vals},
    }
    print("[bench] RESULT " + json.dumps(out), flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=1) + "\n")
        print(f"[bench] wrote {args.out}", flush=True)
    if status != "ok":
        code = BENCH_EXIT_DIVERGED if status == "diverged" else BENCH_EXIT_FAILED
        where = f"record written to {args.out}" if args.out else \
            "record printed above (no --out)"
        print(f"[bench] {status.upper()} after {len(steps)} optimizer "
              f"step(s): {err} — {where}; exit {code}", flush=True)
        raise SystemExit(code)


# --------------------------------------------------------------------------
# recon: the gradient-norm reconnaissance ladder — one process, ONE model
# load, bench-style short trainings at max_grad_norm 1e6 (the trainer logs
# the pre-clip norm only under a FINITE clip, objects.py) across an lr
# ladder spanning the box; the clip is a high percentile of the POOLED
# pre-clip norms of the STABLE rungs.
# --------------------------------------------------------------------------

RECON_UNCLIPPED = 1e6
RECON_BIMODAL_RATIO = 10.0
RECON_EXIT_NO_STABLE = 3


def _recon_alarms(per_lr: list[dict]) -> list[dict]:
    """Bimodality across lr: among the STABLE rungs, sort the certified-
    interval medians; a jump of > RECON_BIMODAL_RATIO between consecutive
    rungs is two regimes, not one distribution — reported with every
    rung's percentiles so the reader sees which rungs sit where."""
    alarms = []
    stable = sorted((x for x in per_lr if x["class"] == "stable"),
                    key=lambda x: x["lr"])
    meds = [(x["lr"], x["quantiles_certified"]["p50"]) for x in stable
            if x.get("quantiles_certified")]
    for (lr_a, m_a), (lr_b, m_b) in zip(meds, meds[1:]):
        lo, hi = sorted((m_a, m_b))
        if lo > 0 and hi / lo > RECON_BIMODAL_RATIO:
            alarms.append({"code": "grad_norm_bimodal_across_lr",
                           "text": (f"the certified-interval median pre-clip norm "
                                    f"jumps {hi / lo:.1f}x between lr={lr_a:.3e} "
                                    f"({m_a:.4g}) and lr={lr_b:.3e} ({m_b:.4g}) — "
                                    "two regimes on the ladder; the pooled "
                                    "percentile mixes them (per-lr percentiles "
                                    "reported)")})
    unstable = [x for x in per_lr if x["class"] != "stable"]
    if unstable:
        alarms.append({"code": "recon_rungs_unstable",
                       "text": ("rungs excluded from the pool: "
                                + ", ".join(f"lr={x['lr']:.3e} ({x['class']}: "
                                            f"{', '.join(x['reasons']) or 'see record'})"
                                            for x in unstable))})
    return alarms


def _recon_ladder(args) -> list[float]:
    ladder = sorted(float(x) for x in str(args.lr_ladder).split(",") if x.strip())
    if len(ladder) < 2 or any(not (math.isfinite(v) and v > 0) for v in ladder):
        raise SystemExit(f"[recon] --lr-ladder needs >= 2 finite positive rates, "
                         f"got {ladder}")
    return ladder


def _recon_rung_path(out: Path, i: int) -> Path:
    """Rung i's bench record: recon_<i>.json beside recon.json — the SAME
    path whether the ladder runs serially or as one array task per rung."""
    return out.parent / f"{out.stem}_{i}.json"


def recon_rung_args(args, ladder: list[float], i: int, out: Path) -> argparse.Namespace:
    """The bench arguments of rung i of the ladder — ONE function for the
    serial ladder and for an array task (`--rung i`), so the two are the
    same run by construction: the rung's lr, the effectively unclipped
    norm, the shared warm-up / steps / seed / init, its record path, no
    gradient parity check."""
    return argparse.Namespace(**{**vars(args), "lr": float(ladder[i]),
                                 "max_grad_norm": RECON_UNCLIPPED,
                                 "warmup_frac": float(args.warmup_frac),
                                 "steps": int(args.steps),
                                 "out": str(_recon_rung_path(out, i)),
                                 "check_grads": False})


#: The recon spec's PROTOCOL keys — what a rung's identity is a function of (the front door's recon_spec.json).
RECON_SPEC_KEYS = ("kind", "lr_ladder", "steps", "clip_percentile", "warmup_frac", "init_scheme", "si",
                   "lora_rank", "lora_modules", "cycled_passes", "effective_batch")
#: The stamp fields the fingerprint hashes (everything but the volatile phase / time / job ids).
RECON_STAMP_FIELDS = ("spec_fingerprint", "cell", "pairs_basename", "pool_sha256", "pool_bytes", "sidecar_sha256", "model",
                      "model_revision", "arm", "lora_rank", "lora_modules", "init_scheme", "steps", "seed", "passes",
                      "ladder", "rung", "lr")


def _canonical_sha256(doc) -> str:
    return hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def recon_spec_fingerprint(spec_path) -> str | None:
    """sha256 of the CANONICAL protocol content of a recon_spec.json (RECON_SPEC_KEYS only, sorted, whitespace-free), so the
    cluster's copy and the grading host's copy of the same spec agree whatever their formatting; None without a spec."""
    if not spec_path:
        return None
    doc = json.loads(sys.stdin.read() if str(spec_path) == "-" else Path(spec_path).read_text())
    return _canonical_sha256({k: doc.get(k) for k in RECON_SPEC_KEYS if k in doc})


def recon_verify_verdict(args, ladder: list[float], out: Path) -> str:
    """An EXISTING recon.json against the CURRENT reconnaissance: its spec_fingerprint is the current spec's canonical
    fingerprint (a verdict without one is a legacy verdict), every rung record beside it verifies (recon_verify_rungs), and its
    rung_fingerprints are exactly those stamps'. Refuses by name; returns the spec fingerprint on success. CPU only, no model."""
    cur = recon_spec_fingerprint(getattr(args, "spec", None))
    if cur is None:
        raise SystemExit("[recon] --verify needs --spec <the current recon_spec.json> (or - on stdin)")
    if not out.is_file():
        raise SystemExit(f"[recon] --verify REFUSED: no verdict at {out}")
    res = json.loads(out.read_text())
    got = res.get("spec_fingerprint")
    if got != cur:
        raise SystemExit(f"[recon] --verify REFUSED: {out} was reduced under another reconnaissance — spec_fingerprint "
                         f"{got[:16] if got else 'none (a legacy verdict)'} != current {cur[:16]} (cell {res.get('si')!r} vs {args.si!r}, "
                         f"ladder {res.get('ladder')} vs {ladder}, steps {res.get('steps')} vs {int(args.steps)})")
    fps = recon_verify_rungs(args, ladder, out, cfg=_load_cfg(args.config))
    if (res.get("rung_fingerprints") or {}) != fps:
        raise SystemExit(f"[recon] --verify REFUSED: {out}'s rung_fingerprints are not the stamps of the rung records beside it "
                         f"({res.get('rung_fingerprints')} vs {fps}) — a verdict reduced from other rung files")
    have = res.get("rung_result_sha256")
    if not isinstance(have, dict):
        raise SystemExit(f"[recon] --verify REFUSED: {out} carries no rung_result_sha256 — a verdict from before the result hashes (legacy); re-run --finalize")
    now = {f"lr_{i}": recon_result_sha256(json.loads(_recon_rung_path(out, i).read_text())) for i in range(len(ladder))}
    bad = sorted(k for k in now if have.get(k) != now[k])
    if bad:
        raise SystemExit(f"[recon] --verify REFUSED: the result payload of rung(s) {bad} differs from what {out} was reduced from "
                         "(a rung re-run under the same spec replaced its record) — re-run --finalize so the clip is this record's")
    # the verdict's own contract: provenance is not enough — a verdict with no stable rung carries no clip,
    # and admitting it would mark the phase DONE over nothing; finalize's exit status (RECON_EXIT_NO_STABLE) is preserved here
    # the verdict's DECISION fields must be what the verified rung records yield: the reduce is re-run here
    # over exactly those records and compared field by field — an edited recon.json (clip 1 -> 100, a rung reclassified) with the rung
    # files unchanged can never reach the design
    records = {f"lr_{i}": json.loads(_recon_rung_path(out, i).read_text()) for i in range(len(ladder))}
    paths = {k: str(_recon_rung_path(out, i)) for i, k in enumerate(records)}
    cfg = _load_cfg(args.config)
    again = recon_reduce(records, paths, ladder=ladder, steps=int(args.steps), pct=float(args.clip_percentile), arm=args.arm,
                         model=cfg.get("model"), model_revision=cfg.get("model_revision"), pairs=args.pairs, si=args.si,
                         warmup_frac=float(args.warmup_frac), lora_rank=getattr(args, "lora_rank", None),
                         lora_modules=getattr(args, "lora_modules", None), init_scheme=getattr(args, "init_scheme", None),
                         passes=int(getattr(args, "passes", 1) or 1))
    diffs = []
    for k in RECON_DECISION_FIELDS:
        if res.get(k) != again.get(k):
            diffs.append(f"{k}: verdict {res.get(k)!r} vs the records' {again.get(k)!r}")
    if [a.get("code") for a in (res.get("alarms") or [])] != [a.get("code") for a in again["alarms"]]:
        diffs.append(f"alarms: verdict {[a.get('code') for a in (res.get('alarms') or [])]} vs the records' {[a.get('code') for a in again['alarms']]}")
    if [x.get("class") for x in (res.get("per_lr") or [])] != [x.get("class") for x in again["per_lr"]]:
        diffs.append(f"per_lr classes: verdict {[x.get('class') for x in (res.get('per_lr') or [])]} vs the records' {[x.get('class') for x in again['per_lr']]}")
    if diffs:
        raise SystemExit(f"[recon] --verify REFUSED: {out}'s decision fields are not what its own verified rung records reduce to — "
                         + "; ".join(diffs) + " — an edited or foreign verdict; re-run --finalize")
    clip = res.get("clip"); stable = res.get("stable_lrs") or []
    if not (isinstance(clip, (int, float)) and not isinstance(clip, bool) and math.isfinite(clip) and clip > 0) or not stable:
        print(f"[recon] --verify REFUSED: {out} is this reconnaissance's but certifies NO clip (clip={clip!r}, stable rungs {stable}; "
              f"classes {[x.get('class') for x in res.get('per_lr') or []]}) — no stable rung on the ladder; the ladder or the cell "
              "needs the coordinator, nothing to mark DONE", flush=True)
        raise SystemExit(RECON_EXIT_NO_STABLE)
    return cur


def _recon_sidecar_sha256(pairs: str) -> str | None:
    p = Path(str(pairs) + ".tokcache.npz")
    if not p.is_file():
        return None
    from sisteer.extraction.tokcache import pool_sha256
    return pool_sha256(p)[0]


def recon_stamp(args, ladder: list[float], i: int, *, cfg: dict, with_pool: bool = True) -> dict:
    """The identity of rung i of THIS reconnaissance: the spec's fingerprint, the cell and pool (file sha256 +
    sidecar sha256), the model + revision, the arm / architecture / init, steps, seed, passes, the ladder, the rung's index and
    lr — and `fingerprint`, the sha256 of the canonical form of all of it. Written at rung start (recon_<i>.stamp.json) and at
    completion (the record's `recon_stamp`); `--finalize` verifies every rung against the CURRENT spec before reducing, so a
    rung file left by an earlier attempt (another cell, spec, model) can never pool into the clip. `with_pool=False` builds the
    expectation on a host without the pool file (the grading host reduce): pool / sidecar shas are then checked for agreement across
    the rungs and against each record's own pool_sha256 instead."""
    from sisteer.extraction.tokcache import pool_sha256
    mods = getattr(args, "lora_modules", None)
    pool_sha = pool_bytes = side = None
    if with_pool:
        pool_sha, pool_bytes = pool_sha256(args.pairs)
        side = _recon_sidecar_sha256(args.pairs)
    st = {"spec_fingerprint": recon_spec_fingerprint(getattr(args, "spec", None)),
          "cell": args.si, "pairs_basename": Path(str(args.pairs)).name,
          "pool_sha256": pool_sha, "pool_bytes": pool_bytes, "sidecar_sha256": side,
          "model": cfg.get("model"), "model_revision": cfg.get("model_revision"), "arm": args.arm,
          "lora_rank": getattr(args, "lora_rank", None),
          "lora_modules": (list(_cast_modules(mods)) if mods else None),
          "init_scheme": getattr(args, "init_scheme", None), "steps": int(args.steps), "seed": int(getattr(args, "seed", 0) or 0),
          "passes": int(getattr(args, "passes", 1) or 1), "ladder": [float(x) for x in ladder], "rung": int(i), "lr": float(ladder[i])}
    st["fingerprint"] = _canonical_sha256({k: st[k] for k in RECON_STAMP_FIELDS})
    return st


#: A rung record's DECISION-CRITICAL payload: what the reduce reads (status, lr, the steps, the pre-clip norm and loss series, the
#: error) plus its stamp's fingerprint — hashed into the verdict (rung_result_sha256) so a rung RE-RUN under the same spec that
#: replaces recon_<i>.json with other norms can never be admitted under the old verdict's clip.
RECON_RESULT_FIELDS = ("status", "lr", "steps_requested", "steps_run", "grad_norm_preclip", "loss", "error", "exception_type")


#: The verdict fields the design READS (and --verify re-derives from the rung records and compares).
RECON_DECISION_FIELDS = ("clip", "clip_raw", "n_pooled", "stable_lrs", "unstable_lrs", "certified_lr_max", "certified_lr_log10_max",
                         "ladder", "steps", "percentile", "warmup_frac", "cycled_passes")


def recon_result_sha256(rec: dict) -> str:
    payload = {k: rec.get(k) for k in RECON_RESULT_FIELDS}
    payload["stamp_fingerprint"] = (rec.get("recon_stamp") or {}).get("fingerprint")
    return _canonical_sha256(payload)


def _recon_stamp_path(out: Path, i: int) -> Path:
    return out.parent / f"{out.stem}_{i}.stamp.json"


def _recon_write_stamp_file(out: Path, i: int, st: dict, phase: str) -> None:
    doc = {**st, "phase": phase, "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID")}
    _recon_stamp_path(out, i).write_text(json.dumps(doc, indent=2) + "\n")


def _recon_run_rung(args, ladder: list[float], i: int, out: Path, *, setup) -> None:
    ns = recon_rung_args(args, ladder, i, out)
    st = recon_stamp(args, ladder, i, cfg=setup.cfg)
    _recon_write_stamp_file(out, i, st, "started")          # a dead task still says which spec it belonged to
    print(f"[recon] rung {i}: lr={ladder[i]:.3e} clip={RECON_UNCLIPPED:g} "
          f"steps={int(args.steps)} fingerprint={st['fingerprint'][:16]} cell={st['cell']}", flush=True)
    try:
        bench(ns, setup=setup)
    except SystemExit as e:
        if e.code not in (BENCH_EXIT_DIVERGED, BENCH_EXIT_FAILED):
            raise
    if not Path(ns.out).is_file():
        raise SystemExit(f"[recon] rung {i} wrote no record at {ns.out}")
    rec = json.loads(Path(ns.out).read_text())
    if rec.get("pool_sha256") not in (None, st["pool_sha256"]):
        raise SystemExit(f"[recon] rung {i}: the record's pool_sha256 {rec.get('pool_sha256')} differs from the stamp's "
                         f"{st['pool_sha256']} — the pool changed under the run; not stamping it")
    rec["recon_stamp"] = {**st, "phase": "completed", "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    Path(ns.out).write_text(json.dumps(rec, indent=2) + "\n")
    _recon_write_stamp_file(out, i, st, "completed")


def recon_verify_rungs(args, ladder: list[float], out: Path, *, cfg: dict) -> dict[str, str]:
    """Every rung record against the CURRENT spec / inputs: each carries a completed `recon_stamp` whose
    fingerprint recomputes from its fields and whose spec fingerprint, cell, pairs, model, revision, arm, architecture, init,
    steps, seed, passes, ladder, index and lr equal this reconnaissance's; the five agree on the pool's sha256 / bytes and the
    sidecar's sha256, and each record's own pool_sha256 is its stamp's. An unstamped (legacy) rung is a mismatch. Refuses by
    name — rung, field, expected, found — on the first difference. Returns {lr_i: fingerprint}."""
    want = recon_stamp(args, ladder, 0, cfg=cfg, with_pool=False)
    # the CURRENT pool's content identity: the front door passes what it read from the cluster's file
    # (--pool-sha256 / --sidecar-sha256 / --rows), and wherever the pool file is reachable it is hashed here too — every rung must
    # have trained on exactly that copy (a pool rebuilt with other content at the same path never admits an old verdict)
    cur_pool = getattr(args, "pool_sha256", None) or None
    cur_side = getattr(args, "sidecar_sha256", None) or None
    if Path(str(args.pairs)).is_file():
        from sisteer.extraction.tokcache import pool_sha256
        here_pool = pool_sha256(args.pairs)[0]; here_side = _recon_sidecar_sha256(args.pairs)
        if cur_pool and cur_pool != here_pool:
            raise SystemExit(f"[recon] REFUSED: --pool-sha256 {cur_pool[:16]} is not the pool file's {here_pool[:16]} ({args.pairs})")
        cur_pool = here_pool; cur_side = cur_side or here_side
    cur_rows = getattr(args, "rows", None)
    if cur_rows is not None and getattr(args, "spec", None):
        spec_rows = json.loads(Path(args.spec).read_text()).get("rows")
        if spec_rows is not None and int(spec_rows) != int(cur_rows):
            raise SystemExit(f"[recon] REFUSED: the cell has {cur_rows} rows now, the reconnaissance's spec recorded {spec_rows} — "
                             "the pool changed; re-run the reconnaissance")
    fps: dict[str, str] = {}; pool_seen: dict[str, tuple] = {}
    for i in range(len(ladder)):
        rec_path = _recon_rung_path(out, i)
        rec = json.loads(rec_path.read_text())
        st = rec.get("recon_stamp")
        if not isinstance(st, dict) or st.get("phase") != "completed":
            raise SystemExit(f"[recon] --finalize REFUSED: rung {i} ({rec_path}) carries no completed recon_stamp — an unstamped "
                             "or unfinished rung record (a legacy or dead task); re-run that rung under the current spec")
        if st.get("fingerprint") != _canonical_sha256({k: st.get(k) for k in RECON_STAMP_FIELDS}):
            raise SystemExit(f"[recon] --finalize REFUSED: rung {i} ({rec_path}): its recon_stamp fingerprint does not recompute "
                             "from its fields — a torn or edited stamp")
        for k in ("spec_fingerprint", "cell", "pairs_basename", "model", "model_revision", "arm", "lora_rank", "lora_modules",
                  "init_scheme", "steps", "seed", "passes", "ladder"):
            if k == "spec_fingerprint" and want[k] is None:
                continue                                      # no --spec on this side: the other fields still bind
            if st.get(k) != want[k]:
                raise SystemExit(f"[recon] --finalize REFUSED: rung {i} ({rec_path}) was run under another reconnaissance — "
                                 f"{k}: expected {want[k]!r}, found {st.get(k)!r}; archive it and re-run the rung under the current spec")
        if int(st.get("rung", -1)) != i or float(st.get("lr", -1)) != float(ladder[i]):
            raise SystemExit(f"[recon] --finalize REFUSED: rung {i} ({rec_path}) is stamped as rung {st.get('rung')} lr {st.get('lr')} — "
                             f"expected rung {i} lr {ladder[i]}; a rung file under the wrong name")
        if rec.get("pool_sha256") not in (None, st.get("pool_sha256")):
            raise SystemExit(f"[recon] --finalize REFUSED: rung {i}: the record's pool_sha256 {rec.get('pool_sha256')} is not its "
                             f"stamp's {st.get('pool_sha256')}")
        if cur_pool and st.get("pool_sha256") != cur_pool:
            raise SystemExit(f"[recon] REFUSED: rung {i} ({rec_path}) trained on another pool copy — pool_sha256 "
                             f"{str(st.get('pool_sha256'))[:16]} vs the current pool's {cur_pool[:16]}; re-run the reconnaissance on the current pool")
        if cur_side and st.get("sidecar_sha256") and st.get("sidecar_sha256") != cur_side:
            raise SystemExit(f"[recon] REFUSED: rung {i} ({rec_path}) trained under another token sidecar — sidecar_sha256 "
                             f"{str(st.get('sidecar_sha256'))[:16]} vs the current {cur_side[:16]}")
        pool_seen[f"lr_{i}"] = (st.get("pool_sha256"), st.get("pool_bytes"), st.get("sidecar_sha256"))
        fps[f"lr_{i}"] = st["fingerprint"]
    if len(set(pool_seen.values())) != 1:
        raise SystemExit(f"[recon] --finalize REFUSED: the rungs do not agree on the pool (sha256, bytes, sidecar sha256): "
                         f"{pool_seen} — rung files from different pool copies or cells")
    return fps


def recon_reduce(records: dict[str, dict], paths: dict[str, str], *, ladder: list[float],
                 steps: int, pct: float, arm: str, model, model_revision, pairs: str,
                 si: str, warmup_frac: float, lora_rank, lora_modules, init_scheme,
                 passes: int = 1) -> dict:
    """The reconnaissance verdict from the rung records: per-rung class,
    the certified-interval pre-clip norms POOLED over the STABLE rungs ->
    clip (ceil2sf of the percentile), the certified lr top, alarms. Pure
    arithmetic on the records — the serial ladder and `--finalize` (the
    array's reduce) call this one function. `clip` is None when no rung is
    stable; the caller exits RECON_EXIT_NO_STABLE."""
    per_lr: list[dict] = []
    pooled: list[float] = []
    for rid, rec in records.items():
        cls = recon_clip.classify_run(rec, recon_steps=steps)
        gn = rec.get("grad_norm_preclip") or {}
        cert = recon_clip.clip_interval_values(gn.get("steps") or [],
                                               gn.get("values") or [],
                                               recon_steps=steps)
        q = recon_clip.quantile_summary(cert) if cert else None
        row = {"run_id": rid, "lr": float(rec["lr"]), "status": rec.get("status"),
               "class": cls["class"], "eligible": cls.get("eligible"),
               "reasons": list(cls.get("reasons") or []),
               "n_certified": len(cert), "quantiles_certified": q,
               "p_target": (recon_clip.quantile(cert, pct / 100.0) if cert else None),
               "quantiles_after_warmup": gn.get("quantiles_after_warmup"),
               "steps_run": rec.get("steps_run"), "record": paths[rid]}
        per_lr.append(row)
        if cls["class"] == "stable":
            pooled.extend(cert)
    alarms = _recon_alarms(per_lr)
    stable_lrs = sorted(x["lr"] for x in per_lr if x["class"] == "stable")
    unstable_lrs = sorted(x["lr"] for x in per_lr if x["class"] != "stable")
    result: dict = {
        "arm": arm, "model": model, "model_revision": model_revision,
        "pool": pairs, "si": si, "ladder": list(ladder), "steps": int(steps),
        "percentile": pct, "unclipped_max_grad_norm": RECON_UNCLIPPED,
        "warmup_frac": float(warmup_frac),
        "lora_rank": lora_rank,
        "lora_modules": (list(_cast_modules(lora_modules)) if lora_modules else None),
        "init_scheme": init_scheme,
        "cycled_passes": int(passes),   # 1 = the paper's one pass over the cell; >1 only by the operator's opt-in
        "per_lr": per_lr, "stable_lrs": stable_lrs, "unstable_lrs": unstable_lrs,
        "certified_lr_max": (max(stable_lrs) if stable_lrs else None),
        "certified_lr_log10_max": (math.log10(max(stable_lrs)) if stable_lrs else None),
        "alarms": alarms, "records": dict(paths),
        "clip_rule": (f"ceil2sf(p{pct:g} of the POOLED certified-interval pre-clip "
                      f"grad norms — steps {recon_clip.BENCH_WARMUP_STEPS + 1}..{steps} "
                      "— over every STABLE ladder rung); unstable rungs are "
                      "diagnostic only and lower the certified lr top"),
    }
    if not pooled:
        result.update(clip=None, clip_raw=None, pooled=None, n_pooled=0)
        return result
    clip_raw = recon_clip.quantile(pooled, pct / 100.0)
    result.update(clip=recon_clip.ceil2sf(clip_raw), clip_raw=clip_raw,
                  pooled=recon_clip.quantile_summary(pooled), n_pooled=len(pooled))
    try:
        d26 = recon_clip.clip_from_records(records, arm=arm,
                                           percentile=pct / 100.0, recon_steps=steps)
        result["clip_max_of_stable_p_rule"] = {k: d26.get(k) for k in
                                               ("clip", "clip_raw", "argmax_run")}
    except (ValueError, KeyError) as e:  # a diagnostic, never the verdict
        result["clip_max_of_stable_p_rule"] = {"error": str(e)}
    return result


def _recon_write_verdict(result: dict, out: Path) -> None:
    out.write_text(json.dumps(result, indent=2) + "\n")
    if result.get("clip") is None:
        print(f"[recon] NO STABLE RUNG on the ladder — every rung is "
              f"{[x['class'] for x in result['per_lr']]}; no clip can be derived; "
              f"record at {out}", flush=True)
        raise SystemExit(RECON_EXIT_NO_STABLE)
    print(f"[recon] clip = {result['clip']!r} (p{result['percentile']:g} of {result['n_pooled']} pooled "
          f"pre-clip norms over {len(result['stable_lrs'])} stable rung(s) {result['stable_lrs']}; "
          f"raw {result['clip_raw']:.4g}); unstable rungs {result['unstable_lrs']}; certified lr max "
          f"{result['certified_lr_max']}; alarms "
          f"{[a['code'] for a in result['alarms']] or 'none'} -> {out}", flush=True)


def recon(args) -> None:
    """The gradient-norm reconnaissance. Three forms of one run:
    the serial ladder (no --rung, no --finalize: every rung in this
    process, one model load, then the reduce); ONE rung as an array task
    (--rung i: rung i's bench record only, recon_<i>.json — the reduce is
    someone else's); the reduce alone (--finalize: no model, read every
    rung's record, write recon.json — refuses while a rung's record is
    missing). The rung specs and the reduce are shared functions, so the
    three forms give the same numbers by construction."""
    if getattr(args, "spec", None) == "-":                    # the spec on stdin is read ONCE (fingerprint, verify and every stamp re-read it)
        import tempfile
        tf = tempfile.NamedTemporaryFile("w", suffix=".recon_spec.json", delete=False)
        tf.write(sys.stdin.read()); tf.close()
        args.spec = tf.name
    if getattr(args, "fingerprint", False):                  # the CPU helper the front door calls for the CURRENT spec
        fp = recon_spec_fingerprint(getattr(args, "spec", None))
        if fp is None:
            raise SystemExit("[recon] --fingerprint needs --spec <recon_spec.json> (or - on stdin)")
        print(fp)
        return
    ladder = _recon_ladder(args)
    pct = float(args.clip_percentile)
    if not (0.0 < pct < 100.0):
        raise SystemExit(f"[recon] --clip-percentile {pct} must lie in (0, 100)")
    steps = int(args.steps)
    out = Path(args.out)
    if getattr(args, "verify", False):                        # an existing verdict against the current reconnaissance
        if getattr(args, "init_scheme", None):
            _check_init_scheme(args.init_scheme)
        fp = recon_verify_verdict(args, ladder, out)
        print(f"[recon] verified: {out} is this reconnaissance's (spec {fp[:16]}, {len(ladder)} stamped rungs)", flush=True)
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    mods = getattr(args, "lora_modules", None)
    if getattr(args, "init_scheme", None):
        _check_init_scheme(args.init_scheme)
    rung = getattr(args, "rung", None)
    finalize = bool(getattr(args, "finalize", False))
    if rung is not None and finalize:
        raise SystemExit("[recon] --rung and --finalize are two different jobs; pass one")
    if rung is not None:
        if not (0 <= int(rung) < len(ladder)):
            raise SystemExit(f"[recon] --rung {rung} is outside the ladder "
                             f"(0..{len(ladder) - 1}: {ladder})")
        setup = _setup(args, "all", arm=args.arm)
        _recon_run_rung(args, ladder, int(rung), out, setup=setup)
        print(f"[recon] rung {int(rung)} record -> {_recon_rung_path(out, int(rung))} "
              f"(the reduce runs with --finalize once every rung's record exists)", flush=True)
        return
    if finalize:
        cfg = _load_cfg(args.config)
        model, model_revision = cfg.get("model"), cfg.get("model_revision")
        missing = [i for i in range(len(ladder)) if not _recon_rung_path(out, i).is_file()]
        if missing:
            raise SystemExit(f"[recon] --finalize: rung record(s) missing for rung(s) {missing} "
                             f"({[str(_recon_rung_path(out, i)) for i in missing]}) — "
                             "every rung's array task must have written its record first")
    else:
        setup = _setup(args, "all", arm=args.arm)   # ONE model load for the ladder
        cfg = setup.cfg
        model, model_revision = cfg.get("model"), cfg.get("model_revision")
        for i in range(len(ladder)):
            _recon_run_rung(args, ladder, i, out, setup=setup)
    # every rung against THIS reconnaissance before anything pools — the serial path too, one contract
    fingerprints = recon_verify_rungs(args, ladder, out, cfg=cfg)
    records: dict[str, dict] = {}
    paths: dict[str, str] = {}
    for i in range(len(ladder)):
        rec_path = _recon_rung_path(out, i)
        records[f"lr_{i}"] = json.loads(rec_path.read_text())
        paths[f"lr_{i}"] = str(rec_path)
    result = recon_reduce(records, paths, ladder=ladder, steps=steps, pct=pct, arm=args.arm,
                          model=model, model_revision=model_revision, pairs=args.pairs,
                          si=args.si, warmup_frac=float(args.warmup_frac),
                          lora_rank=getattr(args, "lora_rank", None), lora_modules=mods,
                          init_scheme=getattr(args, "init_scheme", None),
                          passes=int(getattr(args, "passes", 1) or 1))
    result["spec_fingerprint"] = recon_spec_fingerprint(getattr(args, "spec", None))
    result["rung_fingerprints"] = fingerprints
    result["rung_result_sha256"] = {k: recon_result_sha256(v) for k, v in records.items()}
    st0 = (records["lr_0"].get("recon_stamp") or {})
    result["pool_identity"] = {"pool_sha256": st0.get("pool_sha256"), "pool_bytes": st0.get("pool_bytes"),
                               "sidecar_sha256": st0.get("sidecar_sha256"), "pairs_basename": st0.get("pairs_basename")}
    if finalize:
        result["reduced_by"] = "finalize"
    _recon_write_verdict(result, out)
    for x in result["per_lr"]:
        q = x["quantiles_certified"] or {}
        print(f"  lr={x['lr']:.3e} {x['class']:10s} p50={_fmt_num(q.get('p50'))} "
              f"p90={_fmt_num(q.get('p90'))} p{pct:g}={_fmt_num(x['p_target'])} "
              f"max={_fmt_num(q.get('max'))} n={x['n_certified']}")


# --------------------------------------------------------------------------
# trial
# --------------------------------------------------------------------------

#: The statuses run_trial writes at the END of a trial — the only ones a
#: resume may skip on. Anything else in a summary.json is a torn write or a
#: foreign file, and the trial re-runs.
_TERMINAL_STATUSES = frozenset({"ok", "nan_diverged", "gate_dead",
                                "floor_failed", "ratio_diverged"})

#: Emission EVIDENCE beside a trial's summary.json: written by _record_trial
#: ONLY when the registry writer returned a record path, keyed by
#: (search_id, trial_id, fit_fingerprint); read by the matching-summary
#: resume (ensure_trial_recorded — replays a missing record from the stored
#: summary, never retrains) and by the closers (completed_trials — a
#: terminal trial without it holds the close).
#: A durable summary alone is the trial's terminal marker, not proof of a row.
TRIAL_RECORD_NAME = "fit_trial_record.json"


def _write_json_atomic(path: Path, payload: dict) -> None:
    """temp + rename, so a death mid-write never leaves a torn evidence file."""
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def trial_record_evidence(trial_dir: Path) -> dict | None:
    """The fit_trial emission evidence beside a trial's summary
    (:data:`TRIAL_RECORD_NAME`), or None when absent / unparseable."""
    p = Path(trial_dir) / TRIAL_RECORD_NAME
    if not p.is_file():
        return None
    try:
        ev = json.loads(p.read_text())
    except (ValueError, OSError):
        return None
    return ev if isinstance(ev, dict) else None


def trial_record_matches(ev: dict | None, *, search_id, trial_id: str,
                         fit_fingerprint, campaign=None) -> bool:
    """Evidence that THIS trial of THIS search, under THIS fingerprint — and,
    when the caller names one, under THIS campaign (a record written under
    another campaign is other work, replayed rather than accepted) — was
    written to the registry spool (a record path is the proof)."""
    return bool(ev) and ev.get("search_id") == str(search_id) \
        and ev.get("trial_id") == trial_id \
        and ev.get("fit_fingerprint") == fit_fingerprint \
        and bool(ev.get("record_path")) \
        and (campaign is None or ev.get("campaign") == str(campaign))

#: FIT-TIME provenance writer identity stamped into every trial summary
#: (Evals' ask, 2026-08-28). The evals lane's default-deny provenance gate
#: accepts an object only with a fit-time record carrying the model id,
#: revision and a MEASURED chat-template hash, written by the code that
#: LOADED the tokenizer to fit — ``writer`` names that fit-time writer,
#: distinguishing a real fit record from a backfilled sidecar (which never
#: loaded a tokenizer to measure the hash).
FIT_PROVENANCE_WRITER = "tune_arm.py"


def _bidpo_pool_paths(args, cfg) -> tuple[Path, Path] | None:
    """The bidpo own-pool file paths a trial would consume — the driver's
    defaulting rule (CLI override, else the emit-pairs siblings beside
    --pairs) — or None when pool.bidpo_benign is off. Used for the
    fingerprint's pool-content composite; the actual LOAD still goes
    through driver._load_bidpo_pool (sidecar sha verification included)."""
    bp = _pool_builder()
    if not bp._bidpo_benign_cfg(cfg)["enabled"]:
        return None
    h = (Path(args.bidpo_harmful_pairs) if args.bidpo_harmful_pairs
         else Path(args.pairs).parent / bp.bidpo_harmful_filename(args.si))
    b = (Path(args.bidpo_benign_pairs) if args.bidpo_benign_pairs
         else Path(args.pairs).parent / bp.bidpo_benign_filename(args.si))
    return h, b


def _validate_trials_identity(spec: dict, cfg: dict,
                              config_path: str) -> None:
    """F2: the trials payload names the canonical model repo id + revision
    + config-file sha it was emitted for; a trial refuses any other
    --config BEFORE the idempotent skip or any model load — a trial must
    never train one model while recording another."""
    missing = [k for k in ("model", "model_revision", "config_sha")
               if k not in spec]
    if missing:
        raise SystemExit(
            f"[trial] the trials file carries no canonical model identity "
            f"({missing} absent) — re-emit with `emit-trials --config` so "
            "every trial validates model+revision+config before training")
    checks = (("model", spec["model"], cfg.get("model")),
              ("model_revision", spec["model_revision"],
               cfg.get("model_revision")),
              ("config_sha", spec["config_sha"],
               _config_file_sha(config_path)))
    for key, want, got in checks:
        if want != got:
            raise SystemExit(
                f"[trial] trials file {key}={want!r} does not match "
                f"--config's {got!r} — a trial trained on one model/config "
                "must never be recorded as another; refusing")


def _teacher_topk_path(args) -> Path | None:
    """The --teacher-topk flag -> the artifact path (the sidecar convention
    beside --pairs for "auto"), or None for "off". "auto" REFUSES a missing
    sidecar rather than silently tuning a different (online-teacher)
    objective than the wave intends — pass off explicitly for an online
    cell."""
    from sisteer.extraction.teacher_topk import topk_artifact_path

    flag = str(getattr(args, "teacher_topk", "off") or "off")
    if flag == "off":
        return None
    p = (topk_artifact_path(args.pairs) if flag == "auto" else Path(flag))
    if not p.is_file() or not Path(str(p) + ".json").is_file():
        raise SystemExit(
            f"[tune-arm] --teacher-topk {flag}: no artifact at {p} (+ .json "
            "manifest) — build it first (the offline top-K teacher IS the "
            "re-tune objective), or pass --teacher-topk off for a "
            "deliberate online-teacher cell")
    return p


def _teacher_fingerprint_tag(args) -> str:
    """The teacher-leg identity for the fit fingerprint, deterministic
    BEFORE any model load: consuming a top-K head trains a DIFFERENT
    (renormalized-head) objective than the full-vocab online teacher, so
    the same hp against a different teacher leg is different work. Keyed on
    the manifest bytes (which carry model/SI/pool/max_len/top_k + row
    shas)."""
    p = _teacher_topk_path(args)
    if p is None:
        return "teacher:online"
    manifest_sha = hashlib.sha256(
        Path(str(p) + ".json").read_bytes()).hexdigest()[:16]
    return f"teacher:topk:{manifest_sha}"


def _load_teacher_topk(args, setup, adapter):
    """Load + STRICTLY validate the offline top-K teacher store for this
    trial's world and hand it to the adapter. Unlike the production
    driver's loud ONLINE fallback, a tuning cell REFUSES a mismatched
    artifact outright: half a cell silently falling back would mix two
    objectives (full-vocab online vs renormalized head) inside one design
    — analyze independently re-checks via each summary's recorded
    teacher_forward. Arms with no teacher-logit leg ignore "auto" with a
    note and refuse an explicit path."""
    p = _teacher_topk_path(args)
    if p is None:
        return None
    if not adapter.consumes_teacher_topk:
        flag = str(getattr(args, "teacher_topk", "off") or "off")
        if flag == "auto":
            print(f"[teacher-topk] {adapter.name}: no teacher-logit leg — "
                  "auto ignores the pool's sidecar for this arm")
            return None
        raise SystemExit(
            f"[tune-arm] --teacher-topk {flag}: {adapter.name} has no "
            "teacher-logit leg to replace (psr_mse matches teacher "
            "ACTIVATIONS; bidpo has no teacher forward) — pass "
            "--teacher-topk off")
    from sisteer.extraction.teacher_topk import DEFAULT_TOP_K, TeacherTopK

    store = TeacherTopK.load(p)
    problems = store.validate(
        model=setup.cfg.get("model"),
        model_revision=setup.cfg.get("model_revision"),
        si_id=args.si,
        pool_content_sha256=setup.driver.pool_content_sha256(
            setup.all_pairs),
        max_len=setup.max_len,
        top_k=DEFAULT_TOP_K,
        dtype=setup.cfg.get("dtype"))
    if problems:
        raise SystemExit(
            f"[tune-arm] teacher top-K artifact {p} does not match this "
            f"trial's world: {'; '.join(problems)} — a tuning cell never "
            "falls back to the online teacher (that would mix objectives "
            "inside one design); rebuild the artifact or pass "
            "--teacher-topk off")
    print(f"[teacher-topk] verified {p} (top_k={store.top_k}, "
          f"{len(store)} examples) — the teacher leg is the precomputed "
          "head")
    adapter.teacher_topk = store
    return store


def trial_seed(args) -> int:
    """The seed a trial runs under: ``--seed`` (default 0). Read through
    getattr so the older Namespace builders (no ``seed`` attribute) keep
    the historical seed-0 behaviour exactly."""
    return int(getattr(args, "seed", 0) or 0)


def _trial_fingerprint(args, cfg, spec, trial, arm: str, positions: str,
                       adapter) -> str:
    """The trial's fit_fingerprint (F4): the 8-key contract, with two
    DOCUMENTED composites for the extra tuning dimensions —

      pool_sha256: "pool:<sha>" plus, for bidpo, the own-pool files'
        content shas ("|bidpo_harmful:<sha>|bidpo_benign:<sha>");
      optimizer:   the trial hp {lr, lr_schedule, warmup_frac,
        max_grad_norm} + the optional axes present on the trial row
        (TRIAL_EXTRA_AXES: temperature/weight_decay/lora_rank/gate_reg —
        two designs differing only in one of them at the same index must
        never alias to the same fit) + accum_steps for any arm
        with an effective-batch target (deterministic from bs);
      config_sha:  "config:<file sha>" plus the trial-search coordinates
        and budget (round, trial index, box, warmup levels, batch size,
        epochs, fit-subsample) and the teacher leg ("teacher:online" or
        "teacher:topk:<manifest sha>" — a head-consuming fit is a
        different objective than an online one).

    Deterministic BEFORE the run (no model load), so the pre-skip check
    and the post-run summary record the same value."""
    from sisteer.extraction.fit_fingerprint import fit_fingerprint
    from sisteer.extraction.tokcache import pool_sha256

    pool_id = f"pool:{pool_sha256(args.pairs)[0]}"
    optimizer = {"lr": float(trial["lr"]), "lr_schedule": "warmup_cosine",
                 "warmup_frac": float(trial["warmup_frac"]),
                 "max_grad_norm": float(trial["max_grad_norm"])}
    for ax in TRIAL_EXTRA_AXES:  # raw row values: casting would move fps
        if ax in trial:
            optimizer[ax] = trial[ax]
    accum = adapter.accum_for(args.batch_size)
    if accum != 1:
        optimizer["accum_steps"] = accum
    if adapter.uses_pref_pool:
        paths = _bidpo_pool_paths(args, cfg)
        if paths is not None:
            for tag, p in zip(("bidpo_harmful", "bidpo_benign"), paths):
                sha = (hashlib.sha256(p.read_bytes()).hexdigest()
                       if p.exists() else "MISSING")
                pool_id += f"|{tag}:{sha}"
    # CE arms: the shared held-out pool is part of the trial's identity
    # (default "" for every legacy arm — fingerprints unchanged).
    pool_id += adapter.fingerprint_pool_extra(args, cfg)
    config_id = "|".join([
        f"config:{spec['config_sha']}",
        f"round:{spec['round']}",
        f"trial:{trial['index']}",
        f"box:{json.dumps(spec.get('box'), sort_keys=True)}",
        f"warmup_levels:{json.dumps(spec.get('warmup_levels'))}",
        f"bs:{args.batch_size}",
        f"epochs:{args.epochs}",
        f"fit_subsample:{args.fit_subsample}",
        _teacher_fingerprint_tag(args),
        # A non-default seed is part of the trial's identity (2026-09-15):
        # a seed-1 trial must never be skipped as the "MATCHING fit" of a
        # seed-0 summary of the same design. Emitted ONLY when non-zero, so
        # every fingerprint ever recorded (all at seed 0) is unchanged.
        *([f"seed:{trial_seed(args)}"] if trial_seed(args) else []),
    ])
    return fit_fingerprint({
        "model": cfg["model"], "model_revision": cfg.get("model_revision"),
        "si_id": args.si, "arm": arm, "positions": positions,
        "pool_sha256": pool_id, "optimizer": optimizer,
        "config_sha": config_id})


def _utc_now() -> str:
    """THE registry spelling of a UTC stamp (records.utc_now: ISO-8601 with
    milliseconds), so the summary's started_utc / finished_utc and the
    fit_trial row carry one spelling; the plain second-resolution form only
    on a checkout without the registry package."""
    try:
        from sisteer.registry.records import utc_now
    except ImportError:
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return utc_now()


def _trial_params(trial: dict, summary: dict) -> dict:
    """The fit_trial ``params_json`` vocabulary: lr,
    warmup_frac, max_grad_norm, plus the architecture axes when the row
    carries them (lora_rank, lora_modules, lora_set_id, per_expert_rank,
    init_scheme) and the built object's trainable_params."""
    params = {"lr": float(trial["lr"]), "warmup_frac": float(trial["warmup_frac"]),
              "max_grad_norm": float(trial["max_grad_norm"])}
    for ax in TRIAL_EXTRA_AXES:          # the optional per-trial axes, ONE owner
        if trial.get(ax) is not None:
            params[ax] = trial[ax]
    if params.get("lora_modules") is not None:
        params["lora_modules"] = sorted(params["lora_modules"])
        params["lora_set_id"] = trial.get("lora_set_id") or set_id_of(
            params["lora_modules"])
    if summary.get("trainable_params") is not None:
        params["trainable_params"] = int(summary["trainable_params"])
    return params


def _record_trial(spec: dict, trial: dict, summary: dict, out: Path
                  ) -> dict | None:
    """``record_trial`` for one terminal summary (contract §5): skipped with
    a notice when the design carries no search id (no --campaign at emit)
    or the emitter is absent. ``clip_fraction`` is the trial's own
    clip-activation rate from its metrics.jsonl (never re-measured).

    Returns the emission EVIDENCE it wrote beside the summary
    (:data:`TRIAL_RECORD_NAME`) when the
    writer returned a record path, else None — a disabled writer (no spool,
    emission off) writes no row, so no evidence is written either and the
    next resume replays the record (``ensure_trial_recorded``)."""
    sid = spec.get("search_id")
    campaign = spec.get("campaign")
    if not sid or not campaign:
        print("[registry] the design carries no search_id / campaign — "
              "fit_trial not recorded (emit-trials --campaign opens the search)")
        return None
    eh = _emit_hpo()
    if eh is None:
        return None
    # run_trial's terminal statuses (_TERMINAL_STATUSES) -> the fit_trial
    # CHECK vocabulary (complete | failed | pruned; schema_def.py:289-305):
    # "ok" is complete, every other terminal verdict a FAILED trial with its
    # reason in extra; nothing here prunes, and a status outside the
    # vocabulary is a torn summary, refused rather than filed as failed.
    st = summary.get("status")
    if st not in _TERMINAL_STATUSES:
        raise SystemExit(f"[registry] summary status {st!r} is not a terminal "
                         f"trial status {sorted(_TERMINAL_STATUSES)} — not "
                         "recording a fit_trial for it")
    status = "complete" if st == "ok" else "failed"
    rate, n_steps = _trial_clip_activation(out, summary)
    y = summary.get("best_heldout_loss")
    extra = {
        "trial_role": spec.get("trial_role"),
        "reason": summary.get("error") or None,
        "trial_status": summary.get("status"),
        "best_epoch": summary.get("best_epoch"),
        "heldout_curve": summary.get("heldout_curve"),
        "clip_n_steps": n_steps,
        "fit_fingerprint": summary.get("fit_fingerprint"),
        "round": spec.get("round"),
        "object_desc": summary.get("object_desc"),
        "confirmation_of": spec.get("confirmation_of"),
        "exploration_replicate_of": spec.get("exploration_replicate_of"),
        "confirm_pick_of": spec.get("confirm_pick_of"),
        "refine_of": spec.get("refine_of"),
        "summary_path": str(out / "summary.json"),
    }
    tid = f"r{spec['round']}_t{int(trial['index']):02d}"
    path = eh.record_trial(
        sid, tid, _trial_params(trial, summary), status=status,
        campaign=str(campaign),
        objective_value=(float(y) if y is not None and math.isfinite(float(y))
                         else None),
        clip_fraction=rate, epochs_seen=summary.get("epochs_seen"),
        wandb_run_id=summary.get("wandb_run"),
        started_utc=summary.get("started_utc"),
        finished_utc=summary.get("finished_utc"),
        extra={k: v for k, v in extra.items() if v is not None})
    if not path:
        # the writer is disabled (no spool resolves / emission off): NO row
        # was written, so no evidence is written either — said loudly; a
        # resume replays this, and completed_trials holds the close until
        # the row exists
        print(f"[registry] fit_trial {tid} NOT written for search {sid}: the "
              "record writer is disabled (no spool resolves, or emission is "
              "off) — no emission evidence; a resume replays it and the "
              "search cannot close until the row exists")
        return None
    evidence = {"search_id": str(sid), "trial_id": tid, "status": status,
                "fit_fingerprint": summary.get("fit_fingerprint"),
                "record_path": str(path), "campaign": str(campaign),
                "recorded_utc": _utc_now()}
    _write_json_atomic(out / TRIAL_RECORD_NAME, evidence)
    print(f"registry: fit_trial {tid} ({status}; search {sid})")
    return evidence


def ensure_trial_recorded(spec: dict, trial: dict, summary: dict, out: Path
                          ) -> dict | None:
    """On a matching-summary resume:
    the terminal ``fit_trial`` row is VERIFIED from its evidence (search id,
    trial id, fit fingerprint, a record path) and REPLAYED from the STORED
    summary when the evidence is missing or names other work — idempotent,
    and it never retrains (the summary carries every fact the row needs;
    the clip rate is re-read from the trial's own metrics.jsonl). None when
    the design belongs to no search (nothing to record)."""
    sid = spec.get("search_id")
    if not sid or not spec.get("campaign"):
        return None
    tid = f"r{spec['round']}_t{int(trial['index']):02d}"
    ev = trial_record_evidence(out)
    if trial_record_matches(ev, search_id=sid, trial_id=tid,
                            fit_fingerprint=summary.get("fit_fingerprint"),
                            campaign=spec.get("campaign")):
        print(f"[registry] fit_trial {tid} already recorded for search {sid} "
              f"({ev['record_path']}) — nothing to replay")
        return ev
    print(f"[registry] fit_trial {tid}: terminal summary with no matching "
          f"emission evidence at {out / TRIAL_RECORD_NAME} (the trial ended "
          "between its summary and its record, or the writer refused) — "
          "replaying the record from the stored summary (no retraining)")
    return _record_trial(spec, trial, summary, out)


def spec_with_campaign_fallback(spec: dict, campaign: str | None) -> dict:
    """``trial --campaign C``: the design's ``campaign`` when it carries one,
    else C (a replay of a follow-up design emitted before
    :func:`_art_base_payload` copied the artefact's campaign — stage E
    2026-10-01: the r2/r3 designs carried ``search_id`` alone, so
    :func:`_record_trial` skipped every follow-up trial and the close found
    no fit_trial evidence). A design that names a DIFFERENT campaign wins and
    the flag is refused by name: the record must say what the design says."""
    if not campaign:
        return spec
    have = spec.get("campaign")
    if have and str(have) != str(campaign):
        raise SystemExit(f"[trial] --campaign {campaign!r} contradicts the design's campaign {have!r} "
                         f"(round {spec.get('round')}, search {spec.get('search_id')}) — not overriding")
    if not have:
        # The flag's word is not proof — the design's parent SEARCH must belong to C: the design's own search_opening
        # names C, or the opening fit_search record for its search_id lies under <spool>/<C>/ (emit_hpo's record dir); else refused by name
        sid = spec.get("search_id")
        if not sid:
            raise SystemExit(f"[trial] --campaign {campaign!r}: the design names no search_id — a campaign cannot be asserted for a design "
                             "that belongs to no search; nothing recorded")
        proof = None
        so = spec.get("search_opening") or {}
        if isinstance(so, dict) and so.get("campaign"):
            if str(so["campaign"]) != str(campaign):
                raise SystemExit(f"[trial] --campaign {campaign!r} contradicts the design's search_opening campaign {so['campaign']!r} "
                                 f"(round {spec.get('round')}, search {sid}) — not overriding")
            proof = "the design's search_opening"
        else:
            eh = _emit_hpo()
            opening = _read_opening(eh, str(sid), str(campaign)) if eh is not None else None
            if opening and str(opening.get("search_id")) == str(sid):
                proof = f"the opening fit_search record under the {campaign} spool"
        if proof is None:
            raise SystemExit(f"[trial] --campaign {campaign!r}: the design (round {spec.get('round')}, search {sid}) carries no campaign and "
                             f"nothing proves its search belongs to {campaign!r} — no search_opening naming it, no opening fit_search record for "
                             f"{sid} under the {campaign} spool; the record must say what the search says, not what the flag says; nothing recorded")
        print(f"[trial] design carries search_id {sid} but no campaign — "
              f"--campaign {campaign} supplies it, proven by {proof} (fit_trial record_trial needs both)")
        spec = {**spec, "campaign": str(campaign)}
    return spec


def run_trial(args) -> None:
    import torch
    spec = spec_with_campaign_fallback(json.loads(Path(args.trials).read_text()), getattr(args, "campaign", None))
    trial = spec["trials"][args.index]
    out = Path(args.out) / f"r{spec['round']}_t{trial['index']:02d}"
    arm = spec.get("arm") or args.arm
    if arm is None:
        raise SystemExit("[trial] the trials file carries no 'arm' and no "
                         "--arm was given")
    if args.arm and args.arm != arm:
        raise SystemExit(f"[trial] --arm {args.arm!r} does not match the "
                         f"trials file's arm {arm!r}")
    positions = args.positions or spec.get("position", "all")
    if args.positions and spec.get("position") and \
            args.positions != spec["position"]:
        raise SystemExit(f"[trial] --positions {args.positions!r} does not "
                         f"match the trials file's {spec['position']!r}")
    adapter = make_adapter(arm)
    if positions != "all" and not adapter.supports_response:
        raise SystemExit(
            f"[trial] {arm} cannot honor positions={positions!r} — this "
            "tuning cell is a permanent hole (FIT_MASK_REFUSAL)")
    if "gate_reg" in trial and arm != "psr_ce_all":
        raise SystemExit(
            f"[trial] gate_reg={trial['gate_reg']!r} on a {arm!r} trial — "
            "only psr_ce_all carries the anti-dead-gate regularizer; every "
            "other arm would silently drop the axis and record a sweep "
            "that never happened. Refusing (before any model load).")
    # A design pinned to one data cell (the round-2 gate_reg sweep rides
    # constv2-mix-1m, not the standing HPO_SI) must never silently train
    # on another: the payload's "si" is authoritative when present.
    if spec.get("si") and args.si and args.si != spec["si"]:
        raise SystemExit(
            f"[trial] --si {args.si!r} does not match the trials file's "
            f"pinned data cell {spec['si']!r} — a sweep designed at one "
            "cell must never train at another; refusing")

    # F2: the canonical identity check comes BEFORE the idempotent skip and
    # any model load; F4: the skip itself is fingerprint-gated, so a torn,
    # non-terminal, or different-work summary.json re-runs instead of
    # permanently suppressing the trial.
    cfg = _load_cfg(args.config)
    _validate_trials_identity(spec, cfg, args.config)
    if isinstance(adapter, CESoloAdapter):
        # The fingerprint's accum_steps is the CONFIG's effective batch
        # (64 for Llama, 8 on every frozen key) — bind before it.
        adapter.bind_config(cfg)
    expected_fp = _trial_fingerprint(args, cfg, spec, trial, arm, positions,
                                     adapter)
    summary_path = out / "summary.json"
    if summary_path.exists():
        stored_status = stored_fp = None
        try:
            prev = json.loads(summary_path.read_text())
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            prev = None
        if isinstance(prev, dict):
            stored_status = prev.get("status")
            stored_fp = prev.get("fit_fingerprint")
        from sisteer.extraction.fit_fingerprint import fingerprints_match

        if stored_status in _TERMINAL_STATUSES and \
                fingerprints_match(stored_fp, expected_fp):
            # A durable summary is the terminal marker, not proof of the
            # row or the artefact — verify both before the skip. The saved
            # object is NAMED by the summary (saved before it, so a durable
            # summary implies the file); the fit_trial row is verified from
            # its evidence and replayed from the stored summary when
            # missing (never retrained).
            art = prev.get("artifact")
            if art and not (out / str(art)).is_file():
                raise SystemExit(
                    f"[trial] terminal summary at {out} names its saved object "
                    f"{art!r} but the file is missing — the summary is not "
                    "evidence of a saved trial; restore the file, or delete "
                    "the summary to re-run the trial (nothing is re-trained "
                    "silently)")
            ensure_trial_recorded(spec, trial, prev, out)
            print(f"[trial] terminal summary with a MATCHING fit "
                  f"fingerprint at {out} — skipping "
                  "(resume-after-pod-death idempotence)")
            return
        print(f"[trial] existing summary at {out} is torn, non-terminal, "
              f"or from DIFFERENT work (status={stored_status!r}, "
              f"fingerprint_match="
              f"{fingerprints_match(stored_fp, expected_fp)}) — re-running "
              "rather than trusting it")
    out.mkdir(parents=True, exist_ok=True)

    # The mid-fit KL probe, pre-clip grad tap and continuous-liveness
    # heartbeat (2026-08-25: training must log continuously) all
    # live in the shared library, fitmon.
    from sisteer.extraction.fitmon import (GradNormTap, Heartbeat,
                                           KLRatioProbe, ProbeCadence,
                                           RatioDivergenceGuard,
                                           postclip_grad_norm)

    hb = Heartbeat(total_steps=0)
    hb.phase("loading model+tokenizer, warming token cache, loading+"
             "splitting pool")
    setup = _setup(args, positions, arm=arm)

    # The arm's PRODUCTION pair routing first (bidpo: pairs_for_arm over
    # the own-pool fit sides / legacy routes; every other arm: the full
    # fit pool), THEN the tuning-budget subsample over the routed rows —
    # so bidpo's subsample thins the pool it actually trains on.
    fit_pairs = adapter.route_pairs(setup)
    if args.fit_subsample and args.fit_subsample < len(fit_pairs):
        # Exploration rounds train on a deterministic sha-rank subpool (the
        # SAME subpool for every trial — ranking configs needs a shared
        # horizon, not full-length runs). Keyed on the RAW pair prompt (the
        # split_pairs key) so the identical subpool feeds the distill-
        # example arms and bidpo's pref pool alike. The held-out slice
        # stays FULL, so heldout/probe numbers remain comparable to a
        # full-pool fit's. The winning config's confirm run passes 0.
        from sisteer.extraction.subsample import sha_rank_indices

        idx = sha_rank_indices([p.prompt for p in fit_pairs],
                               args.fit_subsample)
        fit_pairs = [fit_pairs[i] for i in idx]
        print(f"[trial] fit subsampled to {len(fit_pairs)} pairs (sha-rank)")

    n_fit = adapter.prepare_data(setup, fit_pairs,
                                 length_sort=not args.no_length_sort,
                                 batch_size=args.batch_size)
    if not args.no_length_sort:
        print("[trial] length-bucketed batches (sorted by rendered length)")
    hp = TrialHP(lr=float(trial["lr"]),
                 warmup_frac=float(trial["warmup_frac"]),
                 max_grad_norm=float(trial["max_grad_norm"]),
                 # optional axes: absent on the row -> None (not swept)
                 **{ax: cast(trial[ax])
                    for ax, cast in TRIAL_EXTRA_AXES.items() if ax in trial})
    # the v2 design's architecture / init axes ride into the factory; an
    # axis the row did not declare is not passed at all (the other arms'
    # make_object signatures know nothing of them). The bare call is kept
    # verbatim: tests/test_tune_arm.py pins that source line.
    make_kw = {ax: getattr(hp, ax)
               for ax in ("lora_modules", "per_expert_rank", "init_scheme")
               if getattr(hp, ax) is not None}
    if make_kw:
        adapter.make_object(setup, lora_rank=hp.lora_rank, seed=trial_seed(args), **make_kw)
    else:
        adapter.make_object(setup, lora_rank=hp.lora_rank, seed=trial_seed(args))
    if isinstance(adapter, CESoloAdapter):
        ck_dir = getattr(args, "checkpoint_dir", None)
        ck_every = int(getattr(args, "checkpoint_every", 0) or 0)
        if ck_dir and ck_every > 0:
            adapter.checkpoint_dir = str(ck_dir)
            adapter.checkpoint_every = ck_every
            adapter.checkpoint_key = expected_fp
            Path(ck_dir).mkdir(parents=True, exist_ok=True)
        elif ck_dir or ck_every:
            raise SystemExit("[trial] --checkpoint-dir and --checkpoint-every "
                             "come together (both, or neither)")
    _load_teacher_topk(args, setup, adapter)

    total_steps = adapter.optimizer_steps(args.batch_size, args.epochs)
    # The heartbeat counts in the TRAINER's own step units (train_lockstep
    # ticks per micro-batch; the solo trainers per optimizer step).
    hb.total = adapter.progress_steps(args.batch_size, args.epochs)

    sup_per_step = (adapter.sup_tokens_per_epoch(setup.tok)
                    / max(total_steps / args.epochs, 1))

    ratio_guard = RatioDivergenceGuard()
    cadence = ProbeCadence(probe_every=args.probe_every,
                           dense_every=args.dense_every,
                           dense_until=args.dense_until,
                           log_every=args.norms_every)
    if adapter.uses_kl_probe:
        hb.phase(f"encoding KL probe (n={args.probe_n}) + measuring the "
                 "KL(prompted||unsteered) denominator")
        # Soft-token arms carry their intervention in the PROMPT IDS: the
        # probe must encode the injected student rendering (fitmon's inject
        # seam) or obj.apply() is a silent no-op on the probe forwards.
        inject = (adapter.current_object().inject
                  if adapter.probe_needs_inject else None)
        probe = KLRatioProbe(setup.tok, setup.held_ex,
                             max_len=setup.max_len,
                             batch_size=args.batch_size, n=args.probe_n,
                             inject=inject)
        base = probe.measure(setup.model)  # denominator only (no object)
    else:
        # CE arms: no SI teacher, no KL probe — held-out CE is the whole
        # verdict (adapter docstring). Nothing to measure before training.
        probe = None
        base = {}
    hp_extras = "".join(
        f" {k}={_fmt_axis(getattr(hp, k))}"
        for k in TRIAL_EXTRA_AXES if getattr(hp, k) is not None)
    print(f"[trial t{trial['index']:02d}] arm={arm} positions={positions} "
          f"lr={hp.lr:.3e} warmup_frac={hp.warmup_frac:.3f} "
          f"clip={hp.max_grad_norm:.3f}{hp_extras} bs={args.batch_size} "
          f"accum={adapter.accum_for(args.batch_size)} "
          f"total_steps={total_steps} "
          f"noop_floor={base.get('probe/noop_floor', float('nan')):.5f}")

    model_slug = re.sub(r"[^A-Za-z0-9.]+", "-",
                        str(spec.get("model")
                            or setup.cfg["model"].split("/")[-1])).strip("-")
    run = None
    if not args.no_wandb:
        try:
            # the config's wandb_project (the configs write $WANDB_PROJECT), else $WANDB_PROJECT itself; one with no
            # value on this host disables W&B for the trial below, the message naming the variable to set
            from sisteer import site as _site
            project = _site.expand(setup.cfg.get("wandb_project") or "$WANDB_PROJECT", key="wandb_project")
            import wandb
            group = (f"hp-{arm}-{model_slug}-{setup.si}-{positions}"
                     f"-r{spec['round']}")
            # The destination entity comes from the environment first (the
            # Modal secret carries WANDB_ENTITY and the container refused
            # to start unless the key defaults to it), then the config;
            # never wandb's implicit default.
            run = wandb.init(
                project=str(project),
                entity=(os.environ.get("WANDB_ENTITY")
                        or setup.cfg.get("wandb_entity")),
                name=f"{group}-t{trial['index']:02d}",
                group=group,
                config={**trial, "total_steps": total_steps,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size, "si": setup.si,
                        "max_len": setup.max_len, "arm": arm,
                        "model": spec.get("model"), "position": positions,
                        "round": spec["round"]},
                settings=wandb.Settings(console="off", init_timeout=60))
        except Exception as e:  # W&B must never kill a trial
            print(f"[wandb] disabled: {e}")
            run = None

    # One attempt id per process, stamped on every metrics row this process
    # writes AND on the summary it publishes: the file stays append-mode (a
    # re-run's rows land after an earlier attempt's), and the stamp is what
    # lets analyze read exactly the segment the summary describes.
    attempt_id = uuid.uuid4().hex
    jsonl = (out / "metrics.jsonl").open("a")
    grad_params = adapter.grad_params()
    tap = GradNormTap(grad_params) if grad_params else None
    norm_buf = GradNormBuffer()
    state = {"step": 0, "t": time.time(), "disp_curve": [],
             "last_probe_step": -1}

    prof = None
    if args.profile:
        tb_dir = out / "tb"
        tb_dir.mkdir(exist_ok=True)
        prof = torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=20, warmup=3, active=12,
                                             repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(
                str(tb_dir)))
        prof.__enter__()

    def _emit(payload: dict, step: int):
        rec = {k: v for k, v in payload.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
        rec["step"] = step
        jsonl.write(json.dumps({**rec, recon_clip.METRICS_ATTEMPT_KEY:
                                attempt_id}) + "\n")
        jsonl.flush()
        if run is not None:
            run.log(rec)

    def _drain_gradnorms(check: bool = True):
        for s, pre, post in norm_buf.drain(check=check):
            rec = {}
            if not math.isnan(pre):
                rec["train/grad_norm_preclip"] = pre
            if not math.isnan(post):
                rec["train/grad_norm_postclip"] = post
            if rec:
                _emit(rec, s)

    def _probe_now(step: int, obj) -> None:
        """One dual-KL probe at ``step``; deduped (the bidpo trainer's
        probe_cb cadence and the log-cadence check can coincide)."""
        if probe is None or obj is None or step == state["last_probe_step"]:
            return
        state["last_probe_step"] = step
        _drain_gradnorms()
        kl = probe.measure(setup.model, obj)
        state["disp_curve"].append(
            (step, kl.get("probe/displacement_ratio"),
             kl.get("probe/residual_ratio")))
        dr = kl.get("probe/displacement_ratio")
        rr = kl.get("probe/residual_ratio")
        print(f"  step {step}: disp_ratio={dr:.3f} resid_ratio={rr:.3f}")
        _emit(kl, step)
        for k in ("probe/residual_ratio", "probe/displacement_ratio"):
            v = kl.get(k)
            if v is not None and not math.isfinite(v):
                raise TrialDiverged(f"step {step}: {k}={v}")
        # The residual-ratio hard-stop: >1.5 on 2
        # consecutive probe evals = the object is worse than no-op — kill
        # the trial rather than burn its budget. Same guard as the fit
        # path (fitmon.RatioDivergenceGuard); raises TrainingDiverged.
        ratio_guard.update(rr)

    def probe_cb(step: int, obj) -> None:
        # The trainer-side seam (bidpo): called at every probe_every-th
        # optimizer step with the TRANSIENT VectorObject; the dense/sparse
        # schedule is applied here (caller-set cadence).
        if cadence.due(step):
            _probe_now(step, obj)

    def log_fn(payload, step=None):
        step = state["step"] if step is None else step
        if adapter.loss_key not in payload:
            if any("heldout" in k for k in payload):
                hb.phase(f"held-out eval done (epoch boundary, step {step})"
                         " — training resumes")
                _drain_gradnorms()
            _emit(payload, step)
            return
        state["step"] = step
        loss_val = payload[adapter.loss_key]
        hb.step(step, loss_val, payload.get(adapter.lr_key))
        now = time.time()
        dt = now - state["t"]
        state["t"] = now
        extra = {
            "train/sec_per_step": dt,
            "train/supervised_tokens_per_s": sup_per_step / max(dt, 1e-9),
        }
        # Grad norms: accumulate 0-dim GPU tensors, sync ONLY at cadence
        # (the gpu-util rule — a per-step .item() on every norm drops duty
        # cycle exactly like the ragged-shape cuDNN replanning footgun).
        if tap is not None:
            norm_buf.append(step, tap.flush_tensor(),
                            postclip_grad_norm(grad_params))
        if cadence.log_due(step) or step <= 3:
            _drain_gradnorms()
            extra.update(adapter.norm_groups())
        if not adapter.probe_via_trainer and cadence.due(step):
            _probe_now(step, adapter.current_object())
        _emit({**payload, **extra}, step)
        if prof is not None and step <= 40:
            prof.step()
        if math.isnan(loss_val) or math.isinf(loss_val):
            raise TrialDiverged(f"step {step}: loss={loss_val}")

    diag: dict = {}
    status, err = "ok", ""
    cuda = torch.cuda.is_available()
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    started_utc = _utc_now()
    hb.phase(f"{arm} train: encode {n_fit} fit rows (+ held-out), startup "
             f"probes BEFORE step 0 — first step typically lands a few "
             f"minutes in; {total_steps} optimizer steps planned")
    from sisteer.extraction.distill import FloorRatioError
    from sisteer.extraction.fitmon import (
        TrainingDiverged as _LibTrainingDiverged)
    from sisteer.extraction.psr import DeadGateError
    try:
        adapter.train(setup, hp, epochs=args.epochs,
                      batch_size=args.batch_size, seed=trial_seed(args),
                      log_fn=log_fn,
                      probe_cb=probe_cb,
                      probe_every=math.gcd(args.dense_every,
                                           args.probe_every),
                      diagnostics=diag)
        dv = (diag.get(arm) or {}).get("diverged")
        if dv:
            # train_lockstep ISOLATES a diverged arm (marks it in the
            # diagnostics and returns, so production group partners train
            # on) — but a TRIAL is that one arm, so its divergence is the
            # trial's own terminal verdict. Surfacing it here routes the
            # step-divergence guard (and any probe_cb abort) through the
            # same clean nan_diverged summary publish as a raised
            # TrialDiverged; without this a diverged lockstep trial would
            # publish a malformed "ok".
            raise TrialDiverged(dv)
        adapter.post_train_check(setup)
    except TrialDiverged as e:
        status = ("ratio_diverged"
                  if str(e).startswith(RatioDivergenceGuard.PREFIX)
                  else "nan_diverged")
        err = str(e)
    except _LibTrainingDiverged as e:
        # The library trainers' own step-level abort (fitmon.
        # TrainingDiverged — the step-divergence guard, and train_psr's
        # probe_cb contract). A DIFFERENT class from this file's local
        # TrialDiverged; without this clause a guard verdict inside
        # train_psr would crash the trial instead of publishing the
        # terminal nan_diverged summary.
        status, err = "nan_diverged", str(e)
    except DeadGateError as e:
        status, err = "gate_dead", str(e)
    except FloorRatioError as e:
        status, err = "floor_failed", str(e)
    except adapter.diverged_exceptions() as e:
        status, err = "nan_diverged", f"{type(e).__name__}: {e}"
    finally:
        try:
            _drain_gradnorms(check=False)
        except Exception:  # noqa: BLE001 — drain must not mask the verdict
            pass
        if tap is not None:
            tap.remove()
        if prof is not None:
            prof.__exit__(None, None, None)
    wall = time.time() - t0

    hb.phase(f"training done ({status}); final KL probe + saving artifacts")
    final_kl = (probe.measure(setup.model, adapter.current_object())
                if probe is not None and status == "ok"
                and adapter.current_object() is not None
                else {})
    y_val, y_label = adapter.heldout_objective(diag, final_kl)
    floor_ok, floor_ratio, floor_sem = adapter.floor_verdict(diag, final_kl)
    d = diag.get(arm, {})
    resolved = getattr(adapter, "_resolved", {})
    from sisteer.extraction.tokcache import template_hash_of

    # The saved object FIRST: the
    # summary is the terminal marker a resume skips on, so it must not become
    # durable before the artefact it vouches for; it names the file it saved
    # ("artifact"), and a resume refuses a summary whose named file is gone.
    saved_artifact = adapter.save(out) if status == "ok" else None
    summary = {
        "trial": trial, "round": spec["round"], "status": status,
        "error": err, "artifact": saved_artifact,
        "arm": arm, "model": spec.get("model"),
        "model_revision": spec.get("model_revision"),
        # FIT-TIME PROVENANCE: an HPO trial IS a fit, so its summary must
        # carry the fit-driver-equivalent record train_two_si_objects.py
        # stamps into train_budget_meta.json — without it a promoted HPO
        # winner fails the evals default-deny provenance gate, which
        # requires the MEASURED chat-template hash written by the
        # tokenizer-loading fit code. template_hash is measured on the
        # tokenizer THIS fit actually loaded (never re-derived from config
        # at read time), with the canonical helper — sha256 16-hex prefix
        # over the raw chat_template string, the exact
        # train_budget_meta/capeval convention, so fit-time and serve-time
        # hashes compare directly (e.g. NousResearch@d10aef79
        # b48c47f644389271 vs meta-llama@0e9e39f2 e10ca381b1ccc5cf —
        # nominally the same model, different prompt bytes).
        "template_hash": template_hash_of(setup.tok),
        "writer": FIT_PROVENANCE_WRITER,
        "position": positions,
        "si": setup.si,
        # F4: the identity the resume skip is gated on — computed BEFORE
        # the run from the same fields, so pre-skip and post-run agree.
        "fit_fingerprint": expected_fp,
        # The segment of metrics.jsonl this summary describes: every row
        # this process wrote carries the same stamp.
        recon_clip.METRICS_ATTEMPT_KEY: attempt_id,
        "box": spec.get("box"), "warmup_levels": spec.get("warmup_levels"),
        # A confirmation design's stamp rides into its summary:
        # the fold writer reads it as evidence for the promoted point and
        # analyze refuses to pool it. Absent on every search trial.
        **({"confirmation_of": spec["confirmation_of"]}
           if is_confirmation_design(spec) else {}),
        "epochs": args.epochs, "fit_subsample": args.fit_subsample,
        "n_fit": n_fit, "length_sorted": not args.no_length_sort,
        "batch_size": args.batch_size, "max_len": setup.max_len,
        # The seed the trial RAN under (--seed, default 0) — beside the
        # design row, so a promoted winner's provenance says which seed its
        # search was (the seed replicate searches at seed 1).
        "seed": trial_seed(args),
        "warmup_frac": float(trial["warmup_frac"]),
        # The optional axes, echoed flat (they also ride in "trial") so
        # analyze-side greps and W&B summaries see them first-class.
        **{k: trial[k] for k in TRIAL_EXTRA_AXES if k in trial},
        **resolved,
        # The teacher leg the TRAINING steps really ran (train_lockstep
        # records it per arm; a row-binding fallback would surface here and
        # analyze refuses a mixed cell). Absent for solo/legacy arms.
        **({"teacher_forward": d["teacher_forward"]}
           if d.get("teacher_forward") is not None else {}),
        **({"resolved_warmup_steps": d["resolved_warmup_steps"]}
           if d.get("resolved_warmup_steps") is not None else {}),
        **adapter.data_summary(),
        "steps_run": state["step"], "wall_s": wall,
        "peak_alloc_gib": (torch.cuda.max_memory_allocated() / 2**30
                           if cuda else None),
        "best_heldout_loss": y_val,
        "heldout_objective": y_label,
        "best_epoch": d.get("best_epoch"),
        "heldout_curve": d.get("heldout_curve"),
        "stopped_early": d.get("stopped_early"),
        "noop_floor": d.get("noop_floor",
                            final_kl.get("probe/noop_floor",
                                         probe.denom if probe is not None
                                         else None)),
        "final_probe_loss": d.get("final_probe_loss"),
        "floor_ratio": floor_ratio,
        "floor_ratio_sem": floor_sem,
        "floor_ratio_max": adapter.floor_ratio_max,
        "floor_ok": floor_ok,
        "displacement_curve": state["disp_curve"],
        **{k.replace("probe/", "final_"): v for k, v in final_kl.items()},
        "wandb_run": getattr(run, "id", None),
        # The built object's trainable-parameter count, the search this
        # trial belongs to, its role (an exploration replicate is never a
        # search row), the epochs actually seen (held-out evaluations
        # after the untrained one) and the wall-clock bounds the fit_trial
        # record carries.
        "trainable_params": getattr(adapter, "trainable_params", None),
        "object_desc": getattr(adapter, "object_desc", None),
        "search_id": spec.get("search_id"),
        "trial_role": spec.get("trial_role"),
        "started_utc": started_utc,
        "finished_utc": _utc_now(),
        "epochs_seen": (max(len(d["heldout_curve"]) - 1, 0)
                        if d.get("heldout_curve") else None),
        **({"exploration_replicate_of": spec["exploration_replicate_of"]}
           if is_replicate_design(spec) else {}),
        **({"confirm_pick_of": spec["confirm_pick_of"]}
           if is_confirm_pick_design(spec) else {}),
        **({"refine_of": spec["refine_of"]} if spec.get("refine_of") else {}),
        # the v2 design's identity rides every summary so analyze can read
        # the allowed pairs, the box and the set order off the evidence
        # itself (absent on every legacy design -> absent keys)
        **{k: spec[k] for k in ("design", "design_seed", "allowed_pairs",
                                "excluded_pairs", "lr_log10_box", "set_order",
                                "design_extended_to_32", "recon", "campaign",
                                "search_opening", "per_expert_rank_rule",
                                "coverage_repair")
           if k in spec},
    }
    # Atomic terminal marker (F4): temp + rename, so a pod dying mid-write
    # can never leave a torn summary.json that suppresses the retry.
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    tmp.replace(out / "summary.json")
    # The registry's fit_trial, emitted ONCE at the terminal outcome —
    # after the summary is durable, so a crash between
    # the two leaves a summary the next attempt re-reads, never a record
    # without its evidence.
    _record_trial(spec, trial, summary, out)
    if run is not None:
        run.summary.update({k: v for k, v in summary.items()
                            if isinstance(v, (int, float, bool, str))
                            and v is not None})
        run.finish()
    hb.close()
    print(f"[trial t{trial['index']:02d}] {status} "
          f"{y_label}={summary['best_heldout_loss']} "
          f"floor_ratio={summary['floor_ratio']} "
          f"(bar {adapter.floor_ratio_max}) "
          f"disp_ratio={summary.get('final_displacement_ratio')} "
          f"wall={wall:.0f}s -> {out}", flush=True)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("emit-trials")
    e.add_argument("--out", required=True)
    e.add_argument("--arm", required=True, choices=sorted(ADAPTERS))
    e.add_argument("--config", required=True,
                   help="per-model config yaml — the canonical model repo "
                        "id + revision + config sha ride into the payload, "
                        "and `trial` refuses any other --config")
    e.add_argument("--position", default="all", choices=["all", "response"])
    e.add_argument("--n", type=int, default=None,
                   help="total trials (split evenly across the two warmup "
                        "levels; per-level count must be a power of 2). "
                        "Default 32 for the re-tune arms (joint Sobol over "
                        "the expanded axes), 16 for the legacy 2-axis "
                        "arms, or the previous fit's 'next.n' with "
                        "--box-from")
    e.add_argument("--design-salt", type=int, default=0,
                   help="deterministic re-draw of the digital shifts (only "
                        "if a draw fails the emit-time balance "
                        "assertions); 0 = the historical shifts")
    e.add_argument("--pin-axis", action="append", default=[],
                   metavar="AXIS=VALUE",
                   help="hold one of the arm's extra axes (ARM_AXES) at a "
                        "single grid value for a follow-up round — the "
                        "axis leaves the Sobol design and the value is "
                        "stamped on every trial row (e.g. temperature=1.0 "
                        "for the psr_kl_all/lora_r1_all rounds after the "
                        "T!=1 top-K-head divergence). Repeatable; the "
                        "value must sit on the axis's grid; lr/"
                        "max_grad_norm refuse")
    e.add_argument("--addendum-from", default=None,
                   help="D409: the v2 analyze artefact refit over the WIDENED menu (--extends-menu); one trial per "
                        "--pairs item at its predicted lr (else the set's best round-1 lr), round 4, trial_role addendum")
    e.add_argument("--pairs", default=None, help="with --addendum-from: rank:set,… — the pairs the re-verdict certified")
    e.add_argument("--round", type=int, default=None,
                   help="the design's round: 1 for a seed design (the "
                        "default), the next round for --box-from, "
                        f"{CONFIRMATION_ROUND} (reserved) for --confirm-from")
    e.add_argument("--box-from", default=None,
                   help="an analyze fit .json whose 'next' block supplies "
                        "the box (refine/recenter rounds)")
    e.add_argument("--confirm-pick-from", default=None,
                   help="hpo_pick's confirm-pick request (hpo_pick_confirm_"
                        "request.json): emit the ONE-row design at the user's "
                        "picked (rank, set) at its inferred learning rate, "
                        f"round {CONFIRM_PICK_ROUND}, trial_role confirm_pick — "
                        "its measured clip-activation rate decides the "
                        "promotion")
    e.add_argument("--confirm-from", default=None,
                   help="an analyze fit .json with an INTERIOR fitted "
                        "argmin: emit the ONE-row confirmation design at "
                        "exactly that point (sdfbase-llama8b plan §4.4, "
                        "review finding 5 — the promoted point must be one "
                        "that actually ran). Round defaults to "
                        f"{CONFIRMATION_ROUND}; the box, warmup levels and "
                        "data cell are the artefact's; takes no other "
                        "design flag; refused on an edge / non-PD artefact")
    e.add_argument("--lr-min", type=float, default=None)
    e.add_argument("--lr-max", type=float, default=None)
    e.add_argument("--clip-min", type=float, default=None)
    e.add_argument("--clip-max", type=float, default=None)
    e.add_argument("--clip-const", type=float, default=None,
                   help="D26 LR-only design (sdfbase-llama8b): hold "
                        "max_grad_norm at this CONSTANT on every row — the "
                        "recon-derived per-arm clip (plan §4.2) — so the "
                        "box's clip side is [c, c], log(lr) is the single "
                        "Sobol axis, --n defaults to 8, and analyze holds "
                        "the clip fixed. REQUIRED for a seed box that "
                        "declares no clip range (the llama CE rows); "
                        "contradicts --clip-min/--clip-max")
    e.add_argument("--si", default=None,
                   help="pin the design to ONE data cell (payload 'si'): "
                        "run_trial refuses a contradicting --si and the "
                        "Modal spawner reads it (_design_si) — the "
                        "sdfbase-llama8b search rides constv2-mix-10m, not "
                        "the standing HPO_SI")
    e.add_argument("--warmup-levels", default="0.05,0.15")
    # The LoRA architecture search design, its inputs, the refine /
    # exploration follow-ups and the registry open.
    e.add_argument("--design", default="legacy",
                   choices=["legacy", "sobol16_coverage"],
                   help="legacy = build_trials (lr [x clip] per warm-up "
                        "level; the LR-only ReLU search with --clip-const / "
                        "--recon and ONE --warmup-levels value); "
                        "sobol16_coverage = build_trials_v2, the LoRA "
                        "architecture search over --allowed-pairs")
    e.add_argument("--allowed-pairs", default=None,
                   help="the serving probe's allowed_pairs.json (the design "
                        "space P; required by --design sobol16_coverage)")
    e.add_argument("--recon", default=None,
                   help="a `tune_arm.py recon` recon.json: its `clip` is the "
                        "held constant (alternative to --clip-const)")
    e.add_argument("--init-scheme", default=None,
                   help="zero | random_lecun_1e-3 (FD 7): stamped on every "
                        "row and built through object_factory")
    e.add_argument("--lr-log10-min", type=float, default=None,
                   help="v2 design: log10 lr box bottom (default: the "
                        "per-model seed box)")
    e.add_argument("--lr-log10-max", type=float, default=None,
                   help="v2 design: log10 lr box top (default: the "
                        "per-model seed box; the recon's certified rung)")
    e.add_argument("--refine-from", default=None,
                   help="an `analyze --design v2` artefact: emit the THREE "
                        "refine trials at the selected pair on +-0.5 decade "
                        f"around u* (round {REFINE_ROUND}; plan §4 phase 6)")
    e.add_argument("--explore-from", default=None,
                   help="a pair-only refit artefact on the FALLBACK path: "
                        "emit the ONE exploration replicate at the promoted "
                        f"measured trial's coordinates (round "
                        f"{EXPLORATION_ROUND}; never a confirmation)")
    e.add_argument("--campaign", default=None,
                   help="open the registry fit_search under this campaign "
                        "(emit_hpo.open_search; the id rides the payload)")
    e.add_argument("--model-id", default=None,
                   help="the registry model_id of the FIT checkpoint "
                        "(default: the config model's slug)")
    e.add_argument("--trial-epochs", type=int, default=3,
                   help="recorded in search_space_json.trial_budget")
    e.set_defaults(fn=emit_trials)

    def _common(p):
        p.add_argument("--config", default=str(REPO / "configs/two_si.yaml"))
        p.add_argument("--pairs", required=True)
        p.add_argument("--si", default="si_constitution",
                       help="all tuning runs on si_constitution (plan §6)")
        p.add_argument("--batch-size", type=int, default=8)
        p.add_argument("--allow-tokenize", action="store_true")
        p.add_argument("--bidpo-harmful-pairs", default=None,
                       help="bidpo own-pool harmful jsonl (default: the "
                            "emit-pairs sibling bidpo_harmful_<si>.jsonl "
                            "beside --pairs); loaded through the driver's "
                            "own sha-verified loader")
        p.add_argument("--bidpo-benign-pairs", default=None,
                       help="bidpo own-pool REVERSED-polarity benign jsonl "
                            "(default: the sibling bidpo_benign_<si>.jsonl "
                            "beside --pairs)")
        p.add_argument("--teacher-topk", default="off",
                       help="offline top-K teacher store for the lockstep "
                            "KL arms: 'auto' (the <pairs>.teachertopk."
                            "safetensors sidecar, REFUSING if absent), a "
                            "path, or 'off' (the online teacher — the "
                            "historical objective). The re-tune@32 wave "
                            "runs auto; a mismatched artifact refuses "
                            "rather than falling back (a cell must train "
                            "ONE objective)")

    m = sub.add_parser("probe-mem")
    _common(m)
    m.add_argument("--arm", required=True, choices=sorted(ADAPTERS))
    m.add_argument("--positions", default="all",
                   choices=["all", "response"])
    m.add_argument("--steps", type=int, default=30)
    m.add_argument("--no-worst-case", action="store_true",
                   help="probe a sha-rank sample instead of the LONGEST "
                        "rows. Default is WORST-CASE: under length-bucketed "
                        "batching the densest (all-long) batch is the true "
                        "peak-memory shape")
    m.add_argument("--lora-rank", type=int, default=None,
                   help="lora_r1_all only: probe at this LoRA rank "
                        "(default: the largest grid level — the worst-case "
                        "footprint bounds the whole rank-swept cell)")

    def _ce_variant(p):
        # Trainer-variant knobs, probe-mem + bench only (a trial reads
        # the config's own train_kwargs.<arm>.loss_impl).
        p.add_argument("--loss-impl", default="config",
                       choices=["config", "full", "chunked_ce"],
                       help="CE loss implementation override for the CE "
                            "arms: 'config' = the config's own block "
                            "(default 'full')")
        p.add_argument("--chunk-tokens", type=int, default=0,
                       help="chunked_ce rows per LM-head chunk (0 = the "
                            "trainer default)")
        p.add_argument("--gradient-checkpointing", default="config",
                       choices=["config", "on", "off"],
                       help="override the config's per-arm "
                            "gradient_checkpointing for THIS process only")

    _ce_variant(m)
    m.set_defaults(fn=probe_mem)

    b = sub.add_parser("bench", help="R2-D62: bounded throughput measurement "
                       "of the CE train path (never publishes)")
    _common(b)
    b.add_argument("--arm", required=True, choices=["psr_ce_all", "lora_r4_ce"])
    b.add_argument("--steps", type=int, default=200,
                   help="optimizer steps to run (steady window = steps > "
                        f"{BENCH_WARMUP_STEPS})")
    b.add_argument("--accum", type=int, default=0,
                   help="accumulation steps (0 = ceil(8 / batch_size))")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--out", default="", help="write the JSON record here")
    b.add_argument("--check-grads", action="store_true",
                   help="also run chunked-vs-full loss+gradient parity on "
                        "two median-length rows before training")
    b.add_argument("--lr", type=float, default=None,
                   help="per-run learning rate (the gradient-norm "
                        "reconnaissance, sdfbase-llama8b plan §4.2); default: "
                        "the config block's own value when numeric, else 1e-5")
    b.add_argument("--max-grad-norm", type=float, default=None,
                   help="per-run clip — FINITE (the trainer logs the pre-clip "
                        "norm only under a finite clip; 1e6 for an "
                        "effectively unclipped recon run); default: the "
                        "config block's, else 1.0")
    b.add_argument("--warmup-frac", type=float, default=None,
                   help="per-run warmup fraction; default: the config "
                        "block's, else 0.05")
    _ce_variant(b)
    b.set_defaults(fn=bench)

    t = sub.add_parser("trial")
    _common(t)
    t.add_argument("--trials", required=True)
    t.add_argument("--campaign", default=None,
                   help="the registry campaign when the design carries none (a replay of a follow-up design emitted before it inherited the artefact's campaign); a design naming another campaign refuses it")
    t.add_argument("--index", type=int, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--arm", default=None, choices=sorted(ADAPTERS),
                   help="validated against the trials file's arm")
    t.add_argument("--positions", default=None,
                   choices=["all", "response"],
                   help="validated against the trials file's position")
    t.add_argument("--epochs", type=int, default=3,
                   help="the re-tune@32 lockstep cells run 15 (early "
                        "stopping at patience 1 makes the cap cheap); "
                        "psr_mse_all's trainer has no held-out loop, so "
                        "its budget stays a deliberate per-invocation "
                        "choice")
    t.add_argument("--no-length-sort", action="store_true",
                   help="keep the trainers' unsorted batch composition "
                        "(slower: ragged pads + per-shape attention cost)")
    t.add_argument("--fit-subsample", type=int, default=0,
                   help="train on a deterministic sha-rank subpool of this "
                        "many fit pairs (0 = full pool); heldout stays "
                        "full either way")
    t.add_argument("--seed", type=int, default=0,
                   help="the trainer seed the trial runs under (epoch "
                        "shuffle; the LoRA A-factor init on the CE LoRA "
                        "arm) — recorded in summary.json as `seed`. Default "
                        "0 = the historical behaviour of every trial before "
                        "2026-09-15; the seed-replicate lane's search passes "
                        "its config's train_kwargs.default.seed")
    t.add_argument("--probe-n", type=int, default=64)
    t.add_argument("--probe-every", type=int, default=250)
    t.add_argument("--dense-every", type=int, default=25)
    t.add_argument("--dense-until", type=int, default=250)
    t.add_argument("--norms-every", type=int, default=25)
    t.add_argument("--profile", action="store_true")
    t.add_argument("--no-wandb", action="store_true")
    t.add_argument("--checkpoint-dir", default=None,
                   help="CE arms: checkpoint the trial here on a step "
                        "interval under its fit fingerprint and resume from "
                        "the newest matching checkpoint (plan §4 phase 4)")
    t.add_argument("--checkpoint-every", type=int, default=0,
                   help="optimizer steps between checkpoints (0 = off; "
                        "comes with --checkpoint-dir)")
    t.set_defaults(fn=run_trial)

    r = sub.add_parser("recon", help="gradient-norm reconnaissance (plan §4 "
                       "phase 2): bench-style short trainings at an "
                       "effectively unclipped 1e6 across an lr ladder; the "
                       "clip is a high percentile of the pooled pre-clip norm")
    _common(r)
    r.add_argument("--arm", required=True, choices=["psr_ce_all", "lora_r4_ce"])
    r.add_argument("--out", required=True, help="recon.json (per-lr records "
                   "land beside it as recon_<i>.json)")
    r.add_argument("--lr-ladder", required=True,
                   help="comma-separated learning rates spanning the box "
                        "(5 values)")
    r.add_argument("--steps", type=int, default=recon_clip.RECON_STEPS)
    r.add_argument("--clip-percentile", type=float, default=95.0,
                   help="percentile of the pooled pre-clip norm (0-100)")
    r.add_argument("--warmup-frac", type=float, default=0.15)
    r.add_argument("--accum", type=int, default=0)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--lora-rank", type=int, default=None)
    r.add_argument("--lora-modules", default=None,
                   help="a set id (wo|qkvo|attn_mlp|mlp|down) or a comma list")
    r.add_argument("--init-scheme", default=None)
    r.add_argument("--passes", type=int, default=1,
                   help="passes over the cell a rung may take to reach --steps "
                        "(1 = the paper's one epoch; >1 ONLY when the front door "
                        "recorded cycled_passes under the operator's opt-in, D123)")
    r.add_argument("--spec", default=None,
                   help="the front door's recon_spec.json this run realises; its canonical fingerprint stamps every rung "
                        "record and --finalize refuses a rung stamped under another spec")
    r.add_argument("--pool-sha256", default=None,
                   help="the CURRENT pool file's sha256 as the front door read it on the cluster; every rung must "
                        "have trained on exactly that copy (--finalize / --verify)")
    r.add_argument("--sidecar-sha256", default=None, help="the CURRENT token sidecar's sha256 (same contract)")
    r.add_argument("--rows", type=int, default=None, help="the CURRENT cell's row count; must equal the spec's recorded rows")
    r.add_argument("--fingerprint", action="store_true",
                   help="print the canonical fingerprint of --spec and exit (CPU; the front door's "
                        "current-spec check, D133)")
    r.add_argument("--verify", action="store_true",
                   help="verify the EXISTING --out recon.json against --spec and the rung records beside "
                        "it (spec fingerprint, every stamp, rung_fingerprints); refuses by name (D133)")
    r.add_argument("--rung", type=int, default=None,
                   help="train ONLY this rung of the ladder (0-based; an "
                        "array task) and write recon_<i>.json — no reduce")
    r.add_argument("--finalize", action="store_true",
                   help="the reduce alone: read every rung's recon_<i>.json "
                        "(refuses while one is missing) and write recon.json; "
                        "loads no model")
    _ce_variant(r)
    r.set_defaults(fn=recon)

    a = sub.add_parser("analyze")

    a.add_argument("--config", default=None,

                   help="(with --close-search) the config FILE the search was opened on: its full sha256 is recorded as fit_search.config_sha256 and must agree with the opening's value on the 16-hex prefix; without it the opening's value passes through (a non-64-hex is recorded as unverified)")
    a.add_argument("--out", required=True)
    a.add_argument("--rounds", default="1",
                   help="comma-separated rounds to pool (the winner refit "
                        "pools BOTH rounds: --rounds 1,2)")
    a.add_argument("--y-key", default="best_heldout_loss")
    a.add_argument("--design", default="legacy", choices=["legacy", "v2"],
                   help="v2 = the additive model over u = log10(lr) with "
                        "rank / module-set effects (the LoRA architecture "
                        "search) or the plain quadratic in u (the fixed-"
                        "warm-up ReLU search); plan §4 phase 5")
    a.add_argument("--allowed-pairs", default=None,
                   help="v2: the allowed_pairs.json (default: the design's "
                        "own copy on the summaries) — the argmin is taken "
                        "over these pairs only")
    a.add_argument("--source-record", default=None,
                   help="the record this analysis is derived on, as <cluster>:<record dir> — recorded in the artefact; hpo_table refuses a "
                        "derived refit whose source record is not the table's (a residual refit on a shared mirror after a --cluster change)")
    a.add_argument("--fit-name", default=None,
                   help="write the artefact under this file name in --out instead of r<rounds>_fit[_pair].json — the widened-menu "
                        "refit (--extends-menu) writes r1_fit_menu.json and leaves the canonical r1_fit.json untouched on every exit")
    a.add_argument("--extends-menu", action="store_true",
                   help="v2: accept an --allowed-pairs menu that is a SUPERSET of the one the pooled design was drawn "
                        "from (a re-verdict widened it; D409) — the refit predicts the new pairs; recorded as "
                        "menu_extended_from / menu_extension in the artefact")
    a.add_argument("--pair-only", action="store_true",
                   help="v2 refine refit: pool only the selected pair's rows "
                        "(from the r2 refine_of block) and judge the argmin "
                        "against the REFINED box")
    a.add_argument("--close-search", action="store_true",
                   help="v2: close the registry fit_search with this "
                        "artefact's winner (the ReLU search's final refit; "
                        "the LoRA search closes at hpo_pick)")
    a.set_defaults(fn=analyze)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
