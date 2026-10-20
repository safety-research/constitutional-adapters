"""Train several two-SI Class A objects for ONE SI in a SINGLE process, with
ONE model load, using lockstep teacher sharing (efficiency optimisations 3
and 4).

The frozen base is identical across every cell of the matrix (same model, same
revision) — only the trained object differs. The single-cell path loads the 8B
base once PER array task, i.e. ~22 loads. This entry point loads it ONCE for an
SI and trains all its requested arms in one process:

  - the lockstep-COMPATIBLE arms (those consuming the SI-prompted teacher's
    logit distribution — see ``extraction.lockstep``) are trained together in a
    lockstep GROUP, so the teacher forward runs ONCE per batch and feeds every
    arm rather than being recomputed per arm (opt. 4). Bitwise-identical to
    training each solo;
  - the INCOMPATIBLE arms (mse / hard / closed-form / probe fits) are trained
    SOLO through the ordinary registry trainer, in the same process — still one
    model load (opt. 3), never coerced into the group.

Sequence budget and batch size are per-(SI[, arm]) via ``eval.train_budget``:
per-SI ``max_len`` (a memory knob — the completion cap stays shared, so
comparability is untouched), and per-cell batch size with the fisher rung's
O(T^2) hard cap. A lockstep group shares one batch size (it must, for the
teacher batches to align) — the safe shared width across its arms.

This is ADDITIVE: it does not touch ``scripts/two_si_run.py`` (transcript
generation / aggregate) or the single-SI ``scripts/build_safety_defense.py`` /
``scripts/stage2_run.py`` training paths — those keep working unchanged. The
objects this writes land at exactly the cell-id-keyed paths
(``<defenses_dir>/<si>__<arm>/object...``) that ``two_si_run generate`` loads.

Subcommands:
  plan   — CPU only. Print the per-arm partition (lockstep group vs solo),
           resolved per-SI max_len and per-cell batch size, and the
           measurement-harness budget table. No pairs, no training, no GPU.
  train  — On a GPU node. Load the model once and train the arms for one SI
           (lockstep group + solo remainder), saving each object to its
           cell-id path. Pairs come from ``--pairs`` (a jsonl of
           {prompt,pos,neg}) — the matched-pool builder writes the training
           pool; this driver consumes it.

Failure containment + resume (two runs died on CAST's condition-prompt guard
with ZERO objects written — 15:00 and 34:45 of lockstep training for the
OTHER arms discarded):

  - every object is saved the INSTANT its arm completes (the lockstep group's
    members complete together, at ``train_lockstep``'s return), never batched
    to the end of the arm list — a crash in arm N leaves arms 1..N-1 on disk.
  - each arm is ISOLATED: an exception (CAST's guard refusal included) is
    recorded into ``<defenses_dir>/<si>__fit_report.json`` and the run
    CONTINUES; the CLI exits non-zero at the end iff any arm failed, but only
    after every other arm was attempted and saved.
  - a re-run SKIPS arms whose object already exists on disk (``--overwrite``
    retrains everything) — but only after ``check_budget_sidecars`` verifies
    each existing object's recorded ``train_budget_meta.json`` against the
    CURRENT budget knobs AND the object bytes: a ``completion_tokens``
    change, a missing/incomplete sidecar, a foreign ``cell_id``, or an
    ``object_sha256`` that no longer matches the on-disk object REFUSES the
    resume loudly instead of silently mixing supervision budgets or trusting
    a stale pair; a bare ``max_len`` change only warns (a memory knob —
    identical training given preflight_max_len). A partially-done lockstep
    group drops its finished members and trains the remainder at the batch
    size of the FULL requested group — see ``train_si_objects`` for why that
    preserves the bitwise lockstep guarantee.

The category-condition seam (harmful should-fire vs benign should-not-fire
prompts, from ``sisteer.data.content_sets.build_content_contrast`` via the
config's optional ``condition_contrast`` block) is passed ONLY to arms whose
trainer signature declares ``condition_pos_prompts``/``condition_neg_prompts``.
CAST was its one consumer; cast left the registry entirely on 2026-08-20, so
no registered arm declares the kwargs today — the seam stays generic and
signature-introspected, with the builder's leakage guard (refusing
or-bench-hard-1k / XSTest, this lane's over-refusal eval controls) intact,
and the sources actually used still recorded in the arm's
``train_budget_meta.json`` sidecar and the fit report.

CE-lane metric names on the SHARED stream (W&B + the
``reports/metrics_<cell>_all.jsonl`` tee), since 2026-09-03: fitmon's
dual-KL probe vocabulary is teacher-relative, and the document-CE arms
(``CE_ARMS``) have no teacher — doc-mode examples render prompt_with_si ==
prompt_without, so the "teacher" IS the bare base and the same record
measures the object's DISPLACEMENT from base over a ~0 floor. The driver
renames those records for CE arms only (:data:`CE_PROBE_STREAM_KEYS`,
:data:`CE_FLOOR_STREAM_KEYS`); tees written before this carry the old
names, which read as follows:

  probe/<arm>/heldout_loss, probe/<arm>/residual_kl
      -> probe/<arm>/displacement_kl_vs_base   KL(bare || steered) on the
         probe set, nats; RISES as the object learns, by design. It never
         was a held-out loss — the real one is the trainer's
         <arm>/heldout_loss, unchanged (falls).
  probe/<arm>/residual_ratio, probe/<arm>/displacement_ratio,
  kl_ratio_vs_unsteered (the top-level mirror)
      -> dropped; the CE mirror is displacement_kl_vs_base (top-level).
         They were numerator / max(KL(bare || bare), 1e-6) ~ displacement
         x 1e6 — the panel that climbed 1e3 -> 5e5 and read as a blow-up.
  final/<arm>/{residual_kl, residual_ratio, displacement_ratio}
      -> final/<arm>/displacement_kl_vs_base; both ratios dropped.
  <si>__<arm>__heldout_ce/{final_probe_loss, noop_floor, floor_ratio}
      -> .../{heldout_ce_active, heldout_ce_noop, ce_ratio} — the CE floor
         verdict under the fit_record's own field names (never a probe).

Untouched: probe/<arm>/displacement_kl (KL(steered || bare), the reverse
direction) and probe/<arm>/noop_floor; every KL-distillation-lane key, byte
for byte; and the per-CELL ``metrics.jsonl``, which keeps fitmon's flat
vocabulary on both lanes (audit_cell's contract).
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import socket
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # sibling scripts

REPO = Path(__file__).resolve().parents[1]


def _load_cfg(path: str) -> dict:
    import yaml

    return yaml.safe_load(_config_path(path).read_text())


def _config_path(path: str) -> Path:
    """The file :func:`_load_cfg` reads for ``path`` — relative paths are
    repo-rooted, the same resolution, so the sha below hashes the bytes the
    fit actually parsed."""
    p = Path(path)
    if not p.is_absolute():
        p = REPO / path
    return p


def _config_sha(path: str) -> str | None:
    """16-hex sha256 prefix of the config FILE BYTES — the file identity the
    ``[config-resolved]`` banner prints and ``--expect-config-sha`` asserts.
    File bytes, not the parsed dict, because bytes are what a launch tarball
    ships: the 2026-08-22 wave ran all night on pre-fix arm widths from a
    tarball that predated the config commit, and nothing ever compared the
    shipped file against the one the driver intended. ``None`` when the file
    is unreadable (programmatic callers pass placeholder paths; the banner
    then says so instead of crashing)."""
    try:
        return hashlib.sha256(
            _config_path(path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None


def _cfg_dict_sha(cfg: dict) -> str:
    """16-hex sha256 of the canonical config DICT — the programmatic-caller
    fallback for :func:`_config_sha` when there is no config FILE to hash
    (tests and orchestrators pass dicts). Same truncation as the file sha so
    the two spellings are interchangeable inside a fingerprint."""
    blob = json.dumps(cfg or {}, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _fit_par():
    """The ambient fit-parallel context (sisteer.extraction.ddp), or None.
    Lazy import — the driver must stay importable for plan (CPU, no torch
    initialization side effects beyond sisteer's own)."""
    from sisteer.extraction import ddp as _ddp

    return _ddp.active()


def _is_main_rank() -> bool:
    """True in a single-process fit and on rank 0 of a data-parallel one:
    the rank that owns every write (report, checkpoint, sidecars, COMPLETE,
    the registry emission). The replicas are asserted bit-equal inside the
    trainer, so rank 0's record IS the run's record."""
    par = _fit_par()
    return par is None or par.is_main


def fit_width_active() -> int:
    """The data-parallel width THIS process is fitting at: the ambient
    context's world size, else 1. Recorded on every published cell
    (train_budget_meta.json ``fit_width``, fit_selection.json's
    ``fit_width`` stamp, the fit_record)."""
    par = _fit_par()
    return int(par.world_size) if (par is not None and par.enabled) else 1


def _fit_width_expected(args, cfg: dict) -> int:
    """The width a fit entrypoint must be launched at: ``--expected-width``
    when given (the scheduler's arm tasks, a deliberate width-1 run of a
    wide config), else the config's top-level ``fit_width`` (the generated
    fits yaml carries it; absent = 1, the historical single-process fit)."""
    raw = str(getattr(args, "expected_width", "") or "").strip()
    if raw:
        return int(raw)
    return int((cfg or {}).get("fit_width", 1) or 1)


def _enter_fit_process_group(args, cfg: dict, arms: list[str], *, entry: str):
    """The ONE data-parallel gate every fit entrypoint crosses: read
    torchrun's ``WORLD_SIZE`` against the width the config / flag expects,
    refuse every mismatch BY NAME, and — under a real launcher — initialize
    the process group (NCCL on CUDA, ``torch.cuda.set_device(LOCAL_RANK)``
    BEFORE the model load) and set the ambient context the trainer shards
    on. Returns the context, or None for the single-process path.

    Refused: a wide config run outside a launcher (a width-4 fit silently
    running on one GPU for 6 days is the failure this exists for), a
    launcher whose world differs from the width, a non-CE arm (only
    ``train_object_ce`` shards; every other trainer would either replicate
    the whole fit per rank or row-shard on its own terms), the in-job arm
    scheduler (``data_parallel.gpus > 1`` packs arms 1-per-GPU — one or the
    other), and a per-arm ``accum_steps`` the width does not divide (the
    trainer re-checks; this is the fail-in-milliseconds copy)."""
    world = int(os.environ.get("WORLD_SIZE", "1") or 1)
    expected = _fit_width_expected(args, cfg)
    if expected < 1:
        raise SystemExit(f"[{entry}] fit width must be >= 1, got {expected}")
    if world != expected:
        if world == 1:
            raise SystemExit(
                f"[{entry}] the fit width is {expected} (config fit_width / "
                "--expected-width) but this process is not under a "
                f"{expected}-rank launcher (WORLD_SIZE unset) — launch it as "
                f"`torchrun --standalone --nproc_per_node={expected} "
                f"scripts/train_two_si_objects.py {entry} ...` (the fit "
                "sbatch does), or pass --expected-width 1 to run ONE rank "
                "deliberately.")
        raise SystemExit(
            f"[{entry}] WORLD_SIZE={world} but the fit width is {expected} "
            "(config fit_width / --expected-width) — refusing a mis-sized "
            "process group; the launcher's --nproc_per_node must equal the "
            "width.")
    if world == 1:
        return None
    non_ce = [a for a in arms if a not in CE_ARMS]
    if non_ce:
        raise SystemExit(
            f"[{entry}] data-parallel fits (fit_width={world}) are wired for "
            f"the document-CE arms {list(CE_ARMS)} only; got {non_ce}. Run "
            "those arms at width 1.")
    dp = (cfg or {}).get("data_parallel") or {}
    if int(dp.get("gpus", 1) or 1) > 1:
        raise SystemExit(
            f"[{entry}] data_parallel.gpus={dp.get('gpus')} (the in-job arm "
            f"scheduler, arms packed 1-per-GPU) cannot combine with a "
            f"torchrun fit width of {world} — one or the other.")
    tk = (cfg or {}).get("train_kwargs") or {}
    default_accum = (tk.get("default") or {}).get("accum_steps")
    for a in arms:
        accum = (tk.get(a) or {}).get("accum_steps", default_accum)
        if accum is not None and int(accum) % world != 0:
            raise SystemExit(
                f"[{entry}] {a}: fit_width={world} does not divide "
                f"accum_steps={accum} (the effective batch ÷ the micro-batch) "
                "— the per-rank accumulation must split equally; pick a "
                "width that divides it, or run at width 1.")
    from sisteer.extraction import ddp as _ddp

    par = _ddp.setup_from_env()
    assert par is not None and par.world_size == world
    print(f"[{entry}] data-parallel fit: rank {par.rank}/{par.world_size} on "
          f"{par.device} (group-sharded DDP, D96; effective batch unchanged)",
          flush=True)
    return par


def _leave_fit_process_group() -> None:
    """Tear the process group down at the end of a fit entrypoint (a clean
    NCCL shutdown instead of a warning at interpreter exit). No-op when
    none was initialized."""
    from sisteer.extraction import ddp as _ddp

    if _ddp.active() is None:
        return
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
    _ddp.set_active(None)


def _resolved_banner(cfg: dict, sha: str | None, positions: str,
                     config_path: str) -> str:
    """The single greppable ``[config-resolved]`` line every fit entrypoint
    prints FIRST (wave-autopsy lesson 1: a fit must print what the RUNNING
    process actually resolved — sha for the file's byte identity, widths/
    epochs/leg for what a human diffs against intent — in its first log
    lines, not never). Every read tolerates a minimal cfg: programmatic
    callers (tests, the orchestrator) pass dicts with most keys absent."""
    dp = (cfg or {}).get("data_parallel") or {}
    gpus = int(dp.get("gpus", 1) or 1)
    arm_gpus = dp.get("arm_gpus") or {}
    widths = sorted(set(arm_gpus.values()))
    if not arm_gpus:
        amap = "auto"  # no override map: every arm runs width 1 here
    elif len(widths) == 1:
        amap = f"all:{widths[0]}"
    else:
        amap = ",".join(f"{a}:{arm_gpus[a]}" for a in sorted(arm_gpus))
    epochs = _train_kwargs_default(cfg or {}).get("epochs", 6)
    fit_width = int((cfg or {}).get("fit_width", 1) or 1)
    return (f"[config-resolved] sha256={sha or 'unreadable'} gpus={gpus} "
            f"fit_width={fit_width} arm_gpus={amap} epochs={epochs} "
            f"positions={positions} config={_config_path(config_path)}")


def _assert_expected_config_sha(args) -> str | None:
    """The driver-side stale-config CHECK (wave-autopsy lesson 1's second
    half). The wave driver computes the sha of the config it INTENDS and
    passes ``--expect-config-sha``; a launch tarball that predates that
    config then refuses in milliseconds — before any pair or model load —
    instead of fitting the wrong world all night. Returns the actual sha
    (``None`` when the file is unreadable) for the banner. getattr with
    defaults throughout: programmatic callers build bare Namespaces that
    predate the flag."""
    actual = _config_sha(str(getattr(args, "config", "") or ""))
    expected = str(getattr(args, "expect_config_sha", "") or "")
    expected = expected.strip().lower()
    if not expected:
        return actual
    if actual != expected[:16]:
        raise SystemExit(
            f"[config-resolved] REFUSED: expected config sha256="
            f"{expected[:16]} but "
            f"{_config_path(str(getattr(args, 'config', '') or ''))} has "
            f"sha256={actual or 'unreadable'} — the config on disk is not "
            "the one the driver intended (stale launch tarball, or the file "
            "changed after the driver computed its sha). Re-ship the "
            "intended config rather than letting this fit resolve a "
            "different world than the wave was told it ran.")
    return actual


# --------------------------------------------------------------------------- #
# Planning (pure — CPU, no model needed)                                       #
# --------------------------------------------------------------------------- #


def partition_arms(arms: list[str]) -> tuple[list[str], list[str]]:
    """Split ``arms`` into (lockstep-compatible, solo), preserving order. The
    compatible arms can share a teacher forward; the rest train solo."""
    from sisteer.extraction.lockstep import lockstep_compatible

    compatible = [a for a in arms if lockstep_compatible(a)]
    solo = [a for a in arms if not lockstep_compatible(a)]
    return compatible, solo


def default_class_a_arms(cfg: dict) -> list[str]:
    """The registered Class A arms for the two-SI matrix, in config order —
    the arms this driver trains per SI (Class B is SI-agnostic, out of scope)."""
    from sisteer.eval.defense_cell import CLASS_A_ARMS
    from sisteer.methods import ARMS

    arms = list(cfg.get("class_a_arms") or CLASS_A_ARMS)
    return [a for a in arms if a in ARMS]


# Class C — the DECODE-ONLY (positions="response") fits of Class A arms
# (a FIT-TIME taxonomy: an object's class is how it was fitted, never how
# it is later served — the serve phase is the eval cell's own explicit
# choice, the fit x serve 2x2). This driver fits the wave through its
# existing mask leg (--positions response); no separate fit machinery
# exists or is needed. Membership = every Class A arm whose object
# genuinely supports a masked fit AND whose artifact PERSISTS the fit leg
# so a Class C artifact can never be mistaken for a Class A one (the
# 2026-08-21 persistence fixes closed that for psr*/lora*/lofit;
# vector/bidpo persist it in the SteeringVector sidecar metadata).
# configs/two_si.yaml's class_c_arms block records every exclusion and its
# distinct reason (diffmean left 2026-08-25: its capture is hardcoded
# position="last" on prompt-only renderings, so a response "fit" relabels
# the bitwise-identical Class A vector — the alphakl mechanism, verified
# on the wave artifacts; its decode-only leg lives on the SERVE axis, where
# the persisted deployment mode still works unchanged);
# tests/test_class_c_arms.py enforces persistence per member. Order = the
# class_a_arms (arm-major) order, restricted.
# lora_r1_all joined 2026-08-25: it satisfies
# both class-C criteria on this base — genuine masked fit
# (lockstep._MASKABLE_ARMS) and a persisted fit leg (MultiLayerLoRAObject's
# PositionMaskedFit tracker + `fit_positions` in the blob) — placed after
# its single-layer sibling lora_r1 (arm-major order, restricted).
CLASS_C_ARMS: tuple[str, ...] = ("psr_kl_all", "psr_mse_all", "lora_r1",
                                 "lora_r1_all", "lora_r4", "lofit",
                                 "bidpo", "vector", "bidpo_lora")


def default_class_c_arms(cfg: dict) -> list[str]:
    """The registered Class C arms for a decode-only fit wave, in config
    order — config ``class_c_arms`` (the submission-time record) falling
    back to the module tuple, filtered to the registry: identical
    derivation to :func:`default_class_a_arms` above and
    ``fit_class_b.default_class_b_arms``."""
    from sisteer.methods import ARMS

    arms = list(cfg.get("class_c_arms") or CLASS_C_ARMS)
    return [a for a in arms if a in ARMS]


def _accepts_condition_prompts(train_fn) -> bool:
    """Whether a registry trainer's signature DECLARES the category-condition
    kwargs (``condition_pos_prompts`` + ``condition_neg_prompts``). Explicit
    named parameters only — a ``**kw`` catch-all does NOT count, so arms that
    merely swallow kwargs are never handed prompts they don't consume."""
    try:
        params = inspect.signature(train_fn).parameters
    except (TypeError, ValueError):
        return False
    return ("condition_pos_prompts" in params
            and "condition_neg_prompts" in params)


def _accepts_probe_cb(train_fn) -> bool:
    """Whether a registry trainer's signature DECLARES the mid-fit probe
    seam (``probe_cb`` + ``probe_every`` — train_bidpo's shape, and
    train_psr's via methods._make_train_psr since the F7 seam landed).
    Explicit named parameters only, the ``_accepts_condition_prompts``
    convention: a ``**kw`` catch-all does not count, so a trainer that
    merely swallows kwargs is never credited with probes it won't fire."""
    try:
        params = inspect.signature(train_fn).parameters
    except (TypeError, ValueError):
        return False
    return "probe_cb" in params and "probe_every" in params


# Memoized per (canonicalized) condition config: the contrast is shared across
# arms AND SIs, so one process resolves it at most once per distinct config.
_CONTRAST_CACHE: dict[str, object] = {}


def resolve_condition_contrast(cfg: dict):
    """The category-based condition contrast (harmful should-fire vs benign
    should-not-fire prompts) for condition-taking arms (CAST), built by
    ``sisteer.data.content_sets.build_content_contrast`` from the config's
    optional ``condition_contrast`` block (kwargs forwarded verbatim; absent
    -> the builder's defaults). Memoized once per process per config.

    The builder's leakage guard is preserved: naming an over-refusal eval
    control (or-bench-hard-1k, XSTest) as a benign source raises ValueError
    here — this driver adds no bypass, and never sets allow_pairs_fallback."""
    from sisteer.data.content_sets import build_content_contrast

    kw = dict(cfg.get("condition_contrast") or {})
    key = json.dumps(kw, sort_keys=True, default=str)
    if key not in _CONTRAST_CACHE:
        _CONTRAST_CACHE[key] = build_content_contrast(**kw)
    return _CONTRAST_CACHE[key]


def group_batch_size(cfg: dict, si_id: str, compatible_arms: list[str]) -> int:
    """The single batch size a lockstep GROUP uses (it must be shared, so every
    arm's teacher batches align): the SAFE shared width = the min of the
    per-cell resolved batch sizes across the group's arms."""
    from sisteer.eval.train_budget import resolve_batch_size

    if not compatible_arms:
        return 0
    return min(resolve_batch_size(cfg, si_id, a) for a in compatible_arms)


# The optimiser keys a LOCKSTEP arm may opt into through train_kwargs.<arm>
# — the ONE deliberate exception to "lockstep arms' per-arm params are
# pinned to their registry defaults": each alters only that arm's own
# optimiser steps (own optimiser, own RNG stream), so none can perturb a
# partner, and the default (keys absent) is bit-identical to the historical
# loop. Wired for psr_kl_all (config), whose fixed-lr KL fit ran away
# hardest in the wave while its mse sibling got warmup/cosine/clipping
# inside train_psr.
#
# ``lr`` BELONGS HERE and its absence was a silent misconfiguration: a
# config pinning psr_kl_all at 5e-4 merged the schedule keys, dropped the
# rate, and trained at the registry's 5e-3 — the exact rate the pin existed
# to avoid. Nothing failed; the arm simply ran away again, and the config
# read correct the whole time. The rate governs the same private optimiser
# the schedule keys do, so it carries the same no-partner-perturbation
# argument they were admitted under.
# ``warmup_frac`` joined 2026-08-25: HPO winners
# are recorded as a FRACTION so the production fit re-derives warmup steps
# from its own step budget — leaving it off this allow-list is exactly the
# 3dd9e36 class of bug (a configured winner silently dropped, the run
# reading correct throughout). T2 pins it.
# ``accum_steps`` joined 2026-08-27 (re-tune@32): gradient accumulation for
# the lockstep KL arms (effective batch = batch_size x accum_steps, one
# clip on the accumulated gradient — LockstepArmParams.accum_steps owns the
# semantics). Same no-partner-perturbation argument as the schedule keys:
# it alters only that arm's own optimiser cadence.
# ``weight_decay`` joined 2026-08-27 (re-tune): AdamW had silently been
# applying its own 0.01 default with nothing tuning or recording it; the
# exposed default preserves that value, so absence = today's behavior.
# ``temperature`` joined 2026-08-27 (re-tune@32): the KL arms sweep
# T {1,2,4} (tune_arm.TEMPERATURE_GRID) and a winner ships its T in
# train_kwargs.<arm> — off this list it would be refused at launch (the
# dropped-keys guard) or, worse, silently trained at the registry T. Same
# per-arm-params argument as the rest; note the audited-grid gate in
# train_lockstep still bounds which T may consume a precomputed top-K head.
_LOCKSTEP_OPTIMIZER_KEYS = ("lr", "lr_schedule", "warmup_steps",
                            "warmup_frac", "max_grad_norm", "accum_steps",
                            "weight_decay", "temperature")


def lockstep_arm_params(cfg: dict, arm: str):
    """``default_lockstep_params(arm)`` + the per-arm optimiser-schedule
    opt-in from ``train_kwargs.<arm>`` (``_LOCKSTEP_OPTIMIZER_KEYS`` only —
    every other per-arm key stays solo-only, as before). Validation of the
    values lives in ``train_lockstep``, which refuses loudly."""
    from dataclasses import replace as _replace

    from sisteer.extraction.lockstep import default_lockstep_params

    params = default_lockstep_params(arm)
    arm_tk = dict((cfg.get("train_kwargs") or {}).get(arm) or {})
    sched = {k: arm_tk[k] for k in _LOCKSTEP_OPTIMIZER_KEYS if k in arm_tk}
    if sched.get("lr") is not None:
        sched["lr"] = float(sched["lr"])
    if sched.get("warmup_steps") is not None:
        sched["warmup_steps"] = int(sched["warmup_steps"])
    if sched.get("warmup_frac") is not None:
        sched["warmup_frac"] = float(sched["warmup_frac"])
    if sched.get("max_grad_norm") is not None:
        sched["max_grad_norm"] = float(sched["max_grad_norm"])
    if sched.get("accum_steps") is not None:
        sched["accum_steps"] = int(sched["accum_steps"])
    if sched.get("weight_decay") is not None:
        sched["weight_decay"] = float(sched["weight_decay"])
    if sched.get("temperature") is not None:
        sched["temperature"] = float(sched["temperature"])
    return _replace(params, **sched) if sched else params


# Keys a LOCKSTEP arm's ``train_kwargs.<arm>`` block may carry that are NOT
# merged into its params but ARE read elsewhere by the driver, so their
# presence is legitimate rather than a silent drop.
# gradient_checkpointing: consumed by maybe_enable_gradient_checkpointing
# at model-load time (per-arm ckpt compatibility — the Qwen tf canary was
# refused here on 2026-08-27 before this entry existed, proving the guard
# works and that the key needed registering).
# lora_rank: consumed at the driver's build_lockstep_object call (an
# ARCHITECTURE knob — it parametrizes the object, not the optimiser; the
# builder itself refuses it for any arm but lora_r1_all, so a misplaced
# rank fails that arm loudly rather than passing this guard silently).
_LOCKSTEP_DRIVER_KEYS = ("floor_ratio_max", "gradient_checkpointing",
                         "lora_rank",
                         # the fit SELECTION block: serialised by
                         # the driver into the cell's fit_selection.json at
                         # publish, never handed to a trainer
                         "fit_selection")


def lockstep_dropped_keys(cfg: dict, arm: str) -> list[str]:
    """Keys in ``train_kwargs.<arm>`` that a LOCKSTEP arm will silently
    ignore. Anything not merged into its params and not consumed by the
    driver reads as configuration and behaves as nothing."""
    arm_tk = (cfg.get("train_kwargs") or {}).get(arm) or {}
    known = set(_LOCKSTEP_OPTIMIZER_KEYS) | set(_LOCKSTEP_DRIVER_KEYS)
    return sorted(k for k in arm_tk if k not in known)


def effective_optimizer_settings(cfg: dict, arm: str, *, lockstep: bool):
    """``{key: (value, source)}`` for the optimiser settings this arm will
    ACTUALLY train under, with where each value came from.

    Resolved, never echoed: a config block is not an effective setting, and
    the gap between the two is exactly how psr_kl_all trained a whole wave
    at 5e-3 while the config read 5e-4 (the rate was dropped for lockstep
    arms and nothing said so). ``source`` is the point of this function --
    a number alone cannot show that the config was ignored.
    """
    import inspect

    from sisteer.methods import ARMS

    arm_tk = (cfg.get("train_kwargs") or {}).get(arm) or {}
    keys = ("lr", "lr_schedule", "warmup_steps", "warmup_frac",
            "max_grad_norm", "weight_decay", "temperature")
    out: dict[str, tuple] = {}
    if lockstep:
        # accum_steps is part of a lockstep arm's optimiser identity too
        # (re-tune@32): it reaches the trainer via LockstepArmParams, and
        # the cell fingerprint must distinguish an effective-32 fit from
        # an effective-2 one.
        params = lockstep_arm_params(cfg, arm)
        for k in keys + ("accum_steps",):
            src = "config" if k in arm_tk else "registry default"
            out[k] = (getattr(params, k, None), src)
        return out
    # Solo arms additionally resolve accum_steps (the DPO effective-batch
    # knob — T2's bidpo winner check reads it here — and train_psr's mse
    # loop since the re-tune@32 accumulation landed).
    keys = keys + ("accum_steps",)
    default_tk = _train_kwargs_default(cfg)
    sig = None
    if arm in ARMS:
        try:
            sig = inspect.signature(ARMS[arm].train).parameters
        except (TypeError, ValueError):  # builtins / C callables
            sig = None
    for k in keys:
        if k in arm_tk:
            out[k] = (arm_tk[k], "config")
        elif k in default_tk:
            out[k] = (default_tk[k], "config default")
        elif sig is not None and k in sig:
            d = sig[k].default
            out[k] = (None if d is inspect.Parameter.empty else d,
                      "trainer default")
        else:
            out[k] = (None, "n/a")
    return out


def optimizer_banner(cfg: dict, si_id: str, arms: list[str]) -> list[str]:
    """The ``[optimizer-resolved]`` launch banner: one line per arm naming
    the rate it will train under and where that rate came from."""
    compatible, solo = partition_arms(arms)
    lines = []
    for arm in list(compatible) + list(solo):
        eff = effective_optimizer_settings(cfg, arm,
                                           lockstep=arm in compatible)
        mode = "lockstep" if arm in compatible else "solo"
        bits = []
        for k in eff:
            v, src = eff[k]
            if v is None and src in ("n/a", "trainer default",
                                     "registry default"):
                # A knob nobody set and no registry default carries (e.g.
                # warmup_frac unset) stays off the banner; a CONFIGURED
                # value always prints, None included.
                if k in ("warmup_frac", "accum_steps"):
                    continue
                if src in ("n/a", "trainer default"):
                    continue
            bits.append(f"{k}={v} ({src})")
        if arm in CE_ARMS and resolved_loss_impl(cfg, arm) != "full":
            bits.append(f"loss_impl={resolved_loss_impl(cfg, arm)} (config)")
        lines.append(f"[optimizer-resolved] {si_id} {arm} [{mode}]: "
                     + ", ".join(bits))
    return lines


def refuse_silently_dropped_kwargs(cfg: dict, arms: list[str]) -> None:
    """Fail loudly when ``train_kwargs.<arm>`` carries a key its arm will
    ignore. A dropped key is worse than a wrong one: the config documents an
    intent that never reaches the trainer, and the run looks configured."""
    compatible, _solo = partition_arms(arms)
    bad = {a: lockstep_dropped_keys(cfg, a) for a in compatible}
    bad = {a: ks for a, ks in bad.items() if ks}
    if bad:
        detail = "; ".join(f"{a}: {', '.join(ks)}" for a, ks in bad.items())
        raise SystemExit(
            f"[train-two-si] refusing: train_kwargs keys that a LOCKSTEP arm "
            f"silently ignores -- {detail}. Lockstep arms take only "
            f"{', '.join(_LOCKSTEP_OPTIMIZER_KEYS)} (merged into their "
            f"params) plus {', '.join(_LOCKSTEP_DRIVER_KEYS)} (read by the "
            f"driver). Move the key, or make the arm solo. Configuration "
            f"that reads as set and behaves as unset is how this wave "
            f"trained psr_kl_all at 5e-3 against a config saying 5e-4.")


_ABSENT = object()

# The document-CE arms (sdf-base-steering §5): plain hard-target LM
# cross-entropy on document pools, no teacher forward of any kind. Everything
# CE-specific in this driver keys on membership here, never on name prefixes.
CE_ARMS = ("psr_ce_all", "lora_r4_ce")


def _leading_policy(cfg: dict) -> str | None:
    """The config's leading-special-token policy (top-level
    ``leading_special_tokens``: "bos" | "none"; see
    sisteer.extraction.leading_tokens), None when undeclared. Passed to
    EVERY tokenizer construction below so a fit, its sidecar build and its
    budget measurement see one tokenizer; asserted on the encoded examples
    in train_si_objects before any optimizer step."""
    from sisteer.extraction.leading_tokens import LEADING_KEY

    return (cfg or {}).get(LEADING_KEY) or None

#: The CE lane's rename of fitmon's dual-KL probe vocabulary on the SHARED
#: stream (W&B + the reports tee), applied by :func:`ce_lane_probe_record`
#: to the ``probe/`` and ``final/`` records of CE_ARMS only. fitmon's names
#: are TEACHER-relative: residual_kl = KL(prompted || steered), the gap to
#: the SI-prompted teacher, over noop_floor = KL(prompted || unsteered). The
#: doc-CE lane has no teacher — doc-mode examples render prompt_with_si ==
#: prompt_without, so "prompted" IS the bare base — and the same record
#: reads: residual_kl = KL(bare || steered), the object's DISPLACEMENT from
#: base (grows as it learns, by design); noop_floor = KL(bare || bare) ~ 0,
#: clamped to 1e-6; every *_ratio = numerator x ~1e6, not a ratio (the panel
#: that climbed 1e3 -> 5e5 and read as a blow-up, 2026-09-03). ``None`` =
#: dropped from the shared stream. Keys not listed pass through unchanged
#: (displacement_kl = KL(steered || bare), the reverse direction, and
#: noop_floor say what they are already).
CE_PROBE_STREAM_KEYS = {
    "residual_kl": "displacement_kl_vs_base",
    # fitmon's alias of residual_kl ("the held-out distillation loss on the
    # probe subset") — NOT a held-out loss here; the real one is the
    # trainer's <arm>/heldout_loss, which keeps its name.
    "heldout_loss": None,
    "residual_ratio": None,
    "displacement_ratio": None,
}

#: The CE floor verdict on the shared stream — floor_ratio_check's record
#: under the ``<si>__<arm>__heldout_ce`` tag — renamed to the fit_record's
#: own field names: the check's ``final_probe_loss`` is the best-epoch
#: held-out CE with the object active and its ``noop_floor`` the bare base's
#: (SI-teacher probe names for a quantity that is neither).
CE_FLOOR_STREAM_KEYS = {
    "final_probe_loss": "heldout_ce_active",
    "noop_floor": "heldout_ce_noop",
    "floor_ratio": "ce_ratio",
}


def ce_lane_probe_record(rec: dict) -> dict:
    """fitmon's flat ``probe/*`` / ``final/*`` record, renamed for a CE
    arm's SHARED stream per :data:`CE_PROBE_STREAM_KEYS` — dropped keys
    omitted, everything else (train/, norm groups, gate activity,
    displacement_kl, noop_floor) byte for byte. The per-CELL metrics.jsonl
    never passes through here: it keeps fitmon's vocabulary, the audit's
    contract."""
    out: dict = {}
    for k, v in rec.items():
        head, sep, tail = k.partition("/")
        if sep and head in ("probe", "final") and tail in CE_PROBE_STREAM_KEYS:
            new = CE_PROBE_STREAM_KEYS[tail]
            if new is None:
                continue
            k = f"{head}/{new}"
        out[k] = v
    return out


def _ce_floor_stream_record(rec) -> dict:
    """floor_ratio_check's ``<tag>/{final_probe_loss, noop_floor,
    floor_ratio}`` record with the same values under
    :data:`CE_FLOOR_STREAM_KEYS`' names; any other key untouched."""
    if not isinstance(rec, dict):
        return rec
    out: dict = {}
    for k, v in rec.items():
        head, sep, tail = k.rpartition("/")
        if sep and tail in CE_FLOOR_STREAM_KEYS:
            k = f"{head}/{CE_FLOOR_STREAM_KEYS[tail]}"
        out[k] = v
    return out


#: config ``data_mode`` value that switches the example builders to the
#: verbatim document path (docs_to_ce_examples — no chat template, ever).
DOC_DATA_MODE = "documents"


def _data_mode(cfg: dict) -> str:
    return str((cfg or {}).get("data_mode", "pairs"))


#: The per-cell cap keys ``train_kwargs_by_si.<si>`` may carry (the
#: max_len_by_si precedent, generalized). A closed whitelist: an unknown
#: key here is configuration that reads as set and behaves as unset, the
#: exact bug class refuse_silently_dropped_kwargs names.
_BY_SI_KEYS = frozenset({"epochs", "early_stop_patience", "accum_steps"})


def _train_kwargs_for_si(cfg: dict, si_id: str) -> dict:
    """``train_kwargs.default`` with the per-cell ``train_kwargs_by_si.
    <si_id>`` caps merged OVER it (and still UNDER ``train_kwargs.<arm>``,
    which the solo loop applies last) — resolved where ``tk`` is built, the
    ``max_len_by_si`` precedent. Absent block = byte-identical defaults.
    Unknown by-si keys refuse loudly (the whitelist above)."""
    tk = _train_kwargs_default(cfg)
    by_si = dict((cfg.get("train_kwargs_by_si") or {}).get(si_id) or {})
    if not by_si:
        return tk
    unknown = sorted(set(by_si) - _BY_SI_KEYS)
    if unknown:
        raise SystemExit(
            f"[train-two-si] train_kwargs_by_si.{si_id} carries key(s) the "
            f"driver does not consume: {unknown} — only "
            f"{sorted(_BY_SI_KEYS)} are per-cell caps. Configuration that "
            "reads as set and behaves as unset is the refused bug class.")
    tk.update(by_si)
    return tk


def refuse_by_si_for_lockstep_arms(cfg: dict, si_id: str,
                                   arms: list[str]) -> None:
    """A ``train_kwargs_by_si`` block reaches only the SOLO/CE path (it
    merges into the defaults the solo loop reads); a lockstep arm's params
    are pinned to registry defaults + the schedule allow-list, so it would
    silently ignore the per-cell caps — refuse rather than let the config
    read as set and behave as unset."""
    by_si = (cfg.get("train_kwargs_by_si") or {}).get(si_id) or {}
    if not by_si:
        return
    from sisteer.extraction.lockstep import lockstep_compatible

    locked = [a for a in arms if lockstep_compatible(a)]
    if locked:
        raise SystemExit(
            f"[train-two-si] train_kwargs_by_si.{si_id} is set but arm(s) "
            f"{locked} train through the LOCKSTEP path, which pins per-arm "
            "params to registry defaults and would silently ignore the "
            "per-cell caps. Per-cell caps are a solo/CE-path seam — train "
            "those arms solo, or drop the block.")


def refuse_ce_misconfiguration(cfg: dict, arms: list[str]) -> None:
    """Config-shape guard for the document-CE arms, run next to
    :func:`refuse_silently_dropped_kwargs` and again inside
    ``train_si_objects`` (the authoritative choke point every entrypoint
    crosses). Refuses, for any requested CE arm:

    - ``data_mode != "documents"`` — the CE arms' contract is
      documents-in-the-Pair-slots; an SI-pair pool would chat-render;
    - ``train_kwargs.default.positions != "all"`` — the document fits
      steer every position by design (``positions_at_fit="all"``);
    - SI/teacher-shaped ``train_kwargs.<arm>`` keys — ``temperature``,
      ``teacher_prefix_cache`` (CE has no teacher forward at all), and a
      NON-NULL ``floor_ratio_max`` (the SI-teacher floor probe is
      meaningless here; the CE-native ``ce_floor_ratio_max`` replaces it —
      an explicit null is fine and is recorded).

    And, symmetrically: ``data_mode: documents`` with any NON-CE arm is
    refused (the doc pools feed the CE arms only — every other registry
    trainer chat-renders its examples and would silently train a different
    corpus than the sampler budgeted); and ``train_kwargs.default.
    heldout_pool`` requires ``data_mode: documents`` + ``holdout_frac: 0``
    (an explicit shared pool AND a per-fit carve is a contradiction)."""
    cfg = cfg or {}
    arms = list(arms or [])
    tk = _train_kwargs_default(cfg)
    mode = _data_mode(cfg)
    problems: list[str] = []
    ce_requested = [a for a in arms if a in CE_ARMS]
    if ce_requested:
        if mode != DOC_DATA_MODE:
            problems.append(
                f"CE arm(s) {ce_requested} require data_mode: documents "
                f"(got {mode!r}) — the CE contract is "
                "documents-in-the-Pair-slots, never a chat-rendered pool")
        if str(tk.get("positions", "all")) != "all":
            problems.append(
                f"CE arm(s) {ce_requested} require positions: all (got "
                f"{tk.get('positions')!r}) — the document fits steer every "
                "position by design and train_object_ce has no masked leg")
        for arm in ce_requested:
            arm_tk = (cfg.get("train_kwargs") or {}).get(arm) or {}
            for k in ("temperature", "teacher_prefix_cache"):
                if k in arm_tk:
                    problems.append(
                        f"train_kwargs.{arm}.{k} is set, but the CE arms "
                        "have no teacher forward of any kind — the key "
                        "would read as set and behave as unset")
            if ("floor_ratio_max" in arm_tk
                    and arm_tk["floor_ratio_max"] is not None):
                problems.append(
                    f"train_kwargs.{arm}.floor_ratio_max="
                    f"{arm_tk['floor_ratio_max']!r}: the SI-teacher floor "
                    "probe is meaningless for a CE arm (soft-KL vs an SI "
                    "teacher that does not exist). Set floor_ratio_max: "
                    "null explicitly and use ce_floor_ratio_max (default "
                    "1.1) — the guard is replaced, never silently disabled")
            li = arm_tk.get("loss_impl")
            if li is not None:
                from sisteer.extraction.chunked_ce import LOSS_IMPLS
                if li not in LOSS_IMPLS:
                    problems.append(
                        f"train_kwargs.{arm}.loss_impl={li!r} is not one of "
                        f"{list(LOSS_IMPLS)} (R2-D62)")
        if "loss_impl" in tk:
            problems.append(
                "train_kwargs.default.loss_impl is set, but only the "
                "per-arm block (train_kwargs.<arm>.loss_impl) reaches the "
                "CE trainer — the default-level key would read as set and "
                "behave as unset")
    if mode == DOC_DATA_MODE:
        non_ce = [a for a in arms if a not in CE_ARMS]
        if non_ce:
            problems.append(
                f"data_mode: documents but arm(s) {non_ce} are not CE "
                f"arms ({list(CE_ARMS)}) — every other registry trainer "
                "chat-renders its examples and would silently train a "
                "different corpus than the sampler budgeted")
    if tk.get("heldout_pool"):
        if float(tk.get("holdout_frac", 0.1) or 0.0) > 0:
            problems.append(
                "train_kwargs.default.heldout_pool and holdout_frac > 0 "
                "are mutually exclusive — the shared held-out pool "
                "REPLACES the per-fit hash carve (set holdout_frac: 0)")
        if mode != DOC_DATA_MODE:
            problems.append(
                "train_kwargs.default.heldout_pool requires data_mode: "
                "documents — the pool is a document pool and would be "
                "chat-rendered on the pairs path")
    if problems:
        raise SystemExit(
            "[train-two-si] refusing a mis-shaped CE/document config:\n  "
            + "\n  ".join(problems))


def arm_ce_floor_ratio_max(cfg: dict, arm: str) -> float:
    """The CE-native floor bar for one CE arm:
    ``train_kwargs.<arm>.ce_floor_ratio_max`` (default 1.1 — a trained
    object restored to its best held-out epoch should never sit >10% above
    bare base; a degeneration guard, not a did-it-help gate: small cells
    that merely fail to help land at ~1.0 and publish, which the scaling
    study needs)."""
    v = ((cfg.get("train_kwargs") or {}).get(arm) or {}).get(
        "ce_floor_ratio_max", 1.1)
    return float(v)


def _sample_manifest_sha_for(pairs_path, *, cell: str,
                             pool_sha256: str | None = None) -> str | None:
    """sha256 of the sample manifest staged next to a document pool, for
    ``meta_extra.sample_manifest_sha`` — through
    ``sisteer.data.sample_manifest.resolve_sample_manifest``, the ONE
    resolver the Slurm preflight (scripts/sdfbase/slurm_fit_cell.py) also
    uses, so the two cannot name different files. The manifest is
    VALIDATED: its ``cell`` is this SI and its ``pool_sha256`` is the pool
    FILE's byte hash (``pool_sha256`` here only when the caller derived
    exactly that — ``_cli_fit_identity`` — else hashed now; never the
    content sha ``train_si_objects`` derives for programmatic callers).
    None — a stated absence, with a loud line — only when NO manifest exists
    in either spelling (synthetic test pools; production pools always ship
    one). An existing manifest that is empty, unparseable, another cell's
    or another pool's is a SystemExit: false provenance is refused, never
    recorded (2026-09-10; before, the first EXISTING file was hashed unread
    — an empty one included — and nothing tied it to the pool)."""
    from sisteer.data.sample_manifest import (SampleManifestError,
                                              resolve_sample_manifest)

    if not pairs_path:
        return None
    try:
        m = resolve_sample_manifest(pairs_path, cell=cell,
                                    pool_sha256=pool_sha256)
    except SampleManifestError as e:
        raise SystemExit(
            f"[train-two-si] sample manifest beside {pairs_path}: {e}") from e
    if m is None:
        print(f"[train-two-si] no sample manifest beside {pairs_path} "
              "(.manifest.json) — sample_manifest_sha recorded null",
              file=sys.stderr)
        return None
    return m.sha256


def arm_floor_ratio_max(cfg: dict, arm: str, default_frm):
    """The floor-ratio bar THIS arm is judged against:
    ``train_kwargs.<arm>.floor_ratio_max`` when present (null = guard off
    for that arm), else the shared ``default_frm`` (train_kwargs.default).

    Per-arm because the bar's meaning follows the OBJECTIVE: for a
    distillation objective a ratio above 1.0 means the trained object
    moves the unprompted student FURTHER from the SI-prompted teacher
    than doing nothing — not a fit at all — so the default bar is 1.0; an
    arm whose loss is NOT a divergence to the teacher (the DPO preference
    arms; the closed-form diffmean/alphakl fits) may carry its own bar in
    config, with the reason recorded there, rather than silently wearing
    a distillation criterion."""
    v = ((cfg.get("train_kwargs") or {}).get(arm) or {}).get(
        "floor_ratio_max", _ABSENT)
    if v is _ABSENT:
        return default_frm
    return float(v) if v is not None else None


def plan_si(cfg: dict, si_id: str, arms: list[str]) -> dict:
    """The training plan for one SI: partition, per-SI max_len, the lockstep
    group's shared batch size, and each solo arm's resolved batch size. Pure
    CPU — no model, no pairs."""
    from sisteer.eval.train_budget import resolve_batch_size, resolve_max_len

    from sisteer.methods import ARMS

    compatible, solo = partition_arms(arms)
    return {
        "si_id": si_id,
        "max_len": resolve_max_len(cfg, si_id),
        "lockstep_group": compatible,
        "lockstep_batch_size": group_batch_size(cfg, si_id, compatible),
        "solo": {a: resolve_batch_size(cfg, si_id, a) for a in solo},
        # The floor probe's sample size (train_kwargs.default.probe_n; null
        # = every fit example) — a STATISTICS knob, decoupled from the
        # batch widths above so the plan states both.
        "probe_n": _train_kwargs_default(cfg).get("probe_n", 256),
        # The arms that will receive the category condition prompts
        # (signature-introspected), so a plan reader can tell before training.
        "condition_arms": [a for a in arms if a in ARMS
                           and _accepts_condition_prompts(ARMS[a].train)],
    }


# --------------------------------------------------------------------------- #
# Training (one model load; lockstep group + solo remainder)                   #
# --------------------------------------------------------------------------- #


def console_log(data, step=None, commit=None, sync=None):
    """W&B-disabled fallback for ``log_fn``, with ``wandb.log``'s signature.

    The fit jobs run under JB4 containment (WANDB_DISABLED=true), so the
    driver's console logger stands in for ``wandb.log``. It used to be the
    BUILTIN ``print``, which tolerates a positional step — the
    ``log_fn(payload, step)`` shape the lockstep/distill/lofit loops use,
    rendering ``{'lofit/bias_ce': 0.53, 'epoch': 5} 1910`` — but raises
    ``TypeError: 'step' is an invalid keyword argument for print()`` on the
    KEYWORD shape ``log_fn(payload, step=...)`` used by the calibration
    ladder (extraction/calibrate.py, antidote.py) and alphakl's summary
    line. That mismatch killed the cast / alphakl_null / alphakl_fisher
    arms of BOTH SIs (6 object cells) on the n3 fit run. Mirroring
    ``wandb.log(data, step=, commit=, sync=)`` makes EVERY call that is
    valid against the real wandb.log valid here — positional or keyword
    step, and any of its other kwargs — instead of fixing one call shape
    and leaving the next as a landmine. ``commit``/``sync`` are W&B
    batching/sync controls with no console meaning; they are accepted and
    ignored. The positional print format above is preserved exactly.
    """
    del commit, sync
    if step is not None:
        print(data, step)
    else:
        print(data)


def wandb_fit_run(cfg: dict, si_id: str, arms: list[str],
                  positions: str | None = None):
    """A live, METRICS-ONLY W&B run for one SI's fit job, or ``None``.

    Live logging is wanted for the fits, but this lane's stdout prints raw
    attack and pair text on error paths, so W&B console capture must stay
    OFF (``WANDB_CONSOLE=off`` is forced here as a belt on top of the sbatch
    body's export) and only numeric scalars are ever logged
    (``make_metrics_log_fn`` filters). Returns ``None`` — and the fit
    carries on with console + jsonl — when:

    - ``WANDB_DISABLED`` is truthy (transcript-job containment) or
      ``WANDB_MODE=disabled`` (the test env),
    - the config opts out (``wandb_fits: false``),
    - ``wandb.init`` fails for any reason (no login on the node, no route
      to api.wandb.ai): a W&B outage must never cost a fit or its curves —
      the per-step jsonl is written regardless.

    ``TT_WANDB_FIT`` (env, JSON object) lets a sweep driver shape the run
    for cross-trial comparison at init time — the ONLY time W&B accepts
    grouping (a sweep's 16 trials must land as one filterable group with
    the swept hyperparameters as sortable config columns, not 16 orphan
    run pages). Recognised keys: ``name``, ``group``, ``job_type``,
    ``tags`` (list), ``config`` (dict, merged OVER the computed defaults).
    Scalars/identifiers only, same rule as the computed config — never
    prompt/pair/attack text. Malformed JSON is a loud stderr note and the
    computed defaults, never a lost run.
    """
    if os.environ.get("WANDB_DISABLED", "").lower() in ("1", "true"):
        return None
    if os.environ.get("WANDB_MODE", "") == "disabled":
        return None
    if not bool(cfg.get("wandb_fits", True)):
        return None
    os.environ.setdefault("WANDB_CONSOLE", "off")
    override = {}
    raw = os.environ.get("TT_WANDB_FIT", "")
    if raw:
        try:
            override = json.loads(raw)
            if not isinstance(override, dict):
                raise ValueError(f"expected a JSON object, got "
                                 f"{type(override).__name__}")
        except (ValueError, TypeError) as e:
            print(f"[train-two-si] TT_WANDB_FIT ignored (unparseable: {e}) "
                  "— using the computed run name/config", file=sys.stderr)
            override = {}
    # The name has to identify the CELL — model x arm(s) x SI x position
    # (two models now fit concurrently, so the
    # MODEL joins the name; an array wave also runs several arms x both
    # position modes against one SI, and a name carrying only the SI makes
    # those runs distinguishable in the browser solely by job id -- which
    # is exactly the monitoring the run is for).
    slug = str(cfg.get("model") or "").rsplit("/", 1)[-1]
    name = f"fit-{slug}-{si_id}" if slug else f"fit-{si_id}"
    name = f"{name}-{'+'.join(arms)}"
    if positions:
        name = f"{name}-{positions}"
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        name = f"{name}-{job}"
    name = override.get("name") or name
    # Scalar/identifier config only — never prompt/pair/attack text.
    # positions is a first-class axis of the wave, so it belongs in
    # the run config and not only in the name -- grouping an A/C
    # contrast in the UI needs it as a field. model/model_revision
    # likewise (the cross-model comparisons group on them).
    config = {"si_id": si_id, "arms": list(arms),
              "positions": positions,
              "model": cfg.get("model"),
              "model_revision": cfg.get("model_revision"),
              "layer": cfg.get("layer"), "seed":
                  (cfg.get("train_kwargs") or {}).get("default", {})
                  .get("seed", 0)}
    config.update(override.get("config") or {})
    init_kw = {}
    for key in ("group", "job_type", "tags"):
        if override.get(key) is not None:
            init_kw[key] = override[key]
    try:
        # the project: the config's wandb_project (the configs write $WANDB_PROJECT), else $WANDB_PROJECT itself; a
        # placeholder with no value on this host is reported below like an outage (W&B is optional for a fit), its
        # message naming the variable to set
        from sisteer import site as _site
        project = _site.expand(cfg.get("wandb_project") or "$WANDB_PROJECT", key="wandb_project")
        import wandb

        # Destination entity: the environment first (the Modal
        # secret carries WANDB_ENTITY and the container refused to start
        # unless the key defaults to it), then the config; never wandb's
        # implicit default.
        return wandb.init(
            project=str(project),
            entity=os.environ.get("WANDB_ENTITY") or cfg.get("wandb_entity"),
            name=name,
            config=config,
            settings=wandb.Settings(console="off", init_timeout=60),
            **init_kw,
        )
    except Exception as e:  # noqa: BLE001 — resilience, not silence: say so
        print(f"[train-two-si] W&B unavailable ({type(e).__name__}: {e}) — "
              "metrics continue to console + per-step jsonl", file=sys.stderr)
        return None


def make_metrics_log_fn(*, jsonl_path=None, wandb_run=None, echo=None,
                        latest=None, arms=()):
    """Composite ``log_fn``: console echo + per-step metrics jsonl + live
    W&B, with ``wandb.log``'s exact signature (``console_log``'s contract —
    the keyword-step TypeError killed 6 object cells once).

    Metrics-only by construction: every payload is filtered to NUMERIC
    scalars before it touches the jsonl or W&B, so prompt/pair/completion/
    attack text can never leave the process through this path even if a
    trainer logs a string. The jsonl is written and flushed per call, so a
    W&B outage (or a mid-run crash) cannot lose the curves; W&B errors are
    swallowed after one loud stderr note.

    ``latest`` (optional dict) is updated in place with every numeric
    payload plus ``_step`` — the caller's live view of the newest metrics,
    which is what a SIGTERM flush (a sweep monitor's scancel) records as
    the run's last known state before the process dies.

    ``arms`` (the arms this run trains) tells the mirror which lane each
    per-arm probe record belongs to — by arm KIND (membership in
    :data:`CE_ARMS`), never by key shape: a CE arm's record mirrors
    ``probe/<arm>/displacement_kl_vs_base`` to the top-level
    ``displacement_kl_vs_base`` panel and never feeds
    ``kl_ratio_vs_unsteered`` (see :data:`CE_PROBE_STREAM_KEYS`). With no
    CE arm the behaviour is byte-identical to the distillation path's."""
    ce_arms = tuple(a for a in (arms or ()) if a in CE_ARMS)

    def _is_ce_arm_key(key: str) -> bool:
        parts = key.split("/")
        return len(parts) == 3 and parts[1] in ce_arms

    fh = None
    if jsonl_path is not None:
        jsonl_path = Path(jsonl_path)
        jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        fh = open(jsonl_path, "a", encoding="utf-8")
    wandb_broken = [False]

    def log_fn(data, step=None, commit=None, sync=None):
        if echo is not None:
            echo(data, step=step, commit=commit, sync=sync)
        if not isinstance(data, dict):
            # The driver's own status strings (_log) ride the same log_fn;
            # they are console-only by design — text never reaches W&B or
            # the jsonl through this path.
            return
        numeric = {k: v for k, v in data.items()
                   if isinstance(v, (int, float))}
        if not numeric:
            return
        # Mirror buried per-arm keys to prominent top-level panels (they
        # are watched live): the learning rate, and the KL ratio
        # vs the unsteered student (probe/residual_ratio = residual_kl /
        # noop_floor — the gap-to-close, probed ~per-epoch or denser) —
        # or, for a CE arm, the displacement from base.
        for k in list(numeric):
            if k.endswith("/lr"):
                numeric.setdefault("lr", numeric[k])
                break
        for k in list(numeric):
            # fit path namespaces per arm (probe/<arm>/residual_ratio);
            # the trial path is flat (probe/residual_ratio) — match both.
            # Never a CE arm's: there the "ratio" is a displacement over a
            # ~0 floor (CE_PROBE_STREAM_KEYS) — that lane mirrors
            # displacement_kl_vs_base below, chosen by arm kind.
            if (k.endswith("residual_ratio") and k.startswith("probe")
                    and not _is_ce_arm_key(k)):
                numeric.setdefault("kl_ratio_vs_unsteered", numeric[k])
                break
        for arm in ce_arms:
            # The CE lane's top-level panel: the object's displacement from
            # base on the probe set, in nats (grows as it learns, by design
            # — the honest replacement for the x1e6 "ratio").
            k = f"probe/{arm}/displacement_kl_vs_base"
            if k in numeric:
                numeric.setdefault("displacement_kl_vs_base", numeric[k])
                break
        if latest is not None:
            latest.update(numeric)
            if step is not None:
                latest["_step"] = step
        if fh is not None:
            fh.write(json.dumps({"step": step, **numeric}) + "\n")
            fh.flush()
        if wandb_run is not None and not wandb_broken[0]:
            # Deliberately NOT wandb.log(step=...): arms restart their own
            # step counters, and W&B silently drops out-of-order steps —
            # the step rides in the payload instead.
            payload = dict(numeric)
            if step is not None:
                payload.setdefault("step", step)
            try:
                wandb_run.log(payload)
            except Exception as e:  # noqa: BLE001 — outage != lost curves
                wandb_broken[0] = True
                print(f"[train-two-si] W&B log failed ({type(e).__name__}: "
                      f"{e}) — continuing with console + jsonl only",
                      file=sys.stderr)

    return log_fn


# The profiler stops stepping past here — tune_arm.py's cap: the schedule
# (wait=20, warmup=3, active=12, repeat=1) has produced its trace by step 35,
# so later steps would only pay the step() bookkeeping for nothing.
_PROFILE_MAX_STEP = 40


def start_profiler(profile_dir):
    """The ``--profile`` capture (pod_pilot.sh step 7): a torch.profiler ->
    TensorBoard trace of the training steps, mirroring tune_arm.py's inline
    block EXACTLY — same activities (CPU+CUDA), same schedule
    (wait=20, warmup=3, active=12, repeat=1), same
    ``tensorboard_trace_handler`` sink — so the pilot reads one trace shape
    whichever entrypoint produced it. Returns the ENTERED profiler; the
    caller pairs it with ``prof.__exit__`` after training. The handler
    names files per (host, pid), so concurrent arm tasks share one dir
    safely."""
    import torch

    d = Path(profile_dir)
    d.mkdir(parents=True, exist_ok=True)
    prof = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=20, warmup=3, active=12,
                                         repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(d)))
    prof.__enter__()
    return prof


def profiler_stepping_log_fn(log_fn, prof):
    """Wrap ``log_fn`` so the profiler ticks once per OPTIMIZER STEP — the
    driver owns no step loop (the trainers do), but every per-step loss
    record flows through the composite log_fn, so the profiler steps on
    exactly the records the pilot's unified classifier
    (``pod_acceptance_probe.is_step_line``) counts as steps: never on
    startup probes, held-out evals or summary lines, which is what keeps
    the wait/warmup/active windows aligned with real optimizer work."""
    from pod_acceptance_probe import is_step_line

    def wrapped(data, step=None, commit=None, sync=None):
        log_fn(data, step=step, commit=commit, sync=sync)
        if (isinstance(data, dict) and (step or 0) <= _PROFILE_MAX_STEP
                and is_step_line({"step": step, **data})):
            prof.step()

    return wrapped


def _write_fit_summary(wandb_run, outcome, arms: list[str]) -> None:
    """End-of-fit outcome scalars into ``run.summary`` — the leaderboard
    row. Summary values are what the W&B runs table and parallel-coordinate
    panels read as columns/endpoints; a metric logged only per-step is
    invisible there, and a sweep whose FAILING trials carry no floor_ratio
    is a sweep with no data (the floor guard REFUSES >bar and nearly every
    exploration trial is expected to refuse — the ratio is the objective
    either way). The guard's raise is caught per-arm upstream
    (``_record(..., detail=arm_diag)``), so the numbers are here in the
    outcome regardless of pass/fail; this just surfaces them.

    Keys are flat for a single-arm fit (the sweep shape: one trial = one
    arm), ``<arm>/``-prefixed otherwise. ``status`` vocabulary:
    ``trained`` / ``floor_refused`` (FloorRatioError — fit ran, object
    refused) / ``failed`` (anything else) / ``skipped_existing``;
    ``killed_early`` False here — the SIGTERM flush is the one that sets
    it True. Numeric scalars only, never text."""
    if wandb_run is None:
        return
    flat = len(arms) == 1
    for arm, res in outcome.arms.items():
        p = "" if flat else f"{arm}/"
        status = res.status
        if res.status == "failed" and res.error_type == "FloorRatioError":
            status = "floor_refused"
        wandb_run.summary[p + "status"] = status
        wandb_run.summary[p + "killed_early"] = False
        detail = res.detail or {}
        for k in ("floor_ratio", "final_probe_loss", "noop_floor",
                  "floor_ratio_max", "floor_ratio_sem", "probe_n",
                  # The CE arms' floor verdict (sdf-base-steering §5.3) —
                  # the sweep objective for the CE HPO lanes, surfaced
                  # pass or fail exactly like floor_ratio.
                  "heldout_ce_active", "heldout_ce_noop", "ce_ratio",
                  "ce_floor_ratio_max"):
            v = detail.get(k)
            if isinstance(v, (int, float)):
                wandb_run.summary[p + k] = v
        tmeta = detail.get("trainer_metadata") or {}
        for k_src, k_dst in (("final_vec_norm", "final_vec_norm"),
                             ("min_loss", "min_loss"),
                             ("final_loss", "final_loss"),
                             ("final_grad_norm", "final_grad_norm"),
                             ("max_grad_norm_seen", "max_grad_norm_seen"),
                             ("steps_run", "steps_completed"),
                             ("epochs_run", "epochs_completed"),
                             ("effective_batch", "effective_batch")):
            v = tmeta.get(k_src)
            if isinstance(v, (int, float)):
                wandb_run.summary[p + k_dst] = v


def _flush_summary_on_sigterm(wandb_run, latest: dict) -> None:
    """Arm a SIGTERM handler that flushes a last-known-state summary before
    the process dies. A sweep monitor cancels a spiking trial with scancel
    → SIGTERM; without this the run shows as crashed with NO summary and
    the sweep table cannot tell "killed" from "failed". Records
    ``killed_early=True``, ``status="spiked"``, the last logged ``*/loss``
    and its step (``latest`` — make_metrics_log_fn keeps it current), then
    finishes the run and exits 143 (128+TERM) without re-entering the
    training stack. Main process only; W&B trouble must never turn a clean
    kill into a hang, hence the bare finally-exit."""
    import signal

    def _flush(signum, frame):
        del signum, frame
        try:
            wandb_run.summary["killed_early"] = True
            wandb_run.summary["status"] = "spiked"
            losses = {k: v for k, v in latest.items()
                      if k.endswith("/loss")}
            if losses:
                wandb_run.summary["last_loss"] = next(iter(losses.values()))
            if "_step" in latest:
                wandb_run.summary["steps_completed"] = latest["_step"]
            wandb_run.finish(exit_code=143)
        finally:
            os._exit(143)

    signal.signal(signal.SIGTERM, _flush)


def split_pairs(pairs, holdout_frac: float, salt: str):
    """Deterministic hash split of the pair pool → (fit, heldout).

    ``data/prompts.split_prompts``'s exact convention — first byte of
    sha256(salt + prompt) — keyed on the pair's PROMPT, so the split is
    stable across runs and orderings and the same attack prompt can never
    sit on both sides. The held-out pairs are NEVER trained on (they feed
    only the per-epoch held-out loss the early stopping rules on)."""
    if not 0.0 <= holdout_frac < 1.0:
        raise ValueError(f"holdout_frac must be in [0, 1), got {holdout_frac}")
    fit, heldout = [], []
    for p in pairs:
        h = hashlib.sha256((salt + p.prompt).encode()).digest()[0] / 255.0
        (heldout if h < holdout_frac else fit).append(p)
    return fit, heldout


# The DPO PREFERENCE arms — every arm consuming bidpo's pair routing (the
# genuine-compliance rejected side, and the section-13 own pool when
# pool.bidpo_benign is enabled). Gate by SET membership, never a string
# equality on "bidpo": a sibling DPO arm (bidpo_lora — the same bidirectional
# objective on a LoRA r=1 substrate, consuming EXACTLY the same pairs) would
# otherwise silently receive the FULL matched pool and train a preference
# objective on refusal-vs-refusal pairs — the precise degenerate signal this
# routing exists to prevent. A third DPO arm joins by being added HERE.
DPO_PREF_ARMS = frozenset({"bidpo", "bidpo_lora"})


def _is_harmful_req(p) -> bool:
    """A pair on the HARMFUL side of the matched pool (strict or relaxed),
    i.e. not benign. The set bidpo's anti-safety negatives cover."""
    return getattr(p, "requirement_a", None) in ("strict", "relaxed")


def _det_pair_take(pairs: list, n: int) -> list:
    """A deterministic, seedless n-subset of Pair rows: rank by the sha256 of
    the pair's PROMPT (the repo's sha-sort convention — pseudo-random but
    byte-reproducible, and independent of input ordering), keep the selected
    pairs in their original list order. Used to cap bidpo's benign-reversed
    side at ``max_benign_per_harmful`` x the harmful side."""
    if n >= len(pairs):
        return list(pairs)
    ranked = sorted(pairs,
                    key=lambda p: hashlib.sha256(p.prompt.encode()).hexdigest())
    keep = {id(p) for p in ranked[:max(0, n)]}
    return [p for p in pairs if id(p) in keep]


def pairs_for_arm(arm: str, pairs: list, *, anti_neg_enabled: bool = False,
                  bidpo_harmful_pairs: "list | None" = None,
                  bidpo_benign_pairs: "list | None" = None,
                  bidpo_benign_cap: "float | None" = 1.0):
    """The pair subset an arm trains on.

    Every arm trains on the full pool EXCEPT the DPO PREFERENCE arms
    (:data:`DPO_PREF_ARMS` — bidpo and its LoRA-substrate sibling bidpo_lora,
    which consumes exactly the same pairs). A preference arm's rejected
    side must be a genuine harmful COMPLIANCE, or the preference signal is
    degenerate. There are two sources for that rejected side, selected by
    ``anti_neg_enabled``:

    LEGACY (``anti_neg_enabled`` False) — the pair's ``neg`` (the undefended
    completion). The primary pool no longer selects for
    undefended compliance, so ``neg`` is a real compliance ONLY on the STRICT
    subset (``requirement_a == "strict"``); on the relaxed remainder ``neg`` is
    itself a refusal. bidpo therefore trains on the strict subset alone — the
    "undefended complied" requirement it KEEPS. This is bounded by the base
    model's compliance rate (~6% on this pool) and is why the anti path exists.

    ANTI (``anti_neg_enabled`` True, ``pool.anti_neg.enabled``) — the
    anti-safety-elicited harmful compliance carried on ``Pair.anti_neg`` (built
    by ``build_two_si_pool generate-anti-neg`` + ``grade-anti-neg``, one per
    harmful matched pair, accepted only when the grader scored it a real
    compliance). bidpo then trains on EVERY harmful pair that has an accepted
    ``anti_neg`` — not just the ~6% strict subset — with ``neg`` REPLACED by
    ``anti_neg`` so the ordinary ``pairs_to_pref_examples`` uses it as the
    rejected side. Benign pairs are excluded (bidpo never learns "prefer refusal
    over compliance" on a benign prompt — that is over-refusal).

    BIDPO'S OWN POOL — ``bidpo_harmful_pairs`` + ``bidpo_benign_pairs``, the
    two per-SI files emit-pairs writes from the manifest's
    ``bidpo_benign`` block. When supplied (both together, never one) they
    REPLACE the matched pool for bidpo entirely — ``pairs`` is ignored for
    this arm:

    - HARMFUL rows (``bidpo_harmful_<si>.jsonl``): this SI's OWN thwart set
      (not the joint intersection, not the 50/50 balance) with pos = the SI's
      refusal (CHOSEN) and neg = the accepted anti-safety compliance
      (REJECTED). NEVER subsampled — the scarcer, load-bearing side.
    - BENIGN rows (``bidpo_benign_<si>.jsonl``), polarity REVERSED: pos = the
      VERIFIED undefended compliance (CHOSEN), neg = this SI's refusal
      (REJECTED) — so the untouched ``pairs_to_pref_examples`` (pos -> chosen,
      neg -> rejected) yields the reversed preference with no special-casing.
      Capped at ``bidpo_benign_cap`` x the harmful side by a deterministic
      sha-ordered subsample (the section-13 rule at its default 1.0: benign =
      min(harmful, available)); ``None`` = uncapped.

    Only the DPO preference arms ever see these pairs; every other arm's
    return is the SAME object as before, feature on or off.

    Fail LOUDLY rather than silently training bidpo on a refusal (the
    authoritative-source rule; each raise is caught upstream so THIS arm records
    failed and the others continue):
    - NO requirement_a label anywhere -> the pool predates the label (or is the
      native regime); bidpo cannot tell harmful from benign, so refuse;
    - LEGACY, ZERO strict pairs -> the base complied with nothing; refuse and
      say to raise the yield / K, enable the anti path, or drop bidpo;
    - ANTI, ZERO harmful pairs carry an accepted anti_neg -> the anti-neg source
      is enabled but its generate/grade pass did not run or produced no
      compliance above threshold; refuse rather than silently degrade to the
      undefended refusal, and say to run the anti pass (raise K) or disable it;
    - OWN POOL: one side supplied without the other; an EMPTY side; a benign
      row without the ``benign_reversed`` requirement_a stamp (a matched-pool
      file has the OPPOSITE polarity and must never be mixed in); a row with
      an empty chosen/rejected; or a cap that empties the benign side ->
      refuse rather than silently train a lopsided bidpo wearing the
      benign-balanced configuration.
    """
    if arm not in DPO_PREF_ARMS:
        return pairs
    if (bidpo_harmful_pairs is None) != (bidpo_benign_pairs is None):
        raise ValueError(
            "bidpo's own pool takes BOTH sides together (bidpo_harmful_pairs "
            "+ bidpo_benign_pairs, the two files emit-pairs writes) — got "
            f"harmful={'set' if bidpo_harmful_pairs is not None else 'None'}, "
            f"benign={'set' if bidpo_benign_pairs is not None else 'None'}. "
            "A one-sided bidpo is exactly the lopsided-preference failure "
            "this feature exists to fix (spec-bidpo-benign section 13).")

    if bidpo_harmful_pairs is None:
        # -- legacy routes: bidpo trains on the matched pool's harmful side --
        from dataclasses import replace

        labelled = any(getattr(p, "requirement_a", None) is not None
                       for p in pairs)
        if not labelled:
            raise ValueError(
                "bidpo needs the per-pair requirement_a label to tell harmful "
                "from benign pairs and source a genuine harmful compliance as "
                "its rejected side, but the loaded pairs carry none. Emit the "
                "pool from a matched-intersection manifest (build_two_si_pool "
                "emit-pairs) that records requirement_a; a bidpo trained on "
                "refusals has a degenerate preference signal (DECISION 2).")
        if anti_neg_enabled:
            harmful_avail = [p for p in pairs if _is_harmful_req(p)]
            harmful = [replace(p, neg=p.anti_neg)
                       for p in harmful_avail if getattr(p, "anti_neg", None)]
            if not harmful:
                raise ValueError(
                    "bidpo anti-neg source is ENABLED (pool.anti_neg) but ZERO "
                    f"harmful pairs (of {len(harmful_avail)}) carry an accepted "
                    "anti-safety compliance — the generate-anti-neg / "
                    "grade-anti-neg pass did not run, or produced no completion "
                    "the grader scored > threshold. Run the anti-neg pass "
                    "(raise --candidates K), or disable pool.anti_neg. Refusing "
                    "to silently train bidpo on the undefended refusal "
                    "(DECISION 2 / bidpo amendment).")
            return harmful
        strict = [p for p in pairs
                  if getattr(p, "requirement_a", None) == "strict"]
        if not strict:
            raise ValueError(
                "bidpo found ZERO strict (undefended-complied) pairs — its "
                "rejected side would be a refusal, not a harmful compliance. "
                "Enable the anti-neg source (pool.anti_neg + generate-anti-neg), "
                "raise the harmful yield / rejection-sampling K, or drop bidpo "
                "from the arm set (DECISION 2).")
        return strict

    # -- bidpo's OWN pool (section 13): the two emit-pairs files -------------
    import build_two_si_pool as bp  # sibling script; the one REQ_A spelling

    if not bidpo_harmful_pairs:
        raise ValueError(
            "bidpo's own pool has ZERO harmful pairs — the per-SI thwart set "
            "carried no accepted anti-safety compliance (or the holdout carve "
            "emptied it). Run/raise-K on generate-anti-neg + grade-anti-neg "
            "and re-freeze the manifest; a bidpo with no harmful preferences "
            "is untrainable (spec-bidpo-benign section 13).")
    bad_h = sum(1 for p in bidpo_harmful_pairs
                if getattr(p, "requirement_a", None) not in ("strict",
                                                             "relaxed"))
    if bad_h:
        raise ValueError(
            f"bidpo harmful pool: {bad_h} of {len(bidpo_harmful_pairs)} rows "
            "lack the strict/relaxed requirement_a stamp emit-pairs writes on "
            "bidpo_harmful_<si>.jsonl — this does not look like bidpo's "
            "harmful file; refusing rather than training on an unknown pool.")
    if not bidpo_benign_pairs:
        raise ValueError(
            "bidpo's benign-reversed side is ENABLED (pool.bidpo_benign) but "
            "ZERO benign-reversed pairs were supplied (or the holdout carve "
            "emptied them) — refusing to silently train a harmful-only bidpo "
            "wearing the benign-balanced configuration. Run grade-store "
            "(benign none labels) -> manifest -> emit-pairs with "
            "pool.bidpo_benign.enabled, or disable it. (The refusal-label "
            "setting and per-SI counts are in the manifest's bidpo_benign "
            "block and the emit sidecar.)")
    mislabelled = sum(
        1 for p in bidpo_benign_pairs
        if getattr(p, "requirement_a", None) != bp.BIDPO_BENIGN_REQ_A)
    if mislabelled:
        raise ValueError(
            f"bidpo benign-reversed pool: {mislabelled} of "
            f"{len(bidpo_benign_pairs)} rows lack the "
            f"requirement_a == {bp.BIDPO_BENIGN_REQ_A!r} stamp emit-pairs "
            "writes — this is not a reversed-polarity file (a matched-pool "
            "pairs file has pos = the SI response, the OPPOSITE polarity). "
            "Refusing rather than training bidpo on inverted preferences.")
    degenerate = sum(1 for p in list(bidpo_harmful_pairs)
                     + list(bidpo_benign_pairs) if not p.pos or not p.neg)
    if degenerate:
        raise ValueError(
            f"bidpo pool: {degenerate} row(s) have an empty chosen (pos) or "
            "rejected (neg) side — a degenerate preference; re-run emit-pairs "
            "rather than training on it.")
    harmful = list(bidpo_harmful_pairs)      # NEVER subsampled (section 13)
    if bidpo_benign_cap is None:
        benign = list(bidpo_benign_pairs)
    else:
        cap = float(bidpo_benign_cap)
        if cap <= 0:
            raise ValueError(
                "pool.bidpo_benign.max_benign_per_harmful must be > 0 or null "
                f"(uncapped), got {bidpo_benign_cap!r}")
        benign = _det_pair_take(bidpo_benign_pairs,
                                min(len(bidpo_benign_pairs),
                                    int(cap * len(harmful))))
        if not benign:
            raise ValueError(
                f"bidpo benign-reversed side emptied by the cap "
                f"(max_benign_per_harmful={cap} x {len(harmful)} harmful "
                f"pairs selects 0 of {len(bidpo_benign_pairs)} benign pairs) "
                "— raise the cap (or set it null for uncapped) rather than "
                "training a harmful-only bidpo under the benign-balanced "
                "configuration.")
    # Harmful first, then benign, each deterministically ordered; train_bidpo
    # shuffles BATCH ORDER per epoch from its own seeded rng, so block order
    # here does not pin the optimisation path.
    return harmful + benign


def _train_kwargs_default(cfg: dict) -> dict:
    return dict((cfg.get("train_kwargs") or {}).get("default") or {})


def maybe_enable_gradient_checkpointing(model, cfg: dict,
                                        arms: list[str] | None = None) -> bool:
    """Config-gated activation checkpointing for the fit entrypoints:
    ``train_kwargs.<arm>.gradient_checkpointing`` overriding
    ``train_kwargs.default.gradient_checkpointing`` (default OFF — no
    existing fit path changes unless a config asks for it). PER-ARM
    because compatibility is per-arm (measured, Qwen 2026-08-27):
    token_faithful (embedding injection, no layer hooks) checkpoints
    cleanly and fits H200 at bs1/16384, while the all-layer hook arms
    (psr_kl_all/psr_mse_all/lora_r1_all) make the recompute pass save a
    DIFFERENT tensor set than the original forward — cross-layer hook
    state — so non-reentrant checkpointing refuses with CheckpointError
    (49-50 vs 44 saved tensors), correctly: a silent version of that
    mismatch would be silently wrong gradients. Those arms run
    uncheckpointed on a card that holds them. sft.py's enable
    (use_reentrant=False) is the in-repo precedent; use_cache goes off
    with it. Returns True when enabled."""
    tk = _train_kwargs_default(cfg)
    per_arm = {a: bool(((cfg.get("train_kwargs") or {}).get(a) or {})
                       .get("gradient_checkpointing",
                            tk.get("gradient_checkpointing", False)))
               for a in (arms or [])}
    if arms:
        want = set(per_arm.values())
        if len(want) > 1:
            # The enable is model-global per process; a mixed multi-arm
            # single-process invocation cannot honour both. Every launcher
            # in this campaign passes one arm per process (Modal per-cell
            # containers; the cluster scheduler's arm tasks), so refuse
            # the mix rather than silently training half the arms wrong.
            raise SystemExit(
                "[train-two-si] mixed gradient_checkpointing values across "
                f"arms in one process ({per_arm}) — run these arms as "
                "separate invocations/arm-tasks.")
        enabled = want.pop()
    else:
        enabled = bool(tk.get("gradient_checkpointing", False))
    if not enabled:
        return False
    if model is None:
        return False
    if not getattr(model, "supports_gradient_checkpointing", False):
        raise SystemExit(
            "[train-two-si] train_kwargs.default.gradient_checkpointing is "
            f"set but {type(model).__name__} does not support gradient "
            "checkpointing — refusing to run a fit the config sized for "
            "checkpointed memory without it.")
    # HF's decoder layers consult the flag ONLY under ``self.training``
    # (``if self.gradient_checkpointing and self.training:``), and this
    # lane's models load in eval() — the first checkpointed probe wave
    # "enabled" it and measured byte-identical OOMs. So the model must run
    # train-mode, which is only safe when nothing in the forward is
    # stochastic: refuse any nonzero dropout rather than silently changing
    # the forward math (RMSNorm/LayerNorm are mode-invariant; dropout is
    # THE train-mode behavior; these fits keep every base weight frozen,
    # so no batchnorm-style running stats exist to drift).
    confs = [c for c in (getattr(model, "config", None),) if c is not None]
    if confs and getattr(confs[0], "text_config", None) is not None:
        confs.append(confs[0].text_config)
    bad = {}
    for c in confs:
        for k, v in vars(c).items():
            if (k.endswith("dropout") and isinstance(v, (int, float))
                    and float(v) != 0.0):
                bad[k] = v
    if bad:
        raise SystemExit(
            "[train-two-si] gradient_checkpointing needs train-mode "
            f"forwards, but the model config has nonzero dropout {bad} — "
            "train-mode would change the forward math. Refusing.")
    if bool(cfg.get("teacher_prefix_cache", True)):
        # HF forces use_cache=False under train-mode checkpointing, so the
        # teacher prefix KV build comes back past_key_values=None and the
        # cache constructor TypeErrors ('NoneType' object is not iterable
        # — found by the mini-fit test 2026-08-27). The checkpointing
        # consumers (the Qwen fits) run teacher_prefix_cache: false anyway;
        # refuse the combination loudly until someone needs both at once.
        raise SystemExit(
            "[train-two-si] gradient_checkpointing is incompatible with "
            "teacher_prefix_cache: true (HF disables use_cache under "
            "train-mode checkpointing, so the teacher KV build returns "
            "None). Set teacher_prefix_cache: false in this config.")
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False})
    for c in confs:
        c.use_cache = False
    model.train()
    print("[train-two-si] gradient checkpointing ON (train-mode + "
          "use_reentrant=False + use_cache=False; dropout verified 0.0)",
          flush=True)
    return True


def _cfg_with_positions(cfg: dict, positions: str) -> dict:
    """A COPY of ``cfg`` whose ``train_kwargs.default.positions`` is
    ``positions`` — the ``--positions`` CLI override (the Class C
    submission path). A copy, never a mutation, so nothing already derived
    from the caller's cfg shifts under it. The override flows through the
    ONE existing read (:func:`train_si_objects`'s ``tk.get("positions")``)
    into the lockstep builder, the solo arms' PROTECTED kwargs, and each
    artifact's ``positions``/``positions_at_fit`` provenance record —
    deliberately not a second plumbing path."""
    from sisteer.hooks import FIT_POSITION_MODES

    if positions not in FIT_POSITION_MODES:
        raise SystemExit(
            f"[train-two-si] --positions must be one of "
            f"{FIT_POSITION_MODES}, got {positions!r}")
    cfg = dict(cfg)
    tks = dict(cfg.get("train_kwargs") or {})
    tks["default"] = {**(tks.get("default") or {}), "positions": positions}
    cfg["train_kwargs"] = tks
    return cfg


def _effective_positions(args, cfg) -> str:
    """The fit's positions leg for this invocation: the ``--positions``
    flag when given, else the config's ``train_kwargs.default.positions``
    (in-code default ``"all"`` — the same read train_si_objects performs)."""
    return str(getattr(args, "positions", "")
               or _train_kwargs_default(cfg).get("positions", "all"))


def _safe_registry_train(train_fn, model, tok, instruction, pairs, *, layer,
                         candidate_kw, log_fn, protected=frozenset()):
    """Call a registry trainer, passing only the kwargs it accepts. Python
    raises ``TypeError: unexpected keyword argument 'X'`` at CALL time (before
    any training), so on that error we drop X and retry — this lets the driver
    hand every arm its full kwarg set (epochs / batch_size / max_len / seed)
    without crashing a trainer whose signature omits one and takes no
    ``**kw`` (today that is alphasteer, which has no ``seed``; the
    once-cited alphakl / antidote grew ``**kw`` and no longer trip this).
    Only keys WE supplied are ever dropped; a key in
    ``protected`` (per-arm config train_kwargs, the condition prompts) is
    never silently dropped — a typo'd config key must fail LOUDLY, not vanish.

    Returns ``(object, surviving_kwargs)``. The surviving kwargs are the call
    the fit REALLY ran; the sidecar's fit-time provenance reads its
    ``batch_size`` from them, so a width the retry dropped is recorded as
    null — never as a number that never ran (provenance audit 2026-08-21)."""
    kw = dict(candidate_kw)
    while True:
        try:
            return train_fn(model, tok, instruction, pairs, layer=layer,
                            log_fn=log_fn, **kw), kw
        except TypeError as e:
            m = re.search(r"unexpected keyword argument '(\w+)'", str(e))
            if m and m.group(1) in kw and m.group(1) not in protected:
                kw.pop(m.group(1))
                continue
            raise


# --------------------------------------------------------------------------- #
# Per-arm outcome tracking (jobs 148862/148863: one arm's failure must never   #
# discard another arm's finished work)                                         #
# --------------------------------------------------------------------------- #


@dataclass
class ArmOutcome:
    """The terminal state of one requested arm in one run. ``detail``
    carries the numeric fit record (held-out curve, chosen stopping epoch,
    no-op-floor probes — including the probe's per-example values and its
    selection's sha8 keys, so cells are paired-comparable from disk) so the
    report can justify every arm's stopping point. Numbers and hash
    prefixes only, never prompt/pair text."""

    status: str  # "trained" | "skipped_existing" | "failed"
    error_type: str | None = None
    error: str | None = None
    detail: dict | None = None

    def as_dict(self) -> dict:
        d = {"status": self.status}
        if self.status == "failed":
            d["error_type"] = self.error_type
            d["error"] = self.error
        if self.detail is not None:
            d["detail"] = self.detail
        return d


@dataclass
class FitOutcome:
    """What one ``train_si_objects`` run did, arm by arm. ``objects`` holds
    the objects trained IN THIS RUN (a resumed-over arm is on disk, not in
    memory); ``arms`` records every requested arm's terminal state in attempt
    order."""

    si_id: str
    objects: dict[str, object] = field(default_factory=dict)
    arms: dict[str, ArmOutcome] = field(default_factory=dict)
    condition_sources: dict | None = None
    report_path: Path | None = None

    @property
    def failed(self) -> dict[str, ArmOutcome]:
        return {a: o for a, o in self.arms.items() if o.status == "failed"}

    @property
    def skipped(self) -> list[str]:
        return [a for a, o in self.arms.items()
                if o.status == "skipped_existing"]

    @property
    def ok(self) -> bool:
        return not self.failed


def report_path_for(save_dir, si_id: str) -> Path:
    """The per-run fit report beside the cell dirs. A FILE named
    ``<si>__fit_report.json`` cannot collide with a ``<si>__<arm>`` cell
    DIRECTORY, and nothing globs ``defenses_dir`` (readers resolve exact
    cell-id paths via ``object_path_for``)."""
    return Path(save_dir) / f"{si_id}__fit_report.json"


def metrics_path_for(save_dir, si_id: str) -> Path:
    """One SI fit run's shared metrics-jsonl tee. Owned here because
    external monitors (pod_acceptance_probe's step counter) poll this
    exact path — a rename must move every consumer at once."""
    return Path(save_dir) / f"{si_id}__metrics.jsonl"


def arm_report_path_for(save_dir, si_id: str, arm: str) -> Path:
    """One arm task's report FRAGMENT (the parallel driver's unit of
    record): same directory, arm-suffixed filename, merged into the
    canonical ``report_path_for`` file by the scheduler. A separate file
    per arm is what lets concurrent arm processes record their terminal
    states without a shared-file write race."""
    return Path(save_dir) / f"{si_id}__fit_report.arm_{arm}.json"


# --------------------------------------------------------------------------- #
# Fit identity (adversarial review F15, 2026-08-25): a COMPLETE cell is bound  #
# to WHAT was fitted — model+revision, SI, arm, position leg, pool bytes,      #
# resolved optimizer settings, config identity — via the shared               #
# fit_fingerprint utility. Resume matches fingerprints, never names: an       #
# all-token marker can no longer be accepted as the response-leg cell, nor a  #
# Gemma cell as a Qwen one.                                                    #
# --------------------------------------------------------------------------- #


def pool_content_sha256(pairs) -> str:
    """Content hash over LOADED Pair rows — the programmatic-caller fallback
    for the pairs-FILE byte hash (tests and orchestrators hand
    ``train_si_objects`` a list, not a path). Canonical per row (sorted
    keys, every identity-bearing field), so the same pool always hashes the
    same way at publish time and at skip-check time."""
    h = hashlib.sha256()
    for p in pairs:
        h.update(json.dumps({
            "prompt": p.prompt, "pos": p.pos, "neg": p.neg,
            "instruction_id": getattr(p, "instruction_id", ""),
            "requirement_a": getattr(p, "requirement_a", None),
            "anti_neg": getattr(p, "anti_neg", None),
        }, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


#: Slurm lanes (2026-09-06): the campaign configs name their shared held-out
#: pool by its Modal-volume path (``/vol/pool/<key>/pairs_constv2-heldout.
#: jsonl``), and the config FILE's bytes are part of every cell's fit
#: identity, so that path cannot be rewritten per host. Where no ``/vol``
#: exists (a cluster node — ``/`` is not writable there), ``SISTEER_VOL_ROOT``
#: names the directory that plays the volume's root, and every ``/vol/...``
#: path a config declares resolves under it. Unset = today's behaviour byte
#: for byte. The remap is PATH RESOLUTION ONLY: the fit identity hashes the
#: pool's CONTENT (:func:`heldout_pool_identity`'s sha), the budget sidecar
#: records that sha, and nothing records the path — so a Slurm cell and a
#: Modal cell of the same config + pool bytes carry the same fingerprint.
#: Never set it where a real ``/vol`` is mounted.
VOL_ROOT_ENV = "SISTEER_VOL_ROOT"


SHARED_ROOT_ENV = "SISTEER_SHARED_ROOT"   # the front door's job env; its volume is <shared root>/vol


VOL_MOUNT = "/vol"   # the paper's volume mountpoint (a seam the tests point at a scratch directory)


def _mountinfo_text() -> str:
    """This process's mount table (``/proc/self/mountinfo``; empty where there is none). A seam the tests replace."""
    try:
        return Path("/proc/self/mountinfo").read_text()
    except OSError:
        return ""


def _mountinfo_unescape(field: str) -> str:
    # mountinfo octal-escapes space, tab, newline and backslash in path fields
    return field.replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")


def _vol_mounted() -> bool:
    """Whether the volume mountpoint is a REAL mount on this host: a
    pre-created but unattached ``/vol`` directory (a container whose mount
    did not attach) must NOT be read as the volume — it would win over a
    valid pool under ``$SISTEER_SHARED_ROOT/vol`` and the trial would open
    an empty or stale directory under a false source. The mountpoint is
    resolved first (a symlinked ``/vol``); a mount is either what
    ``os.path.ismount`` sees or a mount-table entry whose mount point
    (mountinfo field 5) is the resolved path — bind mounts of a directory
    on the same filesystem are invisible to ismount and visible in the
    table. A plain directory defers to the shared-root fallback."""
    p = Path(VOL_MOUNT)
    if not p.is_dir():
        return False
    try:
        real = str(p.resolve())
    except OSError:
        return False
    if os.path.ismount(real):
        return True
    for line in _mountinfo_text().splitlines():
        parts = line.split()
        if len(parts) >= 5 and _mountinfo_unescape(parts[4]) == real:
            return True
    return False


def vol_root_with_source() -> tuple[str | None, str]:
    """``(root, source)`` — the host's volume root and where it came from,
    in THIS order: the explicit
    ``$SISTEER_VOL_ROOT`` override; else a MOUNTED ``/vol`` directory
    (root None: a ``/vol/…`` path is read as spelled — a host with the
    paper's mount and an inherited front-door shared root must never open
    a different pool under ``$SHARED/vol``); else ``$SISTEER_SHARED_ROOT/vol``
    when the front door's shared root is set (every submission's job env
    carries it; stage E, 2026-09-30: the ReLU trials refused a held-out
    pool spelled ``/vol/pool/…`` while the file sat under
    ``$SISTEER_SHARED_ROOT/vol`` on a cluster with no ``/vol``); else None
    (a plain path)."""
    root = os.environ.get(VOL_ROOT_ENV, "").strip()
    if root:
        return root, VOL_ROOT_ENV
    if _vol_mounted():
        return None, f"{VOL_MOUNT} (mounted)"
    shared = os.environ.get(SHARED_ROOT_ENV, "").strip()
    if shared:
        return str(Path(shared) / "vol"), f"{SHARED_ROOT_ENV}/vol"
    return None, "none"


def vol_root() -> str | None:
    """The host's volume root (:func:`vol_root_with_source`), None when a
    ``/vol/…`` path is to be read as spelled."""
    return vol_root_with_source()[0]


def vol_root_source() -> str:
    """Which root a ``/vol/…`` path resolves through here — recorded by the
    trial / fit summaries beside the resolved path."""
    return vol_root_with_source()[1]


def resolve_vol_path(path) -> str:
    """``path`` as this host reads it: a ``/vol/...`` path under the host's
    volume root (:func:`vol_root`) when one is known, else ``path``
    unchanged. THE one mapping: the fits' held-out pool identity, the
    trials' held-out pool, every ``/vol/`` line of a fits config."""
    s = str(path)
    root = vol_root()
    if not root or not (s == "/vol" or s.startswith("/vol/")):
        return s
    return root if s == "/vol" else str(Path(root) / s[len("/vol/"):])


def vol_path_state() -> str:
    """How a ``/vol/…`` path resolves on this host, for a refusal by name."""
    vol = "mounted" if _vol_mounted() else ("a plain directory (no mount attached)" if Path(VOL_MOUNT).is_dir() else "absent")
    return (f"{VOL_ROOT_ENV}={os.environ.get(VOL_ROOT_ENV) or 'unset'}, {SHARED_ROOT_ENV}={os.environ.get(SHARED_ROOT_ENV) or 'unset'}, "
            f"{VOL_MOUNT} {vol} -> root source {vol_root_source()}")


def heldout_pool_identity(cfg: dict, si_id: str) -> tuple[str | None,
                                                          str | None]:
    """``(path, sha256)`` of the SHARED held-out pool this SI's fit would
    rule its early stopping on (``train_kwargs.default.heldout_pool``,
    per-SI merged exactly the way ``train_si_objects`` reads it), or
    ``(None, None)`` when the config declares none. ONE derivation, shared
    by :func:`cell_fit_fingerprint` (the identity every COMPLETE-skip and
    checkpoint key binds), the fit body (the sha the budget sidecar
    records) and the resume gate (:func:`check_budget_sidecars`), so the
    fingerprint, the record and the comparison can never disagree.

    A declared-but-missing pool file REFUSES loudly here — the sha is part
    of the fit's identity, so it must resolve BEFORE any COMPLETE-skip or
    checkpoint identity is computed, never be guessed around."""
    tk = _train_kwargs_for_si(cfg, si_id)
    hp = tk.get("heldout_pool")
    if not hp:
        return None, None
    # The DECLARED path is what this function returns (the config's own
    # spelling, host-independent); the bytes are read where this host has
    # them (resolve_vol_path — SISTEER_VOL_ROOT on a cluster, identity on
    # Modal / locally).
    p = Path(resolve_vol_path(hp))
    if not p.is_file():
        via = (f" (resolved through {VOL_ROOT_ENV} to {p})"
               if str(p) != str(hp) else "")
        raise SystemExit(
            f"[train-two-si] train_kwargs.default.heldout_pool={hp}{via} "
            "does not exist — the shared held-out pool's CONTENT hash is part "
            "of every cell's fit identity (COMPLETE-skip and checkpoint "
            "key), so it must be resolvable before any resume decision; "
            "stage the pool or drop the key.")
    return str(hp), _sha256_file(p)


def cell_fit_fingerprint(cfg: dict, si_id: str, arm: str, *, positions: str,
                         pool_sha256: str | None,
                         config_sha: str | None,
                         teacher: str = "online",
                         continued_from: dict | None = None) -> str:
    """The canonical fit-input fingerprint for ONE requested cell — the
    identity stored in its COMPLETE marker and required to match before any
    resume skip (F15). ``optimizer`` is the RESOLVED settings mapping from
    :func:`effective_optimizer_settings` (values only — lr, lr_schedule,
    warmup_steps, warmup_frac, max_grad_norm, and accum_steps in BOTH
    modes since the re-tune@32: an effective-32 fit is a different
    identity than an effective-2 one), resolved under the mode the arm
    actually trains in (lockstep vs solo — the same predicate
    ``partition_arms`` uses), so a config pin that never reaches the
    trainer can't alias two different fits.

    Under a SHARED held-out pool (``train_kwargs.default.heldout_pool``,
    sdf-base-steering §5.3) the pool's CONTENT hash joins the identity —
    resolved HERE via :func:`heldout_pool_identity`, so every caller
    (the fit body's ``_cell_fp``, the scheduler's resume gate, the arm
    task) binds it without a call-site change: mutated held-out bytes at
    the same path change the fingerprint, which refuses the COMPLETE skip
    AND the checkpoint load (both key on this value). It rides inside the
    ``optimizer`` mapping — the fingerprint's one extensible slot, the
    ``teacher`` precedent — and ONLY when a pool is declared, so every
    legacy (non-heldout_pool) fingerprint is byte-identical.

    ``continued_from`` (``sisteer.extraction.continuation.
    continuation_marker``): a checkpoint continuation under a raised cap is
    a DIFFERENT fit from a scratch fit at that cap (under warmup_cosine the
    pre-checkpoint steps carry the prior cap's schedule), so the prior fit's
    fingerprint, the exact resume position and the new cap ride in the same
    extensible slot — a scratch cap-N fit never matches a continued cell,
    and the cap-M publication's COMPLETE never skips the continuation."""
    from sisteer.extraction.fit_fingerprint import fit_fingerprint

    return fit_fingerprint(cell_fingerprint_fields(
        cfg, si_id, arm, positions=positions, pool_sha256=pool_sha256,
        config_sha=config_sha, teacher=teacher, continued_from=continued_from))


def cell_fingerprint_fields(cfg: dict, si_id: str, arm: str, *,
                            positions: str, pool_sha256: str | None,
                            config_sha: str | None, teacher: str = "online",
                            continued_from: dict | None = None) -> dict:
    """The un-hashed identity :func:`cell_fit_fingerprint` hashes — exposed
    so a test (or an operator) can show WHICH field two identities differ
    in, not just that the hashes do."""
    from sisteer.extraction.lockstep import lockstep_compatible

    eff = effective_optimizer_settings(cfg, arm,
                                       lockstep=lockstep_compatible(arm))
    # The teacher leg rides inside the optimizer mapping (the
    # fingerprint's one extensible slot): consuming a top-K head
    # trains a DIFFERENT (renormalized-head) objective than the
    # full-vocab online teacher — measured 7-18% apart at T=1 on the
    # production pools — so the same hp against a different teacher
    # leg is different work (tune_arm's trial fingerprint makes the
    # same distinction). "online", or "topk:<manifest sha16>".
    optimizer = {**{k: v for k, (v, _src) in eff.items()},
                 "teacher": teacher}
    _hp_path, hp_sha = heldout_pool_identity(cfg, si_id)
    if hp_sha is not None:
        optimizer["heldout_pool_sha"] = hp_sha
    # The CE trainer's loss IMPLEMENTATION (full-sequence logits
    # vs chunked LM head) is a fit-identity dimension — bf16 numerics
    # differ at the kernel level — and rides in the same extensible slot,
    # ONLY when a config selects the non-default, so every existing
    # fingerprint (and an explicit `loss_impl: full`, the same work) stays
    # byte-identical.
    impl = resolved_loss_impl(cfg, arm)
    if impl != "full":
        optimizer["loss_impl"] = impl
    if continued_from:
        optimizer["continued_from"] = dict(continued_from)
    return {
        "model": cfg.get("model"),
        "model_revision": cfg.get("model_revision"),
        "si_id": si_id,
        "arm": arm,
        "positions": positions,
        "pool_sha256": pool_sha256,
        "optimizer": optimizer,
        "config_sha": config_sha,
    }


def resolved_loss_impl(cfg: dict, arm: str) -> str:
    """The CE loss implementation ``train_kwargs.<arm>.loss_impl`` selects
    for this arm (per-arm block ONLY — that is the one path that reaches
    train_object_ce, as a PROTECTED kwarg; a default-level key is refused
    by refuse_ce_misconfiguration), ``"full"`` when unset. Validated
    against the trainer's own LOSS_IMPLS."""
    from sisteer.extraction.chunked_ce import LOSS_IMPLS

    impl = ((cfg.get("train_kwargs") or {}).get(arm) or {}).get("loss_impl")
    if impl is None:
        return "full"
    if impl not in LOSS_IMPLS:
        raise SystemExit(
            f"[train-two-si] train_kwargs.{arm}.loss_impl={impl!r} is not "
            f"one of {list(LOSS_IMPLS)}")
    return str(impl)


def _teacher_leg_tag(teacher_topk, cfg: dict, arm: str) -> str:
    """The teacher-leg identity for one cell's fingerprint — "online", or
    "topk:<manifest sha16>" when THIS arm would consume the artifact at
    ``teacher_topk`` (lockstep-compatible, plain forward KL at an audited
    temperature, manifest present). Derived from the REQUEST (resolved
    path + config), never from what later ran: the scheduler's skip gate
    and the arm task must derive the identical tag without loading the
    store, so validation-time fallback (a key-mismatched artifact — which
    also logs loudly and stamps meta.teacher_forward) does not shift the
    fingerprint; a REBUILT artifact changes the manifest sha and re-runs
    the cell."""
    if not teacher_topk:
        return "online"
    from sisteer.extraction.lockstep import lockstep_compatible
    from sisteer.extraction.teacher_topk import AUDITED_TOPK_TEMPERATURES

    if not lockstep_compatible(arm):
        return "online"
    params = lockstep_arm_params(cfg, arm)
    if (params.loss != "soft"
            or params.temperature not in AUDITED_TOPK_TEMPERATURES):
        return "online"
    manifest = Path(str(teacher_topk) + ".json")
    if not manifest.is_file():
        return "online"
    return "topk:" + hashlib.sha256(
        manifest.read_bytes()).hexdigest()[:16]


def _cli_fit_identity(args, cfg) -> tuple[str, str]:
    """``(pool_sha256, config_sha)`` as EVERY CLI path derives them — the
    pairs FILE's byte hash and the config FILE's byte hash (falling back to
    the cfg-dict hash when no file is readable). ONE derivation shared by
    ``cmd_train``, the scheduler's resume gate and ``cmd_arm_task``, so the
    fingerprint the arm task stores is byte-identical to the one the
    scheduler checks."""
    pool_sha = _sha256_file(Path(args.pairs))
    config_sha = (_config_sha(str(getattr(args, "config", "") or ""))
                  or _cfg_dict_sha(cfg))
    return pool_sha, config_sha


# --------------------------------------------------------------------------- #
# Checkpoint continuation under a raised cap                                   #
# --------------------------------------------------------------------------- #
# The semantics live on sisteer.extraction.continuation (module docstring):
# the retained <cell>/final_ckpt/ckpt.pt is ADOPTED under the continued fit's
# fingerprint and the ordinary trainer resumes it with epochs = the new cap.
# The driver's job is identity: prove the checkpoint is the prior fit's under
# EXACTLY this config, prove it belongs to the published object, refuse
# everything else loudly, and record what ran.


def _continue_lr_mult_arg(args) -> float:
    """``--continue-lr-mult`` as the driver received it: 1.0 when absent,
    else the raw float — an explicit 0 (or any non-positive value) is
    PRESERVED so ``check_lr_mult`` refuses it, never coerced to 1.0 by a
    truthiness default (`0 or 1.0` would have run a full-rate restart
    under the ordinary fingerprint)."""
    raw = getattr(args, "continue_lr_mult", None)
    return 1.0 if raw is None else float(raw)


def _prepare_continuation(cfg: dict, si_id: str, arms: list[str], *,
                          model, tok, positions: str, pool_sha256,
                          config_sha, teacher_topk, checkpoint_dir,
                          continue_from, epochs_cap, prior_cap: int,
                          continue_lr_mult=1.0) -> dict:
    """Resolve + VERIFY a checkpoint continuation before any arm
    trains. Returns ``{"arm", "state", "summary", "marker", "record"}`` —
    ``marker`` joins the cell's fingerprint (``cell_fit_fingerprint``),
    ``record`` becomes the ``continued_from`` provenance block. Refuses
    (SystemExit): a cap override without a checkpoint (a scratch fit at
    another cap is a config change), a checkpoint without a cap, no
    ``--checkpoint-dir`` (the adopted checkpoint reaches the trainer
    through its fitckpt seam), more than one arm or a non-CE arm, a cap
    below the PRIOR cap (the config's for a first continuation, the last
    chain link's for a chained one) or equal to it with the same lr
    multiplier, a checkpoint whose key the chain from
    this config's plain identity does not reproduce
    (``continuation.resolve_chain`` — anything but the cap and the lr
    multiplier differs, or the publication's recorded chain is broken), or
    a checkpoint that does not belong to the cell's published object
    (``continuation.verify_cell``). When the published best POSTDATES the
    checkpoint, the published object is loaded through its own class
    (``load_object`` — the authoritative ``.trainable`` order) and seated
    as the incumbent best with its patience counter reconstructed from the
    published curve (``continuation.incumbent_stopper``), so early stopping
    in the continuation can never ship an object worse than the one it
    continued from.

    ``continue_lr_mult`` scales the restarted schedule's peak; it rides in
    the marker (so the fingerprint), the provenance block and the lr report,
    and reaches the trainer through
    :func:`_continuation_schedule_report`. Refused without a continuation
    (a multiplier on a scratch fit is a config change) and when not a
    finite positive number.

    The stopper rule is not a knob: every continuation runs under
    ``replay_exempt`` — the replayed remainder of the checkpoint epoch is
    logged but neither a strike nor a new best — and says so in its marker
    (``stopper_rule``), so these fits are a distinct generation from the
    legacy continuations; a same-cap same-multiplier re-issue of a legacy
    chained continuation is therefore admissible. The exemption reaches the
    trainer through :func:`_continuation_stopper_rule` as
    ``replay_exempt_epoch`` = the resume epoch, PROTECTED."""
    from sisteer.extraction.continuation import (
        FINAL_CKPT_DIR, STOPPER_RULE_REPLAY_EXEMPT, ContinuationError,
        check_cap_admissible, check_lr_mult, continuation_marker,
        incumbent_stopper, load_state, marker_stopper_rule, resolve_chain,
        summarize, verify_cell)

    tag = "[train-two-si] continuation"
    # The rule every continuation runs under since this change — the
    # replayed remainder of the checkpoint epoch is logged, never
    # ruled on. It rides in the marker (so the fingerprint) and reaches the
    # trainer PROTECTED (_continuation_stopper_rule).
    stopper_rule = STOPPER_RULE_REPLAY_EXEMPT
    try:
        lr_mult = check_lr_mult(continue_lr_mult)
    except ContinuationError as e:
        raise SystemExit(f"{tag}: --continue-lr-mult: {e}") from e
    if continue_from is None:
        if lr_mult != 1.0:
            raise SystemExit(
                f"{tag}: --continue-lr-mult {lr_mult} without "
                "--continue-from — the multiplier scales a CONTINUATION's "
                "restarted schedule; on a scratch fit the lr is a config "
                "value (train_kwargs.<arm>.lr)")
        raise SystemExit(
            f"{tag}: --epochs-cap without --continue-from — a scratch fit at "
            f"another cap is a CONFIG change (train_kwargs_by_si.{si_id}."
            "epochs), which moves the fingerprint by itself; the cap "
            "override exists only to continue a retained checkpoint")
    if epochs_cap is None:
        raise SystemExit(f"{tag}: --continue-from without --epochs-cap — "
                         "state the raised cap explicitly")
    if checkpoint_dir is None:
        raise SystemExit(
            f"{tag}: needs --checkpoint-dir — the adopted checkpoint reaches "
            "the trainer through its fitckpt seam (<dir>/<cell>/ckpt.pt)")
    if len(arms) != 1 or arms[0] not in CE_ARMS:
        raise SystemExit(
            f"{tag}: exactly ONE document-CE arm per continuation (a "
            f"checkpoint is per cell; only {list(CE_ARMS)} checkpoint the "
            f"full early-stopping state), got {arms}")
    arm = arms[0]
    new_cap = int(epochs_cap)
    ckpt = Path(continue_from)
    try:
        state = load_state(ckpt)
    except ContinuationError as e:
        raise SystemExit(f"{tag}: {e}") from e
    summ = summarize(state, ckpt)
    cell_id = f"{si_id}__{arm}"
    cell_dir = ckpt.parent.parent
    if ckpt.parent.name != FINAL_CKPT_DIR or cell_dir.name != cell_id:
        raise SystemExit(
            f"{tag}: {ckpt} is not <cell>/{FINAL_CKPT_DIR}/ckpt.pt of "
            f"{cell_id} — continue only from a COMPLETE cell's retained "
            "checkpoint")
    # The identity gate: the checkpoint key must be reproduced from THIS
    # config's plain identity through the publication's recorded chain of
    # continuation markers (empty for a first continuation, non-empty
    # for a chained one) — one derivation, shared with the Modal app's
    # pre-spawn fingerprint_cpu and snapshot modes.
    teacher = _teacher_leg_tag(teacher_topk, cfg, arm)

    def _fp_of(marker):
        return cell_fit_fingerprint(
            cfg, si_id, arm, positions=positions, pool_sha256=pool_sha256,
            config_sha=config_sha, teacher=teacher,
            continued_from=marker or None)

    plain_fp = _fp_of(None)
    try:
        chain = resolve_chain(cell_dir, state, plain_fp, _fp_of)
    except ContinuationError as e:
        raise SystemExit(f"{tag}: {e}") from e
    prior_fp = summ["checkpoint_key"]   # the fit the checkpoint belongs to
    # The cap / multiplier / tail rules — the SAME helper the Modal app's
    # pre-spawn derivation runs, so nothing refused here was snapshotted or
    # spawned (continuation.check_cap_admissible).
    try:
        adm = check_cap_admissible(chain=chain, summary=summ, new_cap=new_cap,
                                   lr_mult=lr_mult, config_cap=prior_cap,
                                   stopper_rule=stopper_rule)
    except ContinuationError as e:
        raise SystemExit(f"{tag}: {e} [{si_id}]") from e
    eff_prior_cap = int(adm["prior_cap"])
    ver = verify_cell(cell_dir, state,
                      expect_revision=cfg.get("model_revision"))
    if ver["problems"]:
        raise SystemExit(
            f"{tag}: {ckpt} does not belong to the published cell "
            f"{cell_dir}:\n  - " + "\n  - ".join(ver["problems"]))
    checks = ver["checks"]
    stopper_override = None
    incumbent = {"source": "checkpoint_snapshot",
                 "best_epoch": (summ["stopper"] or {}).get("best_epoch"),
                 "best_heldout_ce": (summ["stopper"] or {}).get("best")}
    if checks.get("incumbent") == "published_object":
        from sisteer.methods import load_object

        pub = load_object(cell_dir / checks["object_file"], model=model,
                          tok=tok)
        try:
            stopper_override = incumbent_stopper(
                state, published_trainable=pub.trainable,
                best_ce=checks["incumbent_best_heldout_ce"],
                best_epoch=checks["incumbent_best_epoch"],
                since_best=checks["incumbent_since_best"])
        except ContinuationError as e:
            raise SystemExit(f"{tag}: {e}") from e
        incumbent = {"source": "published_object",
                     "object_sha256": checks.get("object_sha256"),
                     "best_epoch": checks["incumbent_best_epoch"],
                     "best_heldout_ce": checks["incumbent_best_heldout_ce"],
                     "since_best": checks["incumbent_since_best"],
                     "checkpoint_best_epoch": (summ["stopper"] or {}).get(
                         "best_epoch")}
        print(f"{tag}: the published best (epoch "
              f"{checks['incumbent_best_epoch']}, held-out CE "
              f"{checks['incumbent_best_heldout_ce']:.6f}) postdates the "
              f"checkpoint's (epoch {incumbent['checkpoint_best_epoch']}) — "
              "the PUBLISHED object is the incumbent best; since_best="
              f"{checks['incumbent_since_best']} from the published curve",
              flush=True)
    marker = continuation_marker(summ, new_cap, lr_mult, stopper_rule)
    assert marker_stopper_rule(marker) == stopper_rule == adm["stopper_rule"]
    record = {
        "incumbent": incumbent,
        "prior_fit_fingerprint": prior_fp,
        "root_fit_fingerprint": chain["root_fit_fingerprint"],
        "prior_object_sha256": checks.get("object_sha256"),
        "prior_cap": eff_prior_cap,
        "prior_lr_mult": float(chain["prior_lr_mult"]),
        # the early-stopping rule this continuation runs under and
        # the one the prior link ran under (None: a first continuation);
        # the resume epoch (0-based) is the one the trainer exempts.
        "stopper_rule": stopper_rule,
        "prior_stopper_rule": adm["prior_stopper_rule"],
        "replay_exempt_epoch": int(summ["epoch"]),
        "prior_replayed_partial_epochs": list(
            summ.get("replayed_partial_epochs") or []),
        "prior_epochs_run": checks.get("prior_epochs_run"),
        "prior_best_epoch": checks.get("prior_best_epoch"),
        "prior_optimizer_steps_run": checks.get("prior_optimizer_steps_run"),
        "resume_step": summ["step"],
        "resume_epoch": summ["epoch"],
        "resume_batches_done": summ["batches_done"],
        "n_batches": summ["n_batches"],
        "resume_stopper": summ["stopper"],
        "new_cap": new_cap,
        "lr_mult": lr_mult,
        # The FULL chain this publication descends through: every prior
        # continuation's marker, oldest first, then this one's — what the
        # next continuation walks from the plain identity.
        "chain": [dict(m) for m in chain["chain"]] + [dict(marker)],
        "chain_depth": int(chain["depth"]) + 1,
        "checkpoint": str(ckpt),
        "snapshot": (checks.get("superseded_by") or {}).get("snapshot"),
        "identity_checks": checks,
    }
    print(f"{tag}: {cell_id} resumes the cap-{eff_prior_cap} fit "
          f"{prior_fp[:12]}… (chain depth {chain['depth']}, root "
          f"{plain_fp[:12]}…) from optimizer step {summ['step']} (epoch "
          f"index {summ['epoch']}, micro-batch {summ['batches_done']}/"
          f"{summ['n_batches']}) under cap {new_cap}, lr_mult {lr_mult}, "
          f"stopper rule {stopper_rule!r} (prior link's: "
          f"{adm['prior_stopper_rule']!r}); identity verified against "
          f"{cell_dir}", flush=True)
    return {"arm": arm, "state": state, "summary": summ,
            "marker": marker, "record": record,
            "stopper_override": stopper_override}


def _adopt_continuation(continuation: dict, ck: dict) -> None:
    """Put the prior checkpoint into the continued fit's transient slot
    (``ck`` = the trainer's checkpoint kwargs for the cell). A checkpoint
    ALREADY there under the continued identity at a later step is a
    successor call's own progress (the 24 h Modal ceiling) and wins — never
    re-adopt over it."""
    from sisteer.extraction.continuation import TRAINER, adopt
    from sisteer.extraction.fitckpt import load_checkpoint

    if not ck:
        raise SystemExit("[train-two-si] continuation: no checkpoint kwargs "
                         "resolved — --checkpoint-dir is required")
    summ = continuation["summary"]
    existing = load_checkpoint(
        ck["checkpoint_dir"], ck["checkpoint_key"],
        expect={"trainer": TRAINER, "n_batches": summ["n_batches"]})
    if existing is not None:
        if int(existing["step"]) < int(summ["step"]):
            raise SystemExit(
                f"[train-two-si] continuation: {ck['checkpoint_dir']} holds "
                f"a checkpoint of this continuation at step "
                f"{existing['step']}, EARLIER than the adopted resume step "
                f"{summ['step']} — an impossible state; refusing")
        continuation["record"]["adoption"] = {
            "adopted": False, "transient_step": int(existing["step"]),
            "note": "resumed from this continuation's own later checkpoint "
                    "(a successor call)"}
        print(f"[train-two-si] continuation: {ck['checkpoint_dir']} already "
              f"holds this continuation's checkpoint at step "
              f"{existing['step']} — resuming from it, not re-adopting",
              flush=True)
        return
    path = adopt(continuation["state"], new_key=ck["checkpoint_key"],
                 dest_dir=ck["checkpoint_dir"], marker=continuation["marker"],
                 source=summ["path"],
                 stopper=continuation.get("stopper_override"))
    continuation["record"]["adoption"] = {"adopted": True, "path": str(path)}
    print(f"[train-two-si] continuation: adopted {summ['path']} -> {path} "
          f"under the continuation fingerprint "
          f"{ck['checkpoint_key'][:12]}…", flush=True)


def _continuation_schedule_report(continuation: dict, candidate_kw: dict,
                                  meta: dict) -> dict:
    """Record (and print) what the lr does at the resume step under the
    prior vs the new horizon — the warm-restart size, on the record — and
    APPLY the continuation's lr multiplier: this is the one place
    ``--continue-lr-mult`` reaches the trainer, as ``candidate_kw
    ["lr"] = config lr x lr_mult`` right after the report is taken from the
    config value. The schedule is linear in the peak (lockstep's
    ``_lr_at_factory``), so every step trains at exactly lr_mult x the
    unscaled trajectory; warmup and cosine shape are untouched."""
    from sisteer.extraction.continuation import schedule_report

    rec = continuation["record"]
    lr = candidate_kw.get("lr")
    if lr is None:
        raise SystemExit("[train-two-si] continuation: the arm has no "
                         "resolved lr in its train_kwargs — refusing to "
                         "describe a schedule it would run at an unknown "
                         "rate")
    rep = schedule_report(
        lr=float(lr), lr_schedule=candidate_kw.get("lr_schedule"),
        warmup_frac=meta.get("warmup_frac"),
        warmup_steps=(None if meta.get("warmup_frac") is not None
                      else candidate_kw.get("warmup_steps")),
        warmup_steps_new=(candidate_kw.get("warmup_steps")
                          if meta.get("warmup_frac") is not None else None),
        n_batches=int(rec["n_batches"]),
        accum_steps=int(candidate_kw.get("accum_steps", 1) or 1),
        prior_cap=int(rec["prior_cap"]), new_cap=int(rec["new_cap"]),
        resume_step=int(rec["resume_step"]),
        lr_mult=float(rec.get("lr_mult", 1.0)),
        prior_lr_mult=float(rec.get("prior_lr_mult", 1.0)))
    rec["lr_schedule"] = rep
    if rep["lr_mult"] != 1.0:
        candidate_kw["lr"] = rep["effective_peak_lr"]
        meta["continuation_lr_mult"] = rep["lr_mult"]
        meta["continuation_effective_peak_lr"] = rep["effective_peak_lr"]
    print(f"[train-two-si] continuation lr: {rep['lr_schedule']}, horizon "
          f"{rep['prior_total_steps']} -> {rep['new_total_steps']} optimizer "
          f"steps (warmup {rep['prior_warmup_steps']} -> "
          f"{rep['new_warmup_steps']}); lr at resume step "
          f"{rep['resume_step']}: {rep['lr_at_resume_prior_horizon']:.3e} "
          f"(prior horizon) -> {rep['lr_at_resume_new_horizon']:.3e} (new "
          f"horizon, {rep['lr_at_resume_fraction_of_peak']:.3f} of peak), a "
          f"jump of {rep['lr_jump_fraction_of_peak']:+.3f} of peak "
          f"{rep['peak_lr']:.3e}; lr_mult {rep['lr_mult']} -> effective "
          f"peak {rep['effective_peak_lr']:.3e} — a warm restart, recorded",
          flush=True)
    return rep


def _continuation_stopper_rule(continuation: dict, candidate_kw: dict,
                               protected: frozenset, meta: dict) -> frozenset:
    """Hand the trainer the continuation's stopper rule: under
    ``replay_exempt`` the resume epoch (0-based, from the ADOPTED
    checkpoint — a successor resuming a later checkpoint passes the same
    value and the trainer exempts nothing) goes to ``train_object_ce`` as
    ``replay_exempt_epoch``, PROTECTED: the rule is in this fit's
    fingerprint, so a trainer that lacked the kwarg must fail the arm
    loudly rather than run the legacy rule under a ``replay_exempt``
    identity. Returns the widened ``protected`` set; records the rule on
    the sidecar ``meta``."""
    from sisteer.extraction.continuation import STOPPER_RULE_REPLAY_EXEMPT

    rec = continuation["record"]
    rule = rec["stopper_rule"]
    meta["continuation_stopper_rule"] = rule
    if rule != STOPPER_RULE_REPLAY_EXEMPT:
        return protected
    exempt = int(rec["replay_exempt_epoch"])
    if exempt != int(rec["resume_epoch"]):
        raise SystemExit(
            f"[train-two-si] continuation: replay_exempt_epoch {exempt} != "
            f"resume_epoch {rec['resume_epoch']} — an internally "
            "inconsistent continuation record; refusing")
    candidate_kw["replay_exempt_epoch"] = exempt
    meta["continuation_replay_exempt_epoch"] = exempt
    print(f"[train-two-si] continuation stopper rule {rule!r}: the replayed "
          f"remainder of epoch {exempt + 1} (micro-batches "
          f"{int(rec['resume_batches_done']) + 1}..{rec['n_batches']}) is "
          "logged, not ruled on — patience counts from the first full "
          "post-restart epoch (R2-D77)", flush=True)
    return protected | {"replay_exempt_epoch"}


def _assert_continuation_resumed(continuation: dict, es_diag: dict) -> None:
    """The trainer's own word that it resumed from the adopted checkpoint
    (or a successor's later one) — a continuation that started fresh must
    never publish under a continuation identity — AND that it trained:
    ``optimizer_steps_run`` must exceed the step it resumed from (a
    continuation whose resume position leaves no optimizer group under its
    cap would otherwise publish an untouched object under a new
    fingerprint). Every legitimate continuation takes at least one
    step before its first epoch-end evaluation, so this never refuses a
    fit that early-stopped."""
    got = es_diag.get("resumed_from_step")
    want = int(continuation["summary"]["step"])
    if got is None or int(got) < want:
        raise RuntimeError(
            f"continuation did not resume from the adopted checkpoint "
            f"(trainer reports resumed_from_step={got!r}, adopted step "
            f"{want}) — a scratch fit must never publish under a "
            "continuation identity")
    # …measured against the ADOPTED step, not this call's resume step: a
    # successor that resumes its predecessor's LATER transient checkpoint
    # sitting on the final group boundary (the predecessor trained to the
    # cap and died before publishing) legitimately runs zero steps of its
    # own and only evaluates + publishes — the continuation as a whole
    # trained.
    ran = es_diag.get("optimizer_steps_run")
    if ran is None or int(ran) <= want:
        raise RuntimeError(
            f"continuation ran no optimizer step beyond the adopted "
            f"checkpoint (adopted step {want}, resumed at {got}, "
            f"optimizer_steps_run={ran!r}) — an object the continuation "
            "never updated must not publish under a continuation identity")


# One objects root = ONE (model, revision, position leg) — the F15
# namespacing rule, enforced as a store sentinel because the READERS
# (resolve_defenses_dir -> two_si_run generate) resolve paths without a
# model segment: moving the default path would orphan every store, while a
# sentinel makes mixing two identities in one root impossible at the write
# moment. First writer claims the store; every later writer must match.
_STORE_IDENTITY_NAME = "_store_identity.json"


def assert_store_identity(save_dir, cfg: dict, positions: str) -> None:
    """Claim ``save_dir`` for this (model, revision, positions) leg, or
    REFUSE loudly when the store already belongs to a different one — the
    cross-model / cross-leg clobber guard (two per-model configs share the
    relative ``defenses_dir`` default, and a Class C wave pointed at the
    Class A store would re-run every cell over the Class A artifacts now
    that resume is fingerprint-bound)."""
    save_dir = Path(save_dir)
    identity = {"model": cfg.get("model"),
                "model_revision": cfg.get("model_revision"),
                "positions": positions}
    sentinel = save_dir / _STORE_IDENTITY_NAME
    if sentinel.exists():
        try:
            stored = json.loads(sentinel.read_text())
        except ValueError as e:
            raise SystemExit(
                f"[train-two-si] {sentinel} is unparseable ({e}) — refusing "
                "to write into a store whose identity cannot be read; fix "
                "or remove the sentinel deliberately.")
        if stored != identity:
            raise SystemExit(
                f"[train-two-si] REFUSED: {save_dir} belongs to "
                f"{stored} but this fit is {identity} — one objects root "
                "holds ONE (model, revision, position leg). Point the fit "
                "at its own --out dir instead of mixing two waves' cells "
                "in one store.")
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    tmp = sentinel.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(identity, indent=1, sort_keys=True))
    tmp.replace(sentinel)


# ---------------------------------------------------------------------------- #
# Atomic cell publish + COMPLETE marker + pre-publication audit                 #
# a fit writes into a temp cell dir; a publish-time audit verifies the         #
# artifact set; the dir is renamed into place and a COMPLETE marker written    #
# LAST. Resume validates the marker + sidecars — never bare object existence   #
# (the real bug: object renamed, sidecar unwritten -> resume skips an          #
# unprovenanced object).                                                       #
# ---------------------------------------------------------------------------- #

COMPLETE_MARKER_NAME = "COMPLETE"
#: Registry-emission EVIDENCE beside COMPLETE: the publication event
#: (COMPLETE) is not the record. The
#: emitter writes ``registry_emission.json`` — {object_id, status:
#: emitted | not_emitted | failed, records | reason | error, utc} — after every
#: attempt, plus a bare ``REGISTRY_EMIT_FAILED`` marker on failure (removed by a
#: later success), so a shell (`04_fit_objects.sh`) and a resume can tell
#: "emitted, waiting for the fold" from "never emitted" from "failed". Every
#: validated-COMPLETE skip verifies it and re-emits when missing / stale /
#: failed (ensure_registry_emission) — never by retraining. The literals in
#: _registry_publish's self-contained body equal these (pinned by test).
REGISTRY_EMISSION_NAME = "registry_emission.json"
REGISTRY_EMIT_FAILED_NAME = "REGISTRY_EMIT_FAILED"


def object_format_sidecar_for(obj_path) -> str | None:
    """The FORMAT sidecar this artifact NEEDS in order to load, or None —
    decided by the loader's own CONTENT dispatch
    (``methods._is_safetensors_file``, never the arm or filename): a
    safetensors container hydrates through SteeringVector.load, which reads
    ``<artifact>.with_suffix('.json')`` == ``object.json``, so a cell
    missing that file is UNLOADABLE (bidpo cells have been published with
    the sidecar stranded at ``object.safetensors.json``). A torch blob — or
    a routing test's opaque stand-in bytes — needs none, exactly because
    ``load_object`` would never take the SteeringVector path for it."""
    from sisteer.methods import _is_safetensors_file

    obj_path = Path(obj_path)
    if obj_path.is_file() and _is_safetensors_file(obj_path):
        return str(Path(obj_path.name).with_suffix(".json"))
    return None


class CellAuditError(ValueError):
    """The pre-publication audit refused the cell — the fit is recorded
    failed and its temp dir is left in place for inspection."""


class RegistryEmissionError(RuntimeError):
    """The published object's registry ``object`` record could not be
    written. Raised AFTER the COMPLETE marker is durable, so the fit is kept
    and only the JOB fails — visibly, with REGISTRY_EMIT_FAILED beside
    COMPLETE; a re-run validates COMPLETE and replays the emission without
    retraining."""


def cell_dir_for(save_dir, cell_id: str) -> Path:
    return Path(save_dir) / cell_id


def _cell_tmp_dir(save_dir, cell_id: str) -> Path:
    """The temp cell dir a fit writes into (metrics stream here DURING the
    fit; object + sidecars land here at completion). PID-suffixed so
    concurrent arm processes never share one; an existing same-pid dir is
    a previous failed attempt of THIS process and is cleared."""
    import shutil

    p = Path(save_dir) / f"{cell_id}.tmp-{os.getpid()}"
    if p.exists():
        shutil.rmtree(p)
    p.mkdir(parents=True)
    return p


EPOCH_CKPT_DIRNAME = "epoch_ckpts"
EPOCH_CKPT_INDEX = "epoch_checkpoints.json"


def epoch_checkpoint_root(ckpt_kwargs: dict | None, arm: str):
    """The STABLE, fingerprint-bound home of a cell's epoch snapshots while the fit runs —
    ``<checkpoint_dir>/epoch_ckpts/<arm>`` under the cell's resume-checkpoint dir (per cell, OUTSIDE the
    objects store, keyed by the cell fingerprint like the step checkpoints), so a restarted fit — a NEW
    process, a new PID-suffixed temp cell dir — finds the epochs an earlier attempt saved and publishes them
    with the rest (the temp dir alone loses epoch 1/2 on any restart). None when the checkpoint seam is off
    (the caller then falls back to the temp cell dir)."""
    d = (ckpt_kwargs or {}).get("checkpoint_dir")
    key = (ckpt_kwargs or {}).get("checkpoint_key")
    if not d or not key:
        return None
    # the FINGERPRINT is in the path (second round): an interrupted fit's root under key K1 and a retried
    # cell under a changed config K2 never share a directory, so neither can overwrite the other's artefacts
    return Path(d) / EPOCH_CKPT_DIRNAME / arm / hashlib.sha256(str(key).encode()).hexdigest()[:16]


def save_epoch_checkpoint(root, obj, *, arm: str, si_id: str, epoch: int, step: int, history: list, heldout_curve: list,
                          stopped: bool, epochs_cap: int, checkpoint_key: str | None = None) -> dict:
    """Write ``<root>/epoch_NN/<object artefact>`` + ``provenance.json`` for one arm at the end of epoch
    ``epoch`` (1-based) and update ``<root>/epoch_checkpoints.json`` — the cell-level index (``checkpoint_key`` =
    the cell fingerprint, so a resume can refuse another identity's snapshots). Named roles: ``epoch_1``,
    ``epoch_2``, ``final`` (the last epoch written; re-pointed every epoch so an early stop leaves it on the last
    epoch run); every other epoch is kept too (``intermediate``) — nothing here is ever rotated or deleted.
    ``root`` is the stable per-cell path from :func:`epoch_checkpoint_root`, or the temp cell dir when the
    checkpoint seam is off. Returns the index entry."""
    from sisteer.methods import object_artifact_name
    root = Path(root)
    cell_id = f"{si_id}__{arm}"
    idx_path = root / EPOCH_CKPT_INDEX
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"version": 2, "cell_id": cell_id, "checkpoints": [], "named": {}}
    # identity FIRST, bytes second (second round): a root that belongs to another fingerprint or another cell is
    # refused before any artefact of it is touched — nothing of the other identity's is ever overwritten
    if idx.get("checkpoint_key") not in (None, checkpoint_key):
        raise RuntimeError(f"{idx_path}: index carries checkpoint_key {str(idx.get('checkpoint_key'))[:16]}…, this fit's is "
                           f"{str(checkpoint_key)[:16]}… — another identity's snapshots; refusing to touch them")
    if idx.get("cell_id") not in (None, cell_id):
        raise RuntimeError(f"{idx_path}: index names cell {idx.get('cell_id')!r}, this cell is {cell_id!r} — refusing to touch it")
    d = root / f"epoch_{int(epoch):02d}"
    d.mkdir(parents=True, exist_ok=True)
    art = d / object_artifact_name(arm)
    tmp = art.with_name(art.name + ".tmp")
    obj.save(tmp)
    tmp_side = tmp.with_suffix(".json")
    if tmp_side.exists():
        tmp_side.replace(art.with_suffix(".json"))
    tmp.replace(art)
    sha = _sha256_file(art)
    last_loss = None
    for h in reversed(history):
        v = h.get("loss") if isinstance(h, dict) else h
        if isinstance(v, (int, float)):
            last_loss = float(v); break
    entry = {
        "epoch": int(epoch), "step": int(step), "artifact": f"{EPOCH_CKPT_DIRNAME}/epoch_{int(epoch):02d}/{art.name}",
        "artifact_sha256": sha, "artifact_bytes": art.stat().st_size, "weights_sha256": weights_content_sha256(art), "train_loss_last": last_loss,
        "heldout_loss": (float(heldout_curve[-1]) if heldout_curve else None), "heldout_curve": [float(x) for x in heldout_curve],
        "stopped": bool(stopped), "epochs_cap": int(epochs_cap), "cell_id": cell_id, "si_id": si_id, "arm": arm,
        "epoch_id": f"{cell_id}@epoch_{int(epoch):02d}", "checkpoint_key": checkpoint_key,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "hostname": socket.gethostname(), "code_sha": _git_sha_or_none(), "written_utc": _utc_now_z(),
    }
    (d / "provenance.json").write_text(json.dumps(entry, indent=1, sort_keys=True) + "\n")
    idx["checkpoint_key"] = checkpoint_key
    idx["checkpoints"] = [c for c in idx["checkpoints"] if int(c["epoch"]) != int(epoch)] + [entry]
    idx["checkpoints"].sort(key=lambda c: int(c["epoch"]))
    named = {"final": f"epoch_{int(epoch):02d}"}
    for c in idx["checkpoints"]:
        if int(c["epoch"]) in (1, 2):
            named[f"epoch_{int(c['epoch'])}"] = f"epoch_{int(c['epoch']):02d}"
    idx["named"] = named
    idx["ruling"] = "D546 (2026-10-01): epoch_1, epoch_2 and final are durable, named, registered; never rotated"
    tmp_idx = idx_path.with_suffix(".json.tmp")
    tmp_idx.write_text(json.dumps(idx, indent=1, sort_keys=True) + "\n")
    tmp_idx.replace(idx_path)
    return entry


def finalize_epoch_checkpoints(root, tmp_cell_dir, *, checkpoint_key: str | None, cell_id: str, required: bool = False,
                               expected_final_epoch: int | None = None) -> dict | None:
    """At publish time: bring the cell's epoch snapshots from their stable root into the TEMP CELL DIR
    (so ``publish_cell`` renames them into the published cell atomically), after validating the index's
    ``checkpoint_key`` against this fit's fingerprint (a mismatch is another identity's snapshots: refused by
    name, nothing merged), and give every epoch dir a VALID identity sidecar — the cell's own
    ``train_budget_meta.json`` re-keyed to the epoch artefact's sha with an ``epoch_checkpoint`` block, plus the
    cell's ``fit_selection.json`` — so the production consumers' identity checks (``staged_cell_problems``:
    cell_id + object_sha256 vs bytes) accept each epoch object and the registry hook can emit a per-epoch
    ``object`` row. Returns the merged index, or None when there is nothing."""
    import shutil
    tmp_cell_dir = Path(tmp_cell_dir)
    if root is None or Path(root) == tmp_cell_dir:
        root = tmp_cell_dir
    root = Path(root)
    idx_path = root / EPOCH_CKPT_INDEX
    if not idx_path.exists():
        if required:
            # A fit whose epoch hook WAS wired but saved nothing (every save failed and was logged) must not
            # publish as an epoch-checkpoint fit — the rule's artefacts are part of the deliverable
            raise RuntimeError(f"{cell_id}: the epoch-checkpoint hook was wired but no index exists at {idx_path} — every "
                               f"epoch save failed (see the [epoch-ckpt] lines above); refusing to publish without the "
                               f"D546 artefacts")
        return None
    idx = json.loads(idx_path.read_text())
    if checkpoint_key is not None and idx.get("checkpoint_key") not in (None, checkpoint_key):
        raise RuntimeError(f"{idx_path}: checkpoint_key {str(idx.get('checkpoint_key'))[:16]}… is not this fit's "
                           f"{str(checkpoint_key)[:16]}… — refusing to publish another identity's epoch snapshots")
    if idx.get("cell_id") not in (None, cell_id):
        raise RuntimeError(f"{idx_path}: index names cell {idx.get('cell_id')!r}, this cell is {cell_id!r} — refusing")
    cell_meta_path = tmp_cell_dir / "train_budget_meta.json"
    cell_meta = json.loads(cell_meta_path.read_text()) if cell_meta_path.exists() else None
    fit_sel_path = tmp_cell_dir / FIT_SELECTION_NAME if "FIT_SELECTION_NAME" in globals() else tmp_cell_dir / "fit_selection.json"
    dest_root = tmp_cell_dir / EPOCH_CKPT_DIRNAME
    kept = []
    for c in idx.get("checkpoints") or []:
        src_dir = root / f"epoch_{int(c['epoch']):02d}"
        src_art = root / str(c["artifact"]).split("/", 1)[1] if str(c["artifact"]).startswith(EPOCH_CKPT_DIRNAME + "/") else root / c["artifact"]
        if not src_art.exists():
            print(f"[epoch-ckpt] {cell_id}: epoch {c['epoch']} artefact missing at {src_art} — not published", file=sys.stderr)
            continue
        if _sha256_file(src_art) != c["artifact_sha256"]:
            print(f"[epoch-ckpt] {cell_id}: epoch {c['epoch']} artefact does not hash to its index entry — not published", file=sys.stderr)
            continue
        dst_dir = dest_root / src_dir.name
        if dst_dir.resolve() != src_dir.resolve():
            dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src_dir.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst_dir / f.name)
        if cell_meta is not None:
            side = dict(cell_meta)
            side["object_sha256"] = c["artifact_sha256"]
            side["epoch_checkpoint"] = {k: c.get(k) for k in ("epoch", "step", "epoch_id", "heldout_loss", "train_loss_last", "stopped",
                                                               "epochs_cap", "slurm_job_id", "written_utc", "code_sha")}
            side["epoch_checkpoint"]["parent_object_sha256"] = cell_meta.get("object_sha256")
            side["epoch_checkpoint"]["named_roles"] = sorted(k for k, v in (idx.get("named") or {}).items() if v == src_dir.name)
            t = dst_dir / "train_budget_meta.json.tmp"
            t.write_text(json.dumps(side, indent=1, sort_keys=True, default=str) + "\n"); t.replace(dst_dir / "train_budget_meta.json")
            c = {**c, "identity_sidecar": "train_budget_meta.json", "parent_object_sha256": cell_meta.get("object_sha256")}
        if fit_sel_path.exists():
            shutil.copy2(fit_sel_path, dst_dir / fit_sel_path.name)
        kept.append(c)
    # the three NAMED artefacts must be present and consistent, or the cell does not publish: epoch_1 always;
    # epoch_2 once the fit ran a second epoch; final always (it names the last epoch run)
    kept_dirs = {f"epoch_{int(c['epoch']):02d}" for c in kept}
    max_epoch = max((int(c["epoch"]) for c in idx.get("checkpoints") or []), default=0)
    # LINEAGE-aware: epoch_1 / epoch_2 are required whenever the index reaches them, EXCEPT the epochs a
    # continuation's predecessor trained before epoch checkpoints existed — admitted only through the explicit
    # lineage block `import_predecessor_epoch_checkpoints` writes from the verified continuation marker (predecessor
    # cell + its epoch count + no epoch index there). Never inferred from which save happened to succeed first: a
    # fresh fit whose epoch-1 save failed transiently has no lineage block, so epoch_1 stays required and the cell
    # refuses to publish without it.
    lineage = dict(idx.get("lineage") or {})
    exempt: set = set()
    if lineage.get("continued_from") and lineage.get("predecessor_has_epoch_index") is False:
        exempt = {n for n in (1, 2) if n <= int(lineage.get("predecessor_epochs_done") or 0)}
    reached = {n for n in (1, 2) if n <= max_epoch and n not in exempt}
    required = {"final"} | {f"epoch_{n}" for n in reached}
    not_reached = sorted(n for n in exempt if n <= max_epoch)
    named = dict(idx.get("named") or {})
    missing = [r for r in sorted(required) if named.get(r) not in kept_dirs]
    if missing:
        raise RuntimeError(f"{cell_id}: D546 named epoch checkpoint(s) {missing} missing or inconsistent in {root} (named={named}, "
                           f"kept={sorted(kept_dirs)}) — the cell does not publish without its named artefacts")
    if expected_final_epoch is not None:
        # `final` is bound to the last epoch the TRAINER ran (the driver's callback-invocation count, recorded
        # before any save is attempted), never to the last save that happened to succeed — otherwise a failed
        # last-epoch save leaves `final` on an earlier epoch and the publication certifies the wrong object.
        want = f"epoch_{int(expected_final_epoch):02d}"
        if named.get("final") != want or want not in kept_dirs or max_epoch != int(expected_final_epoch):
            raise RuntimeError(f"{cell_id}: the trainer ran {int(expected_final_epoch)} epoch(s) but the index's final checkpoint is "
                               f"{named.get('final')!r} (kept={sorted(kept_dirs)}, max_epoch={max_epoch}) — the last epoch's save failed "
                               f"or was lost; refusing to certify an earlier epoch as `final`")
    idx = {**idx, "checkpoints": kept, "published_from": str(root), "finalized_utc": _utc_now_z(),
           "not_reached_in_lineage": not_reached}
    out = tmp_cell_dir / EPOCH_CKPT_INDEX
    t = out.with_suffix(".json.tmp"); t.write_text(json.dumps(idx, indent=1, sort_keys=True) + "\n"); t.replace(out)
    return idx


def _git_sha_or_none() -> str | None:
    try:
        import subprocess
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10,
                              cwd=str(Path(__file__).resolve().parent)).stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _utc_now_z() -> str:
    import time as _t
    return _t.strftime("%Y-%m-%dT%H:%M:%SZ", _t.gmtime())




def import_predecessor_epoch_checkpoints(continuation: dict, root, *, checkpoint_key: str, cell_id: str) -> dict | None:
    """For a CONTINUATION: the continued generation adopts its predecessor's step checkpoint under a
    new fingerprint; here it also inherits the predecessor's PUBLISHED epoch checkpoints (epoch_ckpts/* + index in the
    predecessor's cell dir — the cell `--continue-from` names) into its own stable root, each entry carrying
    `imported_from` provenance (predecessor cell dir, fit fingerprint, epoch id, key) and this fit's checkpoint_key, so
    the continued publication keeps epoch_1/epoch_2 and names `final` at the last epoch run. The index ALWAYS gets an
    explicit `lineage` block from the verified continuation marker (predecessor cell, fingerprint, epochs it completed,
    whether it has an epoch index): that block — never the order in which saves happened to succeed — is the ONLY way
    `finalize_epoch_checkpoints` admits a missing early epoch (a predecessor fitted before epoch checkpointing has
    no index; the epochs it trained are recorded as not_reached_in_lineage). Returns the index (None only without a
    root/checkpoint)."""
    import shutil
    if root is None:
        return None
    root = Path(root)
    ck = (continuation.get("record") or {}).get("checkpoint") or continuation.get("checkpoint")
    if not ck:
        return None
    pred_dir = Path(ck).parent.parent
    pred_idx_path = pred_dir / EPOCH_CKPT_INDEX
    pred_idx = json.loads(pred_idx_path.read_text()) if pred_idx_path.exists() else None
    marker = dict(continuation.get("marker") or {})
    pred_fp = marker.get("fit_fingerprint")
    pred_done = marker.get("epoch")
    root.mkdir(parents=True, exist_ok=True)
    idx_path = root / EPOCH_CKPT_INDEX
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {"version": 2, "cell_id": cell_id, "checkpoints": [], "named": {}}
    if idx.get("checkpoint_key") not in (None, checkpoint_key):
        raise RuntimeError(f"{idx_path}: index carries another identity's checkpoint_key — refusing to import into it")
    if idx.get("cell_id") not in (None, cell_id):
        raise RuntimeError(f"{idx_path}: index names cell {idx.get('cell_id')!r}, this cell is {cell_id!r} — refusing to import into it")
    idx["checkpoint_key"] = checkpoint_key
    have = {int(c["epoch"]) for c in idx["checkpoints"]}
    if pred_idx is None:
        print(f"[epoch-ckpt] {cell_id}: predecessor {pred_dir} has no epoch-checkpoint index (pre-D546 cell; it completed "
              f"{pred_done} epoch(s)) — nothing to inherit; those epochs are recorded as not reached in this lineage", file=sys.stderr)
    else:
        for c in pred_idx.get("checkpoints") or []:
            ep = int(c["epoch"])
            if ep in have:
                continue
            src = pred_dir / str(c["artifact"])
            if not src.exists() or _sha256_file(src) != c.get("artifact_sha256"):
                print(f"[epoch-ckpt] {cell_id}: predecessor epoch {ep} artefact missing or altered at {src} — not inherited", file=sys.stderr)
                continue
            dst_dir = root / f"epoch_{ep:02d}"; dst_dir.mkdir(parents=True, exist_ok=True)
            for f in src.parent.iterdir():
                if f.is_file() and f.name not in ("train_budget_meta.json", "fit_selection.json"):   # identity sidecars are re-written for THIS cell at finalize
                    shutil.copy2(f, dst_dir / f.name)
            entry = {**c, "checkpoint_key": checkpoint_key, "cell_id": cell_id,
                     "imported_from": {"cell_dir": str(pred_dir), "fit_fingerprint": pred_fp, "epoch_id": c.get("epoch_id"),
                                       "checkpoint_key": pred_idx.get("checkpoint_key"), "imported_utc": _utc_now_z()}}
            idx["checkpoints"].append(entry)
    idx["checkpoints"].sort(key=lambda c: int(c["epoch"]))
    named = {}
    if idx["checkpoints"]:
        named["final"] = f"epoch_{int(idx['checkpoints'][-1]['epoch']):02d}"
    for c in idx["checkpoints"]:
        if int(c["epoch"]) in (1, 2):
            named[f"epoch_{int(c['epoch'])}"] = f"epoch_{int(c['epoch']):02d}"
    idx["named"] = named
    idx["lineage"] = {"continued_from": str(pred_dir), "predecessor_fit_fingerprint": pred_fp,
                      "predecessor_has_epoch_index": pred_idx is not None,
                      "predecessor_epochs_done": (int(pred_done) if pred_done is not None else None),
                      "inherited_epochs": sorted(int(c["epoch"]) for c in idx["checkpoints"] if c.get("imported_from"))}
    t = idx_path.with_suffix(".json.tmp"); t.write_text(json.dumps(idx, indent=1, sort_keys=True) + "\n"); t.replace(idx_path)
    print(f"[epoch-ckpt] {cell_id}: lineage {idx['lineage']}", file=sys.stderr)
    return idx



def weights_content_sha256(path) -> str | None:
    """hook-fix-5: a CANONICAL hash of an object's WEIGHTS — sha256 over the sorted (key, dtype, shape, contiguous
    little-endian bytes) of every tensor reachable in the saved blob (dicts by sorted key, lists by index) — so two files
    that serialize identical weights to different bytes (the trainer's final `_save_object` vs the epoch saver's
    `obj.save` after the best-epoch restore: bitwise-equal tensors, different file sha) hash the SAME. Non-tensor fields
    are excluded on purpose (they are metadata, recorded elsewhere). None (with a notice) when the file is not a torch
    blob or torch is unavailable — a stated absence, never a guessed identity."""
    try:
        import torch
    except ImportError:
        print(f"[weights-sha] torch unavailable — no weight-content hash for {path}", file=sys.stderr)
        return None
    try:
        try:
            blob = torch.load(str(path), map_location="cpu", weights_only=True)
        except Exception:  # noqa: BLE001 — a blob with non-primitive objects
            blob = torch.load(str(path), map_location="cpu", weights_only=False)
    except Exception as e:  # noqa: BLE001
        print(f"[weights-sha] {path}: not a readable torch blob ({type(e).__name__}) — no weight-content hash", file=sys.stderr)
        return None
    items = []

    def walk(o, pre):
        if torch.is_tensor(o):
            items.append((pre, o))
        elif isinstance(o, dict):
            for k in sorted(o, key=str):
                walk(o[k], f"{pre}.{k}" if pre else str(k))
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                walk(v, f"{pre}[{i}]")
    walk(blob, "")
    if not items:
        return None
    h = hashlib.sha256()
    for key, t in items:
        t = t.detach().contiguous().cpu()
        try:
            raw = t.numpy().tobytes()
        except (TypeError, RuntimeError):            # bf16 and friends: reinterpret at the element width
            width = t.element_size()
            raw = t.view({1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[width]).numpy().tobytes()
        h.update(f"{key}|{str(t.dtype)}|{tuple(t.shape)}|".encode()); h.update(raw)
    return h.hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _step_loss_key_hit(keys, arm: str) -> bool:
    """Whether a metrics record carries per-step training-loss evidence
    for ``arm`` — the monitor's contract key, the lockstep engine's native
    key, or the solo trainer's own prefix (psr's mse loop logs
    ``psr/mse_loss``; bidpo logs ``bidpo/...``)."""
    prefixes = (f"lockstep/{arm}/", f"{arm}/")
    if arm.startswith("psr"):
        prefixes = prefixes + ("psr/",)
    for k in keys:
        if k == "train/loss" or k.startswith(prefixes):
            return True
    return False


def _arm_logs_step_loss(arm: str, *, lockstep: bool) -> bool:
    """Whether this cell's metrics are EXPECTED to carry per-step
    training-loss records: every lockstep-monitored arm (the monitor
    emits ``train/loss``), plus the solo gradient trainers whose loops log
    a step loss (psr's mse loop, the DPO arms). Closed-form / capture
    arms (diffmean, alphakl_*) and trainers with unprefixed keys (loreft)
    train in one shot or log nothing recognisable — the audit must not
    refuse their cells for it."""
    return lockstep or arm.startswith("psr") or arm in DPO_PREF_ARMS


def audit_cell(tmp_dir, *, arm: str, cadence=None,
               monitors_on: bool = True,
               require_step_loss: bool = True) -> dict:
    """The pre-publication audit (P1-8). Verifies, on the TEMP cell dir:

    - the object artifact + ``train_budget_meta.json`` sidecar exist, and
      the sidecar's fit_record carries the floor verdict (the bar it was
      judged against always; the ratio whenever a bar was set);
    - ``metrics.jsonl`` exists, every line parses, every value is finite
      (the last line of defence under the buffered-flush abort);
    - per-step training-loss records are present, the FINAL dual-KL values
      are present, and — where the per-step monitor ran (``train/loss``
      records exist) — a probe record exists at every cadence-due step
      among the observed loss steps.

    ``monitors_on=False`` skips the metric-content requirements (the
    escape-hatch configuration still publishes objects + sidecars).
    Raises :class:`CellAuditError` naming every problem; returns a summary
    dict (embedded in the COMPLETE marker) on success."""
    from sisteer.methods import object_artifact_name

    tmp_dir = Path(tmp_dir)
    problems: list[str] = []
    obj_path = tmp_dir / object_artifact_name(arm)
    if not obj_path.exists():
        problems.append(f"object artifact missing ({obj_path.name})")
    fmt_sidecar = object_format_sidecar_for(obj_path)
    if fmt_sidecar is not None and not (tmp_dir / fmt_sidecar).exists():
        # Review F14: a safetensors artifact without its container sidecar
        # LOADS NOWHERE (SteeringVector.load reads object.json), so a cell
        # missing it must never publish COMPLETE.
        problems.append(
            f"format sidecar missing ({fmt_sidecar}) — a safetensors "
            "artifact is unloadable without it")
    sidecar = tmp_dir / "train_budget_meta.json"
    fit_record = None
    if not sidecar.exists():
        problems.append("train_budget_meta.json sidecar missing")
    else:
        try:
            meta = json.loads(sidecar.read_text())
            if "positions_at_fit" not in meta:
                problems.append("sidecar lacks positions_at_fit")
            fit_record = meta.get("fit_record")
            if not isinstance(fit_record, dict):
                problems.append("sidecar lacks a fit_record")
            else:
                if "floor_ratio_max" not in fit_record:
                    problems.append("fit_record lacks the floor verdict "
                                    "(floor_ratio_max)")
                elif fit_record["floor_ratio_max"] is not None:
                    fr = fit_record.get("floor_ratio")
                    if not (isinstance(fr, (int, float))
                            and not isinstance(fr, bool)
                            and math.isfinite(fr)):
                        problems.append(
                            f"floor bar set but floor_ratio is {fr!r}")
        except ValueError as e:
            problems.append(f"sidecar unparseable: {e}")
    metrics = tmp_dir / "metrics.jsonl"
    rows: list[dict] = []
    if not metrics.exists():
        problems.append("metrics.jsonl missing (the per-cell mirror is "
                        "the contract)")
    else:
        for i, line in enumerate(metrics.read_text().splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                problems.append(f"metrics.jsonl line {i} unparseable")
    if monitors_on and metrics.exists():
        for r in rows:
            for k, v in r.items():
                if (isinstance(v, (int, float)) and not isinstance(v, bool)
                        and not math.isfinite(v)):
                    problems.append(
                        f"non-finite metric {k}={v} at step "
                        f"{r.get('step')} — a diverged fit must never "
                        "publish")
                    break
        if require_step_loss and not any(
                _step_loss_key_hit(r.keys(), arm) for r in rows):
            problems.append("no per-step training-loss records in "
                            "metrics.jsonl")
        if not any("final/residual_ratio" in r for r in rows):
            problems.append("final dual-KL values missing "
                            "(final/residual_ratio)")
        if not any("final/displacement_ratio" in r for r in rows):
            problems.append("final dual-KL values missing "
                            "(final/displacement_ratio)")
        if cadence is not None:
            loss_steps = sorted({r["step"] for r in rows
                                 if "train/loss" in r
                                 and isinstance(r.get("step"), int)})
            probe_steps = {r["step"] for r in rows
                           if "probe/residual_ratio" in r}
            missing = [s for s in loss_steps
                       if cadence.due(s) and s not in probe_steps]
            if missing:
                problems.append(
                    f"probe cadence gap: due step(s) {missing[:8]} carry "
                    "no probe/residual_ratio record")
    if problems:
        raise CellAuditError(
            f"pre-publication audit refused {tmp_dir.name}: "
            + "; ".join(problems))
    return {"metrics_rows": len(rows),
            "object_file": obj_path.name,
            "monitors_on": monitors_on}


#: The fit SELECTION sidecar: the generated fit config's record of WHICH
#: search / ruling picked this cell's lr and LoRA architecture, serialised
#: by the driver into the TEMP cell dir so publish_cell's rename carries it
#: into the published dir, the COMPLETE manifest hashes it (publication
#: evidence, verified on resume) and _registry_publish reads it from the
#: PUBLISHED dir. THE spelling is the hook's
#: (sisteer.registry.hooks.FIT_SELECTION_SIDECAR — the trainer writes what
#: the hook reads); the literal only where the registry package is absent.
try:
    from sisteer.registry.hooks import FIT_SELECTION_SIDECAR as FIT_SELECTION_NAME
except ImportError:
    FIT_SELECTION_NAME = "fit_selection.json"
#: the identity stamps write_fit_selection adds — a config block may not
#: carry them itself (the stamp is the driver's, never the config's claim)
_FIT_SELECTION_STAMPS = ("arm", "si_id", "config_sha", "fit_width")
#: train_kwargs.<arm> keys the DRIVER consumes — never a trainer kwarg, never a protected key: the per-arm floor bars
#: (arm_floor_ratio_max / arm_ce_floor_ratio_max), gradient checkpointing (applied at model load), and the fit SELECTION block
#: (serialised to the sidecar at publish)
DRIVER_ONLY_ARM_KEYS = ("floor_ratio_max", "ce_floor_ratio_max", "gradient_checkpointing", "fit_selection")
#: the trainer-facing ARCHITECTURE keys a fit_selection may claim; the publish holds each against the saved artefact
ARCH_SELECTION_KEYS = ("lora_rank", "lora_modules", "per_expert_rank", "init_scheme")


def arm_trainer_kwargs(per_arm_tk: dict, arm: str) -> dict:
    """The per-arm config kwargs that REACH the trainer (cli.py's train_kwargs.<arm> convention; the caller protects them so a typo'd
    key fails loudly) minus the driver's own knobs (DRIVER_ONLY_ARM_KEYS). ONE construction, shared by the production fit and the
    fit-path test (stamp-fix-2): the architecture keys make_sdfbase_configs promotes top-level — lora_modules,
    per_expert_rank, init_scheme — pass through here exactly as lr / warmup / clip / lora_rank do. Before this the picked architecture
    lived only in the popped `fit_selection`, so the trainer built its defaults under a sidecar claiming the pick."""
    kw = dict((per_arm_tk or {}).get(arm) or {})
    for k in DRIVER_ONLY_ARM_KEYS:
        kw.pop(k, None)
    return kw


def artefact_architecture(blob: dict) -> dict:
    """The architecture a SAVED object embodies, in fit_selection's vocabulary, read from the blob the publish ships: a `lora_all` blob
    carries `rank` (+ lora_modules / per_expert_rank / init_scheme in the general shape; the legacy o_proj shape implies ["o_proj"] and no
    per-expert map), a `psr` blob states init_scheme alone. BOTH writers record init_scheme ONLY when it is not the zero default
    (MultiLayerLoRAObject.to_blob, PSRObject.to_blob), so an absent key IS "zero" — the fact that exposed the published control_relu cell
    (sidecar random_lecun_1e-3, blob without the key). Keys the blob cannot state are absent, never guessed."""
    blob = blob if isinstance(blob, dict) else {}
    out: dict = {"init_scheme": str(blob.get("init_scheme") or "zero")}
    if str(blob.get("kind") or "") == "lora_all":
        out["lora_rank"] = int(blob.get("rank") or 1)
        out["lora_modules"] = sorted(blob.get("lora_modules") or ["o_proj"])
        per = blob.get("per_expert_rank")
        out["per_expert_rank"] = dict(per) if isinstance(per, dict) else per
    return out


def write_fit_selection(tmp_dir, cfg: dict, si_id: str, arm: str, *, artefact_path=None) -> Path | None:
    """Serialise ``train_kwargs.<arm>.fit_selection`` (keys such as
    ``fit_search_id``, ``fit_lr_selected_by``, ``lora_rank``, ``lora_modules``,
    ``per_expert_rank``, ``init_scheme``, ``architecture_selection_ruling_id``
    — written by make_sdfbase_configs from hpo_pick's output) into the TEMP
    cell dir as :data:`FIT_SELECTION_NAME`, stamped with the cell identity
    (``arm``, ``si_id``, ``config_sha``) and the data-parallel ``fit_width``
    this process fitted at (:func:`fit_width_active`; 1 = single-process;
    the registry hook carries it on the object's ``extra``). Returns the
    path, or None when the config carries no block — a pre-selection
    config publishes byte-identical cells. A block that is not a mapping,
    or one that carries a stamp key itself, REFUSES (CellAuditError):
    configuration that reads as set and behaves as unset is the refused
    bug class."""
    block = ((cfg or {}).get("train_kwargs") or {}).get(arm) or {}
    sel = block.get("fit_selection") if isinstance(block, dict) else None
    if sel is None:
        return None
    if not isinstance(sel, dict):
        raise CellAuditError(
            f"train_kwargs.{arm}.fit_selection must be a mapping "
            f"({{fit_search_id, fit_lr_selected_by, lora_rank, ...}}), got "
            f"{type(sel).__name__} — refusing to publish an unrecorded "
            "selection")
    clash = sorted(set(sel) & set(_FIT_SELECTION_STAMPS))
    if clash:
        raise CellAuditError(
            f"train_kwargs.{arm}.fit_selection may not carry the driver's "
            f"identity stamps {clash} — they are written from the cell "
            "being published, never from the config")
    claimed = {k: sel[k] for k in ARCH_SELECTION_KEYS if k in sel}
    if claimed:
        # stamp-fix-2: the sidecar may publish ONLY an architecture the SAVED object embodies. The selection
        # used to live only in this block while the trainer built its defaults, and the publish recorded the false architecture
        # (control_relu: blob without init_scheme = zero init, sidecar random_lecun_1e-3). Read the artefact that ships — `object.pt`
        # in the temp cell dir (`artefact_path` names another) — and refuse the publish by name on any disagreement; no artefact to
        # read against is itself a refusal, never a pass.
        path = Path(artefact_path) if artefact_path is not None else Path(tmp_dir) / "object.pt"
        if not path.is_file():
            raise CellAuditError(
                f"train_kwargs.{arm}.fit_selection for {si_id}__{arm} claims an architecture {claimed} but no saved object is at "
                f"{path} to hold it against — refusing to publish an unverified selection (D875)")
        import torch
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as exc:  # noqa: BLE001 — an opaque / torn artefact cannot certify anything
            raise CellAuditError(
                f"train_kwargs.{arm}.fit_selection for {si_id}__{arm} claims an architecture {claimed} but the saved object {path.name} "
                f"cannot be read as a torch artefact ({type(exc).__name__}: {str(exc)[:80]}) — the selection cannot be held against it; "
                "refusing to publish an unverified selection (D875)") from exc
        arch = artefact_architecture(blob)
        bad = []
        if any(k in claimed for k in ("lora_rank", "lora_modules", "per_expert_rank")) and "lora_rank" not in arch:
            bad.append(f"the artefact is not a LoRA object (kind {(blob or {}).get('kind') if isinstance(blob, dict) else type(blob).__name__!r}) "
                       f"yet the selection claims a LoRA architecture")
        for k, want in claimed.items():
            if k not in arch:
                continue
            if k == "lora_modules":
                want = sorted(want or [])
            if arch[k] != want:
                bad.append(f"{k}: selection {want!r} vs artefact {arch[k]!r}")
        if bad:
            raise CellAuditError(
                f"fit_selection for {si_id}__{arm} does not match the saved object {path.name} — " + "; ".join(bad) +
                " — refusing to publish a selection the object does not embody (D875: the architecture must reach the trainer as "
                "top-level train_kwargs.<arm> keys, see make_sdfbase_configs._arm_block)")
    record = {**sel, "arm": arm, "si_id": si_id,
              "config_sha": _cfg_dict_sha(cfg or {}),
              "fit_width": fit_width_active()}
    p = Path(tmp_dir) / FIT_SELECTION_NAME
    p.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return p


def _required_publication_files(cell_dir, object_file: str) -> list[str]:
    """Every file a published cell MUST carry, by name: the object, its
    format sidecar (when the artifact's CONTENT says it is a safetensors
    container — the loader's own dispatch), the provenance sidecar, the
    per-cell metrics mirror, and — when the cell carries one — the fit
    SELECTION sidecar (:data:`FIT_SELECTION_NAME`; a cell published from a
    pre-selection config has none and is not asked for one). ONE derivation,
    shared by the COMPLETE manifest writer and the resume validator (F16):
    a selection that sits in a published dir without its manifest entry is
    not publication evidence and re-runs the cell."""
    names = [object_file, "train_budget_meta.json", "metrics.jsonl"]
    fmt = object_format_sidecar_for(Path(cell_dir) / object_file)
    if fmt is not None:
        names.insert(1, fmt)
    if (Path(cell_dir) / FIT_SELECTION_NAME).is_file():
        names.append(FIT_SELECTION_NAME)
    # The epoch-boundary checkpoints are PUBLICATION FILES — the finalized index, every artefact it lists and that
    # epoch's identity + provenance sidecars ride the COMPLETE manifest, so a lost or altered epoch artefact makes
    # the cell INCOMPLETE to validated_complete (resume re-runs / re-publishes) instead of failing only when the
    # battery serves it. Relative paths; the writer hashes and the validator checks them like any other.
    idx_path = Path(cell_dir) / EPOCH_CKPT_INDEX
    if idx_path.is_file():
        names.append(EPOCH_CKPT_INDEX)
        try:
            idx = json.loads(idx_path.read_text())
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"{idx_path}: unreadable epoch-checkpoint index ({type(e).__name__}) — not publishable") from e
        for c in idx.get("checkpoints") or []:
            rel = str(c["artifact"])
            names.append(rel)
            ep_dir = str(Path(rel).parent)
            for side in ("train_budget_meta.json", "provenance.json"):
                if (Path(cell_dir) / ep_dir / side).is_file():
                    names.append(f"{ep_dir}/{side}")
    return names


def publish_cell(save_dir, cell_id: str, tmp_dir, *, si_id: str, arm: str,
                 positions: str, audit: dict, fingerprint: str,
                 marker_extra: dict | None = None) -> Path:
    """Rename the audited temp dir into place, then write the COMPLETE
    marker LAST — the marker's existence is the publication event, so a
    kill anywhere before it leaves a cell that resume re-runs (T14).

    The marker is a MANIFEST (F16): every required publication file with
    its sha256 — object + format sidecar + train_budget_meta.json +
    metrics.jsonl — plus the fit-input ``fit_fingerprint`` (F15), so
    resume can prove both WHAT was fitted and that nothing was lost or
    swapped after publication."""
    import shutil
    from sisteer.methods import object_artifact_name

    tmp_dir = Path(tmp_dir)
    final_dir = cell_dir_for(save_dir, cell_id)
    if final_dir.exists():
        # A dir without a validated COMPLETE marker is a torn earlier
        # publish (resume would have skipped a validated one before the
        # fit began), or --overwrite: either way it is superseded.
        shutil.rmtree(final_dir)
    tmp_dir.rename(final_dir)
    obj_path = final_dir / object_artifact_name(arm)
    files = {}
    for name in _required_publication_files(final_dir, obj_path.name):
        p = final_dir / name
        if not p.is_file():
            # Belt on top of audit_cell (which ran on the temp dir): a
            # manifest entry that cannot be hashed must never be promised.
            raise CellAuditError(
                f"publish of {cell_id} lost required file {name} between "
                "audit and marker — refusing to write COMPLETE")
        files[name] = _sha256_file(p)
    marker = {
        "cell_id": cell_id, "si_id": si_id, "arm": arm,
        "positions": positions,
        "object_file": obj_path.name,
        "object_sha256": files[obj_path.name],
        "files": files,
        "fit_fingerprint": fingerprint,
        "published_at": datetime.now(timezone.utc).isoformat(
            timespec="seconds"),
        "audit": audit,
    }
    if marker_extra:
        # Additive only (the continued_from block): the marker's own
        # identity fields are never shadowed by a caller.
        clash = sorted(set(marker_extra) & set(marker))
        if clash:
            raise ValueError(f"marker_extra may not override marker fields "
                             f"{clash}")
        marker.update(marker_extra)
    (final_dir / COMPLETE_MARKER_NAME).write_text(
        json.dumps(marker, indent=1, sort_keys=True) + "\n")
    _registry_publish(final_dir, si_id=si_id, arm=arm)
    # The emitter's outcome is read back from its evidence; a FAILED
    # emission is a visible error of this job (the publication above
    # stands; nothing retrains to repair a record).
    _raise_if_emission_failed(final_dir)
    return final_dir


def _registry_publish(final_dir: Path, *, si_id: str, arm: str, emitter=None, fit_provenance=None, dry_run: bool = False):
    """Results-registry native emission (additive, never raises): the published
    object's ``object`` + ``artifact`` records from its sidecars, AFTER the COMPLETE marker (the
    publication event). The writer states what it fitted: the two-SI arms distil a system
    instruction into the object by KL (a ``document_tokens`` sidecar names a corpus fit by CE
    instead — its role comes from the pool's prefix, hooks.object_role_for_corpus); an arm or SI /
    pool the registry's tables do not name is not recorded, never guessed.

    The fit SELECTION is read from the PUBLISHED dir's
    ``fit_selection.json`` (write_fit_selection put it in the temp dir before publication; the
    COMPLETE manifest hashes it) and the LoRA ARCHITECTURE from the object's own blob (what was
    fitted, not what was configured): ``fit_model_id``, ``fit_search_id``, ``fit_lr_selected_by``,
    ``fit_rank``, ``fit_alpha`` (= rank, scaling 1), ``fit_target_modules_json``, ``fit_layers_json``
    go to the hook as kwargs where its live signature accepts them, everything else —
    ``per_expert_rank``, ``lora_set_id``, ``trainable_params``, ``init_scheme``,
    ``architecture_selected_by``, ``architecture_selection_ruling_id``, ``fit_selection_sha256`` —
    rides ``extra``. A cell without the sidecar (a pre-selection config) emits exactly the legacy
    call. SELF-CONTAINED by design (every import inside, no module-level helper): the hook-coverage
    test execs this function's source alone.

    Never raises — but never silent either: every attempt
    leaves its EVIDENCE beside COMPLETE, ``registry_emission.json`` = {object_id, status, ...} with
    status ``emitted`` (the record paths), ``not_emitted`` (emission off on this host / the registry
    package absent / an undeclared arm or pool — the reason) or ``failed`` (the error), and a failed
    attempt also writes the bare ``REGISTRY_EMIT_FAILED`` marker (a success removes it). publish_cell
    reads the evidence back and raises RegistryEmissionError on ``failed``; the resume path
    (ensure_registry_emission) re-emits on anything but ``emitted`` — the whole hook, or, when the spool already
    holds one of the two records, ONLY the other one through ``emitter``: a
    callable with the hook's call signature that writes one record type (registry_emission's component emitter,
    injected by the caller so this body stays self-contained); default the hook itself."""
    import json
    import time
    final_dir = Path(final_dir)

    def _evidence(status: str, **more) -> None:
        marker: dict = {}
        try:
            marker = json.loads((final_dir / "COMPLETE").read_text())
        except (OSError, ValueError):
            marker = {}
        # "contract": the emission contract this evidence attests to — the literal equals registry_emission.EVIDENCE_CONTRACT
        # (pinned by tests/test_fit_selection_publication.py; this body is self-contained and cannot import it). Evidence from an
        # older contract is never terminal for a replay.
        ev = {"object_id": marker.get("object_sha256"), "status": status,
              "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "writer": "scripts/train_two_si_objects.py publish_cell", "contract": "2", **more}
        text = json.dumps(ev, indent=1, sort_keys=True) + "\n"
        tmp = final_dir / "registry_emission.json.tmp"
        tmp.write_text(text)
        tmp.replace(final_dir / "registry_emission.json")
        failed = final_dir / "REGISTRY_EMIT_FAILED"
        if status == "failed":
            failed.write_text(text)
        elif failed.exists():
            failed.unlink()

    try:
        import hashlib
        import inspect
        try:
            from sisteer.registry import hooks as _rh
        except ImportError as e:
            _evidence("not_emitted", reason=f"registry package unavailable ({e})")
            return
        kind = _rh.object_kind_for_legacy_kind(arm)
        tbm = _rh.read_sidecars(str(final_dir)).get("train_budget_meta.json") or {}
        corpus = tbm.get("training_budget_kind") == "document_tokens"
        # a system instruction's role, or — the corpus-fit pipeline's pools
        # (constv2-mix-* / constv2ctl2-*, review C1) — the role by prefix;
        # the hook's one resolver does both
        role = _rh.object_role_for_si(si_id)
        if kind is None or role is None:
            _rh.notice(f"train_two_si_objects: arm {arm!r} / si_id {si_id!r} not declared in the registry's "
                       f"tables (kind={kind}, role={role}); published object not recorded")
            _evidence("not_emitted", reason=f"arm {arm!r} / si_id {si_id!r} not declared in the registry's "
                                            f"tables (kind={kind}, role={role})")
            return

        def _sha(path) -> str:
            h = hashlib.sha256()
            with open(path, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            return h.hexdigest()

        # -- the published selection sidecar ({} when absent or torn; resume refuses a torn one anyway)
        sel_path = Path(final_dir) / _rh.FIT_SELECTION_SIDECAR
        sel: dict = {}
        if sel_path.is_file():
            try:
                loaded = json.loads(sel_path.read_text())
                sel = loaded if isinstance(loaded, dict) else {}
            except ValueError:
                sel = {}

        # -- the architecture the object's own blob states (a legacy o_proj-only lora_all blob carries
        #    no module keys, D contract §2, and reads as ["o_proj"] / wo; psr has no rank or modules;
        #    a non-torch artifact (safetensors) or an unreadable one is a stated absence, {})
        arch: dict = {}
        art = _rh.find_object_artifact(str(final_dir))
        if art:
            try:
                import torch
                blob = torch.load(art, map_location="cpu", weights_only=True)
            except Exception:  # noqa: BLE001 — not a torch blob
                blob = None
            if isinstance(blob, dict):
                def _numel(d) -> int:
                    return sum(int(t.numel()) for t in (d or {}).values() if hasattr(t, "numel"))
                bkind = blob.get("kind")
                if isinstance(blob.get("layers"), list):
                    arch["layers"] = [int(l) for l in blob["layers"]]
                if bkind == "lora_all":
                    a, b = blob.get("A") or {}, blob.get("B") or {}
                    rank = blob.get("rank")
                    if rank is None and a:
                        t0 = next(iter(a.values()))
                        rank = 1 if t0.dim() == 1 else int(t0.shape[-1])
                    arch.update({
                        "rank": int(rank) if rank is not None else None,
                        "lora_modules": sorted(str(m) for m in (blob.get("lora_modules") or ["o_proj"])),
                        "lora_set_id": blob.get("lora_set_id") or "wo",
                        "per_expert_rank": blob.get("per_expert_rank"),
                        "init_scheme": blob.get("init_scheme", "zero"),
                        "trainable_params": _numel(a) + _numel(b)})
                elif bkind == "psr":
                    n = sum(_numel(blob.get(k)) for k in ("directions", "gate_w", "gate_b")
                            if isinstance(blob.get(k), dict))
                    arch.update({"trainable_params": n or None,
                                 "init_scheme": blob.get("init_scheme", "zero")})
                arch = {k: v for k, v in arch.items() if v is not None}

        # -- the registry model_id for the fit checkpoint the sidecar names (an HF repo) through
        #    decode_policy.json's model_spellings — the seedrep emitter's rule; None when nothing maps it,
        #    so the hook's own fallback (the sidecar's spelling) stands and nothing is guessed
        fit_model_id = None
        if tbm.get("model_id"):
            try:
                from sisteer.registry.decode_policy import PolicyFile
                pf = PolicyFile.load()
                mid = pf.model_id(str(tbm["model_id"]))
                fit_model_id = mid if (mid != str(tbm["model_id"]) or mid in pf.models) else None
            except Exception:  # noqa: BLE001 — no export on this host: stated absence
                fit_model_id = None

        rank = arch.get("rank", sel.get("lora_rank"))
        stated = {
            "fit_model_id": fit_model_id,
            "fit_search_id": sel.get("fit_search_id"),
            "fit_lr_selected_by": sel.get("fit_lr_selected_by"),
            "fit_rank": int(rank) if rank is not None else None,
            "fit_alpha": float(rank) if rank is not None else None,   # LoRA scaling 1: alpha = r
            "fit_target_modules_json": arch.get("lora_modules") or sel.get("lora_modules"),
            "fit_layers_json": arch.get("layers") or tbm.get("target_layers"),
        }
        ruling = sel.get("architecture_selection_ruling_id")
        extra = {
            "per_expert_rank": arch.get("per_expert_rank", sel.get("per_expert_rank")),
            "lora_set_id": arch.get("lora_set_id"),
            "trainable_params": arch.get("trainable_params"),
            "init_scheme": arch.get("init_scheme") or sel.get("init_scheme"),
            "architecture_selection_ruling_id": ruling,
            # a user-picked architecture is recorded `manual`
            "architecture_selected_by": sel.get("architecture_selected_by") or ("manual" if ruling else None),
            "fit_selection_sha256": _sha(sel_path) if sel_path.is_file() else None,
        }
        # -- only the kwargs the installed hook ACCEPTS (an unknown kwarg raises inside its guard and the
        #    record is silently lost); every other fact rides extra; None-valued facts are dropped
        try:
            params = inspect.signature(_rh.object_from_sidecar).parameters
            var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        except (TypeError, ValueError):
            params, var_kw = {}, False
        kwargs: dict = {}
        extra = {k: v for k, v in extra.items() if v is not None}
        # fit-time provenance a REPLAY restates: the object's prior record's statement, never this job's context;
        # the cell's fit_provenance.json sidecar, when present, wins inside the hook. None at the first emission (the fit job is the emitter).
        if fit_provenance:
            stated["fit_provenance"] = dict(fit_provenance)
        for k, v in stated.items():
            if v is None:
                continue
            (kwargs if (k in params or var_kw) else extra)[k] = v
        # the records' campaign is STATED by the job env (REGISTRY_CAMPAIGN; the front door sets it to the paper frame's
        # sdfbase_full22) — never a default here: the old `two_si` default filed this model's fit objects outside the frame
        # and a default-campaign record UPSERTs an adapter-filed object's campaign (stage-E ruling 2026-10-02). Resolved only
        # when emission is on (emission off stays `not_emitted`, not a campaign failure); unset -> CampaignUnset -> `failed`
        # evidence + REGISTRY_EMIT_FAILED through the except below, never a silent record.
        # (inline, not through registry_emission: this function is self-contained by contract — tests exec its source alone)
        campaign = os.environ.get("REGISTRY_CAMPAIGN") or None
        if campaign is None and _rh.emission_enabled(None):
            raise RuntimeError("REGISTRY_CAMPAIGN is unset — a record's campaign is the writer's statement, never a default: "
                               "the front door's job env sets it (SISTEER_REGISTRY_CAMPAIGN, default sdfbase_full22); a manual "
                               "re-emission runs with REGISTRY_CAMPAIGN=<campaign> set explicitly (CampaignUnset)")
        hook = emitter if emitter is not None else _rh.emit_object_from_sidecar
        # hook-fix-5: the WEIGHT-content identity of the published object and of every published epoch checkpoint
        # (path -> canonical hash), so the registry records weight equivalence (same_weights_as_published) in addition to
        # byte identity; passed only when the installed hook accepts the kwarg (an unknown kwarg would raise inside its guard)
        weights_identity: dict = {}
        try:
            hook_params = inspect.signature(hook).parameters
        except (TypeError, ValueError):
            hook_params = {}
        # (resolved through globals(): tests/test_registry_emit_fit.py execs THIS function's source in a tiny namespace to
        #  prove the hook is inert with emission off — there the helpers are absent and the map simply stays empty)
        _wsha = globals().get("weights_content_sha256") or (lambda _p: None)
        _idx_name = globals().get("EPOCH_CKPT_INDEX", "epoch_checkpoints.json")
        if "weights_identity" in hook_params:
            if art:
                wsha = _wsha(art)
                if wsha:
                    weights_identity[str(art)] = wsha
            idx_path = Path(final_dir) / _idx_name
            if idx_path.is_file():
                try:
                    import json as _json
                    for c in (_json.loads(idx_path.read_text()).get("checkpoints") or []):
                        ep_path = Path(final_dir) / str(c.get("artifact") or "")
                        if ep_path.is_file():
                            w = c.get("weights_sha256") or _wsha(ep_path)   # recorded by the saver, else computed here
                            if w:
                                weights_identity[str(ep_path)] = w
                except ValueError:
                    pass
        if dry_run:
            # the preflight's view: the exact records the hook WOULD write, nothing written, no evidence touched
            return hook(
                str(final_dir), campaign=campaign, object_kind=kind, dry_run=True,
                **({"weights_identity": weights_identity} if "weights_identity" in hook_params else {}),
                paper_label=_rh.paper_label_for(kind, role), object_role=role,
                fit_target_kind="corpus_document_fit" if corpus else "system_instruction_distillation",
                fit_objective="ce" if corpus else "kl", fit_thinking_render="off", status="reference",
                fit_teacher_si_id=None if corpus else si_id, fit_corpus_id=si_id if corpus else None,
                script="scripts/train_two_si_objects.py publish_cell", extra=extra or None, **kwargs)
        paths = hook(
            str(final_dir), campaign=campaign, object_kind=kind, **({"weights_identity": weights_identity} if "weights_identity" in hook_params else {}),
            paper_label=_rh.paper_label_for(kind, role), object_role=role,
            fit_target_kind="corpus_document_fit" if corpus else "system_instruction_distillation",
            fit_objective="ce" if corpus else "kl", fit_thinking_render="off", status="reference",
            fit_teacher_si_id=None if corpus else si_id, fit_corpus_id=si_id if corpus else None,
            script="scripts/train_two_si_objects.py publish_cell", extra=extra or None, **kwargs)
        if paths is None:
            # the hook's own guard swallowed a failure (its notice is on stderr): a record was NOT written
            raise RuntimeError("the registry hook failed inside its guard (see the [registry] notice above)")
        if paths:
            _evidence("emitted", records=[str(x) for x in paths], campaign=campaign,
                      spool=_rh.resolve_spool())
        else:
            _evidence("not_emitted", reason="registry emission is off on this host (REGISTRY_EMIT_DISABLED / "
                                            "REGISTRY_EMIT=0, or no spool resolves)")
    except Exception as e:  # noqa: BLE001 — a lost record never costs a publication; it is RECORDED, not swallowed
        try:
            _evidence("failed", error=f"{type(e).__name__}: {e}")
        except Exception:  # noqa: BLE001 — the evidence write itself failing must not mask the notice
            pass
        print(f"[registry] train_two_si_objects: published object not recorded "
              f"({type(e).__name__}: {e}) — REGISTRY_EMIT_FAILED written beside COMPLETE; the artefact "
              "is kept and a re-run replays the record without retraining", file=sys.stderr, flush=True)


def validated_complete(save_dir, cell_id: str,
                       expected_fingerprint: str) -> Path | None:
    """The resume gate: the cell's object path IFF its COMPLETE marker
    validates — marker parses, names this cell, carries a fit_fingerprint
    MATCHING the requested work (F15: identity-bound, so an all-token or
    other-model marker never stands in for this cell), and every file its
    manifest promises exists with the recorded sha256, with the provenance
    sidecar parseable (F16). Anything less — marker absent, fingerprint
    absent/mismatched (a pre-fingerprint marker re-runs BY DESIGN), a
    manifested file missing/modified, a pre-manifest marker — is an
    INCOMPLETE cell and returns None, so resume re-runs it instead of
    counting an unprovenanced artifact as done."""
    final_dir = cell_dir_for(save_dir, cell_id)
    marker_path = final_dir / COMPLETE_MARKER_NAME
    if not marker_path.exists():
        return None
    try:
        marker = json.loads(marker_path.read_text())
    except ValueError:
        print(f"[train-two-si] {marker_path} is unparseable — treating "
              f"{cell_id} as incomplete (will re-run)", file=sys.stderr)
        return None
    if marker.get("cell_id") != cell_id:
        print(f"[train-two-si] {marker_path} names "
              f"{marker.get('cell_id')!r}, not {cell_id!r} — treating the "
              "cell as incomplete", file=sys.stderr)
        return None
    from sisteer.extraction.fit_fingerprint import fingerprints_match

    if not fingerprints_match(marker.get("fit_fingerprint"),
                              expected_fingerprint):
        print(f"[train-two-si] {cell_id}: COMPLETE marker's fit fingerprint "
              f"({marker.get('fit_fingerprint') or 'absent'}) does not match "
              "the requested fit (model/revision, SI, arm, positions, pool, "
              "optimizer, config) — not this cell's work; will re-run",
              file=sys.stderr)
        return None
    obj_file = str(marker.get("object_file") or "")
    files = marker.get("files")
    if not obj_file or not isinstance(files, dict):
        print(f"[train-two-si] {cell_id}: COMPLETE marker predates the "
              "publication manifest (no files/object_file record) — "
              "incomplete, will re-run", file=sys.stderr)
        return None
    try:
        required = _required_publication_files(final_dir, obj_file)
    except RuntimeError as e:
        # A present-but-malformed epoch index is publication corruption like any other —
        # the cell reads INCOMPLETE (resume re-runs / re-publishes it), never an uncaught crash at the gate
        print(f"[resume] {final_dir}: {e} — treating the cell as INCOMPLETE", file=sys.stderr)
        return None
    missing = [n for n in required if n not in files]
    if missing:
        print(f"[train-two-si] {cell_id}: COMPLETE manifest lacks required "
              f"entries {missing} — incomplete, will re-run",
              file=sys.stderr)
        return None
    for name, sha in files.items():
        p = final_dir / str(name)
        if not p.is_file() or _sha256_file(p) != sha:
            print(f"[train-two-si] {cell_id}: manifested file {name} is "
                  "missing or does not match its published sha256 — "
                  "incomplete, will re-run", file=sys.stderr)
            return None
    try:
        json.loads((final_dir / "train_budget_meta.json").read_text())
    except ValueError:
        print(f"[train-two-si] {cell_id}: train_budget_meta.json does not "
              "parse — provenance lost post-publish; incomplete, will "
              "re-run", file=sys.stderr)
        return None
    return final_dir / obj_file


def _emission_mod():
    """scripts/sdfbase/registry_emission.py — the ONE owner of the evidence reader, the spool scan and the
    verify-or-replay step, loaded by path. _registry_publish keeps its inline
    evidence WRITER because that function is pinned self-contained; the literals equal the module's constants."""
    mod = sys.modules.get("registry_emission")
    if mod is None:
        import importlib.util
        path = Path(__file__).resolve().parent / "sdfbase" / "registry_emission.py"
        spec = importlib.util.spec_from_file_location("registry_emission", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["registry_emission"] = mod
        spec.loader.exec_module(mod)
    return mod


def registry_emission_evidence(final_dir) -> dict | None:
    """The emitter's evidence beside COMPLETE (:data:`REGISTRY_EMISSION_NAME`),
    or None when absent / unparseable."""
    return _emission_mod().read_evidence(final_dir)


def _spool_object_records(object_id: str, campaign: str) -> list[str]:
    """registry_emission.spool_object_records — the fallback evidence for a
    publication that predates the evidence file (an ``object`` record of the
    id already in this host's producer spool)."""
    return _emission_mod().spool_object_records(object_id, campaign)


def _raise_if_emission_failed(final_dir) -> dict | None:
    """publish_cell's / the resume's read-back of the emitter's evidence:
    ``failed`` raises RegistryEmissionError naming the marker; anything else
    is returned (``emitted``, or ``not_emitted`` with its stated reason —
    emission off by configuration is the operator's call, said in the
    evidence and refused downstream by 04_fit_objects.sh, not a failure of
    this job)."""
    ev = registry_emission_evidence(final_dir)
    if ev is not None and ev.get("status") == "failed":
        raise RegistryEmissionError(
            f"registry emission of the published object at {final_dir} FAILED "
            f"({ev.get('error')}); {Path(final_dir) / REGISTRY_EMIT_FAILED_NAME} "
            "written — the fit is kept (COMPLETE stands); fix the spool / push "
            "and re-run: the resume validates COMPLETE and replays the record "
            "without retraining")
    return ev


def ensure_registry_emission(final_dir, *, si_id: str, arm: str) -> dict | None:
    """Registry emission as a REPLAYABLE, idempotent step: on a validated
    COMPLETE, verify that the native ``object`` row was emitted (the
    evidence file keyed by the marker's object id) and otherwise RECONCILE
    the producer spool for BOTH records the hook writes: object + artifact
    present → evidence, nothing re-emitted; exactly one present → ONLY the
    missing record is replayed (``_registry_publish`` with the module's
    component emitter — the hook writes both rows unconditionally, so a
    second whole-hook call after a failure between its two writes would
    duplicate the object row); neither → the whole hook. Read on every
    non-emitted status — absent, stale (another object id), ``failed``,
    ``not_emitted``. Never retrains. Returns the evidence; raises
    RegistryEmissionError when the (re-)emission failed."""
    final_dir = Path(final_dir)
    mod = _emission_mod()
    marker = json.loads((final_dir / COMPLETE_MARKER_NAME).read_text())
    object_id = str(marker.get("object_sha256") or "")
    ev = registry_emission_evidence(final_dir)
    # the campaign the job STATES comes first: terminal evidence is evidence for THIS campaign under the
    # CURRENT emission contract — a cell published under the emitters' old `two_si` default is re-emitted here when the job now
    # states sdfbase_full22, through this very replay (validates COMPLETE, never retrains). With emission off no campaign is needed.
    campaign = None
    if mod.emission_on():
        try:
            campaign = mod.campaign_from_env()
        except mod.CampaignUnset as e:
            mod.failed(final_dir, object_id, "scripts/train_two_si_objects.py ensure_registry_emission", "object", f"CampaignUnset: {e}")
            raise RegistryEmissionError(f"{final_dir}: {e}") from e
    if mod.evidence_verifies(ev, object_id, campaign):
        return ev
    why = mod.evidence_mismatch(ev, object_id, campaign)
    writer = "scripts/train_two_si_objects.py ensure_registry_emission"
    rec = mod.reconcile_spool(object_id, campaign)
    if rec.get("refused"):
        why += "; the spool's record(s) for it would be REFUSED at ingest (" + "; ".join(
            f"{Path(f).name}: {fld}" for f, fld, _r in rec["refused"]) + ")"
    # a replay RESTATES the fit's provenance from the object's prior record (the campaign the stale evidence names, else this one);
    # the cell's fit_provenance.json sidecar, when present, wins inside the hook
    prior = mod.fit_provenance_from_prior_record(object_id, (ev or {}).get("campaign") or campaign) \
        or mod.fit_provenance_from_prior_record(object_id, campaign)
    if rec["state"] == "complete":
        print(f"[registry] {final_dir.name}: {why}, but the object + artifact records for {object_id[:16]}… are "
              f"already in the spool ({len(rec['records'])} file(s)) — evidence written, nothing re-emitted")
        return mod.write_evidence(final_dir, "emitted", object_id=object_id, writer=writer,
                                  records=rec["records"], campaign=campaign, verified_from="spool")
    if rec["state"] == "partial":
        missing, = rec["missing"]
        present = [c for c in mod.COMPONENTS if c not in rec["missing"]]
        print(f"[registry] {final_dir.name}: validated COMPLETE, {why}; the spool holds the {present[0]} record for "
              f"{object_id[:16]}… but no {missing} record — replaying ONLY the {missing} record (the fit is not re-run, "
              "no row is duplicated)", flush=True)
        _registry_publish(final_dir, si_id=si_id, arm=arm, emitter=mod.component_emitter(missing), fit_provenance=prior)
        ev2 = _raise_if_emission_failed(final_dir)
        if ev2 is not None and ev2.get("status") == "emitted":
            ev2 = mod.write_evidence(final_dir, "emitted", object_id=object_id, writer=writer,
                                     records=rec["records"] + list(ev2.get("records") or []), campaign=campaign,
                                     spool=ev2.get("spool"), verified_from="spool+replay", replayed=[missing])
        return ev2
    print(f"[registry] {final_dir.name}: validated COMPLETE but {why} and nothing for it in the spool — re-emitting "
          "the object record (the fit is not re-run" + ("; fit provenance restated from its prior record" if prior else "") + ")", flush=True)
    _registry_publish(final_dir, si_id=si_id, arm=arm, fit_provenance=prior)
    return _raise_if_emission_failed(final_dir)


class _ReplicaTrained(Exception):
    """Control flow inside train_si_objects' per-arm try: a data-parallel
    replica (rank > 0) finished its trainer call — record the arm locally
    and skip everything rank 0 owns (guards, save, sidecars, publication)."""


def _write_report(outcome: FitOutcome, save_dir,
                  report_path: Path | None = None) -> None:
    """(Re)write the fit report after every arm reaches a terminal state, so
    a crash mid-run still leaves an accurate record of the arms before it.
    Error MESSAGES are truncated: the full traceback belongs to the job log
    (mode-0700 on the cluster), not a file under ``out/``.

    Under a data-parallel fit only rank 0 writes — the replicas are asserted
    bit-equal inside the trainer, so rank 0's record IS the run's record."""
    if save_dir is None and report_path is None:
        return
    if not _is_main_rank():
        return
    if report_path is not None:
        # The arm-task FRAGMENT convention (parallel driver): a per-arm
        # file, merged into the canonical report by the scheduler, so
        # concurrent arm processes never race one shared file.
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        path = report_path_for(save_dir, outcome.si_id)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    # MERGE, do not replace. This file is per-SI, but a wave trains one arm
    # (or one group) per invocation, so a replacing writer makes each run
    # erase every earlier arm's entry for that SI. That is not hypothetical:
    # job 31850 clobbered si_constitution's lora_r1 entries here with no
    # backup, and only luck (an unrelated lane taking one) preserved
    # si_irrelevant's. The per-cell train_budget_meta.json survives either
    # way, so no measurement was lost -- but any tool pairing
    # <si>__fit_report.json with an object directory silently mis-attributes
    # fit status, which is worse than an absent file.
    carried: list[str] = []
    arms = {}
    if path.exists():
        try:
            prior = json.loads(path.read_text())
            if isinstance(prior.get("arms"), dict):
                arms.update(prior["arms"])
                carried = sorted(arms)
        except (OSError, ValueError) as e:  # unreadable/corrupt: say so, do
            # not let it cost THIS run's record
            print(f"[train-two-si] could not merge prior report at {path} "
                  f"({type(e).__name__}: {e}) -- writing this run's arms only",
                  file=sys.stderr)
            arms, carried = {}, []
    fresh = {a: o.as_dict() for a, o in outcome.arms.items()}
    for a, rec in fresh.items():
        if isinstance(rec, dict):
            rec = dict(rec)
            rec.setdefault("written_at", now)
        arms[a] = rec
    payload = {
        "si_id": outcome.si_id,
        "arms": arms,
        # Entries carried over from earlier runs, i.e. NOT written by this
        # invocation. Stated so a reader can tell an inventory from a
        # transcript: `failed` and `skipped_existing` below describe THIS
        # run only, while `arms` accumulates.
        "arms_from_earlier_runs": sorted(set(carried) - set(fresh)),
        "failed": sorted(outcome.failed),
        "skipped_existing": outcome.skipped,
        "condition_sources": outcome.condition_sources,
        "updated_at": now,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True, default=str))
    tmp.replace(path)
    outcome.report_path = path


def train_si_objects(
    model,
    tok,
    si_id: str,
    pairs,
    arms: list[str],
    *,
    cfg: dict,
    save_dir=None,
    sentinel=None,
    log_fn=None,
    overwrite: bool = False,
    bidpo_harmful_pairs=None,
    bidpo_benign_pairs=None,
    bidpo_benign_meta: dict | None = None,
    report_path=None,
    pool_sha256: str | None = None,
    config_sha: str | None = None,
    teacher_topk: str | None = None,
    pairs_path: str | None = None,
    checkpoint_dir=None,
    checkpoint_every: int | None = None,
    epochs_cap: int | None = None,
    continue_from=None,
    continue_lr_mult: float = 1.0,
) -> FitOutcome:
    """Train every arm in ``arms`` for ``si_id`` in this one process. Returns
    a :class:`FitOutcome` (``.objects`` = the objects trained this run,
    ``.arms`` = every arm's terminal state). If ``save_dir`` is given, each
    object is saved to its cell-id path ``<save_dir>/<si_id>__<arm>/object...``
    (the path ``two_si_run generate`` loads) with a ``train_budget_meta.json``
    sidecar — saved the INSTANT its arm completes, so a crash in a later arm
    cannot discard it (jobs 148862/148863 lost everything to an end-of-list
    save). A per-run fit report is rewritten after every arm.

    Compatible arms share a lockstep teacher forward; the rest train solo. All
    arms use the per-SI ``max_len`` and per-cell batch size from
    ``eval.train_budget``; the lockstep group uses one shared (safe) width.

    Isolation: an exception in one arm (CAST's condition-guard refusal
    included) is recorded and the remaining arms still train and save — it is
    never fatal to the run. The CLI turns ``not outcome.ok`` into a non-zero
    exit AFTER every arm has been attempted.

    Resume: unless ``overwrite``, an arm whose object already exists on disk
    is skipped (recorded ``skipped_existing``). A partially-done lockstep
    group DROPS its finished members and trains only the remainder — at the
    batch size of the FULL requested compatible list, never recomputed over
    the survivors. That choice (drop, not retrain-and-overwrite) is what makes
    a re-run cost nothing for finished arms, and it cannot perturb what the
    remaining arms compute: lockstep gives every arm its own optimiser and
    RNG stream (group membership cannot alter a member's trajectory —
    ``test_lockstep`` proves group == solo bitwise), the object constructors
    draw from their own seeded ``torch.Generator``s (never global RNG), and
    the one group-coupled input — batch composition — is held fixed by
    pinning the batch size to the full requested group.

    Per-arm config ``train_kwargs.<arm>`` (the ``cli.py`` convention) are
    merged over ``train_kwargs.default`` for SOLO arms only — lockstep arms'
    per-arm params are pinned to their registry defaults by the bitwise
    contract. Arms whose trainer signature declares ``condition_pos_prompts``/
    ``condition_neg_prompts`` (none registered today — CAST was the consumer
    until its 2026-08-20 registry removal) additionally receive the category
    condition from :func:`resolve_condition_contrast`: harmful -> should-fire
    (pos), benign -> should-not-fire (neg).

    ``pool_sha256``/``config_sha`` (F15): the pool-file and config-file byte
    hashes the CLI paths pass in (``_cli_fit_identity``); programmatic
    callers may omit them and get the content-derived fallbacks
    (:func:`pool_content_sha256`, :func:`_cfg_dict_sha`) — either way the
    SAME derivation runs at publish time and at every later skip check, so
    resume is bound to the fit's identity rather than to file names.

    ``checkpoint_dir``/``checkpoint_every`` — the mid-fit durability
    opt-in (fit-wave hardening, 2026-08-27): when a dir is given, the
    trainers with a checkpoint seam (train_lockstep, and train_psr's mse
    loop for the solo mse arms) write a resumable full-state snapshot
    every ``checkpoint_every`` optimizer steps (default
    ``fitckpt.DEFAULT_CHECKPOINT_EVERY`` = 100) under
    ``<checkpoint_dir>/<cell_id>/``, keyed to the cell's fit fingerprint
    — the SAME identity the COMPLETE-marker resume validates — so a
    killed fit resumes exactly, and a checkpoint from a different fit
    identity is ignored loudly. Checkpoints live OUTSIDE the cell dirs on
    purpose: the atomic-publish audit and the COMPLETE protocol never see
    them. Default None = off, byte-identical to before the seam.
    """
    from sisteer.data.pairs import pairs_to_distill_examples
    from sisteer.eval.defense_cell import object_path_for
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.eval.train_budget import resolve_batch_size, resolve_max_len
    from sisteer.extraction.distill import (
        EarlyStopSpec, floor_ratio_check, probe_values_record)
    from sisteer.extraction.fitckpt import DEFAULT_CHECKPOINT_EVERY
    from sisteer.extraction.lockstep import (
        LockstepArm, arm_probe_losses, build_lockstep_object, train_lockstep)
    from sisteer.extraction.psr import (
        GUARD_PROBE_FLOOR, PSRObject, check_psr_gates)
    from sisteer.methods import ARMS

    # The two-SI SI resolver handles safety_refusal (the SAFETY_SI module
    # constant, deliberately OUT of the registry) as well as registered SIs.
    instruction = resolve_instruction(si_id)
    layer = int(cfg.get("layer", 15))
    # bidpo's rejected-side source (bidpo amendment 2026-08-20): the anti-safety
    # harmful compliance (pool.anti_neg.enabled) vs the legacy undefended strict
    # subset. Read once so the whole SI's fit routes bidpo consistently and the
    # choice is recorded per object.
    anti_neg_enabled = bool(
        ((cfg.get("pool") or {}).get("anti_neg") or {}).get("enabled", False))
    # bidpo's OWN pool — read
    # through the pool builder's ONE canonical config reader (a sibling
    # script, stdlib-cheap to import; no cycle: it never imports this
    # module). ``bidpo_harmful_pairs``/``bidpo_benign_pairs`` are the two
    # per-SI files cmd_train loaded from emit-pairs (bidpo_harmful_<si> /
    # bidpo_benign_<si>.jsonl); ``bidpo_benign_meta`` carries the emit
    # sidecar's per-SI counts + projected ratios + the refusal-label
    # setting, stamped into bidpo's meta.json and fit-report record
    # (the three objects train at DIFFERENT harmful:benign
    # ratios, so a bidpo cross-SI number is never presented without the
    # per-SI counts AND ratios). Enabled-but-nothing-supplied fails bidpo's
    # arm loudly in the solo loop (isolation: the other arms still train).
    import build_two_si_pool as _bp

    bb_cfg = _bp._bidpo_benign_cfg(cfg)
    if ((bidpo_benign_pairs is not None or bidpo_harmful_pairs is not None)
            and not bb_cfg["enabled"]):
        raise SystemExit(
            "[train-two-si] bidpo own-pool pairs were supplied but "
            "pool.bidpo_benign.enabled is false — a contradictory invocation "
            "(refusing to guess which was intended).")
    max_len = resolve_max_len(cfg, si_id)
    # Per-cell caps (train_kwargs_by_si.<si>) merged over the defaults —
    # solo/CE-path only: a lockstep arm would silently ignore them, refused.
    refuse_by_si_for_lockstep_arms(cfg, si_id, arms)
    tk = _train_kwargs_for_si(cfg, si_id)
    # Config-shape guard for the document-CE arms — the
    # authoritative choke point every entrypoint crosses; cmd_train also
    # runs it next to refuse_silently_dropped_kwargs for the fast refusal.
    refuse_ce_misconfiguration(cfg, arms)
    doc_mode = _data_mode(cfg) == DOC_DATA_MODE
    epochs = int(tk.get("epochs", 6))
    seed = int(tk.get("seed", 0))
    # Stage-3 fit knobs — read with in-code production defaults (the epochs
    # pattern), so absent config keys mean the DEFAULT behavior, not off:
    #   holdout_frac / holdout_salt   — the deterministic held-out slice of
    #                                   the pair pool (never trained on);
    #   early_stop_patience/min_epochs/min_delta — per-arm stopping on the
    #                                   held-out loss; `epochs` is the CAP;
    #   floor_ratio_max               — end-of-training no-op-floor guard
    #                                   (null disables; default 3x);
    #   positions / response_header_len — the fit's mask leg ("all" |
    #                                   "response"; H derived from the
    #                                   tokenizer when unset, the
    #                                   header-steered convention).
    holdout_frac = float(tk.get("holdout_frac", 0.1))
    holdout_salt = str(tk.get("holdout_salt", "two_si_holdout"))
    early_spec_cfg = EarlyStopSpec(
        patience=int(tk.get("early_stop_patience", 2)),
        min_epochs=int(tk.get("early_stop_min_epochs", 1)),
        min_delta=float(tk.get("early_stop_min_delta", 0.0)))
    frm = tk.get("floor_ratio_max", 3.0)
    floor_ratio_max = float(frm) if frm is not None else None
    # probe_n — the floor probe's SAMPLE SIZE (train_kwargs.default.probe_n,
    # in-code default 256 = lockstep.DEFAULT_PROBE_N; null = every fit
    # example). DELIBERATELY not derived from any batch size: the wave's
    # probe read batch 0 only at the arm's TRAINING batch size, so every
    # floor and ratio the guard ruled on was a 2-4 sample estimate of a KL
    # — measured against the true floors (job 187947, n=256/SI: 0.10516 /
    # 0.07129 / 0.06898 with +/-7-9% CIs), si_irrelevant's batch-0 floor
    # (0.1404) was 2x the truth and its whole wave was judged too
    # leniently. Sample size is a statistics knob; batch size is a memory
    # knob; they meet again only at the clamp to the available pool.
    pn = tk.get("probe_n", 256)
    probe_n = int(pn) if pn is not None else None
    if probe_n is not None and probe_n < 1:
        raise SystemExit(
            f"[train-two-si] train_kwargs.default.probe_n must be >= 1 "
            f"(or null for every fit example), got {pn!r}")
    # Production monitoring: dual-KL probes at
    # dense->sparse cadence, per-step grad/object norms, the non-finite
    # abort, per-cell metrics mirror, heartbeat. Default ON — the monitors
    # are observational (no RNG, no writes, buffered sync-free scalars) so
    # the trained objects are bit-identical monitors-on vs off (pinned by
    # tests/test_lockstep.py); `monitors: false` is the escape hatch.
    from pod_acceptance_probe import _NON_STEP_SUFFIXES

    from sisteer.extraction import fitmon

    # A missing model/tokenizer means a routing/stub caller (peer tests
    # drive train_si_objects with fake arms and no model at all): nothing
    # to probe, nothing to tokenize — monitors and bucketing stand down.
    # Data-parallel: rank 0 owns every write and every probe; the
    # replicas train (the trainer shards and asserts them bit-equal) and
    # record their arm locally — no monitor thread, no temp cell dir, no
    # sidecar, no report, no registry emission.
    is_main = _is_main_rank()
    fit_width = fit_width_active()
    monitors_on = (bool(tk.get("monitors", True))
                   and model is not None and tok is not None and is_main)
    cadence = fitmon.ProbeCadence.from_train_kwargs(tk)
    probe_kl_n = int(tk.get("probe_kl_n", 64))
    # Length-bucketed batches in the PRODUCTION path (gpu-util directive
    # 2026-08-25; measured with cuDNN-SDPA-off: 2.6 -> 0.62-0.78 s/step,
    # 33-45% -> 99-100% duty at identical bs). Batch COMPOSITION changes vs
    # an unsorted fit — recorded per cell (`length_bucketed` in the
    # sidecar); the densest batch becomes all-long rows, so memory probes
    # must measure the N-longest rows (gpu_memprobe --worst-case).
    # bidpo's OWN pool files stay in emit order (the DPO
    # trainer's row-weighted accumulation groups are the contract).
    length_bucket = bool(tk.get("length_bucket", True)) and tok is not None
    # Teacher-side prefix KV caching (see the config block): every
    # teacher prompt for one SI shares its rendered prefix (chat header
    # + SI ≈ 95% of the teacher tokens for the long SIs), so the trainers
    # run it once and replay the KV per batch — for GRADIENT-TRAINING
    # forwards only. Every DECISION-BEARING quantity (the floor probes
    # the floor_ratio_max bar rules on, held-out epoch selection) is
    # AUTHORITATIVE AND UNCACHED inside the trainers regardless of this
    # flag: a rejection writes no artifact, so a cache-flipped verdict is
    # unrecoverable after the fact. Config-gated, default ON —
    # `teacher_prefix_cache: false` is the escape hatch for the training
    # legs. The library trainers default OFF so every caller outside this
    # driver stays byte-identical; THIS is where the wave's default-true
    # lives.
    teacher_prefix_cache = bool(cfg.get("teacher_prefix_cache", True))
    positions = str(tk.get("positions", "all"))
    response_header_len = None
    if positions != "all":
        rhl_cfg = tk.get("response_header_len")
        if rhl_cfg is not None:
            response_header_len = int(rhl_cfg)
        else:
            from sisteer.hooks import response_header_len as _derive_h

            response_header_len = _derive_h(tok)
            if response_header_len == 0:
                raise SystemExit(
                    f"[train-two-si] positions={positions!r} but the "
                    "assistant-header length could not be derived from this "
                    "tokenizer (H == 0 — no chat template, or a stripped one "
                    "the marker fallback does not know). A zero H silently "
                    "collapses header-steered to plain decode-only; set "
                    "train_kwargs.default.response_header_len explicitly.")

    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"[train-two-si] unknown arm(s): {unknown}")

    # Fit identity (F15) — resolved ONCE, before any carve or reorder, so
    # the hash covers the pool exactly as supplied and the fingerprint at
    # publish time equals the one every later skip check recomputes.
    if pool_sha256 is None:
        pool_sha256 = pool_content_sha256(pairs)
    if config_sha is None:
        config_sha = _cfg_dict_sha(cfg)

    # Offline top-K teacher artifact (precomputed): load + KEY-CHECK once
    # per SI, before any carve — the artifact is keyed on the FULL pool's
    # content hash, this (model, revision, dtype), this SI, this SI's
    # resolved max_len, and top_k=DEFAULT_TOP_K. Any mismatch (or a
    # missing file) FALLS BACK to the online teacher with a loud line —
    # never a silent consume of a different world's teacher, never a
    # crashed wave over a stale sidecar. Consumption is additionally
    # gated per lockstep group below (forward-KL arms at the audited T
    # grid only) and row-sha-bound inside train_lockstep, so three
    # independent guards sit between a wrong artifact and a trained
    # object.
    topk_store = None
    topk_meta_rec = None
    if teacher_topk:
        from sisteer.extraction.teacher_topk import DEFAULT_TOP_K, TeacherTopK

        try:
            store = TeacherTopK.load(teacher_topk)
            problems = store.validate(
                model=cfg.get("model"),
                model_revision=cfg.get("model_revision"),
                si_id=si_id,
                pool_content_sha256=pool_content_sha256(pairs),
                max_len=max_len,
                top_k=DEFAULT_TOP_K,
                dtype=cfg.get("dtype"))
        except Exception as e:  # noqa: BLE001 — loud fallback, never fatal
            print(f"[teacher-topk] FALLBACK to the ONLINE teacher — could "
                  f"not load {teacher_topk}: {type(e).__name__}: {e}",
                  flush=True)
        else:
            if problems:
                print(f"[teacher-topk] FALLBACK to the ONLINE teacher — "
                      f"{teacher_topk} does not match this fit: "
                      f"{'; '.join(problems)}", flush=True)
            else:
                topk_store = store
                topk_meta_rec = {
                    "path": str(teacher_topk),
                    "top_k": store.top_k,
                    "n_examples": len(store),
                    "n_positions": int(store.idx.shape[0]),
                    "pool_content_sha256": store.meta.get(
                        "pool_content_sha256"),
                    "build": store.meta.get("build"),
                }
                print(f"[teacher-topk] artifact verified for {si_id}: "
                      f"{teacher_topk} (top_k={store.top_k}, "
                      f"{len(store)} examples, "
                      f"{int(store.idx.shape[0])} positions) — the "
                      "lockstep KL arms will consume it instead of "
                      "running the online teacher", flush=True)

    # The deterministic held-out carve (hash on the prompt, split_prompts'
    # convention). Everything downstream — lockstep examples, PSR inits,
    # solo trainers, the condition contrast — sees only fit_pairs; the
    # held-out pairs feed exclusively the per-epoch held-out loss.
    # EXCEPT under an explicit shared held-out pool
    # (train_kwargs.default.heldout_pool): every cell of the campaign — and
    # every HPO trial — evaluates per-epoch CE on exactly the SAME pool, so
    # the carve is skipped (fit_pairs = the whole cell pool) and the
    # held-out side loads from the pool file. Absent key = the existing
    # behavior byte-for-byte. refuse_ce_misconfiguration already refused
    # heldout_pool + holdout_frac > 0 together.
    # Resolved through the ONE derivation the fingerprint uses
    # (heldout_pool_identity), so the sha recorded in every budget sidecar
    # is byte-identically the one bound into the cell's fit identity.
    heldout_pool, heldout_pool_sha = heldout_pool_identity(cfg, si_id)
    if heldout_pool:
        fit_pairs = list(pairs)
        heldout_path = resolve_vol_path(heldout_pool)
        heldout_pairs = _load_pairs(heldout_path)
        via = (f" (read at {heldout_path})"
               if heldout_path != str(heldout_pool) else "")
        print(f"[train-two-si] {si_id}: shared held-out pool "
              f"{heldout_pool}{via} ({len(heldout_pairs)} rows, "
              f"sha256 {heldout_pool_sha[:12]}…) — per-fit carve skipped",
              flush=True)
    else:
        fit_pairs, heldout_pairs = split_pairs(pairs, holdout_frac,
                                               holdout_salt)
    # Document-mode provenance inputs (recorded per object in meta_extra):
    # the sample manifest staged next to the pool, when the caller told us
    # where the pool lives (CLI paths do; programmatic callers may not —
    # a stated absence).
    # (cell = this SI; the pool FILE hash is taken from the file here —
    # pool_sha256 in this scope may be the CONTENT sha derived above for
    # programmatic callers, which is not what the manifest records.)
    sample_manifest_sha = (_sample_manifest_sha_for(pairs_path, cell=si_id)
                          if doc_mode else None)
    # bidpo's own pool gets the SAME deterministic holdout carve (same frac,
    # same salt, keyed on the prompt): held-out prompts never train, exactly
    # like the main pool's.
    bh_fit = bb_fit = None
    if bidpo_harmful_pairs is not None:
        bh_fit, _bh_heldout = split_pairs(bidpo_harmful_pairs, holdout_frac,
                                          holdout_salt)
    if bidpo_benign_pairs is not None:
        bb_fit, _bb_heldout = split_pairs(bidpo_benign_pairs, holdout_frac,
                                          holdout_salt)
    if not fit_pairs:
        raise SystemExit(
            f"[train-two-si] the holdout carve left ZERO fit pairs "
            f"(holdout_frac={holdout_frac}, {len(pairs)} pairs in) — "
            "nothing to train on")
    if holdout_frac > 0 and not heldout_pairs:
        # Loud, not fatal: a pool this small trains at the fixed epoch cap.
        print(f"[train-two-si] WARNING {si_id}: holdout_frac={holdout_frac} "
              f"carved NO held-out pairs from {len(pairs)} — early stopping "
              "is DISABLED for this run (fixed epoch budget)",
              file=sys.stderr)

    # ONE example builder for every driver-side rendering of this SI's
    # pairs (bucketing, monitor probes, held-out examples, the post-fit
    # floor probes): the doc-mode branch builds VERBATIM
    # document examples — pairs_to_distill_examples would chat-render,
    # training/probing a different corpus than the sampler budgeted.
    def _examples_from(prs):
        if doc_mode:
            from sisteer.data.pairs import docs_to_ce_examples

            return docs_to_ce_examples(tok, prs)
        return pairs_to_distill_examples(tok, prs, instruction)

    # Leading special tokens: the fit trains exactly the sequences
    # _encode_aligned builds, so the config's declared policy
    # (leading_special_tokens: "bos" | "none") is asserted on THOSE — every
    # train AND held-out example — before any optimizer step, refusing
    # loudly on a violation; an undeclared policy only OBSERVES the first
    # example (legacy lanes: behaviour untouched, but the convention lands
    # in every fit log as one greppable line either way).
    from sisteer.extraction.distill import _encode_aligned
    from sisteer.extraction.leading_tokens import check_leading_special_tokens

    _lt_policy = _leading_policy(cfg)
    if _lt_policy:
        _lt_exs = _examples_from(fit_pairs) + _examples_from(heldout_pairs or [])
        _lt = check_leading_special_tokens(
            tok, (_encode_aligned(tok, ex, max_len)[1] for ex in _lt_exs),
            _lt_policy, label=f"{si_id}: ")
        print(_lt["line"], flush=True)
    else:
        try:
            _lt = check_leading_special_tokens(
                tok, [_encode_aligned(tok, _examples_from(fit_pairs[:1])[0],
                                      max_len)[1]],
                None, label=f"{si_id}: ")
            print(_lt["line"], flush=True)
        except Exception as e:  # noqa: BLE001 — observe-only, never fatal
            print(f"[leading-tokens] {si_id}: observation skipped "
                  f"({type(e).__name__}: {e})", flush=True)

    if length_bucket:
        # Sort the MATCHED pool by rendered student length (prompt +
        # completion token count, via the token cache), so the engines'
        # in-order chunking pads every batch to near-uniform width. The
        # holdout side too — eval-only there: speed, not semantics.
        # fitmon.length_sorted OWNS the length key (one definition, shared
        # with the tuning harness and the worst-case memory probes); the
        # pairs ride their examples' sorted order.
        def _bucket(prs):
            exs = _examples_from(prs)
            pair_of = {id(e): p for e, p in zip(exs, prs)}
            return [pair_of[id(e)] for e in fitmon.length_sorted(tok, exs)]

        fit_pairs = _bucket(fit_pairs)
        if heldout_pairs:
            heldout_pairs = _bucket(heldout_pairs)

    save_dir = Path(save_dir) if save_dir is not None else None
    if save_dir is not None:
        # One objects root = one (model, revision, position leg) — F15's
        # namespacing rule, enforced at the write moment (the readers
        # resolve paths without a model segment, so a moved default would
        # orphan every existing store; the sentinel forbids mixing instead).
        if is_main:
            # The store sentinel is rank 0's to claim; a replica neither
            # writes nor re-asserts it (rank 0's refusal ends the launcher).
            assert_store_identity(save_dir, cfg, positions)
    outcome = FitOutcome(si_id=si_id)

    # Continuous heartbeat: a ~30s line naming the
    # pre-training phase, then step/total + loss + s/step + ETA, so a
    # quiet GPU never reads as a hang. Closed in the finally below.
    hb = fitmon.Heartbeat(total_steps=0) if monitors_on else None

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    def _record(arm: str, status: str, err: Exception | None = None,
                detail: dict | None = None) -> None:
        outcome.arms[arm] = ArmOutcome(
            status,
            error_type=type(err).__name__ if err is not None else None,
            # Message only, truncated — the full traceback belongs to the
            # job log (mode-0700 on the cluster), not a file under out/.
            error=str(err)[:2000] if err is not None else None,
            detail=detail)
        if err is not None:
            _log(f"[train-two-si] FAILED {si_id}__{arm}: "
                 f"{type(err).__name__}: {err}")
        _write_report(outcome, save_dir, report_path=report_path)

    # Checkpoint continuation under a raised cap (chained, plus the lr
    # multiplier) — resolved BEFORE the fingerprint memo, since the
    # continued fit's identity carries the continuation marker. Overrides
    # `epochs` for the one CE arm; every other knob stays exactly the prior
    # fit's (the chain walk from this config's plain identity inside
    # _prepare_continuation is the proof), except the lr peak when
    # --continue-lr-mult says so — and that rides in the identity.
    continuation = None
    if (continue_from is not None or epochs_cap is not None
            or float(continue_lr_mult) != 1.0):
        continuation = _prepare_continuation(
            cfg, si_id, arms, model=model, tok=tok, positions=positions,
            pool_sha256=pool_sha256, config_sha=config_sha,
            teacher_topk=teacher_topk, checkpoint_dir=checkpoint_dir,
            continue_from=continue_from, epochs_cap=epochs_cap,
            prior_cap=epochs, continue_lr_mult=continue_lr_mult)
        epochs = int(epochs_cap)

    _fp_cache: dict[str, str] = {}

    # the last epoch boundary each arm's epoch callback was INVOKED for — recorded before any save is
    # attempted, so it is independent of save success; finalize binds the index's `final` to exactly this epoch (a failed
    # last-epoch save must refuse the publish, never certify an earlier epoch as final).
    _epoch_seen: dict[str, int] = {}

    def _cell_fp(arm: str) -> str:
        """This request's fit fingerprint for one cell (memoized — the same
        value gates resume AND rides into the COMPLETE marker, so the two
        can never diverge)."""
        if arm not in _fp_cache:
            _fp_cache[arm] = cell_fit_fingerprint(
                cfg, si_id, arm, positions=positions,
                pool_sha256=pool_sha256, config_sha=config_sha,
                teacher=_teacher_leg_tag(teacher_topk, cfg, arm),
                continued_from=(continuation["marker"]
                                if continuation is not None
                                and arm == continuation["arm"] else None))
        return _fp_cache[arm]

    def _existing_path(arm: str):
        """Resume gate: a cell counts as done ONLY with a validated
        COMPLETE marker whose fit fingerprint MATCHES this request and
        whose manifest verifies. A bare object file (a torn publish, a
        pre-marker artifact) or a marker for a different fit identity
        re-runs."""
        if save_dir is None or overwrite:
            return None
        return validated_complete(save_dir, f"{si_id}__{arm}", _cell_fp(arm))

    ckpt_every_resolved = (int(checkpoint_every) if checkpoint_every
                           else DEFAULT_CHECKPOINT_EVERY)

    def _ckpt_kwargs(cell_arms: list[str]) -> dict:
        """The trainer checkpoint trio for one cell (empty dict = seam
        off). The key is the cell's fit fingerprint — for a multi-arm
        lockstep group, every member's fingerprint joined, so group
        MEMBERSHIP is part of the checkpoint identity (a resumed group
        with different members must start fresh; production cells are
        single-arm). The dir is per-cell under the caller's root, OUTSIDE
        the objects store, so audit/publish never see checkpoint files."""
        if checkpoint_dir is None or not cell_arms:
            return {}
        name = f"{si_id}__{'+'.join(cell_arms)}"
        return {
            "checkpoint_every": ckpt_every_resolved,
            "checkpoint_dir": str(Path(checkpoint_dir) / name),
            "checkpoint_key": "|".join(_cell_fp(a) for a in cell_arms),
        }

    if continuation is not None:
        # The prior checkpoint into THIS fit's transient slot, under this
        # fit's identity — the trainer's own resume does the rest.
        _adopt_continuation(continuation, _ckpt_kwargs([continuation["arm"]]))
        # inherit the predecessor's published epoch checkpoints into this fingerprint's root
        if _is_main_rank() and continuation["arm"] in CE_ARMS:
            _ck_cont = _ckpt_kwargs([continuation["arm"]])
            _root_cont = epoch_checkpoint_root(_ck_cont, continuation["arm"])
            if _root_cont is not None:
                import_predecessor_epoch_checkpoints(continuation, _root_cont, checkpoint_key=_cell_fp(continuation["arm"]),
                                                     cell_id=f"{si_id}__{continuation['arm']}")

    def _condition_contrast():
        contrast = resolve_condition_contrast(cfg)
        outcome.condition_sources = dict(contrast.meta)
        return contrast

    # BEFORE any arm can train or skip, verify every existing object's
    # RECORDED budget sidecar against the CURRENT knobs — a
    # completion_tokens change must never be silently mixed in.
    # Raises SystemExit on a hard mismatch. Gate ONLY cells that still HAVE
    # a budget record: a cell with an object but NO sidecar is one the
    # COMPLETE-marker regime already classifies incomplete (the sidecar is
    # in the publication manifest), so it RE-RUNS and re-stamps — refusing
    # there would block exactly the retrain that repairs it (F16). A
    # PRESENT record that contradicts the current knobs still refuses.
    if save_dir is not None and not overwrite:
        from sisteer.eval.defense_cell import object_path_for as _opf

        def _has_record(a: str) -> bool:
            p = _opf(save_dir, f"{si_id}__{a}")
            return (p is not None and p.exists()
                    and (p.parent / "train_budget_meta.json").exists())

        from sisteer.eval.train_budget import resolve_budget_kind

        _bk = resolve_budget_kind(cfg)
        for w in check_budget_sidecars(
                save_dir, si_id, [a for a in arms if _has_record(a)],
                max_len=max_len,
                # Legacy SI-pair path byte-identical (fallback 512); a
                # document_tokens config has no completion cap at all.
                completion_tokens=(None if _bk is not None
                                   else int(cfg.get("completion_tokens",
                                                    512))),
                budget_kind=_bk,
                heldout_pool_sha=heldout_pool_sha):
            _log(f"[train-two-si] WARNING: {w}")

    compatible, solo = partition_arms(arms)

    # -- lockstep group: one shared teacher forward per batch -----------------
    # The batch size is pinned to the FULL requested compatible list BEFORE the
    # resume filter — on a resume, the surviving members must train at the same
    # width as the from-scratch group, or their batch composition (the one
    # group-coupled input) would differ and break the bitwise guarantee.
    gbs = group_batch_size(cfg, si_id, compatible)

    def _skip_existing(arm: str, p: Path) -> None:
        """A validated COMPLETE skips the FIT; its registry emission is
        verified and replayed when missing — a
        failed replay fails the arm visibly, the publication stands."""
        try:
            ev = ensure_registry_emission(p.parent, si_id=si_id, arm=arm)
        except RegistryEmissionError as e:
            _record(arm, "failed", e,
                    detail={"registry_emission":
                            registry_emission_evidence(p.parent)})
            _log(f"[train-two-si] {si_id}__{arm}: object exists at {p} — the "
                 f"fit is skipped (resume) but its registry emission FAILED: {e}")
            return
        _record(arm, "skipped_existing",
                detail={"registry_emission": ev} if ev else None)
        _log(f"[train-two-si] {si_id}__{arm}: object exists at {p} — "
             "skipping (resume); registry emission "
             f"{(ev or {}).get('status', 'unverified')}")

    todo_compatible = []
    for arm in compatible:
        p = _existing_path(arm)
        if p is not None:
            _skip_existing(arm, p)
        else:
            todo_compatible.append(arm)

    # -- per-cell monitoring plumbing (shared by both branches) ---------------
    def _arm_scoped(key: str, arm_name: str) -> str:
        """train/loss -> train/<arm>/loss for the SHARED stream (one W&B
        run may carry several arms); the per-CELL metrics.jsonl keeps the
        contract keys unprefixed."""
        head, sep, tail = key.partition("/")
        return f"{head}/{arm_name}/{tail}" if sep else f"{arm_name}/{key}"

    def _mon_emit(arm_name: str, writer):
        # A CE arm's probe records are renamed for the SHARED stream only
        # (ce_lane_probe_record: fitmon's teacher-relative names mean
        # something else with no teacher — CE_PROBE_STREAM_KEYS); the
        # per-cell file keeps fitmon's flat vocabulary, audit_cell's
        # contract, on both lanes.
        shared = (ce_lane_probe_record if arm_name in CE_ARMS
                  else (lambda r: r))

        def emit(rec, step):
            if writer is not None:
                writer.write(rec, step)
            if log_fn is not None:
                log_fn({_arm_scoped(k, arm_name): v
                        for k, v in shared(rec).items()}, step)
        return emit

    def _close_quietly(mon, writer):
        if mon is not None:
            try:
                mon.close()  # flush any buffered records into the cell file
            except fitmon.TrainingDiverged:
                pass  # the divergence was already recorded for this arm
        if writer is not None:
            writer.close()

    heldout_examples = None
    lock_diag: dict = {}
    if todo_compatible:
        group: list[LockstepArm] = []
        try:
            # Through the ONE example builder too (identical on this path:
            # doc mode never reaches lockstep — refuse_ce_misconfiguration
            # bans non-CE arms under documents and neither CE arm is
            # lockstep-compatible — so this renders the pairs path).
            examples = _examples_from(fit_pairs)
            if heldout_pairs:
                heldout_examples = _examples_from(heldout_pairs)
        except Exception as e:  # noqa: BLE001 — isolate, record, continue
            for arm in todo_compatible:
                _record(arm, "failed", e)
        else:
            sent_kw = {"sentinel": sentinel} if sentinel is not None else {}
            for arm in todo_compatible:
                kw = dict(sent_kw) if arm in ("token_bos",
                                              "token_faithful") else {}
                # lora_rank (re-tune@32 winner key): an ARCHITECTURE knob,
                # consumed HERE at object construction — the builder
                # refuses it for any arm but lora_r1_all, so a misplaced
                # rank fails that arm loudly below.
                rank_cfg = ((cfg.get("train_kwargs") or {}).get(arm)
                            or {}).get("lora_rank")
                if rank_cfg is not None:
                    kw["lora_rank"] = int(rank_cfg)
                try:
                    # positions != "all" REFUSES per arm inside the builder
                    # when the object cannot honor a masked fit — recorded
                    # as that arm's failure, never a silent all-token fit.
                    obj = build_lockstep_object(
                        arm, model, tok, instruction, fit_pairs, layer=layer,
                        positions=positions,
                        response_header_len=response_header_len, **kw)
                except Exception as e:  # noqa: BLE001
                    _record(arm, "failed", e)
                else:
                    # Registry-default params + the per-arm optimiser-
                    # schedule opt-in (train_kwargs.<arm>, schedule keys
                    # only — see lockstep_arm_params).
                    group.append(
                        LockstepArm(arm, obj, lockstep_arm_params(cfg, arm)))
            # Per-cell monitors + metrics writers + temp cell dirs.
            tmp_dirs: dict[str, Path] = {}
            writers: dict[str, object] = {}
            mons: dict[str, object] = {}
            if group and save_dir is not None:
                for a in group:
                    tmp_dirs[a.name] = _cell_tmp_dir(save_dir,
                                                     f"{si_id}__{a.name}")
                    writers[a.name] = fitmon.CellMetricsWriter(
                        tmp_dirs[a.name] / "metrics.jsonl")
            if group and monitors_on:
                probe_src = heldout_examples or examples
                for a in group:
                    probe = fitmon.KLRatioProbe(
                        tok, probe_src, max_len=max_len, batch_size=gbs,
                        n=probe_kl_n, inject=a.obj.inject)
                    # gate_batch rides unconditionally: the monitor itself
                    # dispatches on the OBJECT's gate attributes (fitmon's
                    # never-the-arm-name rule), so a non-gate arm simply
                    # never uses it.
                    mons[a.name] = fitmon.FitMonitor(
                        a.name, model, probe, cadence,
                        _mon_emit(a.name, writers.get(a.name)),
                        gate_batch=(probe.batches[0][1]
                                    if probe.batches else None),
                        heartbeat=hb).attach(a.obj)

            def _group_log(data, step=None, commit=None, sync=None):
                # Tee trainer-native records into each arm's cell file by
                # key match (lockstep keys are arm-scoped by construction);
                # the shared stream is untouched.
                if log_fn is not None:
                    log_fn(data, step=step, commit=commit, sync=sync)
                if isinstance(data, dict) and writers:
                    for arm_name, w in writers.items():
                        pat = f"/{arm_name}/"
                        if any(pat in k or k.startswith(f"{arm_name}/")
                               for k in data):
                            w.write(data, step)

            def _epoch_cb(epoch, step, per_arm):
                """Lockstep side: a DURABLE copy of every arm's object at the end of each epoch under the
                cell's STABLE fingerprint-bound epoch root (epoch_checkpoint_root: beside the resume checkpoints,
                outside the store, so a restarted fit keeps earlier epochs), falling back to the temp cell dir
                when the checkpoint seam is off. finalize_epoch_checkpoints merges them into the publish dir.
                Main rank only. A failed save is loud, never fatal to the fit."""
                if not _is_main_rank():
                    return
                group_ck = _ckpt_kwargs([a_.name for a_ in group])
                for pa in per_arm:
                    a = pa["arm"]
                    _epoch_seen[a.name] = int(epoch) + 1
                    tmp = tmp_dirs.get(a.name)
                    if tmp is None:
                        continue
                    root = epoch_checkpoint_root(group_ck, a.name) or tmp
                    try:
                        save_epoch_checkpoint(root, a.obj, arm=a.name, si_id=si_id, epoch=int(epoch) + 1, step=int(step),
                                              history=list(pa.get("history") or []), heldout_curve=list(pa.get("heldout_curve") or []),
                                              stopped=bool(pa.get("stopped")), epochs_cap=int(epochs), checkpoint_key=_cell_fp(a.name))
                    except Exception as e:  # noqa: BLE001 — the fit goes on; the gap is named
                        log_fn(f"[epoch-ckpt] {si_id}__{a.name} epoch {int(epoch) + 1}: NOT saved ({type(e).__name__}: {e})")

            def _lockstep_probe_cb(step, arm_name, obj_, loss):
                m = mons.get(arm_name)
                if m is not None:
                    m.on_step(step, obj_, loss)

            # Per-group top-K gate: a precomputed head carries the
            # forward-KL objective at the AUDITED temperature grid only
            # (teacher_topk.AUDITED_TOPK_TEMPERATURES — the re-tune's
            # swept T {1,2,4}, where the head KL IS the tuned objective).
            # A group containing an arm outside it (token_faithful's T=10
            # — where the head's mass collapses and the loss would
            # silently differ from anything audited) routes to the online
            # teacher, said out loud. Production cells are single-arm, so
            # this gate is per-arm in practice.
            group_topk = None
            if group and topk_store is not None:
                from sisteer.extraction.teacher_topk import (
                    AUDITED_TOPK_TEMPERATURES)

                unaudited = sorted(
                    a.name for a in group
                    if a.params.loss != "soft"
                    or a.params.temperature not in AUDITED_TOPK_TEMPERATURES)
                if unaudited:
                    print(f"[teacher-topk] {si_id}: ONLINE teacher for "
                          f"group {sorted(a.name for a in group)} — arm(s) "
                          f"{unaudited} train outside the audited "
                          f"forward-KL grid T={AUDITED_TOPK_TEMPERATURES}, "
                          "which a top-K head cannot carry without "
                          "changing the tuned loss", flush=True)
                else:
                    group_topk = topk_store
            if group:
                if hb is not None:
                    hb.total = epochs * math.ceil(
                        max(len(examples), 1) / max(gbs, 1))
                    hb.phase(
                        f"lockstep group {sorted(a.name for a in group)}: "
                        f"encoding {len(examples)} fit examples, startup "
                        f"probes, then {hb.total} steps planned")
                try:
                    train_lockstep(model, tok, group, examples, epochs=epochs,
                                   batch_size=gbs, max_len=max_len, seed=seed,
                                   log_fn=(_group_log if writers else log_fn),
                                   heldout=heldout_examples,
                                   early_stop=early_spec_cfg,
                                   diagnostics=lock_diag, n_probe=probe_n,
                                   teacher_prefix_cache=teacher_prefix_cache,
                                   probe_cb=(_lockstep_probe_cb
                                             if mons else None),
                                   epoch_cb=(_epoch_cb if tmp_dirs else None),
                                   teacher_topk=group_topk,
                                   **_ckpt_kwargs([a.name for a in group]))
                except Exception as e:  # noqa: BLE001 — the group fails together
                    for a in group:
                        _record(a.name, "failed", e)
                        _close_quietly(mons.get(a.name), writers.get(a.name))
                else:
                    # Save NOW, before any solo arm runs.
                    for a in group:
                        arm_diag = dict(lock_diag.get(a.name) or {})
                        mon = mons.get(a.name)
                        writer = writers.get(a.name)
                        # A diverged arm (non-finite metric caught by the
                        # monitor) never ships: recorded failed, its temp
                        # dir left for inspection, partners unaffected.
                        if arm_diag.get("diverged"):
                            _record(a.name, "failed",
                                    fitmon.TrainingDiverged(
                                        arm_diag["diverged"]),
                                    detail=arm_diag)
                            _close_quietly(mon, writer)
                            continue
                        # The bar THIS arm was judged against, in the record
                        # (per-arm: the objective decides the bar — see
                        # arm_floor_ratio_max).
                        arm_frm = arm_floor_ratio_max(cfg, a.name,
                                                      floor_ratio_max)
                        arm_diag["floor_ratio_max"] = arm_frm
                        # The teacher leg the TRAINING steps really ran:
                        # train_lockstep records it per arm ("precomputed_
                        # topk" only when a store was passed AND its rows
                        # bound; a row-binding fallback trains online and
                        # records online) — the meta must never claim a
                        # leg that did not run.
                        meta = {"teacher_forward": arm_diag.get(
                                    "teacher_forward", "shared_lockstep"),
                                "teacher_topk": (
                                    topk_meta_rec
                                    if arm_diag.get("teacher_forward")
                                    == "precomputed_topk" else None),
                                "batch_size": gbs,
                                "positions": positions,
                                "response_header_len": response_header_len,
                                # The teacher-forward leg the group REALLY
                                # ran (the config flag; train_lockstep
                                # received it directly, no retry to drop
                                # it) — fp-tolerance provenance for any
                                # verdict near a guard boundary.
                                "teacher_prefix_cache": teacher_prefix_cache,
                                # The optimiser-schedule opt-in this arm
                                # REALLY trained under (nulls = the
                                # historical fixed-lr loop).
                                "lr_schedule": a.params.lr_schedule,
                                "warmup_steps": a.params.warmup_steps,
                                "warmup_frac": a.params.warmup_frac,
                                "resolved_warmup_steps": arm_diag.get(
                                    "resolved_warmup_steps"),
                                "max_grad_norm": a.params.max_grad_norm,
                                # Gradient accumulation (re-tune@32): the
                                # effective batch this arm REALLY trained
                                # at — clip fires once per optimizer step
                                # on the accumulated gradient.
                                "accum_steps": a.params.accum_steps,
                                "effective_batch":
                                    gbs * int(a.params.accum_steps),
                                # The re-tune's other per-arm winner knobs,
                                # as the arm REALLY trained: T/wd from its
                                # params; rank from the config the builder
                                # consumed (None = the constructor default,
                                # rank 1 — also persisted in the blob).
                                "temperature": a.params.temperature,
                                "weight_decay": a.params.weight_decay,
                                "lora_rank": ((cfg.get("train_kwargs") or
                                               {}).get(a.name)
                                              or {}).get("lora_rank"),
                                # Batch-composition provenance (gpu-util
                                # directive): bucketed batches change
                                # gradient grouping vs an unsorted fit.
                                "length_bucketed": length_bucket,
                                "monitors": monitors_on and mon is not None,
                                "fit_record": arm_diag}
                        try:
                            if mon is not None:
                                # Final dual-KL — the audit requires it in
                                # the per-cell record for every cell.
                                fin = mon.final(a.obj)
                                arm_diag.update(
                                    {k.replace("final/", "final_"): v
                                     for k, v in fin.items()})
                            # Key presence, not dict truthiness: arm_diag now
                            # always carries the bar itself, but the guard
                            # can only rule where a probe actually ran.
                            if (arm_frm is not None
                                    and "final_probe_loss" in arm_diag):
                                # End-of-training no-op-floor guard: raises
                                # FloorRatioError per arm (the 966-step
                                # 4.82-vs-0.148 silent plateau) — the
                                # group's other objects still save.
                                floor_ratio_check(
                                    arm_diag["final_probe_loss"],
                                    arm_diag["noop_floor"],
                                    ratio_max=arm_frm,
                                    tag=f"{si_id}__{a.name}", log_fn=log_fn)
                            if isinstance(a.obj, PSRObject):
                                # Dead-gate guard: the lockstep loop trains
                                # PSR arms without train_psr, so the guard
                                # embedded there never runs on this path —
                                # check here, BEFORE the save, so a PSR
                                # whose gates trained shut records a failed
                                # arm and writes NO artifact (psr_kl_all
                                # once trained 966 steps ~30x worse than a
                                # no-op, silently). The raise is per-arm:
                                # the group's other objects still save.
                                # min_probe: the training set IS the probe
                                # pool here — exhaustive over every example
                                # the fit trained on, so a set smaller than
                                # GUARD_PROBE_FLOOR is complete, not thin.
                                check_psr_gates(
                                    model, tok, obj=a.obj, examples=examples,
                                    max_len=max_len,
                                    tag=f"{si_id}__{a.name}",
                                    min_probe=min(GUARD_PROBE_FLOOR,
                                                  len(examples)))
                            tmp = tmp_dirs.get(a.name)
                            _save_object(a.obj, save_dir, si_id, a.name, cfg,
                                         max_len, meta, model=model, tok=tok,
                                         out_dir=tmp)
                            if tmp is not None:
                                # ATOMIC PUBLISH: audit the temp
                                # cell dir, rename into place, COMPLETE
                                # marker LAST. Any refusal records THIS
                                # arm failed; the temp dir stays for
                                # inspection.
                                stepped = (mon is not None
                                           and mon.steps_seen > 0)
                                _close_quietly(mon, writer)
                                mon = writer = None
                                # Per-step/cadence expectations only where
                                # the monitor OBSERVED steps — a stubbed
                                # engine (routing tests) drives none, and
                                # the audit must not refuse what nothing
                                # was supposed to record.
                                # the fit SELECTION sidecar:
                                # into the temp dir before the audit +
                                # publish; absent block = no file
                                write_fit_selection(tmp, cfg, si_id, a.name, artefact_path=Path(tmp) / "object.pt")
                                if _is_main_rank():
                                    finalize_epoch_checkpoints(
                                        epoch_checkpoint_root(_ckpt_kwargs([a_.name for a_ in group]), a.name), tmp,
                                        checkpoint_key=_cell_fp(a.name), cell_id=f"{si_id}__{a.name}", required=True,
                                        expected_final_epoch=_epoch_seen.get(a.name))
                                audit = audit_cell(
                                    tmp, arm=a.name,
                                    cadence=(cadence if stepped else None),
                                    monitors_on=(monitors_on
                                                 and a.name in mons),
                                    require_step_loss=stepped)
                                publish_cell(save_dir, f"{si_id}__{a.name}",
                                             tmp, si_id=si_id, arm=a.name,
                                             positions=positions,
                                             audit=audit,
                                             fingerprint=_cell_fp(a.name))
                        except Exception as e:  # noqa: BLE001
                            _record(a.name, "failed", e, detail=arm_diag)
                        else:
                            outcome.objects[a.name] = a.obj
                            _record(a.name, "trained", detail=arm_diag)
                        finally:
                            _close_quietly(mon, writer)

    # -- solo remainder: ordinary registry trainer, same process -------------
    # (memoized probe inputs — shared by every solo arm's floor guard)
    _ex_cache: dict = {}

    def _probe_examples(which: str):
        if which not in _ex_cache:
            src = fit_pairs if which == "fit" else heldout_pairs
            _ex_cache[which] = _examples_from(src)
        return _ex_cache[which]

    per_arm_tk = cfg.get("train_kwargs") or {}
    for arm in solo:
        p = _existing_path(arm)
        if p is not None:
            if is_main:
                _skip_existing(arm, p)
            else:
                # a replica of a resumed-and-skipped cell: nothing to fit,
                # nothing to emit (rank 0 verifies / replays the record)
                outcome.arms[arm] = ArmOutcome("skipped_existing")
            continue
        mon = writer = tmp = None
        # Bound BEFORE the try so the except can attach whatever fit
        # record exists at failure time (a pre-fit failure attaches none —
        # the `arm_diag or None` in the except needs the name bound).
        # es_diag too: the trainer fills it in place, so a POST-training
        # refusal (the CE floor, the PSR dead-gate guard) still leaves the
        # stopping record + per-layer gate activity attachable to the
        # failed arm's report (gate activity recorded for every
        # PSR fit, guard pass or FAIL).
        arm_diag: dict = {}
        es_diag: dict = {}
        try:
            bs = resolve_batch_size(cfg, si_id, arm)
            # Per-cell metrics + monitor (same plumbing as the lockstep
            # branch; the solo trainer's own telemetry tees into the cell
            # file, the monitor adds cadence-gated dual-KL probes through
            # the trainer's probe_cb seam where one exists).
            if save_dir is not None and is_main:
                tmp = _cell_tmp_dir(save_dir, f"{si_id}__{arm}")
                writer = fitmon.CellMetricsWriter(tmp / "metrics.jsonl")
            if monitors_on:
                probe_src = _probe_examples(
                    "heldout" if heldout_pairs else "fit")
                probe = fitmon.KLRatioProbe(
                    tok, probe_src, max_len=max_len,
                    batch_size=min(int(bs), max(len(probe_src), 1)),
                    n=probe_kl_n)
                # gate_batch rides unconditionally — the monitor dispatches
                # on the object's own gate attributes (fitmon's
                # never-the-arm-name rule).
                mon = fitmon.FitMonitor(
                    arm, model, probe, cadence, _mon_emit(arm, writer),
                    gate_batch=(probe.batches[0][1]
                                if probe.batches else None),
                    heartbeat=hb,
                    # The residual-ratio hard-stop rules on
                    # KL(prompted||steered) / KL(prompted||unsteered) — a
                    # TEACHER-relative objective. Doc-mode examples have
                    # prompt_with_si == prompt_without, so the denominator
                    # is exactly 0 (clamped to 1e-6) and ANY training
                    # movement reads as a blow-up: the guard's semantics
                    # invert. Off for the CE arms; the probes themselves
                    # stay (residual_kl there = displacement from base —
                    # named so on the shared stream by _mon_emit's
                    # ce_lane_probe_record) and the NaN aborts still hold.
                    ratio_guard=(False if arm in CE_ARMS else None))

            def _solo_log(data, step=None, commit=None, sync=None,
                          _w=writer, _arm=arm):
                if log_fn is not None:
                    log_fn(data, step=step, commit=commit, sync=sync)
                if isinstance(data, dict):
                    if _w is not None:
                        _w.write(data, step)
                    if monitors_on:
                        # The solo abort path: ANY non-finite
                        # numeric the trainer logs unwinds it mid-fit and
                        # records THIS arm failed.
                        fitmon.assert_finite(data, step=step, context=_arm)
                    if hb is not None:
                        # Summary-suffix exclusion shared with the pilot's
                        # step classifier (pod_acceptance_probe): heldout /
                        # step0 / final-probe summaries also end in _loss
                        # but are not optimizer steps and must not tick the
                        # heartbeat. The broader "/loss" match stays — the
                        # unified is_step_line requires an "_loss" suffix
                        # and would drop bidpo/loss + reft_r1/loss, real
                        # per-step keys this heartbeat must keep seeing.
                        # The CE trainer's per-step token count rides the
                        # same record (<arm>/loss_tokens) so the heartbeat
                        # can print tok/s next to s/step (gpu-util skill).
                        toks = next((int(v) for k, v in data.items()
                                     if k.endswith("/loss_tokens")
                                     and isinstance(v, (int, float))), None)
                        for k, v in data.items():
                            if (isinstance(v, (int, float))
                                    and (k.endswith("_loss")
                                         or k.endswith("/loss"))
                                    and not k.endswith(_NON_STEP_SUFFIXES)):
                                hb.step(step or 0, float(v), None,
                                        tokens=toks)
                                break
            candidate_kw = {"epochs": epochs, "batch_size": bs,
                            "max_len": max_len, "seed": seed,
                            # Teacher prefix caching, config-gated (see the
                            # read above). Deliberately NOT protected: a
                            # trainer with no teacher-side forward (bidpo,
                            # loreft, lofit, the Class B closed forms) has
                            # no signature slot for it and the
                            # TypeError-retry drops it — correctly, since
                            # there is nothing to cache there. Every
                            # trainer that DOES run teacher legs declares
                            # the kwarg, so for those it always survives
                            # and the config's false is a working escape
                            # hatch.
                            "teacher_prefix_cache": teacher_prefix_cache}
            if arm in ("token_bos", "token_faithful") and sentinel is not None:
                candidate_kw["sentinel"] = sentinel
            if positions != "all":
                # A masked (decode-only) fit request must never be silently
                # dropped by the TypeError-retry: PROTECTED, so a solo
                # trainer that cannot honor it fails THIS arm loudly.
                candidate_kw["positions"] = positions
                candidate_kw["response_header_len"] = response_header_len
            # Per-arm config kwargs (cli.py's train_kwargs.<arm> convention):
            # protected, so a typo'd config key fails loudly instead of being
            # silently dropped by the TypeError-retry.
            # Per-arm config kwargs (cli.py's train_kwargs.<arm> convention) minus the DRIVER's own knobs (DRIVER_ONLY_ARM_KEYS: the
            # floor bars, gradient checkpointing, the fit SELECTION block) — ONE construction, arm_trainer_kwargs, shared with the
            # fit-path test. Protected below, so a typo'd config key fails loudly instead of being silently dropped by the
            # TypeError-retry; the promoted architecture keys (lora_modules / per_expert_rank / init_scheme) reach the trainer here.
            arm_kw = arm_trainer_kwargs(per_arm_tk, arm)
            # The SI-teacher floor bar is meaningless for a CE arm (no
            # teacher exists): honestly null, recorded as such — the
            # CE-native ce_floor_ratio_max below replaces it, never
            # silently disables it.
            arm_frm = (None if arm in CE_ARMS
                       else arm_floor_ratio_max(cfg, arm, floor_ratio_max))
            if arm in CE_ARMS and "accum_steps" in tk:
                # Per-cell accumulation cap (train_kwargs_by_si /
                # train_kwargs.default) threaded to the CE trainer; the
                # per-arm block still wins via the update below.
                candidate_kw["accum_steps"] = int(tk["accum_steps"])
            protected = frozenset(arm_kw)
            if positions != "all":
                protected = protected | {"positions", "response_header_len"}
            candidate_kw.update(arm_kw)
            if mon is not None:
                # The mid-fit probe seam, where the trainer exposes one
                # (probe_cb(step, obj) + probe_every — train_bidpo's shape,
                # and train_psr's since the F7 seam landed, so BOTH DPO and
                # solo psr cells now carry mid-fit dual-KL like the
                # lockstep arms). The MONITOR owns the real cadence
                # (ProbeCadence's dense->sparse schedule) and self-gates;
                # the gcd is the coarsest trainer cadence that still hits
                # every due step (every due step is a multiple of
                # dense_every or probe_every — tune_arm's trial loop uses
                # the same trick), so the trainer skips building its
                # transient probe object on the steps the monitor would
                # drop anyway. PROTECTED wherever the trainer declares the
                # seam (and unconditionally for the psr arms, whose seam is
                # a shipped contract): a trainer that regresses out of it
                # must fail the arm LOUDLY, never silently degrade back to
                # end-of-fit-only probes. A trainer that genuinely has no
                # seam still drops both via the TypeError-retry and the
                # absence is recorded below, never silent.
                candidate_kw["probe_cb"] = (
                    lambda step, obj, _m=mon: _m.on_probe(step, obj))
                candidate_kw["probe_every"] = math.gcd(
                    cadence.dense_every, cadence.probe_every)
                if _accepts_probe_cb(ARMS[arm].train) or arm.startswith("psr"):
                    protected = protected | {"probe_cb", "probe_every"}
            if arm in CE_ARMS and tmp is not None:
                # Solo side (the ReLU/CE arms train solo — train_object_ce): the epoch-boundary hook saves a
                # durable copy of the object at each epoch end under the stable fingerprint-bound root (or the
                # temp cell dir when the checkpoint seam is off); finalize_epoch_checkpoints merges them at publish.
                def _solo_epoch_cb(epoch, step, obj_, heldout_curve, stopping, _arm=arm, _tmp=tmp):
                    if not _is_main_rank():
                        return
                    _epoch_seen[_arm] = int(epoch) + 1
                    root = epoch_checkpoint_root(_ckpt_kwargs([_arm]), _arm) or _tmp
                    try:
                        save_epoch_checkpoint(root, obj_, arm=_arm, si_id=si_id, epoch=int(epoch) + 1, step=int(step),
                                              history=[], heldout_curve=list(heldout_curve or []), stopped=bool(stopping),
                                              epochs_cap=int(candidate_kw.get("epochs", epochs)), checkpoint_key=_cell_fp(_arm))
                    except Exception as e:  # noqa: BLE001
                        log_fn(f"[epoch-ckpt] {si_id}__{_arm} epoch {int(epoch) + 1}: NOT saved ({type(e).__name__}: {e})")
                candidate_kw["epoch_cb"] = _solo_epoch_cb
                protected = protected | {"epoch_cb"}
            if arm in ("psr", "psr_mse_all") or arm in CE_ARMS:
                # The mid-fit checkpoint seam, solo side: train_psr's mse
                # loop and train_object_ce (the CE arms) are the solo
                # trainers carrying one (the KL-hard psr arms refuse the
                # kwargs by design — their optimizer loop lives in
                # train_object_kl — and the other solo trainers have no
                # seam), so the trio goes to those arms only, PROTECTED: a
                # trainer that regressed out of the seam must fail the arm
                # loudly, never silently fit a long cell with no
                # durability.
                ck = _ckpt_kwargs([arm])
                if ck:
                    candidate_kw.update(ck)
                    protected = protected | set(ck)
            if arm in CE_ARMS and not heldout_pairs:
                # Early stopping + the CE floor are MANDATORY for the CE
                # arms (the floor's denominator is the held-out pool's
                # bare-base CE) — a CE fit with no held-out side has
                # neither, refuse THIS arm loudly.
                raise ValueError(
                    f"{arm}: the CE arms require a held-out pool "
                    "(train_kwargs.default.heldout_pool, or a "
                    "holdout_frac carve that yields pairs) — per-epoch "
                    "held-out early stopping and the ce_floor_ratio_max "
                    "guard are mandatory, and this fit has no held-out "
                    "side to rule on.")
            if ((arm in ("psr", "psr_mse_all") or arm in CE_ARMS)
                    and heldout_pairs):
                # Per-epoch held-out early stopping, solo side: the psr
                # mse arms were the LAST cells running a fixed epoch
                # budget to the cap while every other arm stopped on a
                # held-out curve. They get the SAME 10% carve the
                # lockstep group rules on (split_pairs — the held-out
                # pairs never reach any trainer's fit set) and the
                # config's own spec (early_stop_patience/min_epochs/
                # min_delta); train_psr fills es_diag with the stopping
                # record. The CE arms get the same mandatory seams —
                # for them the held-out side is the SHARED pool's
                # doc-mode examples, and the raise above already made the
                # heldout_pairs condition unconditional. PROTECTED: a
                # trainer that regressed out of the seam must fail the arm
                # loudly, never silently burn the full budget while the
                # record claims early stopping.
                candidate_kw["heldout"] = _probe_examples("heldout")
                candidate_kw["early_stop"] = early_spec_cfg
                candidate_kw["diagnostics"] = es_diag
                protected = protected | {"heldout", "early_stop",
                                         "diagnostics"}
            meta = {"teacher_forward": "solo",
                    # The data-parallel width the fit runs at: N ranks
                    # each accumulate accum_steps/N of every optimizer
                    # step's micro-batches and all-reduce the adapter
                    # gradient; 1 = the single-process path. The effective
                    # batch below is the same number either way.
                    "fit_width": fit_width,
                    "fit_parallel_mode": ("ddp_group_sharded" if fit_width > 1
                                          else "none"),
                    "batch_size": candidate_kw.get("batch_size", bs),
                    # EXPLICIT: gradient accumulation
                    # changes the effective batch (2026-08-25): a trainer
                    # taking accum_steps optimizes at batch_size x
                    # accum_steps rows per step, and that product is the
                    # width a batch-size claim must mean.
                    "effective_batch": (candidate_kw.get("batch_size", bs)
                                        * int(candidate_kw.get(
                                            "accum_steps", 1) or 1)),
                    "accum_steps": int(candidate_kw.get(
                        "accum_steps", 1) or 1),
                    "lr": candidate_kw.get("lr", "trainer_default"),
                    # The seed the fit ran under (train_kwargs.default.seed,
                    # default 0): the trainer's epoch-shuffle seed and, on
                    # the LoRA arm, the A-factor init seed (2026-09-15;
                    # every earlier sidecar read the seed back as absent).
                    "seed": int(candidate_kw.get("seed", 0) or 0),
                    "positions": positions,
                    "length_bucketed": (length_bucket
                                        and not (arm in DPO_PREF_ARMS
                                                 and bh_fit is not None)),
                    "monitors": monitors_on,
                    "response_header_len": response_header_len}
            if arm in CE_ARMS:
                # Which CE loss implementation trained the object (a
                # fingerprinted identity dimension) — the sidecar says it,
                # so no reader has to infer it from the fingerprint.
                meta["loss_impl"] = resolved_loss_impl(cfg, arm)
            if doc_mode:
                # Document-mode provenance (into meta_extra →
                # train_budget_meta.json): template_hash still records the
                # SHIPPED chat template honestly, and this flag says it was
                # never applied; the sample manifest + shared held-out pool
                # are the fit's data identity (null = a stated absence).
                meta.update({
                    "data_mode": DOC_DATA_MODE,
                    "chat_template_bypassed": True,
                    "sample_manifest_sha": sample_manifest_sha,
                    "heldout_pool_sha": heldout_pool_sha,
                })
            if _accepts_condition_prompts(ARMS[arm].train):
                # The category condition: harmful = should-fire (pos),
                # benign = should-not-fire (neg). Resolution failure (offline
                # cache miss, leakage-guard refusal) fails THIS arm only.
                # EQUAL-SIZED sides: the consumer's pairwise condition PCA
                # (CAST's shape — pos[i] - neg[i], refuses unequal sets)
                # needs equal sides, while the contrast is asymmetric — so
                # truncate both to the common min. The lists are already
                # deterministically sha-sorted (_det_sample), so the head IS
                # a deterministic subsample; the counts are recorded below.
                # NOTE this equalization gates the CONDITION sets of a
                # condition-taking arm only — it never touches the Class B
                # content fit, whose sides are deliberately asymmetric
                # (fit_class_b.py + build_content_contrast's min_prompts
                # floor own that path).
                contrast = _condition_contrast()
                n_cond = min(len(contrast.harmful), len(contrast.benign))
                candidate_kw["condition_pos_prompts"] = \
                    list(contrast.harmful)[:n_cond]
                candidate_kw["condition_neg_prompts"] = \
                    list(contrast.benign)[:n_cond]
                protected = protected | {"condition_pos_prompts",
                                         "condition_neg_prompts"}
                meta["condition_sources"] = dict(
                    contrast.meta,
                    n_condition_per_side=n_cond,
                    equalized_from={"harmful": len(contrast.harmful),
                                    "benign": len(contrast.benign)})
            # bidpo trains on a genuine-compliance rejected side, from one
            # of three sources: its OWN pool when pool.bidpo_benign is
            # enabled (per-SI thwart harmful side + the REVERSED-polarity
            # benign side), else the anti-safety negative (pool.anti_neg)
            # or the legacy strict subset over the matched pool. Every
            # other arm trains on the full pool. Raises loudly (caught
            # below, THIS arm recorded failed) if bidpo has no valid
            # rejected side or an empty/mislabelled/one-sided own pool. The
            # chosen sources + realised counts are recorded per object (fit
            # report detail + train_budget_meta.json sidecar), never a
            # silent substitution — counts only, never any text.
            if (arm in DPO_PREF_ARMS and bb_cfg["enabled"]
                    and (bh_fit is None or bb_fit is None)):
                raise ValueError(
                    "pool.bidpo_benign.enabled is true but bidpo's own pool "
                    "was not supplied to train_si_objects — run emit-pairs "
                    "(which writes bidpo_harmful_<si>.jsonl + "
                    "bidpo_benign_<si>.jsonl beside the pairs file) and pass "
                    "both sides through; refusing to silently train a "
                    "matched-pool bidpo wearing the section-13 "
                    "configuration.")
            arm_pairs = pairs_for_arm(
                arm, fit_pairs, anti_neg_enabled=anti_neg_enabled,
                bidpo_harmful_pairs=bh_fit, bidpo_benign_pairs=bb_fit,
                bidpo_benign_cap=bb_cfg["max_benign_per_harmful"])
            neg_meta = None
            benign_meta = None
            if arm in DPO_PREF_ARMS:
                n_benign_sel = sum(
                    1 for p in arm_pairs
                    if getattr(p, "requirement_a", None)
                    == _bp.BIDPO_BENIGN_REQ_A)
                n_harm_sel = len(arm_pairs) - n_benign_sel
                own_pool = bh_fit is not None
                n_harmful = (len(bh_fit) if own_pool else
                             sum(1 for p in fit_pairs if _is_harmful_req(p)))
                neg_meta = {
                    "source": ("anti_safety_compliance_per_si_thwart"
                               if own_pool else
                               "anti_safety_compliance" if anti_neg_enabled
                               else "undefended_strict_subset"),
                    "anti_neg_enabled": anti_neg_enabled,
                    "n_pairs": n_harm_sel,
                    "n_harmful_available": n_harmful,
                    "n_dropped_no_anti_neg": (
                        None if own_pool else
                        n_harmful - n_harm_sel if anti_neg_enabled else None),
                }
                meta["negative_source"] = neg_meta
                if own_pool:
                    # The section-13 confound record: realised counts AND the
                    # realised harmful:benign RATIO here, the per-SI frozen
                    # counts + projected ratios + label setting from the emit
                    # sidecar (bidpo_benign_meta) — so each object's
                    # meta.json carries everything a cross-SI read must be
                    # stated with (the three objects train at DIFFERENT
                    # ratios, trivially misread as an SI effect). Counts
                    # only, never text.
                    benign_meta = {
                        "enabled": True,
                        "harmful_selection": "per_si_thwart_with_anti_neg",
                        "benign_polarity": "reversed_chosen_is_undefended_"
                                           "compliance",
                        "parity": "per_si_unequal_no_parity_subsample",
                        "max_benign_per_harmful":
                            bb_cfg["max_benign_per_harmful"],
                        "n_harmful_available_post_holdout": len(bh_fit),
                        "n_benign_available_post_holdout": len(bb_fit),
                        "n_harmful_pairs": n_harm_sel,
                        "n_benign_pairs": n_benign_sel,
                        "n_total_pairs": len(arm_pairs),
                        "realised_ratio_harmful_to_benign": round(
                            n_harm_sel / max(n_benign_sel, 1), 4),
                        "realised_benign_fraction": round(
                            n_benign_sel / len(arm_pairs), 4),
                        **(bidpo_benign_meta or {}),
                    }
                    meta["bidpo_benign"] = benign_meta
            if ("warmup_frac" in candidate_kw
                    and arm not in DPO_PREF_ARMS):
                # HPO winners carry warmup as a FRACTION; a solo trainer
                # without its own warmup_frac seam (train_psr's mse loop
                # takes warmup_steps) gets the fraction RESOLVED here,
                # against THIS fit's own OPTIMIZER-step budget — the same
                # total-steps arithmetic the trainer runs (epochs x
                # ceil(ceil(n/batch) / accum_steps): accumulation makes an
                # optimizer step span accum_steps micro-batches, and warmup
                # must ramp over optimizer steps); never a copied step
                # count from a subsampled trial. bidpo keeps the fraction:
                # its trainer resolves it natively against its own
                # optimizer-step count (accumulation changes it).
                from sisteer.extraction.lockstep import resolve_warmup_steps

                frac = float(candidate_kw.pop("warmup_frac"))
                eff_epochs = int(candidate_kw.get("epochs", epochs))
                eff_bs = int(candidate_kw.get("batch_size", bs) or bs)
                eff_accum = int(candidate_kw.get("accum_steps", 1) or 1)
                total = eff_epochs * math.ceil(math.ceil(
                    max(len(arm_pairs), 1) / max(eff_bs, 1))
                    / max(eff_accum, 1))
                resolved = resolve_warmup_steps(frac, total)
                candidate_kw["warmup_steps"] = resolved
                protected = (protected - {"warmup_frac"}) | {"warmup_steps"}
                meta["warmup_frac"] = frac
                meta["resolved_warmup_steps"] = resolved
            if continuation is not None and arm == continuation["arm"]:
                # The warm-restart size at the resume step, on the record
                # before a single continued step runs — and the stopper
                # rule the continuation runs under, PROTECTED so a
                # seamless trainer fails the arm loudly.
                _continuation_schedule_report(continuation, candidate_kw,
                                              meta)
                protected = _continuation_stopper_rule(
                    continuation, candidate_kw, protected, meta)
            if hb is not None:
                hb.phase(f"training {si_id}__{arm} solo "
                         f"({len(arm_pairs)} pairs, bs={bs})")
            obj, used_kw = _safe_registry_train(
                ARMS[arm].train, model, tok, instruction, arm_pairs,
                layer=layer, candidate_kw=candidate_kw, log_fn=_solo_log,
                protected=protected)
            if continuation is not None and arm == continuation["arm"]:
                # Fail loudly (no publication) unless the trainer itself
                # says it resumed from the adopted (or a later) step.
                _assert_continuation_resumed(continuation, es_diag)
            if not is_main:
                # A data-parallel replica: the trainer asserted the
                # replicas bit-equal before returning; rank 0 owns the
                # guards, the save, the sidecars, COMPLETE and the registry
                # emission. A guard refusal there exits rank 0 nonzero,
                # which is what the launcher and the sbatch read.
                raise _ReplicaTrained()
            if mon is not None and "probe_cb" not in used_kw:
                # The trainer exposes no mid-fit probe seam: say so in the
                # record (and the log) instead of publishing a cell whose
                # missing probe cadence looks like a monitoring failure.
                # Since the F7 seam this is only ever a genuinely seamless
                # trainer — the psr and DPO arms' seams are PROTECTED
                # above, so a dropped probe_cb there fails the arm loudly
                # and never reaches this fallback.
                print(f"[train-two-si] {si_id}__{arm}: trainer exposes no "
                      "probe_cb seam — mid-fit dual-KL probes unavailable; "
                      "final dual-KL measured post-fit", file=sys.stderr)
            # The width the fit REALLY ran: the TypeError-retry above may
            # have dropped batch_size for a trainer that batches internally,
            # and recording the candidate then would claim a width that
            # never ran. null = "not driver-controlled", a stated absence.
            meta["batch_size"] = used_kw.get("batch_size")
            # Same candidate-vs-survivor rule for the teacher-prefix-cache
            # leg: record the flag the trainer REALLY received. null = the
            # TypeError-retry dropped it, i.e. this trainer declares no
            # teacher-side forward, so the flag had nothing to change —
            # a stated absence, never a claimed leg that did not run.
            meta["teacher_prefix_cache"] = used_kw.get("teacher_prefix_cache")
            # End-of-training no-op-floor guard, uniform across solo arms:
            # the soft-KL probe (lockstep.arm_probe_losses) measures whether
            # the trained object moves the unprompted student TOWARD the
            # SI-prompted teacher or away — the programme's canonical
            # degeneration signal — MEAN over probe_n fit examples, object
            # active vs inactive, with the per-batch sd/sem recorded so the
            # ratio's resolution is part of the record (it used to be batch
            # 0 at the arm's training batch size: a 2-4 sample KL estimate,
            # the wave-wide failure). Most solo trainers own their epoch
            # loops with no held-out seam (recorded honestly below); the
            # psr mse arms carry one since the early-stop seam above, and
            # their stopping record lands lockstep-shaped. The guard still
            # refuses a degenerate arm either way.
            arm_diag = {
                # The bar THIS arm is judged against — recorded even when
                # the guard is off (null), so the audit always finds the
                # verdict field.
                "floor_ratio_max": arm_frm,
                "midfit_probes": (
                    "trainer_probe_cb" if (mon is not None
                                           and "probe_cb" in used_kw)
                    else "unavailable_no_trainer_seam" if mon is not None
                    else "monitors_off"),
                "early_stop": (
                    # The seam that REALLY ran (candidate-vs-survivor):
                    # heldout is PROTECTED for the psr mse arms, so its
                    # presence in used_kw is the trainer's own word.
                    {"supported": True,
                     "heldout_n": len(heldout_pairs),
                     "patience": early_spec_cfg.patience,
                     "min_epochs": early_spec_cfg.min_epochs,
                     "min_delta": early_spec_cfg.min_delta}
                    if "heldout" in used_kw else
                    {"supported": False,
                     "note": "solo registry trainer runs the fixed epoch "
                             "budget; per-epoch held-out stopping is wired "
                             "for the lockstep group and the psr mse arms"}),
                # Same candidate-vs-survivor rule as batch_size above:
                # the TypeError-retry may have dropped `epochs` for a
                # trainer without **kw, and recording the candidate then
                # claims an epoch budget the fit never ran. null = "not
                # driver-controlled" (the trainer owns its own loop), a
                # stated absence rather than a plausible wrong number.
                "epochs_run": used_kw.get("epochs"),
            }
            if "heldout" in used_kw:
                # The trainer's own stopping record, at the SAME top-level
                # keys the lockstep arms publish (best_epoch/heldout_curve/
                # stopped_early — what the sidecar/report tests grep), and
                # epochs_run becomes the count the fit REALLY ran rather
                # than the cap it was handed.
                arm_diag.update({k: es_diag.get(k)
                                 for k in ("heldout_curve", "best_epoch",
                                           "best_heldout_loss",
                                           "stopped_early")})
                if es_diag.get("epochs_run") is not None:
                    arm_diag["epochs_run"] = es_diag["epochs_run"]
            if arm in CE_ARMS:
                # The CE trainer's extra record: the bare-base held-out
                # denominator, the realized effective batch (tiny cells
                # degrade gracefully via the epoch-boundary flush — say
                # what actually ran) WITH its resume marker, and per-layer
                # gate activity for every PSR fit.
                for k in ("heldout_noop_ce", "best_heldout_ce",
                          "realized_eff_batch", "steps_per_epoch",
                          "optimizer_steps_run", "gate_liveness",
                          "replayed_partial_epochs",
                          # The resume marker rides BESIDE the vector: None
                          # = a fresh process (explicit, never an absent
                          # key), an int = the optimizer step a checkpoint
                          # restored this fit to — the vector is then the
                          # resumed epoch's SUFFIX and the realized-batch
                          # validator applies the floor, not the exact
                          # first-epoch reproduction: every fit in the lane
                          # is preemption-tolerant, and a healthy resumed
                          # fit was failing its own acceptance check
                          # without this.
                          "resumed_from_step"):
                    if k in es_diag:
                        arm_diag[k] = es_diag[k]
                if "realized_eff_batch" in es_diag:
                    # The realized-batch verdict, filed by the driver
                    # beside the vector it publishes: n_docs = the
                    # documents THIS arm trained on, bs / accum / seed =
                    # the kwargs the trainer really received (trainer
                    # defaults where absent). Recorded and printed, never
                    # a publication refusal — the validator judges the
                    # record, not the fit.
                    from sisteer.extraction import realized_batch

                    chk = realized_batch.validate_fit_record(
                        arm_diag, n_docs=len(arm_pairs),
                        bs=int(used_kw.get("batch_size") or bs),
                        accum=int(used_kw.get("accum_steps") or 1),
                        seed=int(used_kw.get("seed") or 0))
                    arm_diag["realized_batch_check"] = chk.as_dict()
                    if not chk.ok:
                        print(f"[train-two-si] {si_id}__{arm}: WARNING the "
                              "realized effective-batch vector FAILS "
                              "validation (recorded in fit_record."
                              "realized_batch_check): "
                              + "; ".join(chk.problems),
                              file=sys.stderr, flush=True)
            # The trainer's own numeric self-report (steps_run, min_loss,
            # betas/eps, effective_batch, ...) — the artifact metadata is
            # normally the only carrier, and a floor-guard REFUSAL writes
            # no artifact, which would strand exactly the numbers a sweep
            # harvests from its failing trials. Scalars only, never text.
            # The vector arms return a VectorObject ADAPTER (metadata and
            # vectors live on the wrapped .sv); trainers whose objects
            # carry neither record nothing — a stated absence.
            src = obj if isinstance(getattr(obj, "metadata", None), dict) \
                else getattr(obj, "sv", None)
            tmeta = getattr(src, "metadata", None)
            if isinstance(tmeta, dict):
                arm_diag["trainer_metadata"] = {
                    k: v for k, v in tmeta.items()
                    if isinstance(v, (int, float)) or v is None}
                vecs = getattr(src, "vectors", None)
                if isinstance(vecs, dict) and vecs:
                    arm_diag["trainer_metadata"]["final_vec_norm"] = float(
                        sum(v.float().norm() ** 2
                            for v in vecs.values()) ** 0.5)
            if neg_meta is not None:
                # bidpo's rejected-side provenance in the fit report too, so a
                # reviewer sees which negative source each object used and its
                # realised counts without opening the sidecar.
                arm_diag["negative_source"] = neg_meta
            if benign_meta is not None:
                # ...and the benign-reversed side's counts (per-SI included),
                # so no bidpo cross-SI read can miss the Form B confound.
                arm_diag["bidpo_benign"] = benign_meta
            if arm_frm is not None:
                # batch_size here is ONLY the memory width of one no-grad
                # forward (the arm's measured-safe training width); the
                # SAMPLE SIZE is probe_n, clamped inside the probe to the
                # available pool.
                probe_bs = min(int(bs), len(fit_pairs))
                # AUTHORITATIVE AND UNCACHED (arm_probe_losses takes no
                # cache knob by design): these numbers feed the
                # floor_ratio_check verdict below, and a rejection writes
                # no artifact — see the teacher_prefix_cache read above.
                est = arm_probe_losses(
                    model, tok, obj, _probe_examples("fit"), max_len=max_len,
                    batch_size=probe_bs, n_probe=probe_n)
                act, floor = est
                arm_diag.update({
                    "final_probe_loss": act, "noop_floor": floor,
                    "floor_ratio": act / max(floor, 1e-6),
                    # The bar THIS arm is judged against (per-arm: the
                    # objective decides the bar — arm_floor_ratio_max).
                    "floor_ratio_max": arm_frm,
                    # The estimate's resolution, next to the values the
                    # guard rules on — so a refusal (3.0x today; the
                    # intended < 1.0 criterion) can be read as resolvable
                    # or as noise, and rescored without refitting.
                    "final_probe_sd": est.active_sd,
                    "final_probe_sem": est.active_sem,
                    "noop_floor_sd": est.noop_sd,
                    "noop_floor_sem": est.noop_sem,
                    "floor_ratio_sem": est.ratio_sem,
                    "probe_n": est.n_examples,
                    "probe_n_batches": est.n_batches,
                    "probe_batch_size": est.batch_size,
                    # WHICH examples (the other two write paths already
                    # record this; the solo path dropped it) and the
                    # per-example values themselves, paired and in
                    # selection order — aligned element-wise with
                    # probe_selected_sha8, so any two cells probed on the
                    # same examples are paired-comparable from disk
                    # (correlated-error sem) without re-probing on a GPU.
                    "probe_selection": "sha256_rank(prompt_without)",
                    "probe_selected_sha8": list(est.selected_sha8 or ()),
                    **probe_values_record(active_values=est.active_values,
                                          noop_values=est.noop_values,
                                          per_family=est.per_family),
                })
                if log_fn is not None:
                    # Live (W&B when available, console+jsonl regardless):
                    # the probe mean AND its uncertainty, per arm.
                    rec = {f"probe/{arm}/final_probe_loss": act,
                           f"probe/{arm}/noop_floor": floor,
                           f"probe/{arm}/floor_ratio":
                               arm_diag["floor_ratio"],
                           f"probe/{arm}/probe_n": est.n_examples}
                    for key in ("final_probe_sem", "noop_floor_sem",
                                "floor_ratio_sem"):
                        if arm_diag[key] is not None:
                            rec[f"probe/{arm}/{key}"] = arm_diag[key]
                    log_fn(rec)
                if heldout_pairs:
                    h_est = arm_probe_losses(
                        model, tok, obj, _probe_examples("heldout"),
                        max_len=max_len,
                        batch_size=min(int(bs), len(heldout_pairs)),
                        n_probe=probe_n)
                    arm_diag["heldout_probe_loss"] = h_est.active
                    arm_diag["heldout_probe_sem"] = h_est.active_sem
                    arm_diag["heldout_probe_n"] = h_est.n_examples
                    # Same per-example payload for the held-out leg, under
                    # its own key family (heldout_probe_*), so held-out
                    # cross-cell comparisons pair the same way.
                    arm_diag["heldout_probe_selected_sha8"] = list(
                        h_est.selected_sha8 or ())
                    arm_diag.update(probe_values_record(
                        active_values=h_est.active_values,
                        noop_values=h_est.noop_values,
                        per_family=h_est.per_family,
                        prefix="heldout_probe"))
                floor_ratio_check(act, floor, ratio_max=arm_frm,
                                  tag=f"{si_id}__{arm}", log_fn=log_fn)
            if arm in CE_ARMS:
                # CE-native floor guard, REPLACING — never silently
                # disabling — the SI-teacher probe: the
                # soft-KL-vs-SI-teacher probe is meaningless here (no
                # teacher exists), so the CE arms skip arm_probe_losses
                # entirely and the verdict is
                # heldout_ce_active / heldout_ce_noop <= ce_floor_ratio_max
                # (default 1.1 — a degeneration/pathology bar, not a
                # did-it-help gate: cells that merely fail to help land at
                # ~1.0 and publish, which the scaling study needs). Same
                # refuse-to-publish semantics, same ONE implementation:
                # floor_ratio_check rules (non-finite refusal included) —
                # a second transcription of the ratio could quietly
                # diverge from the verdict it claims to record.
                ce_frm = arm_ce_floor_ratio_max(cfg, arm)
                h_act = es_diag.get("best_heldout_ce")
                h_noop = es_diag.get("heldout_noop_ce")
                arm_diag.update({"heldout_ce_active": h_act,
                                 "heldout_ce_noop": h_noop,
                                 "ce_floor_ratio_max": ce_frm})
                if h_act is None or h_noop is None:
                    raise ValueError(
                        f"{si_id}__{arm}: the CE floor guard needs "
                        "best_heldout_ce + heldout_noop_ce from the "
                        "trainer's diagnostics and one is missing "
                        f"(got {h_act!r}, {h_noop!r}) — the mandatory "
                        "held-out seam did not run; refusing to publish "
                        "an unguarded CE fit.")
                _cap: dict = {}

                def _cap_log(rec, *a, _c=_cap, **kw):
                    # floor_ratio_check logs its ratio BEFORE ruling —
                    # capture it so a REFUSED fit's record still carries
                    # the ratio it was refused at, then forward under the
                    # fit_record's own names (CE_FLOOR_STREAM_KEYS: the
                    # check's SI-teacher vocabulary — final_probe_loss,
                    # noop_floor — names a held-out CE here).
                    for k2, v2 in rec.items():
                        if k2.endswith("/floor_ratio"):
                            _c["ratio"] = v2
                    if log_fn is not None:
                        log_fn(_ce_floor_stream_record(rec), *a, **kw)

                try:
                    floor_ratio_check(
                        h_act, h_noop, ratio_max=ce_frm,
                        tag=f"{si_id}__{arm}__heldout_ce", log_fn=_cap_log)
                finally:
                    if "ratio" in _cap:
                        arm_diag["ce_ratio"] = _cap["ratio"]
            final_measured = False
            if mon is not None and hasattr(obj, "apply"):
                # Final dual-KL for the record + the per-cell audit —
                # measured on the RETURNED object (uniform across solo
                # arms, seam or no seam; the denominator is computed once
                # inside the probe and the base is frozen, so a post-fit
                # floor equals a pre-fit one).
                fin = mon.final(obj)
                arm_diag.update({k.replace("final/", "final_"): v
                                 for k, v in fin.items()})
                final_measured = True
            elif mon is not None:
                # An object without apply() cannot be probed (test doubles
                # only; every registry object exposes it) — a STATED
                # absence; the audit drops to artifact + sidecar checks.
                arm_diag["final_dual_kl"] = "unavailable_object_not_probeable"
            marker_extra = None
            if continuation is not None and arm == continuation["arm"]:
                # The continued_from provenance block: prior object sha +
                # fingerprint, prior cap, resume position, new cap, epochs
                # actually run, stop reason, the identity checks and the
                # lr report — top-level in the sidecar, in the
                # fit_record/report, and on the COMPLETE marker.
                from sisteer.extraction.continuation import outcome_record

                cont_rec = outcome_record(continuation["record"], es_diag)
                meta["continued_from"] = cont_rec
                arm_diag["continued_from"] = cont_rec
                marker_extra = {"continued_from": cont_rec}
            meta["fit_record"] = arm_diag
            # Save NOW, before the next arm trains.
            _save_object(obj, save_dir, si_id, arm, cfg, max_len, meta,
                         model=model, tok=tok, out_dir=tmp)
            if tmp is not None:
                _close_quietly(mon, writer)
                mon = writer = None
                # The fit SELECTION sidecar joins the temp dir
                # here, so the rename below carries it and the COMPLETE
                # manifest hashes it; absent block = no file.
                write_fit_selection(tmp, cfg, si_id, arm, artefact_path=Path(tmp) / "object.pt")
                if _is_main_rank():
                    finalize_epoch_checkpoints(epoch_checkpoint_root(_ckpt_kwargs([arm]), arm), tmp,
                                               checkpoint_key=_cell_fp(arm), cell_id=f"{si_id}__{arm}",
                                               required=(arm in CE_ARMS),   # the hook is wired for the CE arms only
                                               expected_final_epoch=_epoch_seen.get(arm))
                audit = audit_cell(
                    tmp, arm=arm, cadence=None,
                    monitors_on=monitors_on and final_measured,
                    require_step_loss=_arm_logs_step_loss(arm,
                                                          lockstep=False))
                publish_cell(save_dir, f"{si_id}__{arm}", tmp, si_id=si_id,
                             arm=arm, positions=positions, audit=audit,
                             fingerprint=_cell_fp(arm),
                             marker_extra=marker_extra)
        except _ReplicaTrained:
            # this rank's local record only (_record writes nothing off
            # rank 0); the fit_record rides so a replica's log is readable
            outcome.objects[arm] = obj
            _record(arm, "trained", detail={**es_diag,
                                            "fit_width": fit_width,
                                            "replica_rank": True})
        except Exception as e:  # noqa: BLE001 — isolate, record, continue
            # WITH the fit record (2026-08-25): a floor-guard refusal
            # raises AFTER the probe filled arm_diag with the ratio it
            # ruled on — dropping the detail here strands exactly the
            # numbers a sweep harvests from its failing trials (the wd2
            # baseline's report carried status/error only, no ratio; the
            # jsonl was the only carrier). Empty = the failure predates
            # the diag — record the stated absence, not a fake {}.
            # es_diag rides too (arm_diag keys win): the trainer fills it
            # in place, so a POST-training refusal (CE floor, PSR
            # dead-gate) still ships the stopping record + per-layer gate
            # activity with the failed arm (recorded for every PSR fit,
            # guard pass or FAIL).
            _record(arm, "failed", e,
                    detail={**es_diag, **arm_diag} or None)
        else:
            outcome.objects[arm] = obj
            _record(arm, "trained", detail=arm_diag)
        finally:
            _close_quietly(mon, writer)

    _write_report(outcome, save_dir, report_path=report_path)
    if hb is not None:
        hb.close()
    return outcome


# --------------------------------------------------------------------------- #
# Fit-site provenance (sidecar field ``site``, 2026-08-21)                      #
# --------------------------------------------------------------------------- #

# The hook site each object class EDITS at, recorded per artifact for the
# serve lanes: a vLLM residual op applies at BLOCK OUTPUT (resid_post), and
# resid_pre(L) == resid_post(L-1), so serving a resid_pre-fit object without
# the deliberate one-block shift puts the right vector at the wrong layer —
# plausible numbers, never a crash. Vocabulary shared with
# extraction/actadd.py's metadata ("site": "resid_pre"):
#
#   resid_post    — decoder-layer forward hook (the block's OUTPUT)
#   resid_pre     — decoder-layer forward PRE-hook (the block's INPUT)
#   o_proj_out    — forward hook on self_attn.o_proj, INSIDE the block. Not
#                   a residual-stream site: a residual serve op cannot
#                   express it (the serve lane routes these to native LoRA).
#   o_proj_in     — PRE-hook on self_attn.o_proj: the concatenated per-head
#                   attention outputs BEFORE projection. Also not residual —
#                   and deliberately NOT labelled "resid_pre" (a pre-hook,
#                   yes, but on o_proj, not the decoder layer; the residual
#                   label would invite exactly the off-by-one above).
#   embedding_out — forward hook on the input-embedding module.
#
# Keyed by CLASS name, never arm name — bidpo_lora reuses RankOneLoRAObject
# and must inherit its site; a new arm on an existing class needs no entry.
# FRAGILITY, stated plainly: the objects do not self-report their site, and
# their edit hooks only exist inside apply() per batch (interleaved with
# bookkeeping pre-hooks a save-time probe could not tell apart), so this
# table is read from each class's apply()/hook-registration source. An
# object class whose edit hook MOVES must move its entry with it; a class
# may instead declare its own ``fit_site`` attribute, which wins over the
# table (the seam for new classes and test doubles).
_FIT_SITES: dict[str, str] = {
    # hooks.steering-backed: the edit is layers[idx].register_forward_hook
    # (hooks.py; steering's layer-0/model PRE-hooks are position
    # bookkeeping, not the edit):
    "VectorObject": "resid_post",            # methods.py, via hooks.steering
    "MultiLayerVectorObject": "resid_post",  # extraction/multilayer.py, via steering
    "DiffMeanObject": "resid_post",          # extraction/diffmean.py, via steering
    "AntidoteObject": "resid_post",          # extraction/antidote.py, via steering
    "_VectorArmObject": "resid_post",        # extraction/lockstep.py (function-local class), via steering
    # Own decoder-layer forward hooks (where PRE-hooks appear they are
    # capture/bookkeeping; the EDIT is the layer-output hook):
    "PSRObject": "resid_post",               # extraction/psr.py apply()
    "LoReFTObject": "resid_post",            # extraction/reft.py apply()
    "ReFTr1Object": "resid_post",            # extraction/reft.py apply()
    "AlphaKLObject": "resid_post",           # extraction/alphakl.py apply()
    "AlphaSteerObject": "resid_post",        # extraction/alphasteer.py apply()
    "CastObject": "resid_post",              # extraction/cast.py behavior hooks
    # Decoder-layer PRE-hook — the one true resid_pre class:
    "CaaStaticRObject": "resid_pre",         # extraction/caa_static_r.py apply()
    # Inside the attention block:
    "RankOneLoRAObject": "o_proj_out",       # extraction/objects.py o_proj forward hook
    "MultiLayerLoRAObject": "o_proj_out",    # extraction/multilayer.py o_proj forward hooks
    "LoFiTObject": "o_proj_in",              # extraction/lofit.py o_proj PRE-hooks
    # Input embedding (AppendedSoftTokenObject inherits via the MRO walk):
    "SoftTokenObject": "embedding_out",      # extraction/objects.py emb forward hook
}


def object_fit_site(obj) -> str:
    """The hook site ``obj``'s intervention edits at — the sidecar's
    ``site`` field.

    A non-empty ``fit_site`` attribute on the object wins (the self-report
    seam); otherwise the class-name table above, walked up the MRO so
    subclasses inherit their base's site. Unknown classes REFUSE loudly: a
    missing/null site would be read as "no layer-index shift needed" by a
    serve lane, the exact silent off-by-one the field exists to prevent."""
    declared = getattr(obj, "fit_site", None)
    if isinstance(declared, str) and declared:
        return declared
    for klass in type(obj).__mro__:
        site = _FIT_SITES.get(klass.__name__)
        if site is not None:
            return site
    raise ValueError(
        f"cannot derive the fit-time hook site for {type(obj).__name__}: "
        "read its apply()'s EDIT hook registration and add the class to "
        "_FIT_SITES (or set a `fit_site` class attribute) — refusing to "
        "save an artifact whose serve-side layer-index semantics would be "
        "unrecorded.")


def check_budget_sidecars(save_dir, si_id: str, arms: list[str], *,
                          max_len: int, completion_tokens: int | None,
                          budget_kind: str | None = None,
                          heldout_pool_sha: str | None = None) -> list[str]:
    """Refuse a resume across a changed supervision budget or a stale
    object/sidecar pair.

    The resume check is existence-only, and the compiled-objects dir is
    commit-independent ON PURPOSE — so an object trained under an
    OLDER config survives every code/config change. Before any arm is
    skipped as ``skipped_existing``, this validates its recorded
    ``train_budget_meta.json`` against the CURRENT knobs and the ACTUAL
    object bytes:

    - a MISSING, unreadable, or INCOMPLETE sidecar (no exact ``cell_id``,
      ``completion_tokens``, ``max_len``, or ``object_sha256``) next to an
      existing object is HARD (SystemExit): the sidecar is the
      authoritative budget record, never guessed around. A pre-stamp
      legacy sidecar therefore fails LOUDLY — the migration is explicit
      (retrain with --overwrite, or after verifying provenance add the
      printed sha by hand), never silent acceptance.
    - ``completion_tokens`` mismatch is HARD: it is the
      supervision-volume invariant — resuming would silently mix arms
      trained under different caps, breaking exactly the comparison the
      sidecar exists to record.
    - ``object_sha256`` mismatch against the on-disk bytes is HARD: a
      same-budget sidecar over DIFFERENT object bytes is a stale pair
      (a crashed save or an out-of-band overwrite), not a resumable
      state.
    - a ``max_len`` mismatch WARNS and allows the skip: max_len is a
      memory knob, and with ``preflight_max_len`` guaranteeing prompt +
      full completion fit uncut under both the recorded and the current
      value, the trained object is identical (train_budget's
      invariant) — refusing would spuriously void resume for e.g. a
      later ``max_len_by_si`` optimisation.
    - a ``heldout_pool_sha`` mismatch is HARD when the sidecar RECORDED
      one (the shared-held-out-pool campaigns stamp it; the caller passes
      the CURRENT pool's content hash from ``heldout_pool_identity``):
      the held-out pool is the data every epoch's early stopping ruled
      on, so an object trained against different held-out bytes is a
      different budget world — never silently resumed across. A sidecar
      WITHOUT the key (every legacy SI-pair fit) skips the comparison
      byte-for-byte; a recorded ``null`` is a stated absence and must
      match a current absence.

    Returns the warning strings (the caller logs them).
    """
    from sisteer.eval.defense_cell import object_path_for, sha256_file

    problems: list[str] = []
    warnings: list[str] = []
    for arm in arms:
        cell_id = f"{si_id}__{arm}"
        p = object_path_for(save_dir, cell_id)
        if p is None or not p.exists():
            continue  # nothing to resume for this arm
        meta_path = p.parent / "train_budget_meta.json"
        try:
            meta = json.loads(meta_path.read_text())
        except FileNotFoundError:
            problems.append(
                f"{cell_id}: object exists at {p} but its budget "
                f"sidecar {meta_path} is MISSING — the budget it was "
                "trained under cannot be verified")
            continue
        except json.JSONDecodeError as e:
            problems.append(
                f"{cell_id}: unreadable budget sidecar {meta_path}: {e}")
            continue
        # The budget check branches on training_budget_kind:
        # a document_tokens sidecar has NO completion cap (null by schema)
        # and validates sequence_max_len instead; a legacy SI-pair sidecar
        # (no key) keeps today's checks byte-for-byte. A KIND mismatch
        # between the sidecar and the current config is itself HARD —
        # resuming a document fit under an SI-pair config (or vice versa)
        # is mixing two budget worlds in one store.
        recorded_kind = meta.get("training_budget_kind")
        if recorded_kind != budget_kind:
            problems.append(
                f"{cell_id}: sidecar training_budget_kind="
                f"{recorded_kind!r} != current {budget_kind!r} — a "
                "document-token fit and an SI-pair fit are different "
                "budget worlds; never resume one under the other")
            continue
        if recorded_kind is not None:
            required = ("cell_id", "max_len", "object_sha256",
                        "sequence_max_len")
        else:
            required = ("cell_id", "completion_tokens", "max_len",
                        "object_sha256")
        missing = [k for k in required if meta.get(k) is None]
        if missing:
            problems.append(
                f"{cell_id}: INCOMPLETE budget sidecar {meta_path} — "
                f"missing {missing}. A sidecar predating the object-hash "
                "stamp must be migrated deliberately: retrain with "
                "--overwrite, or (only after verifying provenance) add "
                f'"object_sha256": "{sha256_file(p)}" by hand')
            continue
        if meta["cell_id"] != cell_id:
            problems.append(
                f"{cell_id}: sidecar {meta_path} records "
                f"cell_id={meta['cell_id']!r} — the object store is "
                "corrupted")
            continue
        if recorded_kind is not None:
            # The ONE document-kind schema verdict (train_budget.document_
            # budget_problems) — never a local re-implementation. Resume
            # keeps the default strictness: a null sample_manifest_sha is a
            # stated absence (synthetic pools) and stays resumable; only
            # the serving gate (two_si_run) requires it non-null.
            from sisteer.eval.train_budget import document_budget_problems

            problems.extend(f"{cell_id}: {p}"
                            for p in document_budget_problems(meta))
        elif int(meta["completion_tokens"]) != int(completion_tokens):
            problems.append(
                f"{cell_id}: recorded completion_tokens="
                f"{meta['completion_tokens']} != current "
                f"{completion_tokens} — the D9 shared supervision cap "
                "changed under this object")
        if ("heldout_pool_sha" in meta
                and meta.get("heldout_pool_sha") != heldout_pool_sha):
            problems.append(
                f"{cell_id}: recorded heldout_pool_sha="
                f"{str(meta.get('heldout_pool_sha'))[:12]}… != current "
                f"{str(heldout_pool_sha)[:12]}… — the shared held-out "
                "pool changed under this object (its early stopping ruled "
                "on different bytes); never resume across it")
        actual_sha = sha256_file(p)
        if meta["object_sha256"] != actual_sha:
            problems.append(
                f"{cell_id}: object_sha256 mismatch (sidecar "
                f"{meta['object_sha256'][:12]}… != object {actual_sha[:12]}…)"
                " — a stale object/sidecar pair (crashed save or "
                "out-of-band overwrite), not a resumable state")
        if int(meta["max_len"]) != int(max_len):
            warnings.append(
                f"{cell_id}: recorded max_len={meta['max_len']} != current "
                f"{max_len} — benign iff neither truncates "
                "(preflight_max_len guards that); resuming")
    if problems:
        raise SystemExit(
            "[train-two-si] REFUSING to resume across a changed or "
            "unverifiable training budget:\n  " + "\n  ".join(problems) +
            "\nEither restore the config these objects were trained under, "
            "or pass --overwrite to retrain every arm at the current "
            "budget (never mix the two in one comparison).")
    return warnings


def _save_object(obj, save_dir, si_id, arm, cfg, max_len, meta_extra, *,
                 model, tok, out_dir=None) -> None:
    """Save ONE arm's object to its cell-id path + its sidecar. No-op
    without a ``save_dir``.

    ``out_dir`` (the atomic-publish path): when given, the object +
    sidecar land in THAT directory (the temp cell dir) — at their FINAL
    basenames, the dir rename being the atomicity boundary —
    and ``publish_cell`` renames the audited dir into place and writes the
    COMPLETE marker last. Without it, the legacy in-place layout (direct
    callers/tests): temp sibling(s) then rename, so an existing object
    file always implies a complete, loadable artifact set.

    ``model``/``tok`` are the LIVE objects the fit ran on: the fit-time
    provenance fields below read them, never the config, so the record says
    what actually loaded — a config pin and reality can diverge (a moved
    refs/main, a local checkout, a hand-edited template), and reality wins.

    On the LEGACY path the budget sidecar is stamped with the object's
    ``object_sha256`` and written (atomically, tmp sibling + rename) BEFORE
    the object itself is renamed into place. That ordering makes every crash
    state loud rather than plausible:

    - crash before the sidecar rename: the OLD object + OLD sidecar pair is
      untouched and still consistent (a true state);
    - crash between sidecar rename and object rename: the sidecar's sha
      names the NEW bytes while the old object (or nothing) sits at the
      path — the readers' sha check refuses it as a stale pair (or, with
      no object, resume simply retrains);
    - only after the final rename do object and sidecar agree.

    The old order (object first, sidecar second) could crash into an
    OLD-sidecar/NEW-object pair whose budget fields still matched — an
    undetectable stale pair. The sha stamp + this ordering close that.
    On the ATOMIC path the dir rename is the boundary, so in-dir order is
    irrelevant; the sha is taken from the artifact at its final basename."""
    if save_dir is None and out_dir is None:
        return
    from sisteer.capeval.manifest import sha256_hex16
    from sisteer.eval.defense_cell import object_path_for, sha256_file
    from sisteer.eval.train_budget import (DOCUMENT_TOKENS_KIND,
                                           resolve_budget_kind)
    from sisteer.methods import object_artifact_name

    # Budget provenance schema (never the legacy 512 default
    # for a document fit): under ``training_budget_kind: document_tokens``
    # the cell's budget is document tokens under the sampler's official
    # rule, there IS no completion cap, and writing the legacy
    # ``int(cfg.get("completion_tokens", 512))`` fallback would be FALSE
    # provenance the serving-side budget check then reads as a real
    # training cap. The legacy SI-pair path (no ``training_budget_kind``
    # key) stays byte-identical, fallback default included.
    budget_kind = resolve_budget_kind(cfg)
    if budget_kind == DOCUMENT_TOKENS_KIND:
        if cfg.get("completion_tokens") is not None:
            raise SystemExit(
                "[train-two-si] training_budget_kind: document_tokens with "
                f"a non-null completion_tokens={cfg.get('completion_tokens')!r}"
                " — a document fit has no completion cap; set "
                "completion_tokens: null explicitly (the schema records "
                "the absence honestly).")
        completion_tokens = None
    else:
        completion_tokens = int(cfg.get("completion_tokens", 512))
    cell_id = f"{si_id}__{arm}"
    if out_dir is not None:
        # Atomic-publish path: the temp cell DIR is the atomicity boundary
        # (publish_cell renames the whole dir), so the artifact saves at
        # its FINAL basename. A file-level .tmp here is not just redundant
        # but WRONG (review F14): a container that derives a format
        # sidecar from the artifact path (SteeringVector.save writes
        # `<path>.with_suffix('.json')`) would strand it at
        # object.safetensors.json — a name load_object never reads — and
        # the cell would publish COMPLETE while being unloadable.
        path = Path(out_dir) / object_artifact_name(arm)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.save(path)
        tmp = None
        object_sha = sha256_file(path)  # the artifact actually written
    else:
        # Legacy in-place path (direct callers/tests): temp sibling then
        # rename, for BOTH the primary artifact AND any format sidecar the
        # container wrote next to it (F14's second half). BOTH renames are
        # DEFERRED until after the budget sidecar lands (the crash ordering
        # in the docstring), with the format sidecar moving first so the
        # primary's existence still implies a complete, loadable artifact
        # set — what makes the resume check trustworthy.
        path = object_path_for(save_dir, cell_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        obj.save(tmp)
        object_sha = sha256_file(tmp)  # the pre-rename bytes
    # The revision this fit ran against, with its SOURCE recorded. The live
    # model's resolved commit wins when it exists (a config pin and reality
    # can diverge, and reality is what trained). When the loader leaves
    # ``config._commit_hash`` unset — the offline-cache load path does
    # (the qwen35_4b_base bank: every round-1 sidecar recorded null while
    # the bank's _store_identity.json carried the pin, and the derived-
    # object builder had to resolve the revision from the store instead) —
    # fall back to the config's pinned ``model_revision``: that pin was
    # handed to the loader as ``revision=``, so the snapshot that loaded IS
    # that revision, not a guess. ``revision_source`` says which record a
    # reader is looking at; null/null remains the stated absence (no live
    # commit AND no pin, e.g. a local-path load of an unpinned config).
    live_rev = getattr(getattr(model, "config", None), "_commit_hash",
                       None) or None
    if live_rev is not None:
        model_revision, revision_source = str(live_rev), "live_model"
    elif cfg.get("model_revision") is not None:
        model_revision, revision_source = (str(cfg["model_revision"]),
                                           "config_pin")
    else:
        model_revision, revision_source = None, None
    meta = {
        "cell_id": cell_id, "si_id": si_id, "arm": arm,
        "max_len": max_len,
        # the completion-token budget entering the loss, recorded per
        # cell — the SHARED cap, so supervision volume is equal across SIs.
        "completion_tokens": completion_tokens,
        # FIT-PHASE provenance (audit 2026-08-21): the positions leg this
        # object was FITTED under, stamped uniformly here (lockstep and solo
        # branches alike) so a Class C (decode-only) artifact can never be
        # mistaken for a Class A (all-token) one after the fact. Named
        # positions_at_fit because the repo has TWO INCOMPATIBLE decode-only
        # mechanisms — the POSITION MASK (eval/jailbreak.py's response leg:
        # the trailing assistant-header prompt positions AND every generated
        # position are steered, so the forward producing the FIRST generated
        # token IS steered) and the capeval ENGINE-PHASE gate
        # (generated-only serving: the prefill runs unsteered and token 1 is
        # sampled from that unsteered forward; steering begins at token 2) —
        # and an object fit under one must never be served under the other.
        # Class C serves under the POSITION MASK (decision recorded at
        # configs/two_si.yaml class_c_arms, with the rationale); the evals
        # lane asserts fit-phase == serve-phase and refuses a mismatch.
        "positions_at_fit": meta_extra.get("positions", "all"),
        # The exact bytes this sidecar describes — verified before any
        # resume-skip or generation consumes the pair.
        "object_sha256": object_sha,
        **meta_extra,
        # FIT-TIME PROVENANCE (audit 2026-08-21), stamped uniformly here for
        # BOTH the lockstep and solo branches (the positions_at_fit
        # precedent) and AFTER **meta_extra so no caller's meta can shadow
        # what actually ran. All values come from the LIVE model/tokenizer,
        # never re-read from config; null means the component does not exist
        # on what was loaded (e.g. a local-path load has no hub commit) — a
        # stated absence, never a silent omission.
        "model_id": getattr(getattr(model, "config", None),
                            "_name_or_path", None),
        "model_revision": model_revision,
        "revision_source": revision_source,
        # sha256 of the RAW chat-template string, 16-hex prefix — the EXACT
        # capeval convention (manifest.sha256_hex16 over
        # `getattr(tok, "chat_template", None) or ""`, as in
        # scripts/capeval_judged_run.py and scripts/cell_controls_run.py),
        # so fit-time and eval-time hashes compare directly. The known
        # divergence this detects: NousResearch @ d10aef79 renders
        # b48c47f644389271, meta-llama @ 0e9e39f2 renders e10ca381b1ccc5cf —
        # nominally the same model, different prompt bytes.
        "template_hash": (None if tok is None else sha256_hex16(
            (getattr(tok, "chat_template", None) or "").encode("utf-8"))),
        # The width the fit REALLY used — never the arm's own configured
        # entry: a lockstep arm trains at its GROUP's shared min (so its
        # realised width can be SMALLER than its own batch_size_by_arm
        # value), and a solo arm at the kwargs that survived
        # _safe_registry_train (null = the trainer batches internally). The
        # hard KeyError is deliberate: a caller that cannot say what width
        # ran fails the arm loudly instead of writing a guess.
        "resolved_batch_size": meta_extra["batch_size"],
        "site": object_fit_site(obj),
        # The layer indices this object ACTUALLY targets, read off the
        # LIVE object — on hybrid-attention models the all-layer o_proj LoRA
        # is capability-filtered (Qwen3.5-27B: the 16 full-attention layers,
        # not all 64), and this sidecar field makes that auditable without
        # decoding the torch blob. Uniform across arms: any object exposing
        # an int-list `.layers` (multilayer LoRA/vector, PSR) is stamped;
        # null = the object has no such set (single-layer/head objects carry
        # their own placement fields), a stated absence.
        "target_layers": _int_layer_list(obj),
    }
    if budget_kind == DOCUMENT_TOKENS_KIND:
        # The document budget schema: the budget consumers
        # branch on training_budget_kind — a document_tokens fit validates
        # against sequence_max_len + the sample manifest (meta_extra
        # carries sample_manifest_sha), never against a completion cap.
        # Keys added ONLY under the new kind, so legacy sidecars stay
        # byte-identical.
        meta["training_budget_kind"] = DOCUMENT_TOKENS_KIND
        meta["sequence_max_len"] = int(max_len)
        # Schema completeness at the ONE writer: the key must exist even
        # when the caller had no manifest to hash (null = a stated
        # absence; a MISSING key reads as a pre-schema sidecar).
        meta.setdefault("sample_manifest_sha", None)
    meta_path = path.parent / "train_budget_meta.json"
    meta_tmp = meta_path.with_name(meta_path.name + ".tmp")
    meta_tmp.write_text(json.dumps(meta, indent=1, sort_keys=True,
                                   default=str))
    meta_tmp.replace(meta_path)   # sidecar FIRST (see docstring), atomically
    if tmp is not None:
        # legacy path only: then the artifacts the sidecar describes —
        # format sidecar first (F14), then the primary.
        tmp_sidecar = tmp.with_suffix(".json")
        if tmp_sidecar.exists():
            tmp_sidecar.replace(path.with_suffix(".json"))
        tmp.replace(path)


def _int_layer_list(obj) -> list[int] | None:
    """`obj.layers` as a plain int list when that is what it is, else None
    (LoFiT's placement is `.heads` (layer, head) pairs; single-layer objects
    carry `.layer`) — a provenance read, never a behavior seam."""
    layers = getattr(obj, "layers", None)
    if isinstance(layers, (list, tuple)) and all(
            isinstance(l, int) for l in layers):
        return [int(l) for l in layers]
    return None


# --------------------------------------------------------------------------- #
# In-job parallel scheduler (ported from the accel branch,                      #
# train_two_si_objects.py:1840-2366 @ 6aba597).                               #
# THE SCHEDULER PACKS ARMS 1-PER-GPU: data_parallel.arm_gpus must be 1 for    #
# every arm; its job is packing independent arm tasks onto the pod's GPU      #
# pool, cheap-first then longest-first. A WIDE single arm is the OTHER mode:  #
# the fits config's top-level `fit_width: N` + `torchrun --nproc_per_node=N   #
# ... train|arm-task` runs one CE arm as N ranks that group-shard every       #
# optimizer step's micro-batches (train_object_ce; _enter_fit_process_group   #
# is the gate) — the two modes never combine in one job.                      #
# --------------------------------------------------------------------------- #

_ARM_COST: dict[str, float] = {
    "psr_mse_all": 9.0, "lofit": 9.0, "alphakl_fisher": 8.0, "loreft": 8.0,
    "bidpo": 8.0, "psr_kl_all": 7.0, "vector_all": 7.0, "lora_r1_all": 7.0,
    "lora_r4": 7.0,
    "lora_r1": 6.0, "token_bos": 6.0, "token_faithful": 6.0, "vector": 6.0,
    "vector_rkl": 6.0, "vector_js": 6.0, "psr_kl": 6.0, "psr": 6.0,
    "reft_r1": 5.0, "psr_kl_hard": 6.0, "psr_kl_all_hard": 7.0,
    "alphakl_soft": 3.0, "alphakl_null": 2.0, "diffmean": 2.0,
}
_DEFAULT_ARM_COST = 5.0

# Cheap-first threshold (wave-autopsy 2026-08-22, lesson 3): pure
# longest-first placement put the 31k-step gradient arms ahead of diffmean
# on EVERY pod, so when the wave was cut at 6.5h the CHEAPEST cells —
# minutes of forward passes — were the ones missing. Arms whose static cost
# falls strictly BELOW this threshold are scheduled FIRST as a batch; the
# rest keep longest-first. 4.0 sits strictly between the capture arms (2-3)
# and the cheapest gradient rung (5.0 — also _DEFAULT_ARM_COST, so an arm
# MISSING from the table is never classified cheap).
# ``data_parallel.cheap_first_below`` overrides; 0 disables the phase.
_CHEAP_FIRST_BELOW_DEFAULT = 4.0


def cheap_first_threshold(cfg: dict) -> float:
    """The configured cheap-first threshold: ``data_parallel.
    cheap_first_below`` when set, else :data:`_CHEAP_FIRST_BELOW_DEFAULT`."""
    dp = cfg.get("data_parallel") or {}
    v = dp.get("cheap_first_below")
    return _CHEAP_FIRST_BELOW_DEFAULT if v is None else float(v)


def _placement_key(task: dict) -> tuple:
    """Placement order for one planned task: the cheap-first batch (phase
    ``cheap-first``) ahead of everything, then longest-first with the
    deterministic name tiebreak."""
    return (0 if task["phase"] == "cheap-first" else 1,
            -task["est_wall"], task["arm"])


def arm_width(cfg: dict, si_id: str, arm: str, gpus: int) -> int:
    """The GPU width one arm's task uses under the in-job SCHEDULER —
    ALWAYS 1 (arms pack 1-per-GPU; the trainers run their historical
    single-process path byte-identically). An explicit
    ``data_parallel.arm_gpus.<arm>`` > 1 REFUSES loudly: a wide single arm
    is the torchrun mode (the config's top-level ``fit_width``), not
    a scheduler placement."""
    del si_id
    dp = cfg.get("data_parallel") or {}
    override = (dp.get("arm_gpus") or {}).get(arm)
    if override is not None and int(override) > 1:
        raise SystemExit(
            f"[train-two-si] data_parallel.arm_gpus.{arm}={override}: "
            "the in-job scheduler packs arms 1-per-GPU only. Set every "
            "arm_gpus entry to 1 and widen by running more concurrent arm "
            "tasks — or, for ONE data-parallel CE fit, drop the scheduler "
            "and set the config's top-level `fit_width: N` (launched as "
            "`torchrun --standalone --nproc_per_node=N ... train`, the fit "
            "sbatch's path; D96).")
    return max(1, min(int(override or 1), max(gpus, 1)))


def _next_startable(pending: list, free: int):
    """The first pending task the hybrid policy may start with ``free``
    GPUs (``pending`` must be ``_placement_key``-sorted): a main task
    NEVER starts while any cheap-first task is pending (the phase
    barrier), then first-fit by width. ONE implementation, consumed by
    both the ``plan`` simulation and the live scheduler, so the printed
    plan is the placement that actually runs — by construction, not by
    hand-maintained sameness. Returns the task dict, or None."""
    cheap_waiting = bool(pending) and pending[0]["phase"] == "cheap-first"
    for t in pending:
        if cheap_waiting and t["phase"] == "main":
            return None  # the phase barrier
        if t["width"] <= free:
            return t
    return None


def plan_parallel(cfg: dict, si_id: str, arms: list[str]) -> dict:
    """The deterministic parallel plan for one SI: per-arm width/cost/phase
    and a simulated hybrid schedule on the configured pool — the cheap
    closed-form batch first, then longest-first. Pure CPU.

    The task list depends ONLY on (config, si, requested arms) — never on
    which other arms are done (resume skips at launch time) — so placement
    can never make an arm's RESULT depend on the arm set (placement is
    operational: no trainer reads its rank or its siblings)."""
    dp = cfg.get("data_parallel") or {}
    gpus = int(dp.get("gpus", 1) or 1)
    cheap_below = cheap_first_threshold(cfg)
    tasks = []
    for arm in arms:
        w = arm_width(cfg, si_id, arm, gpus)
        cost = _ARM_COST.get(arm, _DEFAULT_ARM_COST)
        tasks.append({"arm": arm, "width": w, "cost": cost,
                      "est_wall": cost / w,
                      "phase": ("cheap-first" if cost < cheap_below
                                else "main")})
    # Hybrid order: the cheap-first batch ahead of everything, then
    # longest-first; at every free-up, start the first pending task that
    # fits (_next_startable — the SAME policy object the live scheduler
    # consumes, which is what makes this simulation authoritative).
    pending = sorted(tasks, key=_placement_key)
    free = gpus
    running: list[tuple[float, dict]] = []  # (end_time, task)
    now = 0.0
    order = []
    while pending or running:
        while (t := _next_startable(pending, free)) is not None:
            free -= t["width"]
            running.append((now + t["est_wall"], t))
            order.append({**t, "start": round(now, 3)})
            pending.remove(t)
        if not running:
            break
        running.sort(key=lambda e: e[0])
        now, done = running.pop(0)
        free += done["width"]
    makespan = max((o["start"] + o["est_wall"] for o in order), default=0.0)
    return {"si_id": si_id, "gpus": gpus, "tasks": order,
            "est_makespan_units": round(makespan, 3)}


def _gpu_pool() -> list[str]:
    """The job's visible GPU ids, in order: CUDA_VISIBLE_DEVICES when set
    (we only ever take SUBSETS of it — cluster hard rule 3), else the
    runtime count. ``SISTEER_FIT_GPU_POOL`` overrides for CPU tests of the
    scheduler (no GPUs)."""
    forced = os.environ.get("SISTEER_FIT_GPU_POOL")
    if forced is not None:
        return [g for g in forced.split(",") if g != ""]
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd:
        return [g.strip() for g in cvd.split(",") if g.strip()]
    try:
        import torch

        return [str(i) for i in range(torch.cuda.device_count())]
    except Exception:  # noqa: BLE001 — no torch/CUDA: no pool
        return []


class _ArmProc:
    """One arm's task: its rank subprocesses, GPU subset, and log paths."""

    def __init__(self, arm: str, width: int, gpu_ids: list[str],
                 procs: list, logs: list):
        self.arm = arm
        self.width = width
        self.gpu_ids = gpu_ids
        self.procs = procs
        self.logs = logs

    def poll(self):
        """None while any rank runs; else the group's exit code (first
        nonzero rank, else 0)."""
        codes = [p.poll() for p in self.procs]
        if any(c not in (None, 0) for c in codes):
            for p in self.procs:
                if p.poll() is None:
                    p.terminate()
            for p in self.procs:
                try:
                    p.wait(timeout=30)
                except Exception:  # noqa: BLE001
                    p.kill()
            return next(c for c in [p.poll() for p in self.procs]
                        if c not in (None, 0))
        if all(c == 0 for c in codes):
            return 0
        return None

    def terminate(self):
        for p in self.procs:
            if p.poll() is None:
                p.terminate()


def _spawn_arm_task(args, arm: str, width: int, gpu_ids: list[str],
                    logdir: Path) -> _ArmProc:
    """Launch one arm's process. Width 1 is this branch's ONLY width
    (``arm_width`` refuses more; ``cmd_arm_task`` re-checks), and it runs
    plain: every distributed env variable is STRIPPED so the trainers take
    their historical single-process path byte-identically. The task's
    CUDA_VISIBLE_DEVICES is a STRICT SUBSET of this job's (cluster hard
    rule 3)."""
    import subprocess

    cmd_common = [
        sys.executable, str(Path(__file__).resolve()), "arm-task",
        "--config", args.config, "--si", args.si, "--arm", arm,
        "--pairs", args.pairs, "--out", str(args.out or ""),
        "--expected-width", str(width),
    ]
    if args.overwrite:
        cmd_common.append("--overwrite")
    # Forward the fit leg and any explicit bidpo pool overrides, so every
    # arm task resolves what cmd_train resolved. Also PIN the config's
    # byte identity: an arm task re-reads the config FILE — possibly hours
    # after launch for a queued arm — so without the pin a mid-wave config
    # edit would fit a different world than the scheduler planned.
    expect_sha = (str(getattr(args, "expect_config_sha", "") or "")
                  or _config_sha(str(args.config)) or "")
    for flag, val in (("--positions", getattr(args, "positions", "")),
                      ("--expect-config-sha", expect_sha),
                      ("--token-cache", getattr(args, "token_cache", "")),
                      ("--teacher-topk", getattr(args, "teacher_topk", "")),
                      # Mid-fit durability rides into every arm task, so a
                      # checkpointed parallel wave never silently drops it.
                      ("--checkpoint-dir",
                       getattr(args, "checkpoint_dir", "")),
                      ("--checkpoint-every",
                       str(getattr(args, "checkpoint_every", 0) or "")),
                      # The TB trace dir rides into every arm task — the
                      # trace handler names files per (host, pid), so
                      # concurrent tasks share the dir without collision.
                      ("--profile", getattr(args, "profile", "")),
                      ("--bidpo-harmful-pairs",
                       getattr(args, "bidpo_harmful_pairs", "")),
                      ("--bidpo-benign-pairs",
                       getattr(args, "bidpo_benign_pairs", ""))):
        if val:
            cmd_common += [flag, str(val)]
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    for k in ("RANK", "WORLD_SIZE", "LOCAL_RANK",
              "MASTER_ADDR", "MASTER_PORT"):
        env.pop(k, None)
    log = logdir / f"{args.si}__{arm}.rank0.log"
    fh = open(log, "a", encoding="utf-8")
    fh.write(f"\n===== launch {datetime.now(timezone.utc).isoformat()} "
             f"arm={arm} rank=0/{width} gpus={gpu_ids} =====\n")
    fh.flush()
    proc = subprocess.Popen(
        cmd_common, stdout=fh, stderr=subprocess.STDOUT, env=env,
        cwd=str(REPO))
    fh.close()  # the child holds the fd; ours can go
    return _ArmProc(arm, width, gpu_ids, [proc], [log])


def run_parallel_fit(args, cfg, arms: list[str], save_dir: Path) -> int:
    """The in-job scheduler: place every requested arm on the GPU pool
    (cheap-first batch, then longest-first — see ``plan_parallel``), run
    each as its own process, merge the per-arm report fragments into the
    canonical fit report, and return the job's exit code (nonzero iff any
    arm failed — after every arm was attempted, the isolation contract).

    Resume-by-skip happens here (validated COMPLETE marker) AND in
    the arm task (idempotent)."""
    sha = _assert_expected_config_sha(args)
    print(_resolved_banner(cfg, sha, _effective_positions(args, cfg),
                           str(getattr(args, "config", "") or "")),
          flush=True)
    import time as _time

    dp = cfg.get("data_parallel") or {}
    gpus = int(dp.get("gpus", 1) or 1)
    pool = _gpu_pool()
    if len(pool) != gpus:
        raise SystemExit(
            f"[train-two-si] data_parallel.gpus={gpus} but this job sees "
            f"{len(pool)} device(s) ({pool or 'none'}). The world size is "
            "PINNED by the shared config so every SI fits identically — "
            "fix the submission or the config; never run one SI at a "
            "different width.")
    # Default-on for every wave: tokenize the pool ONCE (tokenizer-only,
    # CPU) before any arm process exists, so no arm pays the 30-60 min
    # encode phase and none of the others re-pay it.
    tc_flag = getattr(args, "token_cache", "auto") or "auto"
    if tc_flag != "off":
        sidecar = (Path(tc_flag) if tc_flag != "auto"
                   else _token_sidecar_path(args.pairs))
        if not sidecar.exists():
            print(f"[tokcache] sidecar missing; building once for the "
                  f"whole wave -> {sidecar}")
            build_token_sidecar(cfg, args.si, args.pairs, out=str(sidecar))
    logdir = save_dir / "_fit_logs"
    logdir.mkdir(parents=True, exist_ok=True, mode=0o700)

    outcome = FitOutcome(si_id=args.si)

    def _merge_fragment(arm: str) -> dict | None:
        frag = arm_report_path_for(save_dir, args.si, arm)
        if not frag.exists():
            return None
        try:
            payload = json.loads(frag.read_text())
            return payload.get("arms", {}).get(arm)
        except Exception:  # noqa: BLE001 — a torn fragment is a failure
            return None

    def _record(arm: str, status: str, error: str | None = None,
                detail: dict | None = None) -> None:
        outcome.arms[arm] = ArmOutcome(
            status, error_type="SchedulerRecorded" if error else None,
            error=error, detail=detail)
        _write_report(outcome, save_dir)

    # Resume-by-skip at the scheduler: never launch a task for a cell with
    # a validated COMPLETE marker (bare object existence is NOT enough),
    # and the marker must carry THIS request's fit fingerprint — the same
    # identity the arm task recomputes and stores, via the shared
    # _cli_fit_identity derivation, so scheduler skip and arm-task skip
    # can never disagree.
    positions = _effective_positions(args, cfg)
    assert_store_identity(save_dir, cfg, positions)
    pool_sha, config_sha = _cli_fit_identity(args, cfg)
    # The same resolved artifact path the arm tasks will resolve from the
    # forwarded --teacher-topk, so scheduler skip and arm-task skip derive
    # the identical teacher-leg tag (the can-never-disagree invariant).
    tk_path = _resolve_teacher_topk(args)
    todo: list[str] = []
    failed_arms: dict[str, str] = {}
    for arm in arms:
        p = None if args.overwrite else validated_complete(
            save_dir, f"{args.si}__{arm}",
            cell_fit_fingerprint(cfg, args.si, arm, positions=positions,
                                 pool_sha256=pool_sha,
                                 config_sha=config_sha,
                                 teacher=_teacher_leg_tag(tk_path, cfg,
                                                          arm)))
        if p is not None:
            # the fit is skipped; its registry emission is verified and
            # replayed when missing — a failed
            # replay is this run's visible failure, the publication stands
            try:
                ev = ensure_registry_emission(p.parent, si_id=args.si, arm=arm)
            except RegistryEmissionError as e:
                failed_arms[arm] = str(e)
                _record(arm, "failed", error=str(e)[:2000],
                        detail={"registry_emission":
                                registry_emission_evidence(p.parent)})
                print(f"[train-two-si] {args.si}__{arm}: validated COMPLETE "
                      f"cell at {p.parent} — fit skipped (resume) but its "
                      f"registry emission FAILED: {e}", file=sys.stderr)
                continue
            _record(arm, "skipped_existing",
                    detail={"registry_emission": ev} if ev else None)
            print(f"[train-two-si] {args.si}__{arm}: validated COMPLETE "
                  f"cell at {p.parent} — skipping (resume); registry "
                  f"emission {(ev or {}).get('status', 'unverified')}")
        else:
            todo.append(arm)

    plan = plan_parallel(cfg, args.si, todo)
    by_arm = {t["arm"]: t for t in plan["tasks"]}
    cheap_desc = ", ".join(f"{t['arm']}x{t['width']}" for t in plan["tasks"]
                           if t["phase"] == "cheap-first") or "none"
    main_desc = ", ".join(f"{t['arm']}x{t['width']}" for t in plan["tasks"]
                          if t["phase"] == "main") or "none"
    print(f"[train-two-si] parallel fit: SI={args.si} pool={gpus} GPUs, "
          f"{len(todo)} arm task(s); placement — cheap-first (closed-form, "
          f"cost<{cheap_first_threshold(cfg):g}): {cheap_desc}; "
          f"then longest-first: {main_desc}")

    pending = sorted((by_arm[a] for a in todo), key=_placement_key)
    free: list[str] = list(pool)
    live: list[_ArmProc] = []

    import signal as _signal

    def _tear_down(signum, _frame):
        print(f"[train-two-si] scheduler got signal {signum} — "
              "terminating arm tasks", file=sys.stderr)
        for ap in live:
            ap.terminate()
        raise SystemExit(128 + signum)

    prev = _signal.signal(_signal.SIGTERM, _tear_down)
    try:
        while pending or live:
            # _next_startable IS the plan simulation's policy — same phase
            # barrier, same first-fit — so the printed plan is what runs.
            while (t := _next_startable(pending, len(free))) is not None:
                arm, w = t["arm"], t["width"]
                ids, free = free[:w], free[w:]
                ap = _spawn_arm_task(args, arm, w, ids, logdir)
                live.append(ap)
                _record(arm, "running")
                print(f"[train-two-si] LAUNCH {args.si}__{arm} "
                      f"width={w} gpus={ids} log={ap.logs[0]}")
                pending.remove(t)
            if not live:
                break
            _time.sleep(2.0)
            for ap in list(live):
                code = ap.poll()
                if code is None:
                    continue
                live.remove(ap)
                free.extend(ap.gpu_ids)
                rec = _merge_fragment(ap.arm)
                if code == 0 and rec is not None:
                    outcome.arms[ap.arm] = ArmOutcome(
                        rec.get("status", "trained"),
                        error_type=rec.get("error_type"),
                        error=rec.get("error"),
                        detail=rec.get("detail"))
                    _write_report(outcome, save_dir)
                    print(f"[train-two-si] DONE {args.si}__{ap.arm}: "
                          f"{rec.get('status')}")
                    if rec.get("status") == "failed":
                        failed_arms[ap.arm] = rec.get("error") or ""
                else:
                    tail = ""
                    try:
                        tail = "".join(
                            ap.logs[0].read_text().splitlines(True)[-15:])
                    except Exception:  # noqa: BLE001
                        pass
                    msg = (f"arm task exited {code}"
                           + (f"; fragment status: {rec.get('status')}"
                              if rec else "; no report fragment"))
                    failed_arms[ap.arm] = msg
                    _record(ap.arm, "failed", error=msg[:2000],
                            detail=rec.get("detail") if rec else None)
                    print(f"[train-two-si] FAILED {args.si}__{ap.arm}: {msg}\n"
                          f"--- last lines of {ap.logs[0]} ---\n{tail}",
                          file=sys.stderr)
    finally:
        _signal.signal(_signal.SIGTERM, prev)
        for ap in live:
            ap.terminate()

    _write_report(outcome, save_dir)
    n_ok = sum(1 for o in outcome.arms.values()
               if o.status in ("trained", "skipped_existing"))
    print(f"[train-two-si] SI={args.si}: {n_ok} arm(s) ok, "
          f"{len(failed_arms)} failed -> {save_dir} "
          f"(parallel, pool={gpus} GPUs)")
    if failed_arms:
        for arm, msg in failed_arms.items():
            print(f"[train-two-si] FAILED {args.si}__{arm}: {msg}",
                  file=sys.stderr)
        print(f"[train-two-si] {len(failed_arms)} arm(s) failed after every "
              f"arm was attempted; report: "
              f"{report_path_for(save_dir, args.si)}", file=sys.stderr)
        return 1
    return 0


def cmd_arm_task(args, cfg) -> None:
    """ONE arm's fit, run as its own process (group): the scheduler's unit
    of work at width 1 (arm_width pins it; the distributed env is stripped
    at spawn, so the trainers take their historical path byte-identically),
    and — launched under `torchrun --nproc_per_node=N ... arm-task
    --expected-width N` — one rank of a data-parallel CE fit: the
    trainer shards each optimizer step's micro-batches across the ranks,
    rank 0 owns the guards, the save, the report fragment and the metrics
    stream. Same gate as `train` (_enter_fit_process_group)."""
    # Stale-config guard FIRST — before even the imports below — so a
    # stale-tarball task dies in milliseconds, before any pair/model load.
    sha = _assert_expected_config_sha(args)
    pos_flag = getattr(args, "positions", "")
    if pos_flag:
        cfg = _cfg_with_positions(cfg, pos_flag)
    print(_resolved_banner(cfg, sha, _effective_positions(args, cfg),
                           str(getattr(args, "config", "") or "")),
          flush=True)
    par = _enter_fit_process_group(args, cfg, [args.arm], entry="arm-task")
    is_main = par is None or par.is_main

    from sisteer.extraction.fitmon import pin_attention_backends
    from sisteer.models import load_model_and_tokenizer

    # Pools BEFORE the model load (cmd_train's order, and _load_pairs'
    # stated contract): a missing/stale pairs or bidpo pool file fails in
    # milliseconds instead of after minutes of weight loading — per arm
    # task, so a wave pays the mistake N times otherwise. Same for the
    # CE/document config-shape guard.
    refuse_ce_misconfiguration(cfg, [args.arm])
    pairs = _load_pairs(args.pairs)
    # The SAME identity derivation the scheduler's skip gate used, so the
    # fingerprint this task stores is the one the scheduler checks (F15) —
    # derived BEFORE the model load so the sample manifest can be validated
    # against the pool hash (cell + pool_sha256) through the ONE resolver;
    # a stale or foreign manifest is a SystemExit here, never a sidecar row.
    pool_sha, config_sha = _cli_fit_identity(args, cfg)
    if _data_mode(cfg) == DOC_DATA_MODE:
        _sample_manifest_sha_for(args.pairs, cell=args.si, pool_sha256=pool_sha)
    # bidpo's own two-sided pool, on this path too: a DPO arm scheduled by
    # the parallel driver must get the same pool cmd_train would load.
    bidpo_harmful_pairs, bidpo_benign_pairs, bidpo_benign_meta = \
        _load_bidpo_pool(args, cfg, [args.arm])
    pin_attention_backends()  # cuDNN-SDPA off in the production path
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        leading_special_tokens=_leading_policy(cfg))
    maybe_enable_gradient_checkpointing(model, cfg, arms=[args.arm])
    _warm_token_cache(args, cfg, tok)
    from sisteer.eval.defense_cell import resolve_defenses_dir

    save_dir = Path(args.out) if args.out else resolve_defenses_dir(
        cfg, repo_root=REPO)
    wandb_run = (wandb_fit_run(cfg, args.si, [args.arm],
                               positions=getattr(args, "positions", None))
                 if is_main else None)
    log_fn = (make_metrics_log_fn(
        jsonl_path=save_dir / f"{args.si}__{args.arm}__metrics.jsonl",
        wandb_run=wandb_run, echo=console_log, arms=[args.arm])
        if is_main else None)
    # (pool_sha / config_sha: derived above, before the model load.)
    prof = (start_profiler(args.profile)
            if getattr(args, "profile", "") and is_main else None)
    if prof is not None:
        log_fn = profiler_stepping_log_fn(log_fn, prof)
    try:
        outcome = train_si_objects(
            model, tok, args.si, pairs, [args.arm], cfg=cfg,
            save_dir=save_dir, log_fn=log_fn, overwrite=args.overwrite,
            report_path=arm_report_path_for(save_dir, args.si, args.arm),
            bidpo_harmful_pairs=bidpo_harmful_pairs,
            bidpo_benign_pairs=bidpo_benign_pairs,
            bidpo_benign_meta=bidpo_benign_meta,
            pool_sha256=pool_sha, config_sha=config_sha,
            teacher_topk=_resolve_teacher_topk(args),
            pairs_path=args.pairs,
            checkpoint_dir=getattr(args, "checkpoint_dir", "") or None,
            checkpoint_every=(int(getattr(args, "checkpoint_every", 0) or 0)
                              or None))
    finally:
        if prof is not None:
            prof.__exit__(None, None, None)
        if wandb_run is not None:
            wandb_run.finish()
        _leave_fit_process_group()
    if not outcome.ok:
        for arm, res in outcome.failed.items():
            print(f"[arm-task] FAILED {args.si}__{arm}: "
                  f"{res.error_type}: {res.error}", file=sys.stderr)
        raise SystemExit(1)


# --------------------------------------------------------------------------- #
# CLI                                                                          #
# --------------------------------------------------------------------------- #


def _load_pairs(path: str):
    from sisteer.data.pairs import Pair
    from sisteer.extraction.psr import validate_probe_pairs_path

    # Fail in milliseconds with the shared-storage explanation (cluster
    # /home is node-local disk, invisible from compute nodes) instead of a
    # bare FileNotFoundError — and do it here, which cmd_train/cmd_plan
    # reach BEFORE the model load.
    validate_probe_pairs_path(path)
    raw = Path(path).read_bytes()
    # An evaluation-only corpus (the ReNeLLM paper corpus, shipped under terms that forbid training
    # on it) never reaches a fit: refused by path, file hash and content hash, and by each pair's
    # prompt, before anything is loaded (sisteer.eval.attacks.eval_only; counts and hashes only).
    from sisteer.eval.attacks import eval_only
    consumer = f"the training stage's pair loader ({path})"
    eval_only.refuse_corpus(consumer=consumer, path=path, raw=raw)
    rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()
            if line.strip()]
    eval_only.refuse_texts((r.get("prompt") for r in rows), consumer=consumer)
    return [Pair(prompt=r["prompt"], pos=r["pos"], neg=r.get("neg", ""),
                 instruction_id=r.get("instruction_id", ""),
                 requirement_a=r.get("requirement_a"),
                 anti_neg=r.get("anti_neg")) for r in rows]


def cmd_plan(args, cfg) -> None:
    # getattr, not args.positions: programmatic callers (tests, the
    # orchestrator) build bare Namespaces that predate the flag.
    pos_flag = getattr(args, "positions", "")
    positions = _effective_positions(args, cfg)
    if pos_flag:
        cfg = _cfg_with_positions(cfg, pos_flag)
    # A decode-only wave defaults to the Class C arm set (fit-time taxonomy:
    # the same driver, the same builders — only the mask leg and therefore
    # the default membership differ). --arms still overrides either way.
    arms = (args.arms.split(",") if args.arms else
            (default_class_c_arms(cfg) if positions == "response"
             else default_class_a_arms(cfg)))
    plan = plan_si(cfg, args.si, arms)
    print(f"[plan] SI={plan['si_id']}  max_len={plan['max_len']}  "
          f"positions={positions}  (one model load for {len(arms)} arms)")
    print(f"[plan] lockstep group (shared teacher, bs={plan['lockstep_batch_size']}): "
          f"{plan['lockstep_group']}")
    print(f"[plan] solo (own teacher forward): {plan['solo']}")
    if plan["condition_arms"]:
        print(f"[plan] category-condition prompts (harmful=fire / benign=no-fire, "
              f"via condition_contrast config) go to: {plan['condition_arms']}")
    dp = cfg.get("data_parallel") or {}
    if int(dp.get("gpus", 1) or 1) > 1:
        par_plan = plan_parallel(cfg, args.si, arms)
        print(f"[plan] parallel fit (arms 1-per-GPU): "
              f"pool={par_plan['gpus']} GPUs, est makespan "
              f"{par_plan['est_makespan_units']} cost-units; cheap-first "
              f"batch = closed-form arms below cost "
              f"{cheap_first_threshold(cfg):g}")
        for t in par_plan["tasks"]:
            kind = ("singleton-lockstep"
                    if t["arm"] in plan["lockstep_group"] else "solo")
            print(f"[plan]   {t['arm']:<16} width={t['width']} "
                  f"cost~{t['cost']:g} start~{t['start']:g} "
                  f"({kind}, {t['phase']})")
    if not args.pairs:
        print("[plan] (pass --pairs <jsonl> [--with-model] for the "
              "measurement-harness budget table)")
        return
    # Measurement-harness budget table against real prompt lengths. Needs a
    # tokenizer; --with-model loads the pinned one, else a table is skipped.
    if not args.with_model:
        print("[plan] --pairs given without --with-model; skipping the budget "
              "table (it needs a tokenizer to measure prompt lengths)")
        return
    from sisteer.data.pairs import pairs_to_distill_examples
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.eval.train_budget import format_budget_table, measure_si_budget
    from sisteer.extraction.lockstep import lockstep_compatible
    from sisteer.models import load_model_and_tokenizer

    pairs = _load_pairs(args.pairs)
    _m, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        leading_special_tokens=_leading_policy(cfg))
    examples = pairs_to_distill_examples(tok, pairs, resolve_instruction(args.si))
    rows = measure_si_budget(cfg, tok, examples, si_id=args.si, arms=arms,
                             compatible=lockstep_compatible)
    print(format_budget_table(rows))


def _load_bidpo_pool(args, cfg, arms):
    """cmd_train's bidpo own-pool inputs: ``(harmful_pairs,
    benign_pairs, meta)`` — or ``(None, None,
    None)`` when the feature is off or no DPO preference arm
    (:data:`DPO_PREF_ARMS`) is among the requested arms.
    Fails in milliseconds, BEFORE the model load, when the feature is on but
    a file or the sidecar block is missing: an orchestration gap must cost
    seconds, not a model load whose bidpo arm then fails hours in.

    Default locations are the sibling ``bidpo_harmful_<si>.jsonl`` +
    ``bidpo_benign_<si>.jsonl`` emit-pairs writes beside ``--pairs`` (so the
    overnight pairs-dir rsync needs no new plumbing); ``--bidpo-harmful-
    pairs`` / ``--bidpo-benign-pairs`` override. Each file's bytes are
    VERIFIED against the sha256 the emit sidecar recorded — the manifest is
    authoritative, and a hand-swapped or stale file refuses instead of
    silently training a different pool. The sidecar's per-SI counts +
    projected ratios + refusal-label setting are REQUIRED (section 13): they
    ride into the object's meta.json so a bidpo cross-SI number can never be
    read without them."""
    import build_two_si_pool as bp

    bb_cfg = bp._bidpo_benign_cfg(cfg)
    if not bb_cfg["enabled"] or not (DPO_PREF_ARMS & set(arms)):
        return None, None, None
    hpath = (Path(args.bidpo_harmful_pairs) if args.bidpo_harmful_pairs
             else Path(args.pairs).parent / bp.bidpo_harmful_filename(args.si))
    bpath = (Path(args.bidpo_benign_pairs) if args.bidpo_benign_pairs
             else Path(args.pairs).parent / bp.bidpo_benign_filename(args.si))
    missing = [str(p) for p in (hpath, bpath) if not p.exists()]
    if missing:
        raise SystemExit(
            "[train-two-si] pool.bidpo_benign.enabled but bidpo's own pool "
            f"file(s) are missing: {missing} — run build_two_si_pool.py "
            "emit-pairs (with the feature enabled at manifest time, which "
            "writes bidpo_harmful_<si>.jsonl + bidpo_benign_<si>.jsonl "
            "beside the pairs file), or pass --bidpo-harmful-pairs / "
            "--bidpo-benign-pairs. Refusing to silently train a matched-pool "
            "bidpo wearing the section-13 configuration.")
    sidecar_path = bpath.parent / bp.PAIRS_META_NAME
    block = None
    if sidecar_path.exists():
        block = json.loads(sidecar_path.read_text()).get("bidpo_benign")
    if not block:
        raise SystemExit(
            f"[train-two-si] {bpath.name} has no {bp.PAIRS_META_NAME} sidecar "
            "with a bidpo_benign block beside it — the per-SI counts, "
            "projected ratios and the refusal-label setting must ride into "
            "each object's meta.json (section 13: a bidpo cross-SI number is "
            "never presented without them). Re-run emit-pairs rather than "
            "training without the confound record.")
    files = block.get("files", {}).get(args.si, {})
    for side, path in (("harmful", hpath), ("benign", bpath)):
        expected = files.get(side, {}).get("file_sha256")
        if expected is None:
            raise SystemExit(
                f"[train-two-si] the sidecar records no {side} file sha for "
                f"--si {args.si} — the pairs dir predates the section-13 "
                "emit; re-run emit-pairs.")
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != expected:
            raise SystemExit(
                f"[train-two-si] {path.name} does not match the sha256 the "
                f"emit sidecar froze ({got[:12]}… != {expected[:12]}…) — a "
                "hand-swapped or stale bidpo pool file; re-run emit-pairs "
                "rather than training on an unverified pool.")
    meta = {
        "refusal_labels": block["refusal_labels"],
        "per_si_frozen": block["per_si"],
    }
    return _load_pairs(str(hpath)), _load_pairs(str(bpath)), meta



def _token_sidecar_path(pairs_path: str) -> Path:
    """Sidecar convention: the token cache lives NEXT TO the pool file it
    was built from, so shipping a pool to a pod naturally ships its
    tokens (and node-local pools get node-local caches for free)."""
    return Path(str(pairs_path) + ".tokcache.npz")


def _resolve_teacher_topk(args) -> str | None:
    """The ``--teacher-topk`` flag -> the artifact path train_si_objects
    should try, or None (online teacher). 'auto' (the default) looks for
    the sidecar beside ``--pairs`` (the tokcache convention: shipping a
    pool ships its teacher) and says out loud when none is there — the
    normal state until the precompute-teacher step has run for that pool.
    An EXPLICIT path is passed through even when missing, so its failure
    is the loud key-check fallback rather than a silent 'auto' miss."""
    from sisteer.extraction.teacher_topk import topk_artifact_path

    flag = str(getattr(args, "teacher_topk", "auto") or "auto")
    if flag == "off":
        return None
    if flag != "auto":
        return flag
    p = topk_artifact_path(args.pairs)
    if p.is_file():
        return str(p)
    print(f"[teacher-topk] no artifact beside {args.pairs} — ONLINE "
          "teacher (run `precompute-teacher` once per (model, SI) to "
          "remove the training-step teacher forward)", flush=True)
    return None

def _warm_token_cache(args, cfg, tok) -> None:
    """Default-on token-cache warm-in for every fit entrypoint. A PRESENT
    sidecar that mismatches the tokenizer refuses the run (fail loudly —
    a wrong-world cache beside the pool is the stale-artifact bug shape);
    an ABSENT sidecar degrades to an in-process memo, which still
    collapses the preflight + encode re-tokenization within this arm."""
    from sisteer.extraction import tokcache

    flag = getattr(args, "token_cache", "auto") or "auto"
    if flag == "off":
        return
    path = Path(flag) if flag != "auto" else _token_sidecar_path(args.pairs)
    if path.exists():
        cache = tokcache.warm_from_sidecar(path, tok,
                                           expect_model_id=cfg["model"],
                                           expect_pool_path=args.pairs)
        tokcache.set_active(cache)
        print(f"[tokcache] warmed {len(cache)} entries from {path}")
    else:
        tokcache.set_active(tokcache.TokCache())
        print(f"[tokcache] no sidecar at {path}; in-process memo only")

#: The shared held-out pool's id in the SI slot
#: (build_pools.HELDOUT_CELL). It names an EVAL-ONLY pool, never a
#: training cell, so it is deliberately NOT a registered corpus
#: pseudo-instruction: precompute-tokens special-cases it below instead of
#: registering an SI no fit may ever be launched under.
HELDOUT_POOL_SI = "constv2-heldout"

#: EVERY sanctioned held-out pool id (the set the special case below
#: honours): the constitution corpus's shared held-out pool above, and the
#: control corpus's held-out pool ``constv2ctl2-heldout`` (build_ctl_heldout;
#: the sdfbase-llama8b lane scores BOTH held-out pools per fit,
#: so its staging builds a sidecar for each). Eval-only, deliberately
#: unregistered, like the first; HELDOUT_POOL_SI keeps its historical
#: meaning and spelling for every reader that names the constitution pool.
HELDOUT_POOL_SIS = frozenset({HELDOUT_POOL_SI, "constv2ctl2-heldout"})


def build_token_sidecar(cfg, si_id: str, pairs_path: str, out: str = "",
                        tok=None) -> Path:
    """Tokenize the pool ONCE — on any CPU, no GPU or model load — and
    write the sidecar a wave's fits warm from. Renders through the exact
    fit code path (pairs_to_distill_examples + the same tokenizer
    construction), so cached ids are bitwise what a fit would compute.
    Covers both fit legs: Class A and Class C tokenize identical strings
    (positions change loss masks, not ids). bidpo's own pool is NOT
    covered (its examples render separately); bidpo still benefits from
    the in-process memo.

    ``si_id in HELDOUT_POOL_SIS`` (doc mode only) is the SANCTIONED path
    for a held-out pool's sidecar (``constv2-heldout``, and the control
    corpus's ``constv2ctl2-heldout``): the id is not a registered
    instruction (it names no training cell), the doc-mode render consults
    no instruction anyway, and the staging step needs the identical
    sidecar next to ``pairs_<heldout>.jsonl`` — before this special-case
    the operator had to hand-replicate the build because
    resolve_instruction refused the unregistered id."""
    from sisteer.data.pairs import docs_to_ce_examples, pairs_to_distill_examples
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.extraction import tokcache
    from sisteer.models import load_tokenizer

    if tok is None:
        tok = load_tokenizer(cfg["model"], revision=cfg.get("model_revision"),
                             leading_special_tokens=_leading_policy(cfg))
    pairs = _load_pairs(pairs_path)
    if _data_mode(cfg) == DOC_DATA_MODE:
        # Document-mode branch: VERBATIM examples — the pairs
        # path below chat-renders (build_chat_prompt wraps under a shipped
        # template and strip+"\n"-mutates without one), so its cache keys
        # would never be the strings a doc-mode fit encodes. The
        # resolve_instruction render is deliberately not used; the
        # corpus-cell pseudo-instructions exist for cell naming only —
        # but the id must still resolve (typo'd cell ids fail here, in
        # milliseconds, not at fit time). The exceptions are the held-out
        # pool ids (HELDOUT_POOL_SIS): eval-only, deliberately
        # unregistered, stamped into the sidecar as-is.
        if si_id not in HELDOUT_POOL_SIS:
            resolve_instruction(si_id)
        examples = docs_to_ce_examples(tok, pairs)
    else:
        examples = pairs_to_distill_examples(tok, pairs,
                                             resolve_instruction(si_id))
    cache = tokcache.TokCache()
    prev = tokcache.active()
    tokcache.set_active(cache)
    try:
        for ex in examples:
            tokcache.cached_ids(tok, ex.prompt_with_si, True)
            tokcache.cached_ids(tok, ex.prompt_without, True)
            tokcache.cached_ids(tok, ex.completion, False)
    finally:
        tokcache.set_active(prev)
    path = Path(out) if out else _token_sidecar_path(pairs_path)
    header = tokcache.save_sidecar(
        path, cache, tok, model_id=cfg["model"],
        model_revision=cfg.get("model_revision"),
        pool_path=pairs_path,
        extra={"si_id": si_id, "n_pairs": len(pairs)})
    print(f"[tokcache] wrote {header['n_entries']} entries "
          f"({len(pairs)} pairs) -> {path}")
    return path

def cmd_precompute_tokens(args, cfg) -> None:
    build_token_sidecar(cfg, args.si, args.pairs, out=args.out)


def cmd_precompute_teacher(args, cfg) -> None:
    """Build the offline top-K teacher artifact for one (model, SI, pool):
    the SI-prompted frozen teacher run ONCE (bs=1, unpadded, ``no_grad``,
    the config's dtype) over every pool pair, top-``--top-k`` logits kept
    at every decode-only (completion-predicting) position, written as the
    ``<pairs>.teachertopk.safetensors`` sidecar the fits' ``--teacher-topk
    auto`` picks up. Deterministic — the teacher is a pure function of
    (model, revision, dtype, SI, pool, max_len), all of which key the
    manifest — and object-free: no steering arm exists yet at this point,
    and the teacher never depends on one."""
    sha = _assert_expected_config_sha(args)
    from sisteer.data.pairs import pairs_to_distill_examples
    from sisteer.eval.jailbreak import resolve_instruction
    from sisteer.eval.train_budget import resolve_max_len
    from sisteer.extraction.distill import preflight_max_len
    from sisteer.extraction.fitmon import pin_attention_backends
    from sisteer.extraction.subsample import sha_rank_indices
    from sisteer.extraction.teacher_topk import (
        compute_teacher_topk, head_mass_stats, instruction_sha,
        topk_artifact_path)
    from sisteer.models import load_model_and_tokenizer

    print(_resolved_banner(cfg, sha, "n/a (teacher precompute)",
                           str(getattr(args, "config", "") or "")),
          flush=True)
    pairs = _load_pairs(args.pairs)
    instruction = resolve_instruction(args.si)
    max_len = resolve_max_len(cfg, args.si)
    shard_i = int(getattr(args, "shard_index", -1))
    shard_n = int(getattr(args, "shard_count", 0) or 0)
    if (shard_i >= 0) != (shard_n > 0):
        raise SystemExit(
            "[teacher-topk] --shard-index and --shard-count come together")
    if shard_n and not (0 <= shard_i < shard_n):
        raise SystemExit(
            f"[teacher-topk] --shard-index {shard_i} out of range for "
            f"--shard-count {shard_n}")
    if shard_n and getattr(args, "limit", 0):
        raise SystemExit(
            "[teacher-topk] --limit and --shard-* are mutually exclusive "
            "(a limit subset re-keys the pool; a shard keeps the full-pool "
            "key and slices the work)")
    pin_attention_backends()
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        leading_special_tokens=_leading_policy(cfg))
    _warm_token_cache(args, cfg, tok)
    limit = int(getattr(args, "limit", 0) or 0)
    if limit:
        # A verification subset: sha-ranked over the student rendering
        # (never a family-sorted prefix — the probe-selection lesson). The
        # manifest hashes the SUBSET, so this artifact can only ever be
        # consumed by a fit over the same subset — a full-pool fit's key
        # check refuses it.
        keys = [p.prompt for p in pairs]
        pairs = [pairs[i] for i in sha_rank_indices(keys, limit)]
        print(f"[teacher-topk] --limit {limit}: building over a "
              f"{len(pairs)}-pair sha-ranked subset (manifest hashes the "
              "subset; not consumable by full-pool fits)", flush=True)
    # A SHARD hashes and keys the WHOLE pool (the merge validates every
    # shard carries the same full-pool identity) and slices only the WORK
    # — contiguous example ranges in pool order, so the merged
    # concatenation is byte-identical to the unsharded loop's append order.
    full_content_sha = pool_content_sha256(pairs)
    shard_rec = None
    if shard_n:
        n = len(pairs)
        bounds = [round(j * n / shard_n) for j in range(shard_n + 1)]
        lo, hi = bounds[shard_i], bounds[shard_i + 1]
        shard_rec = {"index": shard_i, "of": shard_n, "lo": lo, "hi": hi}
        pairs = pairs[lo:hi]
        print(f"[teacher-topk] shard {shard_i}/{shard_n}: examples "
              f"[{lo}, {hi}) of {n}", flush=True)
    examples = pairs_to_distill_examples(tok, pairs, instruction)
    preflight_max_len(tok, examples, max_len)
    meta = {
        "model": cfg.get("model"),
        "model_revision": cfg.get("model_revision"),
        "dtype": cfg.get("dtype"),
        "si_id": args.si,
        "instruction_sha256": instruction_sha(instruction.text),
        "pool_content_sha256": full_content_sha,
        "pool_file_sha256": _sha256_file(Path(args.pairs)),
        "pool_path": str(args.pairs),
        "max_len": int(max_len),
        "limit": limit or None,
        "config_sha": sha,
        "positions": "completion_decode_only",
    }
    if shard_rec is not None:
        meta["shard"] = shard_rec
    from sisteer.extraction.teacher_topk import DEFAULT_TOP_K

    top_k = int(getattr(args, "top_k", 0) or 0) or DEFAULT_TOP_K
    store = compute_teacher_topk(model, tok, examples, max_len=max_len,
                                 top_k=top_k, meta=meta)
    if args.out:
        out = Path(args.out)
    elif shard_rec is not None:
        from sisteer.extraction.teacher_topk import shard_artifact_path

        out = shard_artifact_path(args.pairs, shard_i, shard_n)
    else:
        out = topk_artifact_path(args.pairs)
    store.save(out)
    stats = head_mass_stats(store)
    print(f"[teacher-topk] wrote {out} (+ .json): top_k={store.top_k}, "
          f"{len(store)} examples, {int(store.idx.shape[0])} positions, "
          f"T=1 head mass mean={stats['mean']:.6f} min={stats['min']:.6f} "
          f"p1={stats['p1']:.6f} p50={stats['p50']:.6f}", flush=True)


def cmd_merge_teacher(args, cfg) -> None:
    """Merge `precompute-teacher --shard-index/--shard-count` shard
    artifacts into the ONE final sidecar the fits consume — CPU only, no
    model load. The merge is byte-identical to an unsharded build on the
    same hardware (pinned by tests/test_teacher_topk.py); every shard's
    full-pool identity, top_k and contiguous coverage are validated before
    a byte is written, and shard files are deleted after the final save
    unless --keep-shards."""
    from sisteer.extraction.teacher_topk import (
        TeacherTopK, head_mass_stats, merge_teacher_topk,
        shard_artifact_path, topk_artifact_path)

    n = int(args.shard_count)
    if n < 1:
        raise SystemExit("[merge-teacher] --shard-count must be >= 1")
    paths = [shard_artifact_path(args.pairs, i, n) for i in range(n)]
    missing = [str(p) for p in paths if not p.is_file()
               or not Path(str(p) + ".json").is_file()]
    if missing:
        raise SystemExit(
            f"[merge-teacher] {len(missing)}/{n} shard artifact(s) missing "
            f"(first: {missing[0]}) — a partial merge would silently drop "
            "pool examples; build every shard first")
    store = merge_teacher_topk([TeacherTopK.load(p) for p in paths])
    out = Path(args.out) if args.out else topk_artifact_path(args.pairs)
    store.save(out)
    stats = head_mass_stats(store)
    print(f"[merge-teacher] wrote {out} (+ .json): top_k={store.top_k}, "
          f"{len(store)} examples, {int(store.idx.shape[0])} positions "
          f"from {n} shards; T=1 head mass mean={stats['mean']:.6f} "
          f"min={stats['min']:.6f} p1={stats['p1']:.6f}", flush=True)
    if not getattr(args, "keep_shards", False):
        for p in paths:
            p.unlink(missing_ok=True)
            Path(str(p) + ".json").unlink(missing_ok=True)
        print(f"[merge-teacher] removed {n} shard files", flush=True)


def cmd_train(args, cfg) -> None:
    # Stale-config guard before EVERYTHING (even the imports below): the
    # wave driver passes --expect-config-sha so a launch tarball that
    # predates the intended config refuses here, in milliseconds.
    sha = _assert_expected_config_sha(args)

    from sisteer.models import load_model_and_tokenizer

    # getattr, not args.positions: programmatic callers (tests, the
    # orchestrator) build bare Namespaces that predate the flag.
    pos_flag = getattr(args, "positions", "")
    positions = _effective_positions(args, cfg)
    if pos_flag:
        cfg = _cfg_with_positions(cfg, pos_flag)
    if positions != "all" and not args.out:
        # Fail in milliseconds, BEFORE any pair/model load: a masked (Class
        # C) wave pointed at the default/env defenses dir would collide with
        # the Class A wave at the SAME cell ids — resume would silently skip
        # every arm (counting an all-token artifact as this wave's fit) and
        # --overwrite would clobber the Class A objects. An explicit --out
        # makes the Class C store a deliberate, separate choice.
        raise SystemExit(
            f"[train-two-si] a masked (positions={positions!r} — Class C, "
            "decode-only) fit wave requires an explicit --out objects dir "
            "of its own: the default/env defenses dir holds the Class A "
            "(all-token) artifacts at the SAME cell ids, so resume there "
            "silently counts an all-token artifact as this wave's fit and "
            "--overwrite clobbers the Class A wave.")
    if positions != "all" and args.out:
        # F15's second half of that guard: --out PRESENT is not enough —
        # passing the Class A store's own path as the Class C --out
        # re-created the exact collision the refusal above exists to
        # prevent. Refuse an --out that RESOLVES to the Class A dir.
        from sisteer.eval.defense_cell import resolve_defenses_dir

        try:
            class_a_store = resolve_defenses_dir(
                cfg, repo_root=REPO).resolve()
        except KeyError:
            class_a_store = None  # cfg names no Class A store to collide with
        if (class_a_store is not None
                and Path(args.out).resolve() == class_a_store):
            raise SystemExit(
                f"[train-two-si] a masked (positions={positions!r} — Class "
                f"C) fit's --out resolves to the Class A store itself "
                f"({class_a_store}): the two legs share cell ids, so this "
                "would count all-token artifacts as decode-only fits and "
                "clobber them under --overwrite. Give the Class C wave a "
                "dir of its own.")
    # A decode-only wave defaults to the Class C arm set (the round-trip-
    # gated membership recorded in configs/two_si.yaml class_c_arms);
    # --arms still overrides for a partial wave.
    arms = (args.arms.split(",") if args.arms else
            (default_class_c_arms(cfg) if positions == "response"
             else default_class_a_arms(cfg)))
    # Checkpoint continuation: both flags, or neither (the driver refuses
    # one without the other inside train_si_objects).
    continue_from = getattr(args, "continue_from", "") or None
    epochs_cap = int(getattr(args, "epochs_cap", 0) or 0) or None
    continue_lr_mult = _continue_lr_mult_arg(args)
    dp = cfg.get("data_parallel") or {}
    if ((continue_from or epochs_cap or continue_lr_mult != 1.0)
            and int(dp.get("gpus", 1) or 1) > 1):
        raise SystemExit(
            "[train-two-si] --continue-from/--epochs-cap/--continue-lr-mult "
            "run the single-process path only (one CE cell per invocation); "
            "this config's data_parallel.gpus > 1 would route through "
            "arm-task, which carries none of the flags")
    if int(dp.get("gpus", 1) or 1) > 1:
        # Parallel driver: one process per arm on the job's GPU pool
        # (arms pack 1-per-GPU — see run_parallel_fit). Dispatched AFTER
        # the positions/arm-set resolution above so a Class C wave
        # schedules the Class C arm set (and inherits its --out refusal).
        if _fit_width_expected(args, cfg) > 1 or os.environ.get(
                "WORLD_SIZE", "1") not in ("", "1"):
            raise SystemExit(
                "[train-two-si] data_parallel.gpus > 1 (the in-job arm "
                "scheduler) cannot combine with a data-parallel fit width "
                "(config fit_width / --expected-width > 1, or a torchrun "
                "launch) — one or the other.")
        from sisteer.eval.defense_cell import resolve_defenses_dir

        save_dir = Path(args.out) if args.out else resolve_defenses_dir(
            cfg, repo_root=REPO)
        raise SystemExit(run_parallel_fit(args, cfg, arms, save_dir))
    # The data-parallel gate: under `torchrun --nproc_per_node=N`
    # every rank runs this whole entrypoint — the trainer shards each
    # optimizer step's micro-batches across the ranks; rank 0 owns every
    # write. A config that declares fit_width > 1 REFUSES to run outside
    # the launcher (see _enter_fit_process_group). Entered BEFORE the model
    # load so torch.cuda.set_device(LOCAL_RANK) places each rank's copy.
    if continue_from or epochs_cap or continue_lr_mult != 1.0:
        if os.environ.get("WORLD_SIZE", "1") not in ("", "1") \
                or _fit_width_expected(args, cfg) > 1:
            raise SystemExit(
                "[train-two-si] --continue-from/--epochs-cap/"
                "--continue-lr-mult run at fit width 1 only (the "
                "continuation adopts a checkpoint into this fit's slot — a "
                "single-process step); pass --expected-width 1 and run "
                "outside torchrun.")
    par = _enter_fit_process_group(args, cfg, arms, entry="train")
    is_main = par is None or par.is_main
    # Single-process path (or one rank of the data-parallel one): the same
    # resolved-config first line the parallel scheduler prints.
    print(_resolved_banner(cfg, sha, positions,
                           str(getattr(args, "config", "") or "")),
          flush=True)
    # CE/document config-shape guard BEFORE the model load (milliseconds,
    # not minutes; train_si_objects re-runs it as the authoritative choke
    # point every entrypoint crosses).
    refuse_ce_misconfiguration(cfg, arms)
    pairs = _load_pairs(args.pairs)
    # Fit identity for the fingerprint-bound resume (F15): the pool file's
    # byte hash + the config file's sha, the one CLI derivation — derived
    # HERE, before the model load, because the sample manifest's validation
    # needs the pool hash.
    pool_sha, config_sha = _cli_fit_identity(args, cfg)
    # The sample manifest beside a document pool, resolved and VALIDATED
    # (its cell is this SI, its pool_sha256 is this pool's bytes) BEFORE the
    # model load — milliseconds, not minutes — through the ONE resolver
    # train_si_objects records from and the Slurm lane preflights with; a
    # stale or foreign manifest is a SystemExit here, never a sidecar row.
    if _data_mode(cfg) == DOC_DATA_MODE:
        _sample_manifest_sha_for(args.pairs, cell=args.si, pool_sha256=pool_sha)
    # bidpo's own pool — resolved, sha-verified and validated BEFORE the
    # model load (milliseconds, not hours).
    bidpo_harmful_pairs, bidpo_benign_pairs, bidpo_benign_meta = \
        _load_bidpo_pool(args, cfg, arms)
    # cuDNN-SDPA off in the production fit path (per-ragged-shape
    # cuDNN plan compilation on CPU halved GPU duty; see fitmon).
    from sisteer.extraction.fitmon import pin_attention_backends

    pin_attention_backends()
    # ONE model load for all arms of this SI (opt. 3).
    model, tok = load_model_and_tokenizer(
        cfg["model"], revision=cfg.get("model_revision"), dtype=cfg["dtype"],
        attn_implementation=cfg.get("attn_implementation", "sdpa"),
        leading_special_tokens=_leading_policy(cfg))
    maybe_enable_gradient_checkpointing(model, cfg, arms=arms)
    # --out (the orchestrator's absolute commit-independent dir) wins;
    # otherwise the ONE shared resolution (TT_TWO_SI_DEFENSES_DIR on the
    # cluster, repo-rooted locally) — never a bare checkout-relative default.
    from sisteer.eval.defense_cell import resolve_defenses_dir

    save_dir = Path(args.out) if args.out else resolve_defenses_dir(
        cfg, repo_root=REPO)
    # Metrics-only live logging: numeric
    # scalars stream to W&B when the env allows it (WANDB_DISABLED unset,
    # console capture forced off), and EVERY step also lands in a per-SI
    # jsonl beside the fit report so a W&B outage cannot lose the curves.
    # The console echo (console_log — the wandb.log-signature shim) stays,
    # so job logs keep the per-step lines.
    # Optimiser transparency, before a single step runs: refuse config keys an
    # arm would silently ignore, then state the rate each arm will ACTUALLY
    # train under and where it came from. Both exist because a pinned 5e-4 was
    # dropped for a lockstep arm and a whole wave trained at 5e-3 with the
    # config reading correct throughout.
    _warm_token_cache(args, cfg, tok)
    refuse_silently_dropped_kwargs(cfg, arms)
    for _line in optimizer_banner(cfg, args.si, arms):
        print(_line, flush=True)

    # Rank 0 owns the metrics stream (W&B, the per-SI jsonl, the console
    # echo); a replica logs nothing — its trainer receives log_fn=None.
    wandb_run = (wandb_fit_run(cfg, args.si, arms,
                               positions=getattr(args, "positions", None))
                 if is_main else None)
    latest_metrics: dict = {}
    log_fn = (make_metrics_log_fn(
        jsonl_path=metrics_path_for(save_dir, args.si),
        wandb_run=wandb_run, echo=console_log, latest=latest_metrics,
        arms=arms) if is_main else None)
    if wandb_run is not None:
        # A sweep monitor's scancel must leave a readable summary row, not
        # a bare crashed run (see _flush_summary_on_sigterm).
        _flush_summary_on_sigterm(wandb_run, latest_metrics)
    # (pool_sha / config_sha: derived above, before the model load.)
    # --profile <dir> (pod_pilot.sh step 7): a torch.profiler TensorBoard
    # trace gated around the training steps, ticked per optimizer-step
    # record through the composite log_fn.
    prof = (start_profiler(args.profile)
            if getattr(args, "profile", "") and is_main else None)
    if prof is not None:
        log_fn = profiler_stepping_log_fn(log_fn, prof)
    outcome = None
    try:
        outcome = train_si_objects(model, tok, args.si, pairs, arms, cfg=cfg,
                                   save_dir=save_dir, log_fn=log_fn,
                                   overwrite=args.overwrite,
                                   bidpo_harmful_pairs=bidpo_harmful_pairs,
                                   bidpo_benign_pairs=bidpo_benign_pairs,
                                   bidpo_benign_meta=bidpo_benign_meta,
                                   pool_sha256=pool_sha,
                                   config_sha=config_sha,
                                   teacher_topk=_resolve_teacher_topk(args),
                                   pairs_path=args.pairs,
                                   checkpoint_dir=(
                                       getattr(args, "checkpoint_dir", "")
                                       or None),
                                   checkpoint_every=(
                                       int(getattr(args, "checkpoint_every",
                                                   0) or 0) or None),
                                   epochs_cap=epochs_cap,
                                   continue_from=continue_from,
                                   continue_lr_mult=continue_lr_mult)
    finally:
        if prof is not None:
            prof.__exit__(None, None, None)
        if wandb_run is not None:
            if outcome is not None:
                # The leaderboard row — floor_ratio and friends land in
                # run.summary PASS OR FAIL (the guard's refusal is caught
                # per-arm upstream; the ratio is the sweep's objective
                # either way).
                _write_fit_summary(wandb_run, outcome, arms)
            wandb_run.finish()
        _leave_fit_process_group()
    print(f"[train-two-si] SI={args.si}: trained {len(outcome.objects)}, "
          f"skipped {len(outcome.skipped)} already on disk -> {save_dir} "
          + ("(one model load)" if par is None else
             f"(rank {par.rank}/{par.world_size}; rank 0 published)"))
    if not outcome.ok:
        # Every arm was attempted and the good ones are saved; NOW fail the
        # job so the failure is visible (never silently, never earlier).
        for arm, res in outcome.failed.items():
            print(f"[train-two-si] FAILED {args.si}__{arm}: "
                  f"{res.error_type}: {res.error}", file=sys.stderr)
        print(f"[train-two-si] {len(outcome.failed)} arm(s) failed after every "
              f"arm was attempted; report: {outcome.report_path}",
              file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan")
    p.add_argument("--config", default="configs/two_si.yaml")
    p.add_argument("--si", required=True)
    p.add_argument("--arms", default="", help="comma-separated; default = the "
                   "registered Class A arms (Class C under "
                   "--positions response)")
    p.add_argument("--pairs", default="", help="optional jsonl to measure the "
                   "budget table against real prompts")
    p.add_argument("--with-model", action="store_true",
                   help="load the tokenizer to measure the budget table")
    p.add_argument("--positions", default="",
                   help="fit-position override, as for `train` (with no "
                        "--arms, 'response' plans the config's class_c_arms)")

    t = sub.add_parser("train")
    t.add_argument("--config", default="configs/two_si.yaml")
    t.add_argument("--si", required=True)
    t.add_argument("--arms", default="")
    t.add_argument("--pairs", required=True, help="jsonl of {prompt,pos,neg}")
    t.add_argument("--token-cache", default="auto",
                   help="'auto' (use/build the sidecar beside --pairs), "
                        "'off' (no cache), or an explicit sidecar path")
    t.add_argument("--teacher-topk", default="auto",
                   help="precomputed top-K teacher artifact: 'auto' (the "
                        "sidecar beside --pairs, "
                        "<pairs>.teachertopk.safetensors, when present), "
                        "'off' (online teacher), or an explicit path. A "
                        "present artifact is key-checked (model+revision, "
                        "SI, pool content sha, max_len, top_k, dtype) and "
                        "row-sha-bound; any mismatch falls back to the "
                        "online teacher loudly. Consumed by the T=1 "
                        "forward-KL lockstep arms only (build it with the "
                        "`precompute-teacher` subcommand).")
    t.add_argument("--positions", default="",
                   help="fit-position override — the Class C submission "
                        "path: 'response' fits the DECODE-ONLY leg of every "
                        "requested arm, overriding train_kwargs.default."
                        "positions (response_header_len stays config-"
                        "resolved: null derives H from the tokenizer). It "
                        "flows into the builders (which refuse per arm when "
                        "an object cannot honour a masked fit) and into "
                        "each artifact's positions/positions_at_fit "
                        "provenance record; with no --arms it selects the "
                        "config's class_c_arms instead of class_a_arms. "
                        "Requires --out (a Class C objects dir of its own). "
                        "Default: the config's value ('all').")
    t.add_argument("--out", default="", help="ABSOLUTE defenses_dir override "
                   "(D58: the orchestrator passes the commit-independent "
                   "objects dir at submit time, so resume survives code "
                   "changes)")
    t.add_argument("--overwrite", action="store_true",
                   help="retrain every requested arm, ignoring objects "
                        "already on disk (default: resume — skip them)")
    t.add_argument("--bidpo-harmful-pairs", default="",
                   help="bidpo's own HARMFUL pairs jsonl (spec-bidpo-benign "
                        "section 13: this SI's own thwart set; pos = the "
                        "SI's refusal, neg = the accepted anti compliance). "
                        "Default when pool.bidpo_benign is enabled: the "
                        "sibling bidpo_harmful_<si>.jsonl emit-pairs writes "
                        "beside --pairs. Consumed by bidpo and ONLY bidpo.")
    t.add_argument("--bidpo-benign-pairs", default="",
                   help="bidpo's benign-REVERSED pairs jsonl (spec-bidpo-"
                        "benign: chosen = the undefended compliance, rejected "
                        "= this SI's refusal). Default when pool.bidpo_benign "
                        "is enabled: the sibling bidpo_benign_<si>.jsonl "
                        "emit-pairs writes beside --pairs. Consumed by bidpo "
                        "and ONLY bidpo.")
    t.add_argument("--expect-config-sha", default="",
                   help="16-hex sha256 of the config FILE BYTES this fit "
                        "must be running (the [config-resolved] banner "
                        "value; a full 64-hex sha is accepted). The wave "
                        "driver computes it from the config it INTENDS, so "
                        "a launch tarball that predates that config "
                        "refuses in milliseconds instead of silently "
                        "fitting stale widths all night.")
    t.add_argument("--profile", default="",
                   help="TensorBoard-trace DIRECTORY: capture a "
                        "torch.profiler trace of the training steps "
                        "(tune_arm.py's schedule — wait=20, warmup=3, "
                        "active=12) into it. The pod pilot passes its "
                        "tb_trace dir here; empty (default) = no "
                        "profiling.")
    t.add_argument("--checkpoint-dir", default="",
                   help="mid-fit durability (fit-wave hardening): a ROOT "
                        "dir for per-cell trainer checkpoints "
                        "(<root>/<cell_id>/ckpt.pt, latest + previous, "
                        "atomic). Keyed to each cell's fit fingerprint; "
                        "on start a matching checkpoint resumes the fit "
                        "EXACTLY, a mismatched one is ignored loudly. "
                        "Keep it OUTSIDE the objects dir — checkpoints "
                        "are transient and must never meet the "
                        "publication audit. Empty (default) = off.")
    t.add_argument("--checkpoint-every", type=int, default=0,
                   help="optimizer steps between checkpoints when "
                        "--checkpoint-dir is set (0/default = the "
                        "production cadence, fitckpt."
                        "DEFAULT_CHECKPOINT_EVERY = 100).")
    t.add_argument("--expected-width", default="",
                   help="D96 data-parallel fits: the rank count this "
                        "process must find itself in (torchrun's "
                        "WORLD_SIZE); default = the config's top-level "
                        "fit_width (absent = 1). Any mismatch refuses by "
                        "name; a wide config outside torchrun refuses too "
                        "— pass 1 to run one rank deliberately.")
    t.add_argument("--continue-from", default="",
                   help="R2-D20: a COMPLETE cell's retained <cell>/final_ckpt/"
                        "ckpt.pt to CONTINUE under --epochs-cap — exactly one "
                        "document-CE arm, --checkpoint-dir required. The "
                        "checkpoint's fit identity must equal this config's "
                        "(only the cap may differ), the cell's publication "
                        "must verify against it, and the continued cell "
                        "publishes under a fingerprint carrying the "
                        "continuation marker (sisteer.extraction."
                        "continuation). R2-D77: the replayed remainder of "
                        "the checkpoint epoch is logged but never a strike "
                        "nor a new best (stopper rule replay_exempt, in the "
                        "marker).")
    t.add_argument("--epochs-cap", type=int, default=0,
                   help="R2-D20: the raised epoch cap for --continue-from "
                        "(must exceed the prior cap: the config's, or for a "
                        "chained continuation the last continuation's). "
                        "Refused without --continue-from: a scratch fit at "
                        "another cap is a config change.")
    t.add_argument("--continue-lr-mult", type=float, default=1.0,
                   help="R2-D44: multiply the continuation's restarted "
                        "schedule PEAK by this (warmup + cosine shape "
                        "unchanged; every step's lr is exactly this x the "
                        "unscaled value). Rides in the continuation "
                        "fingerprint and provenance. Default 1.0 = the "
                        "R2-D20 warm restart. Refused without "
                        "--continue-from.")

    # The scheduler's per-arm unit of work (see run_parallel_fit) — spawned
    # by `train` when data_parallel.gpus > 1; not for humans to invoke.
    at = sub.add_parser("arm-task")
    at.add_argument("--config", default="configs/two_si.yaml")
    at.add_argument("--si", required=True)
    at.add_argument("--arm", required=True)
    at.add_argument("--pairs", required=True)
    at.add_argument("--out", default="")
    at.add_argument("--expected-width", default="1")
    at.add_argument("--overwrite", action="store_true")
    # The scheduler forwards these so an arm task resolves the SAME fit leg
    # and the SAME bidpo pool cmd_train would have: arm-task re-reads the
    # config FILE, so an in-memory positions override in the scheduler is
    # invisible here unless passed explicitly.
    at.add_argument("--positions", default="")
    at.add_argument("--bidpo-harmful-pairs", default="")
    at.add_argument("--bidpo-benign-pairs", default="")
    at.add_argument("--token-cache", default="auto",
                    help="'auto' (sidecar beside --pairs), 'off', or a path")
    at.add_argument("--teacher-topk", default="auto",
                    help="as for `train` — forwarded by the scheduler so "
                         "every arm task consumes (or refuses) the same "
                         "precomputed teacher cmd_train resolved")
    at.add_argument("--profile", default="",
                    help="as for `train` — forwarded by the scheduler so a "
                         "profiled parallel wave traces its arm tasks (the "
                         "TB handler names files per host+pid, so tasks "
                         "share one dir)")
    at.add_argument("--expect-config-sha", default="",
                    help="as for `train`. The scheduler always forwards a "
                         "pinned config sha, because an arm task re-reads "
                         "the config FILE and must refuse to resolve a "
                         "different world than the scheduler planned.")
    at.add_argument("--checkpoint-dir", default="",
                    help="as for `train` — forwarded by the scheduler so a "
                         "checkpointed parallel wave's arm tasks carry the "
                         "same durability the single-process path has")
    at.add_argument("--checkpoint-every", type=int, default=0,
                    help="as for `train`")

    pt = sub.add_parser("precompute-tokens")
    pt.add_argument("--config", default="configs/two_si.yaml")
    pt.add_argument("--si", required=True)
    pt.add_argument("--pairs", required=True)
    pt.add_argument("--out", default="", help="default: <pairs>.tokcache.npz")

    tt = sub.add_parser(
        "precompute-teacher",
        help="run the SI-prompted teacher ONCE over the pool (bs=1, "
             "no_grad, bf16 per the config) and store the top-K logits at "
             "every decode-only position — the artifact the fits consume "
             "instead of an online teacher forward per step")
    tt.add_argument("--config", default="configs/two_si.yaml")
    tt.add_argument("--si", required=True)
    tt.add_argument("--pairs", required=True)
    tt.add_argument("--out", default="",
                    help="default: <pairs>.teachertopk.safetensors "
                         "(+ .json manifest)")
    tt.add_argument("--top-k", type=int, default=0,
                    help="head width; 0/unset = teacher_topk.DEFAULT_TOP_K "
                         "(the one production width consumers validate "
                         "against)")
    tt.add_argument("--token-cache", default="auto",
                    help="as for `train` — warms the tokenizer sidecar so "
                         "the encode phase is not paid twice")
    tt.add_argument("--limit", type=int, default=0,
                    help="build over only the first N sha-ranked pairs (a "
                         "verification subset). The manifest then hashes "
                         "the SUBSET, so a partial artifact can never be "
                         "silently consumed by a full-pool fit.")
    tt.add_argument("--expect-config-sha", default="",
                    help="as for `train` — the artifact must be built from "
                         "the exact config world the fits will run")
    tt.add_argument("--shard-index", type=int, default=-1,
                    help="0-based shard to build (with --shard-count): a "
                         "contiguous example range of the FULL pool, keyed "
                         "to the full pool, written as a .shardNNNofNNN "
                         "sidecar for `merge-teacher`")
    tt.add_argument("--shard-count", type=int, default=0)

    mt = sub.add_parser(
        "merge-teacher",
        help="merge precompute-teacher shard artifacts into the final "
             "sidecar (CPU, no model; byte-identical to an unsharded "
             "build)")
    mt.add_argument("--config", default="configs/two_si.yaml")
    mt.add_argument("--pairs", required=True)
    mt.add_argument("--shard-count", type=int, required=True)
    mt.add_argument("--out", default="",
                    help="default: <pairs>.teachertopk.safetensors")
    mt.add_argument("--keep-shards", action="store_true")

    args = ap.parse_args()
    cfg = _load_cfg(args.config)
    {"plan": cmd_plan, "train": cmd_train,
     "arm-task": cmd_arm_task,
     "precompute-tokens": cmd_precompute_tokens,
     "precompute-teacher": cmd_precompute_teacher,
     "merge-teacher": cmd_merge_teacher}[args.cmd](args, cfg)


if __name__ == "__main__":
    main()
