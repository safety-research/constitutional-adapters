#!/usr/bin/env python3
"""The fit configs and the task table of the Qwen3.5-4B system-instruction distillation refits (``sikl4``).

Derives, from ``configs/two_si_qwen35_4b_fits.yaml`` (the published per-kind recipe):

- ``configs/sikl4_qwen35_4b_fits_s{1,2,3}_lr{0,1,2}.yaml`` — one config per (fit seed, learning-rate step);
- ``configs/sikl4_qwen35_4b_fits_smoke.yaml`` — the mechanics smoke (seed 1, step 0, one epoch);
- ``configs/sikl4/fit_tasks.tsv`` — every job the refits can submit, one row each: 54 fit rows (3 SIs x 2 kinds x 3 seeds x
  steps 0/1/2), 3 teacher rows and 3 smoke rows. The FIRST column is the row id, which is the job name the row runs under
  (``scripts/sbatch/sikl4_fit.sbatch`` refuses any other).

Each fit config differs from the source in exactly these keys (``DELTA_KEYS``):

- ``si_ids`` = safety_refusal, si_constitution, C16, and ``max_len_by_si.C16`` = 8192;
- ``train_kwargs.default.positions`` = all (the adapter acts at every token; the KL loss stays on the response tokens) and
  ``train_kwargs.default.seed`` = the fit seed;
- ``train_kwargs.lora_r4.lr`` and ``train_kwargs.psr_kl_all.lr`` = the source's value x 2^(-n/2) at learning-rate step n (step 0 is
  the source literally; steps 1 and 2 are the half-octave steps a divergence fallback selects, pre-generated so the fallback is a
  choice of rows, never a new config);
- ``registry_paper_label`` = other and ``registry_prompt_pool_ids`` (the registry statement train_two_si_objects.py records);
- the smoke config also sets ``train_kwargs.default.epochs`` = 1.

The W&B project is not a config key here (the source's ``$WANDB_PROJECT`` placeholder stands): the job environment names it.

Every path in the task table is relative: configs to the repository, pools / teachers / output roots to the cluster shared root
the job resolves (``$SHARED``). Each fit row has an output root of its own (``.../fits[_lr<n>]/s<k>/<si>__<arm>``; the object lands
in ``<root>/<si>__<arm>/``), so no two jobs ever share a store, a fit report or a metrics file, and a stopped row's files are
never reused by its fallback row.

    python scripts/sikl4/make_fit_configs.py           # write every file
    python scripts/sikl4/make_fit_configs.py --check   # exit 1 when any committed file differs from a regeneration
"""
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
SOURCE = "configs/two_si_qwen35_4b_fits.yaml"
TASKS = "configs/sikl4/fit_tasks.tsv"

SIS = ("safety_refusal", "si_constitution", "C16")
ARMS = ("lora_r4", "psr_kl_all")
SEEDS = (1, 2, 3)
LR_STEPS = (0, 1, 2)
SMOKE_ATTEMPTS = (1, 2, 3)
#: the smoke's job: the safety LoRA (its mechanics stand for both kinds' plumbing)
SMOKE_SI, SMOKE_ARM, SMOKE_SEED = "safety_refusal", "lora_r4", 1
C16_MAX_LEN = 8192
PAPER_LABEL = "other"
#: the registry's prompt pool ids: <pool model directory>/<pool file stem>, the spelling the published fits' rows carry
POOL_MODEL_DIR = "qwen35_4b"
PROMPT_POOL_IDS = {si: f"{POOL_MODEL_DIR}/pairs_{si}" for si in SIS}
#: checkpoint cadence (optimizer steps): production fits, and the smoke (its one epoch is ~129 steps, so 10 gives a checkpoint
#: about a minute into training, which its resume check needs)
CHECKPOINT_EVERY = 100
SMOKE_CHECKPOINT_EVERY = 10

#: paths under the cluster shared root
POOL_DIR = f"exp/sikl4/pools/{POOL_MODEL_DIR}"
OBJECTS_ROOT = f"exp/objects/sikl4/{POOL_MODEL_DIR}"
SMOKE_ROOT = "exp/sikl4/smoke"

#: every top-level / nested key a fit config may differ from the source in
DELTA_KEYS = ("si_ids", "max_len_by_si.C16", "train_kwargs.default.positions", "train_kwargs.default.seed",
              "train_kwargs.lora_r4.lr", "train_kwargs.psr_kl_all.lr", "registry_paper_label", "registry_prompt_pool_ids")
SMOKE_DELTA_KEYS = DELTA_KEYS + ("train_kwargs.default.epochs",)

TASK_COLUMNS = ("row_id", "mode", "si", "arm", "seed", "lr_step", "config", "pool", "teacher", "out_root", "checkpoint_every")


def config_name(seed: int, lr_step: int) -> str:
    return f"configs/sikl4_qwen35_4b_fits_s{seed}_lr{lr_step}.yaml"


SMOKE_CONFIG = "configs/sikl4_qwen35_4b_fits_smoke.yaml"


def lr_factor(lr_step: int) -> float:
    """2^(-n/2): one half-octave per step."""
    return 2.0 ** (-lr_step / 2.0)


def fit_row_id(si: str, arm: str, seed: int, lr_step: int) -> str:
    return f"sikl-fit-{si}-{arm}-s{seed}" + (f"-lr{lr_step}" if lr_step else "")


def teacher_row_id(si: str) -> str:
    return f"sikl-teacher-{si}"


def smoke_row_id(attempt: int) -> str:
    return f"sikl-fit-smoke-a{attempt}"


def pool_path(si: str) -> str:
    return f"{POOL_DIR}/pairs_{si}.jsonl"


def teacher_path(si: str) -> str:
    """The artifact precompute-teacher writes beside its pool (``<pairs>.teachertopk.safetensors``, teacher_topk's convention)."""
    return f"{pool_path(si)}.teachertopk.safetensors"


def fit_out_root(si: str, arm: str, seed: int, lr_step: int) -> str:
    fits = "fits" if lr_step == 0 else f"fits_lr{lr_step}"
    return f"{OBJECTS_ROOT}/{fits}/s{seed}/{si}__{arm}"


def derive(src: dict, *, seed: int, lr_step: int, smoke: bool = False) -> dict:
    """The source with exactly the declared deltas applied (key order kept; new keys appended)."""
    cfg = copy.deepcopy(src)
    cfg["si_ids"] = list(SIS)
    cfg["max_len_by_si"] = {**cfg["max_len_by_si"], "C16": C16_MAX_LEN}
    tk = cfg["train_kwargs"]
    tk["default"] = {**tk["default"], "positions": "all", "seed": int(seed)}
    if smoke:
        tk["default"]["epochs"] = 1
    for arm in ARMS:
        tk[arm] = {**tk[arm], "lr": float(src["train_kwargs"][arm]["lr"]) * lr_factor(lr_step)}
    cfg["registry_paper_label"] = PAPER_LABEL
    cfg["registry_prompt_pool_ids"] = dict(PROMPT_POOL_IDS)
    return cfg


def header(name: str, *, seed: int, lr_step: int, smoke: bool) -> str:
    what = ("the mechanics smoke: the safety LoRA for ONE epoch (a resume and stop check, never a divergence check)"
            if smoke else "the safety, constitution and C16 refits (lora_r4 and psr_kl_all)")
    lines = [
        f"# {Path(name).name} — DERIVED, do not hand-edit.",
        f"# Regenerate: python scripts/sikl4/make_fit_configs.py (source: {SOURCE})",
        "#",
        f"# Qwen3.5-4B system-instruction distillation, {what}:",
        f"# fit seed {seed}, learning-rate step {lr_step} (both kinds' lr = the source's x 2^(-{lr_step}/2)).",
        "# The rank-4 o_proj LoRA and the all-layer ReLU steer act at EVERY token position; the KL loss",
        "# stays on the response tokens. Every other key is the source's (the published per-kind recipe).",
        f"# Deltas from the source: {', '.join(SMOKE_DELTA_KEYS if smoke else DELTA_KEYS)}.",
        "# C16's text is not in this repository: the fit resolves it from the candidate-SI snapshot",
        "# (SISTEER_LOAD_CANDIDATE_SIS). The W&B project is the job environment's WANDB_PROJECT.",
    ]
    return "\n".join(lines) + "\n"


def render(src: dict, name: str, *, seed: int, lr_step: int, smoke: bool = False) -> str:
    """The config file ``name``: its header, then the source with the deltas applied."""
    body = yaml.safe_dump(derive(src, seed=seed, lr_step=lr_step, smoke=smoke), sort_keys=False, width=79,
                          default_flow_style=False, allow_unicode=True)
    return header(name, seed=seed, lr_step=lr_step, smoke=smoke) + body


def build_configs() -> dict[str, str]:
    """{repo-relative path: file text} for every config this script owns."""
    src = yaml.safe_load((REPO / SOURCE).read_text())
    out = {config_name(seed, step): render(src, config_name(seed, step), seed=seed, lr_step=step)
           for seed in SEEDS for step in LR_STEPS}
    out[SMOKE_CONFIG] = render(src, SMOKE_CONFIG, seed=SMOKE_SEED, lr_step=0, smoke=True)
    return out


def _row(row_id: str, mode: str, si: str, config: str, *, arm: str = "-", seed: int | str = "-", lr_step: int | str = "-",
         out_root: str = "-", checkpoint_every: int | str = "-") -> dict[str, str]:
    """One task-table row (``-`` = the column does not apply to the row's mode); every row reads its SI's pool and teacher."""
    return {"row_id": row_id, "mode": mode, "si": si, "arm": arm, "seed": str(seed), "lr_step": str(lr_step), "config": config,
            "pool": pool_path(si), "teacher": teacher_path(si), "out_root": out_root, "checkpoint_every": str(checkpoint_every)}


def build_tasks() -> list[dict[str, str]]:
    """Every row the refits can submit, in submission-friendly order: teachers, smoke attempts, then the fits by step, seed,
    SI and kind."""
    teacher_cfg = config_name(SEEDS[0], 0)        # any refit config: the teacher reads only the model, the SI and its max_len
    rows = [_row(teacher_row_id(si), "teacher", si, teacher_cfg) for si in SIS]
    rows += [_row(smoke_row_id(a), "smoke", SMOKE_SI, SMOKE_CONFIG, arm=SMOKE_ARM, seed=SMOKE_SEED, lr_step=0,
                  out_root=f"{SMOKE_ROOT}/a{a}", checkpoint_every=SMOKE_CHECKPOINT_EVERY) for a in SMOKE_ATTEMPTS]
    rows += [_row(fit_row_id(si, arm, seed, step), "fit", si, config_name(seed, step), arm=arm, seed=seed, lr_step=step,
                  out_root=fit_out_root(si, arm, seed, step), checkpoint_every=CHECKPOINT_EVERY)
             for step in LR_STEPS for seed in SEEDS for si in SIS for arm in ARMS]
    return rows


def render_tasks(rows: list[dict[str, str]]) -> str:
    lines = ["\t".join(TASK_COLUMNS)]
    for r in rows:
        assert all("\t" not in r[c] and r[c] for c in TASK_COLUMNS), r
        lines.append("\t".join(r[c] for c in TASK_COLUMNS))
    return "\n".join(lines) + "\n"


def build_all() -> dict[str, str]:
    configs = build_configs()
    return {**configs, TASKS: render_tasks(build_tasks())}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check", action="store_true", help="compare the committed files with a regeneration; write nothing")
    args = ap.parse_args(argv)
    files = build_all()
    if args.check:
        stale = [name for name, text in files.items()
                 if not (REPO / name).is_file() or (REPO / name).read_text() != text]
        for name in stale:
            print(f"[sikl4-configs] STALE: {name} differs from a regeneration", file=sys.stderr)
        if stale:
            print("[sikl4-configs] regenerate: python scripts/sikl4/make_fit_configs.py", file=sys.stderr)
            return 1
        print(f"[sikl4-configs] {len(files)} file(s) match a regeneration")
        return 0
    for name, text in files.items():
        path = REPO / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        print(f"wrote {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
