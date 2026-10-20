#!/bin/bash
#SBATCH --job-name=sisteer-frontier15
#SBATCH --partition=general
#SBATCH --qos=high                    # set from gpu-avail at submit time
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=0-04:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# Array: 0 = qwen3.5-9b.
# Submit with the mandatory submit-time SHA pin:
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=0 scripts/sbatch/frontier_stage15.sbatch

set -uo pipefail

OUTS=(stage15_qwen35_9b)
MODELS=("Qwen/Qwen3.5-9B")
REVS=("")
OUT=${OUTS[$SLURM_ARRAY_TASK_ID]}
MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
REV=${REVS[$SLURM_ARRAY_TASK_ID]}

# Ported onto stage2_common: s2_init enforces the submit-time SHA
# pin + EXPECTED_SHA guard, runs the flock'd rc-checked
# `uv sync --frozen --extra cu128` and the srun-time cu128 assert,
# exports the env the old inline block set by hand (hf_env.sh,
# HF_HUB_OFFLINE=1, the UV_* trio — plus WANDB_MODE=offline, which this
# script previously left unset), cds into the repo, and arms
# TERM-escalation + the EXIT figure mirror.
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

s2_run uv run --no-sync python scripts/stage1_frontier.py \
  --out-dir $SISTEER_PROJECT_ROOT/out/$OUT --model "$MODEL" \
  ${REV:+--revision "$REV"}
exit $?
