#!/bin/bash
#SBATCH --job-name=sisteer-objects
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00             # ~25min training + ~2h generation per instruction
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# Array over the 9 non-conditional instructions. Submit with the mandatory
# submit-time SHA pin:
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=0 scripts/sbatch/object_ablation.sbatch      # canary (french)
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=1-8 scripts/sbatch/object_ablation.sbatch

set -uo pipefail

INSTRUCTIONS=(french lowercase concise json bullets include_notably \
              no_sorry pirate multi_french_concise)
INSTR=${INSTRUCTIONS[$SLURM_ARRAY_TASK_ID]}

# Ported onto stage2_common: s2_init enforces the submit-time SHA
# pin + EXPECTED_SHA guard, runs the flock'd rc-checked
# `uv sync --frozen --extra cu128` and the srun-time cu128 assert,
# exports the env the old inline block set by hand (hf_env.sh,
# HF_HUB_OFFLINE=1, the UV_* trio — plus WANDB_MODE=offline, which this
# script previously left unset), cds into the repo, and arms
# TERM-escalation + the EXIT figure mirror (replacing the old bare
# kill-the-process-group cleanup trap).
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# Resume-safe: completed (instruction, object) cells are skipped.
s2_run uv run --no-sync python scripts/object_ablation.py --instruction "$INSTR"
exit $?
