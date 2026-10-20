#!/bin/bash
#SBATCH --job-name=sisteer-stage1
#SBATCH --partition=general
#SBATCH --qos=high                    # user-directed: run stage 1 at high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-07:00:00             # ~3.5h expected per instruction; padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# Submit with an explicit array and the mandatory submit-time SHA pin:
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=0,8 scripts/sbatch/stage1.sbatch     # canaries
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=1-7,9,10 scripts/sbatch/stage1.sbatch

set -uo pipefail

INSTRUCTIONS=(french lowercase concise json bullets include_notably \
              no_sorry pirate cond_medical cond_code multi_french_concise)
INSTR=${INSTRUCTIONS[$SLURM_ARRAY_TASK_ID]}

# Ported onto stage2_common: s2_init enforces the submit-time SHA
# pin + EXPECTED_SHA guard, runs the flock'd rc-checked
# `uv sync --frozen --extra cu128` and the srun-time cu128 assert, and
# exports the same env the old inline block set by hand (hf_env.sh,
# HF_HUB_OFFLINE=1 — Llama-3.1 is gated, weights are cached — offline W&B
# into $SHARED/wandb, the UV_* trio), cds into the repo, and
# arms TERM-escalation + the EXIT figure mirror. The old in-trap
# `wandb sync` is dropped deliberately: compute nodes can't reach
# api.wandb.ai; sync the offline runs from a login node.
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# Resume-safe: completed (instruction, method) cells are skipped.
s2_run uv run --no-sync python -m sisteer fit-eval \
  --config configs/stage1_llama8b.yaml --instruction "$INSTR"
exit $?
