#!/bin/bash
#SBATCH --job-name=s2-frontier
#SBATCH --partition=general
#SBATCH --qos=high                    # the default QoS; re-run gpu-avail at submission
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00             # est ~3 GPU-h (plan 8 F1), padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# F1 stage2_frontier (plan 8/9): within-budget frontier fill-in across the
# A1 matrix + ladder rungs (stage-1 budget rule: MMLU -2pt, distinct3 >=
# 0.35), evaluating the remaining candidate scales stage2_run.py skipped.
#
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=0 scripts/sbatch/stage2_frontier.sbatch
#
# Depends on: A1 + A7 complete.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# frontier.json lands in the ifeval matrix root -- the path judge_batch.py
# reads it from; ladder rung rows fold into the same frontier. --config
# makes the script consume the experiment config's model + PINNED
# model_revision + dtype through the SAME loader stage2_run uses,
# so the MMLU fill-in runs the exact snapshot/precision the matrix cells
# trained on (offline refs/main can dangle on the shared cache).
s2_run uv run --no-sync python scripts/stage2_frontier.py \
  --out-dir "$SISTEER_PROJECT_ROOT/out/stage2_llama8b/ifeval" \
  --ladder-dir "$SISTEER_PROJECT_ROOT/out/stage2_llama8b/ladder" \
  --config configs/stage2_llama8b.yaml
exit $?
