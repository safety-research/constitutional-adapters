#!/bin/bash
#SBATCH --job-name=s2-comp
#SBATCH --partition=general
#SBATCH --qos=high                    # the default QoS; re-run gpu-avail at submission
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-18:00:00             # ~6-9h/task (12-18 GPU-h / 2, plan 8 A4), padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# C4b/A4 stage2_composition: per object kind --
# train 6 per-behavior objects -> monolithic joint objects per k-target ->
# <and> token (token_faithful only) -> joint k-compliance eval over
# seen/unseen/zero-shot targets, averaged over k! orderings.
# Array layout:
#   0  vector (additive-sum composition)
#   1  token_faithful (<and>-token composition)
#   2  C4b canary: ONE two-behavior target end-to-end (separate + monolithic
#      + <and> + eval) -- validates the full path before A4
#
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --time=0-02:00:00 --array=2 scripts/sbatch/stage2_composition.sbatch  # C4b (short walltime override for backfill)
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=0-1 scripts/sbatch/stage2_composition.sbatch                  # A4

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

IDX=${SLURM_ARRAY_TASK_ID:-0}
KINDS=(vector token_faithful)
if [ "$IDX" -eq 2 ]; then
  s2_run uv run --no-sync python scripts/stage2_run.py \
    --config configs/stage2_composition.yaml --suite composition --canary
else
  s2_run uv run --no-sync python scripts/stage2_run.py \
    --config configs/stage2_composition.yaml --suite composition \
    --arm "${KINDS[$IDX]}"
fi
exit $?
