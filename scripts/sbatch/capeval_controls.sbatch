#!/bin/bash
#SBATCH --job-name=capeval-controls
#SBATCH --partition=general
#SBATCH --qos=CAPEVAL_REPLACE_ME      # INVALID on purpose: sbatch refuses unless the submitter passes --qos=<gpu-avail result> (§11.2 step 2; review infra #14)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00             # full cells ~3h planned, padded ~2x; T0-only rungs get --time=0-02:00:00 at submit
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# capeval controls: no-ops, random-vector + noise-LoRA
# ladders, weight-noise. Width = `print-cells` output (F15; currently 15).
# TWO submissions, split by time class (print-cells emits both ranges;
# CAPEVAL_CONTROLS_ARRAY_* mirror them):
#
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD) \
#          --qos=<gpu-avail> --array=$CAPEVAL_CONTROLS_ARRAY_T0ONLY \
#          --time=0-02:00:00 scripts/sbatch/capeval_controls.sbatch   # Tier-0-only rungs (~40min)
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD) \
#          --qos=<gpu-avail> --array=$CAPEVAL_CONTROLS_ARRAY_FULL \
#          scripts/sbatch/capeval_controls.sbatch                     # Tier-1-bearing cells
#
# Weight-mutating cells (noise_lora, weight_noise) are re-exec'd by the
# driver into a fresh subprocess — nothing to do here.
# Depends on: calibration trial 0 complete (the reference + KL caches).

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "${SISTEER_PROJECT_ROOT}-capeval-repo/scripts/sbatch/capeval_common.sh"
s2_init "$@"

# SLURM_ARRAY_TASK_ID is REQUIRED (review infra #13): this is an
# array-only script — a submission without --array would otherwise run
# cell 0 alone and exit green while the rest were never scheduled.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  s2_log "FATAL: SLURM_ARRAY_TASK_ID is unset — submit with --array=<range from print-cells> (F15); never run this script without an array"
  exit 1
fi
case "$SLURM_ARRAY_TASK_ID" in
  ''|*[!0-9]*)
    s2_log "FATAL: SLURM_ARRAY_TASK_ID='$SLURM_ARRAY_TASK_ID' is not a non-negative integer"
    exit 1;;
esac
IDX=$SLURM_ARRAY_TASK_ID
if [ "$IDX" -ge "$CAPEVAL_CONTROLS_CELLS" ]; then
  s2_log "FATAL: array index $IDX >= CAPEVAL_CONTROLS_CELLS=$CAPEVAL_CONTROLS_CELLS — array range must come from print-cells (F15)"
  exit 1
fi
s2_log "cell: controls index $IDX (decode: capeval_run.py print-cells --manifest configs/capeval/manifests/controls.yaml)"

s2_run uv run --no-sync python scripts/capeval_run.py \
  --manifest configs/capeval/manifests/controls.yaml run --cell-index "$IDX"
rc=$?
capeval_revise_time_check $((6 * 3600))
exit $rc
