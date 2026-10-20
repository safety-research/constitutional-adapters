#!/bin/bash
#SBATCH --job-name=capeval-calib
#SBATCH --partition=general
#SBATCH --qos=CAPEVAL_REPLACE_ME      # INVALID on purpose: sbatch refuses unless the submitter passes --qos=<gpu-avail result> (§11.2 step 2; review infra #14)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-06:00:00             # ~3h/trial planned, padded ~2x; trial 0 re-times (REVISE-TIME)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# capeval calibration: k=6 base self-runs, one array task
# = one trial (Tier 0 + Tier 1 + the two S2-parity probes). Trial 0 is the
# DESIGNATED REFERENCE and the ONLY KL-cache builder (§4.3/F5): it must run
# ALONE and finish before ANYTHING else is submitted.
#
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD) \
#          --qos=<gpu-avail> --array=$CAPEVAL_CALIB_ARRAY_REFERENCE scripts/sbatch/capeval_calib.sbatch
#   # ... wait for completion, check REVISE-TIME, then:
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD) \
#          --qos=<gpu-avail> --array=$CAPEVAL_CALIB_ARRAY_REST scripts/sbatch/capeval_calib.sbatch
#
# Ranges/width come from `capeval_run.py print-cells` (F15) — the exported
# CAPEVAL_CALIB_* constants mirror it and the width check below refuses a
# drifted index.

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
if [ "$IDX" -ge "$CAPEVAL_CALIB_CELLS" ]; then
  s2_log "FATAL: array index $IDX >= CAPEVAL_CALIB_CELLS=$CAPEVAL_CALIB_CELLS — array range must come from print-cells (F15)"
  exit 1
fi
s2_log "cell: calibration trial $IDX (trial 0 = reference + sole KL-cache builder)"

s2_run uv run --no-sync python scripts/capeval_run.py \
  --manifest configs/capeval/manifests/calibration.yaml run --cell-index "$IDX"
rc=$?
capeval_revise_time_check $((6 * 3600))
exit $rc
