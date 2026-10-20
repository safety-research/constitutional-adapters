#!/bin/bash
#SBATCH --job-name=capeval-s2
#SBATCH --partition=general
#SBATCH --qos=CAPEVAL_REPLACE_ME      # INVALID on purpose: sbatch refuses unless the submitter passes --qos=<gpu-avail result> (§11.2 step 2; review infra #14)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-07:00:00             # ~3.5h/cell planned (T0+T1+target+parity), padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# capeval S2 rediscovery: 7 cells — 6 divergent frontier
# arms + 1 gentle contrast — each Tier 0 + Tier 1 + target plugin + the two
# S2-parity probes, paired vs calibration trial 0. PILOT FIRST (§9):
#
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD) \
#          --qos=<gpu-avail> --array=$CAPEVAL_S2_ARRAY_PILOT scripts/sbatch/capeval_s2.sbatch
#   # index 0 = number_words/bidpo (the −0.84 cell). Check the parity-probe
#   # ΔGSM8K reproduces the known-size drop and math=degraded, THEN:
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD) \
#          --qos=<gpu-avail> --array=$CAPEVAL_S2_ARRAY_REST scripts/sbatch/capeval_s2.sbatch
#
# EvalPlus execution runs AFTER this array on the sandbox VM, driven from
# the grading host (scripts/capeval_exec_wave.py) — generation cells here write
# samples.jsonl + generation_complete.json and leave results.json to the
# wave (F8). Depends on: calibration trial 0 complete; controls accepted.

set -uo pipefail

# RETIRED. Every cell of configs/capeval/manifests/s2_rediscovery.yaml scores
# one of the IFEval-worded stage-2 instructions, which were removed with the
# vendored IFEval verifier. The target plugin resolves its instruction only at
# scoring time, after the model load and Tier 0/1, so the array refuses here,
# before any sync or model load.
echo "FATAL: the capeval S2 rediscovery sweep is retired: its cells score stage-2 instructions that are no longer registered (number_words, keyword_frequency, end_with_phrase, json_format, number_bullets)" >&2
exit 1

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
if [ "$IDX" -ge "$CAPEVAL_S2_CELLS" ]; then
  s2_log "FATAL: array index $IDX >= CAPEVAL_S2_CELLS=$CAPEVAL_S2_CELLS — array range must come from print-cells (F15)"
  exit 1
fi
s2_log "cell: s2_rediscovery index $IDX (decode: capeval_run.py print-cells --manifest configs/capeval/manifests/s2_rediscovery.yaml)"

s2_run uv run --no-sync python scripts/capeval_run.py \
  --manifest configs/capeval/manifests/s2_rediscovery.yaml run --cell-index "$IDX"
rc=$?
capeval_revise_time_check $((7 * 3600))
exit $rc
