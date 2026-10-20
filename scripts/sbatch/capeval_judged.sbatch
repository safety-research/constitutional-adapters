#!/bin/bash
#SBATCH --job-name=evals_judged
#SBATCH --partition=general
#SBATCH --qos=high                    # standing default; draws on the 16-GPU/user high budget (1 here)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU (hard rule 2)
#SBATCH --time=0-06:00:00             # judged_base gens ~3h, padded ~2x (capeval-controls §6); smoke: submit with --time=0-01:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# capeval judged controls — the GENERATE phase (capeval-controls §3.4/§6).
# Array over the three tasks:
#
#   sbatch --array=0-1 \
#          --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD) \
#          scripts/sbatch/capeval_judged.sbatch                      # base leg
#   sbatch --array=0-1 --time=0-01:00:00 --job-name=evals_judged_smoke \
#          --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-capeval-repo rev-parse HEAD) \
#          scripts/sbatch/capeval_judged.sbatch                      # smoke (set CAPEVAL_JUDGED_LIMIT=20)
#
# Env seams (set in THIS file's body or a wrapper before sourcing — never
# via sbatch --export, hard rule 1):
#   CAPEVAL_JUDGED_SWEEP     sweep name (default judged_base)
#   CAPEVAL_JUDGED_ARM_ARGS  arm flags (default "--base"; a cell leg passes
#                            "--arm-artifact ... --arm-kind ... --arm-name ...
#                             --layer N --coeff C [--system-file ...]")
#   CAPEVAL_JUDGED_LIMIT     optional --limit N (smoke runs)
#   CAPEVAL_REPO_DIR         the evals lane's OWN cluster checkout (§6) —
#                            peers' checkouts and EXPECTED_SHA never touched
#
# The JUDGE phase runs on the grading host (Low-Prio key + shared judge cache
# live there), never here:
#   uv run python scripts/capeval_judged_run.py judge --task <t> --sweep <s>
#
# Preemption-tolerant with no flags: the GenSpool + CellStore resume per
# item; stage2_common's traps handle SIGTERM drain, QoS escalation, the
# STOP sentinel, and the $FIGS_DIR mirror on EVERY exit path.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
# our OWN checkout (capeval-controls §6) — capeval_common.sh pins
# S2_REPO_DIR to CAPEVAL_REPO_DIR and gates on ITS EXPECTED_SHA
export CAPEVAL_REPO_DIR="${CAPEVAL_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-evals-repo}"
source "$CAPEVAL_REPO_DIR/scripts/sbatch/capeval_common.sh"
s2_init "$@"

# cluster NCCL requirements (1-GPU jobs don't collective, but the header
# block is uniform per the cluster rules)
export NCCL_NVLS_ENABLE=0

# SLURM_ARRAY_TASK_ID is REQUIRED: this is an array-only script — a
# submission without --array would run one task and exit green while the
# other two were never scheduled.
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  s2_log "FATAL: SLURM_ARRAY_TASK_ID is unset — submit with --array=0-1 (or a subset)"
  exit 1
fi
TASKS=(or_bench xstest)   # alpaca retired 2026-09-11
case "$SLURM_ARRAY_TASK_ID" in
  0|1|2) TASK="${TASKS[$SLURM_ARRAY_TASK_ID]}" ;;
  *)
    s2_log "FATAL: array index $SLURM_ARRAY_TASK_ID not in 0-1 (or_bench/xstest)"
    exit 1;;
esac

SWEEP="${CAPEVAL_JUDGED_SWEEP:-judged_base}"
LIMIT_ARGS=""
if [ -n "${CAPEVAL_JUDGED_LIMIT:-}" ]; then
  LIMIT_ARGS="--limit $CAPEVAL_JUDGED_LIMIT"
fi
s2_log "judged generate: task=$TASK sweep=$SWEEP arm='${CAPEVAL_JUDGED_ARM_ARGS:---base}' $LIMIT_ARGS"

# shellcheck disable=SC2086  # deliberate word-splitting on the arg seams
s2_run uv run --no-sync python scripts/capeval_judged_run.py generate \
  --task "$TASK" --sweep "$SWEEP" \
  --out-root "$SHARED/exp/capeval_judged" \
  ${CAPEVAL_JUDGED_ARM_ARGS:---base} $LIMIT_ARGS
exit $?
