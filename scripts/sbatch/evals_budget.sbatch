#!/bin/bash
#SBATCH --job-name=evals_budget       # SESSION-UNIQUE evals_* name (capeval-controls §6);
#                                     # cancel by JOB ID only — a shared-user name-scoped
#                                     # scancel once caused a mass kill.
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; draws on the 16-GPU high budget
#SBATCH --gres=gpu:1                  # one H200: the 22 (SI, arm) ladders run sequentially,
#                                     # one fresh trial subprocess per (si, arm, bs)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-06:00:00             # ~3h estimated, ~2x padding (backfill-friendly)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append            # retries append, never clobber
#SBATCH --no-requeue                  # stage2_common self-escalates; Slurm must not double-requeue
#SBATCH --signal=B:TERM@120
#
# The two-SI training-budget GPU pass:
# measured peak memory (alloc + reserved) and tokens/s for every
# (si_id x registered class_a_arm) at ascending batch widths, through each
# arm's OWN production trainer on synthetic worst-case-length pairs. Emits
# the measured table train_budget.measure_si_budget consumes.
#
# Benign job: synthetic filler text only — no attack prompts, no raw
# generations — so logs go to the ordinary world-readable exp/logs and W&B
# stays on (offline; sync from a login node afterwards).
#
# Preemption tolerance: every trial row lands in the fingerprinted trials
# JSONL (fsync per row) on SHARED storage; a resubmitted job resumes by
# exact fingerprint match and re-runs at most the one interrupted trial.
# stage2_common.sh provides the SIGTERM self-escalation, the STOP sentinel,
# the EXPECTED_SHA immutable-checkout gate, the cleanup trap, and the
# $FIGS_DIR mirror on every exit path.
#
# OWN CHECKOUT (peers' checkouts are never repointed): this lane
# runs from ${SISTEER_PROJECT_ROOT}-evals/repo with its OWN EXPECTED_SHA at
# ${SISTEER_PROJECT_ROOT}-evals/EXPECTED_SHA — TrainTest's live jobs keep the
# shared default checkout at THEIR sha, and the root-named uv env keeps the
# two lockfiles out of each other's venvs. S2_ROOT is body-set (never
# sbatch --export, hard rule 1).
#
# Submit (after mirror-repos.sh + syncing OUR checkout + recording its sha):
#   git -C ${SISTEER_PROJECT_ROOT}-evals/repo fetch && \
#     git -C ${SISTEER_PROJECT_ROOT}-evals/repo checkout <sha> && \
#     echo <sha> > ${SISTEER_PROJECT_ROOT}-evals/EXPECTED_SHA
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-evals/repo rev-parse HEAD) \
#          scripts/sbatch/evals_budget.sbatch
# Stop for good with stop-run.sh (a bare scancel looks like preemption and
# self-escalates):
#   ~/.claude/skills/gpu-avail/scripts/stop-run.sh evals_budget

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
S2_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-evals"
source "$S2_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# Cluster NCCL requirements (inert on 1 GPU, kept per the standing sbatch
# checklist so a widened resubmit can never silently miss them). Body-level
# exports — NEVER sbatch --export (hard rule 1).
export NCCL_NVLS_ENABLE=0

# Trials JSONL + measured table on SHARED storage next to the STOP sentinel:
# survives preemption, node reassignment, and repo re-checkouts. The measured
# JSON is pulled back to the grading host and committed as
# configs/two_si_budget_measured.json (§5.3 is a separate, reviewed commit).
TRIALS="$STOP_DIR/trials.jsonl"
MEASURED="$STOP_DIR/two_si_budget_measured.json"

s2_run uv run --no-sync python scripts/measure_train_budget.py run \
  --config configs/two_si.yaml \
  --trials "$TRIALS" \
  --emit "$MEASURED" \
  --figs "$FIGS_DIR"
exit $?
