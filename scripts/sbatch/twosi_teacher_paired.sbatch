#!/bin/bash
#SBATCH --job-name=twosi-teacher-paired
#                                     # Session-unique twosi-teacher-* name: cancel by
#                                     # the JOB ID you submitted (shared-user account —
#                                     # a name/user-scoped scancel would hit peers).
#SBATCH --partition=general
#SBATCH --qos=high                    # cluster A's non-preemptible default QoS;
#                                     # override the QoS name when run on cluster B.
#SBATCH --gres=gpu:1                  # one H200: 2 SIs x 1037 paired prompts, two
#                                     # no-grad forwards each, sequential
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-01:00:00             # measured: 3 SIs x 1037 took 25m55s (job
#                                     # 30542) -> 2 SIs ~18 min; ~3x pad keeps it
#                                     # backfillable
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append            # retries append, never clobber
#SBATCH --no-requeue                  # stage2_common self-escalates; no double-requeue
#SBATCH --signal=B:TERM@120
#
# The PAIRED per-example teacher-signal comparison of si_constitution vs
# si_irrelevant. Job 30542 measured the per-SI floors on 1037 prompt
# identities shared across SIs but persisted only per-family aggregates, so
# the paired per-example test (is the pooled +0.042 con-irr difference real,
# or entirely the bon/encoding families?) could not be computed from its
# output. This rerun is the SAME selection (verified: the 2-SI stratified
# sha-rank selection reproduces 30542's selection.selected exactly) with
# measure_teacher_signal.py's per_example block persisted per SI —
# identity-aligned per-example KLs a reader joins row-by-row.
#
# --sis si_constitution,si_irrelevant: safety_refusal is skipped on purpose
# — two SIs answer the question and halve the cost. Selection is per-SI
# deterministic, so the subset selects exactly what the full run selected.
#
# Inputs: $SHARED/exp/diffmean2x2/pools/pairs_<si>.jsonl (23,002 rows each,
# verified on cluster A). Output json (aggregates + per_example per SI):
# $SHARED/exp/twosi_floors/teacher_paired_<jobid>.json.
#
# Submit (after the repo-mirror tool brings the -twosi clone up to date):
#   cd ${SISTEER_REPO_MIRROR}-twosi
#   git rev-parse HEAD > EXPECTED_SHA
#   sbatch --comment=sha:$(git -C ${SISTEER_REPO_MIRROR}-twosi rev-parse HEAD) \
#     scripts/sbatch/twosi_teacher_paired.sbatch

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
S2_ROOT="${SISTEER_REPO_MIRROR:?set SISTEER_REPO_MIRROR, the cluster clone of this repository (the front door exports it from the site config)}-twosi"
S2_REPO_DIR="$S2_ROOT"
source "$S2_REPO_DIR/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# Standing NCCL exports (inert on 1 GPU; kept so a widened resubmit can
# never silently miss them). Body-level — NEVER sbatch --export (rule 1).
export NCCL_NVLS_ENABLE=0

POOLS="$SHARED/exp/diffmean2x2/pools"
OUT_DIR="$SHARED/exp/twosi_floors"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/teacher_paired_${SLURM_JOB_ID}.json"

s2_run uv run --no-sync python scripts/measure_teacher_signal.py \
  --config configs/two_si.yaml \
  --pairs-dir "$POOLS" \
  --per-family 96 \
  --batch-size 2 \
  --max-len 8192 \
  --sis si_constitution,si_irrelevant \
  --out "$OUT"
exit $?
