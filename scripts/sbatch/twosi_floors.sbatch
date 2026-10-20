#!/bin/bash
#SBATCH --job-name=twosi-floors-strat
#                                     # Session-unique twosi-floors-* name: cancel by
#                                     # the JOB ID you submitted (shared-user account —
#                                     # a name/user-scoped scancel would hit peers).
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; on another cluster pass its own QoS
#                                     # (its site block's CLUSTER_QOS).
#SBATCH --gres=gpu:1                  # one H200: 3 SIs x 1037 paired prompts, two
#                                     # no-grad forwards each, sequential
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-02:00:00             # measured: 3 SIs x 256 prompts took ~18 min
#                                     # (job 187947); 1037/256 ~ 4x -> ~75 min real,
#                                     # ~1.6x pad so backfill can still place it
#SBATCH --output=/workspace/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append            # retries append, never clobber
#SBATCH --no-requeue                  # stage2_common self-escalates; no double-requeue
#SBATCH --signal=B:TERM@120
#
# The per-SI no-op floor re-measure (stratified, paired, per-family) that
# REPLACES the contaminated prefix-sampled floors of job 187947 (235 aim +
# 21 amplegcg_plus standing in for the whole pool — see
# scripts/measure_teacher_signal.py's module docstring for the full
# provenance). Selection: sha256(prompt) rank within every family stratum,
# --per-family 96 -> 1037 shared prompt identities per SI spanning all 11
# families (verified on the real pools, identical across SIs). Estimator:
# per-example token-mean forward KL — the guard probes' own estimator —
# so these floors and the fit probes stay directly comparable.
#
# Inputs: $SHARED/exp/diffmean2x2/pools/pairs_<si>.jsonl (verified 23,002
# rows each on this cluster). Model: the pinned ungated NousResearch
# snapshot in the per-user HF cache (offline load; no token on cluster B,
# by design). Output json (per-family table + two labelled pooled floors +
# selected ids): $SHARED/exp/twosi_floors/floors_stratified_<jobid>.json.
#
# Submit (after mirror-repos.sh --cluster <cluster B's CLUSTER_MIRROR_SELECTOR> <this checkout's name>):
#   cd ${SISTEER_REPO_MIRROR}-twosi
#   git rev-parse HEAD > EXPECTED_SHA
#   sbatch --comment=sha:$(git -C ${SISTEER_REPO_MIRROR}-twosi rev-parse HEAD) \
#     scripts/sbatch/twosi_floors.sbatch

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
OUT="$OUT_DIR/floors_stratified_${SLURM_JOB_ID}.json"

s2_run uv run --no-sync python scripts/measure_teacher_signal.py \
  --config configs/two_si.yaml \
  --pairs-dir "$POOLS" \
  --per-family 96 \
  --batch-size 2 \
  --max-len 8192 \
  --out "$OUT"
exit $?
