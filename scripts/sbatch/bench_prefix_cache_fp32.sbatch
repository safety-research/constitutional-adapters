#!/bin/bash
#SBATCH --job-name=tt-kvbench-fp32    # SESSION-UNIQUE tt-* prefix; cancel by JOB ID only
#SBATCH --partition=general
#SBATCH --qos=high                    # lane default; 1 GPU of the 16-GPU high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-00:20:00             # measured: fp32 control ran 3m52s (job 189420)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# The fp32 CONTROL run for the teacher prefix cache parity numbers (sibling
# of bench_prefix_cache.sbatch — separate file because the dtype cannot ride
# sbatch --export, hard rule 1, and s2_init owns the positional args): bf16
# raw-logit diffs conflate accumulation noise on meaningless logit tails
# with real misalignment. At fp32 precision noise collapses by orders of
# magnitude while a misalignment would not move — one run separates them.
#
# Submit (same pinned checkout as the bf16 sibling):
#   sbatch -J tt-kvbench-fp32 \
#     --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-kvcache/repo rev-parse HEAD) \
#     scripts/sbatch/bench_prefix_cache_fp32.sbatch

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
S2_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-kvcache"
source "$S2_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

export NCCL_NVLS_ENABLE=0

s2_log "bench: teacher prefix KV cache fp32 CONTROL, si_constitution, 16 batches x bs 2"
# fp32 halves the tokens/step memory budget vs bf16; bs 2 x 16 batches keeps
# the parity read wide while staying comfortably inside one H200.
s2_run uv run --no-sync python scripts/bench_prefix_cache.py \
  --config configs/two_si.yaml --si si_constitution \
  --batches 16 --batch-sizes 2 --dtype float32
exit $?
