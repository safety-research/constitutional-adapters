#!/bin/bash
#SBATCH --job-name=tt-kvbench         # SESSION-UNIQUE tt-* prefix: override per
#                                     # submission with -J tt-kvbench-<tag>; cancel by
#                                     # JOB ID only (the shared-user name-scope trap).
#SBATCH --partition=general
#SBATCH --qos=high                    # lane default; 1 GPU of the 16-GPU high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-00:20:00             # measured: both widths + model load ran 3m33s
#                                     # (job 190490) — padded ~5x for a cold HF cache
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# Teacher prefix KV cache: the ACCEPTANCE MEASUREMENT (defined
# 2026-08-24) — one measured before/after tokens-per-second comparison of
# the teacher leg on a real si_constitution batch (pinned Llama-3.1-8B
# mirror, bf16, sdpa, the wave's max_len), plus the bf16 parity numbers.
# Synthetic benchmark prompts only — nothing sensitive, so logs go to the
# ordinary exp/logs (unlike the transcript jobs' private log dir).
#
# Runs from its OWN checkout root (capeval-controls §6 pattern) so the
# shared production checkout (the project root's repo/) — owned by live TrainTest jobs —
# is never repointed. One-time root setup from the mirror, on the login
# node:
#   SHA=<kvcache commit>
#   git clone $SISTEER_REPO_MIRROR \
#       ${SISTEER_PROJECT_ROOT}-kvcache/repo   # once
#   git -C ${SISTEER_PROJECT_ROOT}-kvcache/repo fetch origin && \
#     git -C ${SISTEER_PROJECT_ROOT}-kvcache/repo checkout --detach $SHA && \
#     echo $SHA > ${SISTEER_PROJECT_ROOT}-kvcache/EXPECTED_SHA
# then:
#   sbatch -J tt-kvbench-<tag> \
#     --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-kvcache/repo rev-parse HEAD) \
#     scripts/sbatch/bench_prefix_cache.sbatch

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
# Own checkout root: own EXPECTED_SHA gate, own node-local venv
# (/home/$USER/venvs/sisteer-kvcache), zero contact with the shared repo.
S2_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-kvcache"
source "$S2_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# Single-GPU job, but the standing checklist keeps these explicit:
export NCCL_NVLS_ENABLE=0

s2_log "bench: teacher prefix KV cache, si_constitution, 32 batches, widths 4 and 8"
# bs 4 (the lockstep group width) and bs 8 (the widest configured
# production width — lora_r1/loreft in batch_size_by_arm): peak memory
# must be measured at the widest width because the prefix-KV batch
# expansion is a per-batch COPY (~B x P x 128KiB), not a view.
s2_run uv run --no-sync python scripts/bench_prefix_cache.py \
  --config configs/two_si.yaml --si si_constitution \
  --batches 32 --batch-sizes 4,8
exit $?
