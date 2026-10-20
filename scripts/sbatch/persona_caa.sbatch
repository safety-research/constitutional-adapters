#!/bin/bash
#SBATCH --job-name=sisteer-persona-caa
#SBATCH --partition=general
#SBATCH --qos=high                    # set from gpu-avail at submit time
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00             # ~50min expected; padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# Submit with the mandatory submit-time SHA pin:
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) scripts/sbatch/persona_caa.sbatch

set -uo pipefail

# Ported onto stage2_common: s2_init enforces the submit-time SHA
# pin + EXPECTED_SHA guard, runs the flock'd rc-checked
# `uv sync --frozen --extra cu128` and the srun-time cu128 assert,
# exports the env the old inline block set by hand (hf_env.sh,
# HF_HUB_OFFLINE=1 — gated cached models 403 on revalidation — offline W&B,
# the UV_* trio), cds into the repo, and arms TERM-escalation + the EXIT
# figure mirror.
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

OUT=$SISTEER_PROJECT_ROOT/out

# Clean reruns: strict two-choice items only (the raw files are 26-40%
# (C)-contaminated), matched transfer test sets, pole-correct reporting.
s2_run bash -c "
  set -x
  [ -f $OUT/persona_check_clean/results.json ] || \
    uv run --no-sync python scripts/check_persona_inversion.py \
      --model mistralai/Mistral-7B-Instruct-v0.2 --out $OUT/persona_check_clean
  [ -f $OUT/replicate_bipo_mistral_clean/results.json ] || \
    uv run --no-sync python scripts/replicate_caa_bipo.py \
      --model mistralai/Mistral-7B-Instruct-v0.2 \
      --caa-layers 12,13,14 --bidpo-layer 13 --epochs 10 \
      --out $OUT/replicate_bipo_mistral_clean
  [ -f $OUT/replicate_caa_llama13b_clean/results.json ] || \
    uv run --no-sync python scripts/replicate_caa_bipo.py \
      --model meta-llama/Llama-2-13b-chat-hf \
      --base meta-llama/Llama-2-13b-hf \
      --caa-layers 12,13,14,15,16,17,18 --skip-bipo \
      --n-train 700 \
      --eval-system 'You are a helpful, honest and concise assistant.' \
      --out $OUT/replicate_caa_llama13b_clean
"
exit $?
