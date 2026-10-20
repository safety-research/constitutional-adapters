#!/bin/bash
#SBATCH --job-name=sisteer-replicate
#SBATCH --partition=general
#SBATCH --qos=high                    # set from gpu-avail at submit time
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-05:00:00             # ~2.5h expected; padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# Submit with the mandatory submit-time SHA pin:
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) scripts/sbatch/replicate.sbatch

set -uo pipefail

# Ported onto stage2_common: s2_init enforces the submit-time SHA
# pin + EXPECTED_SHA guard, runs the flock'd rc-checked
# `uv sync --frozen --extra cu128` and the srun-time cu128 assert,
# exports the env the old inline block set by hand (hf_env.sh, offline W&B
# into the shared W&B dir, the UV_* trio), cds into the repo, and
# arms TERM-escalation + the EXIT figure mirror. The old in-trap
# `wandb sync` is dropped deliberately: compute nodes can't reach
# api.wandb.ai, so sync the offline runs from a login node.
#
# s2_init also sets HF_HUB_OFFLINE=1, which this job REQUIRES for the gated
# repos (meta-llama, google/gemma): their weights are cached but the shared
# token is not authorized, so any online revalidation dies with
# GatedRepoError (job 61996's llama step).
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

OUT=$SISTEER_PROJECT_ROOT/out

# Four replications, each resumable/idempotent via its own out dir.
# Models chosen from what is fully cached (Llama-2-7b-chat is gated with
# no cached weights): each paper's second model.
s2_run bash -c "
  set -x
  [ -f $OUT/replicate_actadd/results.json ] || \
    uv run --no-sync python scripts/replicate_actadd.py \
      --model openai-community/gpt2-xl --out $OUT/replicate_actadd
  [ -f $OUT/replicate_bipo_mistral/results.json ] || \
    uv run --no-sync python scripts/replicate_caa_bipo.py \
      --model mistralai/Mistral-7B-Instruct-v0.2 \
      --caa-layers 12,13,14 --bidpo-layer 13 --epochs 10 \
      --out $OUT/replicate_bipo_mistral
  [ -f $OUT/replicate_caa_llama13b/results.json ] || \
    uv run --no-sync python scripts/replicate_caa_bipo.py \
      --model meta-llama/Llama-2-13b-chat-hf \
      --base meta-llama/Llama-2-13b-hf \
      --caa-layers 13,14,15,16,17,18 --skip-bipo \
      --out $OUT/replicate_caa_llama13b
  uv run --no-sync python -m sisteer fit-eval --config configs/replicate_stolfo.yaml
"
exit $?
