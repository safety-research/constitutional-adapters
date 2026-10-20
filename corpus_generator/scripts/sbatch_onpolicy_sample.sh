#!/bin/bash
# On-policy wave-C sampling on cluster A — one model per 1-GPU job.
#
# Usage:  sbatch -J <unique-job-name> scripts/sbatch_onpolicy_sample.sh <e4b|qwen35>
# (model key is POSITIONAL — never pass env via sbatch --export, hard rule)
#
# Reads the repo mirror at $SHARED/repos/corpus_generator and the prompt set
# staged at $SHARED/exp/corpus_generator/onpolicy/; writes samples under
# $SHARED/exp/corpus_generator/onpolicy/<run-id>/.
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

set -euo pipefail

MODEL_KEY="${1:?usage: sbatch ... sbatch_onpolicy_sample.sh <e4b|qwen35>}"

# the cluster's shared root: SISTEER_SHARED_ROOT (the front door's job env), else CLUSTER_SHARED_ROOT
SHARED="${SISTEER_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-}}"
[ -n "$SHARED" ] && [ -d "$SHARED" ] || { echo "FATAL: set SISTEER_SHARED_ROOT to this cluster's shared root (the site config's CLUSTER_SHARED_ROOT)" >&2; exit 2; }
# shared copy, never ~/scripts (cluster /home is node-local)
source "$SHARED/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1               # both models fully cached; fail loudly if not
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false
export VLLM_CACHE_ROOT="$SHARED/.cache/vllm"   # JIT kernel cache survives node changes
export NCCL_NVLS_ENABLE=0

REPO="$SHARED/repos/corpus_generator"
PROMPT_SET="$SHARED/exp/corpus_generator/onpolicy/prompt-set-onpolicy-20260901.jsonl"

case "$MODEL_KEY" in
  e4b)
    MODEL="google/gemma-4-E4B-it"
    REV="ee0ef6023621cff504d758262d4e04895a5af4a2"
    RUN_ID="onpolicy-e4b-20260901"
    # gemma-4 is a HETEROGENEOUS per-layer architecture: transformers 5.15.x
    # raises AmbiguousGlobalPerLayerAttributeError on head_dim (job 328400);
    # 5.14.1 loads it (head_dim uniform 256) — the pin this repository's Modal
    # gemma-4 lane documents. A PYTHONPATH overlay carries just that pin
    # (uv-built venv has no pip; a full clone is 70k files of VAST copies),
    # so the shared venv stays untouched for other campaigns.
    export PYTHONPATH="$SHARED/envs/tf514-overlay${PYTHONPATH:+:$PYTHONPATH}"
    EXTRA=(--max-num-seqs 32)
    ;;
  qwen35)
    MODEL="Qwen/Qwen3.5-4B"           # instruct variant; -Base is the base model
    REV="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
    RUN_ID="onpolicy-qwen35-20260901"
    EXTRA=(--hybrid --max-num-seqs 32)  # hybrid attn: aligned mamba cache
    ;;
  *) echo "unknown model key: $MODEL_KEY" >&2; exit 2 ;;
esac

VENV="$SHARED/envs/vllm-attacker"     # vllm 0.25.1+cu129
export PATH="$VENV/bin:$PATH"         # GDN JIT shells out to ninja

OUT_DIR="$SHARED/exp/corpus_generator/onpolicy/$RUN_ID"
mkdir -p "$OUT_DIR"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

srun --ntasks=1 "$VENV/bin/python" "$REPO/scripts/sample_onpolicy_vllm.py" \
  --prompt-set "$PROMPT_SET" \
  --model "$MODEL" \
  --revision "$REV" \
  --out-dir "$OUT_DIR" \
  --seed 0 \
  --max-model-len 24576 \
  "${EXTRA[@]}"
