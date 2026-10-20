#!/bin/bash
#SBATCH --job-name=plc2gen
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=06:00:00
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append

# Placebo-control regeneration: 5 arms x 4,200 instances = 21,000 greedy/T1
# completions of <=512 tokens on one GPU. --time is ceiling-sized (~3x the
# realistic estimate); the JSONL append + load_done_keys resume makes a rerun
# regenerate only missing blocks. One compute node is excluded: it fails jobs
# at GPU acquisition (cudaErrorDevicesUnavailable) while sinfo reports it
# healthy.

SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"
source "$SHARED/scripts/hf_env.sh"
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# flashinfer JIT-compiles its sampling module with ninja on the first
# generate() call; a venv python invoked by ABSOLUTE PATH does not put its
# bin/ on PATH, so the subprocess dies with FileNotFoundError: 'ninja' (the
# CLAUDE.md "ninja on PATH" trap). Prepend the venv bin so the child sees it.
export VENV=$SHARED/envs/placebo2-vllm
export PATH="$VENV/bin:$PATH"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

REPO=$SHARED/exp/placebo2/repo
OUT=$SHARED/exp/placebo2
cd "$REPO" || exit 1

# Positional overrides (passed positionally, never --export=): $1 prompts,
# $2 rows, $3 comma-list of arms. Defaults reproduce the original 3-family run.
PROMPTS=${1:-$OUT/prompts.jsonl}
ROWS=${2:-$OUT/rows.jsonl}
ARMS=${3:-sr_greedy,si_greedy,si_t1,p2_greedy,p2_t1}

srun "$VENV/bin/python" \
  scripts/placebo2/placebo2_sample.py \
  --prompts "$PROMPTS" \
  --out "$ROWS" \
  --arms "$ARMS" \
  --si-doc scripts/placebo2/irrelevant_si2.md
rc=$?
echo "[plc2gen] srun exit rc=$rc"
exit $rc
