#!/bin/bash
#SBATCH --job-name=hv_asrnoise_renellm
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-00:45:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# ASR noise floor at a MID-RANGE operating point (the LAST measurement on this
# thread). renellm is a strong FROZEN-corpus attack (pure replay, no rewriter at
# run time) — same 40 HarmBench goals, B=1 vs B=8, greedy — to reach a sensitive
# ASR where verdict flips and the length->gradeable-content bias become
# answerable. Corpus staged at SISTEER_HELDOUT_CORPUS_DIR (private, off-repo).
# SHA-gated own checkout, own venv.

set -uo pipefail

SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
source "$SHARED/scripts/hf_env.sh"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1
export SISTEER_HELDOUT_CORPUS_DIR="$SHARED/heldout_corpora"   # frozen renellm.jsonl
export UV_PROJECT_ENVIRONMENT=/home/$USER/venvs/sisteer-hv3
export UV_CACHE_DIR=$SHARED/.cache/uv
export UV_LINK_MODE=copy

RUN_NAME=${SLURM_JOB_NAME}
export FIGS_DIR=$SHARED/exp/figs/$RUN_NAME
mkdir -p "$FIGS_DIR"

finish() { "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" || true; }
cleanup() { finish; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT
trap finish EXIT

REPO=${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-hv3/repo
EXPECTED=$(cat "${SISTEER_PROJECT_ROOT}-hv3/EXPECTED_SHA")
cd "$REPO"
GOT=$(git rev-parse HEAD)
if [ "$GOT" != "$EXPECTED" ]; then
  echo "SHA MISMATCH: checkout $GOT != EXPECTED_SHA $EXPECTED — refusing" >&2
  exit 3
fi
echo "checkout at $GOT (EXPECTED_SHA matched)"

uv sync --frozen --extra cu128 || { echo "FATAL: uv sync --frozen --extra cu128 failed" >&2; exit 4; }
# Runtime GPU guard (the 61380 silent-CPU lesson): the flavored sync
# must have landed the cu128 torch on a live CUDA device — assert, never
# just print.
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__)'
export PYTHONPATH="$REPO/src"
srun "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bench_batch_forward.py \
  --phase asr_noise --family renellm --n-goals 40 --max-new-tokens 256 \
  --out-root "$FIGS_DIR"
