#!/bin/bash
#SBATCH --job-name=hv_backprobe
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-00:55:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# The mechanism-NAMING probe (after 173917 fixed the SCOPE): the ~295 ms cold
# tax per NEW (B, kv_len) attention shape is the signature of cuDNN SDPA plan
# construction (torch enables cuDNN attention by default on sm90 and caches an
# execution plan per problem geometry — exactly a (B, kv_len) key; width
# transfers, batch does not, matching 173917's B/C/D calls precisely). Two arms,
# each a FRESH process (cold state is per-process), same warmup_scope plan:
#   flash — pin SDPBackend.FLASH_ATTENTION. Cold plateau GONE => mechanism
#           named AND the fix becomes a one-line backend pin, superseding
#           warmup choreography entirely.
#   cudnn — pin SDPBackend.CUDNN_ATTENTION, the positive control: it should
#           REPRODUCE the plateau if cuDNN plan build is the mechanism.
# Production allocator setting throughout (exonerated by 173917's A/B).
# SHA-gated own checkout (hv3), own node-local venv, as hv_steptrace.

set -uo pipefail

SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
source "$SHARED/scripts/hf_env.sh"        # HF_HOME + per-user HF_TOKEN_PATH
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1                    # the NousResearch model is cached
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

# Arm 1: flash-pinned — the plateau should VANISH if cuDNN plan build is it.
srun "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bench_batch_forward.py \
  --phase warmup_scope --sdpa-backend flash --out-root "$FIGS_DIR/flash"

# Arm 2: cudnn-pinned — the positive control; should REPRODUCE the plateau.
srun "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bench_batch_forward.py \
  --phase warmup_scope --sdpa-backend cudnn --out-root "$FIGS_DIR/cudnn"
