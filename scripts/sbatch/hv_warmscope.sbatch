#!/bin/bash
#SBATCH --job-name=hv_warmscope
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-00:55:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# warmup_scope on 1xH200: is the 36x per-shape cold-start GLOBAL (one max-shape
# warmup at server start fixes everything) or PER-SHAPE / PER-KV-LENGTH (prompt
# bucketing becomes load-bearing)? Runs the phase TWICE in separate processes:
# once under the production PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# (the arena-growth prime suspect) and once with it unset (native segments) —
# cold-start state is per-process, so each srun is a fresh cold start.
# SHA-gated own checkout (hv3), own node-local venv, as hv_steptrace.

set -uo pipefail

SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
source "$SHARED/scripts/hf_env.sh"        # HF_HOME + per-user HF_TOKEN_PATH
export NCCL_NVLS_ENABLE=0
export HF_HUB_OFFLINE=1                    # the NousResearch model is cached
export UV_PROJECT_ENVIRONMENT=/home/$USER/venvs/sisteer-hv3
export UV_CACHE_DIR=$SHARED/.cache/uv
export UV_LINK_MODE=copy

RUN_NAME=${SLURM_JOB_NAME}
export FIGS_DIR=$SHARED/exp/figs/$RUN_NAME
mkdir -p "$FIGS_DIR"

# push the sanitised summaries to the figure mirror even on crash/preemption.
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

# Arm 1: production allocator setting (expandable segments — the suspect).
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
srun "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bench_batch_forward.py \
  --phase warmup_scope --out-root "$FIGS_DIR/expandable"

# Arm 2: native segments (unset), fresh process = fresh cold start.
unset PYTORCH_CUDA_ALLOC_CONF
srun "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bench_batch_forward.py \
  --phase warmup_scope --out-root "$FIGS_DIR/native"
