#!/bin/bash
#SBATCH --job-name=hv_bench3
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-00:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue

# GPU validation of the serving-phase probe forward-pair + a held-out family
# smoke, on 1xH200. qos=high (16-GPU/user budget; a 1-GPU job fits trivially).
# Runs from an OWN checkout gated on EXPECTED_SHA — never repoints a peer's
# shared checkout. Own node-local venv (sisteer-hv) so a peer's job sharing
# the node keeps its own sisteer editable install.

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

# push the sanitised summary to the figure mirror even on crash/preemption.
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

# build the node-local venv from THIS checkout (sisteer editable -> this repo);
# --frozen uses the lockfile as-is and skips the optional flash-attn extra.
uv sync --frozen --extra cu128 || { echo "FATAL: uv sync --frozen --extra cu128 failed" >&2; exit 4; }
# Runtime GPU guard (the 61380 silent-CPU lesson): the flavored sync
# must have landed the cu128 torch on a live CUDA device — assert, never
# just print.
"$UV_PROJECT_ENVIRONMENT/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__)'

# belt: our checkout's src wins even if the editable install ever points elsewhere.
export PYTHONPATH="$REPO/src"
srun "$UV_PROJECT_ENVIRONMENT/bin/python" scripts/bench_batch_forward.py --phase generate_long \
  --out-root "$FIGS_DIR"
