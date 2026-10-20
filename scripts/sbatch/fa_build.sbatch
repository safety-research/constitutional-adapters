#!/bin/bash
#SBATCH --job-name=fa-build
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=3:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
WHEELS=${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}/wheels
# flash-attn's setup.py IGNORES TORCH_CUDA_ARCH_LIST and builds its full
# default arch set (sm_80/90/100/120) -> 4x the compile memory, OOM-killed
# even at 240G/MAX_JOBS=16 (job 100097). FLASH_ATTN_CUDA_ARCHS is its real
# knob: build only sm_90 (H200). Halve MAX_JOBS too (each nvcc job is
# memory-heavy). Env built + torch 2.11 CUDA confirmed on 100097; the
# compile itself was progressing (16/73 objects) before the OOM.
export UV_PROJECT_ENVIRONMENT=/home/$USER/venvs/fa-build
export UV_CACHE_DIR=$SHARED/.cache/uv UV_LINK_MODE=copy
export FLASH_ATTN_CUDA_ARCHS="90" MAX_JOBS=8 FLASH_ATTENTION_FORCE_BUILD=TRUE
mkdir -p "$WHEELS"
cd "$SISTEER_PROJECT_ROOT/repo" || exit 1
echo "[fa-build] $(date -Is) building flash-attn 2.8.3.post1 (sm_90 only) vs lockfile torch 2.11"
srun bash -c '
  set -e
  # torch-cpu-split: the GPU flavor must be selected explicitly; set -e
  # makes a failed sync abort the build (no unfrozen fallback).
  uv sync --frozen --extra cu128
  source '"$UV_PROJECT_ENVIRONMENT"'/bin/activate
  uv pip install pip ninja packaging psutil setuptools wheel
  # Runtime GPU guard: the wheel must be built against cu128 torch on
  # a live CUDA device — assert, not just print (61380 guard).
  python -c "import torch
assert torch.cuda.is_available(), \"CUDA not available in this env\"
assert torch.cuda.device_count() >= 1, \"no CUDA device visible\"
assert torch.__version__ == \"2.11.0+cu128\", f\"torch {torch.__version__} != 2.11.0+cu128\"
print(\"cu128 runtime assert OK:\", torch.__version__, torch.cuda.get_device_name(0))"
  python -m pip wheel "flash-attn==2.8.3.post1" --no-build-isolation --no-deps -w '"$WHEELS"'
'
RC=$?
echo "[fa-build] wheel build rc=$RC"
if [ "$RC" -eq 0 ]; then
  WHL=$(ls -t "$WHEELS"/flash_attn-*.whl 2>/dev/null | head -1)
  echo "[fa-build] built: $WHL"
  srun bash -c '
    source '"$UV_PROJECT_ENVIRONMENT"'/bin/activate
    uv pip install "'"$WHL"'"
    python - <<PY
import torch, flash_attn
from flash_attn import flash_attn_func
print("flash_attn", flash_attn.__version__, "torch", torch.__version__)
q=torch.randn(1,4,2,16,dtype=torch.bfloat16,device="cuda")
print("smoke OK, out", tuple(flash_attn_func(q,q,q).shape))
PY
  '
  SMOKE_RC=$?
  echo "[fa-build] import+smoke rc=$SMOKE_RC"
  if [ "$SMOKE_RC" -eq 0 ]; then
    # Deposit the verified wheel where uv.lock's [tool.uv.sources] path
    # expects it (repo wheels/ is gitignored — survives git reset --hard,
    # NOT git clean -fdx). `uv sync --frozen --extra cu128 --extra longctx`
    # (the longctx sbatch, via S2_UV_EXTRAS) installs from exactly this
    # path. NOTE: a REBUILT wheel has a new sha256 and uv.lock pins the
    # old one — after any rebuild, re-run `uv lock` on the grading host (wheel
    # rsync'd into the worktree's wheels/) and commit the lock bump.
    mkdir -p "$SISTEER_PROJECT_ROOT/repo/wheels"
    cp -v "$WHL" "$SISTEER_PROJECT_ROOT/repo/wheels/"
  fi
fi
