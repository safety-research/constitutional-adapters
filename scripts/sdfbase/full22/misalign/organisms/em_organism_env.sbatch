#!/bin/bash
# scripts/sdfbase/full22/misalign/organisms/em_organism_env.sbatch — build the `em-organism` venv on a Slurm cluster (CPU-only): the
# sdfbase-fit-tf515 stack (scripts/modal/requirements.txt lockfile export: torch 2.11.0+cu128, transformers 5.15.0,
# accelerate 1.14.0, wandb 0.28.1; hybrid-stack pins triton 3.7.1 + fla 0.5.2; causal-conv1d 1.6.2.post1 source build)
# PLUS peft 0.20.0 (2026-07-28) for the EM model-organism LoRA fits. mamba-ssm is NOT built: Qwen3.5 is
# Gated-DeltaNet + full attention (fla), not Mamba. A shared venv is never modified in place under peers' jobs, so this is
# a NEW env name, not `uv pip install peft` into sdfbase-fit-tf515.
# Stage + submit (from the grading host):
#   ssh <login host> 'mkdir -p <shared root>/envs/em-organism.build && cp <shared root>/envs/sdfbase-fit-tf515.build/requirements.txt <shared root>/envs/em-organism.build/'
#   scp scripts/sdfbase/full22/misalign/organisms/em_organism_env.sbatch <login host>:<shared root>/envs/em-organism.build/
#   ssh <login host> SISTEER_SHARED_ROOT=<shared root> sbatch <shared root>/envs/em-organism.build/em_organism_env.sbatch
#   (<login host>, <shared root>: the cluster block's CLUSTER_SSH and CLUSTER_SHARED_ROOT in the site config)
# Positive completion artifact: the LAST log line is "[env] BUILD-COMPLETE <sha256 of freeze.txt>" and
# $SHARED/envs/em-organism/BUILD-COMPLETE exists (written only after every assertion passed).
#SBATCH --job-name=em-organism-env-build
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
set -uo pipefail
# This is a CPU-only VENV BUILD, not a job that runs project code: it lives beside the misalign eval scripts (outside the
# scripts/sbatch torch-flavor audit, like misalign_proxy.sbatch) and carries no exemption marker — the bake step asserts
# torch == 2.11.0+cu128 by wheel identity before BUILD-COMPLETE is written, and every job that RUNS from the venv
# (em_organism_train.sbatch) runs the live cu128 + CUDA assert before its workload.
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"
export ENV_NAME="${1:-em-organism}"
export BUILD_DIR="$SHARED/envs/$ENV_NAME.build" ENV_DIR="$SHARED/envs/$ENV_NAME"
export PYBASE="$SHARED/.uv-pythons/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12"
export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR="$SHARED/.uv-pythons"
export CUDA_HOME=/usr/local/cuda-12.8 CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='9.0;10.0'
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
export TMPDIR="$SHARED/tmp/env-build-${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR" "$SHARED/envs"
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

build_env() {
  set -euo pipefail
  echo "[env] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-?} name=$ENV_NAME env=$ENV_DIR build=$BUILD_DIR cpus=$MAX_JOBS"
  [ -x "$PYBASE" ] || { echo "[env] FATAL shared interpreter missing: $PYBASE"; exit 3; }
  [ -s "$BUILD_DIR/requirements.txt" ] || { echo "[env] FATAL $BUILD_DIR/requirements.txt missing/empty"; exit 3; }
  [ -x "$CUDA_HOME/bin/nvcc" ] || { echo "[env] FATAL nvcc missing at $CUDA_HOME"; exit 3; }
  rm -f "$ENV_DIR/BUILD-COMPLETE"
  [ -x "$ENV_DIR/bin/python" ] || uv venv --python "$PYBASE" "$ENV_DIR"
  PYBIN="$ENV_DIR/bin/python"
  uvpi() { uv pip install --python "$PYBIN" "$@"; }
  echo "[env] step 1/5 lockfile requirements (cu128 torch tree)"
  uvpi -r "$BUILD_DIR/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
  echo "[env] step 2/5 hybrid-stack pins (triton 3.7.1, fla 0.5.2) + causal-conv1d 1.6.2.post1 source build"
  uvpi triton==3.7.1 fla-core==0.5.2 flash-linear-attention==0.5.2 ninja==1.13.0
  CAUSAL_CONV1D_FORCE_BUILD=TRUE uvpi --no-build-isolation causal-conv1d==1.6.2.post1
  uvpi --no-deps triton==3.7.1
  echo "[env] step 3/5 peft 0.20.0 (--no-deps: every dependency is already pinned by the lockfile export)"
  uvpi --no-deps peft==0.20.0
  echo "[env] step 4/5 bake assertions"
  "$PYBIN" - <<'PY'
import torch, transformers, accelerate, wandb, peft, triton, causal_conv1d, fla, datasets, safetensors
want = {"torch": "2.11.0+cu128", "transformers": "5.15.0", "accelerate": "1.14.0", "wandb": "0.28.1",
        "peft": "0.20.0", "triton": "3.7.1"}
got = {"torch": torch.__version__, "transformers": transformers.__version__, "accelerate": accelerate.__version__,
       "wandb": wandb.__version__, "peft": peft.__version__, "triton": triton.__version__}
bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
assert not bad, f"version drift: {bad}"
from peft import LoraConfig, get_peft_model
LoraConfig(r=32, lora_alpha=64, use_rslora=True, target_modules=["q_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
print("[env] versions OK:", got, "| datasets", datasets.__version__, "| safetensors", safetensors.__version__)
PY
  echo "[env] step 5/5 freeze + completion marker"
  uv pip freeze --python "$PYBIN" > "$BUILD_DIR/freeze.txt"
  sha=$(sha256sum "$BUILD_DIR/freeze.txt" | cut -c1-16)
  printf '%s %s\n' "$(date -Is)" "$sha" > "$ENV_DIR/BUILD-COMPLETE"
  echo "[env] BUILD-COMPLETE $sha"
}
export -f build_env
srun --ntasks=1 bash -c build_env
rc=$?
rm -rf "$TMPDIR"
echo "[env] exit rc=$rc"
exit $rc
