#!/bin/bash
# scripts/sbatch/sdfbase_fit_env.sbatch — build the SHARED sdf-base-steering fit
# venv on a Slurm cluster (cluster A or cluster B), CPU-only, no GPU budget.
#
# The venv reproduces scripts/modal/sdfbase_fits.py's `image_nemotron` recipe
# step for step — the superset of image_std (dense GLM) and image_qwen (hybrid)
# — so every Slurm fit cell runs the trainer on the SAME pinned stack the Modal
# lane runs: the lockfile-exported scripts/modal/requirements.txt (torch
# 2.11.0+cu128, transformers 5.15.0, accelerate 1.14.0, wandb 0.28.1 — no
# peft: the lockfile export pins none), then the hybrid-stack pins (triton 3.7.1 OVER the lockfile's 3.6.0,
# fla-core / flash-linear-attention 0.5.2), causal-conv1d 1.6.2.post1 built from
# source for sm90/sm100 (no wheel exists at torch 2.11), and mamba-ssm
# 2.3.2.post1 built from source with --no-deps (its Mamba3-only tilelang/tvm
# chain is not needed for NemotronH's Mamba2 fast path) — with the SAME bake
# assertions the Modal image runs, so a regressed stack fails the BUILD, not the
# first fit. See the image_nemotron docstring in sdfbase_fits.py for the why of
# every pin.
#
# Layout (shared storage, visible on every compute node — a node-local
# /home/$USER venv would have to be rebuilt per node, and the two CUDA source
# builds take ~30-40 min). <name> is the env-dir name: positional 1, else
# SDFBASE_FIT_ENV_NAME (the submitting shell's environment, through Slurm's
# implicit export — never --export=, hard rule 1), else `sdfbase-fit`
# (today's path, byte for byte). A NEW stack goes into a NEW name
# (`sdfbase-fit-tf515`) and the fit / derived sbatch are pointed at it through
# the same knob or the bank's FIT_ENV_NAME file — a shared venv is never
# rebuilt in place under a peer's running job (O6, 2026-09-16):
#   $SHARED/envs/<name>.build/requirements.txt       copied from the repo
#   $SHARED/envs/<name>/                             the venv (the twosweep-fit
#                                                     precedent on $SHARED/envs)
#   $SHARED/.uv-pythons/cpython-3.12.11-.../python3.12  the shared managed
#                                                     interpreter (twosweep-fit's)
# Stage + submit (from the grading host; <name> defaults to sdfbase-fit):
#   scp scripts/modal/requirements.txt scripts/sbatch/sdfbase_fit_env.sbatch \
#       <login host>:<shared root>/envs/<name>.build/
#   ssh <login host> SISTEER_SHARED_ROOT=<shared root> sbatch <shared root>/envs/<name>.build/sdfbase_fit_env.sbatch <name>
#   (<login host>, <shared root>: the cluster block's CLUSTER_SSH and CLUSTER_SHARED_ROOT in the site config)
# Positive completion artifact: the LAST line of the log is
# "[env] BUILD-COMPLETE <sha256 of freeze.txt>" and $SHARED/envs/<name>/
# BUILD-COMPLETE exists (written only after every assertion passed).
#
# Host knobs (O7, 2026-09-16) — each defaults to today's cluster A behaviour
# byte for byte; a cluster B build sets them (its toolkit is CUDA 13, its
# shared interpreter 3.12.13). Each is read from the submitting shell's
# environment through Slurm's implicit export (never --export=, hard rule 1):
#   SDFBASE_PY             the interpreter: positional 2 > SDFBASE_PY > the shared
#                          3.12.11 (twosweep-fit's — today's path) > the highest-
#                          patch $SHARED/.uv-pythons/cpython-3.12.*-linux-x86_64-gnu
#                          (cluster B carries 3.12.13 only). The log names which.
#                          Nothing is ever `uv python install`ed here, and the uv
#                          minor-version link cpython-3.12-linux-x86_64-gnu (which
#                          peers' strength-dose-fit resolves through) is neither a
#                          candidate nor ever re-pointed.
#   CUDA_HOME              honoured when set, else /usr/local/cuda-12.8; nvcc is
#                          asserted only when a SOURCE build is requested.
#   SDFBASE_KERNEL_SOURCE  `source` (default): the forced source builds above.
#                          `wheel`: the SAME two pinned versions installed from the
#                          prebuilt cu128 wheels in $SDFBASE_WHEELS (default
#                          $SHARED/wheels) with --no-index --no-deps, for a host whose
#                          nvcc refuses to build against torch cu128 (CUDA 13:
#                          "The detected CUDA version (13.0) mismatches ..."). The
#                          wheels are uv's own build output from the cluster A source
#                          build (its sdists cache, 2026-09-07): kernel .so sha256
#                          7f08f446... (causal_conv1d_cuda) / 6aa59172...
#                          (selective_scan_cuda) — the very files installed in
#                          $SHARED/envs/sdfbase-fit, sdfbase-fit-tf515 and
#                          cluster B's strength-dose-fit (dose_env.sbatch carried
#                          them by copy). The log records each wheel's sha256 and
#                          each installed .so's; the bake assertions (5/6) run in
#                          both modes, plus the two kernel versions in wheel mode.
# Cluster B (from a per-sha checkout; the header's --output root does not exist there):
#   SISTEER_SHARED_ROOT=<shared root> SDFBASE_KERNEL_SOURCE=wheel CUDA_HOME=/usr/local/cuda sbatch --qos=<its CLUSTER_QOS> \
#     --output=<shared root>/exp/logs/%x_%j.out scripts/sbatch/sdfbase_fit_env.sbatch <name>
#SBATCH --job-name=sdfbase-fit-env-build
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
set -uo pipefail

# audit-exempt: out-of-lockfile-venv — this CPU-only job BUILDS the shipped
# sdfbase-fit venv with pip from the lockfile EXPORT (scripts/modal/
# requirements.txt) plus the hybrid-stack pins the Modal image_nemotron
# recipe carries; there is no `uv sync` to make frozen and no GPU on the
# allocation to assert CUDA on, so the frozen-cu128-sync + CUDA-assert
# guard does not apply. The bake step (5/6) asserts the wheel identity
# torch == 2.11.0+cu128 before BUILD-COMPLETE is written, and every job
# that RUNS from this venv (sdfbase_fit_cell / sdfbase_derived_build) runs
# the live-CUDA cu128 assert itself before its workload.

SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

# the env-dir NAME: positional 1 > SDFBASE_FIT_ENV_NAME > sdfbase-fit; the two full paths
# below still honour an explicit SDFBASE_BUILD_DIR / SDFBASE_FIT_ENV
export SDFBASE_FIT_ENV_NAME="${1:-${SDFBASE_FIT_ENV_NAME:-sdfbase-fit}}"
export SDFBASE_BUILD_DIR="${SDFBASE_BUILD_DIR:-$SHARED/envs/$SDFBASE_FIT_ENV_NAME.build}"
export SDFBASE_FIT_ENV="${SDFBASE_FIT_ENV:-$SHARED/envs/$SDFBASE_FIT_ENV_NAME}"
# the INTERPRETER: positional 2 > SDFBASE_PY > the shared 3.12.11 (today's path) > the
# highest-patch cpython-3.12.* under $SHARED/.uv-pythons. The glob requires the patch
# component, so the cpython-3.12-linux-x86_64-gnu minor-version link is never a
# candidate; nothing is installed. An empty result fails the preflight below.
SDFBASE_PY_DEFAULT="$SHARED/.uv-pythons/cpython-3.12.11-linux-x86_64-gnu/bin/python3.12"
if [ -n "${2:-}" ]; then SDFBASE_PY="$2"; SDFBASE_PY_FROM="positional 2"
elif [ -n "${SDFBASE_PY:-}" ]; then SDFBASE_PY_FROM="SDFBASE_PY (environment)"
elif [ -x "$SDFBASE_PY_DEFAULT" ]; then SDFBASE_PY="$SDFBASE_PY_DEFAULT"; SDFBASE_PY_FROM="shared default"
else
  SDFBASE_PY="$(ls -d "$SHARED"/.uv-pythons/cpython-3.12.*-linux-x86_64-gnu/bin/python3.12 2>/dev/null | sort -V | tail -1)"
  SDFBASE_PY_FROM="highest cpython-3.12.* under $SHARED/.uv-pythons (shared default absent: $SDFBASE_PY_DEFAULT)"
fi
export SDFBASE_PY SDFBASE_PY_FROM
export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR="$SHARED/.uv-pythons"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.8}" CC=gcc CXX=g++ TORCH_CUDA_ARCH_LIST='9.0;10.0'
# the fused-kernel SOURCE: source (today's forced builds) | wheel (prebuilt cu128 wheels)
export SDFBASE_KERNEL_SOURCE="${SDFBASE_KERNEL_SOURCE:-source}"
export SDFBASE_WHEELS="${SDFBASE_WHEELS:-$SHARED/wheels}"
case "$SDFBASE_KERNEL_SOURCE" in
  source|wheel) ;;
  *) echo "[env] FATAL SDFBASE_KERNEL_SOURCE must be source|wheel, got '$SDFBASE_KERNEL_SOURCE'"; exit 3 ;;
esac
export MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
# Source builds write GBs of intermediates: never /tmp (node-local, small).
export TMPDIR="$SHARED/tmp/env-build-${SLURM_JOB_ID:-local}"
mkdir -p "$TMPDIR" "$SHARED/envs"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

build_env() {
  set -euo pipefail
  echo "[env] $(date -Is) host=$(hostname) job=${SLURM_JOB_ID:-?} name=$SDFBASE_FIT_ENV_NAME env=$SDFBASE_FIT_ENV build=$SDFBASE_BUILD_DIR py=$SDFBASE_PY cpus=$MAX_JOBS"
  echo "[env] interpreter from: $SDFBASE_PY_FROM; CUDA_HOME=$CUDA_HOME; kernels=$SDFBASE_KERNEL_SOURCE (wheels dir $SDFBASE_WHEELS)"
  [ -x "$SDFBASE_PY" ] || { echo "[env] FATAL shared interpreter missing: $SDFBASE_PY ($SDFBASE_PY_FROM)"; exit 3; }
  [ -s "$SDFBASE_BUILD_DIR/requirements.txt" ] || { echo "[env] FATAL $SDFBASE_BUILD_DIR/requirements.txt missing/empty"; exit 3; }
  CC1D_WHL="$SDFBASE_WHEELS/causal_conv1d-1.6.2.post1-cp312-cp312-linux_x86_64.whl"
  MAMBA_WHL="$SDFBASE_WHEELS/mamba_ssm-2.3.2.post1-cp312-cp312-linux_x86_64.whl"
  if [ "$SDFBASE_KERNEL_SOURCE" = source ]; then
    [ -x "$CUDA_HOME/bin/nvcc" ] || { echo "[env] FATAL nvcc missing at $CUDA_HOME"; exit 3; }
  else
    for w in "$CC1D_WHL" "$MAMBA_WHL"; do
      [ -s "$w" ] || { echo "[env] FATAL prebuilt cu128 wheel missing: $w"; exit 3; }
    done
    echo "[env] kernel wheels (sha256):"; sha256sum "$CC1D_WHL" "$MAMBA_WHL"
  fi
  command -v gcc >/dev/null || { echo "[env] FATAL gcc missing"; exit 3; }
  rm -f "$SDFBASE_FIT_ENV/BUILD-COMPLETE"
  if [ ! -x "$SDFBASE_FIT_ENV/bin/python" ]; then
    uv venv --python "$SDFBASE_PY" "$SDFBASE_FIT_ENV"
  fi
  PYBIN="$SDFBASE_FIT_ENV/bin/python"
  # --no-config: the build reads no uv configuration file, so it is the same from whatever directory it was submitted. Submitted
  # from inside a checkout, uv would read pyproject.toml's [tool.uv] table (the project environment's), whose match-runtime
  # torch build dependency for causal-conv1d and mamba-ssm refuses the --no-deps kernel installs below ("`torch` was declared as
  # an extra build dependency with `match-runtime = true`, but was not found in the resolution"), prebuilt wheels included. The
  # documented submission (the staged copy, from the login shell) discovers no configuration file, so there this flag is inert.
  uvpi() { uv pip install --no-config --python "$PYBIN" "$@"; }
  echo "[env] step 1/6 lockfile requirements (cu128 torch tree)"
  uvpi -r "$SDFBASE_BUILD_DIR/requirements.txt" \
       --extra-index-url https://download.pytorch.org/whl/cu128 \
       --index-strategy unsafe-best-match
  echo "[env] step 2/6 hybrid-stack pins (D66)"
  uvpi triton==3.7.1 fla-core==0.5.2 flash-linear-attention==0.5.2 ninja==1.13.0
  if [ "$SDFBASE_KERNEL_SOURCE" = source ]; then
    echo "[env] step 3/6 causal-conv1d 1.6.2.post1 source build (sm90/sm100)"
    CAUSAL_CONV1D_FORCE_BUILD=TRUE uvpi --no-build-isolation causal-conv1d==1.6.2.post1
  else
    echo "[env] step 3/6 causal-conv1d 1.6.2.post1 prebuilt cu128 wheel from $SDFBASE_WHEELS"
    uvpi --no-index --find-links "$SDFBASE_WHEELS" --no-deps causal-conv1d==1.6.2.post1
  fi
  uvpi --no-deps triton==3.7.1
  "$PYBIN" -c "import triton; v = triton.__version__; assert v == '3.7.1', v; import causal_conv1d, fla"
  if [ "$SDFBASE_KERNEL_SOURCE" = source ]; then
    echo "[env] step 4/6 mamba-ssm 2.3.2.post1 source build (--no-deps)"
    MAMBA_FORCE_BUILD=TRUE uvpi --no-build-isolation --no-deps mamba-ssm==2.3.2.post1
  else
    echo "[env] step 4/6 mamba-ssm 2.3.2.post1 prebuilt cu128 wheel from $SDFBASE_WHEELS"
    uvpi --no-index --find-links "$SDFBASE_WHEELS" --no-deps mamba-ssm==2.3.2.post1
  fi
  uvpi --no-deps triton==3.7.1
  echo "[env] step 5/6 bake assertions (= image_nemotron's)"
  "$PYBIN" -c "import triton; v = triton.__version__; assert v == '3.7.1', v; import causal_conv1d, fla, mamba_ssm; from mamba_ssm.ops.triton.selective_state_update import selective_state_update; from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined, mamba_split_conv1d_scan_combined; from causal_conv1d import causal_conv1d_fn, causal_conv1d_update"
  "$PYBIN" - <<'PY'
import torch, transformers, accelerate, wandb, safetensors, tokenizers, numpy
# (no peft: scripts/modal/requirements.txt pins none — the CE arms' LoRA is the
# repo's own MultiLayerLoRAObject; only the unused SFT arm imports peft lazily)
want = {"torch": "2.11.0+cu128", "transformers": "5.15.0",
        "accelerate": "1.14.0", "wandb": "0.28.1"}
got = {"torch": torch.__version__, "transformers": transformers.__version__,
       "accelerate": accelerate.__version__, "wandb": wandb.__version__}
bad = {k: (got[k], want[k]) for k in want if got[k] != want[k]}
assert not bad, f"version drift vs scripts/modal/requirements.txt: {bad}"
print("[env] versions OK:", got, "safetensors", safetensors.__version__,
      "tokenizers", tokenizers.__version__, "numpy", numpy.__version__)
PY
  if [ "$SDFBASE_KERNEL_SOURCE" = wheel ]; then
    # a wheel install can silently be the wrong build: pin the two kernel versions
    # to the source path's, and record which .so bytes landed (cf. the header)
    "$PYBIN" - <<'PY'
import hashlib, importlib.metadata as m, torch, causal_conv1d_cuda, selective_scan_cuda
want = {"causal-conv1d": "1.6.2.post1", "mamba-ssm": "2.3.2.post1"}
got = {k: m.version(k) for k in want}
assert got == want, f"kernel version drift vs the source-build pins: {got} != {want}"
for mod in (causal_conv1d_cuda, selective_scan_cuda):
    digest = hashlib.sha256(open(mod.__file__, "rb").read()).hexdigest()
    print("[env] kernel .so sha256", digest[:16], mod.__file__)
print("[env] kernel versions OK:", got)
PY
  fi
  echo "[env] step 6/6 freeze + completion marker"
  uv pip freeze --python "$PYBIN" > "$SDFBASE_BUILD_DIR/freeze.txt"
  sha=$(sha256sum "$SDFBASE_BUILD_DIR/freeze.txt" | cut -c1-16)
  printf '%s %s\n' "$(date -Is)" "$sha" > "$SDFBASE_FIT_ENV/BUILD-COMPLETE"
  echo "[env] BUILD-COMPLETE $sha"
}
export -f build_env
srun --ntasks=1 bash -c build_env
rc=$?
rm -rf "$TMPDIR"
echo "[env] exit rc=$rc"
exit $rc
