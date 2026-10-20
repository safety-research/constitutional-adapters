#!/bin/bash
#SBATCH --job-name=mamba-build
#SBATCH --partition=general
#SBATCH --qos=high                    # cluster A default (1 of the 16-GPU high budget). #SBATCH headers
#                                     # cannot branch per cluster, so on cluster B (its QoS differs) submit
#                                     # through submit_mamba_build.sh, which derives the QoS and the two
#                                     # log paths below at submit time and overrides all three
#SBATCH --gres=gpu:1                  # the compile needs nvcc only; the USED-not-importable proof needs the GPU
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G                    # NEVER --mem=0 (hard rule 2)
#SBATCH --time=3:00:00                # measured: 1068 s compile at MAX_JOBS=8 (job 1001860) + ~10 min toolkit
#                                     # install where one is needed + the proofs; padded, not 5x
# The optional campaign-seam check reads the class_b prompt lists (benign AND
# harmful, via build_class_b_data), so a traceback could print harmful text:
# logs go to the mode-0700 PRIVATE dir, like alphasteer_fit_f22.sbatch (JB4).
# Cluster A default; Slurm never creates log parents and a nonexistent path
# means the job NEVER STARTS -- submit_mamba_build.sh mkdirs the cluster's
# dir and overrides both paths on either cluster.
#SBATCH --output=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.out
#SBATCH --error=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.err
#SBATCH --open-mode=append
#
# Warm uv's SHARED cache with the fused Mamba kernels, once per cluster, and
# prove they are USED, not merely importable. pyproject's `mamba` extra is
# mamba-ssm 2.3.1 + causal-conv1d 1.6.2.post1 as pinned PyPI SDISTS (since
# 2026-10-12, replacing the self-built path wheels this script used to
# deposit in wheels/): PyPI carries only sdists for both and neither project
# publishes a torch-2.11 wheel (their GitHub releases stop at torch 2.10 plus
# the NGC builds, checked 2026-09-09 and 2026-10-12), so uv COMPILES them
# against the LOCKFILE's torch (2.11.0+cu128, H200 / CUDA 12.8;
# [tool.uv.extra-build-dependencies] match-runtime) inside
# `uv sync --frozen --extra cu128 --extra mamba`, with MAMBA_FORCE_BUILD /
# CAUSAL_CONV1D_FORCE_BUILD set by [tool.uv.extra-build-variables] so neither
# setup.py fetches a GitHub wheel. The built wheels land in UV_CACHE_DIR --
# the SAME $SHARED/.cache/uv every s2_init sync reads -- so after this job a
# Nemotron fit's sync (alphasteer_fit_f22.sbatch, `env_extras: mamba`)
# installs them in seconds instead of compiling for an hour inside its own
# budget. The pins: 2.3.1 is the newest mamba-ssm WITHOUT the tilelang /
# quack-kernels / apache-tvm-ffi tree 2.3.2 grew for Mamba3, which
# transformers' NemotronH never touches.
#
# THE TOOLKIT. The compile needs a CUDA *12* toolkit: torch.utils.cpp_extension
# refuses to build against the cu128 torch with another major ("The detected
# CUDA version (13.0) mismatches the version that was used to compile PyTorch
# (12.8)"). Cluster A's /usr/local/cuda is 12.8; cluster B's is 13.0 with no
# 12.x anywhere (its login host and a compute node, 2026-10-12), which is why
# a cold-cache fit on cluster B died inside uv's build.
# scripts/sbatch/cuda_toolkit.sh (shared with s2_init, so the fit path selects
# exactly what this job warmed with) picks and validates the toolkit; when it
# finds none, THIS job -- the once-per-cluster step -- installs CUDA 12.8.1
# under $SHARED/cuda-12.8 from NVIDIA's runfile (toolkit only, no driver, no
# root, nothing under /usr), pinned by URL, size, NVIDIA's published md5 and
# the sha256 of the download that matched it, and selects it. Compute nodes
# ship their toolkit with nvcc OFF PATH (471132); the selection puts the chosen
# bin first on PATH, so the toolchain the build sees is the one named.
#
# Neither setup.py honours TORCH_CUDA_ARCH_LIST (both pass their own -gencode
# list; torch's cpp_extension then adds none), so the build covers every arch
# the package ships for -- ~10 on CUDA 12.8 -- and takes 18-60 min. MAX_JOBS
# bounds the parallel nvcc jobs (each is memory-heavy: ~15 GB peak nvcc-family
# RSS at MAX_JOBS=8, job 1001860).
#
# Positional (through submit_mamba_build.sh: after a `--`):
#   $1  CODE     the checkout to sync (its pyproject + uv.lock define the
#                build) and run the proof from (default: $S2_ROOT/repo)
#   $2  CONFIG   optional: a Mamba-hybrid target config, e.g.
#                configs/alphasteer_f22_nemotron3nano4b.yaml. When given, the
#                campaign-seam check (scripts/mamba_kernels_check.py: the
#                fused run, then the --fallback baseline) runs after the
#                tiny-model proof; it needs the target in the HF cache and the
#                config's prompt lists on this cluster.
# The proof's JSON readings land in $S2_ROOT/mamba-build/. Nothing is copied
# anywhere: the product is the cache entry (plus, on a cluster without one,
# the toolkit), and the scratch venv below is only the proof's.
set -uo pipefail
# The shared root is SISTEER_SHARED_ROOT, which the front door exports from the cluster's site block; a job
# without it refuses by name (there is no built-in root).
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
S2_ROOT=${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-as22
CODE=${1:-$S2_ROOT/repo}
CONFIG=${2:-}
OUT=$S2_ROOT/mamba-build
export UV_PROJECT_ENVIRONMENT=/home/$USER/venvs/mamba-build   # a SCRATCH env: the warm-up's product is the cache entry
export UV_CACHE_DIR=$SHARED/.cache/uv UV_LINK_MODE=copy       # the SAME cache every s2_init sync reads -- the point
export MAX_JOBS=16
export SHARED CODE CONFIG OUT
[ -s "$SHARED/scripts/hf_env.sh" ] || { echo "FATAL: $SHARED/scripts/hf_env.sh missing" >&2; exit 1; }
source "$SHARED/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# the checks print their readings as they go; under srun a block-buffered
# stdout would hold them all until exit (471132 showed nothing for minutes)
export PYTHONUNBUFFERED=1
mkdir -p "$OUT"
cd "$CODE" || exit 1
echo "[mamba-build] $(date -Is) job ${SLURM_JOB_ID:-?} on $(hostname): SHARED=$SHARED CODE=$CODE @ $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null || echo '?') cache=$UV_CACHE_DIR uv=$(uv --version 2>/dev/null || echo '?')"

# The toolkit (see THE TOOLKIT above). NVIDIA's CUDA 12.8.1 Linux runfile:
# size and md5 from https://developer.download.nvidia.com/compute/cuda/12.8.1/docs/sidebar/md5sum.txt
# (fetched 2026-10-12); the sha256 is of the download that matched both.
CUDA_RUNFILE_URL=https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/cuda_12.8.1_570.124.06_linux.run
CUDA_RUNFILE_SIZE=5382238770
CUDA_RUNFILE_MD5=e281945bf4ea42788db29457ebc1e6ae
CUDA_RUNFILE_SHA256=228f6bcaf5b7618d032939f431914fc92d0e5ed39ebe37098a24502f26a19797
CUDA_TOOLKIT_DIR="${S2_CUDA_SHARED_TOOLKIT:-$SHARED/cuda-12.8}"
export CUDA_RUNFILE_URL CUDA_RUNFILE_SIZE CUDA_RUNFILE_MD5 CUDA_RUNFILE_SHA256 CUDA_TOOLKIT_DIR
# shellcheck disable=SC1091
source "$CODE/scripts/sbatch/cuda_toolkit.sh"
if ! s2_cuda12_select; then
  echo "[mamba-build] $(date -Is) no CUDA 12 toolkit on this cluster (probes above): installing CUDA 12.8.1, toolkit only, under $CUDA_TOOLKIT_DIR"
  srun --ntasks=1 bash -c '
    set -e
    # node-local scratch (750 GB /home per node): the runfile is 5.4 GB and its
    # extraction about as much again; the installer also needs an exec-capable tmpdir
    mkdir -p "/home/$USER/tmp"
    tmp=$(mktemp -d "/home/$USER/tmp/cuda-12.8-install.XXXXXX")
    trap "rm -rf \"$tmp\"" EXIT
    t0=$(date +%s)
    curl -sS -L --retry 3 -o "$tmp/cuda.run" "$CUDA_RUNFILE_URL"
    size=$(stat -c %s "$tmp/cuda.run")
    [ "$size" = "$CUDA_RUNFILE_SIZE" ] || { echo "runfile size $size != $CUDA_RUNFILE_SIZE"; exit 1; }
    echo "$CUDA_RUNFILE_SHA256  $tmp/cuda.run" | sha256sum -c -
    echo "$CUDA_RUNFILE_MD5  $tmp/cuda.run" | md5sum -c -
    echo "[mamba-build] $(date -Is) runfile verified in $(( $(date +%s) - t0 ))s; installing"
    mkdir -p "$(dirname "$CUDA_TOOLKIT_DIR")"
    # --toolkit only (no driver: the node has one), into our own path, no
    # /usr writes (the installer still TRIES the /usr/local/cuda symlink and
    # warns when it cannot, which is fine); --override skips its host-compiler
    # check (gcc 13.3 on the compute nodes is within nvcc 12.8 supported range)
    sh "$tmp/cuda.run" --silent --toolkit --toolkitpath="$CUDA_TOOLKIT_DIR" --override \
      --no-man-page --no-opengl-libs --no-drm --tmpdir="$tmp"
    "$CUDA_TOOLKIT_DIR/bin/nvcc" --version | tail -1
    echo "[mamba-build] $(date -Is) toolkit installed in $(( $(date +%s) - t0 ))s: $(du -sh "$CUDA_TOOLKIT_DIR" | cut -f1) at $CUDA_TOOLKIT_DIR"
  '
  rc=$?
  [ "$rc" -eq 0 ] || { echo "[mamba-build] FATAL: toolkit install rc=$rc"; exit "$rc"; }
  s2_cuda12_select || { echo "[mamba-build] FATAL: the toolkit just installed at $CUDA_TOOLKIT_DIR did not validate"; exit 1; }
fi
echo "[mamba-build] $(date -Is) warming $UV_CACHE_DIR with the mamba extra of $CODE (CUDA_HOME=$CUDA_HOME, nvcc release $S2_CUDA_RELEASE, MAX_JOBS=$MAX_JOBS)"
srun --ntasks=1 bash -c '
  set -e
  nvcc --version | tail -1
  t0=$(date +%s)
  # THE WARM-UP: the frozen sync builds mamba-ssm 2.3.1 + causal-conv1d
  # 1.6.2.post1 from their sdists on a cold cache (18-60 min) and installs
  # the cached build in seconds on a warm one; set -e makes a failed sync
  # abort the job (no unfrozen fallback).
  uv sync --frozen --extra cu128 --extra mamba
  echo "[mamba-build] $(date -Is) sync done in $(( $(date +%s) - t0 ))s (seconds = cache hit; ~20-60 min = the compile)"
  echo "[mamba-build] cached builds now in $UV_CACHE_DIR:"
  find "$UV_CACHE_DIR"/sdists-v*/pypi/mamba-ssm "$UV_CACHE_DIR"/sdists-v*/pypi/causal-conv1d -name "*.whl" 2>/dev/null || true
  source "$UV_PROJECT_ENVIRONMENT/bin/activate"
  # Runtime GPU guard: the proof must run against cu128 torch on a live
  # CUDA device -- assert, never just print (61380 guard).
  python -c "import torch
assert torch.cuda.is_available(), \"CUDA not available in this env\"
assert torch.cuda.device_count() >= 1, \"no CUDA device visible\"
assert torch.__version__ == \"2.11.0+cu128\", f\"torch {torch.__version__} != 2.11.0+cu128\"
print(\"cu128 runtime assert OK:\", torch.__version__, torch.cuda.get_device_name(0), \"cxx11abi\", torch._C._GLIBCXX_USE_CXX11_ABI)"
  # THE PROOF (used, not importable), on a tiny random-init NemotronH hybrid
  # built on the GPU -- no download: the installed distributions are the
  # pins and compiled builds, every kernel wrapper resolves to its package
  # (the repo loader guard, sisteer.models.assert_fused_mamba_kernels), and
  # a profiled forward launches the fused CUDA kernels. PYTHONPATH puts the
  # CODE checkout ahead of the venv editable install.
  cd "$CODE"
  export PYTHONPATH="$CODE/src:$CODE/scripts"
  echo "[mamba-build] $(date -Is) tiny-model proof"
  python scripts/mamba_tiny_proof.py --json "$OUT/tiny_proof.json"
  if [ -n "$CONFIG" ]; then
    # the campaign seam too: the resolver + a profiled capture batch + rows/s
    # through alphasteer_screen.capture_all_layers on the real target, then
    # the same on the blocked (pure-torch) path for the baseline number
    echo "[mamba-build] $(date -Is) campaign-seam check on $CONFIG (fused)"
    python scripts/mamba_kernels_check.py --config "$CONFIG" --rows 128 --json "$OUT/check_fused.json"
    echo "[mamba-build] $(date -Is) campaign-seam baseline (--fallback)"
    python scripts/mamba_kernels_check.py --config "$CONFIG" --rows 128 --fallback --json "$OUT/check_fallback.json"
  fi
'
RC=$?
echo "[mamba-build] rc=$RC"
[ "$RC" -eq 0 ] || exit "$RC"
echo "[mamba-build] $(date -Is) DONE: $UV_CACHE_DIR holds the mamba-ssm 2.3.1 + causal-conv1d 1.6.2.post1 builds (compiled with CUDA_HOME=$CUDA_HOME); readings in $OUT"
