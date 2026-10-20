#!/bin/bash
#SBATCH --job-name=dose-env
#SBATCH --partition=general,overflow
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=0-01:30:00
#SBATCH --open-mode=append
#SBATCH --no-requeue
#
# Build the strength-dose capture venv ONCE, on SHARED storage, for a cluster that has no
# $SHARED/envs/sdfbase-fit (cluster B). CPU only — no --gres: this is a pip install.
#
# WHY NOT NODE-LOCAL, the campaign's usual choice: four capture jobs land on four different nodes, so a
# node-local venv is built four times; the imports off shared storage cost ~1 min per job against ~15 min
# per redundant build. WHY NOT A SOURCE BUILD: the fused Mamba kernels are CARRIED — the exact
# `.so`s and packages cluster A's fit env built (mamba_ssm 2.3.2.post1, causal_conv1d 1.6.2.post1,
# against torch 2.11.0+cu128), copied over and sha-verified, so nvcc is not needed here (cluster B has
# no nvcc on PATH) and the binaries are the ones a Nemotron capture must have.
#
# The pins are the FIT stack's, not the serving stack's: these objects were fit under torch 2.11.0+cu128
# / transformers 5.14.1, and the capture is an HF forward of the same base model. The set is slim on
# purpose (this job runs a forward and reads blobs; it serves nothing), and the exact resolved versions
# are recorded in the artifact.
#
#   sbatch -J dose-env --qos=<cluster high QoS> --output=<shared>/exp/logs/%x_%j.out \
#     scripts/sdfbase/full22/strength/dose_env.sbatch <sha>
set -uo pipefail
SHA="${1:?sha}"
SHARED="${SISTEER_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-}}"   # the cluster's shared root: the job's env prefix, else its block value
[ -n "$SHARED" ] && [ -d "$SHARED" ] || { echo "FATAL: no shared root on $(hostname): set SISTEER_SHARED_ROOT (the site config's CLUSTER_SHARED_ROOT)" >&2; exit 1; }
VENV="$SHARED/envs/strength-dose-fit"
OV="$SHARED/envs/strength-dose-kernels"
export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR="$SHARED/.uv-pythons"
export TMPDIR="$SHARED/tmp/dose-env-${SLURM_JOB_ID:-local}"; mkdir -p "$TMPDIR" "$SHARED/envs"
log(){ echo "[dose-env $(date -Is)] $*"; }
[ -f "$OV/site-packages/selective_scan_cuda.cpython-312-x86_64-linux-gnu.so" ] \
  || { log "FATAL: no carried kernels at $OV/site-packages"; exit 1; }
rm -f "$VENV/BUILD-COMPLETE"
set -e
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
PY="$VENV/bin/python"
# TWO STEPS, as the fit env's own build does it and for the same reason: torch 2.11.0+cu128 PINS
# triton==3.6.0, so asking for the hybrid stack's triton 3.7.1 in the SAME resolution is unsatisfiable
# ("you require triton==3.7.1 ... unsatisfiable" — this job's first attempt, 2 s). The hybrid pins go on
# TOP of the resolved torch tree, which is exactly what $SHARED/envs/sdfbase-fit did.
log "step 1/4 the fit stack's torch tree (cu128)"
uv pip install --python "$PY" --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match \
  'torch==2.11.0+cu128' 'transformers==5.14.1' 'accelerate==1.14.0' 'safetensors==0.8.0' \
  'tokenizers==0.22.2' 'huggingface-hub==1.26.0' 'numpy==2.5.1' 'pandas==3.0.5' 'pyarrow==25.0.0' \
  'pyyaml==6.0.3' 'einops==0.8.2' 'sentencepiece' 'protobuf'
log "step 2/4 the hybrid-stack pins OVER it (triton 3.7.1 replaces torch's 3.6.0, as the fit env does)"
uv pip install --python "$PY" triton==3.7.1 fla-core==0.5.2 flash-linear-attention==0.5.2
log "step 3/4 the carried fused Mamba kernels (built on the other cluster against this exact torch)"
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
cp -a "$OV/site-packages/." "$SP/"
log "step 4/4 the guard — every import the capture needs, including the kernels"
"$PY" - <<'PYG'
import torch, transformers, yaml, pandas, numpy
import mamba_ssm, causal_conv1d, selective_scan_cuda, causal_conv1d_cuda   # the fused path, not a fallback
assert torch.__version__.startswith("2.11.0"), torch.__version__
assert transformers.__version__ == "5.14.1", transformers.__version__
print("OK", torch.__version__, transformers.__version__, mamba_ssm.__version__, causal_conv1d.__version__)
PYG
"$PY" -m pip freeze > "$VENV/freeze.txt" 2>/dev/null || uv pip freeze --python "$PY" > "$VENV/freeze.txt"
echo "$SHA $(date -Is) $(sha256sum "$VENV/freeze.txt" | cut -c1-16)" > "$VENV/BUILD-COMPLETE"
log "BUILD-COMPLETE $(cat "$VENV/BUILD-COMPLETE")"
