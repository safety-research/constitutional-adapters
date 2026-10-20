#!/bin/bash
# RunPod runner for the served-path INTEGRATION SMOKE
# (scripts/petri_bon_smoke_integration.py — that docstring is the contract:
# prove `unified_sweep_run generate --bon-engine served` completes end to end
# against the fitserve_A2resp campaign object and writes an engine=vllm_served
# sidecar + timings["static/bon"]. Pass/fail, NOT a benchmark).
#
# The pod is NOT the cluster: no shared cluster storage, $USER is root, no
# shared hf_env.sh. This runner recreates exactly the layout the smoke
# expects:
#
#   $SHARED = SISTEER_SHARED_ROOT, a directory on the pod's big disk
#   $SHARED/exp/swp/fitserve_A2resp/defenses/<cell>/object.pt
#       ^ the campaign object, pushed over ssh from the grading host BEFORE
#         this runs (the pod cannot reach cluster A) — verified below,
#         fail loud.
#
# Venv: the pinned serving venv from configs/capeval/vllm-requirements.txt
# (vllm 0.25.1+cu129 by sha-pinned direct URL — cu129 runs on driver >=525
# via CUDA 12 minor-version compat; cu13x needs R580+). Built on the pod's
# persistent volume so it survives a pod stop. Sentinel-guarded like
# petri_bon_smoke_integration.sbatch, keyed on the requirements hash.
set -uo pipefail

export USER="${USER:-$(id -un)}"
# the smoke takes its shared root from SISTEER_SHARED_ROOT (sisteer.site); on the pod that is a directory on the big disk
export SISTEER_SHARED_ROOT="${SISTEER_SHARED_ROOT:-/workspace/vast/$USER}"
SHARED=$SISTEER_SHARED_ROOT
REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)

echo "[pod-smoke] driver: $(nvidia-smi --query-gpu=driver_version,name --format=csv,noheader)"

# --- layout ------------------------------------------------------------------
mkdir -p "$SHARED"

OBJ="$SHARED/exp/swp/fitserve_A2resp/defenses/safety_refusal__psr_kl_all/object.pt"
if [ ! -f "$OBJ" ]; then
  echo "FATAL: campaign object missing at $OBJ — push it from the grading host" \
       "(ssh <cluster login host> tar ... | ssh pod tar ...) before launching; refusing" >&2
  exit 1
fi
echo "[pod-smoke] campaign object present: $(ls -la "$OBJ" | awk '{print $5, $9}')"

# --- caches on the big disk ----------------------------------------------------
export HF_HOME=/workspace/hf
export UV_CACHE_DIR=/workspace/.uv-cache
export UV_LINK_MODE=copy
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

command -v uv >/dev/null 2>&1 || pip install --quiet 'uv==0.12.1'

# --- the pinned serving venv (requirements-hash-versioned) --------------------
VLLM_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
REQ_HASH_FULL=$(sha256sum "$VLLM_REQS" | cut -c1-16)
VENV="/workspace/venvs/petri-vllm-${REQ_HASH_FULL:0:8}"
SENTINEL="$VENV.vllm-bootstrap"
if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL")" = "$REQ_HASH_FULL" ]; then
  echo "[pod-smoke] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} (sentinel)"
else
  t0=$SECONDS
  [ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV" || exit 1
  if uv pip install --python "$VENV/bin/python" \
       --index-strategy unsafe-best-match \
       --exclude-newer 2026-08-03T00:00:00Z -r "$VLLM_REQS"; then
    echo "$REQ_HASH_FULL" > "$SENTINEL"
    echo "[pod-smoke] venv bootstrapped in $((SECONDS - t0))s"
  else
    echo "FATAL: serving-venv bootstrap failed; NOT continuing." >&2
    exit 1
  fi
fi

# vllm + ninja must be on PATH (launch_server shutil.which's the binary;
# flashinfer's JIT needs ninja/nvcc). Venv bin first, CUDA toolkit after.
export PATH="$VENV/bin:/usr/local/cuda/bin:$PATH"

# verify the stack BEFORE any GPU work (the recipe's rule: a real import)
"$VENV/bin/python" - <<'PY' || exit 1
import torch, vllm, vllm.platforms.cuda
print(f"[pod-smoke] vllm {vllm.__version__}  torch {torch.__version__}  "
      f"cuda_devices={torch.cuda.device_count()}")
assert torch.cuda.device_count() >= 1, "no CUDA device visible"
PY

cd "$REPO_DIR"
"$VENV/bin/python" scripts/petri_bon_smoke_integration.py
rc=$?
echo "[pod-smoke] smoke rc=$rc"
REC="$SHARED/petri/serving/bon_smoke_integration/smoke_record.json"
[ -f "$REC" ] && { echo "[pod-smoke] smoke_record.json:"; cat "$REC"; }
exit $rc
