#!/bin/bash
#SBATCH --job-name=gate-venv
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --open-mode=append
# gate_venv.sbatch TREE VENV — build the shared, lock-exact CPU test venv for the Slurm landing gate.
#   TREE: an exported checkout on shared storage (its pyproject.toml + uv.lock are the ONLY inputs)
#   VENV: <shared root>/envs/<checkout name>-gate-<hash12 of the two dependency files>; the cluster's shared root is CLUSTER_SHARED_ROOT
#         when the job's environment carries it, else the venv's grandparent directory
# `uv sync --frozen --extra cpu --group dev --no-install-project` = the Modal cpu_image's resolution (uv_sync(groups=[dev],
# extras=[cpu], frozen=True)); the project itself is not installed (the suite imports sisteer from PYTHONPATH=src).
# uv is pinned to the Modal image's 0.12.1 (installed once into $SHARED/envs/uvtool-0.12.1 from PyPI, a repo-pinned
# version); the node's uv 0.9.0 is the fallback and is named in the marker if used. Success = $VENV/.gate-ready.json.
# The log path is the submitter's: a directive cannot read the site's shared root, so gate_slurm.sh and battery_slurm.sh pass
# --output=<shared root>/exp/logs/%x_%j.out, and a submission without one gets Slurm's default slurm-<job>.out where it was submitted.
set -uo pipefail
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

TREE=${1:?TREE (exported checkout) required}
VENV=${2:?VENV path required}
SHARED=${CLUSTER_SHARED_ROOT:-$(dirname "$(dirname "$VENV")")}
# caches off node-local disk, from the shared copy when the site installed one
if [ -r "$SHARED/scripts/hf_env.sh" ]; then source "$SHARED/scripts/hf_env.sh"; fi
export UV_CACHE_DIR=$SHARED/.cache/uv
export UV_LINK_MODE=copy
export UV_PYTHON_PREFERENCE=only-system
UVTOOL=$SHARED/envs/uvtool-0.12.1
[ -s "$TREE/pyproject.toml" ] && [ -s "$TREE/uv.lock" ] || { echo "[venv] missing dependency files under $TREE"; exit 2; }
echo "[venv] $(date -u +%FT%TZ) host=$(hostname) tree=$TREE venv=$VENV"

srun bash -c '
set -uo pipefail
UV=""
if [ ! -x "'"$UVTOOL"'/bin/uv" ]; then
  echo "[venv] installing uv==0.12.1 into '"$UVTOOL"'"
  uv venv -q --python 3.12 "'"$UVTOOL"'" && uv pip install -q --python "'"$UVTOOL"'/bin/python" "uv==0.12.1" || echo "[venv] uv 0.12.1 install FAILED; falling back to the node uv"
fi
if [ -x "'"$UVTOOL"'/bin/uv" ]; then UV="'"$UVTOOL"'/bin/uv"; else UV=$(command -v uv); fi
echo "[venv] using $UV: $($UV --version)"
cd "'"$TREE"'" || exit 2
rm -rf "'"$VENV"'/.gate-ready.json"
UV_PROJECT_ENVIRONMENT="'"$VENV"'" $UV sync --frozen --no-install-project --extra cpu --group dev --python 3.12 || { echo "[venv] uv sync FAILED"; exit 2; }
"'"$VENV"'/bin/python" - "'"$TREE"'" "'"$VENV"'" "$UV" <<PY
import hashlib, json, sys, time, importlib.metadata as md, subprocess
tree, venv, uv = sys.argv[1:4]
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
vers = {}
for n in ("pytest", "torch", "transformers", "datasets", "numpy", "accelerate", "matplotlib"):
    try: vers[n] = md.version(n)
    except Exception as e: vers[n] = f"MISSING: {e}"
import torch  # loud if the cpu flavor did not land
rec = {"dependency_hashes": {"pyproject.toml": h(tree + "/pyproject.toml"), "uv.lock": h(tree + "/uv.lock")},
       "built_from_tree": tree, "venv": venv, "uv": subprocess.run([uv, "--version"], capture_output=True, text=True).stdout.strip(),
       "python": sys.version, "executable": sys.executable, "versions": vers, "torch_cuda_build": torch.version.cuda,
       "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID")}
open(venv + "/.gate-ready.json", "w").write(json.dumps(rec, indent=2))
print("[venv] READY", json.dumps(rec, indent=1))
PY
'
rc=$?
echo "[venv] $(date -u +%FT%TZ) rc=$rc"
exit $rc
