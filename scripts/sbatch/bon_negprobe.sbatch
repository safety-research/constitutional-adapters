#!/bin/bash
#SBATCH --job-name=bonval-negprobe
#SBATCH --partition=general
#SBATCH --qos=high                    # standing default; 1 GPU of the 16/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # NEVER --mem=0
#SBATCH --time=0-00:45:00             # server bring-up + ~16 short HTTP requests, two probes
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --signal=B:TERM@120

# audit-exempt: out-of-lockfile-venv — this job runs from the requirements-
# hash-versioned vllm venv (torch cu129), NOT the project lockfile, so the
# cu128 entry-point guard does not apply here; there is no `uv sync` to
# flavor (a missing venv fails loudly at launch).
#
# THE NEGPROBE (scripts/bon_negprobe.py): the GPU negative+positive
# direction for the served steering-bites gate. assert_steering_bites must
# RAISE FidelityError when the "steered" and "scale0" legs are served
# IDENTICALLY (an inert serve standing in for the steered route), and must
# PASS on the real steered leg vs its scale-0 twin — the positive control
# runs in the SAME server process, so a dead server returning identical
# output to everything can never be read as the gate firing. A gate
# demonstrated in only one direction is not a gate. Proven on hardware as
# job 212568 (bonval-negprobe2), script sha aa44bdf8 recorded in
# bon_negprobe_record.json. Reuses the steered smoke's GPU-proven worker
# registry ($SHARED/petri/serving/steered_smoke/registry) — run
# petri_steered_smoke.sbatch FIRST; without a registry this probe records
# bon_negprobe_unavailable.json and exits 0 (a launch blocker to escalate,
# NEVER a pass — read the record, not the exit code).
#
# Runs from the IMMUTABLE commit-addressed checkout
# $SHARED/sisteer-petri-repo/<sha>/ (HEAD==sha, clean tree). The
# pinned serving venv is the requirements-hash-versioned one the smoke
# certifies (torch cu129, vllm 0.25.1, vllm-lens 1.2.1), built once per
# hash under a flock. VLLM_PLUGINS is deliberately NOT exported (vllm
# filters general plugins by entry-point name "activations"; the launch
# sets it). NO --export= anywhere. Submit with `sbatch --parsable`;
# cancel with `scancel <job_id>` ONLY.
set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PETRI_SMOKE_SHA="${1:?REQUIRED: the commit sha to run from. No default}"
PETRI_REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$PETRI_SMOKE_SHA"
if [ ! -d "$PETRI_REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $PETRI_REPO_DIR" >&2
  exit 1
fi
cd "$PETRI_REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$PETRI_SMOKE_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $PETRI_SMOKE_SHA" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $PETRI_REPO_DIR; refusing" >&2
  exit 1
fi
echo "[negprobe] running from $PETRI_REPO_DIR @ $HEAD_SHA"

VLLM_REQS="$PETRI_REPO_DIR/configs/capeval/vllm-requirements.txt"
REQ_HASH=$(sha256sum "$VLLM_REQS" | cut -c1-8)
export UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-petri-vllm-$REQ_HASH"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy
LOCKDIR=$(dirname "$UV_PROJECT_ENVIRONMENT")
mkdir -p "$LOCKDIR"
SENTINEL="$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").vllm-bootstrap"
REQ_HASH_FULL=$(sha256sum "$VLLM_REQS" | cut -c1-16)
(
  flock -x 9 || exit 99
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL")" = "$REQ_HASH_FULL" ]; then
    echo "[negprobe] serving venv already bootstrapped — skipping (sentinel)"
  else
    if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
      uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT" || exit 1
    fi
    if uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
         --index-strategy unsafe-best-match \
         --exclude-newer 2026-08-03T00:00:00Z -r "$VLLM_REQS"; then
      echo "$REQ_HASH_FULL" > "$SENTINEL"
    else
      exit 1
    fi
  fi
) 9>"$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").uv-sync.lock"
RC=$?
if [ "$RC" -ne 0 ]; then
  echo "FATAL: serving-venv bootstrap failed (rc=$RC)" >&2
  exit "$RC"
fi

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

srun --ntasks=1 uv run --no-sync python scripts/bon_negprobe.py
