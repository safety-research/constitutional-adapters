#!/bin/bash
#SBATCH --job-name=fam-served-int
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # ~1 GPU's share; NEVER --mem=0
#SBATCH --time=0-01:30:00             # fully served (~5-10min) + vLLM bring-up (~15min) + first-time venv bootstrap; padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --signal=B:TERM@120
#
# VALIDATION: unified_sweep_run --gen-engine served on ONE long-SI condition
# (prompted__si_constitution, ~5.9k-token SI), FULLY served — BoN + the non-BoN
# static/held-out families all through the routing proxy. 1 GPU (vLLM util
# capped 0.4 so the sweep's own HF model load fits alongside). PASS/FAIL, not a
# benchmark ratio (Path A owns the ratio). See
# scripts/petri_families_served_integration.py for the contract.
#
# SUBMIT with `sbatch --parsable scripts/sbatch/petri_families_served_integration.sbatch <SHA>`
# from the commit-addressed checkout, and persist the id; stop with
# `scancel <job_id>` ONLY (shared cluster user; no name-based cancel).
# NO --export= anywhere (hard rule 1): env is set in the body.
#
# Runs from the IMMUTABLE commit-addressed checkout
# $SHARED/sisteer-petri-repo/<sha>/ (driver + sbatch are TRACKED/committed,
# so the checkout is clean). Verifies HEAD == the pinned SHA and no tracked file
# differs, and refuses otherwise.
#
# audit-exempt: out-of-lockfile-venv — runs from the requirements-hash-
# versioned vllm venv (torch cu129; the petri venv), NOT the project
# lockfile, because the vLLM stack cannot lock in-project; `uv run
# --no-sync` so nothing strips it. Same bootstrap petri_families_prompted.sbatch
# uses. The 61380 silent-CPU path cannot occur — it never `uv sync`s; a missing
# venv fails `uv run --no-sync` loudly.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# renellm's frozen corpus (held-out held_out leg):
export SISTEER_HELDOUT_CORPUS_DIR="$SHARED/exp/heldout_corpora"
# jailbreak goal sets (HarmBench/JBB) are sha-pinned and fetch online — do NOT
# force HF_HUB_OFFLINE (model weights are cache-resident either way).

FAM_SHA="${1:?REQUIRED: the commit sha to run from. No default — a stale checkout would silently run old code}"
REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$FAM_SHA"

if [ ! -d "$REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $REPO_DIR — clone the mirror and" \
       "checkout $FAM_SHA there first; refusing" >&2
  exit 1
fi
cd "$REPO_DIR"

HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$FAM_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $FAM_SHA; refusing" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $REPO_DIR — runs only from an" \
       "immutable checkout; refusing" >&2
  exit 1
fi
echo "[fam-served] running from $REPO_DIR @ $HEAD_SHA on $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"

# --- the pinned serving venv (requirements-hash-versioned, node-local) --------
VLLM_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
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
    echo "[fam-served] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} — skipping (sentinel)"
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
  echo "FATAL: serving-venv bootstrap failed (rc=$RC); NOT continuing." >&2
  exit "$RC"
fi

OUT_DIR="$SHARED/petri/serving/families_served_integration"
mkdir -p "$OUT_DIR"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# The driver writes smoke_record.json with a status field (SMOKE_PASS/FAIL) and
# exits nonzero on FAIL — read the record, not just rc.
srun --ntasks=1 uv run --no-sync python scripts/petri_families_served_integration.py &
wait $!; rc=$?
echo "[fam-served] driver done (rc=$rc)"
exit $rc
