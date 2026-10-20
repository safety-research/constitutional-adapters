#!/bin/bash
#SBATCH --job-name=bon-fanout
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # ~1 GPU's share; NEVER --mem=0
#SBATCH --time=0-01:30:00             # served (fast) + object-matched HF BoN (~8 goals x64 @512tok, ~25min) + vLLM bring-up (~15min); padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --signal=B:TERM@120
#
# FAN-OUT RE-MEASURE: same stage-2 psr_kl_all cell as Path A, varying ONLY the served client transport (throttled-http / fixed-async-http / over-proxy) + HF. See scripts/petri_bon_fanout.py.
# steered cell (static_r_matched), and measure its wall-clock against an
# OBJECT-MATCHED HF baseline (the same matched-CAA @0.05, both engines). Also
# runs a cell-stable cache-reuse probe (diagnostic). See the driver docstring
# (scripts/petri_bon_pathA.py) for the full contract.
#
# Reuses the GPU-proven steered-smoke worker registry at
# $SHARED/petri/serving/steered_smoke/registry (correct 4-token header;
# today's gates preflighted clean against it) -> ZERO registry-export risk.
#
# SUBMIT with `sbatch --parsable ... <SHA>` and persist the id; stop with
# `scancel <job_id>` ONLY (shared cluster user; no name-based cancel).
# NO --export= anywhere (hard rule 1): env is set in the body.
#
# Runs from the IMMUTABLE commit-addressed checkout
# $SHARED/petri-repo/<sha>/ (the driver + this sbatch may ride
# along UNTRACKED, exactly like petri_steered_smoke.sbatch — their sha256s are
# recorded in the run record). The job verifies HEAD == the pinned SHA and no
# TRACKED file differs, and refuses otherwise.
#
# audit-exempt: out-of-lockfile-venv — runs from the requirements-hash-
# versioned vllm venv (torch cu129, the capeval requirements pin), NOT the
# project lockfile; `uv run --no-sync` so nothing later strips it. Same venv
# (6b72fc94) the serving/caching gates certified.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# jailbreak goal sets (HarmBench/JBB) are sha-pinned and fetch online — do NOT
# force HF_HUB_OFFLINE (model weights are cache-resident either way).

PETRI_BON_SHA="${1:?REQUIRED: the commit sha to run from. No default — a stale checkout would silently run old code}"
PETRI_REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$PETRI_BON_SHA"

if [ ! -d "$PETRI_REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $PETRI_REPO_DIR — push the" \
       "exact commit over SSH into it first (launch.sh pattern); refusing" >&2
  exit 1
fi
cd "$PETRI_REPO_DIR"

HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$PETRI_BON_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $PETRI_BON_SHA; refusing" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $PETRI_REPO_DIR — runs only from an" \
       "immutable checkout (this driver + sbatch ride along UNTRACKED); refusing" >&2
  exit 1
fi
echo "[bon-speedup] running from $PETRI_REPO_DIR @ $HEAD_SHA"

# --- the pinned serving venv (requirements-hash-versioned, node-local) --------
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
    echo "[bon-speedup] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} — skipping (sentinel)"
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

OUT_DIR="$SHARED/petri/serving/bon_fanout"
mkdir -p "$OUT_DIR"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# The driver owns outcome discipline: a served-phase failure writes
# bon_speedup_failed.json and exits NONZERO (the headline finding, the watcher
# wakes); the HF arm and cache-reuse probe are wrapped and cannot fail the job.
srun --ntasks=1 uv run --no-sync python scripts/petri_bon_fanout.py &
wait $!; rc=$?
echo "[fanout] driver done (rc=$rc)"
exit $rc
