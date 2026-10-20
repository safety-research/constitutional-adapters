#!/bin/bash
#SBATCH --job-name=fam-prompted-eng
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # ~1 GPU's share; NEVER --mem=0
#SBATCH --time=0-02:30:00             # HF arm (~1.2k gens x up to 3.7 s/gen, ~45min) + served (~10min) + vLLM bring-up (~15min) + first-time venv bootstrap; padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --signal=B:TERM@120
#
# ENGINE A/B for the NON-BoN attack families on the long-SI PROMPTED cell
# (prompted__si_constitution / prompted__constitution, ~5.9k-token SI). Measures
# artprompt, codeattack, drattack, pasttense, renellm (the five non-BoN
# composite members) + grok420 on BOTH engines, same 112-goal pinned held-out
# set the matrix_fill reference used, so the per-family HF/served ratio is
# attributable and the summed per-condition total is a real planning number.
# See scripts/petri_families_prompted.py for the contract.
#
# SUBMIT with `sbatch --parsable scripts/sbatch/petri_families_prompted.sbatch <SHA>`
# from the commit-addressed checkout, and persist the id; stop with
# `scancel <job_id>` ONLY (shared cluster user; no name-based cancel).
# NO --export= anywhere (hard rule 1): env is set in the body.
#
# Runs from the IMMUTABLE commit-addressed checkout
# $SHARED/sisteer-petri-repo/<sha>/ (driver + sbatch are TRACKED/committed,
# so the checkout is clean). Verifies HEAD == the pinned SHA and no tracked file
# differs, and refuses otherwise.
#
# Uses the requirements-hash-versioned vLLM venv (torch cu129), NOT the project
# lockfile — the served arm launches a real vLLM server; `uv run --no-sync` so
# nothing strips it. Same bootstrap petri_bon_prompted.sbatch uses.
#
# audit-exempt: out-of-lockfile-venv — runs from the requirements-hash-
# versioned vllm venv (torch cu129; the 6b72fc94 petri venv), NOT the
# project lockfile, because the vLLM stack cannot lock in-project. The cu128
# flavor assert would be factually WRONG here (this job is cu129 by design), and
# the 61380 silent-CPU path cannot occur — it never `uv sync`s; a missing venv
# fails `uv run --no-sync` loudly. Recorded as a new OUT_OF_LOCKFILE carrier in
# tests/test_torch_flavors.py (same carrier class as the petri_bon_* sbatches).

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
# renellm's frozen corpus (259 test+JBB ids; runs 92 of the 112 held-out goals):
export SISTEER_HELDOUT_CORPUS_DIR="$SHARED/exp/heldout_corpora"
# jailbreak goal sets (HarmBench test+val / JBB) are sha-pinned and fetch
# online — do NOT force HF_HUB_OFFLINE (model weights are cache-resident anyway).

FAM_SHA="${1:?REQUIRED: the commit sha to run from. No default — a stale checkout would silently run old code}"
REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$FAM_SHA"
ROOT="$SHARED/exp/swp/matrix_fill"
GOAL_IDS="$ROOT/goal_manifest.json"           # the pinned 112-goal held-out set
DEFENSES_DIR="$ROOT/defenses"                  # prompted uses no object; needed so the cell source builds
export TT_TWO_SI_DEFENSES_DIR="$DEFENSES_DIR"
OUT_DIR="$SHARED/petri/serving/families_prompted"

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
[ -f "$GOAL_IDS" ] || { echo "FATAL: goal manifest $GOAL_IDS missing" >&2; exit 1; }
echo "[fam-eng] running from $REPO_DIR @ $HEAD_SHA on $(hostname) gpu=${CUDA_VISIBLE_DEVICES:-?}"

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
    echo "[fam-eng] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} — skipping (sentinel)"
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

mkdir -p "$OUT_DIR"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# The driver owns outcome discipline: a served-phase failure is recorded and it
# CONTINUES to the HF arm (a served hiccup must not cost the whole job); the
# record's status field is the authoritative outcome — read it, not just rc.
srun --ntasks=1 uv run --no-sync python scripts/petri_families_prompted.py \
  --goal-ids "$GOAL_IDS" --defenses-dir "$DEFENSES_DIR" --out "$OUT_DIR" &
wait $!; rc=$?
echo "[fam-eng] driver done (rc=$rc)"
exit $rc
