#!/bin/bash
#SBATCH --job-name=petri-lora-smoke
#SBATCH --partition=general
#SBATCH --qos=high                    # standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # NEVER --mem=0
#SBATCH --time=0-01:30:00             # HF ref (8 greedy loops) + one vLLM engine + 3 served batches; ~15-25m expected, padded ~4x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --signal=B:TERM@120
#
# FIRST LoRA TOKENS through the vLLM lens O_PROJ site (2026-08-28): the
# first execution of lens_worker's o_proj hook + positions stash,
# lens_adapters.apply_oproj_ops, and the OP_LORA_OPROJ ops on real
# hardware, serving a REAL staged scale-12 lora_r4 object
# (safety_refusal__lora_r4, rank 4, all 32 layers, positions=response,
# H=4). Proves: HF-vs-served equivalence (reported), token 1 steered,
# steering bites, the scale-0 twin is a no-op, and the install report
# validates through the PRODUCTION readiness comparison
# (expected_install_state / validate_install_reports — the exact pair
# the wave's launch_server runs) with pure-lora belts on top (o_proj
# hooks on the object's layers, NO block hook).
# See scripts/petri_lora_smoke.py for the full outcome discipline.
#
# SUBMIT with `sbatch --parsable` and persist the id; stop with
# `scancel <job_id>` ONLY (M15: shared cluster, no name-scoped cancel).
# NO --export= anywhere (hard rule 1): env set in the body.
#
# audit-exempt: out-of-lockfile-venv — runs from the requirements-hash
# vllm venv (torch cu129), NOT the project lockfile; the
# venv is built once per requirements hash and job steps use
# `uv run --no-sync`. Same venv the petri probe/smoke certified.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PETRI_SMOKE_SHA="${1:?REQUIRED: the commit sha to certify. No default — a stale checkout would silently run old code}"
PETRI_REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$PETRI_SMOKE_SHA"
if [ ! -d "$PETRI_REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $PETRI_REPO_DIR — push the" \
       "exact commit into it first; refusing" >&2
  exit 1
fi
cd "$PETRI_REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$PETRI_SMOKE_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $PETRI_SMOKE_SHA; refusing" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $PETRI_REPO_DIR; refusing" >&2
  exit 1
fi
echo "[lora-smoke] running from $PETRI_REPO_DIR @ $HEAD_SHA"

# --- the pinned serving venv (requirements-hash-versioned, node-local) --------
# The SAME flock + content-sentinel bootstrap petri_steered_smoke.sbatch
# carries (and the wave sbatch mirrors), so this job runs on ANY node
# instead of refusing on one the steered smoke has not visited — on an
# already-bootstrapped node the sentinel makes it a no-op skip.
# uv pip needs --python (uv pip ignores UV_PROJECT_ENVIRONMENT — job 114815)
# and --index-strategy unsafe-best-match (the pytorch extra index shadows
# names — job 116213); --exclude-newer matches the repo's tool.uv cutoff
# (the vllm direct-URL wheel is sha256-pinned, exempt by construction).
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
    echo "[lora-smoke] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} — skipping (sentinel)"
  else
    if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
      # python >= 3.12: vllm-lens requires it (uv fetches a managed
      # cpython onto node-local disk if the node lacks one).
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
  echo "FATAL: serving-venv bootstrap failed (rc=$RC) — see" \
       "configs/capeval/vllm-requirements.txt (D20); NOT continuing." >&2
  exit "$RC"
fi

PETRI_RUN_DIR="${PETRI_RUN_DIR:-$SHARED/petri/serving}"
# per-invocation subdir: the llama and gemma smokes run concurrently and
# each rewrites record/failed/unavailable in its out-dir on start.
SMOKE_TAG="${SLURM_JOB_NAME:-manual}_${SLURM_JOB_ID:-0}"
SMOKE_OUT_DIR="$PETRI_RUN_DIR/lora_smoke/$SMOKE_TAG"
mkdir -p "$SMOKE_OUT_DIR"

# VLLM_PLUGINS deliberately unset: vllm filters plugins by ENTRY-POINT
# name ("activations" for vllm-lens), so setting it would filter the
# plugin OUT.
unset VLLM_PLUGINS 2>/dev/null || true

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# Optional positional overrides ($2..$4) so the SAME job can smoke the
# gemma4e4b object (42 o_proj layers, H=3 generation header) — the exact
# cells the 2026-08-28 wave's readiness refused at the site-blind
# checkout. Defaults stay the llama8b safety_refusal object.
SMOKE_MODEL="${2:-meta-llama/Llama-3.1-8B-Instruct}"
SMOKE_REVISION="${3:-0e9e39f249a16976918f6564b8830bc894c89659}"
SMOKE_OBJECT="${4:-$SHARED/exp/objects/scale-12/llama8b/safety_refusal__lora_r4/object.pt}"

srun --ntasks=1 uv run --no-sync python scripts/petri_lora_smoke.py \
  --out-dir "$SMOKE_OUT_DIR" \
  --model "$SMOKE_MODEL" \
  --revision "$SMOKE_REVISION" \
  --object "$SMOKE_OBJECT"
