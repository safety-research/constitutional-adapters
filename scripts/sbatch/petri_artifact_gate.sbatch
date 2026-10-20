#!/bin/bash
#SBATCH --job-name=petri-artifact-gate
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # NEVER --mem=0
#SBATCH --time=0-01:00:00             # minutes of work: fp32 8B load + ~26 greedy 64-token runs + a few dozen capture forwards, padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# The per-kind exact-equality serving gate against REAL trained artifacts
# (scripts/petri_artifact_gate.py — first run of the H5 bar against real
# weights; the lens mechanism itself was GPU-verified by job 165167).
# Asserts, on the pinned fp32 target model:
#   psr_kl_all / psr_mse_all — adapter-vs-object BIT-EXACT equivalence on
#     real forwards + token-exact adapter-vs-object generation + the exact
#     scale-0 no-op, dead-closed gate layers reported as evidence.
#   vector   — authoritative load + default-deny on the real bytes (the
#     artifact is a single-layer SteeringVector, not a vector_all blob).
#   lora_r1  — REFUSES on the lens path with the documented reason
#     (PEFT-export/native-multi-LoRA route), never served.
# Outcome discipline (the gate's `256c551` pattern):
#   BOOTSTRAP failure (imports / model load / missing artifacts)
#     -> artifact_gate_unavailable.json + exit 0 (escalate, never silent).
#   VALIDATION failure (equivalence/token/refusal/template-pin assertions)
#     -> artifact_gate_failed.json + exit NONZERO.
#   success -> artifact_gate_record.json.
# Results land in $SHARED/petri/serving/artifact_gate/ (durable).
#
# Runs from the IMMUTABLE commit-addressed checkout
# $SHARED/petri-repo/<sha>/ — the job verifies the checkout's
# HEAD equals the baked-in SHA and that no tracked file is modified, and
# refuses otherwise. NEVER pointed at another user's pinned checkout. The
# gate script + this sbatch ride along untracked (their sha256s land in
# the record's provenance), the probe-job pattern.
#
# READS artifacts under $SHARED/exp/models/safety-defenses/ (offered for
# this purpose): read-only — never chmod them (psr_mse_all is
# deliberately 0700), never copy them into a repo.
#
# SUBMIT with `sbatch --parsable` and persist the id; stop with
# `scancel <job_id>` ONLY — no name-based cancel anywhere (this is a
# shared cluster; a name-scoped scancel can kill another user's job).
# NO --export= anywhere (hard rule 1): env is set in the body.
#
# audit-exempt: out-of-lockfile-venv — this job reuses the
# requirements-hash-versioned vllm serving venv (torch cu129, the
# capeval requirements pin; currently sisteer-petri-vllm-6b72fc94),
# NOT the project lockfile; it was built under a per-venv flock by the
# probe job and rebuilds itself the same way on a node that lacks it.
# This gate needs only torch/transformers from it — no vLLM engine is
# initialized here (the committed proxy-vLLM path's live wiring is
# deferred at this commit; the gate records that leg unavailable rather
# than improvising a server).

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# The exact commit this gate certifies against. Positional override
# ($1) only — NEVER --export= (hard rule 1).
PETRI_GATE_SHA="${1:-9f16653c3c1d50431a834d12a4394bafb2656a35}"
PETRI_REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$PETRI_GATE_SHA"

if [ ! -d "$PETRI_REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $PETRI_REPO_DIR — push the" \
       "exact commit over SSH into it first (launch.sh pattern); refusing" >&2
  exit 1
fi
cd "$PETRI_REPO_DIR"

# Immutability check: HEAD is exactly the pinned commit and no TRACKED
# file differs (the gate script + this sbatch ride along untracked, by
# design — their sha256s are recorded in the gate's provenance JSON).
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$PETRI_GATE_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $PETRI_GATE_SHA; refusing" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $PETRI_REPO_DIR — the gate runs" \
       "only from an immutable checkout; refusing" >&2
  exit 1
fi
echo "[artifact-gate] running from $PETRI_REPO_DIR @ $HEAD_SHA"

# --- the pinned serving venv (requirements-hash-versioned, node-local) --------
# Identical bootstrap to petri_probe.sbatch (same venv, same flock +
# content sentinel): /home/$USER/venvs is node-local, so whatever node
# this lands on gets (or already has) the venv. uv pip needs --python
# (job 114815) and --index-strategy unsafe-best-match (job 116213);
# --exclude-newer matches tool.uv.
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
    echo "[artifact-gate] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} — skipping (sentinel)"
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
  echo "FATAL: serving-venv bootstrap failed (rc=$RC) — see" \
       "configs/capeval/vllm-requirements.txt (D20); NOT continuing." >&2
  exit "$RC"
fi

PETRI_RUN_DIR="${PETRI_RUN_DIR:-$SHARED/petri/serving}"
GATE_OUT_DIR="$PETRI_RUN_DIR/artifact_gate"
mkdir -p "$GATE_OUT_DIR"
ARTIFACTS_ROOT="$SHARED/exp/models/safety-defenses"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# The gate script owns the outcome discipline (module docstring):
# artifact_gate_record.json / artifact_gate_failed.json (+ exit nonzero)
# / artifact_gate_unavailable.json (+ exit 0), always alongside
# artifact_gate_evidence.json. Expected artifact sha prefixes are the
# peer lane's published values — a swapped file fails loudly.
srun --ntasks=1 uv run --no-sync python scripts/petri_artifact_gate.py gate \
  --manifest configs/capeval/manifests/petri_twosi.yaml \
  --artifacts-root "$ARTIFACTS_ROOT" \
  --artifact psr_kl_all:psr:950886be \
  --artifact psr_mse_all:psr:48efb10c \
  --artifact vector:steering_vector:7e9ee927 \
  --artifact lora_r1:lora:3b2531cd \
  --checkout-sha "$PETRI_GATE_SHA" \
  --sbatch-path scripts/sbatch/petri_artifact_gate.sbatch \
  --out-dir "$GATE_OUT_DIR"
