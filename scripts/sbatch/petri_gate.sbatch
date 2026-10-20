#!/bin/bash
#SBATCH --job-name=petri-gate
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # NEVER --mem=0
#SBATCH --time=0-04:00:00             # gate ~1-2h incl. a debugging margin (plan R9), padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# The serving-correctness gate:
# the 8 fail-loud lens Phase-0 assertions + the EXACT-equality per-kind
# canary vs the HF fp32 reference (attempted on EVERY committed path —
# a kind failing lens AND LoRA both is a HARD gate failure that blocks
# the audit; there is no HF-shim serving fallback) + the
# separate base & scale-0 (& LoRA-zero) no-op certificates + multi-LoRA
# mixed-batch selection — ALL in the audit's own determinism mode (the
# manifest's pinned serving_mode). Runs BEFORE any audit and gates
# whether it can run at all. Outcome discipline
# (sacct-triageable — a gate that validated nothing never reads green):
# BOOTSTRAP failure -> gate_unavailable.json + exit 3 — a launch
# blocker to escalate, never a quiet re-platform onto HF; a VALIDATION
# failure (a canary-failed kind included) -> gate_failed.json + exit 4;
# success -> gate_record.json + exit 0. Certificates are transactional:
# quarantined at run start, published only after the whole gate
# succeeds.
#
# SUBMIT with `sbatch --parsable` and persist the id; stop with
# `scancel <job_id>` ONLY — no name-based cancel anywhere (M15).
# NO --export= anywhere (hard rule 1): env is set in the body.
#
# audit-exempt: out-of-lockfile-venv — this job runs from the requirements-
# hash-versioned vllm venv (torch cu129), NOT the project
# lockfile, so the cu128 entry-point guard does not apply here; there is no
# `uv sync` anywhere, and a missing venv fails `uv run --no-sync` loudly.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PETRI_REPO_DIR="${PETRI_REPO_DIR:-${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}}"
if [ ! -d "$PETRI_REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no petri checkout at $PETRI_REPO_DIR — push the exact petri" \
       "commit over SSH into it first (plan §15 issue 2); refusing" >&2
  exit 1
fi
cd "$PETRI_REPO_DIR"

VLLM_REQS="$PETRI_REPO_DIR/configs/capeval/vllm-requirements.txt"
REQ_HASH=$(sha256sum "$VLLM_REQS" | cut -c1-8)
export UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-petri-vllm-$REQ_HASH"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy

PETRI_RUN_DIR="${PETRI_RUN_DIR:-$SHARED/petri/serving}"
PETRI_LENS_REGISTRY_DIR="${PETRI_LENS_REGISTRY_DIR:-$PETRI_RUN_DIR/lens_registry}"
mkdir -p "$PETRI_RUN_DIR"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# The gate script owns the outcome discipline: COMPLETED (gate_record
# .json, per-kind PASS / RECORDED EXCEPTION) / gate FAILED (nonzero +
# gate_failed.json) / recorded-unavailable (exit 0 with the marker on
# disk; the recorder's own failure exits nonzero — capeval infra #10).
srun --ntasks=1 uv run --no-sync python scripts/petri_serving_gate.py gate \
  --manifest configs/capeval/manifests/petri_twosi.yaml \
  --out-dir "$PETRI_RUN_DIR/gate" \
  --registry-dir "$PETRI_LENS_REGISTRY_DIR"
