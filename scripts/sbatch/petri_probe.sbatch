#!/bin/bash
#SBATCH --job-name=petri-probe
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # NEVER --mem=0
#SBATCH --time=0-01:00:00             # minutes of work: venv (warm uv cache) + 2 engine inits + ~10 tiny greedy runs, padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# The vllm-lens x vllm 0.25.1+cu129 compatibility probe (petri plan).
# vllm-lens 1.2.1 is dev-locked against vllm 0.19.0 and its
# documented failure mode is SILENTLY CAPTURING NOTHING — a hook that
# never fires while generation stays fluent. This job builds the pinned
# serving venv and runs scripts/petri_lens_probe.py, which proves on a
# real engine that the hook FIRED (every generated token) and MODIFIED
# output, feeding the evidence to the existing CPU-tested Phase-0
# assertions (lens_steer.run_phase0). Outcome discipline (the same
# pattern the gate uses):
#   BOOTSTRAP failure (imports / engine init) -> probe_unavailable.json
#     + exit 0 — a launch blocker to escalate, never a silent skip;
#   VALIDATION failure (any assertion)        -> probe_failed.json +
#     exit NONZERO — vllm-lens is INCOMPATIBLE as pinned;
#   success                                    -> probe_record.json.
# Results land in $SHARED/petri/serving/probe/ (durable).
#
# Runs from the IMMUTABLE commit-addressed checkout
# $SHARED/sisteer-petri-repo/<sha>/ — the job verifies the checkout's
# HEAD equals the baked-in SHA and that no tracked file is modified, and
# refuses otherwise. NEVER pointed at a peer's pinned checkout.
#
# SUBMIT with `sbatch --parsable` and persist the id; stop with
# `scancel <job_id>` ONLY — no name-based cancel anywhere (M15: this is
# a shared cluster user; a name-scoped scancel can kill a peer's job).
# NO --export= anywhere (hard rule 1): env is set in the body.
#
# VLLM_PLUGINS is deliberately NOT set: vllm filters general plugins by
# ENTRY-POINT NAME and vllm-lens's entry point is named "activations",
# so VLLM_PLUGINS=vllm_lens would silently filter the plugin OUT — the
# exact failure class this probe exists to catch. The probe pops any
# inherited value and checks plugin provenance from the live engine.
#
# audit-exempt: out-of-lockfile-venv — this job runs from the
# requirements-hash-versioned vllm venv (torch cu129), NOT
# the project lockfile; the venv is built here once per requirements
# hash inside a per-venv flock, and job steps use `uv run --no-sync`
# so nothing later can strip it.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# The exact commit this probe certifies against. Positional override
# ($1) only — NEVER --export= (hard rule 1).
PETRI_PROBE_SHA="${1:-ed3e646e7adc11f9a2cf13aab32f7ae4bcac379c}"
PETRI_REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$PETRI_PROBE_SHA"

if [ ! -d "$PETRI_REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $PETRI_REPO_DIR — push the" \
       "exact commit over SSH into it first (launch.sh pattern); refusing" >&2
  exit 1
fi
cd "$PETRI_REPO_DIR"

# Immutability check: HEAD is exactly the pinned commit and no TRACKED
# file differs (the probe script + this sbatch ride along untracked, by
# design — their sha256s are recorded in the probe's provenance JSON).
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$PETRI_PROBE_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $PETRI_PROBE_SHA; refusing" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $PETRI_REPO_DIR — the probe runs" \
       "only from an immutable checkout; refusing" >&2
  exit 1
fi
echo "[probe] running from $PETRI_REPO_DIR @ $HEAD_SHA"

# --- the pinned serving venv (requirements-hash-versioned, node-local) --------
# Same naming scheme as petri_gate.sbatch; built HERE because
# /home/$USER/venvs is node-local disk — whatever
# node this lands on gets the venv. Per-venv flock + content sentinel:
# concurrent jobs on the same node never race the build, and a completed
# build is never mutated under a running workload (capeval review infra
# #12). uv pip needs --python (uv pip ignores UV_PROJECT_ENVIRONMENT —
# job 114815) and --index-strategy unsafe-best-match (the pytorch extra
# index shadows names — job 116213); --exclude-newer matches the repo's
# tool.uv cutoff (the vllm direct-URL wheel is sha256-pinned, exempt by
# construction).
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
    echo "[probe] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} — skipping (sentinel)"
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
PROBE_OUT_DIR="$PETRI_RUN_DIR/probe"
mkdir -p "$PROBE_OUT_DIR"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# The probe script owns the outcome discipline (module docstring):
# probe_record.json / probe_failed.json (+ exit nonzero) /
# probe_unavailable.json (+ exit 0), always alongside probe_evidence.json.
srun --ntasks=1 uv run --no-sync python scripts/petri_lens_probe.py probe \
  --manifest configs/capeval/manifests/petri_twosi.yaml \
  --out-dir "$PROBE_OUT_DIR"
