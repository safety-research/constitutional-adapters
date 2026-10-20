#!/bin/bash
#SBATCH --job-name=pgpsr-live-0829
#SBATCH --partition=general
#SBATCH --qos=high                    # standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # NEVER --mem=0
#SBATCH --time=0-03:00:00             # fp32 ref + 3 in-proc engines + vllm serve + proxy canary; ~45-75m expected, padded ~2.5x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --signal=B:TERM@120
#
# The §10 serving-correctness gate, LIVE, scoped to the psr_kl lens
# cells (si_safety__psr_kl_all, si_constitution__psr_kl_all) — the
# campaign's headline arm, previously UNTESTED on the vllm_lens path
# (certified only for Llama-3.1-8B / vector / all-position). Runs the
# full evidence gathering petri_serving_gate.py now implements: the
# fp32 HF reference oracle, the in-proc fidelity + Phase-0 children,
# the REAL vllm serve + routing proxy (site-aware install attestation,
# no-chunking geometry with KV-coverage + max-concurrency readback,
# prefix caching ON with the salt-isolation proof), then run_gate's
# exact canaries and no-op certificates through the live proxy.
#
# OUTCOME (sacct-triageable, tests/test_petri_serving_gate.py):
#   exit 0 = gate PASSED (gate_record.json + published certificates +
#            gate_evidence.json)
#   exit 3 = COULD NOT RUN (gate_unavailable.json) — env blocker
#   exit 4 = RAN AND DISPROVED (gate_failed.json) — a serving finding
#
# SUBMIT with `sbatch --parsable` and persist the id; stop with
# `scancel <job_id>` ONLY (M15: shared cluster, no name-scoped cancel).
# NO --export= anywhere (hard rule 1): env set in the body.
#
# audit-exempt: out-of-lockfile-venv — runs from the requirements-hash
# vllm venv (torch cu129), NOT the project lockfile; the
# venv is built once per requirements hash and job steps use
# `uv run --no-sync`. Same venv the petri probe/smokes certified.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

PETRI_GATE_SHA="${1:?REQUIRED: the commit sha to certify. No default — a stale checkout would silently run old code}"
PETRI_REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$PETRI_GATE_SHA"
if [ ! -d "$PETRI_REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $PETRI_REPO_DIR — push the" \
       "exact commit into it first; refusing" >&2
  exit 1
fi
cd "$PETRI_REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$PETRI_GATE_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $PETRI_GATE_SHA; refusing" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $PETRI_REPO_DIR; refusing" >&2
  exit 1
fi
echo "[gate-psr] running from $PETRI_REPO_DIR @ $HEAD_SHA"

# --- the pinned serving venv (requirements-hash-versioned, node-local) --------
# The SAME flock + content-sentinel bootstrap the smokes carry, so this
# job runs on ANY node; on an already-bootstrapped node the sentinel
# makes it a no-op skip. uv pip needs --python (uv pip ignores
# UV_PROJECT_ENVIRONMENT — job 114815) and --index-strategy
# unsafe-best-match (the pytorch extra index shadows names — job 116213).
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
    echo "[gate-psr] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} — skipping (sentinel)"
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

# launch_server spawns `vllm serve` via PATH (shutil.which) — the venv's
# bin/ MUST lead PATH or the server child never spawns (this absence
# killed six arrays in this campaign).
SERVE_VENV="$UV_PROJECT_ENVIRONMENT"
export PATH="$SERVE_VENV/bin:$PATH"

# The gate-psr export (registry + routing table + chat template), landed
# by export_gate_registry.py at enumeration time. Loud if absent — the
# gate itself also refuses, with the unavailable marker.
PETRI_RUN_DIR="${PETRI_RUN_DIR:-$SHARED/petri/serving/gate-psr}"
PETRI_LENS_REGISTRY_DIR="${PETRI_LENS_REGISTRY_DIR:-$PETRI_RUN_DIR/lens_registry}"
GATE_OUT_DIR="$PETRI_RUN_DIR/gate_${SLURM_JOB_ID:-manual}"
mkdir -p "$GATE_OUT_DIR"

# VLLM_PLUGINS deliberately unset: vllm filters plugins by ENTRY-POINT
# name ("activations" for vllm-lens); build_server_launch sets the
# correct value in the server child's env itself.
unset VLLM_PLUGINS 2>/dev/null || true

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

srun --ntasks=1 uv run --no-sync python scripts/petri_serving_gate.py gate \
  --manifest configs/capeval/manifests/petri_twosi.yaml \
  --out-dir "$GATE_OUT_DIR" \
  --registry-dir "$PETRI_LENS_REGISTRY_DIR" \
  --cells si_safety__psr_kl_all,si_constitution__psr_kl_all
GATE_RC=$?
echo "[gate-psr] petri_serving_gate exit $GATE_RC" \
     "(0=passed, 3=could-not-run, 4=ran-and-disproved) — out: $GATE_OUT_DIR"
exit "$GATE_RC"
