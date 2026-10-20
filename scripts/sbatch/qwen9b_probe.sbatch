#!/bin/bash
#SBATCH --job-name=qwen9b-probe-f0829
#SBATCH --partition=general
#SBATCH --qos=high                    # standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # NEVER --mem=0
#SBATCH --time=0-01:30:00             # ~5 engine inits + tiny greedy gens (~35-45 min), padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#
# PROBE-FIRST gate for Qwen3.5-9B on the pinned serving stack (vllm
# 0.25.1+cu129, venv sisteer-petri-vllm-<reqhash>): no qwen3_5
# checkpoint has ever loaded under 0.25.1 (proven only on 0.26.0, which is
# unadoptable: protobuf conflict vs evalplus — see
# configs/capeval/vllm-requirements.txt). Nothing about the scale-12 qwen
# wave gets built until this passes. Phases and pass evidence are owned by
# scripts/qwen9b_serve_probe.py (rides along UNTRACKED, sha256 recorded
# below, the petri_probe.sbatch pattern).
#
# Submit: sbatch scripts/sbatch/qwen9b_probe.sbatch [<checkout-sha>]
# Stop:   scancel <job_id> ONLY (M15 — never a name-scoped scancel).
# NO --export= anywhere (hard rule 1): env is set in the body.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export HF_HUB_OFFLINE=1                # model + revision verified in cache from the login node
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# The exact commit this probe runs against. Positional override ($1) only.
PROBE_SHA="${1:-e383011c0bfe3bca3123430d140bc4ec75acfe26}"
REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$PROBE_SHA"

if [ ! -d "$REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $REPO_DIR; refusing" >&2
  exit 1
fi
cd "$REPO_DIR"

HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$PROBE_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $PROBE_SHA; refusing" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $REPO_DIR — refusing" >&2
  exit 1
fi
echo "[probe] running from $REPO_DIR @ $HEAD_SHA on $(hostname)"
echo "[probe] probe script sha256: $(sha256sum scripts/qwen9b_serve_probe.py)"

# --- the pinned serving venv (requirements-hash-versioned, node-local) --------
# audit-exempt: out-of-lockfile-venv — runs from the requirements-hash
# versioned vllm serving venv (torch cu129, capeval), never the
# project lockfile; a cu128 assert would be factually wrong here and
# the silent-CPU path cannot occur (no `uv sync`; a missing venv fails
# `uv run --no-sync` / the venv python loudly).
# Identical bootstrap to petri_probe.sbatch / petri_gate.sbatch: flock +
# content sentinel, uv pip with --index-strategy unsafe-best-match and the
# repo cutoff (the vllm direct-URL wheel is sha256-pinned, exempt).
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
    echo "[probe] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} — skipping (sentinel)"
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

SERVE_VENV="$UV_PROJECT_ENVIRONMENT"
# venv bin MUST be on PATH, not merely installed — a console-script exec
# from a subprocess otherwise dies bare (this killed six job arrays on
# 2026-08-28; see the serving-geometry note).
export PATH="$SERVE_VENV/bin:$PATH"

OUT_DIR="$SHARED/petri/serving/qwen9b_probe/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
echo "[probe] evidence dir: $OUT_DIR"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

run_phase() {
  local ph="$1"
  local log="$OUT_DIR/phase_${ph}.log"
  echo "[probe] ===== phase $ph ====="
  srun --ntasks=1 "$SERVE_VENV/bin/python" scripts/qwen9b_serve_probe.py \
      --phase "$ph" --out "$OUT_DIR/phase_${ph}.json" 2>&1 | tee "$log"
  local rc=${PIPESTATUS[0]}
  local conc
  conc=$(grep -Eo "Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x" "$log" | tail -1 || true)
  echo "[probe] phase $ph rc=$rc concurrency-readback='${conc:-none}'"
  return "$rc"
}

# Phase A: the decisive load+generate test. A failure here is the hard
# blocker the probe exists to surface — stop immediately, exit nonzero.
if ! run_phase A; then
  echo "PROBE_FAIL: Qwen3.5-9B cannot init/generate on vllm 0.25.1 (phase A)" >&2
  exit 1
fi

# Phase B: bracket the no-chunk identity geometry at 16384 tokens.
WINNER=""
if run_phase B32; then
  WINNER="32"
  run_phase B48 && WINNER="48" || echo "[probe] B48 stretch failed (non-fatal); staying at 32"
else
  echo "[probe] B32 failed; falling back to B24"
  if run_phase B24; then
    WINNER="24"
  else
    echo "PROBE_FAIL: no no-chunk geometry passed at 16384 (B32, B24 both failed)" >&2
    exit 1
  fi
fi
echo "[probe] geometry bracket winner: max_num_seqs=$WINNER @ 16384"

# Phase C: blocker-1 recon — prefix caching via mamba align mode. Failure is
# RECORDED, not fatal: it decides how the wave handles caching for qwen.
if run_phase "C$WINNER"; then
  echo "[probe] C$WINNER: prefix caching via --mamba-cache-mode=align WORKS on 0.25.1"
else
  echo "[probe] C$WINNER FAILED: qwen serves caching-off; gate assertion must go model-aware"
fi

echo "PROBE_DONE winner_max_num_seqs=$WINNER evidence=$OUT_DIR"
exit 0
