#!/bin/bash
#SBATCH --job-name=qwen-gate-f0830
#SBATCH --partition=general
#SBATCH --qos=high                    # standing default; 1 GPU of the 16-GPU/user budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # NEVER --mem=0
#SBATCH --time=0-01:00:00             # 1 engine init (+~9min GDN JIT on a fresh node) + 13 greedy batches; padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#
# The Qwen3.5 per-model serving gate (scripts/qwen_serving_gate.py): the
# hook BITES at the RECORDED mask, per fitted cell, on the wave geometry.
# STRICT outcome discipline — could-not-run exits NONZERO (no
# unavailable-exit-0 branch exists in the driver).
#
# Submit (BOTH args required — an unpinned gate certifies nothing):
#   sbatch scripts/sbatch/qwen_gate.sbatch <checkout-sha> <qwen35_9b|qwen35_4b>
# Stop: scancel <job_id> ONLY (M15 — never a name-based cancel).
# NO --export= anywhere (hard rule 1): env is set in the body.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GATE_SHA="${1:?usage: sbatch qwen_gate.sbatch <checkout-sha> <model-key>}"
MODEL_KEY="${2:?usage: sbatch qwen_gate.sbatch <checkout-sha> <model-key>}"
case "$MODEL_KEY" in qwen35_9b|qwen35_4b) ;; *)
  echo "FATAL: model key $MODEL_KEY not in {qwen35_9b,qwen35_4b}" >&2; exit 1;;
esac

REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$GATE_SHA"
if [ ! -d "$REPO_DIR/src/sisteer" ]; then
  echo "FATAL: no commit-addressed checkout at $REPO_DIR; refusing" >&2
  exit 1
fi
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$GATE_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $GATE_SHA; refusing" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $REPO_DIR — refusing" >&2
  exit 1
fi
echo "[gate] running from $REPO_DIR @ $HEAD_SHA on $(hostname) for $MODEL_KEY"

# --- pinned serving venv (identical bootstrap to qwen9b_probe.sbatch) --------
# audit-exempt: out-of-lockfile-venv — runs from the requirements-hash
# versioned vllm serving venv (torch cu129), never the
# project lockfile; a cu128 assert would be factually wrong here and
# the silent-CPU path cannot occur (no `uv sync`; a missing venv fails
# `uv run --no-sync` / the venv python loudly).
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
    echo "[gate] serving venv already bootstrapped @ ${REQ_HASH_FULL:0:8} (sentinel)"
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
[ "$RC" -ne 0 ] && { echo "FATAL: venv bootstrap failed (rc=$RC)" >&2; exit "$RC"; }

SERVE_VENV="$UV_PROJECT_ENVIRONMENT"
export PATH="$SERVE_VENV/bin:$PATH"    # console scripts + subprocess python

BANK="$SHARED/exp/objects/scale-12/$MODEL_KEY"
OUT_DIR="$SHARED/petri/serving/qwen_gate/${MODEL_KEY}/${SLURM_JOB_ID}"
mkdir -p "$OUT_DIR"
CELLS="safety_refusal__psr_kl_all,si_constitution__psr_kl_all,si_irrelevant__psr_kl_all,safety_refusal__lora_r4,si_constitution__lora_r4,si_irrelevant__lora_r4"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

srun --ntasks=1 "$SERVE_VENV/bin/python" scripts/qwen_serving_gate.py \
  --bank "$BANK" --cells "$CELLS" --out-dir "$OUT_DIR" \
  --max-model-len 16384 --max-num-seqs 32 --gpu-memory-utilization 0.90
RC=$?
echo "[gate] exit=$RC record dir: $OUT_DIR"
exit $RC
