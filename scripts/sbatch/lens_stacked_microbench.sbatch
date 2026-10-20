#!/bin/bash
#SBATCH --job-name=lens-stacked-microbench
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                     # no model is loaded; never the whole node (hard rule 2)
#SBATCH --time=0-00:30:00             # all five families' shapes x 6 kinds x 5 batch sizes x 2 profiles: minutes
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#
# THE LENS HOOK MICROBENCH: per-layer apply cost of legacy / per_row / stacked
# reductions and the stacked-vs-per_row elementwise deviation (in bf16 ulps),
# per op kind, at B in {8, 32, 64, 128, 256}, on realistic-shape synthetic
# activations — no model, one GPU (scripts/lens_stacked_microbench.py).
# Results: $SHARED/exp/lens-stacked-scope/microbench/{microbench.json,microbench.md}
#          + per-model PNGs, all mirrored under $SHARED/exp/figs/$SLURM_JOB_NAME/.
# SUBMIT with the commit sha as the positional arg (checkout created from the
# mirror if absent): sbatch --parsable scripts/sbatch/lens_stacked_microbench.sbatch <sha>
# Body-env seams (never --export=): MB_BATCHES, MB_ITERS, MB_TRIALS, MB_MODELS, MB_REPO_DIR, MB_MIRROR.
# audit-exempt: out-of-lockfile-venv — the serving venv of geom_sweep.sbatch (torch is all this needs).

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"
[ -s "$SHARED/scripts/hf_env.sh" ] || { echo "FATAL: $SHARED/scripts/hf_env.sh missing" >&2; exit 1; }
source "$SHARED/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

MB_SHA="${1:?REQUIRED: the commit sha to run from}"
MIR="${MB_MIRROR:-${SISTEER_REPO_MIRROR:?set SISTEER_REPO_MIRROR, the cluster clone of this repository (the front door exports it from the site config)}}"
REPO_DIR="${MB_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-lens-vectorize-repo/$MB_SHA}"
[ -d "$MIR/.git" ] || [ -f "$MIR/HEAD" ] || { echo "FATAL: no repo mirror at $MIR (mirror-repos.sh pushes it)" >&2; exit 1; }
mkdir -p "$(dirname "$REPO_DIR")"
(
  flock -x 8 || exit 99
  if [ ! -d "$REPO_DIR/src/sisteer" ]; then
    rm -rf "$REPO_DIR"
    git clone -q --no-checkout "$MIR" "$REPO_DIR" && git -C "$REPO_DIR" checkout -q "$MB_SHA" || exit 1
  fi
) 8>"$(dirname "$REPO_DIR")/.ab-clone-$MB_SHA.lock"
RC=$?
[ "$RC" -eq 0 ] || { echo "FATAL: could not create the checkout at $REPO_DIR (rc=$RC)" >&2; exit "$RC"; }
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
[ "$HEAD_SHA" = "$MB_SHA" ] || { echo "FATAL: checkout HEAD $HEAD_SHA != pinned $MB_SHA; refusing" >&2; exit 1; }
git diff --quiet HEAD -- 2>/dev/null || { echo "FATAL: tracked files modified in $REPO_DIR; refusing" >&2; exit 1; }
[ -s scripts/lens_stacked_microbench.py ] || { echo "FATAL: no scripts/lens_stacked_microbench.py at $MB_SHA" >&2; exit 1; }

VLLM_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
GEOM_REQS="$REPO_DIR/configs/capeval/geom-sweep-requirements.txt"
[ -s "$VLLM_REQS" ] && [ -s "$GEOM_REQS" ] || { echo "FATAL: requirements files missing" >&2; exit 1; }
REQ_HASH_FULL=$(cat "$VLLM_REQS" "$GEOM_REQS" | sha256sum | cut -c1-16)
REQ_HASH=${REQ_HASH_FULL:0:8}
export UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-geom-sweep-vllm-$REQ_HASH"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy
LOCKDIR=$(dirname "$UV_PROJECT_ENVIRONMENT"); mkdir -p "$LOCKDIR"
SENTINEL="$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").vllm-bootstrap"
(
  flock -x 9 || exit 99
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL")" = "$REQ_HASH_FULL" ]; then
    echo "[mb] serving venv already bootstrapped @ $REQ_HASH — skipping (sentinel)"
  else
    [ -x "$UV_PROJECT_ENVIRONMENT/bin/python" ] || uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT" || exit 1
    uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" --index-strategy unsafe-best-match \
       --exclude-newer 2026-08-03T00:00:00Z -r "$GEOM_REQS" && echo "$REQ_HASH_FULL" > "$SENTINEL" || exit 1
  fi
) 9>"$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").uv-sync.lock"
RC=$?
[ "$RC" -eq 0 ] || { echo "FATAL: serving-venv bootstrap failed (rc=$RC)" >&2; exit "$RC"; }
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"

RESULTS_DIR="$SHARED/exp/lens-stacked-scope/microbench"
export FIGS_DIR="$SHARED/exp/figs/$SLURM_JOB_NAME"
mkdir -p "$RESULTS_DIR" "$FIGS_DIR"
finish() {
  local rc=$? state=COMPLETED; [ "$rc" -ne 0 ] && state=FAILED
  cp "$RESULTS_DIR"/microbench.json "$RESULTS_DIR"/microbench.md "$FIGS_DIR/" 2>/dev/null || true
  echo "$state exit=$rc elapsed_s=$SECONDS job=${SLURM_JOB_ID:-unknown} node=${SLURMD_NODENAME:-$(hostname)} gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)" > "$FIGS_DIR/RESULT" 2>/dev/null || true
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
}
cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap finish EXIT
trap cleanup SIGTERM SIGINT SIGQUIT
unset SISTEER_LENS_STACKED_REDUCTIONS SISTEER_LENS_SHADOW_COMPARE SISTEER_LENS_PER_ROW_APPLY

echo "[mb] $(hostname) $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1) @ $HEAD_SHA"
srun --ntasks=1 uv run --no-sync python scripts/lens_stacked_microbench.py --device cuda \
  --models "${MB_MODELS:-llama8b,gemma4e4b,qwen35_4b,glm46vflash,nemotron3nano4b}" \
  --batches "${MB_BATCHES:-8,32,64,128,256}" --iters "${MB_ITERS:-50}" --trials "${MB_TRIALS:-5}" \
  --out "$RESULTS_DIR/microbench.json" --figs "$FIGS_DIR" &
wait $!; rc=$?
echo "[mb] exit $rc: $RESULTS_DIR/microbench.md"
exit "$rc"
