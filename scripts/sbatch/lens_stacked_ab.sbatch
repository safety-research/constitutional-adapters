#!/bin/bash
#SBATCH --job-name=lens-stacked-ab
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU per task of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; never the whole node (hard rule 2)
#SBATCH --time=0-01:00:00             # 2 cases x 3 standups (~3-4 min ready + waves) + 2 timing legs per case ~ 35-45 min
#SBATCH --array=0-4                   # one task per family, MODELS order below (never a submit loop)
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# THE SERVED A/B OF STACKED vs PER-ROW REDUCTIONS IN THE LENS HOOK
# (scripts/lens_stacked_ab.py — read its docstring: what is compared, what is
# confounded, how to read the report). Scoping evidence on branch
# lens-grouped-scope; NEVER a certification and never a contract change.
# Task i serves MODELS[i] from the commit-addressed checkout of the positional
# sha, three standups per case (per_row / stacked / stacked+shadow), all at
# production VLLM_BATCH_INVARIANT=0 with prefix caching off. Results:
#   $SHARED/exp/lens-stacked-scope/ab/<model>/{lens_stacked_ab.json,report.md}
#   report.md + RESULT under $SHARED/exp/figs/$SLURM_JOB_NAME/<model>/ (mirrored)
#
# SUBMIT with the commit sha as the positional arg (the checkout is created
# here from the cluster mirror if absent — run the repo-mirror script first):
#   sbatch --parsable scripts/sbatch/lens_stacked_ab.sbatch <sha>
# One family: add --array=<idx> (0 llama8b, 1 gemma4e4b, 2 qwen35_4b,
# 3 glm46vflash, 4 nemotron3nano4b).
# Body-env seams (set in a wrapper body, NEVER via --export=, hard rule 1):
#   AB_CASES (comma list of inventory case ids; default psr response + lora_all response)
#   AB_BATCHES (default 8,32)   AB_REPEATS (default 3)   AB_TIMING_BATCHES (default 8,32,64)
#   AB_MEASURE_S (default 30)   AB_NO_TIMING=1   AB_REPO_DIR / AB_MIRROR
# The GPU-visibility variable is Slurm's (hard rule 3): never set here.
#
# audit-exempt: out-of-lockfile-venv — the same requirements-hash-versioned
# SERVING venv as lens_vectorize_gate.sbatch / geom_sweep.sbatch (vllm
# 0.25.1+cu129 lane; the gemma -tf5141 variant), never `uv sync`.

set -uo pipefail

SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

[ -s "$SHARED/scripts/hf_env.sh" ] || { echo "FATAL: $SHARED/scripts/hf_env.sh missing" >&2; exit 1; }
source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

AB_SHA="${1:?REQUIRED: the commit sha to run from. No default — a stale checkout would silently run old code}"
MIR="${AB_MIRROR:-${SISTEER_REPO_MIRROR:?set SISTEER_REPO_MIRROR, the cluster clone of this repository (the front door exports it from the site config)}}"
REPO_DIR="${AB_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-lens-vectorize-repo/$AB_SHA}"
[ -d "$MIR/.git" ] || [ -f "$MIR/HEAD" ] || { echo "FATAL: no repo mirror at $MIR (mirror-repos.sh pushes it)" >&2; exit 1; }
mkdir -p "$(dirname "$REPO_DIR")"
(
  flock -x 8 || exit 99          # five array tasks race to create the same clone
  if [ ! -d "$REPO_DIR/src/sisteer" ]; then
    rm -rf "$REPO_DIR"
    git clone -q --no-checkout "$MIR" "$REPO_DIR" && git -C "$REPO_DIR" checkout -q "$AB_SHA" || exit 1
  fi
) 8>"$(dirname "$REPO_DIR")/.ab-clone-$AB_SHA.lock"
RC=$?
[ "$RC" -eq 0 ] || { echo "FATAL: could not create the checkout at $REPO_DIR from $MIR @ $AB_SHA (rc=$RC)" >&2; exit "$RC"; }
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
[ "$HEAD_SHA" = "$AB_SHA" ] || { echo "FATAL: checkout HEAD $HEAD_SHA != pinned $AB_SHA; refusing" >&2; exit 1; }
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $REPO_DIR — runs only from an immutable checkout; refusing" >&2
  exit 1
fi
[ -s scripts/lens_stacked_ab.py ] || { echo "FATAL: $REPO_DIR has no scripts/lens_stacked_ab.py — is $AB_SHA on branch lens-grouped-scope?" >&2; exit 1; }

MODELS=(llama8b gemma4e4b qwen35_4b glm46vflash nemotron3nano4b)
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: SLURM_ARRAY_TASK_ID is unset — submit as an array (--array=0-$(( ${#MODELS[@]} - 1 )) or one index)" >&2; exit 1; }
[ "$SLURM_ARRAY_TASK_ID" -lt "${#MODELS[@]}" ] || { echo "FATAL: array index $SLURM_ARRAY_TASK_ID >= ${#MODELS[@]} models" >&2; exit 1; }
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# --- the serving venv (requirements-hash-versioned, node-local; geom_sweep's) ---
VLLM_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
case "$MODEL" in
  gemma4e4b|gemma4e2b) GEOM_REQS="$REPO_DIR/configs/capeval/geom-sweep-requirements-tf5141.txt"; GEOM_TF_PIN="transformers==5.14.1 (gemma-4 variant)" ;;
  *)                   GEOM_REQS="$REPO_DIR/configs/capeval/geom-sweep-requirements.txt";        GEOM_TF_PIN="transformers unpinned (the lane's lock, 5.15.0)" ;;
esac
export GEOM_REQS
echo "[ab] task $SLURM_ARRAY_TASK_ID -> model $MODEL @ $HEAD_SHA on $(hostname); reqs=$(basename "$GEOM_REQS") ($GEOM_TF_PIN)"
[ -s "$VLLM_REQS" ] || { echo "FATAL: $VLLM_REQS missing" >&2; exit 1; }
[ -s "$GEOM_REQS" ] || { echo "FATAL: $GEOM_REQS missing" >&2; exit 1; }
REQ_HASH_FULL=$(cat "$VLLM_REQS" "$GEOM_REQS" | sha256sum | cut -c1-16)
REQ_HASH=${REQ_HASH_FULL:0:8}
export UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-geom-sweep-vllm-$REQ_HASH"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy
LOCKDIR=$(dirname "$UV_PROJECT_ENVIRONMENT")
mkdir -p "$LOCKDIR"
SENTINEL="$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").vllm-bootstrap"
(
  flock -x 9 || exit 99
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL")" = "$REQ_HASH_FULL" ]; then
    echo "[ab] serving venv already bootstrapped @ $REQ_HASH — skipping (sentinel)"
  else
    if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
      uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT" || exit 1
    fi
    if uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
         --index-strategy unsafe-best-match \
         --exclude-newer 2026-08-03T00:00:00Z -r "$GEOM_REQS"; then
      echo "$REQ_HASH_FULL" > "$SENTINEL"
    else
      exit 1
    fi
  fi
) 9>"$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").uv-sync.lock"
RC=$?
[ "$RC" -eq 0 ] || { echo "FATAL: serving-venv bootstrap failed (rc=$RC); NOT continuing." >&2; exit "$RC"; }
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
VENV_VERSIONS=$(uv run --no-sync python -c "import importlib.metadata as m; print('transformers', m.version('transformers'), 'vllm', m.version('vllm'))" 2>/dev/null) || VENV_VERSIONS="(version probe failed)"
echo "[ab] serving venv $UV_PROJECT_ENVIRONMENT from $(basename "$GEOM_REQS"): $VENV_VERSIONS"

# --- paths + fail-fast guards -------------------------------------------------
RESULTS_ROOT="$SHARED/exp/lens-stacked-scope/ab"
OBJECTS_ROOT="$SHARED/exp/objects/scale-12"
CANDSI_ROOT="$SHARED/exp/twosweep/fits/candsi_refit"
RESULTS_DIR="$RESULTS_ROOT/$MODEL"
export FIGS_DIR="$SHARED/exp/figs/$SLURM_JOB_NAME/$MODEL"
mkdir -p "$RESULTS_DIR" "$FIGS_DIR"
ENV_LINES=$(uv run --no-sync python scripts/lens_vectorize_gate.py --model "$MODEL" --phase env) || { echo "FATAL: could not resolve $MODEL's pins" >&2; exit 1; }
eval "$ENV_LINES"
echo "[ab] $GATE_HF_ID @ $GATE_HF_REV  config=$GATE_CONFIG  bank=$OBJECTS_ROOT/$GATE_BANK  candsi=$CANDSI_ROOT/$MODEL"
[ -s "$GATE_CONFIG" ] || { echo "FATAL: serving config $GATE_CONFIG missing" >&2; exit 1; }
[ -n "${HF_HOME:-}" ] || { echo "FATAL: HF_HOME unset after hf_env.sh" >&2; exit 1; }
[ -s "$HF_HOME/$GATE_HF_SNAPSHOT_REL/config.json" ] || { echo "FATAL: pinned snapshot $HF_HOME/$GATE_HF_SNAPSHOT_REL has no config.json" >&2; exit 1; }
[ -x "$SHARED/scripts/mirror-figs.sh" ] || echo "[ab] WARNING: $SHARED/scripts/mirror-figs.sh missing — the report will not be mirrored"

_ab_handled=0
finish() {
  local rc=$? state=COMPLETED
  if [ "$_ab_handled" -eq 1 ]; then state=TERMINATED; elif [ "$rc" -ne 0 ]; then state=FAILED; fi
  cp "$RESULTS_DIR/report.md" "$RESULTS_DIR/lens_stacked_ab.json" "$FIGS_DIR/" 2>/dev/null || true
  echo "$state exit=$rc elapsed_s=$SECONDS job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID:-?} model=$MODEL node=${SLURMD_NODENAME:-$(hostname)}" \
    2>/dev/null > "$FIGS_DIR/RESULT" || true
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
}
cleanup() { _ab_handled=1; kill -TERM -$$ 2>/dev/null; wait; }
trap finish EXIT
trap cleanup SIGTERM SIGINT SIGQUIT

# Each standup sets its OWN switches (the driver refuses if any is inherited).
unset SISTEER_LENS_STACKED_REDUCTIONS SISTEER_LENS_SHADOW_COMPARE SISTEER_LENS_PER_ROW_APPLY VLLM_BATCH_INVARIANT
TIMING_FLAG=""; [ "${AB_NO_TIMING:-0}" = "1" ] && TIMING_FLAG="--no-timing"

srun --ntasks=1 uv run --no-sync python scripts/lens_stacked_ab.py \
  --model "$MODEL" --repo-dir "$REPO_DIR" \
  --results-root "$RESULTS_ROOT" --objects-root "$OBJECTS_ROOT" --candsi-root "$CANDSI_ROOT" \
  --extra-objects-dir "$SHARED/exp/objects/stage3-wave/A=llama8b" --extra-objects-dir "$SHARED/exp/objects/stage3-wave/C=llama8b" \
  ${AB_CASES:+--cases "$AB_CASES"} $TIMING_FLAG \
  --batches "${AB_BATCHES:-8,32}" --repeats "${AB_REPEATS:-3}" --timing-batches "${AB_TIMING_BATCHES:-8,32,64}" \
  --measure-s "${AB_MEASURE_S:-30}" --gpu-mem-util 0.90 &
wait $!; rc=$?
echo "[ab] $MODEL exit $rc: $RESULTS_DIR/report.md"
exit "$rc"
