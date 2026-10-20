#!/bin/bash
#SBATCH --job-name=geom-sweep-2026-09-03
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; 1 GPU per task of the 16-GPU/user high budget
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; never the whole node (hard rule 2)
#SBATCH --time=0-03:00:00             # budget walk ~30 min + sweep; the sweep is resumable and stops itself before SLURM_JOB_END_TIME
#SBATCH --array=0-7                   # one task per model, MODELS order below (never a submit loop)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# Per-model batched-token BUDGET + served decode THROUGHPUT sweep on one H200:
# make the per-model budget, then measure throughput for each model. Task i
# serves MODELS[i]: --phase budget (the S x L ladder to the first refusal + one
# bisection, through the production standup path) then --phase sweep (per eval
# window, S = budget // L, the batch ladder, passthrough + one steered psr_kl
# cell). Results:
#   $SHARED/exp/geom-sweep/<model>/{budget.json,sweep.jsonl,sweep_summary.json}
#   figures + RESULT under $SHARED/exp/figs/$SLURM_JOB_NAME/<model>/ (mirrored)
# Fold in afterwards on the grading host:
#   scripts/geom_sweep_analyze.py --results-root <rsync of the root>
#
# SUBMIT with the commit sha as the positional arg, from a commit-addressed
# checkout under $SHARED (the pattern of petri_families_served_integration):
#   sbatch --parsable scripts/sbatch/geom_sweep.sbatch <sha>
# A canary for ONE model: add --array=<idx> (e.g. --array=5 = qwen35_4b).
# NO --export= anywhere (hard rule 1): env is set in the body. The GPU-visibility
# variable is Slurm's (hard rule 3): the sweep READS it for nvidia-smi -i, never sets it.
# Body-env seams: GEOM_MEASURE_S / GEOM_WARMUP_S / GEOM_WINDOWS / GEOM_PROMPT_TOKENS
# (set them in a wrapper body, never via --export=).
#
# audit-exempt: out-of-lockfile-venv — this job runs from a requirements-
# hash-versioned SERVING venv (configs/capeval/geom-sweep-requirements.txt =
# the vllm 0.25.1+cu129 lane plus matplotlib — or, for the two
# gemma-4 models, its -tf5141 variant with transformers==5.14.1: a SECOND venv
# of the same kind, selected per MODEL below), NOT the project lockfile, so the
# cu128 entry-point guard does not apply; there is no `uv sync` anywhere, and a
# missing venv fails `uv run --no-sync` loudly.

set -uo pipefail

# The shared root is SISTEER_SHARED_ROOT, which the front door exports from the cluster's site block; a job
# without it refuses by name (there is no built-in root).
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"

[ -s "$SHARED/scripts/hf_env.sh" ] || { echo "FATAL: $SHARED/scripts/hf_env.sh missing" >&2; exit 1; }
source "$SHARED/scripts/hf_env.sh"     # shared copy — cluster /home is node-local
export HF_HUB_OFFLINE=1                # every pinned snapshot is cache-resident (preflight verifies)
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

GEOM_SHA="${1:?REQUIRED: the commit sha to run from. No default — a stale checkout would silently run old code}"
REPO_DIR="${GEOM_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-geom-sweep-repo/$GEOM_SHA}"
[ -d "$REPO_DIR/src/sisteer" ] || { echo "FATAL: no commit-addressed checkout at $REPO_DIR — clone the mirror and checkout $GEOM_SHA there first; refusing" >&2; exit 1; }
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
[ "$HEAD_SHA" = "$GEOM_SHA" ] || { echo "FATAL: checkout HEAD $HEAD_SHA != pinned $GEOM_SHA; refusing" >&2; exit 1; }
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $REPO_DIR — runs only from an immutable checkout; refusing" >&2
  exit 1
fi

# --- the model this task serves (ONE canonical table: the Python module's) ----
# MODELS mirrors served_decode_sweep.MODELS order (pinned by tests/test_served_decode_sweep.py)
MODELS=(llama8b llama3b gemma4e4b gemma4e2b qwen35_9b qwen35_4b glm46vflash nemotron3nano4b)
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "FATAL: SLURM_ARRAY_TASK_ID is unset — submit as an array (--array=0-$(( ${#MODELS[@]} - 1 )) or one index)" >&2; exit 1; }
[ "$SLURM_ARRAY_TASK_ID" -lt "${#MODELS[@]}" ] || { echo "FATAL: array index $SLURM_ARRAY_TASK_ID >= ${#MODELS[@]} models" >&2; exit 1; }
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# --- the serving venv (requirements-hash-versioned, node-local) ---------------
# PER-FAMILY REQUIREMENTS (fan-out 2026-09-03, cluster B tasks 65057_2/_3):
# vllm-requirements.txt carries no transformers pin, so the lane floats to the
# lock's 5.15.0 (the Nemotron bump — Nemotron-3-Nano NEEDS 5.15.0 for its 'mlp'
# cache layer), and transformers 5.15.0 + vLLM 0.25.1 raises
# AmbiguousGlobalPerLayerAttributeError ('head_dim' is a per-layer attribute) at
# APIServer startup for gemma-4-E4B / E2B, which the Petri venvs at 5.14.1 serve
# fine. So the two gemma-4 models bootstrap the -tf5141 variant (a second
# requirements-hash venv); everything else the default. The hash below covers
# the selected file, so the two venvs never collide.
VLLM_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
case "$MODEL" in
  gemma4e4b|gemma4e2b) GEOM_REQS="$REPO_DIR/configs/capeval/geom-sweep-requirements-tf5141.txt"; GEOM_TF_PIN="transformers==5.14.1 (gemma-4 variant)" ;;
  *)                   GEOM_REQS="$REPO_DIR/configs/capeval/geom-sweep-requirements.txt";        GEOM_TF_PIN="transformers unpinned (the lane's lock, 5.15.0)" ;;
esac
export GEOM_REQS   # the sweep stamps its basename into budget.json / sweep rows
echo "[geom-sweep] task $SLURM_ARRAY_TASK_ID -> model $MODEL @ $HEAD_SHA on $(hostname); reqs=$(basename "$GEOM_REQS") ($GEOM_TF_PIN)"
[ -s "$VLLM_REQS" ] || { echo "FATAL: $VLLM_REQS missing" >&2; exit 1; }
[ -s "$GEOM_REQS" ] || { echo "FATAL: $GEOM_REQS missing" >&2; exit 1; }
REQ_HASH_FULL=$(cat "$VLLM_REQS" "$GEOM_REQS" | sha256sum | cut -c1-16)   # the include AND the outer file
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
    echo "[geom-sweep] serving venv already bootstrapped @ $REQ_HASH — skipping (sentinel)"
  else
    if [ ! -x "$UV_PROJECT_ENVIRONMENT/bin/python" ]; then
      uv venv --python 3.12 "$UV_PROJECT_ENVIRONMENT" || exit 1
    fi
    # --python is REQUIRED (uv pip ignores UV_PROJECT_ENVIRONMENT); unsafe-best-match
    # for the pytorch cu129 extra index; the lane's supply-chain cutoff.
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
if [ "$RC" -ne 0 ]; then
  echo "FATAL: serving-venv bootstrap failed (rc=$RC); NOT continuing." >&2
  exit "$RC"
fi
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts${PYTHONPATH:+:$PYTHONPATH}"
# the RESOLVED versions this task serves with (recorded here and, by the sweep, in budget.json / every row)
VENV_VERSIONS=$(uv run --no-sync python -c "import importlib.metadata as m; print('transformers', m.version('transformers'), 'vllm', m.version('vllm'))" 2>/dev/null) || VENV_VERSIONS="(version probe failed)"
echo "[geom-sweep] serving venv $UV_PROJECT_ENVIRONMENT from $(basename "$GEOM_REQS"): $VENV_VERSIONS"

# --- paths + fail-fast guards on every external input -------------------------
RESULTS_ROOT="$SHARED/exp/geom-sweep"
OBJECTS_ROOT="$SHARED/exp/objects/scale-12"
RESULTS_DIR="$RESULTS_ROOT/$MODEL"
export FIGS_DIR="$SHARED/exp/figs/$SLURM_JOB_NAME/$MODEL"
mkdir -p "$RESULTS_DIR" "$FIGS_DIR"

# the model's pins from the ONE canonical table (GEOM_HF_ID / GEOM_HF_REV / GEOM_CONFIG / GEOM_HF_SNAPSHOT_REL / GEOM_BANK)
ENV_LINES=$(uv run --no-sync python scripts/served_decode_sweep.py --model "$MODEL" --phase env) || { echo "FATAL: could not resolve $MODEL's pins" >&2; exit 1; }
eval "$ENV_LINES"
echo "[geom-sweep] $GEOM_HF_ID @ $GEOM_HF_REV  config=$GEOM_CONFIG  bank=$OBJECTS_ROOT/$GEOM_BANK"
[ -s "$GEOM_CONFIG" ] || { echo "FATAL: serving config $GEOM_CONFIG missing" >&2; exit 1; }
[ -s "$REPO_DIR/configs/capeval/manifests/petri_twosi.yaml" ] || { echo "FATAL: petri manifest missing" >&2; exit 1; }
[ -n "${HF_HOME:-}" ] || { echo "FATAL: HF_HOME unset after hf_env.sh" >&2; exit 1; }
[ -s "$HF_HOME/$GEOM_HF_SNAPSHOT_REL/config.json" ] || { echo "FATAL: pinned snapshot $HF_HOME/$GEOM_HF_SNAPSHOT_REL has no config.json — fetch $GEOM_HF_ID@$GEOM_HF_REV into the shared cache first" >&2; exit 1; }
[ -s "$HF_HOME/$GEOM_HF_SNAPSHOT_REL/tokenizer_config.json" ] || { echo "FATAL: pinned snapshot lacks tokenizer_config.json (the standup derives the chat template from it)" >&2; exit 1; }
if [ -d "$OBJECTS_ROOT/$GEOM_BANK" ]; then
  echo "[geom-sweep] object bank: $(ls "$OBJECTS_ROOT/$GEOM_BANK" | tr '\n' ' ')"
else
  echo "[geom-sweep] NOTE: no object bank at $OBJECTS_ROOT/$GEOM_BANK — passthrough arm only (recorded in preflight.json)"
fi
[ -x "$SHARED/scripts/mirror-figs.sh" ] || echo "[geom-sweep] WARNING: $SHARED/scripts/mirror-figs.sh missing — figures will not be mirrored"

# --- traps: child cleanup + RESULT sentinel + figure mirror (best-effort) ------
_gs_handled=0
finish() {
  local rc=$? state=COMPLETED
  if [ "$_gs_handled" -eq 1 ]; then state=TERMINATED; elif [ "$rc" -ne 0 ]; then state=FAILED; fi
  echo "$state exit=$rc elapsed_s=$SECONDS job=${SLURM_JOB_ID:-unknown} task=${SLURM_ARRAY_TASK_ID:-?} model=$MODEL node=${SLURMD_NODENAME:-$(hostname)}" \
    2>/dev/null > "$FIGS_DIR/RESULT" || true
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
}
cleanup() { _gs_handled=1; kill -TERM -$$ 2>/dev/null; wait; }
trap finish EXIT
trap cleanup SIGTERM SIGINT SIGQUIT

# --- preflight (fail-fast, recorded), then budget, then sweep -----------------
srun --ntasks=1 uv run --no-sync python scripts/served_decode_sweep.py \
  --model "$MODEL" --phase preflight --results-root "$RESULTS_ROOT" --objects-root "$OBJECTS_ROOT" &
wait $!; rc=$?
[ "$rc" -eq 0 ] || { echo "FATAL: preflight failed (rc=$rc) — see $RESULTS_DIR/preflight.json" >&2; exit "$rc"; }

srun --ntasks=1 uv run --no-sync python scripts/served_decode_sweep.py \
  --model "$MODEL" --phase budget --results-root "$RESULTS_ROOT" --objects-root "$OBJECTS_ROOT" \
  --gpu-mem-util 0.90 &
wait $!; rc=$?
[ "$rc" -eq 0 ] || { echo "FATAL: budget phase failed (rc=$rc)" >&2; exit "$rc"; }
[ -s "$RESULTS_DIR/budget.json" ] || { echo "FATAL: budget.json not written" >&2; exit 1; }

srun --ntasks=1 uv run --no-sync python scripts/served_decode_sweep.py \
  --model "$MODEL" --phase sweep --results-root "$RESULTS_ROOT" --objects-root "$OBJECTS_ROOT" \
  --gpu-mem-util 0.90 \
  --measure-s "${GEOM_MEASURE_S:-90}" --warmup-s "${GEOM_WARMUP_S:-20}" \
  --prompt-tokens "${GEOM_PROMPT_TOKENS:-1000}" ${GEOM_WINDOWS:+--windows "$GEOM_WINDOWS"} &
wait $!; rc=$?
[ "$rc" -eq 0 ] || { echo "FATAL: sweep phase failed (rc=$rc)" >&2; exit "$rc"; }
[ -s "$RESULTS_DIR/sweep_summary.json" ] || { echo "FATAL: sweep_summary.json not written" >&2; exit 1; }
cp "$RESULTS_DIR/budget.json" "$RESULTS_DIR/sweep_summary.json" "$FIGS_DIR/" 2>/dev/null || true
echo "[geom-sweep] $MODEL done: $RESULTS_DIR"
exit 0
