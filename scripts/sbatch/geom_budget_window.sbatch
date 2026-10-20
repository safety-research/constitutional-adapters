#!/bin/bash
#SBATCH --job-name=geom-budget-window
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default (on another cluster, pass its own QoS at submit)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G                    # ~200G for the one GPU; never the whole node (hard rule 2)
#SBATCH --time=0-02:30:00             # venv bootstrap (<= 15 min) + ~6 standup probes of 2.5-10 min each; the walk stops itself before SLURM_JOB_END_TIME
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# ONE model's batched-token BUDGET at ONE explicit window, through the production
# standup path — the budget phase of scripts/sbatch/geom_sweep.sbatch alone, at a
# window and an S ladder of the caller's choosing (geom_sweep.sbatch hard-codes
# L = 8192, the default ladder and the throughput sweep). It re-measures
# Qwen3.5-4B's budget AT the MATH-500 window (capability:gen_long_math, L =
# 17,664 = the measured prompt max + the 16,384 cap), on the lens stack, with the
# battery's own instrument, instead of reverting the row by hand.
#
# What runs: `scripts/served_decode_sweep.py --phase budget --budget-len <L>
# --budget-ladder <S_1 x L, S_2 x L, ...>` — the ladder is walked UPWARD through
# `capctrl_serving_up.bring_up` (the SAME composer path the production standup
# uses, explicit max_model_len / max_num_seqs, gpu_memory_utilization 0.90, the
# steered psr_kl cell from the model's scale-12 bank when one exists — the same
# cell the 2026-09-04 probes loaded) until the readiness gate refuses
# (`vllm_server.kv_coverage_problem`) or the engine OOMs, then bisected ONCE.
# Output: <results-root>/<model>/budget.json with every probe, the bracket and a
# paste-ready ModelBudget record; a copy lands beside RESULT under $FIGS_DIR.
#
# SUBMIT with four positional args, from a commit-addressed checkout under $SHARED
# (never --export=, hard rule 1; the body sets its env):
#   sbatch [-J <name>] [--qos=<qos> --output=$SHARED/exp/logs/%x_%j.out] \
#     scripts/sbatch/geom_budget_window.sbatch <sha> <model_key> <L> <S1,S2,...>
#   e.g.  ... <sha> qwen35_4b 17664 20,32,40,44,48,56,64,80
# Body-env seam: GEOM_REPO_DIR (an explicit checkout path; default below).
# The GPU-visibility variable is Slurm's (hard rule 3): the sweep READS it for
# nvidia-smi -i, never sets it.
#
# audit-exempt: out-of-lockfile-venv — the same requirements-hash-versioned
# SERVING venv geom_sweep.sbatch builds (configs/capeval/geom-sweep-requirements.txt
# = the vllm 0.25.1+cu129 lane + matplotlib; the -tf5141 variant for gemma-4),
# NOT the project lockfile; there is no `uv sync` anywhere, and a missing venv
# fails `uv run --no-sync` loudly.

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
MODEL="${2:?REQUIRED: the canonical model key (served_decode_sweep.MODELS: llama8b llama3b gemma4e4b gemma4e2b qwen35_9b qwen35_4b glm46vflash nemotron3nano4b)}"
BUDGET_LEN="${3:?REQUIRED: the probe window L (tokens; a multiple of the vLLM block size 16)}"
SEQS_LADDER="${4:?REQUIRED: the S ladder as a comma list, e.g. 20,32,40,44,48,56,64,80 (probed as S x L products, upward to the first refusal, then one bisection)}"
REPO_DIR="${GEOM_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-geom-sweep-repo/$GEOM_SHA}"
[ -d "$REPO_DIR/src/sisteer" ] || { echo "FATAL: no commit-addressed checkout at $REPO_DIR — clone the mirror and checkout $GEOM_SHA there first; refusing" >&2; exit 1; }
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
[ "$HEAD_SHA" = "$GEOM_SHA" ] || { echo "FATAL: checkout HEAD $HEAD_SHA != pinned $GEOM_SHA; refusing" >&2; exit 1; }
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $REPO_DIR — runs only from an immutable checkout; refusing" >&2
  exit 1
fi

# --- the window and the ladder (fail-fast: integers, block-aligned, ascending) --
case "$BUDGET_LEN" in ''|*[!0-9]*) echo "FATAL: window '$BUDGET_LEN' is not an integer" >&2; exit 1 ;; esac
[ $((BUDGET_LEN % 16)) -eq 0 ] || { echo "FATAL: window $BUDGET_LEN is not a multiple of the vLLM block size 16" >&2; exit 1; }
PRODUCTS=""; PREV=0
IFS=',' read -r -a SEQS <<< "$SEQS_LADDER"
[ "${#SEQS[@]}" -ge 1 ] || { echo "FATAL: empty S ladder" >&2; exit 1; }
for S in "${SEQS[@]}"; do
  case "$S" in ''|*[!0-9]*) echo "FATAL: ladder rung '$S' is not an integer" >&2; exit 1 ;; esac
  [ "$S" -gt "$PREV" ] || { echo "FATAL: ladder must be strictly ascending (got $S after $PREV)" >&2; exit 1; }
  PREV=$S
  PRODUCTS="${PRODUCTS:+$PRODUCTS,}$((S * BUDGET_LEN))"
done
echo "[geom-budget] model $MODEL @ $HEAD_SHA on $(hostname): window L=$BUDGET_LEN, S ladder {$SEQS_LADDER} -> products {$PRODUCTS}"

# --- the serving venv (requirements-hash-versioned, node-local) ---------------
# PER-FAMILY REQUIREMENTS, exactly as geom_sweep.sbatch: the two gemma-4 models
# serve from the transformers==5.14.1 variant venv, everything else from the
# lane's default (vllm-requirements.txt pins transformers itself: 5.14.1 since 2026-09-10).
VLLM_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
case "$MODEL" in
  gemma4e4b|gemma4e2b) GEOM_REQS="$REPO_DIR/configs/capeval/geom-sweep-requirements-tf5141.txt"; GEOM_TF_PIN="transformers==5.14.1 (gemma-4 variant)" ;;
  *)                   GEOM_REQS="$REPO_DIR/configs/capeval/geom-sweep-requirements.txt";        GEOM_TF_PIN="transformers per the lane's own pin (vllm-requirements.txt: 5.14.1 since 2026-09-10)" ;;
esac
export GEOM_REQS   # the sweep stamps its basename into budget.json
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
echo "[geom-budget] serving venv $UV_PROJECT_ENVIRONMENT from $(basename "$GEOM_REQS") ($GEOM_TF_PIN)"
(
  flock -x 9 || exit 99
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL")" = "$REQ_HASH_FULL" ]; then
    echo "[geom-budget] serving venv already bootstrapped @ $REQ_HASH — skipping (sentinel)"
  else
    echo "[geom-budget] bootstrapping the serving venv @ $REQ_HASH (first use on this node)"
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
VENV_VERSIONS=$(uv run --no-sync python -c "import importlib.metadata as m; print('transformers', m.version('transformers'), 'vllm', m.version('vllm'))" 2>/dev/null) || VENV_VERSIONS="(version probe failed)"
echo "[geom-budget] serving venv versions: $VENV_VERSIONS"

# --- paths + fail-fast guards on every external input -------------------------
# A per-WINDOW results root: the L=8192 sweep's <SHARED>/exp/geom-sweep/<model>/
# budget.json is fingerprint-locked to ITS window and ladder and must stay the
# record of that measurement; this one is the record of L=$BUDGET_LEN.
RESULTS_ROOT="$SHARED/exp/geom-sweep-L$BUDGET_LEN"
OBJECTS_ROOT="$SHARED/exp/objects/scale-12"
RESULTS_DIR="$RESULTS_ROOT/$MODEL"
export FIGS_DIR="$SHARED/exp/figs/$SLURM_JOB_NAME/$MODEL"
mkdir -p "$RESULTS_DIR" "$FIGS_DIR"

ENV_LINES=$(uv run --no-sync python scripts/served_decode_sweep.py --model "$MODEL" --phase env) || { echo "FATAL: could not resolve $MODEL's pins" >&2; exit 1; }
eval "$ENV_LINES"
echo "[geom-budget] $GEOM_HF_ID @ $GEOM_HF_REV  config=$GEOM_CONFIG  bank=$OBJECTS_ROOT/$GEOM_BANK  results=$RESULTS_DIR"
[ -s "$GEOM_CONFIG" ] || { echo "FATAL: serving config $GEOM_CONFIG missing" >&2; exit 1; }
[ -s "$REPO_DIR/configs/capeval/manifests/petri_twosi.yaml" ] || { echo "FATAL: petri manifest missing" >&2; exit 1; }
[ -n "${HF_HOME:-}" ] || { echo "FATAL: HF_HOME unset after hf_env.sh" >&2; exit 1; }
[ -s "$HF_HOME/$GEOM_HF_SNAPSHOT_REL/config.json" ] || { echo "FATAL: pinned snapshot $HF_HOME/$GEOM_HF_SNAPSHOT_REL has no config.json — fetch $GEOM_HF_ID@$GEOM_HF_REV into the cache first" >&2; exit 1; }
[ -s "$HF_HOME/$GEOM_HF_SNAPSHOT_REL/tokenizer_config.json" ] || { echo "FATAL: pinned snapshot lacks tokenizer_config.json (the standup derives the chat template from it)" >&2; exit 1; }
if [ -d "$OBJECTS_ROOT/$GEOM_BANK" ]; then
  echo "[geom-budget] object bank: $(ls "$OBJECTS_ROOT/$GEOM_BANK" | tr '\n' ' ')"
else
  echo "[geom-budget] NOTE: no object bank at $OBJECTS_ROOT/$GEOM_BANK — passthrough standup (recorded in preflight.json)"
fi
[ -x "$SHARED/scripts/mirror-figs.sh" ] || echo "[geom-budget] WARNING: $SHARED/scripts/mirror-figs.sh missing — RESULT will not be mirrored"

# --- traps: child cleanup + RESULT sentinel + mirror (best-effort) -------------
_gb_handled=0
finish() {
  local rc=$? state=COMPLETED
  if [ "$_gb_handled" -eq 1 ]; then state=TERMINATED; elif [ "$rc" -ne 0 ]; then state=FAILED; fi
  echo "$state exit=$rc elapsed_s=$SECONDS job=${SLURM_JOB_ID:-unknown} model=$MODEL window=$BUDGET_LEN ladder=$SEQS_LADDER node=${SLURMD_NODENAME:-$(hostname)}" \
    2>/dev/null > "$FIGS_DIR/RESULT" || true
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
}
# DISARM SIGTERM FIRST (the handler re-enters itself otherwise: measured 2026-09-11)
cleanup() { trap '' SIGTERM; _gb_handled=1; kill -TERM -$$ 2>/dev/null; wait; }
trap finish EXIT
trap cleanup SIGTERM SIGINT SIGQUIT

# --- preflight (fail-fast, recorded), then the budget walk --------------------
srun --ntasks=1 uv run --no-sync python scripts/served_decode_sweep.py \
  --model "$MODEL" --phase preflight --results-root "$RESULTS_ROOT" --objects-root "$OBJECTS_ROOT" &
wait $!; rc=$?
[ "$rc" -eq 0 ] || { echo "FATAL: preflight failed (rc=$rc) — see $RESULTS_DIR/preflight.json" >&2; exit "$rc"; }

srun --ntasks=1 uv run --no-sync python scripts/served_decode_sweep.py \
  --model "$MODEL" --phase budget --results-root "$RESULTS_ROOT" --objects-root "$OBJECTS_ROOT" \
  --budget-len "$BUDGET_LEN" --budget-ladder "$PRODUCTS" --gpu-mem-util 0.90 &
wait $!; rc=$?
[ "$rc" -eq 0 ] || { echo "FATAL: budget phase failed (rc=$rc)" >&2; exit "$rc"; }
[ -s "$RESULTS_DIR/budget.json" ] || { echo "FATAL: budget.json not written" >&2; exit 1; }
# the positive artifact: a COMPLETE bracket with a passing probe (a walk that stopped
# for the deadline, an unready engine or a first-rung refusal is not a measurement)
"$UV_PROJECT_ENVIRONMENT/bin/python" - "$RESULTS_DIR/budget.json" <<'PY' || { echo "FATAL: budget.json is not a complete measurement" >&2; exit 1; }
import json, sys
d = json.load(open(sys.argv[1]))
lp = d.get("largest_passing")
ok = d.get("complete") is True and lp is not None and not d.get("aborted") and not d.get("stopped_for_deadline")
print(f"[geom-budget] complete={d.get('complete')} largest_passing={lp and lp['product']} (S={lp and lp['max_num_seqs']} at L={d.get('max_model_len')}) "
      f"first_refused={d.get('first_refused')} probes={d.get('n_probes')} aborted={d.get('aborted')} deadline={d.get('stopped_for_deadline')}")
sys.exit(0 if ok else 1)
PY
cp "$RESULTS_DIR/budget.json" "$RESULTS_DIR/preflight.json" "$FIGS_DIR/" 2>/dev/null || true
echo "[geom-budget] $MODEL @ L=$BUDGET_LEN done: $RESULTS_DIR/budget.json"
exit 0
