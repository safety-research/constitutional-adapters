#!/bin/bash
# scripts/sbatch/fit_refusal_direction.sbatch — the Arditi refusal direction of ONE served checkpoint on a
# Slurm cluster (stage 04; scripts/arditi/fit_refusal_direction.py carries
# the recipe: difference-in-means of the residual stream between the harmful and the harmless instruction sets
# at the last instruction token of every decoder block, kind vector_all at all positions, stored as
# coefficient x raw direction — the arditi_allpos arm, c = 0.3 for the paper battery, spec FD 5).
# One GPU, one hour: a forward pass over ~128 + 128 prompts, no training.
#
# Submit (positional args — never --export, hard rule 1; --comment pins the commit the submitter validated and
# the job REFUSES if the checkout's HEAD differs; -J is the IDENTITY's name — the served id and the staged id, the two
# words that name the publication directory — which the watcher and the STOP sentinel key on and 04_fit_objects.sh's
# queue read coordinates by: every table key that serves the checkpoint sees the one fit queued, and another
# coefficient's fit has another name (when the name came from the table key, a second key missed the first
# key's queued fit and submitted a second fit of the identity to the same directory). Two jobs of one identity can still
# meet — two front doors reading the queue in the same instant, a hand submission — so the fitter publishes by ONE rename
# the kernel refuses while the directory is populated: a publication that appeared since the fit began is kept and this
# fit's bytes discarded, never a second object of the identity):
#   sbatch -J refusal-direction-<model_key>-<staged id> --comment=sha:$(git -C "$REPO" rev-parse HEAD) \
#     "$REPO/scripts/sbatch/fit_refusal_direction.sbatch" <model_key> <serve_hf_id> <revision> <out_root> [attempt] [max_attempts]
#     model_key   the registry model_id of the SERVED checkpoint (reproduce/models.yaml serve.checkpoint.model_id, e.g. qwen35_4b_it —
#                 NOT the models-table key, which need not equal it: the paper's qwen35_4b serves qwen35_4b_it): the fitter's
#                 --model-key, the `fit_model_id` of the object row it emits (the identity 04_fit_objects.sh's registry_done and the
#                 pins writer make_full22_configs.py arditi_pins_from_registry query), AND the PUBLICATION DIRECTORY's name —
#                 <out_root>/<model_key>/<staged id>, the dir 05's cells_json reads for every table key that serves the checkpoint.
#                 The object's identity names its directory, so two keys serving one checkpoint share the publication, the row and
#                 the pin (2026-10-12; when the directory was keyed by the table key,
#                 a second key found the row done, published nothing of its own, and 05 pulled from an empty directory).
#     serve_hf_id its HF repo (serve.checkpoint.hf_repo)      revision  its 40-hex pin (serve.checkpoint.revision)
#     out_root    the objects root; the object lands at <out_root>/<model_key>/ar_c<c>__arditi_allpos/
#   Knobs ride the submitting shell's environment (Slurm's implicit export — never --export=): REFUSAL_COEFFICIENT
#   (0.3), REFUSAL_HARMFUL (arditi:harmful_train), REFUSAL_HARMLESS (arditi:harmless_train), REFUSAL_N (128),
#   REFUSAL_MAX_LEN (512). The default sets are the paper fit's own training draw (sisteer.data.arditi_fit_prompts,
#   data/arditi/fit_prompts.manifest.json: the document the fit behind the committed refusal-direction objects read), read
#   OFFLINE from the verified copy reproduce/00_setup.sh --what third_party installs under $SHARED/third_party and
#   checked again in the preflight below; the builtin harmbench_val / alpaca read the HF cache OFFLINE instead
#   (mariagrandury/harmbench and tatsu-lab/alpaca pre-fetched into $HF_HOME by reproduce/lib/hf_fetch.sh).
# Shared roots: SISTEER_SHARED_ROOT from the front door's env prefix, else the two known per-user roots.
# Venv: the PREBUILT sdfbase-fit venv the fit cells run on ($SHARED/envs/<SDFBASE_FIT_ENV_NAME|sdfbase-fit>;
# SDFBASE_FIT_ENV a full path) — never a uv sync inside a job.
# Preemption-tolerant the cheap way: the fit is idempotent (a COMPLETE publication of the same request is kept),
# so on SIGTERM the job resubmits itself with the attempt counter bumped (positional) unless a STOP sentinel sits at
# $SHARED/exp/training/<job-name>/STOP or the cap is reached. RESULT line + mirror-figs on every exit.
# Stop for good: write the STOP sentinel, then scancel (a bare scancel looks like a preemption and resubmits).
#SBATCH --job-name=refusal-direction
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
set -uo pipefail

MODEL_KEY="${1:?positional 1: model_key (the registry model_id of the SERVED checkpoint: the row's fit_model_id and the publication directory's name)}"
HF_ID="${2:?positional 2: serve_hf_id (HF repo of the served checkpoint)}"
REVISION="${3:?positional 3: revision (40-hex)}"
OUT_ROOT="${4:?positional 4: out_root}"
ATTEMPT="${5:-1}"
MAX_ATTEMPTS="${6:-3}"
COEFFICIENT="$(printf '%g' "${REFUSAL_COEFFICIENT:-0.3}")"   # Python's :g spelling, so ar_c<c> below == fit_refusal_direction.staged_id (0.30 -> 0.3)
HARMFUL="${REFUSAL_HARMFUL:-arditi:harmful_train}"
HARMLESS="${REFUSAL_HARMLESS:-arditi:harmless_train}"
N_PER_SIDE="${REFUSAL_N:-128}"
MAX_LEN="${REFUSAL_MAX_LEN:-512}"

# --- the shared root: SISTEER_SHARED_ROOT from the front door's job env (the site's cluster block), else refused -------
if [ -n "${SISTEER_SHARED_ROOT:-}" ] && [ -d "$SISTEER_SHARED_ROOT" ]; then SHARED="$SISTEER_SHARED_ROOT"; CLUSTER_QOS="${SISTEER_CLUSTER_QOS:-high}"   # the front door names the QoS its escalation resubmits at (sdfbase_fit_cell.sbatch's rule)
else echo "[refusal] SISTEER_SHARED_ROOT is unset or not a directory on $(hostname): the job env names the cluster shared root (the front door exports it from the site config)"; exit 2; fi
ESCALATE_QOS="${ESCALATE_QOS:-$CLUSTER_QOS}"

RUN_NAME="${SLURM_JOB_NAME:-refusal-direction}"
STOP_DIR="$SHARED/exp/training/$RUN_NAME"
export FIGS_DIR="$SHARED/exp/figs/$RUN_NAME"
mkdir -p "$STOP_DIR" "$FIGS_DIR"
log() { echo "[$(date -Is)] [refusal $RUN_NAME attempt $ATTEMPT/$MAX_ATTEMPTS] $*"; }

# --- the checkout + the submit-time sha pin (fail-closed) ------------------------------------------------
JOB_REC="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)"          # ONE controller RPC
SCRIPT_PATH="$(sed -n 's/^ *Command=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
TIMELIMIT="$(sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
COMMENT="$(sed -n 's/^ *Comment=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
STDOUT_PATH="$(sed -n 's/^ *StdOut=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
REPO="${SDFBASE_REPO:-}"
if [ -z "$REPO" ] && [ -n "$SCRIPT_PATH" ]; then REPO="$(cd "$(dirname "$SCRIPT_PATH")/../.." 2>/dev/null && pwd)"; fi
[ -n "$REPO" ] && [ -f "$REPO/scripts/arditi/fit_refusal_direction.py" ] || { log "FATAL cannot resolve the repo checkout (SCRIPT_PATH='$SCRIPT_PATH', SDFBASE_REPO='${SDFBASE_REPO:-}')"; exit 3; }
HEAD_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
case "$COMMENT" in
  *sha:*)
    # the pin is the `sha` FIELD, found BY NAME among `,`/`;`-separated key:value fields (submit_twin.nonce_of_comment's rule): the
    # funnel appends `nonce:<n>` (comment_with_nonce) and may add fields in any order; the literal read of everything after `sha:`
    # refused every nonce-commented job at start (830863/830864)
    PIN=""; _rest="$COMMENT"
    while [ -n "$_rest" ]; do _f="${_rest%%[,;]*}"; case "$_f" in sha:*) PIN="${_f#sha:}" ;; esac; [ "$_f" = "$_rest" ] && break; _rest="${_rest#*[,;]}"; done
    [ -n "$PIN" ] || { log "FATAL --comment '$COMMENT' carries no sha:<40hex> field"; exit 3; } ;;
  *) log "FATAL no --comment=sha:<40hex> pin on this job (got '$COMMENT'); submit with --comment=sha:\$(git -C $REPO rev-parse HEAD)"; exit 3 ;;
esac
[ "$PIN" = "$HEAD_SHA" ] || { log "FATAL checkout $REPO is at $HEAD_SHA, the submission pinned $PIN — refusing to run unvalidated code"; exit 3; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ] || { log "FATAL checkout $REPO has modified tracked files"; exit 3; }

# --- paths + venv ----------------------------------------------------------------------------------------
STAGED_ID="ar_c${COEFFICIENT//./p}__arditi_allpos"
OUT_DIR="$OUT_ROOT/$MODEL_KEY/$STAGED_ID"
FIT_ENV_NAME="${SDFBASE_FIT_ENV_NAME:-sdfbase-fit}"
VENV="${SDFBASE_FIT_ENV:-$SHARED/envs/$FIT_ENV_NAME}"
PY="$VENV/bin/python"

# --- environment (cluster rules) -------------------------------------------------------------------------
source "$SHARED/scripts/hf_env.sh"           # HF_HOME (the cluster's cache) + per-user token path
export HF_HUB_OFFLINE=1                      # the snapshot and the prompt datasets are pre-fetched; never reach out
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off

# --- preflight: every external input, -s (present AND non-empty) -----------------------------------------
SNAP="$HF_HOME/hub/models--${HF_ID//\//--}/snapshots/$REVISION/config.json"
pre_ok=1
for f in "$SNAP" "$SHARED/scripts/mirror-figs.sh"; do
  [ -s "$f" ] || { log "PREFLIGHT missing/empty: $f"; pre_ok=0; }
done
[ -x "$PY" ] || { log "PREFLIGHT no venv python at $PY"; pre_ok=0; }
# an arditi:* set is the paper fit's own draw, read from the third-party root hf_env.sh exports (a helper that predates the
# export leaves it unset: the shared root's third_party/, where 00_setup.sh --what third_party installs it), never fetched
# here; the installed copy is held to every hash of data/arditi/fit_prompts.manifest.json before the GPU work starts
export SISTEER_THIRD_PARTY_DIR="${SISTEER_THIRD_PARTY_DIR:-$SHARED/third_party}" SISTEER_THIRD_PARTY_OFFLINE="${SISTEER_THIRD_PARTY_OFFLINE:-1}"
case " $HARMFUL $HARMLESS " in
  *" arditi:"*)
    [ -x "$PY" ] && "$PY" -m sisteer.data.arditi_fit_prompts fetch --offline >/dev/null \
      || { log "PREFLIGHT the refusal-direction fit prompts ($HARMFUL / $HARMLESS) are not installed and verified under $SISTEER_THIRD_PARTY_DIR (reproduce/00_setup.sh --what third_party installs them; the reason is above)"; pre_ok=0; } ;;
esac
[ "$pre_ok" = 1 ] || exit 3

# audit-exempt: out-of-lockfile-venv — this job runs from the PREBUILT
# sdfbase-fit venv on shared storage (scripts/sbatch/sdfbase_fit_env.sbatch
# builds it once from the lockfile export + the hybrid-stack pins); no job
# ever uv-syncs, so the frozen-cu128-sync guard is factually wrong here. What
# CAN happen is a node where the venv resolves to a CPU-flavored or CUDA-less
# torch, so assert the cu128 GPU torch on a live CUDA device BEFORE the
# workload and exit 92 instead (no self-resubmit).
if ! "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__, torch.cuda.get_device_name(0))'; then
  log "FATAL: $PY is not the cu128 GPU torch (CUDA unavailable, or a wrong-flavor venv on $(hostname)) — NOT running the workload (61380 guard)"
  exit 92
fi
log "repo=$REPO head=$HEAD_SHA model=$HF_ID@${REVISION:0:12} fit_model_id=$MODEL_KEY out=$OUT_DIR c=$COEFFICIENT sets=$HARMFUL/$HARMLESS n=$N_PER_SIDE max_len=$MAX_LEN env=$FIT_ENV_NAME node=$(hostname) gpu=${SLURM_JOB_GPUS:-?} timelimit=$TIMELIMIT"

# --- RESULT + mirror on EVERY exit; self-resubmit on SIGTERM -----------------------------------------------
T_START=$(date +%s)
RC=""
finished=0
finish() {
  [ "$finished" = 1 ] && return; finished=1
  local rc="${RC:-${1:-unknown}}" state
  case "$rc" in 0) state=DONE ;; 143) state=INTERRUPTED ;; *) state=FAILED ;; esac
  local sha; sha="$("$PY" -c "import json,sys; print(json.load(open(sys.argv[1])).get('object_sha256','')[:16])" "$OUT_DIR/COMPLETE" 2>/dev/null || echo none)"
  local line
  line="$state rc=$rc job=${SLURM_JOB_ID:-?} node=$(hostname) object=$MODEL_KEY/$STAGED_ID object_sha256=$sha attempt=$ATTEMPT/$MAX_ATTEMPTS elapsed_s=$(( $(date +%s) - T_START )) head=${HEAD_SHA:0:12} at=$(date -Is)"
  echo "$line" > "$FIGS_DIR/RESULT_${SLURM_JOB_ID:-x}"
  echo "$line" > "$FIGS_DIR/RESULT"
  [ -d "$OUT_DIR" ] && echo "$line" >> "$OUT_DIR/RESULT"
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
  log "RESULT: $line"
}
trap 'finish' EXIT

CHILD=""
escalate() {
  if [ "$ESCALATE_QOS" = "none" ]; then log "escalation disabled"; return; fi
  if [ -e "$STOP_DIR/STOP" ]; then log "STOP sentinel present — deliberate stop, not resubmitting."; return; fi
  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then log "attempt cap $MAX_ATTEMPTS reached — not resubmitting."; return; fi
  if [ -z "$SCRIPT_PATH" ] || [ ! -f "$SCRIPT_PATH" ]; then log "cannot resolve script path — not resubmitting."; return; fi
  log "resubmitting at qos=$ESCALATE_QOS as attempt $((ATTEMPT + 1)) (time=$TIMELIMIT, output=${STDOUT_PATH:-header})"
  SDFBASE_FIT_ENV_NAME="$FIT_ENV_NAME" REFUSAL_COEFFICIENT="$COEFFICIENT" REFUSAL_HARMFUL="$HARMFUL" REFUSAL_HARMLESS="$HARMLESS" \
  REFUSAL_N="$N_PER_SIDE" REFUSAL_MAX_LEN="$MAX_LEN" \
  sbatch --qos="$ESCALATE_QOS" --job-name="$RUN_NAME" ${TIMELIMIT:+--time="$TIMELIMIT"} \
         ${STDOUT_PATH:+--output="$STDOUT_PATH"} --comment="$COMMENT" "$SCRIPT_PATH" \
         "$MODEL_KEY" "$HF_ID" "$REVISION" "$OUT_ROOT" "$((ATTEMPT + 1))" "$MAX_ATTEMPTS" \
    || log "RESUBMIT FAILED — the fit will not continue automatically"
}
# DISARM FIRST: the handler kills the process group, which signals this shell too, and bash runs a trap between
# commands without blocking the signal being handled — an armed handler re-enters itself until the stack overflows
# (measured 2026-09-11: SIGSEGV, exit 139, the RESULT line lost).
on_term() {
  trap '' SIGTERM SIGINT SIGQUIT
  log "SIGTERM (preemption / node drain / cancel) — draining"
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null
  wait "$CHILD" 2>/dev/null
  RC=143
  escalate
  exit 143
}
trap on_term SIGTERM SIGINT SIGQUIT

# --- the fit ---------------------------------------------------------------------------------------------
mkdir -p "$OUT_ROOT/$MODEL_KEY"
srun --ntasks=1 "$PY" "$REPO/scripts/arditi/fit_refusal_direction.py" \
     --model "$HF_ID" --revision "$REVISION" --out "$OUT_DIR" \
     --harmful "$HARMFUL" --harmless "$HARMLESS" --coefficient "$COEFFICIENT" --positions all \
     --n "$N_PER_SIDE" --max-len "$MAX_LEN" --dtype bfloat16 --device cuda --model-key "$MODEL_KEY" &
CHILD=$!
wait "$CHILD"
RC=$?
log "fit exited rc=$RC after $(( $(date +%s) - T_START ))s"
exit "$RC"
