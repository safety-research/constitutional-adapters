#!/bin/bash
# scripts/sbatch/sikl4_fit.sbatch — ONE row of configs/sikl4/fit_tasks.tsv (the Qwen3.5-4B system-instruction distillation
# refits) on a Slurm cluster: a fit (`fit`), the mechanics smoke (`smoke`, the same path at its own checkpoint cadence and spool)
# or a teacher top-K build (`teacher`). The cluster scaffolding is scripts/sbatch/sdfbase_fit_cell.sbatch's: the submit-time sha
# pin against the checkout this script lives in, a prebuilt venv with the cu128 runtime assert, preemption-tolerant (the
# trainer's own --checkpoint-dir resume, keyed to the fit fingerprint), self-resubmitting on SIGTERM with a positional attempt
# counter (NEVER --export; hard rule 1), STOP-sentinel aware, a RESULT line + mirror-figs push on EXIT. The workload differs: a
# fit calls scripts/train_two_si_objects.py `train` directly (always --require-teacher-topk with an explicit --teacher-topk), a
# teacher row calls its `precompute-teacher`.
#
# One job per row, named BY the row: the job name IS the row id (the table's first column), so each row has its own STOP
# sentinel ($SHARED/exp/training/<row id>/STOP), RESULT and figures directory, and the self-resubmit finds its row again by id.
# The job refuses (exit 3, before any venv) unless its name equals the positional row id and the row is in THIS checkout's table.
#
# Submit (positional args, never --export; --comment pins the commit the submitter validated; the job REFUSES if the checkout's
# HEAD differs or has modified tracked files; -J must equal the row id):
#   SISTEER_SHARED_ROOT=<shared root> SISTEER_CLUSTER_QOS=high SISTEER_REGISTRY_SUBSTRATE=<substrate> \
#   SISTEER_REGISTRY_CLUSTER_LABEL=<label> WANDB_ENTITY=<entity> WANDB_PROJECT=<project> \
#   sbatch -J <row id> --comment=sha:<40hex> [--time=0-02:00:00 for a teacher row] \
#     <checkout>/scripts/sbatch/sikl4_fit.sbatch <row id> [attempt] [max_attempts]
#   positional: <row_id> [attempt (default 1)] [max_attempts (default 3)]
# The environment prefix rides Slurm's implicit export. SISTEER_SHARED_ROOT is required (exit 2 without it); every path in the
# task table is relative to it (pools, teachers, output roots) or to the checkout (configs). A fit or smoke row also requires
# SISTEER_REGISTRY_SUBSTRATE (the producer substrate its records state). The registry campaign (si_target_refit) and the spool
# are this script's: a fit row emits into the HOLD spool $SHARED/registry/hold/si_target_refit (released only after an
# independent check), a smoke row into the scratch spool $SHARED/tmp/sikl4-smoke-spool; no push is ever configured here.
# Venv: $SHARED/envs/${SIKL_FIT_ENV:-sdfbase-fit-tf515}. W&B: offline when WANDB_ENTITY and WANDB_PROJECT are both set, else off.
# A C16 row resolves its system instruction from the candidate-SI snapshot (SISTEER_LOAD_CANDIDATE_SIS), pinned by sha below.
# Stop for good: stop-run.sh <row id> (writes $SHARED/exp/training/<row id>/STOP, then scancel) — a bare scancel looks like a
# preemption and resubmits; a STOP makes the name final (this script also refuses to START under it).
#SBATCH --job-name=sikl4-fit
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0-08:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
set -uo pipefail

ROW_ID="${1:-}"
ATTEMPT="${2:-1}"
MAX_ATTEMPTS="${3:-3}"

if [ -n "${SISTEER_SHARED_ROOT:-}" ]; then SHARED="$SISTEER_SHARED_ROOT"; CLUSTER_QOS="${SISTEER_CLUSTER_QOS:-high}"
else echo "[sikl4-fit] SISTEER_SHARED_ROOT is unset: the job env names the cluster shared root"; exit 2; fi
ESCALATE_QOS="${ESCALATE_QOS:-$CLUSTER_QOS}"   # already at high: a resubmit at the same QoS

RUN_NAME="${SLURM_JOB_NAME:-}"
log() { echo "[$(date -Is)] [sikl4-fit ${RUN_NAME:-?} attempt $ATTEMPT/$MAX_ATTEMPTS] $*"; }
[ -n "$ROW_ID" ] || { log "FATAL positional 1 (the row id of configs/sikl4/fit_tasks.tsv) is missing"; exit 3; }
[ -n "$RUN_NAME" ] || { log "FATAL no SLURM_JOB_NAME: submit with -J <row id>"; exit 3; }
STOP_DIR="$SHARED/exp/training/$RUN_NAME"
export FIGS_DIR="$SHARED/exp/figs/$RUN_NAME"
mkdir -p "$STOP_DIR" "$FIGS_DIR"

# --- the checkout + the submit-time sha pin (fail-closed) -------------------
JOB_REC="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)"          # ONE controller RPC
SCRIPT_PATH="$(sed -n 's/^ *Command=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
TIMELIMIT="$(sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
COMMENT="$(sed -n 's/^ *Comment=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
STDOUT_PATH="$(sed -n 's/^ *StdOut=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
REPO=""
if [ -n "$SCRIPT_PATH" ]; then REPO="$(cd "$(dirname "$SCRIPT_PATH")/../.." 2>/dev/null && pwd)"; fi
[ -n "$REPO" ] && [ -f "$REPO/scripts/train_two_si_objects.py" ] || { log "FATAL cannot resolve the repo checkout (SCRIPT_PATH='$SCRIPT_PATH')"; exit 3; }
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

# --- the row: this job's name, in this checkout's table (fail-closed, before any venv) ---------------------------------------
TASKS="$REPO/configs/sikl4/fit_tasks.tsv"
TASK_HEADER=$'row_id\tmode\tsi\tarm\tseed\tlr_step\tconfig\tpool\tteacher\tout_root\tcheckpoint_every'
[ "$ROW_ID" = "$RUN_NAME" ] || { log "FATAL the positional row id '$ROW_ID' is not this job's name '$RUN_NAME' — a row runs only under its own name (its STOP sentinel, RESULT and figures are keyed by it); submit with -J $ROW_ID"; exit 3; }
[ -s "$TASKS" ] || { log "FATAL no task table at $TASKS"; exit 3; }
[ "$(head -1 "$TASKS")" = "$TASK_HEADER" ] || { log "FATAL $TASKS does not start with the expected header (row_id ... checkpoint_every)"; exit 3; }
ROW="$(awk -F'\t' -v id="$ROW_ID" 'NR > 1 && $1 == id' "$TASKS")"
NROWS="$(grep -c . <<<"$ROW")"                                       # 0 when no row matched (an empty here-string)
[ "$NROWS" = 1 ] || { log "FATAL row '$ROW_ID' appears $NROWS time(s) in $TASKS (exactly one required) — not a row of this checkout's table"; exit 3; }
IFS=$'\t' read -r _ID MODE SI ARM SEED LR_STEP CONFIG_REL POOL_REL TEACHER_REL OUT_REL CKPT_EVERY _EXTRA <<< "$ROW"
[ -z "${_EXTRA:-}" ] && [ -n "${CKPT_EVERY:-}" ] || { log "FATAL row '$ROW_ID' does not have exactly the table's 11 columns"; exit 3; }
case "$MODE" in
  fit|smoke)
    case "$ARM" in lora_r4|psr_kl_all) ;; *) log "FATAL row '$ROW_ID': arm '$ARM' is not lora_r4 / psr_kl_all"; exit 3 ;; esac
    case "$SEED$LR_STEP$CKPT_EVERY" in *[!0-9]*|"") log "FATAL row '$ROW_ID': seed/lr_step/checkpoint_every must be integers ('$SEED' '$LR_STEP' '$CKPT_EVERY')"; exit 3 ;; esac
    [ "$OUT_REL" != "-" ] || { log "FATAL row '$ROW_ID' ($MODE) names no output root"; exit 3; } ;;
  teacher)
    [ "$ARM" = "-" ] && [ "$OUT_REL" = "-" ] || { log "FATAL teacher row '$ROW_ID' must carry no arm and no output root"; exit 3; } ;;
  *) log "FATAL row '$ROW_ID': mode '$MODE' is not fit / smoke / teacher"; exit 3 ;;
esac
log "row $ROW_ID: mode=$MODE si=$SI arm=$ARM seed=$SEED lr_step=$LR_STEP config=$CONFIG_REL pool=$POOL_REL teacher=$TEACHER_REL out=$OUT_REL checkpoint_every=$CKPT_EVERY"
# a STOP sentinel makes a name final: a pending successor of a deliberately stopped row never starts
[ ! -e "$STOP_DIR/STOP" ] || { log "FATAL STOP sentinel present at $STOP_DIR/STOP — $RUN_NAME was stopped deliberately and is finished; a restart takes a new row (an -lr<n> row or the next smoke attempt)"; exit 3; }

# --- paths ------------------------------------------------------------------
CONFIG="$REPO/$CONFIG_REL"
PAIRS="$SHARED/$POOL_REL"
TEACHER="$SHARED/$TEACHER_REL"
[ -s "$CONFIG" ] || { log "FATAL no config at $CONFIG"; exit 3; }
# the config's identity, pinned into the trainer (--expect-config-sha): the bytes of the pinned, clean checkout's file
CONFIG_SHA="$(sha256sum "$CONFIG" | cut -c1-16)"
if [ "$MODE" != teacher ]; then
  OUT_ROOT="$SHARED/$OUT_REL"
  CELL_ID="${SI}__${ARM}"
  CELL_DIR="$OUT_ROOT/$CELL_ID"
  CKPT_DIR="$OUT_ROOT/ckpt"
else
  OUT_ROOT="-"; CELL_ID="teacher__${SI}"; CELL_DIR="$(dirname "$TEACHER")"; CKPT_DIR="-"
fi
FIT_ENV_NAME="${SIKL_FIT_ENV:-sdfbase-fit-tf515}"
VENV="$SHARED/envs/$FIT_ENV_NAME"
PY="$VENV/bin/python"
# C16's text lives only in the candidate-SI snapshot (not in this repository), pinned here by its file sha256
CANDIDATE_SIS_SNAPSHOT="${SIKL_CANDIDATE_SIS:-$SHARED/exp/candsi_refit/candidate_sis.snapshot.json}"
CANDIDATE_SIS_SHA256="ba8efe82ece5a35a97e558e31ca9c2660c220bc3677a02193e4b4a126560d60e"

# --- environment (cluster rules) --------------------------------------------
[ -s "$SHARED/scripts/hf_env.sh" ] || { log "FATAL no $SHARED/scripts/hf_env.sh (HF_HOME + the per-user token path)"; exit 3; }
source "$SHARED/scripts/hf_env.sh"           # HF_HOME (shared cache) + per-user token path
export HF_HUB_OFFLINE=1                      # the snapshot is pre-fetched; never reach out
# NCCL's socket interface is the site's to choose (its nccl.conf): the job exports no NCCL_SOCKET_IFNAME
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
if [ "$SI" = C16 ]; then export SISTEER_LOAD_CANDIDATE_SIS="$CANDIDATE_SIS_SNAPSHOT"; fi
# the registry: this script's campaign and spool; never a push (the records wait in the spool for their release)
unset SISTEER_REGISTRY_PUSH SISTEER_REGISTRY_DISABLED
export REGISTRY_CAMPAIGN=si_target_refit
case "$MODE" in
  fit) export SISTEER_REGISTRY_SPOOL="$SHARED/registry/hold/si_target_refit" ;;
  smoke) export SISTEER_REGISTRY_SPOOL="$SHARED/tmp/sikl4-smoke-spool" ;;
  teacher) export SISTEER_REGISTRY_DISABLED=1 ;;   # a teacher build writes no record
esac
# W&B: OFFLINE on the cluster (no key exists here, by rule); synced from the grading host. Entity and project are the job
# environment's; with either missing W&B is OFF for this job (logged).
export WANDB_CONSOLE=off
export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-}"
if [ -n "$WANDB_ENTITY" ] && [ -n "$WANDB_PROJECT" ]; then
  export WANDB_MODE=offline
else
  export WANDB_MODE=disabled
  log "W&B off: WANDB_ENTITY and WANDB_PROJECT are not both set in the job environment"
fi
export WANDB_DIR="$SHARED/exp/wandb/sikl4/$RUN_NAME"
mkdir -p "$WANDB_DIR"
MODEL_SLUG="$(sed -n 's/^model: .*\///p' "$CONFIG" | head -1)"
export TT_WANDB_FIT="{\"name\": \"fit-${MODEL_SLUG}-${SI}-${ARM}-all-s${SEED}-lr${LR_STEP}\", \"tags\": [\"slurm\", \"sikl4\", \"$MODE\", \"job${SLURM_JOB_ID}\", \"attempt${ATTEMPT}\"], \"config\": {\"slurm_job_id\": \"${SLURM_JOB_ID}\", \"slurm_node\": \"$(hostname)\", \"row_id\": \"$ROW_ID\", \"lr_step\": \"$LR_STEP\"}}"

# --- preflight: every external input, -s (present AND non-empty) ------------
MODEL_REPO="$(sed -n 's/^model: //p' "$CONFIG" | head -1)"
MODEL_REV="$(sed -n 's/^model_revision: //p' "$CONFIG" | head -1)"
SNAP="$HF_HOME/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REV/config.json"
pre_ok=1
for f in "$PAIRS" "$PAIRS.tokcache.npz" "$SNAP" "$VENV/BUILD-COMPLETE" "$SHARED/scripts/mirror-figs.sh"; do
  [ -s "$f" ] || { log "PREFLIGHT missing/empty: $f"; pre_ok=0; }
done
[ -x "$PY" ] || { log "PREFLIGHT no venv python at $PY"; pre_ok=0; }
if [ "$MODE" = teacher ]; then
  # a complete artifact (tensors + manifest, the manifest written last) is never overwritten under the fits reading it
  if [ -s "$TEACHER" ] && [ -s "$TEACHER.json" ]; then log "PREFLIGHT the teacher artifact already exists at $TEACHER (+ .json) — move it aside deliberately to rebuild"; pre_ok=0; fi
else
  for f in "$TEACHER" "$TEACHER.json"; do [ -s "$f" ] || { log "PREFLIGHT missing/empty teacher artifact: $f"; pre_ok=0; }; done
  [ -n "${SISTEER_REGISTRY_SUBSTRATE:-}" ] || { log "PREFLIGHT SISTEER_REGISTRY_SUBSTRATE is unset: the records must state their producer substrate"; pre_ok=0; }
fi
if [ "$SI" = C16 ]; then
  if [ ! -s "$CANDIDATE_SIS_SNAPSHOT" ]; then log "PREFLIGHT no candidate-SI snapshot at $CANDIDATE_SIS_SNAPSHOT (C16's text)"; pre_ok=0
  elif [ "$(sha256sum "$CANDIDATE_SIS_SNAPSHOT" | cut -d' ' -f1)" != "$CANDIDATE_SIS_SHA256" ]; then log "PREFLIGHT $CANDIDATE_SIS_SNAPSHOT does not hash to the pinned $CANDIDATE_SIS_SHA256"; pre_ok=0; fi
fi
[ "$pre_ok" = 1 ] || exit 3

# audit-exempt: out-of-lockfile-venv — this job runs from the PREBUILT sdfbase-fit venv on shared storage (the same venv family
# as sdfbase_fit_cell.sbatch: built once by scripts/sbatch/sdfbase_fit_env.sbatch, never uv-synced by a job), so the
# frozen-cu128-sync guard is factually wrong here and the 61380 silent-CPU path cannot occur through a sync. What CAN happen is
# a node where the venv resolves to a CPU-flavored or CUDA-less torch, so assert the cu128 GPU torch on a live CUDA device
# BEFORE the workload and exit 92 instead; the RESULT line carries the FAILED rc, no self-resubmit.
if ! "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__, torch.cuda.get_device_name(0))'; then
  log "FATAL: $PY is not the cu128 GPU torch (CUDA unavailable, or a wrong-flavor venv on $(hostname)) — NOT running the workload (61380 guard)"
  exit 92
fi
log "repo=$REPO head=$HEAD_SHA mode=$MODE config=$CONFIG ($CONFIG_SHA) pairs=$PAIRS teacher=$TEACHER out_root=$OUT_ROOT env=$FIT_ENV_NAME ($VENV) node=$(hostname) gpu=${SLURM_JOB_GPUS:-?} spool=${SISTEER_REGISTRY_SPOOL:-none} campaign=$REGISTRY_CAMPAIGN substrate=${SISTEER_REGISTRY_SUBSTRATE:-unset} timelimit=$TIMELIMIT stdout=${STDOUT_PATH:-?}"

# --- RESULT + mirror on EVERY exit; escalation on SIGTERM -------------------
T_START=$(date +%s)
RC=""
finished=0
finish() {
  [ "$finished" = 1 ] && return; finished=1
  local rc="${RC:-${1:-unknown}}" state
  case "$rc" in 0) state=DONE ;; 143) state=INTERRUPTED ;; *) state=FAILED ;; esac
  local fp; fp="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('fit_fingerprint','')[:16])" "$CELL_DIR/COMPLETE" 2>/dev/null || echo none)"
  local line="$state rc=$rc job=${SLURM_JOB_ID:-?} node=$(hostname) row=$ROW_ID mode=$MODE cell=$CELL_ID attempt=$ATTEMPT/$MAX_ATTEMPTS elapsed_s=$(( $(date +%s) - T_START )) fingerprint=$fp head=${HEAD_SHA:0:12} at=$(date -Is)"
  echo "$line" > "$FIGS_DIR/RESULT_${SLURM_JOB_ID:-x}"
  echo "$line" > "$FIGS_DIR/RESULT"
  [ "$MODE" != teacher ] && [ -d "$CELL_DIR" ] && echo "$line" >> "$CELL_DIR/RESULT"
  [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
  log "RESULT: $line"
}
trap 'finish' EXIT

SRUN_PID=""
handled=0
escalate() {
  if [ "$ESCALATE_QOS" = "none" ]; then log "escalation disabled"; return; fi
  if [ -e "$STOP_DIR/STOP" ]; then log "STOP sentinel present — deliberate stop, not resubmitting."; return; fi
  if [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ]; then log "attempt cap $MAX_ATTEMPTS reached — not resubmitting."; return; fi
  if [ -z "$SCRIPT_PATH" ] || [ ! -f "$SCRIPT_PATH" ]; then log "cannot resolve script path — not resubmitting."; return; fi
  log "resubmitting row $ROW_ID at qos=$ESCALATE_QOS as attempt $((ATTEMPT + 1)) (time=$TIMELIMIT, output=${STDOUT_PATH:-header}, env=$FIT_ENV_NAME)"
  # the SAME job name and the SAME row id (positional 1): the successor finds its row by id and keeps this name's STOP, RESULT
  # and figures; --output = this job's own resolved log path (appends); the shared root, QoS and venv name ride an env prefix
  # (Slurm's implicit export; never --export=), as does everything else the job environment carried at submission
  SISTEER_SHARED_ROOT="$SHARED" SISTEER_CLUSTER_QOS="$CLUSTER_QOS" SIKL_FIT_ENV="$FIT_ENV_NAME" \
  sbatch --qos="$ESCALATE_QOS" --job-name="$RUN_NAME" ${TIMELIMIT:+--time="$TIMELIMIT"} \
         ${STDOUT_PATH:+--output="$STDOUT_PATH"} --comment="$COMMENT" "$SCRIPT_PATH" \
         "$ROW_ID" "$((ATTEMPT + 1))" "$MAX_ATTEMPTS" \
    || log "RESUBMIT FAILED — the run will not continue automatically"
}
on_term() {
  # disarm the signal traps FIRST: a handler re-entered by a second signal mid-drain never runs its escalation or its RESULT
  trap '' SIGTERM SIGINT SIGQUIT
  [ "$handled" -eq 1 ] && return; handled=1
  log "SIGTERM (preemption / node drain / walltime / cancel) — draining the workload (the trainer's checkpoint is the resume point)"
  [ -n "$SRUN_PID" ] && kill -TERM "$SRUN_PID" 2>/dev/null
  wait "$SRUN_PID" 2>/dev/null
  RC=143
  escalate
  exit 143
}
trap on_term SIGTERM SIGINT SIGQUIT

# GPU-duty record (gpu-util skill): one utilization row per minute, into FIGS_DIR so the mirror publishes it with the RESULT.
nvidia-smi dmon -s u -d 60 -o T > "$FIGS_DIR/dmon_${SLURM_JOB_ID:-x}.log" 2>/dev/null &
DMON_PID=$!

# --- the workload -------------------------------------------------------------
cd "$REPO" || { log "FATAL cannot cd to $REPO"; exit 3; }   # the records' fit_code_sha is the HEAD of the working directory
if [ "$MODE" = teacher ]; then
  # no --top-k: precompute-teacher's default is DEFAULT_TOP_K, the width every fit validates the artifact against
  srun --ntasks=1 "$PY" "$REPO/scripts/train_two_si_objects.py" precompute-teacher \
       --config "$CONFIG" --si "$SI" --pairs "$PAIRS" --out "$TEACHER" --token-cache auto \
       --expect-config-sha "$CONFIG_SHA" --code-sha "$HEAD_SHA" &
else
  mkdir -p "$OUT_ROOT" "$SISTEER_REGISTRY_SPOOL"
  srun --ntasks=1 "$PY" "$REPO/scripts/train_two_si_objects.py" train \
       --config "$CONFIG" --si "$SI" --arms "$ARM" --pairs "$PAIRS" --token-cache auto \
       --teacher-topk "$TEACHER" --require-teacher-topk --out "$OUT_ROOT" --expect-config-sha "$CONFIG_SHA" \
       --checkpoint-dir "$CKPT_DIR" --checkpoint-every "$CKPT_EVERY" &
fi
SRUN_PID=$!
wait "$SRUN_PID"
RC=$?
log "workload exited rc=$RC after $(( $(date +%s) - T_START ))s"
# A non-zero rc that is not a signal death is a real failure: no resubmit (a retry would fail the same way — exit 3 is a
# refused configuration, e.g. --require-teacher-topk); the RESULT line + watcher carry it.
exit "$RC"
