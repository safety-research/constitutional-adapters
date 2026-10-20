#!/bin/bash
# scripts/sbatch/hpo_trial.sbatch — ONE trial of a tune_arm.py hyperparameter
# search on a Slurm cluster, array-safe: the array index IS the trial index of the design file, one GPU each.
#
# Submit (positional args — never --export; --comment pins the commit the
# submitter validated and the job REFUSES if the checkout's HEAD differs):
#   sbatch -J hpo-<model>-lora-r1 --array=0-15 --time=12:00:00 \
#     --comment=sha:$(git -C "$REPO" rev-parse HEAD) \
#     "$REPO/scripts/sbatch/hpo_trial.sbatch" \
#     <trials.json> <hpo_root> <arm> <config> <pairs> [epochs=3]
#   positional: trials.json  the `emit-trials` design (its `round` names r<R>)
#               hpo_root     results root: <hpo_root>/r<R>_t<NN>/{summary.json,
#                            metrics.jsonl, ckpt/} — NN is the design ROW's
#                            `index` (trials[$SLURM_ARRAY_TASK_ID].index), the
#                            directory tune_arm.py trial writes; a confirm-pick
#                            design (one row, a reserved index) runs as --array 0-0
#               arm          lora_r4_ce | psr_ce_all
#               config       configs/sdfbase_<fit_key>_fits.yaml (memprobe-folded)
#               pairs        the trial pool jsonl (constv2-mix-10m)
#               epochs       trial budget (default 3)
# Modes (env, Slurm's implicit export; never --export=): HPO_JOB_MODE=trial
# (default) runs `tune_arm.py trial --index $SLURM_ARRAY_TASK_ID`;
# HPO_JOB_MODE=recon reads <trials.json> as a RECON SPEC {lr_ladder, steps,
# clip_percentile, warmup_frac, lora_rank, lora_modules, init_scheme} and runs
# `tune_arm.py recon`. Submitted as an ARRAY (`--array=0-<rungs-1>`, one GPU per
# task) every task trains exactly rung
# $SLURM_ARRAY_TASK_ID of the ladder (`recon --rung i`) and writes
# <hpo_root>/recon_<i>.json; the reduce (`recon --finalize`, no GPU) runs once
# every rung's record exists — the front door's recon phase does it. Without
# an array index the whole ladder runs serially in one job (one model load)
# and writes <hpo_root>/recon.json itself. HPO_CHECKPOINT_EVERY (default 100) is the trial's
# checkpoint interval; a preempted / timed-out trial resumes from its newest
# checkpoint under the same fit fingerprint (the trainer's own logic).
# Venv: the PREBUILT sdfbase-fit venv ($SHARED/envs/${SDFBASE_FIT_ENV_NAME:-
# sdfbase-fit}; SDFBASE_FIT_ENV overrides) — scripts/sbatch/sdfbase_fit_env.sbatch
# builds it; no job ever uv-syncs.
#SBATCH --job-name=hpo-trial
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
set -uo pipefail

TRIALS="${1:?positional 1: trials.json (the emit-trials design, or the recon spec under HPO_JOB_MODE=recon)}"
HPO_ROOT="${2:?positional 2: hpo_root}"
ARM="${3:?positional 3: arm (lora_r4_ce | psr_ce_all)}"
CONFIG="${4:?positional 4: config yaml}"
PAIRS="${5:?positional 5: pairs jsonl}"
EPOCHS="${6:-3}"
MODE="${HPO_JOB_MODE:-trial}"
CKPT_EVERY="${HPO_CHECKPOINT_EVERY:-100}"
TASK="${SLURM_ARRAY_TASK_ID:-0}"

# The shared root: SISTEER_SHARED_ROOT, which rides every submission's env prefix (the front door's job env, from the
# site's cluster block); refused by name when it is unset or not a directory on this node.
if [ -n "${SISTEER_SHARED_ROOT:-}" ] && [ -d "$SISTEER_SHARED_ROOT" ]; then SHARED="$SISTEER_SHARED_ROOT"
else echo "[hpo] SISTEER_SHARED_ROOT is unset or not a directory on $(hostname): the job env names the cluster shared root (the front door exports it from the site config)"; exit 2; fi

RUN_NAME="${SLURM_JOB_NAME:-hpo-trial}"
export FIGS_DIR="$SHARED/exp/figs/$RUN_NAME"
mkdir -p "$FIGS_DIR"
log() { echo "[$(date -Is)] [hpo $RUN_NAME task $TASK mode $MODE] $*"; }

# --- the checkout + the submit-time sha pin (fail-closed) -------------------
JOBQ="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-}}"
JOB_REC="$(scontrol show job "$JOBQ" 2>/dev/null)"          # ONE controller RPC per task
SCRIPT_PATH="$(sed -n 's/^ *Command=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
COMMENT="$(sed -n 's/^ *Comment=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
REPO="${SDFBASE_REPO:-}"
if [ -z "$REPO" ] && [ -n "$SCRIPT_PATH" ]; then REPO="$(cd "$(dirname "$SCRIPT_PATH")/../.." 2>/dev/null && pwd)"; fi
[ -n "$REPO" ] && [ -f "$REPO/scripts/tune_arm.py" ] || { log "FATAL cannot resolve the repo checkout (SCRIPT_PATH='$SCRIPT_PATH', SDFBASE_REPO='${SDFBASE_REPO:-}')"; exit 3; }
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

# --- venv + environment (cluster rules) --------------------------------------
FIT_ENV_NAME="${SDFBASE_FIT_ENV_NAME:-sdfbase-fit}"
VENV="${SDFBASE_FIT_ENV:-$SHARED/envs/$FIT_ENV_NAME}"
PY="$VENV/bin/python"
source "$SHARED/scripts/hf_env.sh"           # HF_HOME (shared cache) + per-user token path
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export SISTEER_VOL_ROOT="$SHARED/vol"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
export WANDB_MODE=offline WANDB_CONSOLE=off
# the site's W&B entity, which the front door's job env carries as SISTEER_WANDB_ENTITY, is the entity a trial's offline run is
# filed under (tune_arm.py reads WANDB_ENTITY; one the job env sets itself wins). With neither the trial logs as before
[ -n "${WANDB_ENTITY:-}" ] || [ -z "${SISTEER_WANDB_ENTITY:-}" ] || export WANDB_ENTITY="$SISTEER_WANDB_ENTITY"
export WANDB_DIR="$SHARED/exp/wandb/hpo/$RUN_NAME"
mkdir -p "$WANDB_DIR" "$HPO_ROOT"

# --- preflight: every external input, -s (present AND non-empty) ------------
pre_ok=1
for f in "$TRIALS" "$CONFIG" "$PAIRS" "$SHARED/scripts/mirror-figs.sh"; do
  [ -s "$f" ] || { log "PREFLIGHT missing/empty: $f"; pre_ok=0; }
done
[ -x "$PY" ] || { log "PREFLIGHT no venv python at $PY"; pre_ok=0; }
[ "$pre_ok" = 1 ] || exit 3

# audit-exempt: out-of-lockfile-venv — this job runs from the PREBUILT
# sdfbase-fit venv on shared storage (scripts/sbatch/sdfbase_fit_env.sbatch
# builds it from the lockfile export + the hybrid-stack pins; no job ever
# uv-syncs), so the frozen-cu128-sync guard does not apply. What CAN happen
# is a node where the venv resolves to a CPU or CUDA-less torch — so assert
# the cu128 GPU torch on a live CUDA device BEFORE the workload (exit 92).
if ! "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__, torch.cuda.get_device_name(0))'; then
  log "FATAL: $PY is not the cu128 GPU torch (CUDA unavailable, or a wrong-flavor venv on $(hostname)) — NOT running the workload (61380 guard)"
  exit 92
fi

# the design's data cell + the config's batch size (the memprobe fold) — read
# from the authoritative files, never guessed from a name
SI="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("si") or "")' "$TRIALS")"
ROUND="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("round", 1))' "$TRIALS")"
BS="$("$PY" -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); v=(c.get("batch_size_by_arm") or {}).get(sys.argv[2]); print(v if isinstance(v,int) else "")' "$CONFIG" "$ARM")"
[ -n "$BS" ] || { log "FATAL $CONFIG carries no integer batch_size_by_arm.$ARM — fold the memprobe first"; exit 3; }
[ -n "$SI" ] || { log "FATAL $TRIALS names no data cell (si) — re-emit with --si"; exit 3; }
# The trial directory is the DESIGN ROW's index, not the array task id: `tune_arm.py trial --index $TASK` writes
# r<R>_t<trials[$TASK].index>, and a confirm-pick design is one row whose index was reserved past the earlier
# confirm trials (r8_t01 submitted as --array 0-0) — deriving the checkpoint / RESULT paths from the task id put
# them under r8_t00 while the summary landed under r8_t01. An array task outside the design refuses before any
# model load. Recon mode has no trial row and no trial dir.
TRIAL_IDX=""; TRIAL_DIR=""
if [ "$MODE" != recon ]; then
  TRIAL_IDX="$("$PY" -c '
import json, sys
d = json.load(open(sys.argv[1])); ts = d.get("trials") or []; k = int(sys.argv[2])
if k < 0 or k >= len(ts): sys.exit(f"array task {k} is outside the design ({len(ts)} trial row(s))")
i = ts[k].get("index")
if isinstance(i, bool) or not isinstance(i, int) or i < 0: sys.exit(f"trial row {k} carries no integer index ({i!r})")
print(i)' "$TRIALS" "$TASK" 2>&1)" || { log "FATAL $TRIALS: $TRIAL_IDX (task $TASK) — refusing to run"; exit 3; }
  TRIAL_DIR="$(printf '%s/r%s_t%02d' "$HPO_ROOT" "$ROUND" "$TRIAL_IDX")"
fi
log "repo=$REPO head=$HEAD_SHA trials=$TRIALS round=$ROUND task=$TASK index=${TRIAL_IDX:-n/a} trial_dir=${TRIAL_DIR:-n/a} arm=$ARM si=$SI bs=$BS epochs=$EPOCHS config=$CONFIG ($(sha256sum "$CONFIG" | cut -c1-16)) pairs=$PAIRS root=$HPO_ROOT env=$VENV node=$(hostname) gpu=${SLURM_JOB_GPUS:-?}"

# --- RESULT + mirror on EVERY exit ----------------------------------------------
T_START=$(date +%s)
RC=""
finished=0
finish() {
  [ "$finished" = 1 ] && return; finished=1
  local rc="${RC:-${1:-unknown}}" state
  case "$rc" in 0) state=DONE ;; 143) state=INTERRUPTED ;; *) state=FAILED ;; esac
  local st=none
  [ -n "$TRIAL_DIR" ] && st="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$TRIAL_DIR/summary.json" 2>/dev/null || echo none)"
  local line="$state rc=$rc job=${SLURM_JOB_ID:-?} array=${SLURM_ARRAY_JOB_ID:-?}_${TASK} node=$(hostname) mode=$MODE arm=$ARM round=$ROUND task=$TASK index=${TRIAL_IDX:-n/a} status=$st elapsed_s=$(( $(date +%s) - T_START )) head=${HEAD_SHA:0:12} at=$(date -Is)"
  echo "$line" > "$FIGS_DIR/RESULT_${SLURM_JOB_ID:-x}"
  echo "$line" >> "$FIGS_DIR/RESULT"
  [ -n "$TRIAL_DIR" ] && [ -d "$TRIAL_DIR" ] && echo "$line" >> "$TRIAL_DIR/RESULT"
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
  log "RESULT: $line"
}
trap 'finish' EXIT

# DISARM SIGTERM FIRST — `kill -TERM -$$` signals this shell too, and bash
# re-enters a trap between commands without blocking the signal being handled
# (measured 2026-09-11: 2,480 re-entries -> SIGSEGV -> exit 139).
SRUN_PID=""
cleanup() {
  trap '' SIGTERM
  log "SIGTERM (preemption / node drain / cancel) — draining the trial (its checkpoint is the resume point)"
  [ -n "$SRUN_PID" ] && kill -TERM "$SRUN_PID" 2>/dev/null
  wait "$SRUN_PID" 2>/dev/null
  kill -TERM -$$ 2>/dev/null
  wait
  RC=143
  exit 143
}
trap cleanup SIGTERM SIGINT SIGQUIT

# --- the workload ------------------------------------------------------------------
if [ "$MODE" = recon ]; then
  # the recon spec in ONE read: the ladder, steps, percentile and warm-up as
  # shell assignments, the optional architecture / init axes as ready argv
  # words (shlex-quoted, so an empty EXTRA is a valid empty array)
  eval "$("$PY" -c '
import json, shlex, sys
d = json.load(open(sys.argv[1]))
print("LADDER=" + shlex.quote(",".join(str(x) for x in d["lr_ladder"])))
for var, key, default in (("STEPS", "steps", 150), ("PCT", "clip_percentile", 95), ("WF", "warmup_frac", 0.15), ("PASSES", "cycled_passes", 1)):
    print(var + "=" + shlex.quote(str(d.get(key, default))))
extra = []
for k in ("lora_rank", "lora_modules", "init_scheme"):
    v = d.get(k)
    if v is not None:
        extra += ["--" + k.replace("_", "-"), ",".join(v) if isinstance(v, list) else str(v)]
print("EXTRA=(" + " ".join(shlex.quote(x) for x in extra) + ")")' "$TRIALS")"
  [ -n "${LADDER:-}" ] || { log "FATAL $TRIALS is not a recon spec (no lr_ladder) — nothing to run"; exit 3; }
  # cycled_passes > 1 is the operator's recorded opt-in (the front door refuses a short cell unless
  # SISTEER_HPO_RECON_ALLOW_CYCLING=1 and then stamps the passes into the spec); absent/1 = the paper's one pass
  PASSES_ARGS=(); [ "${PASSES:-1}" -gt 1 ] 2>/dev/null && { PASSES_ARGS=(--passes "$PASSES"); log "recon: cycled_passes=$PASSES (the operator's opt-in recorded in the spec)"; }
  # an array task is ONE rung: its index must name a rung of the ladder,
  # checked before the model load; a non-array job runs the whole ladder serially
  RUNG=()
  if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    N_RUNGS="$(IFS=,; set -- $LADDER; echo $#)"
    [ "$TASK" -ge 0 ] 2>/dev/null && [ "$TASK" -lt "$N_RUNGS" ] || { log "FATAL array task $TASK is outside the recon ladder (0..$((N_RUNGS - 1)), $N_RUNGS rungs) — refusing to run"; exit 3; }
    RUNG=(--rung "$TASK")
    log "recon array task $TASK: rung $TASK of $N_RUNGS -> $HPO_ROOT/recon_${TASK}.json (the reduce runs with --finalize once every rung's record exists)"
  fi
  srun --ntasks=1 "$PY" "$REPO/scripts/tune_arm.py" recon \
       --config "$CONFIG" --pairs "$PAIRS" --si "$SI" --arm "$ARM" --batch-size "$BS" \
       --out "$HPO_ROOT/recon.json" --lr-ladder "$LADDER" --steps "$STEPS" \
       --clip-percentile "$PCT" --warmup-frac "$WF" --spec "$TRIALS" "${EXTRA[@]}" "${PASSES_ARGS[@]}" "${RUNG[@]}" &
else
  mkdir -p "$TRIAL_DIR/ckpt"
  srun --ntasks=1 "$PY" "$REPO/scripts/tune_arm.py" trial \
       --trials "$TRIALS" --index "$TASK" --out "$HPO_ROOT" --arm "$ARM" \
       --pairs "$PAIRS" --si "$SI" --epochs "$EPOCHS" --config "$CONFIG" --batch-size "$BS" \
       --checkpoint-dir "$TRIAL_DIR/ckpt" --checkpoint-every "$CKPT_EVERY" &
fi
SRUN_PID=$!
wait "$SRUN_PID"
RC=$?
log "workload exited rc=$RC after $(( $(date +%s) - T_START ))s"
exit "$RC"
