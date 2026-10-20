#!/bin/bash
# scripts/sbatch/memprobe_fit.sbatch — the fit-memory probe of ONE (model, arm) on one GPU:
# scripts/sdfbase/memprobe_fit.py walks the micro-batch ladder (1 -> 2 -> 4 by default)
# at 8,192 tokens with gradient checkpointing and bf16 weights, a few REAL optimizer steps per rung through the
# production trainer, and writes memprobe.json + memprobe_report.md (fits / no-fits per micro-batch, peak bytes,
# s/step, the replan flag). Exit 0 = micro-batch 1 fits; 5 = it does not. Submitted by
# reproduce/02_prepare_model.sh --step memprobe through reproduce/lib/slurm.sh (positional args — never --export=,
# hard rule 1; --comment pins the commit the submitter validated and the job REFUSES if the checkout's HEAD differs):
#   sbatch -J memprobe-<model>-<arm> --comment=sha:$(git -C "$REPO" rev-parse HEAD) \
#     "$REPO/scripts/sbatch/memprobe_fit.sbatch" <model_key> <arm> [ladder]
#   positional: <model_key>  the fits config key: configs/sdfbase_<model_key>_fits.yaml
#               <arm>        psr_ce_all | lora_r4_ce
#               [ladder]     comma list of micro-batches (default 1,2,4)
# Job-side knobs ride the submitting shell's environment (Slurm's implicit export; slurm_submit --env prefixes):
#   MEMPROBE_OUT_DIR      where memprobe.json / memprobe_report.md land
#                         (default $SHARED/exp/reproduce/<model_key>/02_prepare_model)
#   MEMPROBE_PAIRS        the pool jsonl whose longest docs seed the rows (default: the config's pool dir's
#                         pairs_constv2-mix-full.jsonl, else its held-out pool, else the probe's synthetic filler)
#   MEMPROBE_LORA_RANK / MEMPROBE_LORA_MODULES / MEMPROBE_INIT_SCHEME   the lora arm's architecture / the init
#   MEMPROBE_STEPS (3); the sequence length is the config's max_len — never an env override (the fold keys the record to it)
#   SDFBASE_FIT_ENV_NAME  the prebuilt fit venv under $SHARED/envs/ (default sdfbase-fit); SDFBASE_FIT_ENV a full path
#   SISTEER_SHARED_ROOT / SISTEER_HF_HOME  the portability prefixes (reproduce/lib/hf_env.sh's rule)
#SBATCH --job-name=memprobe-fit
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=01:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
set -uo pipefail

MODEL_KEY="${1:?positional 1: model_key (e.g. qwen35_4b_demo_base)}"
ARM="${2:?positional 2: arm (psr_ce_all | lora_r4_ce)}"
LADDER="${3:-1,2,4}"

# --- the shared root: the job's site value (the front door passes the cluster block's root); nothing is probed for ----
if [ -n "${SISTEER_SHARED_ROOT:-}" ] && [ -d "$SISTEER_SHARED_ROOT" ]; then SHARED="$SISTEER_SHARED_ROOT"
else echo "[memprobe] no SISTEER_SHARED_ROOT directory on $(hostname): set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)"; exit 2; fi

RUN_NAME="${SLURM_JOB_NAME:-memprobe-fit}"
export FIGS_DIR="$SHARED/exp/figs/$RUN_NAME"
OUT_DIR="${MEMPROBE_OUT_DIR:-$SHARED/exp/reproduce/$MODEL_KEY/02_prepare_model}"
mkdir -p "$FIGS_DIR" "$OUT_DIR"
log() { echo "[$(date -Is)] [memprobe $RUN_NAME $MODEL_KEY/$ARM] $*"; }

# --- the checkout + the submit-time sha pin (fail-closed; the sdfbase_fit_cell.sbatch rule) ----------------------------
JOB_REC="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)"   # ONE scheduler RPC; every field below is read from it
SCRIPT_PATH="$(sed -n 's/^ *Command=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
COMMENT="$(sed -n 's/^ *Comment=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
REPO="${SDFBASE_REPO:-}"
if [ -z "$REPO" ] && [ -n "$SCRIPT_PATH" ]; then REPO="$(cd "$(dirname "$SCRIPT_PATH")/../.." 2>/dev/null && pwd)"; fi
[ -n "$REPO" ] && [ -f "$REPO/scripts/sdfbase/memprobe_fit.py" ] || { log "FATAL cannot resolve the repo checkout (SCRIPT_PATH='$SCRIPT_PATH', SDFBASE_REPO='${SDFBASE_REPO:-}')"; exit 3; }
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

# --- paths ------------------------------------------------------------------------------------------------------------
CONFIG="$REPO/configs/sdfbase_${MODEL_KEY}_fits.yaml"
[ -s "$CONFIG" ] || { log "FATAL no config at $CONFIG"; exit 3; }
POOL_KEY="$(sed -n 's#^ *heldout_pool: /vol/pool/\([^/][^/]*\)/pairs_constv2-heldout\.jsonl *$#\1#p' "$CONFIG" | head -1)"
PAIRS="${MEMPROBE_PAIRS:-}"
if [ -z "$PAIRS" ] && [ -n "$POOL_KEY" ]; then
  for cand in "$SHARED/vol/pool/$POOL_KEY/pairs_constv2-mix-full.jsonl" "$SHARED/vol/pool/$POOL_KEY/pairs_constv2-heldout.jsonl"; do
    [ -s "$cand" ] && { PAIRS="$cand"; break; }
  done
fi
FIT_ENV_NAME="${SDFBASE_FIT_ENV_NAME:-sdfbase-fit}"
VENV="${SDFBASE_FIT_ENV:-$SHARED/envs/$FIT_ENV_NAME}"
PY="$VENV/bin/python"

# --- environment (cluster rules) ----------------------------------------------------------------------------------------
source "$SHARED/scripts/hf_env.sh"           # HF_HOME (the cache) + the per-user token path; honours SISTEER_HF_HOME
export HF_HUB_OFFLINE=1                      # the snapshot is pre-fetched; never reach out
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
export WANDB_MODE=disabled

# --- preflight -----------------------------------------------------------------------------------------------------------
MODEL_REPO="$(sed -n 's/^model: //p' "$CONFIG" | head -1)"
MODEL_REV="$(sed -n 's/^model_revision: //p' "$CONFIG" | head -1)"
SNAP="$HF_HOME/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REV/config.json"
pre_ok=1
for f in "$CONFIG" "$SNAP" "$VENV/BUILD-COMPLETE"; do
  [ -s "$f" ] || { log "PREFLIGHT missing/empty: $f"; pre_ok=0; }
done
[ -x "$PY" ] || { log "PREFLIGHT no venv python at $PY"; pre_ok=0; }
[ -n "$PAIRS" ] || log "PREFLIGHT note: no staged pool for $POOL_KEY under $SHARED/vol/pool — the probe uses its synthetic filler rows"
[ "$pre_ok" = 1 ] || exit 3

# audit-exempt: out-of-lockfile-venv — this job runs from the PREBUILT sdfbase-fit venv on shared storage
# (scripts/sbatch/sdfbase_fit_env.sbatch builds it once from the lockfile export + the hybrid-stack pins); no job ever
# uv-syncs, so the frozen-cu128-sync guard does not apply, and the runtime assert below is the flavour guard: the cu128
# GPU torch on a live CUDA device BEFORE the workload, else exit 92 (the fit cells' rule).
if ! "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__, torch.cuda.get_device_name(0))'; then
  log "FATAL: $PY is not the cu128 GPU torch (CUDA unavailable, or a wrong-flavor venv on $(hostname)) — NOT running the probe"
  exit 92
fi
log "repo=$REPO head=$HEAD_SHA config=$CONFIG pairs=${PAIRS:-synthetic} out=$OUT_DIR env=$FIT_ENV_NAME node=$(hostname) gpu=${SLURM_JOB_GPUS:-?}"

# --- RESULT + mirror on EVERY exit; disarm-first cleanup on a signal -----------------------------------------------------
T_START=$(date +%s)
RC=""
finished=0
finish() {
  [ "$finished" = 1 ] && return; finished=1
  local rc="${RC:-${1:-unknown}}" state
  case "$rc" in 0) state=DONE ;; 5) state=DOES_NOT_FIT ;; 143) state=INTERRUPTED ;; *) state=FAILED ;; esac
  local line="$state rc=$rc job=${SLURM_JOB_ID:-?} node=$(hostname) model=$MODEL_KEY arm=$ARM ladder=$LADDER elapsed_s=$(( $(date +%s) - T_START )) out=$OUT_DIR head=${HEAD_SHA:0:12} at=$(date -Is)"
  echo "$line" > "$FIGS_DIR/RESULT_${SLURM_JOB_ID:-x}"
  echo "$line" > "$FIGS_DIR/RESULT"
  echo "$line" >> "$OUT_DIR/RESULT.$ARM"
  # the report and the JSON ride the figure mirror so the FD 23 verdict is readable remotely
  cp -f "$OUT_DIR/memprobe.$ARM.json" "$FIGS_DIR/" 2>/dev/null || true
  cp -f "$OUT_DIR/memprobe_report.$ARM.md" "$FIGS_DIR/" 2>/dev/null || true
  [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
  log "RESULT: $line"
}
trap 'finish' EXIT
# DISARM FIRST: `kill -TERM -$$` signals this shell too, and bash runs a trap between commands without blocking the
# signal being handled — an armed handler re-enters itself until the stack overflows (measured 2026-09-11).
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; RC=143; exit 143; }
trap cleanup SIGTERM SIGINT SIGQUIT

nvidia-smi dmon -s um -d 30 -o T > "$FIGS_DIR/dmon_${SLURM_JOB_ID:-x}.log" 2>/dev/null &
DMON_PID=$!

# --- the probe --------------------------------------------------------------------------------------------------------
EXTRA=()
[ -n "$PAIRS" ] && EXTRA+=(--pairs "$PAIRS")
[ -n "${MEMPROBE_LORA_RANK:-}" ] && EXTRA+=(--lora-rank "$MEMPROBE_LORA_RANK")
[ -n "${MEMPROBE_LORA_MODULES:-}" ] && EXTRA+=(--lora-modules "$MEMPROBE_LORA_MODULES")
[ -n "${MEMPROBE_INIT_SCHEME:-}" ] && EXTRA+=(--init-scheme "$MEMPROBE_INIT_SCHEME")
srun --ntasks=1 "$PY" "$REPO/scripts/sdfbase/memprobe_fit.py" \
     --config "$CONFIG" --arm "$ARM" --ladder "$LADDER" \
     --steps "${MEMPROBE_STEPS:-3}" \
     --out "$OUT_DIR/memprobe.$ARM.json" --report "$OUT_DIR/memprobe_report.$ARM.md" "${EXTRA[@]}" &
SRUN_PID=$!
wait "$SRUN_PID"
RC=$?
log "probe exited rc=$RC after $(( $(date +%s) - T_START ))s (0 = micro-batch 1 fits; 5 = it does not — FD 23)"
exit "$RC"
