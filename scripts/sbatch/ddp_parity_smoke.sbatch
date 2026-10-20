#!/bin/bash
# scripts/sbatch/ddp_parity_smoke.sbatch — the REAL 2-GPU-vs-1-GPU parity smoke of the data-parallel fit: ONE job on TWO
# GPUs runs scripts/sdfbase/ddp_parity_smoke.py twice on the same node — the reference leg on one
# GPU (`srun --gres=gpu:1`, a strict subset of the allocation through Slurm, never a device list of our own) and the wide leg
# on both (`torchrun --standalone --nproc_per_node=2` inside one srun task) — then `compare` writes parity.json / parity.md
# (loss trajectory within bf16 tolerance, step time ~halved) into the figure dir the mirror publishes.
#   sbatch -J ddpsmoke-<model>-<arm> --comment=sha:$(git -C "$REPO" rev-parse HEAD) \
#     "$REPO/scripts/sbatch/ddp_parity_smoke.sbatch" <model_key> <arm> <pool si> [steps] [accum] [width]
#   positional: <model_key>  the fits config key: configs/sdfbase_<model_key>_fits.yaml
#               <arm>        psr_ce_all | lora_r4_ce
#               <pool si>    the pool whose first rows train (e.g. constv2-mix-10m)
#               [steps]      optimizer steps per leg (default 20)     [accum] accumulation per step (default 4)
#               [width]      the wide leg's rank count (default 2; the header's gpu:2 — submit --gres=gpu:N --cpus-per-task=16N --mem=128N G for more)
# Job-side knobs ride the submitting shell's environment (Slurm's implicit export; never --export=):
#   DDPSMOKE_OUT_DIR   where the two legs + the comparison land (default $SHARED/exp/reproduce/<model_key>/ddp_parity_smoke)
#   DDPSMOKE_LR (1e-5) SDFBASE_FIT_ENV_NAME (sdfbase-fit) SDFBASE_FIT_ENV (a full venv path) SISTEER_SHARED_ROOT / SISTEER_HF_HOME
#SBATCH --job-name=ddpsmoke
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
set -uo pipefail

MODEL_KEY="${1:?positional 1: model_key (e.g. qwen35_4b_base)}"
ARM="${2:?positional 2: arm (psr_ce_all | lora_r4_ce)}"
POOL_SI="${3:?positional 3: pool si (e.g. constv2-mix-10m)}"
STEPS="${4:-20}"
ACCUM="${5:-4}"
WIDTH="${6:-2}"

if [ -n "${SISTEER_SHARED_ROOT:-}" ] && [ -d "$SISTEER_SHARED_ROOT" ]; then SHARED="$SISTEER_SHARED_ROOT"
elif [ -n "${CLUSTER_SHARED_ROOT:-}" ] && [ -d "$CLUSTER_SHARED_ROOT" ]; then SHARED="$CLUSTER_SHARED_ROOT"
else echo "[ddpsmoke] no SISTEER_SHARED_ROOT (the front door's job env) or CLUSTER_SHARED_ROOT naming a directory on $(hostname)"; exit 2; fi

RUN_NAME="${SLURM_JOB_NAME:-ddpsmoke}"
export FIGS_DIR="$SHARED/exp/figs/$RUN_NAME"
OUT_DIR="${DDPSMOKE_OUT_DIR:-$SHARED/exp/reproduce/$MODEL_KEY/ddp_parity_smoke}/${SLURM_JOB_ID:-x}"
mkdir -p "$FIGS_DIR" "$OUT_DIR"
log() { echo "[$(date -Is)] [ddpsmoke $RUN_NAME $MODEL_KEY/$ARM] $*"; }

# --- the checkout + the submit-time sha pin (fail-closed; the fit sbatch's rule) -----------------------------------------
JOB_REC="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null)"
SCRIPT_PATH="$(sed -n 's/^ *Command=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
COMMENT="$(sed -n 's/^ *Comment=\([^ ]*\).*/\1/p' <<<"$JOB_REC" | head -1)"
REPO="${SDFBASE_REPO:-}"
if [ -z "$REPO" ] && [ -n "$SCRIPT_PATH" ]; then REPO="$(cd "$(dirname "$SCRIPT_PATH")/../.." 2>/dev/null && pwd)"; fi
[ -n "$REPO" ] && [ -f "$REPO/scripts/sdfbase/ddp_parity_smoke.py" ] || { log "FATAL cannot resolve the repo checkout (SCRIPT_PATH='$SCRIPT_PATH', SDFBASE_REPO='${SDFBASE_REPO:-}')"; exit 3; }
HEAD_SHA="$(git -C "$REPO" rev-parse HEAD 2>/dev/null)"
case "$COMMENT" in
  *sha:*)
    # the pin is the `sha` FIELD, found BY NAME among `,`/`;`-separated key:value fields (submit_twin.nonce_of_comment's rule): the
    # funnel appends `nonce:<n>` (comment_with_nonce) and may add fields in any order; the literal read of everything after `sha:`
    # refused every nonce-commented job at start (830863/830864)
    PIN=""; _rest="$COMMENT"
    while [ -n "$_rest" ]; do _f="${_rest%%[,;]*}"; case "$_f" in sha:*) PIN="${_f#sha:}" ;; esac; [ "$_f" = "$_rest" ] && break; _rest="${_rest#*[,;]}"; done
    [ -n "$PIN" ] || { log "FATAL --comment '$COMMENT' carries no sha:<40hex> field"; exit 3; } ;;
  *) log "FATAL no --comment=sha:<40hex> pin on this job (got '$COMMENT')"; exit 3 ;;
esac
[ "$PIN" = "$HEAD_SHA" ] || { log "FATAL checkout $REPO is at $HEAD_SHA, the submission pinned $PIN — refusing to run unvalidated code"; exit 3; }
[ -z "$(git -C "$REPO" status --porcelain --untracked-files=no 2>/dev/null)" ] || { log "FATAL checkout $REPO has modified tracked files"; exit 3; }

# --- paths ------------------------------------------------------------------------------------------------------------
CONFIG="$REPO/configs/sdfbase_${MODEL_KEY}_fits.yaml"
[ -s "$CONFIG" ] || { log "FATAL no config at $CONFIG"; exit 3; }
POOL_KEY="$(sed -n 's#^ *heldout_pool: /vol/pool/\([^/][^/]*\)/pairs_constv2-heldout\.jsonl *$#\1#p' "$CONFIG" | head -1)"
[ -n "$POOL_KEY" ] || { log "FATAL could not read the pool dir from $CONFIG"; exit 3; }
PAIRS="$SHARED/vol/pool/$POOL_KEY/pairs_${POOL_SI}.jsonl"
FIT_ENV_NAME="${SDFBASE_FIT_ENV_NAME:-sdfbase-fit}"
VENV="${SDFBASE_FIT_ENV:-$SHARED/envs/$FIT_ENV_NAME}"
PY="$VENV/bin/python"

# --- environment (cluster rules) ----------------------------------------------------------------------------------------
source "$SHARED/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1
# NCCL's socket interface is the site's to choose (its nccl.conf, else NCCL's own detection): the job exports no
# NCCL_SOCKET_IFNAME, because an exported name overrides the site's setting and one the node lacks fails the FIRST collective
# with "Bootstrap : no socket interface found"
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
export WANDB_MODE=disabled NCCL_DEBUG=WARN

# --- preflight ---------------------------------------------------------------------------------------------------------
MODEL_REPO="$(sed -n 's/^model: //p' "$CONFIG" | head -1)"
MODEL_REV="$(sed -n 's/^model_revision: //p' "$CONFIG" | head -1)"
SNAP="$HF_HOME/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REV/config.json"
pre_ok=1
for f in "$CONFIG" "$PAIRS" "$SNAP" "$VENV/BUILD-COMPLETE"; do
  [ -s "$f" ] || { log "PREFLIGHT missing/empty: $f"; pre_ok=0; }
done
[ -x "$PY" ] || { log "PREFLIGHT no venv python at $PY"; pre_ok=0; }
NGPU="${SLURM_GPUS_ON_NODE:-}"; [ -n "$NGPU" ] || NGPU="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
[ "$NGPU" -ge "$WIDTH" ] 2>/dev/null || { log "PREFLIGHT the wide leg needs $WIDTH GPUs, this job holds $NGPU"; pre_ok=0; }
[ "$pre_ok" = 1 ] || exit 3
# audit-exempt: out-of-lockfile-venv — this job runs from the PREBUILT sdfbase-fit venv on shared storage exactly as
# sdfbase_fit_cell.sbatch does (the fit whose data-parallel path it measures; scripts/sbatch/sdfbase_fit_env.sbatch builds it
# once from the lockfile export + the hybrid-stack pins; no job ever uv-syncs), so the frozen-cu128-sync guard does not apply
# and the runtime assert below is the flavour guard: the cu128 GPU torch on a live CUDA device BEFORE the workload, else exit 92.
if ! "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__, torch.cuda.get_device_name(0), torch.cuda.device_count(), "visible")'; then
  log "FATAL: $PY is not the cu128 GPU torch — NOT running the smoke"
  exit 92
fi
log "repo=$REPO head=$HEAD_SHA config=$CONFIG pairs=$PAIRS out=$OUT_DIR env=$FIT_ENV_NAME node=$(hostname) gpus=${SLURM_JOB_GPUS:-?} ngpu=$NGPU nccl_if=${NCCL_SOCKET_IFNAME:-unset} width=$WIDTH steps=$STEPS accum=$ACCUM"

# --- RESULT + mirror on EVERY exit; disarm-first cleanup on a signal ------------------------------------------------------
T_START=$(date +%s)
RC=""
finished=0
finish() {
  [ "$finished" = 1 ] && return; finished=1
  local rc="${RC:-${1:-unknown}}" state
  case "$rc" in 0) state=DONE ;; 1) state=PARITY_FAIL ;; 143) state=INTERRUPTED ;; *) state=FAILED ;; esac
  local line="$state rc=$rc job=${SLURM_JOB_ID:-?} node=$(hostname) model=$MODEL_KEY arm=$ARM pool=$POOL_SI steps=$STEPS accum=$ACCUM width=$WIDTH elapsed_s=$(( $(date +%s) - T_START )) out=$OUT_DIR head=${HEAD_SHA:0:12} at=$(date -Is)"
  echo "$line" > "$FIGS_DIR/RESULT_${SLURM_JOB_ID:-x}"
  echo "$line" > "$FIGS_DIR/RESULT"
  echo "$line" >> "$OUT_DIR/RESULT"
  cp -f "$OUT_DIR"/parity.json "$OUT_DIR"/parity.md "$FIGS_DIR/" 2>/dev/null || true
  for w in w1 w$WIDTH; do cp -f "$OUT_DIR/$w/result.json" "$FIGS_DIR/result_$w.json" 2>/dev/null || true; done
  [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
  log "RESULT: $line"
}
trap 'finish' EXIT
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; RC=143; exit 143; }
trap cleanup SIGTERM SIGINT SIGQUIT

nvidia-smi dmon -s um -d 30 -o T > "$FIGS_DIR/dmon_${SLURM_JOB_ID:-x}.log" 2>/dev/null &
DMON_PID=$!

COMMON=(run --config "$CONFIG" --arm "$ARM" --pairs "$PAIRS" --steps "$STEPS" --accum "$ACCUM" --lr "${DDPSMOKE_LR:-1e-5}")
# --- leg 1: the reference, ONE GPU (a strict subset of the allocation, requested through Slurm) --------------------------
log "leg 1: width 1"
srun --ntasks=1 --gres=gpu:1 "$PY" "$REPO/scripts/sdfbase/ddp_parity_smoke.py" "${COMMON[@]}" --expected-width 1 --out "$OUT_DIR/w1"
RC1=$?
log "leg 1 exited rc=$RC1 after $(( $(date +%s) - T_START ))s"
[ "$RC1" = 0 ] || { RC=$RC1; exit "$RC1"; }
# --- leg 2: the wide leg, torchrun inside ONE srun task holding every GPU -------------------------------------------------
log "leg 2: width $WIDTH (torchrun --standalone --nproc_per_node=$WIDTH)"
mkdir -p "$OUT_DIR/w$WIDTH/torchrun"
# --tee 0:3 = rank 0, BOTH streams, to this .out as well as its file (a bare `--tee 0` is Std.NONE: nothing teed)
srun --ntasks=1 --gres=gpu:"$WIDTH" "$PY" -m torch.distributed.run --standalone --nproc_per_node="$WIDTH" \
     --redirects 3 --tee 0:3 --log-dir "$OUT_DIR/w$WIDTH/torchrun" \
     "$REPO/scripts/sdfbase/ddp_parity_smoke.py" "${COMMON[@]}" --expected-width "$WIDTH" --out "$OUT_DIR/w$WIDTH"
RC2=$?
log "leg 2 exited rc=$RC2 after $(( $(date +%s) - T_START ))s"
[ "$RC2" = 0 ] || { RC=$RC2; exit "$RC2"; }
# --- the comparison ------------------------------------------------------------------------------------------------------
"$PY" "$REPO/scripts/sdfbase/ddp_parity_smoke.py" compare "$OUT_DIR/w1" "$OUT_DIR/w$WIDTH" --out "$OUT_DIR"
RC=$?
log "compare exited rc=$RC (0 = PASS, 1 = parity FAIL) after $(( $(date +%s) - T_START ))s"
exit "$RC"
