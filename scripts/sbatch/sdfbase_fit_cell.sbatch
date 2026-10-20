#!/bin/bash
# scripts/sbatch/sdfbase_fit_cell.sbatch — ONE sdf-base-steering fit cell on a
# Slurm cluster (cluster A or cluster B), the Slurm twin of the Modal app's
# fit_h200_* function (2026-09-06; the user's standing preference: new fit
# work runs on Slurm). The trainer invocation, the identity gates and the
# published artefacts are the Modal lane's exactly (scripts/sdfbase/
# slurm_fit_cell.py carries the how and the why); this file is the cluster
# scaffolding around it: preemption-tolerant (checkpoint/resume through the
# trainer's own --checkpoint-dir, keyed to the fit fingerprint), self-
# resubmitting on SIGTERM with a positional attempt counter (NEVER --export;
# hard rule 1), STOP-sentinel aware, RESULT line + mirror-figs push on EXIT.
#
# Submit (positional args — never --export; --comment pins the commit the
# submitter validated, and the job REFUSES if the checkout's HEAD
# differs; -J gives the unique descriptive name the watcher and STOP
# sentinel key on; --time overrides the header for the GLM cells):
#   sbatch -J sdfbase-fullfit-nemotron-mixfull-psr --time=1-16:00:00 \
#     --comment=sha:$(git -C "$REPO" rev-parse HEAD) \
#     "$REPO/scripts/sbatch/sdfbase_fit_cell.sbatch" \
#     nemotron3nano4b_doc_full constv2-mix-full psr_ce_all newmodels_full
#   positional: <model_key> <si> <arm> <bank> [attempt] [max_attempts]
#     model_key  the fits config key: configs/sdfbase_<model_key>_fits.yaml;
#                a `_full` key's pools live under its parent's pool dir
#                (make_sdfbase_configs pool_key), and so do its further
#                siblings (`_full_r2t12`) — POOL_KEY is READ from the config's
#                train_kwargs.default.heldout_pool (/vol/pool/<POOL_KEY>/...),
#                never inferred from the key's spelling. POOL_KEY names the
#                INPUTS only.
#     bank       the objects bank: outputs land under $SHARED/exp/objects/
#                sdfbase/<bank>/<OUT_KEY>/ (cells_all/, ckpt/, reports/ — the
#                volume's out/<key>/ layout). OUT_KEY is the CONFIG's
#                identity (fit_cell_identity.out_key: the model_key, except
#                the two frozen `_full` lanes pinned at their pool-key dirs
#                in newmodels_full/), NEVER the pool key — configs sharing a
#                pool (`_full`, `_full_r2t12`) must never share a root, or a
#                sibling fit starts fresh in the frozen fit's ckpt/<cell>/
#                slot (2026-09-10). slurm_fit_cell.py re-derives and refuses
#                any other --out-root. Banks: newmodels_full, or
#                newmodels_full_gate for the identity-gate refit of an
#                existing volume cell
# The repo checkout is the one this script lives in (resolved through
# scontrol's Command= — $0 is a spool copy); SDFBASE_REPO overrides.
# Venv: $SHARED/envs/<name>, name = SDFBASE_FIT_ENV_NAME (the submitting
# shell's environment rides Slurm's implicit export — never --export=, hard
# rule 1), else the bank file $SHARED/exp/objects/sdfbase/<bank>/FIT_ENV_NAME
# (one line; durable across manual resubmits, written once per bank), else
# `sdfbase-fit` (today's path, byte for byte); SDFBASE_FIT_ENV (a full path)
# still overrides everything. A lane that needs a rebuilt stack builds
# $SHARED/envs/sdfbase-fit-<tag> (sdfbase_fit_env.sbatch <name>) and points
# its bank at it, never rebuilding the shared venv in place (O6, 2026-09-16).
# The self-resubmit carries the resolved name as an env prefix on its sbatch
# and passes this job's own StdOut path as --output, so a cluster B resubmit
# keeps /workspace/%u/exp/logs/ (the header names cluster A's root, which does
# not exist there — a resubmit without it dies opening its log while the
# queue reads healthy).
# Data-parallel width (2026-09-30): the config's top-level
# `fit_width: N` (absent = 1) is THE width. The header below is the width-1 allocation;
# a width-N submission passes `$(fit_width_flags N)` = --gres=gpu:N --cpus-per-task=16N
# --mem=128N G on the sbatch line (reproduce/04_fit_objects.sh / lib/common.sh
# repro_fit_width_flags do; the self-resubmit below does too) and SDFBASE_FIT_WIDTH=N in
# the job env as a cross-check. In the job the allocation MUST equal the width (refused
# by name otherwise) and slurm_fit_cell.py --fit-width N runs the trainer under
# torchrun --standalone --nproc_per_node=N (rank 0 publishes; every rank's log under
# <out_root>/reports/torchrun_<cell>/). Never --mem=0, never --export=.
# Stop for good: ~/.claude/skills/gpu-avail/scripts/stop-run.sh <job-name>
# (writes $SHARED/exp/training/<job-name>/STOP, then scancel) — a bare
# scancel looks like a preemption and resubmits.
#SBATCH --job-name=sdfbase-fullfit
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=1-16:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
set -uo pipefail

MODEL_KEY="${1:?positional 1: model_key (e.g. nemotron3nano4b_doc_full)}"
SI="${2:?positional 2: si (e.g. constv2-mix-full)}"
ARM="${3:?positional 3: arm (psr_ce_all | lora_r4_ce)}"
BANK="${4:?positional 4: bank (newmodels_full | newmodels_full_gate)}"
ATTEMPT="${5:-1}"
MAX_ATTEMPTS="${6:-3}"
export SDFBASE_ATTEMPT="$ATTEMPT"

# The shared root and the QoS an escalation resubmits at: the front door's job env passes SISTEER_SHARED_ROOT and
# SISTEER_CLUSTER_QOS (from the site's cluster block) on the sbatch line; without the root the job refuses by name.
if [ -n "${SISTEER_SHARED_ROOT:-}" ]; then SHARED="$SISTEER_SHARED_ROOT"; CLUSTER_QOS="${SISTEER_CLUSTER_QOS:-high}"
else echo "[fit] SISTEER_SHARED_ROOT is unset: the job env names the cluster shared root (the front door exports it from the site config)"; exit 2; fi
ESCALATE_QOS="${ESCALATE_QOS:-$CLUSTER_QOS}"   # already at high: a resubmit at the same QoS

RUN_NAME="${SLURM_JOB_NAME:-sdfbase-fullfit}"
STOP_DIR="$SHARED/exp/training/$RUN_NAME"
export FIGS_DIR="$SHARED/exp/figs/$RUN_NAME"
mkdir -p "$STOP_DIR" "$FIGS_DIR"
log() { echo "[$(date -Is)] [fit $RUN_NAME attempt $ATTEMPT/$MAX_ATTEMPTS] $*"; }

# --- the checkout + the submit-time sha pin (fail-closed) -------------------
SCRIPT_PATH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
TIMELIMIT="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p' | head -1)"
COMMENT="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Comment=\([^ ]*\).*/\1/p' | head -1)"
STDOUT_PATH="$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *StdOut=\([^ ]*\).*/\1/p' | head -1)"
REPO="${SDFBASE_REPO:-}"
if [ -z "$REPO" ] && [ -n "$SCRIPT_PATH" ]; then REPO="$(cd "$(dirname "$SCRIPT_PATH")/../.." 2>/dev/null && pwd)"; fi
[ -n "$REPO" ] && [ -f "$REPO/scripts/train_two_si_objects.py" ] || { log "FATAL cannot resolve the repo checkout (SCRIPT_PATH='$SCRIPT_PATH', SDFBASE_REPO='${SDFBASE_REPO:-}')"; exit 3; }
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
# sisteer-attempt-wait-hook: the twin-submit start grant (reproduce/lib/attempt_wait.sh) — active only under SISTEER_ATTEMPT_ID; a
# DENY exits 0 here, before any venv, checkpoint or slurm_fit_cell.py, and doubles as the escalation trap's stop (nothing resubmits)
[ -z "${SISTEER_ATTEMPT_ID:-}" ] || source "$REPO/reproduce/lib/attempt_wait.sh"

# --- paths ------------------------------------------------------------------
CONFIG="$REPO/configs/sdfbase_${MODEL_KEY}_fits.yaml"
# The pool dir is the one the CONFIG names (make_sdfbase_configs pool_key ->
# train_kwargs.default.heldout_pool = /vol/pool/<POOL_KEY>/pairs_constv2-
# heldout.jsonl): a `_full` key shares its parent's, and so does any further
# sibling of that parent (`_full_r2t12`, 2026-09-07). The former suffix strip
# (`${MODEL_KEY%_full}`) read the layout off the key's spelling and would
# have sent a sibling key to a pool dir that does not exist; read it from
# the authoritative source instead, and refuse when it does not parse.
[ -s "$CONFIG" ] || { log "FATAL no config at $CONFIG"; exit 3; }
POOL_KEY="$(sed -n 's#^ *heldout_pool: /vol/pool/\([^/][^/]*\)/pairs_constv2-heldout\.jsonl *$#\1#p' "$CONFIG" | head -1)"
[ -n "$POOL_KEY" ] || { log "FATAL could not read the pool dir from $CONFIG (train_kwargs.default.heldout_pool must be /vol/pool/<key>/pairs_constv2-heldout.jsonl)"; exit 3; }
PAIRS="$SHARED/vol/pool/$POOL_KEY/pairs_${SI}.jsonl"
HELDOUT="$SHARED/vol/pool/$POOL_KEY/pairs_constv2-heldout.jsonl"
# --- the data-parallel width: the CONFIG's top-level `fit_width:` line, absent = 1 -------------------------------
# fit_width_flags <N> -> the sbatch resource flags a width-N fit needs; EMPTY at width 1 (the header's gpu:1 / 16 CPUs / 200G
# stand). reproduce/lib/common.sh repro_fit_width_flags is the submitter's copy; a test pins the two equal.
fit_width_flags() {
  local n="${1:-1}"
  [ "$n" -gt 1 ] 2>/dev/null || return 0
  printf -- '--gres=gpu:%d --cpus-per-task=%d --mem=%dG' "$n" "$((16 * n))" "$((128 * n))"
}
FIT_WIDTH="$(sed -n 's/^fit_width: *\([0-9][0-9]*\) *$/\1/p' "$CONFIG" | head -1)"; FIT_WIDTH="${FIT_WIDTH:-1}"
[ -z "${SDFBASE_FIT_WIDTH:-}" ] || [ "$SDFBASE_FIT_WIDTH" = "$FIT_WIDTH" ] || { log "FATAL the job env says SDFBASE_FIT_WIDTH=$SDFBASE_FIT_WIDTH but $CONFIG declares fit_width $FIT_WIDTH — the submitter and the config disagree; resubmit from the config"; exit 3; }
# the allocation must BE the width: Slurm's per-node GPU count for a gres job, else the devices the cgroup shows
NGPU="${SLURM_GPUS_ON_NODE:-}"; [ -n "$NGPU" ] || NGPU="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
[ "$NGPU" = "$FIT_WIDTH" ] || { log "FATAL fit_width=$FIT_WIDTH (from $CONFIG) but this job holds $NGPU GPU(s) — submit with: sbatch $(fit_width_flags "$FIT_WIDTH") ... (04_fit_objects.sh passes it; the header alone is the width-1 allocation)"; exit 3; }
# The WRITABLE namespace is the CONFIG's, never the pool's: <bank>/<OUT_KEY>
# through the ONE derivation slurm_fit_cell.py enforces (fit_cell_identity
# out-key; stdlib-only, so the node's python3 runs it before the venv is
# touched). FROZEN_OUT_KEYS pins the two published `_full` lanes at their
# pool-key dirs; every other key is its own dir; a key whose own dir is a
# frozen lane's is refused (exit 3, reason on stderr -> this log).
OUT_KEY="$(python3 "$REPO/scripts/sdfbase/fit_cell_identity.py" out-key --bank "$BANK" --config "$CONFIG")" || OUT_KEY=""
[ -n "$OUT_KEY" ] || { log "FATAL no writable namespace for config $MODEL_KEY in bank $BANK (fit_cell_identity out-key refused; its reason is above)"; exit 3; }
OUT_ROOT="$SHARED/exp/objects/sdfbase/$BANK/$OUT_KEY"
CELL_ID="${SI}__${ARM}"
CELL_DIR="$OUT_ROOT/cells_all/$CELL_ID"
# the venv NAME: environment (Slurm's implicit export of the submitting shell) > the
# bank's FIT_ENV_NAME file > sdfbase-fit; SDFBASE_FIT_ENV (a full path) overrides the lot
FIT_ENV_NAME="${SDFBASE_FIT_ENV_NAME:-$( [ -s "$SHARED/exp/objects/sdfbase/$BANK/FIT_ENV_NAME" ] && head -1 "$SHARED/exp/objects/sdfbase/$BANK/FIT_ENV_NAME" | tr -d '[:space:]' )}"
FIT_ENV_NAME="${FIT_ENV_NAME:-sdfbase-fit}"
VENV="${SDFBASE_FIT_ENV:-$SHARED/envs/$FIT_ENV_NAME}"
PY="$VENV/bin/python"

# --- environment (cluster rules) --------------------------------------------
source "$SHARED/scripts/hf_env.sh"           # HF_HOME (shared cache) + per-user token path
export HF_HUB_OFFLINE=1                      # the snapshot is pre-fetched; never reach out
# NCCL's socket interface is the site's to choose (its nccl.conf, else NCCL's own detection): the job exports no
# NCCL_SOCKET_IFNAME, because an exported name overrides the site's setting and one the node lacks fails the FIRST collective
# with "Bootstrap : no socket interface found"
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export SISTEER_VOL_ROOT="$SHARED/vol"        # the config's /vol/... held-out path resolves here
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src"
# W&B: OFFLINE on the cluster (no key exists here, by rule); the run is
# synced from the grading host, where the entity check (sisteer.wandb_entity)
# runs before the upload. The entity is the site's own (WANDB_ENTITY, else
# the site config's SISTEER_WANDB_ENTITY), never a default: with neither in
# the job environment W&B is OFF for this job (WANDB_MODE=disabled, logged).
# Run name = the Modal lane's exact form (the trainer appends -$SLURM_JOB_ID
# otherwise), tags/config carry the Slurm provenance.
export WANDB_CONSOLE=off
export WANDB_ENTITY="${WANDB_ENTITY:-${SISTEER_WANDB_ENTITY:-}}"
export WANDB_PROJECT="${WANDB_PROJECT:-${SISTEER_WANDB_PROJECT:-sisteer-fits}}"
if [ -n "$WANDB_ENTITY" ]; then
  export WANDB_MODE=offline
else
  export WANDB_MODE=disabled
  log "W&B off: no WANDB_ENTITY / SISTEER_WANDB_ENTITY in the job environment"
fi
export WANDB_DIR="$SHARED/exp/wandb/$BANK/$OUT_KEY/$CELL_ID"
mkdir -p "$WANDB_DIR"
MODEL_SLUG="$(sed -n 's/^model: .*\///p' "$CONFIG" | head -1)"
export TT_WANDB_FIT="{\"name\": \"fit-${MODEL_SLUG}-${SI}-${ARM}-all\", \"tags\": [\"slurm\", \"$(basename "$SHARED")\", \"job${SLURM_JOB_ID}\", \"attempt${ATTEMPT}\", \"$BANK\"], \"config\": {\"slurm_job_id\": \"${SLURM_JOB_ID}\", \"slurm_node\": \"$(hostname)\", \"lane\": \"slurm\", \"bank\": \"$BANK\", \"config_key\": \"$MODEL_KEY\"}}"

# --- preflight: every external input, -s (present AND non-empty) ------------
MODEL_REPO="$(sed -n 's/^model: //p' "$CONFIG" | head -1)"
MODEL_REV="$(sed -n 's/^model_revision: //p' "$CONFIG" | head -1)"
SNAP="$HF_HOME/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REV/config.json"
pre_ok=1
for f in "$CONFIG" "$PAIRS" "$HELDOUT" "$SNAP" "$VENV/BUILD-COMPLETE" "$SHARED/scripts/mirror-figs.sh"; do
  [ -s "$f" ] || { log "PREFLIGHT missing/empty: $f"; pre_ok=0; }
done
# the sample manifest: PRESENCE only, in either spelling (<pool>.jsonl.manifest.json,
# or the stem-swapped pairs_<cell>.manifest.json the pool builder ships). The
# wrapper resolves and VALIDATES it (cell, pool_sha256) through
# sisteer.data.sample_manifest — the ONE resolver the trainer records from — so
# an empty or foreign file is ITS refusal, never skipped here for the other spelling
[ -e "$PAIRS.manifest.json" ] || [ -e "${PAIRS%.jsonl}.manifest.json" ] || { log "PREFLIGHT no sample manifest beside $PAIRS (either spelling)"; pre_ok=0; }
[ -x "$PY" ] || { log "PREFLIGHT no venv python at $PY"; pre_ok=0; }
[ -s "$PAIRS.tokcache.npz" ] || log "PREFLIGHT note: no tokcache sidecar at $PAIRS.tokcache.npz (in-process memo only; slower start, same fit)"
[ "$pre_ok" = 1 ] || exit 3

# audit-exempt: out-of-lockfile-venv — this job runs from the PREBUILT
# sdfbase-fit venv on shared storage: scripts/sbatch/sdfbase_fit_env.sbatch
# builds it ONCE (pip, from the lockfile EXPORT scripts/modal/requirements.txt
# plus the hybrid-stack pins) and ships it; no job ever uv-syncs, so the
# frozen-cu128-sync guard is factually wrong here and the 61380 silent-CPU
# path cannot occur through a sync. What CAN happen is a node where the venv
# resolves to a CPU-flavored or CUDA-less torch (a build-time check cannot
# see it; a runtime one can), and a multi-day cell would then run at CPU
# speed while the queue reads healthy — so assert the cu128 GPU torch on a
# live CUDA device BEFORE the workload and exit 92 instead; the RESULT line
# carries the FAILED rc, no self-resubmit.
if ! "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__, torch.cuda.get_device_name(0))'; then
  log "FATAL: $PY is not the cu128 GPU torch (CUDA unavailable, or a wrong-flavor venv on $(hostname)) — NOT running the workload (61380 guard)"
  exit 92
fi
log "repo=$REPO head=$HEAD_SHA config=$CONFIG ($(sha256sum "$CONFIG" | cut -c1-16)) pairs=$PAIRS pool_key=$POOL_KEY out_key=$OUT_KEY out_root=$OUT_ROOT env=$FIT_ENV_NAME ($VENV) node=$(hostname) gpu=${SLURM_JOB_GPUS:-?} fit_width=$FIT_WIDTH ngpu=$NGPU nccl_if=${NCCL_SOCKET_IFNAME:-unset} timelimit=$TIMELIMIT stdout=${STDOUT_PATH:-?}"

# --- RESULT + mirror on EVERY exit; escalation on SIGTERM -------------------
T_START=$(date +%s)
RC=""
finished=0
finish() {
  [ "$finished" = 1 ] && return; finished=1
  local rc="${RC:-${1:-unknown}}" state
  case "$rc" in 0) state=DONE ;; 143) state=INTERRUPTED ;; *) state=FAILED ;; esac
  local fp; fp="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('fit_fingerprint','')[:16])" "$CELL_DIR/COMPLETE" 2>/dev/null || echo none)"
  local line="$state rc=$rc job=${SLURM_JOB_ID:-?} node=$(hostname) cell=$OUT_KEY/$CELL_ID bank=$BANK attempt=$ATTEMPT/$MAX_ATTEMPTS elapsed_s=$(( $(date +%s) - T_START )) fingerprint=$fp head=${HEAD_SHA:0:12} at=$(date -Is)"
  echo "$line" > "$FIGS_DIR/RESULT_${SLURM_JOB_ID:-x}"
  echo "$line" > "$FIGS_DIR/RESULT"
  [ -d "$CELL_DIR" ] && echo "$line" >> "$CELL_DIR/RESULT"
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
  log "resubmitting at qos=$ESCALATE_QOS as attempt $((ATTEMPT + 1)) (time=$TIMELIMIT, output=${STDOUT_PATH:-header}, env=$FIT_ENV_NAME, fit_width=${FIT_WIDTH:-1})"
  # --output = this job's own resolved log path (appends, --open-mode=append): the header's
  # hard-coded log root is cluster A's and does not exist on cluster B. The venv name
  # rides as an env prefix (Slurm's implicit export; never --export=) so the successor runs on
  # the SAME venv whatever the bank file says by then.
  # The site's shared root + QoS ride the successor's env prefix as well (Slurm's implicit export; never --export=), so a
  # configured site's resubmit keeps ITS QoS instead of the literal `high`. On a legacy
  # two-root job SISTEER_SHARED_ROOT is passed EMPTY and the successor probes the roots exactly as before.
  # The width rides too: the successor's allocation is the width's (fit_width_flags — empty at width 1, so the header
  # stands exactly as before) and SDFBASE_FIT_WIDTH lets the successor cross-check the config it reads against the env.
  # shellcheck disable=SC2046
  SISTEER_SHARED_ROOT="${SISTEER_SHARED_ROOT:-}" SISTEER_CLUSTER_QOS="$CLUSTER_QOS" \
  SDFBASE_FIT_ENV_NAME="$FIT_ENV_NAME" SDFBASE_FIT_WIDTH="${FIT_WIDTH:-1}" \
  sbatch --qos="$ESCALATE_QOS" --job-name="$RUN_NAME" ${TIMELIMIT:+--time="$TIMELIMIT"} $(fit_width_flags "${FIT_WIDTH:-1}") \
         ${STDOUT_PATH:+--output="$STDOUT_PATH"} --comment="$COMMENT" "$SCRIPT_PATH" \
         "$MODEL_KEY" "$SI" "$ARM" "$BANK" "$((ATTEMPT + 1))" "$MAX_ATTEMPTS" \
    || log "RESUBMIT FAILED — the run will not continue automatically"
}
on_term() {
  [ "$handled" -eq 1 ] && return; handled=1
  log "SIGTERM (preemption / node drain / cancel) — draining the trainer (its checkpoint is the resume point)"
  [ -n "$SRUN_PID" ] && kill -TERM "$SRUN_PID" 2>/dev/null
  wait "$SRUN_PID" 2>/dev/null
  RC=143
  escalate
  exit 143
}
trap on_term SIGTERM SIGINT SIGQUIT

# GPU-duty record (gpu-util skill): one utilization row per minute, into
# FIGS_DIR so the mirror publishes it with the RESULT.
nvidia-smi dmon -s u -d 60 -o T > "$FIGS_DIR/dmon_${SLURM_JOB_ID:-x}.log" 2>/dev/null &
DMON_PID=$!

# --- the cell --------------------------------------------------------------
mkdir -p "$OUT_ROOT/reports"
# ONE srun task: at width > 1 the wrapper launches torchrun --standalone --nproc_per_node=$FIT_WIDTH INSIDE it (single node —
# the multi-node "torchrun is not srun-wrapped" rule does not apply; the step holds every GPU of the allocation).
srun --ntasks=1 "$PY" "$REPO/scripts/sdfbase/slurm_fit_cell.py" \
     --config "$CONFIG" --si "$SI" --arm "$ARM" --pairs "$PAIRS" \
     --out-root "$OUT_ROOT" --checkpoint-every 100 --python "$PY" --fit-width "$FIT_WIDTH" &
SRUN_PID=$!
wait "$SRUN_PID"
RC=$?
log "cell exited rc=$RC after $(( $(date +%s) - T_START ))s"
# A non-zero rc that is not a signal death is a real failure: no resubmit
# (a retry would fail the same way); the RESULT line + watcher carry it.
exit "$RC"
