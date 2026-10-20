#!/bin/bash
# scripts/sbatch/sdfbase_derived_build.sbatch — the round-2 DERIVED OBJECTS of one (model, corpus,
# control) pair on a Slurm cluster: the Slurm twin of the derived Modal app
# (scripts/modal/sdfbase_derived.py, deployed as sdfbase-derived-6708221d4c2df3f0 for the gemma /
# qwen pairs on 2026-09-04/06; the user's standing preference: new fit work runs on Slurm). The
# COMMANDS are the app's command builders' EXACTLY (lora_subtract_command, compose_command,
# calibrate_command, merge_command, distill_command, liveness_command — same flags, same budget
# 60 / 5, same capture budget 100k tokens, same checkpoint cadence 50, same 3-point lr-mult sweep),
# with the volume's paths mapped onto the cluster's shared storage:
#   /vol/out            -> $SHARED/exp/objects/sdfbase/<bank>          (the fits bank: <key>/cells_all/)
#   /vol/out/derived    -> $SHARED/exp/objects/sdfbase/<bank>/derived  (<kind>/<key>/<derived id>/)
#   /vol/pool/<key>     -> $SHARED/vol/pool/<key>
# and the fan-out expressed as Slurm jobs: stage `prep` (one job), stage `calibrate <m>` (a job
# ARRAY, one task per multiplier), stage `assess` (one job, --dependency=afterok on the array; the
# selection, recorded durably, WITHOUT distilling), stage `finish` (one job, afterok on assess).
# What has no Slurm analogue: W&B (--wandb needs the operator's key, which never lands on a cluster —
# the fit records live in report.json / derived_meta.json / calibration_controls.json, which is what
# the wave reads), the volume commit thread (shared storage is durable on return).
#
# Stages (positional; NEVER --export, hard rule 1):
#   <stage> <model_key> <mix_si> <ctl_si> <bank> [lr_mult|-] [attempt] [max_attempts] [objects]
#   objects    relu | lora | both (default both): the derived objects this chain builds and so the parents its preflight requires —
#              relu = the two psr_ce_all cells (V + the distilled residual; calibrate/assess/finish are ReLU work), lora = the two
#              lora_r4_ce cells (the task-vector subtraction, prep only), both = all four (every existing invocation, byte for byte).
#              prep writes build_provenance.json beside each object it builds (objects; THAT output's parents with their COMPLETE
#              sha/fingerprint — the psr pair for V and the residual, the LoRA pair for the subtraction).
#   prep       THE difference_relu build (since 2026-10-02 the difference object is the DIRECT paired
#              difference V = +P_mix - P_ctl, kind composed / paper label difference_relu — the undistilled
#              paired write; there is no distillation in the method). A 1-GPU allocation (minutes):
#              lora_subtract.py --keep-valid (the LoRA task-vector subtraction, objects lora|both), then
#              psr_residual_distill.py compose-v (V, persisted ONCE; its native registry emission rides it),
#              verify-v, V's gate_liveness.py (seedrep/derived_admission.py refuses a paired write without
#              gate_liveness.json; kept when gate_liveness_receipt.json binds it to V's bytes, re-probed
#              otherwise), the receipt, then emit-v (V's registry record, filed ONLY after the probe)
#              and, as LEGACY scaffolding the three stages below still read under
#              build=chain (positional 10; the front door passes `direct`), capture-set + the ladder
#              DECLARATION <resid>/sweep.json
#   calibrate / assess / finish — LEGACY (2026-10-02): the psr_residual distillation chain
#              (relu_steer_distilled_difference, paper label difference_relu_distilled, in_paper 0 for every
#              model). The front door (04_fit_objects.sh difference_relu) never composes these stages and
#              refuses to advance while one runs; they remain callable for the seedrep chain wrapper
#              (scripts/sdfbase/seedrep/submit_derived_chain.sh) until that lane retires them.
#   calibrate  ONE multiplier's calibration controls (GPU): calibrate --lr-mult <m> --out
#              <resid>/calib_parts/lrmult<m>/calibration_controls.json  (lr_mult from arg 6, or from
#              SLURM_ARRAY_TASK_ID over the sweep 0.3,1,3 when arg 6 is `-`). An EXPLICIT multiplier
#              first runs the idempotent "ensure declared" on sweep.json: a base rung or the already
#              declared extension is a no-op (so an interrupted extension job's self-resubmit passes
#              and resumes its part), an undeclared rung is appended as THE extension, a second
#              extension REFUSES (one rung on an edge, never two)
#   assess     merge-calibration over EXACTLY sweep.json's lr_mults ∪ extensions (any extra or
#              missing part refuses) -> <resid>/calibration_controls.json, then <resid>/assess.json
#              (selected_lr_mult, ladder, on_edge, edge_side, recommended_extension 0.1|10|null, the
#              sha256 of every part and of the merged file). Never distills. On an edge the operator
#              submits ONE `calibrate … <0.1|10>` + a second `assess`, else `finish` directly
#   finish     REQUIRES assess.json and refuses if any part or the merged file no longer hashes to
#              what assess.json recorded, or if the merged selection differs from the assessed one ->
#              distill at the assessed multiplier -> derived_meta.json re-checked against assess.json
#              -> gate_liveness.py on V and the distilled residual (n_probe 32, the mix-full pool)
# Seed: --seed is READ from the parents' config (train_kwargs.default.seed; default 0 = every
# existing key's behaviour, byte for byte) and passed with --holdout-frac 0.1 to capture-set,
# calibrate and distill, so a seed replicate's capture subsample / capture holdout / planted
# controls follow its seed (qwen-base-seed-replicates, 2026-09-15); the RESULT line names seed=.
# Venv: $SHARED/envs/<name>, name = SDFBASE_FIT_ENV_NAME (the submitting shell's environment rides
# Slurm's implicit export — never --export=), else the bank file $SHARED/exp/objects/sdfbase/<bank>/
# FIT_ENV_NAME (one line; durable across manual resubmits), else `sdfbase-fit` (today's path);
# SDFBASE_FIT_ENV (a full path) still overrides everything. The self-resubmit carries the resolved
# name as an env prefix on its sbatch.
# Preemption-tolerant: every fit takes --checkpoint-every 50 (identity-keyed trainer checkpoints under
# <resid>/ckpt/, resumed by default), finished multipliers are reused by the CLI; on SIGTERM the stage
# resubmits ITSELF at high with the attempt counter bumped (positional), unless a STOP sentinel sits at
# $SHARED/exp/training/<job-name>/STOP or the cap is reached — passing the job's own StdOut path as
# --output so a cluster B resubmit keeps /workspace/%u/exp/logs/ (the header names cluster A's
# root). RESULT line + mirror-figs on every exit.
#
# Submit (the checkout is resolved from scontrol's Command=; --comment pins the commit and the job
# REFUSES if the checkout's HEAD differs or is dirty — the fit lane's pin rule; the job env names the shared root):
#   R=<the cluster checkout>; SHA=$(git -C $R rev-parse HEAD); export SISTEER_SHARED_ROOT=<the cluster shared root>
#   P=$(sbatch --parsable -J sdfbase-derived-nemo-prep --time=0:40:00 --comment=sha:$SHA \
#         $R/scripts/sbatch/sdfbase_derived_build.sbatch prep nemotron3nano4b_doc constv2-mix-full constv2ctl2-full newmodels_full)
#   A=$(sbatch --parsable -J sdfbase-derived-nemo-calib --array=0-2 --dependency=afterok:$P --time=6:00:00 --comment=sha:$SHA \
#         $R/scripts/sbatch/sdfbase_derived_build.sbatch calibrate nemotron3nano4b_doc constv2-mix-full constv2ctl2-full newmodels_full -)
#   S=$(sbatch --parsable -J sdfbase-derived-nemo-assess --dependency=afterok:$A --time=0:20:00 --comment=sha:$SHA \
#         $R/scripts/sbatch/sdfbase_derived_build.sbatch assess nemotron3nano4b_doc constv2-mix-full constv2ctl2-full newmodels_full)
#   sbatch -J sdfbase-derived-nemo-finish --dependency=afterok:$S --time=4:00:00 --comment=sha:$SHA \
#         $R/scripts/sbatch/sdfbase_derived_build.sbatch finish nemotron3nano4b_doc constv2-mix-full constv2ctl2-full newmodels_full
# (scripts/sdfbase/seedrep/submit_derived_chain.sh issues exactly this chain, one sbatch at a time.)
# Stop for good: ~/.claude/skills/gpu-avail/scripts/stop-run.sh <job-name> (STOP sentinel, then scancel).
#SBATCH --job-name=sdfbase-derived
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0-06:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@180
set -uo pipefail

STAGE="${1:?positional 1: stage (prep | calibrate | assess | finish)}"
MODEL_KEY="${2:?positional 2: model_key (the fits bank key, e.g. nemotron3nano4b_doc)}"
MIX_SI="${3:?positional 3: mix_si (e.g. constv2-mix-full)}"
CTL_SI="${4:?positional 4: ctl_si (e.g. constv2ctl2-full)}"
BANK="${5:?positional 5: bank (e.g. newmodels_full)}"
LR_MULT_ARG="${6:--}"
ATTEMPT="${7:-1}"
MAX_ATTEMPTS="${8:-3}"
# positional 9: WHICH derived objects this chain builds, and so which
# parents it requires — `relu` (the two psr_ce_all cells: V + the distilled residual), `lora` (the two lora_r4_ce cells: the task-vector
# subtraction, prep only), `both` (all four; the default = every existing invocation, byte for byte). The ReLU residual path never reads
# the LoRA subtraction (psr_residual_distill composes V from the psr cells), so a ReLU-only build is a sibling of the LoRA one, not a re-run.
OBJECTS="${9:-both}"
case "$OBJECTS" in relu|lora|both) ;; *) echo "[derived] FATAL positional 9 objects must be relu | lora | both (got '$OBJECTS')"; exit 2 ;; esac
# positional 10: WHICH build prep runs — `chain` (the default = every existing invocation: compose-v,
# verify-v, the legacy psr_residual capture-set + ladder declaration the calibrate/assess/finish stages read) or `direct` (the front
# door's difference_relu unit: V only — compose-v with its record DEFERRED, verify-v, the liveness probe + its receipt, then emit-v;
# no legacy scaffolding, so nothing avoidable can fail or cost before the probe)
BUILD="${10:-chain}"
case "$BUILD" in chain|direct) ;; *) echo "[derived] FATAL positional 10 build must be chain | direct (got '$BUILD')"; exit 2 ;; esac
derived_parents() {   # <objects> <mix_si> <ctl_si> -> the parent cells the build requires, one per line
  local o="$1" mix="$2" ctl="$3"
  case "$o" in
    relu) printf '%s\n' "${mix}__psr_ce_all" "${ctl}__psr_ce_all" ;;
    lora) printf '%s\n' "${mix}__lora_r4_ce" "${ctl}__lora_r4_ce" ;;
    both) printf '%s\n' "${mix}__psr_ce_all" "${ctl}__psr_ce_all" "${mix}__lora_r4_ce" "${ctl}__lora_r4_ce" ;;
    *) return 2 ;;
  esac
}
builds_relu() { [ "$1" = relu ] || [ "$1" = both ]; }
builds_lora() { [ "$1" = lora ] || [ "$1" = both ]; }
SWEEP=(0.3 1 3)                       # DEFAULT_LR_MULTS, in psr_residual_distill.format_mult's spelling
EPOCHS_CAP=60; PATIENCE=5             # the distillation budget (the app's defaults)
CKPT_EVERY=50                         # sdfbase_derived.DEFAULT_CHECKPOINT_EVERY
CAPTURE_TOKENS=100000                 # the deploy stamp's capture_budget_tokens (derived_full_*.sh)
N_PROBE=32                            # the liveness stage's --n-probe (derived_full_gemma.sh)
HOLDOUT_FRAC=0.1                      # psr_residual_distill's --holdout-frac default, passed explicitly

case "$STAGE" in prep|calibrate|assess|finish) ;; *) echo "[derived] FATAL unknown stage '$STAGE'"; exit 2 ;; esac
if [ "$STAGE" != prep ] && ! builds_relu "$OBJECTS"; then
  echo "[derived] FATAL stage $STAGE is ReLU-residual work; objects='$OBJECTS' builds none (the LoRA subtraction is prep-only)"; exit 2
fi

# the shared root and the QoS an escalation resubmits at: the submission's job env (SISTEER_SHARED_ROOT, SISTEER_CLUSTER_QOS; the
# front door and submit_derived_chain.sh pass both from the site's cluster block); refused by name without the root
if [ -n "${SISTEER_SHARED_ROOT:-}" ]; then SHARED="$SISTEER_SHARED_ROOT"; CLUSTER_QOS="${SISTEER_CLUSTER_QOS:-high}"
else echo "[derived] SISTEER_SHARED_ROOT is unset: the job env names the cluster shared root (the front door exports it from the site config)"; exit 2; fi
ESCALATE_QOS="${ESCALATE_QOS:-$CLUSTER_QOS}"

RUN_NAME="${SLURM_JOB_NAME:-sdfbase-derived}"
STOP_DIR="$SHARED/exp/training/$RUN_NAME"
export FIGS_DIR="$SHARED/exp/figs/$RUN_NAME"
mkdir -p "$STOP_DIR" "$FIGS_DIR"
TASK="${SLURM_ARRAY_TASK_ID:-}"
log() { echo "[$(date -Is)] [derived $RUN_NAME${TASK:+[$TASK]} $STAGE attempt $ATTEMPT/$MAX_ATTEMPTS] $*"; }

# --- the checkout + the submit-time sha pin (fail-closed; the fit lane's rule) ------------------
JOBQ="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"
SCRIPT_PATH="$(scontrol show job "$JOBQ" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
TIMELIMIT="$(scontrol show job "$JOBQ" 2>/dev/null | sed -n 's/.*TimeLimit=\([^ ]*\).*/\1/p' | head -1)"
COMMENT="$(scontrol show job "$JOBQ" 2>/dev/null | sed -n 's/^ *Comment=\([^ ]*\).*/\1/p' | head -1)"
# this job's OWN log path (resolved), carried to a self-resubmit as --output: the header's
# $SHARED/exp/logs/ does not exist on cluster B, and a resubmit without it dies
# opening its log while the queue reads healthy
STDOUT_PATH="$(scontrol show job "${SLURM_JOB_ID:-$JOBQ}" 2>/dev/null | sed -n 's/^ *StdOut=\([^ ]*\).*/\1/p' | head -1)"
REPO="${SDFBASE_REPO:-}"
if [ -z "$REPO" ] && [ -n "$SCRIPT_PATH" ]; then REPO="$(cd "$(dirname "$SCRIPT_PATH")/../.." 2>/dev/null && pwd)"; fi
[ -n "$REPO" ] && [ -f "$REPO/scripts/sdfbase/psr_residual_distill.py" ] || { log "FATAL cannot resolve the repo checkout (SCRIPT_PATH='$SCRIPT_PATH')"; exit 3; }
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

# --- paths (the volume layout, mapped) -----------------------------------------------------------
CONFIG="$REPO/configs/sdfbase_${MODEL_KEY}_full_fits.yaml"     # the PARENTS' config: the full-scale lane's hp
[ -s "$CONFIG" ] || CONFIG="$REPO/configs/sdfbase_${MODEL_KEY}_fits.yaml"
POOL_KEY="$(sed -n 's#^ *heldout_pool: /vol/pool/\([^/][^/]*\)/pairs_constv2-heldout\.jsonl *$#\1#p' "$CONFIG" | head -1)"
[ -n "$POOL_KEY" ] || { log "FATAL could not read the pool dir from $CONFIG"; exit 3; }
# The fit seed (train_kwargs.default.seed; the generator emits the line only for a declared seed,
# so every existing config reads back 0 = psr_residual_distill's own default)
SEED="$(sed -n 's/^ *seed: *\([0-9][0-9]*\) *$/\1/p' "$CONFIG" | head -1)"; SEED="${SEED:-0}"
SEED_ARGS=(--seed "$SEED" --holdout-frac "$HOLDOUT_FRAC")
POOLS="$SHARED/vol/pool/$POOL_KEY"
ROOT="$SHARED/exp/objects/sdfbase/$BANK"            # <root>/<key>/cells_all/<cell>/   (= /vol/out)
DERIVED="$ROOT/derived"                             # <kind>/<key>/<derived id>/       (= /vol/out/derived)
RESID="$DERIVED/psr_residual/$POOL_KEY/${MIX_SI}__psr_ce_all_minus_ctl_distilled"
VDIR="$DERIVED/psr_paired_write/$POOL_KEY/${MIX_SI}__psr_ce_all_minus_ctl"
LORA="$DERIVED/lora_subtraction/$POOL_KEY/${MIX_SI}__lora_r4_ce_minus_ctl"
# the venv NAME: environment (Slurm's implicit export of the submitting shell) > the bank's
# FIT_ENV_NAME file > sdfbase-fit; SDFBASE_FIT_ENV (a full path) overrides the lot
FIT_ENV_NAME="${SDFBASE_FIT_ENV_NAME:-$( [ -s "$ROOT/FIT_ENV_NAME" ] && head -1 "$ROOT/FIT_ENV_NAME" | tr -d '[:space:]' )}"
FIT_ENV_NAME="${FIT_ENV_NAME:-sdfbase-fit}"
VENV="${SDFBASE_FIT_ENV:-$SHARED/envs/$FIT_ENV_NAME}"
PY="$VENV/bin/python"
SWEEP_TOOL="$REPO/scripts/sdfbase/seedrep/derived_sweep.py"   # stdlib; runs under the node's python3
MODEL_REPO="$(sed -n 's/^model: //p' "$CONFIG" | head -1)"
MODEL_REV="$(sed -n 's/^model_revision: //p' "$CONFIG" | head -1)"

# --- environment (cluster rules) ----------------------------------------------------------------
source "$SHARED/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export SISTEER_VOL_ROOT="$SHARED/vol"
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
export PYTHONPATH="$REPO/src:$REPO/scripts/sdfbase"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off

# --- preflight -----------------------------------------------------------------------------------
SNAP="$HF_HOME/hub/models--${MODEL_REPO//\//--}/snapshots/$MODEL_REV/config.json"
pre_ok=1
for f in "$CONFIG" "$SNAP" "$VENV/BUILD-COMPLETE" "$SHARED/scripts/mirror-figs.sh" "$SWEEP_TOOL" \
         "$POOLS/pairs_${MIX_SI}.jsonl" "$POOLS/pairs_${CTL_SI}.jsonl" \
         "$POOLS/pairs_${MIX_SI}.manifest.json" "$POOLS/pairs_${CTL_SI}.manifest.json" \
         "$ROOT/$POOL_KEY/cells_all/_store_identity.json"; do
  [ -s "$f" ] || { log "PREFLIGHT missing/empty: $f"; pre_ok=0; }
done
PARENTS=(); while IFS= read -r c; do PARENTS+=("$c"); done < <(derived_parents "$OBJECTS" "$MIX_SI" "$CTL_SI")
for c in "${PARENTS[@]}"; do
  for f in object.pt COMPLETE train_budget_meta.json; do
    [ -s "$ROOT/$POOL_KEY/cells_all/$c/$f" ] || { log "PREFLIGHT parent $c lacks $f (objects=$OBJECTS)"; pre_ok=0; }
  done
done
[ -x "$PY" ] || { log "PREFLIGHT no venv python at $PY"; pre_ok=0; }
[ "$MODEL_KEY" = "$POOL_KEY" ] || log "note: model_key $MODEL_KEY resolves to pool/bank key $POOL_KEY (the _full lane shares its parent's pool dir)"
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
# live CUDA device BEFORE the workload and exit 92 instead (the RESULT line
# carries the FAILED rc, no self-resubmit).
if ! "$PY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__, torch.cuda.get_device_name(0))'; then
  log "FATAL: $PY is not the cu128 GPU torch (CUDA unavailable, or a wrong-flavor venv on $(hostname)) — NOT running the workload (61380 guard)"
  exit 92
fi
case "$STAGE" in
  calibrate)
    if [ "$LR_MULT_ARG" = "-" ]; then
      [ -n "$TASK" ] || { log "FATAL calibrate with lr_mult '-' needs SLURM_ARRAY_TASK_ID"; exit 3; }
      LR_MULT="${SWEEP[$TASK]:?array task $TASK is outside the sweep ${SWEEP[*]}}"
    else LR_MULT="$LR_MULT_ARG"; fi ;;
  *) LR_MULT="" ;;
esac
log "objects=$OBJECTS parents=${PARENTS[*]} repo=$REPO head=$HEAD_SHA config=$CONFIG ($(sha256sum "$CONFIG" | cut -c1-16)) seed=$SEED pools=$POOLS root=$ROOT derived=$DERIVED model=$MODEL_REPO@${MODEL_REV:0:12} env=$FIT_ENV_NAME ($VENV) node=$(hostname) gpu=${SLURM_JOB_GPUS:-?} timelimit=$TIMELIMIT stdout=${STDOUT_PATH:-?} lr_mult=${LR_MULT:-n/a}"

# --- RESULT + mirror on EVERY exit; escalation on SIGTERM ---------------------------------------
T_START=$(date +%s); RC=""; finished=0
finish_hook() {
  [ "$finished" = 1 ] && return; finished=1
  local rc="${RC:-${1:-unknown}}" state
  case "$rc" in 0) state=DONE ;; 143) state=INTERRUPTED ;; *) state=FAILED ;; esac
  local line="$state rc=$rc stage=$STAGE objects=$OBJECTS build=$BUILD${LR_MULT:+ lr_mult=$LR_MULT} job=${SLURM_JOB_ID:-?}${TASK:+ task=$TASK} node=$(hostname) pair=$POOL_KEY/${MIX_SI}-${CTL_SI} seed=$SEED bank=$BANK attempt=$ATTEMPT/$MAX_ATTEMPTS elapsed_s=$(( $(date +%s) - T_START )) head=${HEAD_SHA:0:12} at=$(date -Is)"
  echo "$line" >> "$FIGS_DIR/RESULT"
  echo "$line" > "$FIGS_DIR/RESULT_${SLURM_JOB_ID:-x}${TASK:+_$TASK}"
  [ -n "${DMON_PID:-}" ] && kill "$DMON_PID" 2>/dev/null
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" 2>/dev/null || true
  log "RESULT: $line"
}
trap 'finish_hook' EXIT
CHILD=""; handled=0
escalate() {
  [ "$ESCALATE_QOS" = "none" ] && { log "escalation disabled"; return; }
  [ -e "$STOP_DIR/STOP" ] && { log "STOP sentinel present — deliberate stop, not resubmitting."; return; }
  [ "$ATTEMPT" -ge "$MAX_ATTEMPTS" ] && { log "attempt cap $MAX_ATTEMPTS reached — not resubmitting."; return; }
  { [ -n "$SCRIPT_PATH" ] && [ -f "$SCRIPT_PATH" ]; } || { log "cannot resolve script path — not resubmitting."; return; }
  # a resubmitted calibrate task carries ITS multiplier explicitly (no array on the retry)
  local mult_arg="${LR_MULT:--}"
  log "resubmitting at qos=$ESCALATE_QOS as attempt $((ATTEMPT + 1)) (time=$TIMELIMIT, stage $STAGE${LR_MULT:+ lr_mult $LR_MULT}, output=${STDOUT_PATH:-header}, env=$FIT_ENV_NAME)"
  # the venv name rides as an env prefix (Slurm's implicit export; never --export=) so the
  # successor runs on the SAME venv whatever the bank file says by then
  SDFBASE_FIT_ENV_NAME="$FIT_ENV_NAME" \
  sbatch --qos="$ESCALATE_QOS" --job-name="$RUN_NAME" ${TIMELIMIT:+--time="$TIMELIMIT"} ${STDOUT_PATH:+--output="$STDOUT_PATH"} --comment="$COMMENT" \
         "$SCRIPT_PATH" "$STAGE" "$MODEL_KEY" "$MIX_SI" "$CTL_SI" "$BANK" "$mult_arg" "$((ATTEMPT + 1))" "$MAX_ATTEMPTS" "$OBJECTS" "$BUILD" \
    || log "RESUBMIT FAILED — the stage will not continue automatically"
}
on_term() {
  [ "$handled" -eq 1 ] && return; handled=1
  log "SIGTERM (preemption / node drain / cancel) — draining (the trainer's checkpoint is the resume point)"
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null; wait "$CHILD" 2>/dev/null
  RC=143; escalate; exit 143
}
trap on_term SIGTERM SIGINT SIGQUIT
nvidia-smi dmon -s u -d 60 -o T > "$FIGS_DIR/dmon_${SLURM_JOB_ID:-x}${TASK:+_$TASK}.log" 2>/dev/null &
DMON_PID=$!

# --- the commands: the derived app's builders, path-mapped ---------------------------------------
run() {   # run one command as the tracked child; return its rc
  log "RUN: $*"
  "$@" & CHILD=$!; wait "$CHILD"; local rc=$?; CHILD=""
  log "rc=$rc"; return $rc
}
RESID_BASE=("$PY" "$REPO/scripts/sdfbase/psr_residual_distill.py")
RESID_COMMON=(--root "$ROOT" --out-root "$DERIVED" --pools "$POOLS" --model-key "$MODEL_KEY"
              --mix-si "$MIX_SI" --ctl-si "$CTL_SI")
RESID_MODEL=(--model "$MODEL_REPO" --revision "$MODEL_REV" --config "$CONFIG"
             --capture-budget-tokens "$CAPTURE_TOKENS")
BUDGET=(--distill-epochs-cap "$EPOCHS_CAP" --distill-patience "$PATIENCE" --checkpoint-every "$CKPT_EVERY")
mkdir -p "$DERIVED"
case "$STAGE" in
  prep)
    # build provenance: WHICH objects this chain builds and FROM WHICH parents (their COMPLETE markers' object sha + fit fingerprint),
    # so a ReLU-only build and a later LoRA-only build read as siblings of one pair, never as a re-run of each other
    write_build_provenance() {   # <dir> <parent cell>... — the parents of THIS output (never the chain's whole list)
      local out_dir="$1"; shift
      mkdir -p "$out_dir"
      "$PY" - "$out_dir/build_provenance.json" "$OBJECTS" "$STAGE" "$HEAD_SHA" "${SLURM_JOB_ID:-}" "$ROOT/$POOL_KEY/cells_all" "$@" <<'PYP'
import json, sys, datetime
out, objects, stage, sha, job, cells, *parents = sys.argv[1:]
rows = []
for c in parents:
    m = {}
    try:
        m = json.load(open(f"{cells}/{c}/COMPLETE"))
    except Exception as e:  # noqa: BLE001
        m = {"error": f"{type(e).__name__}: {e}"}
    rows.append({"cell": c, "path": f"{cells}/{c}", "object_sha256": m.get("object_sha256"), "fit_fingerprint": m.get("fit_fingerprint")})
json.dump({"objects": objects, "relu_only": objects == "relu", "lora_only": objects == "lora", "stage": stage, "code_sha": sha,
           "slurm_job_id": job or None, "parents": rows, "written_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
           "ruling": "per-object parents through the front door (USER RULING 2026-10-01); the ReLU residual path never reads the LoRA subtraction"},
          open(out, "w"), indent=1, sort_keys=True)
PYP
    }
    if builds_lora "$OBJECTS"; then
      # lora_subtract_command: the LoRA task-vector subtraction (CPU, parents admitted by the shared gate)
      run "$PY" "$REPO/scripts/sdfbase/lora_subtract.py" --root "$ROOT" --out-root "$DERIVED/lora_subtraction" \
          --models "$MODEL_KEY" --cells "$MIX_SI" --ctl-si "$CTL_SI" --pools "$POOLS" --keep-valid || { RC=11; exit 11; }
      [ -s "$LORA/object.pt" ] && [ -s "$LORA/derived_meta.json" ] || { log "FATAL lora subtraction wrote no object at $LORA"; RC=11; exit 11; }
      write_build_provenance "$LORA" "${MIX_SI}__lora_r4_ce" "${CTL_SI}__lora_r4_ce"          # the LoRA pair only
    else
      log "prep: objects=$OBJECTS — the LoRA task-vector subtraction is NOT part of this build (no LoRA parents required)"
    fi
    if ! builds_relu "$OBJECTS"; then log "prep: objects=$OBJECTS — no ReLU residual work; done"; RC=0; exit 0; fi
    # compose_command (V once; its registry record DEFERRED to emit-v below: compose-v used to file the
    # composed row long before the liveness probe, so the row alone could satisfy the front door's done-ness
    # while the admission refused the very same V) -> verify_v_command
    if [ -s "$VDIR/object.pt" ]; then
      if ! run "${RESID_BASE[@]}" verify-v "${RESID_COMMON[@]}"; then
        # an ATTESTED V (fit_job_provenance.json beside it) that fails verification is a finding, never a silent rebuild: recomposing
        # would drop the attestation and the emission evidence, and emit-v would then state this job's context as V's provenance
        [ -f "$VDIR/fit_job_provenance.json" ] && { log "FATAL persisted V at $VDIR does not verify but carries fit_job_provenance.json — not recomposing an attested object; investigate"; RC=12; exit 12; }
        log "persisted V does not verify — recomposing (object.pt + derived_meta.json only; other sidecars kept)"
        rm -f "$VDIR/object.pt" "$VDIR/derived_meta.json"
      fi
    fi
    [ -s "$VDIR/object.pt" ] || run "${RESID_BASE[@]}" compose-v "${RESID_COMMON[@]}" --no-emit || { RC=12; exit 12; }
    run "${RESID_BASE[@]}" verify-v "${RESID_COMMON[@]}" || { RC=12; exit 12; }
    write_build_provenance "$VDIR" "${MIX_SI}__psr_ce_all" "${CTL_SI}__psr_ce_all"            # the psr pair only
    if [ "$BUILD" = direct ]; then
      log "prep: build=direct — the legacy psr_residual capture-set + ladder declaration are not part of the direct-object build (nothing avoidable runs before the probe)"
    else
      # the LEGACY scaffolding the calibrate/assess/finish stages still read: the capture manifest + the ladder DECLARATION the
      # merge will cover exactly (idempotent; a different ladder refuses)
      mkdir -p "$RESID"
      run "${RESID_BASE[@]}" capture-set "${RESID_COMMON[@]}" "${SEED_ARGS[@]}" --capture-budget-tokens "$CAPTURE_TOKENS" \
          --out "$RESID/capture_manifest.json" || { RC=13; exit 13; }
      run python3 "$SWEEP_TOOL" declare --resid "$RESID" --lr-mults "$(IFS=,; echo "${SWEEP[*]}")" --declared-by prep || { RC=13; exit 13; }
      [ -s "$RESID/sweep.json" ] || { log "FATAL prep wrote no sweep.json at $RESID"; RC=13; exit 13; }
      write_build_provenance "$RESID" "${MIX_SI}__psr_ce_all" "${CTL_SI}__psr_ce_all"
    fi
    # the paired write V IS the difference_relu object (the undistilled direct difference, kind composed; no
    # distillation in the current method). Its liveness gate belongs to prep: seedrep/derived_admission.py's psr_paired_write
    # branch REFUSES a V without gate_liveness.json. The probe is a property of V's BYTES, so its result is BOUND to them by
    # gate_liveness_receipt.json (object sha + probe-file sha; what 04_fit_objects.sh's done-ness reads): kept when the receipt names
    # the bytes on disk, re-probed when absent or bound to other bytes.
    # BOTH bindings are checked against the files as they are now: the receipt's object_sha256 against V's
    # bytes AND its gate_liveness_sha256 against the probe file; a mismatch on either re-probes (a replaced V, an edited/stale probe)
    V_SHA="$(sha256sum "$VDIR/object.pt" | cut -c1-64)"
    GL_SHA="$( [ -s "$VDIR/gate_liveness.json" ] && sha256sum "$VDIR/gate_liveness.json" | cut -c1-64 || true)"
    read -r RECEIPT_OBJ RECEIPT_GL <<<"$(python3 -c 'import json, os, sys
p = sys.argv[1]
d = json.load(open(p)) if os.path.isfile(p) else {}
print((d.get("object_sha256") or "-"), (d.get("gate_liveness_sha256") or "-"))' "$VDIR/gate_liveness_receipt.json" 2>/dev/null || echo "- -")"
    if [ -s "$VDIR/gate_liveness.json" ] && [ -n "$V_SHA" ] && [ "$RECEIPT_OBJ" = "$V_SHA" ] && [ "$RECEIPT_GL" = "$GL_SHA" ]; then
      log "prep: $VDIR/gate_liveness.json present and bound to V's bytes (${V_SHA:0:16}) and to itself (${GL_SHA:0:16}) — liveness kept"
    else
      [ -s "$VDIR/gate_liveness.json" ] && log "prep: $VDIR/gate_liveness.json present but not bound as it stands (receipt object '${RECEIPT_OBJ:0:16}' vs V ${V_SHA:0:16}; receipt probe '${RECEIPT_GL:0:16}' vs file ${GL_SHA:0:16}) — re-probing"
      run "$PY" "$REPO/scripts/sdfbase/gate_liveness.py" --model "$MODEL_REPO" --revision "$MODEL_REV" \
          --pairs "$POOLS/pairs_${MIX_SI}.jsonl" --n-probe "$N_PROBE" --dirs "$VDIR" --max-len 8192 || { RC=17; exit 17; }
      [ -s "$VDIR/gate_liveness.json" ] || { log "FATAL prep liveness wrote no gate_liveness.json at $VDIR"; RC=17; exit 17; }
      run "${RESID_BASE[@]}" liveness-receipt "${RESID_COMMON[@]}" --writer "scripts/sbatch/sdfbase_derived_build.sbatch prep" \
          --slurm-job-id "${SLURM_JOB_ID:-}" --node "$(hostname)" --code-sha "$HEAD_SHA" --n-probe "$N_PROBE" \
          --pairs "$POOLS/pairs_${MIX_SI}.jsonl" || { RC=17; exit 17; }
      [ -s "$VDIR/gate_liveness_receipt.json" ] || { log "FATAL prep wrote no gate_liveness_receipt.json at $VDIR"; RC=17; exit 17; }
    fi
    # V's registry record — ONLY now, with the liveness bound to its bytes AND not fully dead by the admission's own criterion
    # (emit-v refuses on either: a dead V FAILS here visibly, unpublished); the kept-object replay is
    # idempotent under the current campaign + emission contract (a terminal record is verified, never re-filed)
    run "${RESID_BASE[@]}" emit-v "${RESID_COMMON[@]}" || { log "FATAL emit-v refused (liveness not bound to V's bytes, or a fully dead layer — see above); V stays UNPUBLISHED"; RC=19; exit 19; }
    RC=0 ;;
  calibrate)
    # an EXPLICIT multiplier is an extension declaration (or the no-op re-declaration of a rung
    # already in the ladder — the self-resubmit path); the array indexes the base ladder
    if [ "$LR_MULT_ARG" != "-" ]; then
      run python3 "$SWEEP_TOOL" ensure-extension --resid "$RESID" --lr-mult "$LR_MULT" \
          --default-ladder "$(IFS=,; echo "${SWEEP[*]}")" || { log "FATAL extension $LR_MULT refused by the declaration (above)"; RC=18; exit 18; }
    fi
    # calibrate_command: ONE multiplier's controls into its part file (resumable; finished parts reused)
    PART="$RESID/calib_parts/lrmult${LR_MULT}/calibration_controls.json"
    mkdir -p "$(dirname "$PART")"
    run "${RESID_BASE[@]}" calibrate "${RESID_COMMON[@]}" "${RESID_MODEL[@]}" "${SEED_ARGS[@]}" \
        --lr-mult "$LR_MULT" --out "$PART" "${BUDGET[@]}" || { RC=$?; exit $RC; }
    [ -s "$PART" ] || { log "FATAL calibrate wrote no part at $PART"; RC=14; exit 14; }
    RC=0 ;;
  assess)
    # the ladder is READ from the declaration — lr_mults ∪ extensions, sorted — never typed here
    LADDER="$(python3 "$SWEEP_TOOL" ladder --resid "$RESID")" || { log "FATAL no ladder declaration under $RESID (run prep under this sbatch)"; RC=15; exit 15; }
    PARTS=""; IFS=, read -r -a LADDER_ARR <<< "$LADDER"
    for m in "${LADDER_ARR[@]}"; do PARTS="${PARTS:+$PARTS,}$RESID/calib_parts/lrmult${m}/calibration_controls.json"; done
    for m in "${LADDER_ARR[@]}"; do [ -s "$RESID/calib_parts/lrmult${m}/calibration_controls.json" ] || { log "FATAL missing calibration part for lr_mult $m"; RC=15; exit 15; }; done
    # merge_command: the CPU aggregation with the same selection rule the in-process sweep applies,
    # over EXACTLY the declared ladder (merge-calibration refuses an extra or a missing part)
    run "${RESID_BASE[@]}" merge-calibration "${RESID_COMMON[@]}" --lr-mult "$LADDER" \
        --parts "$PARTS" --out "$RESID/calibration_controls.json" || { RC=15; exit 15; }
    # the DURABLE record of the selection + edge decision; nothing is distilled here
    run python3 "$SWEEP_TOOL" assess --resid "$RESID" || { RC=15; exit 15; }
    [ -s "$RESID/assess.json" ] || { log "FATAL assess wrote no assess.json at $RESID"; RC=15; exit 15; }
    cp "$RESID/assess.json" "$FIGS_DIR/assess_${SLURM_JOB_ID:-x}.json" 2>/dev/null || true
    log "ASSESS: $(tr -d '\n' < "$RESID/assess.json" | tr -s ' ')"
    RC=0 ;;
  finish)
    # the assessed selection is the ONLY thing finish distills at: assess.json must exist, every part
    # and the merged file must still hash to what it recorded, and the merged selection must be it
    SELECTED="$(python3 "$SWEEP_TOOL" check-finish --resid "$RESID")" || { log "FATAL finish refused — no/stale assess.json or a selection that differs (reason above); run assess"; RC=15; exit 15; }
    log "finish: distilling at the ASSESSED lr_mult=$SELECTED"
    # distill_command: production at the SELECTED multiplier (refuses without a complete same-identity sweep)
    run "${RESID_BASE[@]}" distill "${RESID_COMMON[@]}" "${RESID_MODEL[@]}" "${SEED_ARGS[@]}" "${BUDGET[@]}" || { RC=$?; exit $RC; }
    [ -s "$RESID/object.pt" ] && [ -s "$RESID/derived_meta.json" ] && [ -s "$RESID/report.json" ] \
      || { log "FATAL distill published no object at $RESID"; RC=16; exit 16; }
    run python3 "$SWEEP_TOOL" check-distilled --resid "$RESID" || { log "FATAL the distilled object is not at the assessed multiplier $SELECTED"; RC=16; exit 16; }
    # liveness_command: gate liveness of the two psr-gated derived objects on the mix-full pool
    run "$PY" "$REPO/scripts/sdfbase/gate_liveness.py" --model "$MODEL_REPO" --revision "$MODEL_REV" \
        --pairs "$POOLS/pairs_${MIX_SI}.jsonl" --n-probe "$N_PROBE" --dirs "$VDIR,$RESID" --max-len 8192 || { RC=17; exit 17; }
    [ -s "$RESID/gate_liveness.json" ] && [ -s "$VDIR/gate_liveness.json" ] || { log "FATAL liveness wrote no gate_liveness.json"; RC=17; exit 17; }
    # the fit record, in one line, for the RESULT reader
    "$PY" - "$RESID" "$SEED" <<'PY' | tee "$FIGS_DIR/distilled_fit_record.txt"
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); r = json.loads((d / "report.json").read_text()); m = json.loads((d / "derived_meta.json").read_text())
gl = json.loads((d / "gate_liveness.json").read_text()); vals = [float(v) for v in gl.values()]
cal = json.loads((d / "calibration_controls.json").read_text())
a = json.loads((d / "assess.json").read_text()) if (d / "assess.json").is_file() else {}
print(json.dumps({"derived_id": m.get("derived_id"), "recovery_verdict": m.get("recovery_verdict"),
                  "seed": int(sys.argv[2]), "capture_seed": (m.get("trainer") or {}).get("seed"),
                  "best_epoch": r.get("best_epoch"), "epochs_run": r.get("epochs_run"), "epochs_cap": r.get("epochs_cap"),
                  "stopped_early": r.get("stopped_early"), "stop_reason": r.get("stop_reason"),
                  "best_heldout_loss": r.get("best_heldout_loss"), "heldout_init": r.get("heldout_init"),
                  "selected_lr_mult": (cal.get("sweep") or {}).get("selected_lr_mult"),
                  "ladder": a.get("ladder"), "on_edge": a.get("on_edge"), "edge_side": a.get("edge_side"),
                  "extensions": a.get("extensions"), "assess_alarm": a.get("alarm"),
                  "attainable_ed_content": cal.get("attainable_ed_content"), "production_ed": (r.get("ed") or {}),
                  "liveness_min": min(vals), "liveness_mean": sum(vals) / len(vals), "liveness_n_below_0p5": sum(v < 0.5 for v in vals),
                  "object_sha256": m.get("object_sha256")}, default=str))
PY
    RC=0 ;;
esac
log "stage $STAGE done rc=$RC after $(( $(date +%s) - T_START ))s"
exit "$RC"
