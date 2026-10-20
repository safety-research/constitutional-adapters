#!/bin/bash
#SBATCH --job-name=f22-derived-resid
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22: the DISTILLED PSR RESIDUAL on Slurm — the derived Modal app's `all` path
# (scripts/modal/sdfbase_derived.py residual_command) ported to ONE GPU job, one process, one model load:
#   compose-v (CPU; parents identity-gated through parent_admission) -> capture-set -> calibrate at --lr-mult
#   0.3,1,3 (every multiplier resumable, finished ones reused on a retry) -> distill at the calibration's selected
#   multiplier (the budget: --distill-epochs-cap 60 --distill-patience 5; trainer checkpoints every 50 steps).
# Parents: <bank>/<key>/cells_all/{<mix>,<ctl>}__psr_ce_all — the Slurm fit lane's newmodels_full bank; outputs under
# <bank>/derived/{psr_paired_write,psr_residual}/<key>/… (the layout stage_full22.sh's `derived` lane stages from,
# beside the LoRA subtraction). Hyperparameters come from the fit lane's config (train_kwargs.psr_ce_all — the HPO
# winner for glm46vflash), whose sha256 is logged. The fit lane's venv runs it (torch + the trainer deps, proven
# on this model by the fits themselves); PYTHONPATH puts THIS checkout's src + scripts/sdfbase first so the builder
# is the wave sha's. Runs where that venv lives (cluster A, $SHARED/envs/sdfbase-fit).
# Usage (positional only — never --export=):
#   sbatch -J f22-derived-resid-<key> scripts/sdfbase/full22/derived_residual.sbatch \
#       <key> <mix_si> <ctl_si> <wave_sha> <fits_config> [lr_mults=0.3,1,3] [capture_budget_tokens=100000]
set -uo pipefail
KEY="${1:?bank model key, e.g. glm46vflash_doc}"; MIX="${2:?mix si}"; CTL="${3:?ctl si}"; WAVE_SHA="${4:?wave sha}"
FITS_CFG="${5:?fits config path}"; LR_MULTS="${6:-0.3,1,3}"
# CAPTURE BUDGET (2026-09-08): the wave's residuals (gemma/qwen on Modal, nemotron on Slurm) all captured a seeded
# 100,000-official-token subsample of each parent pool (capture_manifest.capture_subsample.budget_tokens=100000 ->
# 72 train / 10 holdout docs). psr_residual_distill.py's default is None = EVERY doc of both pools, and the first run
# of this port (GLM, job 464386) took that default: 30,983 train / 3,442 holdout docs, a 430x larger recipe that sat
# 70 min inside the un-logged paired_gate_liveness pass and could not have finished its 3-mult calibration in 24 h.
CAPTURE_BUDGET_TOKENS="${7:-100000}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_CELLS=()   # this job stands up no target: no steered cell to select a serving path for (the kind-aware default, 2026-09-15)
f22_init "$WAVE_SHA"
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
BANK="$SHARED/exp/objects/sdfbase/newmodels_full"; POOLS="$SHARED/vol/pool/$KEY"; OUT_ROOT="$BANK/derived"
FPY="$SHARED/envs/sdfbase-fit/bin/python"
[ -x "$FPY" ] || f22_fatal "no fit-lane venv at $FPY (this job runs where the fit lane's venv is)"
[ -f "$FITS_CFG" ] || f22_fatal "no fits config at $FITS_CFG"
for si in "$MIX" "$CTL"; do
  d="$BANK/$KEY/cells_all/${si}__psr_ce_all"
  [ -f "$d/COMPLETE" ] && [ -f "$d/object.pt" ] && [ -f "$d/train_budget_meta.json" ] \
    || f22_fatal "parent $d is not a COMPLETE published fit (a .tmp-* dir is a live fit, never a parent)"
done
[ -f "$POOLS/pairs_${MIX}.manifest.json" ] && [ -f "$POOLS/pairs_${CTL}.manifest.json" ] || f22_fatal "pool manifests missing under $POOLS"
MODEL=$("$FPY" -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["model"])' "$FITS_CFG")
REV=$("$FPY" -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["model_revision"])' "$FITS_CFG")
f22_log "residual: key=$KEY mix=$MIX ctl=$CTL model=$MODEL@$REV cfg=$FITS_CFG cfg_sha256=$(sha256sum "$FITS_CFG" | cut -c1-16) lr_mults=$LR_MULTS capture_budget_tokens=$CAPTURE_BUDGET_TOKENS venv=$FPY out=$OUT_ROOT"
RC=1
cleanup(){ f22_result "derived residual $KEY $MIX-$CTL rc=${RC}"; }
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
export PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts/sdfbase${PYTHONPATH:+:$PYTHONPATH}"
srun --ntasks=1 "$FPY" scripts/sdfbase/psr_residual_distill.py all \
  --root "$BANK" --out-root "$OUT_ROOT" --pools "$POOLS" --model-key "$KEY" --mix-si "$MIX" --ctl-si "$CTL" \
  --model "$MODEL" --revision "$REV" --config "$FITS_CFG" --lr-mult "$LR_MULTS" \
  --capture-budget-tokens "$CAPTURE_BUDGET_TOKENS" \
  --distill-epochs-cap 60 --distill-patience 5 --checkpoint-every 50
RC=$?
D="$OUT_ROOT/psr_residual/$KEY/${MIX}__psr_ce_all_minus_ctl_distilled"
f22_log "residual rc=$RC -> $D"
[ -f "$D/derived_meta.json" ] && f22_log "derived_meta: $(tr -d '\n' < "$D/derived_meta.json" | head -c 900)"
[ -f "$D/report.json" ] && f22_log "report: $(tr -d '\n' < "$D/report.json" | head -c 900)"
exit $RC
