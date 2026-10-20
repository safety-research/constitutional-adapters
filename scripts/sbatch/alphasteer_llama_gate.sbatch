#!/bin/bash
#SBATCH --job-name=as22-llama-gate    # the alphasteer-full22 (as22-*) prefix; cancel by
#                                     # JOB ID or `scancel -u $USER -n as22-llama-gate`
#                                     # (never a user-wide pattern: hd-*/f22-* are a peer's)
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; draws 1 of the 16-GPU high budget
#SBATCH --gres=gpu:1                  # one model load; the released reading + the refit screen
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-06:00:00             # right-sized 2026-09-09: the gate reading took 4m21s incl. capture; the 32-layer refit is the rest (3-5 h total), padded <2x
# JB4: the job reads harmful prompt lists, so a stack trace could print harmful
# text — logs go to the mode-0700 PRIVATE dir, NOT world-readable exp/logs.
# Slurm opens the log before the script runs and does NOT create parents:
#   install -d -m 700 $SHARED/exp/jailbreak/_logs   (exists)
#SBATCH --output=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.out
#SBATCH --error=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# alphasteer-full22 PHASE D — the Llama layer-rule validation, the campaign's
# STOP GATE. Runs FIRST and ALONE: no Phase B job
# is submitted until $SHARED/exp/objects/alphasteer_f22/llama_gate.json says
# passed: true. Drives scripts/alphasteer_llama_gate.py:
#   1. D.0 norm probe of the authors' released matrices (re-verifies the recorded file);
#   2. capture of every block on the pinned lists (site=pre, add_special_tokens
#      =False, ids asserted equal to the serving render);
#   3. THE GATE: the F.9 rule on the RELEASED matrices, deployed ||h @ M_l||
#      scoring, projector leakage reconstructed at the paper ratios; recovery
#      vs the paper's 10 layers; passed iff >= 7  -> llama_gate.json
#   4. only if passed: the 32-layer refit through alphasteer_screen.run_screen,
#      reported BESIDE the gate (it cannot set `passed`).
#
# Exit codes: 0 passed (+refit); 3 GATE FAILED (stop, ask the user); 4 passed
# but the refit A5-stopped; 2 the released file is not the recorded one.
#
# Positional: [attempt] [max_attempts] — stage2_common's self-resubmit passes
# them on preemption/TERM; nothing through sbatch --export (hard rule 1).
# Submit (from the grading host, over ssh to the login node):
#   sbatch --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-as22/repo rev-parse HEAD) \
#          ${SISTEER_PROJECT_ROOT}-as22/repo/scripts/sbatch/alphasteer_llama_gate.sbatch
#
# This lane owns its OWN checkout root ($S2_ROOT below) so the peers' shared
# production checkout (the project root's repo/) and its node-local venv are
# never repointed or re-synced under a running job (capeval-controls §6 seam).

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"

S2_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-as22"          # repo at $S2_ROOT/repo, gate file $S2_ROOT/EXPECTED_SHA,
                                            # venv /home/$USER/venvs/sisteer-as22, wandb $S2_ROOT/wandb
source "$S2_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"                                # sha pin + EXPECTED_SHA gate, hf_env, uv sync --frozen, cu128 assert

# Figures are mirrored under the CAMPAIGN name as well as the job name:
# s2_finish mirrors $FIGS_DIR to the figure mirror under <job name>/; this
# second push lands the same files under alphasteer-full22/.
# Wrapping s2_finish keeps its `local rc=$?` reading the trap's entry status
# (it is the first command).
as22_finish() {
  s2_finish
  "$SHARED/scripts/mirror-figs.sh" "$FIGS_DIR" alphasteer-full22 2>/dev/null || true
}
trap as22_finish EXIT

GATE_CONFIG=configs/alphasteer_f22_llama8b_gate.yaml
RELEASED="$SHARED/AlphaSteer/data/steering_matrix/steering_matrix_llama3.1.pt"
LISTS_DIR="$SHARED/exp/jailbreak/contrast"
OBJ_ROOT="$SHARED/exp/objects/alphasteer_f22"
GATE_OUT="$OBJ_ROOT/llama_gate.json"
LLAMA_SNAPSHOT="$HF_HOME/hub/models--NousResearch--Meta-Llama-3.1-8B-Instruct/snapshots/d10aef7999a2b5ba950ab3974312feeedbfe0b77"

# Fail-fast guards on every external input (cluster-submit skill): -s, not -f,
# so an empty-but-present file is caught too. An 11-second death here beats a
# 48-minute one at the first read.
fail=0
for f in "$GATE_CONFIG" "$RELEASED" "$LISTS_DIR/benign_alphasteer.json" \
         "$LISTS_DIR/harmful_alphasteer.json" "$LISTS_DIR/benign_alphasteer.slices.json" \
         "$LISTS_DIR/harmful_alphasteer.slices.json" "$OBJ_ROOT/leakage_report.json"; do
  if [ ! -s "$f" ]; then s2_log "FATAL: required input missing or empty: $f"; fail=1; fi
done
if [ ! -d "$LLAMA_SNAPSHOT" ]; then s2_log "FATAL: Llama snapshot not in the HF cache: $LLAMA_SNAPSHOT"; fail=1; fi
[ "$fail" -eq 0 ] || exit 1
mkdir -p "$OBJ_ROOT/llama8b"
s2_log "inputs present: released=$RELEASED lists=$LISTS_DIR gate_out=$GATE_OUT figs=$FIGS_DIR"
s2_log "leakage_report.json sha256[:16]=$(sha256sum "$OBJ_ROOT/leakage_report.json" | cut -c1-16) (expect 9abd0d24d3baea06)"

# Never set CUDA_VISIBLE_DEVICES here (hard rule 3): Slurm exports it.
# The W&B project: the site's SISTEER_WANDB_PROJECT_ALPHASTEER (the cluster's hf_site.env,
# exported by hf_env.sh in s2_init), else the job's WANDB_PROJECT / SISTEER_WANDB_PROJECT;
# none of them set = W&B off (the script's own default).
AS_WANDB_PROJECT="${SISTEER_WANDB_PROJECT_ALPHASTEER:-${WANDB_PROJECT:-${SISTEER_WANDB_PROJECT:-}}}"
s2_log "W&B project: ${AS_WANDB_PROJECT:-<none: W&B off>}"
s2_run uv run --no-sync python scripts/alphasteer_llama_gate.py \
  --config "$GATE_CONFIG" --released "$RELEASED" --gate-out "$GATE_OUT" \
  --figs-dir "$FIGS_DIR" --resume --wandb-project "$AS_WANDB_PROJECT"
rc=$?
case "$rc" in
  0) s2_log "PHASE D: gate PASSED and the 32-layer refit completed -> $GATE_OUT" ;;
  3) s2_log "PHASE D: STOP GATE FIRED — the F.9 rule recovered fewer than 7 of the paper's 10 Llama layers on the RELEASED matrices. No new-model object may be fit. Ask the user. Verdict: $GATE_OUT" ;;
  4) s2_log "PHASE D: gate PASSED; the 32-layer refit hit spec A5's stop (verdict stands) -> $GATE_OUT" ;;
  2) s2_log "PHASE D: the released matrix file is not the one D8 measured — refused before any reading" ;;
  *) s2_log "PHASE D: FAILED rc=$rc (see the .err sibling for the traceback)" ;;
esac
exit "$rc"
