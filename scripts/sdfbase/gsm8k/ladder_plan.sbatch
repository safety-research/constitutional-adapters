#!/bin/bash
#SBATCH --job-name=g8k-ladder-plan
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-00:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# gsm8k-paired200-paper-objects — the LADDER PLAN: a CPU-only job, one per cluster after that
# cluster's 8,192 rows complete, in the client venv (Inspect reads the .eval logs the adapter classifies). Per launched row
# with a complete store: lane H's inspect_rung.classify_store -> cap_ladder.json (this instrument's rule, threshold, report
# line) and, for a cell over the trigger (1 % since 2026-09-12) with eligible items, rung_subset.json under $WAVE_ROOT/rung16384/<TAG>/<slug>/
# bound to the source store; the rung's T_pred / T_alloc from the ACTUAL subset, S at the rung entry and the cell's observed
# decode rate; the sbatch lines for THIS cluster (Nemotron's under the primary only), the FIRST rung line separated. The
# lines are PRINTED, never run: they are submitted by hand, with a watcher per job at --max-hours =
# 2 x T_alloc. Idempotent: a running / complete rung is never re-planned; a differing subset refuses by name.
# Usage (positional only — never --export=):
#   sbatch -J g8k-ladder-plan scripts/sdfbase/gsm8k/ladder_plan.sbatch <wave_sha> [rows spec] [legs_sha] [matrix]
# <wave_sha> is THIS planner's tree (the checkout the helpers and the adapter come from); [legs_sha] (default <wave_sha>) is
# the pin the RUNG ROWS run at, named in every emitted sbatch line — the trigger harmonisation of 2026-09-12 planned a wave's
# rungs from a later tree while the legs stayed at the wave's pin; [matrix] (default | llama — g8k_cells.py MATRICES; P11) names
# the matrix whose launched rows are planned and whose OWN wave root g8k_init derives. An empty rows spec ("") plans every
# launched row and an empty legs sha ("") means <wave_sha>: each keeps its position.
set -uo pipefail
WAVE_SHA="${1:?wave sha}"; ROWS_ARG="${2:-}"; LEGS_SHA="${3:-$WAVE_SHA}"; G8K_MATRIX="${4:-default}"
case "$LEGS_SHA" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) ;; *) echo "FATAL: legs sha must be hex, got $LEGS_SHA" >&2; exit 1;; esac
case "$G8K_MATRIX" in default|llama|itdoc|qwen_seedrep|arditi_allpos|arditi_seedrep|paired_allmodels) ;; frontdoor_*) ;; *) echo "FATAL: matrix must be one of default|llama|itdoc|qwen_seedrep|arditi_allpos|arditi_seedrep|paired_allmodels (g8k_cells.py MATRICES) or frontdoor_<models.yaml key> (the front door's derived matrices, g8k_cells.py front_door_matrices), got $G8K_MATRIX" >&2; exit 1;; esac
# --- the checkout, located BEFORE either helper is sourced ---------------------------------------------------------
# A Slurm batch process inherits no REPO_DIR (f22_init sets it, later) and $0 is the spool copy, so the path is derived
# the way f22_init derives it: the shared root present on this node + the wave sha; G8K_REPO_DIR overrides it (the
# grading host: the dry run, the startup test; g8k_init refuses if the two ever differ). A missing root or checkout is
# refused BY NAME on stderr before anything else can fail. This block is byte-identical in every launcher (the
# test pins it): the helpers cannot locate themselves.
if [ -n "${G8K_REPO_DIR:-}" ]; then REPO_DIR="$G8K_REPO_DIR"
elif [ -n "${SISTEER_SHARED_ROOT:-}" ]; then REPO_DIR="${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA"
elif [ -n "${SISTEER_EVAL_REPO_ROOT:-}" ]; then REPO_DIR="$SISTEER_EVAL_REPO_ROOT/$WAVE_SHA"
else echo "FATAL: set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config) — nothing to source" >&2; exit 1; fi
[ -r "$REPO_DIR/scripts/sdfbase/full22/common.sh" ] && [ -r "$REPO_DIR/scripts/sdfbase/gsm8k/g8k_common.sh" ] \
  || { echo "FATAL: no checkout at $REPO_DIR (scripts/sdfbase/full22/common.sh + scripts/sdfbase/gsm8k/g8k_common.sh) — README: git -C <the repo mirror> worktree add --detach $REPO_DIR $WAVE_SHA" >&2; exit 1; }
source "$REPO_DIR/scripts/sdfbase/full22/common.sh"; source "$REPO_DIR/scripts/sdfbase/gsm8k/g8k_common.sh"
# --- the cleanup trap, armed BEFORE the first step that can fail (g8k_init) and safe at ANY step: every value the record
# reads has a default; the RESULT record goes to $FIGS_DIR once the wave root is resolved, to stderr (the .out) before
RC=1; G8K_RESULT=INCOMPLETE; PLAN_OUT=""
cleanup(){
  [ -z "${PLAN_OUT:-}" ] || [ -z "${FIGS_DIR:-}" ] || { cp "$PLAN_OUT/ladder_plan.log" "$FIGS_DIR/ladder_plan_${SLURM_JOB_ID:-manual}.log" 2>/dev/null; cp "$PLAN_OUT/sbatch_lines.txt" "$FIGS_DIR/ladder_sbatch_lines_${SLURM_JOB_ID:-manual}.txt" 2>/dev/null; } || true
  local rec="g8k-ladder-plan rc=$RC result=$G8K_RESULT sha=$WAVE_SHA matrix=$G8K_MATRIX cluster=${CLUSTER:-?} out=${PLAN_OUT:-unresolved}"
  if [ -n "${FIGS_DIR:-}" ]; then f22_result "$rec"; else echo "RESULT (before the wave root resolved; no figs dir yet): $rec" >&2; fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
g8k_init "$WAVE_SHA"
f22_stop_requested && { f22_log "wave STOP present — not planning"; RC=0; G8K_RESULT=STOP_REQUESTED; exit 0; }
PLAN_OUT="$WAVE_ROOT/ladder_plan/${SLURM_JOB_ID:-manual}"; mkdir -p "$PLAN_OUT"
f22_client_venv
f22_log "ladder plan: tree=$WAVE_SHA legs=$LEGS_SHA matrix=$G8K_MATRIX cluster=$CLUSTER wave_root=$WAVE_ROOT out=$PLAN_OUT"
( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "$CPY" scripts/sdfbase/gsm8k/ladder_plan.py \
    --wave-root "$WAVE_ROOT" --sha "$LEGS_SHA" --planner-tree-sha "$WAVE_SHA" --cluster "$CLUSTER" --matrix "$G8K_MATRIX" --out "$PLAN_OUT" ${ROWS_ARG:+--rows "$ROWS_ARG"} ) > "$PLAN_OUT/ladder_plan.log" 2>&1
RC=$?
grep -E "^(LADDER|PLAN|NOTE|#|sbatch )" "$PLAN_OUT/ladder_plan.log" || true
if [ "$RC" -eq 0 ]; then G8K_RESULT=PLANNED; f22_log "ladder plan DONE: $PLAN_OUT/sbatch_lines.txt (submit the FIRST rung alone; G-READER-RUNG; then the rest)"
else G8K_RESULT=REFUSED; f22_log "ladder plan REFUSED (rc=$RC): $(tail -3 "$PLAN_OUT/ladder_plan.log" | tr '\n' ' ' | cut -c1-600)"; fi
exit $RC
