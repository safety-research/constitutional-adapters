#!/bin/bash
#SBATCH --job-name=g8k-overlap
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-00:45:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# gsm8k-paired200-paper-objects — the OVERLAP CHECK's stock leg (the analysis reads it): ONE model's `none` cell on ONE
# GPU, inspect/gsm8k_paired200 at the retired instrument's decode settings — thinking OFF, greedy, the OFF table's 512 as
# the cap (statement == resolution; no setter) — identity v2, into overlap/<key>/none/greedy_off512/ under the wave root,
# never a controls slot. The SAME body as gsm8k_thinking.sbatch (the shared core + g8k_common.sh) over the leg row
# g8k_cells.py overlap-leg emits (the same variable names), at the OFF class's entry capability:gen_long (4,608; OFF prompt
# max 2,375 + 512 fits with margin 1,721). Four legs (g8k_cells.py plan-overlap); Nemotron's on the primary cluster only.
# Usage (positional only — never --export=):
#   sbatch -J g8k-overlap-<key> scripts/sdfbase/gsm8k/overlap_off.sbatch <key> <wave_sha>         # one leg
#   sbatch -J g8k-overlap --array=<indices>%w scripts/sdfbase/gsm8k/overlap_off.sbatch ARRAY <wave_sha>   # g8k_cells.py plan-overlap
# Dry run (no GPU, no venv, no bank): G8K_DRY_RUN=1 G8K_DRY_S=<S> bash overlap_off.sbatch <key> <sha>
set -uo pipefail
if [ "${1:-}" = ARRAY ]; then
  WAVE_SHA="${2:?wave sha}"; LEG_ARGS=(ARRAY)
else
  KEY_ARG="${1:?key}"; WAVE_SHA="${2:?wave sha}"; LEG_ARGS=("$KEY_ARG")
fi
# the grading host dry run reads THIS tree (here BASH_SOURCE is the real file, not Slurm's spool copy)
if [ "${G8K_DRY_RUN:-0}" = 1 ]; then G8K_REPO_DIR="${G8K_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"; fi
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
if [ "${G8K_DRY_RUN:-0}" = 1 ]; then
  SHARED="${G8K_DRY_SHARED:-/tmp/g8k-dry}"; WAVE=full22; CLUSTER="${G8K_DRY_CLUSTER:-cluster_a}"
  WAVE_ROOT="$SHARED/$G8K_WAVE_ROOT_REL"; FIGS_DIR="$SHARED/figs"; mkdir -p "$WAVE_ROOT" "$FIGS_DIR"
  export SISTEER_LENS_STACKED_REDUCTIONS=1
  CPY="${G8K_DRY_CPY:-$(command -v python3)}"
  cd "$REPO_DIR" || exit 1
  g8k_overlap_leg "${LEG_ARGS[@]}"
  g8k_resolve_config; g8k_out_root
  SERVE_DIR="$WAVE_ROOT/serving/g8k_overlap_${KEY}_dry"
  TT_TWO_SI_DEFENSES_DIR="$SHARED/exp/objects/sdfbase/full22/$KEY/cells_view/$VIEW"
  PROXY_URL="http://127.0.0.1:0/v1"; TABLE="$SERVE_DIR/$G8K_SERVE_SUBDIR/routing_table.json"; LEDGER_URL="http://127.0.0.1:0/v1"
  TARGET_S="${G8K_DRY_S:?G8K_DRY_S (the resolved S) is required for the dry run}"
  ATTEMPT_ID="${G8K_DRY_ATTEMPT:-a1}"
  g8k_print_plan
  exit 0
fi
# --- the cleanup trap, armed BEFORE the first step that can fail (g8k_init) and safe at ANY step: every value the record
# reads has a default; the RESULT record goes to $FIGS_DIR once the wave root is resolved, to stderr (the .out) before
TARGET_PID=""; STANDUP_STOP=""; DRIVER_PID=""; LEDGER_PID=""; RC=1; DRC=""; G8K_RESULT=INCOMPLETE
cleanup(){
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  [ -n "${DRIVER_PID:-}" ] && kill -TERM "$DRIVER_PID" 2>/dev/null || true
  g8k_ledger_down
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  g8k_attempt_end "$RC" "$G8K_RESULT"
  local rec="g8k-overlap ${KEY:-${KEY_ARG:-?}} none thinking=off greedy cap=512 attempt=${ATTEMPT_ID:-none} rc=$RC driver_rc=${DRC:-none} result=$G8K_RESULT"
  [ -z "${OUT_ROOT:-}" ] || echo "$(date -Is) job=${SLURM_JOB_ID:-manual} array=${SLURM_ARRAY_TASK_ID:-} leg=${G8K_INDEX:-?} ${G8K_JOB_ID:-g8k-overlap-unresolved} $rec" >> "$OUT_ROOT/RESULT"
  if [ -n "${FIGS_DIR:-}" ]; then f22_result "$rec"; else echo "RESULT (before the wave root resolved; no figs dir yet): $rec" >&2; fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit

g8k_init "$WAVE_SHA"
g8k_overlap_leg "${LEG_ARGS[@]}"
g8k_resolve_config; g8k_out_root
f22_stop_requested && { f22_log "wave STOP present — not starting"; RC=0; G8K_RESULT=STOP_REQUESTED; exit 0; }
SERVE_DIR="$WAVE_ROOT/serving/g8k_overlap_${KEY}_${SLURM_JOB_ID:-manual}"
mkdir -p "$SERVE_DIR/$G8K_SERVE_SUBDIR"
TARGET_GPU="$CUDA_VISIBLE_DEVICES"

if [ "$KEY" = nemotron3nano4b ]; then f22_serving_venv nemotron; else f22_serving_venv; fi
f22_client_venv
# --- the datasets probe: the wrapper's own plan OFFLINE, from the client venv, before a GPU minute is spent
g8k_datasets_probe
f22_bank_view_check "$CPY"
g8k_budget_artifact
g8k_attempt_begin
g8k_standup_extra
f22_standup attacks "$SERVE_DIR/$G8K_SERVE_SUBDIR" "" "$G8K_GEOMETRY_EVAL" "" "" "" || { G8K_RESULT=STANDUP_FAILED; exit 1; }
g8k_check_endpoint
g8k_ledger_up || exit 1
g8k_driver_argv
PROXY_TOKEN=$(g8k_proxy_token)
f22_log "driver: ${DRV[*]}"
( cd "$REPO_DIR" && env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" \
    "$G8K_INSPECT_BASE_URL_ENV=$LEDGER_URL" "$G8K_INSPECT_API_KEY_ENV=$PROXY_TOKEN" "${DRV[@]}" ) > "$SERVE_DIR/$G8K_SERVE_SUBDIR/driver.log" 2>&1 &
DRIVER_PID=$!
wait "$DRIVER_PID"; DRC=$?
DRIVER_PID=""
if [ "$DRC" = 0 ]; then G8K_RESULT=COMPLETE; RC=0; else G8K_RESULT=DRIVER_FAILED; RC=2; f22_log "DRIVER FAILED rc=$DRC"; tail -30 "$SERVE_DIR/$G8K_SERVE_SUBDIR/driver.log" || true; fi
g8k_ledger_down
f22_standup_stop
f22_log "done: $G8K_JOB_ID attempt=$ATTEMPT_ID result=$G8K_RESULT driver_rc=$DRC"
exit $RC
