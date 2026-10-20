#!/bin/bash
#SBATCH --job-name=m5t
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-10:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# math500-thinking-on-full22: ONE row of the matrix (scripts/sdfbase/math500/cells.py — rows 0-23 the
# published run, frozen; rows 24-32 the eight-arm EXTENSION) on ONE GPU —
# MATH-500, all 500 items, k = 1, thinking ON on every request, one pass at cap 16384, the model's pinned sampling row
# — on the full22 battery's serving shape: mt_serving_up.py (lens proxy, STACKED, attested) at the ROW's geometry entry
# (capability:gen_long_math, L 17,664, for the bare / safety-SI / object cells; capability:gen_long_math_longsi, L 24,064,
# for the constitution-SI cell — x the model's budget S), --upstream-timeout-s 7200, the USAGE LEDGER in front
# of the proxy, the tinycap driver (served_capability_run.py generate --controls math500 --thinking <row's mode>) from the
# client venv. Every job names its OWN cell to the standup; a FITTED job (an object cell) additionally
# carries the driver-side pin and, under an authorised tuple, the OOS override — UNDECLARED (positions=all,
# lens rule 3), the render certified live by the driver, the served-positions stamp read from the FRESH table.
# An EXTENSION row (wave ext1) also opts the standup into --enable-log-requests and, after the driver, records the
# engine-resolved sampler (record_resolved_sampler.py, P6) and the launcher's row-identity stamp (A2) into its
# WAVE-ISOLATED out root (controls_ext1/...), so a pull of that subtree never carries a published leg.
# Usage (positional only — never --export=):
#   sbatch -J m5t-<key>-<cellshort>[-<mode>] --time=<class> scripts/sdfbase/math500/math500_thinking.sbatch \
#       <key> <cell> <sampling_mode> <wave_sha> <fitted 0|1>          # one row, the plan's grammar
#   sbatch -J m5t-<wave|class> --time=<class> --array=<indices>%w scripts/sdfbase/math500/math500_thinking.sbatch \
#       ARRAY <wave_sha>                                               # the rows of one wave (cells.py plan --wave <name>)
# Either form re-asserts the row's ELIGIBILITY at the batch boundary (m5t_assert_launchable, cells.py's own predicate): a
# published (complete), held or deferred row refuses by name before the out root exists — the submission is not trusted.
# --time is a submitter override per class (cells.py plan): 0-05:00:00 bare/prompted, 0-10:00:00 object, 0-13:00:00 GLM
# object (the published classes); 0-06:00:00 every extension row; %w is that class's share of ONE aggregate allowance
# (cells.py plan --max-concurrent W partitions W across the arrays; peers' jobs count against the same 16-GPU `high`
# budget). High QoS by default (the overflow cluster's own high QoS: submitter override). STOP: touch $WAVE_ROOT/STOP
# (every job) or $SERVE_DIR/math500/STOP (this job), THEN scancel — never a bare scancel first.
# M5T_LOG_REQUESTS_OVERRIDE=0|1 (a plain shell variable in the submitter's environment; never --export=) overrides the
# row's sampler read-back opt-in; the RESULT line records the effective value (logreq=).
# Dry run (no GPU, no venv, no bank): M5T_DRY_RUN=1 M5T_DRY_S=<S> bash math500_thinking.sbatch <key> <cell> <mode> <sha> <fitted> [<view>]
# The optional trailing <view> (E3): `main` by default (every row before E3); `it` names an it-doc row, whose (key, cell, mode) the
# main view shares — the ARRAY form resolves by index and needs no view.
set -uo pipefail
if [ "${1:-}" = ARRAY ]; then
  WAVE_SHA="${2:?wave sha}"; ROW_ARGS=(ARRAY); FITTED_ARG=""
else
  KEY_ARG="${1:?key}"; CELL_ARG="${2:?cell}"; SAMPLING_ARG="${3:?sampling mode}"; WAVE_SHA="${4:?wave sha}"; FITTED_ARG="${5:?fitted 0|1}"; VIEW_ARG="${6:-main}"
  case "$FITTED_ARG" in 0|1) ;; *) echo "FATAL: fitted must be 0 or 1, got $FITTED_ARG" >&2; exit 1;; esac
  ROW_ARGS=("$KEY_ARG" "$CELL_ARG" "$SAMPLING_ARG" "$VIEW_ARG")
fi
# the workstation dry run reads THIS tree (here BASH_SOURCE is the real file, not Slurm's spool copy)
if [ "${M5T_DRY_RUN:-0}" = 1 ]; then M5T_REPO_DIR="${M5T_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"; fi
# --- the checkout, located BEFORE either helper is sourced ---------------------------------------------------------
# A Slurm batch process inherits no REPO_DIR (f22_init sets it, later) and $0 is the spool copy, so the path is derived
# the way f22_init derives it: the shared root present on this node + the wave sha; M5T_REPO_DIR overrides it (the
# grading host: the dry run, the startup test; m5t_init refuses if the two ever differ). A missing root or checkout is
# refused BY NAME on stderr before anything else can fail. This block is byte-identical in every launcher (the
# test pins it): the helpers cannot locate themselves.
if [ -n "${M5T_REPO_DIR:-}" ]; then REPO_DIR="$M5T_REPO_DIR"
elif [ -n "${SISTEER_SHARED_ROOT:-}" ]; then REPO_DIR="${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA"
elif [ -n "${SISTEER_EVAL_REPO_ROOT:-}" ]; then REPO_DIR="$SISTEER_EVAL_REPO_ROOT/$WAVE_SHA"
else echo "FATAL: set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config) — nothing to source" >&2; exit 1; fi
[ -r "$REPO_DIR/scripts/sdfbase/full22/common.sh" ] && [ -r "$REPO_DIR/scripts/sdfbase/math500/m5t_common.sh" ] \
  || { echo "FATAL: no checkout at $REPO_DIR (scripts/sdfbase/full22/common.sh + scripts/sdfbase/math500/m5t_common.sh) — README: git -C <the repo mirror> worktree add --detach $REPO_DIR $WAVE_SHA" >&2; exit 1; }
source "$REPO_DIR/scripts/sdfbase/full22/common.sh"; source "$REPO_DIR/scripts/sdfbase/math500/m5t_common.sh"
if [ "${M5T_DRY_RUN:-0}" = 1 ]; then
  # stubs: the row is read from THIS tree; the shared root, venvs, bank, standup and ledger are placeholders
  SHARED="${M5T_DRY_SHARED:-/tmp/m5t-dry}"; WAVE=full22; CLUSTER=dry
  WAVE_ROOT="$SHARED/$M5T_WAVE_ROOT_REL"; FIGS_DIR="$SHARED/figs"; mkdir -p "$WAVE_ROOT" "$FIGS_DIR"
  export SISTEER_LENS_STACKED_REDUCTIONS=1
  CPY="${M5T_DRY_CPY:-$(command -v python3)}"
  F22_WAVE_ROOT="$WAVE_ROOT"                    # the pin FOLLOWS THE STORE under the wave root: the dry run's is its stub root (no store = unpinned)
  m5t_select_serving_path "${ROW_ARGS[@]}"      # the row's serving ROUTE (F22_LAUNCH_CELLS + the pin + its reason), as the real job declares it before f22_init
  # ...and the path f22_init would then FIX from that declaration — the SAME f22_select_serving_path, its selector on the node's python3
  # exactly as f22_init runs it (never $CPY, a placeholder in the launcher tests; F22_SELECT_PY names another interpreter): EXPLICIT on a
  # pinned row, kind-aware AUTO over the declared cells otherwise. The plan's SERVING_PATH: line prints the SELECTED path, never a
  # re-typed default; a cell the selector refuses dies HERE, before any GPU minute.
  f22_select_serving_path
  m5t_row "${ROW_ARGS[@]}"
  # a HELD row prints its reason and NO argv (E3: the GSM8K launcher's shape) — the batch boundary refuses it in production
  if [ "$M5T_STATUS" = held ]; then printf 'HELD: index=%s job_id=%s wave=%s reason=%s\n' "$M5T_INDEX" "$M5T_JOB_ID" "$M5T_WAVE" "${M5T_HELD_REASON:-}"; exit 0; fi
  [ -z "${FITTED_ARG:-}" ] || [ "$FITTED_ARG" = "$FITTED" ] || f22_fatal "FITTED=$FITTED_ARG given but the matrix says $FITTED for ($KEY, $CELL, $SAMPLING_MODE)"
  m5t_resolve_config; m5t_out_root
  SERVE_DIR="$WAVE_ROOT/serving/m5t_${KEY}_${CELLSLUG}_${SAMPLING_MODE}_dry"
  TT_TWO_SI_DEFENSES_DIR="$SHARED/exp/objects/sdfbase/full22/$KEY/cells_view/main"
  PROXY_URL="http://127.0.0.1:0/v1"; TABLE="$SERVE_DIR/math500/routing_table.json"; LEDGER_URL="http://127.0.0.1:0/v1"
  TARGET_S="${M5T_DRY_S:?M5T_DRY_S (the resolved S) is required for the dry run}"
  m5t_print_plan
  exit 0
fi
# --- the cleanup trap, armed BEFORE the first step that can fail (m5t_init) and safe at ANY step: every value the record
# reads has a default; the RESULT record goes to $FIGS_DIR once the wave root is resolved, to stderr (the .out) before
TARGET_PID=""; STANDUP_STOP=""; DRIVER_PID=""; LEDGER_PID=""; RC=1; DRC=""; M5T_RESULT=INCOMPLETE
cleanup(){
  # the STOP sentinel first (the standup's own stop file), then the driver, then the ledger, then the standup
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  [ -n "${DRIVER_PID:-}" ] && kill -TERM "$DRIVER_PID" 2>/dev/null || true
  m5t_ledger_down
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  local rec="m5t ${KEY:-${KEY_ARG:-?}} ${CELL:-${CELL_ARG:-?}} ${SAMPLING_MODE:-${SAMPLING_ARG:-?}} fitted=${FITTED:-${FITTED_ARG:-?}} rc=$RC driver_rc=${DRC:-none} result=$M5T_RESULT wave=${M5T_WAVE:-?} thinking=${M5T_THINKING_MODE:-?} geometry=${M5T_GEOMETRY_EVAL:-?} logreq=${M5T_LOG_REQUESTS_EFFECTIVE:-?}"
  [ -z "${OUT_ROOT:-}" ] || echo "$(date -Is) job=${SLURM_JOB_ID:-manual} array=${SLURM_ARRAY_TASK_ID:-} row=${M5T_INDEX:-?} ${M5T_JOB_ID:-m5t-unresolved} $rec" >> "$OUT_ROOT/RESULT"
  if [ -n "${FIGS_DIR:-}" ]; then f22_result "$rec"; else echo "RESULT (before the wave root resolved; no figs dir yet): $rec" >&2; fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT

m5t_init "$WAVE_SHA" "${ROW_ARGS[@]}"      # the ROW's serving path is selected inside, before f22_init fixes it from the environment
m5t_row "${ROW_ARGS[@]}"
# the BATCH BOUNDARY: the resolved row must be in its wave's eligible set — a published (complete), held or
# deferred row refuses by name HERE, before the out root exists, the venvs build or anything is written
m5t_assert_launchable
[ -z "${FITTED_ARG:-}" ] || [ "$FITTED_ARG" = "$FITTED" ] || f22_fatal "FITTED=$FITTED_ARG given but the matrix says $FITTED for ($KEY, $CELL, $SAMPLING_MODE)"
m5t_resolve_config; m5t_out_root
f22_stop_requested && { f22_log "wave STOP present — not starting"; RC=0; M5T_RESULT=STOP_REQUESTED; exit 0; }
SERVE_DIR="$WAVE_ROOT/serving/m5t_${KEY}_${CELLSLUG}_${SAMPLING_MODE}_${SLURM_JOB_ID:-manual}"
mkdir -p "$SERVE_DIR/math500"
TARGET_GPU="$CUDA_VISIBLE_DEVICES"

# --- venvs: the serving venv (the Nemotron variant carries the fused kernels), the client venv (inspect_ai + the driver)
if [ "$KEY" = nemotron3nano4b ]; then f22_serving_venv nemotron; else f22_serving_venv; fi
f22_client_venv
# --- the bank view: CHECK-ONLY (never built here); the object cell's bytes held to the pin when fitted. A steering
#     row binds ANOTHER LANE's view instead, an E3 it-doc row its record's bank entry (m5t_bank_view dispatches on the
#     row's object source), and either goes through the DRIVER's provenance mirror HERE, before the standup (m5t_external_mirror_check)
m5t_bank_view
m5t_external_mirror_check
# --- the prompt-budget artifact for this cell class (pre-launch CPU run), into the out root
m5t_budget_artifact
# --- the standup: the battery's, with this experiment's extras
m5t_standup_extra
f22_standup attacks "$SERVE_DIR/math500" "" "$M5T_GEOMETRY_EVAL" "" "" "" || { M5T_RESULT=STANDUP_FAILED; exit 1; }
m5t_check_endpoint
# --- the usage ledger in front of the proxy (the driver is never started without it)
m5t_ledger_up || exit 1
# --- the driver, from the client venv, CPU-only, Inspect pointed at the LEDGER
m5t_driver_argv
PROXY_TOKEN=$(m5t_proxy_token)
f22_log "driver: ${DRV[*]}"
( cd "$REPO_DIR" && env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" \
    "$M5T_INSPECT_BASE_URL_ENV=$LEDGER_URL" "$M5T_INSPECT_API_KEY_ENV=$PROXY_TOKEN" "${DRV[@]}" ) > "$SERVE_DIR/math500/driver.log" 2>&1 &
DRIVER_PID=$!
wait "$DRIVER_PID"; DRC=$?
DRIVER_PID=""
if [ "$DRC" = 0 ]; then M5T_RESULT=COMPLETE; RC=0; else M5T_RESULT=DRIVER_FAILED; RC=2; f22_log "DRIVER FAILED rc=$DRC"; tail -30 "$SERVE_DIR/math500/driver.log" || true; fi
# --- the two RECORDS beside the store (P6 / A2), while the standup is still the job's: the engine-resolved sampler read off the
# server log (iff the row opted in), then the launcher's row-identity stamp (sampling_ref from the driver's manifest)
m5t_record_resolved_sampler
m5t_row_identity
# --- teardown in order: ledger, standup (the cleanup trap repeats each step idempotently)
m5t_ledger_down
f22_standup_stop
f22_log "done: $M5T_JOB_ID result=$M5T_RESULT driver_rc=$DRC"
exit $RC
