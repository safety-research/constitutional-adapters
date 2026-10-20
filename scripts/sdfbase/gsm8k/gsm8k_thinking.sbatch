#!/bin/bash
#SBATCH --job-name=g8k
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-01:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# The GSM8K paired-200 paper-objects battery: ONE row of the matrix (scripts/sdfbase/gsm8k/g8k_cells.py over
# configs/sdfbase/gsm8k/matrix.json: the frozen 42 published rows + the eight-arm extension's appended waves — `ext`, the
# prompted baselines under controls_ext/; `steer`, the AlphaSteer / Arditi rung under controls_steer/, whose objects are
# ANOTHER LANE's, pinned by the P8 admission record and served from the lane's bank view through the row's own overlay pair;
# math-eight-arms-extension E2) on ONE GPU — inspect-ai's stock gsm8k task narrowed to the 200 paired items
# (inspect/gsm8k_paired200), k = 1, thinking ON DECLARED on every request, one pass at the EXPLICIT cap of the rung
# (8,192; the ladder's 16,384 rung over the planner's item subset into a sibling root), the model's thinking-card
# sampling row — on the full22 battery's serving shape: mt_serving_up.py (lens proxy, STACKED, attested) at the rung's
# geometry entry (capability:gen_long_thinking at 8,192; capability:gen_long_gsm8k_rung at 16,384; S from the model's
# budget), --upstream-timeout-s 7200, the USAGE LEDGER in front of the proxy (every row stamped with this attempt's id),
# the tinycap driver (served_capability_run.py generate --controls gsm8k --thinking on) from the client venv. Every job
# names its OWN cell to the standup (with --enable-log-requests: the engine logs the sampler it RESOLVED per request, read
# back after the driver by scripts/sdfbase/record_resolved_sampler.py into <out root>/resolved_sampler.json — a finding beside
# the store, never the job's result); a FITTED row additionally carries the OOS override + the artifact digest under its
# authorised tuple; a DECLARED row (the header-located scale-12 objects of gemma / qwen `main`) names <cell>@thinking; a
# HELD row exits HELD before any venv; a `primary` row refuses every cluster but the primary one at job start. Either
# form re-asserts the row's ELIGIBILITY at the batch boundary (g8k_assert_launchable, g8k_cells.py's own predicate): a
# PUBLISHED (complete) or deferred row refuses by name before the out root exists — the submission is not trusted.
# Usage (positional only — never --export=):
#   sbatch -J g8k-<key>-<view>-<cellshort> --time=<class> scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch \
#       <key> <view> <cell> <rung 8192|16384> <wave_sha> <fitted 0|1> <declared 0|1> [matrix]   # one row, the plan's grammar
#   sbatch -J g8k-<class> --time=<class> --array=<indices>%w scripts/sdfbase/gsm8k/gsm8k_thinking.sbatch \
#       ARRAY <wave_sha> [rung 8192|16384] [matrix]                                          # the matrix rows of one --time class
#                                                                                           # (g8k_cells.py plan; ladder_plan.py for rungs)
# [matrix] (llama-paper-rows P11): `default` (the published 42-row matrix, its wave root) or `llama` (the 11 Llama rows,
# schema g8k_matrix/2, thinking NOT AVAILABLE, the `card` sampling row, their OWN wave root exp/gsm8k_paired200_paper_
# objects_llama); an unknown name is refused HERE, before anything is sourced. The plan prints `ARRAY <sha> 8192 llama`.
# --time is a submitter override per class (g8k_cells.py plan): 0-01:00:00 in-support, 0-01:30:00 fitted, 0-02:00:00 GLM
# fitted; a rung row's --time is ladder_plan.py's computed T_alloc = 2 x T_pred. %w is that class's share of ONE aggregate
# allowance. High QoS by default (the overflow cluster's equivalent QoS: submitter override). STOP: touch $WAVE_ROOT/STOP
# (every job) or $SERVE_DIR/gsm8k/STOP (this job), THEN scancel — never a bare scancel first.
# Dry run (no GPU, no venv, no bank): G8K_DRY_RUN=1 G8K_DRY_S=<S> bash gsm8k_thinking.sbatch <key> <view> <cell> <rung> <sha> <fitted> <declared>
set -uo pipefail
if [ "${1:-}" = ARRAY ]; then
  WAVE_SHA="${2:?wave sha}"; G8K_RUNG_ARG="${3:-8192}"; G8K_MATRIX="${4:-default}"; ROW_ARGS=(ARRAY); FITTED_ARG=""; DECLARED_ARG=""
else
  KEY_ARG="${1:?key}"; VIEW_ARG="${2:?view}"; CELL_ARG="${3:?cell}"; G8K_RUNG_ARG="${4:?rung 8192|16384}"; WAVE_SHA="${5:?wave sha}"
  FITTED_ARG="${6:?fitted 0|1}"; DECLARED_ARG="${7:?declared 0|1}"; G8K_MATRIX="${8:-default}"
  case "$FITTED_ARG" in 0|1) ;; *) echo "FATAL: fitted must be 0 or 1, got $FITTED_ARG" >&2; exit 1;; esac
  case "$DECLARED_ARG" in 0|1) ;; *) echo "FATAL: declared must be 0 or 1, got $DECLARED_ARG" >&2; exit 1;; esac
  ROW_ARGS=("$KEY_ARG" "$VIEW_ARG" "$CELL_ARG")
fi
case "$G8K_RUNG_ARG" in 8192|16384) ;; *) echo "FATAL: rung must be 8192 or 16384, got $G8K_RUNG_ARG" >&2; exit 1;; esac
case "$G8K_MATRIX" in default|llama|itdoc|qwen_seedrep|arditi_allpos|arditi_seedrep|paired_allmodels) ;; frontdoor_*) ;; *) echo "FATAL: matrix must be one of default|llama|itdoc|qwen_seedrep|arditi_allpos|arditi_seedrep|paired_allmodels (g8k_cells.py MATRICES) or frontdoor_<models.yaml key> (the front door's derived matrices, g8k_cells.py front_door_matrices), got $G8K_MATRIX" >&2; exit 1;; esac
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
  # stubs: the row is read from THIS tree; the shared root, venvs, bank, standup and ledger are placeholders; the cluster
  # is the primary one unless the caller names another (the placement refusal is exercised through this seam)
  SHARED="${G8K_DRY_SHARED:-/tmp/g8k-dry}"; WAVE=full22; CLUSTER="${G8K_DRY_CLUSTER:-cluster_a}"
  export SISTEER_LENS_STACKED_REDUCTIONS=1
  CPY="${G8K_DRY_CPY:-$(command -v python3)}"
  cd "$REPO_DIR" || exit 1
  g8k_literals      # the MATRIX's wave root (G8K_WAVE_ROOT_REL), as g8k_init derives it for the real job
  WAVE_ROOT="$SHARED/$G8K_WAVE_ROOT_REL"; FIGS_DIR="$SHARED/figs"; mkdir -p "$WAVE_ROOT" "$FIGS_DIR"
  F22_WAVE_ROOT="$WAVE_ROOT"                    # the pin FOLLOWS THE STORE under the wave root: the dry run's is its stub root (no store = unpinned)
  g8k_select_serving_path "${ROW_ARGS[@]}"      # the row's serving ROUTE (F22_LAUNCH_CELLS + the pin + its reason), as the real job declares it before f22_init
  # ...and the path f22_init would then FIX from that declaration — the SAME f22_select_serving_path, its selector on the node's python3
  # exactly as f22_init runs it (never $CPY, a placeholder in the launcher tests; F22_SELECT_PY names another interpreter): EXPLICIT on a
  # pinned row, kind-aware AUTO over the declared cells otherwise. The plan's SERVING_PATH: line prints the SELECTED path, never a
  # re-typed default; a cell the selector refuses dies HERE, before any GPU minute.
  f22_select_serving_path
  g8k_row "${ROW_ARGS[@]}"
  [ -z "${FITTED_ARG:-}" ] || [ "$FITTED_ARG" = "$FITTED" ] || f22_fatal "FITTED=$FITTED_ARG given but the matrix says $FITTED for ($KEY, $VIEW, $CELL)"
  [ -z "${DECLARED_ARG:-}" ] || [ "$DECLARED_ARG" = "$DECLARED" ] || f22_fatal "DECLARED=$DECLARED_ARG given but the matrix says $DECLARED for ($KEY, $VIEW, $CELL)"
  g8k_resolve_config; g8k_out_root
  SERVE_DIR="$WAVE_ROOT/serving/g8k_${TAG}_${CELLSLUG}_${RUNG}_dry"
  # the bank the row binds: the campaign's view of the row's VIEW, or — a steering row — the EXTERNAL lane's view for the
  # dry run's cluster role (the same dispatch g8k_bank_view makes in production; a row with no view for this role prints none)
  if [ -n "${G8K_OBJECT_SOURCE:-}" ]; then
    case "$CLUSTER" in "$G8K_OVERFLOW_CLUSTER") TT_TWO_SI_DEFENSES_DIR="$SHARED/${G8K_BANK_VIEW_REL_OVERFLOW:-<not-staged-on-this-cluster>}";;
                        *) TT_TWO_SI_DEFENSES_DIR="$SHARED/${G8K_BANK_VIEW_REL_PRIMARY:-<not-staged-on-this-cluster>}";; esac
  else
    TT_TWO_SI_DEFENSES_DIR="$SHARED/exp/objects/sdfbase/full22/$KEY/cells_view/$VIEW"
  fi
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
  # the STOP sentinel first (the standup's own stop file), then the driver, then the ledger, then the standup, then the attempt record
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  [ -n "${DRIVER_PID:-}" ] && kill -TERM "$DRIVER_PID" 2>/dev/null || true
  g8k_ledger_down
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  g8k_attempt_end "$RC" "$G8K_RESULT"
  g8k_registry_end                       # the run's terminal record, before f22_result's final push
  local rec="g8k ${KEY:-${KEY_ARG:-?}} ${VIEW:-${VIEW_ARG:-?}} ${CELL:-${CELL_ARG:-?}} rung=${RUNG:-${G8K_RUNG_ARG:-?}} fitted=${FITTED:-${FITTED_ARG:-?}} declared=${DECLARED:-${DECLARED_ARG:-?}} attempt=${ATTEMPT_ID:-none} rc=$RC driver_rc=${DRC:-none} result=$G8K_RESULT"
  [ -z "${OUT_ROOT:-}" ] || echo "$(date -Is) job=${SLURM_JOB_ID:-manual} array=${SLURM_ARRAY_TASK_ID:-} row=${G8K_INDEX:-?} ${G8K_JOB_ID:-g8k-unresolved} $rec" >> "$OUT_ROOT/RESULT"
  if [ -n "${FIGS_DIR:-}" ]; then f22_result "$rec"; else echo "RESULT (before the wave root resolved; no figs dir yet): $rec" >&2; fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
g8k_registry_end(){   # registry (spec FD 8 "native at every step"): THIS row's run gets its TERMINAL run_state from
                      # the job itself — f22_result's first-word rule reads FAILED/PREEMPTED/CANCELLED records, not `g8k …` ones, so the
                      # outcome is mapped here from G8K_RESULT; a no-op before the job registered (SISTEER_REGISTRY_RUN_IDENTITY unset).
                      # Defined right after the trap is armed (bash binds a function at CALL time; cleanup calls it at exit), so the
                      # helper-free stretch between the source line and the trap stays helper-free.
  [ -n "${SISTEER_REGISTRY_RUN_IDENTITY:-}" ] || return 0
  case "${G8K_RESULT:-INCOMPLETE}" in
    COMPLETE)       f22_registry_state generated COMPLETED final 2>/dev/null || true;;
    STOP_REQUESTED) f22_registry_state cancelled CANCELLED final 2>/dev/null || true;;
    *)              f22_registry_state failed "${G8K_RESULT:-INCOMPLETE}" final 2>/dev/null || true;;
  esac
}
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit

g8k_init "$WAVE_SHA" "${ROW_ARGS[@]}"      # the ROW's serving path is selected inside, before f22_init fixes it from the environment
g8k_row "${ROW_ARGS[@]}"
# the BATCH BOUNDARY (E2; as in the MATH-500 launcher): the resolved row must be in its wave's eligible set — a
# published (complete) or deferred row refuses by name HERE, before the out root exists, the venvs build or anything is written
g8k_assert_launchable
[ -z "${FITTED_ARG:-}" ] || [ "$FITTED_ARG" = "$FITTED" ] || f22_fatal "FITTED=$FITTED_ARG given but the matrix says $FITTED for ($KEY, $VIEW, $CELL)"
[ -z "${DECLARED_ARG:-}" ] || [ "$DECLARED_ARG" = "$DECLARED" ] || f22_fatal "DECLARED=$DECLARED_ARG given but the matrix says $DECLARED for ($KEY, $VIEW, $CELL)"
g8k_resolve_config; g8k_out_root
f22_stop_requested && { f22_log "wave STOP present — not starting"; RC=0; G8K_RESULT=STOP_REQUESTED; exit 0; }
g8k_rung_check
# --- registry (spec FD 8 "native at every step"): THIS row's run is registered by the job itself the moment its
#     overlay is resolved and the row is launchable — f22_registry_register (emit_cli register --from-config, the pending file
#     $SHARED/registry/pending/<jobid>.json, every run `running` at job start), exactly ONE object: the row's cell (F22_REGISTRY_CELLS
#     narrows the overlay's explicit_cells in a job-local copy beside the pending file; the committed configs stay byte-identical), the
#     eval gsm8k (F22_REGISTRY_EVALS from the front door's env prefix, or the default here). SISTEER_ATTEMPT_ID / logical_work_id are
#     stamped from the environment when the row is twinned (emit_cli.attempt_stamp). The terminal run_state is filed by g8k_registry_end
#     in the cleanup trap; f22_result's final push follows it. So a gsm8k unit binds and admits like every other family — no force flag.
export F22_REGISTRY_EVALS="${F22_REGISTRY_EVALS:-gsm8k}" F22_REGISTRY_CELLS="$CELL"
f22_registry_register || true
SERVE_DIR="$WAVE_ROOT/serving/g8k_${TAG}_${CELLSLUG}_${RUNG}_${SLURM_JOB_ID:-manual}"
mkdir -p "$SERVE_DIR/$G8K_SERVE_SUBDIR"
TARGET_GPU="$CUDA_VISIBLE_DEVICES"

# --- venvs: the serving venv (the Nemotron variant carries the fused kernels), the client venv (inspect_ai + the driver)
if [ "$KEY" = nemotron3nano4b ]; then f22_serving_venv nemotron; else f22_serving_venv; fi
f22_client_venv
# --- the datasets probe: the wrapper's own plan OFFLINE, from the client venv, before a GPU minute is spent
g8k_datasets_probe
# --- the bank view of THIS row's VIEW: CHECK-ONLY (never built here); the object's bytes held to the pin when fitted. A
#     steering row binds ANOTHER LANE's view instead (g8k_bank_view dispatches on the row's object source, E2 / P8), and its
#     object goes through the DRIVER's provenance mirror HERE, before the standup (g8k_external_mirror_check, E2-12)
g8k_bank_view
g8k_external_mirror_check
# --- the prompt-budget artifact for (key, class, rung) + its G-FIT sidecar at the pin, into the out root
g8k_budget_make_if_absent   # the front door's row makes its own G-FIT at this tree when none was made before launch (g8k_common.sh)
g8k_budget_artifact
# --- the attempt: minted before the ledger, stamped on every ledger row and request-evidence record
g8k_attempt_begin
# --- the standup: the battery's, with this experiment's extras, at the rung's geometry entry
g8k_standup_extra
f22_standup attacks "$SERVE_DIR/$G8K_SERVE_SUBDIR" "" "$G8K_GEOMETRY_EVAL" "" "" "" || { G8K_RESULT=STANDUP_FAILED; exit 1; }
g8k_check_endpoint
# --- the usage ledger in front of the proxy (the driver is never started without it)
g8k_ledger_up || exit 1
# --- the driver, from the client venv, CPU-only, Inspect pointed at the LEDGER
g8k_driver_argv
PROXY_TOKEN=$(g8k_proxy_token)
f22_log "driver: ${DRV[*]}"
( cd "$REPO_DIR" && env CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" \
    "$G8K_INSPECT_BASE_URL_ENV=$LEDGER_URL" "$G8K_INSPECT_API_KEY_ENV=$PROXY_TOKEN" "${DRV[@]}" ) > "$SERVE_DIR/$G8K_SERVE_SUBDIR/driver.log" 2>&1 &
DRIVER_PID=$!
wait "$DRIVER_PID"; DRC=$?
DRIVER_PID=""
if [ "$DRC" = 0 ]; then G8K_RESULT=COMPLETE; RC=0; else G8K_RESULT=DRIVER_FAILED; RC=2; f22_log "DRIVER FAILED rc=$DRC"; tail -30 "$SERVE_DIR/$G8K_SERVE_SUBDIR/driver.log" || true; fi
# --- the engine-resolved sampler attestation: the server's own request log,
#     read back into the cell's out root beside the store — recorded, never the job's result
g8k_record_resolved_sampler
# --- teardown in order: ledger, standup (the cleanup trap repeats each step idempotently and files the attempt)
g8k_ledger_down
f22_standup_stop
f22_log "done: $G8K_JOB_ID rung=$RUNG attempt=$ATTEMPT_ID result=$G8K_RESULT driver_rc=$DRC"
exit $RC
