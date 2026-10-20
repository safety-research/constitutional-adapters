#!/bin/bash
#SBATCH --job-name=g8k-smoke
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-00:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# gsm8k-paired200-paper-objects / llama-paper-rows — the NO-SWITCH DIAGNOSTIC smoke: the
# certification of a switch-less family's render fact, on the MATH-500 smoke's shape (scripts/sdfbase/math500/smoke_thinking.sbatch).
# ONE GPU, ONE family, the matrix's `none` row on the BASE ROUTE ONLY, G8K_SMOKE_N (24) paired items at G8K_SMOKE_CAP (1,536) under
# the row's `card` sampling row, G8K_SMOKE_CONCURRENCY (8) in flight through the usage ledger — a run that CANNOT produce a production
# result by construction: it writes no tinycap store, its every record carries `diagnostic: true`, and it lives under
# $WAVE_ROOT/$G8K_SMOKE_DIR_REL/<family>/ (never controls/). What it verifies, through contracts.no_switch_render_certified (the
# seam the driver and the analysis' reader share): predicate (A'') — the served header through live /v1/tokenize with NO
# chat_template_kwargs equals the family's render fact (tuple equality, never width); every stamp `unspecified`; every completion
# free of think-tag literals and of a reasoning channel (the `none` split: the content IS the answer); the ledger path with
# `thinking` NULL on every row and body_sha256 == the kwarg-free bytes the probe sent; the serving venv; the client venv; the
# datasets offline; the row's production geometry entry; the smoke's OWN G-FIT at its cap (the endpoint's chat_template_hash held
# to the budget's, as production does); the 7,200 s flag threading; the resolved-sampler recorder over the engine's
# request log. Its completions are committed under data/tinycap/no_switch_smoke/<family>/ and re-read by
# tests/test_no_switch_smoke_real_completions.py; the fact then flips to `certified` (a NEW pin) and the production preflight admits
# the wave. It CANNOT supply cap-8,192 completion lengths (those stay unmeasured until the launch).
# --time (the MATH-500 smoke's rule): the probe sends its completions with `concurrency` in flight (the standup's S is 44 for
# llama8b at this geometry, so 8 draws decode together), so the wall is sized from the REALISTIC per-request time at the diagnostic
# cap — never from the timeout, which no wall can cover — and padded x2, not x5:
#   --time = 2 x (standup 8 min + 2 tokenize probes x 1 min + ceil(24 completions / 8 in flight) x 1 min + teardown 2 min) = 2 x 15 min = 30 min
# where 24 = G8K_SMOKE_N and 8 = G8K_SMOKE_CONCURRENCY (g8k_cells.py); 1 min per cap-1,536 draw is ~26 tok/s per stream at the
# cap (a GSM8K answer without a reasoning channel is far shorter); 8 min stands an 8B engine up (weights, compile, graphs, proxy,
# attestation); the 2 tokenize probes (add_generation_prompt true / false, ONE render) are bounded at 60 s each; teardown is the
# STOP sentinel + the 2 min the trap waits before kill -9. This is the header's `--time 0-00:30:00`; note
# f22_standup's endpoint ceiling (40 min) exceeds it, so a standup that never comes up is filed INCOMPLETE through the
# SIGTERM@120 trap rather than STANDUP_FAILED. A wider n or a lower concurrency needs a matching `sbatch -t`. The test recomputes
# this line.
# Usage (positional only — never --export=):
#   sbatch -J g8k-smoke-<key>-<view> scripts/sdfbase/gsm8k/smoke_not_available.sbatch <key> <view> <wave_sha> [n=24] [cap=1536] [concurrency=8] [matrix=llama]
# [matrix] (P11): `llama` (default — the only matrix with a switch-less family) or `default` (refused at the row: no `not_available`
# row). Dry run (no GPU, no venv, no bank): G8K_DRY_RUN=1 G8K_DRY_S=<S> bash smoke_not_available.sbatch <key> <view> <sha>
set -uo pipefail
KEY_ARG="${1:?key}"; VIEW_ARG="${2:?view}"; WAVE_SHA="${3:?wave sha}"
SMOKE_N_ARG="${4:-}"; SMOKE_CAP_ARG="${5:-}"; SMOKE_CONCURRENCY_ARG="${6:-}"; G8K_MATRIX="${7:-llama}"
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
  # stubs: the row is read from THIS tree; the shared root, venvs, bank, standup and ledger are placeholders
  SHARED="${G8K_DRY_SHARED:-/tmp/g8k-dry}"; WAVE=full22; CLUSTER="${G8K_DRY_CLUSTER:-cluster_a}"
  export SISTEER_LENS_STACKED_REDUCTIONS=1
  CPY="${G8K_DRY_CPY:-$(command -v python3)}"
  cd "$REPO_DIR" || exit 1
  g8k_literals; g8k_smoke_args
  WAVE_ROOT="$SHARED/$G8K_WAVE_ROOT_REL"; FIGS_DIR="$SHARED/figs"; mkdir -p "$WAVE_ROOT" "$FIGS_DIR"
  g8k_row "$KEY_ARG" "$VIEW_ARG" none
  g8k_smoke_check_row
  g8k_resolve_config; g8k_smoke_root
  SERVE_DIR="$SMOKE_ROOT/serving"
  TT_TWO_SI_DEFENSES_DIR="$SHARED/exp/objects/sdfbase/full22/$KEY/cells_view/$VIEW"
  PROXY_URL="http://127.0.0.1:0/v1"; TABLE="$SERVE_DIR/$G8K_SERVE_SUBDIR/routing_table.json"; LEDGER_URL="http://127.0.0.1:0/v1"; PROXY_TOKEN=petri
  TARGET_S="${G8K_DRY_S:?G8K_DRY_S (the resolved S) is required for the dry run}"
  ATTEMPT_ID="${G8K_DRY_ATTEMPT:-a1}"
  g8k_smoke_print_plan
  exit 0
fi
# --- the cleanup trap, armed BEFORE the first step that can fail (g8k_init) and safe at ANY step: every value the record
# reads has a default; the RESULT record goes to $FIGS_DIR once the wave root is resolved, to stderr (the .out) before
TARGET_PID=""; STANDUP_STOP=""; LEDGER_PID=""; RC=1; G8K_RESULT=INCOMPLETE
cleanup(){
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  g8k_ledger_down
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  g8k_attempt_end "$RC" "$G8K_RESULT"
  [ -z "${OUT_ROOT:-}" ] || [ -z "${FIGS_DIR:-}" ] || cp "$OUT_ROOT/smoke_report.json" "$FIGS_DIR/smoke_report_${KEY_ARG}_${VIEW_ARG}_${SLURM_JOB_ID:-manual}.json" 2>/dev/null || true
  local rec="g8k-smoke ${KEY_ARG} ${VIEW_ARG} family=${G8K_FAMILY:-?} thinking=${G8K_THINKING:-?} cap=${SMOKE_CAP:-?} n=${SMOKE_N:-?} attempt=${ATTEMPT_ID:-none} rc=$RC result=$G8K_RESULT diagnostic=true root=${OUT_ROOT:-unresolved}"
  [ -z "${OUT_ROOT:-}" ] || echo "$(date -Is) job=${SLURM_JOB_ID:-manual} $rec" >> "$OUT_ROOT/RESULT"
  if [ -n "${FIGS_DIR:-}" ]; then f22_result "$rec"; else echo "RESULT (before the wave root resolved; no figs dir yet): $rec" >&2; fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit

g8k_init "$WAVE_SHA"
g8k_smoke_args
g8k_row "$KEY_ARG" "$VIEW_ARG" none
g8k_smoke_check_row
g8k_resolve_config; g8k_smoke_root
f22_stop_requested && { f22_log "wave STOP present — not smoking"; RC=0; G8K_RESULT=STOP_REQUESTED; exit 0; }
mkdir -p "$OUT_ROOT"
SERVE_DIR="$SMOKE_ROOT/serving"
mkdir -p "$SERVE_DIR/$G8K_SERVE_SUBDIR"
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
# --- venvs: the serving venv (the Nemotron variant carries the fused kernels), the client venv (inspect_ai + the probe's deps)
if [ "$KEY" = nemotron3nano4b ]; then f22_serving_venv nemotron; else f22_serving_venv; fi
f22_client_venv
# --- the datasets probe: the wrapper's own plan OFFLINE, from the client venv, before a GPU minute is spent
g8k_datasets_probe
# --- the bank view of THIS row's VIEW: CHECK-ONLY (the base route serves no object; no cell, no pin)
f22_bank_view_check "$CPY"
# --- the smoke's OWN G-FIT at its cap, at this tree, held to this run as a production row's is
g8k_smoke_budget
# --- the attempt: minted before the ledger, stamped on every ledger row (read back by the probe)
g8k_attempt_begin
# --- the standup: base route only — the `none` row's extras (--cell none, no digest, no parser, --enable-log-requests), the row's
#     production geometry entry, the 7,200 s timeout and the salt policy, exactly as the production job composes them
g8k_standup_extra
f22_standup attacks "$SERVE_DIR/$G8K_SERVE_SUBDIR" "" "$G8K_GEOMETRY_EVAL" "" "" "" || { G8K_RESULT=STANDUP_FAILED; exit 1; }
g8k_check_endpoint
# --- the usage ledger in front of the proxy (the probe is never started without it)
g8k_ledger_up || exit 1
PROXY_TOKEN=$(g8k_proxy_token)
g8k_smoke_probe_argv
f22_log "smoke probe: ${SPA[*]}"
if ( cd "$REPO_DIR" && env CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "${SPA[@]}" ) > "$SERVE_DIR/$G8K_SERVE_SUBDIR/probe.log" 2>&1; then
  G8K_RESULT=PASS; RC=0; f22_log "diagnostic smoke PASS for $KEY/$VIEW ($G8K_FAMILY, $G8K_THINKING): $OUT_ROOT/smoke_report.json"
else
  G8K_RESULT=FAIL; RC=2; f22_log "diagnostic smoke FAIL for $KEY/$VIEW ($G8K_FAMILY): $(tail -20 "$SERVE_DIR/$G8K_SERVE_SUBDIR/probe.log" | tr '\n' ' ' | head -c 1200)"
fi
# --- the engine-resolved sampler attestation: the server's own request log, read back beside the completions —
#     recorded, never the job's result (--client-cap = the smoke cap)
g8k_record_resolved_sampler
# --- teardown in order: ledger, standup (the cleanup trap repeats each step idempotently and files the attempt)
g8k_ledger_down
f22_standup_stop
f22_log "done: $G8K_JOB_ID smoke cap=$SMOKE_CAP n=$SMOKE_N attempt=$ATTEMPT_ID result=$G8K_RESULT"
exit $RC
