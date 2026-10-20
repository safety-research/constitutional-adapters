#!/bin/bash
#SBATCH --job-name=m5t-smoke
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-00:48:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# math500-thinking-on-full22 — the DIAGNOSTIC smoke standup (G-DIAG): ONE GPU, ONE family,
# BASE ROUTE ONLY, cap 1536 (--time: the formula below) — a run that CANNOT produce a production result by construction: it writes
# no tinycap store, its every record carries `diagnostic: true`, and it lives under $WAVE_ROOT/diagnostic/<family>/
# (never controls/). What it verifies: predicate (A) — the served header ids through live /v1/tokenize under
# enable_thinking true AND false equal the family's render fact (tuple equality, never width); the ledger path;
# the serving venv (the Nemotron variant for nemotron); the client venv; the M-RoPE attestation; the datasets
# offline; the 7,200 s flag threading; and the render the model EMITS under ON — 24 MATH-500 items ON + 4 OFF,
# completions (+ reasoning_content where the server parsed) dumped to smoke_completions.jsonl for G-PARSER (the
# real-completion certification, lane A's test) with render_regime / finish_reason per row. It CANNOT supply
# cap-16384 completion lengths or lens-route throughput (those stay unmeasured until the launch).
# --time: the probe sends its completions with `concurrency` in flight (default 8 — the standup's S is >= 20 for
# every model at this geometry, so 8 draws decode together; each request keeps the 7,200 s upstream timeout of its
# own), so the wall is sized from the REALISTIC per-request time at the diagnostic cap — never from the timeout, which
# no wall can cover — and padded x2, not x5:
#   --time = 2 x (standup 10 min + 4 tokenize probes x 1 min + ceil(28 completions / 8 in flight) x 2 min + teardown 2 min) = 2 x 24 min = 48 min
# where 28 = n_on 24 + n_off 4 (the defaults); 2 min per cap-1536 thinking-ON draw is ~13 tok/s per stream, under the
# 18-35 tok/s per-stream passthrough measured on a 4B model; 10 min stands the engine up (weights, compile, graphs,
# proxy, attestation); the 4 tokenize probes are bounded at 60 s each; teardown is the STOP sentinel + the 2 min the
# trap waits before kill -9. f22_standup's own endpoint ceiling (40 min) fits inside the wall, so a standup that never
# comes up is recorded STANDUP_FAILED, not TIMEOUT. A wider n_on/n_off or a lower concurrency needs a matching
# `sbatch -t`. The test recomputes this line. Serially the same 28 requests did not reliably fit 30 min.
# Usage (positional only — never --export=):
#   sbatch -J m5t-smoke-<key> scripts/sdfbase/math500/smoke_thinking.sbatch <key> <wave_sha> [n_on=24] [n_off=4] [cap=1536] [concurrency=8]
set -uo pipefail
KEY="${1:?key}"; WAVE_SHA="${2:?wave sha}"; N_ON="${3:-24}"; N_OFF="${4:-4}"; SMOKE_CAP="${5:-1536}"; SMOKE_CONCURRENCY="${6:-8}"
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
# --- the cleanup trap, armed BEFORE the first step that can fail (m5t_init) and safe at ANY step: every value the record
# reads has a default; the RESULT record goes to $FIGS_DIR once the wave root is resolved, to stderr (the .out) before
TARGET_PID=""; STANDUP_STOP=""; LEDGER_PID=""; RC=1; M5T_RESULT=INCOMPLETE
cleanup(){
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  m5t_ledger_down
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  [ -z "${SMOKE_ROOT:-}" ] || [ -z "${FIGS_DIR:-}" ] || cp "$SMOKE_ROOT/smoke_report.json" "$FIGS_DIR/smoke_report_${KEY}_${SLURM_JOB_ID:-manual}.json" 2>/dev/null || true
  local rec="m5t-smoke ${KEY} family=${M5T_FAMILY:-?} rc=$RC result=$M5T_RESULT diagnostic=true root=${SMOKE_ROOT:-unresolved}"
  if [ -n "${FIGS_DIR:-}" ]; then f22_result "$rec"; else echo "RESULT (before the wave root resolved; no figs dir yet): $rec" >&2; fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
m5t_init "$WAVE_SHA"
f22_stop_requested && { f22_log "wave STOP present — not smoking"; RC=0; M5T_RESULT=STOP_REQUESTED; exit 0; }
CELL=none; SAMPLING_MODE=thinking_card
m5t_row "$KEY" "$CELL" "$SAMPLING_MODE"
m5t_resolve_config
SMOKE_ROOT="$WAVE_ROOT/$M5T_SMOKE_OUT_REL/smoke_${SLURM_JOB_ID:-manual}"; SERVE_DIR="$SMOKE_ROOT/serving"
mkdir -p "$SMOKE_ROOT" "$SERVE_DIR/math500"
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
if [ "$KEY" = nemotron3nano4b ]; then f22_serving_venv nemotron; else f22_serving_venv; fi
f22_client_venv
f22_bank_view_check "$CPY"
# the standup: base route only — the `none` row's extras (m5t_standup_extra: --cell none, no digest, gemma's parser), the
# experiment's geometry, timeout and salt policy, exactly as the production jobs compose them
m5t_standup_extra
f22_standup attacks "$SERVE_DIR/math500" "" "$M5T_GEOMETRY_EVAL" "" "" "" || { M5T_RESULT=STANDUP_FAILED; exit 1; }
# the ledger in front of the proxy, exactly as the production jobs (the smoke exercises the ledger path)
OUT_ROOT="$SMOKE_ROOT"
m5t_ledger_up || exit 1
PROXY_TOKEN=$(m5t_proxy_token)
if ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" "$CPY" scripts/sdfbase/math500/smoke_thinking_probe.py \
      --ledger-url "$LEDGER_URL" --proxy-url "$PROXY_URL" --proxy-token "$PROXY_TOKEN" --routing-table "$TABLE" \
      --endpoint-json "$SERVE_DIR/math500/endpoint.json" --ledger-path "$OUT_ROOT/$M5T_LEDGER_FILENAME" \
      --model-key "$KEY" --family "$M5T_FAMILY" --sampling-mode "$SAMPLING_MODE" --sampling-params "$M5T_SAMPLING_PARAMS" \
      --n-on "$N_ON" --n-off "$N_OFF" --cap "$SMOKE_CAP" --concurrency "$SMOKE_CONCURRENCY" --out "$SMOKE_ROOT" ) > "$SERVE_DIR/math500/probe.log" 2>&1; then
  M5T_RESULT=PASS; RC=0; f22_log "diagnostic smoke PASS for $KEY ($M5T_FAMILY): $SMOKE_ROOT/smoke_report.json"
else
  M5T_RESULT=FAIL; RC=2; f22_log "diagnostic smoke FAIL for $KEY ($M5T_FAMILY): $(tail -20 "$SERVE_DIR/math500/probe.log" | tr '\n' ' ' | head -c 1200)"
fi
m5t_ledger_down
f22_standup_stop
exit $RC
