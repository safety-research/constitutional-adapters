#!/bin/bash
#SBATCH --job-name=m5t-dry-prod
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
# math500-thinking-on-full22 — G-DRY-prod: the PRODUCTION dry run, run ON THE CLUSTER where the bank
# views exist, as a CPU-only job (no GPU). Per matrix row (cells.py M5T_CELLS) m5t_dry_run.py --stage prod prints the
# verbatim argv, the family's render fact (the production preflight seam must ADMIT every family: certified x4), and the
# FRESH routing table's lens_served_positions stamp for every fitted row, exported over the live bank views by the
# standup's own cell selection + export_routing_table — hence the serving venv is bootstrapped by name and runs the tool;
# the client venv is named as the driver's interpreter in the printed argv. Exit 0 = PASS (the RESULT line is the gate
# record that admits G-LAUNCH); 2 = a refusal, printed with REFUSED:. The grading host --stage pre never admits anything.
# Usage (positional only — never --export=):
#   sbatch -J m5t-dry-prod scripts/sdfbase/math500/dry_run_prod.sbatch <wave_sha>
set -uo pipefail
WAVE_SHA="${1:?wave sha}"
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
RC=1; M5T_RESULT=INCOMPLETE; DRY_OUT=""
cleanup(){
  [ -z "${DRY_OUT:-}" ] || [ -z "${FIGS_DIR:-}" ] || cp "$DRY_OUT/dry_run_prod.log" "$FIGS_DIR/dry_run_prod_${SLURM_JOB_ID:-manual}.log" 2>/dev/null || true
  local rec="m5t-dry-prod rc=$RC result=$M5T_RESULT sha=$WAVE_SHA log=${DRY_OUT:-unresolved}/dry_run_prod.log"
  if [ -n "${FIGS_DIR:-}" ]; then f22_result "$rec"; else echo "RESULT (before the wave root resolved; no figs dir yet): $rec" >&2; fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
m5t_init "$WAVE_SHA"
f22_stop_requested && { f22_log "wave STOP present — not running the dry run"; RC=0; M5T_RESULT=STOP_REQUESTED; exit 0; }
DRY_OUT="$WAVE_ROOT/dry_run_prod/${SLURM_JOB_ID:-manual}"; mkdir -p "$DRY_OUT"
BANK_ROOT="$SHARED/exp/objects/sdfbase/full22"       # <root>/<key>/cells_view/main per model (the check-only views the jobs read)
[ -d "$BANK_ROOT" ] || f22_fatal "no bank views root at $BANK_ROOT — the fresh-table check runs where the views exist"
# the two interpreters the production jobs use: the standup's (the fresh-table export IS the standup's code) runs the tool,
# the driver's is the python the printed argv names
f22_serving_venv
f22_client_venv
f22_log "dry run prod: sha=$WAVE_SHA bank_root=$BANK_ROOT log=$DRY_OUT/dry_run_prod.log"
# a STEERING row's fresh table is exported over ITS external bank view, <shared root>/<bank_view_rel[role]> — the role
# is this cluster's (m5t_common.sh's M5T_OVERFLOW_CLUSTER names the overflow; every other name is the primary)
CLUSTER_ROLE=primary; [ "${CLUSTER:-}" = "$M5T_OVERFLOW_CLUSTER" ] && CLUSTER_ROLE=overflow
( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$SPY" scripts/sdfbase/math500/m5t_dry_run.py --stage prod --sha "$WAVE_SHA" \
    --bank-root "$BANK_ROOT" --shared-root "$SHARED" --cluster-role "$CLUSTER_ROLE" --python "$CPY" --tmp "$DRY_OUT/tmp" ) > "$DRY_OUT/dry_run_prod.log" 2>&1
RC=$?
grep -E "^(STAGE|RENDER_FACT|PREFLIGHT_ADMITS|SERVED_POSITIONS|VERDICT|REFUSED)" "$DRY_OUT/dry_run_prod.log" || true
if [ "$RC" -eq 0 ]; then M5T_RESULT=PASS; f22_log "G-DRY-prod PASS: $DRY_OUT/dry_run_prod.log"
else M5T_RESULT=REFUSED; f22_log "G-DRY-prod REFUSED (rc=$RC): $(grep -cE '^REFUSED:' "$DRY_OUT/dry_run_prod.log" || true) refusal line(s) in $DRY_OUT/dry_run_prod.log"; fi
exit $RC
