#!/bin/bash
#SBATCH --job-name=g8k-dry-prod
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
# gsm8k-paired200-paper-objects — G-DRY-prod: the PRODUCTION dry run, run ON THE CLUSTER where the bank
# views exist, as a CPU-only job (no GPU). Per matrix row (g8k_cells.py over the frozen matrix; held rows printed as HELD
# with no argv) g8k_dry_run.py --stage prod prints the verbatim argv, the family's render fact (the production preflight
# seam must ADMIT every family: certified x4), and the FRESH routing table's lens_served_positions stamp for every fitted
# row against the row's EXPECTED positions (`all` on the all-position objects, `response` on the header-located scale-12
# objects), exported over the live bank views of the row's VIEW by the standup's own cell selection + export_routing_table
# — hence the serving venv is bootstrapped by name and runs the tool; the client venv is named as the driver's interpreter
# in the printed argv. A DECLARED row is exported under its `<cell>@thinking` spelling, so its stamp is looked up under the
# variant's harness id (`si_…@thinking`). Exit 0 = PASS (the RESULT line is the gate record that admits
# G-LAUNCH); 2 = a refusal, printed with REFUSED:. No row is an expected refusal since 2026-09-10 (gemma's two
# header-located rows are servable thinking-ON on the declared path — the live acceptance check is a gemma
# standup of safety_refusal__psr_kl_all@thinking). The grading host's --stage pre never admits anything.
# Usage (positional only — never --export=):
#   sbatch -J g8k-dry-prod scripts/sdfbase/gsm8k/dry_run_prod.sbatch <wave_sha> [matrix]     # matrix: the `case` below (g8k_cells.MATRICES)
set -uo pipefail
WAVE_SHA="${1:?wave sha}"; G8K_MATRIX="${2:-default}"
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
RC=1; G8K_RESULT=INCOMPLETE; DRY_OUT=""
cleanup(){
  [ -z "${DRY_OUT:-}" ] || [ -z "${FIGS_DIR:-}" ] || cp "$DRY_OUT/dry_run_prod.log" "$FIGS_DIR/dry_run_prod_${SLURM_JOB_ID:-manual}.log" 2>/dev/null || true
  local rec="g8k-dry-prod rc=$RC result=$G8K_RESULT sha=$WAVE_SHA matrix=$G8K_MATRIX cluster=${CLUSTER:-?} log=${DRY_OUT:-unresolved}/dry_run_prod.log"
  if [ -n "${FIGS_DIR:-}" ]; then f22_result "$rec"; else echo "RESULT (before the wave root resolved; no figs dir yet): $rec" >&2; fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
g8k_init "$WAVE_SHA"
f22_stop_requested && { f22_log "wave STOP present — not running the dry run"; RC=0; G8K_RESULT=STOP_REQUESTED; exit 0; }
DRY_OUT="$WAVE_ROOT/dry_run_prod/${SLURM_JOB_ID:-manual}"; mkdir -p "$DRY_OUT"
BANK_ROOT="$SHARED/exp/objects/sdfbase/full22"       # <root>/<key>/cells_view/<view> per (key, view): the check-only views the jobs read
[ -d "$BANK_ROOT" ] || f22_fatal "no bank views root at $BANK_ROOT — the fresh-table check runs where the views exist"
# an EXTERNAL-object row exports its fresh table over ANOTHER LANE's bank view, named on the row
# RELATIVE to the shared root (`bank_view_rel` per cluster role) — the dry run resolves it under --shared-root, this cluster's
# the two interpreters the production jobs use: the standup's (the fresh-table export IS the standup's code) runs the tool,
# the driver's is the python the printed argv names
f22_serving_venv
f22_client_venv
f22_log "dry run prod: sha=$WAVE_SHA cluster=$CLUSTER bank_root=$BANK_ROOT log=$DRY_OUT/dry_run_prod.log"
# on the overflow cluster the `primary` rows are not this cluster's to run: the dry run is told the cluster so the
# placement refusal is exercised on the rows that would be refused here, and the verdict covers the rows this cluster carries
( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$SPY" scripts/sdfbase/gsm8k/g8k_dry_run.py --stage prod --sha "$WAVE_SHA" --matrix "$G8K_MATRIX" \
    --cluster "$CLUSTER" --bank-root "$BANK_ROOT" --shared-root "$SHARED" --python "$CPY" --tmp "$DRY_OUT/tmp" ) > "$DRY_OUT/dry_run_prod.log" 2>&1
RC=$?
grep -E "^(STAGE|RENDER_FACT|PREFLIGHT_ADMITS|SERVED_POSITIONS|HELD|EXPECTED_REFUSAL|PLACEMENT|VERDICT|REFUSED)" "$DRY_OUT/dry_run_prod.log" || true
if [ "$RC" -eq 0 ]; then G8K_RESULT=PASS; f22_log "G-DRY-prod PASS: $DRY_OUT/dry_run_prod.log"
else G8K_RESULT=REFUSED; f22_log "G-DRY-prod REFUSED (rc=$RC): $(grep -cE '^REFUSED:' "$DRY_OUT/dry_run_prod.log" || true) refusal line(s) in $DRY_OUT/dry_run_prod.log"; fi
exit $RC
