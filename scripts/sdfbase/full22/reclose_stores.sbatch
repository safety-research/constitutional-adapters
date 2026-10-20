#!/bin/bash
#SBATCH --job-name=f22-reclose
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22 — RECLOSE: re-run a finished eval job's store closer under the driver venv.
# CPU-only: no --gres=gpu:1 here — the closer reads stores and writes records; it serves nothing (tests/test_sdfbase_full22_wave.py exempts
# this one job from the one-GPU server rule). The header above is the wave's required one (append mode so a retry never clobbers the log;
# no-requeue; the TERM signal two minutes before the walltime); the funnel's sbatch line repeats --qos/--partition/--time/--output/-J per site.
# An eval job whose closer REFUSED leaves with exit 75 (common.sh f22_result) and a RECIPE beside its pending file —
# $SHARED/registry/pending/<job>.closer_refused.json: the pending path, spool, push script, wave root, tag, nshards, code sha, captured from the
# job's own environment, nothing re-derived. The funnel (`submit_twin.py reclose --job J`; 06's --resubmit walk on exit 75) submits THIS job,
# CPU-only, at the recipe's sha: it confirms the driver venv by the same f22_driver_venv path the eval job takes (a venv that no longer exists
# refuses by name — never bare python), re-runs the closer from the recipe, pushes the ORIGINAL job's records, and appends its outcome to the
# recipe's `recloses` (the funnel caps at 3 and stops when the same refusal survives a reclose). No GPU, no re-run of the eval.
# Usage (positional only — never --export=):  sbatch -J f22-reclose-<job> [--qos=<qos> --output=...] \
#     scripts/sdfbase/full22/reclose_stores.sbatch <original job id> <wave sha>
set -uo pipefail
ORIG="${1:?original job id}"; WAVE_SHA="${2:?wave sha}"; CLOSER_SHA="${3:-}"   # reclose-at-tree: the closer's tree when it is not the job's
CLOSER_EXTRA="${4:-}"                                                           # The closer's extra flags (one shell-quoted string; submit_twin reclose --closer-arg)
CLOSER_EXTRA_ARGS=(); [ -z "$CLOSER_EXTRA" ] || eval "CLOSER_EXTRA_ARGS=($CLOSER_EXTRA)"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait
F22_LAUNCH_CELLS=()   # this job stands up no target (the kind-aware serving-path default, as attacker_pool.sbatch)
# THE RECIPE, before f22_init (its wave root rides F22_WAVE_ROOT into f22_init's deferred branch); the shared root is derived as f22_init derives it
_shared="${SISTEER_SHARED_ROOT:-}"
[ -n "$_shared" ] || f22_fatal "reclose $ORIG: no shared root: set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)"
RECIPE="$_shared/registry/pending/$ORIG.closer_refused.json"
[ -f "$RECIPE" ] || f22_fatal "reclose $ORIG: no recipe at $RECIPE — the original job's closer refusal wrote none; close by hand under the driver venv"
eval "$(python3 -S -c 'import json, shlex, sys
d = json.load(open(sys.argv[1]))
for k in ("pending", "spool", "push_script", "wave_root", "tag", "nshards", "job_name", "code_sha", "why"):
    print("R_" + k.upper() + "=" + shlex.quote(str(d.get(k) or "")))' "$RECIPE")" || f22_fatal "reclose $ORIG: recipe $RECIPE is not readable JSON"
# this reclose's outcome, in ITS OWN receipt file beside the recipe — $_shared/registry/pending/<orig>.reclose.<this job>.receipt.json, written once,
# atomically, by this job alone (the recipe is immutable after the eval job writes it; a receipt appended onto it raced the controller's
# retry-note writer and the loser's write erased the other's record). The funnel reads every receipt file: rc 0
# closes the matter; the same why twice stops the automation; a `stopped` outcome — the job left under the wave's STOP — counts toward nothing;
# a `preflight` outcome — the recipe or the environment refused BEFORE the closer ran — counts toward nothing and holds until the input changes
reclose_record_outcome(){   # $1 = rc, $2 = why, $3 = stopped (0|1), [$4 = preflight (0|1)]
  RECORDED=1
  R_RECIPE="$RECIPE" R_RECEIPT="$_shared/registry/pending/$ORIG.reclose.${SLURM_JOB_ID:-manual}.receipt.json" R_JOB="${SLURM_JOB_ID:-manual}" R_RC="$1" R_WHY2="$2" R_STOPPED="$3" R_PREFLIGHT="${4:-0}" R_INPUT_PENDING="${R_PENDING:-}" \
  R_INPUT_CHECKOUT="${SISTEER_EVAL_REPO_ROOT:-$_shared/sisteer-eval-repo}/$WAVE_SHA" R_JOB_SHA="$WAVE_SHA" R_CLOSER_SHA="${CLOSER_SHA:-}" R_CLOSER_EXTRA="${CLOSER_EXTRA:-}" python3 -S - <<'PY' || f22_log "reclose: could not record the outcome beside $RECIPE"
import json, os, time
p = os.environ["R_RECIPE"]
with open(p) as fh:
    d = json.load(fh)                      # read-only: the recipe identity below; this job never writes the recipe
rec = {"job_id": os.environ["R_JOB"], "rc": int(os.environ["R_RC"]), "why": os.environ["R_WHY2"], "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "job_sha": os.environ.get("R_JOB_SHA") or None, "closer_sha": os.environ.get("R_CLOSER_SHA") or os.environ.get("R_JOB_SHA") or None,   # reclose-at-tree: both trees, always
       "closer_extra_args": os.environ.get("R_CLOSER_EXTRA") or None}                                                                            # D1168: the closer's extra flags, verbatim
if os.environ.get("R_STOPPED") == "1":
    rec["stopped"] = True
if os.environ.get("R_PREFLIGHT") == "1":
    rec["preflight"] = True
    # closer-fix-6 (a HIGH review finding on c042ab002, D795): the INPUT this preflight saw — the funnel compares it with what it sees later: a repaired input
    # (pending file restored, checkout restored, recipe fixed) is new evidence for one more reclose; the same input is not. The recipe identity is
    # the twin of submit_twin.recipe_identity (every key but recloses / retries / path, canonical JSON, sha256).
    import hashlib
    body = {k: v for k, v in d.items() if k not in ("recloses", "retries", "path")}
    pend = os.environ.get("R_INPUT_PENDING", ""); co = os.environ.get("R_INPUT_CHECKOUT", "")
    rec["input"] = {"recipe_sha256": hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest(),
                    "pending": hashlib.sha256(open(pend, "rb").read()).hexdigest() if pend and os.path.isfile(pend) else "absent",
                    "checkout": "present" if co and os.path.isdir(os.path.join(co, "src", "sisteer")) else "absent"}
out = os.environ["R_RECEIPT"]; tmp = out + ".tmp." + str(os.getpid())
with open(tmp, "w") as fh:
    json.dump(rec, fh, indent=1, sort_keys=True); fh.write("\n"); fh.flush(); os.fsync(fh.fileno())
os.replace(tmp, out)
PY
}
# closer-fix-5: the receipt is armed BEFORE the recipe is validated — a recipe the eval job wrote without its wave
# root or with a bogus pending path used to kill this job here, before any trap, and the funnel's durable cap counted the receipt-less death as a
# spent reclose; now such a refusal is RECORDED on the recipe (rc 1, `preflight`) and the funnel stops on it instead of spending the cap blind
RC=1; WHY="reclose $ORIG: died before its preflight finished"; RECORDED=0
preflight_cleanup(){ trap '' SIGTERM; [ "$RECORDED" = 1 ] || reclose_record_outcome "$RC" "$WHY" 0 1; }
trap preflight_cleanup EXIT SIGTERM SIGINT SIGQUIT
reclose_refuse(){ WHY="reclose $ORIG: $*"; f22_fatal "reclose $ORIG: $*"; }     # f22_fatal exits 1; the EXIT trap writes the preflight receipt first
[ "$R_CODE_SHA" = "$WAVE_SHA" ] || reclose_refuse "the recipe's code sha $R_CODE_SHA is not this job's $WAVE_SHA — the closer runs on the tree the job ran"
case "$R_WAVE_ROOT" in /*) ;; *) reclose_refuse "the recipe carries no absolute wave_root ('${R_WAVE_ROOT:-}') — written by a closer whose wave root never reached the recipe writer (before closer-fix-5); the wave's STOP cannot be honoured; fix the recipe by hand";; esac
[ "$R_PENDING" = "$_shared/registry/pending/$ORIG.json" ] || reclose_refuse "the recipe's pending path ('${R_PENDING:-}') is not this job's pending file $_shared/registry/pending/$ORIG.json — written by a closer whose shared root never reached the recipe writer (before closer-fix-5); fix the recipe by hand"
[ -f "$R_PENDING" ] || reclose_refuse "the recipe's pending file $R_PENDING is gone"
export F22_WAVE_ROOT="$R_WAVE_ROOT"
[ -n "$R_SPOOL" ] && export SISTEER_REGISTRY_SPOOL="$R_SPOOL"
[ -n "$R_PUSH_SCRIPT" ] && export SISTEER_REGISTRY_PUSH="$R_PUSH_SCRIPT"
WHY="reclose $ORIG: f22_init refused (the FATAL line above)"
f22_init "$WAVE_SHA"
RC=0; WHY=""
# an interruption after f22_init (TERM at the walltime, a death in f22_driver_venv or the push) is never a success: the receipt says rc 1
cleanup(){ trap '' SIGTERM; if [ "$RECORDED" != 1 ]; then RC=1; reclose_record_outcome 1 "reclose $ORIG: interrupted before its outcome was recorded (TERM / walltime / a death before the closer answered)" 0; fi; f22_result "reclose orig=$ORIG rc=${RC}"; }      # disarm first: a TERM arriving mid-handler must not re-enter it
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
# closer-fix-4: the wave's STOP is honoured AFTER the trap is armed and RECORDED — a bare exit 0 before the trap left a
# terminal job without a receipt, which the funnel's durable cap counted as a failed reclose; three walks under STOP exhausted the cap
if f22_stop_requested; then
  f22_log "wave STOP present — not reclosing; recorded as stopped (the funnel counts it toward nothing and refuses to submit while the STOP stands)"
  reclose_record_outcome 0 "wave STOP stands" 1
  exit 0
fi
f22_driver_venv      # the SAME path the eval job takes: F22_PY is the confirmed venv; a missing venv refuses by name, never bare python
# reclose-at-tree: the CLOSER may come from another landed tree (a closer fix that post-dates the job); the job's tree stays the
# recipe's identity (the refusal above), the runs' --code-sha stays the job's, and the receipt records both. A closer tree without the closer
# script or src/sisteer refuses by name — never a fallback to the job's tree under a flag that asked for another.
CLOSER_DIR="$REPO_DIR"
if [ -n "$CLOSER_SHA" ]; then
  CLOSER_DIR="${SISTEER_EVAL_REPO_ROOT:-$_shared/sisteer-eval-repo}/$CLOSER_SHA"
  [ -f "$CLOSER_DIR/scripts/sdfbase/full22/registry_close_stores.py" ] && [ -d "$CLOSER_DIR/src/sisteer" ] \
    || reclose_refuse "--tree $CLOSER_SHA: no checkout with the closer at $CLOSER_DIR (scripts/sdfbase/full22/registry_close_stores.py + src/sisteer) — land and mirror it first"
  f22_log "reclose-at-tree: the closer runs from $CLOSER_DIR (tree $CLOSER_SHA); the job's tree $WAVE_SHA stays the recipe's identity and the runs' code sha"
fi
f22_log "reclose: job $ORIG ($R_JOB_NAME) refused its closer — ${R_WHY:0:200}; re-running registry_close_stores.py under $F22_PY (tag=$R_TAG nshards=$R_NSHARDS wave_root=$R_WAVE_ROOT)"
out=$(PYTHONPATH="$CLOSER_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$F22_PY" "$CLOSER_DIR/scripts/sdfbase/full22/registry_close_stores.py" --pending "$R_PENDING" ${R_SPOOL:+--spool "$R_SPOOL"} --code-sha "$WAVE_SHA" \
      ${R_WAVE_ROOT:+--wave-root "$R_WAVE_ROOT"} ${R_TAG:+--tag "$R_TAG"} ${R_JOB_NAME:+--job-name "$R_JOB_NAME"} ${R_NSHARDS:+--nshards "$R_NSHARDS"} ${R_PUSH_SCRIPT:+--push-script "$R_PUSH_SCRIPT"} \
      ${CLOSER_EXTRA_ARGS[@]+"${CLOSER_EXTRA_ARGS[@]}"} 2>&1) || RC=$?
[ -n "$out" ] && printf '%s\n' "$out" | sed 's/^/[registry] /'
WHY=$(printf '%s\n' "$out" | grep -m1 -E "REFUSED|Traceback|Error" | cut -c1-300)
# the ORIGINAL job's final push (its runs' records), whatever the closer said
[ -n "${SISTEER_REGISTRY_SPOOL:-}" ] && PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" python3 -S -m sisteer.registry.emit_cli push --pending-dir "$(dirname "$R_PENDING")" --job "$ORIG" --final 2>/dev/null || true
reclose_record_outcome "$RC" "$WHY" 0
if [ "$RC" -eq 0 ]; then f22_log "reclose: the closer FILED job $ORIG's stores (rc 0) — its runs advance; the gate reads them"
else f22_log "reclose: the closer REFUSED again for job $ORIG (rc $RC): ${WHY:-see the [registry] lines above}"; fi
exit "$RC"
