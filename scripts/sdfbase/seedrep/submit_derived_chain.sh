#!/bin/bash
# scripts/sdfbase/seedrep/submit_derived_chain.sh — submit ONE seed key's derived-object chain on a
# Slurm cluster as afterok-dependent Slurm jobs:
#
#   prep -> calibrate (array 0-2) -> assess          --phase start   (the default)
#   [ calibrate <0.1|10> -> assess2 ]                --phase extend  --after <assess id> --extension <m>
#   finish                                           --phase finish  --after <assess|assess2 id>
#
# Every link is `--dependency=afterok:<prev>`, so no stage starts on an incomplete or failed
# predecessor (a failed link leaves its successors DependencyNeverSatisfied — the watcher's FAILED
# wake triages the link, the operator cancels the dangling successors BY NAME and resubmits from
# the failed link). The assess -> extension decision is the operator's: read <resid>/assess.json
# on the DONE wake of assess (recommended_extension 0.1 | 10 | null) and run --phase extend or
# --phase finish. Each sbatch is its own statement here — never a loop over submissions (hard rule
# 6); the calibrate array is ONE sbatch.
#
#   submit_derived_chain.sh <key> <bank> --sha <F> [--cluster cluster_a|cluster_b] [--repo <checkout>]
#       [--phase start|extend|finish] [--after <jobid>] [--extension <m>]
#       [--mix-si constv2-mix-full] [--ctl-si constv2ctl2-full] [--env-name <venv name>]
#       [--time-prep 0:40:00] [--time-calib 6:00:00] [--time-assess 0:20:00] [--time-finish 4:00:00]
#       [--rs <RS root>] [--local] [--dry-run]
#     key       qwen35_4b_base_eb64_s<s>;  bank  reseed
#     --sha     the 40-hex commit the cluster checkout sits at (the fit sha F); every job carries
#               --comment=sha:<F> and the sbatch refuses if the checkout's HEAD differs
#     --repo    the checkout on the cluster (default <the cluster's eval-repo root>/<sha>)
#     --cluster cluster_a | cluster_b, the registry label. The login host, eval-repo root, partition, QoS and
#               log directory come from the cluster's block in the site config (CLUSTER_SSH, CLUSTER_EVAL_REPO_ROOT,
#               CLUSTER_PARTITION, CLUSTER_QOS, CLUSTER_LOG_DIR; CLUSTER_ACCOUNT and CLUSTER_SBATCH_EXTRA when set);
#               every job carries --output=<log dir>/%x_%j.out
#     --env-name  SDFBASE_FIT_ENV_NAME for the jobs, passed as an env prefix on the REMOTE sbatch
#               (Slurm's implicit export; never --export=). Unset = the bank's FIT_ENV_NAME file /
#               sdfbase-fit, resolved by the sbatch itself
#     --local   run `sbatch` on this host instead of over ssh (on a login node; also the test seam)
# Job names: reseed-derived-<key>-{prep,calib,assess,calib-x<m>,assess2,finish}. Every job id is
# appended to $RS/reseed/submissions.jsonl ({job, name, key, bank, stage, lr_mult, depends_on, sha,
# cluster, qos, host, submitted_at, argv}); the LAST id of the phase is printed alone on stdout so a
# caller can chain: A=$(submit_derived_chain.sh …); …
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { sed -n '2,32p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
KEY="${1:-}"; BANK="${2:-}"; shift 2 2>/dev/null || usage 2
[ -n "$KEY" ] && [ -n "$BANK" ] || usage 2
SHA=""; CLUSTER=cluster_a; REPO=""; PHASE=start; AFTER=""; EXT=""; MIX_SI=constv2-mix-full; CTL_SI=constv2ctl2-full
ENV_NAME=""; T_PREP=0:40:00; T_CALIB=6:00:00; T_ASSESS=0:20:00; T_FINISH=4:00:00; LOCAL=0; DRY=0
RS="${RS:-}"          # default after the options: the run-state tree under the site state root ($STATE)
while [ $# -gt 0 ]; do
  case "$1" in
    --sha) SHA="$2"; shift 2 ;;
    --cluster) CLUSTER="$2"; shift 2 ;;
    --repo) REPO="$2"; shift 2 ;;
    --phase) PHASE="$2"; shift 2 ;;
    --after) AFTER="$2"; shift 2 ;;
    --extension) EXT="$2"; shift 2 ;;
    --mix-si) MIX_SI="$2"; shift 2 ;;
    --ctl-si) CTL_SI="$2"; shift 2 ;;
    --env-name) ENV_NAME="$2"; shift 2 ;;
    --time-prep) T_PREP="$2"; shift 2 ;;
    --time-calib) T_CALIB="$2"; shift 2 ;;
    --time-assess) T_ASSESS="$2"; shift 2 ;;
    --time-finish) T_FINISH="$2"; shift 2 ;;
    --rs) RS="$2"; shift 2 ;;
    --local) LOCAL=1; shift ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "[chain] unknown argument $1" >&2; usage 2 ;;
  esac
done
# the default RS root: the wave's run-state tree under the site state root ($STATE, expanded by sisteer.site from the
# environment, else the site config), refused by name when neither --rs, RS nor the site gives one
if [ -z "$RS" ]; then
  RS=$(PYTHONPATH="$(cd "$HERE/../../.." && pwd)/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.site expand '$STATE/romp/candidate-si-refit/run-state/full22' 2>/dev/null)
  RS="${RS/#\~/$HOME}"
  [ -n "$RS" ] || { echo "[chain] no RS root: pass --rs, set RS, or name the state root in the site config (SISTEER_STATE_ROOT)" >&2; exit 2; }
fi
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "[chain] --sha must be the 40-hex commit the cluster checkout sits at (got '$SHA')" >&2; exit 2; }
case "$CLUSTER" in cluster_a|cluster_b) ;; *) echo "[chain] --cluster must be cluster_a|cluster_b" >&2; exit 2 ;; esac
# The cluster's login host, eval-repo root, partition, QoS and log directory, from its block in the site config (looked up by
# the registry label). They are literal paths, RESOLVED here: the rendered command is shipped over ssh with printf %q, so an
# escaped variable would reach the remote shell unexpanded and sbatch could not open the script.
CL_PY="$(cd "$HERE/../../.." && pwd)/reproduce/lib/clusters.py"
site_get() { python3 "$CL_PY" --get "$CLUSTER" "$1"; }
HOST="$(site_get CLUSTER_SSH)" && EVAL_ROOT="$(site_get CLUSTER_EVAL_REPO_ROOT)" && QOS="$(site_get CLUSTER_QOS)" \
  && PARTITION="$(site_get CLUSTER_PARTITION)" && LOG_DIR="$(site_get CLUSTER_LOG_DIR)" && SHARED_ROOT="$(site_get CLUSTER_SHARED_ROOT)" \
  && ACCOUNT="$(site_get CLUSTER_ACCOUNT)" && SBATCH_EXTRA="$(site_get CLUSTER_SBATCH_EXTRA)" \
  && [ -n "$HOST" ] && [ -n "$EVAL_ROOT" ] && [ -n "$QOS" ] && [ -n "$PARTITION" ] && [ -n "$LOG_DIR" ] && [ -n "$SHARED_ROOT" ] \
  || { echo "[chain] the site config gives no login host, eval-repo root, partition, QoS, log dir and shared root for cluster $CLUSTER (CLUSTER_SSH, CLUSTER_EVAL_REPO_ROOT, CLUSTER_PARTITION, CLUSTER_QOS, CLUSTER_LOG_DIR, CLUSTER_SHARED_ROOT)" >&2; exit 2; }
# every job carries the site's partition, account and log file: the sbatch header names one site's values
SITE_OPT=(--partition="$PARTITION"); [ -n "$ACCOUNT" ] && SITE_OPT+=(--account="$ACCOUNT")
SITE_OPT+=(--output="${LOG_DIR%/}/%x_%j.out")
read -r -a EXTRA_OPT <<< "$SBATCH_EXTRA"; SITE_OPT+=("${EXTRA_OPT[@]}")
[ -n "$REPO" ] || REPO="${EVAL_ROOT%/}/$SHA"
SBATCH_SCRIPT="$REPO/scripts/sbatch/sdfbase_derived_build.sbatch"
case "$PHASE" in
  start) ;;
  extend) [ -n "$AFTER" ] && [ -n "$EXT" ] || { echo "[chain] --phase extend needs --after <assess id> and --extension <m>" >&2; exit 2; }
          [[ "$EXT" =~ ^[0-9.]+$ ]] || { echo "[chain] --extension must be a positive number (0.1 | 10)" >&2; exit 2; } ;;
  finish) [ -n "$AFTER" ] || { echo "[chain] --phase finish needs --after <assess|assess2 id>" >&2; exit 2; } ;;
  *) echo "[chain] --phase must be start|extend|finish" >&2; exit 2 ;;
esac
LEDGER="$RS/reseed/submissions.jsonl"
log() { echo "[chain $KEY $CLUSTER $PHASE] $*" >&2; }

# --- one submission -----------------------------------------------------------------------------------
# submit <stage> <name> <time> <lr_mult|-> <depends_on|-> [extra sbatch flags…]  -> prints the job id
submit() {
  local stage="$1" name="$2" time="$3" mult="$4" dep="$5"; shift 5
  # --qos explicit: the sbatch header's QoS is cluster A's; each cluster's comes from its site config block
  local -a cmd=(sbatch --parsable -J "$name" --qos="$QOS" --time="$time" --comment="sha:$SHA" "${SITE_OPT[@]}")
  [ "$dep" != "-" ] && cmd+=(--dependency="afterok:$dep")
  cmd+=("$@" "$SBATCH_SCRIPT" "$stage" "$KEY" "$MIX_SI" "$CTL_SI" "$BANK")
  cmd+=("$mult")                      # explicit, `-` for the array (the sbatch's own default spelling)
  # the job env the derived build reads: the cluster's shared root and QoS from its site block, and the fit venv's name when given
  local -a env_prefix=(env "SISTEER_SHARED_ROOT=$SHARED_ROOT" "SISTEER_CLUSTER_QOS=$QOS")
  [ -n "$ENV_NAME" ] && env_prefix+=("SDFBASE_FIT_ENV_NAME=$ENV_NAME")
  local rendered; rendered="$(printf '%q ' "${env_prefix[@]}" "${cmd[@]}")"
  if [ "$DRY" = 1 ]; then log "DRY: $rendered"; echo "DRY-$stage"; return 0; fi
  local raw
  if [ "$LOCAL" = 1 ]; then raw="$("${env_prefix[@]}" "${cmd[@]}")" || { log "sbatch FAILED for $name: $raw"; return 1; }
  else raw="$(ssh -o BatchMode=yes "$HOST" "$rendered" 2>&1)" || { log "sbatch FAILED for $name over ssh $HOST: $raw"; return 1; }; fi
  local jid="${raw%%;*}"; jid="${jid//[$'\r\n ']/}"
  [[ "$jid" =~ ^[0-9]+$ ]] || { log "sbatch returned no job id for $name: '$raw'"; return 1; }
  mkdir -p "$(dirname "$LEDGER")"
  python3 - "$LEDGER" "$jid" "$name" "$KEY" "$BANK" "$stage" "$mult" "$dep" "$SHA" "$CLUSTER" "$QOS" "$HOST" "$rendered" <<'PY'
import json, sys
from datetime import datetime, timezone
(ledger, jid, name, key, bank, stage, mult, dep, sha, cluster, qos, host, argv) = sys.argv[1:]
rec = {"kind": "derived", "job": int(jid), "name": name, "key": key, "bank": bank, "stage": stage,
       "lr_mult": (None if mult == "-" else mult), "depends_on": (None if dep == "-" else int(dep)),
       "sha": sha, "cluster": cluster, "qos": qos, "host": host,
       "submitted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "argv": argv.strip()}
with open(ledger, "a", encoding="utf-8") as fh:
    fh.write(json.dumps(rec, sort_keys=True) + "\n")
PY
  log "submitted $name -> job $jid${dep:+ (afterok:$dep)}"
  echo "$jid"
}

N="reseed-derived-$KEY"
case "$PHASE" in
  start)
    P="$(submit prep      "$N-prep"   "$T_PREP"   - -)"    || exit 1
    C="$(submit calibrate "$N-calib"  "$T_CALIB"  - "$P" --array=0-2)" || exit 1
    A="$(submit assess    "$N-assess" "$T_ASSESS" - "$C")" || exit 1
    log "chain start: prep $P -> calib $C (array 0-2) -> assess $A. On assess's DONE wake read <resid>/assess.json: recommended_extension 0.1|10 -> --phase extend --after $A --extension <m>; null -> --phase finish --after $A"
    echo "$A" ;;
  extend)
    E="$(submit calibrate "$N-calib-x$EXT" "$T_CALIB" "$EXT" "$AFTER")" || exit 1
    A2="$(submit assess   "$N-assess2"     "$T_ASSESS" -    "$E")"      || exit 1
    log "extension: calib-x$EXT $E (afterok:$AFTER) -> assess2 $A2. Then --phase finish --after $A2 (a second extension is refused by the sbatch, D7)"
    echo "$A2" ;;
  finish)
    F="$(submit finish "$N-finish" "$T_FINISH" - "$AFTER")" || exit 1
    log "finish $F (afterok:$AFTER); then derived_admission.py on both derived objects"
    echo "$F" ;;
esac
