#!/bin/bash
#SBATCH --job-name=hd-detfloor
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
# HALFDOSE: measure the text-difference noise floor of the served target at the TWO geometries the wave's probes ran at —
# the controls geometry (capability:short, 2048 window: where the steer_is_live counts in evidence.json come from) and the
# static geometry (families_served:short, 4864 window: where the static shards' guard ran). Same standup path as
# target_static.sbatch (f22_standup), then halfdose/determinism_floor.py against the anchor/twin/steered vmodels.
#   sbatch -J hd-detfloor-<key>-<view> scripts/sdfbase/full22/halfdose/determinism_floor.sbatch <key> <view> <wave_sha>
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; WAVE_SHA="${3:?wave sha}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")   # the launch's cells for the kind-aware serving-path default (2026-09-15): the (KEY, VIEW) campaign manifest
f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
SERVE_DIR="$WAVE_ROOT/serving/detfloor_${TAG}_${SLURM_JOB_ID}"; mkdir -p "$SERVE_DIR"
f22_driver_venv; f22_serving_venv; f22_bank_view "$MPY"
TARGET_PID=""; STANDUP_STOP=""; RC=0
cleanup(){ [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  f22_result "detfloor $TAG rc=${RC}"; }
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
for phase in "controls capability:short 2048" "static families_served:short"; do
  read -r PH GEVAL GL <<<"$phase"
  f22_standup attacks "$SERVE_DIR/$PH" "" "$GEVAL" ${GL:-} || { RC=1; f22_log "standup failed for phase $PH"; continue; }
  f22_log "phase $PH: determinism floor at geometry ${TARGET_S}x${TARGET_L}"
  "$MPY" scripts/sdfbase/full22/halfdose/determinism_floor.py --proxy-url "$PROXY_URL" --routing-table "$TABLE" \
    --out "$SERVE_DIR/$PH/determinism_floor.json" --repeats 3 2>&1 | tee -a "$SERVE_DIR/$PH/determinism_floor.log" | grep -E "DETERMINISM_FLOOR|Error|Traceback" | sed "s|^|[$PH ${TARGET_S}x${TARGET_L}] |"
  [ -f "$SERVE_DIR/$PH/determinism_floor.json" ] || RC=1
  f22_standup_stop
done
exit $RC
