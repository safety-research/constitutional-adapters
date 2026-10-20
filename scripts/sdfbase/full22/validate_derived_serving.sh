#!/bin/bash
# Stand up ONE full22 target server on ONE GPU and prove the two DERIVED cells are
# really served (not silently no-op'd). Run under an INTERACTIVE dev allocation —
# `dev` is srun-only, never sbatch:
#
#   srun -p dev,overflow --qos=dev --gres=gpu:1 --cpus-per-task=16 --mem=200G \
#        --time=2:00:00 --job-name=D_$USER_derivserve \
#        bash $SISTEER_EVAL_REPO_ROOT/<sha>/scripts/sdfbase/full22/validate_derived_serving.sh <key> <view> <sha> <cells>
#
# Everything it touches is read-only except its own scratch dir under
# $WAVE_ROOT/validate/. It writes derived_serving_report.json and exits nonzero
# if any check fails.
#   A fifth argument `slice` restricts the standup to `none` + the named cells,
#   through make_validation_slice.py (a scratch roster/config/manifest triple and
#   its own cells_view/validate dir). Use it when an UNRELATED cell of the view
#   cannot be staged — as of 2026-09-06 qwen35_4b/main cannot build at all,
#   because its two si12__safety_refusal__* copies are staged with no
#   train_budget_meta.json. Slicing keeps a serving change verifiable against the
#   real objects on the real path while that staging gap is fixed elsewhere.
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; WAVE_SHA="${3:?wave sha}"
CELLS="${4:?comma-separated cell ids}"; MODE="${5:-full}"
source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
# THE SERVING PATH (the kind-aware default, 2026-09-15): the cell source is the standup's ACTUAL manifest.
#   full  — the whole (KEY, VIEW) campaign view: declared here, selected at f22_init, WAVE_ROOT the campaign's (unchanged).
#   slice — `none` + the named cells ONLY: that manifest is GENERATED below, under OUT, so the selection is DEFERRED and bound
#           from the generated slice manifest after it exists (f22_bind_serving_path). OUT must therefore not follow the
#           path: this script OWNS a scratch root (exp/sdfbase_full22_validate — its outputs are its own report, nothing
#           reads them by root), so OUT="$WAVE_ROOT/validate/..." below stays valid before the path is known. Before this,
#           the parent view's ten cells selected native for a lens-only slice, the sliced PSR cell was skipped and the
#           validation never exercised it. An explicit F22_SERVING_PATH is bound at f22_init in either mode and wins.
if [ "$MODE" = slice ]; then
  shared="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"   # needed before f22_init
  export F22_WAVE_ROOT="$shared/exp/sdfbase_full22_validate"; F22_SERVING_PATH_DEFER=1
else
  F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")
fi
f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"

OUT="$WAVE_ROOT/validate/${TAG}_${SLURM_JOB_ID:-manual}"
mkdir -p "$OUT"
RC=0; TARGET_PID=""
cleanup(){ touch "$OUT/STOP" 2>/dev/null || true
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  f22_log "validation rc=$RC (report: $OUT/derived_serving_report.json)"; }
trap cleanup EXIT SIGTERM SIGINT SIGQUIT

TARGET_GPU="$CUDA_VISIBLE_DEVICES"
f22_driver_venv; f22_serving_venv

if [ "$MODE" = slice ]; then
  SLICE_VIEW=validate
  "$MPY" scripts/sdfbase/full22/make_validation_slice.py --target "$KEY" --view "$VIEW" \
    --slice-view "$SLICE_VIEW" --cells "$CELLS" --out "$OUT/slice" > "$OUT/slice.json" \
    || f22_fatal "slice generation refused"
  cat "$OUT/slice.json"
  CFG="$OUT/slice/slice_config.yaml"; MANIFEST="$OUT/slice/slice_manifest.yaml"
  # the serving path, bound from the GENERATED slice manifest — the exact cells this standup serves (a lens-only slice of a
  # mixed view selects the lens; an explicit F22_SERVING_PATH was bound at f22_init and is left alone)
  [ "${F22_SERVING_PATH_MODE:-}" != deferred ] || f22_bind_serving_path "$MANIFEST"
  "$MPY" scripts/sdfbase/full22/make_bank_view.py --target "$KEY" --view "$SLICE_VIEW" \
    --roster "$OUT/slice/slice_roster.yaml" \
    || f22_fatal "slice bank view for $KEY/$SLICE_VIEW refused"
  export TT_TWO_SI_DEFENSES_DIR="$SHARED/exp/objects/sdfbase/$WAVE/$KEY/cells_view/$SLICE_VIEW"
  mapfile -t ALL_CELLS < <("$MPY" - "$CFG" <<'PY'
import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["explicit_cells"]))
PY
)
  f22_log "SLICE view $SLICE_VIEW: ${ALL_CELLS[*]} (bank $TT_TWO_SI_DEFENSES_DIR)"
else
  f22_bank_view "$MPY"
fi
f22_log "cells in $CFG: ${ALL_CELLS[*]}"

# the standup itself asserts the wave's stacked mode from the authoritative
# stamps and refuses traffic otherwise (f22_standup re-reads endpoint.json AND
# /v1/models); getting past this line is already part of the evidence
f22_standup attacks "$OUT" "" || { RC=1; exit 1; }

"$SPY" "$REPO_DIR/scripts/sdfbase/full22/validate_derived_serving.py" \
  --endpoint "$OUT/endpoint.json" --cells "$CELLS" --repo "$REPO_DIR" \
  --out "$OUT/derived_serving_report.json"
RC=$?
f22_log "validator exit $RC"
exit $RC
