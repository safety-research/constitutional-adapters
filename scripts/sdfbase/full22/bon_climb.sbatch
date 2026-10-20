#!/bin/bash
#SBATCH --job-name=f22-bonclimb
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-05:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22: THE CANONICAL BEST-OF-N at 4,096 for one (target, view), ONE ARRAY TASK = ONE GPU = ONE SHARD.
# The canonical best-of-N is the 512 store PLUS the 4,096 CLIMB of its cap-coherent at-cap draws —
# because the 512 reading is a FLOOR: an at-cap draw is judged jailbroken ~19x more
# often than a terminated one, so a cell with at-cap volume reads low at 512 by construction. A cell with
# NO 512 store (the Llama lane) runs best-of-N NATIVELY at 4,096 instead; the plan makes that branch
# explicit per cell (bon_climb_plan.py -> plan.json `mode`), so a missing store can never be reported as
# "zero escalations, clean". LOOP-FLAGGED DRAWS ARE NOT REGENERATED, by rule — a repeating draw repeats
# further — and are an accepted component of the residual floor.
#
# The standup is the wave's BoN standup (families_served:short = the geometry the 512 draws were served
# at — EXCEPT a job serving a LONG-SI prompted cell on a table view, which stands up at the long-SI climb window
# 10,752 after the rendered-prompt window check, and a view whose roster declares a static window, which stands
# up at it: bon_long_si.sh — at 4,864 the constitution-SI cell's every draw was refused
# prompt_over_window), the lens pinned to STACKED and refused unless attested. Outputs go to a NEW root
# ($WAVE_ROOT/bon_climb/out/<tag>/<cell>/shard<k>.jsonl): the 512 stores are never opened for writing.
# Usage (positional only — never --export=):
#   sbatch -J f22-bonclimb-<tag> --array=0-$((N-1)) scripts/sdfbase/full22/bon_climb.sbatch \
#       <key> <view> <wave_sha> <nshards> [cells=all] [cap=4096]
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; WAVE_SHA="${3:?wave sha}"; NSHARDS="${4:?nshards}"
CELLS_ARG="${5:-}"; CAP="${6:-4096}"
SHARD="${SLURM_ARRAY_TASK_ID:?this script runs as an array; SLURM_ARRAY_TASK_ID is the shard}"
[ "$SHARD" -lt "$NSHARDS" ] || { echo "FATAL: shard $SHARD >= nshards $NSHARDS"; exit 1; }
case "$CAP" in ''|*[!0-9]*) echo "FATAL: cap '$CAP' is not an integer" >&2; exit 1;; esac
# cells-fix-2's rule, the store's sibling (target_static.sbatch): a climb that serves a SUBSET of the view's cells registers that subset —
# F22_REGISTRY_CELLS narrows the registration overlay (common.sh, inside f22_resolve_config below), so the export sits here, before f22_init
[ -n "$CELLS_ARG" ] && export F22_REGISTRY_CELLS="${F22_REGISTRY_CELLS:-$CELLS_ARG}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")   # the launch's cells for the kind-aware serving-path default (2026-09-15): the (KEY, VIEW) campaign manifest
f22_init "$WAVE_SHA"
# THE CLIMB'S REGISTRATION FACTS (bon_long_si.sh): every task registers its runs at the DRAWN cap = rung 1 of the config's
# best-of-N ladder [bon.max_new_tokens, bon.climb_cap] = [512, 4096] (pass ladder_rung) — exported BEFORE f22_resolve_config registers; a
# cap positional the config's climb cap does not spell is refused by name here, never registered under a rung it is not
source scripts/sdfbase/full22/bon_long_si.sh
# THE RERUN PAIR'S CLIMB (stage.py rerun-units -> the plan row's F22_BON_RERUN=<slug>, the ruling's cells on positional 5, afterok on the
# rerun store): re-rooted to $WAVE_ROOT/rerun/<slug> HERE, before ROOT / PRIVATE_ROOT / SERVE_DIR derive from WAVE_ROOT — the climb reads the
# RERUN's 512 store and writes its plan and shards under the rerun's own bon_climb/<tag>, never beside the superseded cell's climb
# (bon_long_si.sh f22_bon_rerun_root); unset = this job byte for byte as before
f22_bon_rerun_root climb "" "$CELLS_ARG"
f22_bon_read_caps "$KEY" "$VIEW"
f22_bon_registry_exports climb "$CAP"
f22_resolve_config "$KEY" "$VIEW"
f22_bon_assert_config
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
ROOT="$WAVE_ROOT/bon_climb/$TAG"; PRIVATE_ROOT="$WAVE_ROOT/transcripts_results"
OUT_ROOT="$ROOT/out"; SERVE_DIR="$ROOT/serving/shard${SHARD}_${SLURM_JOB_ID}"
mkdir -p "$OUT_ROOT" "$SERVE_DIR/drivers"
f22_driver_venv; f22_serving_venv; f22_bank_view "$MPY"
if [ -n "$CELLS_ARG" ]; then IFS=',' read -ra CELLS <<< "$CELLS_ARG"; else CELLS=("${ALL_CELLS[@]}"); fi
# THE CLIMB'S WINDOW (bon_long_si.sh): the roster's static window when the view declares one (a per-leg / one-window
# prompted view — its static window holds BoN at 4,096 beside the static families; f22_roster.view_windows), else — a TABLE view whose job
# serves a LONG-SI prompted cell — the long-SI climb window 10,752 (aligned with the static legs' long-SI window; the BoN prompt under the
# constitution SI + the 4,096 cap needs 10,129 on qwen35_4b), else "" = the table's families_served:short 4,864, the geometry the 512
# draws were served at, byte for byte as before; the long-SI check below steps the window to the measured need if that asks
VIEW_WINDOW_STATIC=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window static --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's static window for $KEY/$VIEW"
f22_bon_detect_long_si "${CELLS[@]}"
CLIMB_L="${VIEW_WINDOW_STATIC:-$(f22_bon_window climb)}"
# THE PLAN (shard 0 writes it; every other shard waits for it): per cell, climb or native, and the climb
# manifest. One writer, so the shards share one classification of one store — of THIS attempt: shard 0 ALWAYS recomputes it (a plan
# left by an earlier attempt classified an earlier store state: the u0 wave's climb reused a plan written at 11:24Z, before its 512
# store existed, on three later attempts and refused every cell against the then-complete store — 2026-09-30), moving the previous
# pair aside; the other shards wait for a plan whose `job_id` is this array's, never for a file that merely exists.
PLAN="$ROOT/plan/plan.json"; MANIFEST_JSONL="$ROOT/plan/climb_manifest.jsonl"; THIS_JOB="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-manual}}"
if [ "$SHARD" = 0 ]; then
  if [ -s "$PLAN" ] || [ -s "$MANIFEST_JSONL" ]; then
    prev="$ROOT/plan/superseded/$(date -u +%Y%m%dT%H%M%SZ).before_$THIS_JOB"; mkdir -p "$prev"
    for f in "$PLAN" "$MANIFEST_JSONL"; do [ -e "$f" ] && mv -f "$f" "$prev/"; done
    f22_log "previous climb plan moved aside -> $prev (a plan classifies ONE store state; this attempt classifies the store as it stands now)"
  fi
  ( cd "$REPO_DIR" && "$MPY" scripts/sdfbase/full22/bon_climb_plan.py --config "$CFG" \
      --private-root "$PRIVATE_ROOT" --out "$ROOT/plan" --lane "$TAG" --job-id "$THIS_JOB" \
      --cells "$(IFS=,; echo "${CELLS[*]}")" ) || f22_fatal "bon_climb_plan failed"
fi
plan_is_ours(){ [ -s "$PLAN" ] && [ "$("$MPY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("job_id",""))' "$PLAN" 2>/dev/null)" = "$THIS_JOB" ]; }
for i in $(seq 1 240); do plan_is_ours && break; sleep 5; done
plan_is_ours || f22_fatal "no plan of THIS attempt (job $THIS_JOB) at $PLAN after 20 min (shard 0 writes it; a plan of another job is not this store's classification)"
f22_log "plan: $("$MPY" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["by_mode"])' "$PLAN")"
TARGET_PID=""; STANDUP_STOP=""; declare -a DRIVER_PIDS=(); declare -A DRIVER_TAG=(); RC=0; TOTAL_FAIL=0
cleanup(){
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  for p in "${DRIVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true; done
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  f22_result "bon-climb $TAG shard=$SHARD/$NSHARDS rc=${RC} failed_drivers=${TOTAL_FAIL}"
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
# the SAME geometry the 512 draws were served at (families_served:short = 4864 x budget-S) — at the climb's window (CLIMB_L above) for a
# long-SI cell on a table view or a roster-windowed view, after the long-SI check at THIS job's cap; the served runs file first, so the
# standup's write_endpoint hook records the attested window per run
if [ -z "$VIEW_WINDOW_STATIC" ] && [ -n "$F22_BON_LONG_SI" ]; then f22_bon_window_check "$SERVE_DIR/static" "$CLIMB_L" bon "$CAP" "${CELLS[@]}"; CLIMB_L="$F22_BON_WINDOW"; fi
f22_registry_runs_file "$SERVE_DIR/static" strongreject_bon
f22_standup attacks "$SERVE_DIR/static" "" families_served:short ${CLIMB_L:+"$CLIMB_L"} || { RC=1; exit 1; }
# the window OBSERVED in the registry spool (bon_long_si.sh): the standup's hook filed each run's patch with the attested window,
# read back from the destination the harvester pulls — a quiet hook would leave context_window_tokens NULL; refused here
f22_registry_serving_recorded "$SERVE_DIR/static"
# native-serving-default: the cells this standup SERVES (PSR / vector cells serve on the native engine's residual slot since native-psr-serving; it skips AlphaSteer / antidote / the stacked wrapper and a position-masked LoRA fit by name — a cells= positional naming a skipped cell is fatal)
f22_served_cells_filter CELLS "$CELLS_ARG"
for pc in "${CELLS[@]}"; do case "$pc" in none|prompted__*) continue;; esac; f22_positive_control "$pc" attacks || { RC=1; exit 1; }; break; done
MAX_DRIVERS=$(( ${#CELLS[@]} + 2 ))
CONC=$(( TARGET_S / ${#CELLS[@]} )); [ "$CONC" -ge 2 ] || CONC=2
# the served model under the launcher's KEY (--model-key; bon_served.served_model_identity): a key that re-pins the paper's checkpoint
# under its own model_id (the worked example) is the identity the sampler resolves for and the rows are filed under, as the 512 store's
# were; the driver reads the store's own recorded key when none is given and refuses a key the store was not drawn under. A NATIVE pass
# (no 512 store) draws at the label the recipe declared for the store -- F22_BON_DECODE_PRESET, a registry label or `canonical` (the
# canonical instrument's label, resolved by the driver) -- while a CLIMB re-draws as_drawn and ignores it; the seed-replicate
# declarations (anchor_match / vendor) are the store driver's explicit blocks and are not forwarded (the climb driver names no block).
# Unset = this argv as before plus the key (tests/test_bon_worked_example_identity.py renders the line).
BON_CLIMB_PRESET=""
case "${F22_BON_DECODE_PRESET:-}" in vendor_preset|harness_default|canonical) BON_CLIMB_PRESET="$F22_BON_DECODE_PRESET";; esac
for cell in "${CELLS[@]}"; do
  f22_stop_requested && break
  MODE=$("$MPY" -c 'import json,sys;d=json.load(open(sys.argv[1]));print(d["cells"].get(sys.argv[2],{}).get("mode","MISSING"))' "$PLAN" "$cell")
  case "$MODE" in
    climb)
      N=$("$MPY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["cells"][sys.argv[2]]["n_climb"])' "$PLAN" "$cell")
      [ "$N" -gt 0 ] || { f22_log "$cell: climb mode, 0 cap-coherent at-cap draws — nothing to climb"; continue; };;
    native_4096) f22_log "$cell: NO 512 store — best-of-N runs NATIVELY at $CAP (not an escalation)";;
    *) f22_fatal "$cell: plan says mode=$MODE";;
  esac
  f22_queue_launch "bonclimb__${cell}" "$MPY" scripts/sdfbase/full22/bon_climb_run.py \
    --config "$CFG" --defense "$cell" --mode "$MODE" --private-root "$PRIVATE_ROOT" \
    --manifest "$MANIFEST_JSONL" --shard "$SHARD" --nshards "$NSHARDS" --lane "$TAG" \
    --out "$OUT_ROOT/$cell/shard${SHARD}.jsonl" --proxy-url "$PROXY_URL" --routing-table "$TABLE" \
    --max-model-len "$TARGET_L" --max-concurrency "$CONC" --cap "$CAP" --require-engine "$TARGET_ENGINE" \
    --vllm-log "$SERVE_DIR/static/vllm_server.log" --model-key "$KEY" \
    ${BON_CLIMB_PRESET:+--bon-decode-preset "$BON_CLIMB_PRESET"}
  f22_log "launched bon-climb driver: cell=$cell mode=$MODE concurrency=$CONC"
  sleep 2
done
f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
f22_standup_stop
RC=$(( TOTAL_FAIL > 0 ? 2 : 0 ))
f22_log "shard $SHARD done: rc=$RC failed=$TOTAL_FAIL"
exit $RC
