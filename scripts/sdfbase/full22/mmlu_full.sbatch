#!/bin/bash
#SBATCH --job-name=mmluF
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-02:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# mmlu_full.sbatch — the served MMLU letter-prefill read over the WHOLE pinned pool (1,531 items) for every cell
# of ONE (key, view), in ONE job, into ITS OWN root
# $WAVE_ROOT/mmlu_full/<TAG>_t<T>/ — never the 200-item roots (sampled/mmlu/, controls/), whose stores it must never
# touch, resume or pool with (served_capability_run.py --mmlu-full: a DIFFERENT store identity by construction —
# resolved_config.mmlu.item_set; the 200 are the 1,531's first rows item for item, so the widths are nested, never pooled).
#
# THE INSTRUMENT IS target_static.sbatch's CONTROLS PHASE, byte for byte apart from the width: the same standup
# (capability:short, --max-logprobs = the driver's MMLU_TOP_LOGPROBS read by import, thinking OFF), the same driver
# argv (--controls mmlu, --top-logprobs, --concurrency TARGET_S / NCELLS), the same steer-is-live positive control.
# The sampled lane's capability_t<T> phase is NOT used because its scope refuses views this panel needs
# (nemotron3nano4b/halfdose, the paired views), and its `mmlu/` root is the 200-item stores'.
#
# LONG-SI cells (prompted__si_constitution / prompted__si_irrelevant: a ~6k-token system turn) do not fit
# capability:short's 2,048 window, and the sampled lane REFUSES them for MMLU on a table view (its 6,656 was measured
# for xstest / or_bench prompts, not MMLU's). Here they get a SECOND standup of their own, after the short-window
# cells, at the view's roster controls window when it declares one (llama8b/shortconst: 6,656, measured for the MMLU
# prefill + its SI) else MMLU_FULL_LONG_SI_WINDOW (default 12,288; a scalar in the submitter's environment, which
# Slurm's implicit export carries — never --export=). The prefill-only read spends nothing on decode, so the smaller
# S at the wider window costs minutes; a prompt that overflows is REFUSED by vLLM (400 -> a failed driver, counted),
# never clamped.
#
# THE JOB NAME (mmluF-*) deliberately matches NO kind in common.sh f22_registry_evals_for_job: for the paper's own tags
# (submit_full22.sh, no F22_REGISTRY_EVALS) the job registers NOTHING here — a self-registration under the 200-item `mmlu`
# identity would file a wider run under the narrower one; their stores are self-describing (resolved_config.mmlu.item_set,
# manifest.mmlu_item_set with the ids, gen_stats.mmlu.item_set) and are ingested deliberately by the registry's owner.
# A FRONT-DOOR key (reproduce/models.yaml; the registry native at every step)
# registers NATIVELY instead: F22_REGISTRY_EVALS=mmlu (set by the front door's plan row, or here for a table key) makes
# f22_resolve_config's f22_registry_register mint the run through the registry's own planner, which for those models
# resolves `mmlu` to the canonical 1,531-item instrument (configs/registry: mmlu_letter_prefill_full_1531 / item set
# mmlu_1531 — the ONE canonical row for them; the 200-item row does not bind to them). The job then REFUSES, by name, a
# pending identity on any other item set, exports the pending file as the store hook's identity (every CellStore below
# routes its rows onto that run: item rows + the aggregate metric, the store uri and the serving facts), and files the
# run's TERMINAL run_state itself at exit (mmlu_full_registry_end: the served driver marks no completion the hook
# could read, so `generated` / `failed` / `cancelled` come from the job — the gsm8k row's precedent).
#
# Usage (positional only — never --export=):
#   sbatch -J mmluF-<key>-<view> scripts/sdfbase/full22/mmlu_full.sbatch <key> <view> <wave_sha> [cells=all] [trial=0]
#   (cluster B: sbatch --qos=<its high QoS> --output=$SHARED/exp/logs/%x_%j.out ...)
#   F22_ROSTER=<roster> in the submitter's environment selects a non-production roster (common.sh), as every lane does.
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; WAVE_SHA="${3:?wave sha}"; CELLS_ARG="${4:-}"; TRIAL="${5:-0}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")   # the kind-aware serving-path default: this job serves the wave's (KEY, VIEW) manifest
f22_init "$WAVE_SHA"
# --- registry (D65) begin: a FRONT-DOOR key registers natively under the 1,531-item instrument; the paper's tags keep the guard ---
MMLU_FULL_ITEM_SET=mmlu_1531   # configs/registry/item_sets.yaml: the item set of mmlu_letter_prefill_full_1531 (tests hold them equal)
# a FRONT-DOOR key = a models-table model with PHASE views (reproduce/lib/models.py front-door; make_full22_configs.front_door_models'
# rule — the paper's own keys sit in the table for their pins with `main` / `it` only and are NOT front-door keys)
if [ -z "${F22_REGISTRY_EVALS:-}" ] && python3 reproduce/lib/models.py front-door "$KEY" 2>/dev/null; then export F22_REGISTRY_EVALS=mmlu; fi
mmlu_full_registry_check(){   # after f22_resolve_config: with a registration REQUESTED, the pending identity MUST be the full-width one
  [ -n "${F22_REGISTRY_EVALS:-}" ] || return 0             # the paper's own tags: nothing registered, nothing to check (the guard)
  local pend="$SHARED/registry/pending/${SLURM_JOB_ID:-none}.json"
  [ -n "${SISTEER_REGISTRY_SPOOL:-}" ] || f22_fatal "front-door key $KEY: F22_REGISTRY_EVALS=$F22_REGISTRY_EVALS but no registry spool on this cluster (SISTEER_REGISTRY_SPOOL) — the gate binds this unit from its native registration, so the job stops here"
  [ -f "$pend" ] || f22_fatal "front-door key $KEY: the native registration under F22_REGISTRY_EVALS=$F22_REGISTRY_EVALS wrote no pending file at $pend (the [registry] lines above name the refusal) — the gate binds this unit from that file, so the job stops here"
  python3 - "$pend" "$MMLU_FULL_ITEM_SET" <<'PY' || f22_fatal "the registered identity is not the full-width instrument's (above): a 1,531-item run is never filed under a narrower item set"
import json, sys
d = json.load(open(sys.argv[1])); runs = d.get("runs") or [d]
bad = [(r.get("run_id"), (r.get("plan") or {}).get("item_set_id")) for r in runs if (r.get("plan") or {}).get("item_set_id") != sys.argv[2]]
if bad:
    sys.exit(f"[registry] mmlu_full: registered run(s) on another item set: {bad} (wanted {sys.argv[2]})")
print(f"[registry] mmlu_full: {len(runs)} run(s) registered natively on item set {sys.argv[2]}: " + ", ".join(str(r.get('run_id')) for r in runs))
PY
}
# --- registry (D65) end ---
f22_resolve_config "$KEY" "$VIEW"
mmlu_full_registry_check
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
VIEW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" controls --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's controls list for $KEY/$VIEW"
case ",$VIEW_CONTROLS," in *,mmlu,*) ;; *) f22_fatal "view $KEY/$VIEW runs no mmlu control (roster controls: $VIEW_CONTROLS)";; esac
VIEW_WINDOW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window controls --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's controls window for $KEY/$VIEW"
TAGT="${TAG}_t${TRIAL}"
SERVE_DIR="$WAVE_ROOT/serving/mmlu_full_${TAGT}_${SLURM_JOB_ID:-manual}"; OUT_ROOT="$WAVE_ROOT/mmlu_full/$TAGT"
mkdir -p "$SERVE_DIR/drivers" "$OUT_ROOT"
f22_log "mmlu_full: view=$VIEW tag=$TAG trial=$TRIAL out=$OUT_ROOT serve_dir=$SERVE_DIR roster_controls_window=${VIEW_WINDOW_CONTROLS:-none}"
f22_driver_venv; f22_serving_venv; f22_bank_view "$MPY"
if [ -n "$CELLS_ARG" ]; then IFS=',' read -ra CELLS <<< "$CELLS_ARG"; else CELLS=("${ALL_CELLS[@]}"); fi
for c in "${CELLS[@]}"; do case " ${ALL_CELLS[*]} " in *" $c "*) ;; *) f22_fatal "cell $c is not in $CFG's explicit_cells (${ALL_CELLS[*]}) — a job serves only cells the config names";; esac; done
# ONE number for the server's --max-logprobs AND the driver's --top-logprobs (the wave's lesson, 447405): read from the driver by import
CTL_TOP_LOGPROBS=$("$MPY" -c 'import importlib.util,sys; s=importlib.util.spec_from_file_location("scr", sys.argv[1]); m=importlib.util.module_from_spec(s); sys.modules["scr"]=m; s.loader.exec_module(m); print(int(m.MMLU_TOP_LOGPROBS))' scripts/served_capability_run.py) \
  || f22_fatal "cannot read MMLU_TOP_LOGPROBS from scripts/served_capability_run.py"
# the opt-in flag by CONSTANT (contracts.FLAG_MMLU_FULL), never a retyped literal: a checkout without it is refused by name
MMLU_FULL_FLAG=$("$MPY" -c 'from sisteer.capeval.tinycap import contracts as tc; print(tc.FLAG_MMLU_FULL)') \
  || f22_fatal "cannot read contracts.FLAG_MMLU_FULL under $MPY — this checkout predates the full-pool opt-in (mmlu-full-1531)"
f22_standup_extra_reset --enable-log-requests
# the two cell classes: SHORT (every cell but a long-SI prompted one) at the wave's controls geometry; LONG-SI at the wide window
LONG_SI_CELLS=(prompted__si_constitution prompted__si_irrelevant)   # common_sampled.sh F22S_LONG_SI_CELLS
SHORT=(); LONG=()
for c in "${CELLS[@]}"; do case " ${LONG_SI_CELLS[*]} " in *" $c "*) LONG+=("$c");; *) SHORT+=("$c");; esac; done
MMLU_FULL_LONG_SI_WINDOW="${MMLU_FULL_LONG_SI_WINDOW:-12288}"
TARGET_PID=""; STANDUP_STOP=""; declare -a DRIVER_PIDS=(); declare -A DRIVER_TAG=(); RC=0; TOTAL_FAIL=0
# registry: the natively registered run's TERMINAL run_state, from the job itself — the served driver marks no completion the
# store hook could read (it never writes results.json), and f22_result's first-word rule reads FAILED/PREEMPTED/CANCELLED records,
# not `mmlu_full …` ones; a no-op before the job registered (SISTEER_REGISTRY_RUN_IDENTITY unset — the paper's own tags)
mmlu_full_registry_end(){   # $1 = RC, $2 = the trap's exit code
  [ -n "${SISTEER_REGISTRY_RUN_IDENTITY:-}" ] || return 0
  if f22_stop_requested; then f22_registry_state cancelled CANCELLED final 2>/dev/null || true
  elif [ "${1:-1}" = 0 ] && [ "${2:-1}" = 0 ]; then f22_registry_state generated COMPLETED final 2>/dev/null || true
  else f22_registry_state failed "rc=${1:-?} exit=${2:-?} failed_drivers=${TOTAL_FAIL:-?}" final 2>/dev/null || true; fi
}
# the wave's cleanup shape (target_static.sbatch; tests/test_sdfbase_full22_wave.py holds every full22 sbatch to it): stop the
# standup, TERM the drivers by pid (never a group kill, so the handler cannot re-enter itself), wait the server out, then the
# RESULT line rides out through the figure mirror
cleanup(){
  local ec=$?; [ "$ec" != 0 ] && [ "${RC:-0}" = 0 ] && RC=$ec
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  for p in "${DRIVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true; done
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  mmlu_full_registry_end "$RC" "$ec"       # the run's terminal record, before f22_result's final push
  f22_result "mmlu_full $TAGT rc=${RC} exit=${ec} failed_drivers=${TOTAL_FAIL}"
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"

run_class(){   # $1 = class tag (short|longsi), $2 = window ("" = the table's capability:short), $3.. = the class's cells
  local cls="$1" L="$2"; shift 2
  local cells=("$@") cell CONC
  [ "${#cells[@]}" -ge 1 ] || return 0
  f22_stop_requested && { f22_log "wave STOP present — not standing up ($cls)"; return 0; }
  # a prompted-only view (llama8b/shortconst): the SI prefix + the longest MMLU user text + 1 must fit the window, checked
  # with the pinned tokenizer BEFORE the standup (f22_window_check is a no-op on every other view)
  f22_window_check "$SERVE_DIR/$cls" "${L:-2048}" capability
  f22_standup attacks "$SERVE_DIR/$cls" "" capability:short "$L" "" "$CTL_TOP_LOGPROBS" || { RC=1; return 1; }
  # native-serving-default: the cells this standup SERVES (a cells= positional naming a skipped cell is fatal)
  f22_served_cells_filter cells "$CELLS_ARG"
  # STEER-IS-LIVE positive control on the first OBJECT cell (a prompted baseline has no scale-0 twin)
  for cell in "${cells[@]}"; do case "$cell" in none|prompted__*) continue;; esac; f22_positive_control "$cell" "mmlu_full_$cls" || { RC=1; return 1; }; break; done
  CONC=$(( TARGET_S / ${#cells[@]} )); [ "$CONC" -ge 8 ] || CONC=8
  MAX_DRIVERS="${#cells[@]}"
  f22_log "mmlu_full $cls: cells=${#cells[@]} (${cells[*]}) window=$TARGET_L S=$TARGET_S concurrency/cell=$CONC logprobs=$CTL_TOP_LOGPROBS flag=$MMLU_FULL_FLAG out=$OUT_ROOT"
  for cell in "${cells[@]}"; do
    f22_stop_requested && { f22_log "wave STOP present — not launching more drivers"; break; }
    f22_queue_launch "mmlu_full_${cls}__${cell}" "$MPY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$OUT_ROOT" \
      --controls mmlu "$MMLU_FULL_FLAG" --trial "$TRIAL" --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS" \
      --vllm-log "$SERVE_DIR/$cls/vllm_server.log"
    sleep 3
  done
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "mmlu_full $cls done: $NFAIL failed drivers"
  f22_standup_stop
}
run_class short "${VIEW_WINDOW_CONTROLS:-}" ${SHORT[@]+"${SHORT[@]}"} || exit 1
run_class longsi "${VIEW_WINDOW_CONTROLS:-$MMLU_FULL_LONG_SI_WINDOW}" ${LONG[@]+"${LONG[@]}"} || exit 1
RC=$(( TOTAL_FAIL > 0 ? 2 : 0 ))
f22_log "mmlu_full $TAGT done: $TOTAL_FAIL failed drivers"
exit $RC
