#!/bin/bash
#SBATCH --job-name=f22s
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-16:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22 SAMPLED re-run (scripts/sdfbase/full22/sampled/DESIGN.md): ONE steered TARGET SERVER (1 GPU) + the drivers
# for one (target, trial, phase), adapted from the wave's target_static.sbatch — same standup, same stacked-lens
# assertions, same bounded driver queue, same cleanup; the only new content is the SAMPLED config (the vendor
# preset in `sampled_decode`, read by the drivers), the `--trial` the drivers assert, and the CAP LADDER. Every
# standup is one geometry at a time on the same GPU (the second starts after the first is stopped).
#
# Phases (T in 0,1 = the two seeded draws; each trial is its own standard-layout run):
#   static_base_t<T>[:fam,...]  the SAME-CAP pass of the static families at 4096 (default all four: pasttense,
#                               codeattack, renellm, grok420; an optional comma subset splits a target over jobs):
#                               --eval families_served:short = 4864 x S (as the wave; gemma 121, qwen 148,
#                               nemotron / glm 161) filled at min(S, 96) jailbreak_run.py generate drivers, one
#                               in-flight request each. Records the base at-cap count per (fam, cell) into the
#                               ladder as rung 4096.
#   static_esc16_t<T>[:fam,...] the static 16,384 RUNG (the ceiling): for each (fam, cell) whose 8,192
#                               store the GATE escalates (ladder_gate.py static: the item-filtered 1 % TRUNCATED verdict
#                               bound to the transport approval), escalate_static.py re-draws EXACTLY the gate's items
#                               (--items) with the committed _cap16384 config into ..._sampled_t<T>_cap16384, at
#                               f22s_rung_window 16384 (fullconst: its own 51,200 window). A pair the 8,192 phase never
#                               drew (no at-cap row at 4,096) left the ladder there and is skipped. No ESCALATE -> exits 0
#                               without a standup; a refused gate -> exit 1 (never a guess). Submit after static_esc_t<T>.
#   static_esc_t<T>[:fam,...]   the static 8192 RUNG: for each (fam, cell) whose base run holds > 0 at-cap rows
#                               (at_cap_static.py; exit 2 = refusal = this job refuses), escalate_static.py
#                               re-draws ONLY those rows at 8192 into the SEPARATE private_run
#                               ..._sampled_t<T>_cap8192 (copied terminated rows + regenerated at-cap rows,
#                               shard00). Geometry: families_served:short with --max-model-len 10240 (S: gemma 57,
#                               qwen 70, nemotron / glm 76); drivers at --concurrency 32, ceil(S / 32) in flight.
#                               Zero at-cap pairs -> exits 0 WITHOUT a standup. Writes static_ladder_t<T>.json.
#   controls_t<T>               this trial's controls, base + ladder, in ONE job: rung 512 = the SAME-CAP pass
#                               (capability:short 2048 x S, --max-logprobs 64 as the wave; every cell, xstest +
#                               or_bench_hard_1k) into sampled/controls_cap512/<TAG>_t<T>; then the rungs 2048,
#                               8192, 16384 (F22S_CONTROLS_RUNGS; windows per f22s_rung_window), each ONLY for the
#                               (cell, control) pairs the GATE escalates (ladder_gate.py: the
#                               item-filtered 1 % TRUNCATED trigger of ladder_decide.py — loops / repetitive /
#                               boundary items never escalate — bound to the transport approval for the rung
#                               above), with --controls-cap <rung> --escalate-from-root <prev root> --items <the
#                               gate's disposition file> --floor-sample 200 into controls_cap<rung>/<TAG>_t<T>.
#                               A pair leaves the ladder on EXIT exit_le_1pct | exit_ceiling (truncated items
#                               UNRESOLVED at the 16,384 ceiling) | transport_refused. Writes ladder.json (per
#                               cell per control: the raw at-cap count and the gate's decision at every rung
#                               attempted, and where it left the ladder).
#   controls_esc_t<T>           the controls LADDER over an EXISTING, COMPLETE 512 root (2026-09-13; the sdfbase-llama8b views): NO
#                               512 pass — refuses unless every job cell's 512 store carries gen_stats.json — then the rungs exactly as
#                               controls_t<T> runs them. The 512 -> 2048 gate reads the 512 standup's record through f22s_prev_endpoint
#                               (live / endpoint.recorded.json / RECONSTRUCTED from standup_jobs.jsonl); every later gate reads
#                               f22s_standup_endpoint (the recorded copy once the rung's server is stopped). Why it exists: the llama8b
#                               controls_t<T> jobs gated their 512 stores after stopping the 512 server and read the live endpoint.json that
#                               mt_serving_up.py unlinks at teardown -> every pair EXIT transport_refused ("no hop timeouts supplied"),
#                               never a verdict; the truncation was far over 1 %. Writes ladder.json like controls_t<T>.
#   ladder_extend_t<T>          (DESIGN.md 2026-09-10; LADDER_RULES.md): the 16,384 CEILING rung of BOTH
#                               ladders (+ the GSM8K readout's when its 8192 root exists) for this trial, entered per
#                               pair ONLY on an item_filtered_v1 `escalate` decision at 8192. Order: the sha-pinned
#                               rep_ratio source -> `ladder_decide.py reclassify` over every rung generated so far
#                               (static, controls, gsm8k; the sibling trial named for the boundary rule) -> the
#                               PRE-standup hop-timeout guards for both clients -> `ladder_extend.py decide --cap 8192
#                               --next-cap 16384` (writes ladders/<TAG>_t<T>/decision_16384.json — the analysis step
#                               reads that path — and the decision events; a hop guard that is not PASS refuses; ZERO
#                               promotions is a valid outcome and still files the record) -> for the promoted pairs,
#                               one standup per geometry at f22s_rung_window 16384 (18432; nemotron 16896): static
#                               (families_served:short) -> POST-standup guard from endpoint.json (mandatory; REFUSE =
#                               no requests) -> escalate_static.py --config <_cap16384> --items <disposition> into
#                               ..._sampled_t<T>_cap16384; controls (capability:short, thinking OFF) -> guard ->
#                               served_capability_run.py --controls-cap 16384 --escalate-from-root <cap8192 root>
#                               --items … into controls_cap16384/<TAG>_t<T>; gsm8k (thinking ON: F22_THINKING_ON=1,
#                               the certification must pass) -> guard -> --controls gsm8k --gsm8k-arm sampled
#                               --standup-endpoint --controls-cap 16384 into gsm8k_thinking_on_cap16384/<TAG>_t<T>.
#                               Then the CEILING REPORT: reclassify again over the 16384 rungs and `ladder_extend.py
#                               decide --cap 16384` (no --next-cap) records every pair's residual SHARE and COUNT at
#                               16,384 (ceiling_16384.json) — the ladder STOPS there; nothing escalates past 16,384.
#                               Cells that exited on the 1% rule with a residual are filed SEPARATELY (exit_kind) from
#                               cells at the ceiling. Folds static_ladder_t<T>.json / ladder.json / gsm8k_ladder.json.
#   capability_t<T>             the capability battery: GSM8K THINKING-ON at the
#                               8,192 cap under the readout preset (Qwen's THINKING row; the others their single
#                               preset), SAMPLED ONLY — there is no matched greedy arm — into the FIXED root
#                               sampled/gsm8k_thinking_on_cap8192/<TAG>_t<T>: PRE-standup hop guard -> F22_THINKING_ON=1
#                               standup at the model's 8192-rung window (the per-template header certification must
#                               pass before any request; the proxy attests thinking_mode on) -> positive control under
#                               --thinking-on -> POST-standup guard from endpoint.json (mandatory) -> per cell
#                               served_capability_run.py --controls gsm8k --gsm8k-arm sampled --standup-endpoint … ->
#                               f22s_assert_gsm8k_items (the greedy baseline's 200 items by sha) + the at-cap count as
#                               the gsm8k ladder's rung 8192 (gsm8k_ladder.json; the 1%/16,384 rule applies above 8k
#                               through a later ladder_extend_t<T>). Then a thinking-OFF standup for MMLU (the wave's
#                               capability:short 2048 with --max-logprobs 64; a letter-logprob argmax, sampler-moot)
#                               into the FIXED root sampled/mmlu/<TAG>_t<T>. Run capability_t<T> BEFORE
#                               ladder_extend_t<T> so the GSM8K root is on that trial's ladder.
# Stores: static under $WAVE_ROOT/transcripts_results/<private_run>/ (the wave layout; the private_run carries
# _sampled_t<T>[_cap8192]); controls under $WAVE_ROOT/sampled/controls_cap<rung>/<TAG>_t<T>/; ladders under
# $WAVE_ROOT/sampled/ladders/<TAG>_t<T>/; this job's serving dir under $WAVE_ROOT/sampled/serving/. NEVER a
# greedy store, NEVER in place (DESIGN.md "Stores"). TAG = <key> for the main view, <key>_it for the instruct-direct
# view (the 6th positional), so the two views of one model never share a store, a ladder or a serving dir.
# STOP: the wave-wide $WAVE_ROOT/STOP, or this job's $SERVE_DIR/STOP — checked before every driver launch and
# every rung; then scancel BY NAME. The RESULT line (f22_result) rides out through the figure mirror.
# LOW-DOSE LIVENESS (common_sampled.sh f22s_low_dose_liveness / f22s_launch_cell): after every standup, before its
# smoke, lora_liveness.py writes <standup dir>/liveness_low_dose_<cell>.json per steered cell (one `liveness <cell>: …`
# log line each); a live_numeric record rides ONLY into that cell's processes as `env SISTEER_LOW_DOSE_LIVENESS_RECORD=…`
# ahead of "$MPY" in the launch line — the smoke when it probes the cell, every controls driver at every rung (16384
# included), every static shard and escalation driver. No record (not_live / unevaluable / writer failed): the launch
# line is byte-identical to before and the text-divergence gate stands.
# Usage (positional only — never --export=):
#   sbatch -J f22s-<key>-<tag> [--time=HH:MM:SS] scripts/sdfbase/full22/sampled/target_sampled.sbatch \
#       <key> <phase> <wave_sha> [nshards=9] [cells=all] [view=main]
#   (cluster B: sbatch --qos=<that cluster's QoS> --output=$SHARED/exp/logs/%x_%j.out ...; submit_sampled.sh prints both)
#   view: `main` = DESIGN.md's scope, TAG = <key>; `it` = the instruct-direct view (a scope addition:
#   none + the two itdoc objects + prompted__safety_refusal), TAG = <key>_it — the sampled configs, the
#   `_it` Petri manifest, the (key, it) bank view, and every store / ladder / serving path keyed on TAG follow. An EMPTY
#   cells arg means all: `... <key> static_base_t0 <sha> 9 "" it`. <key> stays the bare model key (budgets, hybrid facts,
#   rung windows and --model-key are per model).
set -uo pipefail
KEY="${1:?key}"; PHASE="${2:?phase}"; WAVE_SHA="${3:?wave sha}"; NSHARDS="${4:-9}"; CELLS_ARG="${5:-}"; VIEW="${6:-main}"
# cells-fix: a job that serves a SUBSET of the view's cells registers that subset (F22_REGISTRY_CELLS narrows the
# registration overlay inside f22s_resolve_config / f22_registry_register) — exported here, before f22_init
[ -n "$CELLS_ARG" ] && export F22_REGISTRY_CELLS="${F22_REGISTRY_CELLS:-$CELLS_ARG}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/sampled/common_sampled.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/sampled/common_sampled.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")   # the kind-aware serving-path default (2026-09-15): the sampled phases serve the wave's (KEY, VIEW) manifest (f22s_resolve_config)
f22_init "$WAVE_SHA"; f22s_parse_phase "$PHASE"; f22s_resolve_config "$KEY" "$TRIAL" "$VIEW"
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
SERVE_DIR="$WAVE_ROOT/sampled/serving/${PHASE_TAG}_${TAG}_${SLURM_JOB_ID}"; PRIVATE_ROOT="$WAVE_ROOT/transcripts_results"
LADDER_DIR="$WAVE_ROOT/sampled/ladders/${TAG}_t${TRIAL}"; JOB_STOP="$SERVE_DIR/STOP"
mkdir -p "$SERVE_DIR/drivers" "$PRIVATE_ROOT" "$LADDER_DIR"; chmod 700 "$PRIVATE_ROOT"
f22_log "phase=$PHASE kind=$PHASE_KIND trial=$TRIAL view=$VIEW tag=$TAG families=${FAMS[*]:-none} serve_dir=$SERVE_DIR job_stop=$JOB_STOP"
for fam in "${FAMS[@]:-}"; do case "$fam" in renellm)
  export SISTEER_HELDOUT_CORPUS_DIR="${SISTEER_HELDOUT_CORPUS_DIR:-$SHARED/exp/heldout_corpora}"
  # the corpus replay reads: this cluster's copy at $SISTEER_HELDOUT_CORPUS_DIR/renellm.jsonl when there is one, else
  # the paper's corpus as the repository ships it (data/heldout_corpora/renellm.jsonl). The paper's replays as is, any
  # other only when SISTEER_RENELLM_NONPAPER_CORPUS_SHA names it (its results are labelled non-paper). Checked here,
  # under the system python (stdlib only), before anything is stood up
  python3 -I -S scripts/renellm_corpus_check.py \
    || f22_fatal "renellm: the corpus check refused the replay corpus (the reason is above); nothing was stood up";;
esac; done
f22_driver_venv; f22_serving_venv; f22s_assert_vllm_mamba_kernels || f22_fatal "nemotron fused Mamba2 kernels missing in the serving venv (see log)"; f22s_check_config "$MPY"; f22_bank_view "$MPY"
if [ -n "$CELLS_ARG" ]; then IFS=',' read -ra CELLS <<< "$CELLS_ARG"; else CELLS=("${ALL_CELLS[@]}"); fi
NCELLS=${#CELLS[@]}
for c in "${CELLS[@]}"; do case " ${ALL_CELLS[*]} " in *" $c "*) ;; *) f22_fatal "cell $c is not in $CFG's explicit_cells (${ALL_CELLS[*]}) — a job serves only cells the config names";; esac; done
# LONG-SI cells (the constitution-SI prompted baseline, 2026-09-12): a ~6k-token system turn does not fit the :short windows,
# so every window this job stands up comes from f22s_base_window / f22s_rung_window under F22S_LONG_SI (common_sampled.sh)
f22s_detect_long_si "${CELLS[@]}"
# the sampler the ENGINE resolved goes on the record: vLLM logs one INFO line per request (no prompt text);
# f22s_record_resolved_sampler reads it back per phase. A plain Bash array, never exported (hard rule 1). Built ON TOP
# of the inherited value through common.sh's helper, never assigned (2026-09-16): a scalar F22_STANDUP_EXTRA inherited
# from the launch environment is the launch-time argv hook, `=(` clobbered it on this lane (wave 0e21d6dc), and a bare
# `+=` first touch would keep an exported-empty scalar as an empty argv element.
f22_standup_extra_reset --enable-log-requests
ROOT_OF(){ echo "$WAVE_ROOT/sampled/controls_cap$1/${TAG}_t${TRIAL}"; }   # the controls rung roots (every phase that touches one)
f22s_ctl_top_logprobs(){   # -> CTL_TOP_LOGPROBS: ONE number for the server's logprob cap AND the driver's request (the wave's lesson, 447405): read from the driver
  CTL_TOP_LOGPROBS=$("$MPY" -c 'import importlib.util,sys; s=importlib.util.spec_from_file_location("scr", sys.argv[1]); m=importlib.util.module_from_spec(s); sys.modules["scr"]=m; s.loader.exec_module(m); print(int(m.MMLU_TOP_LOGPROBS))' scripts/served_capability_run.py) \
    || { f22_log "cannot read MMLU_TOP_LOGPROBS from scripts/served_capability_run.py"; RC=1; exit 1; }
}
TARGET_PID=""; STANDUP_STOP=""; declare -a DRIVER_PIDS=(); declare -A DRIVER_TAG=(); RC=0; TOTAL_FAIL=0
cleanup(){
  # $? FIRST: the script's exit status (an f22_fatal exits 1 with RC still 0 — the RESULT line must say so, never rc=0)
  local ec=$?; [ "$ec" != 0 ] && [ "${RC:-0}" = 0 ] && RC=$ec
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  for p in "${DRIVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true; done
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  f22_result "sampled $TAG $PHASE rc=${RC} exit=${ec} failed_drivers=${TOTAL_FAIL}"
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"

# ---------------------------------------------------------------- static_base_t<T>: 4864 x S, the 4096 cap
# a prompted-only view (llama8b/fullconst): the load check sends the SI prefix too, at the escalation rung's cap
if f22_view_prompted_only; then
  export F22_LOAD_CHECK_SYSTEM_SI="${F22_LOAD_CHECK_SYSTEM_SI:-$(f22_prompted_si)}"
  # the 8,192 probe cap is the ONE-WINDOW view's (fullconst); a PER-LEG view (shortconst, 2026-09-12) probes each standup at the
  # RUN's cap (the rung phases below pass theirs explicitly) — its base windows are sized for their own caps
  [ -z "$VIEW_WINDOW" ] || export F22_LOAD_CHECK_CAP="${F22_LOAD_CHECK_CAP:-8192}"
fi
if [ "$PHASE_KIND" = static_base ]; then
  f22s_view_has_leg static || f22_fatal "view $KEY/$VIEW has no static leg (legs: $VIEW_LEGS)"
  # a view with a serving window (llama8b/fullconst: 45,056) stands up at it; every other view at the table's 4864
  f22_window_check "$SERVE_DIR/static" "$(f22s_base_window static)" static "$( [ -n "$VIEW_WINDOW" ] || echo base )"   # a per-leg view: the base cap only (its 8,192 rung is its own standup)
  f22_standup attacks "$SERVE_DIR/static" "" families_served:short "$(f22s_base_window static)" || { RC=1; exit 1; }
  f22s_record_si_prefix "$SERVE_DIR/static" || { RC=1; exit 1; }
  f22s_positive_control static_base || { RC=1; exit 1; }
  # the static drivers are one-request-each, so the number in flight IS the client width: it defaults to the
  # attested server width S; F22S_DRIVER_CAP in the launch environment caps it (96 = the campaign's value)
  DRIVER_CAP="${F22S_DRIVER_CAP:-$TARGET_S}"
  MAX_DRIVERS=$(( TARGET_S < DRIVER_CAP ? TARGET_S : DRIVER_CAP ))
  f22_log "phase STATIC_BASE t$TRIAL: families=${FAMS[*]} cells=$NCELLS nshards=$NSHARDS cap=$F22S_STATIC_CAP server_width=$TARGET_S max_drivers=$MAX_DRIVERS (cap $DRIVER_CAP)"
  for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do for s in $(seq 0 $((NSHARDS-1))); do
    f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break 3; }
    f22s_launch_cell "${fam}__${cell}__s$(printf '%02d' "$s")" "$cell" "$MPY" scripts/jailbreak_run.py generate \
      --config "$CFG" --attack "$fam" --defense "$cell" --confirmatory --trial "$TRIAL" --shard "$s" --nshards "$NSHARDS" \
      --private-root "$PRIVATE_ROOT" --target-engine served --target-proxy-url "$PROXY_URL" \
      --target-routing-table "$TABLE" --target-max-model-len "$TARGET_L"
    sleep 1
  done; done; done
  f22_log "phase STATIC_BASE: all ${#DRIVER_PIDS[@]} drivers launched (queued at <= $MAX_DRIVERS in flight); waiting"
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "phase STATIC_BASE done: $NFAIL failed drivers; recording the base at-cap counts (ladder rung $F22S_STATIC_CAP)"
  for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do
    n=$(f22s_at_cap_static "$PRIVATE_RUN_BASE" "$fam" "$cell") || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); continue; }
    f22s_event "$(f22s_event_json kind=static attack="$fam" cell="$cell" rung="$F22S_STATIC_CAP" n_at_cap="$n" root="$PRIVATE_ROOT/$PRIVATE_RUN_BASE/$fam/$cell")"
    f22_log "at-cap @$F22S_STATIC_CAP $fam/$cell: $n"
  done; done
  f22s_write_ladder static "$LADDER_DIR/static_ladder_t${TRIAL}.json"
  f22s_record_resolved_sampler "$SERVE_DIR/static" static "$PRIVATE_ROOT/$PRIVATE_RUN_BASE"/*/"${CELLS[0]}"/shard00/responses.jsonl
  f22_standup_stop

# ---------------------------------------------------------------- static_esc_t<T>: the 8192 rung, at-cap pairs only
elif [ "$PHASE_KIND" = static_esc ]; then
  f22s_view_has_leg static || f22_fatal "view $KEY/$VIEW has no static leg (legs: $VIEW_LEGS)"
  declare -A NBASE=(); NEED=0
  for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do
    n=$(f22s_at_cap_static "$PRIVATE_RUN_BASE" "$fam" "$cell") || { f22_log "REFUSING: the base run's at-cap count for $fam/$cell was refused — fix the base store before escalating"; RC=1; exit 1; }
    NBASE["$fam|$cell"]="$n"
    f22s_event "$(f22s_event_json kind=static attack="$fam" cell="$cell" rung="$F22S_STATIC_CAP" n_at_cap="$n" root="$PRIVATE_ROOT/$PRIVATE_RUN_BASE/$fam/$cell")"
    [ "$n" -gt 0 ] && NEED=$((NEED + 1))
    f22_log "at-cap @$F22S_STATIC_CAP $fam/$cell: $n"
  done; done
  f22s_write_ladder static "$LADDER_DIR/static_ladder_t${TRIAL}.json"
  if [ "$NEED" -eq 0 ]; then f22_log "phase STATIC_ESC t$TRIAL: no (family, cell) pair has at-cap rows at $F22S_STATIC_CAP — every pair left the ladder; no standup"; RC=0; exit 0; fi
  f22s_stop_requested && { f22_log "STOP present — not standing up"; exit 0; }
  # static-reg-fix-2: register EXACTLY the pairs this gate selected (NBASE > 0), once, before any GPU work — the job-start
  # registration was deferred (f22s_parse_phase); a refused registration exits 1 here, standing nothing up
  REG_PAIRS=""; for pc in "${!NBASE[@]}"; do if [ "${NBASE[$pc]}" -gt 0 ]; then REG_PAIRS="${REG_PAIRS:+$REG_PAIRS,}${pc%%|*}:${pc#*|}"; fi; done
  f22s_register_gated_pairs static_esc "$REG_PAIRS"
  f22_window_check "$SERVE_DIR/static_esc" "$(f22s_rung_window 8192)" static 8192
  # the load check probes at THIS rung's cap: the run's cap, never the table's
  F22_LOAD_CHECK_CAP=8192 f22_standup attacks "$SERVE_DIR/static_esc" "" families_served:short "$(f22s_rung_window 8192)" || { RC=1; exit 1; }
  f22s_record_si_prefix "$SERVE_DIR/static_esc" || { RC=1; exit 1; }
  f22s_positive_control static_esc || { RC=1; exit 1; }
  MAX_DRIVERS=$(( (TARGET_S + F22S_ESC_CONC - 1) / F22S_ESC_CONC ))   # ceil(S / 32): the width filled at 32 per driver
  f22_log "phase STATIC_ESC t$TRIAL: $NEED at-cap pairs of ${#FAMS[@]}x$NCELLS; cap $F22S_STATIC_CAP -> $F22S_STATIC_CAP_ESC; server_width=$TARGET_S concurrency/driver=$F22S_ESC_CONC max_drivers=$MAX_DRIVERS"
  for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do
    [ "${NBASE[$fam|$cell]}" -gt 0 ] || continue
    f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break 2; }
    f22s_launch_cell "esc__${fam}__${cell}" "$cell" "$MPY" "$F22S_DIR/escalate_static.py" \
      --config "$CFG_ESC" --attack "$fam" --defense "$cell" --private-root "$PRIVATE_ROOT" \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --max-model-len "$TARGET_L" --concurrency "$F22S_ESC_CONC"
    sleep 2
  done; done
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "phase STATIC_ESC drivers done: $NFAIL failed; reading each escalation.json"
  for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do
    [ "${NBASE[$fam|$cell]}" -gt 0 ] || continue
    ev=$(f22s_escalation_event "$fam" "$cell"); erc=$?
    f22s_event "$ev"
    [ "$erc" = 0 ] || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); f22_log "escalation record PROBLEM for $fam/$cell: ${ev:0:400}"; }
  done; done
  f22s_write_ladder static "$LADDER_DIR/static_ladder_t${TRIAL}.json"
  f22s_record_resolved_sampler "$SERVE_DIR/static_esc" static "$PRIVATE_ROOT/$PRIVATE_RUN_ESC"/*/"${CELLS[0]}"/shard00/responses.jsonl
  f22_standup_stop

# ---------------------------------------------------------------- static_esc16_t<T>: the 16,384 rung — the gate's ESCALATE from the 8,192 store only
#: the static ladder's ceiling rung. ladder_gate.py static reads the 8,192 rung's store (the
# ladder_decide item-filtered 1 % TRUNCATED verdict; loops / repetitive / boundary items never escalate) BOUND to the
# transport approval, and ONLY an ESCALATE pair is re-drawn — exactly the gate's items (escalate_static.py --items) with
# the committed _cap16384 config — at f22s_rung_window 16384 (fullconst: its OWN 51,200 window). A pair the gate
# EXITs leaves the ladder with its reason; a REFUSED gate stops the job: this phase never runs on a guess. The finished
# 16,384 store is gated too (P2-a): its exit_ceiling / exit_le_1pct is filed before the final fold.
elif [ "$PHASE_KIND" = static_esc16 ]; then
  f22s_view_has_leg static || f22_fatal "view $KEY/$VIEW has no static leg (legs: $VIEW_LEGS)"
  # which pairs the 8,192 phase DREW: exactly those with an at-cap row in the base run (that phase's own rule, read from
  # the same authoritative store). A pair with none left the ladder at 4,096 and has no 8,192 store to gate — it is
  # skipped, never gated on a missing store; the count decides nothing about climbing (the gate alone does, below)
  declare -A REACHED=(); NREACHED=0
  for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do
    n=$(f22s_at_cap_static "$PRIVATE_RUN_BASE" "$fam" "$cell") || { f22_log "REFUSING: the base run's at-cap count for $fam/$cell was refused — which pairs reached $F22S_STATIC_CAP_ESC cannot be told"; RC=1; exit 1; }
    if [ "$n" -eq 0 ]; then f22_log "gate @$F22S_STATIC_CAP_ESC $fam/$cell: not drawn — the pair left the ladder at $F22S_STATIC_CAP (no at-cap row in the base run)"; continue; fi
    REACHED["$fam|$cell"]=1; NREACHED=$((NREACHED + 1))
  done; done
  if [ "$NREACHED" -eq 0 ]; then f22_log "phase STATIC_ESC16 t$TRIAL: no (family, cell) pair reached $F22S_STATIC_CAP_ESC — every pair left the ladder at $F22S_STATIC_CAP (the 8,192 phase stood nothing up); nothing to gate, no standup"; RC=0; exit 0; fi
  PREV_EP=$(f22s_prev_endpoint "static_esc_t${TRIAL}" static_esc) || f22_fatal "no endpoint.json of this trial's 8,192 standup under $WAVE_ROOT/sampled/serving/static_esc_t${TRIAL}*_${TAG}_<jobid>/static_esc/ — the transport approval reads its recorded proxy hop, and $NREACHED pair(s) reached $F22S_STATIC_CAP_ESC so that standup must have run; set F22S_PREV_ENDPOINT to point at its record"
  declare -A ITEMS16=(); NEED=0
  for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do
    [ -n "${REACHED[$fam|$cell]:-}" ] || continue
    line=$(f22s_static_gate "$PRIVATE_RUN_ESC" "$fam" "$cell" "$F22S_STATIC_CAP_ESC" "$PREV_EP") || { f22_log "REFUSING: the ladder gate at rung $F22S_STATIC_CAP_ESC for $fam/$cell was refused — the 16,384 rung cannot proceed on a guess"; RC=1; exit 1; }
    gate="$LADDER_DIR/gates/static_cap${F22S_STATIC_CAP_ESC}/${fam}__${cell}.gate.json"
    f22s_event "$(f22s_gate_event "$gate")"
    read -r verb arg1 arg2 <<< "$line"
    case "$verb" in
      ESCALATE)
        [ "$arg1" = "$F22S_STATIC_CAP_ESC16" ] || { f22_log "REFUSING: the gate escalates $fam/$cell to $arg1 but this phase's rung is $F22S_STATIC_CAP_ESC16"; RC=1; exit 1; }
        ITEMS16["$fam|$cell"]="$arg2"; NEED=$((NEED + 1))
        f22_log "gate @$F22S_STATIC_CAP_ESC $fam/$cell: ESCALATE -> $arg1: $(f22s_gate_summary "$gate")";;
      EXIT)
        f22_log "gate @$F22S_STATIC_CAP_ESC $fam/$cell: EXIT $arg1: $(f22s_gate_summary "$gate")";;
    esac
  done; done
  f22s_write_ladder static "$LADDER_DIR/static_ladder_t${TRIAL}.json"
  if [ "$NEED" -eq 0 ]; then f22_log "phase STATIC_ESC16 t$TRIAL: no (family, cell) pair escalates past $F22S_STATIC_CAP_ESC — every pair left the ladder; no standup"; RC=0; exit 0; fi
  f22s_stop_requested && { f22_log "STOP present — not standing up"; exit 0; }
  # static-reg-fix-2: register EXACTLY the gate's ESCALATE pairs (ITEMS16), once, before any GPU work (the job-start registration
  # was deferred); a refused registration exits 1 here, standing nothing up
  REG_PAIRS=""; for pc in "${!ITEMS16[@]}"; do REG_PAIRS="${REG_PAIRS:+$REG_PAIRS,}${pc%%|*}:${pc#*|}"; done
  f22s_register_gated_pairs static_esc16 "$REG_PAIRS"
  WINDOW16=$(f22s_rung_window 16384)   # fullconst: 51,200 (its own window); every other view 18,432 (common_sampled.sh)
  f22_window_check "$SERVE_DIR/static_esc16" "$WINDOW16" static 16384
  F22_LOAD_CHECK_CAP=16384 f22_standup attacks "$SERVE_DIR/static_esc16" "" families_served:short "$WINDOW16" || { RC=1; exit 1; }
  # the transport approval is re-checked against THIS standup's recorded hop before any driver (the gate read the 8,192 rung's)
  f22s_ladder_transport "$SERVE_DIR/static_esc16/endpoint.json" 16384 || { f22_log "REFUSING the 16,384 rung: the standup's recorded proxy hop cannot carry a 16,384-token draw — raise --upstream-timeout-s"; RC=1; exit 1; }
  f22s_positive_control static_esc16 || { RC=1; exit 1; }
  MAX_DRIVERS=$(( (TARGET_S + F22S_ESC_CONC - 1) / F22S_ESC_CONC ))
  f22_log "phase STATIC_ESC16 t$TRIAL: $NEED escalating pairs of ${#FAMS[@]}x$NCELLS; cap $F22S_STATIC_CAP_ESC -> $F22S_STATIC_CAP_ESC16 (window $WINDOW16 x $TARGET_S); concurrency/driver=$F22S_ESC_CONC max_drivers=$MAX_DRIVERS"
  for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do
    [ -n "${ITEMS16[$fam|$cell]:-}" ] || continue
    f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break 2; }
    f22s_launch_cell "esc16__${fam}__${cell}" "$cell" "$MPY" "$F22S_DIR/escalate_static.py" \
      --config "$CFG_ESC16" --attack "$fam" --defense "$cell" --private-root "$PRIVATE_ROOT" \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --max-model-len "$TARGET_L" --concurrency "$F22S_ESC_CONC" \
      --items "${ITEMS16[$fam|$cell]}"
    sleep 2
  done; done
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "phase STATIC_ESC16 drivers done: $NFAIL failed; reading each escalation.json, then the gate at the ceiling"
  for fam in "${FAMS[@]}"; do for cell in "${CELLS[@]}"; do
    [ -n "${ITEMS16[$fam|$cell]:-}" ] || continue
    ev=$(f22s_escalation_event "$fam" "$cell" "$PRIVATE_RUN_ESC16"); erc=$?
    f22s_event "$ev"
    # (P2-α) a counted failure at the ceiling files the rung-keyed UNDECIDED mark before moving on — the
    # fold must not carry the 8,192 ESCALATE forward as an active escalation to a rung that has already run
    [ "$erc" = 0 ] || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); f22_log "escalation record PROBLEM for $fam/$cell: ${ev:0:400}"
                        f22s_event "$(f22s_undecided_event "$fam" "$cell" "$F22S_STATIC_CAP_ESC16" record_problem "$(f22s_json_field "$ev" problem)")"; continue; }
    # (P2-a) the CEILING store is gated too: the pair's exit at 16,384 is the gate's decision over the
    # completed store (exit_ceiling while truncation stays above 1 %, exit_le_1pct otherwise), filed as a gate event so
    # the fold's left_ladder_at is that verdict — never the 8,192 gate's "escalating" carried past the last rung. Nothing
    # runs on this decision (there is no rung above), so a refusal is a counted failure folded as UNDECIDED at 16,384
    # (the mark above, with the gate's own refusal as the reason), not an exit of the job before the fold; the endpoint
    # is THIS standup's record (the store's own hop, recorded on the gate)
    line=$(f22s_static_gate "$PRIVATE_RUN_ESC16" "$fam" "$cell" "$F22S_STATIC_CAP_ESC16" "$SERVE_DIR/static_esc16/endpoint.json") \
      || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); f22_log "gate @$F22S_STATIC_CAP_ESC16 $fam/$cell REFUSED — the pair's exit at the ceiling is undeterminable; folded as undecided at $F22S_STATIC_CAP_ESC16"
           f22s_event "$(f22s_undecided_event "$fam" "$cell" "$F22S_STATIC_CAP_ESC16" gate_refused "$(f22s_last_gate_refusal)")"; continue; }
    gate="$LADDER_DIR/gates/static_cap${F22S_STATIC_CAP_ESC16}/${fam}__${cell}.gate.json"
    f22s_event "$(f22s_gate_event "$gate")"
    read -r verb arg1 arg2 <<< "$line"
    case "$verb" in
      EXIT) f22_log "gate @$F22S_STATIC_CAP_ESC16 $fam/$cell: EXIT $arg1: $(f22s_gate_summary "$gate")";;
      *)    TOTAL_FAIL=$((TOTAL_FAIL + 1)); f22_log "gate @$F22S_STATIC_CAP_ESC16 $fam/$cell printed '$verb $arg1' at the CEILING — there is no rung above $F22S_STATIC_CAP_ESC16; recorded, never acted on";;
    esac
  done; done
  f22s_write_ladder static "$LADDER_DIR/static_ladder_t${TRIAL}.json"
  f22_standup_stop

# ---------------------------------------------------------------- controls_t<T>: rung 512, then 2048, then 8192
elif [ "$PHASE_KIND" = controls ]; then
  # a view without the over-refusal leg (the llama8b fullconst view) has NO controls ladder: refuse, never run
  # xstest / or_bench on a cell they are excluded from
  f22s_view_has_leg controls || f22_fatal "view $KEY/$VIEW has no controls (over-refusal) leg (legs: $VIEW_LEGS): no controls ladder — this phase must not be submitted for it"
  [ "${#F22S_CONTROLS[@]}" -gt 0 ] || f22_fatal "view $KEY/$VIEW: the roster's controls list carries no judged control"
  f22s_ctl_top_logprobs
  CTL_LIST=$(IFS=,; echo "${F22S_CONTROLS[*]}")
  # ---- rung 512: the SAME-CAP base pass, every cell, both controls (capability:short 2048 x S, as the wave; a LONG-SI cell on a table view at f22s_base_window's 6656)
  if [ -z "${F22S_CONTROLS_ESC:-}" ]; then   # controls_t<T>: the 512 pass; controls_esc_t<T> (F22S_CONTROLS_ESC=1): resume below
    f22_window_check "$SERVE_DIR/controls_512" "$(f22s_base_window controls)" controls
    f22_standup attacks "$SERVE_DIR/controls_512" "" capability:short "$(f22s_base_window controls)" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }
    f22s_record_si_prefix "$SERVE_DIR/controls_512" || { RC=1; exit 1; }
    f22s_positive_control controls_512 || { RC=1; exit 1; }
    CONC=$(( TARGET_S / NCELLS )); [ "$CONC" -ge 8 ] || CONC=8
    MAX_DRIVERS=$((NCELLS + 1))   # the queue bound counts LIVE drivers (common.sh); +1 keeps the wave's launch-gate wart moot
    f22_log "phase CONTROLS t$TRIAL rung 512: cells=$NCELLS controls=$CTL_LIST concurrency/cell=$CONC (server width $TARGET_S) logprobs=$CTL_TOP_LOGPROBS out=$(ROOT_OF 512)"
    for cell in "${CELLS[@]}"; do
      f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break; }
      f22s_launch_cell "controls512__${cell}" "$cell" "$MPY" scripts/served_capability_run.py generate \
        --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
        --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$(ROOT_OF 512)" \
        --controls "$CTL_LIST" --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS" \
        --vllm-log "$SERVE_DIR/controls_512/vllm_server.log"
      sleep 3
    done
    f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
    f22_log "phase CONTROLS rung 512 done: $NFAIL failed drivers"
    f22s_record_resolved_sampler "$SERVE_DIR/controls_512" controls "$(ROOT_OF 512)"/cells/*/gen_stats.json
    f22_standup_stop
    EP_512="$(f22s_standup_endpoint "$SERVE_DIR/controls_512")"   # the 512 server is stopped: the READY copy, never the unlinked live file
  else
    # ---- controls_esc_t<T>: resume the ladder over a COMPLETE 512 root — every job cell's store must be finished (gen_stats.json,
    #      manifest cell_id); the phase never generates a 512 store (that is controls_t<T>'s), and never gates on a guessed hop
    COMPLETE_512=$("$MPY" - "$(ROOT_OF 512)" <<'PY'
import json, os, sys
cells = os.path.join(sys.argv[1], "cells")
for s in (sorted(os.listdir(cells)) if os.path.isdir(cells) else []):
    m, g = os.path.join(cells, s, "manifest.json"), os.path.join(cells, s, "gen_stats.json")
    if os.path.isfile(m) and os.path.isfile(g):
        print(json.load(open(m)).get("cell_id", ""))
PY
) || f22_fatal "controls_esc_t$TRIAL: cannot index the 512 root $(ROOT_OF 512)"
    for cell in "${CELLS[@]}"; do
      grep -qx -- "$cell" <<< "$COMPLETE_512" || f22_fatal "controls_esc_t$TRIAL: no COMPLETE 512 store for cell $cell under $(ROOT_OF 512) (a store = manifest.json cell_id + gen_stats.json; complete: $(echo "$COMPLETE_512" | paste -sd, | cut -c1-300)) — this phase resumes a finished 512 pass, never generates one (submit controls_t$TRIAL for that)"
    done
    EP_512=$(f22s_prev_endpoint "controls_t$TRIAL" controls_512) || f22_fatal "controls_esc_t$TRIAL: no endpoint record for $TAG's 512 standup under $WAVE_ROOT/sampled/serving/controls_t${TRIAL}*_${TAG}_<job>/controls_512 (endpoint.json / endpoint.recorded.json / a reconstructible standup_jobs.jsonl) — the transport approval for 2048 is undeterminable; refusing"
    f22_log "phase CONTROLS_ESC t$TRIAL: 512 root $(ROOT_OF 512) complete for ${#CELLS[@]} cells (${CELLS[*]}); no 512 pass; the 512 -> 2048 gate reads $EP_512"
  fi
  # ---- the ladder: 2048, 8192, 16384 (F22S_CONTROLS_RUNGS; windows per f22s_rung_window — a LONG-SI cell on a table view: 8192 / 14848 / 23040 under F22S_LONG_SI) — a (cell, control) climbs a
  # rung ONLY on the gate's ESCALATE: ladder_gate.py binds ladder_decide.py's item-filtered 1 % TRUNCATED
  # verdict (loops / repetitive / boundary items excluded — a loop is non-terminating, never escalated) to the transport
  # approval for the rung above; the driver regenerates EXACTLY the gate's items (--items). An EXIT leaves the ladder
  # with its reason (exit_le_1pct | exit_ceiling = unresolved at the ceiling | transport_refused); a REFUSAL stops the job.
  declare -A ON_LADDER=() ITEMS_OF=()
  for cell in "${CELLS[@]}"; do ON_LADDER[$cell]="$CTL_LIST"; done
  f22s_gate_pairs(){   # $1 = the rung just measured, $2 = its root, $3 = its endpoint.json, $4 = the rung above ("" at the ceiling)
    local rung="$1" root="$2" ep="$3" above="$4" cell ctl keep line verb arg1 arg2 gate ctls
    NRUNG=0
    for cell in "${CELLS[@]}"; do
      [ -n "${ON_LADDER[$cell]:-}" ] || continue
      IFS=',' read -ra ctls <<< "${ON_LADDER[$cell]}"; keep=""
      for ctl in "${ctls[@]}"; do
        line=$(f22s_ladder_gate "$root" "$cell" "$ctl" "$rung" "$ep") || { f22_log "REFUSING: the ladder gate at rung $rung for $cell/$ctl was refused — the ladder cannot proceed on a guess"; RC=1; exit 1; }
        gate="$LADDER_DIR/gates/cap$rung/${cell}__${ctl}.gate.json"
        f22s_event "$(f22s_gate_event "$gate")"
        read -r verb arg1 arg2 <<< "$line"
        case "$verb" in
          ESCALATE)
            [ -n "$above" ] && [ "$arg1" = "$above" ] || { f22_log "REFUSING: the gate escalates $cell/$ctl to $arg1 but the loop's next rung is ${above:-none (ceiling)}"; RC=1; exit 1; }
            keep="${keep:+$keep,}$ctl"; ITEMS_OF["$cell/$ctl"]="$arg2"
            f22_log "gate @$rung $cell/$ctl: ESCALATE -> $arg1: $(f22s_gate_summary "$gate")";;
          EXIT)
            f22_log "gate @$rung $cell/$ctl: EXIT $arg1: $(f22s_gate_summary "$gate")";;
        esac
      done
      ON_LADDER[$cell]="$keep"; [ -n "$keep" ] && NRUNG=$((NRUNG + 1))
    done
  }
  PREV_RUNG=512; LAST_RAN=512; NRUNG=0
  for RUNG in "${F22S_CONTROLS_RUNGS[@]:1}"; do
    WINDOW=$(f22s_rung_window "$RUNG")   # per-model / per-view (common_sampled.sh)
    PREV_ROOT=$(ROOT_OF "$PREV_RUNG"); NEXT_ROOT=$(ROOT_OF "$RUNG")
    # the previous rung's server is STOPPED here, and mt_serving_up.py unlinked its live endpoint.json at teardown: read the
    # 512 standup's record resolved above, else the rung's READY copy (f22s_standup_endpoint) — an absent record still refuses the climb
    if [ "$PREV_RUNG" = 512 ]; then EP_PREV="$EP_512"; else EP_PREV="$(f22s_standup_endpoint "$SERVE_DIR/controls_${PREV_RUNG}")"; fi
    f22s_gate_pairs "$PREV_RUNG" "$PREV_ROOT" "$EP_PREV" "$RUNG"
    f22s_write_ladder controls "$LADDER_DIR/ladder.json"
    if [ "$NRUNG" -eq 0 ]; then f22_log "phase CONTROLS: every (cell, control) left the ladder at rung $PREV_RUNG — no $RUNG rung"; break; fi
    f22s_stop_requested && { f22_log "STOP present — not standing up rung $RUNG"; break; }
    f22_window_check "$SERVE_DIR/controls_${RUNG}" "$WINDOW" controls "$RUNG"
    F22_LOAD_CHECK_CAP="$RUNG" f22_standup attacks "$SERVE_DIR/controls_${RUNG}" "" capability:short "$WINDOW" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }
    f22s_record_si_prefix "$SERVE_DIR/controls_${RUNG}" || { RC=1; exit 1; }
    # the transport approval is re-checked against THIS standup's recorded hop before any driver (the gate read the previous rung's)
    f22s_ladder_transport "$SERVE_DIR/controls_${RUNG}/endpoint.json" "$RUNG" || { f22_log "REFUSING rung $RUNG: the standup's recorded proxy hop cannot carry a $RUNG-token draw — raise --upstream-timeout-s"; RC=1; exit 1; }
    f22s_positive_control "controls_${RUNG}" || { RC=1; exit 1; }
    CONC=$(( TARGET_S / NRUNG )); [ "$CONC" -ge 8 ] || CONC=8
    MAX_DRIVERS=$((NRUNG + 1))
    f22_log "phase CONTROLS t$TRIAL rung $RUNG (window $WINDOW x $TARGET_S): $NRUNG cells still on the ladder, concurrency/cell=$CONC, --floor-sample $F22S_FLOOR_SAMPLE_K, from $PREV_ROOT into $NEXT_ROOT"
    for cell in "${CELLS[@]}"; do
      [ -n "${ON_LADDER[$cell]:-}" ] || continue
      f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break; }
      ITEMS_ARGS=(); IFS=',' read -ra ctls <<< "${ON_LADDER[$cell]}"
      for ctl in "${ctls[@]}"; do ITEMS_ARGS+=(--items "${ITEMS_OF["$cell/$ctl"]}"); done
      f22s_launch_cell "controls${RUNG}__${cell}" "$cell" "$MPY" scripts/served_capability_run.py generate \
        --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
        --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$NEXT_ROOT" \
        --controls "${ON_LADDER[$cell]}" --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS" \
        --vllm-log "$SERVE_DIR/controls_${RUNG}/vllm_server.log" \
        --controls-cap "$RUNG" --escalate-from-root "$PREV_ROOT" --floor-sample "$F22S_FLOOR_SAMPLE_K" "${ITEMS_ARGS[@]}"
      sleep 3
    done
    f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
    f22_log "phase CONTROLS rung $RUNG done: $NFAIL failed drivers"
    f22s_record_resolved_sampler "$SERVE_DIR/controls_${RUNG}" controls "$NEXT_ROOT"/cells/*/gen_stats.json
    f22_standup_stop
    PREV_RUNG=$RUNG; LAST_RAN=$RUNG
  done
  # the pairs still on the ladder after the TOP rung ran get their final decision there: exit_ceiling = truncated
  # items UNRESOLVED at the ceiling (reported as such, distinct from loops), exit_le_1pct = resolved
  if [ "$LAST_RAN" = "${F22S_CONTROLS_RUNGS[-1]}" ] && [ "$NRUNG" -gt 0 ]; then
    # the top rung's endpoint through the SURVIVING record (mt_serving_up.py unlinks the live endpoint.json at teardown; the
    # in-loop gates read it this way since bbd36ebb): the live path here made every pair still > 1 % at the top rung read
    # 'transport_refused: no hop timeouts supplied' (llama8b_it t0/t1 under :8192, 2026-09-14)
    EP_LAST="$(f22s_standup_endpoint "$SERVE_DIR/controls_${LAST_RAN}")"
    if [ -n "${F22S_CONTROLS_CEILING_ABOVE:-}" ]; then
      # a `:<rung>` ceiling BELOW the campaign ceiling: a pair still > 1 % here is ESCALATING to the real next rung, which is due
      # (a later controls_esc_t<T>:<next> job climbs it) — recorded as such, never a ceiling exit, never a refusal
      f22s_gate_pairs "$LAST_RAN" "$(ROOT_OF "$LAST_RAN")" "$EP_LAST" "$F22S_CONTROLS_CEILING_ABOVE"
    else
      f22s_gate_pairs "$LAST_RAN" "$(ROOT_OF "$LAST_RAN")" "$EP_LAST" ""
    fi
  fi
  f22s_write_ladder controls "$LADDER_DIR/ladder.json"

# ---------------------------------------------------------------- ladder_extend_t<T>: the 16,384 CEILING rung of every ladder
elif [ "$PHASE_KIND" = ladder_extend ]; then
  EXT_DIR="$LADDER_DIR/extend_${F22S_CEILING}"; DECISION="$LADDER_DIR/decision_${F22S_CEILING}.json"; CEILING_REC="$LADDER_DIR/ceiling_${F22S_CEILING}.json"
  mkdir -p "$EXT_DIR"
  GSM_ROOT8="$(f22s_gsm8k_root "$F22S_GSM8K_CAP")"; GSM_ROOT16="$(f22s_gsm8k_root "$F22S_CEILING")"
  # 1. the sha-pinned classifier source, then RECLASSIFY every rung generated so far under item_filtered_v1
  f22s_rep_ratio_source || { RC=1; exit 1; }
  f22s_reclassify static "$EXT_DIR/reclassify_static" || { RC=1; exit 1; }
  f22s_reclassify controls "$EXT_DIR/reclassify_controls" || { RC=1; exit 1; }
  RECL=(--reclassify "static=$EXT_DIR/reclassify_static/reclassify.json" --reclassify "controls=$EXT_DIR/reclassify_controls/reclassify.json")
  HAVE_GSM=0
  if [ -d "$GSM_ROOT8/cells" ]; then
    f22s_reclassify gsm8k "$EXT_DIR/reclassify_gsm8k" || { RC=1; exit 1; }
    RECL+=(--reclassify "gsm8k=$EXT_DIR/reclassify_gsm8k/reclassify.json"); HAVE_GSM=1
  else
    f22_log "no GSM8K thinking-ON root at $GSM_ROOT8 — the readout is not on this trial's ladder (run capability_t$TRIAL first if it is wanted)"
  fi
  # 2. the PRE-standup hop-timeout guards for BOTH clients at the ceiling rung (the configured defaults; enforced by the decision)
  f22s_hop_guard "$F22S_CEILING" static "$SERVE_DIR/hop_guard_pre_static.json" || { RC=1; exit 1; }
  f22s_hop_guard "$F22S_CEILING" controls "$SERVE_DIR/hop_guard_pre_controls.json" || { RC=1; exit 1; }
  # 3. the DECISION at 8192 -> decision_16384.json (PINNED PATH: the analysis step reads ladders/<TAG>_t<T>/decision_16384.json)
  f22s_ladder_extend_decide "$F22S_STATIC_CAP_ESC" "$F22S_CEILING" "$DECISION" "${RECL[@]}" \
      --hop-guard "$SERVE_DIR/hop_guard_pre_static.json" --hop-guard "$SERVE_DIR/hop_guard_pre_controls.json" || { RC=1; exit 1; }
  f22s_write_ladder_ext static "$LADDER_DIR/static_ladder_t${TRIAL}.json" "$F22S_STATIC_RUNGS_EXT"
  f22s_write_ladder_ext controls "$LADDER_DIR/ladder.json" "$F22S_CONTROLS_RUNGS_EXT"
  [ "$HAVE_GSM" = 1 ] && f22s_write_ladder_ext controls "$LADDER_DIR/gsm8k_ladder.json" "$F22S_GSM8K_RUNGS" "" gsm8k
  N_PROMOTED=$(f22s_decision_field "$DECISION" n_promoted) || { RC=1; exit 1; }
  if [ "$N_PROMOTED" -eq 0 ]; then
    f22_log "phase LADDER_EXTEND t$TRIAL: ZERO promotions to $F22S_CEILING — every pair exited at $F22S_STATIC_CAP_ESC (decision filed at $DECISION; no standup)"
  else
    # the promoted pairs per kind, read from the record (a failed list is a refusal, never an empty rung)
    for k in static controls gsm8k; do f22s_extend_list "$k" "$DECISION" > "$EXT_DIR/promoted_$k.tsv" || { f22_log "REFUSING: ladder_extend.py list $k failed"; RC=1; exit 1; }; done
    mapfile -t EXT_STATIC < "$EXT_DIR/promoted_static.tsv"
    f22_log "phase LADDER_EXTEND t$TRIAL: $N_PROMOTED promoted pairs (static ${#EXT_STATIC[@]}, controls $(wc -l < "$EXT_DIR/promoted_controls.tsv"), gsm8k $(wc -l < "$EXT_DIR/promoted_gsm8k.tsv")) climb $F22S_STATIC_CAP_ESC -> $F22S_CEILING"
    # 4a. STATIC: one standup at the 16384 window, the POST-standup guard from endpoint.json (mandatory), then the rung drivers with --items
    if [ "${#EXT_STATIC[@]}" -gt 0 ]; then
      f22s_stop_requested && { f22_log "STOP present — not standing up the static $F22S_CEILING rung"; exit 0; }
      f22_standup attacks "$SERVE_DIR/static_16384" "" families_served:short "$(f22s_rung_window "$F22S_CEILING")" || { RC=1; exit 1; }
      f22s_positive_control static_16384 || { RC=1; exit 1; }
      f22s_hop_guard "$F22S_CEILING" static "$SERVE_DIR/hop_guard_post_static.json" "$SERVE_DIR/static_16384/endpoint.json" || { RC=1; exit 1; }
      f22s_hop_guard_event static static "$F22S_CEILING" "$SERVE_DIR/hop_guard_post_static.json"
      MAX_DRIVERS=$(( (TARGET_S + F22S_ESC_CONC - 1) / F22S_ESC_CONC ))   # ceil(S / 32), as the 8192 rung
      f22_log "phase LADDER_EXTEND t$TRIAL static: ${#EXT_STATIC[@]} pairs; server_width=$TARGET_S concurrency/driver=$F22S_ESC_CONC max_drivers=$MAX_DRIVERS"
      for line in "${EXT_STATIC[@]}"; do
        IFS=$'\t' read -r fam cell items <<< "$line"
        f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break; }
        f22s_launch_cell "ext16384__${fam}__${cell}" "$cell" "$MPY" "$F22S_DIR/escalate_static.py" \
          --config "$CFG_ESC16" --attack "$fam" --defense "$cell" --private-root "$PRIVATE_ROOT" \
          --proxy-url "$PROXY_URL" --routing-table "$TABLE" --max-model-len "$TARGET_L" --concurrency "$F22S_ESC_CONC" --items "$items"
        sleep 2
      done
      f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
      for line in "${EXT_STATIC[@]}"; do
        IFS=$'\t' read -r fam cell items <<< "$line"
        ev=$(f22s_escalation_event "$fam" "$cell" "$PRIVATE_RUN_ESC16"); erc=$?
        f22s_event "$ev"
        [ "$erc" = 0 ] || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); f22_log "escalation record PROBLEM for $fam/$cell @$F22S_CEILING: ${ev:0:400}"; }
      done
      f22s_write_ladder_ext static "$LADDER_DIR/static_ladder_t${TRIAL}.json" "$F22S_STATIC_RUNGS_EXT"
      f22_standup_stop
    fi
    # 4b. CONTROLS (thinking OFF): the promoted (cell, control) pairs grouped per cell, one driver per cell with its --items files
    declare -A EXT_CTLS=() EXT_ITEMS=()
    while IFS=$'\t' read -r cell ctl items; do
      [ -n "$cell" ] || continue
      EXT_CTLS[$cell]="${EXT_CTLS[$cell]:+${EXT_CTLS[$cell]},}$ctl"; EXT_ITEMS[$cell]="${EXT_ITEMS[$cell]:+${EXT_ITEMS[$cell]} }$items"
    done < "$EXT_DIR/promoted_controls.tsv"
    if [ "${#EXT_CTLS[@]}" -gt 0 ]; then
      f22s_ctl_top_logprobs
      f22s_stop_requested && { f22_log "STOP present — not standing up the controls $F22S_CEILING rung"; exit 0; }
      f22_standup attacks "$SERVE_DIR/controls_16384" "" capability:short "$(f22s_rung_window "$F22S_CEILING")" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }
      f22s_positive_control controls_16384 || { RC=1; exit 1; }
      f22s_hop_guard "$F22S_CEILING" controls "$SERVE_DIR/hop_guard_post_controls.json" "$SERVE_DIR/controls_16384/endpoint.json" || { RC=1; exit 1; }
      f22s_hop_guard_event controls controls "$F22S_CEILING" "$SERVE_DIR/hop_guard_post_controls.json"
      NRUNG=${#EXT_CTLS[@]}; CONC=$(( TARGET_S / NRUNG )); [ "$CONC" -ge 8 ] || CONC=8; MAX_DRIVERS=$((NRUNG + 1))
      f22_log "phase LADDER_EXTEND t$TRIAL controls: $NRUNG cells, concurrency/cell=$CONC, from $(ROOT_OF "$F22S_STATIC_CAP_ESC") into $(ROOT_OF "$F22S_CEILING")"
      for cell in "${CELLS[@]}"; do
        [ -n "${EXT_CTLS[$cell]:-}" ] || continue
        f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break; }
        items_args=(); for it in ${EXT_ITEMS[$cell]}; do items_args+=(--items "$it"); done
        f22s_launch_cell "controls16384__${cell}" "$cell" "$MPY" scripts/served_capability_run.py generate \
          --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
          --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$(ROOT_OF "$F22S_CEILING")" \
          --controls "${EXT_CTLS[$cell]}" --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS" \
          --vllm-log "$SERVE_DIR/controls_16384/vllm_server.log" \
          --controls-cap "$F22S_CEILING" --escalate-from-root "$(ROOT_OF "$F22S_STATIC_CAP_ESC")" --floor-sample "$F22S_FLOOR_SAMPLE_K" "${items_args[@]}"
        sleep 3
      done
      f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
      for cell in "${CELLS[@]}"; do
        [ -n "${EXT_CTLS[$cell]:-}" ] || continue
        IFS=',' read -ra ctls <<< "${EXT_CTLS[$cell]}"
        for ctl in "${ctls[@]}"; do
          n=$(f22s_at_cap_controls "$(ROOT_OF "$F22S_CEILING")" "$cell" "$ctl") || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); continue; }
          f22s_event "$(f22s_event_json kind=controls cell="$cell" control="$ctl" rung="$F22S_CEILING" n_at_cap="$n" root="$(ROOT_OF "$F22S_CEILING")")"
          f22_log "at-cap @$F22S_CEILING $cell/$ctl: $n$( [ "$n" -gt 0 ] && echo ' (residual at the CEILING: recorded, never escalated further)')"
        done
      done
      f22s_write_ladder_ext controls "$LADDER_DIR/ladder.json" "$F22S_CONTROLS_RUNGS_EXT"
      f22_standup_stop
    fi
    # 4c. GSM8K (thinking ON): its own standup — the certification must pass before any request — and the readout rung driver
    declare -A EXT_GSM=()
    while IFS=$'\t' read -r cell ctl items; do
      [ -n "$cell" ] || continue
      [ "$ctl" = gsm8k ] || f22_fatal "the gsm8k extend list names control $ctl for $cell — not the GSM8K readout"
      EXT_GSM[$cell]="$items"
    done < "$EXT_DIR/promoted_gsm8k.tsv"
    if [ "${#EXT_GSM[@]}" -gt 0 ]; then
      f22s_ctl_top_logprobs
      f22s_stop_requested && { f22_log "STOP present — not standing up the gsm8k $F22S_CEILING rung"; exit 0; }
      F22_THINKING_ON=1
      f22_standup attacks "$SERVE_DIR/gsm8k_thinking_on_16384" "" capability:short "$(f22s_rung_window "$F22S_CEILING")" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }
      f22s_positive_control gsm8k_thinking_on_16384 || { RC=1; exit 1; }
      f22s_hop_guard "$F22S_CEILING" controls "$SERVE_DIR/hop_guard_post_gsm8k.json" "$SERVE_DIR/gsm8k_thinking_on_16384/endpoint.json" || { RC=1; exit 1; }
      f22s_hop_guard_event controls gsm8k "$F22S_CEILING" "$SERVE_DIR/hop_guard_post_gsm8k.json"
      NRUNG=${#EXT_GSM[@]}; CONC=$(( TARGET_S / NRUNG )); [ "$CONC" -ge 8 ] || CONC=8; MAX_DRIVERS=$((NRUNG + 1))
      f22_log "phase LADDER_EXTEND t$TRIAL gsm8k: thinking=$TARGET_THINKING_MODE (certification $TARGET_THINKING_VERDICT) $NRUNG cells, concurrency/cell=$CONC, from $GSM_ROOT8 into $GSM_ROOT16"
      for cell in "${CELLS[@]}"; do
        [ -n "${EXT_GSM[$cell]:-}" ] || continue
        f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break; }
        f22s_launch_cell "gsm8k16384__${cell}" "$cell" "$MPY" scripts/served_capability_run.py generate \
          --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
          --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$GSM_ROOT16" \
          --controls gsm8k --gsm8k-arm sampled --standup-endpoint "$SERVE_DIR/gsm8k_thinking_on_16384/endpoint.json" \
          --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS" --vllm-log "$SERVE_DIR/gsm8k_thinking_on_16384/vllm_server.log" \
          --controls-cap "$F22S_CEILING" --escalate-from-root "$GSM_ROOT8" --floor-sample "$F22S_FLOOR_SAMPLE_K" --items "${EXT_GSM[$cell]}"
        sleep 3
      done
      f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
      for cell in "${CELLS[@]}"; do
        [ -n "${EXT_GSM[$cell]:-}" ] || continue
        f22s_assert_gsm8k_items "$GSM_ROOT16" "$cell" || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); continue; }
        n=$(f22s_at_cap_controls "$GSM_ROOT16" "$cell" gsm8k) || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); continue; }
        f22s_event "$(f22s_event_json kind=controls extend_kind=gsm8k cell="$cell" control=gsm8k rung="$F22S_CEILING" n_at_cap="$n" root="$GSM_ROOT16")"
        f22_log "at-cap @$F22S_CEILING $cell/gsm8k: $n"
      done
      f22s_write_ladder_ext controls "$LADDER_DIR/gsm8k_ladder.json" "$F22S_GSM8K_RUNGS" "" gsm8k
      f22_standup_stop; F22_THINKING_ON=""
    fi
    # 5. the CEILING REPORT: reclassify the rungs now present and record every pair's residual SHARE and COUNT at
    #    16,384 in ceiling_16384.json — the ladder STOPS here; nothing escalates past 16,384. A failed driver leaves a
    #    promoted pair without its 16384 store, which the report refuses (never mistaken for complete): resubmit.
    if [ "$TOTAL_FAIL" -eq 0 ]; then
      f22s_reclassify static "$EXT_DIR/reclassify_static_ceiling" || { RC=1; exit 1; }
      f22s_reclassify controls "$EXT_DIR/reclassify_controls_ceiling" || { RC=1; exit 1; }
      RECL16=(--reclassify "static=$EXT_DIR/reclassify_static_ceiling/reclassify.json" --reclassify "controls=$EXT_DIR/reclassify_controls_ceiling/reclassify.json")
      if [ "$HAVE_GSM" = 1 ]; then
        f22s_reclassify gsm8k "$EXT_DIR/reclassify_gsm8k_ceiling" || { RC=1; exit 1; }
        RECL16+=(--reclassify "gsm8k=$EXT_DIR/reclassify_gsm8k_ceiling/reclassify.json")
      fi
      f22s_ladder_extend_decide "$F22S_CEILING" "" "$CEILING_REC" "${RECL16[@]}" || { RC=1; exit 1; }
      f22s_write_ladder_ext static "$LADDER_DIR/static_ladder_t${TRIAL}.json" "$F22S_STATIC_RUNGS_EXT"
      f22s_write_ladder_ext controls "$LADDER_DIR/ladder.json" "$F22S_CONTROLS_RUNGS_EXT"
      [ "$HAVE_GSM" = 1 ] && f22s_write_ladder_ext controls "$LADDER_DIR/gsm8k_ladder.json" "$F22S_GSM8K_RUNGS" "" gsm8k
    else
      f22_log "phase LADDER_EXTEND t$TRIAL: $TOTAL_FAIL failed drivers / records — the ceiling report is NOT written (a promoted pair lacks its $F22S_CEILING store); fix and resubmit"
    fi
  fi

# ---------------------------------------------------------------- capability_t<T>: GSM8K thinking-ON @8192 (sampled only) + MMLU (thinking OFF)
elif [ "$PHASE_KIND" = capability ]; then
  # a LONG-SI prompted cell (F22S_LONG_SI) is REFUSED here UNLESS the roster windows the leg (merge 2026-09-12, with Sample):
  # the GSM8K prompt (~2.4k tokens) + a ~6k-token SI + the 8,192 cap fits neither the long-SI table's 14,848 nor a per-leg
  # view's 8192 rung (both sized for the static max prompt), so a long-SI cell on a view that RUNS gsm8k refuses; and the
  # long-SI table's 6,656 controls window was measured for xstest / or_bench prompts (max 154) + 512, not the MMLU prompt, so
  # a long-SI cell on a view with NO roster controls window refuses. llama8b/shortconst (capability_controls [mmlu], roster
  # controls window 6,656 measured for the MMLU letter prefill + its SI) passes; gemma / qwen / nemotron main+it with the
  # constitution cell still refuse (both conditions hold) — the capability battery of that arm is another lane's instrument.
  if [ -n "${F22S_LONG_SI:-}" ]; then
    case ",$VIEW_CONTROLS," in *,gsm8k,*) f22_fatal "capability_t$TRIAL: a long-SI prompted cell is among ${CELLS[*]} and $KEY/$VIEW runs gsm8k (roster controls: $VIEW_CONTROLS) — no measured GSM8K window exists for it on any view (the long-SI table's 14,848 and a per-leg view's 8192 rung are sized for the static max prompt, not ~2.4k GSM8K + SI + 8,192); refusing";; esac
    [ -n "${VIEW_WINDOW_CONTROLS:-${VIEW_WINDOW:-}}" ] || f22_fatal "capability_t$TRIAL: a long-SI prompted cell is among ${CELLS[*]} and $KEY/$VIEW declares no roster controls window — the long-SI table's 6,656 was measured for xstest/or_bench prompts (max 154) + 512, not the MMLU prompt; refusing"
  fi
  GSM_ROOT="$(f22s_gsm8k_root "$F22S_GSM8K_CAP")"; MMLU_ROOT="$(f22s_mmlu_root)"
  f22s_ctl_top_logprobs
  # the GSM8K block runs ONLY where the roster's controls list carries gsm8k (capability_omitted names the reason otherwise —
  # the sdfbase-llama8b view omits it: inspect/gsm8k_paired200 is the paper's one GSM8K instrument); a view without
  # it goes straight to MMLU — never a readout the roster omits, never a GSM8K server stood up for nothing
  case ",$VIEW_CONTROLS," in *,gsm8k,*) RUN_GSM8K=1;; *) RUN_GSM8K=""; f22_log "phase CAPABILITY t$TRIAL: gsm8k is not in $KEY/$VIEW's roster controls ($VIEW_CONTROLS) — skipping the GSM8K block (capability_omitted); MMLU only";; esac
  if [ -n "$RUN_GSM8K" ]; then
  # 1. GSM8K, thinking ON, the 8,192 cap: the PRE-standup hop guard (the same censoring failure shape as a rung), the
  #    thinking-ON standup at the model's 8192-rung window (f22_standup refuses unless the certification is servable and
  #    both stamps attest thinking on), the positive control under --thinking-on, the POST-standup guard (mandatory)
  f22s_hop_guard "$F22S_GSM8K_CAP" controls "$SERVE_DIR/hop_guard_pre_gsm8k.json" || { RC=1; exit 1; }
  f22s_stop_requested && { f22_log "STOP present — not standing up"; exit 0; }
  F22_THINKING_ON=1
  f22_standup attacks "$SERVE_DIR/gsm8k_thinking_on" "" capability:short "$(f22s_gsm8k_window)" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }
  f22s_positive_control gsm8k_thinking_on || { RC=1; exit 1; }
  f22s_hop_guard "$F22S_GSM8K_CAP" controls "$SERVE_DIR/hop_guard_post_gsm8k.json" "$SERVE_DIR/gsm8k_thinking_on/endpoint.json" || { RC=1; exit 1; }
  f22s_hop_guard_event controls gsm8k "$F22S_GSM8K_CAP" "$SERVE_DIR/hop_guard_post_gsm8k.json"
  CONC=$(( TARGET_S / NCELLS )); [ "$CONC" -ge 8 ] || CONC=8
  MAX_DRIVERS=$((NCELLS + 1))
  f22_log "phase CAPABILITY t$TRIAL gsm8k: thinking=$TARGET_THINKING_MODE (certification $TARGET_THINKING_VERDICT) cap=$F22S_GSM8K_CAP arm=sampled (no greedy arm, S-D7) cells=$NCELLS concurrency/cell=$CONC (server width $TARGET_S x $TARGET_L) out=$GSM_ROOT"
  for cell in "${CELLS[@]}"; do
    f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break; }
    f22s_launch_cell "gsm8k__${cell}" "$cell" "$MPY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$GSM_ROOT" \
      --controls gsm8k --gsm8k-arm sampled --standup-endpoint "$SERVE_DIR/gsm8k_thinking_on/endpoint.json" \
      --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS" --vllm-log "$SERVE_DIR/gsm8k_thinking_on/vllm_server.log"
    sleep 3
  done
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "phase CAPABILITY gsm8k done: $NFAIL failed drivers; asserting the item set and recording the at-cap counts (gsm8k ladder rung $F22S_GSM8K_CAP)"
  for cell in "${CELLS[@]}"; do
    f22s_assert_gsm8k_items "$GSM_ROOT" "$cell" || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); continue; }
    n=$(f22s_at_cap_controls "$GSM_ROOT" "$cell" gsm8k) || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); continue; }
    f22s_event "$(f22s_event_json kind=controls extend_kind=gsm8k cell="$cell" control=gsm8k rung="$F22S_GSM8K_CAP" n_at_cap="$n" root="$GSM_ROOT")"
    f22_log "at-cap @$F22S_GSM8K_CAP $cell/gsm8k: $n"
  done
  f22s_write_ladder_ext controls "$LADDER_DIR/gsm8k_ladder.json" "$F22S_GSM8K_RUNGS" "" gsm8k
  f22_standup_stop; F22_THINKING_ON=""
  fi   # RUN_GSM8K
  # 2. MMLU, thinking OFF, the wave's controls geometry (capability:short 2048, --max-logprobs 64): a letter-logprob argmax,
  #    sampler-moot by construction — the "sampled" MMLU measures server-to-server logprob nondeterminism only
  f22s_stop_requested && { f22_log "STOP present — not standing up for MMLU"; exit $(( TOTAL_FAIL > 0 ? 2 : 0 )); }
  f22_standup attacks "$SERVE_DIR/mmlu" "" capability:short "$(f22s_base_window controls)" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }
  f22s_positive_control mmlu || { RC=1; exit 1; }
  # the MMLU block's OWN per-cell concurrency: the width filled across the cells, floor 8 — computed here, not inherited from
  # the GSM8K block above, which every roster that omits gsm8k SKIPS (2026-09-16: an MMLU-only capability leg died at this
  # line with `CONC: unbound variable` under set -u, after its standup and positive controls had passed)
  MMLU_CONC=$(( TARGET_S / NCELLS )); [ "$MMLU_CONC" -ge 8 ] || MMLU_CONC=8
  f22_log "phase CAPABILITY t$TRIAL mmlu: thinking=$TARGET_THINKING_MODE cells=$NCELLS concurrency/cell=$MMLU_CONC (server width $TARGET_S) logprobs=$CTL_TOP_LOGPROBS out=$MMLU_ROOT"
  for cell in "${CELLS[@]}"; do
    f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break; }
    f22s_launch_cell "mmlu__${cell}" "$cell" "$MPY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$MMLU_ROOT" \
      --controls mmlu --concurrency "$MMLU_CONC" --top-logprobs "$CTL_TOP_LOGPROBS" \
      --vllm-log "$SERVE_DIR/mmlu/vllm_server.log"
    sleep 3
  done
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "phase CAPABILITY mmlu done: $NFAIL failed drivers"
  f22_standup_stop

# ---------------------------------------------------------------- gsm8k_t<T>: the GSM8K readout, vendor-preset arm (+ greedy comparator in t0)
elif [ "$PHASE_KIND" = gsm8k ]; then
  f22s_view_has_leg capability || f22_fatal "view $KEY/$VIEW has no capability leg (legs: $VIEW_LEGS): no gsm8k readout"
  # the capability leg may be MMLU-only (roster capability_controls; the sdfbase-llama8b roster removed gsm8k, 2026-09-10): refuse, never a readout the roster omits
  case ",$VIEW_CONTROLS," in *,gsm8k,*) ;; *) f22_fatal "view $KEY/$VIEW does not run gsm8k (roster controls: $VIEW_CONTROLS; capability_omitted names the reason) — no gsm8k readout for it";; esac
  # the readout's window: cap 8192 (sampled_decode.readouts.gsm8k.cap) + the gsm8k prompts -> the 8192-rung window of
  # this key / view (10240; nemotron 8704; a view with its own window serves there: llama8b/fullconst 45,056)
  GSM_WINDOW=$(f22s_rung_window 8192)
  f22_window_check "$SERVE_DIR/gsm8k" "$GSM_WINDOW"
  F22_LOAD_CHECK_CAP="${F22_LOAD_CHECK_CAP:-8192}" f22_standup attacks "$SERVE_DIR/gsm8k" "" capability:short "$GSM_WINDOW" || { RC=1; exit 1; }
  f22s_positive_control gsm8k || { RC=1; exit 1; }
  ARMS=(sampled); [ "$TRIAL" = 0 ] && ARMS=(sampled greedy)   # the greedy comparator once, on the same items
  CONC=$(( TARGET_S / (NCELLS * ${#ARMS[@]}) )); [ "$CONC" -ge 8 ] || CONC=8
  MAX_DRIVERS=$((NCELLS * ${#ARMS[@]} + 1))
  f22_log "phase GSM8K t$TRIAL (window $GSM_WINDOW x $TARGET_S): cells=$NCELLS arms=${ARMS[*]} concurrency/driver=$CONC cap=8192 thinking=$(python3 -c 'import sys,yaml; print(yaml.safe_load(open(sys.argv[1]))["sampled_decode"]["readouts"]["gsm8k"]["thinking"])' "$CFG")"
  for arm in "${ARMS[@]}"; do for cell in "${CELLS[@]}"; do
    f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break 2; }
    f22s_launch_cell "gsm8k_${arm}__${cell}" "$cell" "$MPY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$WAVE_ROOT/sampled/gsm8k_${arm}/${TAG}_t${TRIAL}" \
      --controls gsm8k --gsm8k-arm "$arm" --concurrency "$CONC" \
      --vllm-log "$SERVE_DIR/gsm8k/vllm_server.log"
    sleep 3
  done; done
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "phase GSM8K done: $NFAIL failed drivers"
  f22_standup_stop

else
  # ONE dispatch chain, closed by a refusal: a phase kind the grammar accepts but this script does not implement must
  # exit non-zero HERE, never fall through every block and report "done: 0 failed drivers" (a silent no-op success;
  # the 2026-09-10 review caught ladder_extend_t<T> / capability_t<T> doing exactly that before their blocks landed)
  f22_fatal "phase kind $PHASE_KIND ($PHASE) is not implemented in target_sampled.sbatch — refusing to report a phase that ran nothing as done"
fi
RC=$(( TOTAL_FAIL > 0 ? 2 : 0 ))
f22_log "phase $PHASE done: $TOTAL_FAIL failed drivers / records"
exit $RC
