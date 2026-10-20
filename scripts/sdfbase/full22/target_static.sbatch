#!/bin/bash
#SBATCH --job-name=f22-stat
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
# full22: ONE steered TARGET SERVER (1 GPU) + the SINGLE-SHOT drivers for one (target, view) config, in
# up to TWO GEOMETRY PHASES on the same GPU (one standup per geometry; the second starts after the first
# is stopped — never two servers on one card):
#   phase STATIC  (pasttense / codeattack / renellm / grok420 at the 4,096 completion cap, and best-of-75 on
#                 the 100 JBB goals): --eval families_served:short = max_model_len 4864 (the measured p99
#                 static prompt 762 + the 4,096 cap; a 4,096 window would clamp EVERY completion below the
#                 cap) x max_num_seqs sized from the model's KV budget (geometry.ModelBudget.seqs_at:
#                 gemma4e4b 121, qwen35_4b 148 [from the table's 720,896; a Modal-only 786,432 would give 161 —
#                 documented in docs/served_geometry_p99.md §7, not a table row], nemotron3nano4b / glm46vflash 161) — the leg with no attacker
#                 in the loop takes the full stacked target-side gain (step time ~flat in batch to ~100
#                 concurrent on the Modal timing rows; a ~3x advantage at ~100 concurrent is EXTRAPOLATED,
#                 not measured).
#                 LONG-SI CELLS (bon_long_si.sh, the ported F22S_LONG_SI mechanism): a TABLE
#                 view whose job serves prompted__si_constitution / prompted__si_irrelevant stands this phase up at the
#                 long-SI table's window instead — 10,752 with a static family in the job, 6,656 for the BoN-only 512-store
#                 job (-3; the BoN prompt under the SI + the 512 cap) — after the rendered-prompt window check, which steps
#                 the window to the measured need; at 4,864 every constitution-SI BoN slot was refused prompt_over_window.
#   phase CONTROLS (served gsm8k / mmlu / xstest / or_bench_hard_1k, short prompts, 512 completions):
#                 --eval capability:short --max-model-len 2048, S DERIVED from the model's budget at that
#                 window (geometry.ModelBudget.seqs_at: gemma4e4b 288, qwen35_4b 256 = the hybrid cap) — the
#                 launcher's strict identity refuses an explicit S that is not budget // L (the smoke's
#                 `2048 256` was refused: identity_seqs(2048) = 288 for gemma), so only the window is passed.
# The drivers FILL the width: static shards are queued so ~max_num_seqs jailbreak_run processes are in
# flight (the earlier wave was driver-bound at 13% KV); BoN runs the async fan-out driver
# (scripts/sdfbase/full22/bon_served_run.py, bon_served.bon_generate_served) at max_num_seqs / n_cells
# in-flight per cell; the controls driver's --concurrency is 256 / n_cells. Every standup pins the STACKED
# lens, requires the workers to ATTEST it AND the resolved geometry, and is refused otherwise.
# Usage (positional only — never --export=):
#   sbatch -J f22-stat-<key>-<view>-<n> scripts/sdfbase/full22/target_static.sbatch \
#       <key> <view> <families> <wave_sha> [nshards=9] [cells=all]
#     families  comma list from pasttense,codeattack,renellm,grok420,bon,controls
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; FAMILIES="${3:?families}"; WAVE_SHA="${4:?wave sha}"
NSHARDS="${5:-9}"; CELLS_ARG="${6:-}"
# cells-fix-2: a job that serves a SUBSET of the view's cells registers that subset — F22_REGISTRY_CELLS narrows the
# registration overlay (common.sh, inside f22_resolve_config below), so the export sits here, before f22_init; the plan sets both (positional 6 + env)
[ -n "$CELLS_ARG" ] && export F22_REGISTRY_CELLS="${F22_REGISTRY_CELLS:-$CELLS_ARG}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")   # the launch's cells for the kind-aware serving-path default (2026-09-15): the (KEY, VIEW) campaign manifest
f22_init "$WAVE_SHA"
# THE 512 STORE'S REGISTRATION FACTS (bon_long_si.sh): the BoN-only job (f22-stat-<tag>-3, FAMILIES=bon — the recipe's store512
# unit) registers its runs at the DRAWN cap = rung 0 of the config's best-of-N ladder [bon.max_new_tokens, bon.climb_cap] = [512, 4096]
# (pass base_cap; the one-run-per-rung rule) — exported BEFORE f22_resolve_config registers; a mixed job (-all: the statics
# at 4,096 beside the store) registers at the instrument's read-at cap as before, and so does a one-rung lane (a 4,096 store)
source scripts/sdfbase/full22/bon_long_si.sh
# THE RERUN PAIR'S STORE (stage.py rerun-units -> the plan row's F22_BON_RERUN=<slug>, FAMILIES=bon, the cells on positional 6):
# re-rooted to $WAVE_ROOT/rerun/<slug> HERE — before anything derives from WAVE_ROOT (SERVE_DIR, PRIVATE_ROOT, the closer's --wave-root, the
# STOP sentinel) and before f22_resolve_config registers under the producer key the exported segment shapes — so the rerun never resumes the
# superseded store's dir or re-registers its run id (bon_long_si.sh f22_bon_rerun_root); unset = this job byte for byte as before
f22_bon_rerun_root store "$FAMILIES" "$CELLS_ARG"
f22_bon_read_caps "$KEY" "$VIEW"
[ "$FAMILIES" = bon ] && f22_bon_registry_exports store
f22_resolve_config "$KEY" "$VIEW"
f22_bon_assert_config
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
# THE VIEW'S LEGS, WINDOW AND CONTROLS (2026-09-10): read through roster_view.py (the same
# f22_roster readers the generator and the printers use). A family this view does not run is refused up front; a view
# with a serving window (llama8b/fullconst 45,056) stands EVERY phase up at it — a view with PER-LEG windows (llama8b/shortconst)
# stands the static phase up at its static window and the controls phase at its controls window; the controls driver's --controls list is
# the view's (capability mmlu — gsm8k RETIRED 2026-09-14 wave-wide, roster controls_retired — and/or over-refusal xstest,or_bench_hard_1k)
VIEW_LEGS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" legs --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's legs for $KEY/$VIEW"
VIEW_WINDOW=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" max_model_len --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's window for $KEY/$VIEW"
# PER-LEG windows (the llama8b shortconst view, 2026-09-12; f22_roster.view_windows): the static standup's (BoN rides it) and the
# controls standup's; a one-window view (fullconst) answers both with its window, a table view answers "" (the defaults below stand)
VIEW_WINDOW_STATIC=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window static --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's static window for $KEY/$VIEW"
VIEW_WINDOW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window controls --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's controls window for $KEY/$VIEW"
# THE gsm8k CONTROLS WINDOW (2026-09-14): a gsm8k-bearing TABLE view answers the roster's controls_window.gsm8k (4,608: the
# 10-shot Inspect gsm8k prompt, up to 2,462 tokens with the safety turn, + the 512 cap overflowed capability:short's 2,048 and vLLM
# refused 34 of 200 items); a view with its own serving override keeps it (VIEW_WINDOW_CONTROLS wins); "" for a view that runs no
# gsm8k (llama8b: mmlu only), where the 2,048 default stands. f22_gsm8k_window_check holds the chosen window to the derived minimum.
VIEW_WINDOW_GSM8K=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window gsm8k --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's gsm8k controls window for $KEY/$VIEW"
VIEW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" controls --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's controls list for $KEY/$VIEW"
FAMTAG=$(echo "$FAMILIES" | tr ',' '-')
SERVE_DIR="$WAVE_ROOT/serving/static_${TAG}_${FAMTAG}_${SLURM_JOB_ID}"; PRIVATE_ROOT="$WAVE_ROOT/transcripts_results"
mkdir -p "$SERVE_DIR/drivers" "$PRIVATE_ROOT"; chmod 700 "$PRIVATE_ROOT"
IFS=',' read -ra FAMS <<< "$FAMILIES"
STATIC_FAMS=(); BON=0; CONTROLS=0
for fam in "${FAMS[@]}"; do case "$fam" in
  pasttense|codeattack|renellm|grok420) f22_view_has_leg static || f22_fatal "view $KEY/$VIEW has no static leg (legs: $VIEW_LEGS)"; STATIC_FAMS+=("$fam");;
  bon) f22_view_has_leg bon || f22_fatal "view $KEY/$VIEW has no bon leg (legs: $VIEW_LEGS)"; BON=1;;
  controls) [ -n "$VIEW_CONTROLS" ] || f22_fatal "view $KEY/$VIEW has neither a capability nor a controls leg (legs: $VIEW_LEGS)"; CONTROLS=1;;
  *) f22_fatal "unknown family $fam";; esac; done
case ",$FAMILIES," in *,renellm,*)
  export SISTEER_HELDOUT_CORPUS_DIR="${SISTEER_HELDOUT_CORPUS_DIR:-$SHARED/exp/heldout_corpora}"
  # the corpus replay reads: this cluster's copy at $SISTEER_HELDOUT_CORPUS_DIR/renellm.jsonl when there is one, else
  # the paper's corpus as the repository ships it (data/heldout_corpora/renellm.jsonl). The paper's replays as is, any
  # other only when SISTEER_RENELLM_NONPAPER_CORPUS_SHA names it (its results are labelled non-paper). Checked here,
  # under the system python (stdlib only), before anything is stood up
  python3 -I -S scripts/renellm_corpus_check.py \
    || f22_fatal "renellm: the corpus check refused the replay corpus (the reason is above); nothing was stood up";;
esac
f22_driver_venv; f22_serving_venv; f22_bank_view "$MPY"
# THE CONTROLS DRIVER'S INTERPRETER FOLLOWS THE VIEW'S CONTROLS LIST. An INSPECT control — gsm8k on the gemma4e4b /
# qwen35_4b / nemotron3nano4b `main` views (roster_view.py <key> main controls -> gsm8k,mmlu,xstest,or_bench_hard_1k) —
# needs inspect_ai + inspect_evals at the client lock's pins, which the driver venv ($MPY: uv sync --extra cu128) does
# not carry: under it the driver refuses the whole leg by name ("framework 'inspect_ai' is required by a requested
# control but cannot be imported" — smoke 502126). So the CLIENT venv (common.sh f22_client_venv -> $CPY — the gsm8k /
# math500 lanes' path and the Modal launcher's /opt/client, built from the same lock) is bootstrapped and used exactly
# when the view's list intersects the Inspect vocabulary (contracts.INSPECT_CONTROLS, read by import through
# f22_inspect_controls — never a hand-copied list). A LEGACY-ONLY list (the llama8b views: mmlu[,xstest,
# or_bench_hard_1k] — the driver's own SERVED_CONTROLS, no framework) runs under $MPY as it always did: the client lock
# is 259 pinned packages incl. a CPU torch, minutes on a fresh node, and an unrelated Inspect dependency failure must
# not abort a job that never imports Inspect. The Inspect leg ALSO reads the `datasets` library OFFLINE from a prepared
# cache (f22_datasets_guard: HF_DATASETS_CACHE=$SHARED/hf_datasets, as the gsm8k lane does; the shared cache holds
# openai/gsm8k under other load signatures only — smoke 502566), guarded by name here. Both before any standup, so an
# unresolvable lock or a missing cache fails in the job's first minutes and not after the static phase's GPU hours. The
# SMOKE keeps its superset list and therefore always the client venv: it exists to exercise the Inspect path
# (smoke_full22.sbatch).
CTL_PY="$MPY"; CTL_INSPECT=""; CTL_ENV=()
if [ "$CONTROLS" = 1 ]; then
  CTL_INSPECT=$(f22_inspect_controls "$VIEW_CONTROLS") || f22_fatal "cannot read contracts.INSPECT_CONTROLS under $MPY"
  if [ -n "$CTL_INSPECT" ]; then
    f22_client_venv; CTL_PY="$CPY"; f22_datasets_guard "$CTL_INSPECT"
    CTL_ENV=(HF_DATASETS_CACHE="$HF_DATASETS_CACHE" HF_DATASETS_OFFLINE=1)
    f22_log "phase CONTROLS interpreter: the client venv $CTL_PY (Inspect controls in the view's list: $CTL_INSPECT; datasets cache $HF_DATASETS_CACHE)"
  else
    f22_log "phase CONTROLS interpreter: the driver venv $CTL_PY (legacy controls only: $VIEW_CONTROLS; no Inspect framework, no client venv)"
  fi
fi
if [ -n "$CELLS_ARG" ]; then IFS=',' read -ra CELLS <<< "$CELLS_ARG"; else CELLS=("${ALL_CELLS[@]}"); fi
NCELLS=${#CELLS[@]}
# LONG-SI CELLS (the F22S_LONG_SI mechanism, ported in bon_long_si.sh — the static legs already serve the
# constitution cell at 10,752 this way): a TABLE view (no roster static window) whose job serves a long-SI prompted cell stands the static
# phase up at the long-SI table's window, and VIEW_WINDOW_STATIC carries the static phase's window from here on — the roster's when the
# view declares one (per-leg / one-window), else 10,752 with a static family in the job (the static max prompt + the 4,096 cap), else 6,656
# for the BoN-only 512-store job (the BoN max prompt under the SI + the 512 cap; a 4,096 store takes 10,752; the window check in the phase
# steps it to the measured need if that asks) — never the table's 4,864, at which every constitution-SI slot of the demo's store (983259)
# was refused prompt_over_window; no long-SI cell among the job's cells = "" (the table's window), byte for byte as before
f22_bon_detect_long_si "${CELLS[@]}"
BON_LONG_SI_LEGS=""
if [ -z "$VIEW_WINDOW_STATIC" ] && [ -n "$F22_BON_LONG_SI" ]; then
  if [ "${#STATIC_FAMS[@]}" -gt 0 ]; then VIEW_WINDOW_STATIC=$(f22_bon_window static); BON_LONG_SI_LEGS=static; else VIEW_WINDOW_STATIC=$(f22_bon_window store); fi
  [ "$BON" = 1 ] && BON_LONG_SI_LEGS="${BON_LONG_SI_LEGS:+$BON_LONG_SI_LEGS,}bon"
fi
TARGET_PID=""; STANDUP_STOP=""; declare -a DRIVER_PIDS=(); declare -A DRIVER_TAG=(); RC=0; TOTAL_FAIL=0
cleanup(){
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  for p in "${DRIVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true; done
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  f22_result "static $TAG $FAMTAG rc=${RC} failed_drivers=${TOTAL_FAIL}"
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"
# ---------------------------------------------------------------- phase STATIC (+ BoN): 4864 x budget S
# a prompted-only view (llama8b/fullconst): the SI prefix rides every request — the load check sends it too, at the
# static escalation rung's cap (the plan's smoke: 5 concurrent full-prefix requests at 8192), and the window check runs first
if f22_view_prompted_only; then
  export F22_LOAD_CHECK_SYSTEM_SI="${F22_LOAD_CHECK_SYSTEM_SI:-$(f22_prompted_si)}"
  # the 8,192 probe cap is the ONE-WINDOW view's (fullconst: 45,056 carries the full prefix at the escalation rung); a PER-LEG view
  # (shortconst) probes each standup at the RUN's cap — its static window is sized for the 4,096 cap and would refuse an 8,192 probe
  [ -z "$VIEW_WINDOW" ] || export F22_LOAD_CHECK_CAP="${F22_LOAD_CHECK_CAP:-8192}"
fi
if [ "${#STATIC_FAMS[@]}" -gt 0 ] || [ "$BON" = 1 ]; then
  # a PER-LEG view's static standup carries the static families at their BASE cap only (its 8,192 rung has its own standup and check);
  # a ONE-WINDOW view's carries the rung too (the default: base + 8,192)
  # the long-SI window check (bon_long_si.sh; nothing without a long-SI cell on a table view): the served legs at their BASE caps under the
  # long-SI cell's SI with the pinned tokenizer, BEFORE any GPU minute; a refusal steps the window once to the measured need, never a clamp
  if [ -n "$BON_LONG_SI_LEGS" ]; then f22_bon_window_check "$SERVE_DIR/static" "$VIEW_WINDOW_STATIC" "$BON_LONG_SI_LEGS" base "${CELLS[@]}"; VIEW_WINDOW_STATIC="$F22_BON_WINDOW"; fi
  # the served runs file: this standup's runs — the static families' and/or the store's, never the controls' — by served key, so
  # the standup's write_endpoint hook records the attested window / S / engine per run (context_window_tokens was NULL on every sbatch row)
  STATIC_EVALS_CSV="$(IFS=,; echo "${STATIC_FAMS[*]:-}")"; [ "$BON" = 1 ] && STATIC_EVALS_CSV="${STATIC_EVALS_CSV:+$STATIC_EVALS_CSV,}strongreject_bon"
  f22_registry_runs_file "$SERVE_DIR/static" "$STATIC_EVALS_CSV"
  f22_window_check "$SERVE_DIR/static" "${VIEW_WINDOW_STATIC:-4864}" static,bon "$( [ -n "$VIEW_WINDOW" ] || echo base )"
  f22_standup attacks "$SERVE_DIR/static" "" families_served:short ${VIEW_WINDOW_STATIC:+"$VIEW_WINDOW_STATIC"} || { RC=1; exit 1; }
  # the window OBSERVED in the registry spool (bon_long_si.sh): the standup's hook filed each served run's patch with the attested
  # window, read back from the destination the harvester pulls — a quiet hook would leave context_window_tokens NULL; refused here
  f22_registry_serving_recorded "$SERVE_DIR/static"
  # native-serving-default: the cells this standup SERVES (PSR / vector cells serve on the native engine's residual slot since native-psr-serving; it skips AlphaSteer / antidote / the stacked wrapper and a position-masked LoRA fit by name — a cells= positional naming a skipped cell is fatal)
  f22_served_cells_filter CELLS "$CELLS_ARG"; NCELLS=${#CELLS[@]}
  # STEER-IS-LIVE positive control on the first steered cell of this job (common.sh f22_positive_control): the
  # smoke's check, run here so a silently no-op'd object refuses before any driver spends a GPU minute
  for pc in "${CELLS[@]}"; do case "$pc" in none|prompted__*) continue;; esac; f22_positive_control "$pc" static || { RC=1; exit 1; }; break; done   # the first OBJECT cell: a prompted baseline has no scale-0 twin
  # ~one in-flight request per driver: fill the attested width, never more — and never more than DRIVER_CAP
  # processes either (launch-smoke ruling 2026-09-07: concurrency with headroom; a static driver's import
  # baseline is ~0.33 GB RSS, so 96 drivers + the server sit well inside the 200G host-RAM cgroup)
  DRIVER_CAP=96
  MAX_DRIVERS=$(( TARGET_S < DRIVER_CAP ? TARGET_S : DRIVER_CAP ))
  f22_log "phase STATIC: families=${STATIC_FAMS[*]:-none} bon=$BON cells=$NCELLS nshards=$NSHARDS server_width=$TARGET_S max_drivers=$MAX_DRIVERS (cap $DRIVER_CAP)"
  if [ "$BON" = 1 ]; then
    # BoN is ITS OWN SUB-PHASE, run to completion before the static families (2026-09-07): the fan-out driver holds
    # max_num_seqs / n_cells requests in flight per cell, so with BoN and static interleaved the static queue was
    # left with the floor of 8 drivers for the whole phase — the two _it-all jobs took 12 h for ~4 h of work.
    # Sequential phases fill the width twice instead of starving one of them.
    BON_CONC=$(( TARGET_S / NCELLS )); [ "$BON_CONC" -ge 4 ] || BON_CONC=4
    f22_log "phase STATIC/bon: $NCELLS fan-out drivers at $BON_CONC in flight each (~$TARGET_S requests)"
    for cell in "${CELLS[@]}"; do
      f22_stop_requested && break
      # the BoN target's sampler: the SERVED MODEL'S VENDOR PRESET, by declaration
      # (--bon-decode-preset vendor_preset --model-key): the registry's <model>|bon|off|vendor_preset row rides every
      # request (Qwen's presence_penalty 1.5 included), resolved per model by bon_served. F22_BON_DECODE_PRESET=harness_default
      # names the SUPERSEDED grandfathered row (T=1.0, no top_p) — only to FINISH a store drawn under it before
      # 2026-09-17 (the driver refuses to mix or relabel a store), never the default. F22_BON_DECODE_PRESET=canonical (the public
      # recipe's two best-of-N units, reproduce/lib/stage.py) names the registry label whose sampler reproduces the canonical
      # best-of-N instrument the planner binds for the served model (bon_served.canonical_bon_label) — the instrument this job's
      # start-of-job registration binds — so the store draws exactly what it registers under; --model-key carries a key that
      # re-pins the same checkpoint under its own model_id (the worked example) as the identity the rows are filed under.
      # The Qwen base seed-replicate lane's THREE declarations reach the driver as SCALAR env vars —
      # F22_BON_DECODE_PRESET (anchor_match, the driver's explicit block), F22_BON_DECODE_SEED_POLICY (replicate_v1),
      # F22_BON_DECODE_SEED_BASE (the view's seed_base) — set on the submitting shell and carried by Slurm's implicit export
      # (never --export=; a Bash ARRAY never crosses sbatch). Unset = this argv byte for byte as before
      # (tests/test_target_static_bon_argv.py).
      f22_queue_launch "bon__${cell}" "$MPY" scripts/sdfbase/full22/bon_served_run.py \
        --config "$CFG" --defense "$cell" --confirmatory --private-root "$PRIVATE_ROOT" \
        --proxy-url "$PROXY_URL" --routing-table "$TABLE" --max-model-len "$TARGET_L" --max-concurrency "$BON_CONC" \
        --require-engine "$TARGET_ENGINE" \
        --bon-decode-preset "${F22_BON_DECODE_PRESET:-vendor_preset}" --model-key "$KEY" \
        ${F22_BON_DECODE_SEED_POLICY:+--decode-seed-policy "$F22_BON_DECODE_SEED_POLICY"} \
        ${F22_BON_DECODE_SEED_BASE:+--decode-seed-base "$F22_BON_DECODE_SEED_BASE"}
      sleep 3
    done
    f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
    f22_log "phase STATIC/bon done: $NFAIL failed drivers; static families next at max_drivers=$MAX_DRIVERS"
  fi
  for fam in "${STATIC_FAMS[@]}"; do for cell in "${CELLS[@]}"; do for s in $(seq 0 $((NSHARDS-1))); do
    f22_stop_requested && { f22_log "wave STOP present — not launching more drivers"; break 3; }
    f22_queue_launch "${fam}__${cell}__s$(printf '%02d' "$s")" "$MPY" scripts/jailbreak_run.py generate \
      --config "$CFG" --attack "$fam" --defense "$cell" --confirmatory --shard "$s" --nshards "$NSHARDS" \
      --private-root "$PRIVATE_ROOT" --target-engine served --target-proxy-url "$PROXY_URL" \
      --target-routing-table "$TABLE" --target-max-model-len "$TARGET_L"
    sleep 1
  done; done; done
  f22_log "phase STATIC: all ${#DRIVER_PIDS[@]} drivers launched (queued at <= $MAX_DRIVERS in flight); waiting"
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "phase STATIC done: $NFAIL failed drivers"
  f22_standup_stop
fi
# ---------------------------------------------------------------- phase CONTROLS: 2048 x 256
if [ "$CONTROLS" = 1 ]; then
  f22_stop_requested && { f22_log "wave STOP present — skipping controls"; RC=$(( TOTAL_FAIL > 0 ? 2 : 0 )); exit $RC; }
  # ONE number for the server's logprob cap AND the driver's request (2026-09-07): the MMLU leg scores by
  # the top-N logprobs and the engine refuses a request above its --max-logprobs (default 20) — "Requested
  # sample logprobs of 64, which is greater than max allowed: 20" killed all 9 drivers of
  # f22-ctl-gemma4e4b-main 447405 AFTER the fit-provenance step had passed. main's capctrl launcher
  # splices --max-logprobs 64; this lineage's standup never set it. Read the driver's own constant (an
  # import, not a grep of its source) and hand the SAME value to both sides so they cannot drift again.
  CTL_TOP_LOGPROBS=$("$MPY" -c 'import importlib.util,sys; s=importlib.util.spec_from_file_location("scr", sys.argv[1]); m=importlib.util.module_from_spec(s); sys.modules["scr"]=m; s.loader.exec_module(m); print(int(m.MMLU_TOP_LOGPROBS))' scripts/served_capability_run.py) \
    || { f22_log "cannot read MMLU_TOP_LOGPROBS from scripts/served_capability_run.py"; RC=1; exit 1; }
  f22_log "phase CONTROLS: server --max-logprobs = driver --top-logprobs = $CTL_TOP_LOGPROBS"
  # the controls window: the view's own (per-leg / one-window), else the roster's gsm8k window for a gsm8k-bearing table view, else the table's
  CTL_L="${VIEW_WINDOW_CONTROLS:-${VIEW_WINDOW_GSM8K:-2048}}"
  f22_gsm8k_window_check "$CTL_L" "$VIEW_CONTROLS"
  f22_window_check "$SERVE_DIR/controls" "$CTL_L" "$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" legs --roster "$F22_ROSTER" | tr ',' '\n' | grep -x 'controls\|capability' | paste -sd,)"
  f22_standup attacks "$SERVE_DIR/controls" "" capability:short "$CTL_L" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }
  # native-serving-default: the cells this standup SERVES (PSR / vector cells serve on the native engine's residual slot since native-psr-serving; it skips AlphaSteer / antidote / the stacked wrapper and a position-masked LoRA fit by name — a cells= positional naming a skipped cell is fatal)
  f22_served_cells_filter CELLS "$CELLS_ARG"; NCELLS=${#CELLS[@]}
  # the Inspect client's bearer variable (contracts.INSPECT_API_KEY_ENV): the standup's proxy_token else `petri` — the same
  # rule as the smoke and the Modal launcher (smoke 502327 lost its controls leg to the provider's "no key" refusal)
  CTL_PROXY_TOKEN=$("$CTL_PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("proxy_token") or "petri")' "$SERVE_DIR/controls/endpoint.json")
  CONC=$(( TARGET_S / NCELLS )); [ "$CONC" -ge 8 ] || CONC=8
  MAX_DRIVERS="$NCELLS"
  f22_log "phase CONTROLS: cells=$NCELLS concurrency/cell=$CONC (server width $TARGET_S)"
  for cell in "${CELLS[@]}"; do
    # THE INTERPRETER is $CTL_PY (the client venv for an Inspect list, the driver venv for a legacy-only one — chosen above), with
    # the checkout's src + scripts on PYTHONPATH (the client lock installs no sisteer), carried by `env` because f22_queue_launch
    # runs its argv as a command inside `cd $REPO_DIR` with the GPU hidden; CTL_ENV adds the offline datasets cache for an Inspect
    # list (empty otherwise). THE THINKING ARM IS NEVER IMPLICIT (contracts.FLAG_THINKING, P33): with a tinycap control in the
    # view's list (gsm8k) the driver refuses a run with no --thinking; the arm is the standup's ATTESTED regime, TARGET_THINKING_MODE
    # (exported by f22_standup from endpoint.json + /v1/models after refusing a mismatch with F22_THINKING_ON) — lane C's smoke fix,
    # which the wave's leg lacked; inert on a legacy-only list (the driver reads it only for tinycap atoms).
    f22_queue_launch "controls__${cell}" env PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" TINYCAP_API_KEY="$CTL_PROXY_TOKEN" ${CTL_ENV[@]+"${CTL_ENV[@]}"} "$CTL_PY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$WAVE_ROOT/controls/$TAG" \
      --controls "$VIEW_CONTROLS" --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS" \
      --thinking "$TARGET_THINKING_MODE" \
      --vllm-log "$SERVE_DIR/controls/vllm_server.log"
    sleep 3
  done
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "phase CONTROLS done: $NFAIL failed drivers"
  f22_standup_stop
fi
RC=$(( TOTAL_FAIL > 0 ? 2 : 0 ))
f22_log "all phases done: $TOTAL_FAIL failed drivers"
exit $RC
