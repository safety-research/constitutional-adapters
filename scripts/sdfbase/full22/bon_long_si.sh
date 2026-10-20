#!/bin/bash
# full22: the BoN legs' LONG-SI windows and the registration facts that ride with them.
# SOURCED by target_static.sbatch (the 512 store rides its static standup) and bon_climb.sbatch AFTER f22_init (cwd = $REPO_DIR; common.sh's
# f22_log / f22_fatal are bound) and BEFORE f22_resolve_config (which registers the job's runs). NEVER --export= (hard rule 1).
#
# WHY. The sampled lane serves a LONG-SI prompted cell (the ~6k-token si_constitution / si_irrelevant system turn) at windows sized by the
# MAX-PROMPT branch of geometry.py's rule (common_sampled.sh F22S_LONG_SI: static 10,752 / controls 6,656 / rungs 8,192 / 14,848 / 23,040),
# detected from the JOB'S CELLS — but target_static.sbatch (the 512 store) and bon_climb.sbatch had no such rule: both stood up at
# families_served:short = 4,864 (the p99 static prompt 762 + the 4,096 cap) for every cell, so the constitution-SI BoN cell's every slot was
# refused prompt_over_window (the demo's store 983259: 7,500 of 7,500; the paper's u0 store 808185 before it), and the closer
# filed a designed no-fit certificate for a cell that was mis-served, not unfit. The rule:
# the BoN window must hold the whole context — the SI + the (augmented) goal + the cap — and the cell is RE-RUN under a window that fits;
# the approved geometry: per-leg long-SI windows, bon_store >= 6,656 and bon_climb 10,752 (aligned with the static legs' long-SI
# window), the cap unchanged at 4,096, the window check run whenever a prompted cell is SERVED (never gated on a
# prompted-only view: a TABLE view mixing none + the prompted cells is exactly where the standup went up at 4,864).
#
# THE NUMBERS (max_prompt + cap -> round-up-256, the sampled lane's rule), MEASURED 2026-10-11 on the grading host with the pinned tokenizer
# Qwen/Qwen3.5-4B @ 851bf6e8 through prompted_window_check.py (the served render: [system = si_constitution] + the longest JBB goal x 2, the
# sigma-0.4 augmentation bound): system turn 5,973 tokens, BoN prompt 6,033 -> store (cap 512) needs 6,545 -> 6,656 (headroom 111);
# climb (cap 4,096) needs 10,129 -> 10,240 by the rule, served at 10,752 = the static legs' long-SI window (one window for the
# 4,096-cap legs under the SI; the static max prompt governs). The llama8b shortconst measurement (roster: BoN 5,947 + 4,096 = 10,043) sits
# inside both. A store whose config draws at 4,096 (bon.max_new_tokens 4096: the Llama lane's native best-of-N) is a 4,096-cap leg and takes
# the climb's window. A FOURTH tokenizer family is not covered by these numbers — the window check below measures the real need on the job
# and steps the window up to it (never a clamp, never a truncated SI); a roster window (views.<view>.serving: per-leg / one-window) WINS.
#
# REGISTRATION (the sampled lane's one-run-per-rung rule, common_sampled.sh f22s_parse_phase): the 512 store registers its runs
# at the DRAWN cap = rung 0 of the canonical best-of-N ladder [512, 4096] (configs/registry/instruments.yaml token_cap_ladder_json; the
# ladder is read from the CONFIG: bon.max_new_tokens and bon.climb_cap), the climb at ITS cap = rung 1 — before, both registered at the
# instrument's read-at 4,096 with an empty ladder (every demo BoN row: token_cap_tokens 4096, rung NULL, ladder []). A config whose two caps
# are one number (the native 4,096 lane) has a one-rung ladder and keeps the registration as it was. The served WINDOW reaches the registry
# the way every native standup records it: <standup dir>/registry_runs.json (sisteer.registry.hooks.RUNS_FILE, written here from the job's
# pending file) lets native_standup.write_endpoint's hook file one run_patch per served run with the ATTESTED max_model_len as
# context_window_tokens (+ the attested S, the engine version, the adapter fingerprint) — the Modal lane's seedrep_served.write_registry_runs,
# ported; nothing on the cluster wrote the file before, so every sbatch lane's rows carry context_window_tokens NULL.
# THE WRITER THESE JOBS REACH: both BoN jobs stand up through f22_standup -> mt_serving_up.py under
# --serving-path lora_native (the demo's store 983259: serving_path=lora_native serving_path_mode=explicit; endpoint engine=lora_native,
# standup_front=lora_serving_up), i.e. _main_native -> lora_serving_up.bring_up_lora -> native_standup.bring_up_native -> write_endpoint,
# whose hook reads <out>/registry_runs.json — 983259's standup.log carries that hook's own notice ("no run identities for the served keys ...
# write registry_runs.json beside endpoint.json"): the consumer was on the path and found nothing beside endpoint.json. mt_serving_up's
# LENS body (--serving-path lens) wrote its record and called no hook; it calls the same one now (mt_serving_up._lens_registry_serving_facts),
# so the file is consumed under either engine. And because the hook is GUARDED (a lost record is one stderr line, never a refused standup),
# f22_registry_serving_recorded reads the spool BACK after f22_standup and refuses the job unless every run's patch with THIS standup's
# attested window is there (registry_serving_recorded.py) — the window reaching the registry is the rule, not a best effort.

#
# THE RERUN PAIR (reproduce/lib/stage.py `rerun-units` / _rerun_rows): a 512 store that COMPLETED and registered its no-fit
# certificate is re-run as NEW funnel units — bon/store512_<slug> (this file's store leg, FAMILIES=bon, the ruling's cells alone) ->
# bon/climb_<slug> (afterok) — never by a `--resubmit` of the done unit (the funnel refuses a COMPLETED registering unit) and never by a
# same-config job under the campaign root, which would RESUME the old store dir (<private root>/<private_run>/<cell>: bon_errors.jsonl holds
# the 7,500 refusals under a `complete: true` marker; _load_covered reads bon_responses.jsonl, so every slot would be redrawn beside them)
# and register the SAME job-independent run id (sdfbase_full22-stat-3-<tag>). The plan row carries F22_BON_RERUN=<slug>;
# f22_bon_rerun_root (right after f22_init, before anything derives from WAVE_ROOT) re-roots the job to `<campaign root>/rerun/<slug>` —
# transcripts_results (the store), bon_climb/<tag> (the climb's plan + out), serving/, the closer's --wave-root (it finds the store at
# <wave root>/transcripts_results/sdfbase_full22_<tag>/<cell>, registry_close_stores.py's bon layout) and the STOP sentinel (<wave root>/STOP:
# the per-group pool roots own theirs the same way, submit_twin.py §3.9 — a campaign-root STOP does not reach a rerun; stop it at its own root
# or by job name) — and exports F22_REGISTRY_KEY_SEGMENTS=<slug>, so the store registers under `sdfbase_full22-stat-3-<tag>-<slug>` (common.sh
# f22_registry_producer_key; the climb's key stays job-keyed as every climb's is). The plan's coverage root_rel spells the same place.

F22_BON_LONG_SI_CELLS=(prompted__si_constitution prompted__si_irrelevant)   # == common_sampled.sh F22S_LONG_SI_CELLS (tests pin the two equal)
F22_BON_LONG_SI=""
F22_BON_RERUN_ROOT=""                   # the rerun's wave root once f22_bon_rerun_root re-rooted this job; "" = not a rerun

f22_bon_rerun_root(){   # $1 = store|climb  $2 = the store job's FAMILIES ("" for the climb)  $3 = the job's cells positional. Right AFTER f22_init (WAVE_ROOT
                        # bound, nothing derived from it yet) and BEFORE f22_resolve_config (which registers under the producer key the segment shapes).
                        # A no-op unless F22_BON_RERUN names the pair's slug (stage.py BON_RERUN_ENV on the two rerun rows); every other job byte for byte as before.
  F22_BON_RERUN_ROOT=""
  [ -n "${F22_BON_RERUN:-}" ] || return 0
  case "$F22_BON_RERUN" in ''|*[!A-Za-z0-9_]*) f22_fatal "BoN rerun: F22_BON_RERUN='$F22_BON_RERUN' is not one token of [A-Za-z0-9_] (stage.py rerun_slug: <cells>_rerun) — a directory name, a producer-key segment and a unit name at once";; esac
  [ -n "${WAVE_ROOT:-}" ] || f22_fatal "f22_bon_rerun_root: WAVE_ROOT is unbound — call it right after f22_init"
  case "$1" in
    store) [ "$2" = bon ] || f22_fatal "BoN rerun $F22_BON_RERUN: FAMILIES='$2' — the rerun root holds the BoN store alone (the recipe's store512_$F22_BON_RERUN unit, FAMILIES=bon); a static family would write its stores under the rerun root, unread by its own unit";;
    climb) ;;
    *) f22_fatal "f22_bon_rerun_root: unknown leg $1 (store|climb)";;
  esac
  [ -n "$3" ] || f22_fatal "BoN rerun $F22_BON_RERUN: no cells positional — a rerun serves exactly the cells its ruling names (D913 (ii)), never the view's full enumeration"
  F22_BON_RERUN_ROOT="$WAVE_ROOT/rerun/$F22_BON_RERUN"
  WAVE_ROOT="$F22_BON_RERUN_ROOT"; export F22_WAVE_ROOT="$WAVE_ROOT"; mkdir -p "$WAVE_ROOT"
  export F22_REGISTRY_KEY_SEGMENTS="${F22_REGISTRY_KEY_SEGMENTS:+$F22_REGISTRY_KEY_SEGMENTS-}$F22_BON_RERUN"
  f22_log "BoN RERUN $F22_BON_RERUN ($1 leg; cells $3; D913 (ii)): wave root $WAVE_ROOT — its transcripts_results / bon_climb / serving dirs, the closer's --wave-root and the STOP sentinel ($WAVE_ROOT/STOP) are this rerun's own, never the campaign's; producer-key segment '$F22_BON_RERUN' (the store registers as sdfbase_full22-stat-3-<tag>-$F22_BON_RERUN: new run ids beside the superseded cell's)"
}
F22_BON_STATIC_LONG_SI_WINDOW=10752    # the static legs' long-SI window (common_sampled.sh f22s_base_window static under F22S_LONG_SI)
F22_BON_CLIMB_LONG_SI_WINDOW=10752     # the 4,096-cap BoN legs (the climb; a 4,096 store): aligned with the static legs' window
F22_BON_STORE_LONG_SI_WINDOW=6656      # the 512 store: 6,033 + 512 = 6,545 -> 6,656 (measured above)
F22_BON_WINDOW_STEP=128                # the window check's step grid (the 128-token block the per-leg roster windows sit on)
F22_BON_CFG=""; F22_BON_STORE_CAP=""; F22_BON_CLIMB_CAP=""; F22_BON_LADDER=""
F22_BON_WINDOW=""

f22_bon_wave_config(){   # $1 = model key, $2 = view -> the wave config f22_resolve_config binds for (KEY, VIEW) with no arm (its spelling, as f22_full22_manifest's)
  echo "configs/sdfbase_full22_$1$( [ "$2" = main ] || echo "_$2" ).yaml"
}
f22_bon_read_caps(){   # $1 = model key, $2 = view -> F22_BON_CFG, F22_BON_STORE_CAP (bon.max_new_tokens), F22_BON_CLIMB_CAP (bon.climb_cap; the roster's
                       # bon_climb_cap 4,096 when the config names none), F22_BON_LADDER ("<store>,<climb>" when store < climb; "" = a one-rung lane)
  F22_BON_CFG=$(f22_bon_wave_config "$1" "$2")
  [ -f "$F22_BON_CFG" ] || f22_fatal "bon: no wave config at $F22_BON_CFG for $1/$2 (f22_resolve_config would bind this path; generate it: make_full22_configs.py)"
  local caps
  caps=$(python3 - "$F22_BON_CFG" 2>&1 <<'PY'
import sys, yaml
b = (yaml.safe_load(open(sys.argv[1])) or {}).get("bon") or {}
store = b.get("max_new_tokens"); climb = b.get("climb_cap", 4096)
for name, v in (("bon.max_new_tokens", store), ("bon.climb_cap", climb)):
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        sys.exit(f"{sys.argv[1]}: {name} = {v!r} is not a positive integer")
print(store, climb)
PY
) || f22_fatal "bon: cannot read the BoN caps from $F22_BON_CFG — $caps"
  F22_BON_STORE_CAP="${caps% *}"; F22_BON_CLIMB_CAP="${caps#* }"
  if [ "$F22_BON_STORE_CAP" -lt "$F22_BON_CLIMB_CAP" ]; then F22_BON_LADDER="$F22_BON_STORE_CAP,$F22_BON_CLIMB_CAP"
  elif [ "$F22_BON_STORE_CAP" -eq "$F22_BON_CLIMB_CAP" ]; then F22_BON_LADDER=""
  else f22_fatal "bon: $F22_BON_CFG draws the store at bon.max_new_tokens $F22_BON_STORE_CAP ABOVE its climb cap bon.climb_cap $F22_BON_CLIMB_CAP — not a ladder"; fi
}
f22_bon_registry_exports(){   # $1 = store|climb [$2 = the climb's cap positional] (after f22_bon_read_caps, before f22_resolve_config) -> F22_REGISTRY_TOKEN_CAP /
                              # RUNG_INDEX / LADDER / PASS_LABEL for the job-start registration; a caller's values win (the sampled lane's rule); a one-rung
                              # lane exports nothing (jobplan needs a ladder of two to index) and says so
  [ -n "$F22_BON_CFG" ] || f22_fatal "f22_bon_registry_exports: call f22_bon_read_caps first"
  if [ -z "$F22_BON_LADDER" ]; then
    f22_log "registry: $F22_BON_CFG draws the store and the climb at one cap ($F22_BON_STORE_CAP) — a one-rung lane: the $1 registers at the instrument's cap as before (no rung)"
    return 0
  fi
  case "$1" in
    store) export F22_REGISTRY_TOKEN_CAP="${F22_REGISTRY_TOKEN_CAP:-$F22_BON_STORE_CAP}" F22_REGISTRY_RUNG_INDEX="${F22_REGISTRY_RUNG_INDEX:-0}" \
                  F22_REGISTRY_PASS_LABEL="${F22_REGISTRY_PASS_LABEL:-base_cap}";;
    climb) local cap="${2:-$F22_BON_CLIMB_CAP}"
           [ "$cap" = "$F22_BON_CLIMB_CAP" ] || f22_fatal "bon climb: the cap positional $cap is not $F22_BON_CFG's bon.climb_cap $F22_BON_CLIMB_CAP (rung 1 of the ladder [$F22_BON_LADDER]) — the registration cannot name its rung; pass the config's climb cap, or change the config"
           export F22_REGISTRY_TOKEN_CAP="${F22_REGISTRY_TOKEN_CAP:-$cap}" F22_REGISTRY_RUNG_INDEX="${F22_REGISTRY_RUNG_INDEX:-1}" \
                  F22_REGISTRY_PASS_LABEL="${F22_REGISTRY_PASS_LABEL:-ladder_rung}";;
    *) f22_fatal "f22_bon_registry_exports: unknown leg $1 (store|climb)";;
  esac
  export F22_REGISTRY_LADDER="${F22_REGISTRY_LADDER:-$F22_BON_LADDER}"
  f22_log "registry: the BoN $1 registers at cap $F22_REGISTRY_TOKEN_CAP = rung $F22_REGISTRY_RUNG_INDEX of the ladder [$F22_REGISTRY_LADDER] ($F22_REGISTRY_PASS_LABEL; D913 (iii))"
}
f22_bon_assert_config(){   # after f22_resolve_config: the config it bound IS the one the caps were read from (needs CFG)
  [ "${CFG:-}" = "$F22_BON_CFG" ] || f22_fatal "bon: f22_resolve_config bound $CFG but the BoN caps were read from $F22_BON_CFG — the registration named another config's ladder"
}

f22_bon_detect_long_si(){   # $@ = the job's cells -> F22_BON_LONG_SI=1 when any is a long-SI prompted cell (logged), "" otherwise
  local c l
  F22_BON_LONG_SI=""
  for c in "$@"; do for l in "${F22_BON_LONG_SI_CELLS[@]}"; do [ "$c" = "$l" ] && F22_BON_LONG_SI=1; done; done
  if [ -n "$F22_BON_LONG_SI" ]; then
    f22_log "LONG-SI cell among the job's cells ($*): BoN windows store $(f22_bon_window store) (cap ${F22_BON_STORE_CAP:-512}) / climb $(f22_bon_window climb) (cap ${F22_BON_CLIMB_CAP:-4096}) / static $(f22_bon_window static) (max-prompt sizing; see bon_long_si.sh)"
  fi
}
f22_bon_long_si_id(){   # $@ = the job's cells -> the SI id of the first long-SI cell among them (prompted__<si> -> <si>); "" when none
  local c l
  for c in "$@"; do for l in "${F22_BON_LONG_SI_CELLS[@]}"; do [ "$c" = "$l" ] && { echo "${c#prompted__}"; return 0; }; done; done
  echo ""
}
f22_bon_cap_window(){   # $1 = a BoN cap -> the long-SI window for a BoN leg drawing at that cap (the measured table above); an unmeasured cap is fatal
  if [ "$1" -le 512 ]; then echo "$F22_BON_STORE_LONG_SI_WINDOW"
  elif [ "$1" -le 4096 ]; then echo "$F22_BON_CLIMB_LONG_SI_WINDOW"
  else f22_fatal "bon: no long-SI window is measured for a BoN cap of $1 (the table covers 512 and 4,096; measure first — bon_long_si.sh)"; fi
}
f22_bon_window(){   # $1 = store|climb|static -> the leg's long-SI standup window under F22_BON_LONG_SI, "" (the table's) otherwise.
                    # The caller lets a ROSTER window win first (VIEW_WINDOW_STATIC: a per-leg / one-window prompted view sizes its own).
  [ -n "${F22_BON_LONG_SI:-}" ] || { echo ""; return 0; }
  case "$1" in
    store)  f22_bon_cap_window "${F22_BON_STORE_CAP:-512}";;
    climb)  f22_bon_cap_window "${F22_BON_CLIMB_CAP:-4096}";;
    static) echo "$F22_BON_STATIC_LONG_SI_WINDOW";;
    *) f22_fatal "f22_bon_window: unknown leg $1 (store|climb|static)";;
  esac
}
f22_bon_window_check(){   # $1 = the standup out dir, $2 = the window L, $3 = the legs this standup serves (comma list of static,bon), $4 = the cap
                          # (a number: every named leg at it — the climb; `base`: each leg at its own config cap — the store's standup), $5.. = the
                          # job's cells -> F22_BON_WINDOW = L, or L stepped up ONCE to the measured worst need (the F22_BON_WINDOW_STEP grid) when
                          # the check refuses; a second refusal, or a check that cannot run, is fatal. needs KEY VIEW CFG MPY REPO_DIR.
                          # A no-op (F22_BON_WINDOW = L) without a long-SI cell among the cells. Never gated on a prompted-only view.
  local out="$1" L="$2" legs="$3" cap="$4"; shift 4
  local si; si=$(f22_bon_long_si_id "$@")
  F22_BON_WINDOW="$L"
  { [ -n "${F22_BON_LONG_SI:-}" ] && [ -n "$si" ]; } || return 0
  case "$L" in ''|*[!0-9]*) f22_fatal "bon window check: window '$L' is not an integer";; esac
  mkdir -p "$out"
  local attempt rc need stepped
  for attempt in 1 2; do
    if (cd "$REPO_DIR" && "$MPY" scripts/sdfbase/full22/prompted_window_check.py --config "$CFG" --si "$si" --max-model-len "$F22_BON_WINDOW" \
          --legs "$legs" --cap "$cap" --out "$out/long_si_window_check.json" > "$out/long_si_window_check.log" 2>&1); then
      f22_log "long-SI window check PASS for $KEY/$VIEW at L=$F22_BON_WINDOW (cell prompted__$si; legs $legs at cap $cap): $(tail -1 "$out/long_si_window_check.log")"
      return 0
    else
      rc=$?
    fi
    [ "$rc" = 3 ] || f22_fatal "long-SI window check could not run for $KEY/$VIEW (rc $rc): $(tail -3 "$out/long_si_window_check.log" | tr '\n' ' ')"
    [ "$attempt" = 1 ] || f22_fatal "long-SI window check REFUSED for $KEY/$VIEW at L=$F22_BON_WINDOW after one step (legs $legs at cap $cap): $(grep REFUSED "$out/long_si_window_check.log" | tr '\n' ' ') — widen the long-SI table (bon_long_si.sh), never clamp"
    need=$("$MPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print(max(int(r["need"]) for r in d["rows"]))' "$out/long_si_window_check.json" 2>/dev/null) \
      || f22_fatal "long-SI window check: cannot read the worst need from $out/long_si_window_check.json"
    stepped=$(( (need + F22_BON_WINDOW_STEP - 1) / F22_BON_WINDOW_STEP * F22_BON_WINDOW_STEP ))
    [ "$stepped" -gt "$F22_BON_WINDOW" ] || f22_fatal "long-SI window check REFUSED at L=$F22_BON_WINDOW yet its worst need $need fits it — the check's record is inconsistent: $(grep REFUSED "$out/long_si_window_check.log" | tr '\n' ' ')"
    f22_log "long-SI window check REFUSED for $KEY/$VIEW at L=$F22_BON_WINDOW (legs $legs at cap $cap; worst need $need): stepping the window to $stepped (the next $F22_BON_WINDOW_STEP-token block; S follows from the budget) — never a clamp"
    F22_BON_WINDOW="$stepped"
  done
}

f22_registry_runs_file(){   # $1 = the standup out dir, $2 = the evals this standup serves (comma list) -> $1/registry_runs.json (hooks.RUNS_FILE) from the
                            # job's pending file ($SISTEER_REGISTRY_RUN_IDENTITY, exported by f22_registry_register), keyed by each run's SERVED key (the
                            # endpoint's harness spelling), so native_standup.write_endpoint's hook records the attested window per run. needs MPY REPO_DIR.
                            # A job with no run identities (a manual run; a kind that registers nothing) records no serving facts, and says so.
  local out="$1" evals="$2" pend="${SISTEER_REGISTRY_RUN_IDENTITY:-}"
  if [ -z "$pend" ]; then f22_log "registry: no run identities for this job (SISTEER_REGISTRY_RUN_IDENTITY unset: a manual run, or a kind that registers nothing) — the standup records no serving facts"; return 0; fi
  mkdir -p "$out"
  (cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$MPY" scripts/sdfbase/full22/registry_runs_file.py --pending "$pend" --out "$out" --evals "$evals") \
    || f22_fatal "registry: the served runs file $out/registry_runs.json could not be written from $pend (evals $evals; the [registry] lines above) — the standup would record no context_window_tokens for this job's runs (D913 (iii))"
}

f22_registry_serving_recorded(){   # $1 = the standup out dir, right AFTER f22_standup returned 0 -> fatal unless every run of $1/registry_runs.json has
                                   # its run_patch with THIS standup's attested window in the registry spool (registry_serving_recorded.py: the hook's
                                   # record read back from the destination the harvester pulls). A standup with no runs file (no identities:
                                   # f22_registry_runs_file said so) has nothing to observe and passes. needs MPY REPO_DIR.
                                   # F22_REGISTRY_SERVING_RECORDED_TIMEOUT_S (default 120) bounds the wait for the hook's write.
  local out="$1"
  [ -f "$out/registry_runs.json" ] || { f22_log "registry: no served runs file under $out — no serving facts to observe for this standup"; return 0; }
  (cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}" "$MPY" scripts/sdfbase/full22/registry_serving_recorded.py --out "$out" --timeout-s "${F22_REGISTRY_SERVING_RECORDED_TIMEOUT_S:-120}") \
    || f22_fatal "registry: the served window was NOT observed in the registry spool for the runs of $out/registry_runs.json (the [registry] lines above; the standup's own account is in $out/standup.log) — a job whose window never reaches context_window_tokens is the D904 class the ruling retires (D913 (iii)); refused rather than run"
}
