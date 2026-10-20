#!/bin/bash
# full22 SAMPLED re-run — the helpers target_sampled.sbatch adds ON TOP of the wave's common.sh, which is
# SOURCED here (f22_init / f22_resolve_config / f22_bank_view / f22_driver_venv / f22_serving_venv / f22_standup /
# f22_queue_launch / f22_queue_wait / f22_positive_control / f22_result are reused, never copied). Everything
# below is the sampled run's own vocabulary: the phase grammar, the sampled-config resolver, the at-cap counters'
# shell seam, the ladder events. NEVER --export= (hard rule 1): every value travels positionally or here.
F22S_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$F22S_HERE/../common.sh"

F22S_DIR="scripts/sdfbase/full22/sampled"          # relative to $REPO_DIR (f22_init cd's there)
F22S_FAMILIES=(pasttense codeattack renellm grok420)
F22S_CONTROLS=(xstest or_bench_hard_1k)
F22S_STATIC_CAP=4096; F22S_STATIC_CAP_ESC=8192     # == the configs' attacks.static_max_new_tokens (f22s_check_config asserts)
F22S_STATIC_CAP_ESC16=16384                        # the static ladder's ceiling rung (LADDER_RULES.md; phase static_esc16_t<T>)
F22S_CONTROLS_RUNGS=(512 2048 8192 16384)          # LADDER_RULES.md: controls 512 -> 2048 -> 8192 -> 16384 (ceiling); static 4096 -> 8192
                                                   # (the loop iterates THIS list; a rung is climbed only on the gate's ESCALATE)
F22S_FLOOR_SAMPLE_K=200                            # every RUNG invocation of served_capability_run.py --floor-sample K
F22S_ESC_CONC=32                                   # escalate_static.py --concurrency (the brief)
# the pre-decision (count-only) era's controls ladder ended at 8,192 (the 3-rung ladder of 2026-09-10; no gate decisions in its
# events): write_ladder.py --count-only-ceiling reads such a pair as left THERE with its residual instead of "incomplete" under
# today's 4-rung list (the qwen35_4b_it pass-2 refusal of 2026-09-14). Declared here, once; the static ladder never had that era.
F22S_COUNT_ONLY_CEILING=8192
f22s_count_only_flag(){   # $1 = the ladder kind being folded -> the flag for the controls fold, nothing for any other
  [ "$1" = controls ] && echo "--count-only-ceiling $F22S_COUNT_ONLY_CEILING"
  return 0
}
# --- the 16,384 EXTEND (lane G2, 2026-09-10; DESIGN.md "Ceiling 16,384", LADDER_RULES.md): the ceiling rung of BOTH ladders,
# entered per (cell, readout) only on an item_filtered_v1 `escalate` decision at 8192 (ladder_extend.py over ladder_decide.py
# reclassify); the folds of an extended ladder carry the ceiling
F22S_CEILING=16384; F22S_STATIC_CAP_EXT=16384
F22S_CONTROLS_RUNGS_EXT="512,2048,8192,16384"; F22S_STATIC_RUNGS_EXT="4096,8192,16384"; F22S_GSM8K_RUNGS="8192,16384"
F22S_RULE="item_filtered_v1"
# --- the capability battery: GSM8K THINKING-ON
# at 8,192 under the readout preset, vendor-sampled per trial — NO matched greedy arm (the report's comparator is the published
# greedy thinking-OFF 512 baseline, labelled, never pooled) — over the SAME 200 items the greedy wave's capability controls
# used — parity.gsm8k_parity_rows(pins, n=PARITY_N): the pinned gsm8k_platinum revision, shuffle(seed=0),
# select — whose sorted-ids sha256 (the spool header's items_sha256) is asserted per cell against the greedy stores' value
# (read on the grading host from $GRADE/sdfbase_full22/controls/{gemma4e4b,qwen35_4b,gemma4e4b_it}/cells/*/rows.jsonl:
# identical across every cell and TAG); MMLU unchanged (letter-logprob argmax, thinking OFF, --max-logprobs 64)
F22S_GSM8K_CAP=8192; F22S_GSM8K_N=200
F22S_GSM8K_ITEMS_SHA256="249bd4bf24728c39c11a7f79093f2cfb82d1940830af22555225caeefad84445"
F22S_HOP_GUARD_MARGIN=2.0                          # hop_timeout_guard.py --margin: the queue wait (drivers over-subscribe ~96:76)
F22S_REP_RATIO_SOURCE_SHA256="152d180376dd3ce9d6e054354b8bd4a6ddcf6ffd9d4635ab708398d32c5f33ae"   # LADDER_RULES.md §2

# --- the phase grammar: static_base_t<T>[:fam,...] | static_esc_t<T>[:fam,...] | static_esc16_t<T>[:fam,...] | controls_t<T> |
#     ladder_extend_t<T> | capability_t<T> | gsm8k_t<T>  (T in 0,1) -----------------------------------------------------------
# static_esc16_t<T>: the static ladder's 16,384 rung on the GATE's ESCALATE from the
#   8,192 store only (ladder_gate.py static: the item-filtered 1 % TRUNCATED verdict bound to the transport approval).
# ladder_extend_t<T> (lane G2): the 16,384 rung of BOTH ladders (+ the GSM8K readout's when its 8192 root exists) for this
#   trial — reclassify every rung generated so far under item_filtered_v1, decide per pair at 8192, climb ONLY the promoted
#   pairs (zero promotions = a recorded decision, no standup), then the CEILING report at 16,384 (residual share + count; the
#   ladder stops there).
# capability_t<T>: GSM8K thinking-ON @8192 sampled arm (this trial) + MMLU (thinking OFF, unchanged), each into its own root.
# There is NO greedy GSM8K comparator phase (sampled only at the 8,192 cap; the comparator in the
#   report is the published greedy thinking-OFF 512 baseline, labelled, never pooled) — `capability_greedy` is REFUSED.
# gsm8k_t<T>: the GSM8K readout under sampled_decode.readouts.gsm8k — the
#   vendor-preset arm (--gsm8k-arm sampled) and, in trial 0 only, the greedy comparator arm on the same items
#   (--gsm8k-arm greedy); each arm its own store (LADDER_RULES.md §6). A view whose roster OMITS gsm8k (llama8b: the
#   evaluation suite's instrument is the paper's one GSM8K) refuses the phase. MMLU is sampler-moot and runs in the greedy
#   wave's controls leg, labelled so.
f22s_parse_phase(){   # $1 = phase -> PHASE_KIND TRIAL FAMS[] PHASE_TAG
  local phase="$1" spec fams f
  spec="${phase%%:*}"; fams=""; [ "$spec" = "$phase" ] || fams="${phase#*:}"
  F22S_CONTROLS_ESC=""; F22S_CONTROLS_CEILING_ABOVE=""   # the rung above a `:<ceiling>` below the campaign ceiling ("" = none)
  case "$spec" in
    static_base_t[01]) PHASE_KIND=static_base;;
    static_esc_t[01])  PHASE_KIND=static_esc;;
    static_esc16_t[01]) PHASE_KIND=static_esc16;;   # the 16,384 rung: the gate's ESCALATE from the 8,192 store only
    controls_t[01])    PHASE_KIND=controls
                       [ -z "$fams" ] || f22_fatal "phase $phase: a controls phase takes no family subset";;
    controls_esc_t[01]) PHASE_KIND=controls; F22S_CONTROLS_ESC=1   # the controls LADDER over an existing COMPLETE 512 root (no 512 pass; 2026-09-13):
                       # the SAME dispatch arm as controls_t<T> (one chain, one marker), the mode a flag the arm reads;
                       # an optional `:<ceiling>` (2048 | 8192 | 16384) is the LAST rung this job climbs — the rungs above it stay a
                       # priced decision for ladder_extend_t<T> (16k is stopped and priced, never run in-job)
                       if [ -n "$fams" ]; then
                         case "$fams" in 2048|8192|16384) ;; *) f22_fatal "phase $phase: controls_esc takes at most one ceiling rung (2048 | 8192 | 16384), not $fams";; esac
                         local kept=() r_ hit_=""; for r_ in "${F22S_CONTROLS_RUNGS[@]}"; do
                           if [ -n "$hit_" ]; then F22S_CONTROLS_CEILING_ABOVE="$r_"; break; fi   # the real next rung: DUE, not a ceiling
                           kept+=("$r_"); [ "$r_" = "$fams" ] && hit_=1
                         done
                         F22S_CONTROLS_RUNGS=("${kept[@]}")
                       fi;;
    ladder_extend_t[01]) PHASE_KIND=ladder_extend
                       [ -z "$fams" ] || f22_fatal "phase $phase: the extend phase takes no family subset (it decides every pair)";;
    capability_t[01])  PHASE_KIND=capability
                       [ -z "$fams" ] || f22_fatal "phase $phase: a capability phase takes no family subset";;
    capability_greedy) f22_fatal "phase $phase: there is no greedy GSM8K comparator arm (the user, 2026-09-10, S-D7: GSM8K runs SAMPLED only at the 8,192 cap; the report's comparator is the published greedy thinking-OFF 512 baseline)";;
    gsm8k_t[01])       PHASE_KIND=gsm8k
                       [ -z "$fams" ] || f22_fatal "phase $phase: a gsm8k phase takes no family subset";;
    *) f22_fatal "unknown phase $phase (static_base_t<T>[:fam,...] | static_esc_t<T>[:fam,...] | static_esc16_t<T>[:fam,...] | controls_t<T> | controls_esc_t<T> | ladder_extend_t<T> | capability_t<T> | gsm8k_t<T>; T in 0,1)";;
  esac
  TRIAL="${spec##*_t}"
  FAMS=()
  if [ "$PHASE_KIND" = static_base ] || [ "$PHASE_KIND" = static_esc ] || [ "$PHASE_KIND" = static_esc16 ]; then
    if [ -z "$fams" ]; then FAMS=("${F22S_FAMILIES[@]}"); else IFS=',' read -ra FAMS <<< "$fams"; fi
    for f in "${FAMS[@]}"; do case "$f" in
      pasttense|codeattack|renellm|grok420) ;;
      *) f22_fatal "phase $phase: unknown static family $f (pasttense,codeattack,renellm,grok420)";; esac; done
  fi
  PHASE_TAG="$spec"; [ -z "$fams" ] || PHASE_TAG="${spec}_$(echo "$fams" | tr ',' '-')"
  # static-rung-fix: the job registers EXACTLY its phase's evals. f22_registry_register reads F22_REGISTRY_EVALS, else the
  # job-NAME table in common.sh, whose `f22s-*` row is the lane's UNION (the four statics AND xstest / or_bench_hard) — a kind, not a phase: the
  # static 8192 rung 838299 registered 12 runs, 4 of them controls it never produces (the clean-exit closer would then refuse them, exit 75).
  # A caller's value wins (the front door's stage.py sets the unit's evals); the extend / capability / gsm8k phases keep the table for now.
  case "$PHASE_KIND" in
    static_base|static_esc|static_esc16) export F22_REGISTRY_EVALS="${F22_REGISTRY_EVALS:-${FAMS[*]}}";;
    controls)                            export F22_REGISTRY_EVALS="${F22_REGISTRY_EVALS:-xstest or_bench_hard}";;
  esac
  # A static phase registers its runs AT THE RUNG IT DRAWS — ONE RUN PER RUNG, the
  # registry-wide convention: the producer key is the rung's private run name (`sdfbase_full22_<tag>_sampled_t<T>` for the base draw,
  # `…_cap8192` / `…_cap16384` for the rungs; common.sh prefixes the tag), token_cap_tokens the DRAWN cap (never the instrument's read-at
  # 8192, which stamped the 4,096 base draws of 808182/808184/833933/833934 as 8192 with an empty ladder), the rung index and the ladder.
  # A caller's values win. The controls ladder is a different instrument and keeps its own registration.
  case "$PHASE_KIND" in
    static_base)  export F22_REGISTRY_TOKEN_CAP="${F22_REGISTRY_TOKEN_CAP:-$F22S_STATIC_CAP}" F22_REGISTRY_RUNG_INDEX="${F22_REGISTRY_RUNG_INDEX:-0}" \
                         F22_REGISTRY_PASS_LABEL="${F22_REGISTRY_PASS_LABEL:-base_cap}" F22_REGISTRY_RUN_SUFFIX="${F22_REGISTRY_RUN_SUFFIX:-_sampled_t${TRIAL}}";;
    static_esc)   export F22_REGISTRY_TOKEN_CAP="${F22_REGISTRY_TOKEN_CAP:-$F22S_STATIC_CAP_ESC}" F22_REGISTRY_RUNG_INDEX="${F22_REGISTRY_RUNG_INDEX:-1}" \
                         F22_REGISTRY_PASS_LABEL="${F22_REGISTRY_PASS_LABEL:-ladder_rung}" F22_REGISTRY_RUN_SUFFIX="${F22_REGISTRY_RUN_SUFFIX:-_sampled_t${TRIAL}_cap${F22S_STATIC_CAP_ESC}}";;
    static_esc16) export F22_REGISTRY_TOKEN_CAP="${F22_REGISTRY_TOKEN_CAP:-$F22S_STATIC_CAP_ESC16}" F22_REGISTRY_RUNG_INDEX="${F22_REGISTRY_RUNG_INDEX:-2}" \
                         F22_REGISTRY_PASS_LABEL="${F22_REGISTRY_PASS_LABEL:-ladder_rung}" F22_REGISTRY_RUN_SUFFIX="${F22_REGISTRY_RUN_SUFFIX:-_sampled_t${TRIAL}_cap${F22S_STATIC_CAP_ESC16}}";;
  esac
  case "$PHASE_KIND" in
    static_base|static_esc|static_esc16) export F22_REGISTRY_LADDER="${F22_REGISTRY_LADDER:-$F22S_STATIC_RUNGS_EXT}" F22_REGISTRY_TRIAL_INDEX="${F22_REGISTRY_TRIAL_INDEX:-$TRIAL}";;
  esac
  # static-reg-fix-2: the 8,192 / 16,384 rungs draw ONLY the pairs their per-pair gate selects (an at-cap row in the
  # base run; the ladder gate's ESCALATE), and the gate runs after f22_resolve_config — so these phases DEFER the job-start registration
  # (common.sh f22_registry_register returns without registering) and register exactly the selected pairs themselves, before the standup
  # (target_sampled.sbatch -> f22s_register_gated_pairs). A phase whose gate selects nothing registers nothing. The base draw registers every
  # pair at job start, as before. A caller's value wins.
  case "$PHASE_KIND" in
    static_esc|static_esc16) export F22_REGISTRY_DEFER="${F22_REGISTRY_DEFER:-$PHASE_KIND}";;
  esac
}

# --- the sampled SCOPE: ONE list, the generator's SAMPLED_VIEWS (make_sampled_configs.py) — ASKED here, never restated ----
# The scope used to be a second `case main|it` list in this file; two lists drift (the strength frontier's 24 per-dose views,
# 2026-09-10, would have had to land in both). Now the shell asks the generator: exit 0 = (key, view) in scope (the TAG on
# stdout), 3 = NOT in scope (the scope on stderr), anything else = the QUERY ITSELF failed (an interpreter or import problem),
# which is reported as such and never read as "out of scope" — a broken gate must not refuse a valid view with a misleading
# message. The query reads no file and runs before the venvs exist (the sbatch resolves the config before f22_driver_venv), so
# it takes the driver venv python when MPY is already set and the system python3 otherwise (the generator's module imports
# only the stdlib + yaml, both present on the cluster images; tests/test_sdfbase_full22_sampled.py drives this gate against
# SAMPLED_VIEWS so the two can never disagree).
f22s_view_in_scope(){   # $1 = key, $2 = view -> 0 in scope | 3 not in scope | other: the query failed (stderr says why)
  "${MPY:-python3}" "$F22S_HERE/make_sampled_configs.py" --is-in-scope "$1" "$2" >/dev/null
}

# --- the sampled config: the WAVE's checks first, then CFG swapped to the sampled config (MANIFEST stays the wave's) ---
# The VIEW: `main` is DESIGN.md's scope (TAG = KEY); `it` is the instruct-direct view added 2026-09-09 under the identical
# contract (TAG = KEY_it); the strength frontier's per-dose views `d<dose>` / `it_d<dose>` (2026-09-10; TAG = KEY_<view>,
# e.g. nemotron3nano4b_d0p25 / gemma4e4b_it_d0p25) joined under the same rule. For every non-main view
# configs/sdfbase_full22_<TAG>*.yaml, the `<TAG>` Petri manifest, and every store / ladder / serving path keyed on TAG
# follow. KEY stays the bare model key: budgets, hybrid facts, the rung windows (f22s_rung_window) and f22_standup's
# --model-key are per MODEL, not per view; the bank view is (KEY, VIEW)'s. Which views are in scope is the generator's
# SAMPLED_VIEWS, asked through f22s_view_in_scope above — this file holds no list of its own.
f22s_resolve_config(){   # $1 = key, $2 = trial, $3 = view (default main) -> KEY VIEW TAG WAVE_CFG CFG (sampled base) CFG_ESC (cap8192) CFG_ESC16 (cap16384) MANIFEST PRIVATE_RUN_BASE PRIVATE_RUN_ESC PRIVATE_RUN_ESC16 SIBLING_TRIAL
  local key="$1" trial="$2" view="${3:-main}" scope_rc
  # f22_resolve_config derives CFG/MANIFEST from (KEY, VIEW) and runs the wave's job-side belts (the text-only
  # --limit-mm-per-prompt pin for the multimodal keys, glm46vflash's GPU_MEM_UTIL); we keep every one of them and
  # only re-point CFG at the sampled config of THIS trial. The Petri manifest is the wave's for this (KEY, VIEW) (DESIGN.md).
  f22s_view_in_scope "$key" "$view"; scope_rc=$?
  case "$scope_rc" in
    0) ;;
    3) f22_fatal "view $view of $key is not in the sampled scope (make_sampled_configs.py SAMPLED_VIEWS is the ONE list; \`make_sampled_configs.py --tags\` prints it)";;
    *) f22_fatal "the sampled scope query failed (rc=$scope_rc: ${MPY:-python3} $F22S_HERE/make_sampled_configs.py --is-in-scope $key $view) — refusing to guess whether $key/$view is in scope";;
  esac
  f22_resolve_config "$key" "$view"
  # THE VIEW'S LEGS AND WINDOW (2026-09-10; f22_roster.view_legs / view_serving through roster_view.py — the same readers
  # the generator and the printers use): a view that names legs runs exactly those (the llama8b fullconst view: static +
  # bon + capability, NO over-refusal controls -> no ladder job), and a view with a serving window serves EVERY standup
  # at that window (fullconst: 45,056; S = the model budget // L)
  # --roster "$F22_ROSTER" on EVERY call: roster_view.py defaults to the PRODUCTION roster and does NOT read F22_ROSTER
  # itself (target_static.sbatch passes it explicitly; this one never did), so a lane-private roster was invisible to
  # the entire sampled leg -- eight XSTest jobs died with "cannot read the roster's legs" (2026-09-18). Byte-identical
  # when F22_ROSTER is the production default, which every existing lane passes.
  VIEW_LEGS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" legs --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's legs for $KEY/$VIEW"
  VIEW_WINDOW=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" max_model_len --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's serving window for $KEY/$VIEW"
  # PER-LEG windows (the llama8b shortconst view, 2026-09-12: f22_roster.view_windows — the static standup's, the controls standup's,
  # and each ladder rung's; "" = the table's / the model's rung table). A ONE-WINDOW view answers every field with its window, so
  # f22s_base_window / f22s_rung_window below serve fullconst exactly as before
  VIEW_WINDOW_STATIC=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window static --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's static window for $KEY/$VIEW"
  VIEW_WINDOW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window controls --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's controls window for $KEY/$VIEW"
  VIEW_RUNG_WINDOW_2048=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" rung_window 2048 --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's 2048 rung window for $KEY/$VIEW"
  VIEW_RUNG_WINDOW_8192=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" rung_window 8192 --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's 8192 rung window for $KEY/$VIEW"
  VIEW_RUNG_WINDOW_16384=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" rung_window 16384 --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's 16384 rung window for $KEY/$VIEW"
  VIEW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" controls --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's controls list for $KEY/$VIEW"
  # the over-refusal controls of THIS view (the judged, laddered ones); empty = the view has no controls leg
  F22S_CONTROLS=(); local c_; for c_ in ${VIEW_CONTROLS//,/ }; do case "$c_" in xstest|or_bench_hard_1k) F22S_CONTROLS+=("$c_");; esac; done
  WAVE_CFG="$CFG"
  CFG="configs/sdfbase_full22_${TAG}_sampled_t${trial}.yaml"
  CFG_ESC="configs/sdfbase_full22_${TAG}_sampled_t${trial}_cap8192.yaml"
  CFG_ESC16="configs/sdfbase_full22_${TAG}_sampled_t${trial}_cap16384.yaml"
  [ -f "$CFG" ] && [ -f "$CFG_ESC" ] && [ -f "$CFG_ESC16" ] || f22_fatal "missing $CFG, $CFG_ESC or $CFG_ESC16 (regenerate: $F22S_DIR/make_sampled_configs.py)"
  PRIVATE_RUN_BASE="sdfbase_full22_${TAG}_sampled_t${trial}"; PRIVATE_RUN_ESC="${PRIVATE_RUN_BASE}_cap8192"; PRIVATE_RUN_ESC16="${PRIVATE_RUN_BASE}_cap16384"
  SIBLING_TRIAL=$((1 - trial))   # the other seeded draw: the ladder's boundary rule reads its rows (LADDER_RULES.md §3)
  f22_log "sampled config: $CFG (esc $CFG_ESC, ext $CFG_ESC16), manifest $MANIFEST (the wave's), view $VIEW, tag $TAG, private_run $PRIVATE_RUN_BASE (sibling trial $SIBLING_TRIAL)"
}

f22s_check_config(){   # $1 = python (MPY) — the job-side belt: the sampled configs ARE the wave config as data except the permitted keys
  "$1" - "$WAVE_CFG" "$CFG" "$CFG_ESC" "$TRIAL" "$PRIVATE_RUN_BASE" "$F22S_STATIC_CAP" "$F22S_STATIC_CAP_ESC" "$CFG_ESC16" "$F22S_STATIC_CAP_EXT" <<'PY' || f22_fatal "sampled config check refused (regenerate: make_sampled_configs.py; --check)"
import sys, yaml
wave, base, esc, trial, prun, cap, cap_esc, esc16, cap_ext = sys.argv[1:10]
w, b, e, e16 = (yaml.safe_load(open(p)) for p in (wave, base, esc, esc16))
for name, c, extra, want_cap in ((base, b, set(), int(cap)), (esc, e, {"attacks"}, int(cap_esc)), (esc16, e16, {"attacks"}, int(cap_ext))):
    sd = c.get("sampled_decode")
    assert sd, f"{name}: no sampled_decode block"
    assert sd["trial"] == int(trial), f"{name}: sampled_decode.trial {sd['trial']} != this phase's trial {trial}"
    assert c["attacks"]["static_max_new_tokens"] == want_cap, f"{name}: static_max_new_tokens {c['attacks']['static_max_new_tokens']} != {want_cap}"
    assert sd["static_cap"] == c["attacks"]["static_max_new_tokens"], f"{name}: static_cap != attacks.static_max_new_tokens"
    ccap = c["attacks"].get("controls_max_new_tokens", c["attacks"]["target_max_new_tokens"])   # cell_controls.controls_cap's rule
    assert sd["controls_cap"] == ccap, f"{name}: controls_cap != the config's controls cap {ccap} (attacks.controls_max_new_tokens, else attacks.target_max_new_tokens)"
    changed = {k for k in set(c) | set(w) if c.get(k) != w.get(k)}
    assert changed == {"out_dir", "private_run", "sampled_decode"} | extra, f"{name}: differs from {wave} in {sorted(changed)}"
    assert c["explicit_cells"] == w["explicit_cells"], f"{name}: explicit_cells differ from the wave's"
assert b["private_run"] == prun and e["private_run"] == prun + "_cap8192", (b["private_run"], e["private_run"])
assert e["sampled_decode"]["escalated_from"] == {"kind": "static", "cap": int(cap), "private_run": prun}, e["sampled_decode"].get("escalated_from")
# the 16384 rung config chains from the 8192 rung (lane F's generator; the extend phase's escalate_static.py reads it)
assert e16["private_run"] == prun + "_cap16384", e16["private_run"]
assert e16["sampled_decode"]["escalated_from"] == {"kind": "static", "cap": int(cap_esc), "private_run": prun + "_cap8192"}, e16["sampled_decode"].get("escalated_from")
ro = b["sampled_decode"].get("readouts") or {}
g8 = ro.get("gsm8k") or {}
if g8.get("omitted") is True:
    # the roster OMITS gsm8k for this target (the sdfbase-llama8b lane: the sibling lane's inspect/gsm8k_paired200 is the paper's one
    # GSM8K instrument; make_sampled_configs writes the omission with its reason) — the readout block says so, no thinking row
    assert isinstance(g8.get("reason"), str) and g8["reason"].strip(), f"{base}: readouts.gsm8k.omitted needs a reason"
    g8_txt = "omitted"
else:
    assert g8.get("thinking") == "on" and int(g8.get("cap", 0)) == 8192, f"{base}: readouts.gsm8k must be thinking on @8192 (or omitted with a reason), got {ro.get('gsm8k')}"
    g8_txt = f"{g8['thinking']}@{g8['cap']}"
print(f"sampled configs OK: {base} / {esc} / {esc16} (trial {trial}; {sorted(k for k in b['sampled_decode'] if k in ('temperature','top_p','top_k','min_p','presence_penalty','repetition_penalty'))}; gsm8k readout {g8_txt})")
PY
}

# --- the rung windows, PER MODEL (2026-09-10). DESIGN.md: 2048 rung -> 4096, 8192 rung (and the static 8192 rung) -> 10240,
# with S = the model budget // L (identity; the launcher refuses a hand-picked S). Nemotron-H is the exception: its HYBRID KV
# cache (MambaSpec per slot, block aligned to the window) held only 165 concurrent slots at L=4096 against the identity S=192,
# and the standup refused ("supports only 165.36-way concurrency at max_model_len=4096", job 84848) — while 256x2048 and
# 161x4864 fit. So nemotron3nano4b uses the two windows this run and the greedy wave ATTESTED on it: 4864 (S 161, the static
# families' window) for the 2048 rung and 8704 (S 90, the multi-turn window) for the 8192 rungs; at 8704 the served clamp
# trims only a prompt longer than 512 tokens (recorded per item as max_tokens_clamped_from). Every other key keeps 4096 / 10240.
# A view WITH a serving window (the llama8b fullconst view, 45,056: the 34k prefix + the longest static prompt +
# the 8,192 rung fit) serves every rung at that window — EXCEPT a rung the window cannot
# carry: 34,395 + 16,384 = 50,779 > 45,056, so the static 16,384 rung, if the 1% trigger ever reaches it, stands up
# at its OWN window 51,200 (S = 475,136 // 51,200 = 9) — never a clamp (prompted_window_check.py refuses the rung at
# 45,056 before any request).
# The 16,384 rung (2026-09-10) follows the SAME derivation: cap + the prompt headroom each family already runs
# with — 2048 for every non-hybrid key (as 10240 = 8192 + 2048; both static families_served:short p99 prompt 762 and controls
# capability:short p99 prompt 575, incl. the ~90-token prompted safety SI, sit far inside it) -> 18432 (S = budget // 18432:
# gemma 32, qwen 39, glm 42); nemotron keeps its attested 512 headroom (8704 = 8192 + 512) -> 16896 (S 46). Identity S x L
# is what the standup ATTESTS; if the hybrid KV cache cannot hold S slots at that L the standup REFUSES (attested S !=
# resolved S) — never a silently shrunk geometry. Both are 128-token block multiples.
# --- PER-LEG WINDOWS (the constitution-SI prompted view llama8b/shortconst, 2026-09-12 — the long-SI mechanism, the
# numbers from the ROSTER (views.<view>.serving.windows, read once by f22s_resolve_config) rather than a shell table): a ~5.9k-token
# system turn does not fit the :short windows (vLLM rejects prompt + cap > L; the served client's once-only clamp would trim
# max_tokens — not a draw at the rung's cap), so a per-leg view stands every phase up at ITS max-prompt-sized window: static base
# 10,496 / controls base 6,656 / rungs 2048 -> 8,064, 8192 -> 14,592, 16384 -> 22,784 (llama8b; S = budget // L, attested or refused).
# A one-window view (fullconst) answers every field with its window; a table view answers "" and keeps every default below.
# --- LONG-SI PROMPTED CELLS (the constitution-SI prompted baseline, 2026-09-12). Every window
# above was sized for cells WITHOUT an SI prefix (families_served:short p99 prompt 762; capability:short p99 prompt 575; the
# prompted SAFETY cell's ~90-token SI sits inside that headroom). The si_constitution / si_irrelevant system turns alone are
# ~6k tokens (geometry.py: 'the si_constitution / si_irrelevant system turns alone are ~5.9-6.7k tokens'), so a request of
# theirs at the :short windows is REJECTED by vLLM (prompt + cap > L) and the served client's once-only clamp trims max_tokens
# to the room left (served_window.py: MIN_CLAMPED_COMPLETION 64, recorded as max_tokens_clamped_from) — a clamped draw is
# NOT a draw at the rung's cap, and the at-cap counters would read it as one. Refusing to truncate the system prompt or the
# cap, the sampled jobs serve a long-SI cell at windows sized by the MAX-PROMPT branch of geometry.py's rule (the window
# holds EVERY prompt of the class at the cap; L = round-up-256(max_prompt + cap), headroom < 256), from MEASURED numbers:
#   * the SI constant per model, tokenized under the PINNED tokenizer (2026-09-12; the render adds a constant to every item):
#       si_constitution 6,027 (gemma4e4b) / 5,958 (qwen35_4b) / 5,952 (nemotron3nano4b); safety_refusal 87 / 88 / 88,
#       so constitution = the served prompted__safety_refusal prompt + 5,940 / 5,870 / 5,864 tokens;
#   * the served prompt_tokens of the prompted__safety_refusal cell's OWN sampled stores (every row, both trials, base + rung
#       roots; gemma4e4b_it / qwen35_4b_it / nemotron3nano4b): static max 533 / 544 / 566 (grok420), controls max 151 / 154
#       / 153 (or_bench_hard_1k; xstest 120 / 124 / 123);
#   -> constitution max prompt: static 6,473 / 6,414 / 6,430; controls 6,091 / 6,024 / 6,017 (the three models land on the
#      SAME 256-blocks at every cap, so one table serves all; a fourth model is NOT covered by it — measure first).
# Windows (max_prompt + cap -> round-up-256):  static 4096 -> 10,752 (10,569 / 10,510 / 10,526);  controls 512 -> 6,656
# (6,603 / 6,536 / 6,529);  rung 2048 (controls) -> 8,192 (8,139 / 8,072 / 8,065);  rung 8192 (static AND controls, one
# number, the static max governs) -> 14,848 (14,665 / 14,606 / 14,622);  rung 16384 -> 23,040 (22,857 / 22,798 / 22,814).
# S stays the identity budget // L (gemma 54 / 39 / 25, qwen 67 / 48 / 31, nemotron 73 / 52 / 34 at 10,752 / 14,848 /
# 23,040); a hybrid cache that cannot hold S slots at the new L REFUSES at the standup (attested S != resolved S), never a
# silently shrunk geometry. NOT covered: the GSM8K readout under a long SI (its prompt is ~2.4k + the SI; capability_t<T>
# REFUSES a long-SI cell rather than serve it at an unmeasured window). F22S_LONG_SI is set by f22s_detect_long_si from the
# job's CELLS; empty = every window exactly as before (the extracted-function test runs f22s_rung_window with it unset).
F22S_LONG_SI_CELLS=(prompted__si_constitution prompted__si_irrelevant)   # contracts.MATH500_PROMPT_CLASS_OF_SI's long_si class
F22S_LONG_SI=""
f22s_detect_long_si(){   # $@ = the job's cells -> F22S_LONG_SI=1 when any is a long-SI prompted cell (logged), "" otherwise
  local c l
  F22S_LONG_SI=""
  for c in "$@"; do for l in "${F22S_LONG_SI_CELLS[@]}"; do [ "$c" = "$l" ] && F22S_LONG_SI=1; done; done
  if [ -n "$F22S_LONG_SI" ]; then
    f22_log "LONG-SI cell among the job's cells ($*): windows static base $(f22s_base_window static) / controls base $(f22s_base_window controls) / rungs 2048 $(f22s_rung_window 2048), 8192 $(f22s_rung_window 8192), 16384 $(f22s_rung_window 16384) (max-prompt sizing; see common_sampled.sh)"
  fi
}
# --- ONE resolution order for both mechanisms (merge 2026-09-12, sdfbase-llama8b x sampled-constitution-si): a view that DECLARES
# its windows in the roster (llama8b/shortconst: measured for Llama — static 10,496 / controls 6,656 / rungs 8,064 / 14,592 / 22,784;
# fullconst: its one window) wins; else a job serving a LONG-SI cell on a table view (gemma / qwen / nemotron with the constitution
# cell on main / it) takes the measured long-SI table above; else the wave's defaults. The long-SI table is NOT a fourth model's
# measurement (see above) — a fourth model reaches it only through a roster-declared view window.
f22s_base_window(){   # $1 = static|controls -> the BASE phase's --max-model-len (see the resolution order above)
  case "$1" in
    static)
      if [ -n "${VIEW_WINDOW_STATIC:-${VIEW_WINDOW:-}}" ]; then echo "${VIEW_WINDOW_STATIC:-$VIEW_WINDOW}"
      elif [ -n "${F22S_LONG_SI:-}" ]; then echo 10752
      else echo ""; fi;;                                            # "" = the table's families_served:short 4864
    controls)
      if [ -n "${VIEW_WINDOW_CONTROLS:-${VIEW_WINDOW:-}}" ]; then echo "${VIEW_WINDOW_CONTROLS:-$VIEW_WINDOW}"
      elif [ -n "${F22S_LONG_SI:-}" ]; then echo 6656
      else echo 2048; fi;;                                          # the wave's capability:short window
    *) f22_fatal "f22s_base_window: unknown phase kind $1";;
  esac
}
f22s_rung_window(){   # $1 = rung (2048|8192|16384) -> the standup's --max-model-len for $KEY / $VIEW (the resolution order above)
  local vr; case "$1" in 2048) vr="${VIEW_RUNG_WINDOW_2048:-}";; 8192) vr="${VIEW_RUNG_WINDOW_8192:-}";; 16384) vr="${VIEW_RUNG_WINDOW_16384:-}";; *) vr="";; esac
  if [ -n "$vr" ] && [ -z "${VIEW_WINDOW:-}" ]; then echo "$vr"; return 0; fi   # a PER-LEG view: the roster's rung window
  if [ -n "${VIEW_WINDOW:-}" ]; then                                            # a ONE-WINDOW view (fullconst)
    case "$1" in
      2048|8192) echo "$VIEW_WINDOW";;
      16384) if [ "$VIEW_WINDOW" -ge 51200 ]; then echo "$VIEW_WINDOW"; else echo 51200; fi;;
      *) f22_fatal "no rung window for $KEY/$VIEW rung $1";;
    esac
    return 0
  fi
  if [ -n "${F22S_LONG_SI:-}" ]; then                                           # a LONG-SI cell on a table view: the measured long-SI table
    case "$1" in 2048) echo 8192;; 8192) echo 14848;; 16384) echo 23040;; *) f22_fatal "no long-SI rung window for rung $1";; esac
    return
  fi
  case "$KEY:$1" in
    nemotron3nano4b:2048) echo 4864;;
    nemotron3nano4b:8192) echo 8704;;
    nemotron3nano4b:16384) echo 16896;;
    *:2048) echo 4096;;
    *:8192) echo 10240;;
    *:16384) echo 18432;;
    *) f22_fatal "no rung window for $KEY rung $1";;
  esac
}
f22s_view_has_leg(){ f22_view_has_leg "$@"; }   # the name target_sampled.sbatch calls; the predicate is common.sh's
f22s_record_si_prefix(){ f22_record_si_prefix "$@"; }   # the name used here (2026-09-12); common.sh's f22_standup already records at every standup
# GSM8K thinking-ON @8192: the 8192-rung window of the model (10240; nemotron 8704) — the GSM8K prompt (question + the parity
# instruction, + ~90 tokens of the prompted safety SI on the cells that carry it) is under 512 tokens on every model (measured
# from the served prompt_tokens of the sampled controls stores: p99 45-141), so the attested 8192-rung geometry carries the
# 8,192 completion cap with no served clamp. One window per model, already attested on this run.
f22s_gsm8k_window(){ f22s_rung_window 8192; }

# --- STOP: the wave-wide sentinel OR this job's own (README section 8) ----------------------------------------------
f22s_stop_requested(){ f22_stop_requested || [ -e "${JOB_STOP:-/nonexistent/STOP}" ]; }
f22s_register_gated_pairs(){   # $1 = phase kind (the log), $2 = "fam:cell,…" the gate selected -> registers EXACTLY them (static-reg-fix-2)
  # Called by the rung phases after their gate and BEFORE the window check / standup: the job-start registration was deferred
  # (f22s_parse_phase F22_REGISTRY_DEFER), so this is the ONE registration of the job. The pairs are sorted (a stable record). The config is
  # the WAVE's, as the base phase registers it (f22_resolve_config binds CFG to it before f22s_resolve_config re-points CFG at the trial's
  # sampled config), so the three static rungs of one trial carry the same extra.config. A refusal is fatal inside f22_registry_register.
  local kind="$1" pairs n
  pairs=$(printf '%s\n' ${2//,/ } | sed '/^$/d' | sort -u | paste -sd, -)
  [ -n "$pairs" ] || f22_fatal "phase $kind: f22s_register_gated_pairs called with no selected pair — the all-zero phase exits before this (a logic error, not a gate verdict)"
  n=$(printf '%s' "$pairs" | awk -F, '{print NF}')
  f22_log "phase $kind: registering the $n gate-selected pair(s) before any standup: $pairs (D1265)"
  F22_REGISTRY_DEFER= F22_REGISTRY_PAIRS="$pairs" CFG="${WAVE_CFG:-$CFG}" f22_registry_register
}

# --- the at-cap counters (lane A / lane B scripts; exit 2 = REFUSAL, e.g. a greedy store) — never a guess -----------
f22s_at_cap_static(){   # $1 = private_run, $2 = attack, $3 = cell -> echoes the integer; returns 2 on refusal / non-integer
  local out rc err="$SERVE_DIR/at_cap_static.last.err"
  out=$(cd "$REPO_DIR" && "$MPY" "$F22S_DIR/at_cap_static.py" --private-root "$PRIVATE_ROOT" --private-run "$1" --attack "$2" --defense "$3" 2>"$err"); rc=$?
  if [ "$rc" != 0 ]; then f22_log "at_cap_static.py REFUSED rc=$rc for $1/$2/$3: $(tr '\n' ' ' < "$err" | head -c 600)"; return 2; fi
  [[ "$out" =~ ^[0-9]+$ ]] || { f22_log "at_cap_static.py printed a non-integer for $1/$2/$3: ${out:0:200}"; return 2; }
  echo "$out"
}
f22s_at_cap_controls(){   # $1 = out-root, $2 = cell, $3 = control -> echoes the integer; returns 2 on refusal / non-integer
  local out rc err="$SERVE_DIR/at_cap_controls.last.err"
  out=$(cd "$REPO_DIR" && "$MPY" "$F22S_DIR/at_cap_controls.py" --out-root "$1" --cell "$2" --control "$3" 2>"$err"); rc=$?
  if [ "$rc" != 0 ]; then f22_log "at_cap_controls.py REFUSED rc=$rc for $1 $2/$3: $(tr '\n' ' ' < "$err" | head -c 600)"; return 2; fi
  [[ "$out" =~ ^[0-9]+$ ]] || { f22_log "at_cap_controls.py printed a non-integer for $1 $2/$3: ${out:0:200}"; return 2; }
  echo "$out"
}

# --- the cap-ladder GATE: ONE bound decision per (cell, control) at the rung just measured ---------------
# ladder_gate.py decide = ladder_decide.py's item-filtered 1 % TRUNCATED verdict (loops / repetitive / boundary excluded,
# the 16,384 ceiling) BOUND to the transport approval for the rung above (the previous standup's recorded proxy hop +
# the driver's client timeout). Echoes exactly one line: `ESCALATE <next_cap> <items.jsonl>` or `EXIT <reason>`;
# returns 2 on a REFUSAL (never a number the loop could act on). The items file is the driver's --items.
_f22s_gate(){   # $1 = log label, $2 = the gate dir, $3 = "<store> <pair> @<rung>" for the log, then the ladder_gate.py command
  # the ONE body under both gate names: run the tool, refuse (return 2) on a non-zero exit or on anything but a bound
  # decision line — `ESCALATE <next_cap> <items.jsonl>` or `EXIT <reason>` — and echo that line verbatim
  local label="$1" gdir="$2" ctx="$3" out rc err="$SERVE_DIR/ladder_gate.last.err"; shift 3
  mkdir -p "$gdir"
  out=$(cd "$REPO_DIR" && "$@" 2>"$err"); rc=$?
  if [ "$rc" != 0 ]; then f22_log "$label REFUSED rc=$rc for $ctx: $(tr '\n' ' ' < "$err" | head -c 600)"; return 2; fi
  case "$out" in ESCALATE\ [0-9]*\ /*|EXIT\ exit_le_1pct|EXIT\ exit_ceiling|EXIT\ transport_refused) ;;
    *) f22_log "$label printed something other than a bound decision for $ctx: ${out:0:200}"; return 2;; esac
  echo "$out"
}
f22s_ladder_gate(){   # $1 = out-root, $2 = cell, $3 = control, $4 = the rung the store was drawn at, $5 = that rung's endpoint.json
  local gdir="$LADDER_DIR/gates/cap$4"
  local sib="$WAVE_ROOT/sampled/controls_cap$4/${TAG}_t$((1 - TRIAL))"      # the other trial's store at this rung (boundary rule; absent = no boundary calls)
  _f22s_gate "ladder_gate.py" "$gdir" "$1 $2/$3 @$4" \
    "$MPY" "$F22S_DIR/ladder_gate.py" decide --out-root "$1" --cell "$2" --control "$3" --cap "$4" \
          --endpoint "$5" --sibling-out-root "$sib" \
          --out "$gdir/${2}__${3}.gate.json" --decision-out "$gdir/${2}__${3}.decision.json" --items-out "$gdir/${2}__${3}.items.jsonl"
}
f22s_static_gate(){   # $1 = the store's private_run, $2 = attack, $3 = cell, $4 = the rung it was drawn at, $5 = that rung's endpoint.json
  # the static twin of f22s_ladder_gate: ladder_gate.py static over $PRIVATE_ROOT/$1/$2/$3; the
  # items file is escalate_static.py's --items; the sibling is the OTHER trial's run at the same rung (boundary rule)
  local gdir="$LADDER_DIR/gates/static_cap$4"
  local sib="${1/_sampled_t${TRIAL}/_sampled_t$((1 - TRIAL))}"
  _f22s_gate "ladder_gate.py static" "$gdir" "$1 $2/$3 @$4" \
    "$MPY" "$F22S_DIR/ladder_gate.py" static --private-root "$PRIVATE_ROOT" --private-run "$1" --attack "$2" --defense "$3" --cap "$4" \
          --endpoint "$5" --sibling-private-run "$sib" \
          --out "$gdir/${2}__${3}.gate.json" --decision-out "$gdir/${2}__${3}.decision.json" --items-out "$gdir/${2}__${3}.items.jsonl"
}
f22s_standup_endpoint(){   # $1 = a standup dir of THIS job -> the endpoint record a gate reads AFTER f22_standup_stop (2026-09-13):
                           # the live endpoint.json while the server is up; else the byte-identical READY copy f22_standup keeps
                           # (endpoint.recorded.json — mt_serving_up.py UNLINKS the live file at teardown, which is why every
                           # llama8b (cell, control) pair left its 512 ladder `transport_refused`: the controls loop gated the 512
                           # stores after stopping the 512 server and read an unlinked file); else the live path, so the gate refuses
                           # on the absent record exactly as before — never a guessed hop
  if [ -f "$1/endpoint.json" ]; then echo "$1/endpoint.json"
  elif [ -f "$1/endpoint.recorded.json" ]; then echo "$1/endpoint.recorded.json"
  else echo "$1/endpoint.json"; fi
}
f22s_prev_endpoint(){   # $1 = the previous phase's PHASE_TAG prefix (e.g. static_esc_t0), $2 = its standup subdir -> the NEWEST endpoint.json of this TAG
  # the previous rung ran in ANOTHER job (its own SERVE_DIR), so its standup record is found by name under sampled/serving;
  # F22S_PREV_ENDPOINT names one explicitly. Nothing found = the transport approval is undeterminable (the caller refuses)
  local f
  if [ -n "${F22S_PREV_ENDPOINT:-}" ]; then [ -f "$F22S_PREV_ENDPOINT" ] && { echo "$F22S_PREV_ENDPOINT"; return 0; }; return 1; fi
  # the dir is <PHASE_TAG>_<TAG>_<SLURM_JOB_ID>: the job id anchors the TAG, so TAG `llama8b` never reads `llama8b_it`'s or
  # `llama8b_fullconst`'s record (a family-subset PHASE_TAG sits BEFORE the TAG and is what the first * absorbs)
  f=$(ls -t "$WAVE_ROOT"/sampled/serving/"$1"*_"${TAG}"_[0-9]*/"$2"/endpoint.json 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "$f"; return 0; }
  # the previous rung's standup UNLINKS endpoint.json at teardown (mt_serving_up.py), so a rung run as a later
  # job finds none; f22_standup keeps a byte-identical copy at READY (endpoint.recorded.json) — read that; for a standup that
  # ended before the copy existed, RECONSTRUCT the record the transport approval needs from that standup's standup_jobs.jsonl
  # + the tree's proxy hop default (reconstruct_prev_endpoint.py refuses when the record is missing or the phase could have
  # overridden the hop; the reconstruction is stamped as such on the file it writes)
  f=$(ls -t "$WAVE_ROOT"/sampled/serving/"$1"*_"${TAG}"_[0-9]*/"$2"/endpoint.recorded.json 2>/dev/null | head -1)
  [ -n "$f" ] && { echo "$f"; return 0; }
  local sj; sj=$(ls -t "$WAVE_ROOT"/sampled/serving/"$1"*_"${TAG}"_[0-9]*/"$2"/standup_jobs.jsonl 2>/dev/null | head -1)
  [ -n "$sj" ] || return 1
  f=$(cd "$REPO_DIR" && "$MPY" "$F22S_DIR/reconstruct_prev_endpoint.py" --standup-dir "$(dirname "$sj")" 2>>"${F22S_LOG:-/dev/null}") || return 1
  # the note goes to STDERR: callers capture this function's stdout as THE path (EP_512=$(f22s_prev_endpoint …)); a stdout log line
  # made the capture two lines and the gate read a non-path -> "no hop timeouts supplied", every pair transport_refused (100037, 2026-09-13)
  [ -n "$f" ] && [ -f "$f" ] && { f22_log "prev endpoint RECONSTRUCTED for $1/$2: $f (the standup's endpoint.json was unlinked at teardown; source = its standup_jobs.jsonl + the tree's proxy hop default)" >&2; echo "$f"; return 0; }
  return 1
}
f22s_gate_event(){   # $1 = gate.json -> the ladder event carrying the decision (controls: cell/control; static: attack/cell), folded by write_ladder.py
  "$MPY" - "$1" <<'PYG'
import json, sys
from datetime import datetime, timezone
g = json.load(open(sys.argv[1])); c = g["counts"]
head = ({"kind": "static", "attack": g["attack"], "cell": g["cell"]} if g["kind"] == "static"
        else {"kind": "controls", "cell": g["cell"], "control": g["control"]})
print(json.dumps({**head, "rung": int(g["cap"]),
                  "n_at_cap": int(c["n_at_cap"]), "root": g["out_root"], "decision": g["decision"], "next_cap": g["next_cap"],
                  "n_truncated": c["n_truncated"], "n_loop": c["n_loop"], "n_repetitive": c["n_repetitive"],
                  "n_boundary": c["n_boundary"], "share_truncated": c["share_truncated"],
                  "transport_ok": (None if g["transport"] is None else bool(g["transport"]["ok"])),
                  "n_items_escalate": g["n_items_escalate"], "gate": sys.argv[1],
                  "at": datetime.now(timezone.utc).isoformat()}))
PYG
}
f22s_gate_summary(){   # $1 = gate.json -> one human line: what climbs, what stays, and WHY (loops stay distinct from unresolved truncation)
  "$MPY" - "$1" <<'PYS'
import json, sys
g = json.load(open(sys.argv[1])); c = g["counts"]; d = g["decision"]
why = {"escalate": f"{g['n_items_escalate']} truncated items climb to {g['next_cap']}",
       "exit_le_1pct": f"truncated {c['n_truncated']}/{c['n_responses']} <= 1% — resolved",
       "exit_ceiling": f"UNRESOLVED at the ceiling: {c['n_truncated']} truncated items",
       "transport_refused": "the transport cannot carry the rung above: " + str((g.get("transport") or {}).get("reason"))}.get(d, d)
print(f"{why}; at-cap {c['n_at_cap']}/{c['n_responses']}: loops {c['n_loop']} (non-terminating, never escalated), "
      f"repetitive {c['n_repetitive']}, boundary {c['n_boundary']} stay")
PYS
}
f22s_ladder_transport(){   # $1 = the NEW rung's endpoint.json, $2 = the rung -> 0 when its recorded proxy hop carries the rung
  local out rc
  out=$(cd "$REPO_DIR" && "$MPY" "$F22S_DIR/ladder_gate.py" transport --endpoint "$1" --next-cap "$2" 2>&1); rc=$?
  f22_log "transport check for rung $2 against $1: ${out:0:300}"
  return $rc
}

# --- ladder events (one JSON line per measurement, folded by write_ladder.py) ---------------------------------------
f22s_event_json(){   # k=v ... -> one JSON object (rung / n_at_cap as integers; `at` stamped)
  "$MPY" - "$@" <<'PY'
import json, sys
from datetime import datetime, timezone
d = {}
for kv in sys.argv[1:]:
    k, _, v = kv.partition("=")
    d[k] = int(v) if k in ("rung", "n_at_cap") else v
d["at"] = datetime.now(timezone.utc).isoformat()
print(json.dumps(d))
PY
}
f22s_event(){ mkdir -p "$LADDER_DIR"; printf '%s\n' "$1" >> "$LADDER_DIR/events.jsonl"; }
f22s_json_field(){   # $1 = a JSON object, $2 = key -> its value as text ("" when absent / null)
  "$MPY" -c 'import json, sys; v = json.loads(sys.argv[1]).get(sys.argv[2]); print("" if v is None else v)' "$1" "$2"
}
# (P2-α) the rung-keyed UNDECIDED mark: at a rung where the ladder's decision for a pair could not be made —
# the gate REFUSED over a finished store, or the rung's escalation record is missing / unreadable — this event is filed
# before the loop moves on, so write_ladder.py folds the pair as `left_ladder_at = {undecided, rung, cause, reason}` instead
# of carrying the rung below's ESCALATE forward as an active escalation to a rung that has already run. cause is
# gate_refused | record_problem; reason is the tool's own words (one line).
f22s_undecided_event(){   # $1 = attack, $2 = cell, $3 = rung, $4 = cause, $5 = reason -> the static_undecided event
  f22s_event_json kind=static_undecided attack="$1" cell="$2" rung="$3" cause="$4" reason="$5"
}
f22s_last_gate_refusal(){   # -> one line: the stderr head of the LAST ladder_gate.py call (f22s_static_gate / f22s_ladder_gate write $SERVE_DIR/ladder_gate.last.err on every call)
  local err="$SERVE_DIR/ladder_gate.last.err" head=""
  [ -s "$err" ] && head=$(tr '\n' ' ' < "$err" | head -c 600)
  echo "ladder_gate.py static REFUSED: ${head:-(no stderr — the tool printed something other than a bound decision)}"
}
f22s_escalation_event(){   # $1 = attack, $2 = cell, [$3 = the rung's private_run; default PRIVATE_RUN_ESC] -> the static_escalation event (reads escalation.json); returns 3 when it is missing/unreadable
  "$MPY" - "$PRIVATE_ROOT/${3:-$PRIVATE_RUN_ESC}/$1/$2" "$1" "$2" <<'PY'
import json, sys, pathlib
from datetime import datetime, timezone
root, attack, cell = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
cands = [root / "escalation.json", root / "shard00" / "escalation.json"]
found = next((p for p in cands if p.is_file()), None)
ev = {"kind": "static_escalation", "attack": attack, "cell": cell, "escalation": None, "problem": None,
      "at": datetime.now(timezone.utc).isoformat()}
if found is None:
    ev["problem"] = f"escalation.json not found under {root} (looked in {[str(p) for p in cands]})"
else:
    try:
        ev["escalation"] = json.load(open(found)); ev["escalation_path"] = str(found)
    except Exception as e:   # noqa: BLE001 — recorded as the problem, never swallowed
        ev["problem"] = f"{found}: unreadable ({e})"
print(json.dumps(ev))
sys.exit(3 if ev["problem"] else 0)
PY
}
f22s_write_ladder(){   # $1 = kind (controls|static), $2 = the ladder file -> folds $LADDER_DIR/events.jsonl (merge)
  # shellcheck disable=SC2046  # f22s_count_only_flag prints two words or nothing, on purpose
  (cd "$REPO_DIR" && "$MPY" "$F22S_DIR/write_ladder.py" --events "$LADDER_DIR/events.jsonl" --out "$2" --kind "$1" \
      --tag "$TAG" --trial "$TRIAL" --floor-sample-k "$F22S_FLOOR_SAMPLE_K" $(f22s_count_only_flag "$1")) \
    && cp -f "$2" "$SERVE_DIR/$(basename "$2")" 2>/dev/null \
    || f22_log "write_ladder.py FAILED for $1 -> $2 (events kept at $LADDER_DIR/events.jsonl)"
}

# --- the extended ladder folds (16384 in the rung list; write_ladder.py reads the decision + hop-guard events too) ------
f22s_write_ladder_ext(){   # $1 = kind (controls|static), $2 = the ladder file, $3 = rung list (comma) [$4 = events file ("" = the
                           # ladder dir's)] [$5 = extend kind (static|controls|gsm8k; default = kind) — the GSM8K readout's
                           # controls-shaped events share events.jsonl and fold into gsm8k_ladder.json]
  local ev="${4:-}"; [ -n "$ev" ] || ev="$LADDER_DIR/events.jsonl"
  local ext="${5:-$1}"
  # shellcheck disable=SC2046  # the count-only era's ceiling applies to the controls extend fold only (its 4-rung list is what re-read the legacy pairs)
  (cd "$REPO_DIR" && "$MPY" "$F22S_DIR/write_ladder.py" --events "$ev" --out "$2" --kind "$1" --rungs "$3" --extend-kind "$ext" \
      --tag "$TAG" --trial "$TRIAL" --floor-sample-k "$F22S_FLOOR_SAMPLE_K" $(f22s_count_only_flag "$ext")) \
    && cp -f "$2" "$SERVE_DIR/$(basename "$2")" 2>/dev/null \
    || f22_log "write_ladder.py FAILED for $1 ($ext) -> $2 (events kept at $ev)"
}

# --- the extend decision + the ceiling report (ladder_extend.py; LADDER_RULES.md §5) -------------------------------------
f22s_ladder_extend_decide(){   # $1 = cap, $2 = next cap ("" = the CEILING report at $1: nothing escalates, residual share + count),
                               # $3 = out (decision_16384.json | ceiling_16384.json), $4.. = passthrough (--reclassify K=P … --hop-guard R …)
                               # -> returns 2 on a refusal (an incomplete reclassify, a foreign rule, a hop guard that is not PASS)
  local cap="$1" nxt="$2" out="$3" rc; shift 3
  local -a nc=(); [ -n "$nxt" ] && nc=(--next-cap "$nxt")
  (cd "$REPO_DIR" && "$MPY" "$F22S_DIR/ladder_extend.py" decide --cap "$cap" ${nc[@]+"${nc[@]}"} --tag "$TAG" --trial "$TRIAL" \
      --out "$out" --events "$LADDER_DIR/events.jsonl" "$@") > "$out.log" 2>&1; rc=$?
  cp -f "$out" "$SERVE_DIR/$(basename "$out")" 2>/dev/null || true
  if [ "$rc" != 0 ]; then f22_log "ladder_extend.py decide REFUSED rc=$rc (cap $cap${nxt:+ -> $nxt}): $(grep -v '^{' "$out.log" | tail -3 | tr '\n' ' ' | head -c 900)"; return 2; fi
  f22_log "ladder_extend.py decide (cap $cap${nxt:+ -> $nxt}) -> $out: $(f22s_decision_summary "$out")"
}
f22s_decision_summary(){   # $1 = a ladder_extend decision record -> one line: decision, pairs, promotions, exit kinds per kind
  "$MPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ("decision","n_pairs_considered","n_promoted","n_by_exit_kind","at_ceiling")}, {k: (v.get("n_promoted"), v.get("n_by_exit_kind")) for k, v in (d.get("by_kind") or {}).items()})' "$1"
}
f22s_decision_field(){   # $1 = record, $2 = top-level field -> its value (ints print bare); exit 1 when absent
  "$MPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); v=d[sys.argv[2]]; print(v if not isinstance(v, (dict, list)) else json.dumps(v))' "$1" "$2"
}
f22s_extend_list(){   # $1 = kind (static|controls|gsm8k), $2 = the decision record -> TAB-separated promoted pairs on stdout; non-zero = refusal
  (cd "$REPO_DIR" && "$MPY" "$F22S_DIR/ladder_extend.py" list --decision "$2" --kind "$1")
}
f22s_hop_guard_event(){   # $1 = kind (static|controls), $2 = extend kind, $3 = rung, $4 = the POST-standup hop_timeout_guard record -> appended to events.jsonl
  f22s_event "$("$MPY" - "$1" "$2" "$3" "$4" <<'PY'
import json, sys
from datetime import datetime, timezone
kind, ext, rung, rec = sys.argv[1], sys.argv[2], int(sys.argv[3]), json.load(open(sys.argv[4]))
print(json.dumps({"kind": kind, "extend_kind": ext, "rung": rung, "hop_timeout_guard": rec, "at": datetime.now(timezone.utc).isoformat()}))
PY
)"
}
# --- the capability battery's roots (FIXED names — the analysis reads them) ------------------------------------------------
f22s_gsm8k_root(){ echo "$WAVE_ROOT/sampled/gsm8k_thinking_on_cap$1/${TAG}_t${TRIAL}"; }   # $1 = cap (8192 | 16384)
f22s_mmlu_root(){ echo "$WAVE_ROOT/sampled/mmlu/${TAG}_t${TRIAL}"; }

# --- the classifier's rep_ratio source (LADDER_RULES.md §2): ladder_decide.py REFUSES without it; on the cluster it is a copy
# placed under $WAVE_ROOT/sampled/tools/ (the grading host's $STAGE/sdfbase/fullscale/full22_join_scores.py), its sha256
# pinned here and recorded on every decision by the tool itself
f22s_rep_ratio_source(){   # -> REP_SRC; returns 2 when absent or not the pinned bytes
  REP_SRC="$WAVE_ROOT/sampled/tools/full22_join_scores.py"
  [ -f "$REP_SRC" ] || { f22_log "REFUSING: rep_ratio source $REP_SRC is absent (copy the grading host's \$STAGE/sdfbase/fullscale/full22_join_scores.py there; LADDER_RULES.md §2)"; return 2; }
  local sha; sha=$(sha256sum "$REP_SRC" | cut -d' ' -f1)
  [ "$sha" = "$F22S_REP_RATIO_SOURCE_SHA256" ] || { f22_log "REFUSING: rep_ratio source $REP_SRC sha256 $sha != the pinned $F22S_REP_RATIO_SOURCE_SHA256"; return 2; }
  f22_log "rep_ratio source: $REP_SRC sha256 $sha"
}

# --- ladder_decide.py reclassify over the CLUSTER roots (LADDER_RULES.md §5): every rung already generated, no regeneration ---
# $1 = kind: static | controls | gsm8k   $2 = out dir   -> reclassify.json under $2; returns 2 when the pass is not complete
# (a refused store) or the tool refused. The GSM8K readout roots are controls-shaped stores under their own --root-pattern
# (the sampled arm per trial, sibling = the other trial; there is no greedy comparator).
f22s_reclassify(){
  local kind="$1" out="$2" rc log="$2.log"
  mkdir -p "$out"
  case "$kind" in
    static)   (cd "$REPO_DIR" && "$MPY" "$F22S_DIR/ladder_decide.py" reclassify --kind static --root "$PRIVATE_ROOT" \
                 --run "$PRIVATE_RUN_BASE" --sibling-run "sdfbase_full22_${TAG}_sampled_t${SIBLING_TRIAL}" \
                 --rep-ratio-source "$REP_SRC" --out-dir "$out") > "$log" 2>&1; rc=$?;;
    controls) (cd "$REPO_DIR" && "$MPY" "$F22S_DIR/ladder_decide.py" reclassify --kind controls --root "$WAVE_ROOT/sampled" \
                 --tag "$TAG" --trial "$TRIAL" --sibling-trial "$SIBLING_TRIAL" \
                 --rep-ratio-source "$REP_SRC" --out-dir "$out") > "$log" 2>&1; rc=$?;;
    gsm8k)    (cd "$REPO_DIR" && "$MPY" "$F22S_DIR/ladder_decide.py" reclassify --kind controls --root "$WAVE_ROOT/sampled" \
                 --tag "$TAG" --trial "$TRIAL" --sibling-trial "$SIBLING_TRIAL" --root-pattern "gsm8k_thinking_on_cap{cap}/{tag}_t{trial}" \
                 --rep-ratio-source "$REP_SRC" --out-dir "$out") > "$log" 2>&1; rc=$?;;
    *) f22_log "f22s_reclassify: unknown kind $kind"; return 2;;
  esac
  if [ "$rc" != 0 ]; then f22_log "reclassify $kind REFUSED rc=$rc (the pass is not complete): $(grep -v '^{' "$log" | tail -3 | tr '\n' ' ' | head -c 700)"; return 2; fi
  f22_log "reclassify $kind: $(grep '^{' "$log" | tail -1 | "$MPY" -c 'import json,sys; d=json.load(sys.stdin); print({k: d.get(k) for k in ("rungs_present","rungs_absent","n_entries","n_decided","n_refused","n_by_decision","complete")})')"
}

# --- the HOP-TIMEOUT ADEQUACY GUARD (2026-09-10; hop_timeout_guard.py) ---------------------------------------------------
# $1 = rung, $2 = client (static|controls — the GSM8K readout is the controls driver), $3 = record path,
# [$4 = endpoint.json]. TWO phases, neither sufficient alone: WITHOUT $4 the guard runs PRE-standup (`--pre-standup`: the
# configured default the standup will apply, by import — decides whether a standup is worth a GPU minute); WITH $4 it runs
# POST-standup from the RECORD the standup wrote, and that run is MANDATORY before any request (a compiled-in default is the
# value that censored the slow leg once). The floor decode rate comes from this MODEL's most recent rung at the served
# geometry: the newest vllm_server.log under $WAVE_ROOT/sampled/serving/ for this KEY (either view) in the sub-dir the
# client's rung ran in (static: static_esc / static_16384; controls / gsm8k: controls_8192 / controls_16384 /
# gsm8k_thinking_on), falling back to the other kind's log — a model with no rung log at all is undeterminable and refuses.
# Returns 2 on REFUSE.
f22s_newest_vllm_log(){   # $1.. = serving sub-dir names in preference order -> the newest matching log path, or ""
  local sub best=""
  for sub in "$@"; do
    best=$(ls -t "$WAVE_ROOT"/sampled/serving/*_"${KEY}"*_*/"$sub"/vllm_server.log 2>/dev/null | head -1)
    [ -n "$best" ] && { echo "$best"; return 0; }
  done
  echo ""
}
f22s_hop_guard(){
  local rung="$1" client="$2" rec="$3" ep="${4:-}" log rc phase
  local -a mode
  if [ -n "$ep" ]; then mode=(--endpoint "$ep"); phase=post-standup; else mode=(--pre-standup); phase=pre-standup; fi
  case "$client" in
    static)   log=$(f22s_newest_vllm_log static_esc static_16384 controls_8192 controls_16384 static);;
    controls) log=$(f22s_newest_vllm_log controls_8192 controls_16384 gsm8k_thinking_on static_esc static_16384 controls_2048 controls_512);;
    *) f22_log "f22s_hop_guard: unknown client $client"; return 2;;
  esac
  [ -n "$log" ] || { f22_log "hop-timeout guard REFUSED (rung $rung, $client, $phase): no vllm_server.log for $KEY under $WAVE_ROOT/sampled/serving — the floor decode rate is undeterminable"; return 2; }
  (cd "$REPO_DIR" && "$MPY" "$F22S_DIR/hop_timeout_guard.py" --rung "$rung" --client "$client" --vllm-log "$log" \
      --margin "$F22S_HOP_GUARD_MARGIN" --out "$rec" --repo-dir "$REPO_DIR" "${mode[@]}") > "$rec.out" 2> "$rec.err"; rc=$?
  if [ "$rc" != 0 ]; then f22_log "hop-timeout guard REFUSED (rung $rung, $client, $phase): $(tr '\n' ' ' < "$rec.err" | head -c 900)"; return 2; fi
  f22_log "hop-timeout guard PASS (rung $rung, $client, $phase): $("$MPY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print({k: d.get(k) for k in ("phase","binding_s","binding_hop","upstream_timeout_s","upstream_key","client_timeout_s","floor_tok_s","raw_min_tok_s","floor_n_eligible","margin","required_s")})' "$rec")"
}

# --- GSM8K: the item set is the greedy baseline's by construction; asserted per cell from the spool header ------------------
f22s_assert_gsm8k_items(){   # $1 = out-root, $2 = cell id -> 0 when the store's items_sha256 / n_items equal the pinned greedy set
  "$MPY" - "$1" "$2" "$F22S_GSM8K_ITEMS_SHA256" "$F22S_GSM8K_N" <<'PY' || { f22_log "GSM8K ITEM SET MISMATCH for $2 under $1 (see above)"; return 1; }
import json, sys
from pathlib import Path
root, cell, want_sha, want_n = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
found = None
for mp in sorted((root / "cells").glob("*/manifest.json")):
    if json.loads(mp.read_text()).get("cell_id") == cell:
        found = mp.parent; break
if found is None:
    print(f"no store for cell {cell} under {root}", file=sys.stderr); sys.exit(1)
sp = found / "private" / "gens_gsm8k.jsonl"
if not sp.is_file():
    print(f"{found}: no gsm8k spool", file=sys.stderr); sys.exit(1)
head = json.loads(sp.read_text().splitlines()[0])
hdr = head.get("__spool_header__") or head
sha, n = hdr.get("items_sha256"), hdr.get("n_items")
if sha != want_sha or int(n or -1) != want_n:
    print(f"{sp}: items_sha256={sha} n_items={n} != the greedy baseline's {want_sha} / {want_n}", file=sys.stderr); sys.exit(1)
gs = json.loads((found / "gen_stats.json").read_text()).get("gsm8k") or {}
print(f"gsm8k items OK for {cell}: n={n} items_sha256={sha[:16]}... (== the greedy stores'); acc={gs.get('acc')} n_at_cap={gs.get('n_at_cap')} "
      f"thinking={gs.get('thinking')} arm={gs.get('arm')} verdict={(gs.get('thinking_regime') or {}).get('certification_verdict')} "
      f"finish={gs.get('finish_reasons')} thinking_tokens_sum={gs.get('thinking_tokens_sum')} answer_tokens_sum={gs.get('answer_tokens_sum')}")
PY
}

# --- Nemotron: the FUSED Mamba2 kernels of the SERVED path must be present, or the job refuses --------------------------
# This run generates through vLLM (the served/lens path). vLLM's NemotronH does NOT import the `mamba_ssm` /
# `causal_conv1d` pip packages (zero such imports under vllm/model_executor/layers/mamba/ and models/nemotron_h.py at
# the pinned 0.25.1); its fused kernels are vLLM's own Triton ops (ops.ssd_combined = the Mamba2 chunked scan,
# ops.causal_conv1d) plus the compiled selective_scan_fwd custom op. Those are what must be importable in the
# SERVING venv on THIS node — node-local venvs differ per node, so this is asserted inside the job, never assumed
# from the login node. No HF/transformers NemotronH model is instantiated anywhere in this run (the drivers load a
# tokenizer only), so the transformers pure-torch SSM fallback has no code path here; the hybrid-cache attestation
# (f22_model_facts: mamba_ssm_cache_dtype) is the second, live half of the same guarantee.
f22s_assert_vllm_mamba_kernels(){   # needs SPY (the serving venv python), KEY; a no-op for non-Mamba targets
  case "$KEY" in nemotron3nano4b) ;; *) return 0;; esac
  local rep
  rep=$("$SPY" - <<'PYK'
import importlib, sys
mods = ["vllm.model_executor.models.nemotron_h", "vllm.model_executor.layers.mamba.mamba_mixer2",
        "vllm.model_executor.layers.mamba.ops.ssd_combined", "vllm.model_executor.layers.mamba.ops.causal_conv1d"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:  # noqa: BLE001 - reported, then refused
        bad.append(f"{m}: {type(e).__name__}: {str(e)[:160]}")
try:
    import vllm._custom_ops as o
    if "selective_scan_fwd" not in dir(o):
        bad.append("vllm._custom_ops.selective_scan_fwd: absent")
except Exception as e:  # noqa: BLE001
    bad.append(f"vllm._custom_ops: {type(e).__name__}: {str(e)[:160]}")
import vllm
print(f"vllm {vllm.__version__}; fused mamba ops: " + ("ALL PRESENT" if not bad else "MISSING -> " + " | ".join(bad)))
sys.exit(1 if bad else 0)
PYK
  ) || { f22_log "REFUSING: nemotron3nano4b's served path lacks vLLM's fused Mamba2 kernels on $(hostname): $rep"; return 1; }
  f22_log "nemotron fused-kernel check ($(hostname)): $rep"
}

# --- LOW-DOSE NUMERIC LIVENESS — one record per steered cell per standup (lora_liveness.py), for that cell's processes ---
# At a small enough dose a CORRECT object changes no greedy token on a neutral probe, so the wave's steering_bites TEXT probe
# (fidelity._check_steering_bites, run by the smoke and inside every driver) refuses a steer that is live and merely SMALL —
# cluster A job 497871: constv2-mix-full__lora_r4_ce_minus_ctl at nominal 0.1 read 0/5 differing with salt isolation and the
# intervention hashes both PASS, and the ladder refused. At low dose liveness is NUMERIC (2026-09-11).
# After every `target up` and BEFORE the smoke and the drivers, the writer measures the two numeric facts per steered cell
# of this job (never `none`, never a prompted__ baseline: no object, no scale-0 twin) against THIS standup's proxy and
# routing table, into <standup dir>/liveness_low_dose_<cell>.json (the record binds to that routing table; a record from
# another standup admits nothing), and ONE line is logged per cell. A record that reads live_numeric (the writer's exit 0)
# is handed to THAT cell's processes ONLY — the smoke when it probes this cell, and every driver of the cell at every rung
# and shard — as `env SISTEER_LOW_DOSE_LIVENESS_RECORD=<record>` in front of the interpreter INSIDE the f22_queue_launch
# argument list (f22s_launch_cell); it is never exported to the job, never given to another cell. not_live (exit 2),
# unevaluable (exit 3), a writer that failed, or no writer at this sha => nothing is handed over and every process is
# byte-identical to before: the text-divergence gate stands. The policy the consumers apply is
# scripts/sdfbase/full22/low_dose_liveness_policy.py (verdict live_numeric, THIS cell, THIS routing table).
# The low-dose liveness writer and the record-passing launcher now live in common.sh, so the
# ATTACKS leg gets them too (they were sampled-only, and the attacks leg's bare text gate refused
# a live-but-small steer). Names and behaviour are unchanged here.

# --- the steer-is-live positive control on the first OBJECT cell of this job (common.sh f22_positive_control) --------
# The numeric liveness records are written FIRST (every steered cell of this job, this standup), so the smoke — which
# probes one steered cell and consults the same policy on its steering_bites failure — gets THAT cell's record, in its own
# process only (a subshell export; the job shell never carries the variable).
f22s_positive_control(){   # $1 = phase tag -> 0 PASS, 1 FAIL (logged); a job with no object cell records the skip
  local pc
  f22s_low_dose_liveness
  for pc in "${CELLS[@]}"; do
    case "$pc" in none|prompted__*) continue;; esac   # a prompted baseline has no scale-0 twin
    if [ -n "${F22S_LIVENESS_RECORD[$pc]:-}" ]; then
      ( export SISTEER_LOW_DOSE_LIVENESS_RECORD="${F22S_LIVENESS_RECORD[$pc]}"; f22_positive_control "$pc" "$1" ); return $?
    fi
    f22_positive_control "$pc" "$1"; return $?
  done
  f22_log "positive control ($1): no object cell among ${CELLS[*]} — nothing steered to prove; skipped"; return 0
}

# --- the PROMPTED cells' SI TEXT IDENTITY, on the record in FULL (the constitution-SI arm, 2026-09-12) ------------------
# The proxy verifies every prompted request's system prefix against the routing table's `prompted_prefixes`
# (H8: byte_length + sha256_hex16 — 16 hex chars, the manifest's prompted_prefix_hashes vocabulary), so a wrong
# prefix can never be served; but a reader of the stores sees only that 16-char prefix. Two lanes generating the
# SAME arm must be able to prove the two texts identical without recomputing anything, so after every standup
# this writes <standup dir>/si_prefix_record.json: per prompted cell of THIS job, the SI bytes resolved through
# the SAME resolver the standup and the drivers use (registry_adapter._traintest_si_text_bytes -> TrainTest's
# resolve_instruction), their FULL sha256, byte length, first / last line, and the routing table's (byte_length,
# sha16) for that cell's virtual model — REFUSING TRAFFIC (return 1) when the two disagree, or when the table
# names no prefix for a prompted cell. Nothing here touches src/sisteer/capeval/serving/ (pinned by the
# manifests' serving_impl_commit); the record is provenance beside routing_table.json and the mirror carries
# every *.json of a serving dir to the grading host.
f22s_record_si_prefix(){   # $1 = the standup dir (holds routing_table.json) -> writes $1/si_prefix_record.json; 1 = refuse
  local out="$1" rec="$1/si_prefix_record.json"
  [ -f "$out/routing_table.json" ] || { f22_log "REFUSING TRAFFIC: no routing_table.json under $out to check the SI prefixes against"; return 1; }
  (cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" HF_HUB_OFFLINE=1 "$SPY" - "$out/routing_table.json" "$rec" "${SLURM_JOB_ID:-manual}" "$WAVE_SHA" "${CELLS[@]}" <<'PY'
import hashlib, json, sys, datetime
table_path, rec_path, job, sha, *cells = sys.argv[1:]
from sisteer.capeval.petri import registry_adapter as ra
table = json.load(open(table_path))
by_vm = {p["virtual_model"]: p for p in table.get("prompted_prefixes", [])}
routes = {r["cell_id"]: r for r in table.get("routes", [])}
records, problems = [], []
for cell in cells:
    if not cell.startswith("prompted__"):
        continue
    si_id = cell[len("prompted__"):]
    harness = ra.harness_cell_id(cell)                       # prompted__si_constitution -> prompted__constitution
    data = bytes(ra._traintest_si_text_bytes(si_id))         # the resolver the standup + the drivers share
    full = hashlib.sha256(data).hexdigest()
    lines = data.decode("utf-8").splitlines()
    route = routes.get(harness)
    vm = route.get("virtual_model") if route else None
    spec = by_vm.get(vm) if vm else None
    r = {"cell_id": cell, "si_id": si_id, "harness_cell_id": harness, "virtual_model": vm,
         "byte_length": len(data), "sha256": full, "sha256_hex16": full[:16],
         "first_line": lines[0] if lines else "", "last_line": lines[-1] if lines else "", "n_lines": len(lines),
         "resolver": "sisteer.capeval.petri.registry_adapter._traintest_si_text_bytes (TrainTest resolve_instruction)",
         "routing_table": {"si_prefix_hash": route.get("si_prefix_hash") if route else None,
                           "prompted_prefix": spec},
         "verified_per_request": "H8: the proxy hashes every request's leading system text over byte_length and refuses a mismatch (si_prefix_mismatch / si_prefix_missing)"}
    if route is None:
        problems.append(f"{cell}: the routing table has no route {harness!r}")
    elif spec is None:
        problems.append(f"{cell}: the routing table names no prompted_prefix for virtual model {vm!r}")
    elif spec["sha16"] != full[:16] or int(spec["byte_length"]) != len(data) or route.get("si_prefix_hash") != full[:16]:
        problems.append(f"{cell}: resolver bytes sha256 {full} ({len(data)} B) vs routing table sha16 {spec['sha16']} / {spec['byte_length']} B / route {route.get('si_prefix_hash')}")
    records.append(r)
doc = {"job_id": job, "wave_sha": sha, "written_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
       "routing_table": table_path, "cells": records, "problems": problems, "ok": not problems}
json.dump(doc, open(rec_path, "w"), indent=1, ensure_ascii=False)
for r in records:
    print(f"si_prefix {r['cell_id']} -> {r['harness_cell_id']} ({r['virtual_model']}): sha256 {r['sha256']} {r['byte_length']} B; first {r['first_line'][:60]!r}; last {r['last_line'][-60:]!r}; table sha16 {(r['routing_table']['prompted_prefix'] or {}).get('sha16')}")
if problems:
    print("PROBLEMS: " + " | ".join(problems)); sys.exit(1)
PY
  ) > "$out/si_prefix_record.log" 2>&1
  local rc=$?
  f22_log "SI prefix record ($out): $(grep -v '^$' "$out/si_prefix_record.log" | tail -4 | tr '\n' ' ')"
  [ "$rc" = 0 ] || { f22_log "REFUSING TRAFFIC: the SI prefix record disagrees with the routing table (see $out/si_prefix_record.log)"; return 1; }
  return 0
}

# --- the sampler the ENGINE resolved (--enable-log-requests + scripts/sdfbase/record_resolved_sampler.py) ---------------
# target_sampled.sbatch opts the standup in (f22_standup_extra_reset --enable-log-requests); after a phase's drivers finish,
# this reads the standup's vllm_server.log back into <standup dir>/resolved_sampler.json against the rows the clients
# recorded ($2.. = responses.jsonl / gen_stats.json paths). A FINDING, never a job failure (the README's rule: do not
# `|| exit` on it); a missing log is logged and skipped.
f22s_record_resolved_sampler(){   # $1 = the standup dir, $2 = static|controls, $3.. = the --sent-* files
  local out="$1" kind="$2"; shift 2
  local log="$out/vllm_server.log" args=() f
  [ -f "$log" ] || { f22_log "resolved sampler ($out): no vllm_server.log — skipped"; return 0; }
  for f in "$@"; do [ -f "$f" ] && args+=("--sent-$kind" "$f"); done
  [ "$kind" = controls ] && args+=(--probe-cap 48 --probe-cap 1 --probe-cap 256 --client-cap 512)
  # The 8192 rung's false DIVERGED (2026-10-02): the standup's own harness probes — the representative load check (S greedy requests
  # at the CLIENT cap, by design) and the positive control — are TAGGED at the source (X-Tinycap-Task: probe/<kind>) and each emitter writes
  # <stem>.request_evidence.jsonl beside its record; the static read hands every sidecar to the reader (--probe-expect kinds --probe-tags …),
  # so the load check is carved from the client class by its tag instead of being read as six greedy client rows. The controls read keeps
  # its declared caps (its standup_controls probes are untagged by design).
  if [ "$kind" = static ]; then
    local tags=() s first="" ref=""
    for s in "$out"/*.request_evidence.jsonl; do [ -f "$s" ] && tags+=(--probe-tags "$s"); done
    # this file's own positive control (f22s_positive_control <phase>) writes its record and sidecar one level UP, named for the phase
    s="$(dirname "$out")/positive_control_$(basename "$out").request_evidence.jsonl"; [ -f "$s" ] && tags+=(--probe-tags "$s")
    for s in "$@"; do [ -f "$s" ] && { first="$s"; break; }; done
    if [ ${#tags[@]} -gt 0 ]; then
      # kinds mode needs the CLIENT reference as --expect k=v: the sent rows' recorded sampler — the same reference the untagged read derives
      [ -n "$first" ] && ref="$("$SPY" - "$first" <<'PY'
import json, sys
keys = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty")
for line in open(sys.argv[1], encoding="utf-8"):
    line = line.strip()
    if not line:
        continue
    d = json.loads(line).get("decode") or {}
    print(" ".join(f"--expect {k}={d[k]}" for k in keys if k in d))
    break
PY
)"
      if [ -n "$ref" ]; then
        # shellcheck disable=SC2206
        args+=(--probe-expect kinds $ref "${tags[@]}")
      else
        f22_log "resolved sampler ($out): ${#tags[@]} request_evidence sidecar(s) present but no client reference in the sent rows — reading without the tags (the load check reads as client rows)"
      fi
    fi
  fi
  (cd "$REPO_DIR" && "$SPY" scripts/sdfbase/record_resolved_sampler.py --vllm-log "$log" "${args[@]}" --out "$out/resolved_sampler.json") \
    > "$out/resolved_sampler.log" 2>&1
  f22_log "resolved sampler ($out, rc=$?): $(grep -i -m3 'verdict\|resolved\|DIVERGED\|AGREE' "$out/resolved_sampler.log" | tr '\n' ' ' | head -c 600)"
  return 0
}
