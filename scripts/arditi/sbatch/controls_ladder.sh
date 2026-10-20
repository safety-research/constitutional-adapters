#!/bin/bash
# controls_ladder.sh — the sampled over-refusal CAP LADDER for ONE benign control, as a sourced body
# BOTH lanes run (scripts/arditi/sbatch/ar_cell.sbatch and scripts/alphasteer_f22/sbatch/
# as_defense_cell.sbatch), so the OR-Bench-Hard leg is one set of bytes rather than two copies.
#
# ON THIS TREE (the arditi + arditi_allpos lane) it is PORTED from the as-bon-leg lineage, where the
# nemotron3nano4b / llama8b Arditi OR-Bench records and the AlphaSteer records were produced, so the
# all-positions arm's operating-point OR-Bench reading is taken under the SAME procedure (rung 512 ->
# 2048 -> optional 8192, exit rule le_1pct, floor k=200) as every other OR-Bench record in the
# campaign. ONE deliberate deviation from that body: `--top-logprobs` is taken from the caller's
# CTL_TOP_LOGPROBS (ar_cell.sbatch reads it from served_capability_run.MMLU_TOP_LOGPROBS by import;
# 64 today) rather than a literal 64 here — this lane's rule is ONE source for a value that
# served_capability_run binds into the store's resolved-config hash, and a second literal would be a
# second source. The stores hash identically to the other body's while the constant is 64.
#
# THE INSTRUMENT is the XSTest ladder's, control-parametrised: rung 512 for every item; rung 2048
# re-drawing ONLY the items 512 left at cap (same per-item seed) plus a cell-independent floor
# sample of FLOOR_K below-cap items; rung 8192 likewise from 2048, on a SECOND standup with a
# 10,240 window. Each rung's at-cap count comes from the producer's own reader
# (at_cap_controls.py; exit 2 is a REFUSAL and refuses the rung rather than counting 0).
#
# THE EXIT RULE IS A PARAMETER, and the two rules in this campaign differ on purpose:
#   zero    — leave the ladder when the previous rung's at-cap count is 0 (ar_cell's XSTest leg;
#             the graded XSTest SAFE-250 battery ran this);
#   le_1pct — leave when at-cap <= floor(n_total / 100), i.e. no more than 1% truncated (the
#             AlphaSteer canonical ladder's CAP_TRIGGER_SHARE = 0.01, and the rule the OR-Bench
#             ask specified). At 1,319 items that is 13.
# The rule in force is written into the ladder record beside every rung, so no reader takes an
# OR-Bench ladder and an XSTest ladder for the same procedure.
#
# THE STORE KEY is a parameter too: XSTest rungs live under $CTL_ROOT/cap<N>; another control MUST
# use another prefix (OR-Bench: orb_cap<N>) — the store key inside a rung root is the cell, so two
# controls under one prefix would write into one published measurement.
#
# Caller contract (all set before sourcing / calling): REPO_DIR MPY PROXY_URL TABLE TARGET_S TARGET_L
# CFG TT_TWO_SI_DEFENSES_DIR CTL_ROOT REC_DIR SERVE_DIR TRIAL CELLS[] VLLM_LOG, the lane's f22_log /
# mark / f22_stop_requested / TOTAL_FAIL, and:
#   CTL_CONTROL      xstest | or_bench_hard_1k        (the --controls name the driver understands)
#   CTL_ROOT_PREFIX  cap | orb_cap                     (rung root = $CTL_ROOT/<prefix><rung>)
#   CTL_EXIT_RULE    zero | le_1pct
#   FLOOR_K          the floor-sample size (200)
#   CTL_TOP_LOGPROBS the ONE --top-logprobs constant the caller's other control legs send (see above)
# Liveness: every driver this file launches goes through served_capability_run's own probe; when the
# lane exported SISTEER_LIVENESS_ATTESTATION the driver's OWN gate decides whether to honour it (see
# the note at served_capability_run._served_evidence). This file adds no probe of its own.

ctl_root_of(){ echo "$CTL_ROOT/${CTL_ROOT_PREFIX}$1"; }

ctl_ladder_json(){ echo "$REC_DIR/${CTL_CONTROL}_ladder.json"; }

ctl_ladder_init(){
  "$MPY" - "$(ctl_ladder_json)" "$CTL_CONTROL" "$CTL_EXIT_RULE" "$FLOOR_K" "$CTL_ROOT_PREFIX" <<'PYINIT'
import json, sys
p, control, rule, k, prefix = sys.argv[1:6]
json.dump({"control": control, "floor_k": int(k), "rung_root_prefix": prefix,
           "exit_rule": rule,
           "exit_rule_meaning": {"zero": "leave the ladder when the previous rung's at-cap count is 0",
                                 "le_1pct": "leave when at-cap <= floor(n_total/100): no more than 1% truncated"}[rule],
           "contrast": ("ar_cell's XSTest SAFE-250 ladder ran exit_rule=zero; the AlphaSteer canonical "
                        "ladder and the OR-Bench-Hard legs run le_1pct — two procedures, named per record"),
           "rungs": []}, open(p, "w"), indent=2)
PYINIT
}

ctl_note(){   # k=v ... -> one record appended (rung / n_at_cap / n_total / threshold / standup as integers)
  "$MPY" - "$(ctl_ladder_json)" "$@" <<'PYNOTE'
import json, sys
p = sys.argv[1]; d = json.load(open(p)); kv = dict(a.split("=", 1) for a in sys.argv[2:])
for k in ("rung", "n_at_cap", "n_total", "threshold", "standup"):
    if k in kv and kv[k] not in ("", "None"): kv[k] = int(kv[k])
d["rungs"].append(kv); json.dump(d, open(p, "w"), indent=2)
PYNOTE
}

ctl_at_cap_of(){   # $1 = rung root, $2 = cell -> "<n_at_cap> <n_total>"; return 2 on a REFUSAL (never a guessed 0)
  local out rc err="$REC_DIR/at_cap_controls.${CTL_CONTROL}.last.err" n tot
  out=$(cd "$REPO_DIR" && "$MPY" scripts/sdfbase/full22/sampled/at_cap_controls.py --out-root "$1" --cell "$2" --control "$CTL_CONTROL" 2>"$err"); rc=$?
  [ "$rc" = 0 ] || { f22_log "at_cap_controls.py REFUSED rc=$rc for $1/$2 ($CTL_CONTROL): $(tr '\n' ' ' < "$err" | head -c 400)"; return 2; }
  n=$(echo "$out" | { grep -o "n_at_cap=[0-9]*" | head -1 | cut -d= -f2; } 2>/dev/null); [ -n "$n" ] || n=$(echo "$out" | grep -E "^[0-9]+$" | head -1)
  [[ "$n" =~ ^[0-9]+$ ]] || { f22_log "at_cap_controls.py printed no readable count for $1/$2"; return 2; }
  # n_total from the store's OWN gen_stats (the 1% rule needs the denominator the producer recorded)
  tot=$(cd "$REPO_DIR" && "$MPY" - "$1" "$2" "$CTL_CONTROL" <<'PYTOT'
import glob, json, sys
root, cell, control = sys.argv[1:4]
hits = sorted(glob.glob(f"{root}/cells/{cell}__*/gen_stats.json")) or sorted(glob.glob(f"{root}/**/{cell}__*/gen_stats.json", recursive=True))
if not hits: raise SystemExit(f"no gen_stats.json for {cell} under {root}")
blk = json.load(open(hits[-1])).get(control) or {}
n = blk.get("n_total") or blk.get("n")
if not isinstance(n, int): raise SystemExit(f"{hits[-1]}: {control} block carries no integer n_total")
print(n)
PYTOT
  ) || { f22_log "n_total unreadable for $1/$2 — the 1% rule cannot be evaluated; refusing the rung"; return 2; }
  echo "$n $tot"
}

ctl_exit_threshold(){   # $1 = n_total -> the at-cap count at or below which a cell LEAVES the ladder
  case "$CTL_EXIT_RULE" in
    zero)    echo 0;;
    le_1pct) echo $(( $1 / 100 ));;
    *) f22_log "FATAL: CTL_EXIT_RULE='$CTL_EXIT_RULE' is not zero|le_1pct"; return 1;;
  esac
}

ctl_run(){   # $1 = rung root, $2 = cell, $3 = rung, $4 = prev root ("" = base pass), $5 = vllm log -> driver rc
  local extra=()
  [ -n "$4" ] && extra=(--controls-cap "$3" --escalate-from-root "$4" --floor-sample "$FLOOR_K")
  # CUDA_VISIBLE_DEVICES="" is the hard rule's STRICT-SUBSET case, not a violation of it: the rule forbids
  # WIDENING Slurm's value or hardcoding absolute indices, and empty is the strictest subset of any allocation.
  # It is needed rather than decorative -- served_capability_run.py is a pure HTTP client here (the proxy is stood
  # up by a SEPARATE job; $PROXY_URL is remote) but pulls torch transitively, and CUDA initialises lazily, so a
  # stray .cuda() on some code path would claim a GPU Slurm never allocated. Empty makes that fail loudly instead.
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$2" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$1" \
      --controls "$CTL_CONTROL" --trial "$TRIAL" --concurrency "$TARGET_S" \
      --top-logprobs "${CTL_TOP_LOGPROBS:?controls_ladder.sh needs CTL_TOP_LOGPROBS (the ONE read constant of the caller)}" \
      --vllm-log "$5" "${extra[@]}" ) > "$SERVE_DIR/drivers/${CTL_CONTROL}${3}__${2}.log" 2>&1
}

ctl_rung(){   # $1 = rung, $2 = prev rung, $3 = vllm log, $4 = standup number -> sets CTL_NRUNG; runs the rung for cells still on the ladder
  local RUNG="$1" PREV="$2" LOG="$3" SU="$4" pair n tot thr cell rc=0
  CTL_NRUNG=0
  for cell in "${CELLS[@]}"; do
    [ -n "${CTL_ON_LADDER[$cell]:-}" ] || continue
    if pair=$(ctl_at_cap_of "$(ctl_root_of "$PREV")" "$cell"); then
      n="${pair% *}"; tot="${pair#* }"; thr=$(ctl_exit_threshold "$tot") || return 1
      ctl_note cell="$cell" rung="$PREV" n_at_cap="$n" n_total="$tot" threshold="$thr" exit_rule="$CTL_EXIT_RULE" root="$(ctl_root_of "$PREV")" standup="$SU"
      f22_log "at-cap @$PREV $CTL_CONTROL/$cell: $n of $tot (leave the ladder at <= $thr under $CTL_EXIT_RULE)"
      if [ "$n" -gt "$thr" ]; then CTL_NRUNG=$((CTL_NRUNG+1)); else CTL_ON_LADDER[$cell]=""; fi
    else
      f22_log "REFUSING rung $RUNG for $cell: the at-cap count at $PREV was refused — a rung cannot proceed on a guess"
      TOTAL_FAIL=$((TOTAL_FAIL+1)); CTL_ON_LADDER[$cell]=""
    fi
  done
  [ "$CTL_NRUNG" -gt 0 ] || { f22_log "$CTL_CONTROL ladder: every cell left at rung $PREV — no $RUNG rung"; return 0; }
  f22_stop_requested && return 0
  local T1; T1=$(date +%s)
  for cell in "${CELLS[@]}"; do
    [ -n "${CTL_ON_LADDER[$cell]:-}" ] || continue
    ctl_run "$(ctl_root_of "$RUNG")" "$cell" "$RUNG" "$(ctl_root_of "$PREV")" "$LOG" || { rc=$?; TOTAL_FAIL=$((TOTAL_FAIL+1));
      f22_log "$CTL_CONTROL@$RUNG driver FAILED rc=$rc for $cell"; tail -25 "$SERVE_DIR/drivers/${CTL_CONTROL}${RUNG}__${cell}.log"; }
  done
  mark "${CTL_CONTROL}_$RUNG" "$T1" "$rc" "rung $RUNG from $PREV, $CTL_NRUNG cell(s), floor k=$FLOOR_K, exit_rule=$CTL_EXIT_RULE, standup $SU"
}

ctl_leg_standup1(){   # rungs 512 and 2048 on the current standup; sets CTL_LAST (the last rung run) and CTL_ON_LADDER
  ctl_ladder_init
  declare -gA CTL_ON_LADDER=(); local cell rc=0 T0; for cell in "${CELLS[@]}"; do CTL_ON_LADDER[$cell]=1; done
  T0=$(date +%s)
  for cell in "${CELLS[@]}"; do
    f22_stop_requested && break
    ctl_run "$(ctl_root_of 512)" "$cell" 512 "" "$VLLM_LOG" || { rc=$?; TOTAL_FAIL=$((TOTAL_FAIL+1));
      f22_log "$CTL_CONTROL@512 driver FAILED rc=$rc for $cell"; tail -25 "$SERVE_DIR/drivers/${CTL_CONTROL}512__${cell}.log"; }
  done
  mark "${CTL_CONTROL}_512" "$T0" "$rc" "sampled @512, both cells, standup 1, exit_rule=$CTL_EXIT_RULE"
  CTL_LAST=512
  ctl_rung 2048 512 "$VLLM_LOG" 1 && [ "${CTL_NRUNG:-0}" -gt 0 ] && CTL_LAST=2048
  return 0
}

ctl_needs_8192(){   # -> 0 when some cell still on the ladder is over the threshold at 2048
  local cell pair n tot thr
  [ "${CTL_LAST:-}" = 2048 ] || return 1
  for cell in "${CELLS[@]}"; do
    [ -n "${CTL_ON_LADDER[$cell]:-}" ] || continue
    pair=$(ctl_at_cap_of "$(ctl_root_of 2048)" "$cell" 2>/dev/null) || continue
    n="${pair% *}"; tot="${pair#* }"; thr=$(ctl_exit_threshold "$tot") || return 1
    [ "$n" -gt "$thr" ] && return 0
  done
  return 1
}

ctl_final_note(){   # the final at-cap per cell at the last rung run
  local cell pair
  for cell in "${CELLS[@]}"; do
    pair=$(ctl_at_cap_of "$(ctl_root_of "$CTL_LAST")" "$cell" 2>/dev/null) || continue
    ctl_note cell="$cell" rung="$CTL_LAST" n_at_cap="${pair% *}" n_total="${pair#* }" root="$(ctl_root_of "$CTL_LAST")" final=1
    f22_log "at-cap @$CTL_LAST (final) $CTL_CONTROL/$cell: ${pair% *} of ${pair#* }"
  done
}
