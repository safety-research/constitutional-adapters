#!/bin/bash
#SBATCH --job-name=D_f22smoke
#SBATCH --partition=dev,overflow
#SBATCH --qos=dev
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=0-03:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22 SMOKE (README section 5 step 6): ONE GPU, ONE steered cell + the `none` anchor, a SCRATCH
# private root ($WAVE_ROOT/smoke/<tag>_<jobid>/ — never the wave's transcripts_results), and every
# untested seam of the wave exercised end-to-end, gating the fan-out:
#   * the standup ATTESTS the wave's ENGINE — lens_reduction_mode=stacked under F22_SERVING_PATH=lens, engine
#     lora_native (graphs on, the adapters bound by name) under the native default — and the resolved
#     max_num_seqs x max_model_len (f22_standup refuses otherwise) at the static geometry (families_served:short);
#   * the STEER-IS-LIVE positive control (smoke_positive_control.py: proxy identity mode, salt
#     isolation, steering-bites vs the scale-0 twin, intervention hashes, greedy divergence);
#   * static  — one pasttense shard (1 goal: --nshards 259 --shard 0) per cell, served target;
#   * bon     — bon_served_run.py, the fan-out driver that has never run on hardware, --limit-goals 2;
#   * controls— the controls geometry standup (2048 x budget-S, identity, the driver's logprob cap) +
#              served_capability_run.py --limit 8 on the VIEW's roster controls list under the wave's interpreter for that
#              list (the client venv for a gsm8k-bearing view, the driver venv for llama's legacy-only list) — the wave's
#              controls path for the view named, interpreter included.
#   * multiturn — the attacks geometry standup + one crescendo shard per cell against the LIVE
#     attacker pool (run this leg on the cluster that hosts the pool).
#   * output_diagnostics_<phase> (native-psr-serving plan step 6; the NATIVE engine only): after each successful standup
#     (static, controls, attacks) its cells_served.json must record output_diagnostics=false iff any served adapter's
#     petri_source_kind is a residual-slot kind (defense_cells.RESIDUAL_SLOT_KINDS: psr / vector / vector_all / diffmean /
#     static_r — the central no-numerics policy) and true for a LoRA-only standup; the served kinds ride the detail.
# Every check lands in <smoke root>/checks.jsonl; the job exits 0 only if ALL pass. Dev QoS by
# default (srun-style `D_` name so the nightly reaper may clean it); the submit line may override.
# Usage (positional only — never --export=):
#   sbatch [-p dev,overflow --qos=dev] scripts/sdfbase/full22/smoke_full22.sbatch \
#       <key> <view> <wave_sha> <steered_cell> [legs=static,bon,controls]   (legs: any of
#       static,bon,controls,multiturn as a comma list)
#   THE ROSTER (native-psr-serving D15): F22_ROSTER (common.sh; default configs/sdfbase_full22_roster.yaml) selects the
#   roster EVERY reader of this job reads (roster_view.py here, make_bank_view.py / roster_cell_count.py in f22_bank_view).
#   llama8b's CERTIFICATION-ONLY `psrcert` view (the staged all-positions PSR cell + its LoRA twin; no production view of
#   llama8b carries a PSR cell) lives in configs/sdfbase_full22_roster_psrcert.yaml, so its smoke exports the roster on the
#   submit line (a batch job inherits the submitter's exported environment) and never edits the production roster:
#     F22_ROSTER=configs/sdfbase_full22_roster_psrcert.yaml sbatch scripts/sdfbase/full22/smoke_full22.sbatch \
#         llama8b psrcert <wave_sha> constv2-mix-full__psr_ce_all static,controls
#   Without it the job refuses at f22_bank_view's roster count guard (the production roster names no llama8b/psrcert).
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; WAVE_SHA="${3:?wave sha}"; CELL="${4:?steered cell}"; LEGS="${5:-static,bon,controls}"
# THE CONTROLS OVERRIDE (native-psr-serving fallback, 2026-09-15): an optional 6th positional REPLACES the view's
# roster controls list for the controls leg (e.g. `mmlu,xstest,or_bench_hard_1k` to drop the gsm8k Inspect control whose
# 1,537-token prompts overflow main's 2048 controls window — main's debt, since retired upstream). It changes the
# VALUE the driver's `--controls` receives and nothing else: the driver argv stays the wave's, byte for byte
# (tests/test_sdfbase_full22_wave.py pins the two argvs equal), so a driver policy the wave itself would refuse under
# (a model's unverified sampling-card row) is NOT worked around here — that is the wave's debt, recorded, not smoked over.
# Recorded in the controls check's detail; empty = the wave's roster list.
CTL_CONTROLS_OVERRIDE="${6:-}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_MANIFEST=$(f22_full22_manifest "$KEY" "$VIEW")   # the launch's cells for the kind-aware serving-path default (2026-09-15): the (KEY, VIEW) campaign manifest
f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"
f22_stop_requested && { f22_log "wave STOP present — not smoking"; exit 0; }
SMOKE_ROOT="$WAVE_ROOT/smoke/${TAG}_${SLURM_JOB_ID:-manual}"; SERVE_DIR="$SMOKE_ROOT/serving"
PRIVATE_ROOT="$SMOKE_ROOT/transcripts_results"; CHECKS="$SMOKE_ROOT/checks.jsonl"
mkdir -p "$SERVE_DIR/drivers" "$PRIVATE_ROOT"; chmod 700 "$PRIVATE_ROOT"; : > "$CHECKS"
NFAILED=0
check(){   # name PASS|FAIL detail...
  local name="$1" st="$2"; shift 2
  [ "$st" = PASS ] || NFAILED=$((NFAILED+1))
  printf '{"check":"%s","status":"%s","detail":"%s","at":"%s","job":"%s"}\n' "$name" "$st" "$(echo "$*" | tr '"' "'" | tr '\n' ' ')" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-manual}" >> "$CHECKS"
  f22_log "CHECK $st $name: $*"
}
has_leg(){ case ",$LEGS," in *",$1,"*) return 0;; *) return 1;; esac; }
# THE CONTROLS LIST IS THE VIEW'S (it supersedes the earlier fixed four-control superset, which carried the Inspect control
# gsm8k on every view so the Inspect path was always exercised): read through roster_view.py exactly as the wave reads it
# (target_static.sbatch VIEW_CONTROLS — f22_roster.view_controls, the reader the generator and the printers share), before
# any venv bootstrap, so the smoke tests THE WAVE PATH FOR THE VIEW IT NAMES and nothing else. Since 2026-09-14
# EVERY target's list is legacy-only (mmlu / xstest / or_bench_hard_1k — the driver's own SERVED_CONTROLS, no framework):
# the wave's controls-leg gsm8k is RETIRED wave-wide (roster controls_retired — f22_roster.view_controls drops it from every
# list; the paired-200 lane is the paper's one GSM8K instrument; llama8b's own capability_omitted stays), so the Inspect
# client path below is dormant until a roster drops the retirement. WHY NOT THE SUPERSET: smoke 503471
# (llama8b it, native) ran the superset's three legacy controls to their stores and then the Inspect gsm8k control
# OVERFLOWED THE CONTROLS WINDOW — prompts up to 1,537 tokens + max_tokens 512 > capability:short 2048, 34 of 200 sample
# ids refused — a SIZING defect of main's wave for the gsm8k-bearing views (the production wave stood its controls phase
# up at VIEW_WINDOW_CONTROLS, default 2048, for those views too; recorded as wave debt — PAID 2026-09-14: a gsm8k-bearing
# table view reads the roster's controls_window.gsm8k, 4,608, through roster_view.py `window gsm8k` exactly as the wave does,
# and common.sh f22_gsm8k_window_check refuses a window under the derived minimum by name before the standup). The native
# Inspect gsm8k control is proven on Modal (J4, 16,384 window); a control the named view never runs has no place in its
# smoke, and a smoke that names a gsm8k-bearing view meets the wave's own window there, which is the point.
VIEW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" controls --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's controls list for $KEY/$VIEW"
if [ -n "$CTL_CONTROLS_OVERRIDE" ]; then
  f22_log "controls list OVERRIDDEN for this smoke: roster [$VIEW_CONTROLS] -> [$CTL_CONTROLS_OVERRIDE] (D16 fallback; the wave's list is unchanged)"
  VIEW_CONTROLS="$CTL_CONTROLS_OVERRIDE"
fi
VIEW_WINDOW_CONTROLS=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window controls --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's controls window for $KEY/$VIEW"
VIEW_WINDOW_GSM8K=$(python3 scripts/sdfbase/full22/roster_view.py "$KEY" "$VIEW" window gsm8k --roster "$F22_ROSTER") || f22_fatal "cannot read the roster's gsm8k controls window for $KEY/$VIEW"
f22_driver_venv; f22_serving_venv; f22_bank_view "$MPY"
# THE CONTROLS DRIVER'S INTERPRETER FOLLOWS THE VIEW'S CONTROLS LIST — the wave's gate (target_static.sbatch), byte for byte
# from the CTL_INSPECT read down. An INSPECT control (gsm8k; contracts.INSPECT_CONTROLS, read by import through common.sh
# f22_inspect_controls under $MPY, never a hand-copied list) needs inspect_ai + inspect_evals at the client lock's pins,
# which the DRIVER venv ($MPY: uv sync --extra cu128) does not carry — under it the driver refuses the whole leg by name
# ("framework 'inspect_ai' is required by a requested control but cannot be imported", smoke 502126). So the CLIENT venv
# (f22_client_venv -> $CPY: the lock the Modal launcher builds /opt/client from and every gsm8k / math500 lane job runs;
# node-local, hash-versioned, flock-guarded) is bootstrapped and used exactly when the view's list intersects the Inspect
# vocabulary, and the Inspect leg then reads the `datasets` library OFFLINE from the per-account prepared cache
# (f22_datasets_guard: HF_DATASETS_CACHE=$SHARED/hf_datasets, the gsm8k jobs' rule — smoke 502566 failed on the shared
# model cache, which holds openai/gsm8k under other load signatures only; guarded by name here, and HF_DATASETS_CACHE +
# HF_DATASETS_OFFLINE=1 ride the driver's env prefix as CTL_ENV). A LEGACY-ONLY list (the llama8b views) runs under $MPY:
# no 259-package client lock on a fresh node for a job that never imports Inspect. Decided HERE, beside the other venvs,
# only when the controls leg is asked for (the static / bon / multiturn legs never touch it) and before any GPU standup, so
# an unresolvable lock or a missing cache fails the job in its first minutes. A view the roster gives no controls (neither
# a capability nor a controls leg) is refused by name when the leg is asked for — the wave's refusal at its family parse.
CTL_PY="$MPY"; CTL_INSPECT=""; CTL_ENV=()
if has_leg controls; then
  [ -n "$VIEW_CONTROLS" ] || f22_fatal "view $KEY/$VIEW has neither a capability nor a controls leg (the roster names no controls for it): nothing for the smoke's controls leg to run — drop controls from legs="
  CTL_INSPECT=$(f22_inspect_controls "$VIEW_CONTROLS") || f22_fatal "cannot read contracts.INSPECT_CONTROLS under $MPY"
  if [ -n "$CTL_INSPECT" ]; then
    f22_client_venv; CTL_PY="$CPY"; f22_datasets_guard "$CTL_INSPECT"
    CTL_ENV=(HF_DATASETS_CACHE="$HF_DATASETS_CACHE" HF_DATASETS_OFFLINE=1)
    f22_log "phase CONTROLS interpreter: the client venv $CTL_PY (Inspect controls in the view's list: $CTL_INSPECT; datasets cache $HF_DATASETS_CACHE)"
  else
    f22_log "phase CONTROLS interpreter: the driver venv $CTL_PY (legacy controls only: $VIEW_CONTROLS; no Inspect framework, no client venv)"
  fi
fi
case " ${ALL_CELLS[*]} " in *" $CELL "*) ;; *) f22_fatal "$CELL is not in $CFG explicit_cells (${ALL_CELLS[*]})";; esac
CELLS=(none "$CELL")
TARGET_PID=""; STANDUP_STOP=""; declare -a DRIVER_PIDS=(); declare -A DRIVER_TAG=(); RC=1
cleanup(){
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  for p in "${DRIVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true; done
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  cp "$CHECKS" "$FIGS_DIR/smoke_checks_${TAG}_${SLURM_JOB_ID:-manual}.jsonl" 2>/dev/null || true
  f22_result "smoke $TAG legs=$LEGS cell=$CELL rc=${RC} failed_checks=${NFAILED}"
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
TARGET_GPU="$CUDA_VISIBLE_DEVICES"

pc_fail_detail(){   # $1 = the control's record JSON, $2 = its driver log -> ONE line: the FAILED controls FIRST, each with
  # its full payload, then the DIAGNOSTIC (recorded, not gating) ones, the passing names, and the record path — never a
  # truncated log tail (smoke 502002: the FAIL detail was `tail -5 | head -c 600` of the driver log and ended mid-JSON
  # inside a PASS entry, hiding the one failing control's name; the record keeps every probe and text, so it is read
  # whole). No record (the driver died before writing it): the last 20 log lines, whole.
  local rec="$1" log="$2" d
  if [ -f "$rec" ]; then
    d=$("$MPY" - "$rec" <<'PY' 2>&1
import json, sys
d = json.load(open(sys.argv[1]))
checks = d.get("checks", [])
DIAGNOSTIC = "DIAGNOSTIC"     # smoke_positive_control.DIAGNOSTIC_STATUS, pinned equal by test
def payload(c): return json.dumps({k: v for k, v in c.items() if k not in ("check", "status")}, default=str)
failed = [c for c in checks if c.get("status") not in ("PASS", DIAGNOSTIC)]
diag = [c for c in checks if c.get("status") == DIAGNOSTIC]
parts = ["FAILED " + (", ".join(f"{c.get('check')}={payload(c)}" for c in failed) if failed
                      else "no check FAILED, the driver exited non-zero (all_pass=%r) — read the log" % d.get("all_pass"))]
if diag:
    parts.append("DIAGNOSTIC (recorded, not gating) " + ", ".join(f"{c.get('check')}={payload(c)}" for c in diag))
parts.append("PASS " + ",".join(str(c.get("check")) for c in checks if c.get("status") == "PASS"))
parts.append(f"record={sys.argv[1]}")
print(" | ".join(parts))
PY
) || d="record $rec unreadable ($d); log tail: $(tail -20 "$log" | tr '\n' ' ')"
  else
    d="no record JSON at $rec (the driver died before writing it); log tail: $(tail -20 "$log" | tr '\n' ' ')"
  fi
  printf '%s\n' "$d"
}
positive_control(){   # $1 = phase tag; needs PROXY_URL TABLE
  local out="$SMOKE_ROOT/positive_control_$1.json"
  if ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/sdfbase/full22/smoke_positive_control.py \
        --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$CELL" --out "$out" --require-engine "$TARGET_ENGINE" ) > "$SERVE_DIR/drivers/positive_control_$1.log" 2>&1; then
    check "positive_control_$1" PASS "$("$MPY" -c "import json,sys;d=json.load(open(sys.argv[1]));print(' | '.join(c['check']+'='+c['status'] for c in d['checks']), '| greedy:', [c for c in d['checks'] if c['check']=='greedy_divergence'][0])" "$out" 2>/dev/null | head -c 600)"
  else
    check "positive_control_$1" FAIL "$(pc_fail_detail "$out" "$SERVE_DIR/drivers/positive_control_$1.log")"
  fi
}
output_diagnostics_check(){   # $1 = phase tag; needs SERVE_DIR SPY REPO_DIR — the NATIVE engine only (the call sites gate on TARGET_ENGINE)
  # a native standup that serves ANY residual-slot kind
  # (defense_cells.RESIDUAL_SLOT_KINDS — read BY IMPORT from the checkout under $SPY, never a copied list) must record
  # output_diagnostics=false in its cells_served.json (the no-numerics policy: no output-agreement diagnostics on a
  # residual-bearing standup), and a LoRA-only standup output_diagnostics=true — the record the certification admits on
  # (multilora_cert.py) read back here PER PHASE from the standup's OWN record (the cells_served.json endpoint.json names, else
  # the standup dir's; the durable endpoint.recorded.json when the READY record is already gone). The served kinds are the
  # attested adapters' petri_source_kind (the zero adapter is recorded apart and is not a served cell). No record, a missing or
  # non-boolean key, an adapter with no kind, or an unimportable policy constant is a FAIL by name — never a default.
  local out="$SERVE_DIR/$1" ep d
  ep="$out/endpoint.json"; [ -f "$ep" ] || ep="$out/endpoint.recorded.json"
  if d=$(cd "$REPO_DIR" && "$SPY" - "$ep" "$out/cells_served.json" <<'PY' 2>&1
import json, os, sys
sys.path.insert(0, "src")
from sisteer.capeval.petri.defense_cells import RESIDUAL_SLOT_KINDS
ep_path, fallback = sys.argv[1], sys.argv[2]
ep = json.load(open(ep_path)) if os.path.isfile(ep_path) else {}
cs_path = ep.get("cells_served") or fallback
if not os.path.isfile(cs_path):
    sys.exit(f"no cells_served.json for this standup ({cs_path}; endpoint {ep_path})")
d = json.load(open(cs_path))
od = d.get("output_diagnostics")
adapters = d.get("adapters")
if not isinstance(adapters, dict):
    sys.exit(f"{cs_path}: no `adapters` block (a native standup records one per served lora cell)")
kinds = {}
for cell, block in adapters.items():
    k = block.get("petri_source_kind") if isinstance(block, dict) else None
    if not isinstance(k, str) or not k:
        sys.exit(f"{cs_path}: adapters[{cell!r}] carries no petri_source_kind")
    kinds[cell] = k
residual = sorted(c for c, k in kinds.items() if k in RESIDUAL_SLOT_KINDS)
want = not residual
detail = (f"output_diagnostics={json.dumps(od)} want={json.dumps(want)} served_kinds={json.dumps(kinds, sort_keys=True)} "
          f"residual_cells={residual} residual_kinds={list(RESIDUAL_SLOT_KINDS)} record={cs_path}")
if not isinstance(od, bool):
    sys.exit(f"{cs_path}: output_diagnostics is {od!r}, not a boolean — {detail}")
if od != want:
    sys.exit(f"output_diagnostics={json.dumps(od)} but the standup serves {'a residual-slot kind' if residual else 'LoRA kinds only'} — {detail}")
print(detail)
PY
  ); then check "output_diagnostics_$1" PASS "$d"; else check "output_diagnostics_$1" FAIL "$d"; fi
}
count_progress(){   # $1 attack $2 cell -> "unique_goals=N error_outcomes=M latest_error_or_refused=B forwards=F" (error_outcomes = every historical error row; B = goals whose LATEST row is error/refused)
  "$MPY" - "$PRIVATE_ROOT/$("$MPY" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['private_run'])" "$CFG")" "$1" "$2" <<'PY'
import json, sys
from pathlib import Path
root, atk, cell = Path(sys.argv[1]), sys.argv[2], sys.argv[3]
goals, errs, fwd, latest = set(), 0, 0, {}
for p in sorted((root / atk / cell).glob("shard*/progress.jsonl")):
    for line in p.read_text().splitlines():
        if line.strip():
            rec = json.loads(line); goals.add(rec.get("goal_id"))
            errs += str(rec.get("outcome", "")).startswith("error"); fwd += int(rec.get("n_target_forwards") or 0)
            latest[rec.get("goal_id")] = str(rec.get("outcome", ""))
bad = sum(1 for o in latest.values() if o.startswith(("error", "refused")))
print(f"unique_goals={len(goals)} error_outcomes={errs} latest_error_or_refused={bad} forwards={fwd}")
PY
}

# ------------------------------------------------------------------ static geometry: standup + control + static + bon
if has_leg static || has_leg bon; then
  if f22_standup attacks "$SERVE_DIR/static" "" families_served:short; then
    # the smoke's cell is an EXPLICIT ask: a cell the native engine skips by name is fatal here (F22_SERVING_PATH=lens serves it)
    f22_served_cells_filter CELLS "$CELL"
    check standup_static PASS "geometry=${TARGET_S}x${TARGET_L} attested from endpoint.json + /v1/models: $TARGET_ENGINE; table=$TABLE"
    [ "$TARGET_ENGINE" = lora_native ] && output_diagnostics_check static
    positive_control static
    if has_leg static; then
      for cell in "${CELLS[@]}"; do
        ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/jailbreak_run.py generate \
            --config "$CFG" --attack pasttense --defense "$cell" --confirmatory --shard 0 --nshards 259 \
            --private-root "$PRIVATE_ROOT" --target-engine served --target-proxy-url "$PROXY_URL" \
            --target-routing-table "$TABLE" --target-max-model-len "$TARGET_L" ) > "$SERVE_DIR/drivers/pasttense__${cell}.log" 2>&1 &
        pid=$!; DRIVER_PIDS+=("$pid"); DRIVER_TAG[$pid]="pasttense__${cell}"; sleep 2
      done
    fi
    if has_leg bon; then
      # the same three SCALAR declarations the wave job forwards
      # (target_static.sbatch) ride here when set, so the smoke's bon_served_run.json carries the view's preset + policy +
      # base; unset = this argv byte for byte as before
      for cell in "${CELLS[@]}"; do
        ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/sdfbase/full22/bon_served_run.py \
            --config "$CFG" --defense "$cell" --confirmatory --private-root "$PRIVATE_ROOT" \
            --proxy-url "$PROXY_URL" --routing-table "$TABLE" --max-model-len "$TARGET_L" \
            --max-concurrency 40 --limit-goals 2 --require-engine "$TARGET_ENGINE" \
            ${F22_BON_DECODE_PRESET:+--bon-decode-preset "$F22_BON_DECODE_PRESET"} \
            ${F22_BON_DECODE_SEED_POLICY:+--decode-seed-policy "$F22_BON_DECODE_SEED_POLICY"} \
            ${F22_BON_DECODE_SEED_BASE:+--decode-seed-base "$F22_BON_DECODE_SEED_BASE"} ) > "$SERVE_DIR/drivers/bon__${cell}.log" 2>&1 &
        pid=$!; DRIVER_PIDS+=("$pid"); DRIVER_TAG[$pid]="bon__${cell}"; sleep 2
      done
    fi
    for pid in "${DRIVER_PIDS[@]}"; do
      wait "$pid"; rc=$?; tag="${DRIVER_TAG[$pid]}"
      if [ "$rc" = 0 ]; then
        case "$tag" in
          pasttense__*) check "$tag" PASS "$(count_progress pasttense "${tag#pasttense__}") $(grep -o 'n_target_forwards_total=[0-9]*' "$SERVE_DIR/drivers/$tag.log" | tail -1)";;
          bon__*) cell="${tag#bon__}"; rd=$("$MPY" - "$CFG" "$PRIVATE_ROOT" "$cell" <<'PY'
import sys; sys.path.insert(0, "scripts"); import jailbreak_run as jr
cfg = jr._load_cfg(sys.argv[1]); print(jr._bon_defense_dir(cfg, jr._private_root(cfg, sys.argv[2]), sys.argv[3]))
PY
); n=$(wc -l < "$rd/bon_responses.jsonl" 2>/dev/null || echo 0); [ -f "$rd/bon_served_run.json" ] && m=manifest_ok || m=NO_MANIFEST
            if [ "$n" -ge 150 ] && [ "$m" = manifest_ok ]; then check "$tag" PASS "responses=$n (2 goals x n_max 75) $m $(grep -o 'n_target_forwards=[0-9]*' "$SERVE_DIR/drivers/$tag.log" | tail -1) dir=$rd"; else check "$tag" FAIL "responses=$n $m dir=$rd $(tail -3 "$SERVE_DIR/drivers/$tag.log" | tr '\n' ' ' | head -c 400)"; fi;;
        esac
      else
        check "$tag" FAIL "rc=$rc $(tail -8 "$SERVE_DIR/drivers/$tag.log" | tr '\n' ' ' | head -c 800)"
      fi
    done
    DRIVER_PIDS=()
  else
    check standup_static FAIL "$(tail -15 "$SERVE_DIR/static/standup.log" | tr '\n' ' ' | head -c 1200)"
  fi
  f22_standup_stop
fi
# ------------------------------------------------------------------ controls geometry
if has_leg controls; then
  # ONE number for the server's logprob cap AND the driver's request — the wave's controls phase exactly (target_static.sbatch):
  # the MMLU leg scores by the top-N logprobs and the engine refuses a request above its --max-logprobs (default 20) while the
  # driver asks its own MMLU_TOP_LOGPROBS (64) — the refusal that killed all 9 drivers of f22-ctl-gemma4e4b-main 447405. Read
  # the driver's constant under $MPY (an import of the module, which resolves there — the wave and J5's native_capctrl_smoke
  # read it the same way) and hand the SAME value to the standup (7th arg) and the driver (--top-logprobs). Main's smoke stood
  # the controls server up at the engine default and would have lost every MMLU request to that refusal the moment the venv
  # refusal above was out of the way. An unreadable constant is a recorded FAIL, never a silent default.
  CTL_TOP_LOGPROBS=$("$MPY" -c 'import importlib.util,sys; s=importlib.util.spec_from_file_location("scr", sys.argv[1]); m=importlib.util.module_from_spec(s); sys.modules["scr"]=m; s.loader.exec_module(m); print(int(m.MMLU_TOP_LOGPROBS))' scripts/served_capability_run.py) || CTL_TOP_LOGPROBS=""
  # the controls window, as the wave composes it (target_static.sbatch CTL_L): the view's own, else the roster's gsm8k window for a
  # gsm8k-bearing table view, else the table's 2,048; the gsm8k window check runs in a subshell so its refusal is a recorded FAIL
  CTL_L="${VIEW_WINDOW_CONTROLS:-${VIEW_WINDOW_GSM8K:-2048}}"
  if [ -z "$CTL_TOP_LOGPROBS" ]; then
    check standup_controls FAIL "cannot read MMLU_TOP_LOGPROBS from scripts/served_capability_run.py under $MPY (the wave's controls phase reads it the same way); the controls standup was not attempted"
  elif ! ( f22_gsm8k_window_check "$CTL_L" "$VIEW_CONTROLS" ); then
    check standup_controls FAIL "gsm8k window check refused the controls window $CTL_L for the view's controls list ($VIEW_CONTROLS): the 10-shot gsm8k prompt + the controls cap does not fit (D72; see the log); the controls standup was not attempted"
  elif f22_standup attacks "$SERVE_DIR/controls" "" capability:short "$CTL_L" "" "$CTL_TOP_LOGPROBS"; then
    f22_served_cells_filter CELLS "$CELL"
    check standup_controls PASS "geometry=${TARGET_S}x${TARGET_L} attested; $TARGET_ENGINE; max_logprobs=$CTL_TOP_LOGPROBS"
    [ "$TARGET_ENGINE" = lora_native ] && output_diagnostics_check controls
    # THE INSPECT CLIENT NEEDS A BEARER VARIABLE (contracts.INSPECT_API_KEY_ENV): inspect_ai's OpenAI-compatible provider
    # refuses to initialise without it even against this unauthenticated loopback proxy — smoke 502327 lost its controls
    # leg to exactly that refusal once the client venv was in place. The value is the controls standup's proxy_token
    # (endpoint.json; None on this proxy) else the client default `petri` — the Modal launcher's rule
    # (tinycap_served.py INSPECT_API_KEY_ENV: endpoint.get("proxy_token") or "petri") and served_thinking.sh's
    # st_proxy_token, spelled once here for the smoke and the wave alike.
    CTL_PROXY_TOKEN=$("$CTL_PY" -c 'import json,sys;print(json.load(open(sys.argv[1])).get("proxy_token") or "petri")' "$SERVE_DIR/controls/endpoint.json")
    # THE THINKING ARM IS NEVER IMPLICIT (contracts.FLAG_THINKING, P33): served_capability_run refuses a tinycap control
    # (gsm8k) with no --thinking — smoke 502002 lost its controls leg to exactly that refusal, a defect main's smoke
    # carries too (identical invocation since before the fork). The arm is the standup's ATTESTED regime,
    # TARGET_THINKING_MODE (f22_standup exports it from endpoint.json + /v1/models after refusing any mismatch with what
    # F22_THINKING_ON asked for): `off` by default, `on` under F22_THINKING_ON — read from the stamp, never the variable.
    # A no-switch family (llama) records `off` as not_available itself (contracts.effective_arm_for_family). Passed on every
    # list, as the wave passes it: inert on a legacy-only list (the driver reads it only for tinycap atoms).
    # THE INTERPRETER is $CTL_PY (decided beside the other venvs above: the client venv for a view whose list carries an
    # Inspect control, the driver venv for a legacy-only list — the wave's rule), CPU-only (the client lock's torch is the CPU
    # build; the GPU hidden either way), with the checkout's src + scripts on PYTHONPATH the way g8k_common.sh runs it (the
    # client lock installs no sisteer), carried by `env` exactly as f22_queue_launch carries the wave's; CTL_ENV adds the
    # offline datasets cache for an Inspect list (empty otherwise). From `env` on this is THE WAVE'S INVOCATION
    # (target_static.sbatch phase CONTROLS) flag for flag in the same order — --controls the VIEW's roster list — with the
    # smoke's own cell, out-root and width, then the smoke's LIMITS appended (--limit 8 --max-goals 8); the wave test pins
    # the two argv equal after exactly those substitutions.
    if ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" env PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts" TINYCAP_API_KEY="$CTL_PROXY_TOKEN" ${CTL_ENV[@]+"${CTL_ENV[@]}"} "$CTL_PY" scripts/served_capability_run.py generate \
          --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$CELL" --config "$CFG" \
          --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$SMOKE_ROOT/controls/$TAG" \
          --controls "$VIEW_CONTROLS" --concurrency 32 --top-logprobs "$CTL_TOP_LOGPROBS" \
          --thinking "$TARGET_THINKING_MODE" \
          --vllm-log "$SERVE_DIR/controls/vllm_server.log" \
          --limit 8 --max-goals 8 ) > "$SERVE_DIR/drivers/controls__${CELL}.log" 2>&1; then
      check "controls__${CELL}" PASS "controls=$VIEW_CONTROLS${CTL_CONTROLS_OVERRIDE:+ (OVERRIDE)}; $(find "$SMOKE_ROOT/controls" -type f | wc -l) files under $SMOKE_ROOT/controls; $(tail -2 "$SERVE_DIR/drivers/controls__${CELL}.log" | tr '\n' ' ' | head -c 400)"
    else
      check "controls__${CELL}" FAIL "$(tail -10 "$SERVE_DIR/drivers/controls__${CELL}.log" | tr '\n' ' ' | head -c 1000)"
    fi
  else
    check standup_controls FAIL "$(tail -15 "$SERVE_DIR/controls/standup.log" | tr '\n' ' ' | head -c 1200)"
  fi
  f22_standup_stop
fi
# ------------------------------------------------------------------ multi-turn geometry (needs the LIVE pool on THIS cluster)
if has_leg multiturn; then
  if f22_standup attacks "$SERVE_DIR/attacks" ""; then
    f22_served_cells_filter CELLS "$CELL"
    check standup_attacks PASS "geometry=${TARGET_S}x${TARGET_L} attested; $TARGET_ENGINE"
    [ "$TARGET_ENGINE" = lora_native ] && output_diagnostics_check attacks
    positive_control attacks
    ATTACKERS=(); f22_live_attackers
    check attacker_pool_reachable PASS "live=${#ATTACKERS[@]}: ${ATTACKERS[*]}"
    i=0
    for cell in "${CELLS[@]}"; do
      url="${ATTACKERS[$((i % ${#ATTACKERS[@]}))]}"; i=$((i+1))
      ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/jailbreak_run.py generate \
          --config "$CFG" --attack crescendo --defense "$cell" --confirmatory --shard 0 --nshards 259 \
          --private-root "$PRIVATE_ROOT" --attacker-engine vllm --attacker-endpoint "$url" \
          --target-engine served --target-proxy-url "$PROXY_URL" --target-routing-table "$TABLE" \
          --target-max-model-len "$TARGET_L" --expect-pair-n-streams 5 ) > "$SERVE_DIR/drivers/crescendo__${cell}.log" 2>&1 &
      pid=$!; DRIVER_PIDS+=("$pid"); DRIVER_TAG[$pid]="crescendo__${cell}"; sleep 3
    done
    for pid in "${DRIVER_PIDS[@]}"; do
      wait "$pid"; rc=$?; tag="${DRIVER_TAG[$pid]}"
      if [ "$rc" = 0 ]; then
        prog=$(count_progress crescendo "${tag#crescendo__}")
        case "$prog" in unique_goals=1*forwards=0|unique_goals=0*) check "$tag" FAIL "$prog (no target forwards landed)";; unique_goals=1*) check "$tag" PASS "$prog";; *) check "$tag" FAIL "$prog";; esac
      else
        check "$tag" FAIL "rc=$rc $(tail -8 "$SERVE_DIR/drivers/$tag.log" | tr '\n' ' ' | head -c 800)"
      fi
    done
    DRIVER_PIDS=()
  else
    check standup_attacks FAIL "$(tail -15 "$SERVE_DIR/attacks/standup.log" | tr '\n' ' ' | head -c 1200)"
  fi
  f22_standup_stop
fi
f22_log "SMOKE SUMMARY ($CHECKS):"; cat "$CHECKS"
RC=$(( NFAILED > 0 ? 2 : 0 ))
f22_log "SMOKE ${TAG} legs=$LEGS cell=$CELL: $NFAILED failed checks -> rc=$RC"
exit $RC
