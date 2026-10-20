#!/bin/bash
#SBATCH --job-name=asscr
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
# asscreen_controls.sbatch — STRENGTH FRONTIER (paper Figure 5), AlphaSteer lane: the SCREEN of ONE (target, strength view)
# on the LoRA dose screens' EXACT instrument (scripts/alphasteer_f22/screen/README.md).
#
# The screen is target_sampled.sbatch's `controls_t<T>` phase: ONE steered target server (1 GPU, stacked
# reduction mode, attestation required), the steer-is-live positive control, then the judged controls — XSTest (450) +
# OR-Bench-Hard (1319) — under the served MODEL's vendor sampler (configs/sdfbase_sampled_decode.yaml; per-item seed;
# trial T) at the 512 rung into $WAVE_ROOT/sampled/controls_cap512/<TAG>_t<T>/, then the CAP LADDER: rung 2048 and rung
# 8192 re-draw ONLY the (cell, control) pairs whose at-cap count at the previous rung is > 0 (--controls-cap <rung>
# --escalate-from-root <prev> --floor-sample 200). AT-CAP AT 512 IS THE LADDER WORKING, NOT A DEFECT: nothing here
# reads it as a problem signal, and the verdict tool's clause (b) is not amended here.
#
# TWO SHAS, ONE INSTRUMENT. Positional, never --export= (hard rule 1):
#   sbatch -J asscr-<TAG>-ctl<T> [--qos=high --output=$SHARED/exp/logs/%x_%j.out] [--time=HH:MM:SS] \
#       <as_checkout>/scripts/alphasteer_f22/screen/asscreen_controls.sbatch <target> <view> <wave_sha> <as_sha> [trial=0]
#   e.g. sbatch --qos=high --output=$SHARED/exp/logs/%x_%j.out -J asscr-nemotron3nano4b_as_d0p3-ctl0 --time=2:00:00 \
#       $SHARED/eval-repo/<as_sha>/scripts/alphasteer_f22/screen/asscreen_controls.sbatch \
#       nemotron3nano4b as_d0p3 f38d055d59a7102f930e30d3abac64adbcf1270b <as_sha>
#   <wave_sha>  the CODE: $SHARED/eval-repo/<wave_sha> — common.sh, common_sampled.sh, mt_serving_up.py,
#               served_capability_run.py, smoke_positive_control.py, src/sisteer. For a screen that is comparable with the
#               LoRA dose screens this is f38d055d59a7102f930e30d3abac64adbcf1270b (as22_screen.LORA_SCREEN_WAVE_SHA), the sha
#               every f22s-<TAG>-ctl0 dose job ran from; any other sha is a DIFFERENT INSTRUMENT (logged as such).
#   <as_sha>    the DATA: $SHARED/eval-repo/<as_sha>, a checkout of branch strength-asscreen — the strength view's
#               serving config + its sampled variants (make_strength_sampled_configs.py), its Petri manifest, the strength
#               roster and pins. NO code runs from it: as22_screen.py (stdlib + yaml, system python3, before any venv — the
#               wave's --is-in-scope pattern) only RESOLVES paths. Both checkouts must be at HEAD == sha and clean.
# What is the wave's, unchanged, by SOURCING the wave checkout's common_sampled.sh (which sources common.sh): f22_init
# (cluster roots, hf_env.sh, stacked env, the wave root), the phase grammar, f22s_check_config (the sampled config IS the
# serving config as data except {out_dir, private_run, sampled_decode}), the venvs pinned to <wave_sha>, the nemotron
# fused-kernel assertion, f22_standup (stacked + attested, identity S x L, --max-logprobs from the driver), the positive
# control, the bounded driver queue, the at-cap counters, the ladder events, the RESULT line. The controls phase body below
# is the wave sbatch's, VERBATIM (tests/test_alphasteer_f22_screen.py pins it byte-for-byte against
# git show <wave_sha>:scripts/sdfbase/full22/sampled/target_sampled.sbatch), so the driver invocation cannot drift.
# What is THIS lane's, in place of two wave functions: (1) f22s_resolve_config -> as22_screen.py --resolve (the wave's
# gates on make_sampled_configs.SAMPLED_VIEWS and derives sdfbase_full22_* names; ours reads the strength roster and the
# alphasteer_f22_serve_* files); (2) f22_bank_view -> asscr_bank_view (the wave's binds the sdfbase/full22 bank and
# roster_cell_count.py's roster; ours runs the SAME make_bank_view.py from the wave tree over OUR roster + pins + bank,
# $SHARED/exp/objects/alphasteer_f22/<target>/strength, with the same short-view guard). Everything else is byte-identical.
#
# TAG = <target>_<view> (e.g. nemotron3nano4b_as_d0p3): stores, ladders and serving dirs land beside the LoRA screens'
# under the SAME wave root, and screen_verdict.py reads them with --roster configs/alphasteer_f22_strength_roster.yaml.
# Cells: explicit_cells [none, alphasteer] — `none` is this view's OWN same-pass anchor, `alphasteer` the bare Class B
# id served from <view dir>/alphasteer/object.pt (a symlink into the lane's cells/<view>__alphasteer), the twin none__lens0
# derived. STOP: the wave-wide $WAVE_ROOT/STOP, or this job's $SERVE_DIR/STOP — checked before every driver launch and
# every rung.
#
# NOMINAL STRENGTHS ARE LABELS: the grid -0.5, 0.1, 0.3, 0.5, 1.0 is base_scale in the served op's units and is
# UNCHANGED for the re-baked objects (paper-sized layer sets): same views, same TAGs, same launcher, different bytes behind
# cells_view/<view>/alphasteer. A nominal strength is NOT comparable across object types or layer sets; the
# physical comparison is the separately measured realised dose, never the label.
#
# LOW-DOSE LIVENESS (D87 + D88, the coordinator 2026-09-11): the greedy-text divergence probe is an OBSERVATION at low dose,
# not a gate. Liveness is NUMERIC: (A) the intervention demonstrably distinct from the anchor's, and (B) the REALISED DOSE —
# read from the lane's OFFLINE capture $SHARED/exp/strength_dose/<target>/dose_<target>.json and BOUND to the served object
# by sha256, so a re-baked object (different bytes, different layer sets) cannot borrow the as-fit numbers and the gate says
# the re-bake needs its own capture. The serving-time effect above the engine's own repeat noise is measured too and
# recorded beside it; it stands in for (B) only when the artifact does not bind, labelled a proxy.
# Before each standup's positive control this job MEASURES liveness (asscr_liveness.py) and exports
# SISTEER_LOW_DOSE_LIVENESS_RECORD; at the code sha in as22_screen.LOW_DOSE_LIVENESS_WAVE_SHA the smoke's and the driver's
# text-divergence refusals consult that record and, when it admits the cell, record the text result and proceed. A cell whose
# NUMERIC checks fail is genuinely not live and still refuses. See section 4 of the README.
#
# COMPARABILITY CAVEAT THAT MUST TRAVEL WITH EVERY NUMBER THIS WRITES (re-verified here): the
# wave instrument's steer-is-live gate — the smoke's `steering_bites` AND the controls driver's own probe before it
# generates — is greedy TEXT difference on FIVE NEUTRAL prompts vs the scale-0 twin, refusing when NONE differs.
# AlphaSteer nulls its edit on benign input BY CONSTRUCTION, so a correctly served low-strength object can read 0/5 there
# and be refused as "inert" by the very instrument the LoRA cells pass 4/5 on. This wrapper does NOT relax that gate
# (the identical-instrument rule); a refusal whose ONLY failing check is steering_bites, with greedy_divergence PASS, is
# that false negative and needs a decision, not a resubmit (README section 4).
set -uo pipefail
TARGET="${1:?target (gemma4e4b|qwen35_4b|nemotron3nano4b)}"; VIEW="${2:?strength view (as_dm0p5|as_d0p1|as_d0p3|as_d0p5|as_d1)}"
WAVE_SHA="${3:?wave sha (the CODE checkout; f38d055d… for a LoRA-comparable screen)}"; AS_SHA="${4:?alphasteer-screen sha (the DATA checkout)}"
TRIAL="${5:-0}"
source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/sampled/common_sampled.sh"
F22_SERVING_PATH_DEFER=1   # the serving path binds below from THIS screen's manifest (as22_screen.py --resolve), never the wave's petri manifest (2026-09-15)
f22_init "$WAVE_SHA"   # cd $REPO_DIR (the CODE), HEAD == sha, clean, $SHARED / stacked env, the wave's RESULT + figure mirror ($WAVE_ROOT binds with the path below)
# --- the DATA checkout: this branch, at HEAD == <as_sha>, clean — configs, manifest, roster, pins; never code -------------
AS_REPO="$SISTEER_EVAL_REPO_ROOT/$AS_SHA"; AS_SCREEN="$AS_REPO/scripts/alphasteer_f22/screen/as22_screen.py"
[ -f "$AS_SCREEN" ] || f22_fatal "no AlphaSteer-screen checkout at $AS_REPO (README: git -C <the repo mirror> worktree add --detach $AS_REPO $AS_SHA)"
as_head=$(git -C "$AS_REPO" rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < "$AS_REPO/.git/HEAD")
[ "$as_head" = "$AS_SHA" ] || f22_fatal "AlphaSteer-screen checkout HEAD $as_head != $AS_SHA"
git -C "$AS_REPO" diff --quiet HEAD -- || f22_fatal "dirty AlphaSteer-screen checkout $AS_REPO"
LORA_WAVE=$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import as22_screen; print(as22_screen.LORA_SCREEN_WAVE_SHA)' "$(dirname "$AS_SCREEN")") \
  || f22_fatal "cannot read as22_screen.LORA_SCREEN_WAVE_SHA from $AS_SCREEN"
COMPARABLE_NOTE=$(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import as22_screen as a; print(a.COMPARABLE_WAVE_SHAS.get(sys.argv[2], ""))' "$(dirname "$AS_SCREEN")" "$WAVE_SHA" 2>/dev/null)
if [ -n "$COMPARABLE_NOTE" ]; then f22_log "instrument: wave sha $WAVE_SHA — $COMPARABLE_NOTE"
else f22_log "INSTRUMENT DIFFERS: wave sha $WAVE_SHA is neither the LoRA dose screens' $LORA_WAVE nor its D87 low-dose-liveness commit — this screen is NOT on a known-comparable instrument; every number it writes must say so"; fi
PHASE="controls_t${TRIAL}"; f22s_parse_phase "$PHASE"   # the wave's grammar: PHASE_KIND=controls, TRIAL, PHASE_TAG
# --- resolve (target, view): the screen module prints shell assignments (shlex-quoted); ANY non-zero is fatal — a refusal
#     (rc 3: descoped glm46vflash, an unknown view, a missing sampled config, a roster/table disagreement) is named, never guessed
RESOLVED=$(python3 "$AS_SCREEN" --resolve "$TARGET" "$VIEW" --trial "$TRIAL" --repo "$AS_REPO" --shared "$SHARED") \
  || f22_fatal "as22_screen.py --resolve refused for $TARGET/$VIEW (see the message above)"
eval "$RESOLVED"   # KEY TAG STRENGTH CELL STAGED_ID WAVE_CFG CFG CFG_ESC CFG_ESC16 MANIFEST PRIVATE_RUN_* SIBLING_TRIAL AS_ROSTER AS_PINS AS_BANK AS_VIEW_DIR AS_SERVED_CELLS …
[ "$KEY" = "$TARGET" ] && [ -n "${TAG:-}" ] && [ -f "$CFG" ] && [ -f "$MANIFEST" ] || f22_fatal "resolver output incomplete: KEY=${KEY:-} TAG=${TAG:-} CFG=${CFG:-} MANIFEST=${MANIFEST:-}"
# the serving path + engine + WAVE_ROOT, bound from THIS screen's manifest (kind-aware: an AlphaSteer strength view is lens-only);
# an explicit F22_SERVING_PATH was bound at f22_init and is left alone
[ "${F22_SERVING_PATH_MODE:-}" != deferred ] || f22_bind_serving_path "$MANIFEST"
# the wave's per-KEY job-side belt (common.sh f22_resolve_config): TEXT-ONLY serving for the multimodal target
case "$KEY" in qwen35_4b|glm46vflash)
  grep -q -- '--limit-mm-per-prompt' "$MANIFEST" || f22_fatal "$MANIFEST serves multimodal $KEY without --limit-mm-per-prompt (text-only pin) — regenerate (make_strength_configs.py)";;
esac
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
SERVE_DIR="$WAVE_ROOT/sampled/serving/${PHASE_TAG}_${TAG}_${SLURM_JOB_ID}"; PRIVATE_ROOT="$WAVE_ROOT/transcripts_results"
LADDER_DIR="$WAVE_ROOT/sampled/ladders/${TAG}_t${TRIAL}"; JOB_STOP="$SERVE_DIR/STOP"
mkdir -p "$SERVE_DIR/drivers" "$PRIVATE_ROOT" "$LADDER_DIR"; chmod 700 "$PRIVATE_ROOT"
f22_log "asscreen: target=$TARGET view=$VIEW strength=$STRENGTH cell=$CELL staged=$STAGED_ID tag=$TAG phase=$PHASE trial=$TRIAL code=$WAVE_SHA data=$AS_SHA"
f22_log "asscreen: config=$CFG (serving config $WAVE_CFG) manifest=$MANIFEST view_dir=$AS_VIEW_DIR serve_dir=$SERVE_DIR job_stop=$JOB_STOP"
printf '{"job_id":"%s","job_name":"%s","target":"%s","view":"%s","strength":%s,"tag":"%s","trial":%s,"wave_sha":"%s","as_sha":"%s","lora_screen_wave_sha":"%s","same_instrument_as_lora_screens":%s,"instrument_note":"%s","config":"%s","serving_config":"%s","manifest":"%s","view_dir":"%s","cluster":"%s","node":"%s","written_at":"%s"}\n' \
  "${SLURM_JOB_ID:-manual}" "${SLURM_JOB_NAME:-manual}" "$TARGET" "$VIEW" "$STRENGTH" "$TAG" "$TRIAL" "$WAVE_SHA" "$AS_SHA" "$LORA_WAVE" "$( [ -n "$COMPARABLE_NOTE" ] && echo true || echo false )" "${COMPARABLE_NOTE//\"/}" "$CFG" "$WAVE_CFG" "$MANIFEST" "$AS_VIEW_DIR" "$CLUSTER" "$(hostname)" "$(date -u +%FT%TZ)" > "$SERVE_DIR/asscreen_job.json"
f22_driver_venv; f22_serving_venv; f22s_assert_vllm_mamba_kernels || f22_fatal "nemotron fused Mamba2 kernels missing in the serving venv (see log)"
f22s_check_config "$MPY"   # the wave's belt, unchanged: CFG/CFG_ESC/CFG_ESC16 ARE WAVE_CFG (our serving config) as data except the permitted keys
# --- the STRENGTH bank view: the wave tree's make_bank_view.py over OUR roster + pins + bank (the campaign's bank view — the same
#     script over ITS roster + pins — binds
#     lambda_ladder.json's chosen sha and refuses a strength copy by construction; the wave's f22_bank_view binds the sdfbase
#     bank and roster_cell_count.py's roster). Verify first, build only under a per-(target, view) lock (common.sh's rule).
asscr_bank_view(){   # $1 = python -> exports TT_TWO_SI_DEFENSES_DIR, ALL_CELLS; the short-view guard (config == roster == built view)
  local py="${1:?python}" lock
  local -a A=(--roster "$AS_ROSTER" --pins "$AS_PINS" --bank "$AS_BANK" --target "$TARGET" --view "$VIEW")
  lock="$AS_BANK/.$TARGET-$VIEW.view.lock"; mkdir -p "$(dirname "$lock")"
  if ! "$py" scripts/sdfbase/full22/make_bank_view.py --check "${A[@]}" >/dev/null 2>&1; then
    f22_log "strength bank view $TARGET/$VIEW does not verify — building it under $lock"
    (
      flock -w 900 9 || { echo "FATAL: could not take the bank-view lock for $TARGET/$VIEW in 900s" >&2; exit 1; }
      "$py" scripts/sdfbase/full22/make_bank_view.py "${A[@]}"
    ) 9>"$lock" || f22_fatal "strength bank view for $TARGET/$VIEW refused"
  fi
  "$py" scripts/sdfbase/full22/make_bank_view.py --check "${A[@]}" || f22_fatal "strength bank view for $TARGET/$VIEW refused"
  [ -d "$AS_VIEW_DIR" ] || f22_fatal "no view dir at $AS_VIEW_DIR after make_bank_view.py"
  export TT_TWO_SI_DEFENSES_DIR="$AS_VIEW_DIR"
  mapfile -t ALL_CELLS < <("$py" - "$CFG" <<'PY'
import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["explicit_cells"]))
PY
)
  [ "${ALL_CELLS[*]}" = "none $CELL" ] || f22_fatal "$CFG lists explicit_cells [${ALL_CELLS[*]}] — a strength view serves exactly [none $CELL] (its own same-pass anchor + the bare Class B id)"
  # the roster's counts for THIS (target, view), through the wave tree's ONE reader (f22_roster) over OUR roster
  local want want_staged have n_view
  read -r want want_staged < <("$py" - "$AS_ROSTER" "$TARGET" "$VIEW" <<'PY'
import sys, yaml
sys.path.insert(0, "scripts/sdfbase/full22")
from f22_roster import served_cells, staged_cells
r = yaml.safe_load(open(sys.argv[1]))
print(len(served_cells(r, sys.argv[2], sys.argv[3])), len(staged_cells(r, sys.argv[2], sys.argv[3])))
PY
) || f22_fatal "cannot read the strength roster's cell counts for $TARGET/$VIEW"
  have=$(( ${#ALL_CELLS[@]} - 1 ))
  [ "$have" -eq "$want" ] || f22_fatal "$CFG lists $have cells + none but the strength roster names $want for $TARGET/$VIEW — refusing a narrowed view"
  n_view=$(find "$TT_TWO_SI_DEFENSES_DIR" -mindepth 1 -maxdepth 1 ! -name 'VIEW_MANIFEST*' ! -name 'full22_*' 2>/dev/null | wc -l)
  [ "$n_view" -eq "$want_staged" ] || f22_fatal "bank view $TT_TWO_SI_DEFENSES_DIR holds $n_view entries but the strength roster names $want_staged staged cells — a short (or stale) view is an error"
  local f n_id=0 f_id
  for f in object.pt staged_from.json; do [ -f "$TT_TWO_SI_DEFENSES_DIR/$CELL/$f" ] || f22_fatal "bank view $TT_TWO_SI_DEFENSES_DIR/$CELL lacks $f"; done
  for f_id in train_budget_meta.json fit_provenance.json derived_meta.json; do [ -f "$TT_TWO_SI_DEFENSES_DIR/$CELL/$f_id" ] && n_id=$((n_id+1)); done
  [ "$n_id" -eq 1 ] || f22_fatal "bank view $TT_TWO_SI_DEFENSES_DIR/$CELL declares $n_id identity sidecars (need exactly one)"
  f22_log "strength bank view OK: $TT_TWO_SI_DEFENSES_DIR (${#ALL_CELLS[@]} cells incl. none; roster $want served / $want_staged staged, view $n_view — complete; $CELL -> $(readlink -f "$TT_TWO_SI_DEFENSES_DIR/$CELL") sha256 $(sha256sum "$TT_TWO_SI_DEFENSES_DIR/$CELL/object.pt" | cut -c1-16)…)"
}
asscr_bank_view "$MPY"
CELLS=("${ALL_CELLS[@]}"); NCELLS=${#CELLS[@]}
# the served object's sha and the lane's OFFLINE dose capture: fact (B) of the liveness gate, BOUND to these bytes —
# a re-baked object's sha will not match the artifact's rung and the gate then says so rather than borrowing the number
OBJECT_SHA256=$(sha256sum "$TT_TWO_SI_DEFENSES_DIR/$CELL/object.pt" | cut -d" " -f1)
AS_DOSE_ARTIFACT="$SHARED/exp/strength_dose/$TARGET/dose_$TARGET.json"
[ -f "$AS_DOSE_ARTIFACT" ] || f22_log "no offline dose capture at $AS_DOSE_ARTIFACT — the liveness gate's fact (B) will fall back to the serving-time effect (a labelled proxy)"
f22_log "served object: $TT_TWO_SI_DEFENSES_DIR/$CELL/object.pt sha256 $OBJECT_SHA256 (staged $STAGED_ID); dose artifact $AS_DOSE_ARTIFACT"
# --- LOW-DOSE LIVENESS IS NUMERIC, and the greedy-text probe is an OBSERVATION -----------------------------------------
# At a small enough dose a CORRECT object changes no greedy token on a neutral probe, so the wave's `steering_bites` text
# probe reports "not live" when steering is live and merely SMALL — and the stop rule would then stop the low end of every
# ladder for being low. So before each standup's positive control we MEASURE liveness (asscr_liveness.py: the intervention
# distinct from the anchor's, and an effect above the engine's own repeat noise, ruled by the instrument's OWN
# fidelity.check_logprob_steering_bites) and, when it passes, export the record that the refusal branches
# consult. The wave's own control still runs, unmodified, right after — its other six checks are untouched gates.
# The phase text below stays BYTE-IDENTICAL to the wave's: we rename the wave's function and wrap it, never edit the call.
eval "$(declare -f f22s_positive_control | sed '1s/^f22s_positive_control/asscr_wave_positive_control/')"
f22s_positive_control(){   # $1 = phase tag -> 0 PASS, 1 FAIL (the wave's contract, unchanged)
  local ptag="$1"
  local rec="$SERVE_DIR/liveness_low_dose_${ptag}.json"
  local lrc=0
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" "$AS_REPO/scripts/alphasteer_f22/screen/asscr_liveness.py" \
      --wave-repo "$REPO_DIR" --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$CELL" --tag "$TAG" \
      --nominal-dose "$STRENGTH" --dose-artifact "$AS_DOSE_ARTIFACT" --staged-id "$STAGED_ID" \
      --object-sha256 "$OBJECT_SHA256" --arm "$CELL" --out "$rec" ) > "$SERVE_DIR/drivers/liveness_${ptag}.log" 2>&1 || lrc=$?
  case "$lrc" in
    0) export SISTEER_LOW_DOSE_LIVENESS_RECORD="$rec"
       f22_log "low-dose liveness ($ptag): LIVE NUMERIC — $("$MPY" "$AS_REPO/scripts/alphasteer_f22/screen/asscr_liveness.py" --summary "$rec" 2>/dev/null || echo "record $rec"); the text probe is an OBSERVATION (D87/D88)";;
    2) f22_log "REFUSING TRAFFIC: low-dose liveness ($ptag) says NOT LIVE — a numeric check failed, which is a genuine refusal, not the low-dose false negative (D87). See $rec and $SERVE_DIR/drivers/liveness_${ptag}.log"
       return 1;;
    *) unset SISTEER_LOW_DOSE_LIVENESS_RECORD
       f22_log "low-dose liveness ($ptag): UNEVALUABLE (rc=$lrc) — the numeric record admits NOTHING and the wave's text gate governs unchanged; see $SERVE_DIR/drivers/liveness_${ptag}.log";;
  esac
  asscr_wave_positive_control "$ptag"
}

# ---------------------------------------------------------------- RECORD THE RESOLVED SAMPLER
# vLLM fills every sampling key a request OMITS — from the model's generation_config.json when it ships one and from
# its own neutral defaults when it does not — so the sampler that reached the engine is a property of the SERVING
# STACK, not of what the client sent. Measured across this campaign: ZERO of 399 server logs record it, the response
# body carries no sampling fields, and on Qwen the startup line that would state the applied defaults is absent by
# construction. So no existing cell can answer 'which sampler produced this number'.
#
# --enable-log-requests is the only mechanism. It is set HERE, outside the byte-pinned region below, because
# f22_standup reads F22_STANDUP_EXTRA from the shell and appends it to mt_serving_up.py's argv verbatim — so the
# pinned driver invocation is unchanged. The prompt rides a separate DEBUG record and is NOT logged at INFO. Built on top of
# the inherited value through common.sh's helper (2026-09-16): an inherited scalar F22_STANDUP_EXTRA is the launch-time argv
# hook; a bare assignment clobbered it, a bare `+=` first touch would keep an exported-empty scalar as an empty argv element.
f22_standup_extra_reset --enable-log-requests

# ================================================================ VERBATIM from target_sampled.sbatch @ <wave_sha> (pinned by test) — the phase helpers
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
# ================================================================ end verbatim helpers
if [ "$PHASE_KIND" != controls ]; then f22_fatal "this wrapper runs the controls phase only (got $PHASE_KIND)"
# ================================================================ VERBATIM from target_sampled.sbatch @ <wave_sha> (pinned by test) — the controls phase
# ---------------------------------------------------------------- controls_t<T>: rung 512, then 2048, then 8192
elif [ "$PHASE_KIND" = controls ]; then
  f22s_ctl_top_logprobs
  CTL_LIST=$(IFS=,; echo "${F22S_CONTROLS[*]}")
  # ---- rung 512: the SAME-CAP base pass, every cell, both controls (capability:short 2048 x S, as the wave)
  f22_standup attacks "$SERVE_DIR/controls_512" "" capability:short 2048 "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }
  f22s_positive_control controls_512 || { RC=1; exit 1; }
  CONC=$(( TARGET_S / NCELLS )); [ "$CONC" -ge 8 ] || CONC=8
  MAX_DRIVERS=$((NCELLS + 1))   # the queue bound counts LIVE drivers (common.sh); +1 keeps the wave's launch-gate wart moot
  f22_log "phase CONTROLS t$TRIAL rung 512: cells=$NCELLS controls=$CTL_LIST concurrency/cell=$CONC (server width $TARGET_S) logprobs=$CTL_TOP_LOGPROBS out=$(ROOT_OF 512)"
  for cell in "${CELLS[@]}"; do
    f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break; }
    f22_queue_launch "controls512__${cell}" "$MPY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$(ROOT_OF 512)" \
      --controls "$CTL_LIST" --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS" \
      --vllm-log "$SERVE_DIR/controls_512/vllm_server.log"
    sleep 3
  done
  f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
  f22_log "phase CONTROLS rung 512 done: $NFAIL failed drivers"
  f22_standup_stop
  # ---- the ladder: 2048 (window 4096), then 8192 (window 10240) — each ONLY for the pairs still on the ladder
  declare -A ON_LADDER=()
  for cell in "${CELLS[@]}"; do ON_LADDER[$cell]="$CTL_LIST"; done
  PREV_RUNG=512; LAST_RAN=512
  for RUNG in 2048 8192; do
    WINDOW=$(f22s_rung_window "$RUNG")   # per-model: nemotron3nano4b 4864 / 8704, every other key 4096 / 10240 (common_sampled.sh)
    PREV_ROOT=$(ROOT_OF "$PREV_RUNG"); NEXT_ROOT=$(ROOT_OF "$RUNG")
    # count at the PREVIOUS rung for every pair still on the ladder; a zero count leaves it (the exit rule)
    NRUNG=0
    for cell in "${CELLS[@]}"; do
      [ -n "${ON_LADDER[$cell]:-}" ] || continue
      IFS=',' read -ra ctls <<< "${ON_LADDER[$cell]}"; keep=""
      for ctl in "${ctls[@]}"; do
        n=$(f22s_at_cap_controls "$PREV_ROOT" "$cell" "$ctl") || { f22_log "REFUSING: the at-cap count at rung $PREV_RUNG for $cell/$ctl was refused — the ladder cannot proceed on a guess"; RC=1; exit 1; }
        f22s_event "$(f22s_event_json kind=controls cell="$cell" control="$ctl" rung="$PREV_RUNG" n_at_cap="$n" root="$PREV_ROOT")"
        f22_log "at-cap @$PREV_RUNG $cell/$ctl: $n"
        [ "$n" -gt 0 ] && keep="${keep:+$keep,}$ctl"
      done
      ON_LADDER[$cell]="$keep"; [ -n "$keep" ] && NRUNG=$((NRUNG + 1))
    done
    f22s_write_ladder controls "$LADDER_DIR/ladder.json"
    if [ "$NRUNG" -eq 0 ]; then f22_log "phase CONTROLS: every (cell, control) left the ladder at rung $PREV_RUNG — no $RUNG rung"; break; fi
    f22s_stop_requested && { f22_log "STOP present — not standing up rung $RUNG"; break; }
    f22_standup attacks "$SERVE_DIR/controls_${RUNG}" "" capability:short "$WINDOW" "" "$CTL_TOP_LOGPROBS" || { RC=1; exit 1; }
    f22s_positive_control "controls_${RUNG}" || { RC=1; exit 1; }
    CONC=$(( TARGET_S / NRUNG )); [ "$CONC" -ge 8 ] || CONC=8
    MAX_DRIVERS=$((NRUNG + 1))
    f22_log "phase CONTROLS t$TRIAL rung $RUNG (window $WINDOW x $TARGET_S): $NRUNG cells still on the ladder, concurrency/cell=$CONC, --floor-sample $F22S_FLOOR_SAMPLE_K, from $PREV_ROOT into $NEXT_ROOT"
    for cell in "${CELLS[@]}"; do
      [ -n "${ON_LADDER[$cell]:-}" ] || continue
      f22s_stop_requested && { f22_log "STOP present — not launching more drivers"; break; }
      f22_queue_launch "controls${RUNG}__${cell}" "$MPY" scripts/served_capability_run.py generate \
        --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$cell" --config "$CFG" \
        --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$NEXT_ROOT" \
        --controls "${ON_LADDER[$cell]}" --concurrency "$CONC" --top-logprobs "$CTL_TOP_LOGPROBS" \
        --vllm-log "$SERVE_DIR/controls_${RUNG}/vllm_server.log" \
        --controls-cap "$RUNG" --escalate-from-root "$PREV_ROOT" --floor-sample "$F22S_FLOOR_SAMPLE_K"
      sleep 3
    done
    f22_queue_wait; TOTAL_FAIL=$((TOTAL_FAIL + NFAIL))
    f22_log "phase CONTROLS rung $RUNG done: $NFAIL failed drivers"
    f22_standup_stop
    PREV_RUNG=$RUNG; LAST_RAN=$RUNG
  done
  # the residual at the LAST rung that ran (a genuine non-termination at 8192 is recorded as such)
  if [ "$LAST_RAN" != 512 ]; then
    LAST_ROOT=$(ROOT_OF "$LAST_RAN")
    for cell in "${CELLS[@]}"; do
      [ -n "${ON_LADDER[$cell]:-}" ] || continue
      IFS=',' read -ra ctls <<< "${ON_LADDER[$cell]}"
      for ctl in "${ctls[@]}"; do
        n=$(f22s_at_cap_controls "$LAST_ROOT" "$cell" "$ctl") || { TOTAL_FAIL=$((TOTAL_FAIL + 1)); continue; }
        f22s_event "$(f22s_event_json kind=controls cell="$cell" control="$ctl" rung="$LAST_RAN" n_at_cap="$n" root="$LAST_ROOT")"
        f22_log "at-cap @$LAST_RAN $cell/$ctl: $n$( [ "$LAST_RAN" = 8192 ] && [ "$n" -gt 0 ] && echo ' (residual: genuine non-termination)')"
      done
    done
  fi
  f22s_write_ladder controls "$LADDER_DIR/ladder.json"
# ================================================================ end verbatim controls phase
fi


# ---------------------------------------------------------------- the resolved sampler, read off each rung's own log
# A record, never a gate: the screen's data is valid whether or not this reads, so a failure here is logged loudly and
# does NOT change RC. Its ABSENCE is the signal — a missing record means unreadable, which the comparison treats as
# UNREADABLE rather than as agreement.
for _rung_dir in "$SERVE_DIR"/controls_*; do
  [ -f "$_rung_dir/vllm_server.log" ] || continue
  _rec="$_rung_dir/resolved_sampler.json"
  if "$MPY" "$AS_REPO/scripts/sdfbase/record_resolved_sampler.py" \
       --vllm-log "$_rung_dir/vllm_server.log" --out "$_rec" > "$_rung_dir/resolved_sampler.log" 2>&1; then
    f22_log "resolved sampler recorded: $_rec"
  else
    f22_log "WARNING: could not record the resolved sampler for $_rung_dir (rc=$?) — the screen stands, but this rung"
    f22_log "WARNING: has NO resolved-sampler record; see $_rung_dir/resolved_sampler.log"
  fi
done