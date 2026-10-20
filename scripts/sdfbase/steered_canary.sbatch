#!/bin/bash
#SBATCH --job-name=scanary
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# THE STEERED SERVING CANARY of an onboarded model: the first native standup
# that serves a steering vector on the model — ONE ReLU object exported as a residual-slot carrier beside ONE
# certified LoRA adapter and `none`, through scripts/sdfbase/steered_canary.py (lora_serving_up.bring_up_lora,
# the certification driver's seam). The job stages its own canary view from two trial directories (idempotent,
# digest-pinned), derives the canary's config + manifest pair from the roster view's (the same data but the
# cells), runs the standup, reads the residual slot's certificate off the attestation (armed; every block
# hooked; the carrier a rank-1 residual entry; the noop certificate and steering_bites among the six positive
# controls), then the load check three ways (base / LoRA / carrier). Modelled on native_timing_72.sbatch
# (common.sh's init, venvs, RESULT line; the per-sha checkout gate is f22_init's).
# Usage (positional only — never a Slurm export flag; the job inherits the submitter's exported environment):
#   sbatch -J scanary-<key> scripts/sdfbase/steered_canary.sbatch <key> <view> <sha> <relu_cell> <relu_src> <relu_pin> <lora_cell> <lora_src> <lora_pin> [n_layers] [batches] [max_tokens] [S]
#     key        the canonical model key (qwen35_4b_demo ...)
#     view       the roster view whose config + manifest pair the canary derives from (u0 ...)
#     sha        the checkout $SHARED/sisteer-eval-repo/<sha> (HEAD == sha)
#     relu_cell  the carrier's train/test cell id (<si>__psr_ce_all); relu_src the trial dir that holds its object.pt
#     relu_pin   the FROZEN HPO CHOICE, independent of the path: <object_sha256>:<fit_fingerprint>:<search_id>/<trial_id>
#                — the digest the operator obtained elsewhere, and the registry's fit_trial record's fingerprint + ids; the stage step
#                refuses a trial dir whose bytes, summary.json or fit_trial_record.json disagree, or that lacks either record
#     lora_cell  the LoRA cell's train/test cell id (<si>__lora_r4_ce); lora_src its trial dir; lora_pin as above
#     n_layers   the model's block count (default 40); batches (default 32,128); max_tokens (default 64); S (default 64)
# The job NEVER registers in the results registry (F22_REGISTRY_CANARY=1, honoured by every common.sh registry entry point) and
# asserts at exit that no pending file was created for it.
# Records: $SHARED/exp/steered_canary/<key>_<jobid>/ (<key>.json, view/, config/, standup/, windows/, checks.jsonl);
# the JSON + checks copied to $FIGS_DIR with the RESULT line.
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; WAVE_SHA="${3:?wave sha}"
RELU_CELL="${4:?relu cell}"; RELU_SRC="${5:?relu trial dir}"; RELU_PIN="${6:?relu pin sha256:fit_fingerprint:search_id/trial_id}"
LORA_CELL="${7:?lora cell}"; LORA_SRC="${8:?lora trial dir}"; LORA_PIN="${9:?lora pin sha256:fit_fingerprint:search_id/trial_id}"
N_LAYERS="${10:-40}"; BATCHES="${11:-32,128}"; MAX_TOKENS="${12:-64}"; S_REQ="${13:-64}"
source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
[ "${F22_SERVING_PATH:-lora_native}" = lora_native ] || { echo "FATAL: the canary is of the NATIVE engine; F22_SERVING_PATH=$F22_SERVING_PATH"; exit 2; }
export F22_SERVING_PATH=lora_native
# THE CANARY NEVER REGISTERS: common.sh's registry entry points refuse a canary's
# AUTHORITATIVE identity — this script's name (*_canary.sbatch, read off scontrol's Command=) and the derived config's `canary:`
# marker — so no launcher of this script can file the production cells as its runs; the job name `scanary*` (the -J the usage line
# gives) is a cross-check that FAILS FAST on a contradiction, and the flag below is a belt that only annotates. The production pair
# is never passed through f22_resolve_config (which registers): its paths are derived here, read-only.
export F22_REGISTRY_CANARY=1
f22_init "$WAVE_SHA"
TAG="$KEY$( [ "$VIEW" = main ] || echo "_$VIEW" )"
CFG="configs/sdfbase_full22_${TAG}.yaml"; MANIFEST="configs/capeval/manifests/petri_full22_${TAG}.yaml"
[ -f "$CFG" ] && [ -f "$MANIFEST" ] || f22_fatal "missing $CFG or $MANIFEST (the roster view $KEY/$VIEW the canary derives from)"
# THE STAGED SITE GEOMETRY OVERLAY (geometry.site_geometry_dir): the front door's funnel hands a job SISTEER_SHARED_ROOT so the
# standup reads $SHARED/exp/reproduce/geometry/<key>.json — the probe's MEASURED budgets, allowed pairs and adapter parity. This
# job is submitted by hand, so it names the directory itself; without it the standup sized from the PROVISIONAL built-in row
# (job 817770) and the parity gate would have refused the onboarded model for want of an overlay.
export SISTEER_GEOMETRY_DIR="${SISTEER_GEOMETRY_DIR:-$SHARED/exp/reproduce/geometry}"
[ -s "$SISTEER_GEOMETRY_DIR/$KEY.json" ] || f22_fatal "no staged site geometry at $SISTEER_GEOMETRY_DIR/$KEY.json (reproduce/lib/slurm.sh slurm_stage_geometry)"
f22_log "site geometry overlay: $SISTEER_GEOMETRY_DIR/$KEY.json"
OUT="$SHARED/exp/steered_canary/${KEY}_${SLURM_JOB_ID:-manual}"; VIEW_DIR="$OUT/view"; CFG_DIR="$OUT/config"; SERVE_DIR="$OUT/standup"
CHECKS="$OUT/checks.jsonl"; mkdir -p "$VIEW_DIR" "$CFG_DIR" "$SERVE_DIR"; : > "$CHECKS"
NFAILED=0; RC=1; HARNESS_RC=""; TARGET_PID=""
check(){   # name PASS|FAIL detail...
  local name="$1" st="$2"; shift 2
  [ "$st" = PASS ] || NFAILED=$((NFAILED+1))
  printf '{"check":"%s","status":"%s","detail":"%s","at":"%s","job":"%s"}\n' "$name" "$st" "$(echo "$*" | tr '"' "'" | tr '\n' ' ' | head -c 8000)" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-manual}" >> "$CHECKS"
  f22_log "CHECK $st $name: $(echo "$*" | head -c 3000)"
}
cleanup(){
  trap '' SIGTERM
  if [ -n "${TARGET_PID:-}" ] && kill -0 "$TARGET_PID" 2>/dev/null; then
    kill -TERM "$TARGET_PID" 2>/dev/null || true
    for i in $(seq 1 18); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done
    kill -9 "$TARGET_PID" 2>/dev/null || true
  fi
  cp -f "$OUT/$KEY.json" "$FIGS_DIR/steered_canary_${KEY}_${SLURM_JOB_ID:-manual}.json" 2>/dev/null || true
  cp -f "$CHECKS" "$FIGS_DIR/steered_canary_checks_${KEY}_${SLURM_JOB_ID:-manual}.jsonl" 2>/dev/null || true
  for f in endpoint.json endpoint.recorded.json cells_served.json positive_controls.json serving_attestation.json; do
    cp -f "$SERVE_DIR/$f" "$FIGS_DIR/${KEY}_${SLURM_JOB_ID:-manual}_$f" 2>/dev/null || true
  done
  tail -c 300000 "$SERVE_DIR/vllm_server.log" > "$FIGS_DIR/${KEY}_${SLURM_JOB_ID:-manual}_vllm_server.tail.log" 2>/dev/null || true
  # THE REGISTRY ASSERTION: a canary leaves no run identity behind — no pending file for this job, no identity bound
  local pend="$SHARED/registry/pending/${SLURM_JOB_ID:-manual}.json"
  if [ -e "$pend" ] || [ -n "${SISTEER_REGISTRY_RUN_IDENTITY:-}" ]; then
    check registry_untouched FAIL "a canary must register nothing: pending=$pend exists=$([ -e "$pend" ] && echo yes || echo no) identity=${SISTEER_REGISTRY_RUN_IDENTITY:-unset}"
    RC=2
  else
    check registry_untouched PASS "no pending file at $pend; no run identity bound (F22_REGISTRY_CANARY=1)"
  fi
  cp -f "$CHECKS" "$FIGS_DIR/steered_canary_checks_${KEY}_${SLURM_JOB_ID:-manual}.jsonl" 2>/dev/null || true
  f22_result "steered-canary $KEY/$VIEW relu=$RELU_CELL lora=$LORA_CELL harness_rc=${HARNESS_RC:-none} rc=${RC} failed_checks=${NFAILED} out=$OUT"
  kill -TERM -$$ 2>/dev/null; wait
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
f22_driver_venv
f22_serving_venv_for_key "$KEY"   # nemotron3nano4b -> the mamba-ssm overlay; qwen35_* -> the GDN overlay (fla + causal-conv1d); else the base venv
cd "$REPO_DIR" || exit 1

# (1) THE CANARY VIEW: the two trial objects staged under their served cell ids (digest-pinned, idempotent, the
# registry's own staged-cell verdict run on each), with every provenance fact the trials recorded beside them
COMPLETION_CAP=$("$MPY" -c 'import sys,yaml; c=yaml.safe_load(open(sys.argv[1])); print(int(c["completion_tokens"]))' "$CFG") \
  || f22_fatal "cannot read completion_tokens from $CFG"
FIT_MODEL=$("$MPY" -c 'import sys,yaml; m=yaml.safe_load(open(sys.argv[1]))["models"][sys.argv[2]]["fit"]["checkpoint"]; print(m["hf_repo"], m["revision"])' reproduce/models.yaml "$KEY") \
  || f22_fatal "cannot read the fit checkpoint of $KEY from reproduce/models.yaml"
read -r FIT_REPO FIT_REV <<< "$FIT_MODEL"
STAGE=$(PYTHONPATH="$REPO_DIR/src" "$MPY" scripts/sdfbase/steered_canary.py stage --target "$KEY" --view-dir "$VIEW_DIR" \
  --cell "$RELU_CELL=$RELU_SRC" --pin "$RELU_CELL=$RELU_PIN" --cell "$LORA_CELL=$LORA_SRC" --pin "$LORA_CELL=$LORA_PIN" \
  --expect-fit-model "$FIT_REPO" --expect-fit-revision "$FIT_REV" \
  --completion-tokens "$COMPLETION_CAP" --expect-n-layers "$N_LAYERS" 2>&1)
STAGE_RC=$?
case "$STAGE_RC" in 0) check stage PASS "$STAGE";; *) check stage FAIL "rc=$STAGE_RC $STAGE"; exit 1;; esac

# (2) THE CANARY PAIR: the roster view's config + manifest with the cells replaced (the derivation recorded beside them)
DERIVE=$(PYTHONPATH="$REPO_DIR/src" "$MPY" scripts/sdfbase/steered_canary.py derive --target "$KEY" --from-config "$CFG" --from-manifest "$MANIFEST" \
  --cells "none,$RELU_CELL,$LORA_CELL" --defenses-dir "$VIEW_DIR" --out-config "$CFG_DIR/config.yaml" --out-manifest "$CFG_DIR/manifest.yaml" 2>&1)
DERIVE_RC=$?
case "$DERIVE_RC" in 0) check derive PASS "$DERIVE";; *) check derive FAIL "rc=$DERIVE_RC $DERIVE"; exit 1;; esac

# (3) THE PINNED DIGESTS: the OPERATOR's pins (the frozen HPO choice), held to the staged bytes and the sidecar, then handed to the
# standup as --expect-object-sha256 (recorded in endpoint.json) — never a digest re-derived from whatever path was typed
PIN_ARGS=()
for pair in "$RELU_CELL=$RELU_PIN" "$LORA_CELL=$LORA_PIN"; do
  c="${pair%%=*}"; pin="${pair#*=}"; pin="${pin%%:*}"
  disk=$(sha256sum "$VIEW_DIR/$c/object.pt" | cut -c1-64) || f22_fatal "cannot hash $VIEW_DIR/$c/object.pt"
  rec=$("$MPY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["object_sha256"])' "$VIEW_DIR/$c/train_budget_meta.json")
  [ "$pin" = "$disk" ] && [ "$pin" = "$rec" ] || f22_fatal "digest disagreement for $c: pin ${pin:0:12} disk ${disk:0:12} sidecar ${rec:0:12}"
  f22_log "cell $c object.pt sha256 == sidecar == the operator's pin: ${pin:0:12}"
  PIN_ARGS+=(--expect-object-sha256 "$c=$pin")
done

# (4) THE HARNESS (in-process standup through bring_up_lora; the serving venv, the checkout's src first), under srun so
# Slurm tracks the engine child; backgrounded and waited on so the TERM trap runs inside the grace window
PYTHONPATH="$REPO_DIR/src" srun --ntasks=1 "$SPY" scripts/sdfbase/steered_canary.py run \
  --target "$KEY" --config "$CFG_DIR/config.yaml" --manifest "$CFG_DIR/manifest.yaml" --defenses-dir "$VIEW_DIR" \
  --out "$OUT" --repo-dir "$REPO_DIR" --relu-cell "$RELU_CELL" --lora-cell "$LORA_CELL" "${PIN_ARGS[@]}" \
  --expect-n-layers "$N_LAYERS" --batches "$BATCHES" --max-tokens "$MAX_TOKENS" --max-model-len 4096 --max-num-seqs "$S_REQ" \
  ${GPU_MEM_UTIL:+--gpu-mem-util "$GPU_MEM_UTIL"} &
TARGET_PID=$!
wait "$TARGET_PID"; HARNESS_RC=$?; TARGET_PID=""
[ "$HARNESS_RC" -eq 0 ] && check harness_rc PASS "rc=0" || check harness_rc FAIL "rc=$HARNESS_RC (see $OUT/$KEY.json status/refused/checks)"

# (5) THE POST-CHECK: the record's own facts re-read (status complete, every attestation check ok, every planned window
# error-free with an engine tok/s), printed as the one-screen summary the report quotes
POST=$(PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts/sdfbase" "$SPY" - "$OUT/$KEY.json" <<'PY'
import json, sys
import steered_canary as sc
r = json.load(open(sys.argv[1]))
p = []
if r.get("status") != "complete":
    p.append(f"status {r.get('status')!r}: {r.get('refused') or r.get('terminated')}")
failed = [c["name"] for c in (r.get("checks") or []) if not c.get("ok")]
if failed:
    p.append(f"attestation checks failed: {failed}")
plan = sc.window_plan(r["protocol"]["batches"]) if r.get("protocol") else []
p.extend(sc.rows_problems(r.get("rows") or [], plan))
f = r.get("facts") or {}
facts = {"standup_ready_s": r.get("standup_ready_s"), "geometry": f.get("geometry"), "kernel_ceiling": r.get("kernel_ceiling"),
         "attested_residual_slot": f.get("attested_residual_slot"), "worker_residual": {k: v for k, v in (f.get("worker_residual") or {}).items() if k != "hooked_boundaries"},
         "n_hooked_boundaries": len((f.get("worker_residual") or {}).get("hooked_boundaries") or []),
         "patched_classes": list((f.get("patched_classes") or {})), "carrier_attested": f.get("carrier_attested"),
         "adapter_parity_gate": {k: (f.get("adapter_parity_gate") or {}).get(k) for k in ("status", "pairs", "residual")},
         "controls": f.get("controls"), "noop_certificate": {k: (f.get("noop_certificate") or {}).get(k) for k in ("passed", "verdict")},
         "steering_bites": f.get("steering_bites"), "live": {k: (v if not isinstance(v, dict) else {"text": v.get("text")}) for k, v in (r.get("live") or {}).items()},
         "summary": r.get("summary"), "checks": [(c["name"], c["ok"]) for c in (r.get("checks") or [])]}
print(json.dumps({"problems": p, "facts": facts}, default=str))
PY
) || { check post_check FAIL "the post-check could not run: $(echo "$POST" | tail -c 4000)"; }
case "$POST" in *'"problems": []'*) check record PASS "$POST";; *) check record FAIL "$POST";; esac

f22_log "STEERED CANARY ($CHECKS):"; cat "$CHECKS"
RC=$(( NFAILED > 0 ? 2 : 0 ))
f22_log "steered canary $KEY/$VIEW relu=$RELU_CELL lora=$LORA_CELL: $NFAILED failed checks -> rc=$RC (records $OUT)"
exit $RC
