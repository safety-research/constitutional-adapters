#!/bin/bash
#SBATCH --job-name=ntime72
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# THE §7.2 TIMING HARNESS for the NATIVE residual slot:
# ONE native standup per target (scripts/sdfbase/native_timing_72.py through lora_serving_up.bring_up_lora,
# the certification driver's seam) serving the psr cell (+ none, + the zero adapter) at the lens gate's
# 4096 x 64, prefix caching OFF, VLLM_BATCH_INVARIANT unset (0), the residual slot armed, then the gate's
# closed loop per arm {none, psr} x B {8, 32, 64}: the ~800-token passage, max_tokens 128 exact, 10 s
# warm-up, 45 s window, the engine's /metrics through the proxy's launch — ms/step three ways and the
# hook-only subtraction from the SAME run. The vector arm is recorded ABSENT by name unless a 5th
# positional names a staged vector cell. Modelled on native_psr_proof.sbatch (common.sh's init, venvs,
# bank view, digest pins, RESULT line; the per-sha checkout gate is f22_init's).
# Usage (positional only — never a Slurm export flag; the job inherits the submitter's exported environment):
#   sbatch -J ntime72-<key> scripts/sdfbase/native_timing_72.sbatch <target> <view> <sha> [psr_cell] [vector_cell]
#     target    gemma4e4b | qwen35_4b | nemotron3nano4b | llama8b
#     view      psrcert | main | ...              (configs/sdfbase_full22_<key>[_<view>].yaml)
#     sha       the per-sha checkout under $SHARED (HEAD == sha)
#     psr_cell  the all-positions residual cell (cell id; default constv2-mix-full__psr_ce_all)
#     vector_cell  an all-positions vector-kind cell to time as the vector arm (stage 1: none exists)
# Records: $SHARED/exp/native_timing_72/<key>_<jobid>/ (<key>.json, standup/, windows/, checks.jsonl); the
# JSON + checks copied to $FIGS_DIR with the RESULT line. Report: scripts/sdfbase/native_timing_72_report.py.
set -uo pipefail
KEY="${1:?target}"; VIEW="${2:?view}"; WAVE_SHA="${3:?wave sha}"
PSR_CELL="${4:-constv2-mix-full__psr_ce_all}"; VECTOR_CELL="${5:-}"
source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
[ "${F22_SERVING_PATH:-lora_native}" = lora_native ] || { echo "FATAL: the timing is of the NATIVE engine; F22_SERVING_PATH=$F22_SERVING_PATH"; exit 2; }
export F22_SERVING_PATH=lora_native
# the §7.2 protocol is BI unset: a submitter's exported VLLM_BATCH_INVARIANT would be inherited by the engine
unset VLLM_BATCH_INVARIANT 2>/dev/null || true
f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"
case "$KEY" in glm46vflash) f22_fatal "glm46vflash is EXCLUDED from the native engine";; esac
OUT="$SHARED/exp/native_timing_72/${KEY}_${SLURM_JOB_ID:-manual}"; SERVE_DIR="$OUT/standup"
CHECKS="$OUT/checks.jsonl"; mkdir -p "$SERVE_DIR"; : > "$CHECKS"
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
    # the harness writes its partial JSON and tears the standup down on TERM; give it the grace window
    kill -TERM "$TARGET_PID" 2>/dev/null || true
    for i in $(seq 1 18); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done
    kill -9 "$TARGET_PID" 2>/dev/null || true
  fi
  cp -f "$OUT/$KEY.json" "$FIGS_DIR/native_timing_72_${KEY}_${SLURM_JOB_ID:-manual}.json" 2>/dev/null || true
  cp -f "$OUT/timing_prompt.txt" "$FIGS_DIR/native_timing_72_${KEY}_${SLURM_JOB_ID:-manual}_prompt.txt" 2>/dev/null || true
  cp -f "$CHECKS" "$FIGS_DIR/native_timing_72_checks_${KEY}_${SLURM_JOB_ID:-manual}.jsonl" 2>/dev/null || true
  for f in endpoint.json cells_served.json serving_attestation.json; do
    cp -f "$SERVE_DIR/$f" "$FIGS_DIR/${KEY}_${SLURM_JOB_ID:-manual}_$f" 2>/dev/null || true
  done
  tail -c 300000 "$SERVE_DIR/vllm_server.log" > "$FIGS_DIR/${KEY}_${SLURM_JOB_ID:-manual}_vllm_server.tail.log" 2>/dev/null || true
  f22_result "native-timing-72 $KEY/$VIEW psr=$PSR_CELL vector=${VECTOR_CELL:-ABSENT(D4)} harness_rc=${HARNESS_RC:-none} rc=${RC} failed_checks=${NFAILED} out=$OUT"
  kill -TERM -$$ 2>/dev/null; wait
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
f22_driver_venv
case "$KEY" in nemotron3nano4b) f22_serving_venv nemotron;; *) f22_serving_venv;; esac
f22_bank_view "$MPY"
CELLS=("$PSR_CELL"); [ -n "$VECTOR_CELL" ] && CELLS+=("$VECTOR_CELL")
for c in "${CELLS[@]}"; do
  case " ${ALL_CELLS[*]} " in *" $c "*) ;; *) f22_fatal "$c is not in $CFG explicit_cells (${ALL_CELLS[*]})";; esac
done
# the pinned object digests (native_psr_proof.sbatch's block): the view manifest's entry per cell, held to the pins
# and to the bytes on disk, then handed to the harness as --expect-object-sha256 (the standup records them)
PIN_ARGS=()
for c in "${CELLS[@]}"; do
  pin=$("$MPY" - "$TT_TWO_SI_DEFENSES_DIR" "$c" "$REPO_DIR/configs/sdfbase_full22_pins.json" "$KEY" <<'PY'
import hashlib, json, sys
view, cell, pins_path, key = sys.argv[1:]
entries = {e["cell_id"]: e for e in json.load(open(f"{view}/VIEW_MANIFEST.json"))["entries"]}
e = entries.get(cell) or sys.exit(f"no VIEW_MANIFEST entry for {cell}")
pinned = json.load(open(pins_path)).get(key, {}).get(e["staged_id"], {}).get("object_sha256")
disk = hashlib.sha256(open(f"{view}/{cell}/object.pt", "rb").read()).hexdigest()
if not (disk == e["object_sha256"] == pinned):
    sys.exit(f"digest disagreement: disk {disk[:12]} manifest {str(e['object_sha256'])[:12]} pin {str(pinned)[:12]}")
print(disk)
PY
) || f22_fatal "cannot pin $c's object digest"
  f22_log "cell $c object.pt sha256 == view manifest == pins: ${pin:0:12}"
  PIN_ARGS+=(--expect-object-sha256 "$c=$pin")
done
VEC_ARGS=(); [ -n "$VECTOR_CELL" ] && VEC_ARGS=(--vector-cell "$VECTOR_CELL")

# THE HARNESS (in-process standup through bring_up_lora; the serving venv, the checkout's src first), under srun so
# Slurm tracks the engine child; backgrounded and waited on so the TERM trap runs inside the grace window
cd "$REPO_DIR" || exit 1
PYTHONPATH="$REPO_DIR/src" srun --ntasks=1 "$SPY" scripts/sdfbase/native_timing_72.py \
  --target "$KEY" --view "$VIEW" --cell "$PSR_CELL" --out "$OUT" --repo-dir "$REPO_DIR" \
  --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" "${PIN_ARGS[@]}" "${VEC_ARGS[@]}" \
  --batches 8,32,64 --window-s 45 --warmup-s 10 --max-tokens 128 --max-model-len 4096 --max-num-seqs 64 \
  ${GPU_MEM_UTIL:+--gpu-mem-util "$GPU_MEM_UTIL"} &
TARGET_PID=$!
wait "$TARGET_PID"; HARNESS_RC=$?; TARGET_PID=""
[ "$HARNESS_RC" -eq 0 ] && check harness_rc PASS "rc=0" || check harness_rc FAIL "rc=$HARNESS_RC (see $OUT/$KEY.json status/refused)"

# THE POST-CHECK: the record's own facts through THE ONE VALIDATOR (native_timing_72.validate_timing_record — the
# producer stamped its verdict, this re-derives it, the report re-derives it again): complete status, the §7.2
# protocol declared and re-read from the launch facts, every arm x B window error-free with an engine tok/s and
# concurrency >= 0.5 B, hook-only present off the floor. A record that fails is INVALID EVIDENCE, never a table.
POST=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src:$REPO_DIR/scripts/sdfbase" "$SPY" - "$OUT/$KEY.json" <<'PY'
import json, sys
import native_timing_72 as nt
r = json.load(open(sys.argv[1]))
v = nt.validate_timing_record(r)
f = {"status": r.get("status"), "refused": r.get("refused"), "protocol_problems": r.get("protocol_problems"),
     "producer_validation": (r.get("validation") or {}).get("valid"), "n_rows": v.get("n_rows"), "arms": v.get("arms")}
la = r.get("launch") or {}
pc, bi, eng = la.get("prefix_caching") or {}, la.get("batch_invariance") or {}, la.get("engine") or {}
f["prefix_caching"] = {k: pc.get(k) for k in ("composed", "argv_off_token", "attested_enable_prefix_caching")}
f["batch_invariance"] = {k: bi.get(k) for k in ("harness_env", "launch_env", "attested_env", "attested_active", "effective")}
f["engine"] = {k: eng.get(k) for k in ("enforce_eager", "cudagraph_mode", "output_diagnostics")}
f["residual_slot"] = (eng.get("residual_slot") or {}).get("armed"); f["geometry"] = eng.get("geometry")
f["rows"] = [{"arm": x.get("arm"), "B": x.get("B"), "ms_per_step": x.get("ms_per_step"), "hook_only_ms": x.get("hook_only_ms"),
              "tok_s": x.get("tok_s"), "running_mean": x.get("running_mean"), "errors": x.get("errors"),
              "measured_window_s": x.get("measured_window_s")} for x in (r.get("rows") or [])]
f["prompt"] = r.get("prompt"); f["vector"] = (r.get("arms") or {}).get("vector")
p = list(v["problems"])
if (r.get("validation") or {}).get("valid") is not v["valid"]:
    p.append(f"the producer's stamped validation {(r.get('validation') or {}).get('valid')!r} != this re-derivation {v['valid']!r}")
print(json.dumps({"problems": p, "facts": f}, default=str))
PY
) || { check post_check FAIL "the post-check could not run: $(echo "$POST" | tail -c 4000)"; }
case "$POST" in *'"problems": []'*) check record PASS "$POST";; *) check record FAIL "$POST";; esac

f22_log "NATIVE TIMING 72 ($CHECKS):"; cat "$CHECKS"
RC=$(( NFAILED > 0 ? 2 : 0 ))
f22_log "native timing 72 $KEY/$VIEW psr=$PSR_CELL: $NFAILED failed checks -> rc=$RC (records $OUT)"
exit $RC
