#!/bin/bash
#SBATCH --job-name=natcap
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
# native-serving-default: the CLUSTER-SIDE proof of the single-cell capctrl standup on the NATIVE engine:
# scripts/capctrl_serving_up.py under its DEFAULT serving path (lora_native) on ONE (target, view, cell) —
# for nemotron3nano4b the Mamba-hybrid target with the reviewed backend contract, so the PORTED
# sisteer_mamba_backend_report answers on the plugin-free worker and mamba_backend.live_attest passes
# (endpoint.json.mamba_backend_attestation.ok), the §1.8 kernel step FIRES (161 -> 86 at the 4864 window:
# live evidence for the adjustment contract) and is validated by the in-job post-check through
# geometry.geometry_adjustment_problem, readiness attests graphs ON; then the served capability driver
# (scripts/served_capability_run.py generate --controls mmlu --limit 8) against it and the post-check of
# its store (serving_engine lora_native, reduction_mode not_applicable, the twin none__lora0).
# Also runnable for llama8b (no kernel step at 4864: adjustments [] — the identity native standup) as the
# cluster-side check of the capctrl CLI path.
# Usage (positional only — never --export=):
#   sbatch -J natcap-<key> scripts/sdfbase/native_capctrl_smoke.sbatch <key> <view> <cell> <wave_sha>
#     key   nemotron3nano4b | llama8b | gemma4e4b | qwen35_4b   (geometry.MODEL_KEYS; glm46vflash is excluded)
#     view  main | it | ...   (configs/sdfbase_full22_<key>[_<view>].yaml)
#     cell  the LoRA cell to serve (its pinned object.pt sha is read off the bank view's VIEW_MANIFEST)
#     sha   the wave sha (the per-sha checkout under $SHARED)
# Records: $SHARED/exp/native_capctrl_smoke/<key>_<jobid>/ (standup/, the driver's store, checks.jsonl, RESULT),
# the RESULT line mirrored beside the figures so they show the outcome (common.sh f22_result).
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; CELL="${3:?the LoRA cell}"; WAVE_SHA="${4:?wave sha}"
source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
# the native engine is the point of this job: refuse any inherited lens selection rather than run the wrong proof
[ "${F22_SERVING_PATH:-lora_native}" = lora_native ] || { echo "FATAL: this smoke proves the NATIVE engine; F22_SERVING_PATH=$F22_SERVING_PATH"; exit 2; }
export F22_SERVING_PATH=lora_native
f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"
case "$KEY" in glm46vflash) f22_fatal "glm46vflash is EXCLUDED from the native engine (its M-RoPE patch rides the lens plugin)";; esac
OUT="$SHARED/exp/native_capctrl_smoke/${KEY}_${SLURM_JOB_ID:-manual}"; SERVE_DIR="$OUT/standup"; STORE="$OUT/store"
CHECKS="$OUT/checks.jsonl"; mkdir -p "$SERVE_DIR" "$STORE"; : > "$CHECKS"
NFAILED=0; RC=1; TARGET_PID=""; STANDUP_STOP="$SERVE_DIR/STOP"
check(){   # name PASS|FAIL detail...
  local name="$1" st="$2"; shift 2
  [ "$st" = PASS ] || NFAILED=$((NFAILED+1))
  printf '{"check":"%s","status":"%s","detail":"%s","at":"%s","job":"%s"}\n' "$name" "$st" "$(echo "$*" | tr '"' "'" | tr '\n' ' ')" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-manual}" >> "$CHECKS"
  f22_log "CHECK $st $name: $*"
}
cleanup(){
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  cp "$CHECKS" "$FIGS_DIR/native_capctrl_checks_${KEY}_${SLURM_JOB_ID:-manual}.jsonl" 2>/dev/null || true
  f22_result "native-capctrl-smoke $KEY/$VIEW cell=$CELL rc=${RC} failed_checks=${NFAILED} out=$OUT"
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
# the venvs: the driver's, the serving venv (the -nemotron kernel overlay for the Mamba-hybrid target, exactly as
# multilora_cert.sbatch builds it — the fused kernels are a PREREQUISITE, never a fallback), the bank view
f22_driver_venv
case "$KEY" in nemotron3nano4b) f22_serving_venv nemotron;; *) f22_serving_venv;; esac
f22_bank_view "$MPY"
case " ${ALL_CELLS[*]} " in *" $CELL "*) ;; *) f22_fatal "$CELL is not in $CFG explicit_cells (${ALL_CELLS[*]})";; esac
# the pinned object digest: the view manifest's entry for the cell, held to the pins and to the served bytes
PIN=$("$MPY" - "$TT_TWO_SI_DEFENSES_DIR" "$CELL" "$REPO_DIR/configs/sdfbase_full22_pins.json" "$KEY" <<'PY'
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
) || f22_fatal "cannot pin $CELL's object digest"
f22_log "cell $CELL object.pt sha256 == view manifest == pins: ${PIN:0:12}"
CONTRACT="configs/capeval/mamba_backend_contract.json"
[ -f "$REPO_DIR/$CONTRACT" ] || f22_fatal "no Nemotron backend contract at $REPO_DIR/$CONTRACT"
# THE NATIVE CAPCTRL STANDUP (the CLI path, its DEFAULT serving path): the static families' geometry entry at the
# model's budget (families_served:short = 4864 x budget-S; the LoRA kernel ceiling lowers a hybrid's S there),
# --mrope-attest recorded not_applicable, the contract handed in, tool calls OFF (a capability standup)
rm -f "$SERVE_DIR/STOP" "$SERVE_DIR/endpoint.json"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "$SPY" scripts/capctrl_serving_up.py --repo-dir "$REPO_DIR" \
  --config "$CFG" --cell "$CELL" --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out "$SERVE_DIR" --stop-file "$SERVE_DIR/STOP" \
  --eval families_served:short --model-key "$KEY" --serving-path lora_native \
  --expect-object-sha256 "$CELL=$PIN" --manifest "$MANIFEST" --mamba-backend-contract "$CONTRACT" \
  --text-only-mm auto --max-logprobs 64 ${GPU_MEM_UTIL:+--gpu-mem-util "$GPU_MEM_UTIL"} \
  > "$SERVE_DIR/standup.log" 2>&1 &
TARGET_PID=$!
for i in $(seq 1 480); do
  [ -f "$SERVE_DIR/endpoint.json" ] && break
  kill -0 "$TARGET_PID" 2>/dev/null || { check standup FAIL "the native capctrl standup died: $(tail -20 "$SERVE_DIR/standup.log" | tr '\n' ' ' | head -c 1500)"; exit 1; }
  sleep 5
done
[ -f "$SERVE_DIR/endpoint.json" ] || { check standup FAIL "no endpoint.json after 40 min"; exit 1; }
cp -f "$SERVE_DIR/endpoint.json" "$SERVE_DIR/endpoint.recorded.json"
EP="$SERVE_DIR/endpoint.json"
PROXY_URL=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['proxy_url'])" "$EP")
TABLE=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['routing_table'])" "$EP")
# THE POST-CHECK: the ENGINE from the authoritative stamps, the Nemotron backend attestation over the
# ported RPC, the adjustment chain re-derived through geometry.geometry_adjustment_problem, graphs on
POST=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$SPY" - "$EP" "$KEY" "$PROXY_URL" <<'PY'
import json, sys, urllib.request
ep, key, proxy = json.load(open(sys.argv[1])), sys.argv[2], sys.argv[3]
from sisteer.capeval.serving import geometry as sg
from sisteer.capeval.serving import mamba_backend as mb
problems, facts = [], {}
facts["engine"] = ep.get("engine"); facts["serving_engine"] = ep.get("serving_engine")
if ep.get("engine") != "lora_native" or ep.get("serving_engine") != "lora_native":
    problems.append(f"endpoint engine={ep.get('engine')!r} serving_engine={ep.get('serving_engine')!r}")
if "lens_reduction_mode" in ep:
    problems.append("a native endpoint carries lens_reduction_mode")
facts["attested_enforce_eager"], facts["attested_cudagraph_mode"] = ep.get("attested_enforce_eager"), ep.get("attested_cudagraph_mode")
if ep.get("attested_enforce_eager") is not False or ep.get("attested_cudagraph_mode") in (None, "NONE"):
    problems.append("the engine is not graphs-on")
ident = json.load(urllib.request.urlopen(proxy + "/models", timeout=60)).get("petri_proxy") or {}
from sisteer.capeval.serving.proxy_engine import served_engine_problem
p = served_engine_problem(ident, expect="lora_native")
facts["proxy_engine_kind"] = (ident.get("engine") or {}).get("kind")
if p: problems.append(f"/v1/models: {p}")
geom = ep.get("geometry") or {}
L, S = int(ep["max_model_len"]), int(ep["max_num_seqs"])
S_id = sg.resolve_geometry("families_served:short", model_key=key, strict_identity=True).max_num_seqs
facts["geometry"] = {"L": L, "S_served": S, "S_identity": S_id, "adjustments": geom.get("adjustments")}
gp = sg.geometry_adjustment_problem(geom, model_key=key, L=L, S_identity=S_id, S_served=S, native=True)
if gp: problems.append(f"geometry adjustments: {gp}")
expect_kernel = sg.kernel_ceiling_adjustment(key, L, S_id)
kinds = [a.get("kind") for a in (geom.get("adjustments") or [])]
if (expect_kernel is not None) != ("lora_kernel_index_ceiling" in kinds):
    problems.append(f"kernel step expected={expect_kernel} recorded kinds={kinds}")
facts["kernel_step_fired"] = expect_kernel is not None
if key == "nemotron3nano4b":
    att = ep.get("mamba_backend_attestation")
    facts["mamba_backend_attestation"] = {k: (att or {}).get(k) for k in ("ok", "rpc_method", "contract_sha256", "expected_n_mamba_layers")} if isinstance(att, dict) else att
    if not (isinstance(att, dict) and att.get("ok") is True and att.get("rpc_method") == mb.RPC_METHOD and att.get("reports")):
        problems.append(f"mamba backend attestation not passing: {str(att)[:400]} ({ep.get('mamba_backend_attestation_error')})")
    ep_probs = mb.endpoint_attestation_problems(ep)
    if ep_probs: problems.append(f"endpoint_attestation_problems: {ep_probs}")
facts["cells_skipped"] = ep.get("cells_skipped"); facts["served_models"] = ep.get("served_models")
facts["controls"] = {n: (c.get("passed"), c.get("verdict")) for n, c in (json.load(open(ep["positive_controls"])).get("controls") or {}).items()}
print(json.dumps({"problems": problems, "facts": facts}, default=str))
PY
) || { check post_check FAIL "the post-check could not run"; exit 1; }
case "$POST" in *'"problems": []'*) check standup PASS "$POST";; *) check standup FAIL "$POST"; exit 1;; esac
# the served capability driver against the native standup (the tinycap surface's own driver), then its store
CTL_TOP_LOGPROBS=$("$MPY" -c 'import importlib.util,sys; s=importlib.util.spec_from_file_location("scr", sys.argv[1]); m=importlib.util.module_from_spec(s); sys.modules["scr"]=m; s.loader.exec_module(m); print(int(m.MMLU_TOP_LOGPROBS))' scripts/served_capability_run.py) || CTL_TOP_LOGPROBS=64
if ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/served_capability_run.py generate \
      --proxy-url "$PROXY_URL" --routing-table "$TABLE" --cell "$CELL" --config "$CFG" \
      --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" --out-root "$STORE" --controls mmlu --concurrency 8 --limit 8 \
      --top-logprobs "$CTL_TOP_LOGPROBS" --vllm-log "$SERVE_DIR/vllm_server.log" ) > "$OUT/driver.log" 2>&1; then
  # THE DRIVER POST-CHECK: scripts/sdfbase/native_capctrl_postcheck.py — the REAL schema of an MMLU-only
  # run's store (the legacy controls path writes NO tinycap `route` block; the engine is the manifest's TOP-LEVEL
  # serving_engine and the HASHED resolved_config.serving.serving_engine; the bites / MMLU evidence lives in the store's
  # evidence.json; the reduction mode not_applicable is the STANDUP's endpoint.json fact; the positive controls are the
  # standup's controls record), validated on CPU against a store the real driver wrote (tests/test_native_capctrl_
  # postcheck.py) BEFORE this job spends its GPU allocation. The inline check it replaces required manifest.route.
  # {serving_engine, reduction_mode} and would have FAILED a correct run.
  DRV=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$MPY" scripts/sdfbase/native_capctrl_postcheck.py --store "$STORE" --endpoint "$EP" --cell "$CELL" 2>&1)
  case "$DRV" in *'"problems": []'*) check driver PASS "$DRV";; *) check driver FAIL "$DRV";; esac
else
  check driver FAIL "$(tail -12 "$OUT/driver.log" | tr '\n' ' ' | head -c 1200)"
fi
touch "$STANDUP_STOP"; for i in $(seq 1 36); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done
f22_log "NATIVE CAPCTRL SMOKE ($CHECKS):"; cat "$CHECKS"
RC=$(( NFAILED > 0 ? 2 : 0 ))
f22_log "native capctrl smoke $KEY/$VIEW cell=$CELL: $NFAILED failed checks -> rc=$RC (records $OUT)"
exit $RC
