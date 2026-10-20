#!/bin/bash
#SBATCH --job-name=npsr
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
# THE ONE-TARGET GPU PROOF of the residual slot: a NATIVE standup
# (mt_serving_up.py --serving-path lora_native, the wave's own front) serving the roster main-view PSR
# cell beside the target's LoRA cell and `none`, readiness attested with CUDA graphs ON and the
# residual_slot block (serving_attestation schema 2), the five positive controls (steering_bites on
# the PSR cell against its zero twin; selection_honored STRUCTURAL under the no-diagnostics policy),
# the wheel-present class-table + exact-op check (tests/test_native_residual_slot.py (j)) run in the
# serving venv BEFORE the GPU standup, and the worker's residual_slot report — every fact post-checked
# in the job and written to checks.jsonl. Modelled on native_capctrl_smoke.sbatch (common.sh's init,
# venvs, bank view, digest pins, RESULT line; the per-sha checkout gate is f22_init's).
# Usage (positional only — never --export=):
#   sbatch -J npsr-<key> scripts/sdfbase/native_psr_proof.sbatch <key> <view> <psr_cell> <lora_cell> <sha>
#     key       gemma4e4b | qwen35_4b | nemotron3nano4b | llama8b
#     view      main | it | psrcert ...        (configs/sdfbase_full22_<key>[_<view>].yaml)
#     psr_cell  the all-positions residual cell (TrainTest id), e.g. constv2-mix-full__psr_ce_all
#     lora_cell the LoRA cell served beside it, e.g. constv2-mix-full__lora_r4_ce
#     sha       the checkout $SHARED/sisteer-eval-repo/<sha> (HEAD == sha)
# Records: $SHARED/exp/native_psr_proof/<key>_<jobid>/ (standup/, checks.jsonl), the standup's endpoint /
# cells_served / positive_controls / serving_attestation / routing_table copied to $FIGS_DIR with the RESULT line.
set -uo pipefail
KEY="${1:?key}"; VIEW="${2:?view}"; PSR_CELL="${3:?the residual cell}"; LORA_CELL="${4:?the LoRA cell}"; WAVE_SHA="${5:?wave sha}"
source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
[ "${F22_SERVING_PATH:-lora_native}" = lora_native ] || { echo "FATAL: this proof is of the NATIVE engine; F22_SERVING_PATH=$F22_SERVING_PATH"; exit 2; }
export F22_SERVING_PATH=lora_native
f22_init "$WAVE_SHA"; f22_resolve_config "$KEY" "$VIEW"
case "$KEY" in glm46vflash) f22_fatal "glm46vflash is EXCLUDED from the native engine";; esac
OUT="$SHARED/exp/native_psr_proof/${KEY}_${SLURM_JOB_ID:-manual}"; SERVE_DIR="$OUT/standup"
CHECKS="$OUT/checks.jsonl"; mkdir -p "$SERVE_DIR"; : > "$CHECKS"
NFAILED=0; RC=1; TARGET_PID=""; STANDUP_STOP="$SERVE_DIR/STOP"
check(){   # name PASS|FAIL detail...
  local name="$1" st="$2"; shift 2
  [ "$st" = PASS ] || NFAILED=$((NFAILED+1))
  printf '{"check":"%s","status":"%s","detail":"%s","at":"%s","job":"%s"}\n' "$name" "$st" "$(echo "$*" | tr '"' "'" | tr '\n' ' ' | head -c 8000)" "$(date -u +%FT%TZ)" "${SLURM_JOB_ID:-manual}" >> "$CHECKS"
  f22_log "CHECK $st $name: $(echo "$*" | head -c 3000)"
}
cleanup(){
  trap '' SIGTERM
  [ -n "${STANDUP_STOP:-}" ] && touch "$STANDUP_STOP" 2>/dev/null || true
  if [ -n "${TARGET_PID:-}" ]; then
    for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done
    kill -9 "$TARGET_PID" 2>/dev/null || true
  fi
  for f in endpoint.recorded.json cells_served.json positive_controls.json serving_attestation.json routing_table.json; do
    cp -f "$SERVE_DIR/$f" "$FIGS_DIR/${KEY}_${SLURM_JOB_ID:-manual}_$f" 2>/dev/null || true
  done
  cp -f "$CHECKS" "$FIGS_DIR/native_psr_proof_checks_${KEY}_${SLURM_JOB_ID:-manual}.jsonl" 2>/dev/null || true
  tail -c 300000 "$SERVE_DIR/vllm_server.log" > "$FIGS_DIR/${KEY}_${SLURM_JOB_ID:-manual}_vllm_server.tail.log" 2>/dev/null || true
  f22_result "native-psr-proof $KEY/$VIEW psr=$PSR_CELL lora=$LORA_CELL rc=${RC} failed_checks=${NFAILED} out=$OUT"
  kill -TERM -$$ 2>/dev/null; wait
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
f22_driver_venv
case "$KEY" in nemotron3nano4b) f22_serving_venv nemotron;; *) f22_serving_venv;; esac
f22_bank_view "$MPY"
for c in "$PSR_CELL" "$LORA_CELL"; do
  case " ${ALL_CELLS[*]} " in *" $c "*) ;; *) f22_fatal "$c is not in $CFG explicit_cells (${ALL_CELLS[*]})";; esac
done
# the pinned object digests (native_capctrl_smoke's rule): the view manifest's entry per cell, held to the pins
# and to the bytes on disk, then handed to the standup as --expect-object-sha256 (recorded in endpoint.json)
PIN_ARGS=()
for c in "$PSR_CELL" "$LORA_CELL"; do
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

# (1) THE WHEEL-PRESENT CHECK (plan step 1 (j)) in the serving venv BEFORE the GPU standup: the class table resolves
# against the wheel, every defining class's forward returns the pair (or gemma's (h, None)), the op registers into
# vLLM's own library under the ONE name (torch.ops.sisteer.* does NOT resolve), its fake impl and its eager impl
# (on the allocated device) both run, the where-form write-back leaves an unsteered row bit-identical.
WHEEL=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$SPY" - <<'PY' 2>&1
import inspect, json, torch
from sisteer.capeval.serving import native_residual as nr
facts = {}
targets = nr.vllm_patch_targets()
names = sorted(c.__name__ for c in targets.decoder_layer_classes)
facts["classes"] = names
assert names == sorted(n for _, n in nr.VLLM_DECODER_LAYER_CLASSES), names
for cls in targets.decoder_layer_classes:
    own = cls.__dict__.get("forward")
    if own is None:
        facts.setdefault("inherit_forward", []).append(cls.__name__)
        continue
    src = inspect.getsource(own)
    assert ("return hidden_states, residual" in src or "return output, residual" in src
            or "return hidden_states, None" in src), cls.__name__
op = nr.ensure_op_registered()
assert op is torch.ops.vllm.sisteer_residual_slot and nr.op_registered()
facts["op"] = str(torch.ops.vllm.sisteer_residual_slot.default._schema)
try:
    torch.ops.sisteer.residual_slot
    raise SystemExit("KNOWN-BAD: torch.ops.sisteer.residual_slot resolved")
except AttributeError:
    pass
from torch._subclasses.fake_tensor import FakeTensorMode
with FakeTensorMode():
    fo = op(torch.empty(5, 16, dtype=torch.bfloat16), None, torch.arange(5), 0)
    assert tuple(fo.shape) == (5, 16), fo.shape
dev = "cuda" if torch.cuda.is_available() else "cpu"
nr.reset_for_tests()
bank = nr.ResidualSlotBank(n_slots=2, n_boundaries=3, hidden_size=16, max_tokens=8, device=dev, model_dtype=torch.bfloat16)
nr._publish_bank(bank)
g = torch.Generator().manual_seed(1)
blk = {"schema": 1, "kind": "psr", "op": nr.OP_RELU_GATED_VECTOR, "layers": [0, 2], "boundaries": [0, 2],
       "position_mode": "all", "response_header_len": None, "response_header_ids": [], "skip_bos": False,
       "coeff": 1.0, "hidden_size": 16, "n_layers_model": 3, "anchor_layer": 0, "dtype_policy": nr.DTYPE_POLICY_GATED}
tens = {}
for l in (0, 2):
    tens[f"sisteer_residual.layers.{l}.gate_w"] = torch.randn(16, generator=g)
    tens[f"sisteer_residual.layers.{l}.gate_b"] = torch.full((1,), 200.0)
    tens[f"sisteer_residual.layers.{l}.direction"] = torch.randn(16, generator=g)
bank.load_slot(1, nr.payload_from_block_and_tensors(blk, tens), lora_int_id=4)
bank.update_mapping(torch.tensor([1, -1, 1], device=dev), 3)
h = (torch.randn(3, 16, generator=g) * 2).to(torch.bfloat16).to(dev)
r = (torch.randn(3, 16, generator=g) * 2).to(torch.bfloat16).to(dev)
got = op(h, r, torch.arange(3, device=dev), 2)
assert torch.equal(got[1], h[1]) and not torch.equal(got[0], h[0]), "eager impl: unsteered row moved or steered row unchanged"
assert torch.equal(op(h, r, torch.arange(3, device=dev), 1), h), "inactive boundary must be a bit-exact no-op"
facts["eager_device"] = dev; facts["op_invocations"] = bank.op_invocations
nr.reset_for_tests()
print(json.dumps({"ok": True, "facts": facts}))
PY
)
case "$WHEEL" in *'"ok": true'*) check wheel_class_table_and_op PASS "$WHEEL";; *) check wheel_class_table_and_op FAIL "$WHEEL"; exit 1;; esac

# (2) THE NATIVE STANDUP (the wave's front, its default engine): the psr cell + the LoRA cell (+ none, the zero
# twin), the static families' geometry entry, the pinned digests, the residual slot armed by the composer
rm -f "$SERVE_DIR/STOP" "$SERVE_DIR/endpoint.json"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}" "$SPY" scripts/sdfbase/mt_serving_up.py --repo-dir "$REPO_DIR" \
  --config "$CFG" --manifest "$MANIFEST" --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" \
  --out "$SERVE_DIR" --stop-file "$SERVE_DIR/STOP" --mode attacks --eval families_served:short --model-key "$KEY" \
  --serving-path lora_native --cell "$PSR_CELL" --cell "$LORA_CELL" "${PIN_ARGS[@]}" --text-only-mm auto \
  ${GPU_MEM_UTIL:+--gpu-mem-util "$GPU_MEM_UTIL"} \
  > "$SERVE_DIR/standup.log" 2>&1 &
TARGET_PID=$!
for i in $(seq 1 480); do
  [ -f "$SERVE_DIR/endpoint.json" ] && break
  kill -0 "$TARGET_PID" 2>/dev/null || { check standup FAIL "the native standup died: $(tail -40 "$SERVE_DIR/standup.log" | tr '\n' ' ' | head -c 6000)"; exit 1; }
  sleep 5
done
[ -f "$SERVE_DIR/endpoint.json" ] || { check standup FAIL "no endpoint.json after 40 min: $(tail -20 "$SERVE_DIR/standup.log" | tr '\n' ' ' | head -c 3000)"; exit 1; }
cp -f "$SERVE_DIR/endpoint.json" "$SERVE_DIR/endpoint.recorded.json"
EP="$SERVE_DIR/endpoint.json"
PROXY_URL=$("$SPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['proxy_url'])" "$EP")

# (3) THE POST-CHECK: every fact the proof claims, read from the AUTHORITATIVE records and the live proxy
POST=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$SPY" - "$EP" "$KEY" "$PROXY_URL" "$PSR_CELL" "$LORA_CELL" "$SERVE_DIR" <<'PY'
import json, sys, urllib.request
from pathlib import Path
ep_path, key, proxy, psr_tt, lora_tt, serve_dir = sys.argv[1:]
ep = json.load(open(ep_path))
from sisteer.capeval.petri import registry_adapter as ra
from sisteer.capeval.serving import serving_attestation as sa
from sisteer.capeval.serving.proxy_engine import served_engine_problem
psr_h, lora_h = ra.harness_cell_id(psr_tt), ra.harness_cell_id(lora_tt)
problems, facts = [], {}
facts["cells"] = {"psr": psr_h, "lora": lora_h}
facts["engine"] = ep.get("engine"); facts["serving_engine"] = ep.get("serving_engine")
if ep.get("engine") != "lora_native" or ep.get("serving_engine") != "lora_native":
    problems.append(f"endpoint engine={ep.get('engine')!r} serving_engine={ep.get('serving_engine')!r}")
if "lens_reduction_mode" in ep:
    problems.append("a native endpoint carries lens_reduction_mode")
facts["attested_enforce_eager"], facts["attested_cudagraph_mode"] = ep.get("attested_enforce_eager"), ep.get("attested_cudagraph_mode")
if ep.get("attested_enforce_eager") is not False or ep.get("attested_cudagraph_mode") in (None, "NONE"):
    problems.append("the engine is not graphs-on")
facts["geometry"] = {"L": ep.get("max_model_len"), "S": ep.get("max_num_seqs"), "attested_S": ep.get("attested_max_num_seqs"),
                     "adjustments": (ep.get("geometry") or {}).get("adjustments")}
facts["cells_skipped"] = ep.get("cells_skipped"); facts["lora_cells"] = ep.get("lora_cells")
if ep.get("cells_skipped"):
    problems.append(f"cells_skipped is not empty: {ep.get('cells_skipped')}")
if set(ep.get("lora_cells") or []) != {psr_h, lora_h}:
    problems.append(f"lora_cells {ep.get('lora_cells')} != the two served cells")
facts["output_diagnostics"] = ep.get("output_diagnostics")
if ep.get("output_diagnostics") is not False:
    problems.append("output_diagnostics is not False on a residual-bearing standup (D19)")
rs = ep.get("residual_slot") or {}
facts["residual_slot"] = rs
if rs.get("armed") is not True or set(rs.get("cells") or {}) != {psr_h}:
    problems.append(f"residual_slot {rs} does not name exactly the psr cell, armed")
psr_fact = (rs.get("cells") or {}).get(psr_h) or {}
if psr_fact.get("kind") != "psr" or psr_fact.get("position_mode") != "all":
    problems.append(f"psr cell fact {psr_fact}")
ad = ep.get("adapters") or {}
facts["adapter_blocks"] = {c: {k: (b or {}).get(k) for k in ("name", "r", "petri_source_kind", "petri_zero_adapter")} for c, b in ad.items()}
psr_block, lora_block = ad.get(psr_h) or {}, ad.get(lora_h) or {}
if psr_block.get("petri_source_kind") != "psr" or psr_block.get("r") != 1:
    problems.append(f"the psr adapter block is not a rank-1 psr entry: {psr_block}")
psr_name, lora_name = psr_block.get("name"), lora_block.get("name")
# the attestation (schema 2) and the worker's residual report
att = sa.read_serving_attestation(Path(ep["serving_attestation"]))
facts["schema_version"] = att["schema_version"]
if att["schema_version"] != 2:
    problems.append(f"attestation schema {att['schema_version']}")
blk = att[sa.RESIDUAL_SLOT_KEY]
facts["attested_residual_slot"] = {k: blk.get(k) for k in ("armed", "rpc_method", "n_workers")}
facts["attested_residual_expected"] = {k: v for k, v in (blk.get("expected") or {}).items() if k != "adapters"}
if blk.get("armed") is not True:
    problems.append("attestation residual_slot.armed is not True")
w0 = (blk.get("worker_reports") or [{}])[0]
facts["worker_residual"] = {k: w0.get(k) for k in ("armed", "op_name", "op_registered", "runner_bound", "n_layers_hooked",
                                                    "n_layers_model", "hooked_boundaries", "ledger_size", "ledger_fault", "installs", "error")}
facts["worker_bank"] = w0.get("bank")
facts["worker_slots"] = w0.get("slots")
facts["patched_classes"] = w0.get("patched_classes")
facts["machinery"] = w0.get("machinery")
if not w0.get("n_layers_hooked") or w0.get("n_layers_hooked") != w0.get("n_layers_model"):
    problems.append(f"hooked {w0.get('n_layers_hooked')} of {w0.get('n_layers_model')} blocks")
if not any(e.get("forward_wrapped") for e in (w0.get("patched_classes") or {}).values()):
    problems.append("no decoder class reports a wrapped forward")
if int((w0.get("bank") or {}).get("op_invocations") or 0) <= 0:
    problems.append("the residual op never ran on the worker (op_invocations 0)")
if w0.get("ledger_fault") or w0.get("ledger_size"):
    problems.append(f"the request-failure ledger is not clean: size={w0.get('ledger_size')} fault={w0.get('ledger_fault')}")
psr_att = (att.get("adapters") or {}).get(psr_name) or {}
lora_att = (att.get("adapters") or {}).get(lora_name) or {}
facts["psr_attested"] = {k: psr_att.get(k) for k in ("petri_source_kind", "residual_fingerprint", "position_mode", "r", "lora_int_id", "fingerprint_bf16")}
facts["lora_attested"] = {k: lora_att.get(k) for k in ("petri_source_kind", "residual_fingerprint", "position_mode", "r", "lora_int_id")}
psr_id = str(psr_att.get("lora_int_id"))
if psr_id not in (w0.get("slots") or {}):
    problems.append(f"no worker slot for the psr adapter (lora_int_id {psr_id}; slots {list((w0.get('slots') or {}))})")
elif w0["slots"][psr_id].get("fingerprint") != psr_att.get("residual_fingerprint"):
    problems.append("the worker's slot fingerprint != the attested residual_fingerprint")
if psr_att.get("residual_fingerprint") is None or psr_att.get("position_mode") != "all" or psr_att.get("petri_source_kind") != "psr":
    problems.append(f"the attested psr adapter is not a residual entry in mode all: {facts['psr_attested']}")
if lora_att.get("residual_fingerprint") is not None or lora_att.get("position_mode") is not None:
    problems.append(f"the attested LoRA adapter carries residual facts: {facts['lora_attested']}")
# the live proxy: engine kind + the per-adapter kind on /v1/models
ident = json.load(urllib.request.urlopen(proxy + "/models", timeout=60)).get("petri_proxy") or {}
p = served_engine_problem(ident, expect="lora_native")
if p: problems.append(f"/v1/models: {p}")
eng = ident.get("engine") or {}
facts["proxy_residual_slot"] = eng.get("residual_slot")
facts["proxy_psr_entry"] = (eng.get("adapters") or {}).get(psr_name)
if (eng.get("residual_slot") or {}).get("armed") is not True or ((eng.get("adapters") or {}).get(psr_name) or {}).get("petri_source_kind") != "psr":
    problems.append("/v1/models engine block does not carry the armed slot / the psr kind")
# the five positive controls, steering_bites on the psr cell against its twin
pc = json.load(open(ep["positive_controls"]))
facts["controls"] = {n: (c.get("passed"), c.get("verdict")) for n, c in (pc.get("controls") or {}).items()}
facts["controls_output_diagnostics"] = pc.get("output_diagnostics")
facts["controls_order"] = pc.get("order")
if pc.get("passed") is not True:
    problems.append(f"positive controls did not pass: {pc.get('failed_control')} {str(pc.get('error'))[:400]}")
bites = (pc.get("controls") or {}).get("steering_bites") or {}
facts["steering_bites"] = bites
psr_bite = (bites.get("cells") or {}).get(psr_h) or {}
if not psr_bite or not psr_bite.get("n_differing"):
    problems.append(f"steering_bites carries no differing probe for the psr cell: {psr_bite}")
sel = (pc.get("controls") or {}).get("selection_honored") or {}
facts["selection_honored"] = {k: sel.get(k) for k in ("verdict", "n_requests", "n_observed", "n_misrouted", "output_diagnostics")}
if sel.get("verdict") != "routing_attested" or sel.get("output_diagnostics") is not False or "numerics" in sel:
    problems.append(f"selection_honored: verdict={sel.get('verdict')} output_diagnostics={sel.get('output_diagnostics')} numerics_present={'numerics' in sel}")
# cells_served.json carries the kind per adapter and the residual_slot block
cs = json.load(open(ep["cells_served"]))
facts["cells_served"] = {"engine": cs.get("engine"), "cells_skipped": cs.get("cells_skipped"), "lora_cells": cs.get("lora_cells"),
                         "output_diagnostics": cs.get("output_diagnostics"), "residual_slot": cs.get("residual_slot"),
                         "kinds": {c: (b or {}).get("petri_source_kind") for c, b in (cs.get("adapters") or {}).items()}}
if facts["cells_served"]["kinds"].get(psr_h) != "psr" or (cs.get("residual_slot") or {}).get("armed") is not True:
    problems.append(f"cells_served.json does not carry the psr kind / the armed slot: {facts['cells_served']}")
# the worker log line is a FACT, never a gate (job 507580: the `sisteer` logger's INFO line does not reach
# vllm_server.log; the AUTHORITATIVE arming fact is the worker's RPC report above, armed + op_registered)
log = open(f"{serve_dir}/vllm_server.log", errors="replace").read()
facts["armed_log_line"] = "residual slot ARMED" in log
facts["worker_extension_injected"] = "NativeWorkerExtension" in log
facts["vmodels"] = {"base": pc.get("base_vmodel"), "zero": pc.get("zero_vmodel"), "psr": (ep.get("vmodels") or {}).get(psr_h), "lora": (ep.get("vmodels") or {}).get(lora_h)}
print(json.dumps({"problems": problems, "facts": facts}, default=str))
PY
) || { check post_check FAIL "the post-check could not run: $(echo "$POST" | tail -c 4000)"; exit 1; }
case "$POST" in *'"problems": []'*) check standup PASS "$POST";; *) check standup FAIL "$POST"; exit 1;; esac

# (4) a live steered completion through the proxy on base, the psr vmodel and the LoRA vmodel, recorded (a smoke of
# the served route beyond the controls' probes; greedy, so the psr text differing from base is the slot biting)
LIVE=$(cd "$REPO_DIR" && PYTHONPATH="$REPO_DIR/src" "$SPY" - "$EP" "$PROXY_URL" "$PSR_CELL" "$LORA_CELL" <<'PY' 2>&1
import json, sys, urllib.request
ep_path, proxy, psr_tt, lora_tt = sys.argv[1:]
ep = json.load(open(ep_path)); pc = json.load(open(ep["positive_controls"]))
from sisteer.capeval.petri import registry_adapter as ra
vm = ep.get("vmodels") or {}
targets = {"base": pc["base_vmodel"], "psr": vm[ra.harness_cell_id(psr_tt)], "lora": vm[ra.harness_cell_id(lora_tt)]}
out = {}
for label, vmodel in targets.items():
    body = json.dumps({"model": vmodel, "messages": [{"role": "user", "content": "Give one sentence of advice about learning to program."}],
                       "max_tokens": 48, "temperature": 0.0}).encode()
    req = urllib.request.Request(proxy + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    resp = json.load(urllib.request.urlopen(req, timeout=180))
    out[label] = {"vmodel": vmodel, "text": resp["choices"][0]["message"]["content"][:240],
                  "petri": {k: (resp.get("petri") or {}).get(k) for k in ("virtual_model", "cell_id", "engine")}}
out["psr_differs_from_base"] = out["psr"]["text"] != out["base"]["text"]
print(json.dumps({"ok": True, "live": out}, default=str))
PY
)
case "$LIVE" in *'"ok": true'*) check live_completions PASS "$LIVE";; *) check live_completions FAIL "$LIVE";; esac

touch "$STANDUP_STOP"; for i in $(seq 1 36); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done
f22_log "NATIVE PSR PROOF ($CHECKS):"; cat "$CHECKS"
RC=$(( NFAILED > 0 ? 2 : 0 ))
f22_log "native psr proof $KEY/$VIEW psr=$PSR_CELL lora=$LORA_CELL: $NFAILED failed checks -> rc=$RC (records $OUT)"
exit $RC
