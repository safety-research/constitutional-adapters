#!/bin/bash
#SBATCH --job-name=f22-misalign-em
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=0-03:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# em-misalign (WP-C): the Betley 8Q eval over the EM MODEL ORGANISMS — plain vLLM (no proxy, no lens):
# unsloth/Llama-3.1-8B-Instruct served as `llama31-8b-instruct` with the three EM LoRA adapters as
# --lora-modules (em-bad-medical-advice / em-risky-financial-advice / em-extreme-sports, from the OFFLINE HF
# cache: ModelOrganismsForEM/Llama-3.1-8B-Instruct_{bad-medical-advice,risky-financial-advice,extreme-sports}),
# then sisteer.capeval.misalign.runner --plain, family em8q, N=50 samples x 600 tokens, over the pseudo-cells
# {none, prompted__safety, prompted__constitution} x {base, each organism}: cell_id `<served>::<arm>`,
# virtual_model = the served name, si_id on the prompted arms (the SI text as the system message; no proxy,
# so no hash check — the intervention_hash IS sha16(SI bytes), asserted equal to the wave's recorded prefix
# hashes), serving_identity `vllm+lora+plain`. Outputs: $SHARED/exp/misalign/em_organisms/{serving,em8q}/,
# cells.json, serving.json, RESULT. Judged on the grading host afterwards (misalign_judge.py).
#
# Usage (positional only — never --export=; run from the staged checkout, see README.md):
#   ssh <login host> 'cd $SISTEER_EVAL_REPO_ROOT/<sha> && sbatch --qos=high \
#     -J f22-misalign-em scripts/sdfbase/full22/misalign/misalign_em_organisms.sbatch <sha>'
#   (cluster B: its own QoS name and --output=$SHARED/exp/logs/%x_%j.out; its per-user HF cache holds the
#    base + the three adapters as of 2026-09-10.)
#
# Venvs: the wave's serving venv (common.sh f22_serving_venv — vllm 0.25.1+cu129 from
# configs/capeval/vllm-requirements.txt, OUTSIDE the project lockfile) runs `vllm serve`; the driver venv
# (f22_driver_venv, `uv sync --frozen --extra cu128` at <sha>) runs the runner. VLLM_PLUGINS is set EMPTY for
# this launch so vllm-lens (a general plugin in that venv) is not loaded into a server that serves no lens
# registry: a plain engine, the same stack the agentic_serve_run reference used. `--generation-config vllm`
# keeps sampling defaults the engine's (the checkpoint's generation_config.json would otherwise inject
# top_p 0.9 / temperature 0.6 under the runner's explicit T=1.0). This file lives under scripts/sdfbase/full22/,
# outside tests/test_torch_flavors.py's scripts/sbatch audit, like the wave's own jobs; no audit-exempt tag.
set -uo pipefail
WAVE_SHA="${1:?wave sha}"
{ [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_CELLS=()   # this job stands up no target: no steered cell to select a serving path for (the kind-aware default, 2026-09-15)
f22_init "$WAVE_SHA"
MIS_ROOT="$SHARED/exp/misalign/em_organisms"; OUT="$MIS_ROOT/serving"; mkdir -p "$OUT" "$MIS_ROOT/em8q"
f22_stop_requested && { f22_log "wave STOP present — not starting"; exit 0; }
[ -e "$MIS_ROOT/STOP" ] && { f22_log "$MIS_ROOT/STOP present — not starting (remove it to run)"; exit 0; }
f22_driver_venv; f22_serving_venv

BASE_REPO="unsloth/Llama-3.1-8B-Instruct"; SERVED="llama31-8b-instruct"
ORG_NAMES=(em-bad-medical-advice em-risky-financial-advice em-extreme-sports)
ORG_REPOS=(ModelOrganismsForEM/Llama-3.1-8B-Instruct_bad-medical-advice
           ModelOrganismsForEM/Llama-3.1-8B-Instruct_risky-financial-advice
           ModelOrganismsForEM/Llama-3.1-8B-Instruct_extreme-sports)
# --- snapshots from the OFFLINE cache (HF_HUB_OFFLINE=1 from f22_init; hf_env.sh set HF_HOME) -------------------
# Resolve a repo to a COMPLETE cached snapshot dir. `snapshot_download(local_files_only=True)` needs
# refs/main to name a present snapshot; in the shared cohort cache that ref can be missing or stale
# (a peer fetched a pinned revision) -- job 478869 died that way on the EM adapters. So: refs/main when
# it points at a complete snapshot, else the newest snapshot holding config/adapter_config + weights.
snap(){ "$SPY" - "$1" <<'PYSNAP'
import os, sys, glob
from huggingface_hub.constants import HF_HUB_CACHE
repo = sys.argv[1]; d = os.path.join(HF_HUB_CACHE, "models--" + repo.replace("/", "--"))
def complete(s):
    has_cfg = os.path.isfile(os.path.join(s, "config.json")) or os.path.isfile(os.path.join(s, "adapter_config.json"))
    return has_cfg and bool(glob.glob(os.path.join(s, "*.safetensors")))
try:
    main = open(os.path.join(d, "refs", "main")).read().strip()
    cand = os.path.join(d, "snapshots", main)
    if complete(cand):
        print(cand); sys.exit(0)
except OSError:
    pass
snaps = [s for s in glob.glob(os.path.join(d, "snapshots", "*")) if complete(s)]
if not snaps:
    sys.exit(f"no complete cached snapshot for {repo} under {d}")
def stamp(s):
    c = os.path.join(s, "config.json"); a = os.path.join(s, "adapter_config.json")
    return os.path.getmtime(c if os.path.isfile(c) else a)
print(max(snaps, key=stamp))
PYSNAP
}
BASE_SNAP=$(snap "$BASE_REPO") || f22_fatal "$BASE_REPO is not in the offline HF cache on $CLUSTER (hf-cluster-fetch it first)"
LORA_ARGS=(); LORA_SNAPS=()
for i in "${!ORG_NAMES[@]}"; do
  s=$(snap "${ORG_REPOS[$i]}") || f22_fatal "${ORG_REPOS[$i]} is not in the offline HF cache on $CLUSTER"
  [ -f "$s/adapter_config.json" ] || f22_fatal "${ORG_REPOS[$i]}: no adapter_config.json at $s"
  "$SPY" - "$s/adapter_config.json" "$BASE_REPO" <<'PY' || f22_fatal "${ORG_REPOS[$i]}: adapter_config disagrees with the served base or rank cap"
import json, sys
c = json.load(open(sys.argv[1]))
assert c.get("base_model_name_or_path") == sys.argv[2], (c.get("base_model_name_or_path"), sys.argv[2])
assert int(c.get("r", 0)) <= 32, c.get("r")
print(f"adapter ok: r={c.get('r')} alpha={c.get('lora_alpha')} targets={sorted(c.get('target_modules') or [])}")
PY
  LORA_ARGS+=("${ORG_NAMES[$i]}=$s"); LORA_SNAPS+=("$s")
done
f22_log "base $BASE_REPO @ $BASE_SNAP; adapters: ${LORA_ARGS[*]}"

# --- cells.json: pseudo-cells {none, prompted__safety, prompted__constitution} x {base, organisms} -------------------
( cd "$REPO_DIR" && "$MPY" - "$MIS_ROOT/cells.json" "$SERVED" "${ORG_NAMES[@]}" <<'PY' ) || f22_fatal "cells.json refused"
import json, sys
from sisteer.capeval.manifest import sha256_hex16
from sisteer.capeval.petri.registry_adapter import HARNESS_TO_TRAINTEST_SI, _traintest_si_text_bytes
out, served, orgs = sys.argv[1], sys.argv[2], sys.argv[3:]
# the wave's recorded SI prefix hashes (make_full22_configs.PREFIX_HASHES; every misalign manifest carries them):
# the SI bytes this job sends must be THOSE bytes, or the em8q rows would not pair with the proxy lane's prompted cells
want = {"safety": "ac3a58f61d53cf24", "constitution": "21e06f482a6559b3"}
hashes = {}
for si, h in want.items():
    b = _traintest_si_text_bytes(HARNESS_TO_TRAINTEST_SI[si])
    got = sha256_hex16(b)
    if got != h:
        raise SystemExit(f"SI {si}: bytes hash to {got}, the wave records {h} — SI text drift; refusing")
    hashes[si] = got
rows = []
for m in [served] + orgs:
    base_kind = "none" if m == served else "em_lora"
    for arm, si in (("none", None), ("prompted__safety", "safety"), ("prompted__constitution", "constitution")):
        rows.append({"cell_id": f"{m}::{arm}", "virtual_model": m, "serving_identity": "vllm+lora+plain",
                     "intervention_hash": hashes[si] if si else "-", "si_id": si,
                     "kind": "prompted" if si else base_kind})
json.dump(rows, open(out, "w"), indent=2)
print(f"cells.json: {len(rows)} pseudo-cells over {[served] + orgs}")
PY
cp "$MIS_ROOT/cells.json" "$MIS_ROOT/em8q/cells.json"

# --- plain vLLM with the LoRA modules, loopback only ------------------------------------------------------------------
PORT=$("$SPY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
URL="http://127.0.0.1:$PORT/v1"
VLLM_PID=""; RUNNER_PID=""; RC=1; N_E=0
result_line(){ echo "$(date -Is) job=${SLURM_JOB_ID:-manual} target=em_organisms sha=${WAVE_SHA:0:12} cluster=$CLUSTER rc=$RC em8q_responses=$N_E"; }
cleanup(){
  [ -n "${RUNNER_PID:-}" ] && kill -0 "$RUNNER_PID" 2>/dev/null && { kill -TERM "$RUNNER_PID" 2>/dev/null; wait "$RUNNER_PID" 2>/dev/null; }
  if [ -n "${VLLM_PID:-}" ]; then kill -TERM "$VLLM_PID" 2>/dev/null; for i in $(seq 1 24); do kill -0 "$VLLM_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$VLLM_PID" 2>/dev/null || true; fi
  result_line >> "$MIS_ROOT/RESULT"
  f22_result "misalign em_organisms $(result_line)"
}
trap cleanup EXIT
trap 'f22_log "SIGTERM — draining"; RC=143; exit 143' SIGTERM SIGINT SIGQUIT
VLLM_FLAGS=(--served-model-name "$SERVED" --enable-lora --max-lora-rank 32 --max-loras 4 --max-cpu-loras 4
            --max-model-len 8192 --max-num-seqs 128 --dtype bfloat16 --gpu-memory-utilization 0.90
            --generation-config vllm --port "$PORT" --host 127.0.0.1)
f22_log "vllm serve $BASE_SNAP ${VLLM_FLAGS[*]} --lora-modules ${LORA_ARGS[*]} (VLLM_PLUGINS empty)"
VLLM_PLUGINS="" CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" vllm serve "$BASE_SNAP" "${VLLM_FLAGS[@]}" --lora-modules "${LORA_ARGS[@]}" > "$OUT/vllm_server.log" 2>&1 &
VLLM_PID=$!
for i in $(seq 1 360); do   # up to 30 min for engine init
  curl -sf -m 10 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 "$VLLM_PID" 2>/dev/null || { f22_log "vLLM died during startup:"; tail -40 "$OUT/vllm_server.log"; exit 3; }
  sleep 5
done
curl -sf -m 10 "http://127.0.0.1:$PORT/health" >/dev/null || { f22_log "vLLM not healthy after 30 min"; tail -40 "$OUT/vllm_server.log"; exit 3; }
MODELS=$(curl -s -m 20 "$URL/models")
"$SPY" - "$MODELS" "$SERVED" "${ORG_NAMES[@]}" <<'PY' || { f22_log "/v1/models: ${MODELS:0:600}"; exit 3; }
import json, sys
ids = {m["id"] for m in json.loads(sys.argv[1])["data"]}
missing = [m for m in sys.argv[2:] if m not in ids]
if missing: raise SystemExit(f"served models {sorted(ids)} lack {missing}")
print("served:", sorted(ids))
PY
# one-request smoke on EVERY served name (the base and each adapter actually generate)
for m in "$SERVED" "${ORG_NAMES[@]}"; do
  SMOKE=$(curl -s -m 600 -H 'Content-Type: application/json' "$URL/chat/completions" \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":8,\"temperature\":0}")
  "$SPY" -c 'import json,sys; d=json.loads(sys.argv[1]); print("smoke", sys.argv[2], "ok:", repr(d["choices"][0]["message"]["content"][:40]))' "$SMOKE" "$m" \
    || { f22_log "smoke on $m failed: ${SMOKE:0:600}"; exit 3; }
done
"$SPY" - "$OUT/serving.json" "$BASE_REPO" "$BASE_SNAP" "$SERVED" "$PORT" "$WAVE_SHA" "$CLUSTER" "${VLLM_FLAGS[*]}" "${LORA_ARGS[@]}" <<'PY'
import json, os, socket, sys, time, vllm
out, base_repo, base_snap, served, port, sha, cluster, flags = sys.argv[1:9]
loras = dict(a.split("=", 1) for a in sys.argv[9:])
json.dump({"base_repo": base_repo, "base_snapshot": base_snap, "served_model_name": served, "lora_modules": loras,
           "vllm": vllm.__version__, "vllm_flags": flags, "vllm_plugins": "", "port": int(port), "node": socket.gethostname(),
           "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "wave_sha": sha, "cluster": cluster,
           "serving_identity": "vllm+lora+plain", "written_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          open(out, "w"), indent=2)
PY
cp "$OUT/serving.json" "$MIS_ROOT/serving.json"

# --- the runner, plain mode ------------------------------------------------------------------------------------------
LOG="$MIS_ROOT/em8q/runner_${SLURM_JOB_ID:-manual}.log"
f22_log "runner: --plain family em8q --n 50 --max-tokens 600 concurrency 64 -> $MIS_ROOT/em8q (log $LOG)"
# tee into the job log too (watcher liveness; a runner writing only its own file reads as a stall)
( set -o pipefail; cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" -m sisteer.capeval.misalign.runner --plain \
    --proxy-url "$URL" --cells "$MIS_ROOT/cells.json" --family em8q --n 50 --max-tokens 600 --concurrency 64 \
    --out "$MIS_ROOT/em8q" 2>&1 | tee "$LOG" ) &
RUNNER_PID=$!
wait "$RUNNER_PID"; RC=$?; RUNNER_PID=""
kill -0 "$VLLM_PID" 2>/dev/null || { f22_log "vLLM died during the run (see $OUT/vllm_server.log)"; [ "$RC" = 0 ] && RC=3; }
N_E=$(cat "$MIS_ROOT"/em8q/*/responses.jsonl 2>/dev/null | wc -l)
[ "$RC" = 0 ] || { f22_log "runner FAILED rc=$RC; log tail:"; tail -30 "$LOG"; }
f22_log "em_organisms em8q done rc=$RC responses=$N_E; outputs under $MIS_ROOT"
exit $RC
