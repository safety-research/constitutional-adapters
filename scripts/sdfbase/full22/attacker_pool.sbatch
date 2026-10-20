#!/bin/bash
#SBATCH --job-name=f22-atk
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# full22: ONE member of the SHARED ATTACKER POOL — the unsteered huihui-ai/Huihui-Qwen3.6-27B-abliterated
# on vLLM (--max-model-len 131072 by default so Crescendo's accumulating 4,096-cap conversation never 400s; prefix caching ON,
# quantization none = configs/attacker_serving.json, identical for every cell of every target). The
# attacker is identical for every cell, so it is pooled: the previous wave measured crescendo + PAIR
# ATTACKER-bound (383 gen tok/s, 41% KV, a waiting queue) while ActorAttack's attacker idled 41%. Each
# pool member publishes $SHARED/exp/sdfbase_full22/attackers/<jobid>.json {url: http://<node>:<port>};
# target_attacks.sbatch drivers health-check the records and round-robin over the LIVE ones.
# Ends when $SHARED/exp/sdfbase_full22/attackers/STOP (whole pool) or <jobid>.STOP appears, at walltime,
# or — exit 3 — when the vLLM process dies (a watcher sees FAILED, never a live record over a dead engine) — and
# ON ITS OWN when its LAST CONSUMER is terminal on sacct (four members served 1–4 targets at ~37 % busy),
# or when the group's placement is COMPLETE (attackers/PLACED, written by the funnel's `mark-placed`) and no
# target ever bound it: it writes attackers/<jobid>.SELF_EXIT {reason, consumers, utc} and exits 0 (attacker_pool_lifecycle.sh;
# the funnel reads the marker as a teardown, never a result). The group-level `teardown-pool` waiter stays as the backstop.
# Usage (positional only — never --export=):  sbatch -J f22-atk-<n> [--qos=<qos> --output=...] \
#     scripts/sdfbase/full22/attacker_pool.sbatch <wave_sha> [max_model_len=131072]
#   max_model_len: the attacker's serving window. 131072 is the default since 2026-09-11 (the
#   4,096-cap canonical instrument): Crescendo's adversarial chat accumulates every target reply,
#   so 10 turns x 4,096 + its own turns exceeds the 49152 that held every 512-cap conversation
#   (the mt1 / filed-wave value, still passable). Recorded in the pool record (max_model_len).
set -uo pipefail
WAVE_SHA="${1:?wave sha}"
ATK_MAX_LEN="${2:-131072}"
case "$ATK_MAX_LEN" in ''|*[!0-9]*) echo "FATAL: max_model_len '$ATK_MAX_LEN' is not an integer" >&2; exit 1;; esac
source "$(dirname "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)")/common.sh" 2>/dev/null \
  || { [ -n "${SISTEER_SHARED_ROOT:-}" ] && source "${SISTEER_EVAL_REPO_ROOT:-$SISTEER_SHARED_ROOT/sisteer-eval-repo}/$WAVE_SHA/scripts/sdfbase/full22/common.sh" 2>/dev/null; } \
  || source "${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA/scripts/sdfbase/full22/common.sh"
f22_attempt_wait   # sisteer-attempt-wait-hook (reproduce/lib/attempt_wait.sh): a no-op unless SISTEER_ATTEMPT_ID is set — twin submit
F22_LAUNCH_CELLS=()   # this job stands up no target: no steered cell to select a serving path for (the kind-aware default, 2026-09-15)
f22_init "$WAVE_SHA"
source "$REPO_DIR/scripts/sdfbase/full22/attacker_pool_lifecycle.sh" 2>/dev/null \
  || source "$(dirname "$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)")/attacker_pool_lifecycle.sh" \
  || f22_fatal "attacker_pool_lifecycle.sh not found beside common.sh in $REPO_DIR — the member cannot decide its own end"
POOL_DIR="$WAVE_ROOT/attackers"; mkdir -p "$POOL_DIR"
REC="$POOL_DIR/${SLURM_JOB_ID}.json"; SERVE_DIR="$WAVE_ROOT/serving/attacker_${SLURM_JOB_ID}"; mkdir -p "$SERVE_DIR"
f22_stop_requested && { f22_log "wave STOP present — not serving"; exit 0; }
VLLM_VENV="$SHARED/envs/vllm-attacker"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" "$VLLM_VENV/bin/python" -c 'import torch,vllm; torch.zeros(1).cuda(); print("attacker venv OK vllm",vllm.__version__)' || f22_fatal "attacker venv CUDA init failed"
SERVED_NAME=huihui-ai/Huihui-Qwen3.6-27B-abliterated
SNAP=27502c8717fd5a2f8c0c77188c10c243fd4f672e
MODEL_PATH="$HF_HOME/hub/models--huihui-ai--Huihui-Qwen3.6-27B-abliterated/snapshots/$SNAP"
[ -e "$MODEL_PATH/config.json" ] || MODEL_PATH="$SERVED_NAME"   # resolve through the (offline) HF cache
VLLM_PID=""
# THE PENDING MARKER (2026-09-18): announce that an attacker EXISTS the moment this job starts, not when it
# first serves ~22 min later. Without it a target that finds no `.json` cannot tell "a pool is booting, wait
# for it" from "nobody ever submitted one" — and the second case idles the target's GPUs for the whole 90-min
# wait to no purpose. On 2026-09-17 exactly that happened: a launch sequence was interrupted between
# submitting the target arrays and deploying the attacker, so that step never ran, seven targets came up
# healthy and sat idle against a 90-minute budget, and NOTHING in the wave recorded that a stage was missing.
# An absence is invisible unless something asserts presence; this marker is that assertion, and
# target_attacks_goalset.sbatch refuses IMMEDIATELY when neither a record nor a marker exists.
PENDING="$POOL_DIR/${SLURM_JOB_ID}.pending"
printf '{"job_id":"%s","node":"%s","sha":"%s","state":"starting","written_at":"%s"}\n' \
  "$SLURM_JOB_ID" "$(hostname)" "$WAVE_SHA" "$(date -u +%FT%TZ)" > "$PENDING"
f22_log "attacker PENDING marker written ($PENDING) — targets may now wait for this member"
cleanup(){ [ -n "${VLLM_PID:-}" ] && kill -9 "$VLLM_PID" 2>/dev/null || true; rm -f "$REC" "$PENDING"; f22_result "attacker down"; }
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
PORT=$((8137 + SLURM_JOB_ID % 800)); ok=0
for satt in 1 2 3 4; do   # the cluster A cgroup race: retry the serve
  SPORT=$((PORT + satt))
  f22_log "attacker vLLM attempt $satt on gpu $CUDA_VISIBLE_DEVICES port $SPORT"
  # NOT srun-wrapped, deliberately (the mt1 precedent): the serve script backgrounds the vLLM server and
  # returns once healthy, and a step's children die with the step — the batch step is the Slurm-tracked
  # unit here and the EXIT trap kills the server by pid
  bash scripts/vllm_attacker_serve.sh --venv "$VLLM_VENV" --model "$MODEL_PATH" \
    --served-name "$SERVED_NAME" --gpu "$CUDA_VISIBLE_DEVICES" --port "$SPORT" --util 0.85 \
    --max-model-len "$ATK_MAX_LEN" --prefix-caching on --quantization none --wait 2400 \
    --log "$SERVE_DIR/vllm_attacker_$satt.log" --manifest "$SERVE_DIR/attacker_serving_$satt.json" | tee "$SERVE_DIR/attacker_status.txt" || true
  VLLM_PID="$(sed -n 's/^VLLM_SERVER_PID=//p' "$SERVE_DIR/attacker_status.txt" | tail -1)"
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then PORT=$SPORT; ok=1; break; fi
  [ -n "${VLLM_PID:-}" ] && kill -9 "$VLLM_PID" 2>/dev/null || true; VLLM_PID=""; sleep 20
done
[ "$ok" = 1 ] || f22_fatal "attacker not healthy after 4 attempts"
URL="http://$(hostname):$PORT"
# A probe PASSES only when the API answers 2xx (curl --fail — without it a 503 exits 0) AND the body names the
# served model — on EVERY probe, not only at startup: a vLLM that lives but answers 5xx, or serves another model, serves nobody, and a lease it
# renewed would make the retire walk count it as the survivor while withdrawing the working member
# The served model is matched EXACTLY — data[].id == SERVED_NAME for some element of the parsed JSON body — never
# as a substring of the body: an endpoint serving a model whose id merely CONTAINS the expected one (a prefix or a suffix around it) is another
# model, and a lease it renewed would again make the retire walk count it as the survivor. The dependency is explicit: python3 -S (no site),
# which the member image carries (common.sh's recipe and health writers run under it); a body that is not JSON is no server either
pool_probe_model_ok(){   # stdin = the /v1/models body -> 0 iff some data[].id equals SERVED_NAME exactly
  PROBE_MODEL="$SERVED_NAME" python3 -S -c 'import json, os, sys
try:
    d = json.load(sys.stdin)
except ValueError:
    sys.exit(1)
ids = [str(x.get("id")) for x in (d.get("data") or []) if isinstance(x, dict)] if isinstance(d, dict) else []
sys.exit(0 if os.environ["PROBE_MODEL"] in ids else 1)'
}
pool_probe_ok(){ local body; body=$(curl -sf -m 20 "$URL/v1/models" 2>/dev/null) || return 1; printf '%s' "$body" | pool_probe_model_ok; }
# The member SAMPLES its own load every F22_POOL_SAMPLE_EVERY_S (15 s) inside the 60 s tick — the engine's queue
# from vLLM's /metrics (num_requests_running, num_requests_waiting, the request queue-time / e2e / TTFT / inference-time / per-output-token
# histogram sums + counts) and the device's utilization (nvidia-smi over Slurm's OWN device list, never widened) — one JSON line per sample,
# append-only, <pool>/<job>.samples.jsonl beside the health lease. The post-wave analysis (attacker busy fraction and target wait vs concurrent
# targets per member; where attacker queueing binds) replaces the 1-per-3 rule of thumb with a measured number. A failed sample is a null row
# (metrics_ok false, gpu []), never a failed member: nothing here may fail the tick
pool_sample(){   # /metrics is read by the python itself (urllib, 5 s), never through curl: the probe's curl path is the health check's alone
  local gpu
  gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits -i "${CUDA_VISIBLE_DEVICES:-0}" 2>/dev/null || true)
  POOL_SAMPLE_UTC="$(date -u +%FT%TZ)" POOL_SAMPLE_JOB="$SLURM_JOB_ID" POOL_SAMPLE_URL="$URL" POOL_SAMPLE_GPU="$gpu" python3 -S -c '
import json, os, re, sys, urllib.request
try:
    with urllib.request.urlopen(os.environ["POOL_SAMPLE_URL"] + "/metrics", timeout=5) as resp:
        text = resp.read().decode("utf-8", "replace")
except Exception:
    text = ""
m = {}
keep = ("queue_time", "e2e_request_latency", "time_to_first_token", "request_inference_time", "time_per_output_token")
for ln in text.splitlines():
    if not ln or ln.startswith("#"):
        continue
    mm = re.match(r"^([A-Za-z_:][A-Za-z0-9_:]*)(\{[^}]*\})?\s+(\S+)", ln)
    if not mm:
        continue
    name = mm.group(1)
    try:
        v = float(mm.group(3))
    except ValueError:
        continue
    if name in ("vllm:num_requests_running", "vllm:num_requests_waiting") or (name.endswith(("_sum", "_count")) and any(k in name for k in keep)):
        m[name] = m.get(name, 0.0) + v
gpu = []
for ln in (os.environ.get("POOL_SAMPLE_GPU") or "").splitlines():
    f = [x.strip() for x in ln.split(",")]
    if len(f) == 5:
        try:
            gpu.append({"index": int(f[0]), "util_gpu_pct": float(f[1]), "util_mem_pct": float(f[2]), "mem_used_mib": float(f[3]), "mem_total_mib": float(f[4])})
        except ValueError:
            pass
print(json.dumps({"utc": os.environ["POOL_SAMPLE_UTC"], "job_id": os.environ["POOL_SAMPLE_JOB"], "url": os.environ["POOL_SAMPLE_URL"], "metrics_ok": bool(m),
                  "running": m.get("vllm:num_requests_running"), "waiting": m.get("vllm:num_requests_waiting"),
                  "histograms": {k: v for k, v in m.items() if k.endswith(("_sum", "_count"))}, "gpu": gpu}, sort_keys=True))
' >> "$POOL_DIR/${SLURM_JOB_ID}.samples.jsonl" 2>/dev/null || true
}
pool_probe_ok || f22_fatal "attacker answers on loopback but not with a 2xx naming $SERVED_NAME on $URL (vLLM must bind all interfaces for the pool; a 5xx or another model is no server)"
printf '{"url":"%s","node":"%s","port":%s,"job_id":"%s","model":"%s","snapshot":"%s","prefix_caching":true,"quantization":"none","max_model_len":%s,"sha":"%s","serving_manifest":"%s","written_at":"%s"}\n' \
  "$URL" "$(hostname)" "$PORT" "$SLURM_JOB_ID" "$SERVED_NAME" "$SNAP" "$ATK_MAX_LEN" "$WAVE_SHA" "$SERVE_DIR/attacker_serving_$satt.json" "$(date -u +%FT%TZ)" > "$REC"
f22_log "attacker LIVE: $URL (record $REC; max_model_len $ATK_MAX_LEN)"
# 31d-fix-3: my health LEASE — the funnel counts me as a surviving server only while it is fresh and ok (fail closed); written now and every tick
HEALTH_FAILS=0; HEALTH_LEASE_S="${F22_POOL_HEALTH_LEASE_S:-180}"; HEALTH_MAX_FAILS="${F22_POOL_HEALTH_MAX_FAILS:-3}"
f22_pool_health_write "$POOL_DIR" "$SLURM_JOB_ID" "$URL" 1 "$HEALTH_LEASE_S"
LAST_VERDICT_WHY=""
while :; do
  { [ -e "$POOL_DIR/STOP" ] || [ -e "$POOL_DIR/${SLURM_JOB_ID}.STOP" ] || f22_stop_requested; } && { f22_log "STOP requested; leaving"; exit 0; }
  kill -0 "$VLLM_PID" 2>/dev/null || { f22_log "vLLM attacker died (pid $VLLM_PID) — see $SERVE_DIR"; exit 3; }
  if pool_probe_ok; then
    HEALTH_FAILS=0; f22_pool_health_write "$POOL_DIR" "$SLURM_JOB_ID" "$URL" 1 "$HEALTH_LEASE_S"
  else
    # 31d-fix-3: a failed probe is recorded at the source — the lease says ok=false, so no retirement counts me as the
    # survivor while my endpoint is down — and after HEALTH_MAX_FAILS in a row a vLLM that lives but answers nothing serves nobody: withdraw and leave
    HEALTH_FAILS=$((HEALTH_FAILS + 1)); f22_pool_health_write "$POOL_DIR" "$SLURM_JOB_ID" "$URL" 0 "$HEALTH_LEASE_S"
    f22_log "WARNING: health probe failed on $URL — no 2xx answer naming $SERVED_NAME ($HEALTH_FAILS/$HEALTH_MAX_FAILS consecutive; lease marked not ok — the funnel counts me as no server until it passes)"
    if [ "$HEALTH_FAILS" -ge "$HEALTH_MAX_FAILS" ]; then
      f22_pool_member_unhealthy "$POOL_DIR" "$SLURM_JOB_ID" "$URL" "health check failed $HEALTH_FAILS times in a row"
      consumers=$(f22_pool_consumers "$WAVE_ROOT" "$URL" | tr '\n' ',' | sed 's/,$//')
      printf '{"job_id":"%s","reason":"unhealthy","consumers":"%s","why":"health check failed %s times in a row","utc":"%s"}\n' "$SLURM_JOB_ID" "$consumers" "$HEALTH_FAILS" "$(date -u +%FT%TZ)" > "$POOL_DIR/${SLURM_JOB_ID}.SELF_EXIT"
      f22_log "SELF_EXIT (unhealthy): $HEALTH_FAILS failed probes in a row — record withdrawn ($POOL_DIR/${SLURM_JOB_ID}.UNHEALTHY); the drivers bound to me fail over; leaving with exit 3"
      exit 3
    fi
  fi
  # 31d: the funnel marked me RETIRING (the pool is above one member per three live targets): vanish from discovery (my record and
  # pending marker go — new targets stop binding me), drain in-flight requests for the grace period (the drivers bound to me fail
  # over to a surviving member on their next call), then leave as `retired`
  if f22_pool_retire_requested "$POOL_DIR" "$SLURM_JOB_ID"; then
    rm -f "$REC" "$PENDING"; grace="${F22_POOL_RETIRE_GRACE_S:-180}"
    f22_log "RETIRE requested ($POOL_DIR/${SLURM_JOB_ID}.RETIRE) — record withdrawn, draining ${grace}s, then leaving"
    sleep "$grace"
    consumers=$(f22_pool_consumers "$WAVE_ROOT" "$URL" | tr '\n' ',' | sed 's/,$//')
    printf '{"job_id":"%s","reason":"retired","consumers":"%s","why":"retired by the funnel (pool above one member per three live targets); drivers re-bind","utc":"%s"}\n' "$SLURM_JOB_ID" "$consumers" "$(date -u +%FT%TZ)" > "$POOL_DIR/${SLURM_JOB_ID}.SELF_EXIT"
    f22_log "SELF_EXIT (retired) — leaving"; exit 0
  fi
  # 31b: my own end — the last consumer terminal on sacct, or placement complete and nobody bound me (every answer comes from the
  # shared root and Slurm; an undecidable read keeps the member)
  verdict=$(f22_pool_member_verdict "$WAVE_ROOT" "$URL" "$SLURM_JOB_ID" 2> "$SERVE_DIR/lifecycle_why.txt"); why=$(tail -n 1 "$SERVE_DIR/lifecycle_why.txt")
  [ "$why" != "$LAST_VERDICT_WHY" ] && { f22_log "lifecycle: $verdict — $why"; LAST_VERDICT_WHY="$why"; }
  case "$verdict" in
    exit_last_consumer|exit_unbound)
      consumers=$(f22_pool_consumers "$WAVE_ROOT" "$URL" | tr '\n' ',' | sed 's/,$//')
      printf '{"job_id":"%s","reason":"%s","consumers":"%s","why":"%s","utc":"%s"}\n' "$SLURM_JOB_ID" "$verdict" "$consumers" "${why//\"/}" "$(date -u +%FT%TZ)" > "$POOL_DIR/${SLURM_JOB_ID}.SELF_EXIT"
      f22_log "SELF_EXIT ($verdict): $why — leaving; the funnel reads $POOL_DIR/${SLURM_JOB_ID}.SELF_EXIT as a teardown"
      exit 0;;
  esac
  # four samples per 60 s tick (the health probe and the lifecycle verdict keep their cadence)
  for _ in 1 2 3 4; do pool_sample; sleep "${F22_POOL_SAMPLE_EVERY_S:-15}"; done
done
