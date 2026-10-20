#!/bin/bash
# Launch a vanilla vLLM server for the UNSTEERED jailbreak attacker on ONE GPU
# and block until /health is ready (probed through attacker_serving.py `health`,
# stdlib-only -- no curl). Opt-in accelerator for the attack battery:
# scripts/jailbreak_run.py --attacker-engine vllm points at the endpoint this
# starts. See src/sisteer/eval/attacks/vllm_attacker.py for the client side.
#
# WHY ONLY THE ATTACKER. This seam owns only the unsteered adversary. The
# target launcher has separate serving composition and a per-request steering
# cache_salt; importing these prefix-cache/quantization defaults into that path
# would be incorrect. Historical Slurm timing artifacts are permanently gone,
# so this launcher does not repeat the old unsourced attacker-cost/speedup claims.
# scripts/modal/attacker_latency_scope.py is the reproducible Modal measurement.
#
# TWO GPUS PER TASK, NOT ONE. This server takes one GPU (the 27B attacker); the
# separately served steered target takes a SECOND GPU. Pass --gpu with a CUDA
# index that
# is a STRICT SUBSET of Slurm's CUDA_VISIBLE_DEVICES (never widen it, never
# hardcode an absolute index outside the allocation), and give the attack loop
# the other allocated index.
#
# VENV GOTCHA (torchcodec). vllm==0.25.1's server startup imports the multimodal
# video path, which does a GUARDED `from torchcodec.decoders import VideoDecoder`
# -- but the guard only catches ImportError/RuntimeError. In the pinned cu129
# env, torchcodec's prebuilt image lib links libcudart.so.13 (CUDA 13) and fails
# to load with an OSError, which ESCAPES the guard and crashes `vllm serve` even
# though `import vllm` succeeds. torchcodec is video-only and unused here, so the
# fix is to remove it from the vLLM venv:  uv pip uninstall torchcodec  (then the
# guard catches the ModuleNotFoundError and the text server starts fine).
#
# SERVING OPTIONS (quantization, prefix caching) come from ONE definition,
# src/sisteer/eval/attacks/attacker_serving.py, whose defaults live in
# configs/attacker_serving.json (flip a default THERE, not here). --quantization
# none|fp8 and --prefix-caching on|off override the file.
#
# PREFIX CACHING: vLLM's OWN default is OFF for this model, this launcher's is ON. The
# attacker is a HYBRID (gated-delta-net + attention) model and vllm 0.25.1
# disables prefix caching by default for hybrids ("still experimental":
# ModelConfig.is_prefix_caching_supported); every pre-cutover serve log shows
# enable_prefix_caching=False and a 0.0% hit rate. configs/attacker_serving.json
# turns it on, so the module emits --enable-prefix-caching --mamba-cache-mode
# align explicitly (Qwen3.5 refuses mode 'all'); --prefix-caching off is the
# explicit escape hatch and emits --no-enable-prefix-caching. Measured on the
# recorded call bank (scripts/modal/attacker_latency_scope.py): 76-77% prefix-cache
# hit rate, 1.04x / 1.34x / 3.95x batch wall speedup at 1 / 8 / 32 concurrent
# growing-transcript turns. The attacker is unsteered, so caching is SAFE for
# it. If this launcher is ever repurposed to serve a STEERED model, prefix
# caching MUST be off: a steering hook makes an identical token prefix produce
# DIFFERENT activations, so a cached prefix would replay stale, unsteered state.
#
# PROVENANCE. After /health the helper parses the engine's own resolved-config
# log line, checks it agrees with the request (quantization, prefix caching,
# kv-cache dtype), and writes a sidecar manifest (--manifest, default beside the
# log as <log>.serving.json), printing VLLM_SERVING_MANIFEST=<path>. Pass that
# path to the driver as --attacker-serving so attacker_engine.json records the
# quantization (fp8 and bf16 attacker text are NOT cross-comparable; the
# aggregate refuses to merge them). A server whose engine contradicts the
# request is killed and the helper exits 1.
#
# RESULTS ARE NOT CROSS-ENGINE COMPARABLE. vLLM and HF do not produce identical
# attacker text (different kernels/sampling; an equivalence check failed
# over a single trailing newline), so the attack PROMPTS differ and ASR from a
# --attacker-engine vllm run is NOT comparable to an HF-engine battery. Do not
# try to make them bit-identical. The engine is recorded in each run's
# attacker_engine.json so every transcript is self-identifying.
#
# Usage:
#   scripts/vllm_attacker_serve.sh \
#     --venv   $SHARED/envs/vllm-attacker \
#     --model  $HF_HOME/hub/models--huihui-ai--Huihui-Qwen3.6-27B-abliterated/snapshots/27502c8717fd5a2f8c0c77188c10c243fd4f672e \
#     --served-name huihui-ai/Huihui-Qwen3.6-27B-abliterated \
#     --gpu 0 --port 8000 --util 0.85 --max-model-len 8192 \
#     --wait 900 --log /path/to/vllm_serve.log \
#     [--quantization fp8|none] [--prefix-caching on|off] \
#     [--manifest /path/to/attacker_serving.json]
#
# --served-name MUST equal the model id the client sends (cfg["attacker"]["model"])
# or vLLM rejects the request with a model-mismatch error. On success prints the
# server PID and the endpoint URL and exits 0 with the server still running (kill
# the PID to stop it). Exits nonzero if the server never becomes healthy.
set -uo pipefail

VENV="" ; MODEL="" ; SERVED_NAME="" ; GPU="0" ; PORT="8000"
UTIL="0.85" ; MAX_MODEL_LEN="8192" ; WAIT="2400" ; LOG=""
QUANT="" ; PREFIX_CACHING="" ; MANIFEST=""
EXTRA_ARGS=()
# The repo root (this script lives at <repo>/scripts/): the serving-options
# module is run from here, stdlib-only, under the vLLM venv's python (or the
# system python3) -- never the project venv.
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVING_MOD="$REPO_DIR/src/sisteer/eval/attacks/attacker_serving.py"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv)          VENV="$2"; shift 2;;
    --model)         MODEL="$2"; shift 2;;
    --served-name)   SERVED_NAME="$2"; shift 2;;
    --gpu)           GPU="$2"; shift 2;;
    --port)          PORT="$2"; shift 2;;
    --util)          UTIL="$2"; shift 2;;
    --max-model-len) MAX_MODEL_LEN="$2"; shift 2;;
    --wait)          WAIT="$2"; shift 2;;
    --log)           LOG="$2"; shift 2;;
    --quantization)  QUANT="$2"; shift 2;;
    --prefix-caching) PREFIX_CACHING="$2"; shift 2;;
    --manifest)      MANIFEST="$2"; shift 2;;
    --)              shift; EXTRA_ARGS+=("$@"); break;;
    *) echo "vllm_attacker_serve.sh: unknown arg '$1'" >&2; exit 2;;
  esac
done

[ -n "$MODEL" ] || { echo "FATAL: --model is required" >&2; exit 2; }
[ -n "$SERVED_NAME" ] || SERVED_NAME="$MODEL"
[ -n "$LOG" ] || LOG="/tmp/vllm_attacker_${PORT}.log"
[ -n "$MANIFEST" ] || MANIFEST="${LOG%.log}.serving.json"
[ -f "$SERVING_MOD" ] || { echo "FATAL: serving-options module missing at $SERVING_MOD" >&2; exit 2; }

# Resolve the vllm entrypoint from the (separate) vLLM venv.
if [ -n "$VENV" ] && [ -x "$VENV/bin/vllm" ]; then

  VLLM_BIN="$VENV/bin/vllm"
elif [ -n "$VENV" ] && [ -x "$VENV/bin/python" ]; then
  VLLM_BIN="$VENV/bin/python -m vllm.entrypoints.cli.main"
elif command -v vllm >/dev/null 2>&1; then
  VLLM_BIN="vllm"
else
  echo "FATAL: cannot find the vllm CLI (pass --venv <vllm-venv-dir>)" >&2
  exit 2
fi

ENDPOINT="http://127.0.0.1:${PORT}"
# The interpreter for the (stdlib-only) serving-options module.
if [ -n "$VENV" ] && [ -x "$VENV/bin/python" ]; then SERVING_PY="$VENV/bin/python"; else SERVING_PY="python3"; fi
SERVING_OPTS=()
[ -n "$QUANT" ] && SERVING_OPTS+=(--quantization "$QUANT")
[ -n "$PREFIX_CACHING" ] && SERVING_OPTS+=(--prefix-caching "$PREFIX_CACHING")
# ONE definition of the flags: the module prints them, one per line.
if ! _flags_out="$("$SERVING_PY" "$SERVING_MOD" argv "${SERVING_OPTS[@]}")"; then
  echo "FATAL: could not resolve attacker serving options (see above)" >&2; exit 2
fi
SERVE_FLAGS=()
while IFS= read -r _flag; do [ -n "$_flag" ] && SERVE_FLAGS+=("$_flag"); done <<< "$_flags_out"
[ "${#SERVE_FLAGS[@]}" -gt 0 ] || { echo "FATAL: serving-options module printed no flags" >&2; exit 2; }
echo "[serve] model=$MODEL served-name=$SERVED_NAME gpu=$GPU port=$PORT util=$UTIL"
echo "[serve] serving flags (attacker_serving.py; defaults configs/attacker_serving.json): ${SERVE_FLAGS[*]}"
echo "[serve] manifest: $MANIFEST"
# The GDN prefill kernel JIT-compiles via a `ninja` SUBPROCESS, which searches PATH -- not
# the venv we invoke by absolute path. ninja IS installed in the venv; it was simply not
# findable, and the engine died with FileNotFoundError: 'ninja' after ~7min of warmup,
# surfaced only as "server did not become healthy". Keyed on $VENV/bin because VLLM_BIN
# may be a multi-word command (python -m ...), so dirname on it is wrong.
[ -n "$VENV" ] && export PATH="$VENV/bin:$PATH"
# vLLM compiles the model graph on first start (~80s for 27B) and caches it under
# VLLM_CACHE_ROOT, which defaults to ~/.cache. On this cluster /home is NODE-LOCAL disk,
# so that cache is thrown away every time Slurm picks a different node and EVERY run pays
# the cold-compile cost. Point it at shared storage so the second run onward starts fast.
if [ -z "${VLLM_CACHE_ROOT:-}" ]; then
  _sh="${SISTEER_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-}}"   # the cluster's shared root (the job's env prefix, else its block value)
  [ -n "$_sh" ] && [ -d "$_sh" ] && export VLLM_CACHE_ROOT="$_sh/.cache/vllm"
  [ -n "${VLLM_CACHE_ROOT:-}" ] && mkdir -p "$VLLM_CACHE_ROOT"
fi
echo "[serve] VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-<default ~/.cache, NODE-LOCAL>}"
echo "[serve] vllm bin: $VLLM_BIN"
echo "[serve] log: $LOG"

# Pin the server to exactly ONE allocated GPU (strict subset of Slurm's set).
CUDA_VISIBLE_DEVICES="$GPU" \
  $VLLM_BIN serve "$MODEL" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --gpu-memory-utilization "$UTIL" \
    --max-model-len "$MAX_MODEL_LEN" \
    "${SERVE_FLAGS[@]}" \
    "${EXTRA_ARGS[@]}" >>"$LOG" 2>&1 &
SERVER_PID=$!
echo "[serve] started vllm serve pid=$SERVER_PID; waiting up to ${WAIT}s for /health ..."

deadline=$(( $(date +%s) + WAIT ))
while true; do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[serve] FATAL: vllm serve process $SERVER_PID died before becoming healthy." >&2
    echo "------ tail of $LOG ------" >&2
    tail -n 60 "$LOG" >&2
    exit 1
  fi
  # The probe runs under the serving module's interpreter (already required
  # above to compose the flags), NOT curl: a container without curl -- Modal's
  # debian_slim and nvidia/cuda bases ship none -- made this loop read 000
  # against a HEALTHY server until the deadline killed it (2026-09-06).
  code="$("$SERVING_PY" "$SERVING_MOD" health --endpoint "${ENDPOINT}/health" --timeout 5 2>/dev/null || echo 000)"
  if [ "$code" = "200" ]; then
    echo "[serve] HEALTHY: ${ENDPOINT} (pid=$SERVER_PID)"
    # Provenance: the engine's own resolved-config line must agree with the
    # request, else this server is not handed out (exit 1, server killed).
    if ! "$SERVING_PY" "$SERVING_MOD" manifest "${SERVING_OPTS[@]}" \
         --out "$MANIFEST" --log "$LOG" --model "$MODEL" --served-name "$SERVED_NAME" \
         --endpoint "$ENDPOINT" \
         --extra-json "{\"gpu\": \"$GPU\", \"max_model_len_flag\": $MAX_MODEL_LEN, \"gpu_memory_utilization_flag\": $UTIL, \"server_pid\": $SERVER_PID}"; then
      echo "[serve] FATAL: the running engine contradicts the requested serving options (or the options are malformed); killing pid $SERVER_PID" >&2
      kill -TERM "$SERVER_PID" 2>/dev/null
      exit 1
    fi
    echo "VLLM_SERVER_PID=$SERVER_PID"
    echo "VLLM_ENDPOINT=$ENDPOINT"
    echo "VLLM_SERVING_MANIFEST=$MANIFEST"
    exit 0
  fi
  if [ "$(date +%s)" -ge "$deadline" ]; then
    echo "[serve] FATAL: /health did not return 200 within ${WAIT}s (last=$code)." >&2
    echo "------ tail of $LOG ------" >&2
    tail -n 60 "$LOG" >&2
    kill -TERM "$SERVER_PID" 2>/dev/null
    exit 1
  fi
  sleep 3
done
