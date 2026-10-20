#!/bin/bash
#SBATCH --job-name=attacker-pool
#SBATCH --partition=general
#SBATCH --qos=high                     # the standing default; not preempted: the pool must outlive every consumer. 1 of the 16-GPU/user high budget.
#SBATCH --gres=gpu:1                   # ONE 27B abliterated-Qwen attacker replica (POOL_REPLICAS, below: 2 GPUs + 2 ports is documented, not implemented)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                     # ~128G per GPU; NEVER --mem=0
#SBATCH --time=1-00:00:00              # the pool's lifetime; consumers re-verify the job is RUNNING, so a walltime end is a clean, loud refusal downstream
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# THE SHARED ATTACKER POOL for the served multi-turn attack lane: ONE plain
# `vllm serve` of the unsteered 27B attacker (huihui-ai/Huihui-Qwen3.6-27B-
# abliterated @27502c87), bound to this node's ROUTABLE address, serving every
# driver's adaptive attacks (PAIR / Crescendo / ActorAttack + the in-loop
# scorer) over the cluster network — so a target job needs its own GPU only
# (--gres=gpu:1) instead of the 2-GPU target+attacker topology whose attacker
# idles at ~0.6 requests in flight. The contract (endpoint.json, bearer, sizing,
# verification) is src/sisteer/eval/attacks/attacker_pool.py; the rationale,
# numbers and security posture are docs/attacker_pool.md.
#
# SUBMIT (from the login host), positional args = <repo sha> [<pool name>]; the checkout is
# commit-addressed so the pool runs exactly the code that was reviewed:
#   sbatch $CLUSTER_SBATCH_EXTRA scripts/sbatch/attacker_pool.sbatch <sha> [default]
# (node exclusions are a SUBMIT-time flag, the site's CLUSTER_SBATCH_EXTRA; a body-env
# cannot exclude nodes.)
# Then wait for the pool to advertise + verify before submitting consumers:
#   PYTHONPATH=<repo>/src python -m sisteer.eval.attacks.attacker_pool wait \
#       --pool $SHARED/exp/attacker-pool/<pool name>
# Stop it deliberately with the sentinel (a bare scancel also works — the trap
# removes endpoint.json either way):  touch <POOL_DIR>/STOP
#
# BODY-ENV OPTIONS (set in a wrapper's body or `env VAR=… sbatch …` — NEVER via
# --export=, hard rule 1; the defaults are the reviewed ones):
#   POOL_REPO_DIR        commit-addressed checkout (default $SISTEER_PROJECT_ROOT-attacker-pool-repo/<sha>)
#   POOL_PORT            8000
#   POOL_MAX_MODEL_LEN   19200 = attacker_pool.DEFAULT_MAX_MODEL_LEN (the p99 rule on the
#                        attacker's own calls; 20480 = Crescendo-only; 24576 covers every observed call)
#   POOL_MAX_NUM_SEQS    unset = the LARGEST S the memory model fits at the window (37 at 19200)
#                        under the no-chunk identity; an explicit S that does not fit is REFUSED
#   POOL_CHUNKED_PREFILL 0 = the identity max_num_batched_tokens = S x L (vLLM's profiling
#                        forward runs S x L; KV covers S x L, so no preemption at S);
#                        1 = chunked prefill at POOL_CHUNK_TOKENS (16384) — ~1.5x the S
#   POOL_UTIL            0.90 gpu_memory_utilization
#   POOL_REPLICAS        1 (documented only: 2 = --gres=gpu:2, two servers on PORT/PORT+1, two
#                        endpoint records; a consumer picks by hash of its shard. NOT implemented —
#                        this script refuses a value other than 1)
#   POOL_VLLM_VENV       $SHARED/envs/vllm-attacker (vllm 0.25.1+cu129, transformers 5.15.1)
#
# audit-exempt: out-of-lockfile-venv — this job serves from the dedicated vllm
# attacker venv, NOT the project lockfile: no `uv sync` runs here, so the cu128
# entry-point guard does not apply (a missing venv fails vllm_attacker_serve.sh
# loudly). attacker_pool.py is stdlib-only and imports from that venv via
# PYTHONPATH.
#
# RETIRED 2026-09-04: the cluster A/cluster B Slurm clusters and their shared
# filesystems no longer exist. This file remains only as the historical record
# of the measured geometry and old launch contract. The live implementation is
# scripts/modal/attacker_pool.py. It is impossible to launch this script.
echo "FATAL: retired launcher — the Slurm cluster no longer exists; use scripts/modal/attacker_pool.py" >&2
exit 64

set -uo pipefail

# --- the shared root: the job environment's SISTEER_SHARED_ROOT (the front door's job env, from the site's
# cluster block). The pool needs a root that honours modes: its key file must be 0600, and on a filesystem
# that ignores chmod write_api_key refuses (loudly).
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

POOL_SHA="${1:?REQUIRED positional 1: the commit sha to serve from (a commit-addressed checkout under \$SHARED)}"
POOL_NAME="${2:-default}"
REPO_DIR="${POOL_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-attacker-pool-repo/$POOL_SHA}"
VLLM_VENV="${POOL_VLLM_VENV:-$SHARED/envs/vllm-attacker}"
PORT="${POOL_PORT:-8000}"
POOL_DIR="$SHARED/exp/attacker-pool/$POOL_NAME"
UTIL="${POOL_UTIL:-0.90}"
CHUNKED="${POOL_CHUNKED_PREFILL:-0}"
CHUNK_TOKENS="${POOL_CHUNK_TOKENS:-16384}"
REPLICAS="${POOL_REPLICAS:-1}"
CFG="configs/two_si_gemma4e4b_reduced.yaml"     # the attacker block is identical in every two_si config

[ -s "$SHARED/scripts/hf_env.sh" ] || { echo "FATAL: $SHARED/scripts/hf_env.sh missing" >&2; exit 1; }
source "$SHARED/scripts/hf_env.sh"; export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off
export TOKENIZERS_PARALLELISM=false
log(){ echo "[$(date -Is)] [pool] $*"; }

[ "$REPLICAS" = "1" ] || { log "FATAL: POOL_REPLICAS=$REPLICAS is documented but not implemented (one replica per job); refusing"; exit 1; }
[ -d "$REPO_DIR/src/sisteer" ] || { log "FATAL: no commit-addressed checkout at $REPO_DIR — clone the mirror and check out $POOL_SHA there first; refusing"; exit 1; }
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
case "$HEAD_SHA" in "$POOL_SHA"*) ;; *) log "FATAL: checkout HEAD $HEAD_SHA != pinned $POOL_SHA; refusing"; exit 1;; esac
if ! git diff --quiet HEAD -- 2>/dev/null; then
  log "FATAL: tracked files modified in $REPO_DIR — the pool runs only from an immutable checkout; refusing"; exit 1
fi
[ -x "$VLLM_VENV/bin/python" ] || { log "FATAL: no vLLM attacker venv at $VLLM_VENV"; exit 1; }
PY="$VLLM_VENV/bin/python"
export PYTHONPATH="$REPO_DIR/src${PYTHONPATH:+:$PYTHONPATH}"
POOLCTL=(nice -n 5 "$PY" -m sisteer.eval.attacks.attacker_pool)

mkdir -p "$POOL_DIR"
# --- refuse to overwrite a LIVE pool's advertisement (its job is still RUNNING);
# a stale file (job gone) is removed and replaced.
set +e
"${POOLCTL[@]}" live-job --pool "$POOL_DIR"; lj=$?
set -e
case "$lj" in
  0) log "FATAL: $POOL_DIR/endpoint.json belongs to a RUNNING pool (above); refusing to overwrite — use another pool name or stop that pool"; exit 1;;
  2) log "stale endpoint.json in $POOL_DIR (its job is gone); replacing"; rm -f "$POOL_DIR/endpoint.json" "$POOL_DIR/api_key";;
  1) ;;
  *) log "FATAL: live-job check failed rc=$lj"; exit 1;;
esac
set +e

# --- ONE allocated GPU, strict subset of Slurm's set (hard rule 3: read, never set).
IFS=',' read -ra _G <<< "${CUDA_VISIBLE_DEVICES:-}"; GPU="${_G[0]:?no GPU allocated}"

# --- the ROUTABLE address: what the node NAME resolves to (Slurm's NodeAddr is the
# node name; /etc/hosts maps it onto the cluster's internal fabric). NOT `hostname -I |
# awk '{print $1}'`: on these nodes that is the docker-bridge 172.x address, and
# a pool that advertised it served ZERO requests.
NODE="$(hostname)"
NODE_IP="$(getent hosts "$NODE" | awk '{print $1}' | head -1)"
NODE_IP="$("${POOLCTL[@]}" routable "$NODE_IP")" || { log "FATAL: no routable address for $NODE"; exit 1; }
log "pool '$POOL_NAME' on $NODE ($NODE_IP) gpu=$GPU port=$PORT repo=$REPO_DIR @ $HEAD_SHA"

# --- the served attacker: the config's pinned snapshot path + served name. The config writes the
# path as $HF_HOME/hub/...; the helper expands it on this node and refuses an unset placeholder by name.
MODEL_PATH="$("$PY" scripts/attacker_model_path.py "$CFG")" || { log "FATAL: no attacker model path (above); refusing"; exit 1; }
SERVED_NAME="$("$PY" -c "import yaml;print(yaml.safe_load(open('$CFG'))['attacker']['model'])")"
SNAPSHOT="$("$PY" -c "import yaml;print(yaml.safe_load(open('$CFG'))['attacker']['snapshot'])")"
case "$MODEL_PATH" in *"$SNAPSHOT"*) ;; *) log "FATAL: attacker path $MODEL_PATH does not carry snapshot $SNAPSHOT; refusing"; exit 1;; esac
[ -f "$MODEL_PATH/config.json" ] || { log "FATAL: attacker snapshot not cached at $MODEL_PATH"; exit 1; }

# --- geometry: the memory model refuses an S x L that does not fit one H200.
SIZE_ARGS=(size --max-model-len "${POOL_MAX_MODEL_LEN:-19200}" --util "$UTIL" --sh)
[ -n "${POOL_MAX_NUM_SEQS:-}" ] && SIZE_ARGS+=(--max-num-seqs "$POOL_MAX_NUM_SEQS")
[ "$CHUNKED" = "1" ] && SIZE_ARGS+=(--chunked-prefill --chunk-tokens "$CHUNK_TOKENS")
GEOM_SH="$("${POOLCTL[@]}" "${SIZE_ARGS[@]}")" || { log "FATAL: geometry refused (above)"; exit 1; }
eval "$GEOM_SH"
log "geometry: L=$POOL_MAX_MODEL_LEN S=$POOL_MAX_NUM_SEQS ($POOL_SEQS_SOURCE) max_num_batched_tokens=$POOL_MAX_NUM_BATCHED_TOKENS chunked=$CHUNKED util=$UTIL"
"${POOLCTL[@]}" size --max-model-len "$POOL_MAX_MODEL_LEN" --max-num-seqs "$POOL_MAX_NUM_SEQS" --util "$UTIL" $([ "$CHUNKED" = "1" ] && echo "--chunked-prefill --chunk-tokens $CHUNK_TOKENS") | head -1

# --- the bearer: minted per launch into a 0600 file (refused if the FS will not
# hold the mode), exported into the SERVER's env only — vLLM's stock guard reads
# VLLM_API_KEY; it is never on argv (/proc/<pid>/cmdline is world-readable).
KEY_FILE="$POOL_DIR/api_key"
umask 077
"${POOLCTL[@]}" mint --out "$KEY_FILE" >/dev/null || { log "FATAL: could not mint the bearer into a private file"; exit 1; }
umask 022
VLLM_API_KEY="$(cat "$KEY_FILE")"; export VLLM_API_KEY
[ -n "$VLLM_API_KEY" ] || { log "FATAL: empty bearer"; exit 1; }
log "bearer minted -> $KEY_FILE (mode $(stat -c %a "$KEY_FILE")); the value is never printed"

# --- serve. Prefix caching ON (vllm's default, stated explicitly): the attacker is
# UNSTEERED, its requests share the PyRIT system prompt and each call re-sends the
# whole growing transcript, so every turn's prefill re-uses the previous turn's
# blocks — the crescendo p50 prompt is ~9k tokens of which only the last turn is new.
SERVE_LOG="$POOL_DIR/serve_${SLURM_JOB_ID}.log"
STATUS="$POOL_DIR/status_${SLURM_JOB_ID}.txt"
# The no-chunk identity is expressed the way the served lane's composer does it
# (vllm_server.build_server_launch): vLLM V1 keeps chunked prefill on, so the
# identity is max_num_batched_tokens = S x L — a chunk no request can exceed —
# never a --no-enable-chunked-prefill flag. POOL_CHUNKED_PREFILL=1 just sets the
# chunk to POOL_CHUNK_TOKENS instead.
EXTRA=(--host 0.0.0.0 --max-num-seqs "$POOL_MAX_NUM_SEQS" --enable-prefix-caching
       --max-num-batched-tokens "$POOL_MAX_NUM_BATCHED_TOKENS")
VPID=""
cleanup(){
  log "cleanup: retiring the pool advertisement + key, stopping the server"
  # only OUR advertisement is removed (a later pool may already own the file)
  if [ -f "$POOL_DIR/endpoint.json" ] && grep -q "\"job_id\": \"$SLURM_JOB_ID\"" "$POOL_DIR/endpoint.json" 2>/dev/null; then
    rm -f "$POOL_DIR/endpoint.json" "$KEY_FILE"
  fi
  if [ -n "${VPID:-}" ] && kill -0 "$VPID" 2>/dev/null; then
    kill -TERM "$VPID" 2>/dev/null; sleep 20
    kill -0 "$VPID" 2>/dev/null && kill -9 "$VPID" 2>/dev/null   # vLLM's TERM teardown can hang/segfault
  fi
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT
# In-job RETRY (the cluster A cgroup race kills ~half of vLLM starts node-agnostically;
# a failed attempt self-detects in minutes): up to 3 attempts, next port each time.
serve_ok=0
for satt in 1 2 3; do
  SPORT=$((PORT + satt - 1))
  log "vllm serve attempt $satt on port $SPORT"
  bash scripts/vllm_attacker_serve.sh --venv "$VLLM_VENV" --model "$MODEL_PATH" \
    --served-name "$SERVED_NAME" --gpu "$GPU" --port "$SPORT" --util "$UTIL" \
    --max-model-len "$POOL_MAX_MODEL_LEN" --wait 2400 --log "$SERVE_LOG" \
    -- "${EXTRA[@]}" | tee "$STATUS" || true
  VPID="$(sed -n 's/^VLLM_SERVER_PID=//p' "$STATUS" | tail -1)"
  if [ -n "$VPID" ] && kill -0 "$VPID" 2>/dev/null; then PORT=$SPORT; serve_ok=1; break; fi
  log "attempt $satt FAILED; backoff 20s"; [ -n "${VPID:-}" ] && kill -9 "$VPID" 2>/dev/null; VPID=""; sleep 20
done
[ "$serve_ok" = 1 ] || { log "FATAL: attacker pool not healthy after 3 attempts"; exit 1; }
URL="http://$NODE_IP:$PORT"

# --- serving attestation: the launcher verified the engine's resolved-config line
# against the requested weights/prefix-cache options and printed the manifest path;
# the endpoint record DERIVES its precision/cache fields from that manifest (never a
# literal here), and a consumer refuses a record without it.
SERVING_MANIFEST="$(sed -n 's/^VLLM_SERVING_MANIFEST=//p' "$STATUS" | tail -1)"
[ -n "$SERVING_MANIFEST" ] && [ -f "$SERVING_MANIFEST" ] \
  || { log "FATAL: the launcher printed no verified serving manifest; refusing to advertise an unattested pool"; exit 1; }

# --- coverage attestation: vLLM's own 'Maximum concurrency for L tokens per
# request: Xx' must cover the launched S, or the identity does not hold live.
COV_JSON="$("${POOLCTL[@]}" coverage --log "$SERVE_LOG" --max-num-seqs "$POOL_MAX_NUM_SEQS" --max-model-len "$POOL_MAX_MODEL_LEN")"; cov_rc=$?
echo "$COV_JSON"
if [ "$cov_rc" != 0 ]; then
  log "FATAL: KV coverage below the launched S (above) — refusing in both chunked and no-chunk modes; relaunch at the named S"; exit 1
fi

# --- verify the guard from THIS node on the ROUTABLE address before advertising:
# unauthenticated /v1/models must 401, the bearer must be accepted.
unauth=$(curl -s -o /dev/null -m 10 -w '%{http_code}' "$URL/v1/models" || echo 000)
auth=$(curl -s -o /dev/null -m 10 -w '%{http_code}' -H "Authorization: Bearer $VLLM_API_KEY" "$URL/v1/models" || echo 000)
log "guard check at $URL: unauthenticated /v1/models -> $unauth (want 401), bearer -> $auth (want 200)"
[ "$unauth" = "401" ] && [ "$auth" = "200" ] || { log "FATAL: the bearer guard is not enforced as expected; refusing to advertise"; exit 1; }

# --- advertise (atomic write; validated by the contract before anything lands).
"${POOLCTL[@]}" endpoint --pool-name "$POOL_NAME" --pool-dir "$POOL_DIR" --url "$URL" \
  --model "$SERVED_NAME" --snapshot "$SNAPSHOT" --model-path "$MODEL_PATH" \
  --max-model-len "$POOL_MAX_MODEL_LEN" --max-num-seqs "$POOL_MAX_NUM_SEQS" \
  --max-num-batched-tokens "$POOL_MAX_NUM_BATCHED_TOKENS" --util "$UTIL" \
  --api-key-file "$KEY_FILE" --job-id "$SLURM_JOB_ID" --node "$NODE" --node-ip "$NODE_IP" \
  --serve-log "$SERVE_LOG" --repo-sha "$HEAD_SHA" \
  --geometry-json "$POOL_GEOMETRY_JSON" --coverage-json "$COV_JSON" \
  --serving-manifest "$SERVING_MANIFEST" \
  --extra "chunked_prefill=$CHUNKED" --extra "vllm_venv=$VLLM_VENV" \
  || { log "FATAL: could not write endpoint.json"; exit 1; }
log "ADVERTISED $URL -> $POOL_DIR/endpoint.json (job $SLURM_JOB_ID, pid $VPID); consumers: --attacker-pool $POOL_DIR"

# --- stay up until walltime, a STOP sentinel, or the server dies; heartbeat the
# in-flight profile every 10 min so a quiet pool is explainable from one tail.
hb=0
while kill -0 "$VPID" 2>/dev/null; do
  [ -f "$POOL_DIR/STOP" ] && { log "STOP sentinel seen; shutting the pool down"; break; }
  hb=$((hb + 1))
  if [ $((hb % 20)) -eq 0 ]; then
    "${POOLCTL[@]}" logstats --log "$SERVE_LOG" | tr -d '\n' | sed 's/  */ /g'; echo
  fi
  sleep 30
done
kill -0 "$VPID" 2>/dev/null || log "server process $VPID exited on its own (see $SERVE_LOG)"
log "pool exiting"
