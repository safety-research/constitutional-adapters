#!/bin/bash
#SBATCH --job-name=attacker-pool-canary
#SBATCH --partition=general
#SBATCH --qos=high                     # the standing default; 1 of the 16-GPU/user high budget
#SBATCH --gres=gpu:1                   # the TARGET only — the attacker is the shared pool (that is the point)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                     # ~128G per GPU; NEVER --mem=0
#SBATCH --time=0-03:00:00              # ~5 crescendo goals on a prompted cell (~5-10 min each) + venv sync + model load, padded ~2x
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# THE POOL CANARY: stands up NO attacker itself; expects a LIVE pool
# (scripts/sbatch/attacker_pool.sbatch) and runs ONE small multi-turn job
# against it — 5 goals of Crescendo on gemma4e4b's prompted__safety_refusal
# cell (HF target on this job's single GPU) — asserting, in order:
#   1. CROSS-NODE REACHABILITY: this node != the pool's node, and the pool's
#      advertised URL answers /health from here. This is the canary's FIRST
#      assertion because it has never been demonstrated on cluster A: the
#      2026-08-30 pool advertised the docker-bridge address and served zero
#      requests.
#   2. THE GUARD: unauthenticated /v1/models -> 401; the pool's bearer -> 200
#      listing the pinned attacker at the pinned snapshot and window; the
#      pool's Slurm job is RUNNING (attacker_pool verify).
#   3. THE DRIVER PLUMBING: jailbreak_run.py generate --attacker-pool <POOL_DIR>
#      runs to completion; its attacker_engine.json names the pool URL as the
#      attacker endpoint and carries the pool identity; every attacker call in
#      attacker_calls.jsonl / scorer_calls.jsonl succeeded (0 errors).
#   4. THE POOL'S VIEW: the vLLM serve log's 'Running:' stats lines over the
#      canary's window — max/mean requests in flight, generation tok/s, prefix
#      cache hit rate — printed as the throughput report.
# Any failed assertion exits non-zero with the reason; nothing falls back.
#
# SUBMIT (from the login host), AFTER the pool advertises; positional 1 = the repo sha
# (the same commit-addressed checkout the pool serves from), positional 2 =
# the pool name (default 'default'):
#   sbatch $CLUSTER_SBATCH_EXTRA scripts/sbatch/attacker_pool_canary.sbatch <sha> [default]
# BODY-ENV OPTIONS (never --export=): CANARY_ATTACK (crescendo), CANARY_CELL
# (prompted__safety_refusal), CANARY_NSHARDS (52 -> ~5 of the 259 confirmatory
# goals in shard 0), CANARY_SHARD (0), CANARY_CFG (configs/two_si_gemma4e4b_reduced.yaml),
# CANARY_REPO_DIR, CANARY_DEFENSES_DIR ($SHARED/exp/objects/scale-12/gemma4e4b).
#
# RETIRED 2026-09-04: the Slurm cluster this canary targeted no longer exists.
# This file remains as historical evidence only. The live separate-container
# proof is scripts/modal/attacker_pool.py::canary. It is impossible to launch
# this script.
echo "FATAL: retired canary — the Slurm cluster no longer exists; use the Modal attacker-pool canary" >&2
exit 64

set -uo pipefail

SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"

CANARY_SHA="${1:?REQUIRED positional 1: the commit sha (commit-addressed checkout under \$SHARED)}"
POOL_NAME="${2:-default}"
REPO_DIR="${CANARY_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-attacker-pool-repo/$CANARY_SHA}"
POOL_DIR="$SHARED/exp/attacker-pool/$POOL_NAME"
ATTACK="${CANARY_ATTACK:-crescendo}"
CELL="${CANARY_CELL:-prompted__safety_refusal}"
NSHARDS="${CANARY_NSHARDS:-52}"
SHARD="${CANARY_SHARD:-0}"
CFG="${CANARY_CFG:-configs/two_si_gemma4e4b_reduced.yaml}"
OUT_ROOT="$SHARED/exp/attacker-pool/_canary/${SLURM_JOB_ID}"
export TT_TWO_SI_DEFENSES_DIR="${CANARY_DEFENSES_DIR:-$SHARED/exp/objects/scale-12/gemma4e4b}"
export SISTEER_HELDOUT_CORPUS_DIR="$SHARED/exp/heldout_corpora"

[ -s "$SHARED/scripts/hf_env.sh" ] || { echo "FATAL: $SHARED/scripts/hf_env.sh missing" >&2; exit 1; }
source "$SHARED/scripts/hf_env.sh"; export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off
log(){ echo "[$(date -Is)] [canary] $*"; }
fail(){ log "ASSERTION FAILED: $*"; exit 1; }

[ -d "$REPO_DIR/src/sisteer" ] || fail "no commit-addressed checkout at $REPO_DIR"
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
case "$HEAD_SHA" in "$CANARY_SHA"*) ;; *) fail "checkout HEAD $HEAD_SHA != pinned $CANARY_SHA";; esac
mkdir -p "$OUT_ROOT"
T_START="$(date -u '+%m-%d %H:%M:%S')"      # vLLM's log timestamp format, for the pool-log window

# --- the drivers' venv: the project lockfile (cu128 + pyrit), DEDICATED to this
# sha (a shared venv gets re-synced to other commits by concurrent sessions).
MAIN_VENV="/home/$USER/venvs/attacker-pool-canary-${CANARY_SHA:0:12}"
LOCKDIR=$(dirname "$MAIN_VENV"); mkdir -p "$LOCKDIR"
SENTINEL="$LOCKDIR/.$(basename "$MAIN_VENV").synced"
(
  flock -x 9 || exit 99
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL" 2>/dev/null)" = "$CANARY_SHA" ] && [ -x "$MAIN_VENV/bin/python" ]; then
    echo "[canary] dedicated venv ready @ ${CANARY_SHA:0:12}"
  else
    # the sync's result is checked on the SAME command (this shell runs without
    # errexit): a failed sync must never leave an older sha's venv in place
    # behind a fresh sentinel, or the driver would run under a stale env
    UV_PROJECT_ENVIRONMENT="$MAIN_VENV" uv sync --frozen --extra cu128 \
      || { echo "[canary] FATAL: frozen cu128 sync into $MAIN_VENV failed" >&2; exit 98; }
    echo "$CANARY_SHA" > "$SENTINEL"
  fi
) 9>"$LOCKDIR/.attacker-pool-canary-venv.lock" || fail "dedicated venv sync/lock failed (rc=$?)"
MPY="$MAIN_VENV/bin/python"
[ -x "$MPY" ] || fail "no dedicated venv at $MPY (frozen cu128 sync failed)"
# Runtime GPU guard (the silent-CPU lesson) + the editable sisteer resolves to THIS checkout
"$MPY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available"; assert torch.__version__ == "2.11.0+cu128", torch.__version__; print("cu128 runtime assert OK:", torch.__version__)' || fail "venv guard"
SIS_DIR=$("$MPY" -c 'import sisteer,os;print(os.path.dirname(os.path.dirname(sisteer.__file__)))')
case "$SIS_DIR" in "$REPO_DIR"*) log "sisteer pinned OK: $SIS_DIR";; *) fail "sisteer resolves to $SIS_DIR, not $REPO_DIR";; esac
POOLCTL=("$MPY" -m sisteer.eval.attacks.attacker_pool)

# ============ 1. CROSS-NODE REACHABILITY ============================================
[ -f "$POOL_DIR/endpoint.json" ] || fail "no live pool: $POOL_DIR/endpoint.json is absent (submit attacker_pool.sbatch first and wait for it)"
POOL_URL="$("$MPY" -c "import json;print(json.load(open('$POOL_DIR/endpoint.json'))['url'])")"
POOL_NODE="$("$MPY" -c "import json;print(json.load(open('$POOL_DIR/endpoint.json'))['node'])")"
POOL_JOB="$("$MPY" -c "import json;print(json.load(open('$POOL_DIR/endpoint.json'))['job_id'])")"
POOL_LOG="$("$MPY" -c "import json;print(json.load(open('$POOL_DIR/endpoint.json')).get('serve_log') or '')")"
log "pool: $POOL_URL on $POOL_NODE (job $POOL_JOB); canary on $(hostname)"
if [ "$(hostname)" = "$POOL_NODE" ]; then
  fail "the canary landed on the pool's own node; cross-node separation is uncertified (resubmit with --exclude=$POOL_NODE)"
fi
CROSS_NODE=1
code=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$POOL_URL/health" || echo 000)
[ "$code" = "200" ] || fail "cross-node reachability: GET $POOL_URL/health from $(hostname) -> $code (want 200). The pool is not routable from this node"
log "ASSERT 1 OK: $POOL_URL/health -> 200 from $(hostname) (cross_node=$CROSS_NODE)"

# ============ 2. THE GUARD + IDENTITY (the contract's verify) =======================
unauth=$(curl -s -o /dev/null -m 15 -w '%{http_code}' "$POOL_URL/v1/models" || echo 000)
[ "$unauth" = "401" ] || fail "guard: unauthenticated GET $POOL_URL/v1/models -> $unauth (want 401)"
VERIFY_JSON="$("${POOLCTL[@]}" verify --pool "$POOL_DIR")" || fail "attacker_pool verify refused the pool (above)"
echo "$VERIFY_JSON" > "$OUT_ROOT/pool_verify.json"
log "ASSERT 2 OK: unauthenticated -> 401; bearer accepted; model/snapshot/window verified; job RUNNING"

# ============ 3. THE DRIVER, against the pool =======================================
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
log "driver: $ATTACK x $CELL shard $SHARD/$NSHARDS (~5 goals) --attacker-pool $POOL_DIR -> $OUT_ROOT"
# backgrounded + waited so the TERM trap runs inside the grace window; no
# output pipe here — a pipe would make $? the pipe tail's status, not the driver's
"$MPY" scripts/jailbreak_run.py generate \
    --config "$CFG" --attack "$ATTACK" --defense "$CELL" --confirmatory \
    --shard "$SHARD" --nshards "$NSHARDS" --private-root "$OUT_ROOT" \
    --attacker-pool "$POOL_DIR" &
wait $!
rc=$?
T_END="$(date -u '+%m-%d %H:%M:%S')"
[ "$rc" = 0 ] || fail "the driver exited rc=$rc"
PRIVATE_RUN="$("$MPY" -c "import yaml;print(yaml.safe_load(open('$CFG'))['private_run'])")"
SHARD_DIR="$(printf '%s/%s/%s/%s/shard%02d' "$OUT_ROOT" "$PRIVATE_RUN" "$ATTACK" "$CELL" "$SHARD")"
[ -d "$SHARD_DIR" ] || fail "no shard dir at $SHARD_DIR"
"$MPY" - "$SHARD_DIR" "$POOL_URL" "$POOL_JOB" <<'PY' || fail "ledger assertions (above)"
import json, sys
from pathlib import Path
sd, url, job = Path(sys.argv[1]), sys.argv[2].rstrip("/"), sys.argv[3]
m = json.loads((sd / "attacker_engine.json").read_text())
assert m["attacker_engine"] == "vllm", m["attacker_engine"]
assert m["attacker_endpoint"].rstrip("/") == url, (m["attacker_endpoint"], url)
pool = m.get("attacker_pool") or {}
ident = pool.get("identity") or {}
assert ident.get("url", "").rstrip("/") == url and str(ident.get("job_id")) == job, ident
assert ident.get("snapshot") and ident.get("model"), ident
assert (pool.get("evidence") or {}).get("ok") is True, pool.get("evidence")
def rows(name):
    p = sd / name
    if not p.exists():
        return []
    out = []
    for line in p.read_text(errors="replace").splitlines():
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out
att, sco = rows("attacker_calls.jsonl"), rows("scorer_calls.jsonl")
assert att, "no attacker calls recorded"
errs = [r for r in att + sco if r.get("error")]
assert not errs, f"{len(errs)} attacker/scorer call(s) errored; first: {errs[0].get('error')!r}"
goals = {}
for r in rows("progress.jsonl"):
    goals[r.get("goal_id")] = r.get("outcome")
clamped = sum(1 for r in att if (r.get("meta") or {}).get("clamped_max_tokens"))
print(json.dumps({"goals_done": len(goals), "outcomes": goals, "attacker_calls": len(att),
                  "scorer_calls": len(sco), "attacker_call_errors": 0,
                  "attacker_mean_latency_s": round(sum(r["latency_s"] for r in att) / len(att), 2),
                  "attacker_calls_clamped": clamped}, indent=2))
PY
log "ASSERT 3 OK: attacker_engine.json names the pool ($POOL_URL, job $POOL_JOB); all attacker/scorer calls succeeded"

# ============ 4. THE POOL'S VIEW: in-flight + throughput over the canary window =====
if [ -n "$POOL_LOG" ] && [ -f "$POOL_LOG" ]; then
  log "pool serve log window $T_START .. $T_END:"
  "${POOLCTL[@]}" logstats --log "$POOL_LOG" --since "$T_START" --until "$T_END" | tee "$OUT_ROOT/pool_logstats.json"
  grep -c "POST /v1/chat/completions" "$POOL_LOG" >/dev/null 2>&1 && log "pool log: $(grep -c 'POST /v1/chat/completions' "$POOL_LOG") chat completions logged in total"
else
  log "pool serve log not readable from here ($POOL_LOG); in-flight report skipped"
fi
log "CANARY PASSED (cross_node=$CROSS_NODE): pool $POOL_URL served $ATTACK x $CELL; artifacts under $OUT_ROOT"
