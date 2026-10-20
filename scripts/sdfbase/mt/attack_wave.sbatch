#!/bin/bash
#SBATCH --job-name=sdfmt-attacks
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=0-20:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# sdfbase-mt: ONE base model's multi-turn attack job on the vLLM-SERVED lens
# target. SELF-CONTAINED 2-GPU task:
#   GPU A = the vLLM 27B abliterated-Qwen ATTACKER at 127.0.0.1 (in-job serve
#           RETRY loop against cluster A's cgroup race; --max-model-len 49152
#           so Crescendo's accumulating conversation never 400s at turn ~5);
#   GPU B = scripts/sdfbase/mt_serving_up.py --mode attacks --model-key "$SLUG": one vLLM+lens
#           server + RoutingProxy serving EVERY cell of the wave as a virtual
#           model at the `multiturn_attacks` geometry (8704 x 45);
#   drivers = one `jailbreak_run.py generate --target-engine served` process per
#           (cell, attack, shard), ALL concurrent (the target's batch is the
#           throughput lever; a single-stream driver leaves it idle).
# Usage (positional args only — never --export=, hard rule 1):
#   sbatch -J sdfmt-<slug>-<grouptag> scripts/sdfbase/mt/attack_wave.sbatch \
#       <slug> <attacks> <wave_sha> <wave> <root_tag> [nshards] [shards] [cells]
#     slug      qwen35_4b (QWEN ONLY)
#     attacks   comma list of crescendo,actorattack,pair (the job's group)
#     wave_sha  the commit the checkout $SISTEER_EVAL_REPO_ROOT/<sha> is at
#     wave      mt1 (the bank + config + namespace suffix)
#     root_tag  results | smoke_scratch  (transcripts_<root_tag> under the wave root)
#     nshards   default 9; shards default "0,1,2,3,4,5,6,7,8" (the FULL 259-goal confirmatory pool;
#               the §15 87-goal subset == shards 0,1,2; 1-goal smoke: 259 "0")
#     cells     comma list; default = every explicit cell of the wave config
#     lens_per_row  1 = SISTEER_LENS_PER_ROW_APPLY=1 for the target standup (post-vectorization A/B only)
#     reduction     per_row (default, what mt1 ran on) | stacked — NEVER mix within a wave
# Grading is a SEPARATE step on the grading host ($STAGE/sdfbase/mt/mt_grade.py).
# Completeness = UNIQUE goal_ids in progress.jsonl per (attack, cell), never sacct.
set -uo pipefail
SLUG="${1:?slug}"; ATTACKS="${2:?attacks}"; WAVE_SHA="${3:?wave sha}"; WAVE="${4:?wave}"
[ "$SLUG" = qwen35_4b ] || { echo "FATAL: this wave is qwen35_4b only (user scope change 2026-09-04); got $SLUG"; exit 1; }
ROOT_TAG="${5:?root_tag}"; NSHARDS="${6:-9}"; SHARDS="${7:-0,1,2,3,4,5,6,7,8}"; CELLS_ARG="${8:-}"; LENS_PER_ROW="${9:-0}"; REDUCTION_MODE="${10:-per_row}"
# REDUCTION MODE (positional 10, default per_row): the steering hook's reduction, pinned as
# SISTEER_LENS_STACKED_REDUCTIONS. mt1 ran entirely on per_row; every re-run / repair shard /
# top-up / resumed leg of it MUST stay per_row, because these are greedy generations compared
# BETWEEN cells and the stacked reductions differ by 1-16 bf16 ulps — enough to flip a token and
# put an implementation difference inside a between-cell contrast. Exported here as well as passed
# as a flag so anything the standup spawns inherits it.
case "$REDUCTION_MODE" in per_row) export SISTEER_LENS_STACKED_REDUCTIONS=0 ;; stacked) export SISTEER_LENS_STACKED_REDUCTIONS=1 ;; *) echo "FATAL: bad reduction mode $REDUCTION_MODE"; exit 1 ;; esac
# LENS A/B (the vectorized lens hook, branch lens-vectorize, keeps the per-row reference path behind
# SISTEER_LENS_PER_ROW_APPLY=1): positional 9 = 1 selects it for THIS job's target standup — a smoke
# A/B knob, never for a results run. Inert on pre-vectorization code (the env var is unread there).
[ "$LENS_PER_ROW" = 1 ] && export SISTEER_LENS_PER_ROW_APPLY=1
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
source "$SHARED/scripts/hf_env.sh"; export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy UV_PYTHON_INSTALL_DIR=$SHARED/.uv-pythons
export TOKENIZERS_PARALLELISM=false WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off PYTHONUNBUFFERED=1
log(){ echo "[$(date -Is)] $*"; }

REPO_DIR="${SISTEER_EVAL_REPO_ROOT:?set SISTEER_EVAL_REPO_ROOT, the root of the per-commit checkouts (the front door exports it from the site config)}/$WAVE_SHA"
[ -d "$REPO_DIR/src/sisteer" ] || { log "FATAL: no checkout at $REPO_DIR"; exit 1; }
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
[ "$HEAD_SHA" = "$WAVE_SHA" ] || { log "FATAL: HEAD $HEAD_SHA != $WAVE_SHA"; exit 1; }
git diff --quiet HEAD -- || { log "FATAL: dirty checkout $REPO_DIR"; exit 1; }
CFG=configs/sdfbase_transfer_${SLUG}_${WAVE}.yaml
# the manifest follows the wave label: mt1 -> petri_sdfbase_<slug>.yaml; a scratch wave (mt1smoke)
# -> petri_sdfbase_<slug>_<wave>.yaml (the standup refuses a manifest/config cell mismatch)
MANIFEST=configs/capeval/manifests/petri_sdfbase_${SLUG}$( [ "$WAVE" = mt1 ] || echo "_$WAVE" ).yaml
[ -f "$CFG" ] && [ -f "$MANIFEST" ] || { log "FATAL: missing $CFG or $MANIFEST"; exit 1; }
export TT_TWO_SI_DEFENSES_DIR=$SHARED/exp/objects/sdfbase/$WAVE/$SLUG/cells
WAVE_ROOT=$SHARED/exp/sdfbase_mt/$WAVE
GROUPTAG=$(echo "$ATTACKS" | tr ',' '-')
SERVE_DIR=$WAVE_ROOT/serving/attacks_${SLUG}_${GROUPTAG}_${SLURM_JOB_ID:-manual}
PRIVATE_ROOT=$WAVE_ROOT/transcripts_${ROOT_TAG}
mkdir -p "$SERVE_DIR/drivers" "$PRIVATE_ROOT"; chmod 700 "$PRIVATE_ROOT"
log "slug=$SLUG attacks=$ATTACKS sha=$WAVE_SHA wave=$WAVE root=$PRIVATE_ROOT nshards=$NSHARDS shards=$SHARDS reduction=$REDUCTION_MODE (SISTEER_LENS_STACKED_REDUCTIONS=$SISTEER_LENS_STACKED_REDUCTIONS) node=$(hostname)"

# --- GPU split: strict subset of Slurm's set (hard rule 3) ---
IFS=',' read -ra _G <<< "${CUDA_VISIBLE_DEVICES:-}"
[ "${#_G[@]}" -ge 2 ] || { log "FATAL: need 2 GPUs, got '${CUDA_VISIBLE_DEVICES:-}'"; exit 1; }
GPU_A="${_G[0]}"; GPU_B="${_G[1]}"
log "GPU_A(attacker)=$GPU_A GPU_B(target)=$GPU_B"

# --- driver venv: DEDICATED, pinned to WAVE_SHA (a shared venv gets re-synced by peers) ---
MAIN_VENV=/home/$USER/venvs/sdfmt-${WAVE_SHA:0:12}
LOCKDIR=$(dirname "$MAIN_VENV"); mkdir -p "$LOCKDIR"
SENTINEL="$LOCKDIR/.$(basename "$MAIN_VENV").synced"
(
  flock -x 9 || exit 99
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL" 2>/dev/null)" = "$WAVE_SHA" ] && [ -x "$MAIN_VENV/bin/python" ]; then
    log "driver venv ready @ ${WAVE_SHA:0:12}"
  else
    UV_PROJECT_ENVIRONMENT="$MAIN_VENV" uv sync --frozen --extra cu128 && echo "$WAVE_SHA" > "$SENTINEL"
  fi
) 9>"$LOCKDIR/.sdfmt-venv.lock"
MPY="$MAIN_VENV/bin/python"
[ -x "$MPY" ] || { log "FATAL: no driver venv at $MPY"; exit 1; }
"$MPY" -c 'import torch, pyrit, sisteer; print("driver venv OK", torch.__version__, pyrit.__version__, sisteer.__file__)' || { log "FATAL: driver venv guard"; exit 1; }
SIS_DIR=$("$MPY" -c 'import sisteer,os;print(os.path.dirname(os.path.dirname(sisteer.__file__)))')
case "$SIS_DIR" in *"$WAVE_SHA"*) log "sisteer pinned OK: $SIS_DIR";; *) log "FATAL: sisteer resolves to $SIS_DIR not $WAVE_SHA"; exit 1;; esac
"$MPY" - <<'PY' || { log "FATAL: PAIR budget is not 5x3 on this checkout"; exit 1; }
from sisteer.eval.attacks.pair_tap import PAIR_N_STREAMS; assert PAIR_N_STREAMS == 5, PAIR_N_STREAMS
PY

# --- lens serving venv (configs/capeval/vllm-requirements.txt, hash-versioned; the Petri stack) ---
VLLM_REQS="$REPO_DIR/configs/capeval/vllm-requirements.txt"
REQ_HASH=$(sha256sum "$VLLM_REQS" | cut -c1-8); REQ_HASH_FULL=$(sha256sum "$VLLM_REQS" | cut -c1-16)
SERVE_VENV="/home/$USER/venvs/sisteer-petri-vllm-$REQ_HASH"
SSENT="$LOCKDIR/.$(basename "$SERVE_VENV").vllm-bootstrap"
(
  flock -x 9 || exit 99
  if [ -f "$SSENT" ] && [ "$(cat "$SSENT")" = "$REQ_HASH_FULL" ] && [ -x "$SERVE_VENV/bin/python" ]; then
    log "serving venv already bootstrapped"
  else
    [ -x "$SERVE_VENV/bin/python" ] || uv venv --python 3.12 "$SERVE_VENV" || exit 1
    uv pip install --python "$SERVE_VENV/bin/python" --index-strategy unsafe-best-match --exclude-newer 2026-08-03T00:00:00Z -r "$VLLM_REQS" && echo "$REQ_HASH_FULL" > "$SSENT" || exit 1
  fi
) 9>"$LOCKDIR/.$(basename "$SERVE_VENV").uv-sync.lock"
SPY="$SERVE_VENV/bin/python"
[ -x "$SPY" ] || { log "FATAL: serving venv bootstrap failed"; exit 1; }
CUDA_VISIBLE_DEVICES="$GPU_B" "$SPY" -c 'import torch, vllm; torch.zeros(1).cuda(); print("serving venv OK vllm", vllm.__version__)' || { log "FATAL: serving venv CUDA init failed"; exit 1; }
unset VLLM_PLUGINS 2>/dev/null || true

# --- attacker venv (shared, vllm 0.25.1+cu129: it only runs `vllm serve`) ---
VLLM_VENV=$SHARED/envs/vllm-attacker
CUDA_VISIBLE_DEVICES="$GPU_A" "$VLLM_VENV/bin/python" -c 'import torch,vllm; torch.zeros(1).cuda(); print("attacker venv OK vllm",vllm.__version__)' || { log "FATAL: attacker venv CUDA init failed"; exit 1; }

# --- the bank: every explicit cell staged (object + sidecar + COMPLETE) ---
mapfile -t ALL_CELLS < <("$MPY" - "$CFG" <<'PY'
import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["explicit_cells"]))
PY
)
if [ -n "$CELLS_ARG" ]; then IFS=',' read -ra CELLS <<< "$CELLS_ARG"; else CELLS=("${ALL_CELLS[@]}"); fi
for c in "${ALL_CELLS[@]}"; do
  [ "$c" = none ] && continue
  for f in object.pt train_budget_meta.json COMPLETE; do
    [ -f "$TT_TWO_SI_DEFENSES_DIR/$c/$f" ] || { log "FATAL: bank $TT_TWO_SI_DEFENSES_DIR/$c lacks $f"; exit 1; }
  done
done
log "bank OK: $TT_TWO_SI_DEFENSES_DIR (${#ALL_CELLS[@]} cells incl. none); driving cells: ${CELLS[*]}"
# the config writes attacker.path as $HF_HOME/hub/.../snapshots/<revision>: the helper expands it on this node and prints
# it only when that pinned snapshot is here whole; an unset placeholder or a missing snapshot stops the job, no repo-id fallback
MODEL_PATH="$("$MPY" scripts/attacker_model_path.py "$CFG")" || { log "FATAL: no attacker model path (above); refusing"; exit 1; }
SERVED_NAME="$("$MPY" - "$CFG" <<'PY'
import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))["attacker"]["model"])
PY
)"

# --- teardown: SIGKILL the attacker (SIGTERM segfaults vLLM -> false FAILED), stop the target ---
VLLM_PID=""; TARGET_PID=""; declare -a DRIVER_PIDS=()
cleanup(){
  touch "$SERVE_DIR/STOP" 2>/dev/null || true
  for p in "${DRIVER_PIDS[@]:-}"; do [ -n "$p" ] && kill -TERM "$p" 2>/dev/null || true; done
  [ -n "${TARGET_PID:-}" ] && { for i in $(seq 1 24); do kill -0 "$TARGET_PID" 2>/dev/null || break; sleep 5; done; kill -9 "$TARGET_PID" 2>/dev/null || true; }
  [ -n "${VLLM_PID:-}" ] && kill -9 "$VLLM_PID" 2>/dev/null || true
}
trap cleanup EXIT SIGTERM SIGINT SIGQUIT

# --- Phase A: the vLLM attacker on GPU_A (retry loop: the cluster A cgroup race) ---
PORT=$((8137 + ${SLURM_JOB_ID:-0} % 800))
serve_ok=0
for satt in 1 2 3 4; do
  SPORT=$((PORT + satt))
  log "attacker vLLM attempt $satt on GPU $GPU_A port $SPORT (served=$SERVED_NAME)"
  bash scripts/vllm_attacker_serve.sh --venv "$VLLM_VENV" --model "$MODEL_PATH" \
    --served-name "$SERVED_NAME" --gpu "$GPU_A" --port "$SPORT" --util 0.85 \
    --max-model-len 49152 --wait 2400 --log "$SERVE_DIR/vllm_attacker_$satt.log" | tee "$SERVE_DIR/attacker_status.txt" || true
  VLLM_PID="$(sed -n 's/^VLLM_SERVER_PID=//p' "$SERVE_DIR/attacker_status.txt" | tail -1)"
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" 2>/dev/null; then PORT=$SPORT; serve_ok=1; break; fi
  log "attacker attempt $satt FAILED (cgroup race?); killing pid ${VLLM_PID:-none}, backoff 20s"
  [ -n "${VLLM_PID:-}" ] && kill -9 "$VLLM_PID" 2>/dev/null || true
  VLLM_PID=""; sleep 20
done
[ "$serve_ok" = 1 ] || { log "FATAL: attacker not healthy after 4 attempts"; exit 1; }
ATTACKER_URL="http://127.0.0.1:$PORT"
log "attacker healthy pid=$VLLM_PID at $ATTACKER_URL"

# --- Phase B: the served TARGET on GPU_B (all cells; attacks geometry) ---
# the serving venv's bin/ must be on PATH for THIS process: vs.launch_server shells out to `vllm`
# (a venv python invoked by absolute path does not put its bin/ on PATH — smoke job 375475)
CUDA_VISIBLE_DEVICES="$GPU_B" PATH="$SERVE_VENV/bin:$PATH" "$SPY" scripts/sdfbase/mt_serving_up.py --repo-dir "$REPO_DIR" \
  --config "$CFG" --manifest "$MANIFEST" --defenses-dir "$TT_TWO_SI_DEFENSES_DIR" \
  --out "$SERVE_DIR" --stop-file "$SERVE_DIR/STOP" --mode attacks --model-key "$SLUG" --reduction-mode "$REDUCTION_MODE" > "$SERVE_DIR/standup.log" 2>&1 &
TARGET_PID=$!
for i in $(seq 1 480); do
  [ -f "$SERVE_DIR/endpoint.json" ] && break
  kill -0 "$TARGET_PID" 2>/dev/null || { log "FATAL: target standup died (see $SERVE_DIR/standup.log)"; tail -30 "$SERVE_DIR/standup.log"; exit 1; }
  sleep 5
done
[ -f "$SERVE_DIR/endpoint.json" ] || { log "FATAL: target endpoint not up after 40 min"; exit 1; }
PROXY_URL=$("$MPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['proxy_url'])" "$SERVE_DIR/endpoint.json")
TABLE=$("$MPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['routing_table'])" "$SERVE_DIR/endpoint.json")
TARGET_L=$("$MPY" -c "import json,sys;print(json.load(open(sys.argv[1]))['max_model_len'])" "$SERVE_DIR/endpoint.json")
log "target up: $PROXY_URL window=$TARGET_L table=$TABLE"

# --- Phase C: the drivers, ALL concurrent, no GPU of their own ---
IFS=',' read -ra ATKS <<< "$ATTACKS"; IFS=',' read -ra SHARD_LIST <<< "$SHARDS"
declare -A DRIVER_TAG=()
# --target-greedy is PINNED, not inherited: this wave's configs are the mt1-era sdfbase_transfer_*
# ones and its filed stores are greedy. jailbreak_run's default target decode became "sample at the
# served model's own generation config" on 2026-09-11 (the canonical 4,096-cap instrument) and the
# served model's VENDOR PRESET on 2026-09-16; a top-up or repair shard of THIS wave must stay
# comparable with the rows already filed, so the pin stays whatever the default does.
for atk in "${ATKS[@]}"; do for cell in "${CELLS[@]}"; do for s in "${SHARD_LIST[@]}"; do
  tag="${atk}__${cell}__s$(printf '%02d' "$s")"
  ( cd "$REPO_DIR" && CUDA_VISIBLE_DEVICES="" "$MPY" scripts/jailbreak_run.py generate \
      --config "$CFG" --attack "$atk" --defense "$cell" --confirmatory \
      --shard "$s" --nshards "$NSHARDS" --private-root "$PRIVATE_ROOT" \
      --attacker-engine vllm --attacker-endpoint "$ATTACKER_URL" \
      --target-engine served --target-proxy-url "$PROXY_URL" --target-routing-table "$TABLE" \
      --target-max-model-len "$TARGET_L" --expect-pair-n-streams 5 \
      --target-greedy ) > "$SERVE_DIR/drivers/$tag.log" 2>&1 &
  pid=$!; DRIVER_PIDS+=("$pid"); DRIVER_TAG[$pid]="$tag"
  sleep 3   # stagger the per-driver salt-isolation + steering-bites probes
done; done; done
log "launched ${#DRIVER_PIDS[@]} drivers"
nfail=0
for pid in "${DRIVER_PIDS[@]}"; do
  wait "$pid"; rc=$?
  [ "$rc" = 0 ] || { nfail=$((nfail+1)); log "DRIVER FAILED rc=$rc: ${DRIVER_TAG[$pid]} (tail of log follows)"; tail -5 "$SERVE_DIR/drivers/${DRIVER_TAG[$pid]}.log"; }
done
DRIVER_PIDS=()
# --- completeness: UNIQUE goal_ids per (attack, cell) + zero error outcomes ---
PRIVATE_RUN=$("$MPY" -c "import sys,yaml;print(yaml.safe_load(open(sys.argv[1]))['private_run'])" "$CFG")
"$MPY" - "$PRIVATE_ROOT/$PRIVATE_RUN" "$ATTACKS" "$(IFS=,; echo "${CELLS[*]}")" <<'PY' | tee "$SERVE_DIR/completeness.txt"
import json, sys
from pathlib import Path
root, attacks, cells = Path(sys.argv[1]), sys.argv[2].split(","), sys.argv[3].split(",")
for atk in attacks:
    for cell in cells:
        goals, errs = set(), 0
        for sd in sorted((root / atk / cell).glob("shard*")):
            p = sd / "progress.jsonl"
            if not p.is_file():
                continue
            for line in p.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                goals.add(rec.get("goal_id"))
                if str(rec.get("outcome", "")).startswith("error"):
                    errs += 1
        print(f"COMPLETENESS {atk} {cell} unique_goals={len(goals)} error_outcomes={errs}")
PY
touch "$SERVE_DIR/STOP"
log "drivers done: ${nfail} failed; stopping the target"
exit $(( nfail > 0 ? 2 : 0 ))
