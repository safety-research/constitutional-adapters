#!/bin/bash
#SBATCH --job-name=jb-vllm-measure
#SBATCH --partition=general,overflow
#SBATCH --qos=low                     # high budget is full from other sessions; this is a short one-shot measurement. Re-run if preempted (no auto-escalation).
#SBATCH --gres=gpu:2                  # TWO GPUs: one for the vLLM attacker, one for the HF steered target
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=2:30:00
#SBATCH --output=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.out
#SBATCH --error=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#
# One-shot A/B measurement: the SAME small ActorAttack workload run with the
# attacker on HF transformers (--attacker-engine hf, the shipped default) vs on
# a separate vanilla vLLM process (--attacker-engine vllm), reporting wall-clock
# PER GOAL both ways and the resulting end-to-end speedup.
#
# GPU LAYOUT is auto-detected from the allocation. The PRODUCTION design is TWO
# GPUs per task (--gres=gpu:2): GPU A serves the 27B abliterated-Qwen ATTACKER
# over vLLM (unhooked -> vanilla vLLM, ~15x faster generation); GPU B hosts the
# Llama-3.1-8B steered TARGET on HF transformers (a residual-stream forward
# HOOK; vLLM cannot host it). When only ONE GPU is allocated (--gres=gpu:1, the
# budget-constrained measurement mode) it runs CO-RESIDENT: the vLLM attacker
# (util ~0.5, ~70GB of the 141GB H200) and the HF target (~16GB) share the one
# card. Co-residence isolates the engine swap at a FIXED GPU count, which is a
# clean controlled A/B (it can only understate vLLM's win, since the attacker's
# KV cache is squeezed).
#
# CRITICAL -- the HF baseline must NOT pay vLLM's memory reservation. Phase C
# (the HF arm) runs FIRST, with the vLLM server still DOWN, so the HF baseline
# has the whole card; the server is only started (Phase D) after the HF process
# has exited and freed its memory. Otherwise a live server during the HF arm
# would handicap it and OVERSTATE the speedup.
#
# NOT CROSS-ENGINE COMPARABLE. vLLM and HF produce different attacker text, so
# this measures WALL-CLOCK only; the two runs' ASR/transcripts are NOT
# comparable (each run records its engine in attacker_engine.json).
#
# Pre-create the private 0700 log dir once before submitting:
#   install -d -m 700 $SHARED/exp/jailbreak/_logs
#
# Usage:  sbatch scripts/sbatch/jailbreak_vllm_measure.sbatch <repo-dir>
# where <repo-dir> is a clean extract of the commit under test (a git archive
# unpacked on shared storage), not a git checkout -- keeps the run reproducible
# without a git remote on the cluster.
set -uo pipefail

REPO_DIR="${1:?pass the repo dir (clean git-archive extract) as positional arg 1}"
# Optional positional arg 2: reuse a HF-arm wall-clock (seconds) from a prior
# clean run and SKIP Phase C, so a re-run only needs the vLLM arm. Passed
# positionally, never via sbatch --export (hard rule). The prior HF run must
# also have had the vLLM server down (it did -- Phase C always runs server-down).
HF_SECS_OVERRIDE="${2:-}"
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
VLLM_VENV="$SHARED/envs/vllm-attacker"
MAIN_VENV="/home/$USER/venvs/jb-vllm-measure"        # node-local, isolated from the shared project venv
PORT=8137
SCRATCH="$SHARED/exp/jailbreak/_vllm_measure/$SLURM_JOB_ID"
CFG="configs/stage2_jailbreak.yaml"
DEFENSE="prompted"                                   # no compiled artifact needed; attacker dominates so the speedup is defense-independent
ATTACK="actorattack"
N_GOALS_TARGET=4

log() { echo "[$(date -Is)] $*"; }

# -- env (cluster rules: SHARED hf_env.sh, offline HF, no --export) ---------- #
source "$SHARED/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy
export TOKENIZERS_PARALLELISM=false
mkdir -p "$SCRATCH"

# -- GPU split: strict subset of Slurm's allocation (hard rule 3) ------------ #
log "CUDA_VISIBLE_DEVICES from Slurm: ${CUDA_VISIBLE_DEVICES:-unset}"
IFS=',' read -ra _GPUS <<< "${CUDA_VISIBLE_DEVICES:-}"
if [ "${#_GPUS[@]}" -lt 1 ]; then
  log "FATAL: no allocated GPU (CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-none}')"; exit 1
fi
if [ "${#_GPUS[@]}" -ge 2 ]; then
  GPU_A="${_GPUS[0]}"; GPU_B="${_GPUS[1]}"; VLLM_UTIL="0.85"
  MODE="2-GPU (attacker vLLM on GPU A, HF target on GPU B -- separate cards)"
else
  # 1-GPU co-resident: the 27B attacker in bf16 is ~54GB, so vLLM util 0.5
  # (~70GB) leaves ~71GB for the 8B HF target (~16GB) + activations + hooks.
  GPU_A="${_GPUS[0]}"; GPU_B="${_GPUS[0]}"; VLLM_UTIL="0.5"
  MODE="1-GPU co-resident (attacker vLLM + HF target share GPU ${_GPUS[0]})"
fi
log "mode: $MODE ; GPU_A=$GPU_A GPU_B=$GPU_B vllm_util=$VLLM_UTIL"

[ -f "$REPO_DIR/$CFG" ] || { log "FATAL: repo extract $REPO_DIR missing $CFG"; exit 1; }
log "repo dir: $REPO_DIR"
# -- Phase B: build the main (cu128 + pyrit) venv from the warm cache -------- #
log "Phase B: frozen cu128 sync of the main venv at $MAIN_VENV"
mkdir -p "$(dirname "$MAIN_VENV")"
UV_PROJECT_ENVIRONMENT="$MAIN_VENV" uv sync --frozen --extra cu128 || { log "FATAL: frozen cu128 sync failed"; exit 1; }
MPY="$MAIN_VENV/bin/python"
# Runtime GPU guard (the 61380 silent-CPU lesson) for the MAIN venv —
# the vllm venv proves its own CUDA in Phase A below. Runs FIRST so no
# python workload precedes the guard.
"$MPY" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__)'

# the config writes attacker.path as $HF_HOME/hub/...: the helper expands it on this node and refuses an unset placeholder by name
MODEL_PATH="$("$MPY" "$REPO_DIR/scripts/attacker_model_path.py" "$REPO_DIR/$CFG")" || { log "FATAL: no attacker model path (above); refusing"; exit 1; }
SERVED_NAME="$("$MPY" - "$REPO_DIR/$CFG" <<'PY'
import sys, yaml
print(yaml.safe_load(open(sys.argv[1]))["attacker"]["model"])
PY
)"

# -- Phase A: vLLM venv CUDA init on GPU A (catches the driver-570 mismatch) - #
log "Phase A: vLLM venv CUDA sanity on GPU $GPU_A"
CUDA_VISIBLE_DEVICES="$GPU_A" "$VLLM_VENV/bin/python" - <<'PY' || { echo "FATAL: vLLM venv CUDA init failed (report the exact error above; do NOT try random versions)"; exit 1; }
import torch, vllm
print("vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)
x = torch.zeros(1).cuda(); print("cuda tensor OK on", torch.cuda.get_device_name(0))
PY

# -- size the workload to ~N_GOALS_TARGET goals from the real total ---------- #
TOTAL="$(cd "$REPO_DIR" && UV_PROJECT_ENVIRONMENT="$MAIN_VENV" "$MPY" - "$CFG" <<'PY'
import sys, pathlib
sys.path.insert(0, "scripts")
import jailbreak_run as jr
cfg = jr._load_cfg(sys.argv[1])
print(len(jr._eval_goals(cfg, confirmatory=False)))
PY
)"
NSHARDS=$(( (TOTAL + N_GOALS_TARGET - 1) / N_GOALS_TARGET ))
log "total eval goals=$TOTAL -> NSHARDS=$NSHARDS (shard 0 ~= $N_GOALS_TARGET goals)"

run_generate() {  # $1=engine, rest=VAR=val env assignments; writes secs to $SCRATCH/<engine>.secs
  local engine="$1"; shift
  local out="$SCRATCH/$engine"; mkdir -p "$out"
  local t0 t1 rc
  t0="$(date +%s)"
  # `env` applies the VAR=val args (they come from expansion, so a bare prefix
  # would NOT be honored); generate output goes to the job log, not captured.
  ( cd "$REPO_DIR" && env UV_PROJECT_ENVIRONMENT="$MAIN_VENV" "$@" \
      "$MPY" scripts/jailbreak_run.py generate \
      --config "$CFG" --attack "$ATTACK" --defense "$DEFENSE" --target-greedy \
      --shard 0 --nshards "$NSHARDS" --private-root "$out" \
      --attacker-engine "$engine" \
      ${VLLM_ENDPOINT_ARG:-} ) 2>&1 | sed "s/^/[$engine] /"
  rc=${PIPESTATUS[0]}
  t1="$(date +%s)"
  echo "$((t1 - t0))" > "$SCRATCH/$engine.secs"
  return "$rc"
}

# -- Phase C: HF baseline -- vLLM server is DOWN, so it is NOT handicapped ---- #
VLLM_ENDPOINT_ARG=""
if [ -n "$HF_SECS_OVERRIDE" ]; then
  HF_SECS="$HF_SECS_OVERRIDE"
  log "Phase C: SKIPPED -- reusing HF_SECS=${HF_SECS}s from a prior clean server-down run"
else
  log "Phase C: HF-engine baseline on GPU $GPU_A (vLLM server DOWN -- HF has the whole card)"
  run_generate hf CUDA_VISIBLE_DEVICES="$GPU_A" || { log "FATAL: HF-engine run failed"; exit 1; }
  HF_SECS="$(cat "$SCRATCH/hf.secs")"
  log "HF-engine wall-clock: ${HF_SECS}s for $N_GOALS_TARGET goals"
fi

# -- Phase D: start the vLLM attacker server on GPU A ------------------------ #
log "Phase D: start vLLM attacker server on GPU $GPU_A"
SERVE_LOG="$SCRATCH/vllm_serve.log"
if ! bash scripts/vllm_attacker_serve.sh \
      --venv "$VLLM_VENV" --model "$MODEL_PATH" --served-name "$SERVED_NAME" \
      --gpu "$GPU_A" --port "$PORT" --util "$VLLM_UTIL" --max-model-len 8192 \
      --wait 2400 --log "$SERVE_LOG" | tee "$SCRATCH/serve_status.txt"; then
  log "FATAL: vLLM server did not become healthy"; exit 1
fi
VLLM_PID="$(sed -n 's/^VLLM_SERVER_PID=//p' "$SCRATCH/serve_status.txt" | tail -1)"
log "vLLM server pid=$VLLM_PID healthy at http://127.0.0.1:$PORT"
cleanup() { [ -n "${VLLM_PID:-}" ] && kill -TERM "$VLLM_PID" 2>/dev/null; }
trap cleanup EXIT

# -- Phase E: vLLM engine (target HF on GPU B, attacker over HTTP) ----------- #
log "Phase E: vLLM-engine run (target on GPU $GPU_B, attacker via vLLM on GPU $GPU_A)"
VLLM_ENDPOINT_ARG="--attacker-endpoint http://127.0.0.1:$PORT"
run_generate vllm CUDA_VISIBLE_DEVICES="$GPU_B" || { log "FATAL: vLLM-engine run failed"; exit 1; }
VLLM_SECS="$(cat "$SCRATCH/vllm.secs")"
log "vLLM-engine wall-clock: ${VLLM_SECS}s for $N_GOALS_TARGET goals"

# -- Phase F: report --------------------------------------------------------- #
python3 - "$HF_SECS" "$VLLM_SECS" "$N_GOALS_TARGET" "$TOTAL" "$DEFENSE" "$ATTACK" <<'PY'
import sys
hf, vl, n, total, defense, attack = (float(sys.argv[1]), float(sys.argv[2]),
    int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6])
print("\n================ vLLM-attacker A/B measurement ================")
print(f"workload: {attack} x defense={defense}, {n} goals (of {total} total)")
print(f"HF   engine: {hf:8.1f}s total  = {hf/n:7.1f}s/goal")
print(f"vLLM engine: {vl:8.1f}s total  = {vl/n:7.1f}s/goal")
if vl > 0:
    print(f"END-TO-END SPEEDUP: {hf/vl:5.2f}x  (attacker moved HF -> vLLM; target stays HF)")
print("NOTE: wall-clock only. vLLM and HF produce different attacker text, so")
print("ASR/transcripts across the two engines are NOT comparable.")
print("===============================================================\n")
PY
log "done"
