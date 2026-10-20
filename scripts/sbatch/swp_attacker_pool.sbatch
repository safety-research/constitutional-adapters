#!/bin/bash
#SBATCH --job-name=swp_attacker_pool
#SBATCH --partition=general
#SBATCH --qos=high                     # not preempted: the pool must outlive every array task
#SBATCH --gres=gpu:1                    # ONE 27B abliterated-Qwen attacker replica (raise to gpu:2 + two ports for 2 replicas)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00
#SBATCH --output=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.out
#SBATCH --error=/workspace-vast/%u/exp/jailbreak/_logs/%x_%j.err
#SBATCH --open-mode=append
#SBATCH --no-requeue

# audit-exempt: out-of-lockfile-venv — this job serves from the dedicated
# vllm attacker venv ($SHARED/envs/vllm-attacker), NOT the project lockfile:
# no `uv sync` runs here, so the cu128 entry-point guard does not apply (a
# missing venv fails vllm_attacker_serve.sh loudly).
#
# The ONE SHARED ATTACKER DEPLOYMENT for the whole unified sweep (lever 1). A
# single plain `vllm serve` of the unsteered 27B attacker, bound to this node's
# routable IP, serving EVERY condition's adaptive attacks over the cluster
# network. It advertises its endpoint to a shared file the sweep array reads,
# and stays up until walltime or a STOP sentinel. Turns ~1 attacker GPU PER
# CONDITION into a fixed pool — the single biggest lever.
#
# Submit this FIRST; wait for it to write the endpoint file; THEN submit
# swp_sweep.sbatch. Stop it with: touch $SHARED/exp/swp/pool/STOP
set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
REPO_DIR="${SWP_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-swp-repo}"
VLLM_VENV="$SHARED/envs/vllm-attacker"
PORT="${SWP_POOL_PORT:-8000}"
POOL_DIR="$SHARED/exp/swp/pool"
CFG="configs/stage2_jailbreak.yaml"
mkdir -p "$POOL_DIR"
source "$SHARED/scripts/hf_env.sh"; export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off
log(){ echo "[$(date -Is)] $*"; }
# ONE allocated GPU, strict subset of Slurm's set (hard rule 3).
IFS=',' read -ra _G <<< "${CUDA_VISIBLE_DEVICES:-}"; GPU="${_G[0]:?no GPU allocated}"
NODE_IP="$(hostname -I | awk '{print $1}')"
# the config writes attacker.path as $HF_HOME/hub/.../snapshots/<revision>: the helper expands it on this node and prints
# it only when that pinned snapshot is here whole; an unset placeholder or a missing snapshot stops the job, no repo-id fallback
MODEL_PATH="$(python3 "$REPO_DIR/scripts/attacker_model_path.py" "$REPO_DIR/$CFG")" || { log "FATAL: no attacker model path (above); refusing"; exit 1; }
SERVED_NAME="$(python3 -c "import yaml;print(yaml.safe_load(open('$REPO_DIR/$CFG'))['attacker']['model'])")"
log "shared attacker pool: node_ip=$NODE_IP gpu=$GPU port=$PORT model=$MODEL_PATH"
SERVE_LOG="$POOL_DIR/serve_${SLURM_JOB_ID}.log"
cd "$REPO_DIR"
# vllm serve binds 0.0.0.0 by default, so the server is reachable cross-node at
# NODE_IP:PORT; vllm_attacker_serve.sh health-checks locally then exits 0 with
# the server still running.
bash scripts/vllm_attacker_serve.sh --venv "$VLLM_VENV" --model "$MODEL_PATH" \
  --served-name "$SERVED_NAME" --gpu "$GPU" --port "$PORT" --util 0.90 \
  --max-model-len 8192 --wait 2400 --log "$SERVE_LOG" | tee "$POOL_DIR/status_${SLURM_JOB_ID}.txt" \
  || { log "FATAL: pool did not become healthy"; exit 1; }
VPID="$(sed -n 's/^VLLM_SERVER_PID=//p' "$POOL_DIR/status_${SLURM_JOB_ID}.txt" | tail -1)"
ENDPOINT="http://$NODE_IP:$PORT"
echo "$ENDPOINT" > "$POOL_DIR/endpoint.txt"
log "ADVERTISED endpoint=$ENDPOINT (wrote $POOL_DIR/endpoint.txt); pid=$VPID"
cleanup(){ kill -TERM "$VPID" 2>/dev/null; rm -f "$POOL_DIR/endpoint.txt"; }
trap cleanup EXIT SIGTERM SIGINT
# stay up until walltime or a STOP sentinel; drop the endpoint when gone.
while kill -0 "$VPID" 2>/dev/null; do
  [ -f "$POOL_DIR/STOP" ] && { log "STOP sentinel seen; shutting the pool down"; break; }
  sleep 30
done
log "pool exiting"
