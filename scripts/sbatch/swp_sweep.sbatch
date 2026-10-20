#!/bin/bash
#SBATCH --job-name=swp_sweep
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1                    # ONE H200/task: one target model load serves a condition's JB legs; benign reloads once
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-12:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# The unified sweep array (lever 2: data parallelism over conditions). ONE task
# per condition; each loads the target ONCE, runs static + adaptive + benign,
# and points its adaptive attacks at the ONE shared attacker pool
# (swp_attacker_pool.sbatch). Submit AFTER the pool has written its endpoint.
#   sbatch --array=0-<N-1> scripts/sbatch/swp_sweep.sbatch
# where N = the conditions-table length (unified_sweep_run.py plan prints it).
# Do NOT launch the full matrix without an explicit go-ahead (cost model first).
set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
REPO_DIR="${SWP_REPO_DIR:-${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-swp-repo}"
CONDITIONS="${SWP_CONDITIONS:-configs/swp/conditions_focused.tsv}"
OUT_ROOT="${SWP_OUT_ROOT:-$SHARED/exp/swp/out}"
DEFENSES_DIR="${SWP_DEFENSES_DIR:-$SHARED/exp/jailbreak/two_si_defenses/cells}"
# Screening runs override bon.n_max WITHOUT editing the shared configs/two_si.yaml
# (SWP_BON_NMAX=64 to screen, 256 reserved for survivors). Empty -> use the
# config value. Passed as --bon-n-max only when set.
BON_NMAX="${SWP_BON_NMAX:-}"
POOL_DIR="$SHARED/exp/swp/pool"
MAIN_VENV="/home/$USER/venvs/jb-vllm-measure"   # cu129 + pyrit (the attacker-client venv)
source "$SHARED/scripts/hf_env.sh"; export HF_HUB_OFFLINE=1
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_CONSOLE=off
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UV_CACHE_DIR="$SHARED/.cache/uv" UV_LINK_MODE=copy TOKENIZERS_PARALLELISM=false
log(){ echo "[$(date -Is)] $*"; }
# EXPECTED_SHA gate on OUR checkout (never a peer's).
EXP="$(cat "$REPO_DIR/EXPECTED_SHA" 2>/dev/null || true)"
GOT="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || true)"
[ -n "$EXP" ] && [ "$EXP" = "$GOT" ] || { log "FATAL: $REPO_DIR HEAD=$GOT != EXPECTED_SHA=$EXP"; exit 1; }
# read the shared pool endpoint (must be up first).
ENDPOINT="$(cat "$POOL_DIR/endpoint.txt" 2>/dev/null || true)"
[ -n "$ENDPOINT" ] || { log "FATAL: no shared attacker pool endpoint at $POOL_DIR/endpoint.txt — submit swp_attacker_pool.sbatch first"; exit 1; }
log "condition task=$SLURM_ARRAY_TASK_ID pool=$ENDPOINT out=$OUT_ROOT"
cd "$REPO_DIR"
UV_PROJECT_ENVIRONMENT="$MAIN_VENV" uv sync --frozen --extra cu128 || { log "FATAL: frozen cu128 sync failed"; exit 1; }
# Runtime GPU guard (the 61380 silent-CPU lesson): assert the flavored
# sync landed the cu128 torch on a live CUDA device.
"$MAIN_VENV/bin/python" -c 'import torch; assert torch.cuda.is_available(), "CUDA not available in this env"; assert torch.__version__ == "2.11.0+cu128", f"torch {torch.__version__} != 2.11.0+cu128"; print("cu128 runtime assert OK:", torch.__version__)'
cleanup(){ kill -TERM -$$ 2>/dev/null; wait; }; trap cleanup SIGTERM SIGINT SIGQUIT
UV_PROJECT_ENVIRONMENT="$MAIN_VENV" uv run --no-sync python scripts/unified_sweep_run.py generate \
  --conditions "$CONDITIONS" --config configs/two_si.yaml \
  --out-root "$OUT_ROOT" --defenses-dir "$DEFENSES_DIR" \
  --task-id "$SLURM_ARRAY_TASK_ID" --attacker-endpoint "$ENDPOINT" --limit 100 \
  ${BON_NMAX:+--bon-n-max "$BON_NMAX"} &
wait $!
log "condition task $SLURM_ARRAY_TASK_ID done"
