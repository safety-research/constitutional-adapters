#!/bin/bash
#SBATCH --job-name=classc-rr
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --array=0-5
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append

# Class C (response-only fit) rerun on safety_refusal, one cell per array
# task, at classc-response-only fb6327b (functionally identical to
# accel-fit 6aba597 — the branch adds the boundary-anchor tests and a
# docstring fix only). Config = the psr_retest two_si_psr.yaml, sha
# asserted, so these numbers are directly comparable to jobs 193659/193660
# (psr_mse_all A 0.091 / C 14.18). Cells:
#   0 psr_kl_all  Class A   (baseline: unmeasured since the schedule opt-in)
#   1 psr_kl_all  Class C   (the headline question)
#   2 lora_r1     Class C   (old wave: 0.612, batch-0-era probe)
#   3 vector      Class C   (old wave: 0.630)
#   4 lofit       Class C   (old wave: 2.816)
#   5 diffmean    Class C   (old wave: 0.785; fit-position-independent floor)
# bidpo / bidpo_lora are NOT here: their trainer refuses (correctly)
# without the emit-pairs sidecars, which do not exist beside this pool.

set -u
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"
source $SHARED/scripts/hf_env.sh
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export UV_CACHE_DIR=$SHARED/.cache/uv UV_LINK_MODE=copy

TASK=${SLURM_ARRAY_TASK_ID}
export UV_PROJECT_ENVIRONMENT=/home/$USER/venvs/si-classc-t${TASK}

ARMS=(psr_kl_all psr_kl_all lora_r1 vector lofit diffmean)
POSN=(all response response response response response)
CELL=(psr_kl_all_A psr_kl_all_C lora_r1_C vector_C lofit_C diffmean_C)
ARM=${ARMS[$TASK]}
POS=${POSN[$TASK]}
OUT=$SHARED/exp/classc_rerun/${CELL[$TASK]}
REPO=$SHARED/exp/classc_rerun/repo_t${TASK}
WANT_SHA=fb6327b7a4a8c7674566fb61312dc703394e1eb9

mkdir -p "$OUT"
# W&B stays ON (offline on the cluster; synced from the grading host after) —
# never WANDB_DISABLED.
export WANDB_MODE=offline
export WANDB_DIR=$OUT

# Own clone per array task (never share a working tree between jobs).
if [ ! -d "$REPO/.git" ]; then
  git clone -q -b classc-response-only "${SISTEER_REPO_MIRROR:?set SISTEER_REPO_MIRROR, the cluster clone of this repository (the front door exports it from the site config)}" "$REPO"
fi
git -C "$REPO" fetch -q origin classc-response-only
git -C "$REPO" checkout -q "$WANT_SHA"
GOT_SHA=$(git -C "$REPO" rev-parse HEAD)
if [ "$GOT_SHA" != "$WANT_SHA" ]; then
  echo "FATAL: repo at $GOT_SHA, wanted $WANT_SHA" >&2
  exit 90
fi

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

cd "$REPO" || exit 91

POS_FLAG=()
if [ "$POS" != "all" ]; then POS_FLAG=(--positions "$POS"); fi

srun bash -c '
  set -e
  uv sync --frozen --extra cu128 2>&1 | tail -2
  # Runtime GPU guard (61380): the frozen sync must have landed cu128 torch
  # on a live CUDA device — assert, never just print.
  uv run --no-sync python -c "import torch
assert torch.cuda.is_available(), \"CUDA not available in this env\"
assert torch.__version__ == \"2.11.0+cu128\", f\"torch {torch.__version__} != 2.11.0+cu128\""
  uv run --no-sync python scripts/train_two_si_objects.py train \
    --config '"$SHARED"'/psr_retest/two_si_psr.yaml \
    --expect-config-sha 1c8174b2e0fc516c \
    --si safety_refusal --arms '"$ARM"' '"${POS_FLAG[*]:-}"' \
    --pairs '"$SHARED"'/ratio/pool/pairs_safety_refusal.jsonl \
    --out '"$OUT"'
'
rc=$?
echo "[classc-rr] task=$TASK arm=$ARM positions=$POS exit=$rc"
exit $rc
