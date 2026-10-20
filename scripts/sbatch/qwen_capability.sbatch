#!/bin/bash
#SBATCH --job-name=qwen-cap
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # NEVER --mem=0
#SBATCH --time=0-08:00:00             # HF capability, one pinned load, 5 controls; qwen GDN torch fallback is slow; ~2x pad
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# exclude bad nodes at submission: sbatch $CLUSTER_SBATCH_EXTRA <this script> (CLUSTER_SBATCH_EXTRA is site config)
#
# Capability controls for ONE lens-free qwen cell, on HF (never the proxy
# — a served benign leg would change the engine basis and break
# comparability with every other model's capability numbers; the qwen
# scope is JB + capability + Petri). Runs cell_controls_run.py
# generate: one pinned HF load, mmlu parity (free from rows.jsonl; gsm8k
# is the served standard's — cell_controls_run refuses it)
# + xstest/or_bench_hard_1k judged specs (graded LATER on the
# grading host with the sonnet judge). qwen3_5 GDN runs the transformers
# TORCH FALLBACK here (capeval venv has no fla/causal_conv1d — a warned,
# correct slow path, verified in the modeling import guards).
#
# NO equivalence gate: that §4.1 step is for COMPILED cells (engine==lane);
# a lens-free cell (none / prompted, no object) is pure HF, nothing to
# reconcile, and cmd_generate runs no gate itself.
#
# Judge model: LEFT DEFAULT = capeval JUDGE_MODEL = claude-sonnet-5 (the
# benign judge; distinct from the JB haiku judge). It binds at GENERATE
# time into the store key, and the grading host's judge phase must pass
# the SAME (default) value; do not override it.
#
# Submit (self-scoped name; BOTH positional args required):
#   sbatch -J qwen-cap-<mk>-<cell>-f0830 scripts/sbatch/qwen_capability.sbatch \
#          <checkout-sha> <qwen35_9b|qwen35_4b> <none|prompted__safety_refusal|prompted__si_constitution|prompted__si_irrelevant>
# Then JUDGE on the grading host (Low-Prio key; separate, coordinated):
#   uv run --group capeval python scripts/cell_controls_run.py judge \
#     --cell <cell> --config configs/two_si_<mk>.yaml \
#     --out-root <OUT_ROOT> --defenses-dir <bank>
# Stop: scancel <job_id> ONLY. NO --export= (hard rule 1).

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the cluster shared root (the front door exports it from the site config)}"
source "$SHARED/scripts/hf_env.sh"
export HF_HUB_OFFLINE=1
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

SHA="${1:?usage: sbatch qwen_capability.sbatch <sha> <model-key> <cell>}"
MK="${2:?model-key}"
CELL="${3:?cell}"
case "$MK" in qwen35_9b|qwen35_4b) ;; *) echo "FATAL: bad model key $MK" >&2; exit 1;; esac
case "$CELL" in
  none|prompted__safety_refusal|prompted__si_constitution|prompted__si_irrelevant) ;;
  *) echo "FATAL: $CELL is not a lens-free cell" >&2; exit 1;;
esac

REPO_DIR="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$SHA"
[ -d "$REPO_DIR/src/sisteer" ] || { echo "FATAL: no checkout at $REPO_DIR" >&2; exit 1; }
cd "$REPO_DIR"
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
[ "$HEAD_SHA" = "$SHA" ] || { echo "FATAL: HEAD $HEAD_SHA != $SHA" >&2; exit 1; }
git diff --quiet HEAD -- 2>/dev/null || { echo "FATAL: tracked files modified" >&2; exit 1; }
echo "[cap] $REPO_DIR @ $HEAD_SHA on $(hostname): $MK / $CELL"

# --- capeval lane venv (lockfile-governed: uv sync --frozen --group capeval) --
# Same venv name + sync args capeval_common.sh uses (S2_UV_ENV /
# S2_UV_SYNC_ARGS), built inline from THIS checkout so it does not depend
# on the shared evals checkout's sha-gating. The lockfile is identical
# across checkouts, so the resulting venv equals the evals lane's exactly.
export UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/sisteer-capeval"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy
LOCKDIR=$(dirname "$UV_PROJECT_ENVIRONMENT"); mkdir -p "$LOCKDIR"
(
  flock -x 9 || exit 99
  uv sync --frozen --extra cu128 --python 3.12 --group capeval
) 9>"$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").uv-sync.lock"
RC=$?; [ "$RC" -ne 0 ] && { echo "FATAL: capeval venv sync rc=$RC" >&2; exit "$RC"; }
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"

BANK="$SHARED/exp/objects/scale-12/${MK}"
OUT_ROOT="$SHARED/exp/training/qwen_capability/${MK}"
mkdir -p "$OUT_ROOT"

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

# --judge-model LEFT DEFAULT (sonnet-5); --steer-positions all (moot,
# lens-free); --limit 0 (full). mmlu free; xstest,orbench judged.
# Runtime GPU guard (61380): the frozen sync must have landed cu128 torch
# on a live CUDA device — assert, never just print.
uv run --no-sync python -c "import torch; assert torch.cuda.is_available() and torch.__version__ == '2.11.0+cu128', torch.__version__"

srun --ntasks=1 uv run --no-sync python scripts/cell_controls_run.py generate \
  --cell "$CELL" \
  --config "configs/two_si_${MK}.yaml" \
  --controls mmlu,xstest,or_bench_hard_1k \
  --defenses-dir "$BANK" \
  --out-root "$OUT_ROOT" \
  --limit 0
RC=$?
echo "[cap] exit=$RC ($MK/$CELL) out-root=$OUT_ROOT"
exit $RC
