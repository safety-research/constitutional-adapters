#!/bin/bash
#SBATCH --job-name=petri-stage2-check
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0-01:00:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#
# Run a REAL stage-2 object through the lens adapters on the real 8B
# model, in plain HF — deliberately NOT through vLLM. The question is
# whether the ADAPTERS reproduce the object's own intervention and
# produce sensible steered text; isolating that from the serving wiring
# means a later vLLM failure can be attributed rather than guessed at.
#
# No --export= anywhere (hard rule 1); env is set in the body.

set -u
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
source "$SHARED/scripts/hf_env.sh"
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TOKENIZERS_PARALLELISM=false

cleanup() { kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

PETRI_SHA="${1:?usage: sbatch petri_stage2_check.sbatch <commit-sha>}"
REPO="${SISTEER_PETRI_REPO_ROOT:?set SISTEER_PETRI_REPO_ROOT, the root of the per-commit Petri checkouts (the front door exports it from the site config)}/$PETRI_SHA"
[ -d "$REPO/src/sisteer" ] || { echo "FATAL: no checkout at $REPO" >&2; exit 1; }
cd "$REPO"

# Immutability check (the petri_steered_smoke.sbatch pattern): the
# checkout must BE the pinned commit, or the record's provenance would
# name code that did not run.
HEAD_SHA=$(git rev-parse HEAD 2>/dev/null || tr -d '[:space:]' < .git/HEAD)
if [ "$HEAD_SHA" != "$PETRI_SHA" ]; then
  echo "FATAL: checkout HEAD $HEAD_SHA != pinned $PETRI_SHA; refusing" >&2
  exit 1
fi
if ! git diff --quiet HEAD -- 2>/dev/null; then
  echo "FATAL: tracked files modified in $REPO — this check runs only" \
       "from an immutable checkout; refusing" >&2
  exit 1
fi

OUT_DIR="$SHARED/petri/serving/stage2_check"
mkdir -p "$OUT_DIR"

# The pinned model: meta-llama, NOT the NousResearch mirror. The weights
# are byte-identical but the chat templates differ (the official one
# always emits a system block), and for prefill-position arms a
# different render length is a different intervention.
MODEL=meta-llama/Llama-3.1-8B-Instruct
# The pinned snapshot (the stage-2 / smoke pin — cache-resident on
# cluster A): without it the check floats on whatever `main` resolves
# to at run time, and the record cannot say which weights it judged.
REVISION=0e9e39f249a16976918f6564b8830bc894c89659

# The stage-2 PSR objects. PSR steers every position in both phases, so
# it needs no position context — which is exactly why it is the right
# arm for a plain-HF adapter check.
OBJECTS="
$SHARED/exp/training/capbattery/defenses/cells/safety_refusal__psr_kl_all/object.pt
$SHARED/exp/training/capbattery/defenses/cells/safety_refusal__psr_mse_all/object.pt
"

# --- the env, built here rather than found ------------------------------------
# /home/$USER on this cluster is NODE-LOCAL disk, so the set of venvs
# differs per node and "pick one by glob" is a lottery — it drew a venv
# without accelerate on one node and a different one on the next. Build
# a lockfile-pinned env instead, keyed by the lock's hash, with a flock
# and a content sentinel so concurrent jobs on a node neither race the
# build nor mutate a completed one (the petri_probe.sbatch pattern).
LOCK_HASH=$(sha256sum "$REPO/uv.lock" | cut -c1-16)
export UV_PROJECT_ENVIRONMENT="/home/$USER/venvs/petri-stage2-${LOCK_HASH:0:8}"
export UV_CACHE_DIR="$SHARED/.cache/uv"
export UV_LINK_MODE=copy
LOCKDIR=$(dirname "$UV_PROJECT_ENVIRONMENT")
mkdir -p "$LOCKDIR"
SENTINEL="$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").stage2-bootstrap"
(
  flock -x 9 || exit 99
  if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL")" = "$LOCK_HASH" ]; then
    echo "[stage2] env already built @ ${LOCK_HASH:0:8} (sentinel)"
  else
    # --extra cu128 is REQUIRED, not optional: pyproject deliberately keeps
    # torch out of the base deps so a flavor must be chosen explicitly, and
    # a flavor-less sync installs no torch at all — by design, so that a
    # missing choice is a loud ImportError instead of a silent CPU fallback.
    # cu128 is the cluster's flavor (driver 570.x cannot init cu13x wheels).
    uv sync --frozen --no-dev --extra cu128 --project "$REPO" \
      && echo "$LOCK_HASH" > "$SENTINEL"
  fi
) 9>"$LOCKDIR/.$(basename "$UV_PROJECT_ENVIRONMENT").lock"
RC=$?
[ "$RC" -eq 0 ] || { echo "FATAL: env build failed (rc=$RC)" >&2; exit "$RC"; }

PY="$UV_PROJECT_ENVIRONMENT/bin/python"
# THE cu128 GUARD, and it must run BEFORE any workload. Importability is
# not enough: a CPU-flavoured torch imports perfectly and then runs the
# whole job on CPU, silently and about a hundred times slower — the
# 61380 failure. So assert the GPU is actually there AND that the torch
# build is the pinned cu128 one, not merely that torch exists.
"$PY" -c "import importlib.util, torch
assert torch.__version__ == \"2.11.0+cu128\", f\"torch {torch.__version__} != 2.11.0+cu128 — the flavor extra did not take\"
assert torch.cuda.is_available(), \"torch.cuda.is_available() is False — this job would run on CPU and silently produce numbers nobody would question\"
assert torch.cuda.device_count() >= 1, \"no CUDA device visible\"
assert importlib.util.find_spec(\"transformers\") is not None, \"transformers is not importable\"
print(\"[stage2] cu128 guard ok:\", torch.__version__, torch.cuda.get_device_name(0))" \
  || { echo "FATAL: cu128 guard failed — refusing to run" >&2; exit 1; }
echo "[stage2] python=$PY repo=$REPO model=$MODEL"

rc=0
for OBJ in $OBJECTS; do
  [ -f "$OBJ" ] || { echo "MISSING: $OBJ" >&2; rc=1; continue; }
  NAME=$(basename "$(dirname "$OBJ")")
  echo "=================================================================="
  echo "[stage2] $NAME"
  srun --ntasks=1 env PYTHONPATH="$REPO/src" "$PY" \
    scripts/petri_stage2_adapter_check.py \
      --object "$OBJ" --model "$MODEL" --revision "$REVISION" \
      --out "$OUT_DIR/${NAME}.json" || rc=1
done
echo "[stage2] done rc=$rc; records in $OUT_DIR"
exit $rc
