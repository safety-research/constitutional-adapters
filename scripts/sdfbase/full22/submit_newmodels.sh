#!/bin/bash
# full22-newmodels: PRINT (never run) the sbatch commands for the nemotron3nano4b / glm46vflash legs, after
# checking — read-only, over ssh — that every bank view they consume is BUILT and COMPLETE, the model snapshot
# is in each cluster's cache at the pinned revision, the checkout exists, and the budgets. A SHORT view REFUSES
# the whole print (exit 2, nothing submittable printed). The logic lives in submit_newmodels.py (testable);
# this wrapper picks the python (the worktree venv when present — it carries yaml + the repo's src) and passes
# every argument through. Nothing here executes sbatch.
# Usage: submit_newmodels.sh <wave_sha> [--cluster glm46vflash=cluster_a] [--petri-cluster cluster_a]
#        [--cluster-a-budget N] [--cluster-b-budget N] [--offline] [--roster <yaml>] [--grading-host-worktree <path>]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../../.." && pwd)"
PY="${F22_PYTHON:-}"
[ -n "$PY" ] || { [ -x "$REPO/.venv/bin/python" ] && PY="$REPO/.venv/bin/python"; }
[ -n "$PY" ] || PY=$(command -v python3)
exec "$PY" "$HERE/submit_newmodels.py" "$@"
