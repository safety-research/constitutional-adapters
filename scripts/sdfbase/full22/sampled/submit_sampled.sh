#!/bin/bash
# full22 SAMPLED re-run: PRINT (never run) the sbatch commands for the sampled jobs of the four main-view targets,
# after checking — offline: both config generators' --check; over read-only ssh: the checkout at the wave sha, the
# model snapshot, make_bank_view.py --check per target, this user's `high` GPUs vs the budget, the wave STOP. The
# logic lives in submit_sampled.py (testable); this wrapper picks the python (F22_PYTHON, else the repo venv, else
# python3) and passes every argument through. Nothing here executes sbatch.
# Usage: submit_sampled.sh <wave_sha> [--targets k1,k2] [--cluster <key>=cluster_a|cluster_b ...]
#        [--cluster-a-budget N] [--cluster-b-budget N] [--exclude [cluster:]NODES ...] [--nshards 9] [--offline]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../../../.." && pwd)"
PY="${F22_PYTHON:-}"
[ -n "$PY" ] || { [ -x "$REPO/.venv/bin/python" ] && PY="$REPO/.venv/bin/python"; }
[ -n "$PY" ] || PY=$(command -v python3)
exec "$PY" "$HERE/submit_sampled.py" "$@"
