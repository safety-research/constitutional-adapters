#!/bin/bash
#SBATCH --job-name=sisteer-smoke
#SBATCH --partition=general
#SBATCH --qos=high                    # set from gpu-avail at submit time; cluster was <10% idle on 2026-08-10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-04:00:00             # ~2h expected; padded ~2x, not 5x (backfill)
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
# Submit with the mandatory submit-time SHA pin:
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) scripts/sbatch/smoke_qwen7b.sbatch

set -uo pipefail

# Ported onto stage2_common: s2_init enforces the submit-time SHA
# pin + EXPECTED_SHA guard, runs the flock'd rc-checked
# `uv sync --frozen --extra cu128` and the srun-time cu128 assert,
# exports the env the old inline block set by hand (hf_env.sh, offline W&B,
# FIGS_DIR + mkdir, the UV_* trio), cds into the repo, and arms
# TERM-escalation + the EXIT figure mirror (the same mirror-figs the old
# trap ran; the in-trap `wandb sync` is dropped — compute nodes
# can't reach api.wandb.ai; sync the offline runs from a login node).
# BEHAVIOR CHANGE: s2_init exports HF_HUB_OFFLINE=1; this script used
# to run online. The Qwen 7B weights are cached (this job has run), so a
# resubmit is unaffected — for a model not yet in the cache, prefetch it
# from a login node first (or `export HF_HUB_OFFLINE=0` after s2_init).
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# Resume-safe: the CLI skips instruction/method combos whose results.json
# already exists, so a preempted retry appends rather than recomputes.
s2_run uv run --no-sync python -m sisteer fit-eval --config configs/smoke_qwen7b.yaml
exit $?
