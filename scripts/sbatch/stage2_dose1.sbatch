#!/bin/bash
#SBATCH --job-name=s2-dose1
#SBATCH --partition=general
#SBATCH --qos=high                    # the default QoS; re-run gpu-avail at submission
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=0-08:00:00             # ~73 GSM8K gens (~2.5h) + logprob metrics, padded ~2x; cache resumes
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# Fixed-dose panel: every (SI, arm) cell at strength 1.0 — on-target
# effect vs prompting (from recorded per-prompt outcomes) + off-target
# MMLU/GSM8K/GPQA-d/MMLU-Pro deltas vs unsteered, measuring only what the
# panels / F1 / earlier fill-ins did not already pay for (write-through cache
# dose1_fillins.json). Single task; resume-safe.
#
#   sbatch --array=0 --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) \
#     scripts/sbatch/stage2_dose1.sbatch

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

s2_run uv run --no-sync python scripts/stage2_dose1.py \
  --out-dir "$SISTEER_PROJECT_ROOT/out/stage2_llama8b/ifeval" \
  --config configs/stage2_llama8b.yaml
exit $?
