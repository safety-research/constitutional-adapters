#!/bin/bash
#SBATCH --job-name=s2-jailbreak-gcg
#SBATCH --partition=general
#SBATCH --qos=high                    # re-run gpu-avail at submission (10% rule)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00             # GCG: target-only (~16GB) but COMPUTE-heavy; right-size per goal-count + n_steps
# stdout/stderr can carry harmful goal / response text, so logs go to a
# mode-0700 PRIVATE dir (NOT the world-readable exp/logs). Slurm opens the log
# BEFORE the script runs and does NOT create parent dirs, so PRE-CREATE it once
# before submitting:  install -d -m 700 $SHARED/exp/jailbreak/_logs
#SBATCH --output=/workspace-vast/%u/exp/jailbreak/_logs/%x_%A_%a.out
#SBATCH --error=/workspace-vast/%u/exp/jailbreak/_logs/%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# Tier-GCG (white-box, target-only) transcript generation. ONE ARRAY TASK =
# (defense x goal-shard). GCG is WHITE-BOX and TARGET-ONLY: it differentiates
# THROUGH the active steering object and needs NO 27B attacker, so it is CHEAPER
# IN MEMORY (target 8B only, ~16GB) but EXPENSIVE IN COMPUTE (its optimiser does
# ~n_steps * search_width forwards per goal — order 1e5, dwarfing the black-box
# forward_cap; GCG is exempt from that cap and reports its optimiser cost
# separately). This is a SEPARATE script from jailbreak_attacks.sbatch ON
# PURPOSE: adding gcg to that script's ATTACKS=(pair tap crescendo actorattack)
# array would change its stride and silently mislabel every running shard. Do
# NOT merge them. Grading is a SEPARATE grading host Batch post-step
# (scripts/jailbreak_run.py aggregate --attack gcg) — NOT here (JB5/JB6).
#
# Array layout: index = defense_idx * NSHARDS + shard.  DEFENSES order matches
# configs/stage2_jailbreak.yaml `defenses` (none prompted + the 22 compiled S2
# arms in methods.ARMS order, the appended antidote/alphasteer/lofit arms then
# the did/adasteer arms wired 2026-08-19 last); NSHARDS below. Example
# (24 defenses x 4 shards = 96; the axis has grown 4 -> 16 -> 19 -> 22 -> 24
# — never mix indices computed under an older stride):
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=0-95 scripts/sbatch/jailbreak_gcg.sbatch      # full GCG sweep
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=0-3  scripts/sbatch/jailbreak_gcg.sbatch      # 'none' arm, 4 shards
# The 22 compiled arms require the Phase-0 per-arm artifacts from
# build_safety_defense.py (out/stage2_jailbreak/defenses/<arm>/); 'none'/
# 'prompted' need none.
#
# NOTE: this is a TRANSCRIPT-generating job — raw goals+suffixes / responses land
# in the mode-0700 PRIVATE dir
# $SHARED/exp/jailbreak/<run>/gcg/<defense>/shard<NN>/ only. W&B is
# disabled (JB4); curves come from the separate aggregate-only run built from the
# sanitized JSON.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# JB4 transcript-job hardening on top of stage2_common's WANDB_MODE=offline:
# fully disable W&B (the run touches raw goal/response text).
export WANDB_DISABLED=true
export WANDB_CONSOLE=off

# DEFENSES must match configs/stage2_jailbreak.yaml `defenses` in ORDER (the
# defense_idx decode maps to this array). 24 = none prompted + the 22 arms
# (methods.ARMS registry order: the psr_mse_all/psr_kl_hard/psr_kl_all_hard
# arms sit between psr_kl_all and loreft; the divergence arms, the
# appended antidote/alphasteer/lofit arms, then the did/adasteer arms
# (wired 2026-08-19), last).
DEFENSES=(none prompted vector vector_all lora_r1 lora_r1_all token_bos token_faithful psr psr_kl psr_kl_all psr_mse_all psr_kl_hard psr_kl_all_hard loreft reft_r1 bidpo vector_rkl vector_js antidote alphasteer lofit did adasteer)
NSHARDS=${NSHARDS:-4}
IDX=${SLURM_ARRAY_TASK_ID:-0}
DEF=${DEFENSES[$((IDX / NSHARDS))]}
SHARD=$((IDX % NSHARDS))

s2_run uv run --no-sync python scripts/jailbreak_run.py generate \
  --config configs/stage2_jailbreak.yaml \
  --attack gcg --defense "$DEF" --shard "$SHARD" --nshards "$NSHARDS"
exit $?
