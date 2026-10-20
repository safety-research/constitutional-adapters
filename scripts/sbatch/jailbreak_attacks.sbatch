#!/bin/bash
#SBATCH --job-name=s2-jailbreak-attacks
#SBATCH --partition=general
#SBATCH --qos=high                    # re-run gpu-avail at submission (10% rule)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=1-00:00:00             # attacker(27B)+target(8B) co-resident; right-size per (attack,goal-count)
# stdout/stderr can carry PyRIT prompts / error text, so logs go to a
# mode-0700 PRIVATE dir (NOT the world-readable exp/logs). Slurm opens the log
# BEFORE the script runs and does NOT create parent dirs, so PRE-CREATE it once
# before submitting:  install -d -m 700 $SHARED/exp/jailbreak/_logs
#SBATCH --output=/workspace-vast/%u/exp/jailbreak/_logs/%x_%A_%a.out
#SBATCH --error=/workspace-vast/%u/exp/jailbreak/_logs/%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# Tier-1/2 transcript generation (PAIR/TAP/Crescendo/ActorAttack). ONE ARRAY
# TASK = (attack x defense x goal-shard). The abliterated Qwen attacker (27B,
# ~54GB) co-resides with the target (8B, ~16GB) on one H200 (141GB), sdpa. BoN
# has its own target-only script (jailbreak.sbatch); this is the attacker-driven
# battery. Grading is a SEPARATE grading host Batch post-step
# (scripts/jailbreak_run.py aggregate --attack <atk>) — NOT here (JB5/JB6).
#
# Array layout: index = attack_idx*(NDEF*NSHARDS) + defense_idx*NSHARDS + shard.
# ATTACKS order = (pair tap crescendo actorattack); DEFENSES = the 24 of
# configs/stage2_jailbreak.yaml (none prompted + the 22 compiled S2 arms, in
# methods.ARMS order, the appended antidote/alphasteer/lofit arms then the
# did/adasteer arms wired 2026-08-19 last); NSHARDS below. NDEF is derived
# from the array, so the attack stride is 24*4=96 — never mix indices
# computed under a stale defense stride (this axis was 4, then 16, then 19,
# then 22, now 24). Example (4 attacks x 24 defenses x 4 shards = 384):
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=0-383 scripts/sbatch/jailbreak_attacks.sbatch     # full battery
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) --array=0-95  scripts/sbatch/jailbreak_attacks.sbatch     # PAIR only, 24 def x 4 shards
# The 22 compiled arms require the Phase-0 per-arm artifacts from
# build_safety_defense.py (out/stage2_jailbreak/defenses/<arm>/). Run the
# Phase-1 canary FIRST (jailbreak_run.py canary --stage generate) — this array
# is gated on that acceptance check. COMPUTE: the object axis is 6x the
# original 4-defense matrix + 22 distilled artifacts (the final defense set + the
# psr-hard-mse arms + the did/adasteer arms wired 2026-08-19, approved).
#
# NOTE: TRANSCRIPT-generating job — raw attacker prompts / target responses /
# PyRIT SQLite memory land in the mode-0700 PRIVATE dir
# $SHARED/exp/jailbreak/<run>/<attack>/<defense>/shard<NN>/ only.
# W&B is disabled (JB4); curves come from the separate aggregate-only run built
# from the sanitized JSON.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# JB4 transcript-job hardening on top of stage2_common's WANDB_MODE=offline:
# fully disable W&B (this job touches raw attacker/response text; PyRIT/errors
# print prompts to stdout, which W&B would capture and wandb sync would upload).
export WANDB_DISABLED=true
export WANDB_CONSOLE=off

# DEFENSES must match configs/stage2_jailbreak.yaml `defenses` in ORDER (the
# index decode maps defense_idx -> this array; a reorder silently mislabels
# every shard). 24 = none prompted + the 22 methods.ARMS in registry order
# (the psr_mse_all/psr_kl_hard/psr_kl_all_hard arms between psr_kl_all and
# loreft; the divergence arms, the appended antidote/alphasteer/lofit
# arms, then the did/adasteer arms (wired 2026-08-19), last).
ATTACKS=(pair tap crescendo actorattack)
DEFENSES=(none prompted vector vector_all lora_r1 lora_r1_all token_bos token_faithful psr psr_kl psr_kl_all psr_mse_all psr_kl_hard psr_kl_all_hard loreft reft_r1 bidpo vector_rkl vector_js antidote alphasteer lofit did adasteer)
NSHARDS=${NSHARDS:-4}
NDEF=${#DEFENSES[@]}
IDX=${SLURM_ARRAY_TASK_ID:-0}
PER_ATK=$((NDEF * NSHARDS))
ATK=${ATTACKS[$((IDX / PER_ATK))]}
REM=$((IDX % PER_ATK))
DEF=${DEFENSES[$((REM / NSHARDS))]}
SHARD=$((REM % NSHARDS))

# --target-greedy is PINNED: the stage-2 jailbreak wave (pair/tap/crescendo/actorattack on
# configs/stage2_jailbreak.yaml) filed its rows under a greedy target, and jailbreak_run's DEFAULT
# decode became sampled on 2026-09-11. A re-run of this array must reproduce the wave, not restate it.
s2_run uv run --no-sync python scripts/jailbreak_run.py generate \
  --config configs/stage2_jailbreak.yaml \
  --attack "$ATK" --defense "$DEF" --shard "$SHARD" --nshards "$NSHARDS" \
  --target-greedy
exit $?
