#!/bin/bash
#SBATCH --job-name=tt-poolgen          # SESSION-UNIQUE tt-poolgen-* prefix: override
#                                      # per submission with -J tt-poolgen-<tag>; STOP the
#                                      # array with the STOP sentinel + cancel by NAME, never
#                                      # a bare scancel (the escalation trap is armed).
#SBATCH --partition=general
#SBATCH --qos=high                     # the standing default; on another cluster pass its own QoS (its site block's CLUSTER_QOS).
#SBATCH --gres=gpu:1                   # target-only, cell=none (8B target ~16GB); no attacker.
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                     # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-08:00:00              # right-sized ~2x: the long pole is pap (6 variants x
#                                      # ~209 goals x 512-tok greedy) ~4.5h; a TIMEOUT resumes
#                                      # per-goal from progress.jsonl, so this can only over-run
#                                      # into a resubmit, never lose work.
# stdout/stderr can carry attack prompts / response text, so logs go to the
# jailbreak PRIVATE dir (NOT the world-readable exp/logs). Slurm opens the log
# BEFORE the script runs and does NOT create parent dirs, so PRE-CREATE once
# before submitting:  install -d -m 700 $SHARED/exp/jailbreak/_logs
# (Cluster B's /workspace is MooseFS and ignores chmod; the dir already exists.)
# CLUSTER-B-NATIVE header (its QoS above): /workspace, so the escalation
# trap's revert-to-header --error on a resubmit stays a VALID path (stage2_common
# only carries --output forward, not --error). On cluster A override --qos=high
# and both log paths to that cluster's shared storage at submit.
#SBATCH --output=/workspace/%u/exp/jailbreak/_logs/%x_%A_%a.out
#SBATCH --error=/workspace/%u/exp/jailbreak/_logs/%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# Two-SI matched-pool GENERATION: the `none`-condition
# attack transcripts for the static/replayable families. ONE ARRAY TASK = one
# FAMILY (the task's "array over families, not a submit loop"), each run over the
# full eval goal set under the undefended `none` cell — the harmful-refusal
# "neg" side and the attack prompts the matched pool needs. bon is NOT here (it
# already lives in the labelled store from the single-SI lane);
# amplegcg_plus is NOT here either — its transformers<4.56 generator sub-env
# is unprovisioned, so it is skipped-not-attempted (execution decision E13),
# yielding a truthful SIX-family pool.
#
# Grading is a SEPARATE grading host step (the live Low-Prio StrongREJECT key lives in
# ~/.config; cluster /home is node-local) — this job ONLY generates. The per-SI
# `pos` generation is a SECOND GPU wave that depends on the graded `none`, so it
# is not in this array (see the spec decisions for the full pipeline).
#
# Submit (cluster B, with its site block's QoS), pinned to the immutable checkout at S2_ROOT:
#   sbatch --dependency=afterany:<screen job ids> -J tt-poolgen-<tag> --array=0-3 \
#     --comment=sha:$(git -C ${SISTEER_PROJECT_ROOT}-poolgen/repo rev-parse HEAD) \
#     --qos=<the cluster's CLUSTER_QOS> ${SISTEER_PROJECT_ROOT}-poolgen/repo/scripts/sbatch/two_si_pool_gen.sbatch
# The dependency keeps our concurrent GPUs <= 8 (the screen holds 8): this array
# starts only once those finish, and never cancels anything to make room.

set -uo pipefail
# The shared root is SISTEER_SHARED_ROOT, which the front door exports from the cluster's site block; a job
# without it refuses by name (there is no built-in root).
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"

# Dedicated immutable checkout so this job never repoints the shared repo or
# evals-repo checkouts the other running jobs use. stage2_common's s2_init
# verifies its HEAD against EXPECTED_SHA and FAILS LOUD on a stale clone.
export S2_ROOT="${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}-poolgen"
# JB4 transcript-job containment: fully disable W&B (raw attack/response text;
# error paths print prompts, which W&B console capture would upload on sync).
export WANDB_DISABLED=true
export WANDB_CONSOLE=off
# Registry resolution reads defenses_dir; `none` needs no compiled object, but
# name the same commit-independent dir the rest of the lane uses (body-level
# export, never sbatch --export — hard rule 1).
export TT_TWO_SI_DEFENSES_DIR=${TT_TWO_SI_DEFENSES_DIR:-$SHARED/exp/jailbreak/two_si_defenses/cells}

source "$S2_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"
# s2_init sets WANDB_MODE=offline; re-assert the hard JB4 disable on top of it.
export WANDB_DISABLED=true
export WANDB_CONSOLE=off

# One family per array index (bon already in the store; amplegcg_plus
# skipped-not-attempted — E13). A stale --array upper bound fails HERE, before
# any GPU work, rather than silently wrapping.
FAMILIES=(pasttense codeattack renellm grok420)
IDX=${SLURM_ARRAY_TASK_ID:-0}
FAM="${FAMILIES[$IDX]:-}"
if [ -z "$FAM" ]; then
  s2_log "FATAL: no family for array idx $IDX (families: ${FAMILIES[*]}) -- stale --array bound."
  exit 1
fi
s2_log "pool-gen none-condition: family=$FAM cell=none nshards=1"

# --private-root is REQUIRED on cluster B: jailbreak_run._private_root only
# auto-resolves cluster A's shared root; on cluster B it would fall back to
# node-local /home, silently stranding transcripts. Point it at the shared
# 0700 jailbreak root explicitly.
s2_run uv run --no-sync python scripts/two_si_run.py generate \
  --config configs/two_si.yaml \
  --attack "$FAM" --cell none --shard 0 --nshards 1 \
  --private-root "$SHARED/exp/jailbreak"
exit $?
