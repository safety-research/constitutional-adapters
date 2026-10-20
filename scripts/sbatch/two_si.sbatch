#!/bin/bash
#SBATCH --job-name=tt-twosi-gen       # SESSION-UNIQUE tt-twosi-* prefix: override per
#                                     # submission with -J tt-twosi-<tag>; cancel by JOB ID
#                                     # ONLY — a shared-user name-scoped scancel once caused
#                                     # a mass kill.
#SBATCH --partition=general
#SBATCH --qos=high                    # the lane default; draws on the 16-GPU high budget
#SBATCH --gres=gpu:1                  # target-only work (bon/gcg: 8B target ~16GB). The
#                                     # gated tiers co-reside the 27B attacker + 8B target
#                                     # on one H200 too (see jailbreak_attacks.sbatch).
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-10:00:00             # right-sized ~2x, not 5x: a BoN cell shard (~50 goals
#                                     # x 256 variants x 512-tok greedy) runs ~3-4h on the
#                                     # safety cells; the CONSTITUTION prompted baseline
#                                     # pays a 5,908-token prefill on EVERY request (the
#                                     # 46x prompt-length asymmetry), so budget ~2x that.
#                                     # Re-time from the first cell before widening.
# stdout/stderr can carry attack prompts / response text, so logs go to a
# mode-0700 PRIVATE dir (NOT the world-readable exp/logs). Slurm opens the log
# BEFORE the script runs and does NOT create parent dirs, so PRE-CREATE once
# before submitting:  install -d -m 700 $SHARED/exp/jailbreak/_logs
#SBATCH --output=/workspace-vast/%u/exp/jailbreak/_logs/%x_%A_%a.out
#SBATCH --error=/workspace-vast/%u/exp/jailbreak/_logs/%x_%A_%a.err
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120
#
# Two-SI matrix transcript generation. ONE ARRAY TASK = one
# (DefenseCell x goal-shard): index = cell_idx * NSHARDS + shard, and the
# decode runs through the SAME Python the tests pin
# (sisteer.eval.defense_cell.cell_for_array_index via `two_si_run.py cells`)
# — never a shell copy of the cell list, so the array index can never desync
# from the registry order, and the two SI versions of one arm land on
# DISTINCT cell indices instead of colliding. A stale --array upper
# bound FAILS the task (IndexError in the decode), never silently wraps.
#
# Submit with the sha pin (stage2_common verifies the clone against
# EXPECTED_SHA — the immutable-checkout discipline) and a session-unique name:
#   N=$(uv run python scripts/two_si_run.py cells --config configs/two_si.yaml | tail -1)  # prints the bound
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) \
#          -J tt-twosi-bon-<tag> --array=0-<NCELLS*NSHARDS-1> \
#          scripts/sbatch/two_si.sbatch
#   # smoke ONE cell first (cell 0 = none, shard 0):
#   sbatch --comment=sha:$(git -C $SISTEER_PROJECT_ROOT/repo rev-parse HEAD) \
#          -J tt-twosi-smoke-<tag> --array=0-0 scripts/sbatch/two_si.sbatch
# ATTACK defaults to bon; set it in a WRAPPER SCRIPT BODY or edit below —
# NEVER via sbatch --export (hard rule 1: --export crashes slurmd here).
#
# Grading is a SEPARATE grading host post-step (two_si_run.py aggregate) — never
# on-node (JB5/JB6). Compiled cells need their per-cell artifacts at
# <defenses_dir>/<cell.id>/object.* first; `none`/`prompted__*` need none.
#
# NOTE: TRANSCRIPT-generating job — raw variants/responses land ONLY in the
# mode-0700 private dir $SHARED/exp/jailbreak/two_si_v1/<cell.id>/
# (keyed on cell.id, never the arm). W&B is fully disabled (JB4).

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
: "${SISTEER_PROJECT_ROOT:?set SISTEER_PROJECT_ROOT, the cluster project directory (the front door exports it from the site config)}"
source "$SISTEER_PROJECT_ROOT/repo/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# JB4 transcript-job containment on top of stage2_common's WANDB_MODE=offline:
# fully disable W&B (this job touches raw attack/response text; errors print
# prompts to stdout, which W&B console capture would upload on sync).
export WANDB_DISABLED=true
export WANDB_CONSOLE=off

# Compiled objects live in ONE commit-independent dir under the private
# root — NEVER the checkout's out/, which every new sha would orphan. The
# readers here (two_si_run -> registry_from_cfg -> resolve_defenses_dir)
# prefer this env over the config's repo-rooted default; body-level export,
# never sbatch --export (hard rule 1). Must name the SAME dir the fit jobs
# were passed via --out (the overnight orchestrator's default below).
export TT_TWO_SI_DEFENSES_DIR=${TT_TWO_SI_DEFENSES_DIR:-$SHARED/exp/jailbreak/two_si_defenses/cells}

ATTACK=${ATTACK:-bon}       # body-level default — set in a wrapper body, never --export
NSHARDS=${NSHARDS:-4}
IDX=${SLURM_ARRAY_TASK_ID:-0}

# Decode the array index through the registry (single source of truth): a
# failed decode (stale --array bound, config drift) yields empty output and
# fails HERE, before any GPU work.
read -r CELL SHARD < <(uv run --no-sync python scripts/two_si_run.py cells \
  --config configs/two_si.yaml --task-id "$IDX" --nshards "$NSHARDS") || true
if [ -z "${CELL:-}" ] || [ -z "${SHARD:-}" ]; then
  s2_log "FATAL: array decode failed for task $IDX (nshards=$NSHARDS) -- stale --array bound or registry drift; run 'two_si_run.py cells' to see the live bounds."
  exit 1
fi
s2_log "two-si cell: cell=$CELL shard=$SHARD/$NSHARDS attack=$ATTACK"

s2_run uv run --no-sync python scripts/two_si_run.py generate \
  --config configs/two_si.yaml \
  --attack "$ATTACK" --cell "$CELL" --shard "$SHARD" --nshards "$NSHARDS"
exit $?
