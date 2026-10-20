#!/bin/bash
#SBATCH --job-name=twosi-budget-emit
#SBATCH --partition=general
#SBATCH --qos=high                     # the standing default; on another cluster pass its own QoS at submit
#SBATCH --cpus-per-task=8             # CPU-only: emit hashes configs/sources, folds the
#                                     # trials JSONL, and draws the heatmaps — no GPU
#SBATCH --mem=32G
#SBATCH --time=0-00:45:00             # minutes of work (uv sync + imports dominate), ~2x pad
#SBATCH --output=/workspace/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
#SBATCH --signal=B:TERM@120

# Submit (dependent on the three per-SI shard jobs; the sha pin is mandatory; on cluster B pass its QoS):
#   sbatch -J twosi-budget-emit --dependency=afterok:<j1>:<j2>:<j3> \
#          --comment=sha:$(git -C ${SISTEER_REPO_MIRROR}-twosi rev-parse HEAD) \
#          --kill-on-invalid-dep=yes scripts/sbatch/twosi_budget_emit.sbatch
#
# The ASSEMBLING emit of the sharded two-SI training-budget pass (see
# twosi_budget.sbatch): concatenates the three per-SI shards' trial rows and
# runs `measure_train_budget.py emit` under the FULL configs/two_si.yaml —
# the config whose si_ids-bearing measurement_inputs_sha the consumption
# gate (train_budget.measure_si_budget) checks. Rows compose because the
# per-row identity (arm_scoped_inputs_sha) excludes si_ids, and (si, arm,
# bs) keys are disjoint across per-SI shards. emit itself enforces
# acceptance: any (si, arm) without a completed bs=1 trial, or any stale
# row, fails LOUDLY — so this job succeeding IS the 33-pair validity check.
#
# Submit with --dependency=afterok:<shard1>:<shard2>:<shard3>
# --kill-on-invalid-dep=yes — it must only run after all three shards
# succeeded, and must be cancelled (not left pending forever) if one fails.
# NOTE: if a shard self-escalates (timeout/preemption), its SUCCESSOR is a
# new job id and this dependency dies with the predecessor — resubmit this
# job by hand after the successor finishes (it re-validates everything).
#
# The measured table lands at $STOP_DIR/two_si_budget_measured.json (shared
# storage, outside the checkout); pull it back to the grading host and commit it
# as configs/two_si_budget_measured.json in a separate, reviewed commit.

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
S2_ROOT="${SISTEER_REPO_MIRROR:?set SISTEER_REPO_MIRROR, the cluster clone of this repository (the front door exports it from the site config)}-twosi"
S2_REPO_DIR="$S2_ROOT"
source "$S2_REPO_DIR/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# Fold the three shards' rows into one trials file. Keys (si, arm, bs) are
# disjoint across shards; _read_trials still refuses duplicates, so a
# double-cat can never pass silently.
COMBINED="$STOP_DIR/trials_combined.jsonl"
: > "$COMBINED"
for si in safety_refusal si_constitution si_irrelevant; do
  f="$SHARED/exp/training/twosi-budget-$si/trials.jsonl"
  if [ ! -s "$f" ]; then
    s2_log "FATAL: shard trials missing or empty: $f — did twosi-budget-$si finish?"
    exit 1
  fi
  cat "$f" >> "$COMBINED"
done
s2_log "combined $(wc -l < "$COMBINED") trial rows from 3 shards"

MEASURED="$STOP_DIR/two_si_budget_measured.json"

s2_run uv run --no-sync python scripts/measure_train_budget.py emit \
  --config configs/two_si.yaml \
  --trials "$COMBINED" \
  --emit "$MEASURED" \
  --figs "$FIGS_DIR"
rc=$?
[ "$rc" -eq 0 ] && s2_log "measured table -> $MEASURED (copy back to the grading host and commit as configs/two_si_budget_measured.json)"
exit $rc
