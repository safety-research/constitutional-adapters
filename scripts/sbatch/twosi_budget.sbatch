#!/bin/bash
#SBATCH --job-name=twosi-budget-safety_refusal
#                                     # ONE shard = ONE si_id, derived from the job
#                                     # name below — submit with -J twosi-budget-<si_id>
#                                     # (the header default is just the safety shard).
#                                     # Session-unique twosi-budget-* names: cancel with
#                                     # stop-run.sh <name> <cluster B's login host>, never a bare
#                                     # scancel (looks like preemption -> self-escalates)
#                                     # and never a name-scoped scancel wider than this
#                                     # prefix (shared-user account).
#SBATCH --partition=general
#SBATCH --qos=high                    # the standing default; on another cluster pass its own QoS (its site block's CLUSTER_QOS).
#SBATCH --gres=gpu:1                  # one H200 per shard; the 11 arm ladders of one SI
#                                     # run sequentially, one fresh trial subprocess per
#                                     # (si, arm, bs)
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G                    # ~128G per GPU; NEVER --mem=0 (hard rule 2)
#SBATCH --time=0-06:00:00             # 8192-token shards (si_constitution/si_irrelevant):
#                                     # ~3h real (peer 156964: ~3h/22 pairs on cluster A,
#                                     # + MooseFS model-load overhead), ~2x pad. The
#                                     # safety_refusal shard (max_len 2048) is submitted
#                                     # with --time=0-04:00:00 at the command line.
#SBATCH --output=/workspace/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append            # retries append, never clobber
#SBATCH --no-requeue                  # stage2_common self-escalates; Slurm must not double-requeue
#SBATCH --signal=B:TERM@120
#
# ONE per-SI shard of the two-SI training-budget GPU pass (capeval-controls
# lane) on CLUSTER B, re-measured under the NousResearch/
# Meta-Llama-3.1-8B-Instruct pin (two_si.yaml 2026-08-20): the lane's model
# repoint changed measurement_inputs_sha, so the committed meta-llama table
# now (correctly) refuses consumption and plan returns NEEDS_GPU until this
# pass lands.
#
# SHARDING (3 concurrent jobs, one per si_id). Verified composition:
#   * each shard gets its OWN trials JSONL ($STOP_DIR is keyed on the job
#     name) — concurrent `run`s must never share one file (O_APPEND
#     interleaving on MooseFS + _supersede_row's whole-file rewrite);
#   * each shard runs under a GENERATED config identical to configs/
#     two_si.yaml except si_ids reduced to its one SI. This keeps the
#     shard's OWN end-of-run emit acceptance scoped to its SI, so exit 0
#     means "all 11 ladders passed bs=1 acceptance" (with the full config a
#     shard would ALWAYS die at the final acceptance check, which spans all
#     si_ids). Provenance-safe because arm_scoped_inputs_sha — the per-row
#     identity the final emit gates on — deliberately EXCLUDES si_ids and
#     the other SIs' inputs (verified equal for all 33 pairs, 2026-08-20);
#     measurement_inputs_sha DOES include si_ids, which is exactly why the
#     assembling emit (twosi_budget_emit.sbatch) runs with the FULL config.
#   * --emit '' skips the shard-level table write; the committed table is
#     assembled ONLY by the dependent emit job from all three shards' rows.
#
# Model: the pinned NousResearch snapshot is fully present in the per-user
# HF cache (/workspace/$USER/hf_cache, verified 2026-08-20) — ungated, and
# read offline (s2_init exports HF_HUB_OFFLINE=1). No token on cluster B,
# by design.
#
# Preemption tolerance: every trial row lands fsync-per-row in the shard's
# trials JSONL on SHARED storage; a resubmitted shard resumes by exact
# fingerprint match and re-runs at most the one interrupted trial.
# stage2_common.sh provides the SIGTERM self-escalation, the STOP sentinel,
# the EXPECTED_SHA immutable-checkout gate, the cleanup trap (kills the srun
# process group and drains within the grace window), and the $FIGS_DIR
# mirror-figs push on every exit path.
#
# Submit (after mirror-repos.sh --cluster <cluster B's CLUSTER_MIRROR_SELECTOR> <this checkout's name> and
# recording the sha in the checkout):
#   cd ${SISTEER_REPO_MIRROR}-twosi
#   git rev-parse HEAD > EXPECTED_SHA
#   PIN=--comment=sha:$(git -C ${SISTEER_REPO_MIRROR}-twosi rev-parse HEAD)
#   sbatch -J twosi-budget-safety_refusal  --time=0-04:00:00 \
#          --comment=sha:$(git -C ${SISTEER_REPO_MIRROR}-twosi rev-parse HEAD) \
#          scripts/sbatch/twosi_budget.sbatch
#   sbatch -J twosi-budget-si_constitution --time=0-06:00:00 \
#          --comment=sha:$(git -C ${SISTEER_REPO_MIRROR}-twosi rev-parse HEAD) \
#          scripts/sbatch/twosi_budget.sbatch
#   sbatch -J twosi-budget-si_irrelevant   --time=0-06:00:00 \
#          --comment=sha:$(git -C ${SISTEER_REPO_MIRROR}-twosi rev-parse HEAD) \
#          scripts/sbatch/twosi_budget.sbatch
# then the assembler, dependent on all three:
#   sbatch -J twosi-budget-emit --dependency=afterok:<j1>:<j2>:<j3> \
#          --comment=sha:$(git -C ${SISTEER_REPO_MIRROR}-twosi rev-parse HEAD) \
#          --kill-on-invalid-dep=yes scripts/sbatch/twosi_budget_emit.sbatch

set -uo pipefail
SHARED="${SISTEER_SHARED_ROOT:?set SISTEER_SHARED_ROOT, the shared root of the cluster (the front door exports it from the site config)}"
# This job's OWN mirror clone (other jobs' checkouts are never repointed).
# The checkout is not named `repo`, so stage2_common reads EXPECTED_SHA from
# INSIDE it; basename(S2_ROOT) names the node-local venv
# /home/$USER/venvs/sisteer-twosi (the convention).
S2_ROOT="${SISTEER_REPO_MIRROR:?set SISTEER_REPO_MIRROR, the cluster clone of this repository (the front door exports it from the site config)}-twosi"
S2_REPO_DIR="$S2_ROOT"
source "$S2_REPO_DIR/scripts/sbatch/stage2_common.sh"
s2_init "$@"

# Cluster NCCL requirements (inert on 1 GPU, kept per the standing sbatch
# checklist so a widened resubmit can never silently miss them). Body-level
# exports — NEVER sbatch --export (hard rule 1).
export NCCL_NVLS_ENABLE=0

# The shard's SI comes from the job name — the same string that keys
# $STOP_DIR, $FIGS_DIR, and the log file, so one script serves all three
# shards and a self-escalated resubmit (which carries --job-name) keeps its
# identity.
SI="${SLURM_JOB_NAME#twosi-budget-}"
case "$SI" in
  safety_refusal|si_constitution|si_irrelevant) ;;
  *) s2_log "FATAL: job name '$SLURM_JOB_NAME' does not name a shard SI (expect twosi-budget-<si_id>)"; exit 1 ;;
esac

# Generate the shard config: configs/two_si.yaml with si_ids reduced to this
# shard's SI, values otherwise IDENTICAL (a yaml value round-trip; comments
# don't enter any hash — both input shas are computed over config VALUES +
# source-file contents, never config-file bytes).
SHARD_CFG="$STOP_DIR/two_si_shard_$SI.yaml"
uv run --no-sync python - "$SI" "$SHARD_CFG" <<'PY'
import sys
import yaml

si, out = sys.argv[1], sys.argv[2]
cfg = yaml.safe_load(open("configs/two_si.yaml"))
if si not in cfg["si_ids"]:
    raise SystemExit(
        f"[twosi-budget] {si!r} not in configs/two_si.yaml si_ids {cfg['si_ids']}")
cfg["si_ids"] = [si]
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[twosi-budget] shard config -> {out} (si_ids=[{si}])")
PY

TRIALS="$STOP_DIR/trials.jsonl"

s2_run uv run --no-sync python scripts/measure_train_budget.py run \
  --config "$SHARD_CFG" \
  --trials "$TRIALS" \
  --emit '' \
  --figs "$FIGS_DIR"
exit $?
