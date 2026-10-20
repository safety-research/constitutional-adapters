#!/bin/bash
#SBATCH --job-name=gate-array
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --array=0-39
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
# gate_array.sbatch SHA TREE VENV RESULTS_ROOT COUNT RUN_NAME — one array task = one shard of the full pytest suite.
# Submitted by gate_slurm.sh as: sbatch -J gate-<sha7>-<user> --array=0-$((COUNT-1)) [--dependency=afterok:<venv job>] ...
# Each task runs gate_task.py (one shard of the suite) under the shared venv, on a node-local copy of the tree,
# and writes RESULTS_ROOT/SHA/shards/<i>-of-<COUNT>/{pytest.log,junit.xml,summary.json}. Node-local scratch is removed
# on exit. pytest itself is bounded to TIMEOUT_S (default 2400 s; the 45-min walltime leaves room to record a TIMEOUT).
# Re-running ONE timed-out shard: sbatch -J gate-<sha7>-r<i>-<user> --array=<i> --time=02:00:00 gate_array.sbatch ... 6600 1
set -uo pipefail
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

SHA=${1:?SHA} TREE=${2:?TREE} VENV=${3:?VENV} RESULTS_ROOT=${4:?RESULTS_ROOT} COUNT=${5:?COUNT} RUN_NAME=${6:?RUN_NAME}
# the cluster's shared root: CLUSTER_SHARED_ROOT when the job's environment carries it, else the venv's grandparent directory
# (VENV is <shared root>/envs/<name>, the gate venv's layout; gate_venv.sbatch reads it the same way)
SHARED=${CLUSTER_SHARED_ROOT:-$(dirname "$(dirname "$VENV")")}
source "$SHARED/scripts/hf_env.sh"
TIMEOUT_S=${7:-2400}          # pytest's bound; the re-run of a timed-out shard passes a larger one with a matching --time
LOCAL_VENV=${8:-0}            # 1 = run from a node-local byte copy of the venv (the re-run's fix for NFS-bound imports)
HIDE_SLURM=${9:-0}            # 1 = the suite's PATH carries no Slurm command (as on Modal; see gate_task.SLURM_CMDS)
FILES=${10:-}                 # comma-separated tree-relative test files: a targeted re-run into <sha>/reruns/<LABEL>/ (shard mode when empty)
LABEL=${11:-}                 # the re-run's label (default: files-<hash8>)
EXTRA=""
[ "$LOCAL_VENV" = 1 ] && EXTRA="$EXTRA --local-venv"
[ "$HIDE_SLURM" = 1 ] && EXTRA="$EXTRA --hide-slurm"
[ -n "$FILES" ] && EXTRA="$EXTRA --files ${FILES//,/ }"
[ -n "$LABEL" ] && EXTRA="$EXTRA --rerun-label $LABEL"
BIN=$SHARED/gates/bin
WORK=/tmp/gate-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; rm -rf "$WORK"; }
trap cleanup SIGTERM SIGINT SIGQUIT
trap 'rm -rf "$WORK"' EXIT
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}

echo "[gate] $(date -u +%FT%TZ) host=$(hostname) job=${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?} sha=$SHA shard=${SLURM_ARRAY_TASK_ID}/$COUNT"
# fail-fast guards: every external input, before any work
[ -d "$TREE/.git" ]              || { echo "[gate] no exported tree at $TREE"; exit 2; }
[ -s "$VENV/.gate-ready.json" ]  || { echo "[gate] venv not ready: $VENV/.gate-ready.json missing"; exit 2; }
[ -x "$VENV/bin/python" ]        || { echo "[gate] no python in $VENV"; exit 2; }
[ -s "$BIN/gate_task.py" ] && [ -s "$BIN/gate_lib.py" ] || { echo "[gate] gate code missing under $BIN"; exit 2; }
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "[gate] SLURM_ARRAY_TASK_ID is unset (array-only script; a targeted re-run uses --array=0)"; exit 2; }
mkdir -p "$WORK" || { echo "[gate] cannot create $WORK"; exit 2; }

srun "$VENV/bin/python" "$BIN/gate_task.py" --sha "$SHA" --tree "$TREE" --index "$SLURM_ARRAY_TASK_ID" --count "$COUNT" \
  --results-root "$RESULTS_ROOT" --venv "$VENV" --work "$WORK" --run-name "$RUN_NAME" --timeout-s "$TIMEOUT_S" $EXTRA
rc=$?
echo "[gate] $(date -u +%FT%TZ) task ${SLURM_ARRAY_TASK_ID} rc=$rc"
exit $rc
