#!/bin/bash
#SBATCH --job-name=gate-finalize
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
# gate_finalize.sbatch SHA TREE VENV RESULTS_ROOT COUNT RUN_NAME ARRAY_JOB_ID — the server-side finish of a sharded gate
# submitted with --dependency=afterany:<array id>, merges the shard records with gate_lib's own merge_shard_records
# (no suite module is loaded from the tree) into RESULTS_ROOT/SHA/summary.json (a missing shard
# record = TIMEOUT = not a result).
set -uo pipefail
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

SHA=${1:?SHA} TREE=${2:?TREE} VENV=${3:?VENV} RESULTS_ROOT=${4:?RESULTS_ROOT} COUNT=${5:?COUNT} RUN_NAME=${6:?RUN_NAME} ARRAY_JOB_ID=${7:-}
# the cluster's shared root: CLUSTER_SHARED_ROOT when the job's environment carries it, else the venv's grandparent directory
# (VENV is <shared root>/envs/<name>, the gate venv's layout; gate_venv.sbatch reads it the same way)
SHARED=${CLUSTER_SHARED_ROOT:-$(dirname "$(dirname "$VENV")")}
source "$SHARED/scripts/hf_env.sh"
BIN=$SHARED/gates/bin
echo "[gate] $(date -u +%FT%TZ) finalize host=$(hostname) sha=$SHA array=$ARRAY_JOB_ID count=$COUNT"
[ -d "$TREE/.git" ] || { echo "[gate] no exported tree at $TREE"; exit 2; }
PY="$VENV/bin/python"; [ -x "$PY" ] || PY=$(command -v python3)
srun "$PY" "$BIN/gate_lib.py" finalize --sha "$SHA" --tree "$TREE" --results-root "$RESULTS_ROOT" --count "$COUNT" \
  --run-name "$RUN_NAME" ${ARRAY_JOB_ID:+--array-job-id "$ARRAY_JOB_ID"}
rc=$?
echo "[gate] $(date -u +%FT%TZ) finalize rc=$rc"
exit $rc
