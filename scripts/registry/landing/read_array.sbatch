#!/bin/bash
#SBATCH --job-name=reads
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:25:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%A_%a.out
#SBATCH --open-mode=append
# read_array.sbatch WINDOW_ROOT TREE VENV [LOCAL_VENV] -- one array task = one shard of one heavy adapter's read
# (sisteer.registry.cluster_reads.run_shard). Submitted BY THE GRADING HOST over ssh as
#   sbatch --parsable -J reads-<window> --array=0-$((N-1)) read_array.sbatch <window root> <tree> <venv> [0|1]
# (`registry cluster-reads submit`), N = the plan's task count; every parameter is POSITIONAL (never an export flag on the
# submission: it crashes slurmd on this cluster). CPU only, no --gres: a `high` submission costs no GPU budget. Watched in the
# foreground by watch-job.sh from the window script.
#
# The task runs, from the shared gate venv (a reader imports yaml / numpy / pandas and the stdlib, not torch; LOCAL_VENV=1 copies the
# venv node-local first, the gate's fix for NFS-bound pytest imports -- MEASURED 2026-09-28: two concurrent copies of the torch-cpu
# gate venv on one node had not finished after 7.5 min, so the copy is opt-in and the default imports from the shared root),
#   python -m sisteer.registry.cli --root <window root>/root cluster-reads shard --plan <window root>/plan.json --index $SLURM_ARRAY_TASK_ID
# from the exported tree TREE (tree_export.sh; PYTHONPATH=TREE/src): admission of every pinned generation on this node, the home tree
# (the canonical spelling of each store resolving onto its generation under this account's node-local home), the sharded pass against
# the shipped reader-state sidecar, admission again, the shard summary <window root>/<adapter>/<i>-of-<K>.json. Publishes land in
# the window root's spool root/records/adapters/<a>/<rev16>/, which the grading host pulls marker-last. No credential of any kind is
# needed or present; the task reads the shared root and writes the window root and its own home only. HOME is NOT relocated (the
# store indexes embed absolute /home/<user>/ paths); SISTEER_REGISTRY_URI_HOME is exported as this home so the identity map is empty
# by construction and the shard refuses by name if the plan's canonical home is another.
set -uo pipefail

WINDOW_ROOT=${1:?WINDOW_ROOT} TREE=${2:?TREE} VENV=${3:?VENV}
LOCAL_VENV=${4:-0}            # 1 = run from a node-local byte copy of the venv (opt-in; see the header for the measured cost)

# the shared root of whichever cluster this runs on (written against $SHARED, not a literal root)
# (the environment's SISTEER_SHARED_ROOT or CLUSTER_SHARED_ROOT, else the root VENV sits under: <shared>/envs/<venv>, as gate_venv.sbatch)
SHARED=${SISTEER_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-$(dirname "$(dirname "$VENV")")}}
# caches off node-local disk: the shared copy (never a home-directory copy, cluster /home is node-local)
if [ -r "$SHARED/scripts/hf_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$SHARED/scripts/hf_env.sh"
fi
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export SISTEER_REGISTRY_DISABLED=1            # the read emits records through the runner alone; no producer hook fires from a reader
export SISTEER_REGISTRY_URI_HOME="$HOME"      # the canonical home IS this home (module docstring); a differing plan refuses by name
export SISTEER_TREE_SHA=$(git -C "$TREE" rev-parse HEAD 2>/dev/null || echo unverified)     # the producer's code_sha fallback (cluster_producer)

WORK=/tmp/reads-${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}
# DISARM SIGTERM FIRST: `kill -TERM -$$` signals this shell too and bash re-enters the handler otherwise (measured 2026-09-11).
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; rm -rf "$WORK"; }
trap cleanup SIGTERM SIGINT SIGQUIT
trap 'rm -rf "$WORK"' EXIT

log() { echo "[$(date -u +%FT%TZ)] read_array[${SLURM_ARRAY_JOB_ID:-?}_${SLURM_ARRAY_TASK_ID:-?}] $*"; }
log "host=$(hostname) window_root=$WINDOW_ROOT tree=$TREE venv=$VENV cpus=${SLURM_CPUS_PER_TASK:-?}"
# fail-fast guards: every external input, before any work
[ -n "${SLURM_ARRAY_TASK_ID:-}" ]        || { log "SLURM_ARRAY_TASK_ID is unset (array-only script)"; exit 2; }
[ -s "$WINDOW_ROOT/plan.json" ]          || { log "no plan at $WINDOW_ROOT/plan.json (registry cluster-reads stage first)"; exit 2; }
[ -s "$WINDOW_ROOT/reader_state.sqlite" ] || { log "no reader-state sidecar at $WINDOW_ROOT/reader_state.sqlite"; exit 2; }
[ -d "$TREE/src/sisteer/registry" ]      || { log "no exported tree at $TREE (tree_export.sh first)"; exit 2; }
[ -s "$VENV/.gate-ready.json" ]          || { log "venv not ready: $VENV/.gate-ready.json missing"; exit 2; }
[ -x "$VENV/bin/python" ]                || { log "no python in $VENV"; exit 2; }
[ -L "$WINDOW_ROOT/root/rulings" ]       || { log "no rulings symlink at $WINDOW_ROOT/root/rulings (the stage step lays it)"; exit 2; }
mkdir -p "$WORK" || { log "cannot create $WORK"; exit 2; }

PY="$VENV/bin/python"
if [ "$LOCAL_VENV" = 1 ]; then
  # the venv node-local (byte copy; no -a: VAST carries ACL metadata the node's /tmp cannot take), verified by importing from it
  t0=$SECONDS
  cp -dR --preserve=mode,timestamps "$VENV" "$WORK/venv" 2>/dev/null
  if "$WORK/venv/bin/python" -c "import yaml, numpy" 2>/dev/null; then PY="$WORK/venv/bin/python"; else log "node-local venv copy does not import yaml/numpy; using the shared venv"; fi
  log "venv copy took $((SECONDS - t0))s ($PY)"
fi
t0=$SECONDS
"$PY" -c "import yaml, numpy" || { log "the venv at $PY does not import yaml/numpy"; exit 2; }
log "venv import probe $((SECONDS - t0))s ($PY)"

srun --ntasks=1 env PYTHONPATH="$TREE/src" "$PY" -m sisteer.registry.cli --root "$WINDOW_ROOT/root" \
  cluster-reads shard --plan "$WINDOW_ROOT/plan.json" --index "$SLURM_ARRAY_TASK_ID" --work "$WORK" &
wait $!; rc=$?
log "RESULT task=${SLURM_ARRAY_TASK_ID} rc=$rc elapsed=${SECONDS}s"
exit "$rc"
