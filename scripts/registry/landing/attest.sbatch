#!/bin/bash
#SBATCH --job-name=attest
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
# attest.sbatch WINDOW_ROOT TREE -- the per-operation ATTESTATION of every pinned store generation of a cluster-reads plan
# (sisteer.registry.stores.attest_plan). Submitted BY THE GRADING HOST over ssh as
#   sbatch --parsable -J attest-<window> attest.sbatch <window root> <tree>
# (`registry cluster-reads submit`, which then submits the read array behind it: --dependency=afterok:<this job>
# --kill-on-invalid-dep=yes), every parameter POSITIONAL (never an export flag on the submission: it crashes slurmd on this
# cluster). One task, CPU only, no --gres: a `high` submission costs no GPU budget. Stdlib python only (`python3 -S`) from the
# exported tree TREE (tree_export.sh; PYTHONPATH=TREE/src), the same code the shards run; no venv, no credential: the job reads
# the shared root and writes the window root (attestation.json) and, beside a generation whose full content walk passed,
# gen/<slug>/ATTESTED-<sha8>.json.
#
# For every pinned generation of the plan, in parallel threads: (a) the authority facts -- `current` names the pin, the tree exists,
# MANIFEST / VERIFIED / DEST-STAT / SEALED / PUBLISHED stand beside it and agree with the manifest; (b) a FULL DIRECTORY LISTING of the
# generation (recursive scandir: names, types, symlink targets; no stat, no hash) against the manifest's entries and attested links;
# (c) the directory seal (every directory 0555; the count the SEALED record's); (d) the content -- the full admit_generation walk the
# first time (or after a week), else a seeded 64-file stat-and-hash sample. A REFUSAL IS A RESULT: the record says so per generation
# and the job still exits 0 once it is written, so the array runs and every shard reading that pin refuses by name in its summary.
# Only a crash (no record) exits non-zero -- and then the dependency cancels the array, the window finds no shard summaries, and
# every cluster adapter falls back by name.
set -uo pipefail

WINDOW_ROOT=${1:?WINDOW_ROOT} TREE=${2:?TREE}

# the shared root of whichever cluster this runs on (written against $SHARED, not a literal root)
# (the environment's SISTEER_SHARED_ROOT or CLUSTER_SHARED_ROOT, else the root TREE sits under: <shared>/gates/trees/<sha>, the layout
# cluster_reads.TREES_REL fixes)
SHARED=${SISTEER_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-$(dirname "$(dirname "$(dirname "$TREE")")")}}
# caches off node-local disk: the shared copy (never a ~ path, cluster /home is node-local)
if [ -r "$SHARED/scripts/hf_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$SHARED/scripts/hf_env.sh"
fi
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# DISARM SIGTERM FIRST: `kill -TERM -$$` signals this shell too and bash re-enters the handler otherwise (measured 2026-09-11).
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

log() { echo "[$(date -u +%FT%TZ)] attest[${SLURM_JOB_ID:-?}] $*"; }
log "host=$(hostname) window_root=$WINDOW_ROOT tree=$TREE cpus=${SLURM_CPUS_PER_TASK:-?}"
# fail-fast guards: every external input, before any work
[ -s "$WINDOW_ROOT/plan.json" ]     || { log "no plan at $WINDOW_ROOT/plan.json (registry cluster-reads stage first)"; exit 2; }
[ -d "$TREE/src/sisteer/registry" ] || { log "no exported tree at $TREE (tree_export.sh first)"; exit 2; }
[ -w "$WINDOW_ROOT" ]               || { log "window root $WINDOW_ROOT is not writable; the record cannot be written"; exit 2; }

srun --ntasks=1 env PYTHONPATH="$TREE/src" python3 -S -m sisteer.registry.stores attest \
  --plan "$WINDOW_ROOT/plan.json" --out "$WINDOW_ROOT/attestation.json" --threads "${SLURM_CPUS_PER_TASK:-4}" --job-id "${SLURM_JOB_ID:-}" &
wait $!; rc=$?
log "RESULT rc=$rc elapsed=${SECONDS}s record=$WINDOW_ROOT/attestation.json"
exit "$rc"
