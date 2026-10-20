#!/bin/bash
#SBATCH --job-name=registry-replica
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --output=/workspace-vast/%u/exp/logs/%x_%j.out
#SBATCH --open-mode=append
#SBATCH --no-requeue
# replica.sbatch MODE ARGS... -- the ONE writer of the cluster replica of the registry database
# (sisteer.registry.replica). Every invocation is submitted BY THE GRADING HOST over ssh under ONE job name with the singleton
# dependency, so two writers never run at once:
#   sbatch --parsable -J registry-replica --dependency=singleton [--time=…] replica.sbatch <mode> <args…>
# Every parameter is POSITIONAL (never an export flag on the submission: it crashes slurmd on this cluster). CPU only, no --gres: a
# `high` submission costs no GPU budget. The job works NODE-LOCAL (VAST is nfs4.1 with local_lock=none: no SQLite there) under
# /home/$USER/replica-work/<job id> -- cluster /home is node-local disk with a 750 GB quota, never /tmp -- and publishes a finished
# generation back to the shared replica root by compare-and-swap on CURRENT (sisteer.registry.replica.publish). The work directory
# is removed by the trap on every exit.
#
#   build TREE VENV HEAD [REPLICA_ROOT]                        the bootstrap: rebuild EXACTLY the head's allowlist from the
#                                                               record-log mirror into a fresh node-local database with 16 workers,
#                                                               `replica check` against the SAME head, publish only an EQUAL build
#                                                               (an unequal one STOPS -- the record stays under builds/, exit 7).
#                                                               Submit with --time=2-00:00:00 (10-27 h expected).
#   land TREE HEAD VENV BRANCH [REPLICA_ROOT] [OUT_DIR] [EXPECT_SKIPPED] [HASH_MODE]
#                                                               the landing's live-DB step (land.sh step 3 via live_db_slurm.sh): copy
#                                                               CURRENT node-local, every allowlisted file present with the head's sha,
#                                                               ingest the increment, check, `pytest -m live_db` from TREE with
#                                                               REGISTRY_LIVE_DB=<the copy>, publish ({check.json,mirror.json,ingest.json,
#                                                               junit.xml,pytest.log,summary.json} under OUT_DIR, each result stamped with
#                                                               the job's identity). OUT_DIR is THIS submission's own: live_db_slurm.sh
#                                                               mints one per submission; the default by hand is
#                                                               $SHARED/gates/<tip>/live_db/job-<job id>, never a shared directory.
#   migrate TREE VENV [REPLICA_ROOT]                            `registry migrate` on a copy of CURRENT, published (a schema change);
#                                                               a shape no migration expresses exits 8 (fall back to `build`).
#   refold-compare OLD_TREE NEW_TREE VENV OUT_DIR [REPLICA_ROOT] the C1 measurement on a copy of CURRENT.
# TREE: an exported checkout on shared storage (tree_export.sh); VENV: the lock-exact gate venv (under $SHARED/envs);
# HEAD: a head file written by `registry replica head --out` and shipped under <REPLICA_ROOT>/builds/; REPLICA_ROOT default
# $SHARED/registry/replica (laid out once by `replica init --log-root $SHARED/registry/log`).
set -uo pipefail

MODE=${1:?MODE (build|land|migrate|refold-compare)}; shift

# the shared root of whichever cluster this runs on (written against $SHARED, not a literal root)
if [ -d "/workspace-vast/$USER" ]; then SHARED="/workspace-vast/$USER"; else SHARED="/workspace/$USER"; fi
# caches off node-local disk: the shared copy (never a home-directory copy, cluster /home is node-local)
if [ -r "$SHARED/scripts/hf_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$SHARED/scripts/hf_env.sh"
fi
export NCCL_SOCKET_IFNAME=vxlan0
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16} OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export SISTEER_REGISTRY_DISABLED=1            # nothing here is a producer: no registry hook fires from the replica's own code
export PYTHONDONTWRITEBYTECODE=1

WORK=${REPLICA_WORK_ROOT:-/home/$USER/replica-work}/${SLURM_JOB_ID:-local-$$}
# DISARM SIGTERM FIRST: `kill -TERM -$$` signals this shell too and bash re-enters the handler otherwise (measured 2026-09-11).
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; rm -rf "$WORK"; }
trap cleanup SIGTERM SIGINT SIGQUIT
trap 'rm -rf "$WORK"' EXIT

log() { echo "[$(date -u +%FT%TZ)] replica[${SLURM_JOB_ID:-?}] $*"; }
log "mode=$MODE host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} work=$WORK args=$*"
mkdir -p "$WORK" || { log "cannot create the node-local work directory $WORK"; exit 2; }
df -h "$WORK" | tail -n 1 | sed 's/^/[replica] node-local: /'

need_tree() { [ -d "$1/src/sisteer/registry" ] || { log "no exported tree at $1 (tree_export.sh first)"; exit 2; }; }
need_venv() { [ -s "$1/.gate-ready.json" ] && [ -x "$1/bin/python" ] || { log "venv not ready: $1 (gate_venv.sbatch first)"; exit 2; }; }
need_head() { [ -s "$1" ] || { log "no head file at $1 (registry replica head --out, shipped by the grading host)"; exit 2; }; }
need_root() { [ -s "$1/config.toml" ] && [ -L "$1/records" ] || { log "no replica root at $1 (registry replica init --replica-root $1 --log-root $SHARED/registry/log first)"; exit 2; }; }
tree_sha() { git -C "$1" rev-parse HEAD 2>/dev/null || basename "$1"; }

case "$MODE" in
  build)
    TREE=${1:?TREE} VENV=${2:?VENV} HEAD=${3:?HEAD} ROOT=${4:-$SHARED/registry/replica}
    need_tree "$TREE"; need_venv "$VENV"; need_head "$HEAD"; need_root "$ROOT"
    SHA=$(tree_sha "$TREE")
    OUT=$ROOT/builds/build-$(date -u +%Y%m%dT%H%M%SZ)-${SLURM_JOB_ID:-local}
    mkdir -p "$OUT" || { log "cannot create $OUT"; exit 2; }
    log "build: tree=$SHA head=$HEAD root=$ROOT out=$OUT workers=${SLURM_CPUS_PER_TASK:-16}"
    srun --ntasks=1 env PYTHONPATH="$TREE/src" "$VENV/bin/python" -m sisteer.registry.replica build \
      --replica-root "$ROOT" --head "$HEAD" --work "$WORK" --out "$OUT" --workers "${SLURM_CPUS_PER_TASK:-16}" --threads "${SLURM_CPUS_PER_TASK:-16}" \
      --hash "${REPLICA_BUILD_HASH:-all}" --tree-sha "$SHA" --job-id "${SLURM_JOB_ID:-}" &
    wait $!; rc=$?
    if [ "$rc" -eq 7 ] && [ -s "$WORK/registry.sqlite" ]; then
      # an UNEQUAL bootstrap STOPS: the rows are in $OUT/check.json, and the database itself is kept beside it for the open
      # question (never published, never CURRENT; the trap would otherwise remove it with the node-local work)
      log "UNEQUAL build: keeping the work database as $OUT/registry.sqlite.unequal for inspection"
      cp "$WORK/registry.sqlite" "$OUT/registry.sqlite.unequal" || log "copy of the unequal database failed"
    fi
    ;;
  land)
    TREE=${1:?TREE} HEAD=${2:?HEAD} VENV=${3:?VENV} BRANCH=${4:?BRANCH} ROOT=${5:-$SHARED/registry/replica}
    SHA=$(tree_sha "$TREE")
    OUT=${6:-$SHARED/gates/$SHA/live_db/job-${SLURM_JOB_ID:-local-$$}} EXPECT_SKIPPED=${7:-0} HASH_MODE=${8:-all}
    need_tree "$TREE"; need_venv "$VENV"; need_head "$HEAD"; need_root "$ROOT"
    mkdir -p "$OUT" || { log "cannot create $OUT"; exit 2; }
    log "land: tree=$SHA branch=$BRANCH head=$HEAD root=$ROOT out=$OUT expect_skipped=$EXPECT_SKIPPED hash=$HASH_MODE"
    srun --ntasks=1 env PYTHONPATH="$TREE/src" "$VENV/bin/python" -m sisteer.registry.replica land \
      --replica-root "$ROOT" --head "$HEAD" --tree "$TREE" --work "$WORK" --out "$OUT" --pytest-python "$VENV/bin/python" --branch "$BRANCH" \
      --expect-skipped "$EXPECT_SKIPPED" --workers 4 --hash "$HASH_MODE" --threads "${SLURM_CPUS_PER_TASK:-16}" --tree-sha "$SHA" --job-id "${SLURM_JOB_ID:-}" &
    wait $!; rc=$?
    ;;
  migrate)
    TREE=${1:?TREE} VENV=${2:?VENV} ROOT=${3:-$SHARED/registry/replica}
    need_tree "$TREE"; need_venv "$VENV"; need_root "$ROOT"
    SHA=$(tree_sha "$TREE")
    log "migrate: tree=$SHA root=$ROOT"
    srun --ntasks=1 env PYTHONPATH="$TREE/src" "$VENV/bin/python" -m sisteer.registry.replica migrate \
      --replica-root "$ROOT" --work "$WORK" --tree-sha "$SHA" --job-id "${SLURM_JOB_ID:-}" &
    wait $!; rc=$?
    ;;
  refold-compare)
    OLD=${1:?OLD_TREE} NEW=${2:?NEW_TREE} VENV=${3:?VENV} OUT=${4:?OUT_DIR} ROOT=${5:-$SHARED/registry/replica}
    need_tree "$OLD"; need_tree "$NEW"; need_venv "$VENV"; need_root "$ROOT"
    CUR=$(cat "$ROOT/CURRENT" 2>/dev/null) || { log "no CURRENT generation under $ROOT"; exit 11; }
    mkdir -p "$OUT" || { log "cannot create $OUT"; exit 2; }
    log "refold-compare: old=$(tree_sha "$OLD") new=$(tree_sha "$NEW") generation=$CUR out=$OUT"
    cp "$ROOT/gen/$CUR/registry.sqlite" "$WORK/registry.sqlite" || { log "copy of $CUR failed"; exit 2; }
    srun --ntasks=1 env PYTHONPATH="$NEW/src" "$VENV/bin/python" -m sisteer.registry.replica refold-compare \
      --db "$WORK/registry.sqlite" --old-tree "$OLD" --new-tree "$NEW" --out "$OUT" --python "$VENV/bin/python" &
    wait $!; rc=$?
    ;;
  *)
    log "unknown mode $MODE"; exit 2 ;;
esac
log "RESULT mode=$MODE rc=$rc elapsed=${SECONDS}s"
exit "$rc"
