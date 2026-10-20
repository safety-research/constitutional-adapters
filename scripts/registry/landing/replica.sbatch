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
#   land TREE HEAD VENV BRANCH [REPLICA_ROOT] [OUT_DIR] [EXPECT_SKIPPED] [HASH_MODE] [REPLICA_TREE] [ACCEPT_C1_CLASS]   (the `land/3` contract)
#                                                               the landing's live-DB step (land.sh step 3 via live_db_slurm.sh): copy
#                                                               CURRENT node-local, every allowlisted file present with the head's sha,
#                                                               ingest the increment, check, `pytest -m live_db` from TREE with
#                                                               REGISTRY_LIVE_DB=<the copy>, publish ({check.json,mirror.json,ingest.json,
#                                                               junit.xml,pytest.log,summary.json} under OUT_DIR, each result stamped with
#                                                               the job's identity). OUT_DIR is THIS submission's own: live_db_slurm.sh
#                                                               mints one per submission; the default by hand is
#                                                               $SHARED/gates/<tip>/live_db/job-<job id>, never a shared directory.
#                                                               REPLICA_TREE (recovery, recover.sh / LIVE_DB_REPLICA_TREE): the tree
#                                                               whose code runs the REPLICA side (PYTHONPATH and built_by_tree); the
#                                                               tests still run from TREE. Default: TREE itself. In a recovery the
#                                                               script submitted is the CHAIN's own copy of this file, shipped by
#                                                               live_db_slurm.sh beside the head -- never the pinned tree's: a tree
#                                                               older than this contract (the bootstrap generation's) ignores the
#                                                               9th positional and runs TREE's code on the replica side.
#                                                               ACCEPT_C1_CLASS (the 10th positional, land/3): the operator's
#                                                               declaration -- free text naming the refold-compare certification -- under
#                                                               which `replica land` accepts a check that differs from the head on the
#                                                               certified C1 same-stamp class ALONE (lifecycle_state and the CASE columns
#                                                               that read it, every mover in the declared-superseded class) and publishes;
#                                                               passed through as `--accept-c1-class`. Empty or absent: only an equal check
#                                                               passes. The allowance is the cutover landing's (the live database still
#                                                               folds the pre-C1 order) and is retired once the live database adopts the
#                                                               C1 order (`restamp`); no other column is ever accepted. To pass it, the 9th
#                                                               positional must be present (live_db_slurm.sh passes TREE itself when no
#                                                               recovery pin is set); live_db_slurm.sh ships the text `printf %q`-escaped,
#                                                               so it arrives verbatim through ssh.
#   contract                                                    prints the land mode's positional contract (`land/3 TREE HEAD VENV
#                                                               BRANCH REPLICA_ROOT OUT_DIR EXPECT_SKIPPED HASH_MODE REPLICA_TREE
#                                                               ACCEPT_C1_CLASS`) and exits 0 BEFORE any work: live_db_slurm.sh and
#                                                               recover.sh probe the dispatcher they are about to ship with it; a script
#                                                               without the mode (`unknown mode`, exit 2) or with an older contract
#                                                               (`land/2`, no ACCEPT_C1_CLASS) is refused.
#   migrate [--site-aliases FILE] [--expect-site-aliases-sha256 H] TREE VENV [REPLICA_ROOT]
#                                                               `registry migrate` on a copy of CURRENT, published (a schema change);
#                                                               a shape no migration expresses exits 8 (fall back to `build`).
#                                                               The migrations read the replica root's own site_aliases.toml (the
#                                                               table mapping legacy site labels to the registry's); --site-aliases,
#                                                               when given, must name that file, and the table's digest must be
#                                                               --expect-site-aliases-sha256 when given (exit 13 otherwise). The
#                                                               digest is recorded in the generation's meta.json.
#   migrate --db PATH --out OUT_DIR --expect-input-sha256 H --expect-tree SHA --migrate-tree MTREE --expect-migrate-tree SHA --expect-schema N
#           --site-aliases FILE --expect-site-aliases-sha256 H [--head HEAD --expect-head-sha256 H] TREE VENV
#                                                               the --db form: a KEPT database (never CURRENT; no replica root
#                                                               is read, nothing under it written) copied node-local and hashed as read
#                                                               (the hash must be the DECLARED one), migrated by MTREE's own `registry
#                                                               migrate --db` -- MTREE is the tree that carries the migration files the
#                                                               copy needs, e.g. the landed main while this branch has not merged it,
#                                                               bound by commit (--expect-migrate-tree); TREE is the tree whose code
#                                                               runs, bound by commit (--expect-tree) and by the module's own location --
#                                                               its shape re-read by MTREE's `migrate --plan` (pending none, shape
#                                                               matching) and the copy required to stand at the DECLARED schema N (else
#                                                               exit 8, no output), checked against HEAD when one is given (HEAD bound
#                                                               by its declared head_sha256), and written to OUT_DIR/registry.sqlite
#                                                               beside migrate.json (versions before/after, both trees, input and output
#                                                               sha256). Every declaration is required:
#                                                               a record from this mode certifies, so nothing about it is read off the
#                                                               inputs. OUT_DIR may alias nothing (not the input's directory, not the
#                                                               work directory, not a generation). Exit 0; 7 when HEAD is given and the
#                                                               check is UNEQUAL (the output stands); 13 a declaration refused.
#                                                               The site-alias table is declared too: FILE is validated, its digest
#                                                               must be H, and both options go to MTREE's `registry migrate --db` and
#                                                               `migrate --plan --db` (an MTREE whose registry migrate knows no
#                                                               --site-aliases is refused before the copy); migrate.json records the
#                                                               file, its digest and its pairs.
#   update --db PATH --head HEAD --expect-head-sha256 H --out OUT_DIR --expect-input-sha256 H --expect-tree SHA [--expect-site-aliases-sha256 H]
#          TREE VENV [REPLICA_ROOT]
#                                                               the increment over a KEPT database: the input copied
#                                                               node-local and hashed as read (never written; the hash must be the
#                                                               DECLARED one), HEAD bound by its declared head_sha256 and required to
#                                                               strictly supersede the copy's delivered ledger (else exit 7 ahead / 13
#                                                               not newer, before the mirror hash), every allowlisted file of HEAD
#                                                               present on the mirror with its sha, the increment folded by TREE's code
#                                                               (TREE bound by commit with
#                                                               --expect-tree and by the module's own location), `replica check` against
#                                                               HEAD with the full projection written beside the verdict, the result
#                                                               written to OUT_DIR/registry.sqlite beside update.json (input and output
#                                                               sha256, head identity, the bound tree, mirror / ingest / check summaries).
#                                                               The replica root is READ (its records/rulings links) and only the
#                                                               harvester's refusal log under state/ may be written; no generation, no
#                                                               CURRENT; OUT_DIR may alias nothing. Exit 0 equal; 7 UNEQUAL (the output
#                                                               stands); 9 mirror behind; 7 replica ahead of the head (no output); 13 a
#                                                               declaration refused. Submit under the one writer name (-J
#                                                               registry-replica --dependency=singleton). The fold reads the replica
#                                                               root's site_aliases.toml (legacy site labels in old records fold to the
#                                                               registry's labels through it); update.json and check.json record its
#                                                               digest, which must be --expect-site-aliases-sha256 when given.
#   refold-compare [OPTIONS] OLD_TREE NEW_TREE VENV OUT_DIR [REPLICA_ROOT]
#                                                               the C1 measurement: the module copies the
#                                                               database node-local (hashed as read; the input is never written), folds
#                                                               every run key under both trees, projects the copy under the OLD order and
#                                                               the NEW order and reads the verdict. Without options the database is
#                                                               CURRENT's generation. OPTIONS, parsed by this script before the
#                                                               positionals (still positional to sbatch; no export flag anywhere):
#                                                                 --db PATH              a KEPT database to measure (an UNEQUAL
#                                                                                        build's registry.sqlite.unequal); CURRENT is
#                                                                                        never read, no replica root is needed, and
#                                                                                        nothing under the root is written but OUT_DIR
#                                                                 --head HEAD            the head the database was built against: its
#                                                                                        identity is recorded, its projection sha must
#                                                                                        be the expected hash (required with --db)
#                                                                 --expect-head-hash H   the full 64-hex frame_projection/1 sha256 the
#                                                                                        OLD-order fold must equal (required with --db;
#                                                                                        the job exits 1 unless the verdict is PASS)
#                                                                 --expect-new-hash H    the sha the NEW-order fold must equal (optional)
#                                                                 --expect-keys FILE     a JSON list of run_ids the mover set must equal
#                                                                                        exactly (optional)
#                                                                 --expect-input-sha256 H  the --db file's sha256, computed independently;
#                                                                                        the copy's streaming hash must equal it
#                                                                 --expect-as-built old|new  the order the database was built under; the
#                                                                                        copy as built must equal that fold
#                                                                 --expect-old-tree SHA / --expect-new-tree SHA  the commits the trees must
#                                                                                        be checked out at, clean
#                                                               A certification (--expect-head-hash) REQUIRES --head, --expect-input-sha256,
#                                                               --expect-as-built, --expect-old-tree, --expect-new-tree, --expect-new-hash
#                                                               and --expect-keys: the module refuses (exit 13) one that leaves its
#                                                               database, build, trees, NEW hash or mover set undeclared.
# TREE: an exported checkout on shared storage (tree_export.sh); VENV: the lock-exact gate venv (under $SHARED/envs);
# HEAD: a head file written by `registry replica head --out` and shipped under <REPLICA_ROOT>/builds/; REPLICA_ROOT default
# $SHARED/registry/replica (laid out once by `replica init --log-root $SHARED/registry/log`).
set -uo pipefail

MODE=${1:?MODE (build|land|migrate|update|refold-compare|contract)}; shift

# the land mode's positional contract -- answered before anything else runs (no work directory, no trap, nothing on disk), so a
# grading host probe of this file is a pure read. live_db_slurm.sh refuses to ship a dispatcher whose answer is not `land/3 … ACCEPT_C1_CLASS`.
LAND_CONTRACT="land/3 TREE HEAD VENV BRANCH REPLICA_ROOT OUT_DIR EXPECT_SKIPPED HASH_MODE REPLICA_TREE ACCEPT_C1_CLASS"
if [ "$MODE" = contract ]; then echo "$LAND_CONTRACT"; exit 0; fi

# the shared root of whichever cluster this runs on: the environment's SISTEER_SHARED_ROOT or CLUSTER_SHARED_ROOT, else the root
# the gate venv sits under (<shared>/envs/<venv>, the layout gate_venv.sbatch writes). The venv is the argument holding
# .gate-ready.json, the file need_venv checks; every mode but `contract` names one.
SHARED=${SISTEER_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-}}
if [ -z "$SHARED" ]; then
  for _arg in "$@"; do
    if [ -s "$_arg/.gate-ready.json" ]; then SHARED=$(dirname "$(dirname "$_arg")"); break; fi
  done
fi
# caches off node-local disk: the shared copy (never a home-directory copy, cluster /home is node-local)
if [ -r "$SHARED/scripts/hf_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$SHARED/scripts/hf_env.sh"
fi
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
    OUT=${6:-$SHARED/gates/$SHA/live_db/job-${SLURM_JOB_ID:-local-$$}} EXPECT_SKIPPED=${7:-0} HASH_MODE=${8:-all} RTREE=${9:-$TREE} ACCEPT_C1=${10:-}
    need_tree "$TREE"; need_tree "$RTREE"; need_venv "$VENV"; need_head "$HEAD"; need_root "$ROOT"
    RSHA=$(tree_sha "$RTREE")
    # the cutover allowance: the declaration rides to the module as --accept-c1-class, verbatim; absent, only an equal check passes.
    # The module is RTREE's: a tree that predates the allowance knows no such option and would die on it mid-job -- refuse before any work
    if [ -n "$ACCEPT_C1" ] && ! grep -q -- "--accept-c1-class" "$RTREE/src/sisteer/registry/replica.py" 2>/dev/null; then
      log "the replica tree $RTREE predates the C1 allowance (its sisteer/registry/replica.py knows no --accept-c1-class): the declaration cannot be honored; nothing run"; exit 2
    fi
    mkdir -p "$OUT" || { log "cannot create $OUT"; exit 2; }
    ACCEPT_ARGS=(); [ -z "$ACCEPT_C1" ] || ACCEPT_ARGS=(--accept-c1-class "$ACCEPT_C1")
    log "land: tree=$SHA replica_tree=$RSHA branch=$BRANCH head=$HEAD root=$ROOT out=$OUT expect_skipped=$EXPECT_SKIPPED hash=$HASH_MODE accept_c1_class=${ACCEPT_C1:-none}"
    # the replica side's code (PYTHONPATH, built_by_tree) is RTREE's; the tests run from TREE (--tree)
    srun --ntasks=1 env PYTHONPATH="$RTREE/src" "$VENV/bin/python" -m sisteer.registry.replica land \
      --replica-root "$ROOT" --head "$HEAD" --tree "$TREE" --work "$WORK" --out "$OUT" --pytest-python "$VENV/bin/python" --branch "$BRANCH" \
      --expect-skipped "$EXPECT_SKIPPED" --workers 4 --hash "$HASH_MODE" --threads "${SLURM_CPUS_PER_TASK:-16}" --tree-sha "$RSHA" --job-id "${SLURM_JOB_ID:-}" \
      ${ACCEPT_ARGS[@]+"${ACCEPT_ARGS[@]}"} &
    wait $!; rc=$?
    ;;
  migrate)
    # the options come first (positional to sbatch; no export flag): --db PATH --out OUT_DIR --expect-input-sha256 H --expect-tree SHA
    # --migrate-tree MTREE --expect-migrate-tree SHA --expect-schema N [--head HEAD]; then TREE VENV [REPLICA_ROOT]. Without --db:
    # `registry migrate` on CURRENT.
    DBIN= OUT= HEAD= MTREE= EXPECT_IN= EXPECT_MT= EXPECT_TREE= EXPECT_SCHEMA= EXPECT_HEAD= ALIASES= EXPECT_ALIASES=
    while [ $# -gt 0 ]; do
      case "$1" in
        --db) DBIN=${2:?--db PATH}; shift 2 ;;
        --out) OUT=${2:?--out OUT_DIR}; shift 2 ;;
        --site-aliases) ALIASES=${2:?--site-aliases FILE}; shift 2 ;;
        --expect-site-aliases-sha256) EXPECT_ALIASES=${2:?--expect-site-aliases-sha256 HASH}; shift 2 ;;
        --head) HEAD=${2:?--head HEAD}; shift 2 ;;
        --expect-head-sha256) EXPECT_HEAD=${2:?--expect-head-sha256 HASH}; shift 2 ;;
        --migrate-tree) MTREE=${2:?--migrate-tree TREE}; shift 2 ;;
        --expect-input-sha256) EXPECT_IN=${2:?--expect-input-sha256 HASH}; shift 2 ;;
        --expect-migrate-tree) EXPECT_MT=${2:?--expect-migrate-tree SHA}; shift 2 ;;
        --expect-tree) EXPECT_TREE=${2:?--expect-tree SHA}; shift 2 ;;
        --expect-schema) EXPECT_SCHEMA=${2:?--expect-schema N}; shift 2 ;;
        --*) log "unknown migrate option $1"; exit 2 ;;
        *) break ;;
      esac
    done
    TREE=${1:?TREE} VENV=${2:?VENV} ROOT=${3:-$SHARED/registry/replica}
    need_tree "$TREE"; need_venv "$VENV"
    SHA=$(tree_sha "$TREE")
    # the module is TREE's: a tree that predates the site-alias options would die on them mid-job -- refuse before any work
    if [ -n "$ALIASES$EXPECT_ALIASES" ] && ! grep -q -- "--site-aliases" "$TREE/src/sisteer/registry/replica.py" 2>/dev/null; then
      log "the tree $TREE predates the site-alias options (its sisteer/registry/replica.py knows no --site-aliases); nothing run"; exit 2
    fi
    ALIAS_ARGS=()
    [ -z "$ALIASES" ] || ALIAS_ARGS+=(--site-aliases "$ALIASES")
    [ -z "$EXPECT_ALIASES" ] || ALIAS_ARGS+=(--expect-site-aliases-sha256 "$EXPECT_ALIASES")
    if [ -n "$DBIN" ]; then
      # a kept database: the root is not consulted at all -- no CURRENT read, nothing under it written; every declaration required
      { [ -n "$OUT" ] && [ -n "$EXPECT_IN" ] && [ -n "$EXPECT_TREE" ] && [ -n "$MTREE" ] && [ -n "$EXPECT_MT" ] && [ -n "$EXPECT_SCHEMA" ]; } || {
        log "--db needs --out OUT_DIR --expect-input-sha256 H --expect-tree SHA --migrate-tree MTREE --expect-migrate-tree SHA --expect-schema N (a certifying record declares everything)"; exit 2; }
      { [ -n "$ALIASES" ] && [ -n "$EXPECT_ALIASES" ]; } || {
        log "--db needs --site-aliases FILE --expect-site-aliases-sha256 H (the alias table the copy migrates under is declared; an empty table is a file with no pairs)"; exit 2; }
      [ -f "$ALIASES" ] || { log "no site-alias file at $ALIASES"; exit 2; }
      need_tree "$MTREE"
      [ -s "$DBIN" ] || { log "no database at $DBIN"; exit 2; }
      EXTRA=()
      if [ -n "$HEAD" ]; then
        need_head "$HEAD"
        [ -n "$EXPECT_HEAD" ] || { log "--head needs --expect-head-sha256 H (the head a check is read against is declared)"; exit 2; }
        EXTRA+=(--head "$HEAD" --expect-head-sha256 "$EXPECT_HEAD")
      else
        [ -z "$EXPECT_HEAD" ] || { log "--expect-head-sha256 needs --head HEAD"; exit 2; }
      fi
      mkdir -p "$OUT" || { log "cannot create $OUT"; exit 2; }
      log "migrate: tree=$SHA expect_tree=$EXPECT_TREE db=$DBIN migrate_tree=$(tree_sha "$MTREE") expect_migrate_tree=$EXPECT_MT expect_schema=$EXPECT_SCHEMA out=$OUT head=${HEAD:-none} expect_head_sha256=${EXPECT_HEAD:-none} expect_input_sha256=$EXPECT_IN site_aliases=$ALIASES expect_site_aliases_sha256=$EXPECT_ALIASES"
      srun --ntasks=1 env PYTHONPATH="$TREE/src" "$VENV/bin/python" -m sisteer.registry.replica migrate \
        --db "$DBIN" --work "$WORK" --out "$OUT" --tree "$TREE" --expect-tree "$EXPECT_TREE" --migrate-tree "$MTREE" --expect-migrate-tree "$EXPECT_MT" \
        --expect-schema "$EXPECT_SCHEMA" --expect-input-sha256 "$EXPECT_IN" --python "$VENV/bin/python" --job-id "${SLURM_JOB_ID:-}" ${EXTRA[@]+"${EXTRA[@]}"} \
        "${ALIAS_ARGS[@]}" &
    else
      # the site-alias options are the CURRENT form's too (the module reads only the root's own table); every other option needs --db
      [ -z "$OUT$HEAD$MTREE$EXPECT_IN$EXPECT_MT$EXPECT_TREE$EXPECT_SCHEMA$EXPECT_HEAD" ] || { log "the migrate options (--out, --head, --migrate-tree, --expect-* other than --expect-site-aliases-sha256) need --db PATH"; exit 2; }
      need_root "$ROOT"
      log "migrate: tree=$SHA root=$ROOT site_aliases=${ALIASES:-root-table} expect_site_aliases_sha256=${EXPECT_ALIASES:-none}"
      srun --ntasks=1 env PYTHONPATH="$TREE/src" "$VENV/bin/python" -m sisteer.registry.replica migrate \
        --replica-root "$ROOT" --work "$WORK" --tree-sha "$SHA" --job-id "${SLURM_JOB_ID:-}" ${ALIAS_ARGS[@]+"${ALIAS_ARGS[@]}"} &
    fi
    wait $!; rc=$?
    ;;
  update)
    # the options come first (positional to sbatch; no export flag): --db PATH --head HEAD --expect-head-sha256 H --out OUT_DIR
    # --expect-input-sha256 H --expect-tree SHA; then TREE VENV [REPLICA_ROOT]. The root is READ (records / rulings links; the mirror
    # check and the fold read the log through it).
    DBIN= HEAD= OUT= EXPECT_IN= EXPECT_TREE= EXPECT_HEAD= EXPECT_ALIASES=
    while [ $# -gt 0 ]; do
      case "$1" in
        --db) DBIN=${2:?--db PATH}; shift 2 ;;
        --head) HEAD=${2:?--head HEAD}; shift 2 ;;
        --expect-site-aliases-sha256) EXPECT_ALIASES=${2:?--expect-site-aliases-sha256 HASH}; shift 2 ;;
        --expect-head-sha256) EXPECT_HEAD=${2:?--expect-head-sha256 HASH}; shift 2 ;;
        --out) OUT=${2:?--out OUT_DIR}; shift 2 ;;
        --expect-input-sha256) EXPECT_IN=${2:?--expect-input-sha256 HASH}; shift 2 ;;
        --expect-tree) EXPECT_TREE=${2:?--expect-tree SHA}; shift 2 ;;
        --*) log "unknown update option $1"; exit 2 ;;
        *) break ;;
      esac
    done
    TREE=${1:?TREE} VENV=${2:?VENV} ROOT=${3:-$SHARED/registry/replica}
    need_tree "$TREE"; need_venv "$VENV"; need_root "$ROOT"
    { [ -n "$DBIN" ] && [ -n "$HEAD" ] && [ -n "$EXPECT_HEAD" ] && [ -n "$OUT" ] && [ -n "$EXPECT_IN" ] && [ -n "$EXPECT_TREE" ]; } || {
      log "update needs --db PATH --head HEAD --expect-head-sha256 H --out OUT_DIR --expect-input-sha256 H --expect-tree SHA (a certifying record declares everything)"; exit 2; }
    [ -s "$DBIN" ] || { log "no database at $DBIN"; exit 2; }
    need_head "$HEAD"
    if [ -n "$EXPECT_ALIASES" ] && ! grep -q -- "--expect-site-aliases-sha256" "$TREE/src/sisteer/registry/replica.py" 2>/dev/null; then
      log "the tree $TREE predates the site-alias options (its sisteer/registry/replica.py knows no --expect-site-aliases-sha256); nothing run"; exit 2
    fi
    ALIAS_ARGS=(); [ -z "$EXPECT_ALIASES" ] || ALIAS_ARGS=(--expect-site-aliases-sha256 "$EXPECT_ALIASES")
    mkdir -p "$OUT" || { log "cannot create $OUT"; exit 2; }
    SHA=$(tree_sha "$TREE")
    log "update: tree=$SHA expect_tree=$EXPECT_TREE db=$DBIN head=$HEAD expect_head_sha256=$EXPECT_HEAD root=$ROOT out=$OUT expect_input_sha256=$EXPECT_IN workers=${SLURM_CPUS_PER_TASK:-16} expect_site_aliases_sha256=${EXPECT_ALIASES:-none}"
    srun --ntasks=1 env PYTHONPATH="$TREE/src" "$VENV/bin/python" -m sisteer.registry.replica update \
      --replica-root "$ROOT" --db "$DBIN" --head "$HEAD" --expect-head-sha256 "$EXPECT_HEAD" --work "$WORK" --out "$OUT" --tree "$TREE" --expect-tree "$EXPECT_TREE" --expect-input-sha256 "$EXPECT_IN" \
      --workers "${SLURM_CPUS_PER_TASK:-16}" --threads "${SLURM_CPUS_PER_TASK:-16}" --hash "${REPLICA_BUILD_HASH:-all}" --job-id "${SLURM_JOB_ID:-}" ${ALIAS_ARGS[@]+"${ALIAS_ARGS[@]}"} &
    wait $!; rc=$?
    ;;
  refold-compare)
    # the options come first; everything after them is the positional contract OLD_TREE NEW_TREE VENV OUT_DIR [REPLICA_ROOT]
    DBIN= HEAD= EXPECT= EXPECT_NEW= EXPECT_KEYS= EXPECT_IN= EXPECT_BUILT= EXPECT_OLD= EXPECT_NEWT=
    while [ $# -gt 0 ]; do
      case "$1" in
        --db) DBIN=${2:?--db PATH}; shift 2 ;;
        --head) HEAD=${2:?--head HEAD}; shift 2 ;;
        --expect-head-hash) EXPECT=${2:?--expect-head-hash HASH}; shift 2 ;;
        --expect-new-hash) EXPECT_NEW=${2:?--expect-new-hash HASH}; shift 2 ;;
        --expect-keys) EXPECT_KEYS=${2:?--expect-keys FILE}; shift 2 ;;
        --expect-input-sha256) EXPECT_IN=${2:?--expect-input-sha256 HASH}; shift 2 ;;
        --expect-as-built) EXPECT_BUILT=${2:?--expect-as-built old|new}; shift 2 ;;
        --expect-old-tree) EXPECT_OLD=${2:?--expect-old-tree SHA}; shift 2 ;;
        --expect-new-tree) EXPECT_NEWT=${2:?--expect-new-tree SHA}; shift 2 ;;
        --*) log "unknown refold-compare option $1"; exit 2 ;;
        *) break ;;
      esac
    done
    OLD=${1:?OLD_TREE} NEW=${2:?NEW_TREE} VENV=${3:?VENV} OUT=${4:?OUT_DIR} ROOT=${5:-$SHARED/registry/replica}
    need_tree "$OLD"; need_tree "$NEW"; need_venv "$VENV"
    EXTRA=()
    if [ -n "$DBIN" ]; then
      # a kept database: the root is not consulted at all -- no CURRENT read, no head pointer touched
      [ -s "$DBIN" ] || { log "no database at $DBIN"; exit 2; }
      { [ -n "$HEAD" ] && [ -n "$EXPECT" ]; } || { log "--db needs --head HEAD and --expect-head-hash HASH (the identity a certification is read against)"; exit 2; }
      SOURCE="db=$DBIN"
    else
      need_root "$ROOT"
      CUR=$(cat "$ROOT/CURRENT" 2>/dev/null) || { log "no CURRENT generation under $ROOT"; exit 11; }
      DBIN=$ROOT/gen/$CUR/registry.sqlite
      SOURCE="generation=$CUR"
    fi
    if [ -n "$HEAD" ]; then need_head "$HEAD"; EXTRA+=(--head "$HEAD"); fi
    [ -z "$EXPECT" ] || EXTRA+=(--expect-head-hash "$EXPECT")
    [ -z "$EXPECT_NEW" ] || EXTRA+=(--expect-new-hash "$EXPECT_NEW")
    if [ -n "$EXPECT_KEYS" ]; then [ -s "$EXPECT_KEYS" ] || { log "no key list at $EXPECT_KEYS"; exit 2; }; EXTRA+=(--expect-keys "$EXPECT_KEYS"); fi
    [ -z "$EXPECT_IN" ] || EXTRA+=(--expect-input-sha256 "$EXPECT_IN")
    [ -z "$EXPECT_BUILT" ] || EXTRA+=(--expect-as-built "$EXPECT_BUILT")
    [ -z "$EXPECT_OLD" ] || EXTRA+=(--expect-old-tree "$EXPECT_OLD")
    [ -z "$EXPECT_NEWT" ] || EXTRA+=(--expect-new-tree "$EXPECT_NEWT")
    mkdir -p "$OUT" || { log "cannot create $OUT"; exit 2; }
    log "refold-compare: old=$(tree_sha "$OLD") new=$(tree_sha "$NEW") $SOURCE head=${HEAD:-none} expect_head_hash=${EXPECT:-none} expect_new_hash=${EXPECT_NEW:-none} expect_keys=${EXPECT_KEYS:-none} expect_input_sha256=${EXPECT_IN:-none} expect_as_built=${EXPECT_BUILT:-none} expect_old_tree=${EXPECT_OLD:-none} expect_new_tree=${EXPECT_NEWT:-none} out=$OUT"
    # the module copies the database into $WORK (hashing it as it reads) and measures the copy; $DBIN is never written
    srun --ntasks=1 env PYTHONPATH="$NEW/src" "$VENV/bin/python" -m sisteer.registry.replica refold-compare \
      --db "$DBIN" --work "$WORK" --old-tree "$OLD" --new-tree "$NEW" --out "$OUT" --python "$VENV/bin/python" ${EXTRA[@]+"${EXTRA[@]}"} &
    wait $!; rc=$?
    ;;
  *)
    log "unknown mode $MODE"; exit 2 ;;
esac
log "RESULT mode=$MODE rc=$rc elapsed=${SECONDS}s"
exit "$rc"
