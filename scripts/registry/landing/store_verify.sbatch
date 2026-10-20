#!/bin/bash
# store_verify.sbatch -- verify ONE immutable store generation against its manifest, or build the manifest of an in-place
# original (sisteer.registry.stores; `registry stores mirror` / `registry stores check`; spec S12).
#
#   sbatch --parsable [--time=HH:MM:SS] [--job-name=store-verify-<slug> --dependency=singleton] store_verify.sbatch \
#          verify        <generation dir> <MANIFEST-<sha8>.sha256> <VERIFIED-<sha8>.json to write> <workers> [--attestation MANIFEST-<sha8>.<lsha8>.json]
#   sbatch ... store_verify.sbatch \
#          manifest-only <original dir> <out prefix> <workers> [--member GLOB]... [--exclude E]... [--grading-host-root SPELLING]
#   sbatch ... store_verify.sbatch \
#          push <slug> <from dir> <stores dir> <workers> [--note TEXT] [--session NAME] [--member GLOB]... [--exclude E]...
#
# `push` is the PRODUCER'S publication inside its own cluster job (`--on-cluster`): the local copy
# of <from dir> into $SHARED/registry/stores/gen/<slug>/.incoming-<sha8>, the manifest, the full verify IN THIS PROCESS (no second
# job), the directory seal (chmod 0555, SEALED-<sha8>.json), the publish guard, PUBLISHED-<sha8>.json (change_utc = the producer's
# latest file mtime) and `current`. No credential, no ssh, no yaml: stdlib only. A producer that already holds an allocation calls the
# same entry point under its own srun: `python3 -S -m sisteer.registry.stores push <slug> --from DIR --stores-dir $SHARED/registry/stores`.
#
# CPU only (no --gres): a `high` submission costs no GPU budget (QoS caps are gres/gpu only). Submitted BY THE GRADING HOST over ssh
# and watched in the foreground by watch-job.sh; the grading host reads the report it writes and publishes (or not) from it.
# Every parameter is POSITIONAL -- never an export flag on the submission (it crashes slurmd on this cluster). Stdlib python only
# (`python3 -S`) from the cluster's repo mirror (the pinned tree below, else CLUSTER_REPO_MIRROR). No credential of any
# kind is needed or present: the job reads and writes the shared root only.
#
# `verify` hashes EVERY manifest entry of the generation (bytes, never inode identity) and walks the tree for unlisted files;
# a mismatch is a RESULT (the report says ok: false) and the job still exits 0 once the report is written, so the grading host side
# reads the report instead of a failed job. `--time` is set per submission by the grading host: twice the estimate at the measured
# hash rate of that component (default 100 MB/s until a report has measured it). Before hashing, `verify` SEALS every regular
# file of the generation (chmod a-w; directories stay writable so a copy can resume) and, beside the report, writes
# DEST-STAT-<sha8>.json -- the sealed tree's own stat snapshot that `python3 -S -m sisteer.registry.stores admit` (the
# destination change guard the cluster-read task runs before and after a read) checks the tree against. The DIRECTORIES are sealed
# by the PUBLISH step on the grading host side (chmod 0555 deepest first + SEALED-<sha8>.json), and `current` switches only onto a
# generation with MANIFEST, VERIFIED, DEST-STAT and SEALED beside it (the publish guard). Names the destination refuses (':', VAST)
# are stored escaped (':' -> '%3A', '%' -> '%25'); the manifest spells the true names and records the map as `name_escapes`. A
# symlink whose in-tree target carries such a name is stored with its target spelled the same way (the referent is renamed, so the
# true spelling would dangle; `verify` checks the escaped spelling, the manifest keeps the true target as `symlinks` / `target_escapes`).
# The `push` mode runs under the slug's publish lock (gen/<slug>/.publish.lock, taken before the snapshot, released after `current`
# switched; --lock-wait-s bounds the wait) and the switch itself re-reads `current` (a publisher whose reading went stale never
# switches it back): two overlapping jobs of one producer publish in snapshot order. The grading-host-side publish step takes the SAME
# lock for its readlink + comparison + rename (one ssh, `flock` on fd 9), so no publisher's switch interleaves with another's; inside
# that step, `current` must carry its PUBLISHED record (else current_unrecorded: a publisher mid-flight or crashed, adopted only by
# --adopt-unrecorded), the producer rule is applied to that record, and the new PUBLISHED record is written BEFORE `current` switches.
#
# THE CODE IS PINNED. Before a tick copies anything, the grading host pushes its running revision to the repo mirror, checks this
# script exists at that revision, extracts `git archive <sha>` of the verifier (this script, src/sisteer/__init__.py,
# src/sisteer/registry) into $SHARED/registry/stores/trees/<sha>/ and submits THAT copy -- so the job runs exactly the code
# the grading host runs. The tree is found from the submitted script's path (scontrol's Command=; `$0` is a spool copy); only when
# that path yields no tree (a hand submission) does the repo mirror's working tree serve, and the log says so.
#
# The log path is the submitter's: a directive cannot read the site's shared root, so the store mirror passes
# --output=<shared root>/exp/logs/%x_%j.out, and a submission without one gets Slurm's default slurm-<job>.out where it was submitted.
#SBATCH --job-name=store-verify
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --open-mode=append
set -uo pipefail

MODE="${1:-}"
if [ -z "$MODE" ]; then
  echo "usage: store_verify.sbatch verify <generation> <manifest> <out> <workers> [--attestation FILE] | manifest-only <dir> <out prefix> <workers> [--member G]... [--exclude E]... [--grading-host-root S] | push <slug> <from dir> <stores dir> <workers> [--note T] [--session N] [--member G]... [--exclude E]..." >&2
  exit 2
fi
shift

# the cluster's shared root and repo mirror, when the job's environment carries them (the site config's CLUSTER_SHARED_ROOT and
# CLUSTER_REPO_MIRROR); the pinned tree below is found from the submitted path and needs neither
SHARED="${CLUSTER_SHARED_ROOT:-}"
# caches off node-local disk: the shared copy (never a home-directory copy, cluster /home is node-local)
if [ -n "$SHARED" ] && [ -r "$SHARED/scripts/hf_env.sh" ]; then
  # shellcheck disable=SC1091
  source "$SHARED/scripts/hf_env.sh"
fi
export NCCL_NVLS_ENABLE=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
REPO="${CLUSTER_REPO_MIRROR:-${SHARED:+$SHARED/repos/constitutional-adapters}}"
# the pinned tree: <tree>/scripts/registry/landing/<this script>, so the tree is three levels up from the submitted path
SUBMITTED="$(scontrol show job "${SLURM_JOB_ID:-}" 2>/dev/null | sed -n 's/^ *Command=\([^ ]*\).*/\1/p' | head -1)"
if [ -n "$SUBMITTED" ] && [ -d "$(dirname "$SUBMITTED")/../../../src/sisteer/registry" ]; then
  REPO="$(cd "$(dirname "$SUBMITTED")/../../.." && pwd -P)"
  PINNED="$(cat "$REPO/REVISION" 2>/dev/null || echo unknown)"
else
  PINNED="none (running from the repo mirror's working tree: submitted=${SUBMITTED:-?})"
fi

# DISARM SIGTERM FIRST: `kill -TERM -$$` signals this shell too and bash re-enters the handler otherwise (measured 2026-09-11).
cleanup() { trap '' SIGTERM; kill -TERM -$$ 2>/dev/null; wait; }
trap cleanup SIGTERM SIGINT SIGQUIT

log() { echo "[$(date -u +%FT%TZ)] store_verify[${SLURM_JOB_ID:-nojob}] $*"; }
log "mode=$MODE host=$(hostname) cpus=${SLURM_CPUS_PER_TASK:-?} repo=$REPO pinned_revision=$PINNED args: $*"
if [ -z "$REPO" ] || [ ! -d "$REPO/src/sisteer/registry" ]; then
  log "no sisteer.registry under ${REPO:-the repo mirror}/src -- submit the pinned tree's copy of this script, or set CLUSTER_REPO_MIRROR to the cluster's clone"
  exit 3
fi

rc=1
case "$MODE" in
  verify)
    GEN="${1:?generation dir}"; MAN="${2:?manifest}"; OUT="${3:?verified report path}"; WORKERS="${4:-${SLURM_CPUS_PER_TASK:-16}}"
    # what follows the four positionals rides through to the entry point: `--attestation MANIFEST-<sha8>.<lsha8>.json` is a
    # RE-ATTESTATION of a published generation's directory listings (`registry stores attest`), its report and snapshot keyed the same way
    if [ "$#" -ge 4 ]; then shift 4; else shift "$#"; fi
    srun --ntasks=1 env PYTHONPATH="$REPO/src" python3 -S -m sisteer.registry.stores verify "$GEN" "$MAN" \
         --out "$OUT" --workers "$WORKERS" --job-id "${SLURM_JOB_ID:-}" "$@" &
    wait $!; rc=$?
    # exit 1 = the report was written and says ok: false (a result the grading host reads); anything else is a job failure
    if [ "$rc" -eq 1 ] && [ -f "$OUT" ]; then
      log "verify report written with a mismatch (ok: false); rc 1 is a result, not a failure"
      rc=0
    fi
    ;;
  manifest-only)
    DIR="${1:?original dir}"; PREFIX="${2:?out prefix}"; WORKERS="${3:-${SLURM_CPUS_PER_TASK:-16}}"
    shift 3
    srun --ntasks=1 env PYTHONPATH="$REPO/src" python3 -S -m sisteer.registry.stores manifest "$DIR" \
         --out-prefix "$PREFIX" --workers "$WORKERS" --job-id "${SLURM_JOB_ID:-}" "$@" &
    wait $!; rc=$?
    ;;
  push)
    SLUG="${1:?slug}"; FROM="${2:?from dir}"; STORES="${3:?stores dir}"; WORKERS="${4:-${SLURM_CPUS_PER_TASK:-16}}"
    shift 4
    # the producer's publication in this job: copy, manifest, verify in-process, seal, guard, PUBLISHED, `current` -- stdlib only
    srun --ntasks=1 env PYTHONPATH="$REPO/src" python3 -S -m sisteer.registry.stores push "$SLUG" --from "$FROM" \
         --stores-dir "$STORES" --workers "$WORKERS" --json "$@" &
    wait $!; rc=$?
    ;;
  *)
    log "unknown mode $MODE (verify | manifest-only | push)"
    rc=2
    ;;
esac
log "RESULT mode=$MODE rc=$rc elapsed=${SECONDS}s"
exit "$rc"
