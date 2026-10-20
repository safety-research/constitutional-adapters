#!/usr/bin/env bash
# reproduce/lib/mirror_figs.sh — push a job's figures (+ its RESULT line) from a cluster to the grading host over a
# write-only rsync key. Installed by 00_setup.sh as $SHARED/scripts/mirror-figs.sh (the path common.sh f22_mirror calls).
#
# Usage (from an sbatch EXIT trap):  $SHARED/scripts/mirror-figs.sh <figs-dir> [remote-subdir]
# Site facts, all from shared storage or the job's env prefix — nothing personal in this file:
#   FIGS_SHARED_ROOT | SISTEER_SHARED_ROOT   the cluster's shared root (else CLUSTER_SHARED_ROOT, else the root the helper is
#                                            installed under)
#   FIGS_PUSH_KEY                            default $SHARED/.ssh/figs-push   (a key whose authorized_keys line forces
#                                            `rrsync -wo -no-del` into the grading host's figure directory + `restrict`)
#   FIGS_DEST_HOST | $SHARED/.ssh/figs_dest  `user@host` of the grading host (its address is site state, written once)
#   FIGS_URL_BASE  | $SHARED/.ssh/figs_url   optional: the URL the pushed directory is served at, printed for the log
# ALWAYS exits 0: a failed figure push must never fail a finished job.
set -uo pipefail
SRC="${1:-}"
[ -n "$SRC" ] || { echo "mirror-figs: usage: mirror-figs.sh <figs-dir> [remote-subdir]" >&2; exit 0; }
NAME="${2:-${SLURM_JOB_NAME:-job}}"
[ "$NAME" = job ] && [ -n "${SLURM_JOB_ID:-}" ] && NAME="job-$SLURM_JOB_ID"
NAME="$(printf '%s' "$NAME" | tr -c 'a-zA-Z0-9._-' '-')"
# the shared root: FIGS_SHARED_ROOT, else SISTEER_SHARED_ROOT (the job's env prefix), else CLUSTER_SHARED_ROOT, else the root this helper
# is installed under (00_setup.sh installs it as <shared root>/scripts/mirror-figs.sh; a job's EXIT trap calls that copy)
SHARED="${FIGS_SHARED_ROOT:-${SISTEER_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-}}}"
if [ -z "$SHARED" ]; then SHARED="$(dirname "$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)")"; fi
[ -d "$SRC" ] || { echo "mirror-figs: no figures dir at $SRC — nothing to push" >&2; exit 0; }
[ -n "$SHARED" ] || { echo "mirror-figs: no shared root (set SISTEER_SHARED_ROOT / FIGS_SHARED_ROOT)" >&2; exit 0; }
KEY="${FIGS_PUSH_KEY:-$SHARED/.ssh/figs-push}"; KNOWN="$SHARED/.ssh/known_hosts"
DEST_HOST="${FIGS_DEST_HOST:-}"; [ -z "$DEST_HOST" ] && [ -r "$SHARED/.ssh/figs_dest" ] && DEST_HOST="$(tr -d '[:space:]' < "$SHARED/.ssh/figs_dest")"
URL_BASE="${FIGS_URL_BASE:-}"; [ -z "$URL_BASE" ] && [ -r "$SHARED/.ssh/figs_url" ] && URL_BASE="$(tr -d '[:space:]' < "$SHARED/.ssh/figs_url")"
[ -f "$KEY" ] || { echo "mirror-figs: no push key at $KEY (docs/secrets.md: the figure-push key)" >&2; exit 0; }
[ -n "$DEST_HOST" ] || { echo "mirror-figs: no destination (FIGS_DEST_HOST or $SHARED/.ssh/figs_dest) — figures stay at $SRC" >&2; exit 0; }
# a filesystem that ignores chmod (MooseFS reports 0666) makes ssh refuse the key: stage a node-local 0600 copy for the push
KEY_USE="$KEY"; PERMS="$(stat -c %a "$KEY" 2>/dev/null || echo 600)"
if [ "$PERMS" != 600 ] && [ "$PERMS" != 400 ]; then
  if KEY_TMP="$(mktemp "${TMPDIR:-/tmp}/figs-push.XXXXXX" 2>/dev/null)"; then trap 'rm -f "$KEY_TMP"' EXIT; cat "$KEY" > "$KEY_TMP" && KEY_USE="$KEY_TMP"; fi
fi
if rsync -az --prune-empty-dirs --include='*/' --include='*.png' --include='*.jpg' --include='*.pdf' --include='*.svg' \
     --include='*.html' --include='*.csv' --include='*.json' --include='RESULT' --exclude='*' \
     -e "ssh -i '$KEY_USE' -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile='$KNOWN' -o ConnectTimeout=20 -o BatchMode=yes" \
     "$SRC/" "$DEST_HOST:$NAME/"; then
  echo "mirror-figs: pushed $SRC -> $DEST_HOST:$NAME/${URL_BASE:+  ($URL_BASE/$NAME/)}"
else
  echo "mirror-figs: push failed (rc=$?) — figures are still at $SRC" >&2
fi
exit 0
