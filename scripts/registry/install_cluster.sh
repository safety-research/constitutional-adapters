#!/usr/bin/env bash
# install_cluster.sh — provision ONE cluster's registry spool from the repo mirror (run ON the cluster).
#
#   REGISTRY_SHARED_ROOT=<shared root> <repo mirror>/scripts/registry/install_cluster.sh [--dest-host user@host] [--print-key-line]
#
# Creates the layout under $SHARED/registry, where $SHARED is this cluster's shared root (REGISTRY_SHARED_ROOT, else the
# site cluster block's CLUSTER_SHARED_ROOT; neither set refuses by name):
#   registry/records/    the producer spool (RecordWriter writes <campaign>/<run_dir>/<seq>-<type>.jsonl[.gz])
#   registry/spool/      the emitter's scratch (per-run .last_push stamps live beside the records; this holds probes)
#   registry/pending/    the submit wrapper's planned-run files, <jobid>.json (the job finds its own by $SLURM_JOB_ID)
#   registry/snapshot/   where the grading host pushes point-in-time snapshots
#   registry/log/        the mirrored record log (registry mirrors sync)
#   registry/DEST_HOST   user@host of the grading host (written from --dest-host; an ephemeral external IP is
#                        re-pointed with one write)
# and installs push_records.sh + registry_env.sh + sbatch_registered.sh into $SHARED/scripts/registry/ from this
# checkout (the mirror), so every compute node sees them (cluster /home is node-local). registry_env.sh reads the cluster's
# values from registry_site.env beside it and probes nothing, so when that file is absent this writes it with the shared
# root alone; a file already there (reproduce/00_setup.sh --what helpers renders the full one from the site's cluster
# block) is left as it is. Idempotent.
#
# THE KEY IS NOT INSTALLED HERE. The `registry-push` private key must be streamed in by the operator over ssh
# stdin into $SHARED/.ssh/registry-push (never argv, never a file on the grading host); this script only PRINTS the
# grading host's authorized_keys line to add (--print-key-line, or always at the end when the public key is at
# $SHARED/.ssh/registry-push.pub), forced to `rrsync -wo -no-del -no-lock ~/registry/inbox` and, when
# REGISTRY_PUSH_FROM names the cluster's egress ranges, pinned to them with `from=`. A shared filesystem that ignores
# chmod reports 0666 on every file; push_records.sh stages a 0600 copy for the length of each push, so a 0666 key
# there is expected.
#
# Environment (site values, none with a site default):
#   REGISTRY_SHARED_ROOT        this cluster's shared root (else CLUSTER_SHARED_ROOT)
#   REGISTRY_HOST_TAG           this cluster's inbox name on the grading host, shown in the probe hint (default unknown)
#   REGISTRY_PUSH_FROM          the cluster's egress address patterns for `from=`, comma-separated with no spaces,
#                               e.g. 192.0.2.0/24,198.51.100.0/24 (empty: the key line carries no from=)
#   REGISTRY_GRADING_HOST_USER  the grading host account whose ~/registry/inbox receives the pushes (default $USER)
set -euo pipefail

DEST=""; PRINT_KEY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest-host) DEST="$2"; shift 2 ;;
    --print-key-line) PRINT_KEY=1; shift ;;
    -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
    *) echo "install_cluster: unknown option $1" >&2; exit 2 ;;
  esac
done

SHARED="${REGISTRY_SHARED_ROOT:-${CLUSTER_SHARED_ROOT:-}}"
[[ -n "$SHARED" ]] || { echo "install_cluster: set REGISTRY_SHARED_ROOT to this cluster's shared root (the site cluster block's CLUSTER_SHARED_ROOT)" >&2; exit 2; }
[[ -d "$SHARED" ]] || { echo "install_cluster: the shared root $SHARED is not a directory on this host (REGISTRY_SHARED_ROOT / CLUSTER_SHARED_ROOT)" >&2; exit 2; }
HOST_TAG="${REGISTRY_HOST_TAG:-unknown}"
PUSH_FROM="${REGISTRY_PUSH_FROM:-}"
case "$PUSH_FROM" in *[[:space:]\"]*) echo "install_cluster: REGISTRY_PUSH_FROM must be comma-separated address patterns with no spaces or quotes, got '$PUSH_FROM'" >&2; exit 2 ;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REG="$SHARED/registry"
mkdir -p "$REG/records" "$REG/spool" "$REG/pending" "$REG/snapshot" "$REG/log" "$SHARED/scripts/registry" "$SHARED/.ssh"
for f in push_records.sh registry_env.sh sbatch_registered.sh; do
  install -m 0755 "$HERE/$f" "$SHARED/scripts/registry/$f"
done
SITE_ENV="$SHARED/scripts/registry/registry_site.env"
[[ -f "$SITE_ENV" ]] || printf 'REGISTRY_SITE_SHARED_ROOT=%q\n' "$SHARED" > "$SITE_ENV"
if [[ -n "$DEST" ]]; then printf '%s\n' "$DEST" > "$REG/DEST_HOST"; fi
touch "$REG/push.log"

echo "install_cluster: $HOST_TAG shared=$SHARED"
echo "  spool     $REG/records"
echo "  pending   $REG/pending"
echo "  snapshot  $REG/snapshot   log $REG/log"
echo "  scripts   $SHARED/scripts/registry/{push_records,registry_env,sbatch_registered}.sh"
echo "  site      $SITE_ENV"
if [[ -f "$REG/DEST_HOST" ]]; then echo "  DEST_HOST $(cat "$REG/DEST_HOST")"; else echo "  DEST_HOST  (unset — write user@host of the grading host into $REG/DEST_HOST)"; fi
if [[ -f "$SHARED/.ssh/registry-push" ]]; then echo "  key       $SHARED/.ssh/registry-push (perms $(stat -c %a "$SHARED/.ssh/registry-push"))"
else echo "  key       MISSING: stream it in from the grading host over ssh stdin, e.g."
     echo "            cat /run/user/\$UID/registry-push | ssh <cluster> 'umask 077; cat > $SHARED/.ssh/registry-push'"; fi

# python3 -S importability of the emitter from the mirror (the job shell calls it before uv sync); the mirror is the
# checkout this script was run from
MIRROR="$(cd "$HERE/../.." && pwd)"
if [[ -d "$MIRROR/src/sisteer/registry" ]]; then
  if PYTHONPATH="$MIRROR/src" python3 -S -c "import sisteer.registry.emit, sisteer.registry.emit_cli, sisteer.registry.hooks" 2>/dev/null; then
    echo "  emitter   importable under python3 -S from $MIRROR/src"
  else
    echo "  emitter   NOT importable under python3 -S from $MIRROR/src (check the mirror)" >&2
  fi
fi

PUB="$SHARED/.ssh/registry-push.pub"
if [[ "$PRINT_KEY" == "1" || -f "$PUB" ]]; then
  echo
  echo "# grading host ~/.ssh/authorized_keys — ONE line for the registry-push key (leave every other line as it is):"
  if [[ -f "$PUB" ]]; then KEYTXT="$(awk '{print $1" "$2}' "$PUB")"; else KEYTXT="ssh-ed25519 <registry-push public key>"; fi
  FROM_OPT=""; [[ -z "$PUSH_FROM" ]] || FROM_OPT=",from=\"$PUSH_FROM\""
  echo "command=\"/usr/bin/rrsync -wo -no-del -no-lock /home/${REGISTRY_GRADING_HOST_USER:-$USER}/registry/inbox\",restrict$FROM_OPT $KEYTXT registry-push"
  [[ -n "$PUSH_FROM" ]] || echo "# (no from= on this line: set REGISTRY_PUSH_FROM to this cluster's egress ranges to pin the key to them)"
  echo "# then from this cluster: $SHARED/scripts/registry/push_records.sh --probe   (writes inbox/$HOST_TAG/_probe/<utc>.jsonl on the grading host; it stays in the inbox for you to read — not a record)"
fi
