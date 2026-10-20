#!/usr/bin/env bash
# push_records.sh — push ONE run's record directory from a cluster to the grading host's registry inbox
# over the SECOND forced-command, write-only key `registry-push`.
#
# Usage (from the emitter after every write; --final from the job's finish step):
#   $SHARED/scripts/registry/push_records.sh <run_record_dir> [--final] [--probe]
#
# Contract of the `registry-push` key (NOT installed by this script):
#   - the grading host's ~/.ssh/authorized_keys carries ONE line for it:
#       command="/usr/bin/rrsync -wo -no-del -no-lock $HOME/registry/inbox",restrict,from="<cluster egress ranges>" ssh-ed25519 <pub> registry-push
#     so the key can WRITE into ~/registry/inbox and nothing else: no shell, no reads, no delete, no forwarding,
#     pinned to the clusters' egress ranges. The figure-push key and its line are left EXACTLY as they are.
#   - cluster side: the private key lives at $SHARED/.ssh/registry-push (cluster A 0600; cluster B reports 0666
#     under MooseFS — this script stages a mktemp 0600 copy exactly as mirror-figs.sh does), $SHARED/.ssh/known_hosts
#     is reused, and $SHARED/registry/DEST_HOST holds `user@host` of the grading host (its external IP is ephemeral,
#     so a re-point is one write per cluster; REGISTRY_PUSH_DEST=user@host overrides it for one call — the offline
#     drill's lever). Records carry no secrets by construction (the validator refuses secret-shaped payload keys).
#
# Behaviour: derives <host> (cluster_a|cluster_b) from the shared root; honours a per-dir .last_push stamp (skips
# when < 30 s old unless --final); rsyncs the dir to <host>/<campaign>/<run_dir>/ RELATIVE to the rrsync root;
# ALWAYS exits 0 — a failed push is appended to $SHARED/registry/push.log and the harvester's pull is the backstop.
set -uo pipefail

SRC=""
FINAL=0
PROBE=0
for a in "$@"; do
  case "$a" in
    --final) FINAL=1 ;;
    --probe) PROBE=1 ;;
    *) SRC="$a" ;;
  esac
done

log() { echo "push_records: $*" >&2; }

# the cluster's shared root and registry label: REGISTRY_SHARED_ROOT / REGISTRY_HOST_TAG (the front door's job env), else the
# values the helper install rendered from the site's cluster block beside this script (registry_site.env, which registry_env.sh reads)
_here="$(CDPATH='' cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REGISTRY_SITE_SHARED_ROOT=""; REGISTRY_SITE_SUBSTRATE=""
if [[ -r "$_here/registry_site.env" ]]; then source "$_here/registry_site.env"; fi
SHARED="${REGISTRY_SHARED_ROOT:-$REGISTRY_SITE_SHARED_ROOT}"
HOST_TAG="${REGISTRY_HOST_TAG:-${REGISTRY_SITE_SUBSTRATE:-unknown}}"
if [[ -z "$SHARED" ]]; then log "no shared root (REGISTRY_SHARED_ROOT, or registry_site.env beside this script); nothing pushed"; exit 0; fi

REG="$SHARED/registry"
KEY="${REGISTRY_PUSH_KEY:-$SHARED/.ssh/registry-push}"
KNOWN="$SHARED/.ssh/known_hosts"
DEST_FILE="$REG/DEST_HOST"
PUSHLOG="$REG/push.log"
mkdir -p "$REG" 2>/dev/null || true

fail() { echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $HOST_TAG $SRC $*" >> "$PUSHLOG" 2>/dev/null; log "$*"; exit 0; }

if [[ ! -f "$KEY" ]]; then fail "no push key at $KEY; harvester pull is the backstop"; fi
# REGISTRY_PUSH_DEST overrides the DEST_HOST file for ONE invocation (the offline drill points a producer at an
# unreachable address without re-pointing the cluster) — never written anywhere, never a default.
DEST_HOST="${REGISTRY_PUSH_DEST:-}"
if [[ -z "$DEST_HOST" ]]; then
  if [[ ! -f "$DEST_FILE" ]]; then fail "no $DEST_FILE (user@host of the grading host)"; fi
  DEST_HOST="$(tr -d '[:space:]' < "$DEST_FILE")"
  [[ -n "$DEST_HOST" ]] || fail "empty $DEST_FILE"
fi

if [[ "$PROBE" == "1" ]]; then
  # the probe file is NOT a record: the grading host merge leaves inbox/<host>/_probe/ in place (never into the log)
  SRC="$(mktemp -d "${TMPDIR:-/tmp}/registry-probe.XXXXXX")"
  mkdir -p "$SRC/_probe"
  printf '{"probe":"%s","host":"%s"}\n' "$(date -u +%Y-%m-%dT%H:%M:%S.000Z)" "$HOST_TAG" > "$SRC/_probe/$(date -u +%Y%m%dT%H%M%SZ).jsonl"
  REL="$HOST_TAG/"
else
  if [[ -z "$SRC" || ! -d "$SRC" ]]; then log "usage: push_records.sh <run_record_dir> [--final] [--probe]"; exit 0; fi
  STAMP="$SRC/.last_push"
  if [[ "$FINAL" != "1" && -f "$STAMP" ]]; then
    now=$(date +%s); then_=$(stat -c %Y "$STAMP" 2>/dev/null || echo 0)
    if (( now - then_ < 30 )); then exit 0; fi
  fi
  # <host>/<campaign>/<run_dir>/ relative to the rrsync root, from .../records/<campaign>/<run_dir>
  RUN_DIR="$(basename "$SRC")"
  CAMPAIGN="$(basename "$(dirname "$SRC")")"
  REL="$HOST_TAG/$CAMPAIGN/$RUN_DIR/"
fi

# MooseFS ignores chmod (cluster B): stage a node-local 0600 copy for the length of the push.
KEY_USE="$KEY"
KEY_PERMS="$(stat -c %a "$KEY" 2>/dev/null || echo 600)"
if [[ "$KEY_PERMS" != "600" && "$KEY_PERMS" != "400" ]]; then
  if KEY_TMP="$(mktemp "${TMPDIR:-/tmp}/registry-push.XXXXXX" 2>/dev/null)"; then
    trap 'rm -f "$KEY_TMP"' EXIT
    cat "$KEY" > "$KEY_TMP" && KEY_USE="$KEY_TMP"
  fi
fi

RSYNC_SSH="ssh -i '$KEY_USE' -o IdentitiesOnly=yes -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile='$KNOWN'"
# Jittered retry: a transient refusal (sshd MaxStartups under a burst, a dropped
# connection) is retried up to REGISTRY_PUSH_ATTEMPTS times (default 3) after a random 2-6 s pause, so
# concurrent producers de-synchronise instead of hammering the same second; a push that still fails is
# logged and left to the harvester's pull. The stamp is touched only on success, so a failed push is
# retried by the next write's call instead of being silenced by the debounce.
ATTEMPTS="${REGISTRY_PUSH_ATTEMPTS:-3}"
rc=0
for (( try=1; try<=ATTEMPTS; try++ )); do
  # --mkpath: the destination <host>/<campaign>/<run_dir>/ is TWO levels below anything that exists in the inbox for a
  # new campaign, and rsync creates only the final component — every real push failed with `mkdir … No such file or
  # directory` while the one-level `_probe` passed (live drill, 2026-09-17). rrsync allows mkpath; rsync >= 3.2.3 has it.
  if rsync -rt --mkpath --ignore-existing --exclude='.last_push' --exclude='*.tmp' --exclude='*.part' \
       -e "$RSYNC_SSH" "$SRC/" "$DEST_HOST:$REL" 2>>"$PUSHLOG"; then
    [[ "$PROBE" == "1" ]] || touch "$SRC/.last_push" 2>/dev/null || true
    [[ "$PROBE" == "1" ]] && rm -rf "$SRC"
    exit 0
  fi
  rc=$?
  if (( try < ATTEMPTS )); then
    pause=$(( 2 + RANDOM % 5 ))
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) $HOST_TAG $SRC rsync attempt $try/$ATTEMPTS failed (exit $rc); retrying in ${pause}s" >> "$PUSHLOG" 2>/dev/null
    sleep "$pause"
  fi
done
fail "rsync to $DEST_HOST:$REL failed after $ATTEMPTS attempts (last exit $rc); the harvester pull will pick the records up"
