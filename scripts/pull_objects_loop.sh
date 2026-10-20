#!/bin/bash
# Incremental object pull for the two-model sweep waves — used whenever a
# pod has NO network volume, so trained objects,
# HPO summary.jsons and metrics jsonls survive the pod. Runs on the grading host.
#
# Adapted from the stage-3 fit wave's pull loop, KEEPING its two recorded
# fixes (both bit for real, 2026-08-21):
#
#  1. `mapfile`, never `ssh -n` inside a `while read` loop: ssh inside a
#     read loop eats the loop's stdin, but rsync SPEAKS ITS PROTOCOL over
#     that stdin, so adding -n to rsync's transport broke every transfer.
#     mapfile reads the endpoint list up front and nothing in the loop
#     competes for stdin (the reachability probe's ssh keeps -n; rsync's
#     transport ssh must NEVER get it).
#
#  2. LATCHED stop condition: "stop when no pod is reachable" was also true
#     at startup (pods still booting), so v1 exited on cycle 1. Exit only
#     after some pod HAS been reachable at least once, and then only after
#     3 consecutive fully-unreachable cycles.
#
# Endpoints file: one pod per line, whitespace-separated —
#     <name> <host> <port> <remote_dir> [<remote_dir> ...]
# e.g.  gemma4e4b 203.0.113.7 40123 /workspace/pilot_gemma4e4b /workspace/hpo
# Blank lines and #-comments are skipped. Each remote dir lands at
# <dest>/<name>/<basename(remote_dir)>/. Token-cache sidecars and wandb dirs
# are excluded (large, reproducible); everything else in the listed dirs —
# cell dirs with objects, HPO summary.json trees, metrics jsonls — is pulled.
#
# Usage:
#   pull_objects_loop.sh [--endpoints FILE] [--dest DIR] [--interval SEC]
#                        [--max-hours H]
#
# Verify it is working by watching <dest>/pull.log GROW. A live pid is not
# evidence — `pgrep -f pull_objects_loop.sh` also matches any shell whose
# command line contains that string, including the one asking.
set -u

ENDPOINTS=${TWOSWEEP_ENDPOINTS:-$HOME/.local/state/twosweep/pull_endpoints.txt}
DEST=$HOME/si-steering-wave-objects/twosweep
INTERVAL=120          # objects are small; frequent pulls tighten exposure
MAX_HOURS=48
while [ $# -gt 0 ]; do
  case "$1" in
    --endpoints) ENDPOINTS=$2; shift 2 ;;
    --dest)      DEST=$2; shift 2 ;;
    --interval)  INTERVAL=$2; shift 2 ;;
    --max-hours) MAX_HOURS=$2; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

# refuse at start, before creating anything, when there is no endpoint to pull from (a missing or empty file used to
# idle silently until --max-hours)
grep -qvE '^[[:space:]]*(#|$)' "$ENDPOINTS" 2>/dev/null || {
  echo "pull_objects_loop.sh: no endpoint in $ENDPOINTS (missing, empty or comments only). Write one pod per line," \
       "<name> <host> <port> <remote_dir> [<remote_dir> ...], or pass --endpoints FILE" >&2
  exit 2
}
mkdir -p "$DEST"
LOG=$DEST/pull.log
END=$(( $(date +%s) + MAX_HOURS*3600 ))
ever=0
dead_cycles=0

echo "$(date -u +%H:%M:%S) pull loop up: endpoints=$ENDPOINTS dest=$DEST interval=${INTERVAL}s max_hours=$MAX_HOURS" >> "$LOG"

while [ "$(date +%s)" -lt "$END" ]; do
  reachable=0
  total=0
  ROWS=()
  if [ -s "$ENDPOINTS" ]; then
    mapfile -t ROWS < "$ENDPOINTS"
  fi
  for ROW in ${ROWS+"${ROWS[@]}"}; do
    case "${ROW:-}" in ''|'#'*) continue ;; esac
    set -- $ROW
    NAME=${1:-}; H=${2:-}; P=${3:-}
    shift 3 2>/dev/null || true
    [ -z "$H" ] || [ -z "$NAME" ] && continue
    total=$((total + 1))
    # Reachability probe: -n is CORRECT here (no stdin protocol) and keeps
    # the probe from consuming anything.
    if timeout 60 ssh -n -o StrictHostKeyChecking=accept-new \
         -o ConnectTimeout=15 -p "$P" "root@$H" true 2>/dev/null; then
      reachable=$((reachable + 1))
      ever=1
      for RDIR in "$@"; do
        [ -z "$RDIR" ] && continue
        LDIR=$DEST/$NAME/$(basename "$RDIR")
        mkdir -p "$LDIR"
        # NOTE: deliberately NO -n on the transport ssh; rsync needs the
        # ssh stdin channel (recorded fix 1).
        timeout 600 rsync -az --partial \
          --exclude '*.tokcache.npz' --exclude 'wandb/' \
          -e "ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 -p $P" \
          "root@$H:$RDIR/" "$LDIR/" >/dev/null 2>&1
      done
    fi
    # -type f, NOT -name 'object*': a name glob once matched the CONTAINER
    # DIRS and reported "objects=2" for an empty mirror. A count that rises
    # when nothing has happened is worse than no count.
    n=$(find "$DEST/$NAME" -type f ! -name '*.log' 2>/dev/null | wc -l)
    echo "$(date -u +%H:%M:%S) $NAME reachable=$reachable objects=$n" >> "$LOG"
  done
  if [ "$ever" = "1" ] && [ "$reachable" = "0" ] && [ "$total" -gt 0 ]; then
    dead_cycles=$((dead_cycles + 1))
    echo "$(date -u +%H:%M:%S) all $total pod(s) unreachable ($dead_cycles/3)" >> "$LOG"
    if [ "$dead_cycles" -ge 3 ]; then
      echo "$(date -u +%H:%M:%S) wave over; exiting" >> "$LOG"
      break
    fi
  else
    dead_cycles=0
  fi
  sleep "$INTERVAL"
done
echo "$(date -u +%H:%M:%S) pull loop exit (ever_reachable=$ever)" >> "$LOG"
