#!/bin/bash
# compact_swap.sh — the compaction window: the registry's writers and readers stopped
# and drained, the live database compacted and the copy swapped in by `python -m sisteer.registry.compact --swap`, and the units
# restarted ONLY once `compact --validate-live` exits 0.
#
#   compact_swap.sh [--dry-run] [--root DIR] [--attic-root DIR] [--expect-bytes N] [--min-free-bytes N] [--require-file F]...
#                   [--no-post-checks] [--post-wait-s S]
#   compact_swap.sh --finish ATTIC_DIR [--root DIR] [--no-post-checks]      # the EXIT trap's work, by hand (this script was SIGKILLed)
#
# Order (each line stamped in the log):
#   1. preflight (also all of --dry-run): the database exists, the compact module imports under $PY / $PYPATH, every --require-file
#      exists (the grade daemons' dispatch.hold files: the graders are held before the window), the attic root is on the
#      database's filesystem, the attic name is new; then the one-window lock (window.sh's .window.lock: no landing window and no
#      second swap at the same time). Nothing is stopped before this point; a refusal exits 2.
#   2. every unit of $UNITS has its `is-active` state recorded in <attic>/units.before; the maintenance flag
#      (<root>/state/maintenance.json) is written; the active ones are stopped and confirmed inactive.
#   3. the drain: no service of the set `activating` or `active`, and `fuser <root>/harvester.lock` empty.
#   4. `compact --swap --root R --attic A` (the state machine, under the write lock; its state in <attic>/SWAP_STATE.json).
#   5. THE EXIT TRAP (every exit after step 2): `compact --recover` when the swap did not reach a terminal phase, then
#      `compact --validate-live`. Exit 0 -> the maintenance flag removed and exactly the units that were active restarted and
#      confirmed active; then, after a validated swap, the post checks (Datasette answers on 127.0.0.1:8001 and its open
#      descriptors point at the new inode, nobody holds the attic file, one harvest cycle reached `Finished`). Anything else ->
#      the units LEFT STOPPED, the flag kept, <root>/state/swap_blocked.json written and an ALERT line.
#
# Exit: 0 swapped, validated, units restored, post checks pass | 2 refused before anything stopped | 3 not swapped (the swap
# refused or rolled back), units restored on the old file | 4 swapped and restored, a post check failed | 5 BLOCKED: the restart gate
# refused, units left stopped | 6 a unit did not come back active.
#
# Run it detached (systemd-run --user --collect ... compact_swap.sh), so a closed session cannot kill it between the stop and the
# restart. To abort, `kill -TERM <its MainPID>`; never `systemctl stop` its unit (that kills the trap's own children). After a
# SIGKILL nothing restarts on its own (the safe default): run `compact_swap.sh --finish <attic>`.
#
# Test seams (tests/test_registry_compact.py): COMPACT_SWAP_UNITS (the unit set), COMPACT_SWAP_COMPACT (the compact command prefix),
# COMPACT_SWAP_WINDOW_LOCK, COMPACT_SWAP_QUIET_TRIES / COMPACT_SWAP_QUIET_SLEEP_S, COMPACT_SWAP_DATASETTE_URL.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TREE="$(cd "$HERE/../../.." && pwd)"

ROOT="${REGISTRY_ROOT:-$HOME/registry}"
ATTIC_ROOT="$HOME/registry-attic"
EXPECT_BYTES=0
MIN_FREE_BYTES=0
DRY_RUN=0
POST_CHECKS=1
POST_WAIT_S=1200
FINISH=""
REQUIRE=()
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --attic-root) ATTIC_ROOT="$2"; shift 2 ;;
    --expect-bytes) EXPECT_BYTES="$2"; shift 2 ;;
    --min-free-bytes) MIN_FREE_BYTES="$2"; shift 2 ;;
    --require-file) REQUIRE+=("$2"); shift 2 ;;
    --no-post-checks) POST_CHECKS=0; shift ;;
    --post-wait-s) POST_WAIT_S="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --finish) FINISH="$2"; shift 2 ;;
    -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
    *) echo "compact_swap: unknown argument: $1" >&2; exit 2 ;;
  esac
done
for n in "$EXPECT_BYTES" "$MIN_FREE_BYTES" "$POST_WAIT_S"; do
  [[ $n =~ ^[0-9]+$ ]] || { echo "compact_swap: not a non-negative integer: $n" >&2; exit 2; }
done

if [ -n "${COMPACT_SWAP_PY:-}" ]; then PY="$COMPACT_SWAP_PY"
elif [ -x "$TREE/.venv/bin/python" ]; then PY="$TREE/.venv/bin/python"
else PY=python3; fi
PYPATH="${COMPACT_SWAP_PYPATH:-$TREE/src}"
COMPACT="${COMPACT_SWAP_COMPACT:-env PYTHONPATH=$PYPATH $PY -m sisteer.registry.compact}"
# the window's stop set (window.sh TIMERS) + the paper builds + the gate baseline + Datasette (a long reader holding the old inode)
UNITS="${COMPACT_SWAP_UNITS:-registry-harvest.timer registry-cluster-reads.timer registry-adapters.timer registry-modal-pull.timer registry-status.timer registry-status-heavy.timer registry-store-mirror.timer paper-autobuild.timer registry-paper-rebuild.timer registry-snapshot.timer registry-gate-baseline.timer registry-datasette.service}"
SERVICES="${UNITS//.timer/.service}"
WINDOW_LOCK="${COMPACT_SWAP_WINDOW_LOCK:-$HOME/.local/state/registry-landing/windows/.window.lock}"
QUIET_TRIES="${COMPACT_SWAP_QUIET_TRIES:-240}"
QUIET_SLEEP="${COMPACT_SWAP_QUIET_SLEEP_S:-15}"
DATASETTE_URL="${COMPACT_SWAP_DATASETTE_URL:-http://127.0.0.1:8001/registry/}"
LIVE="$ROOT/registry.sqlite"
MAINT_FLAG="$ROOT/state/maintenance.json"
BLOCKED="$ROOT/state/swap_blocked.json"

stamp() { date -u +%H:%M:%S; }
log() { echo "$* $(stamp)"; }
refuse() { echo "compact_swap: REFUSED: $* (nothing stopped) $(stamp)"; exit 2; }
phase_of() {  # the recorded phase of a swap (empty: no state file)
  [ -f "$1/SWAP_STATE.json" ] || return 0
  "$PY" -c 'import json,sys; print(json.load(open(sys.argv[1])).get("phase") or "")' "$1/SWAP_STATE.json" 2>/dev/null
}
is_terminal() { case "$1" in validated|rolled_back) return 0 ;; *) return 1 ;; esac; }

STOPPED=0
FLAG_WRITTEN=0
SWAP_RC=""
ATTIC=""
ACTIVE=""

post_checks() {  # after a validated swap and the restart: 0 when every check passes
  local bad=0 pid code="" i t0 live_ino fd_ino fds=0 st
  live_ino=$(stat -c %i "$LIVE")
  t0=$(date +%s)
  # Datasette opens the database on its first request: ask until it answers 200 (it may need the page cache warmed)
  for i in $(seq 1 60); do
    code=$(curl -s -o /dev/null -m 60 -w '%{http_code}' "$DATASETTE_URL" 2>/dev/null || true)
    [ "$code" = 200 ] && break
    [ $(( $(date +%s) - t0 )) -ge "$POST_WAIT_S" ] && break
    sleep 10
  done
  if [ "$code" = 200 ]; then log "post check: Datasette answers 200 at $DATASETTE_URL"; else log "POST CHECK FAILED: Datasette at $DATASETTE_URL answered '${code:-nothing}'"; bad=1; fi
  pid=$(systemctl --user show -p MainPID --value registry-datasette.service 2>/dev/null || echo 0)
  if [ -n "$pid" ] && [ "$pid" != 0 ] && [ -d "/proc/$pid/fd" ]; then
    for fd in /proc/"$pid"/fd/*; do
      case "$(readlink "$fd" 2>/dev/null)" in
        "$LIVE"|"$LIVE"\ \(deleted\)) fd_ino=$(stat -L -c %i "$fd" 2>/dev/null); fds=$((fds + 1))
          [ "$fd_ino" = "$live_ino" ] || { log "POST CHECK FAILED: Datasette (pid $pid) holds $fd on inode $fd_ino, not the new $live_ino"; bad=1; } ;;
      esac
    done
    log "post check: Datasette (pid $pid) holds $fds descriptor(s) of the live path, all on inode $live_ino"
  else
    log "POST CHECK FAILED: registry-datasette.service has no main PID ('$pid')"; bad=1
  fi
  if fuser "$ATTIC/registry.sqlite" >/dev/null 2>&1; then log "POST CHECK FAILED: the attic file is held open: $(fuser "$ATTIC/registry.sqlite" 2>&1)"; bad=1
  else log "post check: nobody holds $ATTIC/registry.sqlite"; fi
  case " $ACTIVE " in
    *" registry-harvest.timer "*)
      st=""
      while [ $(( $(date +%s) - t0 )) -lt "$POST_WAIT_S" ]; do
        # systemd's own unit messages only (the harvester's output lines may say "failed" about records)
        st=$(journalctl --user -u registry-harvest.service --since "@$RESTART_EPOCH" --no-pager -o cat 2>/dev/null \
             | grep -E '^Finished registry-harvest|^Failed to start registry-harvest|^registry-harvest\.service: (Deactivated successfully|Failed with result)' | head -1)
        [ -n "$st" ] && break
        sleep 15
      done
      case "$st" in
        Finished*|*": Deactivated successfully"*) log "post check: one harvest cycle after the restart: $st" ;;
        *) log "POST CHECK FAILED: no completed harvest cycle within ${POST_WAIT_S}s of the restart (${st:-none})"; bad=1 ;;
      esac ;;
    *) log "post check: the harvest timer was not active before the window; no cycle expected" ;;
  esac
  return $bad
}

finish() {
  local rc=$? vrc rrc phase out not_active="" u result
  # disarmed first: a second signal must not cut the recovery or the restart short (the trap would otherwise re-enter)
  trap '' TERM INT HUP
  trap - EXIT
  [ -n "$SWAP_RC" ] || SWAP_RC=$rc
  if [ "$STOPPED" != 1 ] && [ "$FLAG_WRITTEN" != 1 ]; then
    log "nothing was stopped (rc=$rc): nothing to restore"
    echo "=== compact swap end $(date -u +%FT%TZ) result=not-started rc=$rc"
    exit "$rc"
  fi
  phase=$(phase_of "$ATTIC")
  if [ "$SWAP_RC" != 0 ] || ! is_terminal "$phase"; then
    log "-- recover (swap rc=$SWAP_RC, phase '${phase:-none}')"
    $COMPACT --recover --attic "$ATTIC"; rrc=$?
    phase=$(phase_of "$ATTIC")
    log "recover rc=$rrc, phase now '${phase:-none}'"
  fi
  log "-- validate-live (the restart gate)"
  $COMPACT --validate-live --root "$ROOT" --attic "$ATTIC" > "$ATTIC/validate-live.json"; vrc=$?
  tail -n 40 "$ATTIC/validate-live.json"
  if [ "$vrc" != 0 ]; then
    mkdir -p "$(dirname "$BLOCKED")"
    "$PY" -c 'import json,sys,time; json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "attic": sys.argv[1], "phase": sys.argv[2], "swap_rc": sys.argv[3], "validate_rc": sys.argv[4], "units_before": open(sys.argv[1] + "/units.before").read().split("\n"), "remedy": "read <attic>/SWAP_STATE.json and validate-live.json; compact --recover / --validate-live; then compact_swap.sh --finish <attic>"}, open(sys.argv[5], "w"), indent=1)' \
      "$ATTIC" "${phase:-none}" "$SWAP_RC" "$vrc" "$BLOCKED"
    echo "ALERT: the restart gate REFUSED (validate-live rc=$vrc, phase '${phase:-none}'): every unit LEFT STOPPED, the maintenance flag kept; $BLOCKED written $(stamp)"
    echo "=== compact swap end $(date -u +%FT%TZ) result=BLOCKED rc=5"
    exit 5
  fi
  if [ -e "$MAINT_FLAG" ]; then rm -f "$MAINT_FLAG" && log "maintenance flag removed"; fi
  [ -e "$BLOCKED" ] && mv -f "$BLOCKED" "$ATTIC/swap_blocked.cleared.json" && log "an earlier swap_blocked.json moved to the attic (the gate passes now)"
  RESTART_EPOCH=$(date +%s)
  if [ -n "$ACTIVE" ]; then
    systemctl --user start $ACTIVE || log "start returned non-zero for: $ACTIVE"
  fi
  for u in $ACTIVE; do [ "$(systemctl --user is-active "$u" 2>/dev/null)" = active ] || not_active="$not_active $u"; done
  if [ -n "$not_active" ]; then
    log "units NOT restored (not active after the start):$not_active"
    echo "=== compact swap end $(date -u +%FT%TZ) result=${phase:-not-swapped} rc=6"
    exit 6
  fi
  log "units restored (active before the window): ${ACTIVE:-none}"
  if [ "$phase" = validated ]; then
    result=validated; out=0
    if [ "$POST_CHECKS" = 1 ]; then post_checks || out=4; else log "post checks skipped (--no-post-checks)"; fi
  else
    result="${phase:-not-swapped}"; out=3
  fi
  echo "=== compact swap end $(date -u +%FT%TZ) result=$result rc=$out"
  exit "$out"
}

# ---- --finish ATTIC: the trap's work after this script itself died (SIGKILL), from the attic's records
if [ -n "$FINISH" ]; then
  ATTIC="$(cd "$FINISH" && pwd)" || { echo "compact_swap: no attic $FINISH" >&2; exit 2; }
  [ -s "$ATTIC/units.before" ] || { echo "compact_swap: $ATTIC/units.before is missing: restart the units by hand" >&2; exit 2; }
  ACTIVE=$(awk '$2 == "active" {printf "%s ", $1}' "$ATTIC/units.before")
  STOPPED=1; FLAG_WRITTEN=1; SWAP_RC=1
  echo "=== compact swap finish $(date -u +%FT%TZ) attic=$ATTIC units=${ACTIVE:-none}"
  finish
fi

# ---- 1. preflight (nothing stopped; a refusal exits 2)
echo "=== compact swap start $(date -u +%FT%TZ) root=$ROOT"
[ -f "$LIVE" ] || refuse "$LIVE is not a file"
$COMPACT --help >/dev/null 2>&1 || refuse "the compact module does not run: $COMPACT"
for f in "${REQUIRE[@]+"${REQUIRE[@]}"}"; do [ -e "$f" ] || refuse "required file absent: $f"; done
for c in systemctl fuser flock stat; do command -v "$c" >/dev/null || refuse "$c not found"; done
mkdir -p "$ATTIC_ROOT" 2>/dev/null || refuse "cannot create $ATTIC_ROOT"
[ "$(stat -c %d "$ATTIC_ROOT")" = "$(stat -c %d "$ROOT")" ] || refuse "$ATTIC_ROOT is not on the database's filesystem (the attic is a hard link)"
ATTIC="$ATTIC_ROOT/pre-prune-$(date -u +%Y%m%dT%H%M%SZ)"
[ ! -e "$ATTIC" ] || refuse "$ATTIC exists"
NOW_ACTIVE=$(for u in $UNITS; do [ "$(systemctl --user is-active "$u" 2>/dev/null)" = active ] && printf "%s " "$u"; done)
log "preflight: database $(stat -c %s "$LIVE") B, attic $ATTIC, units active now: ${NOW_ACTIVE:-none}"
if [ "$DRY_RUN" = 1 ]; then
  echo "dry run: would stop ${NOW_ACTIVE:-nothing}, drain [$SERVICES], run: $COMPACT --swap --root $ROOT --attic $ATTIC --expect-bytes $EXPECT_BYTES --min-free-bytes $MIN_FREE_BYTES"
  echo "dry run: then --recover if needed, --validate-live, and restart exactly the units recorded active; nothing was changed"
  exit 0
fi
mkdir -p "$(dirname "$WINDOW_LOCK")"
exec 9>>"$WINDOW_LOCK"
flock -n 9 || refuse "the window lock $WINDOW_LOCK is held (a landing window or another swap is running)"

# ---- 2. record, flag, stop
mkdir -p "$ATTIC"
for u in $UNITS; do printf '%s %s\n' "$u" "$(systemctl --user is-active "$u" 2>/dev/null || true)"; done > "$ATTIC/units.before"
ACTIVE=$(awk '$2 == "active" {printf "%s ", $1}' "$ATTIC/units.before")
trap finish EXIT
trap 'exit 143' TERM INT HUP
mkdir -p "$(dirname "$MAINT_FLAG")"
printf '{"since_utc": "%s", "reason": "compaction swap (compact_swap.sh pid %s)", "pid": %s, "attic": "%s"}\n' "$(date -u +%FT%TZ)" "$$" "$$" "$ATTIC" > "$MAINT_FLAG.tmp.$$" \
  && mv -f "$MAINT_FLAG.tmp.$$" "$MAINT_FLAG" && FLAG_WRITTEN=1 && log "maintenance flag written: $MAINT_FLAG" || { log "maintenance flag could not be written"; exit 3; }
log "units active at the stop (recorded in $ATTIC/units.before): ${ACTIVE:-none}"
STOPPED=1
if [ -n "$ACTIVE" ]; then systemctl --user stop $ACTIVE || log "stop returned non-zero for: $ACTIVE"; fi
still=$(for u in $ACTIVE; do case "$(systemctl --user is-active "$u" 2>/dev/null)" in inactive|failed|unknown|"") ;; *) printf "%s " "$u" ;; esac; done)
[ -z "$still" ] || { log "units not stopped: $still"; SWAP_RC=3; exit 3; }
log "units stopped"

# ---- 3. the drain
quiet() {
  local s st
  for s in $SERVICES; do st=$(systemctl --user is-active "$s" 2>/dev/null || true); case "$st" in activating|active|deactivating|reloading) return 1 ;; esac; done
  ! fuser "$ROOT/harvester.lock" >/dev/null 2>&1
}
i=0
until quiet; do i=$((i + 1)); [ "$i" -ge "$QUIET_TRIES" ] && { log "writers never quiet"; SWAP_RC=3; exit 3; }; sleep "$QUIET_SLEEP"; done
log "writers quiet"

# ---- 4. the swap
log "-- compact --swap"
$COMPACT --swap --root "$ROOT" --attic "$ATTIC" --expect-bytes "$EXPECT_BYTES" --min-free-bytes "$MIN_FREE_BYTES" > "$ATTIC/swap.json"
SWAP_RC=$?
log "compact --swap rc=$SWAP_RC, phase '$(phase_of "$ATTIC")'"
exit "$SWAP_RC"
