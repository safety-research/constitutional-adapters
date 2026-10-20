#!/usr/bin/env bash
# reproduce/lib/watch_job.sh — the DURABLE watcher layer: one line per notable event of ONE Slurm job (or array),
# appended as JSON to --events (the grading host copy) and --remote-events (the cluster's $SHARED/exp/reproduce/<model>/watch/
# <jobid>.jsonl) so an outcome survives a dead session; exits at the terminal state. The stage script launches one per
# submission (lib/common.sh repro_submitted); the EXECUTOR SESSION arms its own Monitor(persistent) from the printed
# `submitted <cluster> <jobid> <unit>` line — this file is the fallback, never the only watcher.
#
#   watch_job.sh --slurm <jobid> --host <ssh alias> --shared-root <root> [--events F] [--remote-events F] [--unit U]
#                [--poll 120] [--stall-min 20] [--pending-min 30] [--max-hours 48] [--tail 30]
# Events: STARTED (node + resolved log), PENDING (once past --pending-min, with squeue's reason), STALL (log idle),
# FAST-FAIL (dead inside the first 2 minutes of RUNNING — never scientific: config, path, args, env), UNREACHABLE,
# NOISE (sacct/squeue disagree), MISSING, then exactly one of DONE / FAILED (with the .out + .err tail), or WATCH-TIMEOUT.
set -uo pipefail
JOB=""; HOST=""; ROOT=""; EVENTS=""; REVENTS=""; UNIT=""; POLL=120; STALL_MIN=20; PENDING_MIN=30; MAX_HOURS=48; TAILN=30
while [ $# -gt 0 ]; do
  case "$1" in
    --slurm) JOB="$2"; shift 2;; --host) HOST="$2"; shift 2;; --shared-root) ROOT="$2"; shift 2;;
    --events) EVENTS="$2"; shift 2;; --remote-events) REVENTS="$2"; shift 2;; --unit) UNIT="$2"; shift 2;;
    --poll) POLL="$2"; shift 2;; --stall-min) STALL_MIN="$2"; shift 2;; --pending-min) PENDING_MIN="$2"; shift 2;;
    --max-hours) MAX_HOURS="$2"; shift 2;; --tail) TAILN="$2"; shift 2;;
    -h|--help) sed -n '2,13p' "$0"; exit 0;;
    *) echo "watch_job.sh: unknown argument $1" >&2; exit 2;;
  esac
done
[ -n "$JOB" ] && [ -n "$HOST" ] && [ -n "$ROOT" ] || { echo "usage: watch_job.sh --slurm <jobid> --host <ssh> --shared-root <root> [...]" >&2; exit 2; }
[ "$POLL" -ge 30 ] || POLL=30
# `-n`: the watcher is forked by a stage that is still reading its plan from stdin; an ssh that inherited that stdin would drain it
SSH=(ssh -n -o ConnectTimeout=15 -o BatchMode=yes -o StrictHostKeyChecking=accept-new)
START=$(date +%s); DEADLINE=$(( START + MAX_HOURS * 3600 ))
hms(){ printf '%dh%02dm' $(($1/3600)) $((($1%3600)/60)); }
json_str(){ python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$1" 2>/dev/null || printf '"%s"' "${1//\"/\\\"}"; }
say(){   # <EVENT> <text>: one stdout line == one notification; the same event as a JSON line in both ledgers
  local ev="$1"; shift; local txt="$*"
  printf '%s job=%s host=%s%s %s\n' "$ev" "$JOB" "$HOST" "${UNIT:+ unit=$UNIT}" "$txt"
  local line; line="{\"event\":\"$ev\",\"utc\":\"$(date -u +%FT%TZ)\",\"job_id\":\"$JOB\",\"host\":\"$HOST\",\"unit\":$(json_str "$UNIT"),\"text\":$(json_str "$txt")}"
  if [ -n "$EVENTS" ]; then mkdir -p "$(dirname "$EVENTS")"; echo "$line" >> "$EVENTS"; fi
  if [ -n "$REVENTS" ]; then printf '%s\n' "$line" | "${SSH[@]}" "$HOST" "mkdir -p '$(dirname "$REVENTS")' && cat >> '$REVENTS'" 2>/dev/null || true; fi
}
unreach=0; announced=0
note_unreachable(){ unreach=$((unreach+1)); if [ $unreach -ge 3 ] && [ $announced -eq 0 ]; then say UNREACHABLE "$unreach consecutive ssh failures; retrying every ${POLL}s"; announced=1; fi; }
note_reachable(){ [ $announced -eq 1 ] && say REACHABLE "ssh recovered"; unreach=0; announced=0; }
seen_running=0; announced_pending=0; last_stall=0; logfile=""; node=""; running_since=0
while :; do
  now=$(date +%s)
  [ "$now" -lt "$DEADLINE" ] || { say WATCH-TIMEOUT "watcher hit its ${MAX_HOURS}h bound; the job may still be alive — re-arm"; exit 3; }
  out=$("${SSH[@]}" "$HOST" "sacct -X -n -P -o State,ExitCode,Elapsed,NodeList,JobID -j $JOB" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ]; then note_unreachable; sleep "$POLL"; continue; fi
  note_reachable
  if [ -z "$out" ]; then
    [ $((now-START)) -gt 600 ] && { say MISSING "no sacct record after 10 min — wrong job id or cluster?"; exit 0; }
    sleep "$POLL"; continue
  fi
  active=0; states=""
  while IFS='|' read -r st ec el nl jid; do
    [ -n "$st" ] || continue; states="$states${states:+,}${st%% *}"
    case "${st%% *}" in PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|COMPLETING|CONFIGURING) active=1;; esac
    [ -n "$nl" ] && [ "$nl" != "None assigned" ] && node="$nl"
  done <<< "$out"
  if [ $active -eq 0 ]; then
    # cross-talk guard: the live queue must not list it; a second sacct read >= 15 s later must agree
    live=$("${SSH[@]}" "$HOST" "squeue -h -j $JOB -o %T 2>/dev/null" 2>/dev/null)
    [ -n "$live" ] && { say NOISE "sacct terminal but squeue shows ${live//$'\n'/,} — still polling"; sleep "$POLL"; continue; }
    first=$("${SSH[@]}" "$HOST" "sacct -X -n -P -o JobName,State -j $JOB" 2>/dev/null | head -1); sleep 15
    summary=$("${SSH[@]}" "$HOST" "sacct -X -n -P -o JobID,JobName,State,ExitCode,Elapsed,NodeList -j $JOB" 2>/dev/null)
    second=$(awk -F'|' 'NR==1{print $2"|"$3}' <<< "$summary")
    [ "$first" = "$second" ] || { say NOISE "two sacct reads disagree ($first vs $second) — still polling"; sleep "$POLL"; continue; }
    grep -qE '\|(PENDING|RUNNING|REQUEUED|RESIZING|SUSPENDED|COMPLETING|CONFIGURING)[^|]*\|' <<< "$summary" && { sleep "$POLL"; continue; }
    failed=0; grep -qE '\|(COMPLETED)\|' <<< "$summary" || failed=1
    grep -qE '\|(FAILED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|PREEMPTED|CANCELLED[^|]*)\|' <<< "$summary" && failed=1
    if [ $failed -eq 0 ]; then say DONE "after $(hms $((now-START))): ${summary//$'\n'/ ; }"; exit 0; fi
    ev=FAILED; [ $seen_running -eq 1 ] && [ $running_since -gt 0 ] && [ $((now-running_since)) -le 180 ] && ev=FAST-FAIL
    say $ev "after $(hms $((now-START))): ${summary//$'\n'/ ; }"
    if [ -z "$logfile" ]; then   # reconstruct <root>/exp/logs/<name>_<rawid>.out (an array's %j is each task's RAW id)
      jn=$(head -1 <<< "$summary" | cut -d'|' -f2); ji=$(head -1 <<< "$summary" | cut -d'|' -f1); ids="${ji%%_*}"
      ids="$ids $("${SSH[@]}" "$HOST" "sacct -X -n -P -o JobIDRaw -j ${ji%%_*} 2>/dev/null | tr '\n' ' '" 2>/dev/null)"
      ids=$(tr ' ' '\n' <<< "$ids" | grep -E '^[0-9]+$' | sort -u | tr '\n' ' ')
      logfile=$("${SSH[@]}" "$HOST" "for i in $ids; do f='$ROOT'/exp/logs/${jn}_\$i.out; [ -f \"\$f\" ] && { echo \"\$f\"; exit 0; }; done; \
        for i in $ids; do g=\$(timeout 60 find '$ROOT' -maxdepth 5 -name \"*\$i*.out\" ! -name '*4294967294*' -size +0 2>/dev/null | head -1); [ -n \"\$g\" ] && { echo \"\$g\"; exit 0; }; done" 2>/dev/null)
    fi
    if [ -n "$logfile" ]; then
      tail_txt=$("${SSH[@]}" "$HOST" "for f in '$logfile' \"\$(printf '%s' '$logfile' | sed 's/\\.out\$/.err/')\"; do [ -s \"\$f\" ] || continue; echo \"### \$f\"; tail -n $TAILN \"\$f\"; done" 2>/dev/null || true)
      say TAIL "$tail_txt"
    else
      say TAIL "(no log file found; expected $ROOT/exp/logs/<jobname>_<jobid>.out)"
    fi
    exit 0
  fi
  if [[ "$states" == *RUNNING* ]] && [ $seen_running -eq 0 ]; then
    seen_running=1; running_since=$now
    jtask=$("${SSH[@]}" "$HOST" "squeue -h -j $JOB -t RUNNING -o '%i' 2>/dev/null | head -1" 2>/dev/null)
    logfile=$("${SSH[@]}" "$HOST" "scontrol show job -o ${jtask:-$JOB} 2>/dev/null | grep -o 'StdOut=[^ ]*' | head -1 | cut -d= -f2-" 2>/dev/null)
    case "$logfile" in *4294967294*) logfile="";; esac
    say STARTED "node=${node:-?} log=${logfile:-unknown}"
  fi
  if [ $seen_running -eq 0 ] && [ "${announced_depfail:-0}" -eq 0 ]; then
    # a Slurm afterok whose predecessor FAILED never starts: the job pends as DependencyNeverSatisfied for ever. Name it (once) — the
    # depended unit is re-run, this job is cancelled by name; nobody hand-releases the dependency
    dreason=$("${SSH[@]}" "$HOST" "squeue -h -j $JOB -o '%R' 2>/dev/null | head -1" 2>/dev/null)
    case "$dreason" in *DependencyNeverSatisfied*)
      say DEPENDENCY-NEVER-SATISFIED "reason=$dreason — the job this one runs after (afterok) FAILED: re-run that unit, then cancel this job by name (scancel -u \$USER --name=<name>); it will never start"
      announced_depfail=1;;
    esac
  fi
  if [ $seen_running -eq 0 ] && [ "$PENDING_MIN" -gt 0 ] && [ $announced_pending -eq 0 ] && [ $((now-START)) -ge $((PENDING_MIN*60)) ]; then
    reason=$("${SSH[@]}" "$HOST" "squeue -h -j $JOB -o '%R' 2>/dev/null | head -1" 2>/dev/null)
    say PENDING "waited=$(hms $((now-START))) reason=${reason:-unknown} — right-size --time, check the largest free block, or twin to the other cluster"
    announced_pending=1
  fi
  if [ $seen_running -eq 1 ] && [ "$STALL_MIN" -gt 0 ] && [ -n "$logfile" ]; then
    mt=$("${SSH[@]}" "$HOST" "stat -c %Y '$logfile' 2>/dev/null" 2>/dev/null)
    if [[ "$mt" =~ ^[0-9]+$ ]]; then idle=$((now-mt))
      if [ $idle -ge $((STALL_MIN*60)) ] && [ $((now-last_stall)) -ge $((STALL_MIN*60)) ]; then
        stail=$("${SSH[@]}" "$HOST" "tail -n 3 '$logfile' 2>/dev/null" 2>/dev/null)
        say STALL "$logfile has not grown in $((idle/60)) min while RUNNING — last lines: ${stail//$'\n'/ | }"
        last_stall=$now
      fi
    fi
  fi
  sleep "$POLL"
done
