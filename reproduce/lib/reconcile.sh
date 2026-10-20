#!/usr/bin/env bash
# reproduce/lib/reconcile.sh — recover what ended while no session was watching: every non-finalised
# submission in the grading host's ledger ($SISTEER_STATE_DIR/<model>/submissions.jsonl) is reconciled against
# BOTH `squeue -u $USER` AND `sacct -X ... -S <ledger start>` on its cluster (watch_job.sh exits at a terminal state,
# so a job that finished during an outage has no watcher event to replay), ownership is validated by job name, the
# outcome is appended to the ledger (`reconciled` events), and any twin election left open is handed to
# `submit_twin.py reconcile` (which fails closed on an inconsistent sentinel state). It then prints the live jobs so the
# session re-arms a watcher for each BEFORE starting new work (the standing reconcile rule).
#
#   reconcile.sh --model <key> [--dry-run] [--config F]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/common.sh"
repro_parse_common_args "$@"
[ "$REPRO_HELP" = 1 ] && { sed -n '2,11p' "$0"; repro_common_help; exit 0; }
repro_load_config; repro_require_model
STATE="$(repro_state_dir "$REPRO_MODEL")"; LEDGER="$STATE/submissions.jsonl"
[ -f "$LEDGER" ] || { repro_say "no submissions ledger at $LEDGER — nothing to reconcile"; exit 0; }
PY="$(repro_python)"
for label in $(cluster_labels); do
  host="$(cluster_var "$label" CLUSTER_SSH)"
  open_jobs="$("$PY" - "$LEDGER" "$label" <<'PYEOF'
import json, sys
ledger, label = sys.argv[1], sys.argv[2]
sub, done = {}, set()
for ln in open(ledger):
    ln = ln.strip()
    if not ln: continue
    d = json.loads(ln)
    if d.get("cluster") != label: continue
    if d.get("event") == "submitted": sub[d["job_id"]] = d
    if d.get("event") in ("reconciled", "terminal", "cancelled"): done.add(d["job_id"])
start = min((d["utc"] for d in sub.values()), default=None)
print(start or "")
for j, d in sorted(sub.items()):
    if j not in done: print(f"{j} {d.get('unit','')}")
PYEOF
)"
  start="$(head -1 <<< "$open_jobs")"; jobs="$(tail -n +2 <<< "$open_jobs")"
  [ -n "$jobs" ] || { repro_say "[$label] nothing open in the ledger"; continue; }
  ids="$(awk '{print $1}' <<< "$jobs" | tr '\n' ',' | sed 's/,$//')"
  if [ "$REPRO_DRY_RUN" = 1 ]; then
    echo "DRY-RUN: ssh $host 'squeue -u \$USER -h -j $ids -o \"%i %j %T %M\"; echo ---; sacct -X -n -P -o JobID,JobName,State,ExitCode,End -S ${start:-now-7days} -j $ids'"
    continue
  fi
  # through repro_ssh (the one ssh face; its local: seam lets tests/test_front_door_wave_integration.py drive this entrypoint)
  rows="$(repro_ssh "$label" "squeue -u \$USER -h -j $ids -o '%i %j %T %M' 2>/dev/null; echo ---; sacct -X -n -P -o JobID,JobName,State,ExitCode,End -S ${start:-now-7days} -j $ids 2>/dev/null")" || { repro_warn "[$label] unreachable ($host) — reconcile again later"; continue; }
  live="${rows%%---*}"; hist="${rows#*---}"
  while read -r jid unit; do
    [ -n "$jid" ] || continue
    if grep -q "^$jid " <<< "$live"; then
      echo "LIVE $label $jid $unit  ->  re-arm: $REPRO_LIB_DIR/watch_job.sh --slurm $jid --host $host --shared-root $(cluster_var "$label" CLUSTER_SHARED_ROOT)"
      continue
    fi
    rec="$(grep -E "^${jid}(_[0-9]+)?\|" <<< "$hist" | head -1)"
    if [ -z "$rec" ]; then echo "UNKNOWN $label $jid $unit  (neither squeue nor sacct knows it — wrong cluster, or older than the sacct window)"; continue; fi
    state="$(cut -d'|' -f3 <<< "$rec")"; jname="$(cut -d'|' -f2 <<< "$rec")"
    echo "TERMINAL $label $jid $unit state=$state name=$jname"
    repro_ledger_append "$REPRO_MODEL" submissions.jsonl "{\"event\":\"reconciled\",\"utc\":\"$(date -u +%FT%TZ)\",\"cluster\":\"$label\",\"job_id\":\"$jid\",\"unit\":\"$unit\",\"state\":\"$state\",\"job_name\":\"$jname\",\"model\":\"$REPRO_MODEL\"}"
  done <<< "$jobs"
done
# the twin elections: replay the ledger, reconcile remote sentinels against it AND the registry, fail closed on inconsistency.
# The global flags (--model/--config/--dry-run) go BEFORE the `reconcile` subcommand, and --dry-run rides ONLY when the stage's
# --dry-run was given (repro_dry_flag reads REPRO_DRY_RUN=1, never a merely-set variable)
if [ -f "$STATE/attempts.jsonl" ]; then
  repro_run "$PY" "$REPRO_LIB_DIR/submit_twin.py" --model "$REPRO_MODEL" --config "$SISTEER_CONFIG" $(repro_dry_flag) reconcile
fi
