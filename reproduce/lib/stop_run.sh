#!/usr/bin/env bash
# reproduce/lib/stop_run.sh — stop a self-escalating / self-resubmitting run FOR GOOD: write its STOP sentinel(s) on the
# cluster, let them propagate over shared storage, THEN scancel by job NAME. A bare scancel looks like a preemption to the
# job's TERM trap and it resubmits itself; writing the sentinel and cancelling in the same instant loses the race.
#
#   stop_run.sh --cluster <label> --name <job-name> [--stop-dir DIR]... [--dry-run] [--config F]
#   The helper's CONTRACT sentinel $SHARED/exp/training/<job-name>/STOP is ALWAYS written (the fit path's location; what
#   the gpu-avail stop-run.sh helper writes too). --stop-dir DIR (repeatable) ADDS a job's own sentinel location — the
#   serving probe's record dir ($SERVING_PROBE_OUT, checked as $OUT/STOP), a wave job's attempt dir (see slurm.sh
#   cancel-attempt — never $WAVE_ROOT/STOP, which stops the WHOLE wave, §3.9). Every sentinel is written BEFORE the
#   cancel, so a job that reads only one of the locations still sees a deliberate stop: the serving probe honours both
#   locations, and this helper writes both.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER=""; NAME=""; STOP_DIRS=(); DRY=0; CONFIG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER="$2"; shift 2;; --name) NAME="$2"; shift 2;; --stop-dir) STOP_DIRS+=("$2"); shift 2;;
    --dry-run) DRY=1; shift;; --config) CONFIG="$2"; shift 2;; -h|--help) sed -n '2,13p' "$0"; exit 0;;
    *) echo "stop_run.sh: unknown argument $1" >&2; exit 2;;
  esac
done
[ -n "$CLUSTER" ] && [ -n "$NAME" ] || { echo "usage: stop_run.sh --cluster <label> --name <job-name> [--stop-dir DIR]..." >&2; exit 2; }
source "$HERE/common.sh"; REPRO_CONFIG="$CONFIG"; REPRO_DRY_RUN=$DRY; repro_load_config; cluster_require "$CLUSTER"
HOST="$(cluster_var "$CLUSTER" CLUSTER_SSH)"; SHARED="$(cluster_var "$CLUSTER" CLUSTER_SHARED_ROOT)"
# the contract location first, then every job-specific one (deduplicated); all written before any cancel
ALL_DIRS=("$SHARED/exp/training/$NAME")
for d in "${STOP_DIRS[@]}"; do
  case " ${ALL_DIRS[*]} " in *" $d "*) ;; *) ALL_DIRS+=("$d");; esac
done
WRITE=""
for d in "${ALL_DIRS[@]}"; do
  WRITE="$WRITE mkdir -p '$d'; touch '$d/STOP'; echo 'STOP sentinel written: $d/STOP';"
done
REMOTE="set -u;$WRITE sleep 5; \
JOBS=\$(squeue -h -u \"\$USER\" -n '$NAME' -o '%i'); [ -n \"\$JOBS\" ] || { echo 'no queued/running jobs named $NAME — nothing to cancel'; exit 0; }; \
echo \"cancelling: \$JOBS\"; scancel -u \"\$USER\" --name='$NAME'; sleep 5; \
LEFT=\$(squeue -h -u \"\$USER\" -n '$NAME' -o '%i'); [ -z \"\$LEFT\" ] && echo 'confirmed: no $NAME jobs remain (no successor was submitted)' || echo \"WARNING: still present after cancel: \$LEFT\""
repro_ssh "$CLUSTER" "$REMOTE"
