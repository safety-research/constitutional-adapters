#!/bin/bash
# baseline_refresh.sh [--sha SHA] [--quiet] — keep the Slurm gate's BASELINE current with origin/main (S6: a landing whose
# main moved under it is never refused for a stale baseline). Run every 2 min by registry-gate-baseline.timer (oneshot,
# Nice 10) and every 30 s by land.sh while it waits for the baseline of the main it merged. One tick at a time: a tick
# takes flock on $STATE/baselines/.lock and exits 0 at once when another tick holds it, so the timer and a waiting land.sh
# never both submit the same main.
#
# One tick, in order:
#   1. NEW = `git ls-remote origin refs/heads/main` (or --sha SHA: land.sh's merged main). No fetch into any checkout.
#   2. Every job recorded under $STATE/baselines/<sha>.job is WATCHED (the timer is the watcher) from ONE squeue read per
#      tick (ids, states, reasons, nodes AND job names) — the venv job the array depends on, the array and the finalizer:
#        in the queue → PENDING (reason) / RUNNING (STARTED once, with the node), one journal line per array state change;
#        a venv job gone from the queue while the array still waits on it → sacct: built → nothing more; failed (seen on
#        two consecutive ticks) → the array and the finalizer are canceled (they could never start) and the failure is
#        handled as below;
#        gone from the queue → sacct, TRUSTED ONLY WHEN: every row kept is this record's by JobID AND JobName (this
#        cluster's sacct can answer for another job on a fresh id — ~/scripts/watch-job.sh; a row with our id under another
#        name is a MISMATCH: journaled, state PENDING_CONFIRM, nothing decided), every kept row is terminal and the
#        finalizer's is among them (else PENDING_CONFIRM), and the SAME terminal rows are read on two consecutive ticks
#        while the jobs stay out of the queue (the first read's digest is kept in CONFIRM; a reappearance in the queue
#        clears it). Only then: finalizer COMPLETED → `gate_slurm.sh <sha> --collect-only --baseline` → $STATE/BASELINE +
#        summaries/summary-<sha>.json, state COLLECTED; a task, the finalizer or the venv NODE_FAIL'd → resubmit with
#        --exclude <those nodes> (at most 3 attempts; the ended attempt's record moves to <sha>.job.attempt<n>, state
#        RESUBMITTED, carrying the exclusion list its replacement uses); any other failure → state FAILED with the
#        finalizer's, the first failed task's and the venv's .out tails in the journal;
#        4 h after the sha's FIRST submission (per sha, not per attempt) a job still queued, or in neither squeue nor a
#        trusted terminal sacct read, → state GAVE_UP (the jobs keep their walltimes; nothing is canceled).
#      A FAILED / GAVE_UP .job blocks a fresh submission of that sha until an operator removes the file — a broken main never
#      loops on the cluster. A CLAIMED .job (step 3) found by a tick means the claiming tick died mid-submission (killed at
#      TimeoutStartSec, a reboot): the queue is read for jobs under the sha's names — array and finalizer live → adopted
#      and watched from here; the array alone → recorded PARTIAL (below). Nothing live → the claim is RECONCILED before it
#      is dropped (a grading host restart may have spanned the whole run): (i) sacct by JobName over the last 24 h — rows under
#      the sha's names → the newest array and finalizer are adopted as an ended attempt and decided below under the
#      two-read rule; (ii) the cluster results dir — a summary.json for the sha → `--collect-only`, COLLECTED when it is
#      terminal. Only when the queue, the accounting and the results dir are all empty is the claim dropped and journaled.
#      A PARTIAL .job (an array queued without its finalizer: the finalizer's sbatch failed, or a claimer died between the
#      two) is canceled, the queue re-read, and the record dropped only once the array is gone — nothing is resubmitted
#      for that sha until then.
#   3. NEW has no terminal summary and no .job → submit: a CLAIMED <sha>.job is written FIRST (ownership persists before any
#      remote work), then `git fetch origin NEW` INTO THE OBJECT STORE (the base checkout never moves — gate_slurm.sh reads
#      pyproject.toml / uv.lock from that store), verify the commit is present (absent → a loud line, the claim is
#      dropped, the tick ends; nothing is submitted), `mirror-repos.sh --cluster <mirror selector> <checkout name>`, then
#      `gate_slurm.sh NEW --baseline --no-wait` (40 × 4 CPU, QoS high, 0 GPU → none of the `high` GPU budget); the job ids
#      land in the .job (state SUBMITTED). A submission that failed after its array was queued (`[gate] PARTIAL … array=N`)
#      is recorded PARTIAL with that id; any other failure drops the claim. When the newest <sha>.job.attempt<n> says
#      RESUBMITTED and no .job exists, the replacement's own submission failed: the tick submits attempt n+1 with that
#      record's exclusion list — a failed replacement is never a fresh attempt 1.
# --quiet mutes the routine per-tick status lines (land.sh ticks this every 30 s); every EVENT is printed and journaled.
# Exit 0 on a normal or a skipped (locked) tick, 1 when the remote could not be read or a submission failed (visible in the
# journal; the timer re-fires — Restart=no on the unit, never a restart loop on the submit path).
#
# State: STATE=${REGISTRY_LANDING_STATE:-~/.local/state/registry-landing}; baselines/<sha>.job (key=value lines: SHA ARRAY
# FINALIZE VENV RUN SUBMITTED_UTC ATTEMPT JOB_STATE LAST_SEEN EXCLUDED CONFIRM UPDATED_UTC; states CLAIMED PARTIAL SUBMITTED
# PENDING RUNNING PENDING_CONFIRM COLLECTED FAILED GAVE_UP — RESUBMITTED only in a .job.attempt<n>), baselines/.lock,
# baselines/journal.log (every line this script prints, stamped). Knobs (env): SISTEER_REPO, GATE_HOST, GATE_SHARED, GATE_RUSER,
# GATE_SLURM_SH (this directory's gate_slurm.sh), MIRROR_REPOS_SH. The login host, the shared root, the mirror selector and the
# checkout come from the site config's primary cluster block, or from the block of the cluster an exported CLUSTER_DEST names
# (site_env.sh beside this script); GATE_HOST, GATE_SHARED and the environment's own CLUSTER_* values win, and a value nothing
# supplies is refused by name (exit 1).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=${REGISTRY_LANDING_STATE:-$HOME/.local/state/registry-landing}
BASELINES=$STATE/baselines; SUMMARIES=$STATE/summaries; JOURNAL=$BASELINES/journal.log
RUSER=${GATE_RUSER:-${USER:-$(id -un)}}
GATE=${GATE_SLURM_SH:-$HERE/gate_slurm.sh}
MIRROR_REPOS=${MIRROR_REPOS_SH:-$HOME/scripts/mirror-repos.sh}
BOUND_H=4; MAX_ATTEMPTS=3      # S6's constants: give up watching 4 h after a sha's first submission; at most 3 attempts
TERMINAL='^(COMPLETED|FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY|PREEMPTED|BOOT_FAIL|DEADLINE)'   # sacct states that end a job
JOB_KEYS="SHA ARRAY FINALIZE VENV RUN SUBMITTED_UTC ATTEMPT JOB_STATE LAST_SEEN EXCLUDED CONFIRM UPDATED_UTC"
SHA_ARG=""; QUIET=0
while [ $# -gt 0 ]; do
  case "$1" in
    --sha) SHA_ARG=$2; shift ;;
    --quiet) QUIET=1 ;;
    -h|--help) awk 'NR == 1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
    *) echo "[baseline] unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
# the cluster's names from the site config (site_env.sh); GATE_HOST and GATE_SHARED win over it, and gate_slurm.sh inherits them
CLUSTER_HOST="${GATE_HOST:-${CLUSTER_HOST:-}}"; CLUSTER_SHARED="${GATE_SHARED:-${CLUSTER_SHARED:-}}"
# shellcheck source=site_env.sh
source "$HERE/site_env.sh"
site_env_load CLUSTER_HOST CLUSTER_SHARED MIRROR_REPOS_CLUSTER SITE_REPO MIRROR_REPOS_NAME || exit 1
REPO=$SITE_REPO; HOST=$CLUSTER_HOST; SHARED=$CLUSTER_SHARED
mkdir -p "$BASELINES" "$SUMMARIES" || { echo "[baseline] cannot create $STATE" >&2; exit 1; }
now_utc() { date -u +%FT%TZ; }
say() { local line="[baseline] $(now_utc) $*"; echo "$line" >> "$JOURNAL"; echo "$line"; }   # an EVENT: journaled and printed
note() { [ "$QUIET" -eq 1 ] || echo "[baseline] $(now_utc) $*"; }                             # a routine status line: muted by --quiet
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=25 "$HOST" "$@"; }
exec 9>"$BASELINES/.lock"
flock -n 9 || { note "another tick holds $BASELINES/.lock — skipping this one"; exit 0; }
rc_fail=0

# ---- the job record: one key=value line per JOB_KEYS entry, read into and written from the variables of those names ----------
read_job() {  # read_job FILE — a value is data, never `source`d
  local k v
  for k in $JOB_KEYS; do printf -v "$k" ''; done
  while IFS='=' read -r k v; do case " $JOB_KEYS " in *" $k "*) printf -v "$k" '%s' "$v" ;; esac; done < "$1"
}
save_job() {  # save_job FILE — UPDATED_UTC stamped now; written whole, then renamed into place
  local k
  { for k in $JOB_KEYS; do
      if [ "$k" = UPDATED_UTC ]; then printf 'UPDATED_UTC=%s\n' "$(now_utc)"; else printf '%s=%s\n' "$k" "${!k}"; fi
    done; } > "$1.tmp" && mv "$1.tmp" "$1"
}
set_state() { JOB_STATE=$2; LAST_SEEN=$(now_utc); save_job "$1"; }   # set_state FILE STATE

summary_terminal() {  # summary_terminal SHA → 0 when summaries/summary-<sha>.json is a terminal (PASS/FAIL, with counts) record
  local f=$SUMMARIES/summary-$1.json
  [ -s "$f" ] && python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if d.get("status") in ("PASS","FAIL") and d.get("counts") else 1)' "$f" 2>/dev/null
}

age_h() {  # age_h ISO-UTC-STAMP → whole hours since it (0 when absent or unparseable)
  local t; [ -n "$1" ] && t=$(date -u -d "$1" +%s 2>/dev/null) || { echo 0; return; }
  echo $(( ( $(date +%s) - t ) / 3600 ))
}

# ---- the queue: read once per tick (id, state, reason, node, name), filtered per job -------------------------------------------
SQ_ALL=""; SQ_RC=""
squeue_all() {  # `squeue -j <ids>` exits 1 once a job has left the queue, which would read as an outage and never reach sacct.
  [ -n "$SQ_RC" ] && return   # `|`-separated: a pending job's node list (%N) is EMPTY and its reason (%R) is parenthesized, so whitespace fields shift
  SQ_ALL=$(rsh "squeue -h -u $RUSER -o '%i|%T|%R|%N|%j'" 2>/dev/null); SQ_RC=$?
}
squeue_fresh() { SQ_RC=""; squeue_all; }
in_queue() { awk -F'|' -v a="$1" '$1 == a || index($1, a "_") == 1 {found = 1} END {exit !found}' <<<"$SQ_ALL"; }   # in_queue ID (an array's tasks count)
cancel_confirmed() {  # cancel_confirmed ID → scancel it, re-read the queue, 0 once it is gone
  rsh "scancel $1" 2>/dev/null
  squeue_fresh; [ "$SQ_RC" -eq 0 ] || return 1
  ! in_queue "$1"
}

# ---- sacct: a row is this record's only by JobID AND JobName ------------------------------------------------------------------------
tag_rows() {  # tag_rows SACCT_ROWS → `OK|row` for this record's jobs, `BAD|row` for its ids under another job's name; other rows dropped
  awk -F'|' -v a="$ARRAY" -v f="$FINALIZE" -v v="$VENV" -v run="$RUN" -v fin="gate-${SHA:0:7}-fin-$RUSER" '
    $1 == a || index($1, a "_") == 1 { print (($5 == run) ? "OK|" : "BAD|") $0; next }
    $1 == f                          { print (($5 == fin) ? "OK|" : "BAD|") $0; next }
    v != "none" && $1 == v           { print ((index($5, "gate-venv-") == 1) ? "OK|" : "BAD|") $0; next }
  ' <<<"$1"
}
confirmed() {  # confirmed FILE ROWS STATE → 0 on the second consecutive identical terminal read; the first is kept (CONFIRM) and journaled
  local d; d=$(sort <<<"$2" | sha256sum | cut -c1-12)
  [ -n "$CONFIRM" ] && [ "$CONFIRM" = "$d" ] && return 0
  CONFIRM=$d; set_state "$1" "$3"
  say "TERMINAL READ 1 of 2 for baseline $SHA (array=$ARRAY finalize=$FINALIZE${VENV:+ venv=$VENV}): $(awk -F'|' '{printf "%s=%s ", $1, $2}' <<<"$2")— the same rows on the next tick decide (digest $d)"
  return 1
}

submit() {  # submit SHA ATTEMPT EXCLUDED FIRST_UTC — the claim, the remote work, then the ids; a failure drops the claim (or records PARTIAL)
  local sha=$1 attempt=$2 excluded=$3 first=$4 f=$BASELINES/$1.job out jobs aj fj vj rj pa pv grc
  SHA=$sha ARRAY="" FINALIZE="" VENV="" RUN="" SUBMITTED_UTC=$first ATTEMPT=$attempt JOB_STATE=CLAIMED LAST_SEEN="" EXCLUDED=$excluded CONFIRM=""
  save_job "$f"
  if [ "$attempt" -eq 1 ]; then
    # fetch the commit into the base checkout's object store WITHOUT moving any branch; then prove it is there
    git -C "$REPO" fetch -q origin "$sha" 2>/dev/null || say "FETCH: git fetch origin $sha failed (network, or the remote refuses fetch-by-sha)"
    if ! git -C "$REPO" cat-file -e "$sha^{commit}" 2>/dev/null; then
      say "REFUSED: $sha is not in $REPO's object store after the fetch — nothing submitted; the gate cannot read pyproject.toml/uv.lock for a commit it does not have"
      rm -f "$f"; rc_fail=1; return 1
    fi
    "$MIRROR_REPOS" --cluster "$MIRROR_REPOS_CLUSTER" "$MIRROR_REPOS_NAME" >/dev/null 2>&1 || say "NOTE: mirror-repos.sh reported a problem for $MIRROR_REPOS_NAME (gate_slurm.sh will push the object directly)"
  fi
  out=$("$GATE" "$sha" --baseline --no-wait ${excluded:+--exclude "$excluded"} 2>&1); grc=$?
  jobs=$(grep '^\[gate\] JOBS ' <<<"$out" | tail -n1)
  aj=$(sed -n 's/.*array=\([0-9]*\).*/\1/p' <<<"$jobs"); fj=$(sed -n 's/.*finalize=\([0-9]*\).*/\1/p' <<<"$jobs")
  vj=$(sed -n 's/.*venv=\([0-9a-z]*\).*/\1/p' <<<"$jobs"); rj=$(sed -n 's/.*run=\([^ ]*\).*/\1/p' <<<"$jobs")
  if [ $grc -ne 0 ] || [ -z "$aj" ] || [ -z "$fj" ]; then
    pa=$(grep '^\[gate\] PARTIAL ' <<<"$out" | tail -n1 | sed -n 's/.*array=\([0-9]*\).*/\1/p')
    if [ -n "$pa" ]; then
      # the array was queued, its finalizer was not: record it so the next tick cancels it and confirms it left the queue before anything resubmits
      pv=$(grep '^\[gate\] PARTIAL ' <<<"$out" | tail -n1 | sed -n 's/.*venv=\([0-9a-z]*\).*/\1/p')
      ARRAY=$pa VENV=${pv:-none} RUN=gate-${sha:0:7}-$RUSER JOB_STATE=PARTIAL
      save_job "$f"
      say "SUBMIT PARTIAL for $sha (attempt $attempt, gate rc=$grc): array=$pa is queued without its finalizer — recorded PARTIAL; canceled and confirmed gone before any resubmission ($(tail -n 2 <<<"$out" | tr '\n' ' '))"
    else
      say "SUBMIT FAILED for $sha (attempt $attempt, gate rc=$grc): $(tail -n 3 <<<"$out" | tr '\n' ' ')"
      rm -f "$f"
    fi
    rc_fail=1; return 1
  fi
  ARRAY=$aj FINALIZE=$fj VENV=${vj:-none} RUN=${rj:-gate-${sha:0:7}-$RUSER} JOB_STATE=SUBMITTED
  save_job "$f"
  say "SUBMITTED baseline for $sha: array=$aj finalize=$fj venv=${vj:-none} attempt=$attempt${excluded:+ exclude=$excluded} (40 × 4 CPU, QoS high, 0 GPU)"
}

retire() {  # retire FILE ROWS — the recorded jobs are out of the queue (or can never start), on trusted rows: collect, resubmit, or fail
  local f=$1 sa=$2 fin_state venv_state venv_name nodefail_nodes task_states
  fin_state=$(awk -F'|' -v j="$FINALIZE" '$1 == j {print $2; exit}' <<<"$sa")
  venv_state=$(awk -F'|' -v j="$VENV" '$1 == j {print $2; exit}' <<<"$sa")
  venv_name=$(awk -F'|' -v j="$VENV" '$1 == j {print $5; exit}' <<<"$sa")
  nodefail_nodes=$(awk -F'|' '$2 ~ /NODE_FAIL/ {print $4}' <<<"$sa" | tr '\n' ',' | sed 's/,$//')
  task_states=$(awk -F'|' -v j="$FINALIZE" -v v="$VENV" '$1 != j && $1 != v {print $2}' <<<"$sa" | sort | uniq -c | awk '{printf "%s×%s ", $2, $1}')
  if [ "$fin_state" = COMPLETED ]; then
    if "$GATE" "$SHA" --collect-only --baseline >/dev/null 2>&1 && summary_terminal "$SHA"; then
      set_state "$f" COLLECTED
      say "TERMINAL COLLECTED baseline $SHA: array=$ARRAY tasks=[${task_states% }] summary=$SUMMARIES/summary-$SHA.json BASELINE=$(cut -c1-8 "$STATE/BASELINE" 2>/dev/null)"
    else
      set_state "$f" FAILED
      say "TERMINAL FAILED baseline $SHA: finalizer COMPLETED but the collect did not yield a terminal summary (missing shards? see $SUMMARIES/summary-$SHA.json); remove $f to retry"
    fi
    return 0
  fi
  if [ -n "$nodefail_nodes" ] && [ "$ATTEMPT" -lt "$MAX_ATTEMPTS" ]; then
    local excl; excl=$(tr ',' '\n' <<<"$EXCLUDED,$nodefail_nodes" | grep -v '^$' | sort -u | tr '\n' ',' | sed 's/,$//')
    # the ended attempt's record becomes the recovery record: attempt n, RESUBMITTED, the exclusion list its replacement uses
    EXCLUDED=$excl; set_state "$f" RESUBMITTED; mv "$f" "$f.attempt$ATTEMPT"
    say "NODE_FAIL on baseline $SHA (array=$ARRAY nodes=$nodefail_nodes; tasks=[${task_states% }] finalize=${fin_state:-absent}${venv_state:+ venv=$venv_state}) — resubmitting attempt $((ATTEMPT+1)) with --exclude $excl"
    submit "$SHA" "$((ATTEMPT+1))" "$excl" "$SUBMITTED_UTC"
    return 0
  fi
  local tail_fin="" tail_task="" tail_venv="" first_failed
  [ -n "$fin_state" ] && tail_fin=$(rsh "tail -n 15 $SHARED/exp/logs/gate-${SHA:0:7}-fin-${RUSER}_$FINALIZE.out 2>/dev/null" 2>/dev/null)
  first_failed=$(awk -F'|' -v j="$FINALIZE" -v v="$VENV" '$1 != j && $1 != v && $2 != "COMPLETED" {print $1; exit}' <<<"$sa")
  [ -n "$first_failed" ] && tail_task=$(rsh "tail -n 15 $SHARED/exp/logs/${RUN}_${first_failed%%_*}_${first_failed##*_}.out 2>/dev/null" 2>/dev/null)
  [ -n "$venv_state" ] && [ "$venv_state" != COMPLETED ] && [ -n "$venv_name" ] && tail_venv=$(rsh "tail -n 15 $SHARED/exp/logs/${venv_name}_$VENV.out 2>/dev/null" 2>/dev/null)
  set_state "$f" FAILED
  say "TERMINAL FAILED baseline $SHA: array=$ARRAY tasks=[${task_states% }] finalize=${fin_state:-absent}${venv_state:+ venv=$venv_state}${nodefail_nodes:+ node_fail=$nodefail_nodes} attempt=$ATTEMPT/$MAX_ATTEMPTS; remove $f to retry"
  [ -n "$tail_fin" ] && say "  finalizer .out tail: $(tr '\n' '|' <<<"$tail_fin")"
  [ -n "$tail_task" ] && say "  task $first_failed .out tail: $(tr '\n' '|' <<<"$tail_task")"
  [ -n "$tail_venv" ] && say "  venv job $VENV .out tail: $(tr '\n' '|' <<<"$tail_venv")"
}

give_up() {  # give_up FILE WHERE — the 4-h bound, per sha
  set_state "$1" GAVE_UP
  say "GAVE_UP watching baseline $SHA: array=$ARRAY finalize=$FINALIZE $2 $BOUND_H h after the first submission; the jobs keep their walltimes; remove $1 to try again"
}

watch() {  # watch FILE — one poll of a recorded baseline job; journal on state change; retire at a trusted terminal
  # shellcheck disable=SC2086
  local f=$1 $JOB_KEYS
  read_job "$f"
  case "$JOB_STATE" in COLLECTED|FAILED|GAVE_UP|RESUBMITTED) return 0 ;; esac
  squeue_all
  if [ "$SQ_RC" -ne 0 ]; then say "UNREACHABLE: squeue on $HOST failed for $SHA (array=${ARRAY:-?}, ssh rc=$SQ_RC); will retry next tick"; return 0; fi
  local run=gate-${SHA:0:7}-$RUSER fin_name=gate-${SHA:0:7}-fin-$RUSER
  if [ "$JOB_STATE" = CLAIMED ]; then
    # this tick holds the lock, so the tick that wrote the claim is gone without recording a submission — but it may have queued
    # jobs under the sha's names before it died: the queue says which
    local la lf
    la=$(awk -F'|' -v n="$run" '$5 == n {sub(/_.*/, "", $1); print $1; exit}' <<<"$SQ_ALL")
    lf=$(awk -F'|' -v n="$fin_name" '$5 == n {print $1; exit}' <<<"$SQ_ALL")
    if [ -n "$la" ] && [ -n "$lf" ]; then
      ARRAY=$la FINALIZE=$lf VENV=none RUN=$run JOB_STATE=SUBMITTED; save_job "$f"
      say "ORPHAN CLAIM on baseline $SHA (attempt $ATTEMPT, claimed $UPDATED_UTC): the claiming tick died after queuing array=$la finalize=$lf — adopted, watched from here"
      # and watched right now: the normal flow below reports its queue state
    elif [ -n "$la" ]; then
      ARRAY=$la RUN=$run; set_state "$f" PARTIAL
      say "ORPHAN CLAIM on baseline $SHA (attempt $ATTEMPT, claimed $UPDATED_UTC): the claiming tick died after queuing array=$la without its finalizer — recorded PARTIAL; canceled and confirmed gone before any resubmission"
      return 0
    else
      # nothing live under the sha's names: a restart may have spanned the whole run — the accounting and the results dir are asked
      # before the claim is dropped, else the same 40-task array is submitted again
      local ha hf
      sa=$(rsh "sacct -X -n -P -S now-24hours -o JobID,State,ExitCode,NodeList,JobName --name=$run,$fin_name" 2>/dev/null)
      ha=$(awk -F'|' -v n="$run" '$5 == n {sub(/_.*/, "", $1); print $1}' <<<"$sa" | sort -n | tail -n 1)
      hf=$(awk -F'|' -v n="$fin_name" '$5 == n {print $1}' <<<"$sa" | sort -n | tail -n 1)
      if [ -n "$ha" ] && [ -n "$hf" ]; then
        ARRAY=$ha FINALIZE=$hf VENV=none RUN=$run JOB_STATE=SUBMITTED; save_job "$f"
        say "ORPHAN CLAIM on baseline $SHA (attempt $ATTEMPT, claimed $UPDATED_UTC): the queue has nothing as $run but the accounting shows array=$ha finalize=$hf under those names — adopted as an ended attempt; its rows decide below under the two-read rule"
        # and decided right now: the normal flow below finds them out of the queue and reads sacct by id
      elif rsh "test -s $SHARED/gates/$SHA/summary.json" 2>/dev/null; then
        if "$GATE" "$SHA" --collect-only --baseline >/dev/null 2>&1 && summary_terminal "$SHA"; then
          ARRAY=${ha:-unknown} FINALIZE=unknown RUN=$run; set_state "$f" COLLECTED
          say "ORPHAN CLAIM on baseline $SHA (attempt $ATTEMPT, claimed $UPDATED_UTC): no job in the queue or the accounting, but $SHARED/gates/$SHA/summary.json is a terminal summary — collected; BASELINE=$(cut -c1-8 "$STATE/BASELINE" 2>/dev/null)"
        else
          say "ORPHAN CLAIM on baseline $SHA (attempt $ATTEMPT, claimed $UPDATED_UTC): $SHARED/gates/$SHA/summary.json exists but is not a terminal summary — dropping the claim; a fresh submission follows"
          rm -f "$f"
        fi
        return 0
      else
        say "ORPHAN CLAIM on baseline $SHA (attempt $ATTEMPT, claimed $UPDATED_UTC): the claiming tick died mid-submission; nothing as $run in the queue, nothing under that name in 24 h of accounting, no summary on the cluster — dropping the claim"
        rm -f "$f"
        return 0
      fi
    fi
  fi
  if [ "$JOB_STATE" = PARTIAL ]; then
    if cancel_confirmed "$ARRAY"; then
      say "PARTIAL baseline $SHA: array=$ARRAY canceled and gone from the queue — record dropped; a fresh submission follows"
      rm -f "$f"
    else
      say "PARTIAL baseline $SHA: array=$ARRAY still in the queue after scancel (or the queue is unreadable) — nothing resubmitted for this sha until it is gone"
    fi
    return 0
  fi
  local sq sa tagged rows bad
  sq=$(awk -F'|' -v a="$ARRAY" -v f="$FINALIZE" -v v="$VENV" '$1 == a || $1 == f || $1 == v || index($1, a "_") == 1' <<<"$SQ_ALL")
  if [ -n "$sq" ]; then
    local arr new reason node
    if [ "$JOB_STATE" = PENDING_CONFIRM ]; then
      say "REAPPEARED in the queue: baseline $SHA array=$ARRAY finalize=$FINALIZE (a false absence; the terminal read is forgotten)"
      CONFIRM=""; JOB_STATE=SUBMITTED
    fi
    arr=$(awk -F'|' -v a="$ARRAY" '$1 == a || index($1, a "_") == 1' <<<"$sq")
    reason=$(awk -F'|' '{print $3; exit}' <<<"$arr")
    if [ "$VENV" != none ] && [[ "$reason" == *Dependency* ]] && ! in_queue "$VENV"; then
      # the array still waits on a venv job that has left the queue: sacct (trusted rows, two consecutive reads) says whether it built or failed
      sa=$(rsh "sacct -X -n -P -o JobID,State,ExitCode,NodeList,JobName -j $VENV" 2>/dev/null)
      rows=$(tag_rows "$sa" | sed -n 's/^OK|//p')
      local vstate; vstate=$(awk -F'|' -v j="$VENV" '$1 == j {print $2; exit}' <<<"$rows")
      if [ -n "$vstate" ] && [[ "$vstate" =~ $TERMINAL ]] && [ "$vstate" != COMPLETED ]; then
        if confirmed "$f" "$rows" "$JOB_STATE"; then
          say "VENV FAILED for baseline $SHA: job $VENV ended $vstate while array=$ARRAY finalize=$FINALIZE wait on it — canceling both (they could never start)"
          rsh "scancel $ARRAY $FINALIZE" 2>/dev/null || say "  scancel $ARRAY $FINALIZE on $HOST failed; they stay queued (reason $reason)"
          retire "$f" "$rows"
        fi
        return 0
      fi
      [ -n "$CONFIRM" ] && { CONFIRM=""; save_job "$f"; }
    fi
    if awk -F'|' '$2 == "RUNNING" {found = 1} END {exit !found}' <<<"$arr"; then new=RUNNING; node=$(awk -F'|' '$2 == "RUNNING" {print $4; exit}' <<<"$arr")
    else new=PENDING; fi
    if [ "$new" != "$JOB_STATE" ]; then
      if [ "$new" = RUNNING ]; then say "STARTED baseline $SHA: array=$ARRAY node=${node:-?} (attempt $ATTEMPT)"
      else say "PENDING baseline $SHA: array=$ARRAY reason=${reason:-?} (attempt $ATTEMPT)"; fi
      set_state "$f" "$new"
    fi
    [ "$(age_h "$SUBMITTED_UTC")" -ge "$BOUND_H" ] && give_up "$f" "still in the queue (state $new)"
    return 0
  fi
  # gone from the queue: sacct decides, on rows that are ours by id AND name, terminal, and read twice the same
  local ids="$ARRAY,$FINALIZE"; [ "$VENV" != none ] && ids="$ids,$VENV"
  sa=$(rsh "sacct -X -n -P -o JobID,State,ExitCode,NodeList,JobName -j $ids" 2>/dev/null)
  tagged=$(tag_rows "$sa"); rows=$(sed -n 's/^OK|//p' <<<"$tagged"); bad=$(sed -n 's/^BAD|//p' <<<"$tagged")
  if [ -n "$bad" ]; then
    CONFIRM=""; set_state "$f" PENDING_CONFIRM
    say "MISMATCH: sacct answered baseline $SHA's job id with another job's name — not trusted (this cluster's sacct can answer for another job on a fresh id): $(tr '\n' ';' <<<"$bad")— PENDING_CONFIRM"
    return 0
  fi
  if [ -z "$rows" ]; then
    if [ "$(age_h "$SUBMITTED_UTC")" -ge "$BOUND_H" ]; then give_up "$f" "in neither squeue nor sacct"
    else say "UNKNOWN: $SHA array=$ARRAY finalize=$FINALIZE in neither squeue nor sacct yet; will retry next tick"; fi
    return 0
  fi
  local nonterm; nonterm=$(awk -F'|' -v t="$TERMINAL" '$2 !~ t' <<<"$rows")
  if [ -n "$nonterm" ] || ! grep -q "^$FINALIZE|" <<<"$rows"; then
    CONFIRM=""; set_state "$f" PENDING_CONFIRM
    if [ "$(age_h "$SUBMITTED_UTC")" -ge "$BOUND_H" ]; then give_up "$f" "out of the queue but not terminal in sacct"
    else say "GONE from the queue but not terminal in sacct yet: baseline $SHA array=$ARRAY finalize=$FINALIZE ($(awk -F'|' '{printf "%s=%s ", $1, $2}' <<<"$rows")) — PENDING_CONFIRM"; fi
    return 0
  fi
  confirmed "$f" "$rows" PENDING_CONFIRM && retire "$f" "$rows"
}

# 1. what main is
if [ -n "$SHA_ARG" ]; then NEW=$SHA_ARG
else
  NEW=$(git -C "$REPO" ls-remote -q origin refs/heads/main 2>/dev/null | cut -f1)
  [ -n "$NEW" ] || { say "ERROR: git ls-remote origin refs/heads/main returned nothing (remote unreachable?)"; exit 1; }
fi
[[ "$NEW" =~ ^[0-9a-f]{40}$ ]] || { say "ERROR: main resolved to '$NEW', not a 40-hex sha"; exit 1; }

# 2. watch every recorded job
for f in "$BASELINES"/*.job; do
  [ -f "$f" ] || continue
  watch "$f"
done

# 3. submit for NEW when it has neither a terminal summary nor a recorded job
if summary_terminal "$NEW"; then
  note "main=$NEW: baseline summary present ($SUMMARIES/summary-$NEW.json; BASELINE=$(cut -c1-8 "$STATE/BASELINE" 2>/dev/null))"
elif [ -f "$BASELINES/$NEW.job" ]; then
  read_job "$BASELINES/$NEW.job"
  case "$JOB_STATE" in
    FAILED|GAVE_UP) say "main=$NEW: baseline attempt is $JOB_STATE (array=$ARRAY); not resubmitting — remove $BASELINES/$NEW.job to try again" ;;
    *) note "main=$NEW: baseline job $JOB_STATE (array=$ARRAY finalize=$FINALIZE attempt=$ATTEMPT, submitted $SUBMITTED_UTC)" ;;
  esac
else
  # no .job: a first submission — unless the newest recovery record says a replacement was owed and never landed
  REC=$(ls -v "$BASELINES/$NEW.job.attempt"* 2>/dev/null | tail -n 1)
  if [ -n "$REC" ] && read_job "$REC" && [ "$JOB_STATE" = RESUBMITTED ]; then
    say "main=$NEW: attempt $ATTEMPT ended NODE_FAIL and its replacement was never recorded — submitting attempt $((ATTEMPT+1)) with --exclude $EXCLUDED, never a fresh attempt 1"
    submit "$NEW" "$((ATTEMPT+1))" "$EXCLUDED" "$SUBMITTED_UTC"
  else
    say "main moved to $NEW: no baseline summary and no job recorded — submitting"
    submit "$NEW" 1 "" "$(now_utc)"
  fi
fi
exit $rc_fail
