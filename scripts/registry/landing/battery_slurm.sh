#!/bin/bash
# battery_slurm.sh — a TARGETED pytest battery on the cluster A Slurm cluster: the test files a change touches (found by grep,
# never a fixed list) run from an exported tree as one CPU array of DISJOINT per-task file lists with unique record paths,
# collected into one verdict only when every task's record is present and is the planned one. The per-commit battery of the
# registry's Slurm-only test runs happen here; the grading host runs no pytest for it.
#
#   battery_slurm.sh <branch-or-sha> --files F [F...] [--label L] [--tasks K] [--out DIR] [--baseline-sha SHA]
#                    [--max-replans N] [--no-wait]
#   battery_slurm.sh <branch-or-sha> --collect-only --label L [--out DIR] [--max-replans N] [--no-wait]
#
# What it does, from the grading host (light orchestration only; nothing heavy runs here; nothing touches the live registry):
#   1. resolves the ref to a 40-hex sha (a branch resolves through the base checkout's refs — a worktree branch is one of
#      them), checks every --files entry exists AT THAT SHA (git cat-file; an empty or duplicated list refuses), and makes
#      sure the sha is on the cluster's clone $CLUSTER_REPO_MIRROR (mirror-repos.sh, else a direct push of the object
#      under refs/heads/battery/<sha7>);
#   2. ships battery.sbatch + battery_collect.py to $SHARED/gates/battery/bin and, on the login node (git only), exports the
#      tree as a standalone shallow checkout at $SHARED/gates/trees/<sha> (tree_export.sh: exact HEAD, clean); the tasks run
#      THAT tree's own gate_task.py, so a commit that changes the gate code is tested under its own gate code;
#   3. reads the duration table: the baseline gate's per-file seconds ($STATE/BASELINE's shard JUnit reports, computed once
#      on the login node by `battery_collect.py durations` and cached at $STATE/battery/durations/<sha>.json) overlaid with
#      the seconds earlier batteries observed ($STATE/battery/observed.json); then PLANS (battery_collect.py plan): K disjoint
#      lists by predicted duration (LPT; unknown files take the median; <= 8 min of predicted tests per task; default
#      K = ceil(total / 6 min); --tasks forces K), pytest's bound and the array's walltime -> $OUT/<label>.plan.json, shipped
#      to a submitter-unique partial path beside the records and moved to $SHARED/gates/<sha>/reruns/<label>.plan.json only
#      under the attempt's reservation (step 4), so no refused submitter can replace the plan the tasks read;
#   4. takes the attempt's RESERVATION on the cluster (below) and submits ONE sbatch array (battery.sbatch, -J
#      battery-<sha7>-<label>-<user>, --array=0-(K-1), 4 CPUs / 16G per task, QoS high, partition general, --time = the plan's
#      walltime) behind a venv-build job (the tree's gate_venv.sbatch) when the lock-exact shared venv
#      $SHARED/envs/<checkout name>-gate-<hash12(pyproject.toml, uv.lock)> is not ready yet; a label that already has records under
#      <sha>/reruns/ is REFUSED (a label is used once per sha: fresh records, never stale ones); each job's log is named with --output
#      under $SHARED/exp/logs at the submission (a log path written into a job script could not follow the site's shared root).
#   5. watches the venv job (if any) and then the array in the foreground (watch-job.sh: STARTED, then DONE/FAILED; the
#      driver returns only at the terminal state), records the job ids in $STATE/jobs.log and $OUT/<label>.jobs;
#   6. copies <sha>/reruns/<label>/ back to $OUT/<label>.records/ and COLLECTS (battery_collect.py collect): exactly K records
#      at t0..t(K-1), each with the tree's sha, its index/count, the plan's list, the ARRAY that wrote it (its slurm_array_job_id
#      must be this attempt's recorded array, its slurm_array_task_id its index) and a terminal status with tests; anything
#      else is `battery incomplete: …` (exit 4, no verdict). Verdict + merged JUnit + every task's log ->
#      $OUT/<label>.{verdict.json,junit.xml,log}; the observed per-file seconds update $STATE/battery/observed.json.
#      A task that ran out of its walltime (a TIMEOUT record, or no record with sacct TIMEOUT) re-plans the whole list with
#      K+1 under <label>-r2, then -r3 … (at most --max-replans times, default 1, saved in <label>.jobs at submission).
#
# ATTEMPTS ARE IMMUTABLE, THE CHAIN IS WALKED FROM ITS ROOT. Every attempt A (the label, then <label>-r2, …) owns
# $OUT/A.{plan.json,jobs,records/,sacct.json,verdict.json,junit.xml,log}, written once and never rewritten by a later attempt:
# a re-plan's A.jobs carries `REPLAN_OF=<label>` and `ATTEMPT=<n>`, which is how the chain is found again. After a re-plan the
# ROOT's <label>.verdict.json is the chain's verdict (the final attempt's result, `final_label`/`final_plan` naming where it
# came from, `attempts` with every attempt's outcome) and <label>.{junit.xml,log} are the final attempt's (the first attempt
# reached no verdict, so it never had them); <label>.{plan.json,jobs,records/} stay the FIRST attempt's. Collection always
# verifies an attempt's records against ITS OWN plan and jobs, never a sibling's.
#
# ONE DRIVER PER CHAIN, ONE SUBMITTER PER ATTEMPT. Two drivers collecting the same asynchronous chain could
# both find <label>-r2 unsubmitted and both submit it: two arrays writing the same task paths, and mixed records. So:
#   (i)  the driver runs under `flock -n -o` on $STATE/battery/locks/<sha>.<root label>.lock for its WHOLE run (collection and any
#        submission): flock(1) itself holds the lock and closes its fd before the script starts, so no child that outlives the run
#        (an ssh ControlMaster, a watcher) can keep it; a second driver on the same (sha, root label) refuses at once (exit 2),
#        naming the holder (host:pid, since, mode) written into the lock file;
#   (ii) on the cluster, right before an attempt's sbatch, an ATOMIC reservation is taken: mkdir of
#        $SHARED/gates/<sha>/reservations/<attempt label>/ (atomic on VAST as on local disk), then its `holder` file (label, sha,
#        host:pid, time, run name, K, walltime). A second submitter — from any host or state dir — finds the directory and REFUSES
#        by name, never submits. The reservation is FINALIZED by its `array` file (the array job id, written the moment sbatch
#        returns); the local <attempt>.jobs records the same ARRAY. A reservation with no `array` is reported as held (the holder,
#        its age, alive/gone when the holder is this host) and, past BATTERY_RESERVATION_STALE_S (default 900 s), as STALE — never
#        silently stolen: the operator confirms nothing is queued under the attempt's run name and removes it by hand. A finalized
#        reservation whose array this state dir does not know is reported with its array id (record it in <attempt>.jobs by hand,
#        never resubmit). An sbatch that FAILS under a reservation leaves it (with a `failed` note): a client-side failure does not
#        prove the controller took nothing, so the next submitter sees it and the operator checks the queue.
#  (iii) every task record names the array that wrote it (gate_task.py: slurm_array_job_id and slurm_array_task_id from Slurm's
#        environment) and collection REFUSES — `battery incomplete`, kind identity, never a PASS — a record whose
#        slurm_array_job_id is not the attempt's recorded ARRAY (or whose slurm_array_task_id is not its index), so records of two
#        arrays can never mix into one verdict.
#   (iv) A SUBMISSION THE CLUSTER TOOK BUT THE GRADING HOST LOST IS RECONCILED, NEVER RESUBMITTED. If the transport
#        drops or the driver dies after sbatch returned (the reservation finalized, the plan in place, the array running) but before
#        <attempt>.plan.json and <attempt>.jobs are written here, this state dir holds only <attempt>.plan.json.partial. Under the
#        chain lock, a finalized reservation met by a submission (HELD with an array) or by --collect-only of a label with that
#        .partial trace is READ BACK: the remote <attempt>.plan.json (verified against the reservation's PLAN_SHA256 and against the
#        attempt's label and sha) becomes <attempt>.plan.json, the reservation's holder + array files become <attempt>.jobs (ARRAY,
#        VENV_JOB, SUBMITTED_UTC, SUBMITTER, RESERVATION, the chain lines REPLAN_OF/ATTEMPT or MAX_REPLANS, RECONCILED_UTC), both
#        written atomically, and the run goes on to watch and collect that array — no sbatch. A plan that is missing or hashes
#        differently, or a reservation naming another sha or label, is refused by name and nothing is written or submitted.
#        A PLAIN SUBMISSION inherits a finalized label only for the SAME REQUEST: the finalized plan's requested
#        files (as a set), task flags and forced task count must equal this invocation's, else `label L is finalized for a
#        different request` refuses by name — an earlier array that tested one file never certifies a request that adds another.
#        (The duration table only shapes the allocation and the walltime, never what runs, so it is not part of the request.)
#        --collect-only carries no request and always reconciles the original. The local record is written as two back-to-back
#        renames (the staged .jobs first, the plan into place, then .jobs); a cut between them leaves a FINAL plan with no .jobs,
#        which is the same recoverable trace as the .partial: its hash must equal the reservation's PLAN_SHA256.
#
# --no-wait stops after step 4 (the records come later); --collect-only resumes a label whose $OUT/<label>.jobs exists: it
# restores the request from what the submission saved (the file list, branch and task flags from <label>.plan.json, the
# re-plan budget from <label>.jobs unless --max-replans is given again), walks the chain from the root collecting each
# attempt, and when the last one ran out of its walltime submits the SAME K+1 re-plan a waiting run would have (then waits,
# unless --no-wait). A re-plan's label given to --collect-only is redirected to its root. Exit: 0 PASS, 1 FAIL,
# 4 INCOMPLETE, 2 a refusal of this driver (named on stderr).
#
# State (never /tmp): STATE=${REGISTRY_LANDING_STATE:-~/.local/state/registry-landing}/{BASELINE, jobs.log, battery/}; OUT defaults
# to $STATE/battery (--out DIR, or BATTERY_OUT). Knobs (env): GATE_HOST (the login host), GATE_RUSER ($USER), GATE_SHARED
# (the shared root), SISTEER_REPO (the checkout), MIRROR_REPOS_SH (~/scripts/mirror-repos.sh), MIRROR_HISTORY_SH
# (this directory's mirror_history.sh: the archived branches and tags reach the mirror before every export), WATCH_JOB_SH
# (~/scripts/watch-job.sh), BATTERY_TASK_FLAGS (gate_task flags per task; default `--local-venv --hide-slurm`, recorded in
# the plan), BATTERY_PY (python3 for the plan and the collection here), BATTERY_REMOTE_PY (python3 on the login node),
# BATTERY_RESERVATION_STALE_S (900: a cluster reservation with no array after this many seconds is reported as stale). The login
# host, the shared root, the mirror selector, the checkout and its clone on the cluster come from the site config's primary cluster
# block, or from the block of the cluster an exported CLUSTER_DEST names (site_env.sh beside this script); GATE_HOST, GATE_SHARED and
# the environment's own CLUSTER_* values win, and a value nothing supplies is refused by name.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SELF=$HERE/$(basename "${BASH_SOURCE[0]}")
ORIG_ARGS=("$@")          # re-passed unchanged to the run under the chain lock
HOLDER="$(hostname):$$"   # who holds a lock or a reservation: this host and this process
STALE_S=${BATTERY_RESERVATION_STALE_S:-900}
STATE=${REGISTRY_LANDING_STATE:-$HOME/.local/state/registry-landing}
JOBS_LOG=$STATE/jobs.log; BASELINE_FILE=$STATE/BASELINE
RUSER=${GATE_RUSER:-${USER:-$(id -un)}}
MIRROR_REPOS=${MIRROR_REPOS_SH:-$HOME/scripts/mirror-repos.sh}
MIRROR_HISTORY=${MIRROR_HISTORY_SH:-$HERE/mirror_history.sh}
WATCH=${WATCH_JOB_SH:-$HOME/scripts/watch-job.sh}
PY=${BATTERY_PY:-python3}
RPY=${BATTERY_REMOTE_PY:-python3}
COLLECT=$HERE/battery_collect.py
TASK_FLAGS=${BATTERY_TASK_FLAGS---local-venv --hide-slurm}
OUT=${BATTERY_OUT:-$STATE/battery}
REF=""; LABEL=""; TASKS=""; BASE_SHA=""; WAIT=1; COLLECT_ONLY=0; MAX_REPLANS=1; MAX_REPLANS_SET=0; FILES=()
die() { echo "[battery] ERROR: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --files) shift; while [ $# -gt 0 ] && [[ "$1" != --* ]]; do FILES+=("$1"); shift; done; continue ;;
    --label) LABEL=$2; shift ;;
    --tasks) TASKS=$2; shift ;;
    --out) OUT=$2; shift ;;
    --baseline-sha) BASE_SHA=$2; shift ;;
    --max-replans) MAX_REPLANS=$2; MAX_REPLANS_SET=1; shift ;;
    --no-wait) WAIT=0 ;;
    --collect-only) COLLECT_ONLY=1 ;;
    -h|--help) awk 'NR == 1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
    -*) die "unknown flag $1" ;;
    *) [ -z "$REF" ] || die "one ref only (got '$REF' and '$1')"; REF=$1 ;;
  esac
  shift
done
[ -n "$REF" ] || die "usage: battery_slurm.sh <branch-or-sha> --files F [F...] [--label L] [--tasks K] [--out DIR] [--baseline-sha SHA] [--max-replans N] [--no-wait] | <branch-or-sha> --collect-only --label L"
# the cluster's names from the site config (site_env.sh); GATE_HOST and GATE_SHARED win over it
CLUSTER_HOST="${GATE_HOST:-${CLUSTER_HOST:-}}"; CLUSTER_SHARED="${GATE_SHARED:-${CLUSTER_SHARED:-}}"
# shellcheck source=site_env.sh
source "$HERE/site_env.sh"
site_env_load CLUSTER_HOST CLUSTER_SHARED MIRROR_REPOS_CLUSTER SITE_REPO MIRROR_REPOS_NAME CLUSTER_REPO_MIRROR || exit 2
HOST=$CLUSTER_HOST; SHARED=$CLUSTER_SHARED; MIRROR=$CLUSTER_REPO_MIRROR; REPO=$SITE_REPO
CL_BBIN=$SHARED/gates/battery/bin; CL_TREES=$SHARED/gates/trees; CL_RESULTS=$SHARED/gates
CL_LOGS=$SHARED/exp/logs     # every job's log, named at submission (<job name>_<job id>.out; an array task's carries its index)
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=25 "$HOST" "$@"; }
mkdir -p "$OUT" "$STATE/battery/durations" || die "cannot create $OUT / $STATE/battery"
[ -s "$COLLECT" ] || die "no battery_collect.py beside this script ($COLLECT)"

json_get() { "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); v=d
for k in sys.argv[2].split("."): v=v.get(k) if isinstance(v,dict) else None
print("" if v is None else v)' "$1" "$2"; }

# --collect-only: the request from the chain root's saved plan and submission record — the file list, the branch and the task flags
# from the plan, the re-plan budget from the record (a --max-replans given on this command line wins)
restore_request() {
  [ -s "$OUT/$LABEL.plan.json" ] || die "no plan at $OUT/$LABEL.plan.json"
  local JSHA PSHA SAVED_MR
  JSHA=$(sed -n 's/^SHA=//p' "$OUT/$LABEL.jobs" | head -n1); PSHA=$(json_get "$OUT/$LABEL.plan.json" sha)
  [ "$JSHA" = "$SHA" ] && [ "$PSHA" = "$SHA" ] || die "label $LABEL was submitted for sha ${JSHA:-?} (plan ${PSHA:-?}), not $SHA"
  mapfile -t FILES < <("$PY" -c 'import json,sys; print("\n".join(json.load(open(sys.argv[1])).get("requested_files") or []))' "$OUT/$LABEL.plan.json")
  [ ${#FILES[@]} -gt 0 ] && [ -n "${FILES[0]}" ] || die "the saved plan $OUT/$LABEL.plan.json names no files"
  [ -n "$BRANCH" ] || BRANCH=$(json_get "$OUT/$LABEL.plan.json" branch)
  TASK_FLAGS=$("$PY" -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1])).get("task_flags") or []))' "$OUT/$LABEL.plan.json")
  SAVED_MR=$(sed -n 's/^MAX_REPLANS=//p' "$OUT/$LABEL.jobs" | head -n1)
  if [ "$MAX_REPLANS_SET" -eq 0 ] && [ -n "$SAVED_MR" ]; then MAX_REPLANS=$SAVED_MR; fi
  [[ "$MAX_REPLANS" =~ ^[0-9]+$ ]] || die "the saved re-plan budget is not an integer ('$MAX_REPLANS')"
}

# a submission the cluster TOOK but this state dir never recorded (the transport dropped, or the driver died, after sbatch returned
# and before <L>.plan.json / <L>.jobs were written): the reservation is FINALIZED (its `array` file exists), the array runs or ran,
# and the grading host holds at most <L>.plan.json.partial. Restore the attempt from the cluster instead of refusing or resubmitting:
# the remote <L>.plan.json — verified against the reservation's PLAN_SHA256 and against this attempt's label and sha — becomes
# <L>.plan.json and the reservation's holder + array files become <L>.jobs (ARRAY, VENV_JOB, SUBMITTED_UTC, SUBMITTER, RESERVATION,
# the chain lines, RECONCILED_UTC), both written atomically; the caller then watches/collects that array, NEVER a new sbatch.
# Refuses by name (exit 2, nothing written) when the plan is missing, hashes differently, or the reservation names another sha or
# label. Returns 1 when the cluster holds no finalized reservation for L (nothing to reconcile). Called under the chain lock.
reconcile_lost() {   # reconcile_lost LABEL [REQUEST_PLAN]  (REQUEST_PLAN: a plain submission's own plan — the finalized one must be the same request)
  local L=$1 REQ=${2:--} RESV=$CL_RESULTS/$SHA/reservations/$1 PLAN_REMOTE=$CL_RESULTS/$SHA/reruns/$1.plan.json TMP=$OUT/$1.reconcile.tmp WHY prc
  rm -rf "$TMP"; mkdir -p "$TMP/resv" || die "cannot create $TMP"
  { rsync -a "$HOST:$RESV/" "$TMP/resv/" 2>/dev/null && [ -s "$TMP/resv/array" ]; } || { rm -rf "$TMP"; return 1; }
  rsync -a "$HOST:$PLAN_REMOTE" "$TMP/plan.json" 2>/dev/null || true
  WHY=$("$PY" - "$TMP" "$L" "$SHA" "$RESV" "$HOST:$PLAN_REMOTE" "$MAX_REPLANS" "$RUSER" "$S7" "$REQ" "$OUT/$L.plan.json" 2>&1 <<'PYR'
import hashlib, json, os, sys, time
tmp, label, sha, resv, plan_remote, max_replans, ruser, s7, req_path, local_final = sys.argv[1:11]
def kv(p):
    try:
        return dict(l.split("=", 1) for l in open(p).read().splitlines() if "=" in l)
    except OSError:
        return {}
holder, array = kv(f"{tmp}/resv/holder"), kv(f"{tmp}/resv/array")
aj = (array.get("ARRAY") or "").strip()
if not aj.isdigit():
    sys.exit(f"the reservation's array file records no numeric ARRAY ({aj!r})")
if holder.get("SHA", sha) != sha:
    sys.exit(f"the reservation names sha {holder.get('SHA')}, not {sha}")
if holder.get("LABEL", label) != label:
    sys.exit(f"the reservation names label {holder.get('LABEL')!r}, not {label!r}")
if not os.path.isfile(f"{tmp}/plan.json"):
    sys.exit(f"no plan at {plan_remote}")
raw = open(f"{tmp}/plan.json", "rb").read()
want, got = (holder.get("PLAN_SHA256") or "").strip(), hashlib.sha256(raw).hexdigest()
if not want:
    sys.exit("the reservation recorded no PLAN_SHA256: the remote plan cannot be verified")
if got != want:
    sys.exit(f"the remote plan hashes to {got}, the reservation recorded {want}")
try:
    plan = json.loads(raw)
except ValueError as e:
    sys.exit(f"the remote plan is not JSON ({e})")
if plan.get("label") != label or plan.get("sha") != sha:
    sys.exit(f"the remote plan is label {plan.get('label')!r} for sha {plan.get('sha')!r}, not {label!r} for {sha}")
if holder.get("TASKS") and str(plan.get("count")) != holder["TASKS"].strip():
    sys.exit(f"the remote plan has {plan.get('count')} tasks, the reservation recorded {holder['TASKS']}")
# a plain submission's request must be the finalized one: the files as a set, the task flags, the forced task count (the duration
# table shapes only the allocation and the walltime, so it is not compared); a difference is exit 3, never a restore
if req_path != "-":
    req = json.load(open(req_path))
    rf, pf = sorted(set(req.get("requested_files") or [])), sorted(set(plan.get("requested_files") or []))
    diffs = []
    if rf != pf:
        diffs.append(f"{len(pf)} files vs {len(rf)} requested (only in the finalized plan: {sorted(set(pf) - set(rf))}; only in this request: {sorted(set(rf) - set(pf))})")
    if list(plan.get("task_flags") or []) != list(req.get("task_flags") or []):
        diffs.append(f"task flags {plan.get('task_flags')} vs {req.get('task_flags')} requested")
    pforced, rforced = bool(plan.get("tasks_forced")), bool(req.get("tasks_forced"))
    if pforced != rforced or (rforced and plan.get("count") != req.get("count")):
        diffs.append(f"tasks {'forced to ' + str(plan.get('count')) if pforced else 'not forced'} vs {'forced to ' + str(req.get('count')) if rforced else 'not forced'} requested")
    if diffs:
        print("; ".join(diffs))
        sys.exit(3)
# a final plan already here with no .jobs (a cut between the two local renames) must be this submission's
local_state = "none"
if os.path.isfile(local_final):
    lh = hashlib.sha256(open(local_final, "rb").read()).hexdigest()
    if lh != want:
        sys.exit(f"a final plan already at {local_final} (with no .jobs) hashes to {lh}, the reservation recorded {want}: not this submission's plan — move it aside")
    local_state = "verified"
lines = [f"SHA={sha}", f"LABEL={label}", f"TASKS={plan.get('count')}", f"WALLTIME={(holder.get('WALLTIME') or plan.get('walltime') or '?').strip()}",
         f"VENV_JOB={(array.get('VENV_JOB') or 'none').strip()}", f"ARRAY={aj}", f"RUN_NAME={(holder.get('RUN_NAME') or f'battery-{s7}-{label}-{ruser}').strip()}",
         f"SUBMITTED_UTC={(array.get('SUBMITTED_UTC') or holder.get('SINCE_UTC') or '?').strip()}", f"SUBMITTER={(holder.get('HOLDER') or '?').strip()}",
         f"RESERVATION={resv}"]
root = (holder.get("REPLAN_OF") or "").strip()
if root:
    lines += [f"REPLAN_OF={root}", f"ATTEMPT={(holder.get('ATTEMPT') or '?').strip()}"]
else:
    lines.append(f"MAX_REPLANS={(holder.get('MAX_REPLANS') or '').strip() or max_replans}")
lines.append("RECONCILED_UTC=" + time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
open(f"{tmp}/plan.json.out", "wb").write(raw)
open(f"{tmp}/jobs.out", "w").write("\n".join(lines) + "\n")
print(f"array={aj} tasks={plan.get('count')} submitter={(holder.get('HOLDER') or '?').strip()} submitted={(array.get('SUBMITTED_UTC') or '?').strip()} plan_sha256={got[:12]} local_plan={local_state}")
PYR
  ); prc=$?
  if [ "$prc" -eq 3 ]; then
    rm -rf "$TMP"
    die "label $L is finalized for a different request: $WHY; use a new label, or \`--collect-only --label $L\` to collect the original request — nothing written, nothing submitted (this request's partial plan stays as the trace --collect-only reads)"
  fi
  [ "$prc" -eq 0 ] || { rm -rf "$TMP"; die "another collector finalized $L on $HOST (reservation $RESV) but it cannot be reconciled here: $WHY — nothing submitted, nothing written; collect it where it was submitted, or repair the reservation by hand"; }
  { mv -f "$TMP/plan.json.out" "$OUT/$L.plan.json" && mv -f "$TMP/jobs.out" "$OUT/$L.jobs"; } || die "cannot write the reconciled $OUT/$L.plan.json / $OUT/$L.jobs"
  rm -f "$OUT/$L.plan.json.partial" "$OUT/$L.jobs.partial"; rm -rf "$TMP"
  echo "$(date -u +%FT%TZ) sha=$SHA branch=${BRANCH:-none} battery=$L reconciled_from=$HOST:$RESV $WHY" >> "$JOBS_LOG"
  echo "[battery] RECONCILED label=$L $WHY: the cluster's finalized reservation had no record here — $OUT/$L.plan.json and $OUT/$L.jobs restored from it, nothing submitted"
  return 0
}

# 1. resolve; the file list
if [[ "$REF" =~ ^[0-9a-f]{40}$ ]]; then SHA=$REF; BRANCH=""
else
  SHA=$(git -C "$REPO" rev-parse --verify --quiet "$REF^{commit}") || die "cannot resolve '$REF' in $REPO"
  BRANCH=$REF
fi
git -C "$REPO" cat-file -e "$SHA^{commit}" 2>/dev/null || die "$SHA is not in $REPO's object store"
S7=${SHA:0:7}
if [ "$COLLECT_ONLY" -eq 0 ]; then
  [ ${#FILES[@]} -gt 0 ] || die "--files is empty: a battery runs the files grep found for the change, never nothing"
  [ -n "$LABEL" ] || LABEL=files-$(printf '%s\n' "${FILES[@]}" | sha256sum | cut -c1-8)
else
  [ -n "$LABEL" ] || die "--collect-only needs --label L (the label that was submitted)"
  [ ${#FILES[@]} -eq 0 ] || die "--collect-only takes no --files: the file list is the saved plan's ($OUT/$LABEL.plan.json)"
  [ -z "$TASKS" ] || die "--collect-only takes no --tasks: a re-plan always takes the previous K + 1"
fi
[[ "$LABEL" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "label '$LABEL' must match [A-Za-z0-9][A-Za-z0-9._-]*"
[ -z "$TASKS" ] || [[ "$TASKS" =~ ^[1-9][0-9]*$ ]] || die "--tasks needs a positive integer"
if [ "$COLLECT_ONLY" -eq 0 ]; then
  DUPS=$(printf '%s\n' "${FILES[@]}" | sort | uniq -d)
  [ -z "$DUPS" ] || die "duplicate file(s) in --files: $(echo "$DUPS" | tr '\n' ' ')"
  MISSING=()
  for f in "${FILES[@]}"; do
    [[ "$f" != /* && "$f" != *..* ]] || die "--files entries are tree-relative paths: '$f'"
    git -C "$REPO" cat-file -e "$SHA:$f" 2>/dev/null || MISSING+=("$f")
  done
  [ ${#MISSING[@]} -eq 0 ] || die "not in the tree at $S7: ${MISSING[*]}"
else
  # --collect-only: find the chain's ROOT, then restore the request from what its submission saved (restore_request)
  REDIRECTED_FROM=""; RECONCILE_LOST=""
  if [ -s "$OUT/$LABEL.jobs" ]; then
    ROOT=$(sed -n 's/^REPLAN_OF=//p' "$OUT/$LABEL.jobs" | head -n1)
    if [ -n "$ROOT" ]; then
      REDIRECTED_FROM=$LABEL; LABEL=$ROOT
      [ -s "$OUT/$LABEL.jobs" ] || die "no $OUT/$LABEL.jobs for the chain's root $LABEL"
    fi
    restore_request
  elif [ -e "$OUT/$LABEL.plan.json.partial" ] || [ -s "$OUT/$LABEL.plan.json" ]; then
    # the trace a submission cut after its sbatch leaves (the shipped .partial, or the final plan with no .jobs when the cut fell
    # between the two local renames): the cluster may hold the finalized reservation this state dir never recorded — reconciled under
    # the chain lock below; only the chain's root is read here (the reservation's REPLAN_OF line), so the lock taken is the chain's
    RECONCILE_LOST=$LABEL
    ROOT=$(rsh "sed -n 's/^REPLAN_OF=//p' $CL_RESULTS/$SHA/reservations/$LABEL/holder 2>/dev/null" 2>/dev/null | head -n1)
    [ -z "$ROOT" ] || { REDIRECTED_FROM=$LABEL; LABEL=$ROOT; }
  else
    die "no $OUT/$LABEL.jobs: label $LABEL was not submitted from here (--out?)"
  fi
fi
[[ "$MAX_REPLANS" =~ ^[0-9]+$ ]] || die "--max-replans needs an integer (got '$MAX_REPLANS')"
PH=$(git -C "$REPO" show "$SHA:pyproject.toml" 2>/dev/null | sha256sum | cut -d' ' -f1) || die "no pyproject.toml at $SHA"
LH=$(git -C "$REPO" show "$SHA:uv.lock" 2>/dev/null | sha256sum | cut -d' ' -f1) || die "no uv.lock at $SHA"
H12=$(printf '%s\n%s\n' "$PH" "$LH" | sha256sum | cut -c1-12)
VENV=$(site_env_gate_venv "$SHARED" "$MIRROR_REPOS_NAME" "$H12")
TREE=$CL_TREES/$SHA

# ONE DRIVER PER (sha, root label): the run happens under `flock -n -o -E 75 LOCK` — flock(1) holds the lock for exactly as long as
# the re-executed script runs and closes its own fd before the exec, so no child (an ssh ControlMaster that persists, a watcher)
# can keep the lock after the run; a second driver on the same chain gets 75 from flock and refuses at once, naming the holder
# the running one wrote into the lock file. The kernel releases the lock when flock exits, so it cannot go stale.
LOCK=$STATE/battery/locks/$SHA.$LABEL.lock
if [ "${BATTERY_CHAIN_LOCK:-}" != "$LOCK" ]; then
  mkdir -p "$STATE/battery/locks" || die "cannot create $STATE/battery/locks"
  BATTERY_CHAIN_LOCK=$LOCK flock -n -o -E 75 "$LOCK" "$SELF" "${ORIG_ARGS[@]}"; lrc=$?
  if [ "$lrc" -eq 75 ]; then
    die "another collector holds $LABEL for $S7: $(tr '\n' ' ' < "$LOCK" 2>/dev/null)— one driver per (sha, label) at a time: wait for it to finish (lock $LOCK)"
  fi
  exit "$lrc"
fi
printf 'HOLDER=%s\nSINCE_UTC=%s\nMODE=%s\nLABEL=%s\nSHA=%s\n' "$HOLDER" "$(date -u +%FT%TZ)" "$([ "$COLLECT_ONLY" -eq 1 ] && echo collect || echo submit)" "$LABEL" "$SHA" > "$LOCK"
if [ -n "${RECONCILE_LOST:-}" ]; then
  # --collect-only of a label this state dir shipped a plan for but never recorded: read the finalized submission back from the
  # cluster (nothing to read -> the submission was cut before its sbatch, and there is nothing to collect)
  reconcile_lost "$RECONCILE_LOST" || die "no $OUT/$RECONCILE_LOST.jobs: label $RECONCILE_LOST was not submitted from here, and the cluster holds no finalized reservation for it (a submission cut before its sbatch leaves nothing to collect: submit it again)"
  if [ "$RECONCILE_LOST" != "$LABEL" ] && [ ! -s "$OUT/$LABEL.jobs" ]; then
    reconcile_lost "$LABEL" || die "no $OUT/$LABEL.jobs for the chain's root $LABEL, and the cluster holds no finalized reservation for it"
  fi
  restore_request
fi
[ -z "${REDIRECTED_FROM:-}" ] || echo "[battery] $REDIRECTED_FROM is a re-plan of $LABEL: a chain is collected from its root"
echo "[battery] ref=$REF sha=$SHA branch=${BRANCH:-none} label=$LABEL files=${#FILES[@]} tasks=${TASKS:-auto} max_replans=$MAX_REPLANS collect_only=$COLLECT_ONLY venv=$VENV out=$OUT state=$STATE holder=$HOLDER"

# 2. mirror, code, tree (once per invocation)
prepare_cluster() {
  if ! rsh "git -C $MIRROR cat-file -e $SHA 2>/dev/null"; then
    echo "[battery] $S7 not on the cluster mirror — running mirror-repos.sh"
    "$MIRROR_REPOS" --cluster "$MIRROR_REPOS_CLUSTER" "$MIRROR_REPOS_NAME" || echo "[battery] mirror-repos.sh reported a problem; trying a direct push of the sha"
    if ! rsh "git -C $MIRROR cat-file -e $SHA 2>/dev/null"; then
      git -C "$REPO" push -q "ssh://$HOST$MIRROR" "$SHA:refs/heads/battery/$S7" || die "direct push of $SHA to $HOST:$MIRROR failed"
      echo "[battery] pushed $S7 to the mirror as refs/heads/battery/$S7"
    fi
    rsh "git -C $MIRROR cat-file -e $SHA" || die "$SHA still not on $HOST:$MIRROR (commit and push first?)"
  fi
  # the history namespaces, before EVERY export (as the gate does): the archived branches and tags reach the mirror, verified, or
  # the battery refuses here -- its tasks read commits only the archive reaches
  HOUT=$("$MIRROR_HISTORY" "$REPO" "ssh://$HOST$MIRROR" 2>&1); hrc=$?
  echo "$HOUT" | sed 's/^/[battery] /'
  [ $hrc -eq 0 ] || die "the archived branches and tags of $REPO did not reach $HOST:$MIRROR (mirror_history.sh rc $hrc); the exported tree would lack commits the tasks read"
  rsh "mkdir -p $CL_BBIN $CL_TREES $CL_RESULTS/$SHA/reruns $CL_LOGS" || die "cannot create cluster dirs"
  rsync -a --chmod=F644 "$HERE/battery.sbatch" "$HERE/battery_collect.py" "$HOST:$CL_BBIN/" || die "rsync of the battery code failed"
  TOUT=$(rsh bash -s -- "$SHA" "$MIRROR" "$TREE" < "$HERE/tree_export.sh"); trc=$?
  echo "$TOUT" | sed 's/^/[battery:cluster] /'
  [ $trc -eq 0 ] && ! grep -q '^ERR' <<<"$TOUT" || die "tree export failed"
}

# 3. the duration table
duration_tables() {
  DURATIONS=()
  BASE=${BASE_SHA:-$(cat "$BASELINE_FILE" 2>/dev/null || true)}
  if [ -n "$BASE" ]; then
    BT=$STATE/battery/durations/$BASE.json
    if [ ! -s "$BT" ]; then
      echo "[battery] reading the baseline gate's per-file durations ($BASE shards) on the login node"
      if rsh "$RPY $CL_BBIN/battery_collect.py durations --shards-dir $CL_RESULTS/$BASE/shards" > "$BT.partial" 2>"$BT.err" && [ -s "$BT.partial" ]; then
        mv "$BT.partial" "$BT"; rm -f "$BT.err"
      else
        echo "[battery] no baseline durations for $BASE ($(tail -n1 "$BT.err" 2>/dev/null)); unknown files take the default"; rm -f "$BT.partial"
      fi
    fi
    [ -s "$BT" ] && DURATIONS+=("$BT")
  else
    echo "[battery] no $BASELINE_FILE and no --baseline-sha: planning without a baseline table"
  fi
  [ -s "$STATE/battery/observed.json" ] && DURATIONS+=("$STATE/battery/observed.json")
}

# a reservation another submitter holds and has NOT finalized: refuse by name (exit 2), classified — stale (no array after STALE_S:
# not stolen, the operator checks the queue and removes it) or in progress (wait). A finalized one is reconcile_lost's business.
refuse_held() {   # refuse_held ATTEMPT RUN_NAME "HELD label=… array= age_s=… holder=host:pid since=… path=…"
  local L=$1 RUN=$2 line=$3 age who since path alive agetxt
  age=$(sed -n 's/.* age_s=\([^ ]*\).*/\1/p' <<<"$line")
  who=$(sed -n 's/.* holder=\([^ ]*\).*/\1/p' <<<"$line"); since=$(sed -n 's/.* since=\([^ ]*\).*/\1/p' <<<"$line")
  path=$(sed -n 's/.* path=\([^ ]*\).*/\1/p' <<<"$line")
  if [ -z "$who" ]; then alive="holder unknown"
  elif [[ "$who" == "$(hostname):"* ]] && [[ "${who##*:}" =~ ^[0-9]+$ ]]; then
    if kill -0 "${who##*:}" 2>/dev/null; then alive="holder process alive on this host"; else alive="holder process GONE from this host"; fi
  else alive="holder on another host"; fi
  if [[ "$age" =~ ^[0-9]+$ ]]; then agetxt="${age}s ago"; else agetxt="age unknown"; fi
  if [[ "$age" =~ ^[0-9]+$ ]] && [ "$age" -ge "$STALE_S" ]; then
    die "another collector holds $L: a STALE reservation on $HOST by ${who:-?} since ${since:-?} (${age}s with no array recorded; $alive) — not stolen: confirm nothing is queued (ssh $HOST squeue -u $RUSER -n $RUN), remove $path by hand, then run again"
  else
    die "another collector holds $L: reservation on $HOST by ${who:-?} since ${since:-?} ($agetxt, no array recorded yet; $alive) — one submitter per attempt: wait for it to finish ($path)"
  fi
}

# the cluster side is prepared once per invocation, and only when something is submitted (a collection that re-plans included)
PREPARED=0
prepare_once() { [ "$PREPARED" -eq 1 ] || { prepare_cluster; PREPARED=1; }; duration_tables; }

# 4+5+6 for one attempt
attempt() {   # attempt LABEL submit [K] ATTEMPT_NO | attempt LABEL collect  -> ATTEMPT_RC (0 PASS, 1 FAIL, 4 INCOMPLETE), ATTEMPT_KIND
  # submit: plan, ship, sbatch, record, watch (unless --no-wait), then collect; collect: an attempt submitted earlier, read with its
  # OWN plan, jobs and records — the three are one attempt's or the collection is meaningless
  local L=$1 MODE=$2 K=${3:-} NO=${4:-1} PLAN=$OUT/$1.plan.json JOBS=$OUT/$1.jobs REC=$OUT/$1.records RUN_NAME=battery-$S7-$1-$RUSER PLAN_REMOTE=$CL_RESULTS/$SHA/reruns/$1.plan.json
  ATTEMPT_RC=2; ATTEMPT_KIND=""
  if [ "$MODE" = submit ]; then
    # a label is submitted once: the local record of a submission (and the cluster's records, checked below) refuse a second one,
    # and the plan becomes <label>.plan.json only once the array is queued, so a refused submission leaves no plan behind
    [ ! -e "$JOBS" ] || die "label $L was already submitted from here ($JOBS exists): --collect-only to finish it, or another label"
    "$PY" "$COLLECT" plan --sha "$SHA" --label "$L" ${BRANCH:+--branch "$BRANCH"} --files "${FILES[@]}" --durations "${DURATIONS[@]}" \
      ${K:+--tasks "$K"} --task-flags="$TASK_FLAGS" --out "$PLAN.partial" || die "planning failed"
    local KN WALL; KN=$(json_get "$PLAN.partial" count); WALL=$(json_get "$PLAN.partial" walltime)
    # the plan travels to a path only this submitter writes and lands at <label>.plan.json under the reservation (REMOTE below)
    local PLAN_TMP=$PLAN_REMOTE.$(hostname)-$$.partial
    rsync -a --chmod=F644 "$PLAN.partial" "$HOST:$PLAN_TMP" || die "rsync of the plan failed"
    local ROUT rrc
    # the chain lines ride along so a lost submission's .jobs can be restored from the reservation (an empty positional would vanish
    # through ssh's re-parse of the command line, hence the `-` sentinels)
    local ROOT_ARG MR_ARG; if [ "$L" = "$LABEL" ]; then ROOT_ARG=-; MR_ARG=$MAX_REPLANS; else ROOT_ARG=$LABEL; MR_ARG=-; fi
    ROUT=$(rsh bash -s -- "$SHA" "$TREE" "$VENV" "$H12" "$CL_BBIN" "$CL_RESULTS" "$L" "$KN" "$WALL" "$PLAN_REMOTE" "$PH" "$LH" "$RUN_NAME" "$HOLDER" "$PLAN_TMP" "$ROOT_ARG" "$NO" "$MR_ARG" "$CL_LOGS" <<'REMOTE'
set -uo pipefail
SHA=$1 TREE=$2 VENV=$3 H12=$4 BBIN=$5 RESULTS=$6 LABEL=$7 K=$8 WALL=$9 PLAN_REMOTE=${10} PH=${11} LH=${12} RUN_NAME=${13} HOLDER=${14} PLAN_TMP=${15} ROOT_LABEL=${16} ATTEMPT_NO=${17} MAX_RP=${18} LOGS=${19:-}
[ "$ROOT_LABEL" != - ] || ROOT_LABEL=""; [ "$MAX_RP" != - ] || MAX_RP=""
bail() { rm -f "$PLAN_TMP"; echo "ERR $*"; exit 2; }   # a refusal before the reservation: nothing of this attempt stays behind
# every job's log goes under the site's shared root, named at submission (it overrides any log directive in the job scripts)
[ -n "$LOGS" ] || bail "no log directory"
[ "$(sha256sum < "$TREE/pyproject.toml" | cut -d' ' -f1)" = "$PH" ] || bail "pyproject.toml hash differs from the grading host's"
[ "$(sha256sum < "$TREE/uv.lock" | cut -d' ' -f1)" = "$LH" ] || bail "uv.lock hash differs from the grading host's"
[ -s "$PLAN_TMP" ] || bail "no plan at $PLAN_TMP"
[ -s "$BBIN/battery.sbatch" ] || bail "no battery.sbatch at $BBIN"
[ -s "$TREE/scripts/registry/landing/gate_task.py" ] || bail "the tree at $TREE has no scripts/registry/landing/gate_task.py"
# the single-owner reservation of this attempt: mkdir is atomic here as on local disk, so exactly one submitter of <sha>/<label>
# gets it; a second finds it and reports it (HELD, exit 3) instead of submitting — with its array when it is finalized, which the
# grading host side reconciles rather than refuses; the `array` file finalizes it
RESV_ROOT=$RESULTS/$SHA/reservations; RESV=$RESV_ROOT/$LABEL
mkdir -p "$RESV_ROOT" || bail "cannot create $RESV_ROOT"
if ! mkdir "$RESV" 2>/dev/null; then
  rm -f "$PLAN_TMP"
  NOW=$(date -u +%s); SINCE=$(sed -n 's/^SINCE_UTC=//p' "$RESV/holder" 2>/dev/null | head -n1); AGE=unknown
  [ -z "$SINCE" ] || AGE=$(( NOW - $(date -u -d "$SINCE" +%s 2>/dev/null || echo "$NOW") ))
  echo "HELD label=$LABEL array=$(sed -n 's/^ARRAY=//p' "$RESV/array" 2>/dev/null | head -n1) age_s=$AGE holder=$(sed -n 's/^HOLDER=//p' "$RESV/holder" 2>/dev/null | head -n1) since=${SINCE:-unknown} path=$RESV$([ -s "$RESV/failed" ] && printf ' failed=%s' "$(tr ' \n' '__' < "$RESV/failed")")"
  exit 3
fi
# records of this label with no reservation (a submission from before reservations existed): the label stays used once per sha
[ ! -e "$RESULTS/$SHA/reruns/$LABEL" ] || { rm -rf "$RESV"; bail "label $LABEL already has records under $RESULTS/$SHA/reruns/ (a label is used once per sha: pick another, or --collect-only)"; }
# under the reservation the plan takes its final name: what the tasks read is this submission's and no later one can replace it
mv -f "$PLAN_TMP" "$PLAN_REMOTE" || { rm -rf "$RESV"; rm -f "$PLAN_TMP"; echo "ERR cannot place the plan at $PLAN_REMOTE"; exit 2; }
printf 'LABEL=%s\nSHA=%s\nHOLDER=%s\nSINCE_UTC=%s\nRUN_NAME=%s\nTASKS=%s\nWALLTIME=%s\nPLAN=%s\nPLAN_SHA256=%s\nREPLAN_OF=%s\nATTEMPT=%s\nMAX_REPLANS=%s\n' "$LABEL" "$SHA" "$HOLDER" "$(date -u +%FT%TZ)" "$RUN_NAME" "$K" "$WALL" "$PLAN_REMOTE" "$(sha256sum < "$PLAN_REMOTE" | cut -d' ' -f1)" "$ROOT_LABEL" "$ATTEMPT_NO" "$MAX_RP" > "$RESV/holder"
DEP=""; VJ=""
if [ -s "$VENV/.gate-ready.json" ]; then echo "VENV ready $VENV"
else
  VJ=$(squeue -h -u "$USER" -n "gate-venv-$H12" -o %i | head -n1)
  if [ -z "$VJ" ]; then
    # nothing of this label is submitted before the venv job exists, so a refusal here releases the reservation
    [ -s "$TREE/scripts/registry/landing/gate_venv.sbatch" ] || { rm -rf "$RESV"; echo "ERR venv not ready and the tree has no gate_venv.sbatch"; exit 2; }
    VJ=$(sbatch --parsable -J "gate-venv-$H12" --output="$LOGS/%x_%j.out" "$TREE/scripts/registry/landing/gate_venv.sbatch" "$TREE" "$VENV" | cut -d';' -f1) || { rm -rf "$RESV"; echo "ERR sbatch venv"; exit 2; }
    echo "VENV-JOB submitted $VJ"
  else echo "VENV-JOB existing $VJ"; fi
  DEP="--dependency=afterok:$VJ"
fi
# a failed array sbatch KEEPS the reservation (noted): the client's failure does not prove the controller took nothing
AJ=$(sbatch --parsable $DEP -J "$RUN_NAME" --array=0-$((K-1)) --time="$WALL" --output="$LOGS/%x_%A_%a.out" "$BBIN/battery.sbatch" "$TREE" "$VENV" "$RESULTS" "$LABEL" "$PLAN_REMOTE" | cut -d';' -f1) || { echo "sbatch array failed $(date -u +%FT%TZ)" > "$RESV/failed"; echo "ERR sbatch array (the reservation $RESV stays: check squeue -n $RUN_NAME before removing it)"; exit 2; }
# ONE stamp per submission: the reservation's array file carries it and the JOBS line hands it to the grading host's record, so the
# record written there and the one reconcile_lost restores from this reservation agree byte for byte (a second clock reading
# on the grading host put the two a second apart whenever the round trip crossed a second boundary, D140)
SUB=$(date -u +%FT%TZ)
printf 'ARRAY=%s\nVENV_JOB=%s\nSUBMITTED_UTC=%s\n' "$AJ" "${VJ:-none}" "$SUB" > "$RESV/array"
echo "JOBS venv=${VJ:-none} array=$AJ reservation=$RESV submitted=$SUB"
REMOTE
    ); rrc=$?
    echo "$ROUT" | sed 's/^/[battery:cluster] /'
    local JL AJ VJ RESV SUB HL HA RECONCILED=0
    if [ $rrc -eq 3 ]; then
      HL=$(grep '^HELD ' <<<"$ROUT" | tail -n1); HA=$(sed -n 's/.* array=\([^ ]*\).*/\1/p' <<<"$HL")
      if [ -n "$HA" ]; then
        # FINALIZED by a submission this state dir has no record of — ours, cut after its sbatch, or another state dir's: read it
        # back and go on to watch/collect it, PROVIDED it is the same request as this invocation's plan ($PLAN.partial); never
        # resubmit (a refusal leaves the .partial trace for a later --collect-only, which collects the original request)
        reconcile_lost "$L" "$PLAN.partial" || die "the reservation of $L reports array $HA but could not be read back from $HOST"
        RECONCILED=1
      else
        rm -f "$PLAN.partial"; refuse_held "$L" "$RUN_NAME" "$HL"
      fi
    else
      { [ $rrc -eq 0 ] && ! grep -q '^ERR' <<<"$ROUT"; } || die "cluster step failed"
      JL=$(grep '^JOBS ' <<<"$ROUT" | tail -n1)
      AJ=$(sed -n 's/.*array=\([0-9]*\).*/\1/p' <<<"$JL"); VJ=$(sed -n 's/.*venv=\([0-9a-z]*\).*/\1/p' <<<"$JL"); RESV=$(sed -n 's/.*reservation=\([^ ]*\).*/\1/p' <<<"$JL")
      SUB=$(sed -n 's/.*submitted=\([^ ]*\).*/\1/p' <<<"$JL")
      [ -n "$AJ" ] || die "could not parse the array job id from: $JL"
      [ -n "$SUB" ] || die "could not parse the submission stamp from: $JL (array $AJ IS submitted: --collect-only --label $LABEL reconciles it)"
      # the submission record: the array (what every task record must name), who submitted it, the cluster reservation that made
      # the submission the only one and the reservation's OWN stamp (the one the JOBS line carried: a reconciled record reads the
      # same stamp back); the root's carries the re-plan budget; a re-plan's names its root and its place in the chain.
      # Written in the order that leaves every cut recoverable: the record is staged first, then two back-to-back renames — the plan
      # into place, then the record (a cut between them leaves a final plan with no .jobs; before them, the .partial: both are the
      # trace --collect-only reconciles from the reservation)
      { printf 'SHA=%s\nLABEL=%s\nTASKS=%s\nWALLTIME=%s\nVENV_JOB=%s\nARRAY=%s\nRUN_NAME=%s\nSUBMITTED_UTC=%s\nSUBMITTER=%s\nRESERVATION=%s\n' "$SHA" "$L" "$KN" "$WALL" "$VJ" "$AJ" "$RUN_NAME" "$SUB" "$HOLDER" "$RESV"
        if [ "$L" = "$LABEL" ]; then printf 'MAX_REPLANS=%s\n' "$MAX_REPLANS"; else printf 'REPLAN_OF=%s\nATTEMPT=%s\n' "$LABEL" "$NO"; fi
      } > "$JOBS.partial" || die "cannot stage $JOBS.partial"
      { mv "$PLAN.partial" "$PLAN" && mv "$JOBS.partial" "$JOBS"; } || die "cannot write $PLAN / $JOBS (array $AJ IS submitted: --collect-only --label $LABEL reconciles it)"
      echo "$(date -u +%FT%TZ) sha=$SHA branch=${BRANCH:-none} battery=$L tasks=$KN walltime=$WALL venv_job=$VJ array=$AJ run=$RUN_NAME" >> "$JOBS_LOG"
      echo "[battery] JOBS sha=$SHA label=$L venv=$VJ array=$AJ run=$RUN_NAME"
      echo "[battery] submitted: array=$AJ (name $RUN_NAME, --array=0-$((KN-1)), QoS high, 4 CPU/16G per task, --time=$WALL, task flags '$TASK_FLAGS')${VJ:+ behind venv job $VJ}"
    fi
    # from here the attempt is the recorded one, submitted just now or read back: watch it (unless --no-wait), then collect it
    AJ=$(sed -n 's/^ARRAY=//p' "$JOBS" | head -n1); VJ=$(sed -n 's/^VENV_JOB=//p' "$JOBS" | head -n1)
    [ "$RECONCILED" -eq 0 ] || echo "[battery] JOBS sha=$SHA label=$L venv=$VJ array=$AJ run=$RUN_NAME (reconciled from the cluster, not submitted by this run)"
    [ "$WAIT" -eq 1 ] || { echo "[battery] --no-wait: finish with: $0 $SHA --collect-only --label $LABEL --out $OUT"; ATTEMPT_RC=0; ATTEMPT_KIND=nowait; return 0; }
    [ "$VJ" = none ] || [ -z "$VJ" ] || { echo "[battery] watching venv job $VJ"; "$WATCH" --slurm "$VJ" --host "$HOST" --poll 30 --pending-min 20 --stall-min 0 --max-hours 2; }
    echo "[battery] watching array $AJ (the driver returns at its terminal state)"
    "$WATCH" --slurm "$AJ" --host "$HOST" --poll 30 --pending-min 20 --stall-min 0 --max-hours 3
  fi
  [ -s "$JOBS" ] || die "no $JOBS: nothing was submitted under label $L"
  [ -s "$PLAN" ] || die "no plan at $PLAN"
  local AJ VJ; AJ=$(sed -n 's/^ARRAY=//p' "$JOBS" | head -n1); VJ=$(sed -n 's/^VENV_JOB=//p' "$JOBS" | head -n1)
  [[ "$AJ" =~ ^[0-9]+$ ]] || die "$JOBS records no array job id (ARRAY=${AJ:-?}): attempt $L's records cannot be bound to a submission"
  [ "$(json_get "$PLAN" label)" = "$L" ] && [ "$(json_get "$PLAN" sha)" = "$SHA" ] || die "$PLAN is not attempt $L's plan for $SHA"
  # 6. records back (by CONTENT, not rsync's size+mtime quick check: a record overwritten with same-size bytes within the same
  # second — the shape a second array writing the same path takes — must reach the collector, never a stale local copy), the
  # array's task states (for the missing-record classification), the verdict
  mkdir -p "$REC"
  rsync -a --checksum --delete "$HOST:$CL_RESULTS/$SHA/reruns/$L/" "$REC/" 2>/dev/null || echo "[battery] no records copied from $HOST:$CL_RESULTS/$SHA/reruns/$L/ (no task wrote one?)"
  rsh "sacct -n -P -X -o JobID,State -j $AJ" 2>/dev/null | "$PY" -c 'import json,sys
d={}
for line in sys.stdin:
    jid,_,state=line.strip().partition("|")
    if "_" in jid and state: d[jid.split("_",1)[1]]=state
print(json.dumps(d))' > "$OUT/$L.sacct.json" 2>/dev/null || echo '{}' > "$OUT/$L.sacct.json"
  "$PY" "$COLLECT" collect --plan "$PLAN" --records "$REC" --out-prefix "$OUT/$L" --sacct "$OUT/$L.sacct.json" --array-job-id "$AJ" \
    ${VJ:+--venv-job-id "$VJ"} --observed "$STATE/battery/observed.json"
  ATTEMPT_RC=$?
  ATTEMPT_KIND=$(json_get "$OUT/$L.verdict.json" refusal_kind)
  return 0
}

L=$LABEL; n=1; ATTEMPTS=("$LABEL")
if [ "$COLLECT_ONLY" -eq 0 ]; then
  # a fresh root: no attempt of this label may exist here, a re-plan recorded under <label>-r<n> included
  [ ! -e "$OUT/$LABEL.jobs" ] || die "label $LABEL was already submitted from here ($OUT/$LABEL.jobs exists): --collect-only to finish it, or another label"
  for j in "$OUT/$LABEL"-r[0-9]*.jobs; do
    [ -e "$j" ] && grep -qx "REPLAN_OF=$LABEL" "$j" && die "label $LABEL already has a re-plan recorded here ($j): --collect-only to finish it, or another label"
  done
  prepare_once
  attempt "$L" submit "${TASKS:-}" 1
else
  attempt "$L" collect
fi
# a task ran out of its walltime: the whole list again with K+1 under the next label of the chain — an attempt already submitted
# (by a run that was cut, or one that did not wait) is collected with its own files; a new one is submitted, in either mode
while [ "$ATTEMPT_RC" -eq 4 ] && [ "$ATTEMPT_KIND" = timeout ]; do
  NEXT=$LABEL-r$((n + 1))
  if [ -s "$OUT/$NEXT.jobs" ] && grep -qx "REPLAN_OF=$LABEL" "$OUT/$NEXT.jobs"; then
    n=$((n + 1)); L=$NEXT; ATTEMPTS+=("$L")
    echo "[battery] re-plan $((n - 1)) of $LABEL was already submitted as $L (array $(sed -n 's/^ARRAY=//p' "$OUT/$L.jobs")): collecting it"
    attempt "$L" collect
    continue
  fi
  if [ "$n" -gt "$MAX_REPLANS" ]; then
    echo "[battery] a task ran out of its walltime and the re-plan budget is spent ($((n - 1)) of $MAX_REPLANS used): INCOMPLETE stands"; break
  fi
  KP=$(json_get "$OUT/$L.plan.json" count)
  if [ "$KP" -ge ${#FILES[@]} ]; then echo "[battery] a task ran out of its walltime with every file already alone in its task (K=$KP): no re-plan can help"; break; fi
  n=$((n + 1)); L=$NEXT; ATTEMPTS+=("$L")
  echo "[battery] a task ran out of its walltime: re-planning the whole list with K=$((KP + 1)) under label $L (re-plan $((n - 1)) of $MAX_REPLANS)"
  prepare_once
  attempt "$L" submit "$((KP + 1))" "$n"
done
[ "$ATTEMPT_KIND" = nowait ] && exit 0
if [ ${#ATTEMPTS[@]} -gt 1 ]; then
  # the chain's verdict at the root: the final attempt's result, where it came from, and every attempt's outcome; the final attempt's
  # junit and log become the root's (the root reached no verdict, so it had none); the root's plan, jobs and records stay its own
  "$PY" - "$OUT" "$LABEL" "$L" "${ATTEMPTS[@]}" <<'PY'
import json, shutil, sys
from pathlib import Path
out, label, final, attempts = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4:]
hist = []
for a in attempts:
    p = out / f"{a}.verdict.json"
    d = json.loads(p.read_text()) if p.is_file() else {}
    hist.append({"label": a, "status": d.get("status"), "refusal": d.get("refusal"), "refusal_kind": d.get("refusal_kind"),
                 "array_job_id": d.get("array_job_id"), "tasks": d.get("tasks"), "plan": str(out / f"{a}.plan.json"),
                 "records": str(out / f"{a}.records"), "jobs": str(out / f"{a}.jobs")})
for ext in ("junit.xml", "log"):
    src = out / f"{final}.{ext}"
    if src.is_file():
        shutil.copyfile(src, out / f"{label}.{ext}")
v = json.loads((out / f"{final}.verdict.json").read_text())
v.update({"final_label": final, "final_plan": str(out / f"{final}.plan.json"), "final_records": str(out / f"{final}.records"),
          "final_jobs": str(out / f"{final}.jobs"), "replans": len(attempts) - 1, "attempts": hist})
tmp = out / f"{label}.verdict.json.tmp"
tmp.write_text(json.dumps(v, indent=2, sort_keys=True))
tmp.replace(out / f"{label}.verdict.json")
PY
fi
STATUS=$(json_get "$OUT/$LABEL.verdict.json" status)
echo "[battery] VERDICT status=${STATUS:-none} label=$LABEL final_label=$L sha=$SHA verdict=$OUT/$LABEL.verdict.json"
exit "$ATTEMPT_RC"
