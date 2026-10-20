#!/bin/bash
# gate_slurm.sh — the results-registry LANDING GATE on the primary Slurm cluster: the full pytest suite
# on 40 CPU shards from an exported tree, merged into one summary.json, judged by the failure-set verdict against
# a baseline summary of main. Since 2026-09-28 it is THE gate: land.sh runs
# it in the foreground when no verdict is recorded, and baseline_refresh.sh runs it with --baseline whenever main moves.
#
#   gate_slurm.sh <branch-or-sha> [--baseline] [--branch NAME] [--shards N] [--no-wait] [--collect-only]
#                 [--baseline-sha SHA] [--exclude NODES]
#
# What it does, from the grading host (light orchestration only — nothing heavy runs here, nothing touches the live registry):
#   1. resolves the ref to a 40-hex sha (a branch resolves through the base checkout's refs; a worktree branch is one
#      of them) and makes sure the sha is on the cluster's clone $CLUSTER_REPO_MIRROR (mirror-repos.sh, else a
#      direct push of the object under refs/heads/gate/<sha7>), then pushes the repository's archived branches and tags to
#      the mirror (mirror_history.sh, verified against the mirror's listing; a history that does not arrive refuses here:
#      the suite reads commits only the archive reaches, and a tree exported without them fails those tests by name);
#   2. ships this directory's gate code to $SHARED/gates/bin and, on the login node (git only), exports the tree as a
#      standalone FULL checkout at $SHARED/gates/trees/<sha> (tree_export.sh: exact HEAD, clean, every head, tag and
#      archived branch; an existing tree is completed against the mirror's archive and tags before it is reused);
#   3. submits ONE sbatch array (gate_array.sbatch, -J gate-<sha7>-<user>, --array=0-(N-1), 4 CPUs / 16G / 1 h 15 per
#      task, QoS high, partition general, --local-venv --hide-slurm, pytest bound 3600 s), behind a venv-build job
#      (gate_venv.sbatch) when the lock-exact shared venv $SHARED/envs/<checkout name>-gate-<hash12(pyproject.toml, uv.lock)>
#      is not ready yet — one build serves every sha with the same dependency files, and a build already queued by
#      another gate is reused, never duplicated; --exclude NODES keeps the array off nodes that failed a previous try;
#   4. submits the finalize step (gate_finalize.sbatch, --dependency=afterany:<array>) that merges the shard records
#      into $SHARED/gates/<sha>/summary.json (a shard without a record = TIMEOUT = no result);
#   5. watches the array and then the finalizer in the foreground (watch-job.sh; exits at the terminal state), copies
#      summary.json back to $STATE/summaries/summary-<sha>.json (and, with --baseline, writes the sha to $STATE/BASELINE),
#      and prints status / counts / failure nodes;
#   6. for a tip (not --baseline): verdict_local.py <baseline summary> <tip summary> > $STATE/verdicts/gate-land-<branch>.out
#      (+ a per-sha copy verdict-<branch>-<sha7>.out) — the file land.sh reuses (`^VERDICT: LAND`, `^MERGE sha=<8 hex>`).
#      The baseline is $STATE/BASELINE unless --baseline-sha names one (land.sh always names origin/main's).
# Job ids are appended to $STATE/jobs.log and printed as one parseable line: `[gate] JOBS sha=… venv=… array=… finalize=… run=…`.
# Every job's log is $SHARED/exp/logs/<job name>_<job id>.out (an array task's <job name>_<array id>_<index>.out), named with --output
# at each submission: a log path written into a job script could not follow the site's shared root.
# A finalizer sbatch that fails AFTER the array was queued leaves an array nobody merges: the array is canceled here and ONE line
# says so — `[gate] PARTIAL sha=… array=… venv=… run=… canceled=yes|no` — which baseline_refresh.sh records (state PARTIAL) so
# nothing resubmits for that sha until the array is confirmed out of the queue.
# --no-wait stops after step 4 (run again with --collect-only later; baseline_refresh.sh is that later).
#
# State (never /tmp): STATE=${REGISTRY_LANDING_STATE:-~/.local/state/registry-landing}/{BASELINE, summaries/, verdicts/, jobs.log}.
# Knobs (env): GATE_HOST (the login host), GATE_SHARED (the shared root), GATE_RUSER ($USER), SISTEER_REPO (the checkout),
# MIRROR_REPOS_SH (~/scripts/mirror-repos.sh), MIRROR_HISTORY_SH (this directory's mirror_history.sh). The login host, the shared root,
# the mirror selector, the checkout and its clone on the cluster come from the site config's primary cluster block, or from the block
# of the cluster an exported CLUSTER_DEST names (site_env.sh beside this script); GATE_HOST, GATE_SHARED and the environment's own
# CLUSTER_* values win, and a value nothing supplies is refused by name.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=${REGISTRY_LANDING_STATE:-$HOME/.local/state/registry-landing}
SUMMARIES=$STATE/summaries; VERDICTS=$STATE/verdicts; JOBS_LOG=$STATE/jobs.log; BASELINE_FILE=$STATE/BASELINE
RUSER=${GATE_RUSER:-${USER:-$(id -un)}}
MIRROR_REPOS=${MIRROR_REPOS_SH:-$HOME/scripts/mirror-repos.sh}
MIRROR_HISTORY=${MIRROR_HISTORY_SH:-$HERE/mirror_history.sh}
WATCH=$HOME/scripts/watch-job.sh
SHARDS=40; BASELINE=0; WAIT=1; COLLECT_ONLY=0; BRANCH=""; BASE_SHA=""; REF=""; EXCLUDE=""
die() { echo "[gate] ERROR: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
  case "$1" in
    --baseline) BASELINE=1 ;;
    --branch) BRANCH=$2; shift ;;
    --shards) SHARDS=$2; shift ;;
    --no-wait) WAIT=0 ;;
    --collect-only) COLLECT_ONLY=1 ;;
    --baseline-sha) BASE_SHA=$2; shift ;;
    --exclude) EXCLUDE=$2; shift ;;
    -h|--help) awk 'NR == 1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
    -*) die "unknown flag $1" ;;
    *) [ -z "$REF" ] || die "one ref only"; REF=$1 ;;
  esac
  shift
done
[ -n "$REF" ] || die "usage: gate_slurm.sh <branch-or-sha> [--baseline] [--branch NAME] [--shards N] [--no-wait] [--collect-only] [--baseline-sha SHA] [--exclude NODES]"
# the cluster's names from the site config (site_env.sh); GATE_HOST and GATE_SHARED win over it
CLUSTER_HOST="${GATE_HOST:-${CLUSTER_HOST:-}}"; CLUSTER_SHARED="${GATE_SHARED:-${CLUSTER_SHARED:-}}"
# shellcheck source=site_env.sh
source "$HERE/site_env.sh"
site_env_load CLUSTER_HOST CLUSTER_SHARED MIRROR_REPOS_CLUSTER SITE_REPO MIRROR_REPOS_NAME CLUSTER_REPO_MIRROR || exit 2
HOST=$CLUSTER_HOST; SHARED=$CLUSTER_SHARED; MIRROR=$CLUSTER_REPO_MIRROR; REPO=$SITE_REPO
CL_BIN=$SHARED/gates/bin; CL_TREES=$SHARED/gates/trees; CL_RESULTS=$SHARED/gates
CL_LOGS=$SHARED/exp/logs     # every job's log, named at submission (<job name>_<job id>.out; an array task's carries its index)
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=25 "$HOST" "$@"; }
mkdir -p "$SUMMARIES" "$VERDICTS" || die "cannot create $STATE"

# 1. resolve
if [[ "$REF" =~ ^[0-9a-f]{40}$ ]]; then SHA=$REF
else
  SHA=$(git -C "$REPO" rev-parse --verify --quiet "$REF^{commit}") || die "cannot resolve '$REF' in $REPO"
  [ -n "$BRANCH" ] || BRANCH=$REF
fi
S7=${SHA:0:7}; S8=${SHA:0:8}
RUN_NAME=gate-$S7-$RUSER
# the dependency files are read from the base checkout's OBJECT STORE: a sha that is only on origin must be fetched first
# (baseline_refresh.sh does `git fetch origin <sha>` before calling here); an absent commit refuses loudly right here
git -C "$REPO" cat-file -e "$SHA^{commit}" 2>/dev/null || die "$SHA is not in $REPO's object store (git fetch origin $SHA first; baseline_refresh.sh does)"
PH=$(git -C "$REPO" show "$SHA:pyproject.toml" 2>/dev/null | sha256sum | cut -d' ' -f1) || die "no pyproject.toml at $SHA"
LH=$(git -C "$REPO" show "$SHA:uv.lock" 2>/dev/null | sha256sum | cut -d' ' -f1) || die "no uv.lock at $SHA"
H12=$(printf '%s\n%s\n' "$PH" "$LH" | sha256sum | cut -c1-12)
VENV=$(site_env_gate_venv "$SHARED" "$MIRROR_REPOS_NAME" "$H12")
TREE=$CL_TREES/$SHA
echo "[gate] ref=$REF sha=$SHA branch=${BRANCH:-none} baseline=$BASELINE shards=$SHARDS venv=$VENV state=$STATE${EXCLUDE:+ exclude=$EXCLUDE}"

if [ "$COLLECT_ONLY" -eq 0 ]; then
  # mirror
  if ! rsh "git -C $MIRROR cat-file -e $SHA 2>/dev/null"; then
    echo "[gate] $S7 not on the cluster mirror — running mirror-repos.sh"
    "$MIRROR_REPOS" --cluster "$MIRROR_REPOS_CLUSTER" "$MIRROR_REPOS_NAME" || echo "[gate] mirror-repos.sh reported a problem; trying a direct push of the sha"
    if ! rsh "git -C $MIRROR cat-file -e $SHA 2>/dev/null"; then
      # mirror-repos.sh pushes the base checkout's LOCAL branches; a sha that is only on origin (a freshly landed main the
      # base checkout has not fast-forwarded to — that checkout is not ours to move) is pushed by object id under a
      # gate/ ref, straight from the local object store
      git -C "$REPO" cat-file -e "$SHA" 2>/dev/null || die "$SHA is not in $REPO's object store (fetch origin first)"
      git -C "$REPO" push -q "ssh://$HOST$MIRROR" "$SHA:refs/heads/gate/$S7" || die "direct push of $SHA to $HOST:$MIRROR failed"
      echo "[gate] pushed $S7 to the mirror as refs/heads/gate/$S7"
    fi
    rsh "git -C $MIRROR cat-file -e $SHA" || die "$SHA still not on $HOST:$MIRROR (commit and push first?)"
  fi
  # the history namespaces, before EVERY export: mirror-repos.sh moves heads and tags only, and ran above only when the sha was absent;
  # the archived branches (refs/archive/*) ride no refspec of its, and the suite reads commits only they reach -- a tree exported
  # without them fails those tests by name, so a history that does not reach the mirror refuses the gate here, not on the shards
  HOUT=$("$MIRROR_HISTORY" "$REPO" "ssh://$HOST$MIRROR" 2>&1); hrc=$?
  echo "$HOUT" | sed 's/^/[gate] /'
  [ $hrc -eq 0 ] || die "the archived branches and tags of $REPO did not reach $HOST:$MIRROR (mirror_history.sh rc $hrc); the exported tree would lack commits the suite reads"
  # 2. ship the gate code, export the tree (login node, git only)
  rsh "mkdir -p $CL_BIN $CL_TREES $CL_RESULTS/$SHA $CL_LOGS" || die "cannot create cluster dirs"
  rsync -a --chmod=F644 "$HERE/gate_lib.py" "$HERE/gate_task.py" "$HERE/gate_array.sbatch" "$HERE/gate_venv.sbatch" "$HERE/gate_finalize.sbatch" "$HOST:$CL_BIN/" || die "rsync of gate code failed"
  TOUT=$(rsh bash -s -- "$SHA" "$MIRROR" "$TREE" < "$HERE/tree_export.sh"); trc=$?
  echo "$TOUT" | sed 's/^/[gate:cluster] /'
  [ $trc -eq 0 ] && ! grep -q '^ERR' <<<"$TOUT" || die "tree export failed"
  # 3+4. venv + submit (one remote bash; sbatch only on the login node)
  # the log directory rides before EXCLUDE: an empty last positional vanishes through ssh's re-parse of the command line, harmlessly
  OUT=$(rsh bash -s -- "$SHA" "$TREE" "$VENV" "$H12" "$CL_BIN" "$CL_RESULTS" "$SHARDS" "$RUN_NAME" "$PH" "$LH" "$S7" "$CL_LOGS" "$EXCLUDE" <<'REMOTE'
set -uo pipefail
SHA=$1 TREE=$2 VENV=$3 H12=$4 BIN=$5 RESULTS=$6 SHARDS=$7 RUN_NAME=$8 PH=$9 LH=${10} S7=${11} LOGS=${12:-} EXCLUDE=${13:-}
# every job's log goes under the site's shared root, named at submission (it overrides any log directive in the job scripts)
[ -n "$LOGS" ] || { echo "ERR no log directory"; exit 2; }
[ "$(sha256sum < "$TREE/pyproject.toml" | cut -d' ' -f1)" = "$PH" ] || { echo "ERR pyproject.toml hash differs from the grading host's"; exit 2; }
[ "$(sha256sum < "$TREE/uv.lock" | cut -d' ' -f1)" = "$LH" ] || { echo "ERR uv.lock hash differs from the grading host's"; exit 2; }
DEP=""; VJ=""
if [ -s "$VENV/.gate-ready.json" ]; then echo "VENV ready $VENV"
else
  VJ=$(squeue -h -u "$USER" -n "gate-venv-$H12" -o %i | head -n1)
  if [ -z "$VJ" ]; then
    VJ=$(sbatch --parsable -J "gate-venv-$H12" --output="$LOGS/%x_%j.out" "$BIN/gate_venv.sbatch" "$TREE" "$VENV" | cut -d';' -f1) || { echo "ERR sbatch venv"; exit 2; }
    echo "VENV-JOB submitted $VJ"
  else echo "VENV-JOB existing $VJ"; fi
  DEP="--dependency=afterok:$VJ"
fi
# defaults since 2026-09-26 04:2xZ (the coordinator): every shard runs from a node-local venv copy (--local-venv: shard 11's
# fresh-interpreter imports from VAST blew the 2400 s bound) with Slurm's commands hidden from PATH (--hide-slurm: a nested
# srun from a test landed on another node); pytest bound 3600 s inside a 1 h 15 min walltime
AJ=$(sbatch --parsable $DEP ${EXCLUDE:+--exclude=$EXCLUDE} -J "$RUN_NAME" --array=0-$((SHARDS-1)) --time=01:15:00 --output="$LOGS/%x_%A_%a.out" "$BIN/gate_array.sbatch" "$SHA" "$TREE" "$VENV" "$RESULTS" "$SHARDS" "$RUN_NAME" 3600 1 1 | cut -d';' -f1) || { echo "ERR sbatch array"; exit 2; }
FJ=$(sbatch --parsable --dependency=afterany:$AJ -J "gate-$S7-fin-$USER" --output="$LOGS/%x_%j.out" "$BIN/gate_finalize.sbatch" "$SHA" "$TREE" "$VENV" "$RESULTS" "$SHARDS" "$RUN_NAME" "$AJ" | cut -d';' -f1) || { echo "PARTIAL array=$AJ venv=${VJ:-none}"; echo "ERR sbatch finalize: array $AJ is queued without its finalizer"; exit 2; }
echo "JOBS venv=${VJ:-none} array=$AJ finalize=$FJ"
REMOTE
  ); rrc=$?
  echo "$OUT" | sed 's/^/[gate:cluster] /'
  if [ $rrc -ne 0 ] || grep -q '^ERR' <<<"$OUT"; then
    PA=$(sed -n 's/^PARTIAL array=\([0-9]*\).*/\1/p' <<<"$OUT" | tail -n1)
    if [ -n "$PA" ]; then
      # the array was queued, the finalizer that merges its shards was not: cancel it now and say whether it left the queue; the
      # `[gate] PARTIAL` line is the record baseline_refresh.sh keeps so nothing resubmits for this sha until the array is confirmed gone
      PV=$(sed -n 's/^PARTIAL .*venv=\([0-9a-z]*\).*/\1/p' <<<"$OUT" | tail -n1)
      rsh "scancel $PA" 2>/dev/null
      if rsh "squeue -h -u $RUSER -o %i" 2>/dev/null | grep -q "^$PA\(_\|\$\)"; then PC=no; else PC=yes; fi
      echo "[gate] PARTIAL sha=$SHA array=$PA venv=${PV:-none} run=$RUN_NAME canceled=$PC — the array was queued without its finalizer"
    fi
    die "cluster step failed"
  fi
  JOBS=$(grep '^JOBS ' <<<"$OUT" | tail -n1)
  AJ=$(sed -n 's/.*array=\([0-9]*\).*/\1/p' <<<"$JOBS"); FJ=$(sed -n 's/.*finalize=\([0-9]*\).*/\1/p' <<<"$JOBS"); VJ=$(sed -n 's/.*venv=\([0-9a-z]*\).*/\1/p' <<<"$JOBS")
  [ -n "$AJ" ] && [ -n "$FJ" ] || die "could not parse job ids from: $JOBS"
  echo "$(date -u +%FT%TZ) sha=$SHA branch=${BRANCH:-none} baseline=$BASELINE shards=$SHARDS venv_job=$VJ array=$AJ finalize=$FJ run=$RUN_NAME${EXCLUDE:+ exclude=$EXCLUDE}" >> "$JOBS_LOG"
  echo "[gate] JOBS sha=$SHA venv=$VJ array=$AJ finalize=$FJ run=$RUN_NAME"
  echo "[gate] submitted: venv=$VJ array=$AJ (name $RUN_NAME, --array=0-$((SHARDS-1)), QoS high, 4 CPU/16G/1h15 per task, --local-venv --hide-slurm, pytest bound 3600 s${EXCLUDE:+, --exclude=$EXCLUDE}) finalize=$FJ"
  [ "$WAIT" -eq 1 ] || { echo "[gate] --no-wait: collect later with: $0 $SHA --collect-only $([ "$BASELINE" -eq 1 ] && echo --baseline) ${BRANCH:+--branch $BRANCH}"; exit 0; }
  # 5. watch (foreground; exits at the terminal state). The array's per-task log is no liveness signal for the array, so
  # --stall-min 0 there; the finalizer is short.
  echo "[gate] watching array $AJ"
  "$WATCH" --slurm "$AJ" --host "$HOST" --poll 60 --pending-min 40 --stall-min 0 --max-hours 4
  echo "[gate] watching finalizer $FJ"
  "$WATCH" --slurm "$FJ" --host "$HOST" --poll 30 --pending-min 15 --stall-min 0 --max-hours 2
fi

# collect
LOCAL="$SUMMARIES/summary-$SHA.json"
scp -q "$HOST:$CL_RESULTS/$SHA/summary.json" "$LOCAL.partial" || die "no summary.json at $HOST:$CL_RESULTS/$SHA/ (finalizer did not run?)"
mv "$LOCAL.partial" "$LOCAL"
python3 - "$LOCAL" <<'PY' || exit 5
import json, sys
d = json.load(open(sys.argv[1]))
print(f"[gate] SUMMARY sha={d.get('sha')} status={d.get('status')} pytest_rc={d.get('pytest_rc')} counts={d.get('counts')} elapsed_s={d.get('elapsed_s')} "
      f"shards={d.get('shards')} layout={(d.get('shard_layout') or {}).get('map_sha256','')[:12]} missing={(d.get('gate') or {}).get('missing_shards')}")
print(f"[gate] FAILURE NODES ({len(d.get('failures') or [])}):")
for f in d.get("failures") or []: print("  " + str(f.get("node")))
if d.get("status") not in ("PASS", "FAIL"):
    print(f"[gate] NON-TERMINAL status {d.get('status')}: not a gate result — re-run before reading"); sys.exit(1)
PY
if [ "$BASELINE" -eq 1 ]; then
  echo "$SHA" > "$BASELINE_FILE.partial" && mv "$BASELINE_FILE.partial" "$BASELINE_FILE"
  echo "[gate] baseline recorded: $LOCAL (BASELINE=$SHA)"
  exit 0
fi
# 6. verdict
BASE=${BASE_SHA:-$(cat "$BASELINE_FILE" 2>/dev/null || true)}
[ -n "$BASE" ] || die "no baseline: run a --baseline gate first or pass --baseline-sha"
BASEF="$SUMMARIES/summary-$BASE.json"; [ -s "$BASEF" ] || die "no baseline summary at $BASEF"
[ -n "$BRANCH" ] || die "a tip verdict needs --branch NAME (names $VERDICTS/gate-land-<branch>.out)"
OUTF="$VERDICTS/gate-land-$BRANCH.out"
python3 "$HERE/verdict_local.py" "$BASEF" "$LOCAL" > "$OUTF.partial" 2>&1; mv "$OUTF.partial" "$OUTF"
cp "$OUTF" "$VERDICTS/verdict-$BRANCH-$S7.out"
grep -E "^(BASELINE|MERGE|WARNING|INTRODUCED|VANISHED|VERDICT)" "$OUTF"
grep -q "^VERDICT: LAND" "$OUTF" && echo "[gate] LAND — recorded in $OUTF (MERGE sha=$S8)" || echo "[gate] NOT LAND — see $OUTF"
