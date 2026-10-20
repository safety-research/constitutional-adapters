#!/bin/bash
# live_db_slurm.sh <tip sha> --branch B --main MAIN [--expect-skipped N] [--out-dir DIR] -- land.sh's step 3 ON SLURM:
# the registry tests that read the live database run against the CLUSTER REPLICA, never on the grading host.
#
#   1. head        `registry replica head --out $STATE/land/<B>.head.json.gz` on the grading host: ONE read transaction of the live database at
#                  a cursor-free boundary (bounded wait, 20 min). Exit 6 (no boundary) = DEFER: wait for the harvester's next completed
#                  cycle (state/harvest_cycle.json moves; bound LIVE_DB_DEFER_MIN, 45) and retry ONCE; still none -> refuse (exit 5).
#   2. mirror      a record-log mirror pass that STARTS after the head (registry-mirrors.service). The unit runs back to back, so a
#                  pass already running when the step enters took its census BEFORE the head: it is waited out (the unit idle, or a newer
#                  start stamp), then a fresh pass is started (`systemctl --user start`, blocking for the oneshot unit; one the timer
#                  started meanwhile is fresh too and the start merges into it) and verified from ITS OWN record: systemd's
#                  ExecMainStartTimestamp after the head's taken_utc; state/mirror_state.json written at that pass's end (no earlier than
#                  its start) with the LIVE_DB_MIRROR_DEST row `switched` -- `mirrors sync` exits 0 for a budget-stopped pass that switched
#                  nothing, so the exit code alone proves nothing; such a pass is followed by another -- and the manifest that row names
#                  carrying EVERY file the head names, with the head's sha where the head has one (an EMPTY_SHA entry, an unreadable
#                  delivery, must still be present: the cluster's require_mirror demands presence of every entry, and this is that check
#                  made before anything ships). Manifest names repeat between ingests (CHECKSUMS-<max ingest_seq>) and a pass REPLACES the
#                  file, so the manifest read here is bound to the pass by the digest its own mirror_ack record (the row's `ack_record`)
#                  carries as `checksums_sha256`; a mismatch means a newer pass has rewritten it, and that pass's record is waited for and
#                  verified instead. Bounded by LIVE_DB_MIRROR_WAIT_S (3600 s, about two passes): past it, refused by name, exit 4, nothing
#                  shipped. LIVE_DB_MIRROR_UNIT=none skips the step (a world with no mirror); LIVE_DB_MIRROR_SYNC, the retired override
#                  command, is refused by name (an override cannot be verified against the head).
#   3. tree        mirror-repos.sh + tree_export.sh on the login node: the tip's exact tree at $SHARED/gates/trees/<tip>; the lock-exact gate
#                  venv must be ready (the gate of step 2 built it; else refuse, naming gate_venv.sbatch).
#   4. ship        EVERY SUBMISSION OWNS IMMUTABLE PATHS, minted here (`<utc>-<pid>-<4 hex>`, the submission token): the head goes to
#                  <replica root>/builds/land-<B>-<tip8>-<token>/head.json.gz and the job writes $SHARED/gates/<tip>/live_db/<token>/, both
#                  created WITHOUT -p (a directory already there refuses: no later submission can replace a pending job's input, and no
#                  earlier job's output can be read as this one's). The paper pin (l1_assembled_pin.jsonl) is shipped once.
#   5. submit      `sbatch --parsable -J registry-replica --dependency=singleton replica.sbatch land <tree> <head> <venv> <B> …`
#                  (16 CPU / 32 G / --time 01:30:00, QoS high, CPU only), the job id printed as `[land] live-db job <id>`; its log is
#                  named at the submission, --output=$SHARED/exp/logs/<job name>_<job id>.out, which step 7 reads back.
#   6. watch       watch-job.sh in the foreground (poll 30 s, stall 40 min, bound 3 h); then the job's TERMINAL STATE is read from sacct
#                  (the watcher exits 0 for DONE and FAILED alike, so the state is read, never inferred): anything but COMPLETED refuses.
#   7. collect     $HOST:<out>/{check.json,mirror.json,ingest.json,junit.xml,pytest.log,summary.json} -> $STATE/land/<B>.live_db/ (recreated
#                  empty first); junit.xml and pytest.log copied to $STATE/land/<B>.{junit.xml,pytest.log} (durations.py reads them).
#   8. verdict     exit 0 only when ALL of: the job COMPLETED; check.json AND summary.json carry an `identity` naming THIS job id, THIS tip
#                  (built_by_tree) and THIS head (head_sha256 of the head taken in step 1); check.json.equal; the job's own status PASS with
#                  pytest rc 0 and skipped == expected; summary.published names a generation AND <replica root>/CURRENT names it now. Else
#                  a named refusal (exit 5), every failed rule printed as `REFUSED: …`. The verdict is written to
#                  $REGISTRY_ROOT/state/replica_check.json (the status page's row: utc, equal, ok, the problems, head n_files, both sha12s,
#                  the projection version, the generation, CURRENT, the job id and state, the submission token, and -- the cutover allowance -- `c1_class`,
#                  `accepted_as` (equal | c1_class | null), `n_movers`, `columns_differing`, the declaration).
# THE CUTOVER ALLOWANCE (landing waits on the live database's adoption of the new order). The live database still folds the pre-C1 order while the
# replica folds C1's deterministic same-stamp tie-break, so until the live database adopts the C1 order (`restamp` of the certified keys
# inside a window) the two legitimately differ on exactly the certified runs: `lifecycle_state` judged → superseded and the CASE columns
# that read it (is_complete, is_current, why), the class refold-compare certified (job 922769). LAND_ACCEPT_C1_CLASS=<certification
# reference> (free text naming that certification) declares the allowance: it rides to the job as the land mode's 10th positional
# ACCEPT_C1_CLASS (`printf %q`-escaped so it arrives verbatim through ssh; the 9th positional is then TREE itself when no recovery pin is
# set; never an export flag) and the verdict here accepts `check.json.equal OR (check.json.c1_class AND the declaration)`, printing
# `ACCEPTED: equal modulo the C1 same-stamp class (N movers) -- <reference>` and recording `accepted_as: c1_class` for land.sh's
# landing-gate description. Empty or unset: the verdict is unchanged (an UNEQUAL check refuses, nothing is published). The job's own
# summary must agree (`check_accepted_as`): a job that published under an allowance this grading host did not declare is refused by name. In a
# RECOVERY the module that folds and checks is the PINNED tree's, so the declaration is refused before any submission unless that tree's
# sisteer/registry/replica.py knows --accept-c1-class (a tree older than the allowance would die on the argument mid-job); a recovery
# pinned to such a tree while the live database still folds the pre-C1 order cannot land on the class -- restamp the certified keys on
# the live database first (the allowance's planned retirement), or pin a generation built by a tree that carries the allowance. NO
# OTHER COLUMN IS EVER ACCEPTED: `c1_class` is false the moment any row differs outside those four columns (decode_policy_match included),
# is on one side only, or any mover fails the same-stamp-pair predicate -- the landing then refuses exactly as before, declaration or not.
# Retire the allowance (stop setting the variable) once the live database carries the C1 order.
# RECOVERY (recover.sh): LIVE_DB_REPLICA_TREE=<40-hex sha> pins the tree whose code runs
# the REPLICA side of the job (the copy of CURRENT, the presence check, the increment, `replica check`, the publication) -- the CURRENT good
# generation's built_by_tree -- while the live_db tests still run from the TIP's tree; it is passed to replica.sbatch as the land mode's
# REPLICA_TREE positional (the `land/3` contract) and must be exported on the cluster (else refuse, naming tree_export.sh). The verdict
# then requires identity.built_by_tree == that tree AND summary.tree to name the tip (the tests' tree); unset, built_by_tree must be the
# tip itself. THE DISPATCHER IN A RECOVERY IS THE CHAIN'S OWN replica.sbatch (LIVE_DB_DISPATCHER, default this directory's), never the
# pinned tree's and never the tip's: the pinned tree may predate the 9th positional -- the R3 bootstrap generation's does
# -- and such a script, handed nine arguments, runs the TIP's code on the replica side and PUBLISHES its generation before the identity
# check here can refuse it. So, before anything else, the dispatcher is probed (`replica.sbatch contract` must answer `land/3 …
# ACCEPT_C1_CLASS`; anything else -- `unknown mode`, or the land/2 contract without the 10th positional -- refuses, exit 5, nothing taken,
# nothing shipped); in step 4 it is shipped beside the head into the submission's own directory and its sha256 read back from the cluster
# must equal the local file's; step 5 submits THAT path. A plain landing keeps the tip's own replica.sbatch (dispatcher and module from
# one tree certify the tip with its own code).
# The grading host runs NO pytest here and never takes the pytest lock. Every line carries `[land] @ live-db-<step>:<start|end> <UTC>` stamps
# (measure/landings.py reads them). Knobs (env): REGISTRY_ROOT (~/registry), REGISTRY_LANDING_STATE, SISTEER_REPO, LAND_PY, LIVE_DB_PYTHONPATH
# (default <worktree>/src), GATE_HOST (the login host), GATE_RUSER, GATE_SHARED (the shared root), REPLICA_ROOT ($SHARED/registry/replica), MIRROR_REPOS_SH,
# MIRROR_HISTORY_SH (this directory's mirror_history.sh: the archived branches and tags reach the repo mirror before the export; a
# problem is a note, the replica side runs the tree's code only), WATCH_JOB_SH, LIVE_DB_WAIT_MIN (20), LIVE_DB_DEFER_MIN (45), LIVE_DB_TIME (01:30:00), LIVE_DB_HASH (all|manifest), LIVE_DB_PIN_L1,
# LIVE_DB_MIRROR_UNIT (registry-mirrors.service; `none` skips step 2), LIVE_DB_MIRROR_DEST (the cluster's registry label: the destination row of
# state/mirror_state.json the landing's cluster is), LIVE_DB_MIRROR_WAIT_S (3600), LIVE_DB_MIRROR_POLL_S (30), LIVE_DB_WATCH_MAX_HOURS (3),
# LIVE_DB_STATE_POLL_S (15: between sacct reads of an empty state), LIVE_DB_DISPATCHER (the replica.sbatch shipped in a recovery; default
# this directory's), LAND_ACCEPT_C1_CLASS (the cutover allowance, above; default none). --out-dir names the BASE under which the
# submission's own directory is created (default $SHARED/gates/<tip>/live_db). The login host, the shared root, the registry label, the
# checkout and its clone on the cluster come from the site config's primary cluster block, or from the block of the cluster an exported
# CLUSTER_DEST names (site_env.sh beside this script); GATE_HOST, GATE_SHARED and the environment's own CLUSTER_* values win, and a
# value nothing supplies is refused by name (exit 5). LIVE_DB_PIN_L1
# names the paper pin shipped to a replica root that has none (no default: nothing is shipped).
# Exit codes: 0 the verdict PASS; 5 a named refusal (`[land] live-db REFUSED: …`, before the job or at the verdict); 4 no record-log
# mirror pass that started after the head completed within LIVE_DB_MIRROR_WAIT_S (step 2; nothing shipped, nothing submitted).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=${REGISTRY_LANDING_STATE:-$HOME/.local/state/registry-landing}
LAND=$STATE/land
REG=${REGISTRY_ROOT:-$HOME/registry}
RUSER=${GATE_RUSER:-${USER:-$(id -un)}}
MIRROR_REPOS=${MIRROR_REPOS_SH:-$HOME/scripts/mirror-repos.sh}
MIRROR_HISTORY=${MIRROR_HISTORY_SH:-$HERE/mirror_history.sh}
WATCH=${WATCH_JOB_SH:-$HOME/scripts/watch-job.sh}
WAIT_MIN=${LIVE_DB_WAIT_MIN:-20}; DEFER_MIN=${LIVE_DB_DEFER_MIN:-45}
JOB_TIME=${LIVE_DB_TIME:-01:30:00}; HASH_MODE=${LIVE_DB_HASH:-all}; MAX_HOURS=${LIVE_DB_WATCH_MAX_HOURS:-3}
STATE_POLL_S=${LIVE_DB_STATE_POLL_S:-15}
PIN_SRC=${LIVE_DB_PIN_L1:-}
RTREE=${LIVE_DB_REPLICA_TREE:-}      # recovery: the tree whose code runs the replica side (default: the tip's)
ACCEPT_C1=${LAND_ACCEPT_C1_CLASS:-}  # the cutover allowance (header): the certification reference, or empty for no allowance
DISPATCHER=${LIVE_DB_DISPATCHER:-$HERE/replica.sbatch}   # recovery: the sbatch script shipped and submitted (the chain's own, never a tree's)
DISPATCH_REMOTE=""; DISPATCH_SHA=""; CONTRACT=""
MIRROR_UNIT=${LIVE_DB_MIRROR_UNIT:-registry-mirrors.service}   # step 2; `none` skips it
MIRROR_WAIT_S=${LIVE_DB_MIRROR_WAIT_S:-3600}; MIRROR_POLL_S=${LIVE_DB_MIRROR_POLL_S:-30}
SHA=""; B=""; MAIN=""; EXPECT_SKIPPED=0; OUT_BASE=""
die() { echo "[land] live-db REFUSED: $*"; exit 5; }
utc() { date -u +%FT%TZ; }
stamp() { echo "[land] @ live-db-$1:$2 $(utc)"; }
while [ $# -gt 0 ]; do
  case "$1" in
    --branch) B=$2; shift ;;
    --main) MAIN=$2; shift ;;
    --expect-skipped) EXPECT_SKIPPED=$2; shift ;;
    --out-dir) OUT_BASE=$2; shift ;;
    -h|--help) awk 'NR == 1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
    -*) die "unknown flag $1" ;;
    *) [ -z "$SHA" ] || die "one sha only"; SHA=$1 ;;
  esac
  shift
done
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || die "usage: live_db_slurm.sh <40-hex tip> --branch B --main MAIN [--expect-skipped N]"
# the cluster's names from the site config (site_env.sh); GATE_HOST and GATE_SHARED win over it
CLUSTER_HOST="${GATE_HOST:-${CLUSTER_HOST:-}}"; CLUSTER_SHARED="${GATE_SHARED:-${CLUSTER_SHARED:-}}"
# shellcheck source=site_env.sh
source "$HERE/site_env.sh"
site_env_load CLUSTER_HOST CLUSTER_SHARED CLUSTER_DEST SITE_REPO MIRROR_REPOS_NAME CLUSTER_REPO_MIRROR \
  || die "the site config does not supply the cluster's names (the line above names the setting); nothing taken"
REPO_DIR=$SITE_REPO; HOST=$CLUSTER_HOST; SHARED=$CLUSTER_SHARED; MIRROR=$CLUSTER_REPO_MIRROR; CL_TREES=$SHARED/gates/trees
REPLICA_ROOT=${REPLICA_ROOT:-$SHARED/registry/replica}
CL_LOGS=$SHARED/exp/logs     # the job's log, named at submission: <job name>_<job id>.out, which the collect step reads back
MIRROR_DEST=${LIVE_DB_MIRROR_DEST:-$CLUSTER_DEST}               # the destination row of state/mirror_state.json the landing's cluster is
[ -n "$B" ] || die "--branch is required"
[ -z "${LIVE_DB_MIRROR_SYNC+x}" ] || die "LIVE_DB_MIRROR_SYNC is retired (D143): step 2 drives $MIRROR_UNIT itself and verifies the pass against the head; an override command cannot be verified. LIVE_DB_MIRROR_UNIT=none skips the step. Nothing taken"
S8=${SHA:0:8}
WT=$REPO_DIR/worktrees/$B
PY=${LAND_PY:-$REPO_DIR/.venv/bin/python}
export PYTHONPATH=${LIVE_DB_PYTHONPATH:-$WT/src}
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=25 "$HOST" "$@"; }
mkdir -p "$LAND" || die "cannot create $LAND"
# the submission token: every path this submission writes on the cluster is its own (step 4 of the header)
TOKEN=$(date -u +%Y%m%dT%H%M%SZ)-$$-$(od -An -N2 -tx1 /dev/urandom | tr -d ' \n')
[ -z "$OUT_BASE" ] && OUT_BASE=$SHARED/gates/$SHA/live_db
OUT_REMOTE=$OUT_BASE/$TOKEN
HEAD_LOCAL=$LAND/$B.head.json.gz
HEAD_REMOTE=$REPLICA_ROOT/builds/land-$B-$S8-$TOKEN/head.json.gz
COLLECT=$LAND/$B.live_db
echo "[land] live-db on Slurm for merged tip $S8 (branch $B, main ${MAIN:0:8}): replica root $REPLICA_ROOT on $HOST; expected skips $EXPECT_SKIPPED; submission $TOKEN"
if [ -n "$ACCEPT_C1" ]; then
  echo "[land] live-db: ALLOWANCE DECLARED (LAND_ACCEPT_C1_CLASS) -- a check equal modulo the certified C1 same-stamp class alone will be accepted: $ACCEPT_C1"
fi
if [ -n "$RTREE" ]; then
  [[ "$RTREE" =~ ^[0-9a-f]{40}$ ]] || die "LIVE_DB_REPLICA_TREE must be a 40-hex sha (got '$RTREE')"
  # the dispatcher must know the pin: `contract` is answered before any work by a script of this generation; a script that predates the
  # 9th positional answers `unknown mode`, one that predates the 10th answers land/2 (REPLICA_WORK_ROOT keeps the probe's stray work
  # directory, if it makes one, out of $HOME)
  [ -s "$DISPATCHER" ] || die "no dispatcher at $DISPATCHER (LIVE_DB_DISPATCHER; default this directory's replica.sbatch)"
  CONTRACT=$(REPLICA_WORK_ROOT=${TMPDIR:-/tmp} bash "$DISPATCHER" contract 2>/dev/null | tail -n 1)
  [[ "$CONTRACT" == "land/3 "* && "$CONTRACT" == *" REPLICA_TREE ACCEPT_C1_CLASS" ]] \
    || die "the dispatcher $DISPATCHER does not carry the land/3 contract (REPLICA_TREE and ACCEPT_C1_CLASS as the land mode's 9th and 10th positionals): answered '${CONTRACT:-nothing}'. A recovery submits the chain's own replica.sbatch, never an older tree's: a script without the positionals runs the tip's code on the replica side and publishes before the identity check, or drops the declaration. Nothing taken, nothing shipped"
  DISPATCH_SHA=$(sha256sum "$DISPATCHER" | cut -d' ' -f1)
  echo "[land] live-db: RECOVERY -- the replica side runs the code of tree ${RTREE:0:12} (LIVE_DB_REPLICA_TREE); the tests run from the tip $S8; the dispatcher is the chain's own $DISPATCHER (sha256 ${DISPATCH_SHA:0:12}; $CONTRACT)"
fi

# 1. the head (cursor-free; defer once on exit 6)
stamp head start
cycle_stamp() { "$PY" -c 'import json,sys
try: print(json.load(open(sys.argv[1])).get("last_completed_cycle_utc") or "")
except Exception: print("")' "$REG/state/harvest_cycle.json" 2>/dev/null; }
take_head() { "$PY" -m sisteer.registry.cli --root "$REG" replica head --out "$HEAD_LOCAL" --wait-min "$WAIT_MIN" > "$LAND/$B.head.log" 2>&1; }
rm -f "$HEAD_LOCAL"
take_head; hrc=$?
if [ $hrc -eq 6 ]; then
  before=$(cycle_stamp)
  echo "[land] live-db: no cursor-free boundary within $WAIT_MIN min ($(tail -n 1 "$LAND/$B.head.log")) -- deferring until the harvester's next completed cycle (bound $DEFER_MIN min)"
  deadline=$(( $(date +%s) + DEFER_MIN * 60 ))
  while [ "$(cycle_stamp)" = "$before" ] && [ "$(date +%s)" -lt "$deadline" ]; do sleep "${LIVE_DB_DEFER_POLL_S:-20}"; done
  [ "$(cycle_stamp)" != "$before" ] || die "the harvester completed no cycle within $DEFER_MIN min; no replayable head (head log: $LAND/$B.head.log)"
  echo "[land] live-db: a cycle completed ($(cycle_stamp)); retrying the head once"
  take_head; hrc=$?
fi
[ $hrc -eq 0 ] && [ -s "$HEAD_LOCAL" ] || die "replica head failed (rc=$hrc): $(tail -n 2 "$LAND/$B.head.log" | tr '\n' ' ')"
HEAD_LINE=$("$PY" - "$HEAD_LOCAL" 2>/dev/null <<'PYH' || true
import gzip, json, sys
d = json.load(gzip.open(sys.argv[1])); p = d["projection"]
print(f"n_files={d['n_files']} allowlist={d['allowlist_sha256'][:12]} projection={p['sha256'][:12]} counts={p['counts']} taken={d['taken_utc']} "
      f"waited_s={d.get('waited_s')} schema={d['schema_version']}")
PYH
)
echo "[land] live-db head: ${HEAD_LINE:-$(tail -n 1 "$LAND/$B.head.log")}"
stamp head end

# 2. the mirror sync, AFTER the head: a pass that STARTS after the head, verified from its own record (header step 2)
stamp mirror-sync start
if [ "$MIRROR_UNIT" != "none" ]; then
  TAKEN=$("$PY" -c 'import gzip, json, sys; print(json.load(gzip.open(sys.argv[1]))["taken_utc"])' "$HEAD_LOCAL" 2>/dev/null)
  [ -n "$TAKEN" ] || die "the head $HEAD_LOCAL carries no taken_utc"
  # the unit's own stamps, microsecond UTC (`show` prints Key=Value in its own order: parsed by key, never by position)
  unit_read() {
    local out; out=$(systemctl --user show --timestamp=us+utc -p ActiveState,Result,ExecMainStartTimestamp,ExecMainExitTimestamp "$MIRROR_UNIT" 2>&1) \
      || die "systemctl --user show $MIRROR_UNIT failed: $out"
    U_STATE=$(sed -n 's/^ActiveState=//p' <<<"$out"); U_RESULT=$(sed -n 's/^Result=//p' <<<"$out")
    U_START=$(sed -n 's/^ExecMainStartTimestamp=//p' <<<"$out"); U_EXIT=$(sed -n 's/^ExecMainExitTimestamp=//p' <<<"$out")
  }
  running() { case "$U_STATE" in activating|active|deactivating|reloading) return 0 ;; *) return 1 ;; esac; }
  iso() { [ -n "$1" ] && date -u -d "${1#* }" +%FT%T.%6NZ 2>/dev/null; }   # "Wed 2026-10-07 00:50:00.976882 UTC" -> ISO; empty -> rc 1
  utc6() { date -u +%FT%T.%6NZ; }
  MIRROR_DEADLINE=$(( $(date +%s) + MIRROR_WAIT_S ))
  mirror_left() { echo $(( MIRROR_DEADLINE - $(date +%s) )); }
  mirror_timeout() { echo "[land] live-db REFUSED: no record-log mirror pass that started after the head (taken $TAKEN) completed within ${MIRROR_WAIT_S}s -- $*; nothing shipped, nothing submitted"; exit 4; }
  # (1) a pass running at entry took its census before the head: wait for it to end (the unit idle, or a newer start stamp)
  unit_read
  if running; then
    S0=$U_START
    echo "[land] live-db: a mirror pass is running (started $(iso "$S0" || echo '?'); the head was taken $TAKEN) -- waiting for it to end"
    while running && [ "$U_START" = "$S0" ]; do
      [ "$(mirror_left)" -gt 0 ] || mirror_timeout "the pass started $(iso "$S0" || echo '?') is still running"
      sleep "$MIRROR_POLL_S"; unit_read
    done
    echo "[land] live-db: stale pass ended $(iso "$U_EXIT" || echo "by $(utc6) (a newer pass started $(iso "$U_START" || echo '?'))")"
  fi
  # (2) a fresh pass run to completion, (3) verified from its own record; a pass stopped on its budget is followed by another
  while :; do
    ASKED=$(utc6); unit_read
    PRE=""   # a pass the timer started since (1): it began after the head too, and the blocking start below merges into it
    if running; then PRE=$U_START; echo "[land] live-db: the timer started a mirror pass at $(iso "$PRE" || echo '?'); waiting for it"
    else echo "[land] live-db: starting a fresh mirror pass (systemctl --user start $MIRROR_UNIT)"; fi
    left=$(mirror_left); [ "$left" -gt 0 ] || mirror_timeout "no verified pass yet"
    timeout "$left" systemctl --user start "$MIRROR_UNIT"; src=$?
    [ $src -ne 124 ] || mirror_timeout "the pass asked for at $ASKED had not ended"
    [ $src -eq 0 ] || die "the record-log mirror pass failed (systemctl --user start $MIRROR_UNIT rc=$src); the cluster would refuse a head it does not hold"
    unit_read
    # the pass just completed: its own stamps when no newer pass has begun; else the stamp read before the start (the merged pass), or
    # the moment the idle unit was asked (a lower bound on a start that a newer pass has since replaced in systemd's properties)
    if ! running && [ -n "$U_START" ]; then START_ISO=$(iso "$U_START"); END_ISO=$(iso "$U_EXIT" || utc6); HOW="ExecMainStartTimestamp"
    elif [ -n "$PRE" ] && [ "$U_START" != "$PRE" ]; then START_ISO=$(iso "$PRE"); END_ISO="$(utc6) at the latest (a newer pass started $(iso "$U_START" || echo '?'))"; HOW="ExecMainStartTimestamp read before the start"
    elif [ -z "$PRE" ] && running; then START_ISO=$ASKED; END_ISO="$(utc6) at the latest (a newer pass started $(iso "$U_START" || echo '?'))"; HOW="the idle unit was asked at this stamp"
    else die "systemctl --user start $MIRROR_UNIT returned 0 but the unit reads ${U_STATE:-?} with start stamp '${U_START}' (asked $ASKED${PRE:+; a pass was running since $(iso "$PRE" || echo '?')}): the pass cannot be identified, nothing shipped"; fi
    [ -n "$START_ISO" ] || die "the mirror pass's start stamp '${U_START:-$PRE}' is not a systemd us+utc timestamp; nothing shipped"
    MV=$("$PY" - "$REG" "$MIRROR_DEST" "$HEAD_LOCAL" "$TAKEN" "$START_ISO" <<'PYM' 2>&1
import hashlib, os, sys
from datetime import datetime
from sisteer.registry import mirrors as M
from sisteer.registry import records as R
from sisteer.registry import replica as RP
root, dest, head_path, taken, start = sys.argv[1:6]
iso = lambda s: datetime.fromisoformat(str(s).replace("Z", "+00:00"))
if not iso(start) > iso(taken):
    print(f"NOT_AFTER the pass started {start}, not after the head taken {taken}"); sys.exit(0)
st = M.read_state(root)
if not st:
    print(f"NO_RECORD no readable {M.state_path(root)}"); sys.exit(0)
if iso(st["utc"]) < iso(start):
    print(f"NO_RECORD {M.state_path(root)} was written {st['utc']}, before the pass started {start}: not this pass's record"); sys.exit(0)
rows = st.get("destinations") or []
row = next((r for r in rows if r.get("destination") == dest), None)
if row is None:
    print(f"NO_RECORD no row for destination {dest!r} in {M.state_path(root)} (rows: {[r.get('destination') for r in rows]})"); sys.exit(0)
if not row.get("switched"):
    kind = "BUDGET" if row.get("stopped") == "budget" and not row.get("error") else "NOT_SWITCHED"
    print(f"{kind} {dest} did not switch to {row.get('manifest')}: stopped={row.get('stopped')} error={row.get('error')} "
          f"missing={row.get('missing')} corrupt={row.get('corrupt')}"); sys.exit(0)
# the manifest on disk is this pass's only while its digest is the one the pass acked: names repeat between ingests and a newer pass
# replaces the file
ack = str(row.get("ack_record") or "")
ack = ack if os.path.exists(ack) else (ack + ".gz" if ack and os.path.exists(ack + ".gz") else "")
if not ack:
    print(f"NO_RECORD the switched {dest} row names no readable mirror_ack record ({row.get('ack_record')!r}); the manifest cannot be bound to the pass"); sys.exit(0)
acked = [rec for _, rec, _ in R.read_record_file(ack, compute_sha=False)]
pay = (acked[0].get("payload") or {}) if acked else {}
mpath = os.path.join(root, M.MANIFEST_DIR, str(row["manifest"]))
try:
    with open(mpath, "rb") as fh:   # ONE read: the bytes hashed are the bytes parsed (a rewrite between two opens could straddle them)
        data = fh.read()
except OSError:
    data = b""
got = hashlib.sha256(data).hexdigest() if data else ""
if pay.get("manifest_name") != row["manifest"] or not pay.get("checksums_sha256") or got != pay["checksums_sha256"]:
    print(f"SUPERSEDED the manifest {row['manifest']} on disk (sha12 {got[:12] or 'absent'}) is not the one the pass acked "
          f"({str(pay.get('manifest_name'))} sha12 {str(pay.get('checksums_sha256'))[:12]}, {ack}): a newer pass has rewritten it"); sys.exit(0)
man = {}
for line in data.decode("utf-8").splitlines():
    if line:
        sha, _, rel = line.partition("  ")
        man[rel] = sha
head = RP.read_head(head_path)
files = head["files"]
missing = [u for u, s, _ in files if u not in man]
differing = [u for u, s, _ in files if s != RP.EMPTY_SHA and u in man and man[u] != s]
if missing or differing:
    print(f"BEHIND the manifest {row['manifest']} the pass switched to lacks {len(missing)} and differs on {len(differing)} of the "
          f"{len(files)} files the head names: " + ", ".join((missing + differing)[:3])); sys.exit(0)
n_sha = sum(1 for _, s, _ in files if s != RP.EMPTY_SHA)
print(f"COVERED manifest {row['manifest']} (sha12 {got[:12]}, acked) holds all {len(files)} files the head names, {n_sha} with the head's sha "
      f"(record {st['utc']}, {dest} row {row.get('utc')})")
PYM
)
    case "$MV" in
      COVERED*)
        echo "[land] live-db: fresh pass started $START_ISO ended $END_ISO (${U_RESULT:-?}; $HOW)"
        echo "[land] live-db: covers head: yes (started after taken_utc $TAKEN; ${MV#COVERED })"; break ;;
      BUDGET*)
        echo "[land] live-db: fresh pass started $START_ISO ended $END_ISO -- ${MV#BUDGET } -- it resumes on the next pass; starting one" ;;
      SUPERSEDED*)
        echo "[land] live-db: fresh pass started $START_ISO ended $END_ISO -- ${MV#SUPERSEDED }; verifying the pass that rewrote it" ;;
      *) die "the mirror pass started $START_ISO ended $END_ISO does not cover the head (taken $TAKEN): $MV" ;;
    esac
  done
fi
stamp mirror-sync end

# 3. the tree and the venv on the cluster
stamp tree start
[ -x "$MIRROR_REPOS" ] && "$MIRROR_REPOS" "$MIRROR_REPOS_NAME" > "$LAND/$B.mirror-repos.log" 2>&1 || echo "[land] live-db: mirror-repos.sh absent or failed (see $LAND/$B.mirror-repos.log); the tree export may refuse"
# the archived branches and tags of the worktree's repository (shared by every worktree): a note on a problem -- the replica side runs
# the tree's code only, and the gate completes a reused tree against the mirror before it runs the suite
HOUT=$("$MIRROR_HISTORY" "$WT" "ssh://$HOST$MIRROR" 2>&1); hrc=$?
[ $hrc -eq 0 ] || echo "[land] live-db: mirror_history.sh reported a problem (rc $hrc): $(tail -n 1 <<<"$HOUT"); the exported tree may lack archived branches or tags"
TREE=$CL_TREES/$SHA
TOUT=$(rsh bash -s -- "$SHA" "$MIRROR" "$TREE" < "$HERE/tree_export.sh" 2>&1); trc=$?
echo "[land] live-db: $(tail -n 1 <<<"$TOUT")"
[ $trc -eq 0 ] || die "tree export of $S8 on $HOST failed: $TOUT"
PH=$(git -C "$WT" show "$SHA:pyproject.toml" 2>/dev/null | sha256sum | cut -d' ' -f1) || die "no pyproject.toml at $SHA"
LH=$(git -C "$WT" show "$SHA:uv.lock" 2>/dev/null | sha256sum | cut -d' ' -f1) || die "no uv.lock at $SHA"
H12=$(printf '%s\n%s\n' "$PH" "$LH" | sha256sum | cut -c1-12)
VENV=$(site_env_gate_venv "$SHARED" "$MIRROR_REPOS_NAME" "$H12")
rsh test -s "$VENV/.gate-ready.json" || die "the gate venv $VENV is not ready on $HOST (gate_venv.sbatch builds it; the gate of step 2 does so)"
if [ -n "$RTREE" ]; then
  rsh test -d "$CL_TREES/$RTREE/src/sisteer/registry" || die "the replica tree ${RTREE:0:12} (LIVE_DB_REPLICA_TREE) is not exported on $HOST at $CL_TREES/$RTREE (tree_export.sh; recover.sh exports it)"
  # the declaration reaches the MODULE that folds and checks, and in a recovery that is the pinned tree's: a tree that predates the
  # allowance knows no --accept-c1-class and would die on the argument after a full submission -- refuse by name here
  if [ -n "$ACCEPT_C1" ]; then
    rsh grep -q -- "--accept-c1-class" "$CL_TREES/$RTREE/src/sisteer/registry/replica.py" \
      || die "the replica tree ${RTREE:0:12} (LIVE_DB_REPLICA_TREE) predates the C1 allowance: its sisteer/registry/replica.py knows no --accept-c1-class, so the declaration (LAND_ACCEPT_C1_CLASS) cannot reach the code that folds and checks. Land with the tip's code (no pin), or pin a tree that carries the allowance; nothing submitted"
  fi
fi
stamp tree end

# 4. ship the head and the pin into this submission's own directories (created without -p: an existing one refuses)
stamp ship start
rsh mkdir -p "$REPLICA_ROOT/builds" "$REPLICA_ROOT/pins" "$OUT_BASE" "$CL_LOGS" || die "cannot create $REPLICA_ROOT/builds, $REPLICA_ROOT/pins, $OUT_BASE or $CL_LOGS on $HOST"
rsh mkdir "$(dirname "$HEAD_REMOTE")" "$OUT_REMOTE" || die "the submission's directories $(dirname "$HEAD_REMOTE") and $OUT_REMOTE could not be created on $HOST (one exists: the token $TOKEN is taken; nothing submitted)"
rsync -q --checksum "$HEAD_LOCAL" "$HOST:$HEAD_REMOTE" || die "cannot ship the head to $HOST:$HEAD_REMOTE"
if [ -n "$RTREE" ]; then
  # the recovery's dispatcher rides beside the head, in the submission's own directory; what the cluster holds must be what was probed
  DISPATCH_REMOTE=$(dirname "$HEAD_REMOTE")/replica.sbatch
  rsync -q --checksum "$DISPATCHER" "$HOST:$DISPATCH_REMOTE" || die "cannot ship the dispatcher to $HOST:$DISPATCH_REMOTE"
  got=$(rsh sha256sum "$DISPATCH_REMOTE" 2>/dev/null | cut -d' ' -f1)
  [ -n "$got" ] && [ "$got" = "$DISPATCH_SHA" ] || die "the shipped dispatcher $HOST:$DISPATCH_REMOTE reads back as sha256 ${got:-nothing}, not the chain's ${DISPATCH_SHA:0:12}; nothing submitted"
  echo "[land] live-db: shipped the chain's dispatcher (sha256 ${DISPATCH_SHA:0:12}) beside the head: $DISPATCH_REMOTE"
fi
if [ -s "$PIN_SRC" ] && ! rsh test -s "$REPLICA_ROOT/pins/l1_assembled_pin.jsonl"; then
  rsync -q "$PIN_SRC" "$HOST:$REPLICA_ROOT/pins/l1_assembled_pin.jsonl" && echo "[land] live-db: shipped the paper pin to $REPLICA_ROOT/pins/"
fi
stamp ship end

# 5. submit, 6. watch, then the terminal state from sacct
stamp job start
# the sbatch script: a plain landing submits the tip's own replica.sbatch (dispatcher and module from one tree); a recovery submits the
# chain's dispatcher shipped in step 4 -- never the pinned tree's script, which may predate the REPLICA_TREE positional (header)
SBATCH_SCRIPT=${RTREE:+$DISPATCH_REMOTE}; SBATCH_SCRIPT=${SBATCH_SCRIPT:-$TREE/scripts/registry/landing/replica.sbatch}
# the land/3 positionals: eight always; the 9th (REPLICA_TREE) in a recovery, or TREE itself when the 10th must follow; the 10th
# (ACCEPT_C1_CLASS) only under the declaration, `printf %q`-escaped so the remote shell hands sbatch ONE word with the text verbatim
POSITIONALS=("$TREE" "$HEAD_REMOTE" "$VENV" "$B" "$REPLICA_ROOT" "$OUT_REMOTE" "$EXPECT_SKIPPED" "$HASH_MODE")
if [ -n "$RTREE" ]; then POSITIONALS+=("$CL_TREES/$RTREE"); elif [ -n "$ACCEPT_C1" ]; then POSITIONALS+=("$TREE"); fi
[ -z "$ACCEPT_C1" ] || POSITIONALS+=("$(printf '%q' "$ACCEPT_C1")")
JOB=$(rsh sbatch --parsable -J registry-replica --dependency=singleton --time="$JOB_TIME" --output="$CL_LOGS/%x_%j.out" "$SBATCH_SCRIPT" land "${POSITIONALS[@]}" 2>&1 | tail -n 1 | cut -d';' -f1)
[[ "$JOB" =~ ^[0-9]+$ ]] || die "sbatch of replica.sbatch land failed: $JOB"
echo "[land] live-db job $JOB (registry-replica, singleton, $JOB_TIME, tree $S8${RTREE:+, replica code from tree ${RTREE:0:8}, the chain dispatcher ${DISPATCH_SHA:0:12}}${ACCEPT_C1:+, C1 allowance declared}) submitted $(utc); out $HOST:$OUT_REMOTE"
echo "$(utc) live-db $B $S8 JOB=$JOB TOKEN=$TOKEN OUT=$OUT_REMOTE" >> "$STATE/jobs.log"
"$WATCH" --slurm "$JOB" --host "$HOST" --poll 30 --stall-min 40 --max-hours "$MAX_HOURS" 2>&1 | sed 's/^/[land] live-db watch: /'; wrc=${PIPESTATUS[0]}
[ "$wrc" -eq 0 ] || echo "[land] live-db: watch-job.sh exited $wrc (a timeout leaves the job to the operator: squeue -n registry-replica); the state below decides"
# the watcher exits 0 for DONE and FAILED alike: the job's state is READ from the accounting, and only COMPLETED passes (step 8)
JOB_STATE=""
for _ in 1 2 3 4; do
  JOB_STATE=$(rsh "sacct -X -n -P -o State -j $JOB" 2>/dev/null | head -n 1 | cut -d' ' -f1 | tr -d '[:space:]')
  [ -n "$JOB_STATE" ] && break
  sleep "$STATE_POLL_S"
done
echo "[land] live-db: job $JOB state ${JOB_STATE:-unknown} $(utc)"
stamp job end

# 7. collect into a fresh directory, 8. verdict
stamp collect start
rm -rf "$COLLECT" && mkdir -p "$COLLECT" || die "cannot recreate $COLLECT"
# --checksum: a same-size rewrite inside one mtime tick hides under rsync's size+mtime quick check (the trap the battery driver hit)
rsync -aq --checksum "$HOST:$OUT_REMOTE/" "$COLLECT/" || echo "[land] live-db: cannot collect $HOST:$OUT_REMOTE (the job wrote no results? state ${JOB_STATE:-unknown})"
[ -s "$COLLECT/junit.xml" ] && cp "$COLLECT/junit.xml" "$LAND/$B.junit.xml"
[ -s "$COLLECT/pytest.log" ] && cp "$COLLECT/pytest.log" "$LAND/$B.pytest.log"
CURRENT=$(rsh cat "$REPLICA_ROOT/CURRENT" 2>/dev/null | tr -d '[:space:]')
VERDICT=$("$PY" - "$COLLECT" "$EXPECT_SKIPPED" "$REG/state/replica_check.json" "$B" "$SHA" "$JOB" "$HEAD_LOCAL" "${JOB_STATE:-}" "$TOKEN" "$OUT_REMOTE" "$CURRENT" \
          "$HOST:$CL_LOGS/registry-replica_$JOB.out" "$RTREE" "$DISPATCH_SHA" "$DISPATCH_REMOTE" "$ACCEPT_C1" <<'PYV'
import json, os, sys, time
from sisteer.registry import replica as RP
d, expect, state_path, branch, sha, job, head_path, job_state, token, out_remote, current, job_log, rtree, dispatch_sha, dispatch_remote, accept_c1 = sys.argv[1:17]
expect = int(expect)
want_built = rtree or sha      # recovery: the replica side ran the pinned tree's code; otherwise the tip's


def load(name):
    try:
        with open(os.path.join(d, name), "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


chk, summ = load("check.json"), load("summary.json")
head = RP.read_head(head_path)
want_head = RP.head_sha256(head)
problems = []
# the results: both files present and both naming THIS job, THIS tip and THIS head
identity_problems = []
for name, doc in (("check.json", chk), ("summary.json", summ)):
    if doc is None:
        identity_problems.append(f"no {name} under {out_remote}: the job left no {'verdict' if name == 'check.json' else 'summary'}")
        continue
    ident = doc.get("identity") or {}
    if not ident:
        identity_problems.append(f"{name} carries no identity (a result of another tree's code?)")
        continue
    if str(ident.get("job_id")) != job:
        identity_problems.append(f"{name} names job {ident.get('job_id')!r}, not the submitted job {job}")
    if ident.get("built_by_tree") != want_built:
        identity_problems.append(f"{name} names tree {str(ident.get('built_by_tree'))[:12]!r}, not the "
                                 + (f"pinned replica tree {want_built[:12]} (LIVE_DB_REPLICA_TREE)" if rtree else f"tip {sha[:12]}"))
    if ident.get("head_sha256") != want_head:
        identity_problems.append(f"{name} names head {str(ident.get('head_sha256'))[:12]!r}, not the head taken for this landing ({want_head[:12]})")
# the job: a terminal COMPLETED state, read from the accounting (the watcher's exit code cannot tell DONE from FAILED)
job_ok = job_state == "COMPLETED"
if not job_ok:
    problems.append(f"job {job} state {job_state or 'unknown'!r}, not COMPLETED (its log: {job_log})")
if rtree and summ is not None:
    # the tests' tree is the TIP's even in a recovery: summary.tree is the exported tree path, whose last component is its sha
    tests_tree = os.path.basename(str(summ.get("tree") or "").rstrip("/"))
    if tests_tree != sha:
        identity_problems.append(f"summary.json's tests tree is {tests_tree[:12] or 'unnamed'!r}, not the tip {sha[:12]} (a recovery runs the tests from the tip)")
problems.extend(identity_problems)
chk = chk or {}
summ = summ or {}
counts = summ.get("counts") or {}
equal = bool(chk.get("equal"))
c1_class = bool(chk.get("c1_class"))
n_movers = chk.get("n_movers")
columns_differing = dict(chk.get("columns_differing") or {})
rc = summ.get("pytest_rc")
skipped = counts.get("skipped")
status = summ.get("status")
published = summ.get("published")
# the check: strict equality, or -- ONLY under the declaration (header: the cutover allowance, D136) -- the certified C1 same-stamp class
# alone; any other column differing, a one-sided row or a mover outside the pair reads c1_class false and refuses as before
accepted_as = RP.CHECK_ACCEPTED_EQUAL if equal else (RP.CHECK_ACCEPTED_C1_CLASS if (c1_class and accept_c1) else None)
if accepted_as is None:
    diff_text = str(chk.get("reason") or "projection differs: " + json.dumps(columns_differing or {k: v.get("rows") for k, v in (chk.get("diff") or {}).get("columns", {}).items()}))
    if c1_class:
        diff_text += f" (the C1 same-stamp class, {n_movers} movers -- NOT declared: no allowance without LAND_ACCEPT_C1_CLASS)"
    elif accept_c1 and chk:
        diff_text += " (not the C1 class alone: the declaration does not cover it)"
    problems.append("replica check UNEQUAL: " + diff_text)
# the job's own record must agree with the rule applied here: a generation published under an allowance this grading host did not declare, or
# without one when the job says it used one, is refused by name
if summ and summ.get("check_accepted_as") not in (None, accepted_as):
    problems.append(f"the job accepted its check as {summ.get('check_accepted_as')!r}, not as {accepted_as!r} here (declaration here: {bool(accept_c1)})")
if rc != 0:
    problems.append(f"pytest rc={rc}")
if skipped != expect:
    problems.append(f"{skipped} skipped, expected {expect}")
if status != "PASS":
    problems.append(f"the job's own verdict is {status!r}" + (f" ({summ.get('reason')})" if summ.get("reason") else ""))
# the generation: published by this job AND what CURRENT names now (an operator's rollback or another writer in between refuses)
if not published:
    problems.append("no generation published: " + str(summ.get("publish_refused") or summ.get("reason") or "summary.published is empty"))
elif current != published:
    problems.append(f"CURRENT names {current or 'nothing'!r}, not the generation {published!r} this job published")
ok = not problems
# the headline the operator reads first: results that are THIS job's explain its failure (an unequal check or red tests exit non-zero,
# so the job reads FAILED too); results that are not this job's are the failure
if ok:
    headline = "pass"
elif identity_problems:
    headline = "job" if not job_ok else "identity"
elif accepted_as is None:
    headline = "unequal"
elif rc != 0 or skipped != expect:
    headline = "tests"
elif not published or current != published:
    headline = "publish"
else:
    headline = "job"
rec = {"schema": "replica_check_state/2", "headline": headline, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "equal": equal, "ok": ok, "problems": problems,
       "branch": branch, "tip": sha, "job": job, "job_state": job_state or None, "submission": token, "out_remote": out_remote,
       "replica_tree": rtree or None,
       # the sbatch script the job ran: the tip's own in a plain landing; in a recovery the chain's, shipped beside the head
       "dispatcher": {"source": "chain", "sha256": dispatch_sha, "path": dispatch_remote} if rtree else {"source": "tip", "sha256": None, "path": None},
       "head": {"n_files": head.get("n_files"), "taken_utc": head.get("taken_utc"), "allowlist_sha12": str(head.get("allowlist_sha256", ""))[:12],
                "projection_sha12": str(head["projection"]["sha256"])[:12], "head_sha12": want_head[:12]},
       "replica": {"projection_sha12": (chk.get("replica") or {}).get("sha12"), "counts": (chk.get("replica") or {}).get("counts")},
       "projection_version": chk.get("projection_version"), "generation": published, "current_generation": current or None,
       "parent_generation": summ.get("parent_generation"), "schema_version": head.get("schema_version"),
       "pytest": {"rc": rc, "counts": counts, "expected_skipped": expect, "status": status}, "reason": chk.get("reason") or summ.get("reason"),
       # the cutover allowance (D136): what the check read, what rule passed it, the declaration it passed under
       "c1_class": c1_class, "accepted_as": accepted_as, "n_movers": n_movers, "columns_differing": columns_differing,
       "declaration": accept_c1 or None, "job_check_accepted_as": summ.get("check_accepted_as")}
os.makedirs(os.path.dirname(state_path), exist_ok=True)
tmp = state_path + ".tmp"
with open(tmp, "w", encoding="utf-8") as fh:
    json.dump(rec, fh, indent=1, sort_keys=True)
os.replace(tmp, state_path)
h, r = rec["head"], rec["replica"]
print(f"replica check: {'equal' if equal else 'UNEQUAL'} -- {rec['projection_version']}; replica {r['projection_sha12']} vs head {h['projection_sha12']} "
      f"(head {h['taken_utc']}, {h['n_files']} files); generation {published} (CURRENT {current or 'none'}); job {job} {job_state or 'unknown'}; "
      f"pytest rc={rc} counts={counts} expected_skipped={expect}")
if accepted_as == RP.CHECK_ACCEPTED_C1_CLASS:
    print(f"ACCEPTED: equal modulo the C1 same-stamp class ({n_movers} movers) -- {accept_c1}")
elif not equal and chk:
    print(f"REASON: {chk.get('reason') or 'projection differs: ' + json.dumps(columns_differing or {k: v.get('rows') for k, v in (chk.get('diff') or {}).get('columns', {}).items()})}")
for p in problems:
    print(f"REFUSED: {p}")
print(f"HEADLINE: {headline}")
sys.exit(0 if ok else 5)
PYV
); vrc=$?
echo "$VERDICT"
stamp collect end
if [ $vrc -ne 0 ]; then
  case "$(sed -n 's/^HEADLINE: //p' <<<"$VERDICT" | tail -n 1)" in
    job) echo "LIVE-DB JOB $JOB DID NOT COMPLETE (state ${JOB_STATE:-unknown}) -- nothing it may have written certifies tip $S8; NOT pushed" ;;
    identity) echo "LIVE-DB RESULT IDENTITY MISMATCH (job $JOB) -- the collected results are not this job's, tip's or head's; see $COLLECT/; NOT pushed" ;;
    unequal) echo "LIVE-DB REPLICA UNEQUAL (job $JOB) -- see $COLLECT/check.json; NOT pushed" ;;
    tests) echo "LIVE-DB TESTS RED or skips unaccounted (job $JOB) -- see $LAND/$B.pytest.log and $COLLECT/summary.json; NOT pushed" ;;
    *) echo "LIVE-DB GENERATION NOT PUBLISHED or CURRENT moved (job $JOB) -- the next landing would not start from this one's fold; see $COLLECT/summary.json; NOT pushed" ;;
  esac
  exit 5
fi
ACCEPTED_LINE=$(sed -n 's/^ACCEPTED: //p' <<<"$VERDICT" | head -n 1)
echo "[land] live-db on Slurm PASS for merged tip $S8 (job $JOB, generation $(grep -oE 'generation [^ ]+' <<<"$VERDICT" | head -n 1 | cut -d' ' -f2)${ACCEPTED_LINE:+; accepted as $ACCEPTED_LINE})"
exit 0
