#!/bin/bash
# recover.sh <revert-or-fix branch> <title> <body-file> [--rollback] [--dry-run]   |   recover.sh --plan
# The Slurm-only RECOVERY of a bad landing: the targeted revert-or-fix branch
# lands through the ordinary chain, but the REPLICA SIDE of its live-DB step runs the code of the CURRENT GOOD GENERATION
# (`meta.json.built_by_tree` of what CURRENT names on the cluster) instead of the merged tip's -- so a tip whose replica code is
# the very thing being reverted never builds the generation that certifies it -- with a NEWLY verified head (live_db_slurm.sh takes
# one for every landing). Honest time ~ 1 h with a good generation; the multi-hour rebuild (replica build) only when no kept
# generation is good.
#   1. read        the chain's DISPATCHER first (LIVE_DB_DISPATCHER, default this directory's replica.sbatch): `replica.sbatch contract`
#                  must answer `land/3 … REPLICA_TREE ACCEPT_C1_CLASS` -- the land mode's 9th and 10th positionals (the replica-side tree;
#                  the cutover declaration), which live_db_slurm.sh ships beside the head and
#                  submits in step 4 instead of ANY tree's own script (a tree older than the contract, the R3 bootstrap generation's
#                  among them, would run the tip's code on the replica side and publish before the identity check); then
#                  CURRENT and gen/<CURRENT>/meta.json on the cluster (ssh, read-only): the good generation and its built_by_tree; a
#                  dispatcher without the contract, no CURRENT, no meta, no 40-hex built_by_tree -> refuse (exit 5), nothing else runs
#   2. --rollback  a CORRUPT CURRENT: `python -m sisteer.registry.replica rollback --replica-root <root>` on the login node under the
#                  CURRENT generation's own tree and the lock-exact gate venv (a pointer compare-and-swap under the publish lock,
#                  minutes, nothing heavy) -> CURRENT = its parent; then step 1 again for the generation that is now current
#   3. tree        the good generation's tree must be exported on the cluster ($SHARED/gates/trees/<built_by_tree>); absent -> mirror
#                  (mirror-repos.sh) + tree_export.sh over ssh; a tree that cannot be exported refuses (exit 5)
#   4. land        `LIVE_DB_REPLICA_TREE=<built_by_tree> land.sh <branch> <title> <body>` in the foreground: live_db_slurm.sh ships the
#                  chain's dispatcher beside the head (sha256 read back), passes the tree to its land mode as REPLICA_TREE, and its
#                  verdict requires identity.built_by_tree == that tree while the tests ran from the tip (summary.tree); land.sh's exit
#                  status is this script's
# THE CUTOVER ALLOWANCE DOES NOT REACH A PIN THAT PREDATES IT. The module that folds and checks in a recovery
# is the PINNED tree's, so LAND_ACCEPT_C1_CLASS is honored only when that tree's sisteer/registry/replica.py knows --accept-c1-class;
# live_db_slurm.sh refuses the declaration by name before any submission otherwise. While the live database still folds the pre-C1 order,
# a recovery pinned to a generation built by a tree WITH C1's reducer but WITHOUT the allowance (the generation current at the cutover)
# therefore cannot land on the certified class: either restamp the certified keys on the live database first (the planned retirement of
# the allowance -- the live head then carries the C1 order and no allowance is needed by any tree with C1's reducer), or pin a generation
# built by a tree that carries the allowance. This is a recorded limitation of the cutover window, not a path this script papers over.
# --dry-run stops before any write (no rollback, no export, no landing) and prints what would run. Every step is stamped
# `[recover] @ <step>:<start|end> <UTC>`; the outcome is recorded in $STATE/recover/<branch>.json (utc, generation before/after, the
# tree, rolled_back, tree_present, land_rc). Knobs (env): GATE_HOST (the login host), GATE_RUSER ($USER), GATE_SHARED (the shared root),
# REPLICA_ROOT ($SHARED/registry/replica), REGISTRY_LANDING_STATE, SISTEER_REPO (the checkout whose archived branches and
# tags mirror_history.sh pushes before the export), LAND_SH (this directory's land.sh), MIRROR_REPOS_SH, MIRROR_HISTORY_SH (this directory's),
# RECOVER_PY (python3 for the JSON reads here), LIVE_DB_DISPATCHER (the replica.sbatch probed here and shipped by live_db_slurm.sh, which
# reads the same variable; default this directory's). Exit: 0 landed; 2 usage; 5 a refusal here; else land.sh's code. The login host,
# the shared root, the checkout and its clone on the cluster come from the site config's primary cluster block, or from the block of
# the cluster an exported CLUSTER_DEST names (site_env.sh beside this script); GATE_HOST, GATE_SHARED and the environment's own
# CLUSTER_* values win, and a value nothing supplies is refused by name.
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
STATE=${REGISTRY_LANDING_STATE:-$HOME/.local/state/registry-landing}
RUSER=${GATE_RUSER:-${USER:-$(id -un)}}
LAND_SH=${LAND_SH:-$HERE/land.sh}
MIRROR_REPOS=${MIRROR_REPOS_SH:-$HOME/scripts/mirror-repos.sh}
MIRROR_HISTORY=${MIRROR_HISTORY_SH:-$HERE/mirror_history.sh}
PY3=${RECOVER_PY:-python3}
DISPATCHER=${LIVE_DB_DISPATCHER:-$HERE/replica.sbatch}
plan() {
  cat <<'PLAN'
PLAN 1 read :: the chain's dispatcher answers `replica.sbatch contract` with land/3 … REPLICA_TREE ACCEPT_C1_CLASS (else refuse: an older script would run the tip's code on the replica side or drop the declaration); then ssh <host> cat <replica root>/CURRENT and gen/<CURRENT>/meta.json: the good generation and its built_by_tree (refuse without them)
PLAN 2 rollback :: only with --rollback: `python -m sisteer.registry.replica rollback --replica-root <root>` on the login node under the current generation's own tree and the lock-exact gate venv (CAS under the publish lock); then read again
PLAN 3 tree :: the good generation's tree exported on the cluster ($SHARED/gates/trees/<built_by_tree>), else mirror-repos.sh + tree_export.sh over ssh
PLAN 4 land :: LIVE_DB_REPLICA_TREE=<built_by_tree> land.sh <branch> <title> <body> in the foreground: the replica side of the live-DB step runs the good generation's code with a newly verified head; the tests run from the tip; the sbatch script is the chain's own replica.sbatch shipped beside the head (sha256 read back), never the pinned tree's
PLAN
}
ROLLBACK=0; DRY=0; ARGS=()
for arg in "$@"; do
  case "$arg" in
    --plan) plan; exit 0 ;;
    --rollback) ROLLBACK=1 ;;
    --dry-run) DRY=1 ;;
    -h|--help) awk 'NR == 1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
    *) ARGS+=("$arg") ;;
  esac
done
[ ${#ARGS[@]} -eq 3 ] || { echo "usage: recover.sh <branch> <title> <body-file> [--rollback] [--dry-run] | recover.sh --plan" >&2; exit 2; }
B=${ARGS[0]}; TITLE=${ARGS[1]}; BODY=${ARGS[2]}
# the cluster's names from the site config (site_env.sh); GATE_HOST and GATE_SHARED win over it, and land.sh inherits them
CLUSTER_HOST="${GATE_HOST:-${CLUSTER_HOST:-}}"; CLUSTER_SHARED="${GATE_SHARED:-${CLUSTER_SHARED:-}}"
# shellcheck source=site_env.sh
source "$HERE/site_env.sh"
site_env_load CLUSTER_HOST CLUSTER_SHARED SITE_REPO MIRROR_REPOS_NAME CLUSTER_REPO_MIRROR \
  || { echo "[recover] REFUSED: the site config does not supply the cluster's names (the line above names the setting)"; exit 5; }
HOST=$CLUSTER_HOST; SHARED=$CLUSTER_SHARED; MIRROR=$CLUSTER_REPO_MIRROR; REPO=$SITE_REPO; CL_TREES=$SHARED/gates/trees
REPLICA_ROOT=${REPLICA_ROOT:-$SHARED/registry/replica}
utc() { date -u +%FT%TZ; }
stamp() { echo "[recover] @ $1:$2 $(utc)"; }
die() { echo "[recover] REFUSED: $*"; record "refused: $*"; exit 5; }
rsh() { ssh -o BatchMode=yes -o ConnectTimeout=25 "$HOST" "$@"; }
CUR_BEFORE=""; CUR=""; BUILT=""; ROLLED=0; TREE_PRESENT=""; LAND_RC=""
record() {
  mkdir -p "$STATE/recover" 2>/dev/null || return 0
  "$PY3" - "$STATE/recover/$B.json" "$B" "$CUR_BEFORE" "$CUR" "$BUILT" "$ROLLED" "$TREE_PRESENT" "$LAND_RC" "$DRY" "${1:-}" <<'PY' 2>/dev/null
import json, sys, time
p, b, cb, cur, built, rolled, present, rc, dry, note = sys.argv[1:11]
rec = {"schema": "recover/1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "branch": b, "current_before": cb or None, "generation": cur or None,
       "built_by_tree": built or None, "rolled_back": rolled == "1", "tree_present": present or None, "land_rc": int(rc) if rc.isdigit() else None,
       "dry_run": dry == "1", "note": note or None}
json.dump(rec, open(p, "w"), indent=1, sort_keys=True)
PY
}
read_current() {  # CUR and BUILT from the cluster's CURRENT + meta.json (read-only)
  CUR=$(rsh "cat '$REPLICA_ROOT/CURRENT' 2>/dev/null" | tr -d '[:space:]')
  [ -n "$CUR" ] || die "no CURRENT generation under $REPLICA_ROOT on $HOST (bootstrap with replica build)"
  META=$(rsh "cat '$REPLICA_ROOT/gen/$CUR/meta.json' 2>/dev/null")
  [ -n "$META" ] || die "generation $CUR has no meta.json under $REPLICA_ROOT/gen/$CUR on $HOST"
  BUILT=$("$PY3" -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d.get("built_by_tree") or "")' <<<"$META" 2>/dev/null)
  [[ "$BUILT" =~ ^[0-9a-f]{40}$ ]] || die "generation $CUR's meta.json names no 40-hex built_by_tree (got '${BUILT:-none}'): the code that built it is unknown"
}
echo "[recover] branch $B: the replica side of the live-DB step runs the CURRENT good generation's code (replica root $REPLICA_ROOT on $HOST)$([ $DRY = 1 ] && echo ' -- DRY RUN')"
stamp read start
# the dispatcher before the cluster: a script without the land/2 contract would be shipped by live_db_slurm.sh only to be refused there
# -- after a --rollback had already moved CURRENT. Refuse here first, before anything moves.
[ -s "$DISPATCHER" ] || die "no dispatcher at $DISPATCHER (LIVE_DB_DISPATCHER; default this directory's replica.sbatch)"
CONTRACT=$(REPLICA_WORK_ROOT=${TMPDIR:-/tmp} bash "$DISPATCHER" contract 2>/dev/null | tail -n 1)
[[ "$CONTRACT" == "land/3 "* && "$CONTRACT" == *" REPLICA_TREE ACCEPT_C1_CLASS" ]] \
  || die "the dispatcher $DISPATCHER carries no land/3 contract (REPLICA_TREE and ACCEPT_C1_CLASS as the land mode's 9th and 10th positionals): answered '${CONTRACT:-nothing}'. Shipped, it would run the tip's code on the replica side or drop the declaration; nothing runs"
echo "[recover] dispatcher $DISPATCHER (sha256 $(sha256sum "$DISPATCHER" | cut -c1-12); $CONTRACT)"
read_current
CUR_BEFORE=$CUR
echo "[recover] CURRENT $CUR built by tree ${BUILT:0:12} ($(utc))"
stamp read end
if [ "$ROLLBACK" = 1 ]; then
  stamp rollback start
  # the rollback runs the CURRENT generation's own code (its tree) under the lock-exact venv of that tree's dependency files
  TREE=$CL_TREES/$BUILT
  H12=$(rsh "cat '$TREE/pyproject.toml' '$TREE/uv.lock' >/dev/null 2>&1 && { sha256sum '$TREE/pyproject.toml' | cut -d' ' -f1; sha256sum '$TREE/uv.lock' | cut -d' ' -f1; }" | sha256sum | cut -c1-12)
  VENV=$(site_env_gate_venv "$SHARED" "$MIRROR_REPOS_NAME" "$H12")
  if [ $DRY = 1 ]; then
    echo "[recover] would run: ssh $HOST env PYTHONPATH=$TREE/src $VENV/bin/python -m sisteer.registry.replica rollback --replica-root $REPLICA_ROOT"
  else
    rsh "test -s '$VENV/.gate-ready.json'" || die "the gate venv $VENV for tree ${BUILT:0:12} is not ready on $HOST (gate_venv.sbatch builds it); no rollback"
    ROUT=$(rsh "env PYTHONPATH='$TREE/src' '$VENV/bin/python' -m sisteer.registry.replica rollback --replica-root '$REPLICA_ROOT'" 2>&1); rrc=$?
    echo "$ROUT" | tail -n 3
    [ $rrc -eq 0 ] || die "replica rollback failed (rc=$rrc): $(tail -n 1 <<<"$ROUT")"
    ROLLED=1
    read_current
    echo "[recover] rolled back: CURRENT $CUR_BEFORE -> $CUR (built by tree ${BUILT:0:12})"
  fi
  stamp rollback end
fi
stamp tree start
if rsh "test -d '$CL_TREES/$BUILT/src/sisteer/registry'"; then
  TREE_PRESENT=existing; echo "[recover] tree ${BUILT:0:12} present on $HOST at $CL_TREES/$BUILT"
elif [ $DRY = 1 ]; then
  TREE_PRESENT=would-export; echo "[recover] would export tree ${BUILT:0:12} to $CL_TREES/$BUILT (mirror-repos.sh + tree_export.sh)"
else
  [ -x "$MIRROR_REPOS" ] && "$MIRROR_REPOS" "$MIRROR_REPOS_NAME" >/dev/null 2>&1 || echo "[recover] mirror-repos.sh absent or failed; tree_export verifies the sha on the mirror"
  # the archived branches and tags: a note on a problem (the replica side runs the pinned tree's code only)
  HOUT=$("$MIRROR_HISTORY" "$REPO" "ssh://$HOST$MIRROR" 2>&1); hrc=$?
  [ $hrc -eq 0 ] || echo "[recover] mirror_history.sh reported a problem (rc $hrc): $(tail -n 1 <<<"$HOUT"); the exported tree may lack archived branches or tags"
  TOUT=$(rsh bash -s -- "$BUILT" "$MIRROR" "$CL_TREES/$BUILT" < "$HERE/tree_export.sh" 2>&1); trc=$?
  echo "[recover] $(tail -n 1 <<<"$TOUT")"
  [ $trc -eq 0 ] && grep -q '^TREE ' <<<"$TOUT" || die "tree ${BUILT:0:12} could not be exported on $HOST: $TOUT"
  TREE_PRESENT=exported
fi
stamp tree end
stamp land start
if [ $DRY = 1 ]; then
  echo "[recover] would run: LIVE_DB_REPLICA_TREE=$BUILT $LAND_SH $B <title> $BODY (live_db_slurm.sh ships the dispatcher $DISPATCHER beside the head and submits it)"
  stamp land end
  record "dry run"
  exit 0
fi
echo "[recover] landing $B with LIVE_DB_REPLICA_TREE=${BUILT:0:12} (generation $CUR) $(utc)"
LIVE_DB_REPLICA_TREE=$BUILT "$LAND_SH" "$B" "$TITLE" "$BODY"; LAND_RC=$?
stamp land end
echo "[recover] land.sh exited $LAND_RC for $B (replica code from tree ${BUILT:0:12}, generation $CUR)"
record
exit "$LAND_RC"
