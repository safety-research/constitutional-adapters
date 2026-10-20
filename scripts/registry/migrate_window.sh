#!/usr/bin/env bash
# The BOUNDED LANDING WINDOW for a registry schema migration.
#
# A migration that rewrites a CHECK rebuilds its tables in place, and `db.open_or_create` refuses the live database the moment the
# code's version is ahead of the file's -- so the PR that carries the migration may merge ONLY after the live database has been
# brought to the new version, with every writer stopped for the minutes the copy takes. This script is that window, stage by stage,
# each stage refusing BY NAME and touching nothing when its precondition fails:
#
#   plan        writers stopped (every unit of WRITER_UNITS reads inactive to `systemctl --user is-active`) -> `registry migrate --plan`
#               lists EXACTLY the expected migration as pending; the rebuilt tables and their row counts are recorded
#   apply       `registry migrate` (the apply runs under the registry's write lock; a held lock refuses, exit 4)
#   verify      db version == code version, `registry migrate --plan` reports the shape matches, `PRAGMA integrity_check` == ok,
#               every rebuilt table's row count unchanged -> the VERDICT file <state>/migration-<expect>.json
#   merge-gate  exit 0 iff the verdict file says every check held for <expect> (what the landing step runs before merging the PR)
#   all         plan, apply, verify
#
# Resuming the writers is the operator's last act (`systemctl --user start <the timers>`): this script never starts a unit.
#
# usage: scripts/registry/migrate_window.sh <stage> --expect 0018_substrate_slurm [--root ~/registry] [--state DIR] [--python PY]
#          [--src <repo>/src] [--units "unit ..."]
set -euo pipefail

STAGE="${1:-}"; shift || true
ROOT="${REGISTRY_ROOT:-$HOME/registry}"
STATE="${REGISTRY_LANDING_STATE:-$HOME/.local/state/registry-landing}"
PY="${REGISTRY_PYTHON:-${REGISTRY_PY:-python3}}"                    # REGISTRY_PY / REGISTRY_PYTHONPATH: the knobs landing/window.sh reads, honoured too
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="${REGISTRY_SRC:-${REGISTRY_PYTHONPATH:-$(cd "$HERE/../.." && pwd)/src}}"
EXPECT=""
# every unit that WRITES the database or feeds its inbox (the harvester; the cluster-reads tick, whose pull writes the inbox and the
# re-activation bundles, in place of the retired registry-adapters units; the mirrors feed its
# inbox; the status page's heavy refresh reads only but is stopped with the rest so the window is quiet) -- the snapshot timer is NOT here:
# it is left stopped by standing rule and this script never touches it
WRITER_UNITS="${REGISTRY_WRITER_UNITS:-registry-harvest.timer registry-harvest.service registry-cluster-reads.timer registry-cluster-reads.service registry-mirrors.timer registry-mirrors-full.timer registry-status-heavy.timer registry-grade-daemon.service}"
while [ $# -gt 0 ]; do
  case "$1" in
    --expect) EXPECT="$2"; shift 2;;
    --root) ROOT="$2"; shift 2;;
    --state) STATE="$2"; shift 2;;
    --python) PY="$2"; shift 2;;
    --src) SRC="$2"; shift 2;;
    --units) WRITER_UNITS="$2"; shift 2;;
    -h|--help) sed -n '2,24p' "${BASH_SOURCE[0]}"; exit 0;;
    *) echo "migrate_window: unknown argument $1" >&2; exit 2;;
  esac
done
[ -n "$STAGE" ] || { echo "migrate_window: a stage is required (plan | apply | verify | merge-gate | all)" >&2; exit 2; }
[ -n "$EXPECT" ] || { echo "migrate_window: --expect <NNNN_slug> is required (the ONE migration this window lands)" >&2; exit 2; }
mkdir -p "$STATE"
VERDICT="$STATE/migration-$EXPECT.json"
REG=("$PY" -m sisteer.registry.cli --root "$ROOT")
export PYTHONPATH="$SRC${PYTHONPATH:+:$PYTHONPATH}"
DBFILE="$ROOT/registry.sqlite"
log(){ echo "[migrate-window $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

writers_active(){   # prints every unit of WRITER_UNITS that is active (empty = all stopped); a missing systemctl reads as "unknown" -> refuse
  local u out active=()
  command -v systemctl >/dev/null 2>&1 || { echo "systemctl-unavailable"; return 0; }
  for u in $WRITER_UNITS; do
    out="$(systemctl --user is-active "$u" 2>/dev/null || true)"
    case "$out" in active|activating|reloading|deactivating) active+=("$u");; esac
  done
  printf '%s\n' "${active[@]:-}"
}

require_writers_stopped(){   # <stage>: refuse (exit 5) while any writer unit is active -- nothing touched
  local active; active="$(writers_active | sed '/^$/d')"
  [ -z "$active" ] || { echo "migrate_window $1 REFUSED: writer unit(s) still active: $(echo "$active" | tr '\n' ' ')-- stop them first (systemctl --user stop <unit>); nothing touched" >&2; exit 5; }
}

sqlq(){   # <sql> -> the first column of its first row, read from the database read-only
  "$PY" - "$DBFILE" "$1" <<'PY'
import sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
print(con.execute(sys.argv[2]).fetchone()[0])
PY
}

row_counts(){   # JSON {table: rows} for the tables the plan rebuilds (arg: comma list)
  "$PY" - "$DBFILE" "$1" <<'PY'
import json, sqlite3, sys
con = sqlite3.connect(f"file:{sys.argv[1]}?mode=ro", uri=True)
print(json.dumps({t: con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in sys.argv[2].split(",") if t}, sort_keys=True))
PY
}

db_version(){ sqlq "SELECT COALESCE(MAX(version), 0) FROM schema_migration"; }
integrity(){ sqlq "PRAGMA integrity_check"; }
code_version(){ "$PY" -c "from sisteer.registry.migrations import code_version; print(code_version())"; }

stage_plan(){
  local plan pending rebuilt
  require_writers_stopped plan
  [ -f "$DBFILE" ] || { echo "migrate_window plan REFUSED: no database at $DBFILE" >&2; exit 6; }
  plan="$("${REG[@]}" migrate --plan 2>&1 || true)"
  pending="$(printf '%s\n' "$plan" | sed -n 's/^schema version [0-9]*; code [0-9]*; pending: //p')"
  if [ "$pending" != "$EXPECT" ]; then
    echo "migrate_window plan REFUSED: pending migration(s) are '$pending', expected exactly '$EXPECT' (registry migrate --plan):" >&2
    printf '%s\n' "$plan" >&2
    exit 7
  fi
  rebuilt="$(printf '%s\n' "$plan" | sed -n 's/^rebuild table \([a-z_]*\): .*/\1/p' | paste -sd, -)"
  log "plan OK: pending exactly $EXPECT; rebuilds: ${rebuilt:-none}"
  printf '%s\n' "$plan"
  "$PY" - "$STATE/window-$EXPECT.plan.json" "$EXPECT" "$rebuilt" "$(db_version)" "$(row_counts "$rebuilt")" <<'PY'
import json, sys, datetime
out, expect, rebuilt, version, counts = sys.argv[1:6]
json.dump({"expect": expect, "rebuilt_tables": [t for t in rebuilt.split(",") if t], "version_before": int(version),
           "row_counts_before": json.loads(counts), "planned_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()},
          open(out, "w"), indent=1, sort_keys=True)
PY
  log "wrote $STATE/window-$EXPECT.plan.json"
}

stage_apply(){
  require_writers_stopped apply
  [ -f "$STATE/window-$EXPECT.plan.json" ] || { echo "migrate_window apply REFUSED: run the plan stage first (no $STATE/window-$EXPECT.plan.json)" >&2; exit 8; }
  log "applying $EXPECT under the registry's write lock"
  "${REG[@]}" migrate --lock-timeout 0 || { rc=$?; echo "migrate_window apply FAILED (registry migrate exit $rc); the database is as it was (one transaction)" >&2; exit "$rc"; }
}

stage_verify(){
  local want have plan rebuilt before after ok=1 shape=0 integ
  want="$(code_version)"; have="$(db_version)"
  plan="$("${REG[@]}" migrate --plan 2>&1 || true)"
  printf '%s\n' "$plan" | grep -q '^shape: matches the code' && shape=1
  integ="$(integrity)"
  rebuilt="$("$PY" -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['rebuilt_tables']))" "$STATE/window-$EXPECT.plan.json" 2>/dev/null || true)"
  before="$("$PY" -c "import json,sys; print(json.dumps(json.load(open(sys.argv[1]))['row_counts_before'], sort_keys=True))" "$STATE/window-$EXPECT.plan.json" 2>/dev/null || echo '{}')"
  after="$(row_counts "$rebuilt")"
  [ "$have" = "$want" ] || { echo "migrate_window verify: db version $have != code version $want" >&2; ok=0; }
  [ "$shape" = 1 ] || { echo "migrate_window verify: the shape still differs from the code's schema" >&2; printf '%s\n' "$plan" >&2; ok=0; }
  [ "$integ" = ok ] || { echo "migrate_window verify: PRAGMA integrity_check said '$integ'" >&2; ok=0; }
  [ "$before" = "$after" ] || { echo "migrate_window verify: row counts moved: before $before after $after" >&2; ok=0; }
  "$PY" - "$VERDICT" "$EXPECT" "$have" "$want" "$shape" "$integ" "$before" "$after" "$ok" <<'PY'
import json, sys, datetime
out, expect, have, want, shape, integ, before, after, ok = sys.argv[1:10]
json.dump({"expect": expect, "db_version": int(have), "code_version": int(want), "shape_matches": shape == "1", "integrity": integ,
           "row_counts_before": json.loads(before), "row_counts_after": json.loads(after), "ok": ok == "1",
           "verified_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}, open(out, "w"), indent=1, sort_keys=True)
PY
  if [ "$ok" = 1 ]; then log "verify OK: version $have, shape matches, integrity ok, row counts unchanged -> $VERDICT"; else
    echo "migrate_window verify FAILED -> $VERDICT (ok false); do NOT merge" >&2; exit 9; fi
}

stage_merge_gate(){
  [ -f "$VERDICT" ] || { echo "migrate_window merge-gate REFUSED: no verdict at $VERDICT -- the window has not run (plan, apply, verify)" >&2; exit 10; }
  "$PY" - "$VERDICT" "$EXPECT" "$(code_version)" <<'PY' || exit 11
import json, sys
v = json.load(open(sys.argv[1])); expect, want = sys.argv[2], int(sys.argv[3])
problems = []
if v.get("expect") != expect: problems.append(f"verdict is for {v.get('expect')!r}, not {expect!r}")
if not v.get("ok"): problems.append("verdict ok is false")
if int(v.get("db_version", -1)) != want: problems.append(f"verdict db_version {v.get('db_version')} != code version {want}")
if not v.get("shape_matches"): problems.append("shape did not match")
if v.get("integrity") != "ok": problems.append(f"integrity {v.get('integrity')!r}")
if v.get("row_counts_before") != v.get("row_counts_after"): problems.append("row counts moved")
if problems:
    print("migrate_window merge-gate REFUSED: " + "; ".join(problems), file=sys.stderr); sys.exit(1)
print(f"merge-gate OK: {expect} landed on the live database at version {want} (verified {v.get('verified_utc')})")
PY
}

case "$STAGE" in
  plan) stage_plan;;
  apply) stage_apply;;
  verify) stage_verify;;
  merge-gate) stage_merge_gate;;
  all) stage_plan; stage_apply; stage_verify;;
  *) echo "migrate_window: unknown stage $STAGE (plan | apply | verify | merge-gate | all)" >&2; exit 2;;
esac
