#!/usr/bin/env bash
# paper_rebuild.sh — the registry-driven paper rebuild trigger (replaces paper-autobuild.sh's fingerprint list).
#
# Runs from registry-paper-rebuild.timer (every 2 min) or by hand. The trigger is the paper-input head
# (`registry paper-head` = MAX(ingest_seq) over what the paper reads: selected rows, roster, rulings, dispositions,
# accepted source revisions, withdrawals) KEYED WITH THE SCHEMA VERSION `paper-head --json` reports beside it: a migration
# that recreates a v_* object changes what the same rows mean without moving a rank (0021 retired 361 anchored rows and 7
# superseded-judgment rows that way, 2026-10-02), so a page built at this head under another schema -- or before the
# schema term was recorded (a two-field `paper_last_head`) -- is rebuilt once, at the first tick after `registry migrate`.
# The build runs only when that (head, schema) differs from the last SUCCESSFUL build's and from an already-FAILED /
# REFUSED one (a refusal needs a person, not a retry loop).
#
#   1. head + schema version + roster gaps from `registry paper-head --json`; unchanged / already failed → exit 0.
#   2. REFUSE (no build, page untouched) when any roster entry of the open round has neither a judged run nor a
#      disposition, or a ruling's evidence is indeterminate — the registry says the paper's inputs are incomplete.
#   3. Take the BUILDER's own lock (`[paper] build_lock`, ~/.local/state/paper-autobuild/lock) NON-BLOCKING (`flock -n`):
#      busy → exit 0 and try next tick. Never the harvester's lock: the build only reads the database (WAL), so the
#      harvester never waits on this script and this script never waits on the harvester.
#   4. Keep a copy of the published page, run `PAPER_BUILD_LOCK_HELD=1 paper/build.sh` from `[paper] checkout` (its own
#      gates: figures drawn by this run, BUILD DONE), then register every figure file as an artifact
#      (`registry publish-figures --checkout … --verify-dir <publish_dir>/figures`), which REFUSES an untracked figure
#      or a published file with no record. Any failure restores the previous page byte for byte and keeps the failed
#      output beside it (`<publish_dir>.failed-<utc>`, `[paper] keep_failed_builds` of them).
#   5. Write `state/paper_rebuild.RESULT` (one line: OK|FAILED|REFUSED|BUSY <head> <utc> <detail>), `paper_last_head`
#      on success, `paper_failed_head` on failure/refusal (both `<head> <utc> schema=<v> ...`). The status page shows all three.
#
# PAPER_AUTOBUILD_COMMIT=1 also commits and pushes the artefacts on the paper branch (publish-only by default, as before).
# Environment: REGISTRY_ROOT, REGISTRY_PY (python with the registry package; default: the venv of the checkout this script
# sits in), REGISTRY_PYTHONPATH (default: that checkout's src/).
set -uo pipefail

ROOT="${REGISTRY_ROOT:-$HOME/registry}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${REGISTRY_PY:-$REPO_DIR/.venv/bin/python}"
export PYTHONPATH="${REGISTRY_PYTHONPATH:-$REPO_DIR/src}${PYTHONPATH:+:$PYTHONPATH}"
reg() { "$PY" -m sisteer.registry.cli --root "$ROOT" "$@"; }

STATE="$ROOT/state"; mkdir -p "$STATE"
LAST="$STATE/paper_last_head"; FAILED="$STATE/paper_failed_head"; RESULT="$STATE/paper_rebuild.RESULT"; LOG="$STATE/paper_rebuild.log"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
result() {  # result <STATE> <head> <detail…>
  local st="$1" h="$2"; shift 2
  printf '%s %s %s %s\n' "$st" "$h" "$(now)" "$*" > "$RESULT.tmp" && mv -f "$RESULT.tmp" "$RESULT"
  echo "$(now) $st $h $*" >> "$LOG"
}

CHECKOUT="$(reg config paper.checkout)" || { result FAILED unknown "registry config unreadable"; exit 1; }
PUBLISH="$(reg config paper.publish_dir)"
LOCK="$(reg config paper.build_lock)"
KEEP="$(reg config paper.keep_failed_builds)"; KEEP="${KEEP:-3}"

# 1. the head, the schema version and the roster gaps
HEADJSON="$(reg paper-head --json)" || { result FAILED unknown "registry paper-head failed (database unreadable?)"; exit 1; }
read -r head gaps indet schema < <("$PY" -c 'import json,sys; d=json.load(sys.stdin); print(d["head"], d["roster_gaps"], d["indeterminate"], d.get("schema_version", 0))' <<<"$HEADJSON")
KEY="$head schema=$schema"
built_key() {  # built_key <file>: the (head, schema) a watermark file records, as "<head> schema=<v>" ("<head> " when it predates the term)
  [ -f "$1" ] || return 0
  awk 'NR == 1 { s = ""; for (i = 1; i <= NF; i++) if ($i ~ /^schema=/) s = $i; print $1 " " s }' "$1"
}
if [ "$(built_key "$LAST")" = "$KEY" ]; then exit 0; fi       # built already at this head under this schema
if [ "$(built_key "$FAILED")" = "$KEY" ]; then exit 0; fi     # failed/refused at this head under this schema; a person must look

# 2. refuse on incomplete inputs — no build, the published page stays exactly as it is
if [ "$gaps" != "0" ] || [ "$indet" != "0" ]; then
  names="$(HEADJSON="$HEADJSON" "$PY" -c 'import json, os; d = json.loads(os.environ["HEADJSON"]); print(", ".join("%s (%s: %s x %s)" % (g["roster_entry_id"], g["comparison_id"], g["model_id"], g["eval_id"]) for g in d["gaps"][:10]))')"
  echo "$head $(now) schema=$schema refused" > "$FAILED"
  result REFUSED "$head" "roster entries with neither a judged run nor a disposition: $gaps; indeterminate rulings: $indet; $names"
  exit 0
fi

# 3. the builder's own lock, never waited on
mkdir -p "$(dirname "$LOCK")"
exec 9>"$LOCK"
if ! flock -n 9; then result BUSY "$head" "another build holds $LOCK"; exit 0; fi

[ -d "$CHECKOUT/paper" ] || { echo "$head $(now) schema=$schema" > "$FAILED"; result FAILED "$head" "paper checkout missing: $CHECKOUT/paper"; exit 1; }

# 4. keep the current page, build, register
PREV="$PUBLISH.prev"
rm -rf "$PREV"
if [ -d "$PUBLISH" ]; then cp -a "$PUBLISH" "$PREV"; fi
restore() {  # put the previous page back; keep the failed output beside it
  local ts; ts="$(date -u +%Y%m%dT%H%M%SZ)"
  if [ -d "$PREV" ]; then
    [ -d "$PUBLISH" ] && mv "$PUBLISH" "$PUBLISH.failed-$ts"
    mv "$PREV" "$PUBLISH"
  elif [ -d "$PUBLISH" ]; then
    mv "$PUBLISH" "$PUBLISH.failed-$ts"   # there was no page before; a half-built one must not stand in for it
  fi
  ls -1d "$PUBLISH".failed-* 2>/dev/null | sort | head -n -"$KEEP" | xargs -r rm -rf
}

echo "$(now) head $head moved -> building from $CHECKOUT" >> "$LOG"
cd "$CHECKOUT/paper" || { echo "$head $(now) schema=$schema" > "$FAILED"; result FAILED "$head" "cannot enter $CHECKOUT/paper"; exit 1; }
if PAPER_BUILD_LOCK_HELD=1 ./build.sh >> "$LOG" 2>&1 && [ "$(tail -1 build.log 2>/dev/null | grep -c 'BUILD DONE')" = 1 ]; then
  sha="$(git -C "$CHECKOUT" rev-parse HEAD 2>/dev/null || echo unverified)"
  if reg publish-figures --dir "$CHECKOUT/paper/figures" --notebook-commit "$sha" --checkout "$CHECKOUT" --verify-dir "$PUBLISH/figures" >> "$LOG" 2>&1; then
    if [ "${PAPER_AUTOBUILD_COMMIT:-0}" = 1 ]; then
      git add -u -- GAPS.md build.log constitutional_adapters_figures.html constitutional_adapters_figures.ipynb paperlib/lane_watermarks.json >/dev/null 2>&1
      git add -- figures >/dev/null 2>&1
      if ! git diff --cached --quiet; then
        git commit -q -m "paper: rebuild at registry head $head ($(now))" && git push -q >/dev/null 2>&1 && echo "$(now) committed $(git rev-parse --short HEAD)" >> "$LOG"
      fi
    fi
    echo "$head $(now) schema=$schema" > "$LAST"; rm -f "$FAILED"; rm -rf "$PREV"
    result OK "$head" "built $sha from $CHECKOUT; figures registered; published $PUBLISH; schema v$schema"
    exit 0
  fi
  restore
  echo "$head $(now) schema=$schema" > "$FAILED"
  result FAILED "$head" "publish-implies-recorded refused (an untracked figure or a published file with no record); previous page restored"
  exit 1
fi
restore
echo "$head $(now) schema=$schema" > "$FAILED"
result FAILED "$head" "build failed (see $LOG); previous page restored; not retried until the head moves"
exit 1
