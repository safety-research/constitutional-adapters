#!/bin/bash
# frame_identity.sh -- the frame-identity check: the three v_paper_current counts and one sha256 over its ORDERED
# canonical-JSON rows, so a Slurm-read window on unchanged inputs can be shown to leave the paper frame byte-identical.
#
#   scripts/registry/measure/frame_identity.sh [--root DIR] [--out FILE] [--label start|end|...]
#
# Read-only (sqlite3 -readonly, a 120 s busy timeout): ONE read of v_paper_current (~30 s live), the window's one non-fold read, named
# in its log. Prints one JSON line {utc, label, in_paper, selected, rows, sha256, elapsed_s} and appends it to --out (default
# ~/.local/state/registry-landing/measure/p3/frame_identity.jsonl). Run it at the window's start and end and compare the two lines:
# equal counts and an equal sha mean the frame is byte-identical. The row sha is over `json_object(...)` of every column, ordered by
# run_id (the view's natural key), so column order and float spelling are SQLite's own on both reads.
set -euo pipefail
ROOT="${REGISTRY_ROOT:-$HOME/registry}"
OUT="$HOME/.local/state/registry-landing/measure/p3/frame_identity.jsonl"
LABEL="ad-hoc"
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift ;;
    --out) OUT="$2"; shift ;;
    --label) LABEL="$2"; shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
DB="$ROOT/registry.sqlite"
[ -f "$DB" ] || { echo "no database at $DB" >&2; exit 2; }
t0=$(date +%s.%N)
# the columns, in the view's own order, for the canonical-JSON row
COLS=$(sqlite3 -readonly "$DB" ".timeout 120000" "SELECT group_concat(quote(name) || ', ' || name, ', ') FROM pragma_table_info('v_paper_current');")
[ -n "$COLS" ] || { echo "v_paper_current has no columns (view absent?)" >&2; exit 2; }
read -r IN_PAPER SELECTED ROWS SHA < <(sqlite3 -readonly -separator ' ' "$DB" ".timeout 120000" \
  "SELECT SUM(in_current_paper = 1), SUM(selection_state = 'selected'), COUNT(*), '' FROM v_paper_current;" | head -n 1)
# one pass over the ordered rows, hashed as they stream (never held whole)
SHA=$(sqlite3 -readonly "$DB" ".timeout 120000" "SELECT json_object($COLS) FROM v_paper_current ORDER BY run_id;" | sha256sum | cut -d' ' -f1)
t1=$(date +%s.%N)
LINE=$(printf '{"utc": "%s", "label": "%s", "root": "%s", "in_paper": %s, "selected": %s, "rows": %s, "sha256": "%s", "elapsed_s": %.1f}' \
  "$(date -u +%FT%TZ)" "$LABEL" "$ROOT" "${IN_PAPER:-0}" "${SELECTED:-0}" "${ROWS:-0}" "$SHA" "$(echo "$t1 - $t0" | bc)")
echo "$LINE"
mkdir -p "$(dirname "$OUT")"
echo "$LINE" >> "$OUT"
