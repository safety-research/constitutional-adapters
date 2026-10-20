#!/usr/bin/env bash
# inbox_to_log.sh — the grading host side: move what the clusters pushed (inbox/<host>/<campaign>/<run_dir>/*.jsonl[.gz])
# into the merged record log (records/<campaign>/<run_dir>/) ATOMICALLY (os.replace on one filesystem), the
# same merge every harvester cycle performs, callable alone — from a timer, from the
# operator, or by the harvester wrapper before an ingest.
#
#   scripts/registry/inbox_to_log.sh [--root ~/registry] [--verbose]
#
# A same-named file whose logical sha differs is a PRODUCER BUG: kept beside the original as
# `.conflict-<sha8>` and counted, never silently overwritten; an identical duplicate is dropped. Prints one
# JSON line {merged, duplicate, conflict, files}. Runs the registry CLI from the repo venv when present
# (the harvester's own code), else the checkout's src/ under python3; exit 0 unless the merge itself failed.
set -uo pipefail

ROOT="${REGISTRY_ROOT:-$HOME/registry}"; VERBOSE=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --root) ROOT="$2"; shift 2 ;;
    --verbose) VERBOSE=(--verbose); shift ;;
    -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "inbox_to_log: unknown option $1" >&2; exit 2 ;;
  esac
done
[[ -d "$ROOT" ]] || { echo "inbox_to_log: no registry root at $ROOT" >&2; exit 2; }

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
MAIN="$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; MAIN="${MAIN%/.git}"   # the main checkout of a worktree
if [[ -x "$REPO/.venv/bin/python" ]]; then PY="$REPO/.venv/bin/python"
elif [[ -n "$MAIN" && -x "$MAIN/.venv/bin/python" ]]; then PY="$MAIN/.venv/bin/python"
else PY=python3; fi
PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" nice -n 10 "$PY" -m sisteer.registry.cli --root "$ROOT" inbox-merge "${VERBOSE[@]}"
