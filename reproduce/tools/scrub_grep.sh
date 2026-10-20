#!/usr/bin/env bash
# reproduce/tools/scrub_grep.sh: the publication scan (reproduce/tools/scrub_scan.py) as a grep. Prints every hit outside
# the allowlist as path:line:text under the given roots (default: the front door, src/, docs/, README.md, LICENSE and the
# shipped generation-config table); --all widens it to the whole publication set. Every other option goes to the scan
# (--no-allowlist, --scope tree|comment|leaf, --text-only, ...). Exit 0 whatever it finds; 2 when the scan cannot run.
#   reproduce/tools/scrub_grep.sh [--all] [OPTION...] [ROOT...]
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
PY="${SCRUB_PYTHON:-${SISTEER_PYTHON:-python3}}"
ALL=0; ROOTS=(); OPTS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --all) ALL=1 ;;
    --scope|--tokens|--allowlist|--tmpdir|--jobs) OPTS+=("$1" "${2:?$1 takes a value}"); shift ;;
    -*) OPTS+=("$1") ;;
    *) ROOTS+=("$1") ;;
  esac
  shift
done
if [ "$ALL" = 0 ] && [ ${#ROOTS[@]} -eq 0 ]; then
  ROOTS=(reproduce src docs README.md LICENSE configs/sdfbase_shipped_generation_config.json)
fi
exec "$PY" "$HERE/scrub_scan.py" --root "$REPO" --grep ${OPTS[@]+"${OPTS[@]}"} ${ROOTS[@]+"${ROOTS[@]}"}
