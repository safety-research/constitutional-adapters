#!/bin/bash
# check_no_modal.sh [--repo DIR] [--inventory JSON] [--mode report|refuse] [--json OUT] [--print-pattern]
# The no-Modal scan: every line of the REPO TREE's registry
# ownership and of the OPERATING EXECUTABLES is matched against the Modal reach patterns, with no allowlist.
#   repo scope (same as tests/test_registry_no_modal.py): src/sisteer/registry/**, scripts/registry/**, systemd/user/registry-*,
#     configs/registry/*.yaml, tests/test_registry_*.py (text files only)
#   operating scope: the files an inventory (operating_scripts_inventory.py --out JSON) classifies `executable`, under its root,
#     `retired/` excluded; its documents are LISTED (counted) and not scanned
#   patterns: a Modal import (the package named by `import` or `from`, or its App / Function / Volume / Image / Cls attributes), a
#     launcher (the CLI's run and volume verbs, the Modal venv), the token variable, the registry's volume app, smoke producer and
#     transport modules (the four module names), the retired gates launcher directory, the Modal-era suite module, and the Modal
#     configuration table's header. The regex is COMPOSED from fragments at run time (see PAT) so this script's own text is not a hit
#     of its own scan; `--print-pattern` prints the composed regex for the test that pins the two scanners together.
# Output: one `HIT <path>:<line>: <text>` per match, then one summary line
#   `no-modal scan: <n> hit(s) in <k> file(s); repo <dir> (<r> files scanned); operating executables <m> scanned, <d> documents
#    listed, retired excluded; mode=<mode>`
# Exit: report -> 0 whatever the count (the hits are the report); refuse -> 1 when any hit exists, and refuse REQUIRES --inventory
# (the scan is mandatory only once the inventory records zero live hits); 2 usage / an unreadable inventory.
# land.sh runs it after the merge under LAND_NO_MODAL_SCAN=report|refuse|off (default report; a refusal exits 14 there).
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../../.." && pwd)
INV=""; MODE=report; JSON_OUT=""; PRINT=0
while [ $# -gt 0 ]; do
  case "$1" in
    --repo) REPO=$2; shift ;;
    --inventory) INV=$2; shift ;;
    --mode) MODE=$2; shift ;;
    --json) JSON_OUT=$2; shift ;;
    --print-pattern) PRINT=1 ;;
    -h|--help) awk 'NR == 1 {next} /^#/ {print; next} {exit}' "$0"; exit 0 ;;
    *) echo "check_no_modal.sh: unknown argument $1" >&2; exit 2 ;;
  esac
  shift
done
case "$MODE" in report|refuse) ;; *) echo "check_no_modal.sh: --mode must be report or refuse (got '$MODE')" >&2; exit 2 ;; esac
# the regex, composed so no literal pattern sits in this file: M = the package, U = its upper-case form, S = an underscore
M=modal; U=MODAL; S=_; V=volume; R=registry; G=gates/; L=land/; C=cpu
PAT="import $M\b|from $M\b|$M\.(App|Function|Volume|Image|Cls)|$M run|$M $V|venvs/$M|${U}${S}TOKEN|${R}${S}${V}|${R}${S}smoke|${M}${S}ops|${V}${S}ops|${G}${L}|${C}${S}suite|\[$M\]"
if [ "$PRINT" = 1 ]; then echo "$PAT"; exit 0; fi
[ -d "$REPO" ] || { echo "check_no_modal.sh: no repo at $REPO" >&2; exit 2; }
if [ "$MODE" = refuse ] && [ -z "$INV" ]; then echo "check_no_modal.sh: refuse mode requires --inventory (the operating executables are part of the scan)" >&2; exit 2; fi
HITS_FILE=$(mktemp); trap 'rm -f "$HITS_FILE"' EXIT
# ---- the repo scope: a file list, then one grep over it (binary files skipped: -I)
REPO_FILES=$(cd "$REPO" && {
  [ -d src/sisteer/registry ] && find src/sisteer/registry -type f
  [ -d scripts/registry ] && find scripts/registry -type f
  [ -d systemd/user ] && find systemd/user -maxdepth 1 -type f -name 'registry-*'
  [ -d configs/registry ] && find configs/registry -maxdepth 1 -type f -name '*.yaml'
  [ -d tests ] && find tests -maxdepth 1 -type f -name 'test_registry_*.py'
} | grep -v '/__pycache__/' | sort)
N_REPO=$(printf '%s\n' "$REPO_FILES" | grep -c .)
if [ "$N_REPO" -gt 0 ]; then
  (cd "$REPO" && printf '%s\n' "$REPO_FILES" | xargs -r grep -nIHE -e "$PAT" -- 2>/dev/null) | sed "s|^|$REPO/|" >>"$HITS_FILE" || true
fi
# ---- the operating scope from the inventory: executables scanned, documents counted
N_EXEC=0; N_DOC=0
if [ -n "$INV" ]; then
  [ -s "$INV" ] || { echo "check_no_modal.sh: inventory $INV is missing or empty" >&2; exit 2; }
  LISTS=$(python3 - "$INV" <<'PY'
import json, os, sys
d = json.load(open(sys.argv[1]))
root = d.get("root") or ""
ex = [os.path.join(root, f["path"]) for f in d.get("files", []) if f.get("class") == "executable" and not f["path"].startswith("retired/")]
docs = [f["path"] for f in d.get("files", []) if f.get("class") == "document" and not f["path"].startswith("retired/")]
print(len(ex)); print(len(docs))
print("\n".join(ex))
PY
) || { echo "check_no_modal.sh: inventory $INV is not readable as an inventory" >&2; exit 2; }
  N_EXEC=$(sed -n 1p <<<"$LISTS"); N_DOC=$(sed -n 2p <<<"$LISTS")
  EXECS=$(sed -n '3,$p' <<<"$LISTS")
  if [ -n "$EXECS" ]; then
    printf '%s\n' "$EXECS" | while IFS= read -r f; do [ -f "$f" ] && grep -nIHE -e "$PAT" -- "$f" 2>/dev/null; done >>"$HITS_FILE" || true
  fi
fi
N_HITS=$(grep -c . "$HITS_FILE" || true)
N_FILES=$(cut -d: -f1 "$HITS_FILE" | sort -u | grep -c . || true)
# `HIT <path>:<line>: <text>` -- grep -n prints `path:line:text`; the documented shape has a space before the text
sed -E 's/^([^:]+):([0-9]+):/\1:\2: /; s/^/HIT /' "$HITS_FILE"
echo "no-modal scan: $N_HITS hit(s) in $N_FILES file(s); repo $REPO ($N_REPO files scanned); operating executables $N_EXEC scanned, $N_DOC documents listed, retired excluded; mode=$MODE"
if [ -n "$JSON_OUT" ]; then
  python3 - "$HITS_FILE" "$JSON_OUT" "$MODE" "$REPO" "$N_REPO" "$N_EXEC" "$N_DOC" "${INV:-}" <<'PY'
import json, sys, time
hits_f, out, mode, repo, n_repo, n_exec, n_doc, inv = sys.argv[1:9]
hits = []
for line in open(hits_f, encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    path, _, rest = line.partition(":")
    ln, _, text = rest.partition(":")
    hits.append({"path": path, "line": int(ln) if ln.isdigit() else None, "text": text.lstrip()})
rec = {"schema": "no_modal_scan/1", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "mode": mode, "repo": repo,
       "repo_files_scanned": int(n_repo), "operating_executables_scanned": int(n_exec), "operating_documents_listed": int(n_doc),
       "inventory": inv or None, "hits": hits, "n_hits": len(hits), "n_files": len({h["path"] for h in hits})}
json.dump(rec, open(out, "w", encoding="utf-8"), indent=1, sort_keys=True)
PY
fi
if [ "$MODE" = refuse ] && [ "$N_HITS" -gt 0 ]; then exit 1; fi
exit 0
