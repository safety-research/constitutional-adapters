#!/bin/bash
# refusal_drill.sh -- the LIVE refusal drill of the freshness gate: in a
# SCRATCH COPY of the registry's cluster-read state, plant a stale generation pin for ONE component of one adapter, plan a window
# against it, and assert that the plan records that adapter REFUSED by name (route grading_host = not read; C6) and that the status page names the refusal. Nothing
# here touches the live registry, the live state, the stores or the cluster: the scratch root holds a copy of `state/store_manifests`
# and `state/store_mirror.json` only (no database is needed for `cluster-reads plan`; the page's section reads the plan file), and the
# drill is removed unless --keep.
#
#   scripts/registry/measure/refusal_drill.sh [--root ~/registry] [--component misalign_grades] [--adapter em_misalign]
#                                            [--config configs/registry/cluster_reads.yaml] [--python PY] [--keep]
#
# Dry-runnable against shims: with REGISTRY_PY pointing at a shim that answers `cluster-reads plan` and `cluster-reads status --json`,
# the drill exercises the same steps and assertions (tests/test_registry_cluster_read.py drives it that way).
set -euo pipefail
ROOT="${REGISTRY_ROOT:-$HOME/registry}"
COMPONENT="misalign_grades"
ADAPTER="em_misalign"
CONFIG=""
PY="${REGISTRY_PY:-}"
KEEP=0
while [ $# -gt 0 ]; do
  case "$1" in
    --root) ROOT="$2"; shift ;;
    --component) COMPONENT="$2"; shift ;;
    --adapter) ADAPTER="$2"; shift ;;
    --config) CONFIG="$2"; shift ;;
    --python) PY="$2"; shift ;;
    --keep) KEEP=1 ;;
    -h|--help) sed -n '2,15p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
[ -n "$PY" ] || PY="$REPO/.venv/bin/python"
[ -n "$CONFIG" ] || CONFIG="$REPO/configs/registry/cluster_reads.yaml"
export PYTHONPATH="${REGISTRY_PYTHONPATH:-$REPO/src}${PYTHONPATH:+:$PYTHONPATH}"
REG="$PY -m sisteer.registry.cli"
[ -f "$ROOT/state/store_mirror.json" ] || { echo "DRILL: no $ROOT/state/store_mirror.json -- no published generation to plant a stale pin against"; exit 3; }

SCRATCH=$(mktemp -d "${TMPDIR:-/tmp}/registry-refusal-drill.XXXXXX")
trap '[ "$KEEP" = 1 ] && echo "DRILL: scratch kept at $SCRATCH" || rm -rf "$SCRATCH"' EXIT
mkdir -p "$SCRATCH/state"
cp -a "$ROOT/state/store_mirror.json" "$SCRATCH/state/"
cp -a "$ROOT/state/store_manifests" "$SCRATCH/state/" 2>/dev/null || mkdir -p "$SCRATCH/state/store_manifests"
echo "DRILL: scratch root $SCRATCH (a copy of the store state; the live root is untouched)"

# the plant: the published pin of ONE component points at a generation whose manifest names a sha no grading host file has -- the admission's
# stat walk finds the manifest's stat snapshot for every file, hashes the sample, and refuses `content_changed` by name
STALE="deadbeef"
$PY - "$SCRATCH" "$COMPONENT" "$STALE" <<'EOF'
import json, os, sys, shutil
scratch, slug, stale = sys.argv[1:4]
st = json.load(open(f"{scratch}/state/store_mirror.json"))
rec = st.setdefault("components", {}).setdefault(slug, {})
published = rec.get("published")
sdir = f"{scratch}/state/store_manifests/{slug}"
os.makedirs(sdir, exist_ok=True)
if published and os.path.isfile(f"{sdir}/MANIFEST-{published}.sha256"):
    # a copy of the real manifest with every sha flipped: same paths, same sizes, other bytes -> content_changed on the sample
    lines = open(f"{sdir}/MANIFEST-{published}.sha256").read().splitlines()
    out = []
    for l in lines:
        parts = l.split("  ", 2)
        if len(parts) == 3 and len(parts[0]) == 64:
            parts[0] = ("f" if parts[0][0] != "f" else "0") + parts[0][1:]
        out.append("  ".join(parts))
    open(f"{sdir}/MANIFEST-{stale}.sha256", "w").write("\n".join(out) + "\n")
    shutil.copy(f"{sdir}/MANIFEST-{published}.stat.json", f"{sdir}/MANIFEST-{stale}.stat.json")
    if os.path.isfile(f"{sdir}/MANIFEST-{published}.json"):
        shutil.copy(f"{sdir}/MANIFEST-{published}.json", f"{sdir}/MANIFEST-{stale}.json")
    print(f"DRILL: planted MANIFEST-{stale} for {slug} (every sha flipped; sizes and paths kept) beside the real {published}")
else:
    print(f"DRILL: {slug} has no published generation with a manifest in the state copy; the plan will refuse it as no_manifest instead")
rec["published"] = stale
json.dump(st, open(f"{scratch}/state/store_mirror.json", "w"), indent=1)
EOF

WINDOW="drill-$(date -u +%Y%m%dT%H%M%SZ)"
# the JSON alone on stdout (the plan constructs the adapters, and the adapters package logs its store-override line on
# import -- to stderr, kept apart so the JSON below parses); both to the drill's own output
$REG --root "$SCRATCH" cluster-reads plan --window "$WINDOW" --adapter "$ADAPTER" --tree-sha "dddddddddddddddddddddddddddddddddddddddd" --config "$CONFIG" --json > "$SCRATCH/plan.out" 2> "$SCRATCH/plan.err" || true
cat "$SCRATCH/plan.err" "$SCRATCH/plan.out"
ROUTE=$($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["routes"].get(sys.argv[2], "?"))' "$SCRATCH/plan.out" "$ADAPTER" 2>/dev/null || echo "?")
REASON=$($PY -c 'import json,sys; d=json.load(open(sys.argv[1])); r=d.get("refusals") if d.get("refusals") is not None else d.get("fallback", {}); print("; ".join(r.get(sys.argv[2], [])))' "$SCRATCH/plan.out" "$ADAPTER" 2>/dev/null || echo "")
PAGE=$($REG --root "$SCRATCH" cluster-reads status --json 2>/dev/null || echo "{}")
NAMED=$($PY - "$PAGE" "$ADAPTER" "$COMPONENT" <<'PYEOF' 2>/dev/null || echo "no"
import json, sys
page, adapter, component = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
rows = [r for r in page.get("rows", []) if r.get("adapter") == adapter]
print("yes" if rows and rows[0].get("route") == "grading_host" and component in (rows[0].get("refusals") or rows[0].get("fallback") or "") else "no")
PYEOF
)
echo "DRILL: window $WINDOW: $ADAPTER route=$ROUTE reason=[$REASON] named_on_page=$NAMED"
if [ "$ROUTE" = grading_host ] && [[ "$REASON" == *"$COMPONENT"* ]] && [ "$NAMED" = yes ]; then
  echo "DRILL PASS: a stale pin for $COMPONENT records $ADAPTER refused by name, and the page names it"
  exit 0
fi
echo "DRILL FAIL: expected route=grading_host naming $COMPONENT and the page to name it"
exit 1
