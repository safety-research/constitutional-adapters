#!/bin/bash
# reproduce/tools/snapshot_paper_frame.sh — cut the paper frame: the committed registry export the companion repo carries
# (`registry snapshot --scope paper_frame`). A COMMAND, run on demand from the grading host outside a
# landing window; never the registry's snapshot timer, never the registry root's own snapshots/ + CURRENT.
#
#   reproduce/tools/snapshot_paper_frame.sh [--dest-root DIR] [--source-root ROOT] [--model-id ID]... [--paper-models-only]
#                                           [--item-judge ID]... [--no-item-judges] [--item-projection NAME]
#                                           [--precondition-override REASON] [--no-space-check]
#                                           [--config F] [--dry-run] [--help]
#
#   --dest-root DIR      where the export is written: <DIR>/CURRENT + <DIR>/snapshots/<id>/{MANIFEST.json, *.parquet (every
#                        table but item_result and item_flags, which ship in the SQLite only), paper_frame.sqlite.gz.NN}
#                        (default: <repo>/paper/snapshot)
#   --source-root ROOT   the registry root to read       (default: $SISTEER_REGISTRY_ROOT from the site config, else ~/registry)
#   --model-id ID        also export this model's runs (repeatable; the paper-frame scope's own model set — the paper's
#                        four with their fit checkpoints, plus every run of the registry's test models — is always in)
#   --paper-models-only  leave the registry's test models out (the committed frame is cut so)
#   --item-judge ID      add every complete judgment of this judge on a run of the model set to the item scope (repeatable).
#                        Default: the paper policy's judges, paper/notebooks/data/paper_judges.json, in its item_projection
#   --no-item-judges     cut without item judges (a frame only the `primary` policy can render)
#   --item-projection NAME
#                        ship the item judges' rows in this projection (reader: extra_json reduced to the keys the notebook's
#                        item reader extracts, record_sha256 NULL). Default: paper_judges.json's with its judges, else verbatim
#   --precondition-override REASON
#                        skip the tree precondition (relu_native PAIRED_ROOT off an unlanded working tree, main merged
#                        afterwards); the reason is recorded in MANIFEST.precondition
#   --no-space-check     skip the free-space rule (2 x estimated output + 2 GB, measured at --dest-root)
#
# Prints `registry: snapshot <id>` and `head <ingest_seq>` for the export it cut; refuses (non-zero, the reason on stderr)
# whenever the registry CLI refuses. The interpreter is $SISTEER_PYTHON, else the repo's .venv, else python3.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/../lib/common.sh"
usage(){ sed -n '2,33p' "$0"; repro_common_help; }
DEST=""; SOURCE=""; MODEL_IDS=(); OVERRIDE=""; NO_SPACE=0; PAPER_ONLY=0; ITEM_JUDGES=(); NO_ITEM_JUDGES=0; PROJECTION=""
repro_parse_common_args "$@"; set -- "${REPRO_ARGS[@]+"${REPRO_ARGS[@]}"}"
while [ $# -gt 0 ]; do
  case "$1" in
    --dest-root) DEST="${2:?--dest-root needs a directory}"; shift 2;;
    --dest-root=*) DEST="${1#--dest-root=}"; shift;;
    --source-root) SOURCE="${2:?--source-root needs a registry root}"; shift 2;;
    --source-root=*) SOURCE="${1#--source-root=}"; shift;;
    --model-id) MODEL_IDS+=("${2:?--model-id needs an id}"); shift 2;;
    --paper-models-only) PAPER_ONLY=1; shift;;
    --item-judge) ITEM_JUDGES+=("${2:?--item-judge needs a judge id}"); shift 2;;
    --no-item-judges) NO_ITEM_JUDGES=1; shift;;
    --item-projection) PROJECTION="${2:?--item-projection needs a name}"; shift 2;;
    --precondition-override) OVERRIDE="${2:?--precondition-override needs a reason}"; shift 2;;
    --no-space-check) NO_SPACE=1; shift;;
    *) echo "REFUSED: unknown argument $1" >&2; usage >&2; exit 2;;
  esac
done
if [ "$REPRO_HELP" = 1 ]; then usage; exit 0; fi
[ "$NO_ITEM_JUDGES" = 0 ] || [ "${#ITEM_JUDGES[@]}" = 0 ] || { echo "REFUSED: --no-item-judges and --item-judge contradict each other" >&2; exit 2; }
[ "$NO_ITEM_JUDGES" = 0 ] || [ -z "$PROJECTION" ] || { echo "REFUSED: --item-projection applies to item judges, and --no-item-judges names none" >&2; exit 2; }
# the site config when there is one (--config, $SISTEER_CONFIG, $SISTEER_SITE_DIR/config.env); without one the defaults below stand
if [ -r "$(repro_config_path)" ]; then repro_load_config; else repro_note "no site config at $(repro_config_path); using the defaults"; fi
[ -n "$DEST" ] || DEST="$REPRO_REPO/paper/snapshot"
[ -n "$SOURCE" ] || SOURCE="$(eval echo "${SISTEER_REGISTRY_ROOT:-\$HOME/registry}")"
PY="$(repro_python)"
[ -n "$PY" ] || repro_fatal "no Python interpreter (set SISTEER_PYTHON in the site config)"
if [ "$REPRO_DRY_RUN" != 1 ]; then
  [ -f "$SOURCE/registry.sqlite" ] || repro_fatal "no registry database at $SOURCE/registry.sqlite (pass --source-root ROOT or set SISTEER_REGISTRY_ROOT)"
  # the destination against the registry root, resolved PHYSICALLY through the exporter's own check (snapshot.paper_frame_dest_root:
  # equal, inside or containing — a symlink alias included). The CLI runs the same check again
  # before it reclaims anything; this call only names the refusal at the front door.
  env "PYTHONPATH=$REPRO_REPO/src" "$PY" - "$SOURCE" "$DEST" <<'PYEOF' || repro_fatal "--dest-root $DEST is refused against the registry root $SOURCE (the reason is above)"
import sys
from sisteer.registry import snapshot as SN
try:
    SN.paper_frame_dest_root(sys.argv[1], sys.argv[2])
except SN.SnapshotError as exc:
    print(f"REFUSED: {exc}", file=sys.stderr)
    sys.exit(2)
PYEOF
fi
# the paper policy's judges and their projection (paper/notebooks/data/paper_judges.json) unless named or refused here
JUDGES_FILE="$REPRO_REPO/paper/notebooks/data/paper_judges.json"
if [ "$NO_ITEM_JUDGES" = 0 ] && [ "${#ITEM_JUDGES[@]}" = 0 ]; then
  [ -r "$JUDGES_FILE" ] || repro_fatal "no paper judges at $JUDGES_FILE (pass --item-judge ID, or --no-item-judges for a primary-only frame)"
  while IFS= read -r j; do [ -z "$j" ] || ITEM_JUDGES+=("$j"); done < <("$PY" -c 'import json, sys; [print(j["judge_id"]) for j in json.load(open(sys.argv[1]))["judges"]]' "$JUDGES_FILE")
  [ "${#ITEM_JUDGES[@]}" -gt 0 ] || repro_fatal "$JUDGES_FILE names no judge"
  [ -n "$PROJECTION" ] || PROJECTION="$("$PY" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("item_projection") or "")' "$JUDGES_FILE")"
fi
ARGS=(-m sisteer.registry.cli --root "$SOURCE" snapshot --scope paper_frame --dest-root "$DEST")
for m in "${MODEL_IDS[@]+"${MODEL_IDS[@]}"}"; do ARGS+=(--model-id "$m"); done
[ "$PAPER_ONLY" = 0 ] || ARGS+=(--paper-models-only)
for j in "${ITEM_JUDGES[@]+"${ITEM_JUDGES[@]}"}"; do ARGS+=(--item-judge "$j"); done
[ -z "$PROJECTION" ] || ARGS+=(--item-projection "$PROJECTION")
[ -z "$OVERRIDE" ] || ARGS+=(--precondition-override "$OVERRIDE")
[ "$NO_SPACE" = 0 ] || ARGS+=(--no-space-check)
repro_say "paper frame: source=$SOURCE dest=$DEST python=$PY"
if [ "$REPRO_DRY_RUN" = 1 ]; then
  repro_run env "PYTHONPATH=$REPRO_REPO/src" "$PY" "${ARGS[@]}"
  echo "DRY-RUN: registry: snapshot <id>  (printed from MANIFEST.json after the build)"
  exit 0
fi
mkdir -p "$DEST"
MANIFEST_OUT="$(mktemp "${TMPDIR:-/tmp}/paper_frame_manifest.XXXXXX")"
trap 'rm -f "$MANIFEST_OUT"' EXIT
rc=0
env "PYTHONPATH=$REPRO_REPO/src" "$PY" "${ARGS[@]}" > "$MANIFEST_OUT" || rc=$?
if [ "$rc" != 0 ]; then
  REPRO_REFUSE_RC="$rc" repro_fatal "registry snapshot --scope paper_frame refused (rc=$rc; the reason is above)"
fi
read -r SNAP_ID HEAD_SEQ < <("$PY" -c 'import json, sys; m = json.load(open(sys.argv[1])); print(m["snapshot_id"], m["ingest_seq"])' "$MANIFEST_OUT")
repro_registry_id snapshot "$SNAP_ID"
echo "head $HEAD_SEQ"
repro_say "paper frame cut at head $HEAD_SEQ -> $DEST/snapshots/$SNAP_ID (CURRENT -> $SNAP_ID); commit $DEST to publish it"
