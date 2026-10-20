#!/bin/bash
# PADDED STATIC leg (docs/static_padding.md): mirror ONE (padding level, family) of the pad_static stores from the cluster
# into the grading host analysis tree (read-only rsync), then grade it there with grade_padded_static.py through the shared
# StrongREJECT body. One (level, family) per call — the wave runs one job per family and each is graded as it comes off.
#
#   grade_padded_family.sh <pad> <family> [--check-only | --dry-run] [options]
#     <pad>                 the padding level: 0 (the control), 8192, 16384, 32768 or 65536 (make_padding_configs.PADS — five for
#                           EVERY config family since 2026-09-24: 8192 / 32768 joined the steered family first and the prompted-only
#                           family the same day as the REFERENCE standups the registry files the steered rungs on; the grader is
#                           the authority on which family has which level, make_padding_configs.pads_of)
#     <family>              pasttense | codeattack | renellm | grok420 (the ATTACK family)
#     --config-family CFAM  grade a STEERED wave's stores instead of the prompted-only wave's (pad_static_<key>_p<pad>): `obj`
#                           (pad_static_<key>_obj_p<pad>, the ten-cell padstatic_obj configs; make_padding_configs.OBJ_FAMILY) or a
#                           seed replicate `obj_s2` / `obj_s3` (pad_static_<key>_obj_s<N>_p<pad>, the seven-cell padstatic_obj_s<N>
#                           configs; make_padding_configs.OBJ_SEED_FAMILIES). The spelling is asked of the generator, never listed
#                           here: any family make_padding_configs.FAMILIES knows is accepted, anything else refused by name.
#                           Same grader, same instrument: only the run name and config name change (the steered wave ran on
#                           cluster A: --cluster with its label; seed 2 ran on cluster B, the default cluster)
#     --check-only          completeness + identity + the padding invariants only; writes nothing (exit 0 when gradeable)
#     --dry-run             no API: scores 0.0; exercises the judge-payload guard on the real rows; writes under …/dry_run/ only
#     --key KEY             the leg's target key (default qwen35_4b; make_padding_configs.KEYS)
#     --cluster LABEL       the cluster to mirror from, by registry label (default the prompted-only wave's) or site label; its login
#                           host (CLUSTER_SSH) and shared root (CLUSTER_SHARED_ROOT) come from the site config (reproduce/lib/clusters.py)
#     --host HOST           the login host's ssh alias, given directly instead (needs --remote-root); `local` = --remote-root is a
#                           local directory, no ssh
#     --remote-root DIR     the cluster's transcripts_results (default <shared root>/exp/sdfbase_full22_native/static_padding/transcripts_results)
#     --dest DIR            the grading host analysis root (default the full22 run-state tree's static_padding/, under the state
#                           root $STATE through sisteer.site; $F22P_DEST)
#                           the mirror lands in <dest>/transcripts_padded/<private_run>/<family>/, grades in <dest>/grades_padded/<private_run>/<family>/
#     --no-mirror           grade the existing mirror without touching the cluster
#     --mirror-only         mirror and stop
#     --concurrency N       the haiku grader's in-flight ceiling (default 200; Low Prio only — the grader accepts no other org)
#     --nshards N           shards per (family, cell) store (default 3, the sbatch's default, the filed rungs' and the prompted-only
#                           reference standups' at every level; 16 for the STEERED added rungs p8192 / p32768, sharded across four
#                           jobs into one shard00..shard15 layout; 12 for every rung of a seed replicate, three jobs into
#                           shard00..shard11 — the grader requires exactly the store's)
#     --python BIN          interpreter (default: the first of <worktree>/.venv/bin/python, the main checkout's .venv/bin/python, python3 on PATH
#                           that can import yaml + make_padding_configs — probed, not assumed)
#     --grader-arg ARG      appended to the grader call (repeatable; e.g. --grader-arg --seed-cache=<jsonl>)
#     --print               print the rsync and grader commands and exit (nothing runs)
#
# rsync flags follow mirror_sampled.sh: -rlt --no-owner --no-group --no-perms --partial (MooseFS refuses chown; the stores are
# append-only JSONL, so re-syncing a growing store is safe). Never --delete: the mirror only ever gains files. The grader's
# stdout/stderr also land in <dest>/logs/grade[_obj|_obj_s<N>]_p<pad>_<family>.<UTC stamp>.log.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
WT=$(cd "$HERE/../../../.." && pwd)
PAD=""; FAM=""; MODE=""; KEY=qwen35_4b; CFAM=""; CLUSTER=cluster_b; HOST=""; RROOT=""; DEST=${F22P_DEST:-}
MIRROR=1; GRADE=1; CONC=200; NSHARDS=3; PY=""; PRINT=0; GRADER_ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --check-only|--dry-run) MODE=$1; shift;;
    --key) KEY=$2; shift 2;;
    --config-family) CFAM=$2; shift 2;;
    --cluster) CLUSTER=$2; shift 2;;
    --host) HOST=$2; shift 2;;
    --remote-root) RROOT=$2; shift 2;;
    --dest) DEST=$2; shift 2;;
    --no-mirror) MIRROR=0; shift;;
    --mirror-only) GRADE=0; shift;;
    --concurrency) CONC=$2; shift 2;;
    --nshards) NSHARDS=$2; shift 2;;
    --python) PY=$2; shift 2;;
    --grader-arg) GRADER_ARGS+=("$2"); shift 2;;
    --print) PRINT=1; shift;;
    -h|--help) sed -n '2,44p' "$0"; exit 0;;
    -*) echo "unknown arg: $1" >&2; exit 2;;
    *) if [ -z "$PAD" ]; then PAD=$1; elif [ -z "$FAM" ]; then FAM=$1; else echo "unexpected positional: $1" >&2; exit 2; fi; shift;;
  esac
done
[ -n "$PAD" ] && [ -n "$FAM" ] || { echo "usage: grade_padded_family.sh <pad> <family> [--check-only|--dry-run] [options]" >&2; exit 2; }
# the default analysis root: the full22 run-state tree's static_padding/, under the state root ($STATE), expanded through sisteer.site
if [ -z "$DEST" ]; then
  DEST=$(PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.site expand '$STATE/romp/candidate-si-refit/run-state/full22/static_padding') \
    || { echo "grade_padded_family.sh: set SISTEER_STATE_ROOT (the state root), or pass --dest / F22P_DEST" >&2; exit 2; }
  DEST="${DEST/#\~/$HOME}"
fi
case "$PAD" in 0|8192|16384|32768|65536) ;; *) echo "pad '$PAD' is not a level of this leg (0, 8192, 16384, 32768, 65536)" >&2; exit 2;; esac
case "$FAM" in pasttense|codeattack|renellm|grok420) ;; *) echo "family '$FAM' is not a static family" >&2; exit 2;; esac
case "$CONC" in ''|*[!0-9]*|0) echo "--concurrency '$CONC' is not a positive integer" >&2; exit 2;; esac
if [ -n "$HOST" ]; then
  [ -n "$RROOT" ] || { echo "--host $HOST needs --remote-root <dir>" >&2; exit 2; }
elif [ "$MIRROR" = 1 ]; then
  # the login host and shared root of the cluster, from its block in the site config (looked up by registry or site label)
  CL_PY="$WT/reproduce/lib/clusters.py"
  HOST=$(python3 "$CL_PY" --get "$CLUSTER" CLUSTER_SSH) && [ -n "$HOST" ] \
    || { echo "the site config gives no login host (CLUSTER_SSH) for cluster $CLUSTER; describe it there, or pass --host <ssh alias> --remote-root <dir>" >&2; exit 2; }
  if [ -z "$RROOT" ]; then
    CL_SHARED=$(python3 "$CL_PY" --get "$CLUSTER" CLUSTER_SHARED_ROOT) && [ -n "$CL_SHARED" ] \
      || { echo "the site config gives no shared root (CLUSTER_SHARED_ROOT) for cluster $CLUSTER; pass --remote-root <dir>" >&2; exit 2; }
    RROOT=$CL_SHARED/exp/sdfbase_full22_native/static_padding/transcripts_results
  fi
fi
# THE INTERPRETER (2026-09-28): --python wins; otherwise the first candidate that can import the generator's dependencies —
# <worktree>/.venv, the repo checkout's .venv, then the python on PATH. A candidate is PROBED (`import yaml, make_padding_configs`),
# not just tested for existence: the Modal cpu_suite image carries a tree with no usable venv at either venv path, and the
# old existence-only pick handed a bare interpreter to the generator, which died with `No module named 'yaml'` inside the
# suite (three permanent reds on main until this line; the grading host and the Slurm gate both have the venv, so neither saw it).
# Every candidate that fails is named in the refusal so the environment, not the driver, is what gets fixed.
if [ -z "$PY" ]; then
  _tried=()
  _main="$(git -C "$HERE" rev-parse --path-format=absolute --git-common-dir 2>/dev/null)"; _main="${_main%/.git}"   # the main checkout of a worktree
  for _cand in "$WT/.venv/bin/python" "${_main:+$_main/.venv/bin/python}" "$(command -v python3 2>/dev/null)" "$(command -v python 2>/dev/null)"; do
    [ -n "$_cand" ] && [ -x "$_cand" ] || continue
    if "$_cand" -c 'import sys; sys.path.insert(0, sys.argv[1]); import yaml, make_padding_configs' "$HERE" >/dev/null 2>&1; then PY=$_cand; break; fi
    _tried+=("$_cand")
  done
  [ -n "$PY" ] || { echo "grade_padded_family.sh: no interpreter can import the generator's dependencies (yaml, make_padding_configs); tried: ${_tried[*]:-none} — pass --python <interpreter>" >&2; exit 2; }
fi
# THE CONFIG FAMILY'S NAME SEGMENT — asked of the generator (make_padding_configs.seg_of: empty for the prompted-only wave, `_obj`
# for the seed-1 steered family, `_obj_s<N>` for a seed replicate), never a case arm per family retyped here: a family the
# generator gains is served by this driver without an edit, and a family it does not know is refused by name, with the
# generator's own list, before anything is mirrored or graded. Needs $PY (resolved just above), the same interpreter the
# grader runs under; a $PY that cannot import the generator is its own, distinct, refusal.
FAMSEG=$("$PY" -c '
import sys
sys.path.insert(0, sys.argv[1])
import make_padding_configs as gen
try:
    print(gen.seg_of(sys.argv[2] or None))
except ValueError as e:
    sys.stderr.write(f"--config-family {sys.argv[2]!r} is not a padded-static config family: {e}\n")
    sys.exit(2)
' "$HERE" "$CFAM") || { echo "grade_padded_family.sh: could not derive the config family segment for '$CFAM' from $HERE/make_padding_configs.py under $PY (see above)" >&2; exit 2; }
RUN=pad_static_${KEY}${FAMSEG}_p${PAD}
CFG=configs/sdfbase/static_padding/sdfbase_full22_${KEY}.padstatic${FAMSEG}_p${PAD}.yaml
if [ "$HOST" = local ]; then SRC="$RROOT/$RUN/$FAM/"; else SRC="$HOST:$RROOT/$RUN/$FAM/"; fi
DST="$DEST/transcripts_padded/$RUN/$FAM/"
RSYNC=(rsync -rlt --no-owner --no-group --no-perms --partial --stats "$SRC" "$DST")
GRADER=("$PY" scripts/sdfbase/full22/static_padding/grade_padded_static.py --target "$KEY" --config "$CFG"
        --private-root "$DEST/transcripts_padded" --grades-root "$DEST/grades_padded" --attacks "$FAM"
        --nshards "$NSHARDS" --concurrency "$CONC")
[ -n "$MODE" ] && GRADER+=("$MODE")
[ ${#GRADER_ARGS[@]} -gt 0 ] && GRADER+=("${GRADER_ARGS[@]}")
q() { printf '%q ' "$@"; echo; }
say() { echo "[pad-grade $(date -u +%H:%M:%SZ)] $*"; }
if [ $PRINT -eq 1 ]; then
  echo "mirror: $([ $MIRROR -eq 1 ] && q "${RSYNC[@]}" || echo '(skipped: --no-mirror)')"
  echo "grade:  $([ $GRADE -eq 1 ] && { printf 'cd %q && PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 ' "$WT"; q "${GRADER[@]}"; } || echo '(skipped: --mirror-only)')"
  exit 0
fi
if [ $MIRROR -eq 1 ]; then
  mkdir -p "$DST"
  say "mirror $SRC -> $DST"
  out=$("${RSYNC[@]}" 2>&1) || { say "FAILED rsync: $(tail -1 <<<"$out")"; exit 1; }
  say "mirror ok: $(grep -E 'Number of regular files transferred|Total transferred file size' <<<"$out" | tr '\n' ';')"
  n_shards=$(find "$DST" -mindepth 2 -maxdepth 2 -type d -name 'shard*' | wc -l)
  say "mirror holds $n_shards shard dir(s) under $DST"
fi
[ $GRADE -eq 1 ] || exit 0
mkdir -p "$DEST/logs"
LOG="$DEST/logs/grade${FAMSEG}_p${PAD}_${FAM}.$(date -u +%Y%m%dT%H%M%SZ).log"
say "grade ${MODE:-LIVE (Low Prio, concurrency $CONC)}: $RUN/$FAM -> $DEST/grades_padded/$RUN/$FAM/ (log $LOG)"
cd "$WT" || exit 1
PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 "${GRADER[@]}" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
say "grader exit $rc"
exit "$rc"
