#!/bin/bash
# The post-judging pipeline for one lane, from a cache that already holds every verdict
# (e.g. after batch_judge_fill.py harvest): file rows/summaries with --no-live (any cache
# miss raises), re-emit statistics with the run cutoff, cap-condition, pair against each
# view's none, report, append-verified push, read-back, cache merge.
#   post_pipeline.sh TAG TARGETS MATCH LOGIN_HOST REMOTE_ROOT CACHE RUN_STARTED_AT_ISO
set -euo pipefail
TAG=$1; TARGETS=$2; MATCH=$3; LOGIN=$4; RROOT=$5; CACHE=$6; STARTED=$7
J=$(cd "$(dirname "$0")" && pwd); WT=$(cd "$J/../../../.." && pwd)
G=${F22_GRADE_ROOT:-$(PYTHONPATH="$WT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.site expand '$GRADE/sdfbase_full22' || :)}; G="${G/#\~/$HOME}"
[ -n "$G" ] || { echo "post_pipeline.sh: set SISTEER_GRADE_DIR (the grade root) or F22_GRADE_ROOT" >&2; exit 2; }
LOGDIR=$G/logs/$TAG; mkdir -p "$LOGDIR"
export F22_LOGIN_HOST=$LOGIN F22_CONTROLS_REMOTE=$RROOT PYTHONPATH=src:scripts HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
[ -n "$MATCH" ] && MATCHARG=(--match "$MATCH") || MATCHARG=()
cd "$WT"
echo "[$TAG post] file from cache (no live)"
CAPEVAL_JUDGE_CACHE=$CACHE .venv/bin/python scripts/served_capability_judge.py --controls-root "$G/controls" --targets "$TARGETS" "${MATCHARG[@]}" --no-live --cell-parallel 4 --sentinel-dir "$LOGDIR/file" 2>&1 | grep -E "DONE |REFUSED|FAILED|JUDGE_|Traceback|no-live" | cut -c1-220
echo "[$TAG post] re-emit (cutoff $STARTED)"
CAPEVAL_JUDGE_CACHE=$CACHE .venv/bin/python scripts/served_capability_judge.py --controls-root "$G/controls" --targets "$TARGETS" "${MATCHARG[@]}" --reemit-only --run-started-at "$STARTED" --cell-parallel 4 --sentinel-dir "$LOGDIR/reemit" 2>&1 | grep -E "JUDGE_"
echo "[$TAG post] cap conditioning"
.venv/bin/python "$J/cap_condition.py" --root "$G/controls" --targets "$TARGETS" "${MATCHARG[@]}" 2>&1 | grep -v "^$" | cut -c1-230
echo "[$TAG post] paired vs none"
python3 "$J/paired_vs_none.py" --targets "$TARGETS" 2>&1 | grep -E "^target|${MATCH:-.}" | cut -c1-150
echo "[$TAG post] report"
python3 "$J/report_table.py" --targets "$TARGETS" --json "$G/report_rows_$TAG.json" 2>&1 | cut -c1-200
echo "[$TAG post] push-back to $LOGIN"
python3 "$J/pushback_judged.py" --targets "$TARGETS" 2>&1 | grep -vE "^IN-SYNC" | tail -8
for t in ${TARGETS//,/ }; do rsync -t --no-owner --no-group --no-perms "$G/controls/$t/judge_paired_vs_none.json" "$LOGIN:$RROOT/$t/"; done
echo "[$TAG post] read-back"
ssh "$LOGIN" "F22_CONTROLS_REMOTE=$RROOT python3 -" < "$J/verify_remote_stores.py" 2>&1 | grep -E "${MATCH:-.}|TOTAL" | cut -c1-200
echo "[$TAG post] cache merge"
python3 "$J/merge_cache_back.py" --private "$CACHE" 2>&1 | tail -1
echo "[$TAG post] DONE $(date -u +%FT%TZ)"
