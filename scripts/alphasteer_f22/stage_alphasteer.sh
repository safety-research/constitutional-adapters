#!/usr/bin/env bash
# stage_alphasteer.sh — alphasteer-full22 Phase F: stage each served target's SELECTED baked rung as the
# staged cell the Phase G legs serve, verify it, stamp it, record it, and build + check the bank view.
# Modelled on the wave's stage_full22.sh (rsync flags, sha + file-count verification on the cluster side,
# staged_from.json + staging_manifest.json). Runs on the grading host; the copies are WITHIN-cluster on
# cluster B (the rung already lives there) driven over ssh; cluster A is gated on reachability.
#
#   bash scripts/alphasteer_f22/stage_alphasteer.sh <W_A> [--target T ...] [--dry-run] [--skip-view] [--skip-cluster-a]
#
# Per served target (configs/alphasteer_f22_pins.json, status served — a pending or not-admissible target is
# REFUSED if named and SKIPPED with a printed reason otherwise; glm46vflash never stages):
#   1. <ROOT>/<t>/ladder/lam<chosen.i>/cells_view/<view>/alphasteer/  ->  <ROOT>/<t>/cells/selected__alphasteer/
#      rsync -rlt --no-owner --no-group --no-perms --partial (NEVER -a: MooseFS refuses chown and leaves a
#      silently incomplete tree), within the cluster
#   2. the parent's fit_meta.json (+ .COMPLETE) and the target's screen_f9.json copied in; the selection
#      record lambda_ladder.json (+ .COMPLETE) pushed in from the grading host
#   3. object.pt sha256 == the pin, checked ON THE CLUSTER; every file hashed for the stamp
#   4. staged_from.json written (the wave's hashcheck_and_stamp schema + the rung and the ladder record; the
#      bake's own stamp is embedded as rung_staged_from, never lost) and pushed; the file COUNT re-verified
#   5. a row appended to the campaign's staging_manifest.json (same row schema as the wave's) and pushed to
#      <ROOT>/staging_manifest.json
#   6. make_bank_view.py --roster OURS --pins OURS --bank <ROOT>/<t> --target <t> --view <view>, then --check,
#      from the W_A checkout with the shared vllm-attacker python (torch for the fit_provenance verdict)
# cluster A: if the login host answers, the staged dir is routed cluster B -> grading host -> cluster A and verified the
# same way; otherwise "cluster_a: skipped: unreachable" and exit 0 for that half.
#
# STOP discipline: this script writes no STOP file of any kind. Runs no GPU job. Touches no git.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
PINS="$REPO/configs/alphasteer_f22_pins.json"; ROSTER="$REPO/configs/alphasteer_f22_roster.yaml"
# a path under the grade root, $GRADE, expanded by the site module (empty when the grade root is not set on this host)
grade_path(){ local p; p=$(PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.site expand "\$GRADE/$1") || return 0; printf '%s\n' "${p/#\~/$HOME}"; }
RECORDS_ROOT="${AS22_RECORDS_ROOT:-$(grade_path alphasteer_f22/ladder)}"
STAGE_STATE="${AS22_STAGE_STATE:-$(grade_path alphasteer_f22/staging)}"
[ -n "$RECORDS_ROOT" ] && [ -n "$STAGE_STATE" ] \
  || { echo "stage_alphasteer.sh: set SISTEER_GRADE_DIR (the grade root) or AS22_RECORDS_ROOT and AS22_STAGE_STATE" >&2; exit 2; }
RECORDS="$STAGE_STATE/_records"; LOG="$STAGE_STATE/stage_alphasteer.log"; MANIFEST="$STAGE_STATE/staging_manifest.json"
REL=exp/objects/alphasteer_f22
# Each cluster's login host, shared root and per-commit checkouts root come from its block in the site config
# (reproduce/lib/clusters.py, looked up by the cluster's registry label); the labels stay keys only.
declare -A HOST=() SHARED=() EVAL=()
site_cluster(){   # <registry label> [optional]: HOST, SHARED and EVAL of that cluster from the site config, else a refusal by name
  local h="" s="" e=""
  h=$(python3 "$REPO/reproduce/lib/clusters.py" --get "$1" CLUSTER_SSH) \
    && s=$(python3 "$REPO/reproduce/lib/clusters.py" --get "$1" CLUSTER_SHARED_ROOT) \
    && e=$(python3 "$REPO/reproduce/lib/clusters.py" --get "$1" CLUSTER_EVAL_REPO_ROOT) || true
  if [ -z "$h" ] || [ -z "$s" ] || [ -z "$e" ]; then
    [ "${2:-}" = optional ] \
      || { echo "stage_alphasteer.sh: the site config gives no login host, shared root and checkouts root (CLUSTER_SSH, CLUSTER_SHARED_ROOT, CLUSTER_EVAL_REPO_ROOT) for cluster $1" >&2; exit 2; }
    h=""; s=""; e=""
  fi
  HOST[$1]=$h; SHARED[$1]=${s%/}; EVAL[$1]=${e%/}
}
RSYNC_FLAGS=(-rlt --no-owner --no-group --no-perms --partial --inplace)
STAGED_ID=selected__alphasteer; CELL=alphasteer
WAVE_SHA=""; TARGETS_ARG=(); DRY_RUN=false; SKIP_VIEW=false; SKIP_BULB=false
while [ $# -gt 0 ]; do
  case "$1" in
    --target) TARGETS_ARG+=("$2"); shift ;;
    --dry-run) DRY_RUN=true ;;
    --skip-view) SKIP_VIEW=true ;;
    --skip-cluster-a) SKIP_BULB=true ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    -*) echo "unknown arg: $1" >&2; exit 2 ;;
    *) [ -z "$WAVE_SHA" ] && WAVE_SHA="$1" || { echo "unexpected positional $1" >&2; exit 2; } ;;
  esac; shift
done
[ -n "$WAVE_SHA" ] || { echo "usage: stage_alphasteer.sh <W_A> [--target T] [--dry-run] [--skip-view] [--skip-cluster-a]" >&2; exit 2; }
site_cluster cluster_b
if $SKIP_BULB; then site_cluster cluster_a optional; else site_cluster cluster_a; fi
mkdir -p "$RECORDS"
log(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG" >&2; }
fatal(){ log "FATAL: $*"; exit 1; }

# a bash snippet on a cluster host over STDIN into `bash -s` (the remote login shell is zsh: one non-matching
# glob aborts a whole line there); 3 attempts
remote(){ local host=$1 script=$2 a; for a in 1 2 3; do printf '%s\n' "$script" | ssh -o BatchMode=yes -o ConnectTimeout=30 "$host" bash -s && return 0; sleep $((a*5)); done; return 1; }
# two lines: sha256 of <dir>/object.pt (or MISSING), then the file count under <dir>
remote_probe(){ remote "$1" "d=$(printf '%q' "$2")
if [ -f \"\$d/object.pt\" ]; then sha256sum \"\$d/object.pt\" | cut -d' ' -f1; else echo MISSING; fi
find \"\$d\" -type f 2>/dev/null | wc -l"; }
# every file's sha under <dir>, as `<relpath> <sha>` lines
remote_shas(){ remote "$1" "cd $(printf '%q' "$2") && find . -type f -print0 | sort -z | xargs -0 sha256sum | sed 's#  \./#  #'"; }

# --- the pins decide what is staged --------------------------------------------------------------------------
mapfile -t PIN_ROWS < <(python3 - "$PINS" "$ROSTER" <<'PY'
import json, sys, yaml
sys.path.insert(0, sys.argv[2].rsplit("/configs/", 1)[0] + "/scripts/alphasteer_f22")
import as22_targets as tg
pins = json.load(open(sys.argv[1]))
for t in yaml.safe_load(open(sys.argv[2]))["targets"]:
    e = pins[t]["selected__alphasteer"]
    print(t, e["status"], e.get("chosen_i"), e.get("object_sha256"), tg.view_of(t), e.get("ladder_record_sha256"), e.get("lambda"), e.get("intervention_hash"))
PY
) || fatal "cannot read $PINS / $ROSTER"

want_target(){ [ "${#TARGETS_ARG[@]}" -eq 0 ] && return 0; local t; for t in "${TARGETS_ARG[@]}"; do [ "$t" = "$1" ] && return 0; done; return 1; }
for t in "${TARGETS_ARG[@]}"; do
  st=$(printf '%s\n' "${PIN_ROWS[@]}" | awk -v t="$t" '$1==t{print $2}')
  [ -n "$st" ] || fatal "--target $t is not a campaign target"
  [ "$st" = served ] || fatal "--target $t has pin status $st — REFUSING to stage a target that is not served (glm46vflash: no admissible operating point, D72; pending: regenerate the pins first)"
done

# --- reachability ---------------------------------------------------------------------------------------------
reach(){ ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" true 2>/dev/null; }
reach "${HOST[cluster_b]}" || fatal "cluster_b unreachable — the rungs and the run tree live there; nothing to stage without it"
BULB_OK=false
if $SKIP_BULB; then log "cluster_a: skipped (--skip-cluster-a)"; elif reach "${HOST[cluster_a]}"; then BULB_OK=true; else log "cluster_a: skipped: unreachable (D75 ruling 1) — the cluster B half proceeds; exit 0 for this half"; fi

# --- the W_A checkout on cluster_b (make_bank_view runs from it) ------------------------------------------------
CH_REPO="${EVAL[cluster_b]}/$WAVE_SHA"
CH_PY="${SHARED[cluster_b]}/envs/vllm-attacker/bin/python"
remote "${HOST[cluster_b]}" "[ -f $(printf '%q' "$CH_REPO")/scripts/sdfbase/full22/make_bank_view.py ] && [ -x $(printf '%q' "$CH_PY") ]" \
  || fatal "cluster_b: no W_A checkout at $CH_REPO (README: git -C <the repo mirror> worktree add --detach $CH_REPO $WAVE_SHA) or no $CH_PY"
remote "${HOST[cluster_b]}" "cd $(printf '%q' "$CH_REPO") && [ \"\$(git rev-parse HEAD)\" = $(printf '%q' "$WAVE_SHA") ] && git diff --quiet HEAD --" \
  || fatal "cluster_b: $CH_REPO is not a clean checkout at $WAVE_SHA"

# --- stamp writer (the wave's hashcheck_and_stamp schema + the rung / ladder facts) -------------------------------
write_stamp(){ # out target i view sha lambda ihash ladder_sha source_dir shas_file rung_stamp_json
  python3 - "$@" <<'PY'
import json, re, sys, time
out, target, i, view, sha, lam, ihash, ladder_sha, source, shas_file, rung_stamp = sys.argv[1:]
hexre = re.compile(r"^[0-9a-f]{64}$")
staged_shas = {}
for line in open(shas_file):
    line = line.rstrip("\n")
    if not line: continue
    h, rel = line.split("  ", 1)
    staged_shas[rel] = h
assert staged_shas.get("object.pt") == sha, ("object.pt sha", staged_shas.get("object.pt"), sha)
rs = None
try: rs = json.load(open(rung_stamp))
except Exception as e: rs = {"unreadable": str(e)}
json.dump({"target": target, "lane": "selected", "cell_id": "alphasteer", "staged_id": "selected__alphasteer",
           "source": f"cluster_b:{source}", "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "object_sha256": sha, "staged_shas": staged_shas,
           "recorded_hash_check": {"status": "match", "object_sha256_recorded_in": ["lambda_ladder.json (chosen.object_sha256)", "dose_scaled_from.json"],
                                   "object_sha256_recorded_match": True},
           "rung": {"i": int(i), "lambda": float(lam), "view": view, "intervention_hash": ihash},
           "ladder_record_sha256": ladder_sha, "identity_sidecar": "fit_provenance.json",
           "rung_staged_from": rs, "staged_by": "stage_alphasteer.sh"},
          open(out, "w"), indent=2, sort_keys=True)
PY
}

log "=== stage_alphasteer.sh start: W_A=$WAVE_SHA targets=${TARGETS_ARG[*]:-all} dry_run=$DRY_RUN skip_view=$SKIP_VIEW cluster_a=$($BULB_OK && echo yes || echo skipped)"
NSTAGED=0; NFAIL=0
for row in "${PIN_ROWS[@]}"; do
  read -r target status i sha view ladder_sha lam ihash <<<"$row"
  want_target "$target" || continue
  if [ "$status" != served ]; then log "--- $target: SKIP ($status) — nothing staged, nothing served"; continue; fi
  ROOT_CH="${SHARED[cluster_b]}/$REL"
  SRC="$ROOT_CH/$target/ladder/lam$i/cells_view/$view/$CELL"
  DST="$ROOT_CH/$target/cells/$STAGED_ID"
  PARENT="$ROOT_CH/$target/cells/$CELL"
  LOCAL_REC="$RECORDS_ROOT/$target/lambda_ladder.json"
  key="${target}__${STAGED_ID}"; rec="$RECORDS/$key.json"
  log "--- $key <- cluster_b:$SRC (rung $i, lambda $lam, pin ${sha:0:16})"
  [ -f "$LOCAL_REC" ] && [ -f "$LOCAL_REC.COMPLETE" ] || { log "  no local selection record $LOCAL_REC (+.COMPLETE)"; NFAIL=$((NFAIL+1)); continue; }
  lsha=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['sha256'])" "$LOCAL_REC")
  [ "$lsha" = "$ladder_sha" ] || { log "  local record sha $lsha != the pins' $ladder_sha — regenerate the pins"; NFAIL=$((NFAIL+1)); continue; }
  probe=$(remote_probe "${HOST[cluster_b]}" "$SRC") || probe=$'SSH_FAILED\n-1'
  ssha=$(sed -n 1p <<<"$probe")
  [ "$ssha" = "$sha" ] || { log "  SOURCE rung object sha ${ssha:0:16} != pin ${sha:0:16} — refusing to stage"; NFAIL=$((NFAIL+1)); continue; }
  if $DRY_RUN; then log "  would rsync $SRC -> $DST, copy fit_meta/screen_f9, push lambda_ladder.json, stamp, verify, build view"; continue; fi
  # 1 + 2: within-cluster copy + the record files
  remote "${HOST[cluster_b]}" "set -e; mkdir -p $(printf '%q' "$DST")
rsync -rlt --no-owner --no-group --no-perms --partial --inplace $(printf '%q' "$SRC")/ $(printf '%q' "$DST")/
cp -f $(printf '%q' "$PARENT")/fit_meta.json $(printf '%q' "$DST")/fit_meta.json
[ -f $(printf '%q' "$PARENT")/fit_meta.json.COMPLETE ] && cp -f $(printf '%q' "$PARENT")/fit_meta.json.COMPLETE $(printf '%q' "$DST")/ || true
cp -f $(printf '%q' "$ROOT_CH/$target")/screen_f9.json $(printf '%q' "$DST")/screen_f9.json
[ -f $(printf '%q' "$DST")/staged_from.json ] && mv -f $(printf '%q' "$DST")/staged_from.json $(printf '%q' "$DST")/.rung_staged_from.json || true" \
    || { log "  COPY FAILED (cluster_b within-cluster)"; NFAIL=$((NFAIL+1)); continue; }
  rsync "${RSYNC_FLAGS[@]}" "$LOCAL_REC" "$LOCAL_REC.COMPLETE" "${HOST[cluster_b]}:$DST/" >>"$LOG" 2>&1 \
    || { log "  PUSH of lambda_ladder.json failed"; NFAIL=$((NFAIL+1)); continue; }
  # 3: sha on the cluster, every file hashed
  shas_file=$(mktemp); remote_shas "${HOST[cluster_b]}" "$DST" > "$shas_file" || { log "  remote hashing failed"; NFAIL=$((NFAIL+1)); continue; }
  rsha=$(awk '$2=="object.pt"{print $1}' "$shas_file")
  [ "$rsha" = "$sha" ] || { log "  STAGED object sha ${rsha:0:16} != pin ${sha:0:16}"; NFAIL=$((NFAIL+1)); continue; }
  # the bake's own stamp, embedded (pulled back for the record)
  rung_stamp=$(mktemp); rsync "${RSYNC_FLAGS[@]}" "${HOST[cluster_b]}:$DST/.rung_staged_from.json" "$rung_stamp" >>"$LOG" 2>&1 || echo '{}' > "$rung_stamp"
  # 4: stamp + push + count
  stamp=$(mktemp); write_stamp "$stamp" "$target" "$i" "$view" "$sha" "$lam" "$ihash" "$ladder_sha" "$SRC" "$shas_file" "$rung_stamp" \
    || { log "  stamp refused"; NFAIL=$((NFAIL+1)); continue; }
  rsync "${RSYNC_FLAGS[@]}" "$stamp" "${HOST[cluster_b]}:$DST/staged_from.json" >>"$LOG" 2>&1 || { log "  PUSH of staged_from.json failed"; NFAIL=$((NFAIL+1)); continue; }
  n_expect=$(( $(wc -l < "$shas_file") + 1 ))     # every hashed file + staged_from.json (the bake's stamp is hidden and counted already)
  probe=$(remote_probe "${HOST[cluster_b]}" "$DST") || probe=$'SSH_FAILED\n-1'
  rsha=$(sed -n 1p <<<"$probe"); rcount=$(sed -n 2p <<<"$probe")
  ch_ok=false; [ "$rsha" = "$sha" ] && [ "$rcount" = "$n_expect" ] && ch_ok=true
  log "  [cluster_b] verified=$ch_ok remote_sha=${rsha:0:16} n_files=$rcount (expected $n_expect)"
  # Cluster A half (routed cluster_b -> the grading host -> cluster_a), gated on reachability
  bu_ok=skipped; bu_path="${SHARED[cluster_a]:+${SHARED[cluster_a]}/$REL/$target/cells/$STAGED_ID}"
  if $BULB_OK; then
    tmpd=$(mktemp -d)
    if rsync "${RSYNC_FLAGS[@]}" "${HOST[cluster_b]}:$DST/" "$tmpd/" >>"$LOG" 2>&1 \
       && remote "${HOST[cluster_a]}" "mkdir -p $(printf '%q' "$bu_path")" \
       && rsync "${RSYNC_FLAGS[@]}" "$tmpd/" "${HOST[cluster_a]}:$bu_path/" >>"$LOG" 2>&1; then
      bprobe=$(remote_probe "${HOST[cluster_a]}" "$bu_path") || bprobe=$'SSH_FAILED\n-1'
      [ "$(sed -n 1p <<<"$bprobe")" = "$sha" ] && [ "$(sed -n 2p <<<"$bprobe")" = "$n_expect" ] && bu_ok=true || bu_ok=false
    else bu_ok=false; fi
    rm -rf "$tmpd"; log "  [cluster_a] verified=$bu_ok"
  fi
  # 5: the record row (the wave's row schema)
  python3 - "$rec" "$target" "$i" "$lam" "$sha" "$n_expect" "$SRC" "$DST" "$ch_ok" "$rsha" "$rcount" "$bu_path" "$bu_ok" "$ladder_sha" <<'PY'
import json, sys, time
rec, target, i, lam, sha, n, src, dst, ch_ok, rsha, rcount, bu_path, bu_ok, ladder_sha = sys.argv[1:]
row = {"target": target, "lane": "selected", "cell_id": "alphasteer", "staged_id": "selected__alphasteer",
       "source_path": f"cluster_b:{src}", "rung_i": int(i), "lambda": float(lam), "object_sha256": sha, "n_files": int(n),
       "cluster_b_path": dst, "cluster_b_verified": ch_ok == "true", "cluster_b_action": "copied_within_cluster",
       "cluster_b_remote_sha256": rsha, "cluster_b_remote_n_files": int(rcount) if rcount.lstrip("-").isdigit() else None,
       "cluster_a_path": bu_path, "cluster_a_verified": (True if bu_ok == "true" else False if bu_ok == "false" else None),
       "cluster_a_action": ("routed_via_grading_host" if bu_ok in ("true", "false") else "skipped: unreachable"),
       "ladder_record_sha256": ladder_sha, "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
       "status": "staged" if ch_ok == "true" and bu_ok != "false" else "verify_failed", "staged_by": "stage_alphasteer.sh"}
json.dump(row, open(rec, "w"), indent=2, sort_keys=True)
PY
  rm -f "$shas_file" "$stamp" "$rung_stamp"
  $ch_ok || { NFAIL=$((NFAIL+1)); continue; }
  NSTAGED=$((NSTAGED+1))
  # 6: the bank view, OUR roster / pins / bank, from the W_A checkout — then --check
  if ! $SKIP_VIEW; then
    remote "${HOST[cluster_b]}" "cd $(printf '%q' "$CH_REPO") && $(printf '%q' "$CH_PY") scripts/sdfbase/full22/make_bank_view.py --roster configs/alphasteer_f22_roster.yaml --pins configs/alphasteer_f22_pins.json --bank $(printf '%q' "$ROOT_CH/$target") --target $(printf '%q' "$target") --view $(printf '%q' "$view") && $(printf '%q' "$CH_PY") scripts/sdfbase/full22/make_bank_view.py --check --roster configs/alphasteer_f22_roster.yaml --pins configs/alphasteer_f22_pins.json --bank $(printf '%q' "$ROOT_CH/$target") --target $(printf '%q' "$target") --view $(printf '%q' "$view")" \
      && log "  [cluster_b] bank view $target/$view built + verified" || { log "  [cluster_b] bank view $target/$view REFUSED"; NFAIL=$((NFAIL+1)); }
    if [ "$bu_ok" = true ]; then
      BU_REPO="${EVAL[cluster_a]}/$WAVE_SHA"; BU_PY="${SHARED[cluster_a]}/envs/vllm-attacker/bin/python"
      remote "${HOST[cluster_a]}" "cd $(printf '%q' "$BU_REPO") && $(printf '%q' "$BU_PY") scripts/sdfbase/full22/make_bank_view.py --roster configs/alphasteer_f22_roster.yaml --pins configs/alphasteer_f22_pins.json --bank $(printf '%q' "${SHARED[cluster_a]}/$REL/$target") --target $(printf '%q' "$target") --view $(printf '%q' "$view") && $(printf '%q' "$BU_PY") scripts/sdfbase/full22/make_bank_view.py --check --roster configs/alphasteer_f22_roster.yaml --pins configs/alphasteer_f22_pins.json --bank $(printf '%q' "${SHARED[cluster_a]}/$REL/$target") --target $(printf '%q' "$target") --view $(printf '%q' "$view")" \
        && log "  [cluster_a] bank view $target/$view built + verified" || log "  [cluster_a] bank view $target/$view not built (no W_A checkout there, or refused) — recorded, not fatal"
    fi
  fi
done

# --- the manifest: every record row, pushed beside the objects ------------------------------------------------
python3 - "$RECORDS" "$MANIFEST" <<'PY'
import glob, json, os, sys
recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))]
json.dump(recs, open(sys.argv[2], "w"), indent=2, sort_keys=True)
print(f"MANIFEST {sys.argv[2]}: {len(recs)} rows, {sum(r.get('status') == 'staged' for r in recs)} staged")
PY
if ! $DRY_RUN && [ -s "$MANIFEST" ]; then
  rsync "${RSYNC_FLAGS[@]}" "$MANIFEST" "${HOST[cluster_b]}:${SHARED[cluster_b]}/$REL/staging_manifest.json" >>"$LOG" 2>&1 && log "manifest -> cluster_b OK" || log "manifest -> cluster_b FAILED"
  $BULB_OK && { rsync "${RSYNC_FLAGS[@]}" "$MANIFEST" "${HOST[cluster_a]}:${SHARED[cluster_a]}/$REL/staging_manifest.json" >>"$LOG" 2>&1 && log "manifest -> cluster_a OK" || log "manifest -> cluster_a FAILED"; }
fi
log "=== stage_alphasteer.sh done: staged=$NSTAGED failed=$NFAIL"
[ "$NFAIL" -eq 0 ]
