#!/usr/bin/env bash
# stage_strength.sh — STRENGTH FRONTIER (paper Figure 5), AlphaSteer lane: verify the strength cells bake_strength.py
# wrote on cluster B against the pins, mirror them to cluster A, verify there, and build + check the five per-target
# bank views on BOTH clusters with plain make_bank_view.py over OUR roster + pins (exactly as stage_alphasteer.sh step 6
# does for the campaign — the campaign's bank view (make_bank_view.py over its roster + pins) binds every served object to lambda_ladder.json's chosen sha
# and would refuse these copies, so it is not used). Runs on the grading host, driving both clusters over ssh.
#
#   bash scripts/alphasteer_f22/stage_strength.sh <SHA> [--lane L] [--target T ...] [--skip-copy] [--skip-view] [--dry-run]
#
# --lane (as22_strength.LANES; default alphasteer_f22): picks the roster, the pins, the bank root under $SHARED and the
# staging state on the grading host; the lane's TARGETS are its pins' keys (which come from its roster). For a lane whose
# parents exist on cluster B only (the re-baked parents), the PARENT dir <ROOT>/<t>/cells/alphasteer/ is MIRRORED to
# cluster A first (rsync, sha-verified against cluster B's; an existing parent with a DIFFERENT sha is refused, never
# overwritten) because the as_d1 anchor symlinks to it and bake_strength.py --check loads it there.
#
# <SHA> is the commit checked out at $SHARED/eval-repo/<SHA> on both clusters (the README convention:
# git -C $CLUSTER_REPO_MIRROR worktree add --detach $SHARED/eval-repo/<SHA> <SHA>); it must carry
# configs/alphasteer_f22_strength_{roster.yaml,pins.json}. Python: $SHARED/envs/vllm-attacker/bin/python (torch +
# yaml, the one stage_alphasteer.sh uses).
#
# Per target (configs/alphasteer_f22_strength_pins.json — every entry must be `served`; a pending target refuses):
#   1. cluster_b: <ROOT>/<t>/strength/cells/<sid>/object.pt sha256 == the pin for every staged id (sha256sum follows
#      the as_d1 symlink to the parent's bytes); bake_strength.py --check re-verifies the whole bank from disk
#   2. copy (unless --skip-copy, which instead VERIFIES the already-copied bank on cluster_a in place):
#      <ROOT>/<t>/strength/{cells/,strength_bake.json} cluster_b -> grading host tmp -> cluster_a,
#      rsync -rlt --no-owner --no-group --no-perms --partial --inplace (NEVER -a: MooseFS refuses chown and leaves a
#      silently incomplete tree; -l keeps the anchor's symlink a symlink — the parent exists on cluster A at the same sha)
#   3. cluster_a: every sha == pin, the file COUNT equal to cluster_b's, and bake_strength.py --check against
#      that cluster's own parent
#   4. views (unless --skip-view): on each cluster, for each of the five views, make_bank_view.py --roster OURS --pins
#      OURS --bank <ROOT>/<t>/strength --target <t> --view <view>, then --check. Steps 1, 3 and 4 torch.load the
#      objects, so they run under short CPU-ONLY srun allocations on general at the high QoS (no --gres), never on
#      the login shell and never on the two-node dev partition
#   5. a row per (target, view) appended to the lane's staging manifest, pushed to <ROOT>/strength_staging_manifest.json
#
# Writes ONLY under <ROOT>/<t>/strength/ and <ROOT>/strength_staging_manifest.json. No STOP file, no GPU job, no git.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
LANE=alphasteer_f22
for ((i=1; i<=$#; i++)); do [ "${!i}" = --lane ] && { j=$((i+1)); LANE="${!j}"; }; done
# the lane's static facts (as22_strength.LANES; no roster read, stdlib only)
read -r REL PINS_REL ROSTER_REL RECORDS_SUBDIR < <(python3 -c 'import sys; sys.path.insert(0, sys.argv[1]); import as22_strength as st; ln = st.lane(sys.argv[2]); print(ln.bank_root_rel, ln.pins_rel, ln.roster_rel, ln.records_subdir)' "$HERE" "$LANE") \
  || { echo "unknown --lane $LANE" >&2; exit 2; }
PINS="$REPO/$PINS_REL"; ROSTER="$REPO/$ROSTER_REL"
STAGE_STATE="${AS22_STRENGTH_STAGE_STATE:-$(PYTHONPATH="$REPO/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m sisteer.site expand "\$GRADE/$RECORDS_SUBDIR/staging")}"
STAGE_STATE="${STAGE_STATE/#\~/$HOME}"   # the site module prints a leading ~ as written
[ -n "$STAGE_STATE" ] || { echo "stage_strength.sh: set SISTEER_GRADE_DIR (the grade root) or AS22_STRENGTH_STAGE_STATE" >&2; exit 2; }
RECORDS="$STAGE_STATE/_records"; LOG="$STAGE_STATE/stage_strength.log"; MANIFEST="$STAGE_STATE/strength_staging_manifest.json"
# Each cluster's login host, shared root, job partition and QoS and per-commit checkouts root come from its block in the site
# config (reproduce/lib/clusters.py: CLUSTER_SSH, CLUSTER_SHARED_ROOT, CLUSTER_PARTITION, CLUSTER_QOS, CLUSTER_EVAL_REPO_ROOT,
# looked up by the cluster's registry label); the labels stay keys only.
declare -A HOST=() SHARED=() PART=() HIQOS=() EVAL=()
site_cluster(){   # <registry label>: HOST, SHARED, PART, HIQOS and EVAL of that cluster from the site config, else a refusal by name
  local h="" s="" p="" q="" e=""
  h=$(python3 "$REPO/reproduce/lib/clusters.py" --get "$1" CLUSTER_SSH) \
    && s=$(python3 "$REPO/reproduce/lib/clusters.py" --get "$1" CLUSTER_SHARED_ROOT) \
    && p=$(python3 "$REPO/reproduce/lib/clusters.py" --get "$1" CLUSTER_PARTITION) \
    && q=$(python3 "$REPO/reproduce/lib/clusters.py" --get "$1" CLUSTER_QOS) \
    && e=$(python3 "$REPO/reproduce/lib/clusters.py" --get "$1" CLUSTER_EVAL_REPO_ROOT) || true
  [ -n "$h" ] && [ -n "$s" ] && [ -n "$p" ] && [ -n "$q" ] && [ -n "$e" ] \
    || { echo "stage_strength.sh: the site config gives no login host, shared root, partition, QoS and checkouts root (CLUSTER_SSH, CLUSTER_SHARED_ROOT, CLUSTER_PARTITION, CLUSTER_QOS, CLUSTER_EVAL_REPO_ROOT) for cluster $1" >&2; exit 2; }
  HOST[$1]=$h; SHARED[$1]=${s%/}; PART[$1]=$p; HIQOS[$1]=$q; EVAL[$1]=${e%/}
}
RSYNC_FLAGS=(-rlt --no-owner --no-group --no-perms --partial --inplace)
# the object-loading steps (bake_strength.py --check, make_bank_view.py's fit_provenance verdict) torch.load up to
# 1.2 GB per object: run them under a short CPU-ONLY allocation on each cluster, never on the login shell. On the
# BIG partition (general, 28 nodes) at the default QoS with NO --gres — a zero-GPU job draws nothing from the 16-GPU
# budget and places at once — never on `dev` (two nodes, shared with everyone's interactive GPU sessions, where the
# first pass sat PENDING on priority for symlink work). The D_ job-name prefix keeps the nightly reaper's view of it.
srun_view(){ echo "srun -p ${PART[$1]} --qos=${HIQOS[$1]} --cpus-per-task=2 --mem=8G --time=0:10:00 --job-name=D_${USER}_asview-$2"; }
srun_check(){ echo "srun -p ${PART[$1]} --qos=${HIQOS[$1]} --cpus-per-task=4 --mem=40G --time=0:30:00 --job-name=D_${USER}_ascheck-$2"; }
SHA=""; TARGETS_ARG=(); DRY_RUN=false; SKIP_COPY=false; SKIP_VIEW=false
while [ $# -gt 0 ]; do
  case "$1" in
    --lane) shift ;;                                   # consumed above
    --target) TARGETS_ARG+=("$2"); shift ;;
    --dry-run) DRY_RUN=true ;;
    --skip-copy) SKIP_COPY=true ;;
    --skip-view) SKIP_VIEW=true ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    -*) echo "unknown arg: $1" >&2; exit 2 ;;
    *) [ -z "$SHA" ] && SHA="$1" || { echo "unexpected positional $1" >&2; exit 2; } ;;
  esac; shift
done
[ -n "$SHA" ] || { echo "usage: stage_strength.sh <SHA> [--lane L] [--target T] [--skip-copy] [--skip-view] [--dry-run]" >&2; exit 2; }
site_cluster cluster_b; site_cluster cluster_a
mkdir -p "$RECORDS"
log(){ printf '%s %s\n' "$(date -u +%FT%TZ)" "$*" | tee -a "$LOG" >&2; }
fatal(){ log "FATAL: $*"; exit 1; }

# a bash snippet on a cluster host over STDIN into `bash -s` (the remote login shell is zsh: one non-matching glob
# aborts a whole line there); 3 attempts
remote(){ local host=$1 script=$2 a; for a in 1 2 3; do printf '%s\n' "$script" | ssh -o BatchMode=yes -o ConnectTimeout=30 "$host" bash -s && return 0; sleep $((a*5)); done; return 1; }
# `<sid> <sha-or-MISSING>` per staged id under <bank>/cells, then a line `COUNT <n regular files, symlinks followed>`
remote_probe(){ remote "$1" "b=$(printf '%q' "$2")
for d in \"\$b\"/cells/*/; do s=\$(basename \"\$d\"); if [ -f \"\$d/object.pt\" ]; then echo \"\$s \$(sha256sum \"\$d/object.pt\" | cut -d' ' -f1)\"; else echo \"\$s MISSING\"; fi; done
echo COUNT \$(find -L \"\$b\"/cells -type f 2>/dev/null | wc -l)"; }

# --- the pins decide what is staged ------------------------------------------------------------------------------
mapfile -t PIN_ROWS < <(python3 - "$PINS" "$ROSTER" <<'PY'
import json, sys, yaml
pins = json.load(open(sys.argv[1]))
for t in yaml.safe_load(open(sys.argv[2]))["targets"]:
    for sid, e in pins[t].items():
        print(t, sid, e["status"], e.get("object_sha256"), e.get("view"), e.get("strength"), e.get("is_parent"))
PY
) || fatal "cannot read $PINS / $ROSTER"
want_target(){ [ "${#TARGETS_ARG[@]}" -eq 0 ] && return 0; local t; for t in "${TARGETS_ARG[@]}"; do [ "$t" = "$1" ] && return 0; done; return 1; }
mapfile -t TARGETS < <(printf '%s\n' "${PIN_ROWS[@]}" | awk '{print $1}' | awk '!seen[$0]++')
for t in "${TARGETS_ARG[@]}"; do printf '%s\n' "${TARGETS[@]}" | grep -qx "$t" || fatal "--target $t is not a lane target"; done
for row in "${PIN_ROWS[@]}"; do read -r t sid st sha _ <<<"$row"; want_target "$t" || continue
  [ "$st" = served ] && [ "${#sha}" -eq 64 ] || fatal "$t/$sid has pin status $st (sha ${sha:-null}) — regenerate the pins from the bake records first"; done

# --- reachability + checkouts --------------------------------------------------------------------------------------
reach(){ ssh -o BatchMode=yes -o ConnectTimeout=15 "$1" true 2>/dev/null; }
reach "${HOST[cluster_b]}" || fatal "cluster_b unreachable — the bake wrote there; nothing to stage without it"
reach "${HOST[cluster_a]}" || fatal "cluster_a unreachable — this lane verifies on BOTH clusters; retry when its login host answers"
for c in cluster_b cluster_a; do
  R="${EVAL[$c]}/$SHA"; PY="${SHARED[$c]}/envs/vllm-attacker/bin/python"
  remote "${HOST[$c]}" "[ -f $(printf '%q' "$R")/scripts/alphasteer_f22/bake_strength.py ] && [ -f $(printf '%q' "$R/$PINS_REL") ] && [ -x $(printf '%q' "$PY") ]" \
    || fatal "$c: no checkout carrying bake_strength.py + $PINS_REL at $R (git -C <the repo mirror> worktree add --detach $R $SHA), or no $PY"
  remote "${HOST[$c]}" "cd $(printf '%q' "$R") && [ \"\$(git rev-parse HEAD)\" = $(printf '%q' "$SHA") ] && git diff --quiet HEAD --" \
    || fatal "$c: $R is not a clean checkout at $SHA"
done

bank_view(){ # cluster target view  -> build (if it does not verify) then --check, from the checkout
  local c=$1 t=$2 v=$3 R PY bank
  R="${EVAL[$c]}/$SHA"; PY="${SHARED[$c]}/envs/vllm-attacker/bin/python"; bank="${SHARED[$c]}/$REL/$t/strength"
  remote "${HOST[$c]}" "cd $(printf '%q' "$R") && A=(--roster $(printf '%q' "$ROSTER_REL") --pins $(printf '%q' "$PINS_REL") --bank $(printf '%q' "$bank") --target $(printf '%q' "$t") --view $(printf '%q' "$v"))
$(srun_view "$c" "$t") $(printf '%q' "$PY") scripts/sdfbase/full22/make_bank_view.py --check \"\${A[@]}\" >/dev/null 2>&1 || $(srun_view "$c" "$t") $(printf '%q' "$PY") scripts/sdfbase/full22/make_bank_view.py \"\${A[@]}\"
$(srun_view "$c" "$t") $(printf '%q' "$PY") scripts/sdfbase/full22/make_bank_view.py --check \"\${A[@]}\""
}
bake_check(){ # cluster target -> bake_strength.py --check from the checkout against THAT cluster's parent
  local c=$1 t=$2 R PY
  R="${EVAL[$c]}/$SHA"; PY="${SHARED[$c]}/envs/vllm-attacker/bin/python"
  remote "${HOST[$c]}" "cd $(printf '%q' "$R") && $(srun_check "$c" "$t") $(printf '%q' "$PY") scripts/alphasteer_f22/bake_strength.py --check --lane $(printf '%q' "$LANE") --target $(printf '%q' "$t") --root $(printf '%q' "${SHARED[$c]}/$REL")"
}

export STAGE_LANE="$LANE"
log "=== stage_strength.sh start: SHA=$SHA lane=$LANE root=$REL targets=${TARGETS_ARG[*]:-all} dry_run=$DRY_RUN skip_copy=$SKIP_COPY skip_view=$SKIP_VIEW"
NOK=0; NFAIL=0
for t in "${TARGETS[@]}"; do
  want_target "$t" || continue
  BANK_CH="${SHARED[cluster_b]}/$REL/$t/strength"; BANK_BU="${SHARED[cluster_a]}/$REL/$t/strength"
  mapfile -t ROWS < <(printf '%s\n' "${PIN_ROWS[@]}" | awk -v t="$t" '$1==t')
  log "--- $t: ${#ROWS[@]} cells"
  # 1. cluster_b shas + count + the bake's own re-verification
  probe=$(remote_probe "${HOST[cluster_b]}" "$BANK_CH") || { log "  cluster_b probe failed"; NFAIL=$((NFAIL+1)); continue; }
  ch_ok=true
  for row in "${ROWS[@]}"; do read -r _ sid _ sha _ <<<"$row"; got=$(awk -v s="$sid" '$1==s{print $2}' <<<"$probe")
    [ "$got" = "$sha" ] || { log "  cluster_b $sid sha ${got:-absent} != pin ${sha:0:16}"; ch_ok=false; }; done
  ch_count=$(awk '$1=="COUNT"{print $2}' <<<"$probe")
  $ch_ok || { NFAIL=$((NFAIL+1)); continue; }
  if $DRY_RUN; then log "  would: bake --check (cluster_b), copy -> cluster_a, verify, build 5 views on both"; continue; fi
  bake_check cluster_b "$t" >>"$LOG" 2>&1 && log "  [cluster_b] bake_strength --check VERIFIED (n_files=$ch_count)" \
    || { log "  [cluster_b] bake_strength --check REFUSED (see $LOG)"; NFAIL=$((NFAIL+1)); continue; }
  # the PARENT on cluster_a: the as_d1 anchor links to it and --check loads it. Mirror it when absent (sha-verified
  # against cluster_b's), refuse when present with a different sha, never overwrite.
  PARENT_CH="${SHARED[cluster_b]}/$REL/$t/cells/alphasteer"; PARENT_BU="${SHARED[cluster_a]}/$REL/$t/cells/alphasteer"
  psha_ch=$(remote "${HOST[cluster_b]}" "sha256sum $(printf '%q' "$PARENT_CH")/object.pt | cut -d' ' -f1") || { log "  cluster_b parent probe failed"; NFAIL=$((NFAIL+1)); continue; }
  psha_bu=$(remote "${HOST[cluster_a]}" "[ -f $(printf '%q' "$PARENT_BU")/object.pt ] && sha256sum $(printf '%q' "$PARENT_BU")/object.pt | cut -d' ' -f1 || echo ABSENT") || psha_bu=PROBE_FAILED
  if [ "$psha_bu" = ABSENT ]; then
    tmpp=$(mktemp -d "${TMPDIR:-/tmp}/as-parent-$t.XXXXXX")
    if rsync "${RSYNC_FLAGS[@]}" "${HOST[cluster_b]}:$PARENT_CH/" "$tmpp/" >>"$LOG" 2>&1 \
       && remote "${HOST[cluster_a]}" "mkdir -p $(printf '%q' "$PARENT_BU")" \
       && rsync "${RSYNC_FLAGS[@]}" "$tmpp/" "${HOST[cluster_a]}:$PARENT_BU/" >>"$LOG" 2>&1; then
      psha_bu=$(remote "${HOST[cluster_a]}" "sha256sum $(printf '%q' "$PARENT_BU")/object.pt | cut -d' ' -f1") || psha_bu=PROBE_FAILED
      if [ "$psha_bu" = "$psha_ch" ]; then log "  [cluster_a] parent MIRRORED from cluster_b (${psha_ch:0:16}, $(ls "$tmpp" | wc -l) files)"
      else log "  [cluster_a] parent mirror sha ${psha_bu:0:16} != cluster_b ${psha_ch:0:16}"; rm -rf "$tmpp"; NFAIL=$((NFAIL+1)); continue; fi
    else log "  PARENT MIRROR FAILED (see $LOG)"; rm -rf "$tmpp"; NFAIL=$((NFAIL+1)); continue; fi
    rm -rf "$tmpp"
  elif [ "$psha_bu" != "$psha_ch" ]; then
    log "  [cluster_a] parent at $PARENT_BU has sha ${psha_bu:0:16} != cluster_b ${psha_ch:0:16} — REFUSING to touch it"; NFAIL=$((NFAIL+1)); continue
  else log "  [cluster_a] parent present (${psha_ch:0:16})"; fi
  # 2 + 3. the copy, routed through the grading host, then verify on cluster_a
  bu_ok=skipped
  if ! $SKIP_COPY; then
    tmpd=$(mktemp -d "${TMPDIR:-/tmp}/as-strength-$t.XXXXXX")
    if rsync "${RSYNC_FLAGS[@]}" "${HOST[cluster_b]}:$BANK_CH/cells/" "$tmpd/cells/" >>"$LOG" 2>&1 \
       && rsync "${RSYNC_FLAGS[@]}" "${HOST[cluster_b]}:$BANK_CH/strength_bake.json" "$tmpd/strength_bake.json" >>"$LOG" 2>&1 \
       && remote "${HOST[cluster_a]}" "mkdir -p $(printf '%q' "$BANK_BU")/cells" \
       && rsync "${RSYNC_FLAGS[@]}" "$tmpd/cells/" "${HOST[cluster_a]}:$BANK_BU/cells/" >>"$LOG" 2>&1 \
       && rsync "${RSYNC_FLAGS[@]}" "$tmpd/strength_bake.json" "${HOST[cluster_a]}:$BANK_BU/strength_bake.json" >>"$LOG" 2>&1; then
      bprobe=$(remote_probe "${HOST[cluster_a]}" "$BANK_BU") || bprobe=""
      bu_ok=true
      for row in "${ROWS[@]}"; do read -r _ sid _ sha _ <<<"$row"; got=$(awk -v s="$sid" '$1==s{print $2}' <<<"$bprobe")
        [ "$got" = "$sha" ] || { log "  cluster_a $sid sha ${got:-absent} != pin ${sha:0:16}"; bu_ok=false; }; done
      bu_count=$(awk '$1=="COUNT"{print $2}' <<<"$bprobe")
      [ "$bu_count" = "$ch_count" ] || { log "  cluster_a file count $bu_count != cluster_b $ch_count"; bu_ok=false; }
      if [ "$bu_ok" = true ]; then
        bake_check cluster_a "$t" >>"$LOG" 2>&1 && log "  [cluster_a] copied + shas == pins + bake_strength --check VERIFIED (n_files=$bu_count)" \
          || { log "  [cluster_a] bake_strength --check REFUSED (see $LOG)"; bu_ok=false; }
      fi
    else log "  COPY FAILED (see $LOG)"; bu_ok=false; fi
    rm -rf "$tmpd"
  else
    # --skip-copy is a RESUME after a completed copy: verify what is on cluster_a in place (shas, count, --check)
    bprobe=$(remote_probe "${HOST[cluster_a]}" "$BANK_BU") || bprobe=""
    bu_ok=true
    for row in "${ROWS[@]}"; do read -r _ sid _ sha _ <<<"$row"; got=$(awk -v s="$sid" '$1==s{print $2}' <<<"$bprobe")
      [ "$got" = "$sha" ] || { log "  cluster_a $sid sha ${got:-absent} != pin ${sha:0:16} (--skip-copy: nothing copied)"; bu_ok=false; }; done
    bu_count=$(awk '$1=="COUNT"{print $2}' <<<"$bprobe")
    [ "$bu_count" = "$ch_count" ] || { log "  cluster_a file count ${bu_count:-?} != cluster_b $ch_count"; bu_ok=false; }
    if [ "$bu_ok" = true ]; then
      bake_check cluster_a "$t" >>"$LOG" 2>&1 && log "  [cluster_a] already staged: shas == pins + bake_strength --check VERIFIED (n_files=$bu_count)" \
        || { log "  [cluster_a] bake_strength --check REFUSED (see $LOG)"; bu_ok=false; }
    fi
  fi
  [ "$bu_ok" = false ] && { NFAIL=$((NFAIL+1)); continue; }
  # 4. the views, both clusters
  views_ok=true
  if ! $SKIP_VIEW; then
    for row in "${ROWS[@]}"; do read -r _ sid _ sha view _ <<<"$row"
      for c in cluster_b cluster_a; do
        [ "$c" = cluster_a ] && [ "$bu_ok" != true ] && continue
        if bank_view "$c" "$t" "$view" >>"$LOG" 2>&1; then log "  [$c] view $t/$view built + verified (${sha:0:12})"
        else log "  [$c] view $t/$view REFUSED (see $LOG)"; views_ok=false; fi
      done
    done
  fi
  # 5. rows
  for row in "${ROWS[@]}"; do read -r _ sid _ sha view s isp <<<"$row"
    python3 - "$RECORDS/${t}__${sid}.json" "$t" "$sid" "$view" "$s" "$isp" "$sha" "$BANK_CH" "$BANK_BU" "$bu_ok" "$views_ok" "$SHA" "$ch_count" "${bu_count:-}" <<'PY'
import json, os, sys, time
rec, t, sid, view, s, isp, sha, bch, bbu, bu_ok, views_ok, code_sha, nch, nbu = sys.argv[1:]
json.dump({"target": t, "lane": os.environ.get("STAGE_LANE", "alphasteer_f22"), "cell_id": "alphasteer", "staged_id": sid, "view": view, "strength": float(s),
           "is_parent": isp == "True", "object_sha256": sha,
           "cluster_b_path": f"{bch}/cells/{sid}", "cluster_b_verified": True, "cluster_b_action": "baked_in_place",
           "cluster_a_path": f"{bbu}/cells/{sid}", "cluster_a_verified": (True if bu_ok == "true" else False if bu_ok == "false" else None),
           "cluster_a_action": ("routed_via_grading_host_or_verified_in_place" if bu_ok in ("true", "false") else "skipped"),
           "n_files_cluster_b": int(nch) if nch.isdigit() else None, "n_files_cluster_a": int(nbu) if nbu.isdigit() else None,
           "views_built_and_checked": views_ok == "true", "checkout_sha": code_sha,
           "pulled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "status": "staged" if bu_ok != "false" and views_ok == "true" else "verify_failed", "staged_by": "stage_strength.sh"},
          open(rec, "w"), indent=2, sort_keys=True)
PY
  done
  $views_ok && NOK=$((NOK+1)) || NFAIL=$((NFAIL+1))
done

python3 - "$RECORDS" "$MANIFEST" <<'PY'
import glob, json, os, sys
recs = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json")))]
json.dump(recs, open(sys.argv[2], "w"), indent=2, sort_keys=True)
print(f"MANIFEST {sys.argv[2]}: {len(recs)} rows, {sum(r.get('status') == 'staged' for r in recs)} staged")
PY
if ! $DRY_RUN && [ -s "$MANIFEST" ]; then
  for c in cluster_b cluster_a; do
    rsync "${RSYNC_FLAGS[@]}" "$MANIFEST" "${HOST[$c]}:${SHARED[$c]}/$REL/strength_staging_manifest.json" >>"$LOG" 2>&1 && log "manifest -> $c OK" || log "manifest -> $c FAILED"
  done
fi
log "=== stage_strength.sh done: targets_ok=$NOK failed=$NFAIL"
[ "$NFAIL" -eq 0 ]
