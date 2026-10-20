#!/bin/bash
# scripts/sdfbase/seedrep/colocate_cells.sh — put a seed key's parent cells (and its pool dir) on
# the DERIVATION cluster before the derived chain runs there.
# A fit that overflowed to the other cluster is COPIED, never moved; the copy is verified
# by sha256 against the location ledger on BOTH sides; the mirror gets its OWN artifact-location
# record (copied_from = the source cluster; `kind` PROPAGATED from the source record — SCHEMAS.md
# §2, the stager requires it — and `fit` when the source carries none: every cell this script
# moves is a parent fit under cells_all/). Runs on the grading host: the two clusters have no
# credentials for each other, so every byte goes cluster -> the grading host staging -> cluster.
#
#   colocate_cells.sh <key> <target-cluster> [--bank reseed] [--cells c1,c2,...] [--rs <RS root>]
#                     [--staging <dir>] [--dry-run]
#     key             qwen35_4b_base_eb64_s<s>
#     target-cluster  cluster_a | cluster_b (the cluster the derived chain will run on: the one
#                     holding >= 3 of the seed's 4 parents, cluster A on ties — the operator's call)
#     --cells         default: the seed's four parents (constv2-mix-full / constv2ctl2-full x
#                     psr_ce_all / lora_r4_ce)
# Reads where each cell is from `artifact_locations.py view --key <key>` (records the fit watcher's
# DONE handler wrote per (key, cell, cluster)); a cell already recorded on the target is verified
# (object.pt hashes to its record) and skipped. rsync flags are the MooseFS-safe set on BOTH legs
# (-rlt --no-owner --no-group --no-perms --partial --inplace: cluster B's /workspace refuses chown
# and a plain -a leaves a silently incomplete tree); a sha or file-count mismatch REFUSES (exit 4),
# writes no record and deletes nothing.
#
# Test seams (never set in production): SEEDREP_CLUSTER_ROOT_<cluster>=<dir> makes that cluster a
# local directory standing in for its $SHARED (no ssh); SEEDREP_RSYNC=<binary> replaces rsync.
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCATIONS="$HERE/artifact_locations.py"
RSYNC_BIN="${SEEDREP_RSYNC:-rsync}"
RSYNC_FLAGS=(-rlt --no-owner --no-group --no-perms --partial --inplace)
DEFAULT_CELLS="constv2-mix-full__psr_ce_all,constv2-mix-full__lora_r4_ce,constv2ctl2-full__psr_ce_all,constv2ctl2-full__lora_r4_ce"

usage() { sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
KEY="${1:-}"; TARGET="${2:-}"; shift 2 2>/dev/null || usage 2
[ -n "$KEY" ] && [ -n "$TARGET" ] || usage 2
BANK=reseed; CELLS="$DEFAULT_CELLS"; RS_ARG=(); STAGING="${SEEDREP_STAGING:-$HOME/.cache/seedrep-colocate}"; DRY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --bank) BANK="$2"; shift 2 ;;
    --cells) CELLS="$2"; shift 2 ;;
    --rs) RS_ARG=(--rs "$2"); shift 2 ;;
    --staging) STAGING="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "[colocate] unknown argument $1" >&2; usage 2 ;;
  esac
done
case "$TARGET" in cluster_a|cluster_b) ;; *) echo "[colocate] target cluster must be cluster_a|cluster_b, got '$TARGET'" >&2; exit 2 ;; esac
log() { echo "[colocate $KEY -> $TARGET] $*"; }

# --- the two clusters -------------------------------------------------------------------------------
# Each cluster's login host and shared root come from its block in the site config (reproduce/lib/clusters.py, looked up by
# the cluster's registry label), resolved once per cluster in this shell before the first command that needs them; a cluster
# the site config does not describe stops the run by name. The labels themselves stay record keys only.
CL_PY="$(cd "$HERE/../../.." && pwd)/reproduce/lib/clusters.py"
declare -A SSH_OF=() SHARED_OF=()
need() {
  local c="$1" v="SEEDREP_CLUSTER_ROOT_$1" h s
  [ -n "${SHARED_OF[$c]:-}" ] && return 0
  if [ -n "${!v:-}" ]; then SHARED_OF[$c]="${!v}"; return 0; fi
  h="$(python3 "$CL_PY" --get "$c" CLUSTER_SSH)" && [ -n "$h" ] \
    && s="$(python3 "$CL_PY" --get "$c" CLUSTER_SHARED_ROOT)" && [ -n "$s" ] \
    || { echo "[colocate] REFUSED: the site config gives no login host and shared root (CLUSTER_SSH, CLUSTER_SHARED_ROOT) for cluster $c" >&2; exit 2; }
  SSH_OF[$c]="$h"; SHARED_OF[$c]="$s"
}
host_of()   { echo "${SSH_OF[$1]}"; }
shared_of() { echo "${SHARED_OF[$1]}"; }
is_local()  { local v="SEEDREP_CLUSTER_ROOT_$1"; [ -n "${!v:-}" ]; }
rsh()       { local c="$1"; shift; if is_local "$c"; then bash -c "$*"; else ssh -o BatchMode=yes "$(host_of "$c")" "$*"; fi; }
rspec()     { if is_local "$1"; then echo "$2"; else echo "$(host_of "$1"):$2"; fi; }
q()         { printf '%q' "$1"; }
# sha256 of the three identity files, "name sha" per line, in a fixed order
hash_three() { rsh "$1" "cd $(q "$2") && for f in object.pt train_budget_meta.json COMPLETE; do sha256sum \"\$f\" | awk '{print \$2\" \"\$1}'; done"; }
count_files() { rsh "$1" "find $(q "$2") -type f | wc -l | tr -d ' '"; }
tree_hashes() { rsh "$1" "cd $(q "$2") && find . -type f | LC_ALL=C sort | xargs -r sha256sum"; }

# --- the ledger's view of the key ---------------------------------------------------------------------
VIEW="$(python3 "$LOCATIONS" "${RS_ARG[@]}" view --key "$KEY" --json)" || { log "FATAL artifact_locations view failed"; exit 3; }
# cell|cluster|bank_dir|object_sha256|fit_fingerprint|copied_from|kind   (kind empty when the record predates the flag)
ROWS="$(printf '%s' "$VIEW" | python3 -c '
import json, sys
d = json.load(sys.stdin)
for r in d["records"]:
    print("|".join(str(r.get(k) if r.get(k) is not None else "") for k in ("cell", "cluster", "bank_dir", "object_sha256", "fit_fingerprint", "copied_from", "kind")))')"
rows_for() { printf '%s\n' "$ROWS" | awk -F'|' -v c="$1" '$1 == c'; }

need "$TARGET"
TSHARED="$(shared_of "$TARGET")"
mkdir -p "$STAGING/$KEY"
fail=0; copied=0; present=0
IFS=, read -r -a CELL_ARR <<< "$CELLS"
for cell in "${CELL_ARR[@]}"; do
  dst="$TSHARED/exp/objects/sdfbase/$BANK/$KEY/cells_all/$cell"
  on_target="$(rows_for "$cell" | awk -F'|' -v t="$TARGET" '$2 == t')"
  if [ -n "$on_target" ]; then
    rec_sha="$(printf '%s' "$on_target" | cut -d'|' -f4)"; rec_dir="$(printf '%s' "$on_target" | cut -d'|' -f3)"
    if [ "$DRY" = 1 ]; then log "$cell: recorded on $TARGET at $rec_dir (would verify)"; present=$((present+1)); continue; fi
    got="$(hash_three "$TARGET" "$rec_dir" | awk '$1=="object.pt"{print $2}')"
    if [ "$got" = "$rec_sha" ]; then log "$cell: already on $TARGET at $rec_dir (object.pt sha verified ${rec_sha:0:12})"; present=$((present+1))
    else log "REFUSE $cell: recorded on $TARGET at $rec_dir with sha ${rec_sha:0:12} but object.pt hashes to ${got:0:12} — the record and the disk disagree; not touching it"; fail=1; fi
    continue
  fi
  # pick the source: the fit's own publication (copied_from empty) first, else any other record
  src_row="$(rows_for "$cell" | awk -F'|' '$6 == ""' | head -1)"
  [ -n "$src_row" ] || src_row="$(rows_for "$cell" | head -1)"
  [ -n "$src_row" ] || { log "REFUSE $cell: no artifact-location record at all — the fit's DONE handler never recorded it (or it is not done)"; fail=1; continue; }
  IFS='|' read -r _c src_cluster src_dir src_sha src_fp _cf src_kind <<< "$src_row"
  need "$src_cluster"
  # the mirror's record carries the source record's kind; a source written before the tool had
  # --kind carries none, and every cell this script copies is a parent FIT (cells_all/), so `fit`
  if [ -z "$src_kind" ]; then
    src_kind=fit
    log "$cell: source record on $src_cluster carries no kind (written before --kind existed); the mirror records kind=fit (a cells_all/ parent)"
  fi
  [ "$src_cluster" != "$TARGET" ] || { log "REFUSE $cell: the only record is the target's own but was not matched (internal)"; fail=1; continue; }
  if [ "$DRY" = 1 ]; then log "$cell: would copy $src_cluster:$src_dir -> $TARGET:$dst (sha ${src_sha:0:12})"; continue; fi
  log "$cell: copying $src_cluster:$src_dir -> $TARGET:$dst via $STAGING"
  # the source must hash to its OWN record before a byte moves (ledger vs disk on the source side)
  src_got="$(hash_three "$src_cluster" "$src_dir" | awk '$1=="object.pt"{print $2}')"
  [ "$src_got" = "$src_sha" ] || { log "REFUSE $cell: source $src_cluster:$src_dir object.pt hashes to ${src_got:0:12}, its record says ${src_sha:0:12} — not copying a cell whose ledger and disk disagree"; fail=1; continue; }
  stage="$STAGING/$KEY/$cell"; mkdir -p "$stage"
  "$RSYNC_BIN" "${RSYNC_FLAGS[@]}" "$(rspec "$src_cluster" "$src_dir")/" "$stage/" || { log "REFUSE $cell: rsync pull from $src_cluster failed"; fail=1; continue; }
  rsh "$TARGET" "mkdir -p $(q "$dst")" || { log "REFUSE $cell: cannot mkdir $dst on $TARGET"; fail=1; continue; }
  "$RSYNC_BIN" "${RSYNC_FLAGS[@]}" "$stage/" "$(rspec "$TARGET" "$dst")/" || { log "REFUSE $cell: rsync push to $TARGET failed"; fail=1; continue; }
  # verify: the three identity files hash identically on both clusters, object.pt to the record,
  # and the file COUNT matches (rsync on MooseFS can exit 0 on an incomplete tree)
  src_h="$(hash_three "$src_cluster" "$src_dir")"; dst_h="$(hash_three "$TARGET" "$dst")"
  src_n="$(count_files "$src_cluster" "$src_dir")"; dst_n="$(count_files "$TARGET" "$dst")"
  dst_obj="$(printf '%s\n' "$dst_h" | awk '$1=="object.pt"{print $2}')"
  if [ "$src_h" != "$dst_h" ] || [ "$dst_obj" != "$src_sha" ] || [ "$src_n" != "$dst_n" ]; then
    log "REFUSE $cell: copy does not verify — object.pt dst ${dst_obj:0:12} vs record ${src_sha:0:12}; files src=$src_n dst=$dst_n; identity hashes $( [ "$src_h" = "$dst_h" ] && echo equal || echo DIFFER ). The copy is left in place for inspection; NO record written"
    fail=1; continue
  fi
  python3 "$LOCATIONS" "${RS_ARG[@]}" write --key "$KEY" --cell "$cell" --cluster "$TARGET" --bank-dir "$dst" \
      --object-sha256 "$src_sha" --fit-fingerprint "$src_fp" --copied-from "$src_cluster" --kind "$src_kind" \
    || { log "REFUSE $cell: the copy verified but its record could not be written (above)"; fail=1; continue; }
  log "$cell: verified on $TARGET ($dst_n files, object.pt ${src_sha:0:12}); record written (kind=$src_kind)"
  copied=$((copied+1))
done

# --- the seed's pool dir --------------------------------------------------------------------------------
pool_dst="$TSHARED/vol/pool/$KEY"
if rsh "$TARGET" "test -d $(q "$pool_dst")"; then
  log "pool: $pool_dst present on $TARGET"
else
  # the pool comes from whichever cluster holds an ORIGIN record (the fit ran there, so its pool is there)
  pool_src="$(printf '%s\n' "$ROWS" | awk -F'|' -v t="$TARGET" '$6 == "" && $2 != t {print $2}' | head -1)"
  [ -z "$pool_src" ] || need "$pool_src"
  if [ -z "$pool_src" ]; then log "REFUSE pool: $pool_dst absent on $TARGET and no origin record names another cluster to copy it from"; fail=1
  elif [ "$DRY" = 1 ]; then log "pool: would copy $pool_src:$(shared_of "$pool_src")/vol/pool/$KEY -> $TARGET:$pool_dst"
  else
    psrc="$(shared_of "$pool_src")/vol/pool/$KEY"; pstage="$STAGING/$KEY/_pool"; mkdir -p "$pstage"
    log "pool: copying $pool_src:$psrc -> $TARGET:$pool_dst"
    if "$RSYNC_BIN" "${RSYNC_FLAGS[@]}" "$(rspec "$pool_src" "$psrc")/" "$pstage/" \
       && rsh "$TARGET" "mkdir -p $(q "$pool_dst")" \
       && "$RSYNC_BIN" "${RSYNC_FLAGS[@]}" "$pstage/" "$(rspec "$TARGET" "$pool_dst")/"; then
      if [ "$(tree_hashes "$pool_src" "$psrc")" = "$(tree_hashes "$TARGET" "$pool_dst")" ]; then
        log "pool: verified ($(count_files "$TARGET" "$pool_dst") files, every sha equal)"
      else log "REFUSE pool: tree hashes differ between $pool_src and $TARGET after the copy"; fail=1; fi
    else log "REFUSE pool: rsync failed"; fail=1; fi
  fi
fi

log "summary: ${#CELL_ARR[@]} cell(s): $present already on $TARGET, $copied copied now, fail=$fail${DRY:+ (dry-run=$DRY)}"
[ "$fail" = 0 ] || exit 4
[ "$DRY" = 1 ] || [ $((present + copied)) -eq ${#CELL_ARR[@]} ] || exit 4
exit 0
