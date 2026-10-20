#!/usr/bin/env bash
# reproduce/lib/third_party.sh — install the run-time-fetched third-party files the jobs read onto ONE cluster's shared root.
# Neither is distributed with this repository. Both are fetched HERE, on the setup host, from their pinned upstream commits and
# verified before anything is shipped, then each is shipped by content to <CLUSTER_SHARED_ROOT>/third_party/<relpath> and verified
# there again (lib/common.sh repro_ssh_put: the final name only ever holds verified bytes):
#   - the G0DM0D3 templates (sisteer.eval.attacks_godmode; AGPL-3.0): the file's sha256, then every template's;
#   - the refusal-direction fit prompts (sisteer.data.arditi_fit_prompts): andyrdt/refusal_direction's four split files, each by
#     sha256, re-drawn into the paper fit's document, every set and the document held to data/arditi/fit_prompts.manifest.json;
#     the document alone is shipped (stage 04's refusal direction reads it).
# Compute nodes have no egress: jobs read those copies offline, because lib/hf_env.sh exports SISTEER_THIRD_PARTY_DIR=$SHARED/third_party
# and SISTEER_THIRD_PARTY_OFFLINE=1. Re-running is a no-op once the bytes are there.
#
#   third_party.sh --cluster <label> [--cache-dir D] [--dry-run] [--config F]
#
#   --cache-dir D   the setup host's third-party root (default $SISTEER_THIRD_PARTY_DIR, else $XDG_CACHE_HOME/sisteer/third_party)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER=""; CACHE=""; DRY=0; CONFIG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER="$2"; shift 2;; --cache-dir) CACHE="$2"; shift 2;;
    --dry-run) DRY=1; shift;; --config) CONFIG="$2"; shift 2;;
    -h|--help) sed -n '2,15p' "$0"; exit 0;;
    *) echo "third_party.sh: unknown argument $1" >&2; exit 2;;
  esac
done
[ -n "$CLUSTER" ] || { echo "REFUSED: --cluster <label> is required" >&2; exit 2; }
source "$HERE/common.sh"
[ "$REPRO_DRY_RUN" = 1 ] && DRY=1          # fail-closed: a caller that exported REPRO_DRY_RUN=1 stays dry (lib/common.sh)
REPRO_CONFIG="$CONFIG"; REPRO_DRY_RUN=$DRY; repro_load_config; cluster_require "$CLUSTER"
HOST="$(cluster_var "$CLUSTER" CLUSTER_SSH)"; SHARED="$(cluster_var "$CLUSTER" CLUSTER_SHARED_ROOT)"
[ -n "$SHARED" ] || repro_fatal "cluster $CLUSTER has no CLUSTER_SHARED_ROOT -- nowhere to install the third-party files"
PY="$(repro_python)"
agm(){ PYTHONPATH="$REPRO_REPO/src${PYTHONPATH:+:$PYTHONPATH}" "$PY" "$@"; }

# what (as the messages name it) <TAB> module (its `fetch --json` makes the file present and verified here and reports status, path,
# relpath, sha256) <TAB> the pins alone (relpath and sha256, for a dry run: an import reads nothing and touches no network)
ITEMS=(
  $'G0DM0D3\tsisteer.eval.attacks_godmode\tfrom sisteer.eval import attacks_godmode as m; print(m.CACHE_RELPATH, m.FILE_SHA256, sep="\\t")'
  $'refusal-direction fit prompts\tsisteer.data.arditi_fit_prompts\tfrom sisteer.data import arditi_fit_prompts as m; f = m.load_manifest(); print(m.document_relpath(f), f["document"]["sha256"], sep="\\t")'
)
fetch_cmd(){ FETCH=(-m "$1" fetch --json); [ -n "$CACHE" ] && FETCH+=(--cache-dir "$CACHE"); }

if [ "$DRY" = 1 ]; then
  # the pins only: the exact destinations, nothing fetched, nothing shipped
  for item in "${ITEMS[@]}"; do
    IFS=$'\t' read -r what mod pins <<<"$item"
    IFS=$'\t' read -r rel sha < <(agm -c "$pins") && [ -n "$sha" ] || repro_fatal "could not read the $what pins with $PY"
    fetch_cmd "$mod"; printf 'DRY-RUN:'; printf ' %q' "$PY" "${FETCH[@]}"; echo
    repro_ssh_put "$CLUSTER" "<the verified local copy of $rel>" "$SHARED/third_party/$rel" "$sha"
  done
  exit 0
fi

# 1. the setup host: fetch (or reuse) and verify EVERY file before any is shipped; a refusal prints its own reason on stderr
SHIP=()
for item in "${ITEMS[@]}"; do
  IFS=$'\t' read -r what mod pins <<<"$item"
  fetch_cmd "$mod"
  REPORT="$(agm "${FETCH[@]}")" || repro_fatal "the $what fetch on this host refused (its reason is above) -- nothing was shipped to $CLUSTER"
  IFS=$'\t' read -r status src rel sha < <(printf '%s' "$REPORT" | "$PY" -c 'import json,sys; r=json.load(sys.stdin); print(r["status"], r["path"], r["relpath"], r["sha256"], sep="\t")') \
    && [ -n "$sha" ] || repro_fatal "could not read the $what fetch report: $REPORT"
  SHIP+=("$what"$'\t'"$status"$'\t'"$src"$'\t'"$rel"$'\t'"$sha")
done

# 2. the cluster: ship each by content, verified at the destination
for row in "${SHIP[@]}"; do
  IFS=$'\t' read -r what status src rel sha <<<"$row"
  DEST="$SHARED/third_party/$rel"
  repro_say "third_party -> $CLUSTER: $what $rel (sha256 ${sha:0:16}..., $status on this host) -> $HOST:$DEST"
  repro_ssh_put "$CLUSTER" "$src" "$DEST" "$sha"; rc=$?
  case "$rc" in
    0) repro_say "  verified on $CLUSTER: $DEST";;
    255) repro_fatal "cluster $CLUSTER's login host ($HOST) could not be reached -- the $what file is NOT installed there";;
    *) repro_fatal "the $what file never verified at $HOST:$DEST (rc $rc) -- it is NOT installed there";;
  esac
done
