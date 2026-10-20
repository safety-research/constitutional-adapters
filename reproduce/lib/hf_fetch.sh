#!/usr/bin/env bash
# reproduce/lib/hf_fetch.sh — pre-download ONE HuggingFace repo into a cluster's HF cache without a token on the cluster.
# The token (read-only, from lib/secrets.sh: a secret-manager item / HF_TOKEN / none) travels over ssh's STDIN — never
# argv, never a file — into ONE srun'd `hf download` as an env var for the life of that job, and is gone when it exits.
# Jobs then load offline (HF_HUB_OFFLINE=1). ONE repo per call: `hf download` treats a 2nd positional as a filename
# inside the first repo and fetches 0 files with exit 0.
#
#   hf_fetch.sh --cluster <label> --repo <org/name> [--revision <sha>] [--repo-type model|dataset] [--include GLOB]...
#               [--time H:MM:SS] [--dry-run] [--config F]
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER=""; REPOID=""; REV=""; RTYPE=model; WALL=4:00:00; DRY=0; CONFIG=""; INCLUDES=()
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER="$2"; shift 2;; --repo) REPOID="$2"; shift 2;; --revision) REV="$2"; shift 2;;
    --repo-type) RTYPE="$2"; shift 2;; --include) INCLUDES+=("$2"); shift 2;; --time) WALL="$2"; shift 2;;
    --dry-run) DRY=1; shift;; --config) CONFIG="$2"; shift 2;;
    -h|--help) sed -n '2,10p' "$0"; exit 0;;
    *) echo "hf_fetch.sh: unknown argument $1 (one --repo per call)" >&2; exit 2;;
  esac
done
[ -n "$CLUSTER" ] && [ -n "$REPOID" ] || { echo "REFUSED: --cluster <label> and --repo <org/name> are required" >&2; exit 2; }
case "$REPOID" in */*) ;; *) echo "REFUSED: --repo must be <org>/<name>, got '$REPOID'" >&2; exit 2;; esac
case "$REPOID" in *" "*) echo "REFUSED: one repo per call (got '$REPOID')" >&2; exit 2;; esac
source "$HERE/common.sh"; source "$HERE/secrets.sh"
REPRO_CONFIG="$CONFIG"; REPRO_DRY_RUN=$DRY; repro_load_config; cluster_require "$CLUSTER"
HOST="$(cluster_var "$CLUSTER" CLUSTER_SSH)"; SHARED="$(cluster_var "$CLUSTER" CLUSTER_SHARED_ROOT)"; HFH="$(cluster_var "$CLUSTER" CLUSTER_HF_HOME)"
TSRC="$(cluster_var "$CLUSTER" CLUSTER_HF_TOKEN_SOURCE)"; PART="$(cluster_var "$CLUSTER" CLUSTER_DEV_PARTITION)"; QOS="$(cluster_var "$CLUSTER" CLUSTER_DEV_QOS)"
[ -n "$PART" ] || PART="$(cluster_var "$CLUSTER" CLUSTER_PARTITION)"; [ -n "$QOS" ] || QOS="$(cluster_var "$CLUSTER" CLUSTER_QOS)"
ACCT="$(cluster_var "$CLUSTER" CLUSTER_ACCOUNT)"; ENVS="$(cluster_var "$CLUSTER" CLUSTER_ENVS_DIR)"
HF_BIN="${SISTEER_CLUSTER_HF_BIN:-$ENVS/hf-cli/bin/hf}"
inc=""; for g in ${INCLUDES[@]+"${INCLUDES[@]}"}; do inc="$inc --include $(printf '%q' "$g")"; done
REMOTE="read -r T; export HF_HOME=$(printf '%q' "$HFH") HF_HUB_ENABLE_HF_TRANSFER=0; [ -n \"\$T\" ] && export HF_TOKEN=\"\$T\"; unset T; mkdir -p \"\$HF_HOME\"; \
srun -p $PART --qos=$QOS ${ACCT:+--account=$ACCT} --cpus-per-task=8 --mem=16G --time=$WALL --job-name=D_\${USER}_hffetch \
  $HF_BIN download --repo-type $RTYPE $(printf '%q' "$REPOID")${REV:+ --revision $REV}$inc && echo \"[$CLUSTER] $REPOID cached under \$HF_HOME\""
if [ "$DRY" = 1 ]; then
  echo "DRY-RUN: <token from source '$TSRC' on stdin> | ssh $HOST '$REMOTE'"
  exit 0
fi
# the token: an empty line for source `none` (public repos); the value never appears on argv or in this log
{ secret_hf_cluster_read "$TSRC" || exit 1; echo; } | ssh -o BatchMode=yes -o ConnectTimeout=30 "$HOST" "$REMOTE"
