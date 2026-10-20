#!/usr/bin/env bash
# reproduce/lib/mirror_repo.sh — put the committed tree at ONE sha on a cluster: the repo mirror (a working clone under
# CLUSTER_REPO_MIRROR, updated by `git push` over ssh — no GitHub credential ever lands on cluster storage) and the clean,
# detached checkout the wave shell requires at <CLUSTER_EVAL_REPO_ROOT>/<sha> (common.sh f22_init refuses when HEAD !=
# the positional wave sha or the tree is dirty).
#
#   mirror_repo.sh --cluster <label> --sha <40hex> [--repo DIR] [--dry-run] [--config F]
#   mirror_repo.sh --ssh HOST --mirror DIR --eval-root DIR --sha <40hex> [--repo DIR]   # the raw form (tests)
# Idempotent: an existing checkout at the sha with a clean tree is verified and left alone.
# ONE run per mirror clone at a time: two lanes' runs against the same cluster clone raced — a `git worktree
# add` in progress read as "dirty checkout" to the other run. The whole remote section (setup, push, checkout) holds a flock on
# THIS host (every front-door lane runs here), keyed by the target (ssh host + mirror path), under
# $SISTEER_MIRROR_LOCK_DIR (default $SISTEER_STATE_DIR/locks, else ~/.local/state/sisteer-reproduce/locks); a held lock is waited
# on for SISTEER_MIRROR_LOCK_WAIT_S (default 900), then refused by name. SISTEER_MIRROR_HOLD_S (tests) holds the lock that long.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER=""; SHA=""; REPO=""; SSH_HOST=""; MIRROR=""; EVAL_ROOT=""; DRY=0; CONFIG=""
while [ $# -gt 0 ]; do
  case "$1" in
    --cluster) CLUSTER="$2"; shift 2;; --sha) SHA="$2"; shift 2;; --repo) REPO="$2"; shift 2;;
    --ssh) SSH_HOST="$2"; shift 2;; --mirror) MIRROR="$2"; shift 2;; --eval-root) EVAL_ROOT="$2"; shift 2;;
    --dry-run) DRY=1; shift;; --config) CONFIG="$2"; shift 2;;
    -h|--help) sed -n '2,11p' "$0"; exit 0;;
    *) echo "mirror_repo.sh: unknown argument $1" >&2; exit 2;;
  esac
done
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
# A TEST WORLD NEVER MIRRORS THE REAL REPO (2026-09-30): a front-door test whose fake cluster lives under a pytest basetemp ran this
# for real once per test — a full checkout of the repo (~6 GB each: corpora/*.jsonl.gz) — and filled the grading host root in 30 minutes,
# because the world's MIRRORED_SHA (HEAD read at import) fell behind a HEAD that peers moved during the run. A test world exports
# SISTEER_TEST_NO_MIRROR=1 and this refuses BY NAME (loud, nothing written) instead of copying the repo; a dry run is still described.
if [ "${SISTEER_TEST_NO_MIRROR:-0}" = 1 ] && [ "$DRY" != 1 ]; then
  echo "REFUSED: mirror_repo.sh under SISTEER_TEST_NO_MIRROR=1 — a test world never mirrors the real repo (sha ${SHA:0:12}, target ${CLUSTER:-$SSH_HOST}); the world's MIRRORED_SHA must equal the run's HEAD, or the test stubs the mirror" >&2
  exit 3
fi
case "$SHA" in [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]*) [ ${#SHA} -eq 40 ] || { echo "REFUSED: --sha must be 40 hex" >&2; exit 2; };; *) echo "REFUSED: --sha <40hex> is required" >&2; exit 2;; esac
if [ -n "$CLUSTER" ]; then
  source "$HERE/common.sh"; REPRO_CONFIG="$CONFIG"; REPRO_DRY_RUN=$DRY; repro_load_config; cluster_require "$CLUSTER"
  SSH_HOST="$(cluster_var "$CLUSTER" CLUSTER_SSH)"; MIRROR="$(cluster_var "$CLUSTER" CLUSTER_REPO_MIRROR)"; EVAL_ROOT="$(cluster_var "$CLUSTER" CLUSTER_EVAL_REPO_ROOT)"
fi
[ -n "$SSH_HOST" ] && [ -n "$MIRROR" ] && [ -n "$EVAL_ROOT" ] || { echo "REFUSED: need --cluster <label> (config) or --ssh/--mirror/--eval-root" >&2; exit 2; }
git -C "$REPO" cat-file -e "$SHA^{commit}" 2>/dev/null || { echo "REFUSED: $SHA is not a commit of $REPO" >&2; exit 2; }
# the remote side: ssh (BatchMode) and an ssh:// push URL — or, the test seam every front-door ssh face shares (lib/common.sh
# repro_ssh, submit_twin.py Remote), `local:<dir>` runs each command with `bash -c` on this host and pushes to the mirror path
if [[ "$SSH_HOST" == local:* ]]; then rsh(){ bash -c "$1"; }; URL="$MIRROR"; else rsh(){ ssh -o BatchMode=yes -o ConnectTimeout=30 "$SSH_HOST" "$1"; }; URL="ssh://$SSH_HOST$MIRROR"; fi
# 1. the mirror clone exists and refuses dirty overwrites; 2. push the sha (and every branch/tag we hold — forced ref
# update is safe: the ls-remote guard below proves every remote tip is already in our object DB); 3. the detached
# checkout at <eval-root>/<sha>, created as a worktree of the mirror (one object store per cluster), verified clean.
REMOTE_SETUP="set -e; mkdir -p '$(dirname "$MIRROR")' '$EVAL_ROOT'; [ -e '$MIRROR/.git' ] || git init -q -b main '$MIRROR'; git -C '$MIRROR' config receive.denyCurrentBranch updateInstead"
REMOTE_CHECKOUT="set -e; if [ -d '$EVAL_ROOT/$SHA/.git' ] || [ -f '$EVAL_ROOT/$SHA/.git' ]; then cd '$EVAL_ROOT/$SHA'; [ \"\$(git rev-parse HEAD)\" = '$SHA' ] || { echo 'checkout at another sha' >&2; exit 3; }; git diff --quiet HEAD -- || { echo 'dirty checkout' >&2; exit 3; }; echo verified; else git -C '$MIRROR' worktree add --detach '$EVAL_ROOT/$SHA' '$SHA' >/dev/null && echo created; fi"
if [ "$DRY" = 1 ]; then
  echo "DRY-RUN: ssh $SSH_HOST \"$REMOTE_SETUP\""
  echo "DRY-RUN: git -C $REPO push -q $URL '+refs/heads/*:refs/heads/*' '+refs/tags/*:refs/tags/*' $SHA:refs/heads/sisteer-eval-$SHA"
  echo "DRY-RUN: ssh $SSH_HOST \"$REMOTE_CHECKOUT\""
  exit 0
fi
LOCK_DIR="${SISTEER_MIRROR_LOCK_DIR:-${SISTEER_STATE_DIR:-$HOME/.local/state/sisteer-reproduce}/locks}"
mkdir -p "$LOCK_DIR" || { echo "REFUSED: cannot create the lock dir $LOCK_DIR" >&2; exit 1; }
LOCK="$LOCK_DIR/mirror.$(printf '%s' "$SSH_HOST$MIRROR" | sha256sum | cut -c1-16).lock"
LOCK_WAIT="${SISTEER_MIRROR_LOCK_WAIT_S:-900}"
exec 9>"$LOCK"
flock -w "$LOCK_WAIT" 9 || { echo "REFUSED: another mirror_repo.sh run holds the mirror $SSH_HOST:$MIRROR (lock $LOCK, waited ${LOCK_WAIT}s) — a mirror clone takes one run at a time; retry when it finishes" >&2; exit 1; }
echo "mirror_repo: lock acquired $(date -u +%s.%N) $SSH_HOST:$MIRROR ($LOCK)" >&2
trap 'echo "mirror_repo: lock released $(date -u +%s.%N) $SSH_HOST:$MIRROR" >&2' EXIT
[ -z "${SISTEER_MIRROR_HOLD_S:-}" ] || sleep "$SISTEER_MIRROR_HOLD_S"      # test seam: hold the lock (two concurrent runs serialize)
rsh "$REMOTE_SETUP" || { echo "mirror_repo: setup over ssh failed ($SSH_HOST)" >&2; exit 1; }
foreign=""
while read -r rsha ref; do [ -n "$rsha" ] || continue; git -C "$REPO" cat-file -e "$rsha" 2>/dev/null || foreign="$foreign $ref@${rsha:0:8}"; done < <(git -C "$REPO" ls-remote "$URL" 'refs/heads/*' 2>/dev/null)
[ -z "$foreign" ] || { echo "REFUSED: mirror $URL has commits not present locally:$foreign (a mirror is one-way; reconcile by hand)" >&2; exit 1; }
git -C "$REPO" push -q "$URL" '+refs/heads/*:refs/heads/*' '+refs/tags/*:refs/tags/*' "$SHA:refs/heads/sisteer-eval-$SHA" 2>&1 | sed 's/^/mirror_repo: /' >&2 || { echo "mirror_repo: push failed" >&2; exit 1; }
out="$(rsh "$REMOTE_CHECKOUT" 2>&1)" || { echo "REFUSED: eval checkout at $SSH_HOST:$EVAL_ROOT/$SHA — $out" >&2; exit 1; }
echo "mirror_repo: $SSH_HOST:$EVAL_ROOT/$SHA $out"
