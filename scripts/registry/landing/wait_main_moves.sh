#!/bin/bash
# wait_main_moves.sh <current-main-sha> [<sha-that-must-be-contained>] [poll-s]
# Polls origin/main (git ls-remote, no fetch into any checkout; the base checkout is site_env.sh's SITE_REPO: SISTEER_REPO, else the site
# config's, else this repository's main working tree) every poll-s (default 30) and exits 0 with
# "MOVED <new-sha>" once it differs from <current-main-sha>; with a second argument it also requires the new
# main to contain that sha (a landed branch tip) and keeps polling until it does. Bound the call with `timeout`.
set -u
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OLD=$1; MUST=${2:-}; POLL=${3:-30}
REPO=$(source "$HERE/site_env.sh" && site_env_load SITE_REPO && printf '%s\n' "$SITE_REPO") || exit 2
# an absent checkout refuses loudly, never a silent poll forever
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || {
  echo "wait_main_moves.sh: $REPO is not a git checkout; set SISTEER_REPO to the base checkout" >&2; exit 2; }
while :; do
  NEW=$(git -C "$REPO" ls-remote -q origin refs/heads/main | cut -f1)   # its errors reach stderr
  if [ -n "$NEW" ] && [ "$NEW" != "$OLD" ]; then
    git -C "$REPO" fetch -q origin 2>/dev/null
    if [ -z "$MUST" ] || git -C "$REPO" merge-base --is-ancestor "$MUST" "$NEW" 2>/dev/null; then
      echo "MOVED $NEW"; exit 0
    fi
    echo "moved to $NEW but does not yet contain $MUST" >&2
  fi
  sleep "$POLL"
done
