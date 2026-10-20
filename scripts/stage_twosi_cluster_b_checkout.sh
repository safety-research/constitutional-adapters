#!/bin/bash
# Stage the IMMUTABLE commit-addressed checkout of THIS worktree's HEAD
# on a cluster, so the overnight orchestrator's generated fit sbatch scripts
# (whose pin guard requires <project directory>/checkouts/<sha>) can run.
#
# ADDITIVE ONLY, by construction:
#   - never runs mirror-repos.sh and never pushes to the branch mirror
#     (the repository copy's -twosi sibling) — moving that mirror's HEAD trips the
#     EXPECTED_SHA gate of any job running FROM it (jobs 23789/23790 died at
#     start exactly that way on 2026-08-21);
#   - each sha gets its own directory; an existing correct checkout is a
#     no-op, so re-running is free;
#   - missing commits travel by a direct `git push` INTO the new clone (to a
#     staging ref, never a checked-out branch), the same
#     no-credentials-on-the-cluster pattern launch.sh uses for pods.
#
# Usage: scripts/stage_twosi_cluster_b_checkout.sh [host shared_root]
#   host        ssh alias of the cluster login node
#   shared_root the cluster's per-user shared root
#   With neither, both come from cluster B's block in the site config (CLUSTER_SSH, CLUSTER_SHARED_ROOT, read through
#   reproduce/lib/clusters.py); a site config without that block is refused by name.
#   The checkout goes under the cluster's project directory, cloned from its copy of this repository: SISTEER_PROJECT_ROOT
#   and SISTEER_REPO_MIRROR when set, else CLUSTER_PROJECT_ROOT and CLUSTER_REPO_MIRROR from cluster B's site block.
set -euo pipefail

REPO_DIR=$(cd "$(dirname "$0")/.." && pwd)
CLUSTER=cluster_b
if [ $# -ge 2 ]; then
  HOST=$1
  SHARED_REMOTE=$2
elif [ $# -eq 1 ]; then
  echo "FATAL: a login host was given without its shared root; pass both, or neither to read cluster B's block of the site config" >&2
  exit 2
else
  CL_PY="$REPO_DIR/reproduce/lib/clusters.py"
  HOST=$(python3 "$CL_PY" --get "$CLUSTER" CLUSTER_SSH) && [ -n "$HOST" ] \
    || { echo "FATAL: no login host (CLUSTER_SSH) for cluster $CLUSTER in the site config; pass <host> <shared_root>" >&2; exit 2; }
  SHARED_REMOTE=$(python3 "$CL_PY" --get "$CLUSTER" CLUSTER_SHARED_ROOT) && [ -n "$SHARED_REMOTE" ] \
    || { echo "FATAL: no shared root (CLUSTER_SHARED_ROOT) for cluster $CLUSTER in the site config; pass <host> <shared_root>" >&2; exit 2; }
fi

# The fit jobs run exactly HEAD (the orchestrator pins --comment=sha:HEAD and
# the in-script guard verifies it), so a dirty tree means the launched code
# would NOT be the code being looked at. Refuse; commit first.
if ! git -C "$REPO_DIR" diff --quiet --ignore-submodules HEAD; then
  echo "FATAL: this worktree has uncommitted changes -- the fit jobs run" >&2
  echo "exactly HEAD, so commit (or stash) first." >&2
  exit 1
fi

# The cluster's project directory and its clone of this repository: SISTEER_PROJECT_ROOT and SISTEER_REPO_MIRROR when set,
# else cluster B's block of the site config (CLUSTER_PROJECT_ROOT, CLUSTER_REPO_MIRROR); refused by name with neither.
PROJECT=${SISTEER_PROJECT_ROOT:-}
MIRROR_ROOT=${SISTEER_REPO_MIRROR:-}
[ -n "$PROJECT" ] || PROJECT=$(python3 "$REPO_DIR/reproduce/lib/clusters.py" --get "$CLUSTER" CLUSTER_PROJECT_ROOT) || PROJECT=""
[ -n "$MIRROR_ROOT" ] || MIRROR_ROOT=$(python3 "$REPO_DIR/reproduce/lib/clusters.py" --get "$CLUSTER" CLUSTER_REPO_MIRROR) || MIRROR_ROOT=""
if [ -z "$PROJECT" ] || [ -z "$MIRROR_ROOT" ]; then
  echo "FATAL: no project directory or repository clone for cluster $CLUSTER: set SISTEER_PROJECT_ROOT and SISTEER_REPO_MIRROR," \
    "or CLUSTER_PROJECT_ROOT and CLUSTER_REPO_MIRROR in its site config block" >&2
  exit 2
fi

SHA=$(git -C "$REPO_DIR" rev-parse HEAD)
DEST="$PROJECT/checkouts/$SHA"
MIRROR="$MIRROR_ROOT-twosi"

# Already staged and verified -> no-op (idempotent re-runs).
if ssh -o BatchMode=yes "$HOST" \
    "git -C '$DEST' rev-parse HEAD 2>/dev/null" | grep -qx "$SHA"; then
  echo "already staged: $HOST:$DEST"
  exit 0
fi

# Clone from the mirror READ-ONLY (objects hardlink on the same filesystem,
# so this is cheap) without checking anything out yet.
ssh -o BatchMode=yes "$HOST" \
  "install -d -m 700 '$PROJECT/checkouts' && \
   [ -d '$DEST/.git' ] || git clone -q --no-checkout '$MIRROR' '$DEST'"

# Ship whatever commits the mirror does not have yet, INTO THE NEW CLONE
# ONLY, on a staging ref (never a checked-out branch, so the push is always
# accepted and nothing running elsewhere can be disturbed).
git -C "$REPO_DIR" push -q "$HOST:$DEST" "$SHA:refs/heads/twosi-staged"

# Pin the working tree to the exact commit and verify the round trip.
GOT=$(ssh -o BatchMode=yes "$HOST" \
  "git -C '$DEST' checkout -q --detach '$SHA' && git -C '$DEST' rev-parse HEAD")
if [ "$GOT" != "$SHA" ]; then
  echo "FATAL: staged checkout HEAD '$GOT' != expected '$SHA'" >&2
  exit 1
fi
echo "staged: $HOST:$DEST"
