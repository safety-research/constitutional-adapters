#!/bin/bash
# tree_export.sh SHA MIRROR TREE — run ON THE CLUSTER LOGIN NODE (git only, nothing heavy): export commit SHA from the
# repo mirror MIRROR as a standalone FULL checkout at TREE (exact HEAD, clean, detached) — used by gate_slurm.sh for the
# test suite's tree and by the cluster read window for the reader's tree (the same exported tree serves both). Idempotent:
# an existing tree at TREE with HEAD == SHA is reused; a tree at that path with any other HEAD refuses (a path is one
# commit, forever). The export is staged at TREE.tmp and published by one rename, so a killed export leaves no half tree.
#
# FULL means the mirror's whole object store: every branch head (as refs/remotes/mirror/*), every tag, and every ARCHIVED
# branch (refs/archive/<date>/<branch>, the namespace kept as the grading host spells it -- never a live head under refs/remotes),
# with their commits, trees and blobs. The archive matters: on 2026-10-09 an archiving pass moved 80 branches on the grading host
# into refs/archive/<date>/ and the mirror's heads were pruned, and a tree exported from heads and tags alone lost every commit
# those branches alone reached -- three history commits the seedrep ws1c / ws4h tests read out of the object store
# (tests/seedrep_history_fixtures.py: `git cat-file -e <sha>^{commit}` on a non-shallow checkout, by object presence, never by a
# ref name), four nodes INTRODUCED on every tree exported after that moment. The suite reads the repository's history, not
# just HEAD's files: the reclose funnel's closer-tree guard
# proves `git merge-base --is-ancestor <closer-fix-3> HEAD` on a fixture that shares this object store
# (tests/test_closer_refusal_outcome._git_tree); the frozen-fixture and carry-over tests `git show <old sha>:<path>`,
# `git diff <a> <b>`, `git archive <sha>` and `git worktree add` at pinned commits — several from OTHER branches
# (seedrep-ws*), and they refuse by name, never skip, when a FULL (non-shallow) checkout lacks the commit. Two shapes
# were measured wrong on the 2026-10-08 stage-e-run gate: a depth-1 export (the Modal gate's shallow_clone_tarball twin,
# the shape until that day) made every reclose fixture refuse as "commit <HEAD>, which does not descend from closer-fix-3"
# (21 INTRODUCED nodes), and a commits-only `--filter=tree:0` deepen made the history tests RUN against absent trees and
# blobs (51 INTRODUCED). Cost of the full shape: .git ~1.5 GB against ~0.9 GB for depth-1 (HEAD's own blobs are the bulk;
# history and the other branches add ~0.6 GB); the fetch runs once per sha on the login node from the local mirror.
# Nothing here prunes older trees under $(dirname TREE): one directory per exported sha accumulates until someone does.
#
# The archive reaches the mirror only if something pushes it: mirror-repos.sh transfers heads and tags alone, so the grading host side runs
# mirror_history.sh (beside this script) before every export -- every refs/archive/* ref and every tag of the checkout, verified against
# the mirror's own listing. A tree exported BEFORE that (the trees of 2026-10-09 lacked the archive; a tree exported before a pin tag
# lacks the pin) is not reused as it stands: an existing tree with HEAD == SHA is held against the mirror's archive-and-tag inventory
# (ref name AND object), and what it lacks is fetched into it IN PLACE -- objects and refs under refs/archive and refs/tags only, with
# gc.auto off so nothing repacks under a job that may be reading the checkout; HEAD, the index and the working files are untouched, so
# the path is still one commit forever and the `TREE existing` line then says what was completed. The heads are the export-time snapshot
# (refs/remotes/mirror/*) and are not refreshed: the tree is one commit's tests against one history, not a tracking clone.
#
# Invocation from the grading host:   ssh $HOST bash -s -- "$SHA" "$MIRROR" "$TREE" < tree_export.sh
# Output: `TREE existing <path>`, `TREE existing <path> (completed: <k> archived branches / tags fetched)` or
# `TREE exported <path> (<n> files, <c> commits, <refs> refs)` on success; any `ERR ...` line + exit 2 refuses.
set -uo pipefail
SHA=${1:?SHA (40 hex)} MIRROR=${2:?MIRROR (the repo mirror on shared storage)} TREE=${3:?TREE (destination directory)}
[[ "$SHA" =~ ^[0-9a-f]{40}$ ]] || { echo "ERR SHA must be 40 hex, got '$SHA'"; exit 2; }
git -C "$MIRROR" rev-parse --git-dir >/dev/null 2>&1 || { echo "ERR no repo mirror at $MIRROR (mirror-repos.sh first)"; exit 2; }
# the mirror is a local path, so the server side is a `git upload-pack` this very process spawns: its knobs ride --upload-pack
# (a client-side `-c uploadpack.*` is read by the client only — measured 2026-10-08); the bare sha is wanted beside every head and tag
UP='git -c uploadpack.allowAnySHA1InWant=true upload-pack'
# the history inventory a tree must carry: every archived branch and tag of the mirror, at the mirror's object
inventory() { git -C "$1" for-each-ref --format='%(objectname) %(refname)' refs/archive refs/tags | LC_ALL=C sort; }
if [ -d "$TREE/.git" ]; then
  head=$(git -C "$TREE" rev-parse HEAD 2>/dev/null); [ "$head" = "$SHA" ] || { echo "ERR existing tree at $TREE has HEAD '$head'"; exit 2; }
  missing=$(LC_ALL=C comm -23 <(inventory "$MIRROR") <(inventory "$TREE"))
  if [ -n "$missing" ]; then
    k=$(wc -l <<<"$missing")
    # additive only; a fetch that fails is judged by the inventory afterwards (a concurrent export of the same sha may have completed it)
    git -C "$TREE" -c gc.auto=0 fetch -q --upload-pack "$UP" "$MIRROR" '+refs/archive/*:refs/archive/*' '+refs/tags/*:refs/tags/*' 2>/dev/null
    still=$(LC_ALL=C comm -23 <(inventory "$MIRROR") <(inventory "$TREE"))
    [ -z "$still" ] || { echo "ERR existing tree at $TREE lacks $(wc -l <<<"$still") of the mirror's archived branches / tags after a fetch: $(head -n 3 <<<"$still" | awk '{print $2"@"substr($1,1,8)}' | tr '\n' ' ')"; exit 2; }
    # the objects came, not just the names (the fixture's question is object presence)
    while read -r obj ref; do
      git -C "$TREE" cat-file -e "$obj" 2>/dev/null || { echo "ERR existing tree at $TREE: $ref names $obj, which is not in its object store"; exit 2; }
    done <<<"$missing"
    [ "$(git -C "$TREE" rev-parse HEAD 2>/dev/null)" = "$SHA" ] || { echo "ERR existing tree at $TREE: HEAD moved during the completion"; exit 2; }
    echo "TREE existing $TREE (completed: $k archived branches / tags fetched)"; exit 0
  fi
  echo "TREE existing $TREE"; exit 0
fi
git -C "$MIRROR" cat-file -e "$SHA^{commit}" 2>/dev/null || { echo "ERR $SHA is not in the mirror $MIRROR (mirror-repos.sh first)"; exit 2; }
mkdir -p "$(dirname "$TREE")" || { echo "ERR cannot create $(dirname "$TREE")"; exit 2; }
rm -rf "$TREE.tmp"; mkdir -p "$TREE.tmp" && git init -q "$TREE.tmp" || { echo "ERR git init"; exit 2; }
git -C "$TREE.tmp" fetch -q --upload-pack "$UP" "$MIRROR" '+refs/heads/*:refs/remotes/mirror/*' '+refs/tags/*:refs/tags/*' \
  '+refs/archive/*:refs/archive/*' "$SHA" \
  || { echo "ERR full fetch (every head, tag and archived branch + $SHA) from $MIRROR"; exit 2; }
git -C "$TREE.tmp" checkout -q --detach "$SHA" || { echo "ERR checkout"; exit 2; }
[ "$(git -C "$TREE.tmp" rev-parse HEAD)" = "$SHA" ] || { echo "ERR exported HEAD != $SHA"; exit 2; }
[ "$(git -C "$TREE.tmp" rev-parse --is-shallow-repository)" = false ] || { echo "ERR exported tree is shallow (the history did not come)"; exit 2; }
# the history is READABLE, not just enumerable: the root commit's tree (the oldest object HEAD descends from) resolves
root=$(git -C "$TREE.tmp" rev-list --max-parents=0 HEAD | tail -n 1)
[ -n "$root" ] && git -C "$TREE.tmp" cat-file -e "$root^{tree}" 2>/dev/null || { echo "ERR the root commit's tree does not resolve (history without its objects: a filtered or partial fetch)"; exit 2; }
git -C "$TREE.tmp" diff --quiet HEAD -- || { echo "ERR exported tree not clean"; exit 2; }
mv "$TREE.tmp" "$TREE" || { echo "ERR mv $TREE.tmp -> $TREE"; exit 2; }
echo "TREE exported $TREE ($(find "$TREE" -type f | wc -l) files, $(git -C "$TREE" rev-list --count HEAD) commits, $(git -C "$TREE" for-each-ref | wc -l) refs)"
