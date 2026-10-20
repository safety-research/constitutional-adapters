#!/bin/bash
# mirror_history.sh REPO URL — run ON THE GRADING HOST, right before every tree export (git only, nothing heavy): push the repository's
# HISTORY NAMESPACES, every archived branch (refs/archive/<date>/<branch>) and every tag, from the checkout REPO to the repo mirror at
# URL, then read the mirror's own listing back and verify it carries each of them at the same object. Idempotent: a mirror that
# already has them is `HISTORY current`; nothing here deletes a ref on the mirror (an archived branch the grading host later dropped stays
# there; the export carries it harmlessly) and nothing here touches a branch head.
#
# Why this exists beside mirror-repos.sh: the repo mirror is kept by `mirror-repos.sh`, whose push transfers `+refs/heads/*` and
# `+refs/tags/*` and nothing else, and the gate / battery drivers run it only when the sha under test is absent from the mirror. An
# archived branch rides no refspec of its: on 2026-10-09 an archiving pass moved 80 branches on the grading host into refs/archive/<date>/
# and pruned their heads, and the three history commits the seedrep ws1c / ws4h tests read by object presence
# (tests/seedrep_history_fixtures.py: `git cat-file -e <sha>^{commit}` on a FULL, non-shallow checkout) reached the mirror through no
# ref at all once the heads were gone -- so tree_export.sh's `+refs/archive/*:refs/archive/*` fetch had nothing to fetch, and the three
# were pinned as tags (pin/<branch>-<sha8>) to travel at all. This script is the general remedy: the whole archive namespace reaches
# the mirror before the export that reads it, and the tags too (the gate skips mirror-repos.sh when the sha is already there, so a tag
# made after the sha's first push -- a pin -- would otherwise wait for someone else's push). REPO may be any worktree of the
# repository: refs/archive and refs/tags are shared by every worktree.
#
# Invocation (the gate, the battery, the window, the live-DB step, recover.sh, reads_protocol.ensure_tree):
#   mirror_history.sh "$REPO" "ssh://$HOST$MIRROR"       # URL: anything `git push` takes; a local path in the tests
# Output, one line: `HISTORY current <n> refs on <url>` (nothing to push), `HISTORY mirrored <n> refs on <url> (<k> pushed, <n-k>
# current)`, or an `ERR ...` line + exit 2 (REPO is not a checkout; the push failed or a ref was rejected, the push's own report
# indented under it; the mirror's listing still lacks a ref after the push). Exit 0 means: every archived branch and tag REPO has, the
# mirror has, at the same object -- the inventory tree_export.sh then completes an existing tree against.
set -uo pipefail
REPO=${1:?REPO (a checkout of the repository)} URL=${2:?URL (the repo mirror: a git push destination)}
git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1 || { echo "ERR $REPO is not a git checkout"; exit 2; }
inventory() { git -C "$REPO" for-each-ref --format='%(objectname) %(refname)' refs/archive refs/tags | LC_ALL=C sort; }
want=$(inventory) || { echo "ERR cannot list refs/archive and refs/tags of $REPO"; exit 2; }
n=0; [ -z "$want" ] || n=$(wc -l <<<"$want")
if [ "$n" -eq 0 ]; then echo "HISTORY current 0 refs on $URL (no archived branch, no tag)"; exit 0; fi
# the push's own per-ref report (--porcelain: `<flag>\t<from>:<to>\t<summary>`; `=` up to date, `*` new, `+` forced, ` ` fast-forward,
# `!` rejected); a glob refspec that matches nothing pushes nothing and is not an error (measured 2026-10-09, git 2.43)
out=$(git -C "$REPO" push --porcelain "$URL" '+refs/archive/*:refs/archive/*' '+refs/tags/*:refs/tags/*' 2>&1); rc=$?
rejected=$(grep -c $'^!\t' <<<"$out" || true)
if [ "$rc" -ne 0 ] || [ "${rejected:-0}" -gt 0 ]; then
  echo "ERR push of the archived branches and tags of $REPO to $URL failed (rc $rc, $rejected ref(s) rejected)"
  sed 's/^/    /' <<<"$out"; exit 2
fi
pushed=$(grep -c $'^[* +]\t' <<<"$out" || true); pushed=${pushed:-0}
# the mirror's own word: its listing (peeled `^{}` lines dropped; an annotated tag is listed at its tag object, as for-each-ref spells it)
have=$(git -C "$REPO" ls-remote "$URL" 'refs/archive/*' 'refs/tags/*' 2>/dev/null | grep -v $'\\^{}$' | awk '{print $1" "$2}' | LC_ALL=C sort) \
  || { echo "ERR cannot list refs/archive and refs/tags of the mirror $URL after the push"; exit 2; }
missing=$(LC_ALL=C comm -23 <(printf '%s\n' "$want") <(printf '%s\n' "$have"))
if [ -n "$missing" ]; then
  echo "ERR the mirror $URL still lacks $(wc -l <<<"$missing") of $n archived branches / tags after the push: $(head -n 3 <<<"$missing" | awk '{print $2"@"substr($1,1,8)}' | tr '\n' ' ')"
  exit 2
fi
if [ "$pushed" -eq 0 ]; then echo "HISTORY current $n refs on $URL"; else echo "HISTORY mirrored $n refs on $URL ($pushed pushed, $((n - pushed)) current)"; fi
