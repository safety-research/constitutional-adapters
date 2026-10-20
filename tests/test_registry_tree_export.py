"""tree_export.sh carries the mirror's ARCHIVED branches: a commit only ``refs/archive/<date>/<branch>`` reaches travels with the tree.

The export fetched ``+refs/heads/*`` (as ``refs/remotes/mirror/*``) and ``+refs/tags/*`` from the repo mirror. At 2026-10-09T01:58Z an
archiving pass moved 80 branches on the grading host into ``refs/archive/<date>/<branch>`` and the mirror's heads were pruned, so every tree
exported since lacked the commits those branches alone reached -- among them three history commits the seedrep ws1c / ws4h tests read
out of the object store through ``tests/seedrep_history_fixtures.py::pinned_commit_or_skip``, and four nodes read INTRODUCED on any tree
exported after that moment. That fixture resolves a commit by OBJECT PRESENCE (``git cat-file -e <sha>^{commit}`` on a non-shallow
checkout), never by a ref name, so the export now also fetches ``+refs/archive/*:refs/archive/*`` -- the namespace kept as it is: the
archive stands where the grading host put it and nothing of it is mistaken for a live head under ``refs/remotes/mirror`` -- and the full-clone
behavior (every head, every tag, the whole history) is unchanged. The tests here run the script against a synthetic bare mirror whose
one archived commit no head or tag reaches and ask the exported tree the fixture's own question.

Review round 1 (2026-10-09) added the two halves that make the fetch mean something. (HIGH) The mirror is kept by ``mirror-repos.sh``,
whose push moves ``+refs/heads/*`` and ``+refs/tags/*`` and nothing else (:data:`MIRROR_REPOS_REFSPECS`), so an archived branch reached
the mirror through no refspec at all and the new fetch had nothing to fetch: ``mirror_history.sh REPO URL`` (beside the export script,
run by every caller right before the export) pushes ``refs/archive/*`` and ``refs/tags/*`` and verifies the mirror's own listing. The
mirrors here are built BY the production refspecs, never by pre-populating the archive, so the gap is shown before it is filled.
(MEDIUM) A tree exported before the archive reached the mirror used to be reused as it stood (``TREE existing`` on HEAD alone): an
existing tree is now held against the mirror's archive-and-tag inventory and completed IN PLACE (additive objects and refs, HEAD and
the working files untouched), and the line says what was fetched.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from seedrep_history_fixtures import pinned_commit_or_skip

TESTS_DIR = Path(__file__).resolve().parent
LANDING = TESTS_DIR.parent / "scripts" / "registry" / "landing"
DATE = "2026-10-09"


def _env(tmp_path: Path) -> dict:
    """A hermetic git: no user or system config, a synthetic identity (the script commits nothing, the fixture does)."""
    return {"PATH": os.environ["PATH"], "HOME": str(tmp_path / "home"), "GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_AUTHOR_NAME": "tester", "GIT_AUTHOR_EMAIL": "tester@example.invalid", "GIT_COMMITTER_NAME": "tester",
            "GIT_COMMITTER_EMAIL": "tester@example.invalid", "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}


def _git(env: dict, cwd: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, env=env)
    assert r.returncode == 0, (args, r.stdout, r.stderr)
    return r.stdout.strip()


def _export(env: dict, sha: str, mirror: Path, tree: Path) -> subprocess.CompletedProcess:
    """The script as the grading host invokes it on the login node: over stdin, with the sha, the mirror and the destination."""
    return subprocess.run(["bash", "-s", "--", sha, str(mirror), str(tree)], input=(LANDING / "tree_export.sh").read_text(),
                          capture_output=True, text=True, env=env, timeout=120)


#: mirror-repos.sh's push (its one `git push` line, as of 2026-10-09): every branch head and every tag, forced; NO refs/archive/*.
MIRROR_REPOS_REFSPECS = ("+refs/heads/*:refs/heads/*", "+refs/tags/*:refs/tags/*")


def _history(env: dict, repo: Path, url: str) -> subprocess.CompletedProcess:
    """mirror_history.sh as every caller runs it: the checkout and the mirror's push URL (a path here; ``ssh://host/path`` in production)."""
    return subprocess.run(["bash", str(LANDING / "mirror_history.sh"), str(repo), url], capture_output=True, text=True, env=env, timeout=120)


def _source_after_the_archive_pass(tmp_path: Path, env: dict):
    """The grading host checkout: ``main`` (one commit, tagged ``v1``) and a branch ``seedrep-probe`` one commit ahead, then the archiving pass as it
    ran on 2026-10-09 -- the branch becomes ``refs/archive/<date>/seedrep-probe`` and its head is deleted, so the checkout holds a commit no
    head and no tag reaches. Returns ``(work, main sha, the archived commit's sha)``."""
    work = tmp_path / "work"
    work.mkdir()
    _git(env, work, "init", "-q", "-b", "main")
    (work / "a.txt").write_text("main\n")
    _git(env, work, "add", "-A")
    _git(env, work, "commit", "-q", "-m", "main")
    main_sha = _git(env, work, "rev-parse", "HEAD")
    _git(env, work, "tag", "-a", "v1", "-m", "v1")
    _git(env, work, "checkout", "-q", "-b", "seedrep-probe")
    (work / "probe.txt").write_text("only the archived branch carries this file\n")
    _git(env, work, "add", "-A")
    _git(env, work, "commit", "-q", "-m", "probe")
    archived = _git(env, work, "rev-parse", "HEAD")
    _git(env, work, "checkout", "-q", "main")
    # the archiving pass: the branch moves under refs/archive/<date>/ and its head goes
    _git(env, work, "update-ref", f"refs/archive/{DATE}/seedrep-probe", archived)
    _git(env, work, "branch", "-D", "-q", "seedrep-probe")
    assert _git(env, work, "for-each-ref", "--format=%(refname)").splitlines() == [f"refs/archive/{DATE}/seedrep-probe", "refs/heads/main", "refs/tags/v1"]
    assert archived not in _git(env, work, "rev-list", "refs/heads/main", "refs/tags/v1").splitlines(), "no head or tag reaches the archived commit"
    return work, main_sha, archived


def _mirror_by_the_production_refspecs(tmp_path: Path, env: dict, work: Path) -> Path:
    """The repo mirror as mirror-repos.sh leaves it: a bare repository pushed with exactly its two refspecs, named (as the tool names every
    clone) after the source checkout's directory."""
    mirror = tmp_path / "shared" / "repos" / work.name
    mirror.parent.mkdir(parents=True, exist_ok=True)
    _git(env, tmp_path, "init", "-q", "--bare", str(mirror))
    _git(env, work, "push", "-q", str(mirror), *MIRROR_REPOS_REFSPECS)
    return mirror


def _has_commit(env: dict, repo: Path, sha: str) -> bool:
    return subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True, env=env).returncode == 0


def _mirror_with_an_archived_branch(tmp_path: Path, env: dict):
    """The 2026-10-09 state and its remedy, through the production path: the source after the archive pass, the mirror pushed by
    mirror-repos.sh's refspecs (which leave the archived commit behind -- asserted), then mirror_history.sh (the step every caller now
    runs before the export), after which the mirror holds the archived commit under its archive ref and still no head reaches it.
    Returns ``(mirror, main sha, the archived commit's sha)``."""
    work, main_sha, archived = _source_after_the_archive_pass(tmp_path, env)
    mirror = _mirror_by_the_production_refspecs(tmp_path, env, work)
    assert _git(env, mirror, "for-each-ref", "--format=%(refname)").splitlines() == ["refs/heads/main", "refs/tags/v1"], "heads and tags: the production push"
    assert not _has_commit(env, mirror, archived), "the production refspecs carry no archived branch: the mirror lacks the commit outright"
    r = _history(env, work, str(mirror))
    assert r.returncode == 0 and r.stdout.strip() == f"HISTORY mirrored 2 refs on {mirror} (1 pushed, 1 current)", r.stdout + r.stderr
    assert _git(env, mirror, "for-each-ref", "--format=%(refname)").splitlines() == [f"refs/archive/{DATE}/seedrep-probe", "refs/heads/main", "refs/tags/v1"]
    assert archived not in _git(env, mirror, "rev-list", "refs/heads/main", "refs/tags/v1").splitlines(), "no head or tag reaches the archived commit"
    assert _has_commit(env, mirror, archived)
    return mirror, main_sha, archived


def test_the_exported_tree_carries_a_commit_only_an_archived_branch_reaches(tmp_path):
    """The defect and the fix in one shape: with the branch archived and its head pruned, the exported tree must hold the archived commit
    (``cat-file -e <sha>^{commit}``, the fixture's question; ``rev-parse --verify``; its blob readable), under the archive ref AS the
    mirror spells it -- not as a remote head -- while every head and tag arrives as before and the checkout is not shallow."""
    env = _env(tmp_path)
    mirror, main_sha, archived = _mirror_with_an_archived_branch(tmp_path, env)
    tree = tmp_path / "shared" / "gates" / "trees" / main_sha
    r = _export(env, main_sha, mirror, tree)
    assert r.returncode == 0 and r.stdout.startswith(f"TREE exported {tree} ("), r.stdout + r.stderr
    assert _git(env, tree, "rev-parse", "HEAD") == main_sha and _git(env, tree, "rev-parse", "--is-shallow-repository") == "false"
    # the fixture's own question, in the fixture's words
    assert subprocess.run(["git", "-C", str(tree), "cat-file", "-e", f"{archived}^{{commit}}"], capture_output=True, text=True, env=env).returncode == 0, \
        f"the archived commit {archived[:8]} did not travel with the tree (refs: {_git(env, tree, 'for-each-ref', '--format=%(refname)')})"
    assert _git(env, tree, "rev-parse", "--verify", f"{archived}^{{commit}}") == archived
    assert _git(env, tree, "show", f"{archived}:probe.txt") == "only the archived branch carries this file"
    # the namespace is kept: the archive ref stands where the grading host put it, never as a live head of the mirror
    refs = _git(env, tree, "for-each-ref", "--format=%(refname)").splitlines()
    assert f"refs/archive/{DATE}/seedrep-probe" in refs and _git(env, tree, "rev-parse", f"refs/archive/{DATE}/seedrep-probe") == archived
    assert "refs/remotes/mirror/seedrep-probe" not in refs, refs
    # the full-clone behavior as it was: every head as refs/remotes/mirror/*, every tag, the whole history
    assert _git(env, tree, "rev-parse", "refs/remotes/mirror/main") == main_sha and _git(env, tree, "rev-parse", "refs/tags/v1^{commit}") == main_sha
    assert r.stdout.rstrip().endswith("1 commits, 3 refs)"), r.stdout        # HEAD's own history is one commit; the refs: the head, the tag, the archive
    # idempotent: a second export of the same sha at the same path is the existing tree
    r2 = _export(env, main_sha, mirror, tree)
    assert r2.returncode == 0 and r2.stdout.strip() == f"TREE existing {tree}", r2.stdout + r2.stderr


def test_the_history_fixture_finds_an_archived_commit_in_the_exported_tree_without_a_skip_or_a_fail(tmp_path):
    """The seedrep tests' fixture against the exported tree, verbatim: ``pinned_commit_or_skip`` RETURNS for the archived commit (object
    present, non-shallow checkout) -- where a tree exported without the archive namespace made it fail by name ("is not in this FULL
    checkout"), the four INTRODUCED nodes of 2026-10-09."""
    env = _env(tmp_path)
    mirror, main_sha, archived = _mirror_with_an_archived_branch(tmp_path, env)
    tree = tmp_path / "shared" / "gates" / "trees" / main_sha
    r = _export(env, main_sha, mirror, tree)
    assert r.returncode == 0, r.stdout + r.stderr
    pinned_commit_or_skip(archived, "seedrep-probe", "the archived branch's commit", repo=tree, anchor=tree / "a.txt")
    # and a commit that is in NO store still fails by name through the same fixture (the rule's other half is untouched)
    with pytest.raises(BaseException, match="is not in this FULL checkout"):
        pinned_commit_or_skip("0" * 40, "nowhere", "a commit nobody has", repo=tree, anchor=tree / "a.txt")


def test_the_production_mirroring_refspecs_leave_the_archive_behind_and_mirror_history_carries_it_verified(tmp_path):
    """(HIGH) The gap and its fill, each shown: a mirror pushed by mirror-repos.sh's two refspecs after the archive pass has neither the
    archive ref nor the commit, so the export's archive fetch had nothing to fetch (the tree lacks the commit, by the fixture's own
    question); mirror_history.sh pushes the archived branch (the tag is already current) and reads the mirror's listing back; run again it
    pushes nothing and says so; the export then carries the commit. The constant is held against the grading host's own mirror-repos.sh when
    that file is present (an operator machine), so a change to its refspecs is noticed here."""
    env = _env(tmp_path)
    work, main_sha, archived = _source_after_the_archive_pass(tmp_path, env)
    mirror = _mirror_by_the_production_refspecs(tmp_path, env, work)
    assert not _has_commit(env, mirror, archived) and f"refs/archive/{DATE}/seedrep-probe" not in _git(env, mirror, "for-each-ref", "--format=%(refname)")
    # the export against that mirror: the fetch of refs/archive/* has nothing to fetch; the tree lacks the commit
    tree0 = tmp_path / "shared" / "gates" / "trees0" / main_sha
    r0 = _export(env, main_sha, mirror, tree0)
    assert r0.returncode == 0 and r0.stdout.startswith("TREE exported "), r0.stdout + r0.stderr
    assert not _has_commit(env, tree0, archived), "a mirror without the archive exports a tree without it: the 2026-10-09 trees"
    # the fill: the archived branch goes; the tag was already there (1 pushed, 1 current); the mirror's listing is the proof
    r = _history(env, work, str(mirror))
    assert r.returncode == 0 and r.stdout.strip() == f"HISTORY mirrored 2 refs on {mirror} (1 pushed, 1 current)", r.stdout + r.stderr
    assert _git(env, mirror, "rev-parse", f"refs/archive/{DATE}/seedrep-probe") == archived and _has_commit(env, mirror, archived)
    assert _git(env, mirror, "rev-parse", "refs/heads/main") == main_sha, "no head moved"
    r2 = _history(env, work, str(mirror))
    assert r2.returncode == 0 and r2.stdout.strip() == f"HISTORY current 2 refs on {mirror}", r2.stdout + r2.stderr
    # a worktree of the checkout spells the same inventory (refs/archive and refs/tags are shared), so any caller's checkout serves
    wt = tmp_path / "wt"
    _git(env, work, "worktree", "add", "-q", "--detach", str(wt), main_sha)
    r3 = _history(env, wt, str(mirror))
    assert r3.returncode == 0 and r3.stdout.strip() == f"HISTORY current 2 refs on {mirror}", r3.stdout + r3.stderr
    # the export now carries the commit, under the archive ref, as before
    tree = tmp_path / "shared" / "gates" / "trees" / main_sha
    r4 = _export(env, main_sha, mirror, tree)
    assert r4.returncode == 0 and r4.stdout.startswith("TREE exported "), r4.stdout + r4.stderr
    assert _has_commit(env, tree, archived) and _git(env, tree, "rev-parse", f"refs/archive/{DATE}/seedrep-probe") == archived
    pinned_commit_or_skip(archived, "seedrep-probe", "the archived branch's commit", repo=tree, anchor=tree / "a.txt")
    # the constant against the operator's own mirror-repos.sh, where it exists: its one push line carries exactly these refspecs
    grading_host_script = Path(os.path.expanduser("~")) / "scripts" / "mirror-repos.sh"
    if grading_host_script.is_file():
        push_lines = [l for l in grading_host_script.read_text().splitlines() if "git -C" in l and " push " in l and "refs/heads/*" in l]
        assert len(push_lines) == 1, push_lines
        assert all(f"'{spec}'" in push_lines[0] for spec in MIRROR_REPOS_REFSPECS) and "refs/archive" not in push_lines[0], push_lines[0]


def test_mirror_history_says_when_there_is_nothing_to_push_and_refuses_by_name(tmp_path):
    """A checkout with no archived branch and no tag pushes nothing (no network at all: the URL may not even exist); a path that is not a
    checkout refuses; a URL that is not a repository refuses with the push's own report; a mirror whose hook rejects the archive namespace
    refuses naming the rejected count -- never a silent `HISTORY current`."""
    env = _env(tmp_path)
    bare = tmp_path / "bare"
    bare.mkdir()
    _git(env, bare, "init", "-q", "-b", "main")
    (bare / "f").write_text("f\n")
    _git(env, bare, "add", "-A")
    _git(env, bare, "commit", "-q", "-m", "f")
    r = _history(env, bare, str(tmp_path / "no-such-mirror"))
    assert r.returncode == 0 and r.stdout.strip() == f"HISTORY current 0 refs on {tmp_path / 'no-such-mirror'} (no archived branch, no tag)", r.stdout + r.stderr
    r = _history(env, tmp_path / "not-a-checkout", str(tmp_path / "no-such-mirror"))
    assert r.returncode == 2 and r.stdout.startswith("ERR ") and "is not a git checkout" in r.stdout, r.stdout + r.stderr
    work, main_sha, archived = _source_after_the_archive_pass(tmp_path, env)
    r = _history(env, work, str(tmp_path / "no-such-mirror"))
    assert r.returncode == 2 and r.stdout.startswith("ERR push of the archived branches and tags of ") and "failed (rc 128, 0 ref(s) rejected)" in r.stdout, r.stdout + r.stderr
    assert "    fatal:" in r.stdout, "the push's own report, indented under the ERR line"
    mirror = _mirror_by_the_production_refspecs(tmp_path, env, work)
    hook = mirror / "hooks" / "pre-receive"
    hook.write_text("#!/bin/bash\nrc=0\nwhile read -r old new ref; do case \"$ref\" in refs/archive/*) echo \"archive refused by the hook: $ref\" >&2; rc=1 ;; esac; done\nexit $rc\n")
    hook.chmod(0o755)
    r = _history(env, work, str(mirror))
    assert r.returncode == 2 and r.stdout.startswith("ERR push of the archived branches and tags of ") and "1 ref(s) rejected)" in r.stdout, r.stdout + r.stderr
    assert "archive refused by the hook" in r.stdout and f"refs/archive/{DATE}/seedrep-probe" not in _git(env, mirror, "for-each-ref", "--format=%(refname)")


def test_an_existing_tree_exported_before_the_archive_reached_the_mirror_is_completed_in_place_on_reuse(tmp_path):
    """(MEDIUM) A tree exported from a mirror without the archive (the 2026-10-09 trees), reused after the mirror was repaired: `TREE existing`
    is no longer HEAD alone -- the tree is held against the mirror's archive-and-tag inventory, what it lacks is fetched into it, the line
    says so, and the fixture's question is then answered. HEAD, the working files and the non-shallow shape are untouched; a third run is
    the plain `TREE existing`; a tag made later (a pin) completes the same way; a tree whose HEAD is another commit still refuses by name."""
    env = _env(tmp_path)
    work, main_sha, archived = _source_after_the_archive_pass(tmp_path, env)
    mirror = _mirror_by_the_production_refspecs(tmp_path, env, work)
    tree = tmp_path / "shared" / "gates" / "trees" / main_sha
    r = _export(env, main_sha, mirror, tree)
    assert r.returncode == 0 and r.stdout.startswith("TREE exported "), r.stdout + r.stderr
    assert not _has_commit(env, tree, archived), "exported before the archive was mirrored: the tree lacks the commit"
    with pytest.raises(BaseException, match="is not in this FULL checkout"):
        pinned_commit_or_skip(archived, "seedrep-probe", "the archived branch's commit", repo=tree, anchor=tree / "a.txt")
    before = {"head": _git(env, tree, "rev-parse", "HEAD"), "status": _git(env, tree, "status", "--porcelain"), "a": (tree / "a.txt").read_text()}
    # the mirror is repaired (mirror_history.sh, as every caller now runs it before the export); the same sha is exported again
    assert _history(env, work, str(mirror)).returncode == 0
    r2 = _export(env, main_sha, mirror, tree)
    assert r2.returncode == 0 and r2.stdout.strip() == f"TREE existing {tree} (completed: 1 archived branches / tags fetched)", r2.stdout + r2.stderr
    assert _has_commit(env, tree, archived) and _git(env, tree, "rev-parse", f"refs/archive/{DATE}/seedrep-probe") == archived
    pinned_commit_or_skip(archived, "seedrep-probe", "the archived branch's commit", repo=tree, anchor=tree / "a.txt")
    assert _git(env, tree, "show", f"{archived}:probe.txt") == "only the archived branch carries this file"
    refs = _git(env, tree, "for-each-ref", "--format=%(refname)").splitlines()
    assert "refs/remotes/mirror/seedrep-probe" not in refs and "refs/remotes/mirror/main" in refs, refs
    after = {"head": _git(env, tree, "rev-parse", "HEAD"), "status": _git(env, tree, "status", "--porcelain"), "a": (tree / "a.txt").read_text()}
    assert after == before == {"head": main_sha, "status": "", "a": "main\n"}, "HEAD, the index and the working files are untouched"
    assert _git(env, tree, "rev-parse", "--is-shallow-repository") == "false" and not (tree.with_name(tree.name + ".tmp")).exists()
    r3 = _export(env, main_sha, mirror, tree)
    assert r3.returncode == 0 and r3.stdout.strip() == f"TREE existing {tree}", r3.stdout + r3.stderr
    # a pin tag made after the export (the remedy in force for the three seedrep commits) completes the same way
    _git(env, work, "tag", "-a", f"pin/seedrep-probe-{archived[:8]}", "-m", "pin", archived)
    assert _history(env, work, str(mirror)).stdout.strip() == f"HISTORY mirrored 3 refs on {mirror} (1 pushed, 2 current)"
    r4 = _export(env, main_sha, mirror, tree)
    assert r4.returncode == 0 and r4.stdout.strip() == f"TREE existing {tree} (completed: 1 archived branches / tags fetched)", r4.stdout + r4.stderr
    assert _git(env, tree, "rev-parse", f"refs/tags/pin/seedrep-probe-{archived[:8]}^{{commit}}") == archived
    # another commit at the same path still refuses by name: a path is one commit, forever
    r5 = _export(env, archived, mirror, tree)
    assert r5.returncode == 2 and r5.stdout.strip() == f"ERR existing tree at {tree} has HEAD '{main_sha}'", r5.stdout + r5.stderr
    # and a mirror path that is not a repository refuses before anything is read or written
    r6 = _export(env, main_sha, tmp_path / "no-mirror", tmp_path / "shared" / "gates" / "trees" / "x" / main_sha)
    assert r6.returncode == 2 and r6.stdout.startswith("ERR no repo mirror at "), r6.stdout + r6.stderr


def test_the_python_reads_path_pushes_the_history_between_the_mirror_and_the_export(tmp_path):
    """reads_protocol.ensure_tree (the window's cluster reads): mirror-repos.sh, then mirror_history.sh with the destination's source
    checkout and its mirror URL (the script beside tree_export.sh by default), then the export over ``bash -s``; a failing history push is a
    logged note and the export still runs (the reads need the tree's code only). Then the same function end to end on a local destination
    with the real scripts: the exported tree carries the archived commit."""
    from sisteer.registry import reads_protocol as RP
    from sisteer.registry import stores as ST
    env = _env(tmp_path)
    calls, notes = [], []

    def runner(argv, timeout_s, input=None):
        calls.append(list(argv))
        if argv[0].endswith("mirror_history.sh"):
            return subprocess.CompletedProcess(argv, 2, stdout="ERR the mirror still lacks 1 of 3 archived branches / tags after the push: x\n", stderr="")
        return subprocess.CompletedProcess(argv, 0, stdout="TREE existing /shared/gates/trees/" + "a" * 40 + "\n", stderr="")
    dest = ST.Destination(name="cluster_a", host="node-shim", shared="/shared", source_repo="/checkouts/constitutional-adapters")
    export_script = tmp_path / "landing" / "tree_export.sh"
    export_script.parent.mkdir()
    export_script.write_text("#!/bin/bash\necho TREE existing\n")
    tree = RP.ensure_tree(dest, "a" * 40, "/x/mirror-repos.sh", str(export_script), runner=runner, log=notes.append)
    assert tree == "/shared/gates/trees/" + "a" * 40
    assert [c[0] for c in calls] == ["/x/mirror-repos.sh", str(tmp_path / "landing" / "mirror_history.sh"), "ssh"], calls
    assert calls[1] == [str(tmp_path / "landing" / "mirror_history.sh"), "/checkouts/constitutional-adapters", "node-shim:/shared/repos/constitutional-adapters"], calls[1]
    assert calls[2][-6:] == ["bash", "-s", "--", "a" * 40, "/shared/repos/constitutional-adapters", "/shared/gates/trees/" + "a" * 40], calls[2]
    assert notes == ["tree: mirror_history.sh reported rc 2 (ERR the mirror still lacks 1 of 3 archived branches / tags after the push: x); "
                     "the exported tree may lack archived branches or tags"], notes
    # end to end on a local destination: the real scripts, a mirror pushed by the production refspecs, the window's own source checkout
    work, main_sha, archived = _source_after_the_archive_pass(tmp_path, env)
    mirror = _mirror_by_the_production_refspecs(tmp_path, env, work)
    assert not _has_commit(env, mirror, archived)
    shim = tmp_path / "mirror-repos.sh"
    shim.write_text("#!/bin/bash\nexit 0\n")
    shim.chmod(0o755)
    local = ST.Destination(name="cluster_a", host=None, shared=str(tmp_path / "shared"), source_repo=str(work))
    notes2 = []
    with pytest.MonkeyPatch.context() as mp:
        for k, v in env.items():
            mp.setenv(k, v)
        out = RP.ensure_tree(local, main_sha, str(shim), str(LANDING / "tree_export.sh"), log=notes2.append)
    assert out == str(tmp_path / "shared" / "gates" / "trees" / main_sha) and notes2 == [], notes2
    assert _has_commit(env, mirror, archived), "mirror_history.sh ran against the mirror"
    assert _has_commit(env, Path(out), archived) and _git(env, Path(out), "rev-parse", "HEAD") == main_sha
    pinned_commit_or_skip(archived, "seedrep-probe", "the archived branch's commit", repo=Path(out), anchor=Path(out) / "a.txt")
