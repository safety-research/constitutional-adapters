"""``seedrep_history_fixtures.pinned_commit_or_skip`` -- the exact-absence rule for the seedrep tests that read a pinned commit
out of the object store (a full checkout under another owner -- rc 128 "detected dubious ownership" -- is a Git access failure,
never "not a repository"; the shallow-status command must itself succeed). Every cell has its known-bad twin in the table: the
``cat-file -e``-only predicate skipped ``full_missing``; the ``--is-inside-work-tree``-rc-only predicate skipped
``dubious_ownership_*``; the marker-file-only predicate skipped a development checkout with a stray marker.

The published repository is the third positively established absence, by ``publication_root``'s whole rule: not shallow, the
marker file present with content ``squashed``, HEAD's one root commit carrying it. Commits made on top of the publication
(the committed configs the reproduction asks for) are descendants of that root and change nothing -- never a commit count."""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from seedrep_history_fixtures import (HISTORY_SQUASHED_MARKER, MARKER_CONTENT, NotPublished, git_metadata, history_squashed_marker,
                                      pinned_commit_or_skip, publication_root)

REPO = Path(__file__).resolve().parents[1]
SHA = "0123456789abcdef0123456789abcdef01234567"
ROOT = "1111111111111111111111111111111111111111"
OK_GIT_DIR = (0, ".git\n", "")
NOT_REPO = (128, "", "fatal: not a git repository (or any of the parent directories): .git")
DUBIOUS = (128, "", "fatal: detected dubious ownership in repository at '/mnt/src'\nTo add an exception for this directory, call:\n"
                    "\tgit config --global --add safe.directory /mnt/src")
FULL, SHALLOW = (0, "false\n", ""), (0, "true\n", "")
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
           "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z", "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z"}


class _FakeGit:
    """``subprocess.run`` for the helper's git calls, each answered by an (rc, stdout, stderr) triple: the repository's metadata, its shallow
    status, the pinned commit's presence, HEAD's root commits (``roots``) and what the root commit carries as the marker (``root_marker``:
    its content, or None for no such file)."""

    def __init__(self, *, git_dir=OK_GIT_DIR, shallow=FULL, present=False, roots=(ROOT,), root_marker=MARKER_CONTENT):
        self.git_dir, self.shallow, self.present, self.roots, self.root_marker, self.calls = git_dir, shallow, present, roots, root_marker, []

    def __call__(self, argv, **kw):
        assert argv[:2] == ["git", "-C"] and kw == {"capture_output": True, "text": True, "check": False}
        sub = list(argv[3:])
        self.calls.append(sub)
        if sub == ["rev-parse", "--git-dir"]:
            rc, out, err = self.git_dir
        elif sub == ["rev-parse", "--is-shallow-repository"]:
            rc, out, err = self.shallow
        elif sub[:2] == ["cat-file", "-e"]:
            rc, out, err = (0, "", "") if self.present else (128, "", f"fatal: Not a valid object name {sub[2]}")
        elif sub == ["rev-list", "--max-parents=0", "HEAD"]:
            rc, out, err = 0, "".join(r + "\n" for r in self.roots), ""
        elif sub[:2] == ["cat-file", "-p"] and sub[2] == f"{ROOT}:{HISTORY_SQUASHED_MARKER}":
            rc, out, err = (0, self.root_marker + "\n", "") if self.root_marker is not None else (128, "", f"fatal: path '{HISTORY_SQUASHED_MARKER}' does not exist in '{ROOT}'")
        else:
            raise AssertionError(sub)
        return subprocess.CompletedProcess(argv, rc, stdout=out, stderr=err)


def _tree(tmp_path: Path, metadata, marker=False):
    """A repo-shaped tree: ``metadata`` None = source-only (no ``.git``); "dir" = a checkout; "file" = a linked worktree's ``.git`` file.
    ``marker`` True writes the publish step's ``.history-squashed`` at the root (content ``squashed``); a string writes that content."""
    repo = tmp_path / "src"
    anchor = repo / "tests" / "test_probe.py"
    anchor.parent.mkdir(parents=True)
    anchor.write_text("")
    if metadata == "dir":
        (repo / ".git").mkdir()
    elif metadata == "file":
        (repo / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    if marker:
        (repo / HISTORY_SQUASHED_MARKER).write_text((MARKER_CONTENT if marker is True else marker) + "\n")
    return repo, anchor


CASES = [
    # id,                                     .git,   marker,  git,                                              outcome,  text in the message
    ("not_a_repo",                            None,   False,   _FakeGit(git_dir=NOT_REPO),                       "skip",   "not a git repository: no .git entry"),
    ("dubious_ownership_full_checkout",       "dir",  False,   _FakeGit(git_dir=DUBIOUS),                        "fail",   "detected dubious ownership"),
    ("dubious_ownership_linked_worktree",     "file", False,   _FakeGit(git_dir=DUBIOUS),                        "fail",   "detected dubious ownership"),
    ("not_a_repo_said_but_metadata_present",  "dir",  False,   _FakeGit(git_dir=NOT_REPO),                       "fail",   "not an absence of history"),
    ("no_metadata_but_an_access_error",       None,   False,   _FakeGit(git_dir=DUBIOUS),                        "fail",   "detected dubious ownership"),
    ("shallow_status_command_fails",          "dir",  False,   _FakeGit(shallow=(128, "", "fatal: bad config")), "fail",   "--is-shallow-repository failed"),
    ("shallow_missing",                       "dir",  False,   _FakeGit(shallow=SHALLOW),                        "skip",   "shallow clone"),
    ("full_missing",                          "dir",  False,   _FakeGit(),                                       "fail",   "run `git fetch origin some-branch`"),
    # the published repository: the publish step's root marker, borne out by the repository -- not shallow, HEAD's one root commit carrying it --
    # is the repository's own statement that it carries no development history, so a full clone lacking the commit SKIPS there, naming the
    # publication root; its known-bad twin is `full_missing` above: the same git answers without the marker still FAIL, naming the branch
    ("squashed_marker_missing",               "dir",  True,    _FakeGit(),                                       "skip",   f"committed by its publication root {ROOT[:8]}"),
    ("squashed_marker_present_commit",        "dir",  True,    _FakeGit(present=True),                           "return", ""),
    # the marker decides nothing before the object store is asked: a Git access failure or a shallow status failure still FAILS with it
    ("squashed_marker_dubious_ownership",     "dir",  True,    _FakeGit(git_dir=DUBIOUS),                        "fail",   "detected dubious ownership"),
    ("squashed_marker_shallow_status_fails",  "dir",  True,    _FakeGit(shallow=(128, "", "fatal: bad config")), "fail",   "--is-shallow-repository failed"),
    # ...and a marker the repository does not bear out FAILS by name, never skips: the root commit does not carry it (a development checkout
    # with a stray marker file), HEAD has two roots, or the marker's content is not the publish step's
    ("squashed_marker_root_does_not_carry_it", "dir", True,    _FakeGit(root_marker=None),                       "fail",   f"the root commit {ROOT[:12]} does not carry {HISTORY_SQUASHED_MARKER}"),
    ("squashed_marker_root_other_content",    "dir",  True,    _FakeGit(root_marker="squished"),                 "fail",   "not `squashed`"),
    # whitespace INSIDE the content is other content (only surrounding whitespace is stripped); around it, it is the marker
    ("squashed_marker_root_spaced_inside",    "dir",  True,    _FakeGit(root_marker="squa shed"),                "fail",   "reading 'squa shed', not `squashed`"),
    ("squashed_marker_file_spaced_inside",    "dir",  "squa shed", _FakeGit(),                                   "fail",   "reads 'squa shed', not `squashed`"),
    ("squashed_marker_padded",                "dir",  "  squashed \t", _FakeGit(root_marker="\n squashed  "),    "skip",   f"committed by its publication root {ROOT[:8]}"),
    ("squashed_marker_two_roots",             "dir",  True,    _FakeGit(roots=(ROOT, "2" * 40)),                 "fail",   "HEAD has 2 root commits"),
    ("squashed_marker_file_other_content",    "dir",  "nope",  _FakeGit(),                                       "fail",   "reads 'nope', not `squashed`"),
    # a shallow clone with the marker file skips as a shallow clone (the pre-existing exception, read first) -- never as a publication
    ("squashed_marker_shallow_clone",         "dir",  True,    _FakeGit(shallow=SHALLOW),                        "skip",   "shallow clone"),
    ("present",                               "dir",  False,   _FakeGit(present=True),                           "return", ""),
    ("present_shallow",                       "dir",  False,   _FakeGit(shallow=SHALLOW, present=True),          "return", ""),
]


@pytest.mark.parametrize("metadata, marker, git, outcome, text", [c[1:] for c in CASES], ids=[c[0] for c in CASES])
def test_the_history_skip_fires_only_for_a_source_only_tree_a_shallow_clone_or_a_borne_out_publication_and_everything_else_fails(
        tmp_path, metadata, marker, git, outcome, text):
    repo, anchor = _tree(tmp_path, metadata, marker)
    if outcome == "return":
        assert pinned_commit_or_skip(SHA, "some-branch", "the probe", repo=repo, anchor=anchor, run=git) is None
        assert git.calls[-1][:2] == ["cat-file", "-e"]
        return
    with pytest.raises(pytest.skip.Exception if outcome == "skip" else pytest.fail.Exception) as ei:
        pinned_commit_or_skip(SHA, "some-branch", "the probe", repo=repo, anchor=anchor, run=git)
    msg = str(ei.value)
    assert text in msg and "01234567" in msg and "the probe" in msg, msg
    if outcome == "fail" and "dubious" in text:
        assert "safe.directory" in msg                                            # the fix is named, the skip is refused
    if outcome == "fail" and marker and "dubious" not in text and "--is-shallow" not in text:
        assert "does not bear it out" in msg and "a development checkout never carries the marker" in msg, msg
    if metadata is None and outcome == "skip":
        assert git.calls == [["rev-parse", "--git-dir"]]                        # the absence is decided before any object read
    if outcome == "skip" and "publication root" in text:
        assert ["rev-list", "--max-parents=0", "HEAD"] in git.calls and ["cat-file", "-p", f"{ROOT}:{HISTORY_SQUASHED_MARKER}"] in git.calls


def test_git_metadata_walks_up_to_a_directory_or_a_worktree_file_and_is_none_for_a_source_only_tree(tmp_path):
    for kind, expect in ((None, None), ("dir", "src/.git"), ("file", "src/.git")):
        repo, anchor = _tree(tmp_path / str(kind), kind)
        got = git_metadata(anchor)
        assert (got is None) == (expect is None) and (got is None or got == tmp_path / str(kind) / expect and got.exists())


def test_on_this_checkout_the_real_git_answers_and_head_is_present():
    """The real thing, once: this tree (a checkout or a linked worktree, whose ``.git`` is a FILE) has metadata, HEAD is present, so
    the helper returns; an unknown sha on a FULL checkout fails naming the branch (skips only when this checkout is shallow, or when
    the repository was published without its development history and says so with the publish step's root marker, borne out)."""
    assert git_metadata(Path(__file__)) is not None
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert pinned_commit_or_skip(head, "main", "HEAD itself", repo=REPO, anchor=Path(__file__)) is None
    shallow = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--is-shallow-repository"], capture_output=True, text=True, check=True).stdout.strip()
    squashed = publication_root(REPO) is not None
    if shallow == "true":
        exc, text = pytest.skip.Exception, "shallow clone"
    elif squashed:
        exc, text = pytest.skip.Exception, "carries no development history"
    else:
        exc, text = pytest.fail.Exception, "git fetch origin some-branch"
    with pytest.raises(exc, match=text):
        pinned_commit_or_skip(SHA, "some-branch", "a sha nobody has", repo=REPO, anchor=Path(__file__))


def test_the_squashed_marker_is_consistent_with_this_checkouts_shape():
    """The marker is the repository's statement that it was published without its development history: a development checkout never
    carries it (the publish step alone writes it, into the publication's root commit), and a checkout that carries it must be borne out --
    not shallow, the one root commit carrying it with content ``squashed`` -- whatever was committed on top of the publication."""
    marker = history_squashed_marker(REPO)
    if marker is None:
        assert publication_root(REPO) is None
        return
    assert marker.read_text().strip() == MARKER_CONTENT, marker.read_text()[:80]
    try:
        root = publication_root(REPO)
    except NotPublished as exc:
        pytest.fail(f"{HISTORY_SQUASHED_MARKER} is present but this checkout does not bear it out: {exc} -- the marker belongs only to a publication")
    assert root is not None and subprocess.run(["git", "-C", str(REPO), "rev-list", "--max-parents=0", "HEAD"], capture_output=True, text=True,
                                               check=True).stdout.split() == [root]


# ------------------------------------------------------------------------------------------------------ real repositories, real git
def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, env={**os.environ, **GIT_ENV})


def _repo(path: Path, *, marker_in_root: bool, commits: int) -> Path:
    """A repository of its own (none of this repo's commits): ``marker_in_root`` commits the publish step's marker in the FIRST commit;
    ``commits`` more empty commits follow."""
    path.mkdir(parents=True)
    assert _git(path, "init", "-q").returncode == 0
    (path / "tests").mkdir()
    (path / "tests" / "test_probe.py").write_text("")
    _git(path, "add", "tests/test_probe.py")
    if marker_in_root:
        (path / HISTORY_SQUASHED_MARKER).write_text(MARKER_CONTENT + "\n")
        _git(path, "add", HISTORY_SQUASHED_MARKER)
    assert _git(path, "commit", "-q", "-m", "c0").returncode == 0
    for i in range(1, commits):
        assert _git(path, "commit", "-q", "--allow-empty", "-m", f"c{i}").returncode == 0
    return path


def test_a_publication_with_a_users_commit_on_top_still_skips_and_a_marker_the_root_does_not_carry_fails(tmp_path):
    """The regression the one-commit rule failed: clone the publication, commit the generated configs (as the reproduction asks) -- the
    marker is still borne out by the root commit, so the pinned-commit check skips with the published-repository reason. Its known-bad twins:
    the marker committed in a LATER commit (the root does not carry it), a stray uncommitted marker, two roots, other content."""
    one = _repo(tmp_path / "one", marker_in_root=True, commits=1)
    root = _git(one, "rev-parse", "HEAD").stdout.strip()
    assert publication_root(one) == root
    with pytest.raises(pytest.skip.Exception, match=f"committed by its publication root {root[:8]}"):
        pinned_commit_or_skip(SHA, "some-branch", "the probe", repo=one, anchor=one / "tests" / "test_probe.py")
    # a commit on top: configs committed after the publication -- a descendant of the root, still the publication
    later = _repo(tmp_path / "later", marker_in_root=True, commits=1)
    root = _git(later, "rev-parse", "HEAD").stdout.strip()
    (later / "configs").mkdir()
    (later / "configs" / "generated.yaml").write_text("x: 1\n")
    _git(later, "add", "configs/generated.yaml")
    assert _git(later, "commit", "-q", "-m", "the generated configs, committed").returncode == 0
    assert _git(later, "rev-list", "--count", "HEAD").stdout.strip() == "2" and _git(later, "rev-parse", "HEAD").stdout.strip() != root
    assert publication_root(later) == root
    with pytest.raises(pytest.skip.Exception, match=f"carries no development history .* committed by its publication root {root[:8]}"):
        pinned_commit_or_skip(SHA, "some-branch", "the probe", repo=later, anchor=later / "tests" / "test_probe.py")
    # a present commit returns before the marker is consulted, on the publication too
    assert pinned_commit_or_skip(root, "some-branch", "the probe", repo=later, anchor=later / "tests" / "test_probe.py") is None
    # the marker committed LATER: the root does not carry it -- a development checkout's shape, failed by name
    dev = _repo(tmp_path / "dev", marker_in_root=False, commits=2)
    (dev / HISTORY_SQUASHED_MARKER).write_text(MARKER_CONTENT + "\n")
    with pytest.raises(NotPublished, match="does not carry"):
        publication_root(dev)
    with pytest.raises(pytest.fail.Exception, match="does not bear it out: the root commit .* does not carry"):
        pinned_commit_or_skip(SHA, "some-branch", "the probe", repo=dev, anchor=dev / "tests" / "test_probe.py")
    _git(dev, "add", HISTORY_SQUASHED_MARKER)
    assert _git(dev, "commit", "-q", "-m", "a marker committed later").returncode == 0
    with pytest.raises(pytest.fail.Exception, match="does not bear it out: the root commit .* does not carry"):
        pinned_commit_or_skip(SHA, "some-branch", "the probe", repo=dev, anchor=dev / "tests" / "test_probe.py")
    # no marker at all: the full-checkout failure, naming the branch (unchanged)
    plain = _repo(tmp_path / "plain", marker_in_root=False, commits=1)
    assert publication_root(plain) is None
    with pytest.raises(pytest.fail.Exception, match="run `git fetch origin some-branch`"):
        pinned_commit_or_skip(SHA, "some-branch", "the probe", repo=plain, anchor=plain / "tests" / "test_probe.py")
    # other content in the committed marker
    other = _repo(tmp_path / "other", marker_in_root=False, commits=1)
    (other / HISTORY_SQUASHED_MARKER).write_text("something else\n")
    _git(other, "add", HISTORY_SQUASHED_MARKER)
    assert _git(other, "commit", "-q", "--amend", "-m", "c0 with another marker").returncode == 0
    with pytest.raises(NotPublished, match="not `squashed`"):
        publication_root(other)


def test_a_depth_one_shallow_clone_of_a_publication_is_a_shallow_clone_never_a_publication(tmp_path):
    """A depth-one clone of a marked publication: HEAD's tree carries the marker file and ``rev-list --max-parents=0 HEAD`` answers HEAD
    itself (its parents were cut), so the root rule alone would read it as published -- the shallow clause refuses that, and the
    pinned-commit helper skips it as what it is, a shallow clone (the pre-existing exception), by the shallow sentence."""
    src = _repo(tmp_path / "src", marker_in_root=True, commits=3)
    clone = tmp_path / "clone"
    r = subprocess.run(["git", "clone", "-q", "--depth", "1", f"file://{src}", str(clone)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert _git(clone, "rev-parse", "--is-shallow-repository").stdout.strip() == "true" and (clone / HISTORY_SQUASHED_MARKER).is_file()
    assert _git(clone, "rev-list", "--max-parents=0", "HEAD").stdout.split() == [_git(clone, "rev-parse", "HEAD").stdout.strip()]
    with pytest.raises(NotPublished, match="the clone is shallow"):
        publication_root(clone)
    with pytest.raises(pytest.skip.Exception, match="shallow clone") as ei:
        pinned_commit_or_skip(SHA, "some-branch", "the probe", repo=clone, anchor=clone / "tests" / "test_probe.py")
    assert "publication root" not in str(ei.value)
