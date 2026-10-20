"""Shared history fixture for the seedrep tests that read a PINNED COMMIT out of the object store (an oracle script, the
producer's file, main's registry package): the exact-absence rule (delta 2).

A skip fires ONLY on the positively established ABSENCE of history:

* a SOURCE-ONLY tree -- no ``.git`` entry (directory, or a linked worktree's file) anywhere up from the calling test file AND
  ``git rev-parse --git-dir`` saying "not a git repository";
* a genuinely SHALLOW clone (``git rev-parse --is-shallow-repository`` == ``true``; the Modal CPU suite stages ``git init`` +
  ``fetch --depth 1`` WITH ``.git`` -- ``cpu_suite.shallow_clone_tarball`` / ``unpack_tree``) that lacks the commit;
* a PUBLISHED repository without its development history, which SAYS so: not shallow, the publish step's root marker
  (``HISTORY_SQUASHED_MARKER``, content ``squashed``) present at the root, and the repository's one root commit -- the publication
  root -- carrying that marker (:func:`publication_root`). Commits made on top of the publication are descendants of that root
  and change nothing; a marker the root commit does not carry (a development checkout with a stray marker file) FAILS.

Everything else FAILS, in Git's words: repository metadata present but Git refusing (rc 128 "detected dubious ownership" on a
checkout mounted under another owner, a broken config), the shallow-status command itself failing, or a FULL checkout that lacks
the commit (the old workstream objects are not ancestors of the lane's re-authored tip, so a single-branch clone lacks them without
being shallow) -- that last one naming the branch to fetch. ``git cat-file -e`` alone cannot carry the decision: it answers rc 128
for a missing object AND for an access failure, so repository access is established first.

Importable by every test under ``tests/`` (pytest's prepend import mode puts the directory on ``sys.path``), the way
``seedrep_audit_fixtures`` is."""
from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Callable, Optional

import pytest

NOT_A_REPOSITORY = "not a git repository"
#: the root marker the PUBLISH step writes beside a squash of the repository (content ``squashed``), committed in the publication's
#: root commit: the repository's own positive statement that it carries no development history. Never present in a development
#: checkout, a gate tree or the base checkout; tests that exercise it create it in tmp repos. ``sisteer.registry.snapshot.HISTORY_SQUASHED_MARKER``
#: is the same name.
HISTORY_SQUASHED_MARKER = ".history-squashed"
MARKER_CONTENT = "squashed"


class NotPublished(Exception):
    """The marker is present but the repository does not bear it out (why, in the message): never a published repository."""


def history_squashed_marker(repo: Path) -> Optional[Path]:
    """The publish step's root marker FILE under ``repo``; None where there is none (the marker is a file at the repository root,
    nothing else counts). Its presence alone decides nothing: :func:`publication_root` is the rule."""
    p = Path(repo) / HISTORY_SQUASHED_MARKER
    return p if p.is_file() else None


def publication_root(repo: Path, *, run: Callable = subprocess.run) -> Optional[str]:
    """The sha of the publication root when ``repo`` is a repository published without its development history, None when no marker
    file is present (a development checkout). The rule, every clause required: the marker file is present at the root with content
    ``squashed``; ``git rev-parse --is-shallow-repository`` succeeds and prints ``false``; HEAD has exactly ONE root commit; that root
    commit carries the marker with the same content. Commits on top of the publication are descendants of the root and pass.
    Raises :class:`NotPublished` naming the clause that fails when a marker is present but the repository does not bear it out --
    a shallow clone, a stray marker in a development checkout (the root commit does not carry it), a second root, other content --
    and a git failure is raised as :class:`NotPublished` too, never read as an absence."""
    marker = history_squashed_marker(repo)
    if marker is None:
        return None

    def git(*args):
        return run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)

    def said(r) -> str:
        return (r.stderr or r.stdout or "").strip()

    content = marker.read_text(encoding="utf-8").strip()
    if content != MARKER_CONTENT:
        raise NotPublished(f"{HISTORY_SQUASHED_MARKER} reads {content[:40]!r}, not `{MARKER_CONTENT}` -- not the publish step's marker")
    shallow = git("rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0:
        raise NotPublished(f"git rev-parse --is-shallow-repository failed (rc {shallow.returncode}): {said(shallow)[:300]} -- the clone's shape "
                           "cannot be established, so neither can a publication")
    if shallow.stdout.strip() != "false":
        raise NotPublished(f"the clone is shallow (--is-shallow-repository: {shallow.stdout.strip()!r}) -- a cut history, not a publication's absent one")
    roots = git("rev-list", "--max-parents=0", "HEAD")
    shas = roots.stdout.split()
    if roots.returncode != 0:
        raise NotPublished(f"git rev-list --max-parents=0 HEAD failed (rc {roots.returncode}): {said(roots)[:300]}")
    if len(shas) != 1:
        raise NotPublished(f"HEAD has {len(shas)} root commits, not the one publication root a published repository has")
    committed = git("cat-file", "-p", f"{shas[0]}:{HISTORY_SQUASHED_MARKER}")
    if committed.returncode != 0:
        raise NotPublished(f"the root commit {shas[0][:12]} does not carry {HISTORY_SQUASHED_MARKER} -- a development checkout with a stray marker "
                           "file is not a published repository")
    if committed.stdout.strip() != MARKER_CONTENT:
        raise NotPublished(f"the root commit {shas[0][:12]} carries {HISTORY_SQUASHED_MARKER} reading {committed.stdout.strip()[:40]!r}, not `{MARKER_CONTENT}`")
    return shas[0]


def git_metadata(anchor: Path) -> Optional[Path]:
    """The nearest ``.git`` entry -- a directory, or a linked worktree's FILE -- walking up from ``anchor``; ``None`` for a
    source-only tree (a tarball unpack that carries no ``.git``)."""
    start = Path(anchor).resolve()
    for d in [start, *start.parents]:
        if (d / ".git").exists():
            return d / ".git"
    return None


def pinned_commit_or_skip(commit: str, branch: str, what: str, *, repo: Path, anchor: Path,
                          run: Callable = subprocess.run) -> None:
    """Skip / fail / return for a check that needs ``commit`` out of ``repo``'s object store (module docstring). ``anchor`` is the
    calling test file (the ``.git`` walk starts there); ``run`` is ``subprocess.run`` (injected by this module's own tests)."""
    def git(*args):
        return run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)

    def said(r) -> str:
        return (r.stderr or r.stdout or "").strip()

    meta = git_metadata(anchor)
    gd = git("rev-parse", "--git-dir")
    if gd.returncode != 0:
        if meta is None and NOT_A_REPOSITORY in said(gd).lower():
            pytest.skip(f"not a git repository: no .git entry up from {anchor} and git agrees ({what} needs {commit[:8]} out of the object store)")
        pytest.fail(f"git rev-parse --git-dir failed (rc {gd.returncode}; repository metadata {'at ' + str(meta) if meta else 'absent'}): "
                    f"{said(gd)[:300]} -- a Git access/config failure, not an absence of history ({what} needs {commit[:8]}); fix it "
                    f"(e.g. `git config --global --add safe.directory {repo}`) rather than skipping")
    shallow = git("rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0:
        pytest.fail(f"git rev-parse --is-shallow-repository failed (rc {shallow.returncode}): {said(shallow)[:300]} -- the clone's shape "
                    f"cannot be established, so neither can an absence ({what} needs {commit[:8]})")
    if git("cat-file", "-e", f"{commit}^{{commit}}").returncode == 0:
        return
    if shallow.stdout.strip() == "true":
        pytest.skip(f"shallow clone: {commit[:8]} is not in this checkout ({what}; the Modal CPU suite stages fetch --depth 1)")
    # the third positively established absence: the repository SAYS it was published without its development history -- the publish
    # step's root marker, borne out by the publication root (publication_root's whole rule; later commits on top pass). Without
    # the marker a full checkout lacking the commit still FAILS, naming the branch to fetch; a marker the repository does not bear out FAILS.
    try:
        root = publication_root(repo, run=run)
    except NotPublished as exc:
        pytest.fail(f"{HISTORY_SQUASHED_MARKER} is present at {repo} but the repository does not bear it out: {exc} -- a development checkout "
                    f"never carries the marker; remove it, or fix the publication ({what} needs {commit[:8]}); a skip here would drop the "
                    "regression check silently")
    if root is not None:
        pytest.skip(f"the published repository carries no development history ({HISTORY_SQUASHED_MARKER} at its root, committed by its "
                    f"publication root {root[:8]}); this check reads commit {commit[:8]} and is unavailable here ({what})")
    pytest.fail(f"{commit[:8]} is not in this FULL checkout ({what}) -- branch {branch} carries it: run `git fetch origin {branch}` "
                "and re-run; a skip here would drop the regression check silently")
