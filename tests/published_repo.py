"""The published repository's shape, for tests that read maintainer-only files.

The published repository is a squash of main without CLAUDE.md (the maintainers' instructions), and the publish step
commits a root marker, `.history-squashed` (content `squashed`; the name seedrep_history_fixtures.HISTORY_SQUASHED_MARKER
and sisteer.registry.snapshot.HISTORY_SQUASHED_MARKER share), in the publication's root commit. A test that pins
CLAUDE.md's text skips there, by that positive statement -- the marker present, not shallow, the one root commit carrying
it (seedrep_history_fixtures.publication_root), so a user's later commits on top of the publication change nothing -- and
nowhere else: without the marker, a missing CLAUDE.md still fails the test loudly, and a marker the repository does not
bear out (a stray file in a development checkout) fails by name.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from seedrep_history_fixtures import HISTORY_SQUASHED_MARKER, NotPublished, publication_root  # noqa: F401  (the name is re-exported)

REPO = Path(__file__).resolve().parents[1]
#: the repository paths the publication leaves out; reproduce/tools/scrub_scan.py reads this literal for its publication set
UNPUBLISHED = ("CLAUDE.md",)


def is_published(root: Path = REPO) -> bool:
    """Whether ``root`` is a repository published without its development history (``publication_root``'s whole rule); a marker the
    repository does not bear out FAILS the test by name rather than answering either way."""
    try:
        return publication_root(Path(root)) is not None
    except NotPublished as exc:
        pytest.fail(f"{HISTORY_SQUASHED_MARKER} is present at {root} but the repository does not bear it out: {exc} -- a development checkout "
                    "never carries the marker; remove it, or fix the publication")


def claude_md_text(root: Path = REPO) -> str:
    """CLAUDE.md's text; skips only in the published repository, where the file is deliberately absent."""
    path = Path(root) / "CLAUDE.md"
    if not path.is_file() and is_published(root):
        pytest.skip("CLAUDE.md holds the maintainers' instructions and is not part of the published repository")
    return path.read_text(encoding="utf-8")
