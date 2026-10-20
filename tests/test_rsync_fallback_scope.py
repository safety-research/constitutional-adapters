"""The pure-Python ``rsync`` fallback (tests/_rsync_fallback.py) is scoped to the test that asked for it.

On a host without rsync a world builder's ``ensure_rsync_on_path()`` prepends the fallback's directory to the process-wide PATH;
the ``rsync_on_path`` fixture and the conftest belt (``_rsync_fallback_off_path_after_each_test``) take it off again when that test
ends, so a later test's own ``shutil.which("rsync")`` reads the host, never a shim it did not request. Exercised here on a PATH
with no rsync on it whatever the host has (one empty directory as the whole PATH), so these run the same on the gate, a developer's
machine and the CPU-suite container.
"""
from __future__ import annotations

import os
import shutil
import stat

import _rsync_fallback as RF


def _no_rsync_path(tmp_path) -> str:
    """A PATH on which ``rsync`` does not resolve: one empty directory."""
    d = tmp_path / "empty-bin"
    d.mkdir(exist_ok=True)
    return str(d)


def test_the_fallback_is_installed_only_where_rsync_is_absent_and_removed_exactly(tmp_path, monkeypatch):
    bare = _no_rsync_path(tmp_path)
    monkeypatch.setenv("PATH", bare)
    assert shutil.which("rsync") is None
    d = RF.ensure_rsync_on_path()
    assert d and d in RF.installed_dirs() and os.environ["PATH"] == d + os.pathsep + bare
    wrapper = os.path.join(d, "rsync")
    assert shutil.which("rsync") == wrapper and os.access(wrapper, os.X_OK)
    # idempotent: an rsync now resolves (the fallback), so nothing is installed twice
    assert RF.ensure_rsync_on_path() is None and os.environ["PATH"] == d + os.pathsep + bare
    assert RF.remove_fallback_from_path() is True
    assert os.environ["PATH"] == bare and shutil.which("rsync") is None
    assert RF.remove_fallback_from_path() is False                        # nothing left to remove
    # where an rsync resolves, nothing is installed and PATH is left byte-for-byte
    has = tmp_path / "has-rsync"
    has.mkdir()
    (has / "rsync").write_text("#!/bin/sh\nexit 0\n")
    (has / "rsync").chmod((has / "rsync").stat().st_mode | stat.S_IXUSR)
    monkeypatch.setenv("PATH", f"{has}{os.pathsep}{bare}")
    assert RF.ensure_rsync_on_path() is None and os.environ["PATH"] == f"{has}{os.pathsep}{bare}"
    assert RF.remove_fallback_from_path() is False
    # the executable a shim may exec by absolute path: the resolving rsync, else the wrapper, and PATH untouched either way
    assert RF.rsync_executable() == str(has / "rsync") and os.environ["PATH"] == f"{has}{os.pathsep}{bare}"
    monkeypatch.setenv("PATH", bare)
    assert RF.rsync_executable() == wrapper and os.environ["PATH"] == bare


def test_a_fallback_a_test_leaves_on_path_is_gone_before_the_next_test_part_one(tmp_path):
    """Leaves the fallback on PATH deliberately, the way a world builder's ``ensure_rsync_on_path()`` with no fixture around it did:
    the real PATH put back by hand UNDER the fallback, so no ``monkeypatch`` undoes it and only the conftest belt can."""
    saved = os.environ["PATH"]
    os.environ["PATH"] = _no_rsync_path(tmp_path)
    try:
        d = RF.ensure_rsync_on_path()
    finally:
        os.environ["PATH"] = saved
    assert d
    os.environ["PATH"] = d + os.pathsep + saved                           # the leak, on purpose
    assert shutil.which("rsync") == os.path.join(d, "rsync")


def test_a_fallback_a_test_leaves_on_path_is_gone_before_the_next_test_part_two():
    """Runs after part one (pytest keeps a file's order): no fallback directory is on PATH, none is left on disk, and ``rsync`` resolves
    to the host's own or not at all. Alone (``-k part_two``) there is nothing installed to check, and the test passes on that."""
    dirs = RF.installed_dirs()
    entries = os.environ["PATH"].split(os.pathsep)
    assert not any(e in dirs for e in entries), (dirs, entries)
    assert not any(os.path.lexists(d) for d in dirs), dirs                  # the belt removed the wrapper directory too, not only the PATH entry
    found = shutil.which("rsync")
    assert found is None or os.path.dirname(found) not in dirs, found


def test_teardown_removes_the_wrapper_directory_and_the_next_call_makes_a_fresh_one(tmp_path, monkeypatch):
    """The wrapper directory lives for the test that asked and no longer: ``teardown_fallback`` (the fixture's and the belt's one call)
    takes the fallback off PATH and removes the directory from disk, and a later ``ensure_rsync_on_path`` / ``rsync_executable`` makes a
    new one rather than handing out the removed path."""
    bare = _no_rsync_path(tmp_path)
    monkeypatch.setenv("PATH", bare)
    d = RF.ensure_rsync_on_path()
    assert d and os.path.isfile(os.path.join(d, "rsync"))
    assert RF.teardown_fallback() is True
    assert os.environ["PATH"] == bare and not os.path.lexists(d) and d in RF.installed_dirs()
    assert RF.teardown_fallback() is False                                  # nothing left on PATH or on disk
    d2 = RF.ensure_rsync_on_path()
    assert d2 and d2 != d and os.path.isfile(os.path.join(d2, "rsync")) and os.environ["PATH"] == d2 + os.pathsep + bare
    # the executable a shim holds by absolute path is a fresh wrapper too, never the removed one
    assert RF.teardown_fallback() is True
    exe = RF.rsync_executable()
    assert exe not in (os.path.join(d, "rsync"), os.path.join(d2, "rsync")) and os.path.isfile(exe) and os.environ["PATH"] == bare
