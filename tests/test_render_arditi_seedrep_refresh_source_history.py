"""``render_arditi_seedrep_configs --refresh-source`` on a clone that does not carry MIN_SOURCE_SHA (a published repository without its
development history, a clone that never fetched it): refused BY NAME before any ``rev-parse --verify`` can raise a CalledProcessError
traceback, naming the commit and the missing history. ``--check`` reads only the committed source copies and is untouched."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "seedrep"))
sys.path.insert(0, str(REPO / "scripts" / "sdfbase" / "full22" / "sampled"))
import render_arditi_seedrep_configs as ra  # noqa: E402


def test_refresh_source_refuses_by_name_when_the_clone_does_not_carry_the_minimum_source_commit(monkeypatch):
    monkeypatch.setattr(ra, "_has_commit", lambda sha: False)
    asked = []

    def no_git(*args):
        asked.append(args)
        raise AssertionError(f"git must not be asked after the presence probe refused: {args}")

    monkeypatch.setattr(ra, "_git", no_git)
    with pytest.raises(SystemExit) as ei:
        ra.refresh_source("deadbeef")
    msg = str(ei.value)
    assert f"--refresh-source needs commit {ra.MIN_SOURCE_SHA}, which this clone does not carry" in msg, msg
    assert "a published repository without its development history" in msg and "needs the full development history" in msg, msg
    assert "--check reads them" in msg and asked == [], msg                                  # no rev-parse, no traceback, nothing written


def test_the_presence_probe_reads_this_clones_object_store():
    assert ra._has_commit("0" * 40) is False
    head = subprocess.run(["git", "-C", str(ra.REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert ra._has_commit(head) is True
    # the probe is the FIRST thing refresh_source does, before any verify of the named sha or the minimum
    src = Path(ra.__file__).read_text()
    body = src[src.index("def refresh_source("):]
    body = body[:body.index("\ndef ", 1)]
    assert body.index("_has_commit(MIN_SOURCE_SHA)") < body.index('rev-parse", "--verify"'), body


def test_refresh_source_still_refuses_a_sha_that_does_not_descend_from_the_minimum_where_the_clone_carries_it(monkeypatch):
    """Where the history is present the descent check runs and refuses (the probe answered True)."""
    if not ra._has_commit(ra.MIN_SOURCE_SHA):
        pytest.skip(f"this clone does not carry {ra.MIN_SOURCE_SHA} (a published repository without its development history); the D18 descent "
                    "refusal is not exercised here")
    root = subprocess.run(["git", "-C", str(ra.REPO), "rev-list", "--max-parents=0", "HEAD"], capture_output=True, text=True, check=True).stdout.split()[0]
    with pytest.raises(SystemExit, match=f"does not descend from {ra.MIN_SOURCE_SHA}"):
        ra.refresh_source(root)
