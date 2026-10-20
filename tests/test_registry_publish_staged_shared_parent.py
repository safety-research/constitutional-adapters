"""``publish_staged`` leaves the adapter-wide staging parent STANDING.

Concurrent cluster-read shards of ONE adapter share the window root on the cluster's shared storage, so they share
``<root>/staging/adapters/<adapter>/``: each stages its own revision under it and renames the revision into
``records/``. The parent is a SHARED directory, and the tidy-up ``publish_staged`` used to run after the rename --
``if os.path.isdir(parent) and not os.listdir(parent): os.rmdir(parent)`` -- was a non-atomic check-then-act on it.
A landing window lost two ``objects_bank`` shards to exactly that: one shard's ``rmdir`` landed between another's
check and act, the ``OSError`` escaped ``_read_and_publish`` (which catches ``AdapterRefusal`` only), the shard wrote no
summary and the window lost its revisions. The same race has two more faces: a peer stages ITS revision between the
listing and the ``rmdir`` (``ENOTEMPTY``), and a peer's ``os.makedirs`` of ``<parent>/<rev>`` hits ``ENOENT`` when the
parent is removed under it between its exists-check and its ``mkdir``.

The fix deletes the tidy-up (a tolerant ``rmdir`` or a tidy at pass end would still remove the parent under a peer);
the empty parent is harmless because ``staging/`` is outside ``records/`` and ``rulings/``, which is all the harvester
scans, and nothing lists ``staging/adapters``. The tests here inject each peer action deterministically at the point
between the check and the act -- the tidy-up's own ``os.listdir`` of the parent -- and pin that ``publish_staged``
returns the published files, the revision stands in ``records/adapters/<adapter>/<rev>`` with its marker, nothing
raises, and the parent still exists afterwards. Every case FAILS on the code before the fix (e293e9ff1) and passes
after it; the injection-free case pins the parent's survival directly, so re-adding any tidy-up fails it.
"""
from __future__ import annotations

import os
from typing import Callable, List

import pytest

from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth
from sisteer.registry.adapters import runner as RUN

ADAPTER = "synthetic_store"
REV = "a" * 16
PEER_REV = "b" * 16
MARKER_FILE = R.record_file_name(1, "source_revision")


def _root(tmp_path) -> str:
    root = str(tmp_path / "root")
    H.ensure_layout(root)
    return root


def _stage_one_file_revision(root: str, rev: str, now: str) -> str:
    """A one-file revision under ``staging/adapters/<ADAPTER>/<rev>/``: the marker file alone (what a shard renames into records/)."""
    staged = RUN.staging_dir(root, ADAPTER, rev)
    marker = synth.rec("source_revision", {"revision_id": rev, "source_id": f"{ADAPTER}:s1", "adapter": ADAPTER, "observed_utc": now,
                                           "inputs_json": {"x": "1" * 64}, "accepted": 1}, now)
    R.write_records(os.path.join(staged, MARKER_FILE), [marker])
    return staged


def _out_dir(root: str, rev: str) -> str:
    return os.path.join(H.records_dir(root), "adapters", ADAPTER, rev)


def _assert_published(root: str, staged: str, rev: str, files: List[str]) -> None:
    """The revision stands in records/ with its marker, the staged directory moved, and the SHARED parent still exists."""
    out_dir = _out_dir(root, rev)
    assert files == [os.path.join(out_dir, MARKER_FILE)], files
    assert os.path.isdir(out_dir) and sorted(os.listdir(out_dir)) == [MARKER_FILE]
    assert RUN._is_marker_file(os.path.basename(files[0]))
    recs = [rec for _, rec, _ in R.read_record_file(files[0])]
    assert [r["record_type"] for r in recs] == ["source_revision"] and recs[0]["payload"]["revision_id"] == rev
    assert not os.path.exists(staged)                                  # the staged revision moved, whole
    parent = os.path.dirname(staged)
    assert os.path.isdir(parent), f"the shared staging parent was removed: {parent}"           # It stands for the life of the root
    assert parent == os.path.join(root, "staging", "adapters", ADAPTER)


def _peer_lands_at_the_check(mp: pytest.MonkeyPatch, parent: str, action: Callable[[], None], before_listing: bool) -> List[str]:
    """Patch the runner's ``os.listdir`` so a peer's ``action`` lands beside the FIRST listing of ``parent`` -- the tidy-up's check:
    ``before_listing`` puts it between the ``isdir`` and the ``listdir`` (the observed trace: the listing itself raises ENOENT), else between
    the listing and the ``rmdir`` (the listing said empty; the act then finds the parent gone, or no longer empty). Returns the list
    the spy appends to when it fires, so a test can tell whether the check ran at all (after the fix it never does)."""
    real = os.listdir
    fired: List[str] = []

    def spy(path=".", *a, **k):
        if not fired and isinstance(path, str) and path == parent:
            fired.append(path)
            if before_listing:
                action()
                return real(path, *a, **k)
            out = real(path, *a, **k)
            action()
            return out
        return real(path, *a, **k)
    mp.setattr(RUN.os, "listdir", spy)
    return fired


@pytest.mark.parametrize("face, before_listing", [
    ("peer_rmdir_between_isdir_and_listdir", True),      # observed: "[Errno 2] No such file or directory: <root>/staging/adapters/<adapter>"
    ("peer_rmdir_between_listdir_and_rmdir", False),     # ENOENT from the act itself
])
def test_publish_staged_survives_a_peer_shard_removing_the_shared_parent_between_the_check_and_the_act(tmp_path, face, before_listing):
    root = _root(tmp_path)
    staged = _stage_one_file_revision(root, REV, synth.t(1000))
    parent = os.path.dirname(staged)
    with pytest.MonkeyPatch.context() as mp:
        fired = _peer_lands_at_the_check(mp, parent, lambda: os.rmdir(parent), before_listing)
        files = RUN.publish_staged(staged, _out_dir(root, REV))          # used to escape as OSError: the shard died with no summary
    _assert_published(root, staged, REV, files)
    assert fired == [], f"publish_staged still runs the check-then-act on the shared parent ({face})"


def test_publish_staged_survives_a_peer_shard_staging_its_revision_between_the_check_and_the_act(tmp_path):
    """The ENOTEMPTY face: the listing says empty, a peer creates ``<parent>/<its rev>`` (its own ``write_records`` ``makedirs``), the
    ``rmdir`` then refuses. Afterwards the peer's staged revision must stand too: it is the peer's to publish."""
    root = _root(tmp_path)
    staged = _stage_one_file_revision(root, REV, synth.t(1000))
    parent = os.path.dirname(staged)
    peer_dir = os.path.join(parent, PEER_REV)
    with pytest.MonkeyPatch.context() as mp:
        fired = _peer_lands_at_the_check(mp, parent, lambda: os.makedirs(peer_dir), before_listing=False)
        files = RUN.publish_staged(staged, _out_dir(root, REV))          # used to escape as OSError(ENOTEMPTY)
    _assert_published(root, staged, REV, files)
    if fired:                                                            # had the check run, the peer's revision in flight is still its own
        assert os.path.isdir(peer_dir)
    assert fired == [], "publish_staged still runs the check-then-act on the shared parent"


def test_publish_staged_leaves_the_shared_parent_standing_with_no_peer_at_all(tmp_path):
    """Injection-free: the parent stands after a first publish, after the no-op re-publish of identical content (the ``rmtree`` branch),
    and after a second shard's revision -- for the life of the root. Re-adding ANY tidy-up (tolerant or not) fails this."""
    root = _root(tmp_path)
    staged = _stage_one_file_revision(root, REV, synth.t(1000))
    parent = os.path.dirname(staged)
    files = RUN.publish_staged(staged, _out_dir(root, REV))
    _assert_published(root, staged, REV, files)
    assert os.listdir(parent) == []                                      # empty, and standing
    # a re-run of the same deterministic revision: identical content is left alone, the staged copy dropped, the parent stands
    staged_again = _stage_one_file_revision(root, REV, synth.t(1000))
    assert RUN.publish_staged(staged_again, _out_dir(root, REV)) == files
    _assert_published(root, staged_again, REV, files)
    # a second revision (another shard's group) under the same parent
    staged2 = _stage_one_file_revision(root, PEER_REV, synth.t(2000))
    files2 = RUN.publish_staged(staged2, _out_dir(root, PEER_REV))
    _assert_published(root, staged2, PEER_REV, files2)
    assert sorted(os.listdir(os.path.join(H.records_dir(root), "adapters", ADAPTER))) == [REV, PEER_REV]
    assert os.listdir(parent) == []
    # the harvester's namespace is records/ and rulings/: the standing parent is outside both
    assert not parent.startswith(H.records_dir(root) + os.sep) and not parent.startswith(os.path.join(root, "rulings") + os.sep)


def test_publish_staged_still_refuses_different_content_under_one_revision_id(tmp_path):
    """The loud refusal the tidy-up sat beside is unchanged: the same revision id with different bytes is a bug, and the staged copy is
    left in place for the operator (nothing of it reaches records/)."""
    root = _root(tmp_path)
    staged = _stage_one_file_revision(root, REV, synth.t(1000))
    RUN.publish_staged(staged, _out_dir(root, REV))
    other = _stage_one_file_revision(root, REV, synth.t(1001))         # a different observed_utc: different bytes, same id
    with pytest.raises(RuntimeError, match="already published with different content"):
        RUN.publish_staged(other, _out_dir(root, REV))
    assert os.path.isdir(other) and os.path.isdir(os.path.dirname(other))
    assert sorted(os.listdir(_out_dir(root, REV))) == [MARKER_FILE]
