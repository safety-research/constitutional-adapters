"""stager-fix-4: the stager's refusals stay recoverable.

* (1) a refusal NOTE this bank wrote (`identity_locked` — a peer held the identity) carries no object_sha256 and is not an object: once the peer
  has published that identity (the same bytes), the rerun on the same bank succeeds instead of refusing "two objects claim one pin" against a
  MISSING local sha — the preflight and the shared union read object records only (one filter: object_records);
* (2) a transient remote failure AFTER the identity locks were acquired (the live manifest read raising, the test-only hook) releases the
  locks on both clusters — nothing copied, no lock directory left behind — and an immediate retry stages and publishes;
* a `remote_manifest_conflict` note is kept out of the shared union, and at the publish the peer's pin for that identity stays a CONFLICT
  (both shas shown) rather than a fold the bank never verified; the function shape is pinned: the loop from acquisition through
  the publish sits under one try/finally whose finally releases the locks; acquire_identity_lock states what the locks rely on and the
  stale-lock policy (no automatic breaking; the operator confirms the owner is gone).

Local clusters under tmp (the stager's own Host over a local root); synthetic hosts, pids and digests."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)
from test_stage_objects_locks import _lock_dir  # noqa: E402
from test_stage_objects_manifest_merge import KEY, REL, SO, _ids, _read, _rec, _remote, _site, _source, _stage  # noqa: E402

SRC = REPO / "scripts/sdfbase/full22/stage_objects.py"


def test_after_an_identity_locked_refusal_the_peers_publish_of_the_same_digest_lets_the_rerun_succeed(tmp_path):
    site, roots = _site(tmp_path, ["cluster_a", "cluster_b"])
    base = tmp_path / "bank_run"; base.mkdir()
    d, sha = _source(tmp_path, "cellX", b"bytes-of-X")
    cells = [{"cell_id": "cellX", "lane": "derived", "kind": "derived", "source": f"local:{d}"}]
    peer = _lock_dir(roots["cluster_b"], "derived__cellX"); peer.mkdir(parents=True)
    (peer / "owner.json").write_text(json.dumps({"host": "TESTHOST", "pid": 4242, "utc": "2026-10-03T00:00:00Z", "bank": "/tmp/peer_bank", "target": KEY}))
    rc, out = _stage(base, site, cells)
    assert rc == 2 and "is being staged by another bank on cluster_b" in out, out
    note = json.loads((base / "bank" / "_records" / f"{KEY}__derived__cellX.json").read_text())
    assert note["status"] == "identity_locked" and "object_sha256" not in note and note["object_sha256_found"] == sha
    # the peer finishes: its lock goes, its record (the SAME digest) sits in both shared manifests
    (peer / "owner.json").unlink(); peer.rmdir()
    peer_rec = _rec(KEY, "derived__cellX", sha=sha, cell_id="cellX", lane="derived")
    for shared in roots.values():
        _remote(shared, [peer_rec])
    # the rerun on the same bank (the note still in _records): the preflight reads object records only, so the note is NO conflict — without
    # --merge-remote the peer's record is a foreign record (a push would erase it), with it the fold succeeds: staged + published
    rc, out = _stage(base, site, cells)
    assert rc == 2 and "two objects claim one pin" not in out and f"this bank lacks ({KEY}/derived__cellX)" in out and "re-run with --merge-remote" in out, out
    rc, out = _stage(base, site, cells, "--merge-remote")
    assert rc == 0 and "two objects claim one pin" not in out and "REFUSED" not in out, out
    rec = json.loads((base / "bank" / "_records" / f"{KEY}__derived__cellX.json").read_text())
    assert rec["status"] == "staged" and rec["object_sha256"] == sha
    for shared in roots.values():
        assert (shared / REL / KEY / "cells" / "derived__cellX" / "object.pt").read_bytes() == b"bytes-of-X"
        assert _ids(_read(shared)) == {(KEY, "derived__cellX"): sha} and not _lock_dir(shared, "derived__cellX").exists()
    assert all(r.get("object_sha256") for r in json.loads((base / "bank" / "staging_manifest.json").read_text()))


def test_a_transient_remote_failure_after_acquisition_releases_the_identity_locks_and_the_immediate_retry_succeeds(tmp_path, monkeypatch):
    site, roots = _site(tmp_path, ["cluster_a", "cluster_b"])
    base = tmp_path / "bank_run"; base.mkdir()
    d, sha = _source(tmp_path, "cellW", b"bytes-of-W")
    cells = [{"cell_id": "cellW", "lane": "derived", "kind": "derived", "source": f"local:{d}"}]
    marker = tmp_path / "failed_once"
    monkeypatch.setenv("SO_TEST_FAIL_LIVE_CHECK_ONCE", str(marker))                  # the live read raises once, AFTER the locks were taken
    rc, out = _stage(base, site, cells)
    assert rc != 0 and "simulated transient ssh failure after the identity locks were acquired" in out, out
    assert marker.exists()
    for shared in roots.values():
        assert not _lock_dir(shared, "derived__cellW").exists(), f"lock left behind on {shared}"      # released by the finally
        assert not (shared / REL / KEY / "cells" / "derived__cellW" / "object.pt").exists()          # nothing copied
    # the immediate retry (the marker exists: no failure): stages and publishes; no "being staged by another bank"
    rc, out = _stage(base, site, cells)
    assert rc == 0 and "is being staged by another bank" not in out, out
    for shared in roots.values():
        assert (shared / REL / KEY / "cells" / "derived__cellW" / "object.pt").read_bytes() == b"bytes-of-W"
        assert _ids(_read(shared)) == {(KEY, "derived__cellW"): sha} and not _lock_dir(shared, "derived__cellW").exists()


def test_a_remote_conflict_note_never_enters_the_shared_union_and_the_shape_is_pinned(tmp_path, monkeypatch):
    site, roots = _site(tmp_path, ["cluster_a", "cluster_b"])
    base = tmp_path / "bank_run"; base.mkdir()
    d, sha = _source(tmp_path, "cellV", b"bytes-of-V")
    peer_rec = _rec(KEY, "derived__cellV", sha="9" * 64, cell_id="cellV", lane="derived")
    hook = tmp_path / "peer_publishes.sh"
    hook.write_text(f"#!/bin/bash\nset -e\nmkdir -p {roots['cluster_a'] / REL}\nprintf %s {json.dumps(json.dumps([peer_rec]))} > {roots['cluster_a'] / REL / 'staging_manifest.json'}\n")
    monkeypatch.setenv("SO_TEST_AFTER_PREFLIGHT_HOOK", str(hook))
    rc, out = _stage(base, site, [{"cell_id": "cellV", "lane": "derived", "kind": "derived", "source": f"local:{d}"}])
    assert rc == 2 and "pins this identity to another object" in out, out
    note = json.loads((base / "bank" / "_records" / f"{KEY}__derived__cellV.json").read_text())
    assert note["status"] == "remote_manifest_conflict" and "object_sha256" not in note
    assert all(r.get("object_sha256") for r in json.loads((base / "bank" / "staging_manifest.json").read_text()))   # the note is not in the union
    # the publish: the peer's pin for the identity this bank REFUSED stays a conflict (both shas), never a fold — nothing published anywhere,
    # the note untouched (with --merge-remote the plain foreign fold must not adopt the refused pin either)
    rc2, out2 = _stage(base, site, [{"cell_id": "cellV", "lane": "derived", "kind": "derived", "source": f"local:{d}"}], "--merge-remote")
    assert rc2 == 2 and "MANIFEST REFUSED" in out2 and "two objects claim one pin" in out2 and f"local object_sha256 {sha} vs cluster_a {'9' * 64}" in out2, out2
    note2 = json.loads((base / "bank" / "_records" / f"{KEY}__derived__cellV.json").read_text())
    assert note2["status"] == "remote_manifest_conflict" and note2["object_sha256_found"] == sha                      # the note was not overwritten by the peer's record
    assert _ids(_read(roots["cluster_a"])) == {(KEY, "derived__cellV"): "9" * 64} and _read(roots["cluster_b"]) is None   # the peer's pin stands; nothing published
    for shared in roots.values():
        assert not _lock_dir(shared, "derived__cellV").exists()
    # the shape: object_records feeds the preflight and the union; acquisition -> live read -> transfer -> publish sit under one try/finally
    src = SRC.read_text()
    assert "remote0 = preflight_remote_manifests(hosts, labels, a.rel, object_records(records)" in src and "recs = object_records(records)" in src
    i_held = src.index("    held: List[Tuple[Host, str]] = []"); i_try = src.index("    try:\n        owner = {")   # the try opens right after `held` (the owner block and the loop inside it)
    i_acq = src.index("acquire_identity_lock(hosts[lab], a.rel, a.target, staged_id, owner)"); i_live = src.index("clash = live_identity_conflicts(")
    i_stage = src.index("r = stage_to(hosts[lab], local_dir, rdir, sha, n)"); i_pub = src.index("recs = publish_union(hosts, labels, a.rel, manifest, records")
    i_fin = src.index("    finally:\n        release_identity_locks(held)")
    assert i_held < i_try < i_acq < i_live < i_stage < i_pub < i_fin
    assert src.count("release_identity_locks(held)") == 1                                                            # the one release, in the finally
    doc = src[src.index("def acquire_identity_lock("):src.index("def release_identity_locks(")]
    assert "Stale-lock policy" in doc and "NOT broken automatically" in doc and "MooseFS's flock guarantees are not established" in doc and "--break-stale-locks" in doc
