"""stager-fix-3 (two High): (1) the manifest publish is a compare-and-swap only under ONE writer lock per cluster that
spans the digest compare and the rename, and the manifest is read as one byte snapshot whose digest describes exactly the parsed bytes; (2)
every (target, staged_id) is guarded by a per-identity mkdir lock on each cluster, held from a LIVE re-check of the shared manifests through
the transfer to the publication — a peer mid-stage refuses by name, a peer that pinned the identity after the preflight is caught by the live
check, nothing is copied either way. Local hosts (`local:`) stand in for the clusters; synthetic records and bytes; placeholder ids."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from front_door_testlib import REPO  # noqa: E402
from _rsync_fallback import rsync_on_path  # noqa: E402,F401  (autouse: the fallback rsync on PATH where the host has none; a no-op where it has one)
from test_stage_objects_manifest_merge import KEY, REL, SO, _bank, _host, _ids, _read, _rec, _remote, _site, _source, _stage  # noqa: E402

SRC = REPO / "scripts/sdfbase/full22/stage_objects.py"


def test_the_manifest_is_read_as_one_byte_snapshot_and_its_digest_describes_those_bytes(tmp_path):
    h, shared = _host(tmp_path)
    (shared / REL).mkdir(parents=True)
    body = json.dumps([_rec("gemma4e4b", "base__x")], indent=1) + "\n\n   \n"                        # trailing whitespace: still the bytes on disk
    f = shared / REL / "staging_manifest.json"; f.write_bytes(body.encode())
    recs, digest = SO.remote_manifest_with_digest(h, REL)
    assert _ids(recs) == {("gemma4e4b", "base__x"): "a" * 64} and digest == hashlib.sha256(f.read_bytes()).hexdigest()
    assert digest == subprocess.run(["sha256sum", str(f)], capture_output=True, text=True).stdout.split()[0]   # the digest the CAS step compares against
    assert SO.remote_manifest_with_digest(h, "exp/objects/elsewhere") == ([], SO.ABSENT)
    f.write_text("{not json")
    with pytest.raises(RuntimeError, match="is not JSON"):
        SO.remote_manifest_with_digest(h, REL)
    src = SRC.read_text()
    assert 'base64 -w0 "$f"' in src and 'sha256sum "$f" | cut -d" " -f1; cat "$f"' not in src     # one open, no separate hash-then-cat


_PUBLISHER = textwrap.dedent("""
    import json, sys
    from pathlib import Path
    sys.path.insert(0, {tests!r}); sys.path.insert(0, {full22!r})
    import stage_objects as SO
    label, shared, bank = sys.argv[1], sys.argv[2], Path(sys.argv[3])
    h = SO.Host(label, "local:" + shared, shared, False)
    recs = SO.publish_union({{label: h}}, [label], {rel!r}, bank / "staging_manifest.json", bank / "_records", merge_remote=True)
    print("PUBLISHED", json.dumps(sorted(r["staged_id"] for r in recs)))
""")


def test_two_publishers_paused_between_check_and_rename_serialize_under_the_writer_lock_and_neither_record_is_lost(tmp_path):
    """Publisher A is held INSIDE its critical section (after the digest check, before the rename) by the test-only pause; publisher
    B, staging a disjoint cell, must wait on the writer lock — it cannot pass the digest test and rename over A. When A is released, B's
    check sees a moved digest, re-reads, folds A's record and publishes the union: both records survive on the shared manifest."""
    h, shared = _host(tmp_path)
    R0 = _rec("gemma4e4b", "base__x"); _remote(shared, [R0])
    bankA, recA, manA = _bank(tmp_path / "A", [_rec(KEY, "base__cellA", sha="1" * 64)])
    bankB, recB, manB = _bank(tmp_path / "B", [_rec(KEY, "base__cellB", sha="2" * 64)])
    code = _PUBLISHER.format(tests=str(REPO / "tests"), full22=str(REPO / "scripts/sdfbase/full22"), rel=REL)
    sentinel = tmp_path / "release_A"
    envA = {**os.environ, "SO_PUBLISH_PAUSE_UNTIL": str(sentinel)}; envB = {k: v for k, v in os.environ.items() if k != "SO_PUBLISH_PAUSE_UNTIL"}
    A = subprocess.Popen([sys.executable, "-c", code, "cluster_a", str(shared), str(bankA)], env=envA, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.time() + 60
    while time.time() < deadline and not list((shared / REL).glob(".staging_manifest.json.*.tmp")):   # A has rsynced its temp and is in the remote step
        time.sleep(0.1)
    time.sleep(0.5)                                                                                       # past A's digest check, into the pause
    assert A.poll() is None and _ids(_read(shared)) == {("gemma4e4b", "base__x"): "a" * 64}            # A holds the lock, paused; nothing published yet
    B = subprocess.Popen([sys.executable, "-c", code, "cluster_b_same_root", str(shared), str(bankB)], env=envB, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(2.0)
    assert B.poll() is None and _ids(_read(shared)) == {("gemma4e4b", "base__x"): "a" * 64}             # B is blocked on the writer lock: no second rename, A's digest check not bypassed
    sentinel.write_text("go")
    outA, errA = A.communicate(timeout=90); outB, errB = B.communicate(timeout=120)
    assert A.returncode == 0 and B.returncode == 0, (outA, errA, outB, errB)
    want = {("gemma4e4b", "base__x"): "a" * 64, (KEY, "base__cellA"): "1" * 64, (KEY, "base__cellB"): "2" * 64}
    assert _ids(_read(shared)) == want, _read(shared)                                                     # A's record was never erased; B folded it
    assert "changed since it was read" in outB and "PUBLISHED" in outA and "PUBLISHED" in outB, (outA, outB)
    assert not list((shared / REL).glob(".staging_manifest.json.*.tmp")) and (shared / REL / ".staging_manifest.lock").exists()
    src = SRC.read_text()
    assert "LOCK-TIMEOUT" in src and "refusing to publish without it" in src


def _lock_dir(shared, staged_id):
    return shared / REL / KEY / "cells" / SO.IDENTITY_LOCK_DIR / f"{staged_id}.lock"


def test_a_peer_holding_the_identity_lock_refuses_by_name_and_nothing_is_copied(tmp_path):
    """A peer mid-stage on this identity (its mkdir lock present on one cluster) — the stager refuses by name BEFORE any transfer,
    records identity_locked, leaves the shared manifest and its own locks clean."""
    site, roots = _site(tmp_path, ["cluster_a", "cluster_b"])
    base = tmp_path / "bank_run"; base.mkdir()
    d, sha = _source(tmp_path, "cellX", b"bytes-of-X")
    peer = _lock_dir(roots["cluster_b"], "derived__cellX"); peer.mkdir(parents=True)
    (peer / "owner.json").write_text(json.dumps({"host": "TESTHOST", "pid": 4242, "bank": "/tmp/peer_bank"}))
    rc, out = _stage(base, site, [{"cell_id": "cellX", "lane": "derived", "kind": "derived", "source": f"local:{d}"}])
    assert rc == 2 and "is being staged by another bank on cluster_b" in out and "TESTHOST" in out and "nothing copied" in out, out
    for lab, shared in roots.items():
        assert not (shared / REL / KEY / "cells" / "derived__cellX" / "object.pt").exists()             # no transfer on either cluster
        published = _read(shared) or []
        assert all((r["target"], r["staged_id"]) != (KEY, "derived__cellX") for r in published), published   # the refusal note never enters the shared union (it would conflict the peer's publish)
    assert not _lock_dir(roots["cluster_a"], "derived__cellX").exists() and peer.is_dir()               # our own cluster_a lock released; the peer's untouched
    rec = json.loads((base / "bank" / "_records" / f"{KEY}__derived__cellX.json").read_text())
    assert rec["status"] == "identity_locked" and rec["identity_lock"]["cluster"] == "cluster_b"       # the bank's own log keeps the refusal
    assert all(r.get("status") != "identity_locked" for r in json.loads((base / "bank" / "staging_manifest.json").read_text()))


def test_a_peer_pinning_the_identity_after_the_preflight_is_caught_by_the_live_check_and_nothing_is_copied(tmp_path, monkeypatch):
    """The shared manifest is clean at the preflight; a peer publishes a pin for this identity to OTHER bytes before the stager
    reaches the cell (the test-only post-preflight hook) — the live re-check under the identity lock refuses by name; nothing is copied; the
    peer's pin stands; every identity lock is released."""
    site, roots = _site(tmp_path, ["cluster_a", "cluster_b"])
    base = tmp_path / "bank_run"; base.mkdir()
    d, sha = _source(tmp_path, "cellY", b"bytes-of-Y")
    peer_rec = _rec(KEY, "derived__cellY", sha="9" * 64, cell_id="cellY", lane="derived")
    hook = tmp_path / "peer_publishes.sh"
    hook.write_text(f"#!/bin/bash\nset -e\nmkdir -p {roots['cluster_a'] / REL}\nprintf %s {json.dumps(json.dumps([peer_rec]))} > {roots['cluster_a'] / REL / 'staging_manifest.json'}\n")
    monkeypatch.setenv("SO_TEST_AFTER_PREFLIGHT_HOOK", str(hook))
    rc, out = _stage(base, site, [{"cell_id": "cellY", "lane": "derived", "kind": "derived", "source": f"local:{d}"}])
    assert rc == 2 and "pins this identity to another object" in out and "9" * 12 in out and "nothing copied" in out, out
    for shared in roots.values():
        assert not (shared / REL / KEY / "cells" / "derived__cellY" / "object.pt").exists()
        assert not _lock_dir(shared, "derived__cellY").exists()                                          # released on the refusal path
    assert _ids(_read(roots["cluster_a"])) == {(KEY, "derived__cellY"): "9" * 64}                        # the peer's pin stands, untouched
    rec = json.loads((base / "bank" / "_records" / f"{KEY}__derived__cellY.json").read_text())
    assert rec["status"] == "remote_manifest_conflict"


def test_a_clean_stage_holds_its_identity_locks_through_the_publish_and_releases_them(tmp_path):
    site, roots = _site(tmp_path, ["cluster_a", "cluster_b"])
    base = tmp_path / "bank_run"; base.mkdir()
    d, sha = _source(tmp_path, "cellZ", b"bytes-of-Z")
    rc, out = _stage(base, site, [{"cell_id": "cellZ", "lane": "derived", "kind": "derived", "source": f"local:{d}"}])
    assert rc == 0, out
    for shared in roots.values():
        assert (shared / REL / KEY / "cells" / "derived__cellZ" / "object.pt").read_bytes() == b"bytes-of-Z"
        assert _ids(_read(shared)) == {(KEY, "derived__cellZ"): sha}
        assert not _lock_dir(shared, "derived__cellZ").exists() and (shared / REL / KEY / "cells" / SO.IDENTITY_LOCK_DIR).is_dir()
        assert (shared / REL / ".staging_manifest.lock").exists()
    src = SRC.read_text()
    i_acq = src.index("d, who = acquire_identity_lock(hosts[lab], a.rel, a.target, staged_id, owner)"); i_live = src.index("clash = live_identity_conflicts(")
    i_stage = src.index("r = stage_to(hosts[lab], local_dir, rdir, sha, n)"); i_pub = src.index("recs = publish_union(hosts, labels, a.rel, manifest, records"); i_rel = src.index("release_identity_locks(held)                                   # the identity locks outlive")
    assert i_acq < i_live < i_stage < i_pub < i_rel                                                     # lock -> live check -> transfer -> publish -> release
