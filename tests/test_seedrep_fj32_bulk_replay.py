"""fj32 (pre-launch, blocks certify): the clause-(e) ladder replay is BULK -- O(1) volume operations in the queue's size.

Known-bad at d19583ff0: `replay_ladder_queue` moved the re-keyed queue ONE FILE AT A TIME through the Volume API (per record one `read_file`;
per target one listdir for `exists` + one `batch_upload(force=True)` context = one upload + one commit) at ~5 s a record: ~30 min for
pasttense's 200-item queue, 8-15 h for controls' 5,872. The fix reads the prior queue in ONE bulk read, re-keys locally with the SAME bytes and
paths, lists this wave's ladder root ONCE, ships every new file in ONE upload + ONE commit, reads the committed files back in ONE bulk read and
closes from those. Every case here runs on the real `seedrep_ladder` protocol over its in-memory volume doubles (a `CountingVolume` tallying
the calls), a fake `modal.Volume` client for the RemoteVolume adapter (listdir / read_file / batch_upload counted; the bulk pull injected as a
directory copy), the synthetic two-tree world of the fj2 / fj3 tests -- no Modal, no real RS, no credentials, nothing under /tmp/lane-final2.
"""
from __future__ import annotations

import copy
import io
import json
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO / "src", REPO / "scripts", REPO / "scripts" / "modal", REPO / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import _seedrep_testlib as T  # noqa: E402
import seedrep_carry_over as co  # noqa: E402
import test_seedrep_fj2_carry_over as F  # noqa: E402
import test_seedrep_fj3_carry_over as J  # noqa: E402  -- the synthetic prior wave (a real Producer over the fake volume), the two plans

L = T.load_ladder()
W_PREV, W_THIS, WAVE, WAVE_PREV, AID = J.W_PREV, J.W_THIS, J.WAVE, J.WAVE_PREV, J.AID
PRE_FJ32 = "d19583ff0d03a97d410798fad149220ee49e8f12"          # the launch tree: the per-file implementation this one must match byte for byte
BULK_KINDS = ("listdir", "read_bulk", "upload", "upload_bulk", "commit")


# --------------------------------------------------------------------------- the world: N at-cap records enqueued by the real protocol

def _world(tmp_path, n):
    prev, this = J._plans()
    shared = L.SharedVolume()
    accepted_prev = L.MemoryKV(prev["dicts"]["accepted"])
    spec_prev, root_rel, recs = J._prior_wave(shared, accepted_prev, n=n)
    prior = co.load_prior_source(F._prior_dir(tmp_path, prev, name=f"prior{n}"))
    this_spec = L.QueueSpec.build(W_THIS[:7], WAVE, "pasttense", "s1", "static", 8192, ["0"])
    return types.SimpleNamespace(prev=prev, this=this, shared=shared, accepted_prev=accepted_prev, accepted_this=L.MemoryKV(this["dicts"]["accepted"]),
                                 spec_prev=spec_prev, root_rel=root_rel, recs=recs, prior=prior, prev_row=prev["standups"][0], this_spec=this_spec, n=n)


def _replay(w, vol, **kw):
    return co.replay_ladder_queue(L=L, vol=vol, accepted_kv=w.accepted_this, prev_row=w.prev_row, prior=w.prior, this_plan=w.this, aid=AID,
                                  prev_root=w.root_rel, expected_n=w.n, prev_wave_uuid=WAVE_PREV, **kw)


def _expected_rekey(w):
    """The re-key as the per-file implementation spelled it: the two fields, canonical JSON + newline, the attempt-unique path under THIS root."""
    out = {}
    for r in w.recs:
        r2 = dict(r)
        r2["W7"], r2["wave_uuid"] = w.this_spec.W7, w.this_spec.wave_uuid
        out[L.queue_entry_rel(w.this_spec.ladder_root, r2["arm_s"], r2["item_key"], AID)] = (L.canonical_json(r2) + "\n").encode("utf-8")
    return out


def _under(files, root):
    return {k: v for k, v in files.items() if k.startswith(root + "/")}


def _consumer_sees(w, n):
    view = w.shared.read_only_view()
    assert L.read_closed(view, w.this_spec)["n_items"] == n
    acc = L.accepted_producer_attempts(w.accepted_this, w.this_spec)
    assert acc == {"0": AID}
    entries = L.accepted_queue_entries(view, None, w.this_spec, acc)
    assert len(entries) == n and all((r["W7"], r["wave_uuid"]) == (W_THIS[:7], WAVE) for r in entries.values())


# --------------------------------------------------------------------------- (1) O(1) volume operations, counted

class TestBulkReplayIsO1:
    def test_n50_takes_at_most_three_calls_per_bulk_kind_and_the_same_number_as_n5(self, tmp_path):
        counts = {}
        for n in (5, 50):
            w = _world(tmp_path, n)
            cv = L.CountingVolume(w.shared.read_only_view())
            rp = _replay(w, cv)
            assert rp["ok"] and rp["closed"] and rp["queue_closed"] and rp["accepted"] == "seeded"
            assert (rp["n_records"], rp["n_written"], rp["n_already_present"]) == (n, n, 0)
            assert rp["counts"] == {"read": n, "rekeyed": n, "skipped_identical": 0, "uploaded": n + 2, "uploaded_queue_files": n, "uploaded_carried_records": 2}
            assert rp["volume_ops"]["ops"] == cv.ops and rp["volume_ops"]["files_uploaded"] == cv.files_uploaded == n + 2 + 2   # + marker + CLOSED.json
            for kind in BULK_KINDS:
                assert cv.ops[kind] <= 3, (kind, cv.ops)
            assert cv.ops["upload_bulk"] == 1 and cv.ops["read_bulk"] == 2 and cv.ops["listdir"] == 3           # prior queue, this root, the closure's
            _consumer_sees(w, n)
            counts[n] = dict(cv.ops)
        assert counts[5] == counts[50], counts          # EVERY kind, point lookups and single reads included: O(1) in the queue's size

    def test_the_rerun_uploads_and_commits_nothing_and_stays_o1(self, tmp_path):
        counts = {}
        for n in (5, 50):
            w = _world(tmp_path, n)
            _replay(w, w.shared.read_only_view())
            before = dict(w.shared.files)
            cv = L.CountingVolume(w.shared.read_only_view())
            rp = _replay(w, cv)
            assert rp["ok"] and rp["queue_closed"] and rp["accepted"] == "present"
            assert (rp["n_written"], rp["n_already_present"], rp["counts"]["uploaded"], rp["counts"]["skipped_identical"]) == (0, n, 0, n)
            assert cv.ops["upload_bulk"] == 0 and cv.ops["upload"] == 0 and cv.ops["commit"] == 0 and cv.ops["read_bulk"] == 2 and cv.ops["listdir"] == 3
            assert w.shared.files == before
            counts[n] = dict(cv.ops)
        assert counts[5] == counts[50], counts

    def test_a_volume_without_the_bulk_verbs_is_still_correct_and_counted_honestly(self, tmp_path):
        w = _world(tmp_path, 7)

        class _PerFile:
            def __init__(self, inner):
                self._i = inner
            read_bytes = property(lambda self: self._i.read_bytes)
            write_bytes = property(lambda self: self._i.write_bytes)
            exists = property(lambda self: self._i.exists)
            listdir = property(lambda self: self._i.listdir)
            walk = property(lambda self: self._i.walk)
            commit = property(lambda self: self._i.commit)
            reload = property(lambda self: self._i.reload)
        cv = L.CountingVolume(_PerFile(w.shared.read_only_view()))
        rp = _replay(w, cv)
        assert rp["ok"] and rp["queue_closed"] and rp["counts"]["uploaded"] == 9
        assert cv.ops["read_bulk"] == 0 and cv.ops["upload_bulk"] == 0 and cv.ops["read"] >= 2 * 7 and cv.ops["upload"] >= 9
        _consumer_sees(w, 7)


# --------------------------------------------------------------------------- (2) identical bytes skip, different bytes refuse, a torn upload completes on re-run

class _TornUploadView(L.SnapshotVolume):
    """A view whose ONE bulk upload lands (and commits) the first ``k`` files, then fails -- the worst case of a midway failure."""

    def __init__(self, shared, k):
        super().__init__(shared, "torn", autoreload=True)
        self.k = k

    def write_many(self, items):
        for i, (rel, data) in enumerate(items.items()):
            if i == self.k:
                self.commit()
                raise RuntimeError("upload torn midway")
            self.write_bytes(rel, data)


class TestIdempotence:
    def test_identical_bytes_already_present_are_skipped_and_only_the_rest_uploaded(self, tmp_path):
        w = _world(tmp_path, 50)
        expected = _expected_rekey(w)
        for rel in sorted(expected)[:10]:
            w.shared.files[rel] = expected[rel]
        cv = L.CountingVolume(w.shared.read_only_view())
        rp = _replay(w, cv)
        assert rp["ok"] and rp["queue_closed"] and (rp["n_written"], rp["n_already_present"]) == (40, 10)
        assert rp["counts"] == {"read": 50, "rekeyed": 50, "skipped_identical": 10, "uploaded": 42, "uploaded_queue_files": 40, "uploaded_carried_records": 2}
        assert cv.ops["read_bulk"] == 3 and cv.ops["upload_bulk"] == 1 and cv.ops["listdir"] == 3        # prior, the present ten, the read-back
        assert _under(w.shared.files, f"{w.this_spec.ladder_root}/queue") == expected
        _consumer_sees(w, 50)

    def test_a_target_with_different_bytes_refuses_before_any_write(self, tmp_path):
        w = _world(tmp_path, 50)
        victim = sorted(_expected_rekey(w))[7]
        w.shared.files[victim] = b'{"other": true}\n'
        cv = L.CountingVolume(w.shared.read_only_view())
        with pytest.raises(ValueError, match="DIFFERENT bytes"):
            _replay(w, cv)
        assert cv.ops["upload_bulk"] == 0 and cv.ops["upload"] == 0 and cv.ops["commit"] == 0
        assert _under(w.shared.files, w.this_spec.ladder_root) == {victim: b'{"other": true}\n'}
        assert w.accepted_this.get(L.key_str(w.this_spec.producers[0].key)) is None

    def test_a_torn_bulk_upload_leaves_a_partial_target_that_the_rerun_completes(self, tmp_path):
        w = _world(tmp_path, 50)
        expected = _expected_rekey(w)
        with pytest.raises(RuntimeError, match="torn midway"):
            _replay(w, _TornUploadView(w.shared, 20))
        landed = _under(w.shared.files, w.this_spec.ladder_root)
        assert len(landed) == 20 and all(expected[k] == v for k, v in landed.items())           # queue files only: no inventory, stub, marker, CLOSED
        assert w.accepted_this.get(L.key_str(w.this_spec.producers[0].key)) is None                # the acceptance follows the upload; none happened
        cv = L.CountingVolume(w.shared.read_only_view())
        rp = _replay(w, cv)
        assert rp["ok"] and rp["queue_closed"] and rp["accepted"] == "seeded" and (rp["n_written"], rp["n_already_present"]) == (30, 20)
        assert rp["counts"]["uploaded"] == 32 and cv.ops["upload_bulk"] == 1 and cv.ops["read_bulk"] == 3
        assert _under(w.shared.files, f"{w.this_spec.ladder_root}/queue") == expected
        _consumer_sees(w, 50)
        rp2 = _replay(w, w.shared.read_only_view())
        assert rp2["accepted"] == "present" and rp2["n_already_present"] == 50 and rp2["counts"]["uploaded"] == 0

    def test_the_read_back_refuses_when_the_committed_bytes_are_not_the_upload(self, tmp_path):
        w = _world(tmp_path, 5)

        class _Lossy(L.SnapshotVolume):
            def write_many(self, items):
                for i, (rel, data) in enumerate(items.items()):
                    self.write_bytes(rel, data if i != 2 else data[:-3] + b'x"}\n')
        with pytest.raises(ValueError, match="read-back after the upload"):
            _replay(w, _Lossy(w.shared, "lossy", autoreload=True))
        root = w.this_spec.ladder_root
        assert not any(k.startswith(f"{root}/producers") or k == L.closed_rel(root) for k in w.shared.files)   # no marker, no CLOSED.json under THIS root
        assert w.accepted_this.get(L.key_str(w.this_spec.producers[0].key)) is None


# --------------------------------------------------------------------------- (3) byte identity with the per-file implementation

def _sha_exists(sha):
    return subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{sha}^{{commit}}"], capture_output=True).returncode == 0


def _pre_fj32_module():
    src = subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_FJ32}:scripts/modal/seedrep_carry_over.py"], capture_output=True, text=True, check=True).stdout
    mod = types.ModuleType("seedrep_carry_over_pre_fj32")
    mod.__file__ = str(REPO / "scripts" / "modal" / "seedrep_carry_over.py")
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


def _strip_times(b):
    d = json.loads(b.decode("utf-8"))
    return {k: v for k, v in d.items() if k not in ("written_utc", "closed_utc", "replayed_utc")}


class TestByteIdentity:
    def test_rekeyed_bytes_and_paths_equal_the_local_rekey_and_the_d148_digest_set(self, tmp_path):
        w = _world(tmp_path, 50)
        _replay(w, w.shared.read_only_view())
        expected = _expected_rekey(w)
        got = _under(w.shared.files, f"{w.this_spec.ladder_root}/queue")
        assert got == expected                                                           # every path, byte for byte
        assert {L.sha256_hex(b) for b in got.values()} == {L.sha256_hex(b) for b in expected.values()} and len(got) == 50
        inv2 = json.loads(w.shared.files[L.inventory_rel(co.carried_producer_root(w.this_spec.ladder_root, "0", AID))])
        assert {e["path"]: e["sha256"] for e in inv2["entries"]} == {k: L.sha256_hex(v) for k, v in expected.items()}

    @pytest.mark.skipif(not _sha_exists(PRE_FJ32), reason="the launch tree's sha is not in this clone")
    def test_the_bulk_replay_writes_what_the_per_file_implementation_wrote_on_the_same_input(self, tmp_path):
        w = _world(tmp_path, 50)
        old = _pre_fj32_module()
        assert "read_many" not in subprocess.run(["git", "-C", str(REPO), "show", f"{PRE_FJ32}:scripts/modal/seedrep_carry_over.py"],
                                                 capture_output=True, text=True, check=True).stdout    # it IS the per-file form
        # two copies of ONE prior world; the old implementation on A, the new on B
        a, b = L.SharedVolume(), L.SharedVolume()
        a.files, b.files = dict(w.shared.files), dict(w.shared.files)
        acc_a, acc_b = L.MemoryKV("acc-a"), L.MemoryKV("acc-b")
        kw = dict(L=L, prev_row=w.prev_row, prior=w.prior, this_plan=w.this, aid=AID, prev_root=w.root_rel, expected_n=50, prev_wave_uuid=WAVE_PREV, utc="2026-09-20T00:00:00Z")
        rp_old = old.replay_ladder_queue(vol=a.read_only_view(), accepted_kv=acc_a, **kw)
        rp_new = co.replay_ladder_queue(vol=b.read_only_view(), accepted_kv=acc_b, **kw)
        root = w.this_spec.ladder_root
        fa, fb = _under(a.files, root), _under(b.files, root)
        assert set(fa) == set(fb) and len(fa) == 50 + 2 + 1 + 1                          # queue files, inventory + stub, marker, CLOSED.json
        timed = {L.producer_marker_rel(root, "0", AID), L.closed_rel(root)}
        assert {k: v for k, v in fa.items() if k not in timed} == {k: v for k, v in fb.items() if k not in timed}     # byte for byte
        assert all(_strip_times(fa[k]) == _strip_times(fb[k]) for k in timed)
        assert acc_a.get(L.key_str(w.this_spec.producers[0].key)) == acc_b.get(L.key_str(w.this_spec.producers[0].key)) == AID
        assert {k: v for k, v in rp_new.items() if k not in ("counts", "volume_ops")} == rp_old      # the report, but for the two new fields
        assert rp_new["schema"] == rp_old["schema"] == co.REPLAY_SCHEMA
        # the prior queue is untouched by either
        assert _under(a.files, w.spec_prev.ladder_root) == _under(b.files, w.spec_prev.ladder_root) == _under(w.shared.files, w.spec_prev.ladder_root)


# --------------------------------------------------------------------------- (4) the RemoteVolume adapter: one pull, one batch_upload, no per-file API call

class _FakeModalVolume:
    """A modal.Volume handle as the adapter calls it (listdir / read_file / batch_upload), over one directory, every call counted."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.listdir_calls, self.read_file_calls, self.batch_contexts, self.put_files, self.pulls = 0, 0, 0, 0, 0

    def listdir(self, path, *, recursive=False):
        self.listdir_calls += 1
        p = self.root / path
        if not p.is_dir():
            raise FileNotFoundError(path)
        it = p.rglob("*") if recursive else p.iterdir()
        return [types.SimpleNamespace(path="/" + q.relative_to(self.root).as_posix(), type=types.SimpleNamespace(name="DIRECTORY" if q.is_dir() else "FILE"))
                for q in sorted(it)]

    def read_file(self, path):
        self.read_file_calls += 1
        data = (self.root / path).read_bytes()
        yield data[:7]
        yield data[7:]

    def batch_upload(self, force=False):
        vol = self
        vol.batch_contexts += 1

        class _Batch:
            def __enter__(self):
                return self

            def put_file(self, fileobj, remote):
                dest = vol.root / remote.lstrip("/")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(fileobj.read())
                vol.put_files += 1

            def __exit__(self, *a):
                return False
        return _Batch()

    def bulk_get(self, name, remote_dir, local_dir):
        """`modal volume get <name> <remote_dir> <local_dir>` as the CLI lays it out: the directory's last component under local_dir."""
        self.pulls += 1
        src = self.root / remote_dir.lstrip("/")
        assert src.is_dir(), remote_dir
        shutil.copytree(src, Path(local_dir) / src.name, dirs_exist_ok=True)


def _materialise(shared, root: Path):
    for rel, data in shared.files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)


class TestRemoteVolumeBulkVerbs:
    def test_read_many_is_one_pull_and_write_many_one_batch_context(self, tmp_path):
        fake = _FakeModalVolume(tmp_path / "vol")
        rels = [f"seedrep/x/ladder/queue/none__s1/item{i:03d}/{AID}.json" for i in range(50)]
        for i, r in enumerate(rels):
            (fake.root / r).parent.mkdir(parents=True)
            (fake.root / r).write_bytes(f'{{"i": {i}}}\n'.encode())
        rv = L.RemoteVolume(fake, name="vol-x", scratch=tmp_path / "scratch", bulk_get=fake.bulk_get)
        got = rv.read_many(rels)
        assert got == {r: f'{{"i": {i}}}\n'.encode() for i, r in enumerate(rels)}
        assert (fake.pulls, fake.read_file_calls) == (1, 0) and list((tmp_path / "scratch").iterdir()) == []    # the scratch is removed after a full pull
        rv.write_many({f"seedrep/x/ladder/queue/none__s1/new{i:03d}/{AID}.json": b"n\n" for i in range(50)})
        assert (fake.batch_contexts, fake.put_files) == (1, 50)
        assert rv.walk("seedrep/x/ladder/queue") and len(rv.walk("seedrep/x/ladder/queue")) == 100
        # a pull that does not deliver a listed file refuses and keeps the scratch for inspection
        (fake.root / rels[3]).unlink()
        with pytest.raises(FileNotFoundError, match="delivered 49/50"):
            rv.read_many(rels)
        assert len(list((tmp_path / "scratch").iterdir())) == 1
        # no name -> the per-file path (correct, counted as read_file calls); paths with no common directory below the root refuse
        rv2 = L.RemoteVolume(fake, scratch=tmp_path / "scratch2")
        assert rv2.read_many(rels[:2]) == {r: got[r] for r in rels[:2]} and fake.read_file_calls == 2
        with pytest.raises(L.LadderRefusal, match="no directory below the volume root"):
            rv.read_many(["a/x.json", "b/y.json"])
        assert rv.read_many([]) == {}

    def test_the_whole_replay_over_the_remote_adapter_makes_no_per_record_api_call(self, tmp_path):
        counts = {}
        for n in (5, 50):
            w = _world(tmp_path, n)
            fake = _FakeModalVolume(tmp_path / f"vol{n}")
            _materialise(w.shared, fake.root)
            rv = L.RemoteVolume(fake, name="seedrep-vol", scratch=tmp_path / f"scratch{n}", bulk_get=fake.bulk_get)
            rp = _replay(w, rv)
            assert rp["ok"] and rp["queue_closed"] and rp["n_written"] == n and rp["counts"]["uploaded"] == n + 2
            assert fake.pulls == 2 and fake.batch_contexts == 3 and fake.put_files == n + 2 + 2               # prior + read-back; bulk + marker + CLOSED
            assert fake.read_file_calls <= 6, fake.read_file_calls                                             # inventory, COMPLETE, marker, inventory again
            local = L.LocalVolume(fake.root)
            assert L.read_closed(local, w.this_spec)["n_items"] == n
            assert _under({k: local.read_bytes(k) for k in local.walk(f"{w.this_spec.ladder_root}/queue")}, f"{w.this_spec.ladder_root}/queue") == _expected_rekey(w)
            counts[n] = (fake.listdir_calls, fake.read_file_calls, fake.pulls, fake.batch_contexts)
        assert counts[5] == counts[50], counts
