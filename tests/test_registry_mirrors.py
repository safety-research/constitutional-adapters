"""Record-log fan-out (F2, S9) on local destinations: resumable sync, verify detects missing and
corrupt files, the manifest pointer switches only after every named record arrived, the ack record,
rulings ride the same file set, gzip preserves the logical sha, a rebuild from the mirror equals the live
database, and the verifier hashes decompressed content everywhere.
"""
from __future__ import annotations

import gzip
import json
import os

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import mirrors as M
from sisteer.registry import records as R
from sisteer.registry import synth


def _setup(tmp_path, n_runs=2):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=n_runs, n_items=10)
    R.write_records(os.path.join(root, "rulings", "rulings_r_synth.jsonl"),
                    [synth.rec("ruling", {"ruling_id": "R1", "round_id": "r_synth", "action": "reference", "scope_kind": "run",
                                          "scope_json": {"run_id": "x"}, "reason": "s", "ruled_utc": synth.t(1), "ruled_by": "u",
                                          "source_ref": "t"}, synth.t(1))])
    H.ingest_once(root, workers=0, min_free=0)
    dest = M.Destination("cluster_a", "local", str(tmp_path / "mirror"))
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": dest.name, "kind": "local", "path": dest.path}]}, fh)
    return root, dest


def _seq(root):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0]
    finally:
        conn.close()


def test_sync_is_resumable_after_interrupt(tmp_path):
    root, dest = _setup(tmp_path)
    files = M.log_files(root)
    # simulate an interrupted first sync: half the files present, one of them partial
    os.makedirs(dest.path)
    for rel, path in files[: len(files) // 2]:
        d = os.path.join(dest.path, os.path.relpath(path, root))
        os.makedirs(os.path.dirname(d), exist_ok=True)
        with open(path, "rb") as src, open(d, "wb") as out:
            out.write(src.read()[:-3] if rel == files[0][0] else src.read())
    res = M.sync(root, dest, _seq(root))
    assert res.switched and res.verify.clean
    assert res.copied == len(files) - len(files) // 2 + 1   # the partial one was re-copied
    assert M.current_manifest(dest.path) == res.manifest


def test_verify_detects_missing_and_corrupt_file(tmp_path):
    root, dest = _setup(tmp_path)
    res = M.sync(root, dest, _seq(root))
    assert res.switched
    mpath = os.path.join(dest.path, M.MANIFEST_DIR, res.manifest)
    entries = M.read_manifest(mpath)
    victim_missing = os.path.join(dest.path, entries[0][1])
    victim_corrupt = os.path.join(dest.path, entries[1][1])
    os.remove(victim_missing if os.path.exists(victim_missing) else victim_missing + ".gz")
    with open(victim_corrupt if os.path.exists(victim_corrupt) else victim_corrupt + ".gz", "ab") as fh:
        fh.write(b"\n{}")
    rep = M.verify_manifest(dest.path, mpath)
    assert rep.missing == [entries[0][1]] and rep.corrupt == [entries[1][1]] and not rep.clean
    vd = M.verify_destination(root, dest)
    assert vd["missing"] and vd["corrupt"]


def test_manifest_pointer_switches_only_after_all_named_records_arrive(tmp_path, monkeypatch):
    root, dest = _setup(tmp_path)
    # make the copy step "lose" one file: the pointer must NOT switch
    real_copy = M._copy_local
    def lossy(src_root, dest_root):
        n = real_copy(src_root, dest_root)
        rel, path = M.log_files(src_root)[-1]
        d = os.path.join(dest_root, os.path.relpath(path, src_root))
        os.remove(d)
        return n
    monkeypatch.setattr(M, "_copy_local", lossy)
    res = M.sync(root, dest, _seq(root))
    assert not res.switched and res.verify.missing
    assert M.current_manifest(dest.path) is None
    monkeypatch.setattr(M, "_copy_local", real_copy)
    res2 = M.sync(root, dest, _seq(root))
    assert res2.switched and M.current_manifest(dest.path) == res2.manifest


def test_ack_records_through_seq(tmp_path):
    root, dest = _setup(tmp_path)
    seq = _seq(root)
    res = M.sync(root, dest, seq)
    mpath = os.path.join(root, M.MANIFEST_DIR, res.manifest)
    payload = M.ack_payload(res, seq, mpath, now=synth.t(1000))
    H.write_harvester_record(root, "mirror_ack", payload, synth.t(1000))
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT destination, acked_through_seq, file_count, manifest_name FROM mirror_ack").fetchone()
    assert tuple(row) == ("cluster_a", seq, res.verify.ok, res.manifest)
    conn.close()


def test_rulings_in_same_file_set(tmp_path):
    root, dest = _setup(tmp_path)
    res = M.sync(root, dest, _seq(root))
    names = [rel for _, rel in M.read_manifest(os.path.join(root, M.MANIFEST_DIR, res.manifest))]
    assert any(n.startswith("rulings/") for n in names) and any(n.startswith("records/") for n in names)
    assert os.path.exists(os.path.join(dest.path, "rulings", "rulings_r_synth.jsonl"))


def test_second_sync_then_compression_then_fresh_rebuild(tmp_path):
    root, dest = _setup(tmp_path)
    res1 = M.sync(root, dest, _seq(root))
    assert res1.switched
    # ack, then gzip after ingest (min age 0), then sync again: the gz travels, the logical sha is unchanged
    H.write_harvester_record(root, "mirror_ack", M.ack_payload(res1, _seq(root), os.path.join(root, M.MANIFEST_DIR, res1.manifest),
                                                               now=synth.t(1000)), synth.t(1000))
    H.ingest_once(root, workers=0, min_free=0, gzip_min_age_s=0)
    H.ingest_once(root, workers=0, min_free=0, gzip_min_age_s=0)   # the ack record itself gets compressed on the next pass
    gz_files = [p for _, p in M.log_files(root) if p.endswith(".gz")]
    assert gz_files
    seq = _seq(root)
    res2 = M.sync(root, dest, seq)
    assert res2.switched
    m1 = dict((rel, sha) for sha, rel in M.read_manifest(os.path.join(root, M.MANIFEST_DIR, res1.manifest)))
    m2 = dict((rel, sha) for sha, rel in M.read_manifest(os.path.join(root, M.MANIFEST_DIR, res2.manifest)))
    for rel in m1:
        assert m2[rel] == m1[rel], rel     # gz predecessor's logical sha equals the plain one
    # the verifier pruned plain copies beside verified gz copies on the destination
    for rel, path in M.log_files(root):
        if path.endswith(".gz"):
            assert not os.path.exists(os.path.join(dest.path, rel))
    # a fresh rebuild FROM THE MIRROR equals the live database
    out = str(tmp_path / "from_mirror.sqlite")
    H.rebuild(dest.path, out, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    diff = H.verify_equal(a, b)
    assert not {k: v for k, v in diff.items() if v}, {k: v for k, v in diff.items() if v}
    a.close()
    b.close()


def test_verifier_hashes_decompressed_content_everywhere(tmp_path):
    root, dest = _setup(tmp_path, n_runs=1)
    res = M.sync(root, dest, _seq(root))
    mpath = os.path.join(dest.path, M.MANIFEST_DIR, res.manifest)
    sha, rel = M.read_manifest(mpath)[0]
    plain = os.path.join(dest.path, rel)
    assert os.path.exists(plain)
    with open(plain, "rb") as fh:
        data = fh.read()
    with gzip.open(plain + ".gz", "wb") as fh:
        fh.write(data)
    os.remove(plain)
    rep = M.verify_manifest(dest.path, mpath)
    assert rep.clean
    # the remote entry point (python3 -S -m sisteer.registry.mirrors verify) returns 0 and JSON
    import subprocess, sys
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    out = subprocess.run(["python3", "-S", "-m", "sisteer.registry.mirrors", "verify", dest.path, mpath, "--json"],
                         capture_output=True, text=True, env={"PYTHONPATH": src, "PATH": os.environ["PATH"]}, cwd="/")
    assert out.returncode == 0, out.stderr
    assert json.loads(out.stdout)["missing"] == []
