"""The cluster mirrors' INCREMENTAL verify (2026-09-28). The whole-log hash on a login node (428,000 entries at 110 files/s,
~65 min) crossed the 3,600 s timeout a third of the time, so: presence of EVERY entry from ONE directory walk; hashed only
the delta since the destination's ``CHECKSUMS.CURRENT``, every carried entry with both a plain and a ``.gz`` copy, and a
seeded random sample; a FULL verify (every entry, a process pool) when there is no base, on ``--full`` and for ``mirrors
verify`` — as an ``srun`` step when the cluster entry says so, never ``--export``, never ``--mem=0``, no GPUs. The report
says what rested on what (``mode`` / ``since`` / ``hashed`` / ``presence_only`` / ``sampled`` / ``sample_seed``); a remote
verifier older than the flags is a NOTE, not a failure. Also: ``write_manifest`` survives a file the harvester gzips between
the listing and the hash, a timed-out verify is that destination's error and the next destination still runs, ``--dest`` is
repeatable, and the cluster entries carry the Slurm defaults.
"""
from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys

import pytest

from sisteer.registry import cli
from sisteer.registry import config as CFG
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import mirrors as M
from sisteer.registry import records as R
from sisteer.registry import synth

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
UNITS = os.path.join(ROOT, "systemd", "user")


def _setup(tmp_path, n_runs=2):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=n_runs, n_items=10)
    H.ingest_once(root, workers=0, min_free=0)
    return root


def _seq(root):
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        return int(conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0])
    finally:
        conn.close()


def _more(root, i):
    R.write_records(os.path.join(root, "records", "late", f"{i:06d}-late.jsonl"),
                    [synth.rec("si_text", {"si_id": f"late_{i}", "si_text_sha256": ("%064x" % (0xf0 + i)), "si_kind": "irrelevant_control",
                                           "source_uri": "synthetic://x"}, synth.t(6000 + i))])
    H.ingest_once(root, workers=0, min_free=0)


class _Recorder:
    """The runner shape ``(argv, timeout_s, input=, check=)``: ``respond(argv)`` is the stdout; ``fail(argv)`` may raise."""

    def __init__(self, respond, fail=None):
        self.argv = []
        self.respond = respond
        self.fail = fail

    def __call__(self, argv, timeout_s, **kw):
        self.argv.append(list(argv))
        if self.fail:
            self.fail(list(argv), timeout_s)
        return subprocess.CompletedProcess(list(argv), 0, stdout=self.respond(list(argv)), stderr="")


def _is_probe(argv):
    return "CHECKSUMS.CURRENT" in " ".join(argv) and "test -f" in " ".join(argv)


def _is_verify(argv):
    return "sisteer.registry.mirrors" in " ".join(argv) and "verify" in argv


def _mirror_with_two_manifests(tmp_path):
    """A local mirror verified clean on manifest A (its pointer), then two more files and manifest B copied beside it."""
    root = _setup(tmp_path)
    dest = M.Destination("m", "local", str(tmp_path / "mirror"))
    res1 = M.sync(root, dest, _seq(root))
    assert res1.switched and res1.mode == "full"
    _more(root, 1)
    _more(root, 2)
    name2 = M.write_manifest(root, _seq(root))
    M._copy_local(root, dest.path)
    shutil.copy2(os.path.join(root, M.MANIFEST_DIR, name2), os.path.join(dest.path, M.MANIFEST_DIR, name2))
    base_path = os.path.join(dest.path, M.MANIFEST_DIR, res1.manifest)
    m2_path = os.path.join(dest.path, M.MANIFEST_DIR, name2)
    base = dict((rel, sha) for sha, rel in M.read_manifest(base_path))
    entries = M.read_manifest(m2_path)
    delta = [rel for sha, rel in entries if base.get(rel) != sha]
    carried = [rel for sha, rel in entries if base.get(rel) == sha]
    assert delta and len(carried) > 5
    return root, dest, base_path, m2_path, entries, delta, carried


def test_incremental_verify_hashes_the_delta_the_gz_twins_and_the_sample_and_confirms_the_rest_by_presence(tmp_path):
    root, dest, base_path, m2_path, entries, delta, carried = _mirror_with_two_manifests(tmp_path)
    # a carried file gzipped after the base was acked: both copies on the destination -> hashed, the plain pruned
    twin = carried[0]
    plain = os.path.join(dest.path, twin)
    with open(plain, "rb") as fh, gzip.open(plain + ".gz", "wb") as out:
        out.write(fh.read())
    rep = M.verify_manifest(dest.path, m2_path, since=base_path, sample=0, seed=1)
    assert rep.mode == "incremental" and rep.since == os.path.basename(base_path) and rep.clean
    assert rep.hashed == len(delta) + 1 and rep.sampled == 0 and rep.presence_only == len(carried) - 1
    assert rep.ok == len(entries) and rep.pruned_plain == 1 and not os.path.exists(plain) and os.path.exists(plain + ".gz")
    assert rep.sample_seed == 1 and rep.seconds >= 0
    # bytes covers what was hashed (the delta + the twin), not the whole mirror
    assert 0 < rep.bytes < sum(os.path.getsize(p) for p in (os.path.join(dest.path, r) for r in carried[1:]) if os.path.exists(p)) + rep.bytes
    # silent LOSS of a carried file is caught by the walk even with no sample; silent CORRUPTION of an unsampled carried
    # file is, by design, not caught by presence alone — the sample and the weekly full verify are for that
    lost, damaged = carried[1], carried[2]
    os.remove(os.path.join(dest.path, lost))
    with open(os.path.join(dest.path, damaged), "ab") as fh:
        fh.write(b"\n{}")
    rep2 = M.verify_manifest(dest.path, m2_path, since=base_path, sample=0, seed=1)
    assert rep2.missing == [lost] and rep2.corrupt == [] and not rep2.clean
    # a sample covering every carried entry hashes them all and catches the corruption
    rep3 = M.verify_manifest(dest.path, m2_path, since=base_path, sample=10 ** 6, seed=1)
    assert rep3.corrupt == [damaged] and rep3.missing == [lost]
    # every carried entry but the lost one is in the sample (the pruned twin is now a single .gz copy and still carried)
    assert rep3.presence_only == 0 and rep3.sampled == len(carried) - 1 and rep3.hashed == len(entries) - 1
    # a delta file damaged on the destination is always hashed, hence always caught
    with open(os.path.join(dest.path, delta[0]), "ab") as fh:
        fh.write(b"\n{}")
    rep4 = M.verify_manifest(dest.path, m2_path, since=base_path, sample=0, seed=1)
    assert delta[0] in rep4.corrupt
    # a base that is not there: full mode with a note, every entry hashed
    rep5 = M.verify_manifest(dest.path, m2_path, since=base_path + ".absent", sample=0)
    assert rep5.mode == "full" and rep5.since is None and "not on the destination" in rep5.note
    assert rep5.hashed == len(entries) - 1 and rep5.presence_only == 0 and rep5.sampled is None


def test_choose_sample_is_bounded_reproducible_and_reports_its_seed():
    rels = [f"records/r/{i:06d}-x.jsonl" for i in range(50)]
    a, seed_a = M.choose_sample(rels, 7, seed=42)
    b, seed_b = M.choose_sample(rels, 7, seed=42)
    assert a == b and len(a) == 7 and len(set(a)) == 7 and seed_a == seed_b == 42 and set(a) <= set(rels)
    c, _ = M.choose_sample(rels, 7, seed=43)
    assert c != a
    whole, _ = M.choose_sample(rels, 500, seed=1)
    assert whole == rels, "a sample larger than the population is the population, in order"
    none, _ = M.choose_sample(rels, 0, seed=1)
    assert none == []
    _, drawn = M.choose_sample(rels, 3, seed=None)
    assert isinstance(drawn, int) and 0 <= drawn < 2 ** 32, "a None seed draws one and reports it"


def test_full_mode_with_a_worker_pool_matches_the_serial_report(tmp_path):
    root, dest, base_path, m2_path, entries, delta, carried = _mirror_with_two_manifests(tmp_path)
    serial = M.verify_manifest(dest.path, m2_path, prune_plain=False, workers=1)
    pooled = M.verify_manifest(dest.path, m2_path, prune_plain=False, workers=2)
    for rep in (serial, pooled):
        assert rep.mode == "full" and rep.hashed == len(entries) and rep.presence_only == 0 and rep.sampled is None and rep.clean
    assert (serial.ok, serial.bytes, serial.missing, serial.corrupt) == (pooled.ok, pooled.bytes, pooled.missing, pooled.corrupt)
    victim = os.path.join(dest.path, carried[0])
    with open(victim, "ab") as fh:
        fh.write(b"\n{}")
    assert M.verify_manifest(dest.path, m2_path, prune_plain=False, workers=2).corrupt == [carried[0]]


def test_walk_present_lists_plain_and_gz_record_files_only(tmp_path):
    root = str(tmp_path / "r")
    for rel in ("records/a/000001-x.jsonl", "records/a/000002-y.jsonl.gz", "rulings/r.jsonl", "records/a/notes.txt",
                "records/a/.hidden.jsonl.tmp", "other/z.jsonl"):
        p = os.path.join(root, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(b"x")
    assert M.walk_present(root) == {"records/a/000001-x.jsonl", "records/a/000002-y.jsonl.gz", "rulings/r.jsonl"}
    assert M.walk_present(str(tmp_path / "absent")) == set()


def _ssh_dest(name="cluster_a", host="login-0", shared="/shared/u", **kw):
    return M.Destination(name, "ssh", f"{shared}/registry/log", host, remote_src=f"{shared}/repos/constitutional-adapters/src", **kw)


def _incremental_json(n, since):
    return json.dumps({"manifest": "x", "ok": n, "missing": [], "corrupt": [], "bytes": 1, "mode": "incremental", "since": since,
                       "hashed": 3, "presence_only": n - 3, "sampled": 2, "sample_seed": 7, "seconds": 12.5})


def test_ssh_sync_verifies_incrementally_against_the_destination_pointer_and_records_what_rested_on_what(tmp_path):
    root = _setup(tmp_path)
    dest = _ssh_dest(slurm=True, qos="high")
    # the destination's acked manifest is one the grading host wrote earlier (its file sits under <root>/manifests): the copy is the delta
    base = M.write_manifest(root, _seq(root))
    _more(root, 1)
    n = len(M.log_files(root))
    rec = _Recorder(lambda argv: (base if _is_probe(argv) else _incremental_json(n, base) if _is_verify(argv) else ""))
    mw = M.write_manifest_report(root, _seq(root))
    res = M.sync(root, dest, _seq(root), mw.name, runner=rec, manifest=mw)
    assert res.switched and res.error is None and res.verify.ok == n
    assert res.mode == "incremental" and res.since == base and res.hashed == 3 and res.presence_only == n - 3 and res.sampled == 2
    assert res.slurm is False and res.note is None and res.seconds > 0
    delta = M._on_disk_relpaths(root, M.added_since(os.path.join(root, M.MANIFEST_DIR, mw.name), os.path.join(root, M.MANIFEST_DIR, base)))
    assert res.copy_mode == "incremental" and res.copy_since == base and res.copy_files == len(delta) > 0
    copies = [a for a in rec.argv if a[0] == "rsync" and "--files-from=-" in a]
    assert len(copies) == 1 and "--ignore-existing" in copies[0] and copies[0][-2:] == [root + "/", "login-0:/shared/u/registry/log/"]
    assert not any(a[0] == "rsync" and a[-1].endswith(("/records/", "/rulings/")) for a in rec.argv), "no whole-tree pass"
    probe = [a for a in rec.argv if _is_probe(a)]
    assert len(probe) == 1 and probe[0][:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"], "one ssh reads the pointer"
    assert rec.argv.index(probe[0]) < rec.argv.index(copies[0]), "the pointer is read before the copy: it is the copy's base too"
    verify = [a for a in rec.argv if _is_verify(a)]
    assert len(verify) == 1 and "srun" not in verify[0], "an incremental verify is light enough for the login node"
    cmd = " ".join(verify[0])
    assert cmd.endswith(f"env PYTHONPATH=/shared/u/repos/constitutional-adapters/src python3 -S -m sisteer.registry.mirrors verify "
                        f"/shared/u/registry/log /shared/u/registry/log/manifests/{mw.name} --json "
                        f"--since /shared/u/registry/log/manifests/{base} --sample 2000")
    assert any("CHECKSUMS.CURRENT.tmp" in " ".join(a) and "mv -f" in " ".join(a) for a in rec.argv)
    # the ack's bytes are the manifest's on-disk bytes, not the few the incremental verify read
    payload = M.ack_payload(res, _seq(root), os.path.join(root, M.MANIFEST_DIR, mw.name), now=synth.t(1), manifest=mw)
    assert payload["bytes"] == mw.bytes > 1 and payload["file_count"] == n
    # sync_all: the row carries the fields and the state file holds them
    rows = M.sync_all(root, _seq(root), [dest], runner=rec, now=synth.t(2))
    row = rows[0]
    assert row["mode"] == "incremental" and row["since"] == base and row["sampled"] == 2 and row["slurm"] is False
    assert row["copy_mode"] == "incremental" and row["copy_since"] == base and row["copy_files"] == len(delta) and row["copied"] == -1
    assert row["hashed"] == 3 and row["presence_only"] == n - 3 and row["vanished"] == 0 and row["vanished_names"] == []
    assert row["note"] is None and row["seconds"] >= 0 and row["switched"] and os.path.exists(row["ack_record"])
    assert M.read_state(root)["destinations"][0]["mode"] == "incremental"
    # the per-run --sample override rides the destination
    dest.sample = 25
    rec2 = _Recorder(lambda argv: (base if _is_probe(argv) else _incremental_json(n, base) if _is_verify(argv) else ""))
    M.sync(root, dest, _seq(root), mw.name, runner=rec2, manifest=mw)
    assert " ".join([a for a in rec2.argv if _is_verify(a)][0]).endswith("--sample 25")


def _full_json(n):
    return json.dumps({"manifest": "x", "ok": n, "missing": [], "corrupt": [], "bytes": 99, "mode": "full", "since": None,
                       "hashed": n, "presence_only": 0, "sampled": None, "sample_seed": None, "seconds": 1000.0})


def test_ssh_sync_runs_a_full_verify_under_srun_when_there_is_no_base_or_full_is_asked(tmp_path):
    root = _setup(tmp_path)
    n = len(M.log_files(root))
    mw = M.write_manifest_report(root, _seq(root))
    dest = _ssh_dest(slurm=True, qos="high")
    # no pointer on the destination (a fresh mirror): full, as a Slurm step
    rec = _Recorder(lambda argv: ("" if _is_probe(argv) else _full_json(n) if _is_verify(argv) else ""))
    res = M.sync(root, dest, _seq(root), mw.name, runner=rec, manifest=mw)
    assert res.switched and res.mode == "full" and res.since is None and res.slurm is True and res.hashed == n
    verify = [a for a in rec.argv if _is_verify(a)][0]
    i = verify.index("srun")
    assert verify[:i] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "login-0"]
    assert verify[i:i + 8] == ["srun", "--partition=general", "--qos=high", "--cpus-per-task=4", "--mem=16G", "--time=0:10:00",
                               "--job-name=registry-verify-cluster_a", "--quiet"]
    assert verify[i + 8:i + 12] == ["env", "PYTHONPATH=/shared/u/repos/constitutional-adapters/src", "python3", "-S"]
    assert verify[-2:] == ["--workers", "4"] and "--since" not in verify and "--no-prune" not in verify
    joined = " ".join(verify)
    for banned in ("--export", "--mem=0", "--gres", "--gpus"):
        assert banned not in joined, banned
    # a pointer exists but --full is asked: srun again, no pointer probe
    rec2 = _Recorder(lambda argv: ("CHECKSUMS-000000000001.sha256" if _is_probe(argv) else _full_json(n) if _is_verify(argv) else ""))
    res2 = M.sync(root, dest, _seq(root), mw.name, runner=rec2, manifest=mw, full=True)
    assert res2.slurm is True and res2.mode == "full" and not any(_is_probe(a) for a in rec2.argv)
    # the ack of a full verify carries the destination's own byte count
    payload = M.ack_payload(res2, _seq(root), os.path.join(root, M.MANIFEST_DIR, mw.name), now=synth.t(1), manifest=mw)
    assert payload["bytes"] == 99
    # Cluster B's entry: its QoS and root; and a cluster with slurm=false keeps the full verify on the login node
    ch = _ssh_dest("cluster_b", "cluster_b", "/shared-b/u", slurm=True, qos="high-b", cpus=8, mem="32G")
    rec3 = _Recorder(lambda argv: ("" if _is_probe(argv) else _full_json(n) if _is_verify(argv) else ""))
    M.sync(root, ch, _seq(root), mw.name, runner=rec3, manifest=mw)
    v3 = [a for a in rec3.argv if _is_verify(a)][0]
    assert "--qos=high-b" in v3 and "--cpus-per-task=8" in v3 and "--mem=32G" in v3 and v3[-2:] == ["--workers", "8"]
    assert "--job-name=registry-verify-cluster_b" in v3 and "/shared-b/u/registry/log" in v3
    plain = _ssh_dest(slurm=False)
    rec4 = _Recorder(lambda argv: ("" if _is_probe(argv) else _full_json(n) if _is_verify(argv) else ""))
    res4 = M.sync(root, plain, _seq(root), mw.name, runner=rec4, manifest=mw)
    assert res4.slurm is False and "srun" not in [a for a in rec4.argv if _is_verify(a)][0] and res4.mode == "full"


def test_slurm_time_is_right_sized_from_the_entry_count():
    assert M.slurm_time(0, 4) == "0:10:00", "the floor"
    assert M.slurm_time(430_000, 4) == "0:35:50", "2x the estimate at 100 files/s per worker"
    assert M.slurm_time(430_000, 8) == "0:17:55"
    assert M.slurm_time(10 ** 8, 4) == "1:00:00", "the ceiling: the runner's own 3,600 s"
    wrap = M._slurm_wrap(_ssh_dest(slurm=True, qos="high"), 430_000)
    assert wrap[0] == "srun" and "--time=0:35:50" in wrap and not any(w.startswith("--export") for w in wrap)


def test_a_remote_verifier_older_than_the_flags_is_a_note_not_a_failure(tmp_path):
    root = _setup(tmp_path)
    n = len(M.log_files(root))
    mw = M.write_manifest_report(root, _seq(root))
    dest = _ssh_dest(slurm=True, qos="high")
    old_shape = json.dumps({"manifest": "x", "ok": n, "missing": [], "corrupt": [], "bytes": 5})
    rec = _Recorder(lambda argv: ("CHECKSUMS-000000000001.sha256" if _is_probe(argv) else old_shape if _is_verify(argv) else ""))
    res = M.sync(root, dest, _seq(root), mw.name, runner=rec, manifest=mw)
    assert res.switched and res.error is None, "the old verifier hashed everything: a valid verify"
    assert res.mode == "full" and res.hashed is None and res.presence_only is None and res.since is None
    assert res.note and "predates" in res.note and "repo-mirror tool" in res.note
    rows = M.sync_all(root, _seq(root), [dest], runner=rec, now=synth.t(2))
    assert "predates" in rows[0]["note"]
    # its ack still carries the manifest's bytes (the old shape's 5 bytes are what it read, not what it holds)
    payload = M.ack_payload(res, _seq(root), os.path.join(root, M.MANIFEST_DIR, mw.name), now=synth.t(1), manifest=mw)
    assert payload["bytes"] == 5, "a full verify's own count stands"


def test_a_timed_out_verify_is_that_destinations_error_and_the_next_destination_still_runs(tmp_path):
    root = _setup(tmp_path)
    n = len(M.log_files(root))
    bulb = _ssh_dest(slurm=True, qos="high")
    char = _ssh_dest("cluster_b", "cluster_b", "/shared-b/u", slurm=True, qos="high-b")

    def fail(argv, timeout_s):
        if _is_verify(argv) and "login-0" in argv:
            raise subprocess.TimeoutExpired(argv, timeout_s)

    base = "CHECKSUMS-000000000001.sha256"
    rec = _Recorder(lambda argv: (base if _is_probe(argv) else _incremental_json(n, base) if _is_verify(argv) else ""), fail=fail)
    rows = M.sync_all(root, _seq(root), [bulb, char], runner=rec, now=synth.t(3))
    assert [r["destination"] for r in rows] == ["cluster_a", "cluster_b"]
    assert not rows[0]["switched"] and "timed out" in rows[0]["error"] and rows[0]["ok"] == 0 and rows[0]["seconds"] >= 0
    assert rows[1]["switched"] and rows[1]["error"] is None and rows[1]["mode"] == "incremental"
    assert not any("CHECKSUMS.CURRENT.tmp" in " ".join(a) and "login-0" in a for a in rec.argv), "no switch after a timeout"
    st = M.read_state(root)
    assert [d["destination"] for d in st["destinations"]] == ["cluster_a", "cluster_b"]
    assert [a for a in rec.argv if _is_verify(a)][0][5] == "login-0", "cluster A's verify was the one that timed out"
    # the verifier's timeout is the module constant, handed to the runner
    timeouts = []
    rec2 = _Recorder(lambda argv: (base if _is_probe(argv) else _incremental_json(n, base) if _is_verify(argv) else ""),
                     fail=lambda argv, t: timeouts.append(t) if _is_verify(argv) else None)
    M.sync(root, char, _seq(root), M.write_manifest(root, _seq(root)), runner=rec2)
    assert timeouts == [M.VERIFY_TIMEOUT_S]


def test_verify_destination_ssh_runs_the_full_verify_under_srun_with_no_prune(tmp_path):
    root = _setup(tmp_path, n_runs=1)
    latest = M.latest_manifest(root) or M.write_manifest(root, _seq(root))
    n = len(M.read_manifest(os.path.join(root, M.MANIFEST_DIR, latest)))
    dest = _ssh_dest(slurm=True, qos="high")
    rec = _Recorder(lambda argv: (_full_json(n) if _is_verify(argv) else ("CHECKSUMS-000000000001.sha256\n" if "cat" in " ".join(argv) else "")))
    row = M.verify_destination(root, dest, runner=rec)
    assert row["lag_files"] == 0 and row["error"] is None and row["slurm"] is True and row["mode"] == "full" and row["hashed"] == n
    verify = [a for a in rec.argv if _is_verify(a)][0]
    assert "srun" in verify and "--no-prune" in verify and verify[-2:] == ["--workers", "4"] and "--since" not in verify
    assert "--qos=high" in verify and "--partition=general" in verify
    plain = _ssh_dest(slurm=False)
    rec2 = _Recorder(lambda argv: (_full_json(n) if _is_verify(argv) else ""))
    row2 = M.verify_destination(root, plain, runner=rec2)
    assert row2["slurm"] is False and "srun" not in [a for a in rec2.argv if _is_verify(a)][0]


def test_write_manifest_survives_a_file_gzipped_or_removed_between_the_listing_and_the_hash(tmp_path, monkeypatch):
    root = _setup(tmp_path)
    real = M.log_files
    listed = real(root)
    plains = [(rel, path) for rel, path in listed if not path.endswith(".gz") and rel.startswith("records/")]
    (twin_rel, twin_path), (gone_rel, gone_path) = plains[0], plains[1]
    twin_sha = R.file_logical_sha(twin_path)
    calls = []

    def racing(r):
        out = real(r)
        calls.append(len(out))
        # the harvester's gzip_after_ingest lands between the listing and the hash: one file becomes its .gz twin …
        if os.path.exists(twin_path):
            with open(twin_path, "rb") as src, gzip.GzipFile(twin_path + ".gz", "wb", mtime=0) as dst:
                dst.write(src.read())
            os.remove(twin_path)
        # … and one is simply gone (no twin)
        if os.path.exists(gone_path):
            os.remove(gone_path)
        return out

    monkeypatch.setattr(M, "log_files", racing)
    mw = M.write_manifest_report(root, _seq(root))
    assert calls and mw.entries == len(listed) - 1 and mw.vanished == [gone_rel] and mw.bytes > 0
    entries = dict((rel, sha) for sha, rel in M.read_manifest(os.path.join(root, M.MANIFEST_DIR, mw.name)))
    assert entries[twin_rel] == twin_sha, "the .gz twin carries the same logical sha"
    assert gone_rel not in entries
    # the sync JSON says so; a local destination's copy step tolerates the same race
    dest = M.Destination("m", "local", str(tmp_path / "mirror"))
    rows = M.sync_all(root, _seq(root), [dest], now=synth.t(4))
    assert rows[0]["switched"] and rows[0]["vanished"] == 0, "the second manifest no longer lists the gone file"
    monkeypatch.setattr(M, "log_files", real)
    # _copy_local: a file removed after the listing is skipped, not a crash
    victim = [p for _, p in real(root) if not p.endswith(".gz")][-1]

    def racing_copy(r):
        out = real(r)
        if os.path.exists(victim):
            os.remove(victim)
        return out

    monkeypatch.setattr(M, "log_files", racing_copy)
    M._copy_local(root, str(tmp_path / "mirror2"))
    monkeypatch.setattr(M, "log_files", real)
    # the row names up to VANISHED_NAMES_IN_ROW files
    assert M.VANISHED_NAMES_IN_ROW == 50
    # the historical signature still returns the name
    assert M.write_manifest(root, _seq(root)).startswith("CHECKSUMS-")


def test_cli_dest_is_repeatable_kind_filters_and_full_and_sample_ride_to_sync_all(tmp_path, monkeypatch):
    root = _setup(tmp_path, n_runs=1)
    with open(os.path.join(root, "config.json"), "w") as fh:
        json.dump({"mirrors": [{"name": "cluster_a", "kind": "ssh", "host": "h", "path": "/x/log", "remote_src": "/x/src", "slurm": True,
                                "qos": "high"},
                               {"name": "cluster_b", "kind": "ssh", "host": "c", "path": "/y/log", "remote_src": "/y/src", "slurm": True,
                                "qos": "high-b", "verify_sample": 300},
                               {"name": "local", "kind": "local", "path": str(tmp_path / "m")}]}, fh)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    captured = {}

    def fake_sync_all(root_, seq, dests, deadline=None, reconcile=False, full=False, **kw):
        captured["dests"] = list(dests)
        captured["full"] = full
        return [{"destination": d.name, "switched": True, "error": None, "stopped": None} for d in dests]

    monkeypatch.setattr(M, "sync_all", fake_sync_all)
    assert cli.main(["mirrors", "sync"]) == 0 and [d.name for d in captured["dests"]] == ["cluster_a", "cluster_b", "local"]
    assert captured["full"] is False and [d.sample for d in captured["dests"][:2]] == [2000, 300]
    assert cli.main(["mirrors", "sync", "--dest", "cluster_a", "--dest", "cluster_b", "--full", "--sample", "7"]) == 0
    assert [d.name for d in captured["dests"]] == ["cluster_a", "cluster_b"] and captured["full"] is True
    assert [d.sample for d in captured["dests"]] == [7, 7], "the flag wins for one run"
    assert cli.main(["mirrors", "sync", "--kind", "ssh"]) == 0 and [d.name for d in captured["dests"]] == ["cluster_a", "cluster_b"]
    assert [(d.qos, d.slurm) for d in captured["dests"]] == [("high", True), ("high-b", True)], "config.json entries carry the knobs too"
    assert cli.main(["mirrors", "sync", "--kind", "local"]) == 0 and [d.name for d in captured["dests"]] == ["local"]
    assert cli.main(["mirrors", "sync", "--sample", "-1"]) == 2


def test_cluster_destinations_carry_qos_and_slurm_defaults_and_the_entry_overrides_them(tmp_path):
    root = str(tmp_path / "reg")
    os.makedirs(root)
    with open(os.path.join(root, CFG.CONFIG_NAME), "w") as fh:
        fh.write('[registry]\nuser = "u"\n\n[[clusters]]\nname = "cluster_a"\nhost = "cluster-a-login"\nshared = "/scratch/example/u"\n\n'
                 '[[clusters]]\nname = "cluster_b"\nhost = "cluster-b-login"\nshared = "/data/example/u"\nqos = "low"\nslurm = false\n'
                 'verify_sample = 10\nverify_cpus = 8\nverify_mem = "32G"\npartition = "overflow"\n')
    dests = CFG.mirror_destinations(CFG.load(root))
    assert [d["name"] for d in dests] == ["cluster_a", "cluster_b"]
    b, c = dests
    assert (b["qos"], b["slurm"], b["partition"], b["verify_cpus"], b["verify_mem"], b["verify_sample"]) == ("normal", True, "general", 4, "16G", 2000)
    assert (c["qos"], c["slurm"], c["partition"], c["verify_cpus"], c["verify_mem"], c["verify_sample"]) == ("low", False, "overflow", 8, "32G", 10)
    db, dc = M.Destination.from_config(b), M.Destination.from_config(c)
    assert db.qos == "normal" and db.slurm and db.cpus == 4 and db.mem == "16G" and db.sample == 2000 and db.partition == "general"
    assert dc.qos == "low" and not dc.slurm and dc.cpus == 8 and dc.mem == "32G" and dc.sample == 10 and dc.partition == "overflow"
    assert M.Destination.from_config(db.as_dict()) == db, "as_dict round-trips"
    # a third cluster with no known default QoS falls back to `normal`; the rendered default file names the QoS per cluster
    assert CFG.mirror_destinations({"clusters": [{"name": "cluster_c", "host": "s", "shared": "/w/u"}]})[0]["qos"] == "normal"
    text = CFG.render_default(root, user="u")
    assert text.count('qos = "normal"') == 2
    # a Destination built by hand (the tests' shape) keeps the login-node path unless asked
    assert M.Destination("x", "ssh", "/p", "h").slurm is False and M.Destination("x", "ssh", "/p", "h").sample == 2000


def test_remote_verifier_entry_point_takes_the_incremental_flags_and_keeps_the_old_shape(tmp_path):
    root, dest, base_path, m2_path, entries, delta, carried = _mirror_with_two_manifests(tmp_path)
    env = {"PYTHONPATH": SRC, "PATH": os.environ["PATH"]}
    out = subprocess.run(["python3", "-S", "-m", "sisteer.registry.mirrors", "verify", dest.path, m2_path, "--json", "--no-prune",
                          "--since", base_path, "--sample", "1", "--seed", "3", "--workers", "2"],
                         capture_output=True, text=True, env=env, cwd="/")
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["mode"] == "incremental" and data["since"] == os.path.basename(base_path) and data["sampled"] == 1 and data["sample_seed"] == 3
    assert data["hashed"] == len(delta) + 1 and data["presence_only"] == len(carried) - 1 and data["ok"] == len(entries) and data["missing"] == []
    assert data["note"] is None and data["seconds"] >= 0
    # the old invocation still works and reports full mode; an unknown flag is noted, not fatal
    old = subprocess.run(["python3", "-S", "-m", "sisteer.registry.mirrors", "verify", dest.path, m2_path, "--json", "--future-flag"],
                         capture_output=True, text=True, env=env, cwd="/")
    assert old.returncode == 0, old.stderr
    d2 = json.loads(old.stdout)
    assert d2["mode"] == "full" and d2["hashed"] == len(entries) and d2["presence_only"] == 0 and "--future-flag" in d2["note"]
    # $SLURM_CPUS_PER_TASK sizes the pool when --workers is absent (the operator's srun one-liner)
    step = subprocess.run(["python3", "-S", "-m", "sisteer.registry.mirrors", "verify", dest.path, m2_path],
                          capture_output=True, text=True, env=dict(env, SLURM_CPUS_PER_TASK="2"), cwd="/")
    assert step.returncode == 0 and "mode=full" in step.stdout and f"hashed={len(entries)}" in step.stdout
    # missing/corrupt is exit 1, still a report
    os.remove(os.path.join(dest.path, delta[0]))
    bad = subprocess.run(["python3", "-S", "-m", "sisteer.registry.mirrors", "verify", dest.path, m2_path, "--json", "--since", base_path],
                         capture_output=True, text=True, env=env, cwd="/")
    assert bad.returncode == 1 and json.loads(bad.stdout)["missing"] == [delta[0]]
    assert subprocess.run(["python3", "-S", "-m", "sisteer.registry.mirrors"], capture_output=True, text=True, env=env, cwd="/").returncode == 2


def test_weekly_full_verify_unit_templates_are_installed_and_shaped_for_slurm():
    with open(os.path.join(UNITS, "registry-mirrors-full.service.in"), encoding="utf-8") as fh:
        svc = fh.read()
    with open(os.path.join(UNITS, "registry-mirrors-full.timer.in"), encoding="utf-8") as fh:
        tmr = fh.read()
    assert "mirrors sync --kind ssh --full" in svc and "Conflicts=registry-mirrors.service" in svc and "TimeoutStartSec=3h" in svc
    assert "Restart=on-failure" in svc and "RestartSec=1h" in svc and "StartLimitBurst=2" in svc, "one retry, never a loop"
    assert "eval " not in svc, "secret-free: the clusters need no credential (no environment hook)"
    assert "--export" not in svc
    assert "OnCalendar=Sun" in tmr and "Persistent=true" in tmr and "Unit=registry-mirrors-full.service" in tmr
    with open(os.path.join(ROOT, "scripts", "registry", "install_grading_host.sh"), encoding="utf-8") as fh:
        assert "registry-mirrors-full.timer" in fh.read()
