"""Snapshot operations: the two scopes (a run-level set ships no item Parquet
and keeps the latest full set resident; a full build supersedes everything), the ``--if-due`` cadence policy,
push + commit on local / ssh / Modal destinations (fake runners record the argv; nothing leaves the box), a
failing destination reported rather than raised, the stdlib remote ``commit`` entry point over a subprocess,
the two lease races with a PAUSED READER PROCESS (before lease acquisition; during protected consumption),
the footprint file the CLI writes, and the free-space guard paths.
"""
from __future__ import annotations

import fcntl
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import snapshot as SN
from sisteer.registry import synth

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def _setup(tmp_path, n_runs=2, n_items=10):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=n_runs, n_items=n_items)
    H.ingest_once(root, workers=0, min_free=0)
    return root


def _build(root, scope="full", now=None):
    return SN.build(root, DB.db_path(root), check_free_space=False, scope=scope, now=now)


def _ingest_more(root, i):
    R.write_records(os.path.join(root, "records", "late", f"{i:06d}-late.jsonl"),
                    [synth.rec("si_text", {"si_id": f"late_si_{i}", "si_text_sha256": ("%064x" % (0xe0 + i)), "si_kind": "irrelevant_control",
                                           "source_uri": "synthetic://x"}, synth.t(5000 + i))])
    H.ingest_once(root, workers=0, min_free=0)


# --------------------------------------------------------------------------------------------
# scopes
# --------------------------------------------------------------------------------------------


def test_runlevel_scope_ships_no_item_parquet_and_keeps_latest_full(tmp_path):
    root = _setup(tmp_path)
    full = _build(root, "full")
    _ingest_more(root, 1)
    rl = _build(root, "runlevel")
    assert rl.scope == "runlevel" and rl.snapshot_id != full.snapshot_id
    assert os.path.isdir(full.dir), "a run-level build keeps the latest FULL set resident"
    assert full.snapshot_id not in rl.reclaimed
    files = set(rl.manifest["files"])
    assert "registry_runlevel.sqlite" in files and "run.parquet" in files
    assert not any(f.startswith(("item_result", "item_flags", "run_degeneration")) and f.endswith(".parquet") for f in files)
    assert rl.manifest["parquet_tables"] == list(SN.RUNLEVEL_PARQUET_TABLES) and "run_degeneration" not in rl.manifest["parquet_tables"]
    assert rl.manifest["runlevel_tables"] == list(SN.RUNLEVEL_TABLES) and "run_degeneration" in rl.manifest["runlevel_tables"]
    assert SN.latest_full_id(root) == full.snapshot_id and SN.pointer(root) == rl.snapshot_id
    with SN.snapshot_open(root) as snap:
        assert snap.scope == "runlevel" and snap.parquet("run").num_rows == 2
        with pytest.raises(SN.SnapshotError) as ei:
            snap.parquet("item_result")
        assert "scope runlevel" in str(ei.value) and "item_result.parquet" in str(ei.value)


def test_full_build_supersedes_every_previous_directory(tmp_path):
    root = _setup(tmp_path)
    a = _build(root, "full")
    _ingest_more(root, 1)
    b = _build(root, "runlevel")
    _ingest_more(root, 2)
    c = _build(root, "full")
    assert set(c.reclaimed) == {a.snapshot_id, b.snapshot_id}
    assert SN.snapshot_ids(root) == [c.snapshot_id] and SN.latest_full_id(root) == c.snapshot_id


def test_unknown_scope_refused():
    with pytest.raises(SN.SnapshotError):
        SN.build("/nonexistent", "/nonexistent/x.sqlite", check_free_space=False, scope="items")


# --------------------------------------------------------------------------------------------
# --if-due policy
# --------------------------------------------------------------------------------------------


def test_should_build_policy_walks_the_cadence(tmp_path):
    root = _setup(tmp_path)
    policy = {"in_flight_every_s": 300, "idle_every_s": 3600, "full_every_s": 86400, "runlevel_only_in_flight": True}
    conn = DB.connect(DB.db_path(root), readonly=True)
    t0 = R.utc_to_epoch_ms(synth.t(0)) / 1000.0 + 10_000_000

    def utc(offset):
        return R.epoch_ms_to_utc(int((t0 + offset) * 1000))
    assert SN.should_build(conn, root, policy, now_s=t0) == ("full", "no full snapshot is resident")
    _build(root, "full", now=utc(0))
    scope, why = SN.should_build(conn, root, policy, now_s=t0 + 60)
    assert scope is None and "already carries the index head" in why
    _ingest_more(root, 1)                                   # the head moves, nothing in flight → idle rule
    assert SN.should_build(conn, root, policy, now_s=t0 + 600)[0] is None
    assert SN.should_build(conn, root, policy, now_s=t0 + 3601) == ("runlevel", SN.should_build(conn, root, policy, now_s=t0 + 3601)[1])
    # a queued run puts the registry in flight → run-level every 5 min
    R.write_records(os.path.join(root, "records", "late", "000009-q.jsonl"),
                    [synth.rec("run", synth.run_payload("inflight_k", i=900), synth.t(5900))])
    H.ingest_once(root, workers=0, min_free=0)
    assert conn.execute("SELECT COUNT(*) FROM v_in_flight").fetchone()[0] >= 1
    assert SN.should_build(conn, root, policy, now_s=t0 + 100)[0] is None
    scope, why = SN.should_build(conn, root, policy, now_s=t0 + 301)
    assert scope == "runlevel" and "in flight" in why
    policy2 = dict(policy, runlevel_only_in_flight=False)
    assert SN.should_build(conn, root, policy2, now_s=t0 + 301)[0] == "full"
    # a stale full set wins over everything
    assert SN.should_build(conn, root, policy, now_s=t0 + 86401)[0] == "full"
    conn.close()


def test_should_build_treats_a_dangling_current_pointer_as_due(tmp_path):
    """2026-09-17 live: a run-level build reclaimed CURRENT's directory (retention runs BEFORE the build), then failed
    on a schema drift; every later ``--if-due`` cycle raised ``OverflowError: cannot convert float infinity to integer``
    over the age of a set that was not there. A pointer to no resident set — or no pointer at all — is due at once."""
    root = _setup(tmp_path)
    policy = {"in_flight_every_s": 300, "idle_every_s": 3600, "full_every_s": 86400, "runlevel_only_in_flight": True}
    conn = DB.connect(DB.db_path(root), readonly=True)
    t0 = R.utc_to_epoch_ms(synth.t(0)) / 1000.0 + 10_000_000

    def utc(offset):
        return R.epoch_ms_to_utc(int((t0 + offset) * 1000))
    full = _build(root, "full", now=utc(0))
    _ingest_more(root, 1)
    rl = _build(root, "runlevel", now=utc(3601))
    assert SN.pointer(root) == rl.snapshot_id and SN.should_build(conn, root, policy, now_s=t0 + 3700)[0] is None
    shutil.rmtree(rl.dir)                       # what the failed successor left: CURRENT names a reclaimed directory
    scope, why = SN.should_build(conn, root, policy, now_s=t0 + 4000)
    assert scope == "runlevel" and rl.snapshot_id in why and full.snapshot_id in why and "no resident snapshot directory" in why
    os.remove(SN.current_path(root))            # no pointer at all is the same situation
    scope, why = SN.should_build(conn, root, policy, now_s=t0 + 4000)
    assert scope == "runlevel" and "no CURRENT pointer" in why
    healed = _build(root, "runlevel", now=utc(4000))   # the build it asks for heals the pointer
    assert SN.pointer(root) == healed.snapshot_id and SN.latest_full_id(root) == full.snapshot_id
    assert SN.should_build(conn, root, policy, now_s=t0 + 4001)[0] is None
    conn.close()


def test_a_failed_build_leaves_no_building_directory_and_the_next_cycle_is_due(tmp_path, monkeypatch):
    """The live failure's other trace: ``snapshots/building-<id>.building`` left behind by the export that raised —
    ``reclaim`` never touches a ``.building`` directory, so it would count against the footprint forever."""
    root = _setup(tmp_path)
    policy = {"in_flight_every_s": 300, "idle_every_s": 3600, "full_every_s": 86400, "runlevel_only_in_flight": True}
    first = _build(root, "full")

    def drifted(conn, table, out_path):
        raise sqlite3.OperationalError("no such column: field_confidence_json")
    monkeypatch.setattr(SN, "_export_parquet", drifted)
    with pytest.raises(sqlite3.OperationalError):
        _build(root, "full")                    # retention reclaimed `first` before the export raised
    assert not [n for n in os.listdir(SN.snapshots_dir(root)) if n.endswith(".building")]
    assert SN.pointer(root) == first.snapshot_id and not os.path.isdir(first.dir)
    monkeypatch.undo()
    conn = DB.connect(DB.db_path(root), readonly=True)
    assert SN.should_build(conn, root, policy)[0] == "full"          # no full set is resident any more: due, not a crash
    conn.close()
    healed = _build(root, "full")
    assert SN.pointer(root) == healed.snapshot_id


def test_if_due_cli_exits_zero_when_not_due(tmp_path, monkeypatch, capsys):
    root = _setup(tmp_path)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["snapshot", "--no-space-check"]) == 0
    capsys.readouterr()
    assert cli.main(["snapshot", "--if-due", "--no-space-check"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["built"] is False and "already carries" in out["reason"]


# --------------------------------------------------------------------------------------------
# push + commit
# --------------------------------------------------------------------------------------------


def test_push_to_local_destinations_keeps_the_destinations_full_set(tmp_path):
    root = _setup(tmp_path)
    d1, d2 = str(tmp_path / "cluster_a"), str(tmp_path / "cluster_b")
    dests = [{"name": "cluster_a", "kind": "local", "path": d1}, {"name": "cluster_b", "kind": "local", "path": d2}]
    full = _build(root, "full")
    res = SN.push(root, full.snapshot_id, dests)
    assert [r["ok"] for r in res] == [True, True] and SN.pointer(d1) == full.snapshot_id == SN.pointer(d2)
    assert res[0]["remote_manifest_sha"] and res[0]["error"] is None
    _ingest_more(root, 1)
    rl = _build(root, "runlevel")
    res2 = SN.push(root, rl.snapshot_id, dests)
    assert all(r["ok"] for r in res2) and SN.pointer(d1) == rl.snapshot_id
    assert os.path.isdir(os.path.join(SN.snapshots_dir(d1), full.snapshot_id)), "the destination keeps its full set under a run-level CURRENT"
    assert res2[0]["reclaimed"] == []
    with SN.snapshot_open(d1) as snap:
        assert snap.scope == "runlevel"
    _ingest_more(root, 2)
    full2 = _build(root, "full")
    res3 = SN.push(root, full2.snapshot_id, dests)
    assert set(res3[0]["reclaimed"]) == {full.snapshot_id, rl.snapshot_id}
    assert SN.snapshot_ids(d1) == [full2.snapshot_id]
    # a second push of the same id is idempotent
    res4 = SN.push(root, full2.snapshot_id, dests[:1])
    assert res4[0]["ok"] and SN.pointer(d1) == full2.snapshot_id


class _Recorder:
    def __init__(self, stdout_for=None):
        self.argv = []
        self.stdout_for = stdout_for or (lambda argv: "")

    def __call__(self, argv, timeout_s):
        self.argv.append(list(argv))
        return subprocess.CompletedProcess(list(argv), 0, stdout=self.stdout_for(list(argv)), stderr="")


def test_push_ssh_destination_argv_with_fake_runner(tmp_path):
    root = _setup(tmp_path)
    info = _build(root)
    dest = {"name": "cluster_a", "kind": "ssh", "host": "login-0", "path": "/shared/u/registry/snapshot",
            "remote_src": "/shared/u/repos/constitutional-adapters/src"}
    commit_json = json.dumps({"snapshot_id": info.snapshot_id, "current": info.snapshot_id, "remote_manifest_sha": "ab" * 32,
                              "reclaimed": ["old"], "leases_blocking": []})
    rec = _Recorder(lambda argv: commit_json if "commit" in " ".join(argv) else "")
    res = SN.push(root, info.snapshot_id, [dest], runner=rec)[0]
    assert res["ok"] and res["reclaimed"] == ["old"] and res["remote_manifest_sha"] == "ab" * 32
    cmds = [" ".join(a) for a in rec.argv]
    assert cmds[0].startswith("ssh -o BatchMode=yes -o ConnectTimeout=15 login-0 mkdir -p /shared/u/registry/snapshot/snapshots")
    assert rec.argv[1][:5] == ["rsync", "-rt", "--partial", "--timeout=600", "-e"]
    assert rec.argv[1][-1] == f"login-0:/shared/u/registry/snapshot/snapshots/{info.snapshot_id}.building/"
    assert rec.argv[1][-2] == info.dir + "/"
    assert cmds[2].endswith(f"env PYTHONPATH=/shared/u/repos/constitutional-adapters/src python3 -S -m sisteer.registry.snapshot commit "
                            f"/shared/u/registry/snapshot {info.snapshot_id} --json")


def test_push_failure_is_reported_per_destination_not_raised_and_the_removed_kind_is_an_error_row(tmp_path, monkeypatch):
    """A destination kind without a transport is that destination's error row, by name, and the next destination still runs. The
    removed Modal kind is one of them: an entry handed to push directly (the loader would have refused it
    from a file) reaches no volume and runs no subprocess."""
    root = _setup(tmp_path)
    info = _build(root)
    dests = [{"name": "x", "kind": "carrier-pigeon", "path": "/nowhere"},
             {"name": "legacy", "kind": "modal", "volume": "v", "path": "/registry/snapshot", "python": "/no/such/python", "app": "/no/app"},
             {"name": "cluster_a", "kind": "local", "path": str(tmp_path / "ok")}]
    calls = []
    res = SN.push(root, info.snapshot_id, dests, runner=lambda argv, t: calls.append(list(argv)))
    assert res[0]["ok"] is False and "carrier-pigeon" in res[0]["error"]
    assert res[1]["ok"] is False and "unknown snapshot destination kind 'modal'" in res[1]["error"] and "removed" in res[1]["error"]
    assert res[2]["ok"] is True and calls == [], "no subprocess ran for a kind without a transport"
    # a snapshot that is not on the grading host is reported too
    assert SN.push(root, "no-such-id", dests[2:])[0]["error"].startswith("snapshots/no-such-id is not on the grading host")


def test_remote_commit_entry_point_over_a_subprocess(tmp_path):
    root = _setup(tmp_path)
    info = _build(root)
    dest = str(tmp_path / "cluster")
    staged = os.path.join(SN.snapshots_dir(dest), info.snapshot_id + ".building")
    import shutil
    shutil.copytree(info.dir, staged)
    out = subprocess.run([sys.executable, "-S", "-m", "sisteer.registry.snapshot", "commit", dest, info.snapshot_id, "--json"],
                         capture_output=True, text=True, env={"PYTHONPATH": SRC, "PATH": os.environ["PATH"]}, cwd="/")
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data["current"] == info.snapshot_id and SN.pointer(dest) == info.snapshot_id and not os.path.exists(staged)
    cur = subprocess.run([sys.executable, "-S", "-m", "sisteer.registry.snapshot", "current", dest], capture_output=True, text=True,
                         env={"PYTHONPATH": SRC, "PATH": os.environ["PATH"]}, cwd="/")
    assert info.snapshot_id in cur.stdout
    # a torn staged set is refused by name and CURRENT is left alone
    _ingest_more(root, 1)
    info2 = _build(root)
    staged2 = os.path.join(SN.snapshots_dir(dest), info2.snapshot_id + ".building")
    shutil.copytree(info2.dir, staged2)
    with open(os.path.join(staged2, "run.parquet"), "ab") as fh:
        fh.write(b"torn")
    out2 = subprocess.run([sys.executable, "-S", "-m", "sisteer.registry.snapshot", "commit", dest, info2.snapshot_id, "--json"],
                          capture_output=True, text=True, env={"PYTHONPATH": SRC, "PATH": os.environ["PATH"]}, cwd="/")
    assert out2.returncode == 1 and "torn" in out2.stderr and SN.pointer(dest) == info.snapshot_id


# --------------------------------------------------------------------------------------------
# lease races with a PAUSED READER PROCESS (S10)
# --------------------------------------------------------------------------------------------

READER = textwrap.dedent('''
    import json, os, sys, time
    from sisteer.registry import snapshot as SN
    root, where, flagdir = sys.argv[1], sys.argv[2], sys.argv[3]
    def wait_for(name, timeout=30):
        t0 = time.time()
        while not os.path.exists(os.path.join(flagdir, name)):
            if time.time() - t0 > timeout: raise SystemExit("timeout waiting for " + name)
            time.sleep(0.01)
    def touch(name):
        open(os.path.join(flagdir, name), "w").close()
    def hook(stage):
        if where == "before_lease" and stage == "after_read_current":
            touch("paused"); wait_for("go")
    with SN.snapshot_open(root, pause_hook=hook) as snap:
        n_run = snap.parquet("run").num_rows
        if where == "during":
            touch("paused"); wait_for("go")
        n_items = snap.parquet("item_result").num_rows
        print(json.dumps({"id": snap.snapshot_id, "run": n_run, "items": n_items}))
''')


def _spawn_reader(root, where, flagdir):
    return subprocess.Popen([sys.executable, "-c", READER, root, where, flagdir], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            env={"PYTHONPATH": SRC, "PATH": os.environ["PATH"]})


def _wait_flag(flagdir, name, timeout=30):
    t0 = time.time()
    while not os.path.exists(os.path.join(flagdir, name)):
        assert time.time() - t0 < timeout, f"reader never reached {name}"
        time.sleep(0.01)


def test_reader_process_paused_before_lease_acquisition_survives_reclaim(tmp_path):
    root = _setup(tmp_path)
    a = _build(root)
    flags = str(tmp_path / "flags")
    os.makedirs(flags)
    proc = _spawn_reader(root, "before_lease", flags)
    try:
        _wait_flag(flags, "paused")
        done = {}

        def publisher():
            done["result"] = SN.reclaim(root, keep=[], timeout_s=10)
        t = threading.Thread(target=publisher)
        t.start()
        time.sleep(0.4)
        assert not done, "reclamation must block while the reader holds the shared lock between CURRENT and its lease"
        assert os.path.isdir(a.dir)
        open(os.path.join(flags, "go"), "w").close()
        out, err = proc.communicate(timeout=30)
        t.join(timeout=15)
        assert proc.returncode == 0, err
        data = json.loads(out)
        assert data == {"id": a.snapshot_id, "run": 2, "items": 20}
        rec, blk = done["result"]
        assert rec == [a.snapshot_id] or blk   # reclaimed only AFTER the reader released, or skipped while it held the lease
    finally:
        if proc.poll() is None:
            proc.kill()


def test_reader_process_paused_during_protected_consumption_survives_reclaim(tmp_path):
    root = _setup(tmp_path)
    a = _build(root)
    flags = str(tmp_path / "flags")
    os.makedirs(flags)
    proc = _spawn_reader(root, "during", flags)
    try:
        _wait_flag(flags, "paused")
        assert SN.leases_of(root, a.snapshot_id), "the reader registered its lease"
        _ingest_more(root, 1)
        b = _build(root)                  # publishes twice while the reader holds its lease
        _ingest_more(root, 2)
        c = _build(root)
        assert os.path.isdir(a.dir) and any(l.startswith(a.snapshot_id + "/") for l in c.leases_blocking)
        assert b.snapshot_id in c.reclaimed
        open(os.path.join(flags, "go"), "w").close()
        out, err = proc.communicate(timeout=30)
        assert proc.returncode == 0, err
        assert json.loads(out) == {"id": a.snapshot_id, "run": 2, "items": 20}
    finally:
        if proc.poll() is None:
            proc.kill()
    rec, blk = SN.reclaim(root, keep=[SN.pointer(root)])
    assert a.snapshot_id in rec and not os.path.isdir(a.dir) and not blk


# --------------------------------------------------------------------------------------------
# footprint + free-space guard
# --------------------------------------------------------------------------------------------


def test_cli_snapshot_writes_footprint_and_snapshot_record(tmp_path, monkeypatch, capsys):
    root = _setup(tmp_path)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    assert cli.main(["snapshot", "--no-space-check", "--scope", "runlevel"]) == 0
    out = json.loads(capsys.readouterr().out)
    fp = SN.read_footprint(root)
    assert fp["snapshot_id"] == out["snapshot_id"] and fp["scope"] == "runlevel" and fp["peak_bytes"] > 0
    assert set(fp["peak_breakdown"]) >= {DB.db_path(root), os.path.join(root, "records"), SN.snapshots_dir(root)}
    assert fp["budget_bytes"] == 10 * 1024 ** 3 and fp["pushed_to"] == []
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT snapshot_id, extra_json FROM snapshot").fetchone()
    assert row[0] == out["snapshot_id"] and json.loads(row[1])["scope"] == "runlevel"
    conn.close()
    stamp = json.load(open(os.path.join(root, "state", "last_run", "snapshot.json")))
    assert stamp["rc"] == 0 and stamp["command"] == "snapshot"


def test_free_space_guard_paths(tmp_path, monkeypatch, capsys):
    root = _setup(tmp_path)
    monkeypatch.setenv("REGISTRY_ROOT", root)
    # the snapshot precondition: a run-level set needs the floor alone, a full set 0.4 × DB on top
    ok_rl, free, need_rl = SN.free_space_precondition(root, DB.db_path(root), scope="runlevel")
    ok_full, _, need_full = SN.free_space_precondition(root, DB.db_path(root), scope="full")
    assert need_rl == 2 * 1024 ** 3 and need_full == need_rl + int(0.4 * DB.file_size(DB.db_path(root)))
    # below the floor the CLI refuses, records the reason in the footprint file, exits 1, and CURRENT is untouched
    monkeypatch.setattr(SN, "free_space_precondition", lambda *a, **k: (False, 123, 10 ** 12))
    assert cli.main(["snapshot"]) == 1
    assert "free space 123 B" in capsys.readouterr().err
    fp = SN.read_footprint(root)
    assert fp["snapshot_id"] is None and "not building" in fp["error"]
    assert SN.pointer(root) is None
    monkeypatch.undo()
    monkeypatch.setenv("REGISTRY_ROOT", root)
    # the harvester's guard pauses ingest and the status page carries the banner and the headroom numbers.
    # A controlled reading (1 GB free, whatever the host has) rather than an "impossible" floor: the Modal CPU
    # suite's container reports ~9.2e18 free bytes AND an equally absurd total, so db.free_bytes's clamp cannot
    # bring it under any floor a test could name; the one reader the harvester and the status page share is patched.
    monkeypatch.setattr(H, "free_bytes", lambda path: 1024 ** 3)
    rep = H.ingest_once(root, workers=0, min_free=H.MIN_FREE_BYTES)
    assert rep.paused and rep.free_bytes == 1024 ** 3 and H.paused(root) is not None
    from sisteer.registry import status
    ctx = status.status_json(root)
    assert ctx["paused"] and ctx["disk"]["floor_gb"] == 5.0 and ctx["disk"]["free_gb"] == 1.0 and ctx["disk"]["headroom_gb"] == -4.0
    assert ctx["footprint"]["error"]
    out = str(tmp_path / "index.html")
    status.render(root, out_path=out, register=False)
    html = open(out).read()
    assert "Ingest paused" in html and "not building" in html and "Disk headroom" in html
