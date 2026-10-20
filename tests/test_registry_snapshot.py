"""Snapshot publisher and reader lease protocol: two consecutive
publications; atomic pointer; a reader paused BEFORE lease acquisition and one paused DURING protected
consumption both survive reclamation; retry-on-disappearance refuses after three attempts; a live Modal
consumer across the heartbeat-expiry boundary is not reclaimed (a verified-finished one is; an
unresolvable call id keeps its snapshot); concurrent ingest during the build; the run-level file is
allowlist-only and the allowlist is the complement of the item-scale tables, the staging shadows and the history
ledger (``row_history`` ships in NEITHER scope: it is the reducer's fold ledger, read only on the grading host against the
live database — an inflated ledger enters no snapshot, and the size assertion names the tables it fires on); torn set
refused; retention 1 once readers release; push to a local destination commits and points CURRENT.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import snapshot as SN
from sisteer.registry import synth


def _setup(tmp_path, n_runs=2, n_items=10):
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    synth.write_corpus(root, n_runs=n_runs, n_items=n_items)
    H.ingest_once(root, workers=0, min_free=0)
    return root


def _build(root):
    return SN.build(root, DB.db_path(root), check_free_space=False)


def test_runlevel_allowlist_is_complement_of_item_scale():
    # every table but the item-scale ones, the transient staging shadows (inactive revision rows are not data of record)
    # and the history ledger (the reducer's fold ledger: read by no view and no snapshot reader; the record log is the history)
    # ... except `run_degeneration` (item-scale by key, run-grain in count), which the paper views read since migration 0010
    assert SN.RUNLEVEL_ITEM_TABLES == ("run_degeneration",) and set(SN.RUNLEVEL_ITEM_TABLES) <= set(S.ITEM_SCALE_TABLES)
    assert set(SN.RUNLEVEL_TABLES) == (set(S.ALL_TABLES) - set(S.ITEM_SCALE_TABLES) - set(S.STAGING_TABLES) - set(SN.HISTORY_TABLES)
                                       | set(SN.RUNLEVEL_ITEM_TABLES))
    assert set(SN.RUNLEVEL_TABLES) & set(R.ITEM_SCALE_TABLES) == set(SN.RUNLEVEL_ITEM_TABLES)
    assert not set(SN.RUNLEVEL_TABLES) & set(S.STAGING_TABLES)
    assert set(SN.ROW_FILTERS) == {"record_delivery", "observation", "blocked_records", "run_degeneration"}
    assert SN.ROW_FILTERS["run_degeneration"] == S.item_visible()          # the visible rows only: a superseded revision's are read by nothing
    assert set(SN.RUNLEVEL_PARQUET_TABLES) == set(SN.RUNLEVEL_TABLES) - set(SN.RUNLEVEL_ITEM_TABLES)
    # the history ledger is a ledger (not record-fed), is exactly row_history, is read by no view, and ships in neither scope
    assert SN.HISTORY_TABLES == ("row_history",) and set(SN.HISTORY_TABLES) <= set(S.LEDGER_TABLES)
    assert not any(set(SN.HISTORY_TABLES) & S.VIEW_TABLES[v] for v in S.VIEW_NAMES)
    assert set(SN.FULL_TABLES) == set(S.ALL_TABLES) - set(S.STAGING_TABLES) - set(SN.HISTORY_TABLES)
    assert set(SN.RUNLEVEL_TABLES) <= set(SN.FULL_TABLES) and set(S.ITEM_SCALE_TABLES) <= set(SN.FULL_TABLES)
    # the paper and source-check views the clusters read stay in the run-level file (v_source_status reads the item-scale
    # source_status table, so it never was and is not); the degeneration views join them with their table (0010)
    for v in ("v_paper_long", "v_paper_current", "v_multi_source_check", "v_in_flight", "v_run_degeneration_current", "v_run_degeneration_wide"):
        assert v in SN.RUNLEVEL_VIEWS, v
    assert "v_source_status" not in SN.RUNLEVEL_VIEWS and "v_item_headline" not in SN.RUNLEVEL_VIEWS


def test_two_consecutive_publications_succeed(tmp_path):
    root = _setup(tmp_path)
    a = _build(root)
    assert SN.pointer(root) == a.snapshot_id and os.path.isdir(a.dir)
    synth.write_corpus(str(tmp_path / "more"), n_runs=1, n_items=5)  # unrelated
    b = _build(root)
    assert b.snapshot_id != a.snapshot_id and SN.pointer(root) == b.snapshot_id
    assert a.snapshot_id in b.reclaimed and not os.path.isdir(a.dir)   # retention 1, deleted before the build
    with SN.snapshot_open(root) as snap:
        assert snap.snapshot_id == b.snapshot_id
        assert snap.parquet("run").num_rows == 2


def test_pointer_switch_is_atomic(tmp_path):
    root = _setup(tmp_path)
    a = _build(root)
    seen = set()
    stop = threading.Event()

    bad = []

    def reader():
        while not stop.is_set():
            p = SN.pointer(root)
            if p:
                seen.add(p)
                if not (os.path.isdir(os.path.join(SN.snapshots_dir(root), p)) or p == a.snapshot_id):
                    bad.append(p)
            time.sleep(0.001)
    t = threading.Thread(target=reader, daemon=True)
    t.start()
    try:
        b = _build(root)
    finally:
        stop.set()
        t.join(timeout=10)
    assert not bad and seen <= {a.snapshot_id, b.snapshot_id}


def test_runlevel_sqlite_is_allowlist_only(tmp_path):
    root = _setup(tmp_path)
    # put rows in the mixed tables that must be filtered
    conn = DB.connect(DB.db_path(root))
    with DB.transaction(conn):
        conn.execute("INSERT INTO blocked_records(record_sha256, record_uri, record_type, missing_parent_table, missing_parent_id, "
                     "first_seen_utc, last_checked_utc) VALUES (?, 'u#1', 'item_result', 'run', 'x', ?, ?)",
                     ("9" * 64, synth.t(0), synth.t(0)))
        conn.execute("INSERT INTO blocked_records(record_sha256, record_uri, record_type, missing_parent_table, missing_parent_id, "
                     "first_seen_utc, last_checked_utc) VALUES (?, 'u#2', 'run_state', 'run', 'x', ?, ?)",
                     ("8" * 64, synth.t(0), synth.t(0)))
    conn.close()
    info = _build(root)
    rl = sqlite3.connect(os.path.join(info.dir, "registry_runlevel.sqlite"))
    tables = {r[0] for r in rl.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert tables == set(SN.RUNLEVEL_TABLES)
    # every view whose base tables (through any view it references) are all in the run-level file is in it — and queryable
    assert {r[0] for r in rl.execute("SELECT name FROM sqlite_master WHERE type = 'view'")} == set(SN.RUNLEVEL_VIEWS)
    assert set(SN.RUNLEVEL_VIEWS) == {n for n in S.VIEW_NAMES if S.VIEW_TABLES[n] <= set(SN.RUNLEVEL_TABLES)}
    assert set(SN.RUNLEVEL_VIEWS) == ({n for n in S.VIEW_NAMES if not S.VIEW_TABLES[n] & (set(S.ITEM_SCALE_TABLES) - set(SN.RUNLEVEL_ITEM_TABLES))}
                                      - {n for n in S.VIEW_NAMES if S.VIEW_TABLES[n] & (set(S.STAGING_TABLES) | set(SN.HISTORY_TABLES))})
    assert "v_paper_current" in SN.RUNLEVEL_VIEWS and "v_item_headline" not in SN.RUNLEVEL_VIEWS
    for v in SN.RUNLEVEL_VIEWS:
        rl.execute(f"SELECT COUNT(*) FROM {v}").fetchone()
    # the one item-scale table the file carries: its visible rows, keyed for the current-row anti-join its views run
    assert {r[0] for r in rl.execute("SELECT name FROM sqlite_master WHERE type = 'index'")} == {"idx_run_degeneration_snapshot_key"}
    assert "row_history" not in tables                      # the history ledger ships in neither scope
    assert rl.execute("SELECT COUNT(*) FROM blocked_records WHERE record_type = 'item_result'").fetchone()[0] == 0
    assert rl.execute("SELECT COUNT(*) FROM blocked_records").fetchone()[0] == 1
    # item-only record files ride Parquet only
    for uri, in rl.execute("SELECT file_uri FROM record_delivery"):
        assert "item_result" not in uri
    rl.close()
    assert info.runlevel_bytes <= SN.RUNLEVEL_MAX_BYTES
    assert info.manifest["tables"]["item_result"] == 20
    assert os.path.exists(os.path.join(info.dir, "item_result.parquet"))
    assert "row_history.parquet" not in info.manifest["files"] and "row_history" not in info.manifest["tables"]
    assert info.manifest["parquet_tables"] == list(SN.FULL_TABLES)


def _inflate_row_history(root, n_rows, payload_bytes):
    """Append fat run-scale history events (valid rows: run-scale table, a real record type, hex sha, UTC-ms stamp, JSON payloads) —
    the shape that put the live ledger at 679 MB, 77 % of a 922 MB run-level set (2026-09-18)."""
    conn = DB.connect(DB.db_path(root))
    fat = json.dumps({"blob": "x" * payload_bytes})
    with DB.transaction(conn):
        for i in range(n_rows):
            conn.execute("INSERT INTO row_history(table_name, natural_key_json, written_utc, record_sha256, record_type, record_uri, "
                         "payload_json, folded_state_json) VALUES ('run', ?, ?, ?, 'run_state', 'records/synthetic/inflate.jsonl', ?, ?)",
                         (json.dumps({"run_id": f"inflate_{i}"}), synth.t(9000 + i), "%064x" % (0xf0000 + i), fat, fat))
    conn.close()


def _runlevel_tables(info):
    rl = sqlite3.connect(os.path.join(info.dir, "registry_runlevel.sqlite"))
    try:
        return {r[0] for r in rl.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        rl.close()


def test_inflated_row_history_enters_neither_scope(tmp_path, monkeypatch):
    """A synthetic root whose ``row_history`` alone is several times the run-level bound: the run-level build succeeds
    without it, the full build's manifest lists no ``row_history.parquet``, a reader asking for it is refused by name
    (the record log is the history), and the ledger itself is untouched in the live database. The KNOWN-BAD half first:
    with the ledger back in the allowlist the same build fails and the assertion names ``row_history`` largest, with
    its measured bytes."""
    root = _setup(tmp_path)
    base = _build(root)
    bound = 2 * base.runlevel_bytes
    n_rows, payload = 90, 16 * 1024      # 60 -> 90 (2026-09-21): the run-level views gained columns and the 3x premise below sat within 2 % of them
    _inflate_row_history(root, n_rows, payload)
    live = DB.connect(DB.db_path(root), readonly=True)
    n_hist = live.execute("SELECT COUNT(*) FROM row_history").fetchone()[0]
    live.close()
    assert n_hist >= n_rows
    inflated = SN.table_bytes(DB.db_path(root))["row_history"]
    assert inflated > 3 * bound, (inflated, bound)      # the premise: the ledger alone breaches the bound several times over
    monkeypatch.setattr(SN, "RUNLEVEL_MAX_BYTES", bound)
    # KNOWN-BAD: the pre-change allowlist (ledger included) fails the build, naming the ledger first with its bytes
    with monkeypatch.context() as m:
        m.setattr(SN, "RUNLEVEL_TABLES", SN.RUNLEVEL_TABLES + ("row_history",))
        with pytest.raises(SN.SnapshotError) as ei:
            SN.build(root, DB.db_path(root), check_free_space=False, scope="runlevel")
    msg = str(ei.value)
    assert f"> {bound} B" in msg
    top = re.search(r"largest tables: (\w+) (\d+) B \(([\d.]+) %\)", msg)
    assert top and top.group(1) == "row_history", msg
    assert int(top.group(2)) >= n_rows * 2 * payload and float(top.group(3)) > 50.0, msg
    assert SN.pointer(root) == base.snapshot_id and os.path.isdir(base.dir)          # a failed build moves nothing
    assert not [n for n in os.listdir(SN.snapshots_dir(root)) if n.endswith(".building")]  # and leaves no half-written directory
    # the change: the ledger is outside both scopes, so the same database builds under the same bound
    rl = SN.build(root, DB.db_path(root), check_free_space=False, scope="runlevel")
    assert rl.runlevel_bytes <= bound
    assert _runlevel_tables(rl) == set(SN.RUNLEVEL_TABLES) and "row_history" not in _runlevel_tables(rl)
    assert "row_history.parquet" not in rl.manifest["files"] and rl.manifest["parquet_tables"] == list(SN.RUNLEVEL_PARQUET_TABLES)
    full = SN.build(root, DB.db_path(root), check_free_space=False, scope="full")
    assert full.runlevel_bytes <= bound and "row_history" not in _runlevel_tables(full)
    assert "row_history.parquet" not in full.manifest["files"] and "row_history" not in full.manifest["tables"]
    assert full.manifest["parquet_tables"] == list(SN.FULL_TABLES) and "item_result.parquet" in full.manifest["files"]
    with SN.snapshot_open(root) as snap:
        assert snap.scope == "full" and snap.parquet("run").num_rows == 2
        with pytest.raises(SN.SnapshotError) as ei2:
            snap.parquet("row_history")
        assert "history ledger" in str(ei2.value) and "record log" in str(ei2.value) and "row_history.parquet" in str(ei2.value)
        for v in SN.RUNLEVEL_VIEWS:                        # every run-level view still compiles and runs without the ledger
            snap.conn.execute(f"SELECT COUNT(*) FROM {v}").fetchone()
    live = DB.connect(DB.db_path(root), readonly=True)
    assert live.execute("SELECT COUNT(*) FROM row_history").fetchone()[0] == n_hist    # the ledger stays in the live database
    live.close()


def test_runlevel_size_assertion_names_offending_tables(tmp_path, monkeypatch):
    """When the bound fires the message carries the file's bytes against the bound AND the largest tables with measured
    bytes and share, largest first — names from the allowlist, at most RUNLEVEL_SIZE_REPORT_TABLES of them."""
    root = _setup(tmp_path)
    base = _build(root)
    monkeypatch.setattr(SN, "RUNLEVEL_MAX_BYTES", 1)
    with pytest.raises(SN.SnapshotError) as ei:
        _build(root)
    msg = str(ei.value)
    assert "run-level file is" in msg and "> 1 B" in msg and "largest tables:" in msg
    named = re.findall(r"(\w+) (\d+) B \(([\d.]+) %\)", msg)
    assert 1 <= len(named) <= SN.RUNLEVEL_SIZE_REPORT_TABLES
    assert {n for n, _, _ in named} <= set(SN.RUNLEVEL_TABLES)
    sizes = [int(b) for _, b, _ in named]
    assert sizes == sorted(sizes, reverse=True) and all(b > 0 for b in sizes)
    assert sum(float(p) for _, _, p in named) <= 100.5
    assert SN.pointer(root) == base.snapshot_id
    assert not [n for n in os.listdir(SN.snapshots_dir(root)) if n.endswith(".building")]


def test_runlevel_bound_is_512_mib_and_the_assertion_still_names_offenders(tmp_path):
    """The bound moved 100 MiB → 512 MiB on 2026-09-18 (the live set without the ledger measured ~207 MiB; the caveats sit
    beside the constant). Pinned here so a silent change to it is a test change; and the message the assertion raises
    at THIS bound still names allowlist tables with measured bytes and share, largest first."""
    assert SN.RUNLEVEL_MAX_BYTES == 512 * 1024 * 1024
    root = _setup(tmp_path)
    info = _build(root)
    assert info.runlevel_bytes <= SN.RUNLEVEL_MAX_BYTES
    rl_path = os.path.join(info.dir, "registry_runlevel.sqlite")
    msg = SN.runlevel_size_message(rl_path, SN.RUNLEVEL_MAX_BYTES + 1, SN.RUNLEVEL_MAX_BYTES)
    assert f"> {512 * 1024 * 1024} B" in msg and "largest tables:" in msg
    named = re.findall(r"(\w+) (\d+) B \(([\d.]+) %\)", msg)
    assert 1 <= len(named) <= SN.RUNLEVEL_SIZE_REPORT_TABLES and {n for n, _, _ in named} <= set(SN.RUNLEVEL_TABLES)
    sizes = [int(b) for _, b, _ in named]
    assert sizes == sorted(sizes, reverse=True) and all(b > 0 for b in sizes)
    measured = SN.table_bytes(rl_path)
    assert named[0][0] == max(measured, key=lambda k: (measured[k], k)) or measured[named[0][0]] == max(measured.values())
    assert "row_history" not in measured                # the ledger is not in the file the bound governs


def test_table_bytes_measures_every_table_and_attributes_indexes(tmp_path):
    """A measured reading (dbstat), never an estimate: every table's b-tree bytes, with its indexes' pages attributed to it."""
    p = str(tmp_path / "x.sqlite")
    c = sqlite3.connect(p)
    c.execute("CREATE TABLE a(x TEXT PRIMARY KEY, y TEXT)")
    c.execute("CREATE TABLE b(x TEXT)")
    c.executemany("INSERT INTO a VALUES (?, ?)", [("k%06d" % i, "v" * 500) for i in range(2000)])
    c.execute("INSERT INTO b VALUES ('one')")
    c.commit()
    c.close()
    sizes = SN.table_bytes(p)
    assert set(sizes) >= {"a", "b"} and sizes["a"] > 2000 * 500 and sizes["b"] >= 4096
    assert "sqlite_autoindex_a_1" not in sizes          # the PK index's pages are attributed to `a`, not listed by their own name


def test_torn_set_refused(tmp_path):
    root = _setup(tmp_path)
    info = _build(root)
    with open(os.path.join(info.dir, "run.parquet"), "ab") as fh:
        fh.write(b"torn")
    with pytest.raises(SN.SnapshotError) as ei:
        with SN.snapshot_open(root):
            pass
    assert info.snapshot_id in str(ei.value)
    # CURRENT naming a dir without a MANIFEST
    SN.switch_pointer(root, "no-such-snapshot")
    with pytest.raises(SN.SnapshotError) as ei2:
        with SN.snapshot_open(root):
            pass
    assert "no-such-snapshot" in str(ei2.value)


def test_retry_on_disappearance_refuses_after_three_attempts(tmp_path):
    root = _setup(tmp_path)
    _build(root)
    SN.switch_pointer(root, "stale-mirror-id")
    attempts = []
    with pytest.raises(SN.SnapshotError) as ei:
        with SN.snapshot_open(root, pause_hook=lambda where: attempts.append(where)):
            pass
    assert len(attempts) == 3 and "3 attempts" in str(ei.value)
    assert not os.listdir(os.path.join(SN.snapshots_dir(root), SN.LEASES_DIR, "stale-mirror-id"))  # its leases were removed


def test_reader_paused_before_lease_acquisition_survives_reclaim(tmp_path):
    """Paused AFTER reading CURRENT but BEFORE creating the lease file: the reader holds the shared
    lock, so the publisher's exclusive reclamation waits; when it resumes its reads succeed."""
    root = _setup(tmp_path)
    a = _build(root)
    gate = threading.Event()
    resumed = threading.Event()
    result = {}

    def pause(where):
        if where == "after_read_current":
            gate.set()
            resumed.wait(timeout=10)

    def reader():
        with SN.snapshot_open(root, pause_hook=pause) as snap:
            result["id"] = snap.snapshot_id
            result["rows"] = snap.parquet("run").num_rows
    t = threading.Thread(target=reader)
    t.start()
    gate.wait(timeout=10)
    # the publisher tries to reclaim while the reader is inside the shared-lock window → must block
    done = {}

    def publisher():
        rec, blk = SN.reclaim(root, keep=[], timeout_s=5)
        done["reclaimed"], done["blocking"] = rec, blk
    p = threading.Thread(target=publisher)
    p.start()
    time.sleep(0.3)
    assert not done, "reclamation must wait for the reader's shared lock"
    resumed.set()
    t.join(timeout=10)
    p.join(timeout=10)
    assert result["id"] == a.snapshot_id and result["rows"] == 2
    # the reader released its lease → the publisher's reclamation ran afterwards (or found it leased and skipped)
    assert done["reclaimed"] == [a.snapshot_id] or done["blocking"]


def test_reader_paused_during_protected_consumption_survives_reclaim(tmp_path):
    root = _setup(tmp_path)
    a = _build(root)
    with SN.snapshot_open(root) as snap:
        n1 = snap.parquet("run").num_rows
        b = _build(root)               # publishes twice while the lease is registered
        c = _build(root)
        assert a.snapshot_id in c.leases_blocking[0] if c.leases_blocking else True
        assert os.path.isdir(a.dir)    # survives until release
        n2 = snap.parquet("item_result").num_rows
        assert n1 == 2 and n2 == 20
    rec, blk = SN.reclaim(root, keep=[SN.pointer(root)])
    assert a.snapshot_id in rec and not os.path.isdir(a.dir) and not blk


def test_modal_live_consumer_across_heartbeat_expiry_is_not_reclaimed(tmp_path):
    root = _setup(tmp_path)
    a = _build(root)
    lease_dir = os.path.join(SN.snapshots_dir(root), SN.LEASES_DIR, a.snapshot_id)
    os.makedirs(lease_dir)
    stale_heartbeat = json.dumps({"touched_utc": "2026-01-01T00:00:00.000Z"})
    for tok in ("fc-RUNNING.aaaa", "fc-FINISHED.bbbb", "fc-UNKNOWN.cccc"):
        with open(os.path.join(lease_dir, tok), "w") as fh:
            fh.write(stale_heartbeat)
    calls = {"fc-RUNNING": "running", "fc-FINISHED": "finished"}

    def resolver(snapshot_id, token):
        call_id = token.split(".")[0]
        state = calls.get(call_id)
        return state in ("finished", "failed", "cancelled")   # unknown → False → keep
    rec, blk = SN.reclaim(root, keep=[], lease_resolver=resolver)
    assert rec == [] and sorted(blk) == [f"{a.snapshot_id}/fc-RUNNING.aaaa", f"{a.snapshot_id}/fc-UNKNOWN.cccc"]
    assert not os.path.exists(os.path.join(lease_dir, "fc-FINISHED.bbbb"))
    calls["fc-RUNNING"] = "finished"
    os.remove(os.path.join(lease_dir, "fc-UNKNOWN.cccc"))  # explicit release
    rec2, blk2 = SN.reclaim(root, keep=[], lease_resolver=resolver)
    assert rec2 == [a.snapshot_id] and not blk2


def test_concurrent_ingest_during_build_does_not_block_or_tear(tmp_path):
    root = _setup(tmp_path)
    conn_seq = DB.connect(DB.db_path(root), readonly=True)
    head_before = conn_seq.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0]
    conn_seq.close()
    errors = []

    def ingest_more():
        try:
            R.write_records(os.path.join(root, "records", "late", "000001-late.jsonl"),
                            [synth.rec("si_text", {"si_id": "late_si", "si_text_sha256": "e" * 64, "si_kind": "irrelevant_control",
                                                   "source_uri": "synthetic://x"}, synth.t(5000))])
            H.ingest_once(root, workers=0, min_free=0)
        except Exception as exc:  # pragma: no cover
            errors.append(exc)
    t = threading.Thread(target=ingest_more)
    t.start()
    info = _build(root)
    t.join()
    assert not errors
    assert info.ingest_seq in (head_before, head_before + 1)
    with SN.snapshot_open(root) as snap:
        assert snap.manifest["ingest_seq"] == info.ingest_seq


def test_retention_one_once_readers_release(tmp_path):
    root = _setup(tmp_path)
    a = _build(root)
    with SN.snapshot_open(root):
        b = _build(root)
    assert os.path.isdir(a.dir)
    c = _build(root)
    assert set(c.reclaimed) >= {a.snapshot_id, b.snapshot_id}
    dirs = [d for d in os.listdir(SN.snapshots_dir(root)) if not d.startswith(".")]
    assert dirs == [c.snapshot_id]


def test_free_space_precondition_and_push_to_a_local_destination(tmp_path, monkeypatch):
    root = _setup(tmp_path)
    monkeypatch.setattr(SN, "free_bytes", lambda path: 1024 ** 3)      # a controlled reading: 1 GB free, whatever the host has
    ok, free, need = SN.free_space_precondition(root, DB.db_path(root), floor_bytes=2 * 1024 ** 3)
    assert not ok and free == 1024 ** 3 and need >= 2 * 1024 ** 3
    with pytest.raises(SN.SnapshotError) as bad:
        SN.build(root, DB.db_path(root), check_free_space=True)         # the default 2 GB floor is above the 1 GB reading
    assert "free space" in str(bad.value)
    assert SN.free_space_precondition(root, DB.db_path(root), floor_bytes=0)[0]   # and a floor under the reading passes
    # a run-level set needs the floor alone: the 0.4 × DB term is a full set's
    assert SN.free_space_precondition(root, DB.db_path(root), floor_bytes=0, scope="runlevel") == (True, 1024 ** 3, 0)
    # push: a local destination receives the directory, commits it and points CURRENT at it
    info = _build(root)
    dest = str(tmp_path / "dest")
    results = SN.push(root, info.snapshot_id, [{"name": "cluster_a", "kind": "local", "path": dest}])
    assert results[0]["ok"] and results[0]["error"] is None and SN.pointer(dest) == info.snapshot_id
    assert results[0]["remote_manifest_sha"] == R.file_sha256(os.path.join(info.dir, "MANIFEST.json"))
    with SN.snapshot_open(dest) as snap:
        assert snap.parquet("run").num_rows == 2


def test_snapshot_record_payload_ingests(tmp_path):
    root = _setup(tmp_path)
    info = _build(root)
    H.write_harvester_record(root, "snapshot", SN.snapshot_record_payload(info))
    H.ingest_once(root, workers=0, min_free=0)
    conn = DB.connect(DB.db_path(root), readonly=True)
    row = conn.execute("SELECT snapshot_id, head_ingest_seq FROM snapshot").fetchone()
    assert row[0] == info.snapshot_id and row[1] == info.ingest_seq
    conn.close()


def test_footprint_watcher_samples_peak(tmp_path):
    root = _setup(tmp_path)
    with SN.footprint_watcher(root, interval_s=0.05) as w:
        _build(root)
    assert w.peak > 0 and w.samples and set(w.peak_breakdown) >= {DB.db_path(root)}
    assert w.peak_rss_bytes > 0   # peak resident memory is sampled beside the disk bytes (the gate must see it)
