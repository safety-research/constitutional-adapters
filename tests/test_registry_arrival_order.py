"""``ingest_seq`` is the dense rank of record FILES in ARRIVAL order — ``(arrival_utc, first_written_utc, sha,
uri)`` — where ``arrival_utc`` is the live registry's clock at the file's FIRST fold, strictly after the head's. A file
the registry folds is always head + 1 whatever its producer stamp, nothing already ranked moves, the cycle's dense
pass has no work, and "everything above a head a reader has seen is new" holds. The order is in the log: the cycle
writes a ``file_arrival`` fact per file, so a rebuild reproduces every rank; a file with no fact (never folded live,
or the harvester's own) takes its ``first_written_utc``, and a log with no facts replays in the legacy order.
"""
from __future__ import annotations

import json
import os
import shutil
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import synth

A_STAMP, B_STAMP = synth.t(500), synth.t(600)
C_STAMP = synth.t(500 - 86400)                 # a day OLDER than A's producer stamp
NOW1, NOW2, NOW3 = synth.t(9000), synth.t(9001), synth.t(9002)


def _root(tmp_path, name="reg"):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _si(i, stamp):
    return synth.rec("si_text", {"si_id": f"si_{i}", "si_text_sha256": ("%064x" % (0x100 + i)), "si_kind": "irrelevant_control",
                                 "source_uri": "synthetic://x"}, stamp)


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _ingest(root, now):
    return H.ingest_once(root, workers=0, min_free=0, now=now)


def _metrics(root):
    path = os.path.join(H.state_dir(root), "metrics.jsonl")
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _ledger(conn):
    return {r[0]: dict(sha=r[1], first=r[2], arrival=r[3], n=r[4], seq=r[5]) for r in
            conn.execute("SELECT file_uri, file_sha256, first_written_utc, arrival_utc, n_records, ingest_seq FROM record_ingest").fetchall()}


def _seq_rows(conn):
    """Every seq-bearing table, whole rows, plus the ledger — the state a rank change would show up in."""
    out = {"record_ingest": sorted(tuple(r) for r in conn.execute(f"SELECT {H.LEDGER_ORDER_SQL}, n_records, ingest_seq FROM record_ingest"))}
    for t in H.SEQ_TABLES:
        out[t] = sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}").fetchall())
    return out


def _plus_ms(stamp, ms):
    return R.epoch_ms_to_utc(R.utc_to_epoch_ms(stamp) + ms)


def _a_b_then_older_c(tmp_path):
    """Cycle 1 folds A and B; cycle 2 folds C, stamped a day before A."""
    root = _root(tmp_path)
    _write(root, "early", "000001-a.jsonl", [_si(1, A_STAMP)])
    _write(root, "early", "000002-b.jsonl", [_si(2, B_STAMP)])
    rep1 = _ingest(root, NOW1)
    with DB.reading(DB.db_path(root)) as conn:
        before = _seq_rows(conn)
        head_before = conn.execute("SELECT MAX(ingest_seq) FROM record_ingest").fetchone()[0]
    _write(root, "late", "000001-c.jsonl", [_si(3, C_STAMP)])
    rep2 = _ingest(root, NOW2)
    return root, rep1, rep2, before, head_before


def test_a_late_stamped_file_is_head_plus_one_and_nothing_moves(tmp_path):
    """(a) C, stamped a day older than A, takes head + 1; every existing row of every seq-bearing table keeps its
    ``ingest_seq``; the cycle's ``rerank`` metric reports 0 rows; C's ``arrival_utc`` is the fold clock."""
    root, rep1, rep2, before, head_before = _a_b_then_older_c(tmp_path)
    # `ok` counts the folded arrival facts too (two for A and B, one for C); the files' own records are 2 and 1
    assert rep1.ok - rep1.arrival_records == 2 and rep2.ok - rep2.arrival_records == 1
    assert rep2.reranks == 0 and rep2.rerank_rows == 0
    with DB.reading(DB.db_path(root)) as conn:
        led = _ledger(conn)
        after = _seq_rows(conn)
        c = led["records/late/000001-c.jsonl"]
        assert c["seq"] == head_before + 1 and c["first"] == C_STAMP and c["arrival"] == NOW2
        # A took the cycle's clock; B, folded in the same cycle, one millisecond after it (strictly after the head)
        assert led["records/early/000001-a.jsonl"]["arrival"] == NOW1
        assert led["records/early/000002-b.jsonl"]["arrival"] == _plus_ms(NOW1, 1)
        assert led["records/early/000001-a.jsonl"]["seq"] == 1 and led["records/early/000002-b.jsonl"]["seq"] == 2
        # the ledger is dense and ordered by arrival, whatever the producer stamps say
        order = [r[0] for r in conn.execute(f"SELECT file_uri FROM record_ingest ORDER BY {H.LEDGER_ORDER_SQL}").fetchall()]
        seqs = [led[u]["seq"] for u in order]
        assert seqs == list(range(1, len(order) + 1))
        assert order.index("records/late/000001-c.jsonl") > order.index("records/early/000002-b.jsonl")
    # nothing that existed before cycle 2 moved: every pre-existing row is present unchanged
    for t, rows in before.items():
        assert set(rows) <= set(after[t]), t
    rerank2 = [m for m in _metrics(root) if m["kind"] == "rerank" and m["utc"] == NOW2]
    assert len(rerank2) == 1 and rerank2[0]["rows"] == 0


def test_the_cycle_writes_one_arrival_fact_per_file_and_folds_it(tmp_path):
    """The order is in the log: each cycle writes ONE ``file_arrival`` file under ``records/harvester/<day>/`` carrying a
    fact per file it folded, folds it in the same cycle, and counts it apart from the run_ingest / ruling facts."""
    root, rep1, rep2, _, _ = _a_b_then_older_c(tmp_path)
    assert rep1.arrival_facts == 2 and rep1.arrival_records == 2 and rep1.harvester_records == 0
    assert rep2.arrival_facts == 1 and rep2.arrival_records == 1 and rep2.harvester_records == 0
    hv = os.path.join(root, "records", H.HARVESTER_SUBDIR)
    facts_files = sorted(os.path.join(d, f) for d, _, fs in os.walk(hv) for f in fs if f.startswith(H.ARRIVAL_FILE_PREFIX))
    assert len(facts_files) == 2
    recs = [rec for p in facts_files for _, rec, _ in R.read_record_file(p)]
    assert {r["record_type"] for r in recs} == {"file_arrival"}
    by_uri = {r["payload"]["file_uri"]: r["payload"] for r in recs}
    with DB.reading(DB.db_path(root)) as conn:
        led = _ledger(conn)
        table = {r[0]: (r[1], r[2]) for r in conn.execute("SELECT file_uri, file_sha256, arrival_utc FROM file_arrival").fetchall()}
    assert set(by_uri) == {"records/early/000001-a.jsonl", "records/early/000002-b.jsonl", "records/late/000001-c.jsonl"}
    for uri, p in by_uri.items():
        assert p["arrival_utc"] == led[uri]["arrival"] and p["file_sha256"] == led[uri]["sha"]
        assert table[uri] == (led[uri]["sha"], led[uri]["arrival"])
    # the facts files are harvester-originated: ordered by their own stamp (= the arrival), stamped strictly after the
    # head their cycle left, and they carry no fact about themselves
    for p in facts_files:
        uri = os.path.relpath(p, root)
        assert H.is_harvester_file(uri) and uri not in by_uri
        assert led[uri]["arrival"] == led[uri]["first"]
    stamps = {led[os.path.relpath(p, root)]["first"] for p in facts_files}
    assert stamps == {_plus_ms(NOW1, 2), _plus_ms(NOW2, 1)}              # A at NOW1, B at + 1 ms, the facts at + 2; C at NOW2, facts + 1
    # an idle cycle writes nothing
    rep3 = _ingest(root, NOW3)
    assert rep3.arrival_facts == 0 and not rep3.changed and rep3.files == 0


def test_re_ingesting_a_file_keeps_its_arrival(tmp_path):
    """(b) A file read again -- the same bytes -- keeps its first arrival; other bytes at its uri are REFUSED by name and the ledger row
    stands whole (an ingested record file is immutable; before, the appended file re-keyed the row's sha and first stamp and its new record
    folded, which this test pinned until 2026-10-06)."""
    root, _, _, _, _ = _a_b_then_older_c(tmp_path)
    c_uri = "records/late/000001-c.jsonl"
    with DB.reading(DB.db_path(root)) as conn:
        c0 = _ledger(conn)[c_uri]
    # the same bytes again: forget the delivery so the cycle re-reads the file
    conn = DB.connect(DB.db_path(root))
    conn.execute("DELETE FROM record_delivery WHERE file_uri = ?", (c_uri,))
    conn.commit()
    conn.close()
    rep = _ingest(root, NOW3)
    assert rep.files == 1 and rep.duplicates == 1 and rep.reranks == 0 and rep.arrival_facts == 0
    with DB.reading(DB.db_path(root)) as conn:
        c1 = _ledger(conn)[c_uri]
    assert c1 == c0
    # changed bytes (a record appended, stamped older still): the file is refused by name as a replacement of an ingested record file;
    # the ledger row -- sha, first stamp, arrival, rank, count -- stands exactly, no fact is added and the appended record never folds
    path = os.path.join(root, c_uri)
    R.write_records(path, [_si(4, synth.t(-86400 * 2))])
    rep = _ingest(root, synth.t(9003))
    assert rep.ok == 0 and rep.refused == 1 and rep.files_replaced_refused == 1 and rep.reranks == 0 and rep.files == 2   # + the refusal-fact file
    assert len(rep.refusals) == 1 and rep.refusals[0].startswith(f"{c_uri}: replaced: ") and c0["sha"] in rep.refusals[0]
    with DB.reading(DB.db_path(root)) as conn:
        c2 = _ledger(conn)[c_uri]
        assert conn.execute("SELECT COUNT(*) FROM file_arrival WHERE file_uri = ?", (c_uri,)).fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM si_text WHERE si_id = 'si_4'").fetchone()[0] == 0
        assert tuple(conn.execute("SELECT record_uri, record_type FROM record_refusal WHERE record_sha256 = ?", (R.file_logical_sha(path),)).fetchone()) == (c_uri, None)
    assert c2 == c0


def _copy_log(root, dest, drop_prefix=None):
    for f in R.iter_record_files(os.path.join(root, "records")):
        if drop_prefix and os.path.basename(f).startswith(drop_prefix):
            continue
        rel = os.path.relpath(f, os.path.join(root, "records"))
        dest_f = os.path.join(dest, "records", rel)
        os.makedirs(os.path.dirname(dest_f), exist_ok=True)
        shutil.copy2(f, dest_f)


def test_rebuild_reproduces_every_rank_from_the_arrival_facts(tmp_path):
    """(c) A rebuild of the root's log gives the IDENTICAL ``ingest_seq`` on every row of every seq-bearing table and
    the identical ledger (arrival included), though the replay folds the files in producer-stamp order."""
    root, _, _, _, _ = _a_b_then_older_c(tmp_path)
    _write(root, "late", "000002-d.jsonl", [_si(5, synth.t(700))])           # a fourth cycle, one more file
    _ingest(root, NOW3)
    copy = str(tmp_path / "copy")
    _copy_log(root, copy)
    out = str(tmp_path / "rebuilt.sqlite")
    rep = H.rebuild(copy, out, workers=0)
    assert rep.reranks > 0                        # the replay folds C (oldest stamp) first, then A and B whose arrivals precede C's: inversions seen
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        assert _seq_rows(a) == _seq_rows(b)
        assert _ledger(a) == _ledger(b)
        diff = H.verify_equal(a, b)
        assert not {k: v for k, v in diff.items() if v}, diff
    finally:
        a.close()
        b.close()


def test_rebuild_of_a_log_without_facts_orders_by_stamp_sha_uri(tmp_path):
    """(d) A log the live registry never folded has no ``file_arrival`` facts: the replay takes ``arrival_utc =
    first_written_utc`` for every file — the legacy order ``(first_written_utc, sha, uri)`` exactly, C first."""
    root = _root(tmp_path, "fresh")
    _write(root, "early", "000001-a.jsonl", [_si(1, A_STAMP)])
    _write(root, "early", "000002-b.jsonl", [_si(2, B_STAMP)])
    _write(root, "late", "000001-c.jsonl", [_si(3, C_STAMP)])
    assert H.read_arrival_facts(root) == {}
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    with DB.reading(out) as conn:
        led = _ledger(conn)
        assert conn.execute("SELECT COUNT(*) FROM file_arrival").fetchone()[0] == 0
    assert all(v["arrival"] == v["first"] for v in led.values())
    expect = sorted(led, key=lambda u: (led[u]["first"], led[u]["sha"], u))
    assert [led[u]["seq"] for u in expect] == [1, 2, 3]
    assert expect[0] == "records/late/000001-c.jsonl"
    # the same log with its facts REMOVED replays in that order too — and differs from the live order it came from
    live, _, _, _, _ = _a_b_then_older_c(tmp_path)
    stripped = str(tmp_path / "stripped")
    _copy_log(live, stripped, drop_prefix=H.ARRIVAL_FILE_PREFIX)
    assert H.read_arrival_facts(stripped) == {}
    out2 = str(tmp_path / "rebuilt2.sqlite")
    H.rebuild(stripped, out2, workers=0)
    with DB.reading(out2) as conn:
        led2 = _ledger(conn)
    assert led2["records/late/000001-c.jsonl"]["seq"] == 1
    with DB.reading(DB.db_path(live)) as conn:
        assert _ledger(conn)["records/late/000001-c.jsonl"]["seq"] == 4      # A, B, their facts file, then C


def test_read_arrival_facts_keeps_the_earliest_and_skips_malformed(tmp_path):
    root = _root(tmp_path)
    day = os.path.join(root, "records", H.HARVESTER_SUBDIR, "2026-01-01")
    prod = H.harvester_producer(root)
    good = R.make_record("file_arrival", {"file_uri": "records/x/1.jsonl", "file_sha256": "a" * 64, "arrival_utc": synth.t(20)}, prod, "id-1", synth.t(20))
    earlier = R.make_record("file_arrival", {"file_uri": "records/x/1.jsonl", "file_sha256": "a" * 64, "arrival_utc": synth.t(10)}, prod, "id-2", synth.t(21))
    bad = R.make_record("file_arrival", {"file_uri": "records/x/2.jsonl", "file_sha256": "b" * 64, "arrival_utc": "not a stamp"}, prod, "id-3", synth.t(22))
    other = R.make_record("run_ingest", {"run_id": "r", "ingested_utc": synth.t(23)}, prod, "id-4", synth.t(23))
    R.write_records(os.path.join(day, f"{H.ARRIVAL_FILE_PREFIX}aaaaaaaaaaaa.jsonl"), [good, bad])
    R.write_records(os.path.join(day, f"{H.ARRIVAL_FILE_PREFIX}bbbbbbbbbbbb.jsonl.gz"), [earlier])
    R.write_records(os.path.join(day, "run_ingest-cccccccccccc.jsonl"), [other])
    with open(os.path.join(day, f"{H.ARRIVAL_FILE_PREFIX}dddddddddddd.jsonl"), "w") as fh:
        fh.write("{not json\n")
    assert H.read_arrival_facts(root) == {"records/x/1.jsonl": synth.t(10)}


def test_harvester_originated_files_are_ordered_by_their_own_stamp_on_both_paths(tmp_path):
    """A file under ``records/harvester/`` takes its stamp as its arrival live and on replay, and no fact is written for
    it; one stamped below the head is the one case the dense pass still has work for — and the replay agrees."""
    root = _root(tmp_path)
    _write(root, "early", "000001-a.jsonl", [_si(1, A_STAMP)])
    _ingest(root, NOW1)
    _write(root, os.path.join(H.HARVESTER_SUBDIR, "2025-12-31"), "si_text-000000000001.jsonl", [_si(9, C_STAMP)])
    rep = _ingest(root, NOW2)
    assert rep.reranks == 1 and rep.rerank_rows > 0 and rep.arrival_facts == 0
    hv_uri = f"records/{H.HARVESTER_SUBDIR}/2025-12-31/si_text-000000000001.jsonl"
    with DB.reading(DB.db_path(root)) as conn:
        led = _ledger(conn)
        assert led[hv_uri]["arrival"] == C_STAMP == led[hv_uri]["first"] and led[hv_uri]["seq"] == 1
        assert conn.execute("SELECT COUNT(*) FROM file_arrival WHERE file_uri = ?", (hv_uri,)).fetchone()[0] == 0
    out = str(tmp_path / "rebuilt.sqlite")
    H.rebuild(root, out, workers=0)
    a = DB.connect(DB.db_path(root), readonly=True)
    b = DB.connect(out, readonly=True)
    try:
        assert _ledger(a) == _ledger(b) and _seq_rows(a) == _seq_rows(b)
    finally:
        a.close()
        b.close()


def test_the_harvester_stamps_its_facts_after_the_head_it_left(tmp_path):
    """``ingest_once`` stamps the cycle's facts at ``Ranker.next_arrival``: head + 1 ms when the cycle's arrivals ran
    past the clock, else the clock — so the facts files (ordered by their stamp) never sort below the files they follow."""
    root = _root(tmp_path)
    for i in range(5):
        _write(root, "early", f"{i:06d}-si.jsonl", [_si(i, synth.t(100 + i))])
    rep = _ingest(root, NOW1)
    assert rep.reranks == 0 and rep.rerank_rows == 0
    with DB.reading(DB.db_path(root)) as conn:
        led = _ledger(conn)
        arrivals = sorted(v["arrival"] for v in led.values())
    assert arrivals == [_plus_ms(NOW1, k) for k in range(6)]                 # five files, then the facts file at + 5
    conn = DB.connect(DB.db_path(root), readonly=True)
    try:
        rk = H.Ranker(conn)
        assert rk.next_arrival(NOW1) == _plus_ms(NOW1, 6)                     # the clock has not passed the head
        assert rk.next_arrival(NOW2) == NOW2                                   # it has
    finally:
        conn.close()


def test_a_database_at_the_old_ledger_shape_is_refused_by_the_writer(tmp_path):
    """(e) The schema-drift guard refuses a database built before arrival ranking (no ``arrival_utc``, no ``file_arrival``
    table) by name, pointing at the rebuild — the intended behaviour of the live writer until the one rebuild after landing."""
    path = str(tmp_path / "r.sqlite")
    DB.open_fresh(path).close()
    conn = DB.connect(path)
    conn.execute("DROP INDEX idx_record_ingest_order")
    conn.execute("ALTER TABLE record_ingest DROP COLUMN arrival_utc")
    conn.execute("DROP TABLE file_arrival")
    conn.commit()
    conn.close()
    with DB.reading(path) as ro:
        drift = DB.schema_drift(ro)
    assert "table record_ingest lacks column(s) arrival_utc" in drift and "table file_arrival is absent" in drift
    with pytest.raises(RuntimeError) as ei:
        DB.open_or_create(path)
    assert "record_ingest lacks column(s) arrival_utc" in str(ei.value) and "registry rebuild" in str(ei.value)


def test_schema_pins_for_the_arrival_columns():
    """(f) The ledger carries ``arrival_utc`` with the UTC-ms CHECK and the order index; ``file_arrival`` is a run-scale
    record-fed table (so it is in SEQ_TABLES and every snapshot roster) whose payload is exactly the three fields."""
    t = S.TABLE_BY_NAME["record_ingest"]
    assert t.column("arrival_utc").nn and t.column("arrival_utc").pattern == "UTC_MS"
    assert ("idx_record_ingest_order", ("arrival_utc", "first_written_utc", "file_sha256", "file_uri"), None) in t.indexes
    ddl = S.render_schema_sql()
    assert "arrival_utc TEXT NOT NULL CHECK (arrival_utc GLOB" in ddl
    fa = S.TABLE_BY_NAME["file_arrival"]
    assert fa.scale == "run" and "file_arrival" in H.SEQ_TABLES and "file_arrival" in S.RUN_SCALE_TABLES
    assert S.required_fields("file_arrival") == ["file_uri", "file_sha256", "arrival_utc"]
    assert R.RECORD_TYPE_TABLE["file_arrival"] == "file_arrival" and "file_arrival" in R.RECORD_TYPES
    assert H.LEDGER_ORDER_SQL == "arrival_utc, first_written_utc, file_sha256, file_uri"
    from sisteer.registry.migrations import apply_migrations
    conn = DB.connect_memory()                                                   # a fresh database enforces the CHECK
    apply_migrations(conn)
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO record_ingest(file_uri, record_file_id, file_sha256, first_written_utc, arrival_utc, n_records, ingest_seq) "
                     "VALUES ('f', 1, ?, '2026-01-01T00:00:00.000Z', 'yesterday', 1, 1)", ("0" * 64,))
    conn.close()
