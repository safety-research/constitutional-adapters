"""The live harvester cycle ranks its files provisionally per chunk and makes them dense ONCE at the end of the
cycle (``Ingestor(defer_ranks=True)`` + one ``finalize_ranks`` transaction logged as ``kind = "rerank"``), the way a
replay already did. A dense rank is a pure function of the ledger's keys, so the ranks written are the same as the
eager per-chunk path's; only WHEN they are made dense changes. Under the producer-stamp order every pulled file was a
late arrival on the live instance and the eager path rewrote every seq-bearing table once per 50-file chunk — 376–759 s
per chunk on the live database. The ledger is now in ARRIVAL order, so a plain file can no longer sort below the
head at all; the one file that still can is a harvester-originated one (``records/harvester/…``, ordered by its own
stamp on every path) stamped below the head — the late files here are written there, so the inversion the deferral
exists for is still exercised. Also: the end of a writer pass attempts one ``PRAGMA wal_checkpoint(TRUNCATE)`` and
logs SQLite's triple, never failing the cycle on it.
"""
from __future__ import annotations

import json
import os

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import records as R
from sisteer.registry import synth


def _root(tmp_path, name):
    root = str(tmp_path / name)
    H.ensure_layout(root)
    return root


def _si(i):
    return synth.rec("si_text", {"si_id": f"si_{i}", "si_text_sha256": ("%064x" % (0x100 + i)), "si_kind": "irrelevant_control",
                                 "source_uri": "synthetic://x"}, synth.t(i))


def _write(root, sub, name, recs):
    return R.write_records(os.path.join(root, "records", sub, name), recs)


def _metrics(root):
    path = os.path.join(H.state_dir(root), "metrics.jsonl")
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _state(root):
    """The ledger and every seq-bearing table, whole rows, in a key order that does not depend on ingest_seq."""
    out = {}
    with DB.reading(DB.db_path(root)) as conn:
        out["record_ingest"] = [tuple(r) for r in conn.execute("SELECT file_uri, file_sha256, first_written_utc, arrival_utc, n_records, "
                                                               "ingest_seq FROM record_ingest ORDER BY file_uri").fetchall()]
        for t in H.SEQ_TABLES:
            out[t] = sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}").fetchall())
        dense = [r[0] for r in conn.execute(f"SELECT ingest_seq FROM record_ingest ORDER BY {H.LEDGER_ORDER_SQL}").fetchall()]
    return out, dense


def _two_cycles_with_late_files(root, n_late):
    """Cycle 1 ranks files stamped t500..; cycle 2 brings ``n_late`` HARVESTER-originated files stamped EARLIER — a
    file under ``records/harvester/`` is ordered by its own stamp, so every one sorts below the ledger's head and
    every chunk of cycle 2 sees an inversion."""
    for i in range(3):
        _write(root, "early", f"{i:06d}-si.jsonl", [_si(500 + i)])
    rep1 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(9000))
    for i in range(n_late):
        _write(root, os.path.join(H.HARVESTER_SUBDIR, "late"), f"{i:06d}-si.jsonl", [_si(10 + i)])
    rep2 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(9001))
    return rep1, rep2


def test_deferred_live_cycle_ranks_equal_the_eager_path_and_emit_one_rerank_per_cycle(tmp_path, monkeypatch):
    """(a) The same log ingested by the deferred live path and by the eager per-chunk path gives the same ledger and the
    same ``ingest_seq`` on every row of every seq-bearing table; the deferred path emits exactly ONE ``rerank``
    transaction per cycle however many chunks the cycle had, and the ranks it leaves are dense."""
    monkeypatch.setattr(H, "FILES_PER_TRANSACTION", 1)          # five late files → five chunk transactions in cycle 2
    deferred = _root(tmp_path, "deferred")
    rep1, rep2 = _two_cycles_with_late_files(deferred, 5)
    # cycle 1's ok counts the three files' records plus their three arrival facts; the late harvester files write none
    assert rep1.ok == 6 and rep1.arrival_records == 3 and rep2.ok == 5 and rep2.arrival_records == 0
    assert rep2.reranks == 5 and rep2.rerank_rows > 0

    real_init = H.Ingestor.__init__

    def eager_init(self, *args, **kwargs):
        kwargs["defer_ranks"] = False
        real_init(self, *args, **kwargs)
    monkeypatch.setattr(H.Ingestor, "__init__", eager_init)
    eager = _root(tmp_path, "eager")
    erep1, erep2 = _two_cycles_with_late_files(eager, 5)
    assert erep2.ok == 5 and erep2.reranks == 5 and erep2.rerank_rows == 0   # the eager path had already made them dense

    d_state, d_dense = _state(deferred)
    e_state, e_dense = _state(eager)
    assert d_dense == list(range(1, 10)) and e_dense == d_dense       # 3 early + their file_arrival facts file + 5 late
    for t, rows in e_state.items():
        assert d_state[t] == rows, t

    kinds1 = [m["kind"] for m in _metrics(deferred) if m["utc"] == synth.t(9000)]
    kinds2 = [m["kind"] for m in _metrics(deferred) if m["utc"] == synth.t(9001)]
    assert kinds1.count("rerank") == 1 and kinds2.count("rerank") == 1
    assert kinds2.count("transaction") == 5
    rerank2 = [m for m in _metrics(deferred) if m["kind"] == "rerank" and m["utc"] == synth.t(9001)]
    assert rerank2[0]["rows"] == rep2.rerank_rows and rerank2[0]["files"] == 0
    # the rerank is the LAST rank-affecting transaction of the cycle (after every chunk)
    order2 = [m["kind"] for m in _metrics(deferred) if m["utc"] == synth.t(9001) and m["kind"] in ("transaction", "rerank")]
    assert order2[-1] == "rerank" and order2[:-1] == ["transaction"] * 5
    assert rep2.transactions == 6                                          # five chunks + the one dense pass


def test_no_non_final_finalize_does_work_in_a_live_cycle(tmp_path, monkeypatch):
    """The per-chunk ``finalize()`` calls stay in place (the replay's contract) and do nothing under deferral; the one
    ``final=True`` call carries the whole re-rank."""
    monkeypatch.setattr(H, "FILES_PER_TRANSACTION", 2)
    calls = []
    real = H.Ranker.finalize

    def spy(self, final=False):
        n = real(self, final=final)
        calls.append((self.defer, final, n))
        return n
    monkeypatch.setattr(H.Ranker, "finalize", spy)
    root = _root(tmp_path, "reg")
    _two_cycles_with_late_files(root, 4)
    assert calls and all(defer for defer, _, _ in calls)
    assert all(n == 0 for _, final, n in calls if not final)
    finals = [c for c in calls if c[1]]
    assert len(finals) == 2 and finals[0][2] == 0 and finals[1][2] > 0     # one per cycle; only cycle 2 had a late file


def test_live_ingestor_is_constructed_deferred(tmp_path, monkeypatch):
    """(b) ``ingest_once`` builds its ``Ingestor`` with ``defer_ranks=True``; ``rebuild`` keeps doing the same."""
    seen = []
    real_init = H.Ingestor.__init__

    def spy(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        seen.append((kwargs.get("defer_ranks"), self.ranker.defer))
    monkeypatch.setattr(H.Ingestor, "__init__", spy)
    root = _root(tmp_path, "reg")
    _write(root, "a", "000001-si.jsonl", [_si(1)])
    H.ingest_once(root, workers=0, min_free=0)
    assert seen == [(True, True)]
    with DB.reading(DB.db_path(root)) as conn:
        assert H.Ingestor(conn, root).ranker.defer is False                    # the constructor default is unchanged


def test_writer_pass_ends_with_a_wal_truncate_attempt_logged_as_a_metric(tmp_path):
    """The end of ``ingest_once`` attempts ``wal_checkpoint(TRUNCATE)`` once and logs SQLite's (busy, log, checkpointed)
    triple; with no reader holding a snapshot the WAL is folded back and truncated to nothing."""
    root = _root(tmp_path, "reg")
    _write(root, "a", "000001-si.jsonl", [_si(1)])
    conn = DB.open_or_create(DB.db_path(root))
    try:
        H.ingest_once(root, conn, workers=0, min_free=0, now=synth.t(9000))
        wal = DB.db_path(root) + "-wal"
        assert os.path.exists(wal) and os.path.getsize(wal) == 0
        assert int(conn.execute("PRAGMA busy_timeout").fetchone()[0]) == DB.BUSY_TIMEOUT_MS   # restored after the short wait
    finally:
        conn.close()
    lines = [m for m in _metrics(root) if m["kind"] == "wal_checkpoint"]
    assert len(lines) == 1
    assert lines[0]["busy"] == 0 and lines[0]["checkpointed"] >= 0 and lines[0]["log"] == lines[0]["checkpointed"]
    assert lines[0]["utc"] == synth.t(9000)


def test_wal_truncate_that_is_busy_or_failing_never_fails_the_cycle(tmp_path, monkeypatch):
    """A reader's snapshot leaves ``busy = 1`` (logged, cycle succeeds); an error from the pragma is logged as its own
    metric line and the cycle still returns its report."""
    root = _root(tmp_path, "reg")
    _write(root, "a", "000001-si.jsonl", [_si(1)])
    DB.open_or_create(DB.db_path(root)).close()                              # the file exists before the reader opens it
    reader = DB.connect(DB.db_path(root), readonly=True)
    try:
        reader.execute("BEGIN")
        reader.execute("SELECT COUNT(*) FROM record_ingest").fetchone()      # a snapshot held across the writer pass
        rep = H.ingest_once(root, workers=0, min_free=0, now=synth.t(9000))
    finally:
        reader.close()
    assert rep.ok == 2                                                       # the file's record + its arrival fact
    lines = [m for m in _metrics(root) if m["kind"] == "wal_checkpoint"]
    assert len(lines) == 1 and lines[0]["busy"] in (0, 1)

    import sqlite3

    def boom(conn, busy_timeout_ms=1000):
        raise sqlite3.OperationalError("synthetic checkpoint failure")
    monkeypatch.setattr(DB, "wal_checkpoint_truncate", boom)
    _write(root, "a", "000002-si.jsonl", [_si(2)])
    rep2 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(9001))
    assert rep2.ok == 2                                                      # the file's record + its arrival fact
    errs = [m for m in _metrics(root) if m["kind"] == "wal_checkpoint_error"]
    assert len(errs) == 1 and "synthetic" in errs[0]["error"] and errs[0]["utc"] == synth.t(9001)
