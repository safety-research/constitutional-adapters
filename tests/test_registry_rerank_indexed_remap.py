"""The dense re-rank rewrites the item tables through their revision index, never by a scan (2026-09-24).

``Ranker.finalize`` remaps ``ingest_seq`` in every seq-bearing table when a file sorted below the ledger's head. The item
tables have no index on ``ingest_seq``, so the remap read every row of ``item_result`` (11.5 M) and ``item_flags`` (6.4 M) to
move the rows of a handful of files: 729-1,029 s per pass, 13-20k s a day of write-lock time on the live database (the
2026-09-24 fold-time profile, specs: results-registry/fold-perf-profile.md). ``Ranker._remap_pairs`` finds the moved files'
rows through ``(source_revision_id, record_file_id)`` instead. These tests hold the two paths to the same rows and pin the
query plan.
"""
from __future__ import annotations

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


def _state(root):
    """Every seq-bearing table and the ledger, whole rows, in a key order that does not depend on ingest_seq."""
    out = {}
    with DB.reading(DB.db_path(root)) as conn:
        out["record_ingest"] = sorted(tuple(r) for r in conn.execute("SELECT file_uri, file_sha256, first_written_utc, arrival_utc, n_records, "
                                                                     "ingest_seq FROM record_ingest").fetchall())
        for t in H.SEQ_TABLES:
            out[t] = sorted(tuple(r) for r in conn.execute(f"SELECT * FROM {t}").fetchall())
    return out


def _cycles(root, indexed, monkeypatch):
    """Cycle 1 folds a corpus with item rows; cycle 2 brings harvester-originated files stamped EARLIER (ordered by their own
    stamp), so every ranked file -- the item files included -- moves down and the dense pass remaps the item tables."""
    monkeypatch.setattr(H, "INDEXED_SEQ_REMAP", indexed)
    synth.write_corpus(root, n_runs=2, n_items=25)
    rep1 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(9000))
    assert rep1.refused == 0 and rep1.blocked == 0, rep1.refusals
    for i in range(4):
        R.write_records(os.path.join(root, "records", H.HARVESTER_SUBDIR, "late", f"{i:06d}-si.jsonl"), [_si(10 + i)])
    rep2 = H.ingest_once(root, workers=0, min_free=0, now=synth.t(9001))
    assert rep2.refused == 0 and rep2.reranks >= 1 and rep2.rerank_rows > 0, (rep2.reranks, rep2.rerank_rows)
    return rep2


def test_indexed_remap_leaves_the_same_rows_as_the_scan_and_keeps_every_item_row_at_its_files_rank(tmp_path, monkeypatch):
    indexed = _root(tmp_path, "indexed")
    scanned = _root(tmp_path, "scanned")
    rep_i = _cycles(indexed, True, monkeypatch)
    rep_s = _cycles(scanned, False, monkeypatch)
    assert rep_i.rerank_rows == rep_s.rerank_rows
    st_i, st_s = _state(indexed), _state(scanned)
    for t, rows in st_s.items():
        assert st_i[t] == rows, t
    with DB.reading(DB.db_path(indexed)) as c:
        n_items = c.execute("SELECT COUNT(*) FROM item_result").fetchone()[0]
        assert n_items == 2 * 25
        # the schema's invariant the indexed path rests on: a row's ingest_seq is its producing file's ledger rank
        for t in H.REVISION_INDEXED_TABLES:
            off = c.execute(f"SELECT COUNT(*) FROM {t} x JOIN record_ingest r ON r.record_file_id = x.record_file_id "
                            "WHERE x.ingest_seq <> r.ingest_seq").fetchone()[0]
            assert off == 0, t
        dense = [r[0] for r in c.execute(f"SELECT ingest_seq FROM record_ingest ORDER BY {H.LEDGER_ORDER_SQL}")]
        assert dense == list(range(1, len(dense) + 1))
    assert set(H.REVISION_INDEXED_TABLES) >= {"item_result", "item_flags"}


def test_the_item_table_remap_is_an_index_search_not_a_scan(tmp_path):
    """The statement the dense pass issues for an item table searches ``idx_<table>_revision``; the scan it replaced read
    every row."""
    conn = DB.open_fresh(str(tmp_path / "r.sqlite"))
    ranker = H.Ranker(conn)
    with DB.transaction(conn):
        conn.execute("CREATE TEMP TABLE seqmap(old INTEGER, new INTEGER)")
        conn.execute("CREATE INDEX temp.idx_seqmap_old ON seqmap(old)")
        assert ranker._remap_pairs([1234567]) is True
        for t in ("item_result", "item_flags"):
            plan = " | ".join(r[3] for r in conn.execute(
                f"EXPLAIN QUERY PLAN UPDATE {t} SET ingest_seq = (SELECT new FROM temp.seqmap WHERE seqmap.old = {t}.ingest_seq) "
                f"WHERE (source_revision_id, record_file_id) IN (SELECT rev, rfid FROM temp.seqpairs) AND ingest_seq IN (SELECT old FROM temp.seqmap)"))
            assert f"SEARCH {t} USING INDEX idx_{t}_revision" in plan, plan
            scan = " | ".join(r[3] for r in conn.execute(
                f"EXPLAIN QUERY PLAN UPDATE {t} SET ingest_seq = (SELECT new FROM temp.seqmap WHERE seqmap.old = {t}.ingest_seq) "
                f"WHERE ingest_seq IN (SELECT old FROM temp.seqmap)"))
            assert f"SCAN {t}" in scan, scan
        # the pair set is (every revision, and the native '' tag) x (moved files)
        assert conn.execute("SELECT COUNT(*) FROM temp.seqpairs").fetchone()[0] == 1
        assert tuple(conn.execute("SELECT rev, rfid FROM temp.seqpairs").fetchone()) == ("", 1234567)
        # too many pairs: the scan path, equally exact
        assert ranker._remap_pairs(list(range(H.SEQ_REMAP_PAIRS_MAX + 1))) is False
        assert ranker._remap_pairs([]) is False
        conn.execute("DROP TABLE temp.seqmap")
    conn.close()
