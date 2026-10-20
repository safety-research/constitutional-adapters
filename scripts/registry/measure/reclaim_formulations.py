"""Measure the harvester's reclaim UPDATEs on a synthetic ``item_result`` (2026-09-29).

``harvest.Ingestor.reclaim`` marks the rows of a SUPERSEDED revision that its successor no longer carries (pass ii) and clears the
marks of a HEAD revision (pass i), in batches of RECLAIM_BATCH_ROWS, with ``UPDATE ... WHERE (pk) IN (SELECT pk FROM item_result i
WHERE i.source_revision_id = ? ... NOT EXISTS (SELECT 1 FROM item_result h WHERE <same key> AND h.source_revision_id = ?) LIMIT ?)``.
The question: does that correlated NOT EXISTS scan the table per batch, and is a formulation that computes the key set once (EXCEPT)
and updates by primary key cheaper? This script builds the real ``item_result`` DDL (schema.sql) in a scratch database, fills it with
N rows over two revisions (A superseded by B, half their keys shared), and times both passes under both formulations, reporting wall
time, batches, WAL bytes written (autocheckpoint off) and the query plans. ``item_result`` is WITHOUT ROWID, so every
alternative addresses rows by primary key.

    python scripts/registry/measure/reclaim_formulations.py [--rows 1000000] [--batch 10000] [--out /tmp/reclaim_formulations.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "..", "..", "..", "src", "sisteer", "registry", "schema.sql")
PK = ("run_id", "item_id", "metric_id", "trial_index", "sample_index", "turn_index", "judgment_id", "source_revision_id")
KEY = [c for c in PK if c != "source_revision_id"]
COLS = ", ".join(PK)
SAME = " AND ".join(f"h.{c} = i.{c}" for c in KEY)


def item_result_ddl() -> list:
    sql = open(SCHEMA, encoding="utf-8").read()
    m = re.search(r"CREATE TABLE item_result \(.*?\n\)( WITHOUT ROWID)?;", sql, re.DOTALL)
    assert m, "item_result DDL not found"
    table = re.sub(r"REFERENCES \w+\(\w+\)", "", m.group(0))          # no parents in the scratch database (WITHOUT ROWID kept: the PK is the key)
    indexes = re.findall(r"CREATE INDEX idx_item_result_\w+ ON item_result \([^;]*;", sql)
    return [table] + indexes


def build(path: str, rows: int) -> None:
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executescript("\n".join(item_result_ddl()))
    half = rows // 2
    quarter = rows // 4

    def row(rev: str, i: int):
        return ("run-1", f"item-{i:08d}", "asr", 0, 0, -1, "judgment-1", rev, 0.5, None, None, "2026-01-01T00:00:00.000Z", None, None,
                1767225600000 + i, bytes([i % 251] * 32), 1, 1)
    cols = ("run_id, item_id, metric_id, trial_index, sample_index, turn_index, judgment_id, source_revision_id, value, label, api_pool, "
            "created_utc, judged_utc, withdrawn_by_revision_id, written_utc, record_sha256, record_file_id, ingest_seq")
    t0 = time.monotonic()
    conn.execute("BEGIN")
    conn.executemany(f"INSERT INTO item_result ({cols}) VALUES ({','.join('?' * 18)})", (row("A", i) for i in range(half)))
    conn.executemany(f"INSERT INTO item_result ({cols}) VALUES ({','.join('?' * 18)})", (row("B", i) for i in range(quarter, quarter + half)))
    conn.execute("COMMIT")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    print(f"built {rows} rows (A: {half}, B: {half}, shared keys: {half - quarter}) in {time.monotonic() - t0:.1f} s", flush=True)


def wal_bytes(path: str) -> int:
    try:
        return os.path.getsize(path + "-wal")
    except OSError:
        return 0


def timed(conn, path, label, fn):
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    w0 = wal_bytes(path)
    t0 = time.monotonic()
    out = fn()
    dt = time.monotonic() - t0
    w1 = wal_bytes(path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    res = {"label": label, "seconds": round(dt, 3), "wal_mib": round((w1 - w0) / 1048576, 1), **out}
    print(json.dumps(res), flush=True)
    return res


def plan(conn, sql, params):
    return [r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()]


def mark_current(conn, rev, to_rev, batch):
    n_batches = 0
    total = 0
    while True:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(
            f"UPDATE item_result SET withdrawn_by_revision_id = ? WHERE ({COLS}) IN "
            f"(SELECT {COLS} FROM item_result i WHERE i.source_revision_id = ? AND i.withdrawn_by_revision_id IS NULL "
            f" AND NOT EXISTS (SELECT 1 FROM item_result h WHERE {SAME} AND h.source_revision_id = ?) LIMIT ?)", (to_rev, rev, to_rev, batch))
        n = cur.rowcount
        conn.execute("COMMIT")
        n_batches += 1
        total += n
        if n < batch:
            break
    return {"batches": n_batches, "rows": total}


def mark_except_pk(conn, rev, to_rev, batch):
    """The key set once (EXCEPT over the revision index), then bounded batches of primary-key UPDATEs."""
    t0 = time.monotonic()
    keys = conn.execute(f"SELECT {', '.join(KEY)} FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NULL "
                        f"EXCEPT SELECT {', '.join(KEY)} FROM item_result WHERE source_revision_id = ?", (rev, to_rev)).fetchall()
    t_keys = time.monotonic() - t0
    where = " AND ".join(f"{c} = ?" for c in KEY) + " AND source_revision_id = ?"
    n_batches = 0
    for i in range(0, len(keys), batch):
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(f"UPDATE item_result SET withdrawn_by_revision_id = ? WHERE {where}", [(to_rev, *k, rev) for k in keys[i:i + batch]])
        conn.execute("COMMIT")
        n_batches += 1
    return {"batches": n_batches, "rows": len(keys), "keys_s": round(t_keys, 3)}


def unmark_current(conn, rev, batch):
    n_batches = 0
    total = 0
    while True:
        conn.execute("BEGIN IMMEDIATE")
        cur = conn.execute(f"UPDATE item_result SET withdrawn_by_revision_id = NULL WHERE ({COLS}) IN "
                           f"(SELECT {COLS} FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NOT NULL LIMIT ?)", (rev, batch))
        n = cur.rowcount
        conn.execute("COMMIT")
        n_batches += 1
        total += n
        if n <= 0:
            break
    return {"batches": n_batches, "rows": total}


def unmark_pk(conn, rev, batch):
    """The marked rows' primary keys once (the revision index, then the row's flag), then bounded batches of primary-key UPDATEs
    (``item_result`` is WITHOUT ROWID: the primary key IS the row's b-tree key)."""
    t0 = time.monotonic()
    keys = conn.execute(f"SELECT {COLS} FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NOT NULL", (rev,)).fetchall()
    t_keys = time.monotonic() - t0
    where = " AND ".join(f"{c} = ?" for c in PK)
    n_batches = 0
    for i in range(0, len(keys), batch):
        conn.execute("BEGIN IMMEDIATE")
        conn.executemany(f"UPDATE item_result SET withdrawn_by_revision_id = NULL WHERE {where}", keys[i:i + batch])
        conn.execute("COMMIT")
        n_batches += 1
    return {"batches": n_batches, "rows": len(keys), "keys_s": round(t_keys, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=1_000_000)
    ap.add_argument("--batch", type=int, default=10_000)
    ap.add_argument("--out", default="/tmp/reclaim_formulations.json")
    a = ap.parse_args()
    d = tempfile.mkdtemp(prefix="reclaim-measure-")
    path = os.path.join(d, "scratch.sqlite")
    build(path, a.rows)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA synchronous=NORMAL")
    results = {"rows": a.rows, "batch": a.batch, "plans": {}, "runs": []}
    results["plans"]["mark_current"] = plan(conn,
        f"UPDATE item_result SET withdrawn_by_revision_id = ? WHERE ({COLS}) IN (SELECT {COLS} FROM item_result i WHERE i.source_revision_id = ? "
        f"AND i.withdrawn_by_revision_id IS NULL AND NOT EXISTS (SELECT 1 FROM item_result h WHERE {SAME} AND h.source_revision_id = ?) LIMIT ?)",
        ("B", "A", "B", a.batch))
    results["plans"]["mark_except_keys"] = plan(conn,
        f"SELECT {', '.join(KEY)} FROM item_result WHERE source_revision_id = ? AND withdrawn_by_revision_id IS NULL "
        f"EXCEPT SELECT {', '.join(KEY)} FROM item_result WHERE source_revision_id = ?", ("A", "B"))
    results["plans"]["unmark_current"] = plan(conn,
        f"UPDATE item_result SET withdrawn_by_revision_id = NULL WHERE ({COLS}) IN (SELECT {COLS} FROM item_result WHERE source_revision_id = ? "
        f"AND withdrawn_by_revision_id IS NOT NULL LIMIT ?)", ("A", a.batch))
    for k, v in results["plans"].items():
        print(k, "->", " | ".join(v), flush=True)
    # pass (ii) mark: A superseded by B -- the current formulation, then the EXCEPT + PK one (marks reset in between)
    results["runs"].append(timed(conn, path, "mark_current", lambda: mark_current(conn, "A", "B", a.batch)))
    conn.execute("UPDATE item_result SET withdrawn_by_revision_id = NULL WHERE source_revision_id = 'A'")
    results["runs"].append(timed(conn, path, "mark_except_pk", lambda: mark_except_pk(conn, "A", "B", a.batch)))
    # pass (i) unmark: A becomes the head again -- the current formulation, then the rowid one
    results["runs"].append(timed(conn, path, "unmark_current", lambda: unmark_current(conn, "A", a.batch)))
    conn.execute("UPDATE item_result SET withdrawn_by_revision_id = 'B' WHERE source_revision_id = 'A' AND item_id < ?", (f"item-{a.rows // 4:08d}",))
    results["runs"].append(timed(conn, path, "unmark_pk", lambda: unmark_pk(conn, "A", a.batch)))
    conn.close()
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("written", a.out, "scratch db at", path, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
