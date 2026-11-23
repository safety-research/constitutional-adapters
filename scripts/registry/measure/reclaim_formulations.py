"""Measure the harvester's reclaim DELETE batches on a synthetic ``item_result`` (2026-10-19; the 2026-09-29 version timed the mark UPDATEs
the deletion replaced).

Since migration 0033 ``harvest.Ingestor.reclaim`` DELETES the rows of an accepted, superseded revision, carrier by carrier
(``record_file_id``), in batches of ``RECLAIM_BATCH_ROWS``: each batch is one transaction that selects the next rows of the carrier, ALL
columns, through ``idx_item_result_revision``; digests them (``prune.row_digest``, the additive ``D_fold`` / ``D_bind`` the deletion ledger
carries); deletes exactly those primary keys under ``EXISTS (SELECT 1 FROM source_revision s WHERE s.revision_id = ? AND s.accepted = 1 AND
s.superseded_by_revision_id IS NOT NULL)``; and requires the delete count to equal the selected count. The questions this script answers on
a scratch database built from the real DDL (``schema.sql``):

* what one batch costs in wall time and in WAL bytes -- lower the batch size if one batch writes more than 128 MB of WAL;
* how much of it is the digest pass;
* which DELETE formulation is cheapest: ``executemany`` of a primary-key DELETE per row, a temp table of the selected keys joined by
  ``(pk) IN (SELECT ...)``, or (the reference, with no exact digest of what it deletes) ``(pk) IN (SELECT pk ... LIMIT n)``.

The live table holds 12-20 copies of a visible key, one per revision, and they are NEIGHBOURS in the primary key (``source_revision_id``
is its last column): deleting one revision's rows touches a page per row or so, not a contiguous range. The build reproduces that --
``--revisions`` copies of every key, interleaved by the primary key, with ``--payload-bytes`` of filler per row (the live average row is
~375 bytes, dbstat 2026-10-19) -- and deletes the OLDEST revision's rows. ``item_result`` is WITHOUT ROWID: rows are addressed by
primary key. Every formulation runs on its own copy of the built file; the WAL is truncated before every batch, so each batch's WAL is
measured alone (autocheckpoint off).

    PYTHONPATH=$PWD/src python scripts/registry/measure/reclaim_formulations.py [--keys 100000] [--revisions 12] [--batch 10000] \\
        [--carrier-rows 50000] [--payload-bytes 300] [--out /tmp/reclaim_formulations.json]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sqlite3
import statistics
import tempfile
import time
from typing import Callable, Dict, List, Sequence, Tuple

from sisteer.registry import prune as P

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEMA = os.path.join(HERE, "..", "..", "..", "src", "sisteer", "registry", "schema.sql")
TABLE = "item_result"
PK = P.PK[TABLE]
PK_SQL = ", ".join(PK)
GUARD = P.DELETE_GUARD


def ddl() -> List[str]:
    """``item_result`` and its indexes from schema.sql (parents dropped: the scratch database has none), plus a minimal ``source_revision``
    for the visibility guard."""
    sql = open(SCHEMA, encoding="utf-8").read()
    m = re.search(r"CREATE TABLE item_result \(.*?\n\)( WITHOUT ROWID)?;", sql, re.DOTALL)
    assert m, "item_result DDL not found"
    table = re.sub(r"REFERENCES \w+\(\w+\)", "", m.group(0))
    indexes = re.findall(r"CREATE INDEX idx_item_result_\w+ ON item_result \([^;]*;", sql)
    revision = ("CREATE TABLE source_revision (revision_id TEXT PRIMARY KEY, accepted INTEGER NOT NULL, "
                "superseded_by_revision_id TEXT) WITHOUT ROWID;")
    return [table] + indexes + [revision]


def columns(conn: sqlite3.Connection) -> List[str]:
    return [str(r[1]) for r in conn.execute(f"PRAGMA table_info({TABLE})").fetchall()]


def build(path: str, keys: int, revisions: int, carrier_rows: int, payload_bytes: int) -> Dict[str, object]:
    """``revisions`` copies of ``keys`` item keys (one run, one judgment), revision ``r00`` the oldest; every revision but the newest is
    accepted and superseded. Rows of one revision are split into carriers of ``carrier_rows`` (``record_file_id``)."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.executescript("\n".join(ddl()))
    cols = columns(conn)
    revs = [f"r{i:02d}" for i in range(revisions)]
    filler = "x" * max(0, int(payload_bytes))       # carried in extra_json, the column the live rows' width varies in
    t0 = time.monotonic()
    conn.execute("BEGIN")
    for i, rev in enumerate(revs):
        conn.execute("INSERT INTO source_revision VALUES (?, 1, ?)", (rev, revs[i + 1] if i + 1 < len(revs) else None))

    def row(rev_i: int, k: int) -> Tuple[object, ...]:
        base = {"run_id": "run-1", "item_id": f"item-{k:08d}", "metric_id": "asr", "trial_index": 0, "sample_index": 0, "turn_index": -1,
                "judgment_id": "judgment-1", "source_revision_id": revs[rev_i], "value": (k % 7) / 7.0, "label": None, "api_pool": None,
                "extra_json": json.dumps({"pad": filler}),
                "created_utc": "2026-01-01T00:00:00.000Z", "judged_utc": None, "withdrawn_by_revision_id": None,
                "written_utc": 1767225600000 + rev_i * 1000 + k, "record_sha256": bytes([(k + rev_i) % 251] * 32),
                "record_file_id": 1000 + rev_i * 1000 + k // carrier_rows, "ingest_seq": 1 + rev_i * 1000 + k // carrier_rows}
        return tuple(base.get(c) for c in cols)
    ph = ",".join("?" * len(cols))
    for rev_i in range(len(revs)):
        conn.executemany(f"INSERT INTO {TABLE} ({', '.join(cols)}) VALUES ({ph})", (row(rev_i, k) for k in range(keys)))
    conn.execute("COMMIT")
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    pages = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    conn.close()
    out = {"rows": keys * len(revs), "keys": keys, "revisions": len(revs), "carrier_rows": carrier_rows, "payload_bytes": payload_bytes,
           "pages": pages, "page_size": page_size, "build_s": round(time.monotonic() - t0, 1)}
    print("built", json.dumps(out), flush=True)
    return out


def wal_bytes(path: str) -> int:
    try:
        return os.path.getsize(path + "-wal")
    except OSError:
        return 0


def carriers(conn: sqlite3.Connection, rev: str) -> List[int]:
    return [int(r[0]) for r in conn.execute(f"SELECT DISTINCT record_file_id FROM {TABLE} WHERE source_revision_id = ? ORDER BY 1", (rev,))]


def select_batch(conn: sqlite3.Connection, rev: str, carrier: int, batch: int) -> List[Tuple[object, ...]]:
    return conn.execute(f"SELECT * FROM {TABLE} INDEXED BY idx_item_result_revision WHERE source_revision_id = ? AND record_file_id = ? LIMIT ?",
                        (rev, carrier, batch)).fetchall()


def digest(rows: Sequence[Tuple[object, ...]]) -> float:
    """The batch's ``D_fold`` and ``D_bind`` (the ledger's two running digests) through the fold's own ``prune.row_digest``; returns the
    seconds it took."""
    t0 = time.monotonic()
    fold = P.Digest()
    bind = P.Digest()
    for r in rows:
        fold.add(P.row_digest(TABLE, r, exclude=P.FOLD_EXCLUDE))
        bind.add(P.row_digest(TABLE, r, exclude=P.BIND_EXCLUDE))
    return time.monotonic() - t0


def delete_executemany(conn: sqlite3.Connection, rev: str, carrier: int, batch: int, pk_pos: Sequence[int]) -> Dict[str, object]:
    rows = select_batch(conn, rev, carrier, batch)
    dig_s = digest(rows)
    where = " AND ".join(f"{c} = ?" for c in PK)
    before = conn.total_changes
    conn.executemany(f"DELETE FROM {TABLE} WHERE {where} AND {GUARD}", [tuple(r[i] for i in pk_pos) + (rev,) for r in rows])
    return {"selected": len(rows), "deleted": conn.total_changes - before, "digest_s": dig_s}


def delete_temp_table(conn: sqlite3.Connection, rev: str, carrier: int, batch: int, pk_pos: Sequence[int]) -> Dict[str, object]:
    rows = select_batch(conn, rev, carrier, batch)
    dig_s = digest(rows)
    conn.execute("DELETE FROM temp.reclaim_keys")
    conn.executemany(f"INSERT INTO temp.reclaim_keys ({PK_SQL}) VALUES ({','.join('?' * len(PK))})", [tuple(r[i] for i in pk_pos) for r in rows])
    cur = conn.execute(f"DELETE FROM {TABLE} WHERE ({PK_SQL}) IN (SELECT {PK_SQL} FROM temp.reclaim_keys) AND {GUARD}", (rev,))
    return {"selected": len(rows), "deleted": cur.rowcount, "digest_s": dig_s}


def delete_subquery_limit(conn: sqlite3.Connection, rev: str, carrier: int, batch: int, pk_pos: Sequence[int]) -> Dict[str, object]:
    """The reference: no exact record of what it deletes (no select, no digest) -- the floor the other two are compared with."""
    cur = conn.execute(f"DELETE FROM {TABLE} WHERE ({PK_SQL}) IN (SELECT {PK_SQL} FROM {TABLE} INDEXED BY idx_item_result_revision "
                       f"WHERE source_revision_id = ? AND record_file_id = ? LIMIT ?) AND {GUARD}", (rev, carrier, batch, rev))
    return {"selected": cur.rowcount, "deleted": cur.rowcount, "digest_s": 0.0}


FORMULATIONS: Dict[str, Callable[..., Dict[str, object]]] = {
    "executemany_pk": delete_executemany, "temp_table_in": delete_temp_table, "subquery_limit_reference": delete_subquery_limit}


def run_formulation(template: str, workdir: str, name: str, rev: str, batch: int) -> Dict[str, object]:
    path = os.path.join(workdir, f"{name}.sqlite")
    shutil.copyfile(template, path)
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA wal_autocheckpoint=0")
    conn.execute(f"CREATE TEMP TABLE reclaim_keys ({PK_SQL}, PRIMARY KEY ({PK_SQL})) WITHOUT ROWID")
    cols = columns(conn)
    pk_pos = [cols.index(c) for c in PK]
    fn = FORMULATIONS[name]
    batches: List[Dict[str, object]] = []
    t_all = time.monotonic()
    for carrier in carriers(conn, rev):
        while True:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            w0 = wal_bytes(path)
            t0 = time.monotonic()
            conn.execute("BEGIN IMMEDIATE")
            res = fn(conn, rev, carrier, batch, pk_pos)
            if res["deleted"] != res["selected"]:
                conn.execute("ROLLBACK")
                raise SystemExit(f"{name}: deleted {res['deleted']} of {res['selected']} selected rows -- the batch refuses")
            conn.execute("COMMIT")
            dt = time.monotonic() - t0
            if not res["selected"]:
                break
            batches.append({"carrier": carrier, "rows": res["selected"], "seconds": round(dt, 4), "digest_s": round(float(res["digest_s"]), 4),
                            "wal_bytes": wal_bytes(path) - w0})
            if int(res["selected"]) < batch:   # type: ignore[arg-type]
                break
    total_s = time.monotonic() - t_all
    left = conn.execute(f"SELECT COUNT(*) FROM {TABLE} WHERE source_revision_id = ?", (rev,)).fetchone()[0]
    conn.close()
    secs = [float(b["seconds"]) for b in batches] or [0.0]   # type: ignore[arg-type]
    wal = [int(b["wal_bytes"]) for b in batches] or [0]   # type: ignore[call-overload]
    rows = sum(int(b["rows"]) for b in batches)   # type: ignore[call-overload]
    p99 = statistics.quantiles(secs, n=100, method="inclusive")[98] if len(secs) >= 2 else secs[0]
    out = {"formulation": name, "batches": len(batches), "rows": rows, "rows_left": left, "total_s": round(total_s, 3),
           "rows_per_s": round(rows / total_s, 1) if total_s else None,
           "batch_s_p50": round(statistics.median(secs), 4), "batch_s_p99": round(p99, 4), "batch_s_max": round(max(secs), 4),
           "digest_share": round(sum(float(b["digest_s"]) for b in batches) / max(sum(secs), 1e-9), 3),   # type: ignore[arg-type]
           "wal_mib_per_batch_p50": round(statistics.median(wal) / 1048576, 2), "wal_mib_per_batch_max": round(max(wal) / 1048576, 2),
           "batches_over_128_mib": sum(1 for w in wal if w > 128 * 1048576)}
    print(json.dumps(out), flush=True)
    return out


def plans(template: str, rev: str, batch: int) -> Dict[str, List[str]]:
    conn = sqlite3.connect(f"file:{template}?mode=ro", uri=True)
    try:
        carrier = carriers(conn, rev)[0]
        where = " AND ".join(f"{c} = ?" for c in PK)
        return {
            "select_batch": [r[3] for r in conn.execute(f"EXPLAIN QUERY PLAN SELECT * FROM {TABLE} INDEXED BY idx_item_result_revision "
                                                        "WHERE source_revision_id = ? AND record_file_id = ? LIMIT ?", (rev, carrier, batch))],
            "delete_pk": [r[3] for r in conn.execute(f"EXPLAIN QUERY PLAN DELETE FROM {TABLE} WHERE {where} AND {GUARD}",
                                                     ("run-1", "item-00000000", "asr", 0, 0, -1, "judgment-1", rev, rev))],
            "delete_subquery_limit": [r[3] for r in conn.execute(
                f"EXPLAIN QUERY PLAN DELETE FROM {TABLE} WHERE ({PK_SQL}) IN (SELECT {PK_SQL} FROM {TABLE} INDEXED BY idx_item_result_revision "
                f"WHERE source_revision_id = ? AND record_file_id = ? LIMIT ?) AND {GUARD}", (rev, carrier, batch, rev))],
        }
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(description="Time the reclaim DELETE batches on a synthetic item_result (see the module doc).")
    ap.add_argument("--keys", type=int, default=100_000, help="item keys per revision")
    ap.add_argument("--revisions", type=int, default=12, help="copies of every key, one per revision (the oldest is deleted)")
    ap.add_argument("--batch", type=int, default=10_000, help="rows per DELETE batch (harvest.RECLAIM_BATCH_ROWS)")
    ap.add_argument("--carrier-rows", type=int, default=50_000, help="rows per carrier file (record_file_id)")
    ap.add_argument("--payload-bytes", type=int, default=300, help="filler per row, so a row is about as wide as a live one")
    ap.add_argument("--formulations", default=",".join(FORMULATIONS), help="comma-separated subset of " + ", ".join(FORMULATIONS))
    ap.add_argument("--out", default="/tmp/reclaim_formulations.json")
    ap.add_argument("--keep", action="store_true", help="keep the scratch directory")
    a = ap.parse_args()
    d = tempfile.mkdtemp(prefix="reclaim-measure-")
    template = os.path.join(d, "template.sqlite")
    results: Dict[str, object] = {"build": build(template, a.keys, a.revisions, a.carrier_rows, a.payload_bytes), "batch": a.batch,
                                  "plans": plans(template, "r00", a.batch), "runs": []}
    for k, v in results["plans"].items():   # type: ignore[union-attr]
        print(k, "->", " | ".join(v), flush=True)
    for name in [f.strip() for f in a.formulations.split(",") if f.strip()]:
        results["runs"].append(run_formulation(template, d, name, "r00", a.batch))   # type: ignore[union-attr]
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1)
    print("written", a.out, flush=True)
    if a.keep:
        print("scratch directory", d, flush=True)
    else:
        shutil.rmtree(d, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
