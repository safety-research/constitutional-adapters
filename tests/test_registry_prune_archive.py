"""The superseded-revision pruning's archive, restore, digests, cohort binding and handoff (prune_archive.py).

* the round trip: archive -> reclaim -> the deletion contract -> restore into a scratch copy equals the pre-prune item tables exactly,
  marks included (a marked deletable carrier comes back from the archive; a marked protected carrier is never touched);
* the row encoding's extremes and the archive verification (a flipped byte, a dropped row, a consistently re-summed tamper);
* cohort mode: a same-count change, a supersession after the manifest (then a delta archive), an ``ingest_seq``
  remap, a stop mid-carrier and its resumption, a remaining row changed between passes, a carrier deleted over three passes, and the
  explicit mode on a non-live file against ``--mode normal``;
* the handoff: a copied archive gates ``mode cohort`` only when every file is present and intact, the cluster's VERIFIED.json
  names its MANIFEST, and the local re-verification passes;
* the refusals, ``status``, ``verify-deleted``, ``mode normal``, the digests and the CLI (``rebuild --diff-out``, ``compare``).
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import sqlite3
import subprocess

import pytest

from sisteer.registry import cli
from sisteer.registry import db as DB
from sisteer.registry import equality as EQ
from sisteer.registry import harvest as H
from sisteer.registry import prune as P
from sisteer.registry import prune_archive as PA

import test_registry_prune_fold as F
from test_registry_core_fixes import _q, _store


# --------------------------------------------------------------------------------------------
# one world, built once: rev_a and rev_b accepted and superseded by rev_c, reclaim held (nothing deleted)
# --------------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def base(tmp_path_factory):
    d = tmp_path_factory.mktemp("prune_archive_base")
    root, store_root, ad = F._world(d / "w", mode="hold")
    rev_a = F._read(root, ad, 2000)
    _store(store_root, "s1", "store_1", F.LORA, [0.9, 0.9, 0.9, 0.7])
    rev_b = F._read(root, ad, 3000)
    _store(store_root, "s1", "store_1", F.LORA, [0.1, 0.2])
    rev_c = F._read(root, ad, 4000)
    assert F._head(root) == rev_c
    return {"dir": str(d / "w"), "revs": (rev_a, rev_b, rev_c)}


class W:
    def __init__(self, base, tmp_path, name="w"):
        self.dir = str(tmp_path / name)
        shutil.copytree(base["dir"], self.dir)
        self.root = os.path.join(self.dir, "reg")
        self.store_root = os.path.join(self.dir, "stores")
        self.rev_a, self.rev_b, self.rev_c = base["revs"]
        self.tmp = tmp_path

    def snap(self, name, src=None):
        """A consistent copy of the root's database (or of ``src``) at ``tmp/<name>.sqlite``."""
        out = str(self.tmp / f"{name}.sqlite")
        with DB.reading(src or DB.db_path(self.root)) as conn:
            conn.execute(f"VACUUM INTO '{out}'")
        return out


def _raw(path):
    conn = sqlite3.connect(path, isolation_level=None)
    return conn


def _count(path, table, rev=None, rfid=None):
    sql, args = f"SELECT COUNT(*) FROM {table}", []
    if rev is not None:
        sql += " WHERE source_revision_id = ?"
        args.append(rev)
        if rfid is not None:
            sql += " AND record_file_id = ?"
            args.append(rfid)
    with DB.reading(path) as conn:
        return conn.execute(sql, args).fetchone()[0]


def _groups(archive_dir):
    return {(g.table, g.revision_id, g.record_file_id): g for g in P.read_groups(archive_dir)}


_group_dicts = PA.group_dicts


def _archive(w, src, name="A", verify=True, **kw):
    out = str(w.tmp / name)
    m = PA.archive(src, w.root, out, **kw)
    if verify:
        PA.verify_archive(out, db_path=src, root=w.root)
    return out, m


def _first_carrier(path, table, rev):
    with DB.reading(path) as conn:
        return conn.execute(f"SELECT MIN(record_file_id) FROM {table} WHERE source_revision_id = ?", (rev,)).fetchone()[0]


ALL_COLUMNS = {"exclude_tables": (), "exclude_columns": {}, "tables": P.PRUNE_TABLES}


# --------------------------------------------------------------------------------------------
# the round trip
# --------------------------------------------------------------------------------------------

def test_round_trip_archive_reclaim_contract_restore_reproduces_the_pre_prune_tables_marks_included(base, tmp_path):
    w = W(base, tmp_path)
    s1 = w.snap("s1")
    c = _raw(s1)
    # a MARKED deletable carrier (rev_a's item_result: the mark is inert, so the carrier stays authentic) and a marked PROTECTED one
    # (rev_b's item_result made unreproducible: one value differs from what the current code parses)
    c.execute("UPDATE item_result SET withdrawn_by_revision_id = 'mark-a' WHERE source_revision_id = ?", (w.rev_a,))
    c.execute("UPDATE item_result SET value = value + 0.5, withdrawn_by_revision_id = 'mark-b' WHERE source_revision_id = ? AND item_id = 'q0'",
              (w.rev_b,))
    c.close()
    orig = w.snap("orig", s1)
    d1 = PA.digest_database(s1, str(tmp_path / "d1.json"))
    A, m = _archive(w, s1)
    assert m["census"]["item_result"]["unauthentic_digest"] == {"rows": 4, "carriers": 1}
    assert m["groups"]["count"] == 5 and not m["skipped_partial"]                          # rev_a x 3 tables, rev_b x 2
    assert {k[:2] for k in _groups(A)} == {(t, w.rev_a) for t in P.PRUNE_TABLES} | {("item_flags", w.rev_b), ("run_degeneration", w.rev_b)}
    assert all(g.klass == P.DELETABLE for g in _groups(A).values())
    n_arch = sum(e["rows"] for e in m["tables"].values())
    assert n_arch == 3 + 3 + 1 + 4 + 1
    res = PA.reclaim_file(s1, w.root, PA.mode_for("cohort", [A]))
    # rev_b's unarchived carrier is unauthentic, which the normal rule protects in cohort mode too: it never waits for a delta archive,
    # so the cohort pass itself deletes every archived row and records both revisions reclaimed; `mode normal` then finds nothing
    assert res["rows_deleted"] == n_arch and res["stopped"] is None and res["pending_revisions"] == 0
    assert os.path.exists(os.path.join(A, P.VERIFIED_LOCAL_FILE))                         # the gate ran for the explicit cohort mode
    res = PA.reclaim_file(s1, w.root, P.ReclaimMode.normal())
    assert res["rows_deleted"] == 0 and res["pending_revisions"] == 0
    d2 = PA.digest_database(s1, str(tmp_path / "d2.json"))
    ok = EQ.check_contract(d1, d2, "deletion", PA.read_manifest(A), _group_dicts(A))
    assert ok["ok"], ok["failures"]
    assert _q_db(s1, "SELECT COUNT(*), MIN(withdrawn_by_revision_id) FROM item_result WHERE source_revision_id = ?", w.rev_b) == [(4, "mark-b")]
    assert PA.verify_deleted(s1, [A])["ok"]
    # S2 -> S3 (compaction) is identity
    s3 = w.snap("s3", s1)
    d3 = PA.digest_database(s3)
    assert EQ.check_contract(d2, d3, "identity")["ok"]
    # restore into a scratch copy of the pruned database: the three item tables equal the pre-prune copy on every column
    scratch = w.snap("scratch", s1)
    r = PA.restore(A, scratch)
    assert r["ok"] and r["conflicts"] == 0 and r["ledger_removed"] == 5 and r["ledger_kept"] == 0
    assert sum(t["inserted"] for t in r["tables"].values()) == n_arch
    with DB.reading(orig) as a, DB.reading(scratch) as b:
        assert EQ.verify_equal(a, b, **ALL_COLUMNS) == {t: 0 for t in P.PRUNE_TABLES}
    assert _q_db(scratch, "SELECT COUNT(*) FROM item_reclaim_ledger") == [(0,)]
    assert _q_db(scratch, "SELECT DISTINCT withdrawn_by_revision_id FROM item_result WHERE source_revision_id = ?", w.rev_a) == [("mark-a",)]
    assert _q_db(scratch, "SELECT reclaimed FROM source_revision WHERE revision_id IN (?, ?) ORDER BY 1", w.rev_a, w.rev_b) == [(0,), (0,)]


def _q_db(path, sql, *params):
    with DB.reading(path) as conn:
        return [tuple(r) for r in conn.execute(sql, params).fetchall()]


def test_table_digest_is_the_archive_row_digest_and_parallel_digests_agree(base, tmp_path):
    w = W(base, tmp_path)
    db = DB.db_path(w.root)
    with DB.reading(db) as conn:
        for t in P.PRUNE_TABLES:
            row = tuple(conn.execute(f"SELECT {', '.join(P.columns_of(t))} FROM {t} LIMIT 1").fetchone())
            assert PA.generic_row_digest(t, row) == P.row_digest(t, row)
    one = PA.digest_database(db, objects="v_item_result_current,v_paper_long")
    two = PA.digest_database(db, objects="v_item_result_current,v_paper_long", jobs=2)
    strip = lambda d: {k: {n: {f: v for f, v in e.items() if f != "seconds"} for n, e in d[k].items()} for k in ("tables", "objects")}
    assert strip(one) == strip(two) and one["extras"] == two["extras"] and one["schema"] == two["schema"]
    assert set(one["objects"]) == {"v_item_result_current", "v_paper_long"}
    with pytest.raises(PA.ArchiveError):
        PA.digest_database(db, objects="v_no_such_object")


# --------------------------------------------------------------------------------------------
# encoding and verification
# --------------------------------------------------------------------------------------------

def test_object_digests_do_not_depend_on_the_clock_and_still_compare_the_timestamps_the_ages_come_from(tmp_path):
    """``v_in_flight.age_s`` and ``v_pending_judgment.claim_age_s`` are ``julianday('now')`` minus a timestamp of the same row: two digests
    of one unchanged database taken a second apart agree (the ages are left out and the document says so), and a change to the
    timestamp an age is computed from still changes the object's digest."""
    import time
    from sisteer.registry import pending as PEND
    from sisteer.registry import records as R
    from sisteer.registry import synth
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=4)
    rid_running, recs = synth.run_records("inflight", 500, through="running")
    R.write_records(os.path.join(root, "records", "c", "000001-run.jsonl"), recs)
    H.ingest_once(root, workers=0, min_free=0)
    rid = ids["run_ids"][0]
    PEND.claim(root, rid, synth.JUDGE_ID_2, "session-A", now=synth.t(1000))
    db = DB.db_path(root)
    with DB.reading(db) as conn:
        assert conn.execute("SELECT COUNT(*) FROM v_in_flight").fetchone()[0] >= 1
        assert conn.execute("SELECT COUNT(*) FROM v_pending_judgment WHERE claim_utc IS NOT NULL").fetchone()[0] == 1
    objs = "v_in_flight,v_pending_judgment"
    d1 = PA.digest_database(db, objects=objs)
    time.sleep(1.1)                                    # SQLite's 'now' is the wall clock: the ages differ by at least a second
    d2 = PA.digest_database(db, objects=objs)
    assert d1["projection"] == {"clock_derived_columns_left_out": {"v_in_flight": ["age_s"], "v_pending_judgment": ["claim_age_s"]}}
    assert d1["objects"]["v_in_flight"]["left_out"] == ["age_s"] and "age_s" not in d1["objects"]["v_in_flight"]["columns"]
    assert {"last_event_utc", "queued_utc"} <= set(d1["objects"]["v_in_flight"]["columns"])
    assert "claim_utc" in d1["objects"]["v_pending_judgment"]["columns"]
    ok = EQ.check_contract(d1, d2, "identity")
    assert ok["ok"], ok["failures"]
    moved = str(tmp_path / "moved.sqlite")
    with DB.reading(db) as conn:
        conn.execute(f"VACUUM INTO '{moved}'")
    c = _raw(moved)
    c.execute("UPDATE judgment SET claimed_utc = ? WHERE run_id = ? AND claimed_by_session = 'session-A'", (synth.t(1500), rid))
    c.close()
    d3 = PA.digest_database(moved, objects=objs)
    assert EQ.dsum(d3["objects"]["v_pending_judgment"]) != EQ.dsum(d1["objects"]["v_pending_judgment"])
    assert EQ.dsum(d3["objects"]["v_in_flight"]) == EQ.dsum(d1["objects"]["v_in_flight"])
    # two documents taken under different projections never pass a contract
    d4 = dict(d2, projection={"clock_derived_columns_left_out": {}})
    assert not EQ.check_contract(d1, d4, "identity")["ok"]


def test_row_encoding_round_trips_extreme_values_through_a_zstd_row_file(tmp_path):
    rows = [(None, 0, -1, 2 ** 63 - 1, -(2 ** 63), 0.0, -0.0, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308, float("inf"),
             "", "é☃\U0001F600\t\\\"", b"", b"\x00\xff" * 16)]
    path = str(tmp_path / "x.jsonl.zst")
    z = PA.ZstdLines(path)
    for r in rows:
        z.write_line(P.encode_row(r))
    meta = z.close()
    assert meta["sha256"] == PA.sha256_file(path) and meta["bytes"] == os.path.getsize(path)
    back = [P.decode_row(ln) for ln in PA.iter_zstd_lines(path)]
    assert len(back) == 1
    for x, y in zip(rows[0], back[0]):
        assert type(x) is type(y) and (x.hex() == y.hex() if isinstance(x, float) else x == y)
    assert PA.generic_row_digest("t", rows[0]) == PA.generic_row_digest("t", back[0])


def _rewrite_row_file(archive_dir, table, keep):
    """Rewrite ``table``'s row file keeping the lines ``keep`` selects, and re-sum it CONSISTENTLY (SHA256SUMS, the MANIFEST's sha and
    bytes, and SHA256SUMS's line for the re-written MANIFEST): a tamper only the row digests can catch."""
    path = os.path.join(archive_dir, PA.row_file(table))
    lines = [ln for i, ln in enumerate(PA.iter_zstd_lines(path)) if keep(i)]
    z = PA.ZstdLines(path)
    for ln in lines:
        z.write_line(ln)
    meta = z.close()
    mpath = os.path.join(archive_dir, P.MANIFEST_FILE)
    m = json.load(open(mpath))
    m["tables"][table]["sha256"], m["tables"][table]["bytes"] = meta["sha256"], meta["bytes"]
    json.dump(m, open(mpath, "w"), indent=1, sort_keys=True)
    sums = PA.read_sums(archive_dir)
    sums[PA.row_file(table)] = meta["sha256"]
    sums[P.MANIFEST_FILE] = PA.sha256_file(mpath)
    with open(os.path.join(archive_dir, PA.SUMS_FILE), "w") as fh:
        for n, s in sums.items():
            fh.write(f"{s}  {n}\n")


def _flip(path, at=None):
    data = bytearray(open(path, "rb").read())
    i = len(data) // 2 if at is None else at
    data[i] ^= 0x01
    open(path, "wb").write(bytes(data))


def test_verify_archive_catches_a_flipped_byte_a_dropped_row_and_a_consistent_re_sum(base, tmp_path):
    w = W(base, tmp_path)
    A, _ = _archive(w, DB.db_path(w.root), verify=False)
    rec = PA.verify_archive(A, db_path=DB.db_path(w.root), root=w.root)
    assert rec[P.VERIFIED_MANIFEST_KEY] == PA.sha256_file(os.path.join(A, P.MANIFEST_FILE))
    assert json.load(open(os.path.join(A, P.VERIFIED_FILE)))[P.VERIFIED_MANIFEST_KEY] == rec[P.VERIFIED_MANIFEST_KEY]
    flipped = str(tmp_path / "flipped")
    shutil.copytree(A, flipped)
    os.remove(os.path.join(flipped, P.VERIFIED_FILE))
    _flip(os.path.join(flipped, PA.row_file("item_flags")))
    with pytest.raises(PA.ArchiveInvalid) as e:
        PA.verify_archive(flipped)
    assert any("item_flags.jsonl.zst" in f and "sha256" in f for f in e.value.failures)
    assert not os.path.exists(os.path.join(flipped, P.VERIFIED_FILE))                     # nothing written on a failure
    dropped = str(tmp_path / "dropped")
    shutil.copytree(A, dropped)
    _rewrite_row_file(dropped, "item_result", lambda i: i != 1)
    with pytest.raises(PA.ArchiveInvalid) as e:
        PA.verify_archive(dropped, verified_name=P.VERIFIED_LOCAL_FILE)
    assert any("group" in f and "differ" in f for f in e.value.failures), e.value.failures
    # the database comparison: an archive that no longer equals the database's deletable set fails too
    c = _raw(DB.db_path(w.root))
    c.execute("UPDATE run_degeneration SET ingest_seq = ingest_seq + 1 WHERE source_revision_id = ?", (w.rev_a,))
    c.close()
    with pytest.raises(PA.ArchiveInvalid) as e:
        PA.verify_archive(A, db_path=DB.db_path(w.root), root=w.root, write=False)
    assert any("database: group" in f for f in e.value.failures), e.value.failures


# --------------------------------------------------------------------------------------------
# cohort mode: the binding, resumption, the explicit mode
# --------------------------------------------------------------------------------------------

def test_a_same_count_change_after_the_archive_leaves_the_group_undeleted_and_listed(base, tmp_path):
    w = W(base, tmp_path)
    work = w.snap("work")
    A, _ = _archive(w, work)
    rf = _first_carrier(work, "item_result", w.rev_a)
    c = _raw(work)
    c.execute("UPDATE item_result SET withdrawn_by_revision_id = 'later' WHERE source_revision_id = ? AND item_id = 'q1'", (w.rev_a,))
    c.close()
    PA.reclaim_file(work, w.root, PA.mode_for("cohort", [A]))
    assert _count(work, "item_result", w.rev_a, rf) == 3                                   # the changed group survives, whole
    with DB.reading(work) as conn:
        plans = {(p.table, p.record_file_id): p for p in P.revision_plan(conn, w.root, w.rev_a, P.ReclaimMode.cohort([A]))}
    assert plans[("item_result", rf)].klass == P.COHORT_UNCOVERED and "digest_bind" in plans[("item_result", rf)].detail
    left = [(t, rev) for (t, rev, rfid) in _groups(A) if _count(work, t, rev, rfid)]
    assert left == [("item_result", w.rev_a)]                                              # every other archived group is gone


def test_a_revision_superseded_after_the_manifest_waits_for_a_delta_archive(tmp_path):
    root, store_root, ad = F._world(tmp_path / "w", mode="hold")
    rev_a = F._read(root, ad, 2000)
    _store(store_root, "s1", "store_1", F.LORA, [0.9, 0.9, 0.9])
    rev_b = F._read(root, ad, 3000)
    A = str(tmp_path / "A")
    PA.archive(DB.db_path(root), root, A)
    PA.verify_archive(A, db_path=DB.db_path(root), root=root)
    _store(store_root, "s1", "store_1", F.LORA, [0.4, 0.5, 0.6])                          # rev_b is superseded AFTER the manifest
    rev_c = F._read(root, ad, 4000)
    assert F._head(root) == rev_c
    PA.set_mode(root, "cohort", [A])
    F._ingest(root, 4100)                                                                  # the live cycle, cohort mode from state
    assert F._rows(root, rev_a) == {t: 0 for t in P.PRUNE_TABLES}
    assert F._rows(root, rev_b) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}
    st = PA.status(root, uncovered_out=str(tmp_path / "delta.json"))
    assert st["mode"]["kind"] == "cohort" and st["uncovered_groups"] == 3 and st["cohort"]["groups_with_rows"] == 0
    # the JSON lists them too (what the live driver turns into a delta archive's group list), each with table, revision and file id
    assert [g["table"] for g in st["uncovered"]] == sorted(P.PRUNE_TABLES) and {g["revision_id"] for g in st["uncovered"]} == {rev_b}
    assert PA.load_groups_filter(str(tmp_path / "delta.json")) == {(g["table"], g["revision_id"], g["record_file_id"]) for g in st["uncovered"]}
    assert PA.status(root, uncovered=True)["uncovered"] == st["uncovered"] and "uncovered" not in PA.status(root)
    with pytest.raises(PA.ArchiveError, match="still deletable"):
        _refuse_normal_while_uncovered(root, A, tmp_path)
    D = str(tmp_path / "D1")
    m = PA.archive(DB.db_path(root), root, D, groups=PA.load_groups_filter(str(tmp_path / "delta.json")), kind="live-delta-1")
    assert m["groups"]["count"] == 3 and m["groups_filter"]["not_found"] == []
    PA.verify_archive(D, db_path=DB.db_path(root), root=root)
    PA.set_mode(root, "cohort", [A, D])
    F._ingest(root, 4200)
    assert F._rows(root, rev_b) == {t: 0 for t in P.PRUNE_TABLES}
    assert PA.verify_deleted(DB.db_path(root), [A, D])["ok"]
    out = PA.set_mode(root, "normal")
    assert out["mode"] == "normal" and not os.path.exists(P.ReclaimMode.state_path(root))


@pytest.mark.parametrize("protect", ["unauthentic", "missing"])
def test_cohort_mode_keeps_an_unarchived_protected_carrier_without_holding_its_revision(base, tmp_path, protect):
    """Cohort mode classifies by the normal rule first. A carrier the normal rule protects (unauthentic: the current code does not
    reproduce its rows; missing: its log file is not the ingested bytes) is never archived, and in cohort mode it stays protected rather
    than ``cohort_uncovered``: it waits for no delta archive. The revision's archived carriers go; an unauthentic carrier is a class, not
    work (the revision is recorded reclaimed in the cohort pass itself), while a missing one keeps its revision due, so the file's return
    is seen. The deletion contract holds between the two states either way (additive against the archive; ``reclaimed`` = 1 only on an
    accepted, superseded row, and 0 on one only while rows of it remain)."""
    w = W(base, tmp_path)
    s1 = w.snap("s1")
    rf = _first_carrier(s1, "item_result", w.rev_b)
    c = _raw(s1)
    if protect == "unauthentic":                                                           # one stored value the parser does not produce
        c.execute("UPDATE item_result SET value = value + 0.5 WHERE source_revision_id = ? AND record_file_id = ? AND item_id = 'q0'",
                  (w.rev_b, rf))
    else:                                                                                  # the file on disk is not the bytes ingested
        c.execute("UPDATE record_ingest SET file_sha256 = ? WHERE record_file_id = ?", ("0" * 64, rf))
    c.close()
    d1 = PA.digest_database(s1, str(tmp_path / "d1.json"))
    A, m = _archive(w, s1)
    mode = PA.mode_for("cohort", [A])
    with DB.reading(s1) as conn:
        plans = {}
        for rev in (w.rev_a, w.rev_b):
            plans[rev] = {(p.table, p.record_file_id): p for p in P.revision_plan(conn, w.root, rev, mode)}
    want = P.UNAUTHENTIC_DIGEST if protect == "unauthentic" else P.MISSING
    kept = {(rev, k): p.rows for rev, ps in plans.items() for k, p in ps.items() if p.klass != P.DELETABLE}
    assert kept and all(plans[rev][k].klass == want for rev, k in kept) and (w.rev_b, ("item_result", rf)) in kept
    assert not any(p.klass == P.COHORT_UNCOVERED for ps in plans.values() for p in ps.values())
    gone = {(rev, k) for rev, ps in plans.items() for k, p in ps.items() if p.klass == P.DELETABLE}
    assert {(rev, (t, rfid)) for (t, rev, rfid) in _groups(A)} == gone                     # exactly the deletable carriers were archived
    assert any(rev == w.rev_b for rev, _ in gone) or protect == "missing"                  # rev_b keeps carriers to delete beside the kept one
    res = PA.reclaim_file(s1, w.root, mode)
    assert res["stopped"] is None and res["pending_revisions"] == (1 if protect == "missing" else 0)
    assert res["rows_deleted"] == sum(e["rows"] for e in m["tables"].values())
    for rev, ps in plans.items():
        for (t, rfid), p in ps.items():
            assert _count(s1, t, rev, rfid) == kept.get((rev, (t, rfid)), 0), (rev, t, rfid)
    reclaimed = dict(_q_db(s1, "SELECT revision_id, reclaimed FROM source_revision WHERE revision_id IN (?, ?)", w.rev_a, w.rev_b))
    assert reclaimed == {w.rev_a: 1, w.rev_b: 0 if protect == "missing" else 1}
    ok = EQ.check_contract(d1, PA.digest_database(s1, str(tmp_path / "d2.json")), "deletion", PA.read_manifest(A), _group_dicts(A))
    assert ok["ok"], ok["failures"]
    assert PA.verify_deleted(s1, [A])["ok"]


def _refuse_normal_while_uncovered(root, A, tmp_path):
    """``mode normal`` refuses while a cohort group still has rows and is deletable: rebuild that state on a fresh cohort over the
    still-undeleted rev_b groups (the delta archive), before the cycle that deletes them."""
    D0 = str(tmp_path / "D0")
    PA.archive(DB.db_path(root), root, D0, groups=PA.load_groups_filter(str(tmp_path / "delta.json")))
    PA.verify_archive(D0)
    PA.set_mode(root, "cohort", [A, D0])
    try:
        PA.set_mode(root, "normal")
    finally:
        PA.set_mode(root, "cohort", [A])


def test_an_ingest_seq_remap_between_archive_and_deletion_still_deletes_and_restore_re_derives_the_rank(base, tmp_path):
    w = W(base, tmp_path)
    work = w.snap("work")
    A, _ = _archive(w, work)
    c = _raw(work)                                                                         # Ranker.finalize moves a file's rank and its rows' together
    c.execute("UPDATE record_ingest SET ingest_seq = ingest_seq + 100000")
    for t in P.PRUNE_TABLES:
        c.execute(f"UPDATE {t} SET ingest_seq = ingest_seq + 100000")
    c.close()
    remapped = w.snap("remapped", work)
    res = PA.reclaim_file(work, w.root, PA.mode_for("cohort", [A]))
    assert res["rows_deleted"] == sum(g.rows for g in _groups(A).values())
    scratch = w.snap("scratch", work)
    r = PA.restore(A, scratch)
    assert r["ok"] and r["conflicts"] == 0
    with DB.reading(remapped) as a, DB.reading(scratch) as b:
        assert EQ.verify_equal(a, b, **ALL_COLUMNS) == {t: 0 for t in P.PRUNE_TABLES}      # ingest_seq = the CURRENT rank


def _stop_after(n_batches):
    calls = []

    def hook(ing, plan):
        calls.append(plan)
        return "budget" if len(calls) > n_batches else None
    return hook


def test_cohort_stop_mid_carrier_resumes_from_present_plus_ledger_and_ends_additive(base, tmp_path, monkeypatch):
    w = W(base, tmp_path)
    work = w.snap("work")
    A, _ = _archive(w, work)
    d1 = PA.digest_database(work)
    rf = _first_carrier(work, "item_result", w.rev_a)
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", _stop_after(1))
    res = PA.reclaim_file(work, w.root, PA.mode_for("cohort", [A]))
    assert res["rows_deleted"] == 1 and res["stopped"] == "budget"
    assert _count(work, "item_result", w.rev_a, rf) == 2
    assert _q_db(work, "SELECT table_name, rows_deleted FROM item_reclaim_ledger") == [("item_result", 1)]
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", None)
    res = PA.reclaim_file(work, w.root, PA.mode_for("cohort", [A]))
    assert res["stopped"] is None and all(_count(work, t, rev, rfid) == 0 for (t, rev, rfid) in _groups(A))
    ok = EQ.check_contract(d1, PA.digest_database(work), "deletion", PA.read_manifest(A), _group_dicts(A))
    assert ok["ok"], ok["failures"]


def test_a_remaining_row_changed_between_passes_stops_the_group_and_its_rows_survive(base, tmp_path, monkeypatch):
    w = W(base, tmp_path)
    work = w.snap("work")
    A, _ = _archive(w, work)
    rf = _first_carrier(work, "item_result", w.rev_a)
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", _stop_after(1))
    PA.reclaim_file(work, w.root, PA.mode_for("cohort", [A]))
    assert _count(work, "item_result", w.rev_a, rf) == 2
    c = _raw(work)
    c.execute("UPDATE item_result SET withdrawn_by_revision_id = 'between' WHERE source_revision_id = ? AND record_file_id = ? "
              "AND item_id = (SELECT MIN(item_id) FROM item_result WHERE source_revision_id = ? AND record_file_id = ?)", (w.rev_a, rf, w.rev_a, rf))
    c.close()
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", None)
    PA.reclaim_file(work, w.root, PA.mode_for("cohort", [A]))
    assert _count(work, "item_result", w.rev_a, rf) == 2                                   # stopped as cohort_uncovered: both remaining rows stay
    with DB.reading(work) as conn:
        p = [p for p in P.revision_plan(conn, w.root, w.rev_a, P.ReclaimMode.cohort([A])) if p.table == "item_result"][0]
    assert p.klass == P.COHORT_UNCOVERED
    assert all(_count(work, t, rev, rfid) == 0 for (t, rev, rfid) in _groups(A) if (t, rev) != ("item_result", w.rev_a))


def test_a_carrier_deleted_across_three_passes_finishes(base, tmp_path, monkeypatch):
    w = W(base, tmp_path)
    work = w.snap("work")
    A, _ = _archive(w, work)
    rf = _first_carrier(work, "item_result", w.rev_a)
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    for k in (1, 2, 3):
        monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", _stop_after(1))
        PA.reclaim_file(work, w.root, PA.mode_for("cohort", [A]))
        assert _count(work, "item_result", w.rev_a, rf) == 3 - k
        if k < 3:
            assert _q_db(work, "SELECT rows_deleted FROM item_reclaim_ledger WHERE record_file_id = ?", rf) == [(k,)]
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", None)
    PA.reclaim_file(work, w.root, PA.mode_for("cohort", [A]))
    assert all(_count(work, t, rev, rfid) == 0 for (t, rev, rfid) in _groups(A))
    assert _q_db(work, "SELECT rows_deleted FROM item_reclaim_ledger WHERE record_file_id = ?", rf) == [(3,)]


@pytest.mark.parametrize("perturb", ["mark", "rank"])
def test_explicit_cohort_mode_on_a_non_live_file_keeps_the_perturbed_group_and_normal_mode_deletes_it(base, tmp_path, perturb):
    """The binding, not a live flag, keeps the group: the same perturbed copy under ``--mode normal`` loses it. The perturbation keeps
    the carrier AUTHENTIC (an inert mark, or a rank that is not the carrier's), so only the archive binding can tell."""
    w = W(base, tmp_path)
    A, _ = _archive(w, DB.db_path(w.root))
    table = "item_result" if perturb == "mark" else "run_degeneration"
    sql = (f"UPDATE item_result SET withdrawn_by_revision_id = 'perturbed' WHERE source_revision_id = ? AND item_id = 'q0'" if perturb == "mark"
           else "UPDATE run_degeneration SET ingest_seq = ingest_seq + 1 WHERE source_revision_id = ?")
    out = {}
    for mode in ("cohort", "normal"):
        work = w.snap(f"work-{mode}")
        c = _raw(work)
        c.execute(sql, (w.rev_a,))
        c.close()
        args = ["--root", w.root, "prune-superseded", "reclaim", "--db", work, "--mode", mode] + (["--manifest", A] if mode == "cohort" else [])
        assert cli.main(args) == 0
        out[mode] = {(t, rev): _count(work, t, rev, rfid) for (t, rev, rfid) in _groups(A)}
    assert out["cohort"][(table, w.rev_a)] > 0 and sum(n for k, n in out["cohort"].items() if k != (table, w.rev_a)) == 0
    assert sum(out["normal"].values()) == 0


def test_a_groups_file_swapped_after_mode_cohort_binds_nothing_and_holds_deletion(tmp_path):
    """The cohort binding reads ``groups.tsv.zst`` every cycle, and the verification the mode was set with names the MANIFEST only: the
    groups bytes the loader reads must be the ones that MANIFEST names. Swap in another archive's groups file -- valid, verified in its own
    archive, listing a revision superseded after the first archive was taken -- and the mode no longer validates: the live cycle deletes
    nothing (``reclaim_stopped = "mode"``), so that revision's rows, archived nowhere the mode names, stay. With the file put back, the
    first archive's groups go and the later revision still waits."""
    root, store_root, ad = F._world(tmp_path / "w", mode="hold")
    rev_a = F._read(root, ad, 2000)
    _store(store_root, "s1", "store_1", F.LORA, [0.9, 0.9, 0.9])
    rev_b = F._read(root, ad, 3000)
    A = str(tmp_path / "A")
    PA.archive(DB.db_path(root), root, A)
    PA.verify_archive(A, db_path=DB.db_path(root), root=root)
    _store(store_root, "s1", "store_1", F.LORA, [0.4, 0.5, 0.6])                          # rev_b is superseded after A
    F._read(root, ad, 4000)
    A2 = str(tmp_path / "A2")
    PA.archive(DB.db_path(root), root, A2)                                                 # a later archive: rev_a's groups and rev_b's
    PA.verify_archive(A2, db_path=DB.db_path(root), root=root)
    assert {g[1] for g in _groups(A2)} == {rev_a, rev_b} and {g[1] for g in _groups(A)} == {rev_a}
    PA.set_mode(root, "cohort", [A])
    assert P.ReclaimMode.from_state(root).kind == "cohort"
    gpath = os.path.join(A, P.GROUPS_FILE)
    kept = gpath + ".orig"
    shutil.copyfile(gpath, kept)
    shutil.copyfile(os.path.join(A2, P.GROUPS_FILE), gpath)
    with pytest.raises(P.ModeStateInvalid, match="sha256"):
        P.ReclaimMode.from_state(root)
    st = PA.status(root)
    assert st["mode"]["valid"] is False and "sha256" in st["mode"]["error"]
    rep = F._ingest(root, 4100)
    assert rep.reclaim_stopped == "mode" and rep.reclaim_rows_deleted == 0
    assert F._rows(root, rev_b) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}
    assert F._rows(root, rev_a) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}
    os.replace(kept, gpath)
    rep = F._ingest(root, 4200)
    assert rep.reclaim_stopped is None and F._rows(root, rev_a) == {t: 0 for t in P.PRUNE_TABLES}
    assert F._rows(root, rev_b) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}


def test_a_live_cycle_in_cohort_mode_and_mode_normal_refusal_then_release(base, tmp_path):
    w = W(base, tmp_path)
    A, _ = _archive(w, DB.db_path(w.root))
    PA.set_mode(w.root, "cohort", [A])
    st = PA.status(w.root)
    assert st["mode"]["kind"] == "cohort" and st["mode"]["valid"]
    assert st["cohort"] == {"groups": 6, "groups_with_rows": 6, "rows_remaining": 3 + 3 + 1 + 4 + 4 + 1} and st["pending_revisions"] == 2
    with pytest.raises(PA.ArchiveError, match="still deletable"):
        PA.set_mode(w.root, "normal")
    F._ingest(w.root, 5000)
    st = PA.status(w.root)
    assert st["cohort"]["groups_with_rows"] == 0 and st["pending_revisions"] == 0
    assert st["ledger"]["item_result"] == {"carriers": 2, "rows_deleted": 7}
    assert PA.verify_deleted(DB.db_path(w.root), [A])["ok"]
    PA.set_mode(w.root, "normal")
    assert PA.status(w.root)["mode"]["kind"] == "normal"


def _cohort_then_a_late_supersession(tmp_path):
    """rev_a archived (A) and deleted in cohort mode; then rev_b superseded AFTER the archive: its carriers are normally deletable and
    archived nowhere, so cohort mode keeps them -- while ``verify-deleted`` over A, which lists only rev_a's groups, passes."""
    root, store_root, ad = F._world(tmp_path / "w", mode="hold")
    rev_a = F._read(root, ad, 2000)
    _store(store_root, "s1", "store_1", F.LORA, [0.9, 0.9, 0.9])
    rev_b = F._read(root, ad, 3000)
    A = str(tmp_path / "A")
    PA.archive(DB.db_path(root), root, A)
    PA.verify_archive(A, db_path=DB.db_path(root), root=root)
    PA.set_mode(root, "cohort", [A])
    F._ingest(root, 3100)
    assert F._rows(root, rev_a) == {t: 0 for t in P.PRUNE_TABLES}
    _store(store_root, "s1", "store_1", F.LORA, [0.4, 0.5, 0.6])
    F._read(root, ad, 4000)
    assert F._rows(root, rev_b) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}
    assert PA.verify_deleted(DB.db_path(root), [A])["ok"]
    return root, store_root, ad, rev_a, rev_b, A


def test_mode_normal_waits_until_every_normally_deletable_carrier_is_archived_and_deleted_and_a_hold_keeps_that_obligation(tmp_path):
    """Leaving cohort mode for normal is the last moment the archive can still cover what normal mode deletes: ``mode normal`` refuses
    while any accepted, superseded revision has a normally deletable carrier no verified archive binds (here rev_b, superseded after the
    archive), and while a bound one still has rows. A ``mode hold`` in between keeps the cohort's archives in the hold file, so hold then
    normal runs the same check. Once a delta archive covers rev_b and the cohort has deleted it, normal is allowed."""
    root, store_root, ad, rev_a, rev_b, A = _cohort_then_a_late_supersession(tmp_path)
    with pytest.raises(PA.ArchiveError, match="no verified archive covers"):
        PA.set_mode(root, "normal")
    assert P.ReclaimMode.from_state(root).kind == "cohort"
    for _ in range(2):                                                                     # a pause, and a second one, keep the obligation
        out = PA.set_mode(root, "hold")
        raw = json.load(open(P.ReclaimMode.state_path(root)))
        assert raw["mode"] == "hold" and [e["dir"] for e in raw[P.OBLIGATION_KEY]] == [os.path.abspath(A)], raw
        assert P.ReclaimMode.from_state(root).kind == "hold" and out["cohort_obligation"] == raw[P.OBLIGATION_KEY]
        with pytest.raises(PA.ArchiveError, match="no verified archive covers"):
            PA.set_mode(root, "normal")
    assert PA.status(root)["cohort_obligation"] == raw[P.OBLIGATION_KEY]
    F._ingest(root, 4100)
    assert F._rows(root, rev_b) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}   # nothing was deleted unarchived
    PA.set_mode(root, "cohort", [A])
    delta = str(tmp_path / "delta.json")
    st = PA.status(root, uncovered_out=delta)
    assert {g["revision_id"] for g in st["uncovered"]} == {rev_b} and st["uncovered_groups"] == 3
    D = str(tmp_path / "D1")
    PA.archive(DB.db_path(root), root, D, groups=PA.load_groups_filter(delta), kind="live-delta-1")
    PA.verify_archive(D, db_path=DB.db_path(root), root=root)
    PA.set_mode(root, "cohort", [A, D])
    with pytest.raises(PA.ArchiveError, match="still deletable"):                          # covered now, not yet deleted
        PA.set_mode(root, "normal")
    F._ingest(root, 4200)
    assert F._rows(root, rev_b) == {t: 0 for t in P.PRUNE_TABLES}
    out = PA.set_mode(root, "normal")
    assert out["obligation_checked"] == [os.path.abspath(A), os.path.abspath(D)] and not os.path.exists(P.ReclaimMode.state_path(root))


def test_mode_normal_checks_again_under_the_writer_lock(tmp_path, monkeypatch):
    """The first check runs without the lock; a cycle that supersedes another revision between it and the lock is caught by the second,
    which runs under the writer lock in the snapshot the switch is made in: the mode stays cohort."""
    root, store_root, ad, rev_a, rev_b, A = _cohort_then_a_late_supersession(tmp_path)
    delta = str(tmp_path / "delta.json")
    PA.status(root, uncovered_out=delta)
    D = str(tmp_path / "D1")
    PA.archive(DB.db_path(root), root, D, groups=PA.load_groups_filter(delta))
    PA.verify_archive(D, db_path=DB.db_path(root), root=root)
    PA.set_mode(root, "cohort", [A, D])
    F._ingest(root, 4200)
    real = DB.write_lock
    raced = []

    @contextlib.contextmanager
    def racing(lock_root, *a, **kw):
        if not raced:
            raced.append(1)                                                                # a harvest cycle lands first: rev_c superseded
            _store(store_root, "s1", "store_1", F.LORA, [0.3, 0.3, 0.3])
            F._read(root, ad, 5000)
        with real(lock_root, *a, **kw):
            yield
    monkeypatch.setattr(DB, "write_lock", racing)
    with pytest.raises(PA.ArchiveError, match="under the writer lock"):
        PA.set_mode(root, "normal")
    monkeypatch.setattr(DB, "write_lock", real)
    assert raced and P.ReclaimMode.from_state(root).kind == "cohort"


@pytest.mark.parametrize("installed", ["cohort", "unreadable"])
def test_mode_normal_from_a_plain_hold_reads_the_mode_file_again_under_the_writer_lock(tmp_path, monkeypatch, installed):
    """``mode normal`` started from a plain hold reads no obligation without the lock. A ``mode cohort`` (or a mode file that does not
    read) landing between that read and the lock is what the switch must judge: the cohort's obligation is checked under the lock --
    here it refuses, since rev_b was superseded after the archive and rev_a's archived rows are not yet deleted -- and an unreadable file
    refuses. The file stands, and the next cycle deletes only archived rows."""
    root, store_root, ad = F._world(tmp_path / "w", mode="hold")
    rev_a = F._read(root, ad, 2000)
    _store(store_root, "s1", "store_1", F.LORA, [0.9, 0.9, 0.9])
    rev_b = F._read(root, ad, 3000)
    A = str(tmp_path / "A")
    PA.archive(DB.db_path(root), root, A)
    PA.verify_archive(A, db_path=DB.db_path(root), root=root)
    _store(store_root, "s1", "store_1", F.LORA, [0.4, 0.5, 0.6])
    F._read(root, ad, 4000)                                                                # rev_b superseded after the archive
    assert PA._obligation_entries(root) == [] and P.ReclaimMode.from_state(root).kind == "hold"
    path = P.ReclaimMode.state_path(root)
    real = DB.write_lock
    raced = []

    @contextlib.contextmanager
    def racing(lock_root, *a, **kw):
        if not raced:
            raced.append(1)                                                                # another operator's command lands first
            if installed == "cohort":
                PA.set_mode(root, "cohort", [A])
            else:
                with open(path, "w", encoding="utf-8") as fh:
                    fh.write("{torn")
        with real(lock_root, *a, **kw):
            yield
    monkeypatch.setattr(DB, "write_lock", racing)
    # under the lock the check stops at the first revision with a finding (rev_a's archived rows, or rev_b's unarchived ones: the order
    # follows revision ids, which hash the store's path)
    with pytest.raises(PA.ArchiveError, match="no verified archive covers|still deletable" if installed == "cohort" else "cannot be read"):
        PA.set_mode(root, "normal")
    monkeypatch.setattr(DB, "write_lock", real)
    assert raced and os.path.exists(path)                                                  # the file the switch did not judge safe stands
    if installed == "cohort":
        assert P.ReclaimMode.from_state(root).kind == "cohort"
        F._ingest(root, 4100)
        assert F._rows(root, rev_a) == {t: 0 for t in P.PRUNE_TABLES}                      # archived: deleted
        assert F._rows(root, rev_b) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}   # archived nowhere: kept
    else:
        assert open(path, encoding="utf-8").read() == "{torn"
        F._ingest(root, 4100)                                                              # the harvester refuses the torn file: deletes nothing
        assert F._rows(root, rev_a) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}
        assert F._rows(root, rev_b) == {"item_result": 3, "item_flags": 3, "run_degeneration": 1}


def test_verify_deleted_seeks_each_group_and_counts_only_the_failures_it_reports(base, tmp_path, monkeypatch):
    w = W(base, tmp_path)
    work = w.snap("work")
    A, _ = _archive(w, work)
    counted = []
    real = PA._present
    monkeypatch.setattr(PA, "_present", lambda conn, *k: (counted.append(k), real(conn, *k))[1])
    res = PA.verify_deleted(work, [A])                                                     # nothing deleted yet: every group fails
    assert not res["ok"] and res["n_failures"] == 6 and len(counted) == 6
    want = sorted([{"table": t, "revision_id": rev, "record_file_id": rf, "rows_present": _count(work, t, rev, rf), "rows_archived": g.rows}
                   for (t, rev, rf), g in _groups(A).items()], key=lambda f: (f["table"], f["revision_id"], f["record_file_id"]))
    assert res["failures"] == want and (res["empty"], res["revision_no_longer_superseded"]) == (0, 0)
    PA.reclaim_file(work, w.root, PA.mode_for("cohort", [A]))
    counted.clear()
    res = PA.verify_deleted(work, [A])
    assert res == {"ok": True, "groups": 6, "empty": 6, "revision_no_longer_superseded": 0, "failures": [], "n_failures": 0}
    assert counted == []                                                                   # no count when nothing is reported
    monkeypatch.setattr(PA, "VERIFY_DELETED_REPORTED", 2)
    res = PA.verify_deleted(w.snap("undeleted"), [A])
    assert res["n_failures"] == 6 and len(res["failures"]) == 2 and len(counted) == 2


def test_status_no_coverage_leaves_out_the_cohort_progress_and_nothing_else(base, tmp_path, monkeypatch, capsys):
    w = W(base, tmp_path)
    A, _ = _archive(w, DB.db_path(w.root))
    PA.set_mode(w.root, "cohort", [A])
    full = PA.status(w.root)
    assert full["cohort"]["groups"] == 6
    monkeypatch.setattr(PA, "cohort_coverage", lambda *a, **k: pytest.fail("--no-coverage must not count the cohort's groups"))
    assert cli.main(["--root", w.root, "prune-superseded", "status", "--json", "--no-coverage"]) == 0
    lean = json.loads(capsys.readouterr().out)
    assert "cohort" not in lean
    drop = ("cohort", "recent_rehydrations", "last_reclaim_stop", "recent_reclaim_rows")
    assert {k: v for k, v in lean.items() if k not in drop} == json.loads(json.dumps({k: v for k, v in full.items() if k not in drop}))


def test_rehydrate_from_archive_reads_the_archive_once_and_before_the_write_lock(base, tmp_path, monkeypatch):
    w = W(base, tmp_path)
    A, _ = _archive(w, DB.db_path(w.root))
    os.remove(P.ReclaimMode.state_path(w.root))
    F._ingest(w.root, 5000)
    assert _q(w.root, "SELECT COUNT(*) FROM item_reclaim_ledger WHERE revision_id = ?", w.rev_a) == [(3,)]
    events = []
    real_groups, real_iter, real_lock = P.read_groups, PA.iter_archive_groups, DB.write_lock
    monkeypatch.setattr(P, "read_groups", lambda d: (events.append("groups"), real_groups(d))[1])

    def iter_groups(d, t, cols):
        events.append(f"rows:{t}")
        yield from real_iter(d, t, cols)
    monkeypatch.setattr(PA, "iter_archive_groups", iter_groups)

    @contextlib.contextmanager
    def lock(*a, **kw):
        events.append("lock")
        with real_lock(*a, **kw):
            yield
    monkeypatch.setattr(DB, "write_lock", lock)
    res = PA.rehydrate(w.root, w.rev_a, archive_dirs=[A])
    assert res["rows_restored"] == 7
    assert events.count("groups") == 1 and events.index("lock") > max(i for i, e in enumerate(events) if e != "lock"), events
    assert sorted(e for e in events if e.startswith("rows:")) == sorted(f"rows:{t}" for t in P.PRUNE_TABLES)


# --------------------------------------------------------------------------------------------
# the handoff: a local copy of the cluster's archive
# --------------------------------------------------------------------------------------------

def _copy_like_the_runbook(src, dst):
    """``rsync -a`` of every file, then ``sha256sum -c SHA256SUMS`` in the copy (copytree / the Python check where the tools are absent)."""
    if shutil.which("rsync"):
        subprocess.run(["rsync", "-a", src.rstrip("/") + "/", dst.rstrip("/") + "/"], check=True)
    else:
        shutil.copytree(src, dst)
    if shutil.which("sha256sum"):
        subprocess.run(["sha256sum", "--quiet", "-c", PA.SUMS_FILE], cwd=dst, check=True)
    else:
        assert all(PA.sha256_file(os.path.join(dst, n)) == s for n, s in PA.read_sums(dst).items())


def test_the_handoff_gate_refuses_every_damaged_or_unverified_copy_and_accepts_a_whole_one(base, tmp_path):
    w = W(base, tmp_path)
    A, _ = _archive(w, DB.db_path(w.root))                                                # VERIFIED.json: the cluster's independent pass
    good = str(tmp_path / "B")
    _copy_like_the_runbook(A, good)

    def damaged(name, how):
        d = str(tmp_path / name)
        shutil.copytree(A, d)
        how(d)
        return d

    cases = {
        "no row file": damaged("B1", lambda d: os.remove(os.path.join(d, PA.row_file("item_flags")))),
        "truncated row file": damaged("B2", lambda d: os.truncate(os.path.join(d, PA.row_file("item_result")),
                                                                os.path.getsize(os.path.join(d, PA.row_file("item_result"))) - 9)),
        "flipped byte": damaged("B3", lambda d: _flip(os.path.join(d, PA.row_file("run_degeneration")))),
        "no VERIFIED.json": damaged("B4", lambda d: os.remove(os.path.join(d, P.VERIFIED_FILE))),
        "VERIFIED.json names another MANIFEST": damaged("B5", lambda d: json.dump({P.VERIFIED_MANIFEST_KEY: "0" * 64},
                                                                                 open(os.path.join(d, P.VERIFIED_FILE), "w"))),
    }
    state = P.ReclaimMode.state_path(w.root)
    before = open(state).read()
    for what, d in cases.items():
        with pytest.raises(PA.ArchiveInvalid):
            PA.set_mode(w.root, "cohort", [d])
        assert open(state).read() == before, what                                          # the mode is still hold
        assert not os.path.exists(os.path.join(d, P.VERIFIED_LOCAL_FILE)), what
    assert cli.main(["--root", w.root, "prune-superseded", "mode", "cohort", "--manifest", cases["flipped byte"]]) == 2
    out = PA.set_mode(w.root, "cohort", [good])
    assert out["manifests"][0]["dir"] == os.path.abspath(good) and os.path.exists(os.path.join(good, P.VERIFIED_LOCAL_FILE))
    assert P.ReclaimMode.from_state(w.root).kind == "cohort"


# --------------------------------------------------------------------------------------------
# refusals, the partial group, status
# --------------------------------------------------------------------------------------------

def test_restore_refuses_the_live_database_and_reports_conflicts_without_overwriting(base, tmp_path):
    w = W(base, tmp_path)
    A, _ = _archive(w, DB.db_path(w.root))
    with pytest.raises(PA.LiveRefused):
        PA.restore(A, DB.db_path(w.root), root=w.root)
    with pytest.raises(PA.LiveRefused, match="maintenance"):
        PA.restore(A, DB.db_path(w.root), root=w.root, live=True)
    assert cli.main(["--root", w.root, "prune-superseded", "restore", A, "--db", DB.db_path(w.root)]) == 2
    copy = w.snap("unpruned")                                                               # nothing deleted: every key is present
    r = PA.restore(A, copy)
    assert r["ok"] and sum(t["inserted"] for t in r["tables"].values()) == 0
    assert sum(t["present_identical"] for t in r["tables"].values()) == 16
    c = _raw(copy)
    c.execute("UPDATE item_result SET value = 0.123 WHERE source_revision_id = ? AND item_id = 'q0'", (w.rev_a,))
    c.close()
    r = PA.restore(A, copy)
    assert not r["ok"] and r["conflicts"] == 1 and r["conflict_examples"][0]["table"] == "item_result"
    assert _q_db(copy, "SELECT value FROM item_result WHERE source_revision_id = ? AND item_id = 'q0'", w.rev_a) == [(0.123,)]


def test_restore_journals_its_groups_in_a_temp_file_and_a_live_restore_needs_room_for_it(base, tmp_path, monkeypatch):
    # each group is a SAVEPOINT and each of its INSERTs a statement savepoint (the item tables' RAISE(ABORT) triggers): with
    # temp_store=MEMORY every release walks the in-memory sub-journal, so a group costs time quadratic in its rows. The setting that
    # governs a transaction is the one in force when it BEGINs, so the spy reads it there, on restore's own write connection
    w = W(base, tmp_path)
    work = w.snap("work")
    A, _ = _archive(w, work)
    deleted = PA.reclaim_file(work, w.root, PA.mode_for("cohort", [A]))["rows_deleted"]
    assert deleted > 0
    at_begin = []
    real = DB.transaction

    @contextlib.contextmanager
    def spy(conn, *args, **kw):
        at_begin.append(conn.execute("PRAGMA temp_store").fetchone()[0])
        with real(conn, *args, **kw) as c:
            yield c

    monkeypatch.setattr(DB, "transaction", spy)
    r = PA.restore(A, work)
    assert r["ok"] and sum(t["inserted"] for t in r["tables"].values()) == deleted
    assert at_begin and set(at_begin) == {1}                                                # 1 = FILE (db.connect sets 2 = MEMORY)
    # live (the base world holds reclaim mode hold): refused while SQLite's temp directory lacks room for one group's sub-journal
    H.write_maintenance(w.root, "test: live restore")
    monkeypatch.setenv("SQLITE_TMPDIR", str(tmp_path))
    assert PA.sqlite_temp_dir() == str(tmp_path)
    monkeypatch.setattr(PA, "RESTORE_TEMP_MIN_FREE_BYTES", 1 << 62)
    with pytest.raises(PA.LiveRefused, match="SQLITE_TMPDIR"):
        PA.restore(A, DB.db_path(w.root), root=w.root, live=True)
    monkeypatch.setattr(PA, "RESTORE_TEMP_MIN_FREE_BYTES", 0)
    at_begin.clear()
    r = PA.restore(A, DB.db_path(w.root), root=w.root, live=True)
    assert r["ok"] and sum(t["present_identical"] for t in r["tables"].values()) == 16     # nothing was deleted from the live file
    assert at_begin and set(at_begin) == {1}


def test_reclaim_refuses_the_live_database_and_a_partial_group_is_not_archived(base, tmp_path, monkeypatch):
    w = W(base, tmp_path)
    with pytest.raises(PA.LiveRefused):
        PA.reclaim_file(DB.db_path(w.root), w.root, P.ReclaimMode.normal())
    work = w.snap("work")
    monkeypatch.setattr(H, "RECLAIM_BATCH_ROWS", 1)
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", _stop_after(1))
    PA.reclaim_file(work, w.root, P.ReclaimMode.normal())
    monkeypatch.setattr(H, "RECLAIM_BEFORE_BATCH_HOOK", None)
    rf = _first_carrier(work, "item_result", w.rev_a)
    A, m = _archive(w, work)                                                                # verified against the database too
    assert m["skipped_partial"] == [{"table": "item_result", "revision_id": w.rev_a, "record_file_id": rf, "rows_present": 2, "rows_deleted": 1}]
    assert ("item_result", w.rev_a, rf) not in _groups(A) and m["groups"]["count"] == 5


def test_mode_writes_and_status_reads_the_state(base, tmp_path):
    w = W(base, tmp_path)
    st = PA.status(w.root)
    assert st["mode"]["kind"] == "hold" and st["pending_revisions"] == 2 and st["superseded_revisions"] == 2
    assert st["held_rehydrations"] == [] and st["ledger"] == {t: {"carriers": 0, "rows_deleted": 0} for t in P.PRUNE_TABLES}
    full = PA.status(w.root, census=True)
    assert full["census"]["by_table_class"]["item_result"]["deletable"] == {"rows": 7, "carriers": 2}
    with pytest.raises(PA.ArchiveError):
        PA.set_mode(w.root, "hold", ["/nowhere"])
    with pytest.raises(PA.ArchiveError):
        PA.set_mode(w.root, "cohort")
    plan = PA.plan(DB.db_path(w.root), w.root)
    assert plan["by_table_class"]["run_degeneration"]["deletable"] == {"rows": 2, "carriers": 2}
    PA.set_mode(w.root, "normal")
    assert not os.path.exists(P.ReclaimMode.state_path(w.root)) and PA.status(w.root)["mode"]["kind"] == "normal"
    PA.set_mode(w.root, "hold")
    assert json.load(open(P.ReclaimMode.state_path(w.root)))["mode"] == "hold"


# --------------------------------------------------------------------------------------------
# the CLI: digest + compare, rebuild --diff-out, compare --expect rebuild with a baseline
# --------------------------------------------------------------------------------------------

def test_cli_digest_compare_and_rebuild_diff_out(base, tmp_path, capsys):
    w = W(base, tmp_path)
    db = DB.db_path(w.root)
    d1, d2 = str(tmp_path / "d1.json"), str(tmp_path / "d2.json")
    assert cli.main(["--root", w.root, "prune-superseded", "digest", "--db", db, "--out", d1, "--objects", "none"]) == 0
    assert cli.main(["--root", w.root, "prune-superseded", "digest", "--db", db, "--out", d2, "--objects", "none"]) == 0
    assert cli.main(["prune-superseded", "compare", d1, d2, "--expect", "identity"]) == 0
    assert cli.main(["prune-superseded", "compare", d1, d2, "--expect", "deletion"]) == 2                       # needs --archive
    # rebuild: the live root holds every stale copy (mode hold), the rebuild deleted them -- the diff file names each one
    out, diff = str(tmp_path / "rebuilt.sqlite"), str(tmp_path / "diff.jsonl")
    assert cli.main(["--root", w.root, "rebuild", "--from", w.root, "--into", out, "--verify", "--diff-out", diff, "--workers", "0"]) == 1
    capsys.readouterr()
    lines = list(EQ.read_diffs(diff))
    stale = [d for d in lines if d["table"] in P.PRUNE_TABLES]
    assert stale and all(d["a"] and not d["b"] for d in stale)                              # live-only rows
    assert sum(len(d["a"]) for d in stale) == 16
    with DB.reading(db) as a, DB.reading(out) as b:
        counts = EQ.verify_equal(a, b)
    assert sum(len(d["a"]) + len(d["b"]) for d in lines) == sum(counts.values())
    # the same difference against itself as the baseline: nothing new, so the contract holds
    diff2 = str(tmp_path / "diff2.jsonl")
    assert cli.main(["prune-superseded", "compare", db, out, "--expect", "rebuild", "--diff-out", diff2, "--baseline-diff", diff]) == 0
    assert cli.main(["prune-superseded", "compare", db, out, "--expect", "rebuild"]) == 1
    assert cli.main(["--root", w.root, "rebuild", "--from", w.root, "--into", str(tmp_path / "x.sqlite"), "--diff-out", diff, "--workers", "0"]) == 2


def _at_schema_before_0033(src, out):
    """A copy of ``src`` taken back to the schema before migration 0033 (no ledger table, the head's old text, the old index, no 0033
    row): what the code before the pruning builds and reads."""
    from test_registry_migrations import _strip_prune_shape
    with DB.reading(src) as conn:
        conn.execute(f"VACUUM INTO '{out}'")
    c = _raw(out)
    _strip_prune_shape(c)
    c.execute("DELETE FROM schema_migration WHERE version = ?", (EQ.MIGRATION_VERSION,))
    c.close()
    return out


def test_compare_rebuild_reads_two_files_at_the_schema_before_0033_only_under_the_declared_legacy_contract(base, tmp_path, capsys):
    """The rehearsal's control is the OLD code's rebuild against the unpruned database, both at the schema before 0033 (no
    ``item_reclaim_ledger``). The ordinary rebuild comparison refuses a file that lacks a table the code declares (exit 2, the table
    named); ``--legacy-schema 32`` requires both files at that version and compares every table the code declares but the ledger, and
    nothing else changes: its diff is the current-schema diff without the ledger's lines."""
    w = W(base, tmp_path)
    db = DB.db_path(w.root)
    out = str(tmp_path / "rebuilt.sqlite")
    assert cli.main(["--root", w.root, "rebuild", "--from", w.root, "--into", out, "--workers", "0"]) == 0
    capsys.readouterr()
    cur_diff = str(tmp_path / "cur.jsonl")
    assert cli.main(["prune-superseded", "compare", db, out, "--expect", "rebuild", "--diff-out", cur_diff]) == 1
    old_live, old_rebuilt = _at_schema_before_0033(db, str(tmp_path / "old_live.sqlite")), _at_schema_before_0033(out, str(tmp_path / "old_rebuilt.sqlite"))
    legacy = EQ.MIGRATION_VERSION - 1
    assert legacy == 32 and EQ.LEGACY_SCHEMAS == {32: ("item_reclaim_ledger",)}
    capsys.readouterr()
    assert cli.main(["prune-superseded", "compare", old_live, old_rebuilt, "--expect", "rebuild"]) == 2           # refused, not a crash
    err = capsys.readouterr().err
    assert "has no table item_reclaim_ledger" in err and "--legacy-schema" in err
    assert cli.main(["prune-superseded", "compare", db, old_rebuilt, "--expect", "rebuild"]) == 2                 # one side at 32
    leg_diff, leg_out = str(tmp_path / "legacy.jsonl"), str(tmp_path / "legacy.json")
    assert cli.main(["prune-superseded", "compare", old_live, old_rebuilt, "--expect", "rebuild", "--legacy-schema", "32",
                     "--diff-out", leg_diff, "--out", leg_out]) == 1                                              # the stale copies differ
    res = json.load(open(leg_out))
    assert res["legacy_schema"] == 32 and "item_reclaim_ledger" not in res["tables_compared"] and "item_result" in res["tables_compared"]
    key = lambda d: (d["table"], json.dumps(d["key"]), json.dumps(d["a"]), json.dumps(d["b"]))
    cur = [d for d in EQ.read_diffs(cur_diff)]
    assert any(d["table"] == "item_reclaim_ledger" for d in cur)
    assert sorted(map(key, EQ.read_diffs(leg_diff))) == sorted(key(d) for d in cur if d["table"] != "item_reclaim_ledger")
    # the legacy contract is explicit: both files at the declared version, a declared version only, a rebuild comparison only
    assert cli.main(["prune-superseded", "compare", db, out, "--expect", "rebuild", "--legacy-schema", "32"]) == 2
    assert cli.main(["prune-superseded", "compare", old_live, old_rebuilt, "--expect", "rebuild", "--legacy-schema", "31"]) == 2
    assert cli.main(["prune-superseded", "compare", leg_out, leg_out, "--expect", "identity", "--legacy-schema", "32"]) == 2
    same = PA.compare_databases(old_live, old_live, legacy_schema=32)
    assert same["ok"] and same["differing"] == {}


# --------------------------------------------------------------------------------------------
# the migration contract against the real migration; the rehydrate remedies' refusals and the held listing
# --------------------------------------------------------------------------------------------

def test_the_migration_contract_holds_across_the_real_migration_and_names_any_other_change(tmp_path):
    """A database at the version before the pruning's migration, with a superseded revision already flagged ``reclaimed = 1`` (the
    flag's old meaning), digested; migrated; digested again: S0 -> S1 is exactly the migration contract."""
    from sisteer.registry import migrations as M
    files = M.migration_files()
    assert (EQ.MIGRATION_VERSION, EQ.MIGRATION_SLUG) in {(v, s) for v, s, _ in files}
    path = str(tmp_path / "before.sqlite")
    conn = DB.connect(path)
    for version, slug, mpath in files:
        if version < EQ.MIGRATION_VERSION:
            with open(mpath, "r", encoding="utf-8") as fh:
                M._apply_one(conn, version, slug, fh.read())
    t = "2026-01-01T00:00:00.000Z"
    for rev, sup, rec in (("aaaaaaaaaaaaaaaa", "bbbbbbbbbbbbbbbb", 1), ("bbbbbbbbbbbbbbbb", None, 0)):
        conn.execute("INSERT INTO source_revision (revision_id, source_id, adapter, observed_utc, inputs_json, accepted, refusal, "
                     "superseded_by_revision_id, priority, reclaimed, record_uri, ingest_seq) VALUES (?, 's', 'a', ?, '{}', 1, NULL, ?, 0, ?, 'x', 1)",
                     (rev, t, sup, rec))
    conn.close()
    d0 = PA.digest_database(path)
    assert d0["extras"]["source_revision"]["reclaimed_nonzero"] == 1 and EQ.LEDGER_TABLE not in d0["tables"]
    conn = DB.connect(path)
    assert M.apply_migrations(conn) == [EQ.MIGRATION_VERSION]
    conn.close()
    d1 = PA.digest_database(path)
    r = EQ.check_contract(d0, d1, "migration")
    assert r["ok"], r["failures"]
    assert EQ.check_contract(d1, PA.digest_database(path), "identity")["ok"]
    assert not EQ.check_contract(d0, d1, "identity")["ok"]
    bad = json.loads(json.dumps(d1))
    bad["schema"]["index:idx_item_result_withdrawn"] = bad["schema"].pop("index:idx_item_result_withdrawn")[:1] + ["CREATE INDEX x"]
    assert any("changed" in f for f in EQ.check_contract(d0, bad, "migration")["failures"])


def test_rehydrate_refuses_a_mismatched_archive_and_an_unreasoned_accept_and_status_lists_the_hold(base, tmp_path, monkeypatch):
    from sisteer.registry import validate as VAL
    w = W(base, tmp_path)
    A, _ = _archive(w, DB.db_path(w.root))
    other = w.snap("other")                                                                 # an archive of ANOTHER state of rev_a's rows
    c = _raw(other)
    c.execute("UPDATE item_result SET withdrawn_by_revision_id = 'other' WHERE source_revision_id = ? AND item_id = 'q0'", (w.rev_a,))
    c.close()
    A2 = str(tmp_path / "A2")
    PA.archive(other, w.root, A2)
    os.remove(P.ReclaimMode.state_path(w.root))                                            # normal mode: the live cycle deletes
    F._ingest(w.root, 5000)
    assert F._rows(w.root, w.rev_a) == {t: 0 for t in P.PRUNE_TABLES}
    # a marker-only re-activation of rev_a whose rehydration no longer authenticates: held, listed by status
    monkeypatch.setattr(VAL, "validate_record", F._refuse_q1_of(w.rev_a))
    F._bundle(w.root, w.rev_a, 6000, observed_i=6000, files=lambda fn: "source_revision" in fn)
    F._ingest(w.root, 6001)
    held = PA.status(w.root)["held_rehydrations"]
    assert [(h["revision"], h["kind"]) for h in held] == [(w.rev_a, H.REHYDRATION_KIND)] and "rehydration of" in held[0]["reason"]
    ledger = _q(w.root, "SELECT table_name, rows_deleted FROM item_reclaim_ledger WHERE revision_id = ? ORDER BY 1", w.rev_a)
    with pytest.raises(PA.ArchiveError, match="nothing written"):
        PA.rehydrate(w.root, w.rev_a, archive_dirs=[A2])
    with pytest.raises(PA.ArchiveError, match="--reason"):
        PA.rehydrate(w.root, w.rev_a, accept_reparse=True)
    with pytest.raises(PA.ArchiveError, match="exactly one"):
        PA.rehydrate(w.root, w.rev_a)
    assert _q(w.root, "SELECT table_name, rows_deleted FROM item_reclaim_ledger WHERE revision_id = ? ORDER BY 1", w.rev_a) == ledger
    assert F._rows(w.root, w.rev_a) == {t: 0 for t in P.PRUNE_TABLES}
    assert PA.rehydrate(w.root, w.rev_c, archive_dirs=[A])["carriers"] == 0                 # the head holds no ledger rows
    res = PA.rehydrate(w.root, w.rev_a, archive_dirs=[A])
    assert res["rows_restored"] == 7 and res["kind"] == "rehydrate_from_archive"
    log = str(tmp_path / "decisions.jsonl")
    assert cli.main(["--root", w.root, "prune-superseded", "rehydrate", "--revision", w.rev_b, "--accept-reparse",
                     "--reason", "synthetic: the current code's rows stand", "--decisions-log", log]) == 0
    lines = [json.loads(ln) for ln in open(os.path.join(w.root, "state", PA.DECISIONS_FILE))]
    assert [ln["kind"] for ln in lines] == ["rehydrate_from_archive", "rehydration_accepted"]
    assert lines[1]["reason"].startswith("synthetic") and json.loads(open(log).read())["revision"] == w.rev_b
    assert not _q(w.root, "SELECT 1 FROM item_reclaim_ledger WHERE revision_id IN (?, ?)", w.rev_a, w.rev_b)


def test_cli_archive_under_a_parent_names_the_directory_kind_utc_dbid(base, tmp_path, capsys):
    import re
    w = W(base, tmp_path)
    parent = str(tmp_path / "archives")
    os.makedirs(parent)
    db = DB.db_path(w.root)
    assert cli.main(["--root", w.root, "prune-superseded", "archive", "--db", db, "--out-parent", parent]) == 2      # no --kind
    assert cli.main(["--root", w.root, "prune-superseded", "archive", "--db", db, "--out-parent", parent, "--kind", "rehearsal"]) == 0
    out = json.loads(capsys.readouterr().out)
    (name,) = os.listdir(parent)
    assert re.fullmatch(r"rehearsal-\d{8}T\d{6}Z-[0-9a-f]{8}", name) and out["archive"] == os.path.join(parent, name)
    assert out["groups"] == 6 and PA.read_manifest(os.path.join(parent, name))["kind"] == "rehearsal"
    assert cli.main(["--root", w.root, "prune-superseded", "verify-archive", os.path.join(parent, name), "--db", db]) == 0
