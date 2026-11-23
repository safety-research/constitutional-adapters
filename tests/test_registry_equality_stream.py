"""The streaming ``verify_equal`` (equality.py) against the set-based comparison it replaced, and the pruning's comparison
contracts.

The stream merges two databases on each table's primary key; the set-based form held every row of both sides in memory. They must
report the same symmetric-difference size per table on every input: one-sided keys, changed values, floats that differ only below the
12-place rounding, BLOBs, NULLs, mixed-type keys and keys that repeat (a rowid table's PRIMARY KEY admits several NULL keys). A
merge whose order SQLite does not share fails loudly (``OrderMismatch``), never with a wrong count.
"""
from __future__ import annotations

import json
import os
import random
import shutil
import sqlite3

import pytest

from sisteer.registry import db as DB
from sisteer.registry import equality as EQ
from sisteer.registry import harvest as H
from sisteer.registry import schema_def as S
from sisteer.registry import synth

from test_registry_core_fixes import StoreAdapter, _adapter_setup, _pass, _store


# ---- the merge on synthetic tables -------------------------------------------------------------------------------------------

KEY_POOL = [None, 0, 1, 2, -5, 2.5, 7.25, "a", "b", "B", "zz", "é", b"\x00", b"\x01\x02", b"\xff"]
VALUE_POOL = [None, 0, 1, 3, 0.1, 1 / 3, 1e300, -0.0, "x", "y", b"\x00", b"\xab"]


def _toy(path, rows):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("CREATE TABLE t (k1, k2 INTEGER, v, w REAL, PRIMARY KEY (k1, k2))")
    conn.executemany("INSERT OR IGNORE INTO t VALUES (?, ?, ?, ?)", rows)
    return conn


def _sets(a, b):
    sql = "SELECT k1, k2, v, w FROM t"
    ra = {EQ.norm_row(r) for r in a.execute(sql).fetchall()}
    rb = {EQ.norm_row(r) for r in b.execute(sql).fetchall()}
    return len(ra ^ rb)


def _random_rows(rng, n):
    rows = []
    for _ in range(n):
        rows.append((rng.choice(KEY_POOL), rng.randrange(4), rng.choice(VALUE_POOL), rng.choice([None, 0.5, 1.0, 2.0 / 3])))
    return rows


def _perturb(rng, rows):
    out = list(rows)
    for _ in range(rng.randrange(6)):
        if not out:
            break
        i = rng.randrange(len(out))
        k1, k2, v, w = out[i]
        op = rng.randrange(5)
        if op == 0:
            out.pop(i)                                                        # a one-sided key
        elif op == 1:
            out[i] = (k1, k2, rng.choice(VALUE_POOL), w)                      # a changed value (sometimes the same)
        elif op == 2 and isinstance(w, float):
            out[i] = (k1, k2, v, w + 1e-14)                                   # below the rounding: equal
        elif op == 3 and isinstance(w, float):
            out[i] = (k1, k2, v, w + 1e-6)                                    # above it: different
        else:
            out.append((rng.choice(KEY_POOL), rng.randrange(4), rng.choice(VALUE_POOL), None))
    return out


@pytest.mark.parametrize("seed", range(40))
def test_stream_equals_the_set_based_count_on_random_tables(tmp_path, seed):
    rng = random.Random(seed)
    rows = _random_rows(rng, rng.randrange(0, 40))
    a = _toy(str(tmp_path / "a.sqlite"), rows)
    b = _toy(str(tmp_path / "b.sqlite"), _perturb(rng, rows))
    seen = []
    got = EQ.table_diff(a, b, "t", ["k1", "k2", "v", "w"], ["k1", "k2"], diff_sink=lambda *x: seen.append(x), fetch=3)
    assert got == _sets(a, b)
    assert sum(len(x[2]) + len(x[3]) for x in seen) == got                   # the sink sees exactly the counted rows


def test_null_keys_repeat_and_compare_as_sets(tmp_path):
    rows = [(None, 1, "x", None), (None, 1, "y", None), (None, 1, "x", None)]        # three rows, two distinct, under one NULL key
    a = _toy(str(tmp_path / "a.sqlite"), rows)
    b = _toy(str(tmp_path / "b.sqlite"), [(None, 1, "y", None), (None, 1, "z", None)])
    assert EQ.table_diff(a, b, "t", ["k1", "k2", "v", "w"], ["k1", "k2"]) == _sets(a, b) == 2


def test_float_rounding_and_blobs_follow_the_set_based_rule(tmp_path):
    a = _toy(str(tmp_path / "a.sqlite"), [("k", 0, b"\x01", 0.1), ("k", 1, "t", 1.0)])
    b = _toy(str(tmp_path / "b.sqlite"), [("k", 0, b"\x01", 0.1 + 1e-14), ("k", 1, "t", 1.0 + 1e-9)])
    assert EQ.table_diff(a, b, "t", ["k1", "k2", "v", "w"], ["k1", "k2"]) == _sets(a, b) == 2


def test_an_order_sqlite_does_not_share_fails_loudly(tmp_path):
    """A NOCASE key orders 'a' before 'B'; the merge's BINARY order puts 'B' first. The stream refuses rather than mis-merge."""
    conn = sqlite3.connect(str(tmp_path / "n.sqlite"), isolation_level=None)
    conn.execute("CREATE TABLE t (k TEXT COLLATE NOCASE PRIMARY KEY, v)")
    conn.executemany("INSERT INTO t VALUES (?, ?)", [("a", 1), ("B", 2)])
    with pytest.raises(EQ.OrderMismatch):
        EQ.table_diff(conn, conn, "t", ["k", "v"], ["k"])


def test_value_encoding_round_trips_exactly():
    vals = [None, 0, -1, 2 ** 63 - 1, -(2 ** 63), 0.0, -0.0, 5e-324, 2.2250738585072014e-308, 1e308, float("inf"), 1 / 3,
            "", "é☃\U0001F600", b"", b"\x00\xff", "{\"r\": \"x\"}"]
    for v in vals:
        enc = json.loads(json.dumps(EQ.encode_value(v)))
        back = EQ.decode_value(enc)
        assert type(back) is type(v) and (back == v or (back != back and v != v))
        if isinstance(v, float):
            assert back.hex() == v.hex()                                    # -0.0 and subnormals keep their bits


# ---- REAL key columns: the merge normalizes the key as the set-based form did ------------------------------------------------

X03 = 0.1 + 0.2                                       # 0.30000000000000004: rounds to 0.3 at 12 places
M_COLS = ["run", "thr", "comp", "v"]
M_KEY = ["run", "thr", "comp"]


def _mtoy(path, rows):
    """``run_metric``'s key shape: a REAL in the middle of the primary key, a TEXT after it."""
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("CREATE TABLE m (run TEXT, thr REAL, comp TEXT, v, PRIMARY KEY (run, thr, comp))")
    conn.executemany("INSERT OR IGNORE INTO m VALUES (?, ?, ?, ?)", rows)
    return conn


def _msets(a, b):
    sql = "SELECT run, thr, comp, v FROM m"
    return len({EQ.norm_row(r) for r in a.execute(sql)} ^ {EQ.norm_row(r) for r in b.execute(sql)})


def test_rounded_real_keys_on_both_sides_are_one_key(tmp_path):
    a = _mtoy(str(tmp_path / "a.sqlite"), [("r", X03, "c1", 1), ("r", 0.5, "c1", 2), ("s", X03, "c1", 3)])
    b = _mtoy(str(tmp_path / "b.sqlite"), [("r", 0.3, "c1", 1), ("r", 0.5, "c1", 2), ("s", 0.3, "c1", 4)])
    seen = []
    assert EQ.table_diff(a, b, "m", M_COLS, M_KEY, diff_sink=lambda *x: seen.append(x)) == _msets(a, b) == 2
    assert [(x[1][0], len(x[2]), len(x[3])) for x in seen] == [("s", 1, 1)]          # one key: the rounded threshold matched


def test_raw_keys_collapsing_onto_one_normalized_key_compare_as_one_set(tmp_path):
    # side a: two raw keys in one rounding bucket; SQLite orders them by the raw threshold BEFORE the computation id, so the stream meets
    # (r, 0.3, "z") before (r, 0.3..04, "a") -- normalized, out of order: the block of that prefix is regrouped
    a = _mtoy(str(tmp_path / "a.sqlite"), [("r", 0.3, "z", 1), ("r", X03, "a", 1), ("r", 0.7, "a", 5)])
    b = _mtoy(str(tmp_path / "b.sqlite"), [("r", 0.3, "a", 1), ("r", 0.3, "z", 1), ("r", 0.7, "a", 5)])
    assert EQ.table_diff(a, b, "m", M_COLS, M_KEY, fetch=1) == _msets(a, b) == 0
    # two raw keys of one normalized key with different values: one key, its rows compared as one set
    a2 = _mtoy(str(tmp_path / "a2.sqlite"), [("r", 0.3, "a", 1), ("r", X03, "a", 2)])
    b2 = _mtoy(str(tmp_path / "b2.sqlite"), [("r", 0.3, "a", 1)])
    seen = []
    assert EQ.table_diff(a2, b2, "m", M_COLS, M_KEY, diff_sink=lambda *x: seen.append(x)) == _msets(a2, b2) == 1
    assert len(seen) == 1 and [EQ.norm_row(r) for r in seen[0][2]] == [("r", 0.3, "a", 2)] and seen[0][3] == []


THR_POOL = [None, 0, 1, 0.3, X03, 0.30000000000001, 1 / 3, 0.333333333333, 2.5, "t"]


@pytest.mark.parametrize("seed", range(30))
def test_stream_equals_the_set_based_count_on_random_real_keyed_tables(tmp_path, seed):
    rng = random.Random(1000 + seed)
    rows = [(rng.choice(["r", "s"]), rng.choice(THR_POOL), rng.choice(["a", "b", "z"]), rng.choice([1, 2, 0.5, 0.5 + 1e-14]))
            for _ in range(rng.randrange(0, 30))]
    other = [(r[0], X03 if r[1] == 0.3 else r[1], r[2], r[3]) for r in rows if rng.random() > 0.15]
    other += [(rng.choice(["r", "s"]), rng.choice(THR_POOL), rng.choice(["a", "b", "z"]), 1) for _ in range(rng.randrange(3))]
    a = _mtoy(str(tmp_path / "a.sqlite"), rows)
    b = _mtoy(str(tmp_path / "b.sqlite"), other)
    seen = []
    got = EQ.table_diff(a, b, "m", M_COLS, M_KEY, diff_sink=lambda *x: seen.append(x), fetch=2)
    assert got == _msets(a, b)
    assert sum(len(x[2]) + len(x[3]) for x in seen) == got


def test_a_float_in_an_integer_key_column_stops_the_merge(tmp_path):
    """The exact path normalizes no key: a non-integral value an INTEGER key column kept would be compared raw, so the merge refuses."""
    a = _toy(str(tmp_path / "a.sqlite"), [])
    a.execute("CREATE TABLE i (k TEXT, n INTEGER, v, PRIMARY KEY (k, n))")
    a.execute("INSERT INTO i VALUES ('k', 1.5, 1)")
    with pytest.raises(EQ.OrderMismatch, match="REAL key value"):
        EQ.table_diff(a, a, "i", ["k", "n", "v"], ["k", "n"])


# ---- a real registry: the stream, the set-based form and harvest.verify_equal agree -------------------------------------------

def _registry(tmp_path):
    root, store_root = _adapter_setup(tmp_path, [("s1", "store_1", "constitution_lora@aaaa", [0.2, 0.8, 0.4]),
                                                 ("s2", "store_2", "none", [0.1, 0.9])])
    ad = StoreAdapter(store_root)
    _pass(root, ad, synth.t(2000))
    _store(store_root, "s1", "store_1", "constitution_lora@aaaa", [0.3, 0.8, 0.4, 0.6])
    _pass(root, ad, synth.t(3000))
    return root


def _raw(path):
    conn = sqlite3.connect(path, isolation_level=None)
    conn.execute("PRAGMA foreign_keys=OFF")
    return conn


def test_stream_set_based_and_harvest_agree_on_a_registry_and_on_its_perturbed_copy(tmp_path):
    root = _registry(tmp_path)
    live = DB.db_path(root)
    copy = str(tmp_path / "copy.sqlite")
    with DB.reading(live) as conn:
        conn.execute(f"VACUUM INTO '{copy}'")
    with DB.reading(live) as a, DB.reading(copy) as b:
        assert EQ.verify_equal(a, b) == EQ.verify_equal_sets(a, b) == H.verify_equal(a, b)
        assert not any(EQ.verify_equal(a, b).values())
    c = _raw(copy)
    first = c.execute("SELECT run_id, item_id, metric_id, source_revision_id FROM item_result ORDER BY 1, 2, 3, 4").fetchall()
    assert len(first) >= 3
    c.execute("UPDATE item_result SET value = value + 1e-14 WHERE run_id = ? AND item_id = ? AND metric_id = ? AND source_revision_id = ?",
              first[0])                                                     # below the rounding: still equal
    c.execute("UPDATE item_result SET value = value + 0.5 WHERE run_id = ? AND item_id = ? AND metric_id = ? AND source_revision_id = ?",
              first[1])                                                     # a changed value: 2
    c.execute("DELETE FROM item_result WHERE run_id = ? AND item_id = ? AND metric_id = ? AND source_revision_id = ?", first[2])  # 1
    c.execute("UPDATE run_metric SET value = 0.123")                        # every run_metric row differs
    c.execute("DELETE FROM record_delivery")                                # an excluded table: not compared
    c.close()
    with DB.reading(live) as a, DB.reading(copy) as b:
        got = EQ.verify_equal(a, b)
        assert got == EQ.verify_equal_sets(a, b)
        assert got["item_result"] == 3
        assert got["run_metric"] == 2 * a.execute("SELECT COUNT(*) FROM run_metric").fetchone()[0]
        assert "record_delivery" not in got
        assert set(EQ.verify_equal(a, b, tables=["item_result"])) == {"item_result"}


def test_run_metric_thresholds_below_the_rounding_merge_as_one_key_on_a_registry(tmp_path):
    """``run_metric`` carries the REAL ``threshold`` in its primary key: a threshold that moved below the 12-place rounding is the same
    key, and a second raw key in one rounding bucket collapses into it, exactly as the set-based comparison counts them."""
    root = _registry(tmp_path)
    copies = []
    for name in ("a", "b"):
        out = str(tmp_path / f"{name}.sqlite")
        with DB.reading(DB.db_path(root)) as conn:
            conn.execute(f"VACUUM INTO '{out}'")
        copies.append(out)
    ca, cb = _raw(copies[0]), _raw(copies[1])
    n = ca.execute("SELECT COUNT(*) FROM run_metric").fetchone()[0]
    assert n >= 2
    for c in (ca, cb):
        c.execute("UPDATE run_metric SET threshold = 0.3")
    cb.execute("UPDATE run_metric SET threshold = ?", (X03,))                       # every key moved below the rounding
    first = ca.execute("SELECT * FROM run_metric ORDER BY run_id, judgment_id, metric_id, computation_id LIMIT 1").fetchone()
    cols = [r[1] for r in ca.execute("PRAGMA table_info(run_metric)")]
    twin = dict(zip(cols, first), threshold=0.3 + 1e-14)                             # a second raw key of the same normalized key
    ca.execute(f"INSERT INTO run_metric ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})", [twin[c] for c in cols])
    ca.close()
    cb.close()
    with DB.reading(copies[0]) as a, DB.reading(copies[1]) as b:
        assert EQ.verify_equal(a, b, tables=["run_metric"]) == EQ.verify_equal_sets(a, b, tables=["run_metric"]) == {"run_metric": 0}
    cb = _raw(copies[1])
    cb.execute("UPDATE run_metric SET value = COALESCE(value, 0) + 0.25 WHERE run_id = ? AND computation_id = ?",
               (first[cols.index("run_id")], first[cols.index("computation_id")]))
    cb.close()
    with DB.reading(copies[0]) as a, DB.reading(copies[1]) as b:
        got = EQ.verify_equal(a, b, tables=["run_metric"])
        assert got == EQ.verify_equal_sets(a, b, tables=["run_metric"]) and got["run_metric"] > 0


# ---- classifying a diff against a baseline: only an identical residual is waived -----------------------------------------------

def _diff_file(tmp_path, name, a, b):
    path = str(tmp_path / f"{name}.jsonl")
    with EQ.DiffWriter(path) as w:
        EQ.table_diff(a, b, "t", ["k1", "k2", "v", "w"], ["k1", "k2"], diff_sink=w)
    return path


def test_a_baseline_key_whose_rows_changed_is_a_new_difference(tmp_path):
    ref = _toy(str(tmp_path / "ref.sqlite"), [("k", 0, "x", None), ("k", 1, "y", 0.5)])
    old = _toy(str(tmp_path / "old.sqlite"), [("k", 0, "x", None), ("k", 1, "Y", 0.5)])           # the baseline residual at (k, 1)
    base = _diff_file(tmp_path, "base", ref, old)
    same = _toy(str(tmp_path / "same.sqlite"), [("k", 0, "x", None), ("k", 1, "Y", 0.5 + 1e-14)])  # the same residual (normalized)
    assert EQ.classify_diffs(_diff_file(tmp_path, "same", ref, same), base)["ok"]
    changed = _toy(str(tmp_path / "changed.sqlite"), [("k", 0, "x", None), ("k", 1, "Z", 0.5)])  # the key remains; its rows changed
    cls = EQ.classify_diffs(_diff_file(tmp_path, "changed", ref, changed), base)
    assert not cls["ok"] and (cls["new"], cls["baseline"], cls["changed_at_baseline_key"]) == (1, 0, 1)
    assert cls["tables"]["t"] == {"new": 1, "baseline": 0, "changed_at_baseline_key": 1} and cls["new_examples"][0]["key"] == ["k", 1]


def test_a_baseline_key_whose_candidate_lost_the_row_is_a_new_difference(tmp_path):
    ref = _toy(str(tmp_path / "ref.sqlite"), [("k", 0, "x", None), ("k", 1, "y", None)])
    old = _toy(str(tmp_path / "old.sqlite"), [("k", 0, "x", None), ("k", 1, "Y", None)])
    base = _diff_file(tmp_path, "base", ref, old)
    lost = _toy(str(tmp_path / "lost.sqlite"), [("k", 0, "x", None)])                             # (k, 1) gone on the candidate's side
    cls = EQ.classify_diffs(_diff_file(tmp_path, "lost", ref, lost), base)
    assert not cls["ok"] and cls["changed_at_baseline_key"] == 1 and cls["baseline"] == 0
    # and a candidate that FIXED the residual has nothing to classify
    assert EQ.classify_diffs(_diff_file(tmp_path, "fixed", ref, ref), base)["ok"]


def test_default_exclusions_are_harvests(tmp_path):
    tables, columns = EQ.default_exclusions()
    assert tables == tuple(H.VERIFY_EXCLUDE_TABLES) and columns == dict(H.VERIFY_EXCLUDE_COLUMNS)
    names = [t for t, _c, _k in EQ.compared_tables()]
    assert not set(names) & set(H.VERIFY_EXCLUDE_TABLES)
    for t, cols, key in EQ.compared_tables():
        assert key == [c for c in S.TABLE_BY_NAME[t].pk if c in cols]
        assert not set(cols) & set(H.VERIFY_EXCLUDE_COLUMNS.get(t, ()))


def test_diff_file_names_each_differing_key_and_classifies_against_a_baseline(tmp_path):
    a = _toy(str(tmp_path / "a.sqlite"), [("k", 0, "x", None), ("k", 1, "y", None), ("k", 2, "z", None)])
    b = _toy(str(tmp_path / "b.sqlite"), [("k", 0, "x", None), ("k", 1, "Y", None)])
    base = _toy(str(tmp_path / "base.sqlite"), [("k", 0, "x", None), ("k", 1, "Y", None), ("k", 2, "z", None)])
    with EQ.DiffWriter(str(tmp_path / "new.jsonl")) as w:
        assert EQ.table_diff(a, b, "t", ["k1", "k2", "v", "w"], ["k1", "k2"], diff_sink=w) == 3
    with EQ.DiffWriter(str(tmp_path / "base.jsonl")) as w:
        assert EQ.table_diff(a, base, "t", ["k1", "k2", "v", "w"], ["k1", "k2"], diff_sink=w) == 2
    diffs = list(EQ.read_diffs(str(tmp_path / "new.jsonl")))
    assert [(d["key"], len(d["a"]), len(d["b"])) for d in diffs] == [(["k", 1], 1, 1), (["k", 2], 1, 0)]
    cls = EQ.classify_diffs(str(tmp_path / "new.jsonl"), str(tmp_path / "base.jsonl"))
    assert cls["baseline"] == 1 and cls["new"] == 1 and not cls["ok"]
    assert cls["tables"]["t"] == {"new": 1, "baseline": 1, "changed_at_baseline_key": 0}
    assert cls["new_examples"][0]["key"] == ["k", 2]
    assert EQ.classify_diffs(str(tmp_path / "base.jsonl"), str(tmp_path / "base.jsonl"))["ok"]


# ---- the comparison contracts over digest documents ---------------------------------------------------------------------------

def _d(rows, s):
    return {"rows": rows, "sum256": f"{s:064x}"}


def _doc(**over):
    doc = {
        "schema": {"table:item_result": ["item_result", "CREATE TABLE item_result (…)"],
                   "index:idx_item_result_withdrawn_seq": ["item_result", "CREATE INDEX idx_item_result_withdrawn_seq …"],
                   "view:v_paper_inputs_head": ["v_paper_inputs_head", "CREATE VIEW v_paper_inputs_head AS old"]},
        "tables": {"item_result": _d(10, 100), "item_flags": _d(4, 40), "run_degeneration": _d(2, 20), "run": _d(3, 30),
                   "source_revision": _d(2, 5), "schema_migration": _d(2, 7)},
        "objects": {"v_paper_inputs_head": _d(1, 9), "v_item_result_current": _d(5, 50)},
        "extras": {"source_revision": {"without_reclaimed": _d(2, 11), "reclaimed_nonzero": 1, "reclaimed_mismatch": 1},
                   "schema_migration": [[1, "2026-01-01T00:00:00.000Z", "initial"], [32, "2026-10-01T00:00:00.000Z", "x"]]},
    }
    doc.update(over)
    return doc


def _migrated(s0):
    s1 = json.loads(json.dumps(s0))
    del s1["schema"]["index:idx_item_result_withdrawn_seq"]
    s1["schema"]["view:v_paper_inputs_head"] = ["v_paper_inputs_head", "CREATE VIEW v_paper_inputs_head AS new"]
    s1["schema"]["table:item_reclaim_ledger"] = ["item_reclaim_ledger", "CREATE TABLE item_reclaim_ledger (…)"]
    s1["tables"]["item_reclaim_ledger"] = _d(0, 0)
    s1["tables"]["source_revision"] = _d(2, 6)                                   # reclaimed reset: the full digest moves
    s1["tables"]["schema_migration"] = _d(3, 8)
    s1["extras"]["source_revision"] = {"without_reclaimed": _d(2, 11), "reclaimed_nonzero": 0, "reclaimed_mismatch": 1}
    s1["extras"]["schema_migration"] = s0["extras"]["schema_migration"] + [[33, "2026-10-20T00:00:00.000Z", "prune_superseded_item_rows"]]
    s1["extras"]["item_reclaim_ledger"] = []
    return s1


def test_migration_contract_holds_and_catches_any_other_change():
    s0 = _doc()
    s1 = _migrated(s0)
    ok = EQ.check_contract(s0, s1, "migration")
    assert ok["ok"], ok["failures"]
    bad = json.loads(json.dumps(s1))
    bad["tables"]["item_result"] = _d(10, 101)                                  # an item row changed
    bad["extras"]["source_revision"]["reclaimed_nonzero"] = 2
    bad["schema"]["index:idx_other"] = ["run", "CREATE INDEX idx_other …"]
    r = EQ.check_contract(s0, bad, "migration")
    assert not r["ok"]
    assert any("item_result" in f for f in r["failures"]) and any("reclaimed" in f for f in r["failures"])
    assert any("added" in f for f in r["failures"])
    two = json.loads(json.dumps(s1))
    two["extras"]["schema_migration"].append([34, "x", "other"])
    assert not EQ.check_contract(s0, two, "migration")["ok"]


def _deleted(s1, archive):
    s2 = json.loads(json.dumps(s1))
    for t, (n, s) in archive.items():
        rows, sm = EQ.dsum(s1["tables"][t])
        s2["tables"][t] = EQ.dfmt((rows - n, (sm - s) % EQ.M256))
    s2["tables"]["source_revision"] = _d(2, 77)
    s2["extras"]["source_revision"] = {"without_reclaimed": _d(2, 11), "reclaimed_nonzero": 1, "reclaimed_mismatch": 0,
                                       "reclaimed_unexplained": 0}
    s2["extras"]["item_reclaim_ledger"] = [["item_result", "rev_a", 11, 6], ["item_flags", "rev_a", 12, 4]]
    return s2


def test_deletion_contract_is_additive_against_the_archive():
    s1 = _migrated(_doc())
    archive = {"item_result": (6, 60), "item_flags": (4, 40)}
    manifest = {"tables": {t: {"rows": n, "digest_all": _d(n, s)} for t, (n, s) in archive.items()}}
    groups = [{"table": "item_result", "revision_id": "rev_a", "record_file_id": 11, "rows": 6, "class": "deletable"},
              {"table": "item_flags", "revision_id": "rev_a", "record_file_id": 12, "rows": 4, "class": "deletable"},
              {"table": "item_flags", "revision_id": "rev_x", "record_file_id": 99, "rows": 7, "class": "unauthentic_digest"}]
    s2 = _deleted(s1, archive)
    r = EQ.check_contract(s1, s2, "deletion", manifest, groups)
    assert r["ok"], r["failures"]
    short = json.loads(json.dumps(s2))
    short["tables"]["item_result"] = EQ.dfmt((3, 1))                            # a row deleted that the archive lacks
    assert not EQ.check_contract(s1, short, "deletion", manifest, groups)["ok"]
    ledger = json.loads(json.dumps(s2))
    ledger["extras"]["item_reclaim_ledger"] = ledger["extras"]["item_reclaim_ledger"][:1]
    assert not EQ.check_contract(s1, ledger, "deletion", manifest, groups)["ok"]
    view = json.loads(json.dumps(s2))
    view["objects"]["v_item_result_current"] = _d(5, 51)                        # a visible row changed
    assert not EQ.check_contract(s1, view, "deletion", manifest, groups)["ok"]
    rec = json.loads(json.dumps(s2))
    rec["extras"]["source_revision"]["reclaimed_unexplained"] = 1                # a 1 on a head, or a 0 with nothing left to delete
    assert not EQ.check_contract(s1, rec, "deletion", manifest, groups)["ok"]
    kept = json.loads(json.dumps(s2))
    kept["extras"]["source_revision"]["reclaimed_mismatch"] = 1                  # a revision kept due by a missing carrier: its rows remain
    assert EQ.check_contract(s1, kept, "deletion", manifest, groups)["ok"]
    old = json.loads(json.dumps(s2))
    del old["extras"]["source_revision"]["reclaimed_unexplained"]                 # a digest taken before the check existed: taken again
    assert not EQ.check_contract(s1, old, "deletion", manifest, groups)["ok"]
    with pytest.raises(ValueError):
        EQ.check_contract(s1, s2, "deletion")


def test_identity_contract_requires_everything_equal():
    s2 = _migrated(_doc())
    assert EQ.check_contract(s2, json.loads(json.dumps(s2)), "identity")["ok"]
    s3 = json.loads(json.dumps(s2))
    s3["tables"]["source_revision"] = _d(2, 0)
    assert not EQ.check_contract(s2, s3, "identity")["ok"]
    with pytest.raises(ValueError):
        EQ.check_contract(s2, s3, "nonsense")
