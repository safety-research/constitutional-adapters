"""Migration 0032 (``site_label_rename``): the five tables whose CHECK names a site label -- fit_search, object, run, mirror_ack and
analysis -- rebuild to the code's neutral labels, every label column copied through the runner's alias table.

The tree never spells a real legacy label, so the fixtures build a database at version 31 from this tree's 0001-0031, widen the
five CHECK lists to also admit invented spellings (``oldsite_a``, ``oldsite_b``, ``oldhost``), put those spellings in rows of
all five tables, and migrate with a synthetic alias table. The same migration on a copy of a populated live replica, with the
site's real alias file, is the proof on real spellings (outside the tree).
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil

import pytest

from sisteer.registry import db as DB
from sisteer.registry import harvest as H
from sisteer.registry import migrations as M
from sisteer.registry import records as R
from sisteer.registry import schema_def as S
from sisteer.registry import site_aliases as SA
from sisteer.registry import synth
from test_registry_schema import FROZEN_MIGRATIONS

SLUG = "site_label_rename"
LEGACY = ("oldsite_a", "oldsite_b", "oldhost")
ALIASES = {"oldsite_a": "cluster_a", "oldsite_b": "cluster_b", "oldhost": "grading_host"}
LABEL_TABLES = ("fit_search", "object", "run", "mirror_ack", "analysis")
#: what each label column holds in the populated fixture before the migration, and the label it must hold after
LEGACY_ROWS = {("run", "substrate"): ("oldsite_a", "cluster_a"), ("object", "fit_substrate"): ("oldsite_b", "cluster_b"),
               ("fit_search", "substrate"): ("oldhost", "grading_host"), ("analysis", "substrate"): ("oldsite_b", "cluster_b"),
               ("mirror_ack", "destination"): ("oldsite_a", "cluster_a")}


@pytest.fixture(autouse=True)
def _empty_active_table():
    SA.reset()
    yield
    SA.reset()


def _migration():
    return next(f for f in M.migration_files() if f[1] == SLUG)


def _through_31(conn) -> None:
    _through(conn, _migration()[0] - 1)


def _through(conn, last: int) -> None:
    """This tree's 0001..``last`` applied file by file (each file alone maps through the empty alias table)."""
    for v, slug, path in M.migration_files():
        if v <= last:
            with open(path, encoding="utf-8") as fh:
                M._apply_one(conn, v, slug, fh.read())


def _widen(conn) -> None:
    """The five CHECK lists also admit the invented legacy spellings: the shape of a database built before the labels were
    neutral, with the legacy spelling made up."""
    ver = int(conn.execute("PRAGMA schema_version").fetchone()[0])
    conn.execute("PRAGMA writable_schema=ON")
    try:
        for t, c, _ in SA.substrate_columns():
            sql = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (t,)).fetchone()[0]
            new, n = re.subn(rf"(CHECK \({c} IN \()([^)]*)\)", lambda m: m.group(1) + m.group(2) + ", " + ", ".join(f"'{v}'" for v in LEGACY) + ")",
                             sql, count=1)
            assert n == 1, (t, c)
            conn.execute("UPDATE sqlite_master SET sql = ? WHERE type = 'table' AND name = ?", (new, t))
        conn.execute(f"PRAGMA schema_version = {ver + 1}")
    finally:
        conn.execute("PRAGMA writable_schema=OFF")
    conn.commit()


def _statements(sql: str):
    """The statements of a migration body, comments stripped and whitespace collapsed (string literals kept)."""
    text = DB._outside_strings(sql, r"--[^\n]*|/\*.*?\*/", " ")
    return [" ".join(s.split()) for s in DB._split_top_level(text, ";") if s.strip()]


# --------------------------------------------------------------------------------------------
# the file
# --------------------------------------------------------------------------------------------


def test_0032_is_frozen_and_its_body_is_the_render_with_the_label_columns_mapped():
    version, slug, path = _migration()
    assert version == 32 and os.path.basename(path) == "0032_site_label_rename.sql"
    with open(path, encoding="utf-8") as fh:
        committed = fh.read()
    assert FROZEN_MIGRATIONS[os.path.basename(path)] == hashlib.sha256(committed.encode("utf-8")).hexdigest()
    assert M.reserved_statement(committed) is None and slug in M.PRE_CHECK and slug not in M.POST_APPLY
    conn = DB.connect_memory()
    _through_31(conn)
    _widen(conn)
    plan = M.shape_plan(conn)
    assert [r["table"] for r in plan["rebuild"]] == ["fit_search", "object", "run", "mirror_ack", "analysis"]
    assert plan["create"] == [] and plan["add_columns"] == [] and plan["undeclared_tables"] == []
    assert plan["objects"] == {"index": {"recreate": [], "drop": []}, "trigger": {"recreate": [], "drop": []}, "view": {"recreate": [], "drop": []}}
    rendered = M.render_shape_migration(conn, plan, map_site_labels=True)
    assert _statements(committed) == _statements(rendered)
    for t, c in SA.LABEL_COLUMNS:
        assert M.site_label_select(t, c) in committed, (t, c)
    assert committed.count("CREATE TABLE ") == 5 and committed.count("ALTER TABLE ") == 5
    for name in [re.match(r"CREATE TRIGGER (\w+)", s).group(1) for s in S.triggers_on("run")]:
        assert f"CREATE TRIGGER {name} " in committed, name                     # the fence triggers ride the run rebuild
    for word in LEGACY:
        assert word not in committed


def test_a_file_applied_directly_maps_through_the_empty_alias_table():
    """A caller that applies 0032 on its own (a test building a shape file by file) gets the empty alias table for the file and none
    after it: on a database with no legacy value the copy is the identity."""
    conn = DB.connect_memory()
    _through_31(conn)
    version, slug, path = _migration()
    with open(path, encoding="utf-8") as fh:
        M._apply_one(conn, version, slug, fh.read())
    assert M.db_version(conn) == 32 and DB.schema_drift(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM sqlite_temp_master WHERE name = 'site_alias'").fetchone()[0] == 0


def test_on_a_fresh_database_0032_only_rebuilds():
    conn = DB.connect_memory()
    _through_31(conn)
    assert M.db_version(conn) == 31 and DB.schema_drift(conn) == []         # 0001-0031 already create the neutral CHECKs
    assert M.apply_migrations(conn) == [32]
    assert DB.schema_drift(conn) == [] and all(conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] == 0 for t in LABEL_TABLES)


# --------------------------------------------------------------------------------------------
# a populated database at version 31 that carries legacy spellings
# --------------------------------------------------------------------------------------------


def _populated(tmp_path):
    """A registry root whose database is at version 31 with a legacy spelling in every label column, in the live rows and in
    row_history alike (as code from before the labels wrote them). Returns (root, run_id)."""
    root = str(tmp_path / "reg")
    H.ensure_layout(root)
    ids = synth.write_corpus(root, n_runs=2, n_items=5)
    extra = [
        synth.rec("fit_search", {"search_id": "hpo:00000000000000a1", "campaign": synth.CAMPAIGN, "target_object_class_json": {"kind": "lora"},
                                 "method": "sobol", "n_trials": 2, "n_trials_completed": 2, "search_space_json": {"lr": [1e-4, 1e-3]},
                                 "objective": "heldout_loss", "objective_direction": "minimize", "substrate": "grading_host",
                                 "code_sha": synth.CODE_SHA, "started_utc": synth.t(900)}, synth.t(900)),
        synth.rec("analysis", {"analysis_id": hashlib.sha256(b"synthetic analysis").hexdigest(), "name": "study.capture", "code_sha": "c" * 40,
                               "code_ref": "branch study", "cluster_job_ids_json": ["1"], "substrate": "cluster_b", "report_uri": None,
                               "report_sha256": None, "filed_utc": synth.t(901), "filed_by": "the authors", "notes": None}, synth.t(901)),
        synth.rec("mirror_ack", {"destination": "cluster_a", "acked_through_seq": 0, "manifest_name": "CHECKSUMS-probe.sha256", "file_count": 0,
                                 "bytes": 0, "checksums_sha256": synth.SHA256_A, "acked_utc": synth.t(902), "last_error": None}, synth.t(902)),
    ]
    R.write_records(os.path.join(root, "records", "extra", "000001-extra.jsonl"), extra)
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0
    conn = DB.connect(DB.db_path(root))
    try:
        for t in LABEL_TABLES:
            assert conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] > 0, t
        conn.execute("UPDATE object SET fit_substrate = 'cluster_b' WHERE object_id = ?", (synth.OBJECT_SHA,))
        _widen(conn)
        for (t, c), (legacy, label) in LEGACY_ROWS.items():
            conn.execute(f"UPDATE {t} SET {c} = ? WHERE {c} IS NOT NULL", (legacy,))
        conn.execute("UPDATE row_history SET payload_json = json_set(payload_json, '$.substrate', 'oldsite_a'), "
                     "folded_state_json = json_set(folded_state_json, '$.substrate', 'oldsite_a') "
                     "WHERE table_name = 'run' AND json_extract(payload_json, '$.substrate') IS NOT NULL")
        # the mirror's history is filed under its destination, the natural key: old code filed it under the legacy spelling
        conn.execute("UPDATE row_history SET natural_key_json = json_set(natural_key_json, '$.destination', 'oldsite_a'), "
                     "payload_json = json_set(payload_json, '$.destination', 'oldsite_a'), "
                     "folded_state_json = json_set(folded_state_json, '$.destination', 'oldsite_a') WHERE table_name = 'mirror_ack'")
        conn.execute("DELETE FROM schema_migration WHERE version = 32")
        conn.commit()
        assert M.db_version(conn) == 31
    finally:
        conn.close()
    return root, ids["run_ids"][0]


def _label_values(conn):
    return {(t, c): sorted({r[0] for r in conn.execute(f"SELECT {c} FROM {t} WHERE {c} IS NOT NULL").fetchall()}) for (t, c) in LEGACY_ROWS}


def _counts(conn):
    return {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in LABEL_TABLES}


def test_without_aliases_0032_refuses_naming_each_legacy_value_and_applies_nothing(tmp_path):
    root, _ = _populated(tmp_path)
    conn = DB.connect(DB.db_path(root))
    before, counts = _label_values(conn), _counts(conn)
    with pytest.raises(M.MigrationFailed) as exc:
        M.apply_migrations(conn)
    msg = str(exc.value)
    assert msg.startswith("site label rename refused before anything was applied")
    for (t, c), (legacy, _) in LEGACY_ROWS.items():
        n = counts[t] if t != "object" else 1
        assert f"{t}.{c} holds {legacy!r} on {n} row(s): not a label" in msg, (t, c, msg)
    assert "site_aliases.toml" in msg and "--site-aliases" in msg
    assert M.db_version(conn) == 31 and _label_values(conn) == before and _counts(conn) == counts
    assert conn.execute("SELECT COUNT(*) FROM sqlite_temp_master WHERE name = 'site_alias'").fetchone()[0] == 0
    # an alias whose label the column does not admit (a host as a mirror destination) is refused by name too
    with pytest.raises(M.MigrationFailed) as exc2:
        M.apply_migrations(conn, site_aliases={"oldsite_a": "grading_host", "oldsite_b": "cluster_b", "oldhost": "grading_host"})
    assert "mirror_ack.destination holds 'oldsite_a' on 1 row(s): aliased to 'grading_host', which MIRROR_DESTINATIONS does not admit" in str(exc2.value)
    assert M.db_version(conn) == 31
    conn.close()


def test_with_the_alias_table_0032_maps_every_label_column_and_the_shape_is_the_codes(tmp_path):
    root, _ = _populated(tmp_path)
    conn = DB.connect(DB.db_path(root))
    counts = _counts(conn)
    assert DB.schema_drift(conn) != []
    assert M.apply_migrations(conn, site_aliases=ALIASES) == [32]
    assert _label_values(conn) == {k: [label] for k, (_, label) in LEGACY_ROWS.items()}
    assert _counts(conn) == counts
    assert DB.schema_drift(conn) == [] and DB.ddl_master(conn) == DB.code_ddl_master()
    assert M.foreign_key_violations(conn, list(LABEL_TABLES)) == []
    assert conn.execute("SELECT description FROM schema_migration WHERE version = 32").fetchone()[0] == SLUG
    conn.close()


def test_after_0032_a_lifecycle_event_over_legacy_history_folds_to_the_label(tmp_path):
    """The migrated database keeps row_history as the old code wrote it; a later event that refolds that history (a run_state)
    writes the label through the root's alias file, never the legacy spelling the new CHECK would refuse."""
    root, rid = _populated(tmp_path)
    conn = DB.connect(DB.db_path(root))
    assert M.apply_migrations(conn, site_aliases=ALIASES) == [32]
    conn.close()
    with open(os.path.join(root, SA.FILE_NAME), "w", encoding="utf-8") as fh:
        fh.write('[site_aliases]\noldsite_a = "cluster_a"\noldsite_b = "cluster_b"\noldhost = "grading_host"\n')
    patch = synth.rec("run_patch", {"run_id": rid, "fields": {"notes": "re-observed after the migration"}}, synth.t(950))
    R.write_records(os.path.join(root, "records", "extra", "000002-after.jsonl"), [patch])
    rep = H.ingest_once(root, workers=0, min_free=0)
    assert rep.refused == 0
    with DB.reading(DB.db_path(root)) as c:
        assert tuple(c.execute("SELECT substrate, notes FROM run WHERE run_id = ?", (rid,)).fetchone()) == ("cluster_a", "re-observed after the migration")
        spelled = {r[0] for r in c.execute("SELECT json_extract(payload_json, '$.substrate') FROM row_history WHERE table_name = 'run' "
                                             "AND json_extract(payload_json, '$.substrate') IS NOT NULL").fetchall()}
    assert spelled == {"oldsite_a"}                                            # history stays as written


# --------------------------------------------------------------------------------------------
# a mirror's history follows its destination's label
# --------------------------------------------------------------------------------------------


def _mirror_history(conn):
    return sorted((r[0], r[1]) for r in conn.execute("SELECT natural_key_json, json_extract(payload_json, '$.destination') FROM row_history "
                                                     "WHERE table_name = 'mirror_ack'").fetchall())


def test_0032_files_a_mirrors_history_under_its_label_so_a_later_acknowledgment_joins_it(tmp_path):
    """A mirror's natural key is its destination. 0032 files the history old code wrote under the legacy spelling under the label
    (each record still as written), so the fold reads one history per mirror: a delayed acknowledgment written before the standing
    one leaves the watermark where it is, and a later one moves it."""
    root, _ = _populated(tmp_path)
    conn = DB.connect(DB.db_path(root))
    assert _mirror_history(conn) == [('{"destination":"oldsite_a"}', "oldsite_a")]
    assert M.apply_migrations(conn, site_aliases=ALIASES) == [32]
    assert _mirror_history(conn) == [('{"destination":"cluster_a"}', "oldsite_a")]
    standing = tuple(conn.execute("SELECT destination, acked_through_seq FROM mirror_ack").fetchone())
    conn.close()
    assert standing == ("cluster_a", 0)
    with open(os.path.join(root, SA.FILE_NAME), "w", encoding="utf-8") as fh:
        fh.write('[site_aliases]\noldsite_a = "cluster_a"\noldsite_b = "cluster_b"\noldhost = "grading_host"\n')

    def ack(seq, i):
        return synth.rec("mirror_ack", {"destination": "cluster_a", "acked_through_seq": seq, "manifest_name": f"CHECKSUMS-{seq}.sha256",
                                        "file_count": 0, "bytes": 0, "checksums_sha256": synth.SHA256_A, "acked_utc": synth.t(i),
                                        "last_error": None}, synth.t(i))

    for name, rec, want in (("000003-delayed.jsonl", ack(7, 800), 0), ("000004-later.jsonl", ack(9, 990), 9)):
        R.write_records(os.path.join(root, "records", "extra", name), [rec])
        rep = H.ingest_once(root, workers=0, min_free=0)
        assert rep.refused == 0
        with DB.reading(DB.db_path(root)) as c:
            assert tuple(c.execute("SELECT destination, acked_through_seq FROM mirror_ack").fetchone()) == ("cluster_a", want), name
            keys = {r[0] for r in c.execute("SELECT natural_key_json FROM row_history WHERE table_name = 'mirror_ack'").fetchall()}
        assert keys == {'{"destination":"cluster_a"}'}, name


# --------------------------------------------------------------------------------------------
# a database older than 31 that carries legacy spellings
# --------------------------------------------------------------------------------------------


def _at_30(tmp_path):
    """The populated root taken back to the version-30 shape, as code from before the labels left it: the version-31 database's
    tables keep their rows and their legacy spellings, and the two tables 0031 rebuilds get their version-30 text (the CHECK lists
    of that shape, widened to admit the legacy spellings). Returns the root."""
    root, _ = _populated(tmp_path)
    older = DB.connect_memory()
    _through(older, 30)
    _widen(older)
    want = {str(r[0]): str(r[1]) for r in older.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall()}
    conn = DB.connect(DB.db_path(root))
    try:
        ver = int(conn.execute("PRAGMA schema_version").fetchone()[0])
        conn.execute("PRAGMA writable_schema=ON")
        for name, sql in conn.execute("SELECT name, sql FROM sqlite_master WHERE type = 'table'").fetchall():
            if name in want and DB.normalize_ddl(str(sql)) != DB.normalize_ddl(want[name]):
                conn.execute("UPDATE sqlite_master SET sql = ? WHERE type = 'table' AND name = ?", (want[name], name))
        conn.execute(f"PRAGMA schema_version = {ver + 1}")
        conn.execute("PRAGMA writable_schema=OFF")
        conn.execute("DELETE FROM schema_migration WHERE version = 31")
        conn.commit()
        assert M.db_version(conn) == 30 and DB.ddl_master(conn) == DB.ddl_master(older)
    finally:
        conn.close()
        older.close()
    return root


def test_a_database_older_than_31_with_legacy_spellings_is_refused_by_name_and_migrates_once_it_is_at_31(tmp_path):
    """The migrations before 0032 rebuild tables with CHECK lists that admit only the labels and copy every value as stored, so a
    legacy spelling in a column one of them copies cannot pass; only 0032 maps spellings. The runner refuses such a database before
    anything is applied, naming the file, the column, the value and its rows, and the way through: version 31 under the migrations
    of the code that wrote it, then 0032 here. The same database, brought to 31 that way, migrates and maps every column."""
    root = _at_30(tmp_path)
    conn = DB.connect(DB.db_path(root))
    before, counts = _label_values(conn), _counts(conn)
    for table in (ALIASES, None):
        with pytest.raises(M.MigrationFailed) as exc:
            M.apply_migrations(conn, site_aliases=table)
        msg = str(exc.value)
        assert msg.startswith("a database at version 30 is refused before anything was applied"), msg
        assert f"0031_non_template_positions rebuilds run and copies run.substrate, which holds 'oldsite_a' on {counts['run']} row(s)" in msg, msg
        assert ("(aliased to 'cluster_a')" if table else "(not aliased)") in msg, msg
        assert "version 31 with the code that wrote it" in msg and "0032_site_label_rename" in msg, msg
        assert "object.fit_substrate" not in msg                     # 0031 does not rebuild object: 0032's own check reads it
        assert M.db_version(conn) == 30 and _label_values(conn) == before and _counts(conn) == counts
        assert conn.execute("SELECT COUNT(*) FROM sqlite_temp_master WHERE name = 'site_alias'").fetchone()[0] == 0
    # the way through: 0031 as the code that wrote the database had it (its CHECK lists admit the spellings), then 0032 maps them
    v31, slug31, path31 = next(f for f in M.migration_files() if f[0] == 31)
    with open(path31, encoding="utf-8") as fh:
        body, n = re.subn(r"(CHECK \(substrate IN \()([^)]*)\)", lambda m: m.group(1) + m.group(2) + ", " + ", ".join(f"'{v}'" for v in LEGACY) + ")",
                          fh.read())
    assert n == 1
    M._apply_one(conn, v31, slug31, body)
    assert M.db_version(conn) == 31
    assert M.apply_migrations(conn, site_aliases=ALIASES) == [32]
    assert _label_values(conn) == {k: [label] for k, (_, label) in LEGACY_ROWS.items()} and _counts(conn) == counts
    assert DB.schema_drift(conn) == []
    conn.close()


def test_a_database_older_than_31_that_holds_only_labels_migrates_straight_through():
    conn = DB.connect_memory()
    _through(conn, 29)
    assert M.apply_migrations(conn) == [30, 31, 32] and DB.schema_drift(conn) == []
