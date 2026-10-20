"""Ordered schema migrations (``NNNN_<slug>.sql``), each applied in one transaction and recorded in
``schema_migration``. A fresh database is created by applying ALL migrations, never by executing
``schema.sql`` directly; ``tests/test_registry_schema.py::test_schema_sql_equals_migrations`` asserts
the two produce the same normalised ``sqlite_master``, and ``test_landed_migrations_are_frozen`` pins every
landed file's sha256 — a schema change is a NEW file, never an edit of a landed one (the in-place edits of
``0001_initial.sql`` left the live database at an older shape than the code at the same version, 2026-09-17 and
2026-09-19; ``db.schema_drift`` catches that, this rule prevents it).

The runner (``apply_migrations``) wraps each file in ``BEGIN`` … ``COMMIT`` with ``PRAGMA foreign_keys=OFF`` and
``PRAGMA legacy_alter_table=ON`` for the file's duration — the documented table-rebuild dance needs both: SQLite
cannot alter a CHECK, NOT NULL or DEFAULT in place, so a constraint change creates ``<table>__new`` from the code's
DDL, copies the rows, drops the old table and renames the new one; ``DROP TABLE`` under ``foreign_keys=ON`` would
refuse a parent with children, and a non-legacy ``RENAME`` rewrites every other table's ``REFERENCES`` and fails on a
view that names the dropped table. After the file the runner runs ``PRAGMA foreign_key_check`` over every table whose
b-tree the file recreated and every table referencing one, and rolls the migration back on a violation. A migration
file is a bare statement list: no ``BEGIN``/``COMMIT``, no ``PRAGMA``.

A migration may carry a PYTHON post-step beside its SQL (``POST_APPLY``, keyed by the file's slug so a renumbering never detaches it):
a data lift SQL cannot express byte for byte -- the lift of ``object.parent_object_id`` / ``object.named_role`` from ``extra_json``
is a REFOLD of every object key under the new fold (``reducer.fold``), whose ``field_versions_json`` and ``folded_state_json`` are
canonical JSON (sorted keys) that ``json_set`` cannot produce. The step runs inside the file's transaction, after its statements and
before the version row commits, is idempotent (a second run changes nothing), and returns the line ``registry migrate`` prints
("lifted object lineage: ..."). It is migration code, never a hand edit of the database.

``shape_plan`` / ``render_shape_migration`` derive the migration a drifted database needs from ``schema_def`` — ``ALTER TABLE
… ADD COLUMN`` for a table that only lacks columns the code declares LAST (``add_columns``: O(1), no row copied — SQLite
appends an added column after the last existing one, so the stored CREATE text equals the code's only then; 0020 is the
worked example), the rebuild dance for every other table whose CREATE text differs, ``DROP``/``CREATE`` for every differing
index, trigger and ``v_*`` object — for ``registry migrate --plan [--sql]``; the rendered body is saved as the next
``NNNN_<slug>.sql`` and reviewed, never applied unsaved (the files are the executable history).

A migration that renames vocabulary values reads the site's alias table, never a spelling in its own text: before any pending
file the runner creates ``TEMP TABLE site_alias(old, new)`` on the connection and fills it from the ``site_aliases`` argument
(empty by default; ``site_aliases.py``), and the file maps a column through it
(``COALESCE((SELECT new FROM temp.site_alias WHERE old = t.c), t.c)``). A check registered in ``PRE_CHECK`` under the file's slug
runs before the file and refuses by name what the file could not map (0032: a stored value that is neither a label nor aliased).
The files before that one copy a label column as stored, so a database below it that holds a site's own spelling in a table they
rebuild is refused by name before anything is applied (:func:`check_labels_before_rename`).
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
_MIG_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.sql$")
#: The suffix of the table a rebuild creates beside the old one before the swap.
REBUILD_SUFFIX = "__new"


class MigrationFailed(RuntimeError):
    """A migration file did not apply cleanly: a statement failed, it managed its own transaction, or it left a
    foreign-key violation. The database is as it was before the file (the transaction rolled back)."""


def migration_files() -> List[Tuple[int, str, str]]:
    """[(version, slug, path)] in version order; a duplicate or non-contiguous version is an error."""
    out = []
    for fn in sorted(os.listdir(HERE)):
        m = _MIG_RE.match(fn)
        if m:
            out.append((int(m.group(1)), m.group(2), os.path.join(HERE, fn)))
    versions = [v for v, _, _ in out]
    if versions != list(range(1, len(versions) + 1)):
        raise RuntimeError(f"migrations must be contiguous from 0001: {versions}")
    return out


def code_version() -> int:
    return len(migration_files())


def db_version(conn: sqlite3.Connection) -> int:
    has = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migration'").fetchone()
    if not has:
        return 0
    row = conn.execute("SELECT COALESCE(MAX(version), 0) FROM schema_migration").fetchone()
    return int(row[0])


def pending_migrations(conn: sqlite3.Connection) -> List[Tuple[int, str, str]]:
    """The migration files newer than the database's version, in order."""
    have = db_version(conn)
    return [(v, slug, path) for v, slug, path in migration_files() if v > have]


def _now() -> str:
    from ..records import utc_now
    return utc_now()


def _rootpages(conn: sqlite3.Connection) -> Dict[str, int]:
    return {str(r[0]): int(r[1]) for r in conn.execute(
        "SELECT name, rootpage FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'").fetchall()}


def _tables(conn: sqlite3.Connection) -> List[str]:
    return [str(r[0]) for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name").fetchall()]


def referencing_tables(conn: sqlite3.Connection, parents: Set[str]) -> Set[str]:
    """Every table with a ``REFERENCES`` (``PRAGMA foreign_key_list``) naming a table in ``parents``."""
    out: Set[str] = set()
    for t in _tables(conn):
        for fk in conn.execute(f"PRAGMA foreign_key_list({t})").fetchall():
            if str(fk[2]) in parents:
                out.add(t)
                break
    return out


def foreign_key_violations(conn: sqlite3.Connection, tables: Sequence[str]) -> List[Tuple[str, Optional[int], str, int]]:
    """``PRAGMA foreign_key_check(<t>)`` over ``tables``: ``[(table, rowid, parent, fk_index)]``, empty when clean."""
    out = []
    for t in sorted(set(tables)):
        for row in conn.execute(f"PRAGMA foreign_key_check({t})").fetchall():
            out.append((str(row[0]), row[1], str(row[2]), int(row[3])))
    return out


#: The temporary table the runner fills from the site's alias table before any pending file (module docstring).
SITE_ALIAS_TABLE = "site_alias"


def install_alias_table(conn: sqlite3.Connection, site_aliases: Optional[Mapping[str, str]]) -> Dict[str, str]:
    """(Re)create ``temp.site_alias(old PRIMARY KEY, new)`` on ``conn`` and fill it from the validated ``site_aliases``
    (``site_aliases.validate``; empty when None). Returns the validated table. A temporary table lives in the connection's own
    ``temp`` schema: it is never part of the database file, ``sqlite_master`` or the schema the drift check compares."""
    from .. import site_aliases as SA
    table = SA.validate(site_aliases or {}, "the migration alias table")
    conn.execute(f"DROP TABLE IF EXISTS temp.{SITE_ALIAS_TABLE}")
    conn.execute(f"CREATE TEMP TABLE {SITE_ALIAS_TABLE} (old TEXT PRIMARY KEY, new TEXT NOT NULL)")
    conn.executemany(f"INSERT INTO temp.{SITE_ALIAS_TABLE}(old, new) VALUES (?, ?)", SA.pairs(table))
    return table


def drop_alias_table(conn: sqlite3.Connection) -> None:
    conn.execute(f"DROP TABLE IF EXISTS temp.{SITE_ALIAS_TABLE}")


def apply_migrations(conn: sqlite3.Connection, notes: Optional[List[str]] = None,
                     site_aliases: Optional[Mapping[str, str]] = None) -> List[int]:
    """Apply every migration newer than the database's version; returns the versions applied. Each file runs in one
    transaction with foreign keys off and legacy ALTER semantics (module docstring); the version row commits with the
    DDL or not at all; a foreign-key violation the file leaves behind rolls it back (:class:`MigrationFailed`). ``notes``,
    when given, collects the line each Python post-step (:data:`POST_APPLY`) returns, for the caller's log.

    ``site_aliases`` is the site's alias table (``site_aliases.load_root`` / ``load_table``), empty by default; it is
    validated, loaded into ``temp.site_alias`` before the first pending file and dropped after the last. A file's
    :data:`PRE_CHECK` runs before the file and refuses by name, applying nothing of that file."""
    applied: List[int] = []
    have = db_version(conn)
    pending = [(v, slug, path) for v, slug, path in migration_files() if v > have]
    if not pending:
        return applied
    table = install_alias_table(conn, site_aliases)
    try:
        check_labels_before_rename(conn, pending, table)
        for version, slug, path in pending:
            with open(path, "r", encoding="utf-8") as fh:
                sql = fh.read()
            check = PRE_CHECK.get(slug)
            if check is not None:
                check(conn, table)
            note = _apply_one(conn, version, slug, sql)
            if note and notes is not None:
                notes.append(f"{version:04d}_{slug}: {note}")
            applied.append(version)
    finally:
        drop_alias_table(conn)
    return applied


def lift_object_lineage(conn: sqlite3.Connection) -> str:
    """The one-shot lift of the existing object rows' ``extra_json.parent_object_id`` / ``extra_json.named_role`` into the two columns the
    record-refusal-fact migration adds -- as a REFOLD of every object key's history under the new fold (``reducer.fold``, whose
    ``lift_object_fields`` reads the keys from each event's own parked extra), exactly what a rebuild produces: every event's
    ``folded_state_json`` rewritten where it changed, the live row rewritten from the last state (its ``record_uri`` and ``ingest_seq``
    kept), so ``registry rebuild --verify`` holds after the window as it did before. Idempotent: a second run refolds to the same bytes
    and counts nothing. Runs under the runner's transaction with foreign keys off; the parents are checked afterwards
    (``PRAGMA foreign_key_check(object)``) and a violation fails the migration. Live, read-only, 2026-10-03: 412 objects, 27 carrying
    both keys under extra (4 distinct parents, all present), 3,467 object events. Returns the log line."""
    from .. import reducer as RED
    keys = [r[0] for r in conn.execute("SELECT DISTINCT natural_key_json FROM row_history WHERE table_name = 'object' ORDER BY natural_key_json").fetchall()]
    red = RED.Reducer(conn)
    n_parent = n_role = n_rows = n_events = 0
    for key_json in keys:
        events = red._load_events("object", key_json)
        if not events:
            continue
        folded = RED.fold("object", events)
        for ev, (state, _) in zip(events, folded):
            new_json = RED.canonical_json(state)
            cur = conn.execute("UPDATE row_history SET folded_state_json = ? WHERE table_name = 'object' AND natural_key_json = ? "
                               "AND written_utc = ? AND record_sha256 = ? AND folded_state_json <> ?",
                               (new_json, key_json, ev.written_utc, ev.sha, new_json))
            n_events += cur.rowcount
        key = json.loads(key_json)
        live = conn.execute("SELECT parent_object_id, named_role, record_uri, ingest_seq FROM object WHERE object_id = ?", (key["object_id"],)).fetchone()
        if live is None:
            continue
        state, fv = folded[-1]
        if (live[0], live[1]) == (state.get("parent_object_id"), state.get("named_role")):
            continue
        n_parent += int(live[0] is None and state.get("parent_object_id") is not None)
        n_role += int(live[1] is None and state.get("named_role") is not None)
        n_rows += 1
        red._write_live_row("object", key, state, fv, str(live[2]), int(live[3]))
    bad = foreign_key_violations(conn, ["object"])
    if bad:
        raise MigrationFailed(f"the object lineage lift leaves {len(bad)} foreign-key violation(s) on object (a parent_object_id naming no object row): "
                              + "; ".join(f"rowid {rid} -> {parent}" for _, rid, parent, _ in bad[:5]))
    return (f"lifted object lineage from extra_json: parent_object_id {n_parent}, named_role {n_role} ({n_rows} object row(s) rewritten, "
            f"{n_events} history event(s) refolded, {len(keys)} object key(s) checked)")


def derive_fold_observations(conn: sqlite3.Connection) -> str:
    """The one-shot derivation of ``fold_observation`` for the run keys already folded when the fold-observation migration lands
    (2026-10-04): every live run whose ``extra_json`` carries ``serving_jobs`` (exactly the keys with an event stating it -- the fold never
    removes an ``extra`` key of a run) has its ``row_history`` events read back, folded and observed through the SAME ``observations.observe_key``
    the live reducer calls after every refold, so the rows are the ones a rebuild writes. Idempotent (the rows are replaced). Live, read-only,
    2026-10-04: 162 of 12,699 run rows carry the key, every value a list. Returns the log line (``observations.backfill``)."""
    from .. import observations as OBS
    return OBS.backfill(conn)


#: Python post-steps by migration SLUG (module docstring): run inside the file's transaction after its statements, before the version row
#: commits; each returns the line `registry migrate` prints. Keyed by slug, not number, so renumbering a file never
#: detaches its step.
POST_APPLY: Dict[str, Callable[[sqlite3.Connection], str]] = {
    "record_refusal_fact": lift_object_lineage,
    "fold_observation": derive_fold_observations,
}


def check_site_labels(conn: sqlite3.Connection, site_aliases: Mapping[str, str]) -> None:
    """The site-label rename's pre-check: every stored value of every label column (``site_aliases.substrate_columns``) must
    be a label that column admits, or a spelling ``site_aliases`` maps to one. Refuses by name (:class:`MigrationFailed`)
    naming each value, its table and column, its row count and the entry to add to the root's ``site_aliases.toml``; reads
    only."""
    from .. import site_aliases as SA
    bad = [r for r in SA.census(conn, site_aliases) if r["status"] in ("unaliased", "alias_not_admitted")]
    if not bad:
        return
    parts = []
    for r in bad:
        where = f"{r['table']}.{r['column']} holds {r['value']!r} on {r['rows']} row(s)"
        if r["status"] == "unaliased":
            parts.append(f"{where}: not a label ({r['vocab']}) and not aliased")
        else:
            parts.append(f"{where}: aliased to {r['label']!r}, which {r['vocab']} does not admit")
    raise MigrationFailed(f"site label rename refused before anything was applied: {'; '.join(parts)}. Name each spelling's label in "
                          f"the registry root's {SA.FILE_NAME} under [{SA.TABLE_NAME}] (<spelling> = \"<label>\"), or pass that file "
                          "to `registry migrate --site-aliases FILE`")


#: The slug of the migration that maps site labels through the alias table (0032).
SITE_LABEL_RENAME = "site_label_rename"

#: Checks that run BEFORE a file's statements, keyed by migration SLUG like :data:`POST_APPLY`: each takes the connection and
#: the validated alias table and raises :class:`MigrationFailed` by name, before the file's transaction opens.
PRE_CHECK: Dict[str, Callable[[sqlite3.Connection, Mapping[str, str]], None]] = {
    SITE_LABEL_RENAME: check_site_labels,
}


def rebuilt_tables(sql: str) -> List[str]:
    """The tables a migration body rebuilds (``CREATE TABLE <t>__new``), in file order; comments and string literals are skipped."""
    from .. import db as DB
    text = DB._outside_strings(sql, r"--[^\n]*|/\*.*?\*/", " ")
    return [m.group(1) for m in re.finditer(rf"\bCREATE\s+TABLE\s+(\w+?){REBUILD_SUFFIX}\b", text, re.IGNORECASE)]


def check_labels_before_rename(conn: sqlite3.Connection, pending: Sequence[Tuple[int, str, str]], site_aliases: Mapping[str, str]) -> None:
    """Refuse by name, before anything is applied, a database older than the label migration whose label columns hold a value
    outside the labels in a table a pending earlier file rebuilds. Those files create the table with CHECK lists that admit only the
    labels and copy every value as stored; only the label migration maps a site's own spelling through the alias table. So such a
    database is first brought to the version before the label migration by the code that wrote it, whose migrations admit its
    spellings, and then migrated here. Reads only; a database at that version or later, or one that holds only labels, passes."""
    from .. import site_aliases as SA
    rename = next((v for v, slug, _ in migration_files() if slug == SITE_LABEL_RENAME), None)
    earlier = [(v, slug, path) for v, slug, path in pending if rename is not None and v < rename]
    if not earlier:
        return
    stray = [r for r in SA.census(conn, site_aliases) if r["status"] != "label"]     # empty on a fresh database: no file is read
    if not stray:
        return
    first: Dict[str, str] = {}                     # table -> the first pending earlier file that rebuilds it
    for version, slug, path in earlier:
        with open(path, "r", encoding="utf-8") as fh:
            for t in rebuilt_tables(fh.read()):
                first.setdefault(t, f"{version:04d}_{slug}")
    bad = [r for r in stray if r["table"] in first]
    if not bad:
        return
    parts = []
    for r in bad:
        alias = f"aliased to {r['label']!r}" if r["status"] != "unaliased" else "not aliased"
        parts.append(f"{first[r['table']]} rebuilds {r['table']} and copies {r['table']}.{r['column']}, which holds {r['value']!r} on "
                     f"{r['rows']} row(s) ({alias})")
    name = f"{rename:04d}_{SITE_LABEL_RENAME}"
    raise MigrationFailed(f"a database at version {db_version(conn)} is refused before anything was applied: {'; '.join(parts)}. The "
                          f"migrations before {name} copy a label column as stored into CHECK lists that admit only the labels; {name} "
                          f"is the one that maps a site's own spellings. Bring the database to version {rename - 1} with the code that "
                          f"wrote it (its migrations admit those spellings), then run `registry migrate` here with the site's alias table")


#: Statements a migration file may not carry: the runner owns the transaction and the pragmas, and a COMMIT inside a
#: file would make whatever precedes it durable before the runner's checks (or a later statement's failure) could roll
#: it back.
_RESERVED_STATEMENTS = ("BEGIN", "COMMIT", "END", "ROLLBACK", "SAVEPOINT", "RELEASE", "PRAGMA", "VACUUM", "ATTACH", "DETACH")


def reserved_statement(sql: str) -> Optional[str]:
    """The first statement of ``sql`` (comments and string literals aside) that starts with a reserved word, or None.
    A trigger's body (``CREATE TRIGGER … BEGIN <statements>; END;``) is skipped: its closing ``END`` is not a
    transaction statement, and a ``CASE … END`` inside one of its statements never stands alone."""
    from .. import db as DB
    text = DB._outside_strings(sql, r"--[^\n]*|/\*.*?\*/", " ")
    in_trigger = False
    for stmt in DB._split_top_level(text, ";"):
        words = stmt.split()
        if not words:
            continue
        word = words[0].upper()
        if in_trigger:
            if word == "END" and len(words) == 1:
                in_trigger = False
            continue
        if word == "CREATE" and "TRIGGER" in (w.upper() for w in words[1:3]):
            in_trigger = True
            continue
        if word in _RESERVED_STATEMENTS:
            return stmt.strip()[:80]
    return None


def _apply_one(conn: sqlite3.Connection, version: int, slug: str, sql: str) -> Optional[str]:
    name = f"{version:04d}_{slug}"
    note: Optional[str] = None
    bad_stmt = reserved_statement(sql)
    if bad_stmt is not None:
        raise MigrationFailed(f"migration {name} carries a statement the runner owns ({bad_stmt!r}): a migration file is a bare "
                              "statement list — no BEGIN/COMMIT, no PRAGMA (the runner wraps it in one transaction with foreign "
                              "keys off and checks them after)")
    made_alias = False
    if f"temp.{SITE_ALIAS_TABLE}" in sql and conn.execute(
            "SELECT 1 FROM sqlite_temp_master WHERE type = 'table' AND name = ?", (SITE_ALIAS_TABLE,)).fetchone() is None:
        # a caller applying one file directly (a test building a shape file by file) maps through the EMPTY alias table: the
        # copy is the identity, and a legacy value then meets the file's CHECK and fails by name. apply_migrations fills the
        # table from the site's alias file before the first pending file, and runs the file's pre-check.
        install_alias_table(conn, None)
        made_alias = True
    before = _rootpages(conn)
    # both pragmas are no-ops inside a transaction, so they are set around it — and RESTORED to what the caller had, never
    # forced: a test connection opened with foreign_keys=OFF to insert rows without their parents must still be off after
    # (the gate's four `FOREIGN KEY constraint failed` in test_registry_adapter_declared, 2026-09-19, were this runner
    # switching the caller's connection to ON)
    fk_was = int(conn.execute("PRAGMA foreign_keys").fetchone()[0])
    legacy_was = int(conn.execute("PRAGMA legacy_alter_table").fetchone()[0])
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("PRAGMA legacy_alter_table=ON")
    try:
        # executescript COMMITs any open transaction before it runs, so the transaction begins INSIDE the script and
        # is left open for the version row and the checks below: DDL + version row commit together or not at all
        # (SQLite DDL is transactional)
        try:
            conn.executescript("BEGIN;\n" + sql)
        except sqlite3.Error as exc:
            raise MigrationFailed(f"migration {name}: {exc}") from exc
        if not conn.in_transaction:
            raise MigrationFailed(f"migration {name} manages its own transaction (BEGIN/COMMIT): a migration file is a bare "
                                  "statement list, the runner owns the transaction")
        conn.execute("INSERT INTO schema_migration(version, applied_utc, description) VALUES (?, ?, ?)", (int(version), _now(), slug))
        post = POST_APPLY.get(slug)
        if post is not None:
            note = post(conn)              # the file's Python post-step (a data lift), inside the same transaction
        recreated = {n for n, rp in _rootpages(conn).items() if before.get(n) != rp}
        bad = foreign_key_violations(conn, sorted(recreated | referencing_tables(conn, recreated)))
        if bad:
            shown = "; ".join(f"{t}(rowid {rid}) → {parent}" for t, rid, parent, _ in bad[:5]) + ("; …" if len(bad) > 5 else "")
            raise MigrationFailed(f"migration {name} leaves {len(bad)} foreign-key violation(s) ({shown}): rolled back")
        conn.execute("COMMIT")
        return note
    except BaseException:
        if conn.in_transaction:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
        raise
    finally:
        conn.execute(f"PRAGMA legacy_alter_table={legacy_was}")
        conn.execute(f"PRAGMA foreign_keys={fk_was}")
        if made_alias:
            drop_alias_table(conn)


# --------------------------------------------------------------------------------------------
# deriving the migration a drifted database needs (registry migrate --plan [--sql])
# --------------------------------------------------------------------------------------------


#: The SELECT expression a rebuild uses for a label column when the render maps site labels (``map_site_labels``): the
#: alias table's label for a legacy spelling, else the stored value.
SITE_LABEL_SELECT = "COALESCE((SELECT new FROM temp.{alias_table} WHERE old = {table}.{column}), {table}.{column})"


def site_label_select(table_name: str, column: str) -> str:
    return SITE_LABEL_SELECT.format(alias_table=SITE_ALIAS_TABLE, table=table_name, column=column)


def render_table_rebuild(table_name: str, common_columns: Sequence[str], map_site_labels: bool = False) -> str:
    """The documented rebuild of ``table_name`` to the code's DDL: ``CREATE TABLE <t>__new`` (``schema_def``'s
    columns and constraints), ``INSERT … SELECT`` of ``common_columns`` in primary-key order, ``DROP`` the old table,
    ``RENAME`` the new one, recreate the code's indexes and triggers ON the table, ``ANALYZE`` it. Runs under the
    runner's ``foreign_keys=OFF`` / ``legacy_alter_table=ON`` (module docstring). A code column not in
    ``common_columns`` takes its DEFAULT — the migration author supplies a value when it has none. With
    ``map_site_labels`` every label column (``site_aliases.label_columns``) is copied through :data:`SITE_LABEL_SELECT`, and a label
    in the table's natural key re-keys its ``row_history`` the same way (:func:`render_history_rekey`)."""
    from .. import schema_def as S
    t = S.TABLE_BY_NAME[table_name]
    new = table_name + REBUILD_SUFFIX
    cols = [c for c in t.column_names if c in set(common_columns)]
    missing = [c.name for c in t.columns if c.name not in set(common_columns)]
    dropped = [c for c in common_columns if c not in t.column_names]
    lines = [f"-- rebuild {table_name}: SQLite cannot alter a constraint in place (migrations/README.md)"]
    for c in missing:
        col = t.column(c)
        note = "takes its DEFAULT" if col.default is not None else ("nullable: NULL" if not col.nn else
                                                                       "NOT NULL without a DEFAULT: the INSERT below needs a value for it")
        lines.append(f"-- column {c} is new in the code: {note}")
    for c in dropped:
        lines.append(f"-- column {c} is not declared by the code: its values are NOT copied")
    lines.append(S.render_create_table(t, new))
    col_list = ", ".join(cols)
    sel_list = col_list
    if map_site_labels:
        from .. import site_aliases as SA
        mapped = set(SA.label_columns(table_name))
        sel_list = ", ".join(site_label_select(table_name, c) if c in mapped else c for c in cols)
    lines.append(f"INSERT INTO {new} ({col_list}) SELECT {sel_list} FROM {table_name} ORDER BY {', '.join(t.pk)};")
    lines.append(f"DROP TABLE {table_name};")
    lines.append(f"ALTER TABLE {new} RENAME TO {table_name};")
    lines += S.render_indexes(t)
    lines += S.triggers_on(table_name)
    lines.append(f"ANALYZE {table_name};")
    if map_site_labels:
        lines += render_history_rekey(table_name)
    return "\n".join(lines)


#: The UPDATE that files a table's ``row_history`` events under the label when a label column is part of its natural key: the key's
#: JSON names the label wherever the alias table maps the stored spelling, as the fold keys a new event (``reducer._key_of``).
HISTORY_REKEY_SQL = ("UPDATE row_history SET natural_key_json = json_set(natural_key_json, '$.{column}', (SELECT new FROM temp.{alias_table} "
                     "WHERE old = json_extract(row_history.natural_key_json, '$.{column}'))) WHERE table_name = '{table}' AND "
                     "json_extract(natural_key_json, '$.{column}') IN (SELECT old FROM temp.{alias_table});")


def render_history_rekey(table_name: str) -> List[str]:
    """For every label column of ``table_name`` that is part of its natural key (``records.NATURAL_KEYS``; only
    ``mirror_ack.destination`` today), the statement that re-keys the table's ``row_history`` through the alias table. Each record's
    payload stays as written; only the key a history is filed under moves, so a mirror's events written under either spelling form
    one history and a later event of either spelling joins it."""
    from .. import records as R
    from .. import site_aliases as SA
    keyed = [c for c in SA.label_columns(table_name) if c in R.NATURAL_KEYS.get(table_name, ())]
    if not keyed:
        return []
    out = [f"-- {table_name}'s history is filed under its natural key, which names a site label: re-keyed through the alias table"]
    out += [HISTORY_REKEY_SQL.format(column=c, alias_table=SITE_ALIAS_TABLE, table=table_name) for c in keyed]
    return out


def add_column_sql(table_name: str, column_name: str) -> str:
    """``ALTER TABLE <t> ADD COLUMN <definition>`` for a code column, the definition rendered as the CREATE text renders it
    (``schema_def.render_column``) without its trailing comment -- SQLite stores the definition's tokens verbatim, so the
    altered table's CREATE text normalizes equal to the code's when the column is declared last."""
    from .. import schema_def as S
    definition = S.render_column(S.TABLE_BY_NAME[table_name].column(column_name)).split("  -- ", 1)[0].rstrip()
    return f"ALTER TABLE {table_name} ADD COLUMN {definition};"


def addable_columns(conn: sqlite3.Connection, table_name: str) -> Optional[List[str]]:
    """The columns ``ALTER TABLE ADD COLUMN`` takes ``table_name`` from ``conn``'s shape to the code's with, or None when
    only a rebuild does: the live columns must be a prefix-closed subset of the code's (every missing column after the last
    live one in the code's order -- SQLite appends an added column after the last existing one, before the table
    constraints), each must be addable (nullable, or NOT NULL with a DEFAULT), and the live CREATE text with them appended --
    tried for real in a scratch database -- must normalize equal to the code's. Verified by execution, never by inference:
    a difference anywhere else in the text (a CHECK list, a NOT NULL, a constraint) makes it a rebuild."""
    from .. import db as DB
    from .. import schema_def as S
    t = S.TABLE_BY_NAME[table_name]
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)).fetchone()
    if row is None:
        return None
    live_cols = [str(r[1]) for r in conn.execute(f"PRAGMA table_info({table_name})").fetchall()]
    code_cols = t.column_names
    missing = [c for c in code_cols if c not in live_cols]
    if not missing or any(c not in code_cols for c in live_cols):
        return None
    last_live = max(code_cols.index(c) for c in live_cols)
    if any(code_cols.index(c) < last_live for c in missing):
        return None
    scratch = sqlite3.connect(":memory:")
    try:
        scratch.execute(str(row[0]))
        for c in missing:
            scratch.execute(add_column_sql(table_name, c).rstrip(";"))
        got = scratch.execute("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table_name,)).fetchone()
    except sqlite3.Error:
        return None
    finally:
        scratch.close()
    if got is None or DB.normalize_ddl(str(got[0])) != DB.code_ddl_master()[("table", table_name)]:
        return None
    return missing


def shape_plan(conn: sqlite3.Connection) -> Dict[str, object]:
    """What a migration must do to bring ``conn``'s shape to the code's (read-only): the drift lines, the tables to take
    forward by ``ALTER TABLE ADD COLUMN`` (``add_columns``: they lack only columns the code declares last, no row copied),
    the tables to rebuild (CREATE text differs otherwise) with their row counts and common columns, the tables to create,
    the tables the code does not declare (a human decision, never generated), and the indexes, triggers and ``v_*``
    objects to (re)create or drop — those on a rebuilt table excluded, the rebuild recreates them."""
    from .. import db as DB
    from .. import schema_def as S
    want, have = DB.code_ddl_master(), DB.ddl_master(conn)
    have_tables = {n for (t, n) in have if t == "table"}
    rebuild: List[Dict[str, object]] = []
    add_columns: List[Dict[str, object]] = []
    create: List[str] = []
    for t in S.TABLES:
        if t.name not in have_tables:
            create.append(t.name)
        elif have[("table", t.name)] != want[("table", t.name)]:
            live_cols = [str(r[1]) for r in conn.execute(f"PRAGMA table_info({t.name})").fetchall()]
            n = int(conn.execute(f"SELECT COUNT(*) FROM {t.name}").fetchone()[0])
            addable = addable_columns(conn, t.name)
            if addable:
                add_columns.append({"table": t.name, "rows": n, "columns": addable, "scale": t.scale})
                continue
            common = [c for c in live_cols if c in set(t.column_names)]
            rebuild.append({"table": t.name, "rows": n, "common_columns": common, "scale": t.scale,
                            "dropped_columns": [c for c in live_cols if c not in set(t.column_names)]})
    undeclared_tables = sorted(n for (t, n) in have if t == "table" and n not in S.TABLE_BY_NAME)
    touched = set(create) | {str(r["table"]) for r in rebuild}
    owner: Dict[Tuple[str, str], str] = {}
    for t in S.TABLES:
        for name, _, _ in t.indexes:
            owner[("index", name)] = t.name
    for sql in S.render_triggers():
        m = re.search(r"^CREATE TRIGGER (\w+) .*? ON (\w+)$", sql.split("\n", 1)[0])
        if m:
            owner[("trigger", m.group(1))] = m.group(2)
    objects: Dict[str, Dict[str, List[str]]] = {}
    for typ in ("index", "trigger", "view"):
        recreate = sorted(n for (t2, n), sql in want.items() if t2 == typ and have.get((typ, n)) != sql and owner.get((typ, n)) not in touched)
        drop = sorted(n for (t2, n) in have if t2 == typ and (typ, n) not in want)
        objects[typ] = {"recreate": recreate, "drop": drop}
    return {"drift": DB.schema_drift(conn), "rebuild": rebuild, "add_columns": add_columns, "create": create,
            "undeclared_tables": undeclared_tables, "objects": objects}


def render_shape_migration(conn: sqlite3.Connection, plan: Optional[Dict[str, object]] = None, title: str = "",
                           map_site_labels: bool = False) -> str:
    """The SQL body of the migration ``shape_plan`` describes, ready to save as the next ``NNNN_<slug>.sql`` (the file
    is the history; the body is derived from the code's DDL at render time, so it is saved and reviewed, never applied
    from memory). Empty string when the shape already matches. ``map_site_labels`` copies every label column of a
    rebuilt table through the runner's alias table (:func:`render_table_rebuild`)."""
    from .. import schema_def as S
    plan = plan or shape_plan(conn)
    parts: List[str] = []
    if title:
        parts.append(f"-- {title}")
    for line in plan["drift"]:
        parts.append(f"-- drift: {line}")
    if plan["drift"]:
        parts.append("")
    for name in plan["undeclared_tables"]:
        parts.append(f"-- table {name} is not declared by the code: dropping it is a human decision (not generated)")
    for name in plan["create"]:
        t = S.TABLE_BY_NAME[name]
        parts.append(f"-- create {name}")
        parts.append(S.render_create_table(t))
        parts += S.render_indexes(t)
        parts += S.triggers_on(name)
        parts.append("")
    for r in plan["rebuild"]:
        parts.append(render_table_rebuild(str(r["table"]), list(r["common_columns"]), map_site_labels=map_site_labels))
        parts.append("")
    for a in plan.get("add_columns", ()):
        t = S.TABLE_BY_NAME[str(a["table"])]
        parts.append(f"-- add column(s) to {t.name}: ALTER TABLE ADD COLUMN, no row copied (the code declares them after every live "
                     "column, so SQLite's appended definition equals the code's CREATE text; migrations/README.md)")
        for c in a["columns"]:
            col = t.column(c)
            parts.append(f"-- column {c}: " + ("derived, " if col.derived else "") + (f"why-null: {col.why_null}" if col.why_null else col.doc))
            parts.append(add_column_sql(t.name, c))
        parts.append("")
    objs = plan["objects"]
    for name in objs["index"]["drop"]:
        parts.append(f"DROP INDEX IF EXISTS {name};")
    for name in objs["trigger"]["drop"]:
        parts.append(f"DROP TRIGGER IF EXISTS {name};")
    for name in objs["view"]["drop"]:
        parts.append(f"DROP VIEW IF EXISTS {name};")
    index_sql = {n: sql for t in S.TABLES for n, sql in zip([i[0] for i in t.indexes], S.render_indexes(t))}
    for name in objs["index"]["recreate"]:
        parts.append(f"DROP INDEX IF EXISTS {name};")
        parts.append(index_sql[name])
    trigger_sql = {re.match(r"^CREATE TRIGGER (\w+)", sql).group(1): sql for sql in S.render_triggers()}
    for name in objs["trigger"]["recreate"]:
        parts.append(f"DROP TRIGGER IF EXISTS {name};")
        parts.append(trigger_sql[name])
    for name in objs["view"]["recreate"]:
        parts.append(f"DROP VIEW IF EXISTS {name};")
        parts.append(S.VIEW_SQL[name].strip())
    body = "\n".join(parts).strip()
    return body + "\n" if body else ""
