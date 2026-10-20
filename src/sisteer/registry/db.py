"""SQLite connection discipline for the results registry.

* ``connect(path, readonly=False)``: WAL journal, ``synchronous=NORMAL``, ``foreign_keys=ON``,
  ``busy_timeout`` — readers are never blocked, the write lock is held only around
  ``executemany`` + commit.
* ``executemany_batched(conn, sql, rows, batch=5000)``: no row-at-a-time inserts anywhere.
* ``upsert_sql(table, cols, pk, guard=, keep=, update=)``: the one ``INSERT … ON CONFLICT DO UPDATE`` shape.
* ``write_lock(root, shared=)``: an ``flock`` on ``<root>/<name>`` — the harvester's ``harvester.lock`` (any
  process may call ``harvest.ingest_once`` but only under it; never shared with the paper build's lock) and,
  shared for readers / exclusive for reclamation, the snapshot readers' coordination lock.
* ``reading(path)``: a read-only connection for the length of a ``with`` block.
* ``open_fresh(path)``: a new database with every migration applied (never ``schema.sql`` directly).
* ``open_or_create(path)``: the writers' open. Refuses — :class:`SchemaDrift`, naming the object — a database whose
  migration version is behind the code's OR whose SHAPE differs from the code's DDL at the same version:
  ``schema_drift`` compares every table's CREATE text (columns, types, NOT NULL, DEFAULT, CHECK lists, REFERENCES,
  table constraints, WITHOUT ROWID), every index, trigger and ``v_*`` object against ``schema_def``'s render,
  normalised (comments, whitespace and identifier quoting aside). The version guard alone could not see an in-place
  edit of a migration: the live database (created 2026-09-18) kept the old ``object.paper_label`` CHECK after
  ``vocab.PAPER_LABELS`` grew, the harvester folded until the CHECK refused a record mid-revision, and whole source
  revisions were held on ``IntegrityError: CHECK constraint failed`` (2026-09-19). The remedy for drift is a
  migration (``registry migrate --plan [--sql]`` renders one), never an edit of a landed migration file.
* ``require_fold_order(conn, path)``: the writers' FOLD-ORDER check, run by ``open_or_create`` after the shape check and by every
  fold writer that takes a connection it did not open. The reducer's same-stamp order has a version
  (``reducer.FOLD_ORDER_VERSION``), recorded in the same ledger as the DDL version -- the ``schema_migration`` row
  ``version = 0`` (:data:`FOLD_ORDER_ROW_VERSION`), outside the migration sequence and invisible to ``MAX(version)`` -- by the
  first writer (``stamp_fold_order``; ``open_fresh`` stamps a database as it creates it). A database recording the code's order
  proceeds; one recording none is stamped and proceeds (the bootstrap: every database from before the stamp); one recording
  ANOTHER order refuses by name (:class:`FoldOrderDrift`, a :class:`SchemaDrift`), both orders in the message. Readers never
  check (``require_shape`` over a read-only connection checks without stamping). A metadata row is not a shape:
  ``schema_drift`` and ``migrations.shape_plan`` read ``sqlite_master``, so the stamp never reads as drift or as a migration.
* ``FOLD_ORDER_FUNCTION`` (``registry_fold_order()``): the FENCE the database enforces itself. Every connection :func:`connect` opens
  registers this SQL function returning the code's fold order, and the two triggers ``trg_run_fold_order_fence_ins`` /
  ``trg_run_fold_order_fence_upd`` (``schema_def.fold_order_fence_triggers``, migration 0030) refuse any write of a ``run`` row whose
  connection's order is not the one the database records in row 0 -- and a connection that never registered the function cannot write a
  ``run`` row at all (SQLite resolves the function when it compiles the write: ``no such function: registry_fold_order``). So the
  stamp fences every writer, not only the ones that read it: a checkout from before the stamp -- commit 5ea3fa411's ``replica
  refold-apply``, which opened the LIVE file with ``connect`` and rewrote run rows under the pre-tie-break order with no version, shape
  or order check at all -- is refused at the row, with nothing written. The Python check (``require_fold_order``) still runs first on
  every writer of this code, so a mismatch is named by both orders before a row is touched; the trigger is the backstop for the
  connections that skip it. Only ``run`` is fenced: ``tie_rank`` tells ``run_state`` events apart and nothing else, so ``run`` is the
  one table whose rows the two orders fold differently. A database lacking the triggers reads as shape drift and every writer refuses it.
"""
from __future__ import annotations

import contextlib
import fcntl
import functools
import os
import re
import shutil
import sqlite3
import time
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

DEFAULT_ROOT = os.path.join(os.path.expanduser("~"), "registry")
DB_NAME = "registry.sqlite"
#: A rebuild writes into ``<into>.building`` and renames it into place once the replay is complete and fsynced
#: (``harvest.rebuild``); the footprint watcher samples the same suffix so a build in flight counts toward the budget.
REBUILD_TMP_SUFFIX = ".building"
BATCH_ROWS = 5000
BUSY_TIMEOUT_MS = 30000
WRITER_CACHE_KIB = 262144   # 256 MiB page cache on a writer connection (a maximum, not an allocation)
#: The reserved ``schema_migration.version`` of the FOLD-ORDER stamp (:func:`stamp_fold_order`): the row recording which same-stamp
#: order (``reducer.FOLD_ORDER_VERSION``) folded the database's run-scale rows. 0 is outside the migration sequence --
#: ``migrations.migration_files`` requires the files contiguous from 0001, so no migration can ever take it -- and invisible to every
#: reader of the schema version (``MAX(version)``; a database holding only this row reads 0, "no migration applied", which is what it
#: is). Its ``description`` IS the fold order's version string and its ``applied_utc`` the moment the stamp was made: a database this code
#: created, at its creation; a database from before the stamp, at the first writer's open after the code landed (the bootstrap), which is
#: also the moment before which its rows may carry an earlier order (``replica.residual_same_stamp_class`` lists the rows that do). No new
#: table: the stamp rides the ledger that already records the DDL version, and a shape comparison reads ``sqlite_master``, never rows.
FOLD_ORDER_ROW_VERSION = 0
#: The SQL function every connection :func:`connect` opens registers (``registry_fold_order()`` -> ``reducer.FOLD_ORDER_VERSION``, the
#: order THIS code folds under), read by the fold-order fence triggers on ``run`` (``schema_def.fold_order_fence_triggers``; migration
#: 0030): a write of a ``run`` row is refused when the connection's order is not the one row :data:`FOLD_ORDER_ROW_VERSION` records, and a
#: connection that never registered the function cannot write a ``run`` row at all (``no such function: registry_fold_order`` as SQLite
#: compiles the write). The fence is what reaches the writers the stamp cannot: a pre-stamp checkout opening the live file with a bare
#: ``connect`` (commit 5ea3fa411's ``replica refold-apply``), a ``sqlite3`` shell, any process whose code predates the rule.
FOLD_ORDER_FUNCTION = "registry_fold_order"


def db_path(root: Optional[str] = None) -> str:
    return os.path.join(root or DEFAULT_ROOT, DB_NAME)


def connect(path: str, readonly: bool = False, busy_timeout_ms: int = BUSY_TIMEOUT_MS) -> sqlite3.Connection:
    """Open the registry database with the registry's pragmas. ``readonly`` opens via URI ``mode=ro``."""
    if readonly:
        uri = "file:" + os.path.abspath(path) + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=busy_timeout_ms / 1000.0, isolation_level=None)
    else:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        conn = sqlite3.connect(path, timeout=busy_timeout_ms / 1000.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    if not readonly:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA wal_autocheckpoint=10000")
        conn.execute(f"PRAGMA cache_size=-{WRITER_CACHE_KIB}")   # the item b-trees stay resident while a batch lands (F2)
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA temp_store=MEMORY")
    register_fold_order(conn)
    return conn


def wal_checkpoint_truncate(conn: sqlite3.Connection, busy_timeout_ms: int = 1000) -> Tuple[int, int, int]:
    """``PRAGMA wal_checkpoint(TRUNCATE)`` with a SHORT busy wait, the connection's own ``busy_timeout`` restored after:
    the end-of-writer-pass attempt to fold the WAL back into the database file. Returns SQLite's
    ``(busy, log, checkpointed)`` triple — ``busy = 1`` when a reader's snapshot kept the WAL alive; the caller logs it
    and never fails on it. Call outside any transaction."""
    prev = int(conn.execute("PRAGMA busy_timeout").fetchone()[0])
    conn.execute(f"PRAGMA busy_timeout={int(busy_timeout_ms)}")
    try:
        row = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    finally:
        conn.execute(f"PRAGMA busy_timeout={prev}")
    return int(row[0]), int(row[1]), int(row[2])


def connect_memory() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    register_fold_order(conn)
    return conn


def register_fold_order(conn: sqlite3.Connection) -> None:
    """Register :data:`FOLD_ORDER_FUNCTION` on ``conn``: a zero-argument SQL function returning the order this code folds under
    (``reducer.FOLD_ORDER_VERSION``), the token the fold-order fence triggers on ``run`` compare with the stamp in row
    :data:`FOLD_ORDER_ROW_VERSION`. :func:`connect` and :func:`connect_memory` call it on every connection they open, read-only ones
    included (a reader never fires the trigger; registering costs nothing). A connection opened any other way carries no such function,
    and the fence refuses its writes of ``run`` rows as SQLite compiles them."""
    from .reducer import FOLD_ORDER_VERSION
    conn.create_function(FOLD_ORDER_FUNCTION, 0, lambda: FOLD_ORDER_VERSION, deterministic=True)


def open_fresh(path: str) -> sqlite3.Connection:
    """Create a brand-new database at ``path`` (refusing to overwrite) and apply every migration."""
    if os.path.exists(path):
        raise FileExistsError(f"registry.db.open_fresh: {path} exists; refusing to overwrite a database")
    conn = connect(path)
    from .migrations import apply_migrations
    apply_migrations(conn)
    stamp_fold_order(conn)           # a database this code creates is folded under this code's order from its first row
    return conn


class SchemaDrift(RuntimeError):
    """A writer refused the database: its migration version is behind the code's (``have`` < ``want``), or its shape
    differs from the code's DDL at the same version (``lines``: one per differing object — the table and column or
    constraint, the index, trigger or ``v_*`` object — from :func:`schema_drift`). The message names the remedy:
    ``registry migrate`` for a pending migration; ``registry migrate --plan --sql`` to render the missing one; a
    rebuild from the record log only when no migration can express the change. Never an edit of the database by hand."""

    def __init__(self, path: str, have: int, want: int, lines: Sequence[str]):
        self.path, self.have, self.want, self.lines = path, int(have), int(want), list(lines)
        super().__init__(drift_message(path, self.have, self.want, self.lines))


def drift_message(path: str, have: int, want: int, lines: Sequence[str], limit: int = 8) -> str:
    shown = "; ".join(lines[:limit]) + ("; …" if len(lines) > limit else "")
    if have != want:
        msg = (f"registry database at {path} is at schema version {have}, code expects {want}: run `registry migrate` "
               f"(`registry migrate --plan` lists what it applies)")
        if lines:
            msg += f"; its shape also differs from the code's schema ({shown})"
        return msg
    return (f"registry database at {path} is at schema version {have} like the code, but its shape differs from the code's "
            f"schema ({shown}): a landed migration was edited in place, or the code's DDL changed without a migration — the "
            "version guard cannot see either. Bring the file to the code's shape with a migration: `registry migrate --plan` "
            "names what differs, `registry migrate --plan --sql` renders the table rebuilds for a new "
            "src/sisteer/registry/migrations/NNNN_<slug>.sql (landed migration files are frozen), `registry migrate` applies "
            "it. When no migration can express the change, rebuild from the record log — `registry rebuild --from <root> "
            "--into <fresh.sqlite>` — and swap the file in with the harvester stopped; "
            "never edit the database by hand")


class FoldOrderDrift(SchemaDrift):
    """A writer refused the database: the fold order it RECORDS (``have_fold_order``, the ``schema_migration`` row
    :data:`FOLD_ORDER_ROW_VERSION`) is not the order this code folds under (``want_fold_order``, ``reducer.FOLD_ORDER_VERSION``). The
    order is a property of the rows, and two orders in one database cannot be told apart afterwards: the live database carried rows
    folded under the pre-tie-break order by a harvester two days behind main (2026-10-04 to 2026-10-06) and a judgment daemon on an older
    checkout until 2026-10-08, and nothing in it said so until a landing's replica check met same-stamp movers -- so the writer stops by
    name instead of folding a second order in. A :class:`SchemaDrift` by type, so every refusal path a writer already has (``registry
    ingest`` exit 3 and the status page's banner, the grade daemon's schema hold, the replica's exit 12) treats it as the drift it is;
    ``have`` and ``want`` as SchemaDrift readers know them are the (equal) schema version, ``lines`` the one line naming the two orders.
    Never raised for a database with NO stamp: the first writer stamps the code's order and proceeds (:func:`require_fold_order`)."""

    def __init__(self, path: str, have_fold_order: str, want_fold_order: str, schema_version: int):
        self.path, self.have_fold_order, self.want_fold_order = path, str(have_fold_order), str(want_fold_order)
        self.have = self.want = int(schema_version)
        self.lines = [f"fold order: the database records {self.have_fold_order!r}, the code folds under {self.want_fold_order!r}"]
        RuntimeError.__init__(self, fold_order_message(path, self.have_fold_order, self.want_fold_order))


def fold_order_message(path: str, have: str, want: str) -> str:
    return (f"registry database at {path} records fold order {have!r} but this code folds under {want!r}: its run-scale rows were folded "
            "under another same-stamp rule, and a fold by this code would leave rows of two orders no reader could tell apart; nothing is "
            "written. When the CODE is the older side (a checkout behind main -- the harvester's base, a daemon's worktree), run the code the "
            "database was folded by: fast-forward the checkout. When the DATABASE is the older side, refold it under this code's order: "
            "`registry rebuild --from <root> --into <fresh.sqlite>` and a swap with the timers stopped -- a rebuild stamps the fresh "
            "database as it creates it. The stamp is never edited by hand")


def open_or_create(path: str) -> sqlite3.Connection:
    """Open the database, creating and migrating it when absent; refuse (:class:`SchemaDrift`) a database whose schema
    version is behind the code (``registry migrate`` upgrades) or whose shape differs from the code's DDL at the same
    version (a migration is missing: ``registry migrate --plan --sql`` renders it); then the fold order (:func:`require_fold_order`:
    stamped when absent, refused by name -- :class:`FoldOrderDrift` -- when another order's)."""
    from .migrations import apply_migrations, code_version, db_version
    if not os.path.exists(path):
        return open_fresh(path)
    conn = connect(path)
    have, want = db_version(conn), code_version()
    if have == 0:
        apply_migrations(conn)
        have = want
    elif have != want:
        conn.close()
        raise SchemaDrift(path, have, want, [])
    drift = schema_drift(conn)
    if drift:
        conn.close()
        raise SchemaDrift(path, have, want, drift)
    try:
        require_fold_order(conn, path)
    except FoldOrderDrift:
        conn.close()
        raise
    return conn


def require_shape(conn: sqlite3.Connection, path: str) -> None:
    """Raise :class:`SchemaDrift` unless ``conn``'s version and shape are the code's (the check ``open_or_create`` runs,
    for a connection opened another way), then :class:`FoldOrderDrift` when the database records another fold order --
    WITHOUT stamping one that records none, so a read-only connection (the grade daemon's schema preflight) can run it."""
    from .migrations import code_version, db_version
    have, want = db_version(conn), code_version()
    drift = schema_drift(conn) if have == want else []
    if have != want or drift:
        raise SchemaDrift(path, have, want, drift)
    require_fold_order(conn, path, stamp=False)


def connection_path(conn: sqlite3.Connection) -> str:
    """The file behind ``conn``'s main database (``PRAGMA database_list``), ``:memory:`` for none -- the name a refusal carries when
    the caller holds only the connection."""
    for row in conn.execute("PRAGMA database_list").fetchall():
        if str(row[1]) == "main":
            return str(row[2]) or ":memory:"
    return ":memory:"


def fold_order_version(conn: sqlite3.Connection) -> Optional[str]:
    """The fold order the database RECORDS -- the ``description`` of the ``schema_migration`` row :data:`FOLD_ORDER_ROW_VERSION` -- or
    None for a database that records none (one from before the stamp, or one with no ``schema_migration`` table at all)."""
    has = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migration'").fetchone()
    if not has:
        return None
    row = conn.execute("SELECT description FROM schema_migration WHERE version = ?", (FOLD_ORDER_ROW_VERSION,)).fetchone()
    return str(row[0]) if row is not None else None


def stamp_fold_order(conn: sqlite3.Connection, now: Optional[str] = None) -> str:
    """Record the code's fold order (``reducer.FOLD_ORDER_VERSION``) as the ``schema_migration`` row :data:`FOLD_ORDER_ROW_VERSION`, once:
    ``INSERT OR IGNORE``, then the row read back -- what the database records after the call, which may be an earlier stamp (the
    caller compares; :func:`require_fold_order` refuses a different one). Runs inside the caller's transaction when one is open, else as
    its own statement; the writers all hold the registry's write lock, so two first writers never race, and the ``OR IGNORE`` makes a
    race harmless anyway. Returns the recorded version."""
    from .records import utc_now
    from .reducer import FOLD_ORDER_VERSION
    conn.execute("INSERT OR IGNORE INTO schema_migration(version, applied_utc, description) VALUES (?, ?, ?)",
                 (FOLD_ORDER_ROW_VERSION, now or utc_now(), FOLD_ORDER_VERSION))
    return str(conn.execute("SELECT description FROM schema_migration WHERE version = ?", (FOLD_ORDER_ROW_VERSION,)).fetchone()[0])


def require_fold_order(conn: sqlite3.Connection, path: Optional[str] = None, stamp: bool = True) -> Optional[str]:
    """The writers' fold-order check, before anything folds: the database records the code's order (``reducer.FOLD_ORDER_VERSION``) --
    proceed; it records NONE -- stamp the code's order (:func:`stamp_fold_order`) and proceed, the bootstrap every database from before
    the stamp takes exactly once (the live registry at the first harvester tick after the code lands; a refusal there would stop the
    registry); it records ANOTHER -- refuse by name, :class:`FoldOrderDrift`, both orders in the message, nothing written. ``stamp=False``
    is the reader's form (:func:`require_shape` over a read-only connection): a missing stamp is then no finding. Returns the recorded
    version (None when absent and not stamped); ``path`` names the database in the refusal (read off the connection when omitted).
    The database enforces the same rule on every write of a ``run`` row itself (the fence triggers of migration 0030 against
    :data:`FOLD_ORDER_FUNCTION`), so a writer that skips this check -- code from before it, a shell -- is refused at the row."""
    from .migrations import db_version
    from .reducer import FOLD_ORDER_VERSION
    have = fold_order_version(conn)
    if have is None and stamp:
        have = stamp_fold_order(conn)
    if have is not None and have != FOLD_ORDER_VERSION:
        raise FoldOrderDrift(path or connection_path(conn), have, FOLD_ORDER_VERSION, db_version(conn))
    return have


# -- DDL normalisation and comparison -----------------------------------------------------------------------------
#: A single-quoted SQL string literal ('' is the escaped quote); every normalisation step leaves literals untouched.
_DDL_STRING = r"'(?:[^']|'')*'"


def _outside_strings(sql: str, pattern: str, repl) -> str:
    """``re.sub(pattern, repl)`` over the parts of ``sql`` outside single-quoted string literals."""
    rx = re.compile(_DDL_STRING + "|" + pattern, re.DOTALL)

    def sub(m):
        if m.group(0).startswith("'"):
            return m.group(0)
        return repl(m) if callable(repl) else repl
    return rx.sub(sub, sql)


def normalize_ddl(sql: str) -> str:
    """The comparable form of a CREATE statement: comments removed, plain identifiers unquoted (SQLite rewrites a
    renamed table's stored text as ``CREATE TABLE "name"``), whitespace collapsed, no space around ``(``, ``)`` and
    ``,``, no trailing ``;`` — string literals byte-for-byte. Two statements with the same normal form create the
    same object."""
    s = _outside_strings(sql, r"--[^\n]*|/\*.*?\*/", " ")
    s = _outside_strings(s, r'"([A-Za-z_][A-Za-z0-9_]*)"|\[([A-Za-z_][A-Za-z0-9_]*)\]|`([A-Za-z_][A-Za-z0-9_]*)`',
                         lambda m: m.group(1) or m.group(2) or m.group(3))
    s = _outside_strings(s, r"\s+", " ")
    s = _outside_strings(s, r"\s*([(),])\s*", lambda m: m.group(1))
    return s.strip().rstrip(";").strip()


def ddl_master(conn: sqlite3.Connection) -> Dict[Tuple[str, str], str]:
    """``{(type, name): normalised CREATE text}`` for every user object in ``conn``'s ``sqlite_master`` (SQLite's own
    ``sqlite_*`` objects — autoindexes, ``sqlite_stat1`` — excluded)."""
    out: Dict[Tuple[str, str], str] = {}
    for typ, name, sql in conn.execute("SELECT type, name, sql FROM sqlite_master WHERE sql IS NOT NULL").fetchall():
        if str(name).startswith("sqlite_"):
            continue
        out[(str(typ), str(name))] = normalize_ddl(str(sql))
    return out


@functools.lru_cache(maxsize=1)
def code_ddl_master() -> Dict[Tuple[str, str], str]:
    """The code's shape: ``schema_def.render_schema_sql()`` executed into an empty database, read back through
    :func:`ddl_master` (~20 ms, once per process)."""
    from . import schema_def as S
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(S.render_schema_sql())
        return ddl_master(conn)
    finally:
        conn.close()


def schema_drift(conn: sqlite3.Connection) -> List[str]:
    """How this database's shape differs from the code's DDL — one line per differing object, empty when they agree.
    Tables: absent, lacking columns (the pre-2026-09-19 lines, kept verbatim), then the CREATE text itself — per
    column the type, NOT NULL, DEFAULT, CHECK (an ``IN (...)`` list by the values it lacks or adds) and REFERENCES,
    the table constraints, WITHOUT ROWID; then every index and trigger and every ``v_*`` object by name and normalised
    text; objects the code does not declare. The migration version alone cannot tell: a database created from an
    older text of a migration reports the same version (``table judge has no column named field_confidence_json``,
    36 harvester restarts, 2026-09-17; ``CHECK constraint failed: paper_label IN (...)`` holding whole source revisions
    after ``vocab.PAPER_LABELS`` grew without a migration, 2026-09-19)."""
    from . import schema_def as S
    want = code_ddl_master()
    have = ddl_master(conn)
    out: List[str] = []
    have_tables = {n for (t, n) in have if t == "table"}
    for t in S.TABLES:
        if t.name not in have_tables:
            out.append(f"table {t.name} is absent")
            continue
        cols = {r[1] for r in conn.execute(f"PRAGMA table_info({t.name})").fetchall()}
        absent = [c for c in t.column_names if c not in cols]
        if absent:
            out.append(f"table {t.name} lacks column(s) {', '.join(absent)}")
        h, w = have[("table", t.name)], want[("table", t.name)]
        if h != w:
            out += describe_table_ddl_drift(t.name, h, w, skip_columns=absent)
    for (typ, name) in sorted(have):
        if typ == "table" and name not in S.TABLE_BY_NAME:
            out.append(f"table {name} is not declared by the code")
    for name in S.VIEW_NAMES:
        if ("view", name) not in have:
            out.append(f"{name} is absent")
        elif have[("view", name)] != want[("view", name)]:
            out.append(f"{name} differs from the code's SQL")
    for (typ, name) in sorted(have):
        if typ == "view" and ("view", name) not in want:
            out.append(f"{name} is not declared by the code")
    for typ in ("index", "trigger"):
        for (t2, name), sql in sorted(want.items()):
            if t2 != typ:
                continue
            if (typ, name) not in have:
                out.append(f"{typ} {name} is absent")
            elif have[(typ, name)] != sql:
                out.append(f"{typ} {name} differs from the code's SQL")
        for (t2, name) in sorted(have):
            if t2 == typ and (typ, name) not in want:
                out.append(f"{typ} {name} is not declared by the code")
    return out


def _scan_top_level(s: str):
    """Yield ``(index, depth)`` for every character of ``s`` outside string literals (depth = open parens before it)."""
    depth = 0
    in_str = False
    i, n = 0, len(s)
    while i < n:
        ch = s[i]
        if in_str:
            if ch == "'":
                if i + 1 < n and s[i + 1] == "'":
                    i += 2
                    continue
                in_str = False
            i += 1
            continue
        if ch == "'":
            in_str = True
        elif ch == "(":
            yield i, depth
            depth += 1
            i += 1
            continue
        elif ch == ")":
            depth -= 1
        yield i, depth
        i += 1


def _split_top_level(s: str, sep: str = ",") -> List[str]:
    """Split ``s`` at depth-0 occurrences of ``sep`` outside string literals."""
    parts, last = [], 0
    for i, depth in _scan_top_level(s):
        if depth == 0 and s[i] == sep:
            parts.append(s[last:i].strip())
            last = i + 1
    parts.append(s[last:].strip())
    return [p for p in parts if p]


_COLUMN_CLAUSE = re.compile(r"(NOT NULL|DEFAULT|CHECK|REFERENCES|COLLATE|UNIQUE|PRIMARY KEY|GENERATED ALWAYS)(?![A-Za-z0-9_])")
_CONSTRAINT_HEAD = re.compile(r"^(CONSTRAINT [A-Za-z_][A-Za-z0-9_]*|PRIMARY KEY|UNIQUE|CHECK|FOREIGN KEY)(?![A-Za-z0-9_])")


def _table_body(norm_sql: str) -> Tuple[List[str], str]:
    """``(top-level items of the parenthesised body, the tail after it — 'WITHOUT ROWID' or '')`` of a normalised
    CREATE TABLE statement."""
    open_at = norm_sql.find("(")
    if open_at < 0:
        return [], norm_sql
    close_at = None
    for i, depth in _scan_top_level(norm_sql[open_at:]):
        if depth == 0 and norm_sql[open_at + i] == ")":
            close_at = open_at + i
            break
    if close_at is None:
        return [], norm_sql
    return _split_top_level(norm_sql[open_at + 1:close_at]), norm_sql[close_at + 1:].strip()


def _column_clauses(coldef: str) -> Dict[str, List[str]]:
    """``{'head': ['name TYPE'], 'NOT NULL': [...], 'DEFAULT': [...], 'CHECK': [..., ...], 'REFERENCES': [...]}`` — the
    column definition cut at its depth-0 clause keywords."""
    cuts = [0]
    for i, depth in _scan_top_level(coldef):
        if depth == 0 and (i == 0 or not (coldef[i - 1].isalnum() or coldef[i - 1] == "_")) and _COLUMN_CLAUSE.match(coldef, i):
            if i not in cuts:
                cuts.append(i)
    cuts.append(len(coldef))
    out: Dict[str, List[str]] = {"head": [coldef[cuts[0]:cuts[1]].strip()]}
    for a, b in zip(cuts[1:], cuts[2:]):
        seg = coldef[a:b].strip()
        kw = _COLUMN_CLAUSE.match(seg).group(1)
        out.setdefault(kw, []).append(seg)
    return out


def _in_list_values(check: str) -> Optional[Tuple[str, List[str]]]:
    """``('col', ['a', 'b'])`` for a normalised ``CHECK(col IN('a','b'))``, else None."""
    m = re.match(r"^CHECK\((?:\()?([A-Za-z_][A-Za-z0-9_]*) IN\((.*)\)\)?\)$", check)
    if not m:
        return None
    return m.group(1), _split_top_level(m.group(2))


def _excerpt(text: str, limit: int = 160) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def describe_table_ddl_drift(table: str, have_sql: str, want_sql: str, skip_columns: Sequence[str] = ()) -> List[str]:
    """Name what differs between two normalised CREATE TABLE texts of ``table`` (``have``: the database, ``want``: the
    code): per column the type, NOT NULL, DEFAULT, CHECK — an ``IN (...)`` list by its missing or undeclared values —
    and REFERENCES; the table constraints; WITHOUT ROWID. Columns in ``skip_columns`` (already reported as lacking)
    are not repeated. Falls back to ``CREATE text differs`` when the texts differ in no nameable part."""
    out: List[str] = []
    h_items, h_tail = _table_body(have_sql)
    w_items, w_tail = _table_body(want_sql)

    def classify(items):
        cols: Dict[str, str] = {}
        keyed: Dict[str, str] = {}
        by_text: Set[str] = set()
        for it in items:
            m = _CONSTRAINT_HEAD.match(it)
            if m:
                head = m.group(1)
                if head.startswith("CONSTRAINT ") or head == "PRIMARY KEY":
                    keyed[head] = it
                else:
                    by_text.add(it)
            else:
                cols[it.split(" ", 1)[0].split("(", 1)[0]] = it
        return cols, keyed, by_text

    h_cols, h_keyed, h_text = classify(h_items)
    w_cols, w_keyed, w_text = classify(w_items)
    skip = set(skip_columns)
    for name in h_cols:
        if name not in w_cols:
            out.append(f"table {table}: column {name} is not declared by the code")
    for name, wdef in w_cols.items():
        if name in skip or name not in h_cols:
            continue
        hdef = h_cols[name]
        if hdef == wdef:
            continue
        hc, wc = _column_clauses(hdef), _column_clauses(wdef)
        named = False
        if hc["head"] != wc["head"]:
            out.append(f"table {table}: column {name} type differs — have «{hc['head'][0]}» want «{wc['head'][0]}»")
            named = True
        for kw in ("NOT NULL", "DEFAULT", "REFERENCES", "COLLATE", "UNIQUE", "PRIMARY KEY", "GENERATED ALWAYS"):
            hv, wv = hc.get(kw, []), wc.get(kw, [])
            if hv != wv:
                out.append(f"table {table}: column {name} {kw} differs — have «{'; '.join(hv) or 'none'}» want «{'; '.join(wv) or 'none'}»")
                named = True
        h_checks, w_checks = set(hc.get("CHECK", [])), set(wc.get("CHECK", []))
        for wchk in sorted(w_checks - h_checks):
            w_in = _in_list_values(wchk)
            paired = None
            if w_in is not None:
                for hchk in sorted(h_checks - w_checks):
                    h_in = _in_list_values(hchk)
                    if h_in is not None and h_in[0] == w_in[0]:
                        paired = h_in
                        break
            if paired is not None:
                lacks = [v for v in w_in[1] if v not in paired[1]]
                extra = [v for v in paired[1] if v not in w_in[1]]
                if lacks:
                    out.append(f"table {table}: column {name} CHECK ({w_in[0]} IN (...)) lacks {', '.join(lacks)}")
                if extra:
                    out.append(f"table {table}: column {name} CHECK ({w_in[0]} IN (...)) has undeclared {', '.join(extra)}")
                h_checks.discard(next(h for h in h_checks if _in_list_values(h) == paired))
                named = True
            else:
                out.append(f"table {table}: column {name} CHECK is absent — want «{_excerpt(wchk)}»")
                named = True
        for hchk in sorted(h_checks - w_checks):
            out.append(f"table {table}: column {name} CHECK is not declared by the code — have «{_excerpt(hchk)}»")
            named = True
        if not named:
            out.append(f"table {table}: column {name} differs — have «{_excerpt(hdef)}» want «{_excerpt(wdef)}»")
    for key, wdef in w_keyed.items():
        if key not in h_keyed:
            out.append(f"table {table}: {key} is absent")
        elif h_keyed[key] != wdef:
            out.append(f"table {table}: {key} differs — have «{_excerpt(h_keyed[key])}» want «{_excerpt(wdef)}»")
    for key in h_keyed:
        if key not in w_keyed:
            out.append(f"table {table}: {key} is not declared by the code")
    for text in sorted(w_text - h_text):
        out.append(f"table {table}: {_excerpt(text)} is absent")
    for text in sorted(h_text - w_text):
        out.append(f"table {table}: {_excerpt(text)} is not declared by the code")
    if h_tail != w_tail:
        out.append(f"table {table}: table tail differs — have «{h_tail or 'rowid table'}» want «{w_tail or 'rowid table'}»")
    if not out:
        out.append(f"table {table}: CREATE text differs — have «{_excerpt(have_sql)}» want «{_excerpt(want_sql)}»")
    return out


@contextlib.contextmanager
def reading(path: str) -> Iterator[sqlite3.Connection]:
    """A read-only connection, closed when the block ends."""
    conn = connect(path, readonly=True)
    try:
        yield conn
    finally:
        conn.close()


def upsert_sql(table: str, cols: Sequence[str], pk: Sequence[str], guard: Optional[str] = None, keep: Sequence[str] = (),
               update: Optional[Mapping[str, str]] = None, source: Optional[str] = None) -> str:
    """``INSERT INTO table (cols) VALUES (?…) ON CONFLICT(pk) DO UPDATE SET <non-key col> = excluded.<col> …``.
    ``keep`` names columns left untouched on conflict (first-arrival facts); ``update`` overrides the expression
    for a column (``{"n_ok": "n_ok + excluded.n_ok"}``); ``guard`` is a ``WHERE`` on the update (the item
    tables' order-key rule). With no updatable column the conflict does nothing. ``source`` is a SELECT (with a
    WHERE clause — SQLite's upsert grammar requires one) used in place of ``VALUES``: the set-based form."""
    cols = list(cols)
    sets = [c for c in cols if c not in pk and c not in keep]
    values = source if source is not None else "VALUES (" + ", ".join("?" for _ in cols) + ")"
    sql = (f"INSERT INTO {table} (" + ", ".join(cols) + f") {values} "
           f"ON CONFLICT(" + ", ".join(pk) + ") ")
    if not sets:
        return sql + "DO NOTHING"
    sql += "DO UPDATE SET " + ", ".join(f"{c} = {(update or {}).get(c, f'excluded.{c}')}" for c in sets)
    if guard:
        sql += f" WHERE {guard}"
    return sql


def executemany_batched(conn: sqlite3.Connection, sql: str, rows: Iterable[Sequence], batch: int = BATCH_ROWS) -> int:
    """``executemany`` in batches of at least ``batch`` rows; returns the row count. The caller owns
    the transaction (``with transaction(conn):``)."""
    buf: list = []
    n = 0
    for r in rows:
        buf.append(r)
        if len(buf) >= batch:
            conn.executemany(sql, buf)
            n += len(buf)
            buf = []
    if buf:
        conn.executemany(sql, buf)
        n += len(buf)
    return n


@contextlib.contextmanager
def transaction(conn: sqlite3.Connection, immediate: bool = True) -> Iterator[sqlite3.Connection]:
    """One transaction; ``BEGIN IMMEDIATE`` takes the write lock up front so a writer never waits
    mid-transaction."""
    conn.execute("BEGIN IMMEDIATE" if immediate else "BEGIN")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    else:
        conn.execute("COMMIT")


@contextlib.contextmanager
def savepoint(conn: sqlite3.Connection, name: str = "sp") -> Iterator[None]:
    conn.execute(f"SAVEPOINT {name}")
    try:
        yield
    except BaseException:
        conn.execute(f"ROLLBACK TO {name}")
        conn.execute(f"RELEASE {name}")
        raise
    else:
        conn.execute(f"RELEASE {name}")


class LockHeld(RuntimeError):
    """Another process holds the harvester lock."""


@contextlib.contextmanager
def write_lock(root: str, timeout_s: float = 0.0, name: str = "harvester.lock", shared: bool = False) -> Iterator[None]:
    """``flock`` on ``<root>/<name>`` (exclusive, or ``shared`` for a reader); raises :class:`LockHeld` after
    ``timeout_s`` (0 = try once)."""
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, name)
    fh = open(path, "a+")
    deadline = time.monotonic() + timeout_s
    try:
        while True:
            try:
                fcntl.flock(fh.fileno(), (fcntl.LOCK_SH if shared else fcntl.LOCK_EX) | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise LockHeld(f"lock held: {path}")
                time.sleep(0.02)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    finally:
        fh.close()


def drop_indexes(conn: sqlite3.Connection, keep: Sequence[str] = ()) -> list:
    """Drop every explicit index (bulk backfill) except the ones named in ``keep``; returns the dropped indexes'
    CREATE statements for :func:`recreate_indexes`. ``keep`` names the indexes the load itself reads on every step
    (a replay's revision path looks item rows up by ``(source_revision_id, record_file_id)`` and by the withdrawn
    mark): without them each such lookup is a full scan of a table that grows to millions of rows."""
    rows = conn.execute("SELECT name, sql FROM sqlite_master WHERE type='index' AND sql IS NOT NULL ORDER BY name").fetchall()
    kept = set(keep)
    out = []
    for name, sql in rows:
        if name in kept:
            continue
        conn.execute(f"DROP INDEX {name}")
        out.append(sql)
    return out


def recreate_indexes(conn: sqlite3.Connection, statements: Sequence[str]) -> None:
    for sql in statements:
        conn.execute(sql)
    conn.execute("ANALYZE")


def free_bytes(path: str) -> int:
    """Free bytes on the filesystem holding ``path``, clamped to its total: a container filesystem can report an
    absurd free figure (~9.2e18 on the Modal CPU suite), which would make every free-space guard think space is
    infinite. Never more than ``total``."""
    du = shutil.disk_usage(path)
    return min(int(du.free), int(du.total))


def fsync_file_and_dir(path: str) -> None:
    """``fsync`` a file and the directory holding it (a rename is durable only once the directory entry is)."""
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
    dfd = os.open(os.path.dirname(os.path.abspath(path)) or ".", os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def file_size(path: str) -> int:
    total = 0
    for suffix in ("", "-wal", "-shm"):
        p = path + suffix
        if os.path.exists(p):
            total += os.path.getsize(p)
    return total
