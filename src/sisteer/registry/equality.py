"""Equality of two registry databases, streamed: the primary-key merge behind ``harvest.verify_equal``, and the comparison
contracts of the superseded-revision pruning.

**The merge** (:func:`verify_equal`). Each compared table is read from both databases in primary-key order (``ORDER BY`` the
key columns, which every table's primary key index serves) and the two streams are merged one key at a time, so memory holds one
key's rows, never a table. The result is EXACTLY the set-based comparison it replaces (:func:`verify_equal_sets`, kept as the
reference the tests pin the stream against): per table, the size of the symmetric difference of the two sides' DISTINCT rows,
each value normalized the way the set-based form did (a float rounded to 12 places, a BLOB as hex). A key present on one side
counts its rows; a key present on both with unequal rows counts both. Rows are compared raw first and normalized only when the raw
values differ, so the common case costs one tuple comparison.

The merge depends on Python ordering the keys the way SQLite does (no registry column declares a collation, so SQLite's order is
BINARY: NULL, then numbers, then text by UTF-8 bytes — the order of Python's code points —, then BLOBs bytewise). The stream checks
that order as it reads: a key that sorts before its predecessor raises :class:`OrderMismatch` instead of mis-merging, so a merge
that cannot be trusted fails loudly rather than reporting a wrong count.

A merge key holding a column that can store a REAL (declared REAL, NUMERIC or with no type: ``run_metric`` and ``run_contrast`` carry
``threshold`` in their primary keys) is merged on the NORMALIZED key, as the set-based form compared it: two keys whose floats round to
the same 12 places are one key, and their rows are compared as one set. Rounding is monotone, so in SQLite's raw order the normalized
prefix through the first such column never decreases; the stream reads one such prefix (a block: the few rows that share it) at a time
and orders the block's rows by their whole normalized key, so a collapse anywhere in the key is merged correctly with memory for one
block. A float in any other key column (an INTEGER column holding a non-integral value) stops the merge loudly.

**The diff sink**. ``diff_sink(table, key, rows_a, rows_b)`` is called once per differing key with that key's distinct rows on each
side (one side empty for a one-sided key); :class:`DiffWriter` writes them as JSON lines (``registry rebuild --diff-out``,
``registry prune-superseded compare --expect rebuild``), and :func:`classify_diffs` splits one diff file by whether a baseline diff
(the old code's rebuild against the unpruned database) shows the SAME residual: the same table and normalized key
with the same rows on each side. Anything else at a key the baseline also differs at -- a row changed, added or lost -- is new.

**The contracts** (:func:`check_contract`). Each pruning step is checked across exactly one change, over two digest documents
(``registry prune-superseded digest``; built by ``prune_archive.digest_database``):

- ``migration`` (before and after the migration): every table and object equal, except ``schema_migration`` (the rows before the
  migration plus exactly one new row of :data:`MIGRATION_VERSION` / :data:`MIGRATION_SLUG`), ``source_revision`` (equal except
  ``reclaimed``, which is 0 on every row after the migration), ``item_reclaim_ledger`` (new and empty), the ``v_paper_inputs_head``
  text (its value equal) and the dropped index ``idx_item_result_withdrawn_seq``.
- ``deletion`` (before and after the deletion, one schema): per item table, ``digest(before) = digest(after) + digest(archive)`` and
  the counts likewise; ``item_reclaim_ledger`` equal to the archive's group list; ``source_revision`` equal except ``reclaimed``,
  which after the deletion is 1 only on an accepted, superseded row and 0 on such a row only while item rows of it remain (a revision
  with a kept carrier whose log file is missing stays due; the digest's ``reclaimed_unexplained``); every other table,
  ``schema_migration`` included, and every ``v_*`` object equal.
- ``identity`` (before and after the compaction): everything equal.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Set, Tuple

from . import prune as P
from . import schema_def as S

#: The item tables the pruning deletes from (never ``source_status``, never ``observation``): ``prune.PRUNE_TABLES``.
PRUNED_TABLES: Tuple[str, ...] = P.PRUNE_TABLES
#: The migration that arms the new reclaim (``migrations/0033_prune_superseded_item_rows.sql``). Renumbering it (another migration
#: landing first) changes the version only.
MIGRATION_SLUG = "prune_superseded_item_rows"
MIGRATION_VERSION = 33
#: The derived table the new reclaim writes (``prune.LEDGER_TABLE``).
LEDGER_TABLE = P.LEDGER_TABLE
#: The schema objects the migration changes: the paper-input head's text, the dropped partial index.
MIGRATION_CHANGED_OBJECTS: Tuple[str, ...] = ("view:v_paper_inputs_head",)
MIGRATION_DROPPED_OBJECTS: Tuple[str, ...] = ("index:idx_item_result_withdrawn_seq",)
CONTRACTS: Tuple[str, ...] = ("migration", "deletion", "identity")
#: The older schemas a rebuild comparison may read (``registry prune-superseded compare --expect rebuild --legacy-schema N``): per
#: version, the code's tables that schema does not have. Both files must be at that version; every other table is compared as the
#: code declares it. The schema before the migration lacks only the ledger it creates (the schema of the old code's rebuild against the
#: unpruned database).
LEGACY_SCHEMAS: Dict[int, Tuple[str, ...]] = {MIGRATION_VERSION - 1: (LEDGER_TABLE,)}
DIFF_FORMAT = "registry-diff/1"
FETCH_ROWS = 5000

DiffSink = Callable[[str, list, List[tuple], List[tuple]], None]


class OrderMismatch(RuntimeError):
    """A stream returned a key that sorts before its predecessor under the merge's order: the merge cannot be trusted."""


# --------------------------------------------------------------------------------------------
# values: the set-based normalization, SQLite's order, the exact JSON encoding
# --------------------------------------------------------------------------------------------


def norm(v):
    """The set-based ``verify_equal``'s value normalization (``harvest._norm``): a BLOB as hex, a float rounded to 12 places."""
    if isinstance(v, (bytes, memoryview)):
        return bytes(v).hex()
    if isinstance(v, float):
        return round(v, 12)
    return v


def norm_row(row: Sequence) -> tuple:
    return tuple(norm(v) for v in row)


def key_norm(v):
    """A key value as the merge compares it: a float rounded to 12 places (:func:`norm`'s rounding; the value keeps its type, so SQLite's
    order is kept), anything else as it is."""
    return round(v, 12) if isinstance(v, float) else v


def affinity(declared: Optional[str]) -> str:
    """SQLite's column affinity of a declared type (``INTEGER`` / ``TEXT`` / ``BLOB`` / ``REAL`` / ``NUMERIC``)."""
    t = (declared or "").upper()
    if "INT" in t:
        return "INTEGER"
    if "CHAR" in t or "CLOB" in t or "TEXT" in t:
        return "TEXT"
    if "BLOB" in t or not t:
        return "BLOB"
    if "REAL" in t or "FLOA" in t or "DOUB" in t:
        return "REAL"
    return "NUMERIC"


#: The affinities whose columns can store a REAL value as it was given (an INTEGER column keeps only a non-integral one).
REAL_AFFINITIES = ("REAL", "NUMERIC", "BLOB")


def _declared_types(conn, table: str) -> Dict[str, str]:
    return {str(r[1]): str(r[2] or "") for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def order_key(v) -> tuple:
    """``v``'s position in SQLite's BINARY order: NULL < INTEGER/REAL (numerically) < TEXT (UTF-8 bytes, which is code-point order)
    < BLOB (bytewise)."""
    if v is None:
        return (0, 0)
    if isinstance(v, (int, float)):
        return (1, v)
    if isinstance(v, str):
        return (2, v)
    return (3, bytes(v))


#: One SQLite value as exact JSON: NULL ``null``, INTEGER an int, TEXT a string, REAL ``{"r": float.hex()}``, BLOB ``{"x": hex}``
#: (``prune.encode_value``, the archive's row encoding; :func:`decode_value` inverts it exactly).
encode_value = P.encode_value


def decode_value(j):
    """The strict inverse of :func:`encode_value`: an encoded REAL or BLOB is an object of exactly its one key."""
    if j is None or isinstance(j, (int, str)):
        return j
    if isinstance(j, dict):
        if set(j) == {"r"}:
            return float.fromhex(j["r"])
        if set(j) == {"x"}:
            return bytes.fromhex(j["x"])
    raise ValueError(f"equality.decode_value: not an encoded value: {j!r}")


# --------------------------------------------------------------------------------------------
# the streaming merge
# --------------------------------------------------------------------------------------------


def default_exclusions() -> Tuple[Tuple[str, ...], Dict[str, Tuple[str, ...]]]:
    """Rebuild equality's exclusions, read from ``harvest`` (imported here, never at module load: harvest imports this module)."""
    from . import harvest as H
    return tuple(H.VERIFY_EXCLUDE_TABLES), dict(H.VERIFY_EXCLUDE_COLUMNS)


def legacy_tables(version: int) -> List[str]:
    """The code's tables a file at the older schema ``version`` (:data:`LEGACY_SCHEMAS`) has, in the code's table order."""
    if version not in LEGACY_SCHEMAS:
        raise ValueError(f"equality: schema {version} is not a declared legacy schema ({sorted(LEGACY_SCHEMAS)})")
    absent = set(LEGACY_SCHEMAS[version])
    return [t.name for t in S.TABLES if t.name not in absent]


def compared_tables(exclude_tables: Optional[Iterable[str]] = None, exclude_columns: Optional[Mapping[str, Sequence[str]]] = None,
                    tables: Optional[Iterable[str]] = None) -> List[Tuple[str, List[str], List[str]]]:
    """``(table, compared columns, merge key)`` per compared table, in the code's table order. The merge key is the primary key's
    compared columns: a primary-key column excluded from the comparison leaves several rows per merge key, compared as sets."""
    if exclude_tables is None or exclude_columns is None:
        d_tables, d_columns = default_exclusions()
        exclude_tables = d_tables if exclude_tables is None else exclude_tables
        exclude_columns = d_columns if exclude_columns is None else exclude_columns
    skip = set(exclude_tables)
    only = None if tables is None else set(tables)
    if only is not None:
        unknown = sorted(only - set(S.TABLE_BY_NAME))
        if unknown:
            raise ValueError(f"equality: unknown table(s) {unknown}")
    out = []
    for t in S.TABLES:
        if t.name in skip or (only is not None and t.name not in only):
            continue
        drop = set(exclude_columns.get(t.name, ()))
        cols = [c for c in t.column_names if c not in drop]
        key = [c for c in t.pk if c in cols]
        out.append((t.name, cols, key))
    return out


def _stream(conn, table: str, cols: Sequence[str], key: Sequence[str], fetch: int = FETCH_ROWS) -> Iterator[tuple]:
    cur = conn.cursor()
    cur.row_factory = None          # plain tuples, whatever the connection's factory
    order = ", ".join(key) if key else ", ".join(cols)
    cur.execute(f"SELECT {', '.join(cols)} FROM {table} ORDER BY {order}")
    try:
        while True:
            batch = cur.fetchmany(fetch)
            if not batch:
                return
            yield from batch
    finally:
        cur.close()


def _groups(rows: Iterator[tuple], key_idx: Sequence[int], side: str, table: str) -> Iterator[Tuple[tuple, List[tuple]]]:
    """``(order key, rows)`` per distinct merge key, in order, for a key with no column that can hold a REAL; raises
    :class:`OrderMismatch` on a key below its predecessor, or on a float in a key column (the key would need normalizing)."""
    prev = None
    group: List[tuple] = []
    for r in rows:
        k = tuple(order_key(r[i]) for i in key_idx)
        if prev is None or k != prev:
            if any(type(r[i]) is float for i in key_idx):
                raise OrderMismatch(f"equality: {table} on side {side} holds a REAL key value in a column of another affinity "
                                    f"({[r[i] for i in key_idx]!r}); the merge does not normalize that key -- the comparison cannot be trusted")
        if prev is None or k == prev:
            group.append(r)
        elif k > prev:
            yield prev, group
            group = [r]
        else:
            raise OrderMismatch(f"equality: {table} on side {side} returned key {k!r} after {prev!r}; SQLite's order is not "
                                "the merge's -- the comparison cannot be trusted")
        prev = k
    if group:
        yield prev, group


def _normalized_groups(rows: Iterator[tuple], key_idx: Sequence[int], first_real: int, side: str,
                       table: str) -> Iterator[Tuple[tuple, List[tuple]]]:
    """``(order key, rows)`` per distinct NORMALIZED merge key (:func:`key_norm`), in order: the rows of one block -- one normalized key
    prefix through ``key_idx[first_real]``, the first column that can hold a REAL -- are read whole and grouped by their full normalized
    key; raises :class:`OrderMismatch` on a block below its predecessor."""
    pre = list(key_idx[:first_real + 1])
    prev = None
    block: Dict[tuple, List[tuple]] = {}
    for r in rows:
        b = tuple(order_key(key_norm(r[i])) for i in pre)
        if b != prev:
            if prev is not None:
                if b < prev:
                    raise OrderMismatch(f"equality: {table} on side {side} returned key prefix {b!r} after {prev!r}; SQLite's order "
                                        "is not the merge's -- the comparison cannot be trusted")
                for k in sorted(block):
                    yield k, block[k]
                block = {}
            prev = b
        block.setdefault(tuple(order_key(key_norm(r[i])) for i in key_idx), []).append(r)
    for k in sorted(block):
        yield k, block[k]


def _distinct(rows: List[tuple]) -> List[tuple]:
    """``rows`` deduplicated under the set-based comparison's equality (normalized values), first occurrence kept."""
    if len(rows) == 1:
        return rows
    seen: Set[tuple] = set()
    out = []
    for r in rows:
        n = norm_row(r)
        if n not in seen:
            seen.add(n)
            out.append(r)
    return out


def _group_diff(ra: List[tuple], rb: List[tuple]) -> Tuple[int, List[tuple], List[tuple]]:
    """The symmetric-difference size of one key's rows and the rows on each side that the other lacks."""
    if len(ra) == 1 and len(rb) == 1:
        a, b = ra[0], rb[0]
        if a == b or norm_row(a) == norm_row(b):
            return 0, [], []
        return 2, [a], [b]
    na = {norm_row(r): r for r in _distinct(ra)}
    nb = {norm_row(r): r for r in _distinct(rb)}
    only_a = [r for n, r in na.items() if n not in nb]
    only_b = [r for n, r in nb.items() if n not in na]
    return len(only_a) + len(only_b), only_a, only_b


def table_diff(a, b, table: str, cols: Sequence[str], key: Sequence[str], diff_sink: Optional[DiffSink] = None,
               fetch: int = FETCH_ROWS) -> int:
    """The symmetric-difference size of ``table`` between connections ``a`` and ``b`` over ``cols``, merged on ``key`` (normalized when a
    key column can hold a REAL: module doc)."""
    key_cols = list(key) if key else list(cols)
    key_idx = [list(cols).index(c) for c in key_cols]
    ta, tb = _declared_types(a, table), _declared_types(b, table)
    first_real = next((j for j, c in enumerate(key_cols)
                       if affinity(ta.get(c)) in REAL_AFFINITIES or affinity(tb.get(c)) in REAL_AFFINITIES), None)

    def grouped(conn, side: str) -> Iterator[Tuple[tuple, List[tuple]]]:
        rows = _stream(conn, table, cols, key, fetch)
        if first_real is None:
            return _groups(rows, key_idx, side, table)
        return _normalized_groups(rows, key_idx, first_real, side, table)
    ga = grouped(a, "a")
    gb = grouped(b, "b")
    n = 0
    xa = next(ga, None)
    xb = next(gb, None)
    while xa is not None or xb is not None:
        if xb is None or (xa is not None and xa[0] < xb[0]):
            rows = _distinct(xa[1])
            n += len(rows)
            if diff_sink is not None:
                diff_sink(table, [rows[0][i] for i in key_idx], rows, [])
            xa = next(ga, None)
        elif xa is None or xb[0] < xa[0]:
            rows = _distinct(xb[1])
            n += len(rows)
            if diff_sink is not None:
                diff_sink(table, [rows[0][i] for i in key_idx], [], rows)
            xb = next(gb, None)
        else:
            d, only_a, only_b = _group_diff(xa[1], xb[1])
            if d:
                n += d
                if diff_sink is not None:
                    first = (only_a or only_b)[0]
                    diff_sink(table, [first[i] for i in key_idx], only_a, only_b)
            xa = next(ga, None)
            xb = next(gb, None)
    return n


def verify_equal(a, b, exclude_tables: Optional[Iterable[str]] = None, exclude_columns: Optional[Mapping[str, Sequence[str]]] = None,
                 tables: Optional[Iterable[str]] = None, diff_sink: Optional[DiffSink] = None, fetch: int = FETCH_ROWS) -> Dict[str, int]:
    """Row-level diff per compared table: ``{table: symmetric-difference size}`` over distinct normalized rows, streamed by
    primary-key merge. ``exclude_tables`` / ``exclude_columns`` default to rebuild equality's (``harvest.VERIFY_EXCLUDE_TABLES`` /
    ``VERIFY_EXCLUDE_COLUMNS``); ``tables`` restricts the comparison to the named tables; ``diff_sink`` receives every differing
    key (module doc)."""
    out: Dict[str, int] = {}
    for table, cols, key in compared_tables(exclude_tables, exclude_columns, tables):
        out[table] = table_diff(a, b, table, cols, key, diff_sink=diff_sink, fetch=fetch)
    return out


def verify_equal_sets(a, b, exclude_tables: Optional[Iterable[str]] = None, exclude_columns: Optional[Mapping[str, Sequence[str]]] = None,
                      tables: Optional[Iterable[str]] = None) -> Dict[str, int]:
    """The set-based comparison the stream replaced (every row of both sides in memory): the reference
    ``tests/test_registry_equality_stream.py`` pins :func:`verify_equal` against. Not for a database of any size."""
    out: Dict[str, int] = {}
    for table, cols, _key in compared_tables(exclude_tables, exclude_columns, tables):
        sql = f"SELECT {', '.join(cols)} FROM {table}"
        ra = set(norm_row(r) for r in a.execute(sql).fetchall())
        rb = set(norm_row(r) for r in b.execute(sql).fetchall())
        out[table] = len(ra ^ rb)
    return out


# --------------------------------------------------------------------------------------------
# diff files: write, read, classify against a baseline
# --------------------------------------------------------------------------------------------


class DiffWriter:
    """A :data:`DiffSink` writing one JSON line per differing key: ``{"table", "key", "a", "b"}`` with every value exactly encoded
    (:func:`encode_value`), preceded by a header line naming the format and the two sides."""

    def __init__(self, path: str, side_a: str = "a", side_b: str = "b"):
        self.path = path
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self.fh = open(path, "w", encoding="utf-8")
        self.lines = 0
        self.fh.write(json.dumps({"format": DIFF_FORMAT, "a": side_a, "b": side_b}, separators=(",", ":")) + "\n")

    def __call__(self, table: str, key: list, rows_a: List[tuple], rows_b: List[tuple]) -> None:
        line = {"table": table, "key": [encode_value(v) for v in key],
                "a": [[encode_value(v) for v in r] for r in rows_a], "b": [[encode_value(v) for v in r] for r in rows_b]}
        self.fh.write(json.dumps(line, separators=(",", ":"), ensure_ascii=False) + "\n")
        self.lines += 1

    def close(self) -> None:
        self.fh.close()

    def __enter__(self) -> "DiffWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def read_diffs(path: str) -> Iterator[Dict[str, object]]:
    """The differing keys of a diff file (the header line checked and skipped)."""
    with open(path, "r", encoding="utf-8") as fh:
        head = json.loads(fh.readline() or "{}")
        if head.get("format") != DIFF_FORMAT:
            raise ValueError(f"equality: {path} is not a {DIFF_FORMAT} file (header {head!r})")
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _diff_id(d: Mapping[str, object]) -> str:
    """A differing key's identity: its table and its NORMALIZED key (the merge's), so a residual is found again whatever float of a
    rounding bucket either diff file happened to name."""
    return json.dumps([d["table"], [key_norm(decode_value(v)) for v in d["key"]]], separators=(",", ":"), ensure_ascii=False)


def _residual(rows: Iterable[Sequence[object]]) -> Set[tuple]:
    """One side's rows of a differing key as the comparison sees them: distinct normalized rows."""
    return {norm_row([decode_value(v) for v in r]) for r in rows}


def classify_diffs(path: str, baseline_path: str, examples: int = 5) -> Dict[str, object]:
    """Split the differences of ``path`` by whether ``baseline_path`` shows the SAME residual: the same table and normalized key, and on
    each side the same distinct normalized rows. ``baseline`` = waived (a residual the old code's rebuild already had, unchanged); ``new``
    = anything else, ``changed_at_baseline_key`` among them when the baseline differs at that key with other rows (a row changed, added
    or lost since). Per table counts, a few examples of the new ones, and ``ok`` = no new difference."""
    base: Dict[str, Tuple[Set[tuple], Set[tuple]]] = {}
    for d in read_diffs(baseline_path):
        have = base.setdefault(_diff_id(d), (set(), set()))
        have[0].update(_residual(d.get("a") or []))
        have[1].update(_residual(d.get("b") or []))
    per: Dict[str, Dict[str, int]] = {}
    new_examples: List[Mapping[str, object]] = []
    n_new = n_base = n_changed = 0
    for d in read_diffs(path):
        c = per.setdefault(str(d["table"]), {"new": 0, "baseline": 0, "changed_at_baseline_key": 0})
        b = base.get(_diff_id(d))
        if b is not None and (_residual(d.get("a") or []), _residual(d.get("b") or [])) == b:
            c["baseline"] += 1
            n_base += 1
            continue
        c["new"] += 1
        n_new += 1
        if b is not None:
            c["changed_at_baseline_key"] += 1
            n_changed += 1
        if len(new_examples) < examples:
            new_examples.append(d)
    return {"ok": n_new == 0, "new": n_new, "baseline": n_base, "changed_at_baseline_key": n_changed, "tables": per,
            "new_examples": new_examples}


# --------------------------------------------------------------------------------------------
# the comparison contracts over digest documents
# --------------------------------------------------------------------------------------------

M256 = P.MOD


def dsum(entry: Mapping[str, object]) -> Tuple[int, int]:
    """``(rows, sum)`` of one digest entry (``{"rows": n, "sum256": hex}``)."""
    return int(entry["rows"]), int(str(entry["sum256"]), 16)


def dadd(x: Tuple[int, int], y: Tuple[int, int]) -> Tuple[int, int]:
    return x[0] + y[0], (x[1] + y[1]) % M256


def dfmt(x: Tuple[int, int]) -> Dict[str, object]:
    return {"rows": x[0], "sum256": f"{x[1]:064x}"}


class _Check:
    def __init__(self):
        self.failures: List[str] = []
        self.checked: List[str] = []

    def eq(self, what: str, a, b) -> None:
        self.checked.append(what)
        if a != b:
            self.failures.append(f"{what}: {_short(a)} != {_short(b)}")

    def fail(self, what: str) -> None:
        self.failures.append(what)


def _short(v, limit: int = 300) -> str:
    s = json.dumps(v, sort_keys=True, default=str) if not isinstance(v, str) else v
    return s if len(s) <= limit else s[:limit] + " …"


def _schema_delta(a: Mapping[str, object], b: Mapping[str, object]) -> Tuple[Set[str], Set[str], Set[str]]:
    """Schema entries (``"<type>:<name>" -> [tbl_name, normalized sql]``) added, removed and changed from ``a`` to ``b``."""
    added = set(b) - set(a)
    removed = set(a) - set(b)
    changed = {k for k in set(a) & set(b) if list(a[k]) != list(b[k])}
    return added, removed, changed


def archive_table_sums(archive_manifest: Mapping[str, object]) -> Dict[str, Tuple[int, int]]:
    """``{table: (rows, digest_all sum)}`` of an archive MANIFEST (``prune_archive``)."""
    out = {}
    for t, e in dict(archive_manifest.get("tables") or {}).items():
        out[t] = (int(e["rows"]), int(str(e["digest_all"]["sum256"]), 16))
    return out


def check_contract(a: Mapping[str, object], b: Mapping[str, object], expect: str, archive_manifest: Optional[Mapping[str, object]] = None,
                   archive_groups: Optional[Iterable[Mapping[str, object]]] = None) -> Dict[str, object]:
    """Check one contract (module doc) between digest documents ``a`` (before) and ``b`` (after). ``deletion`` needs the archive's
    MANIFEST and its group list (``prune_archive.read_groups``). Returns ``{"ok", "expect", "failures", "checked"}``."""
    if expect not in CONTRACTS:
        raise ValueError(f"equality: unknown contract {expect!r} (one of {', '.join(CONTRACTS)})")
    if expect == "deletion" and (archive_manifest is None or archive_groups is None):
        raise ValueError("equality: the deletion contract needs the archive (its MANIFEST and groups)")
    ck = _Check()
    ta, tb = dict(a.get("tables") or {}), dict(b.get("tables") or {})
    oa, ob = dict(a.get("objects") or {}), dict(b.get("objects") or {})
    xa, xb = dict(a.get("extras") or {}), dict(b.get("extras") or {})
    sa, sb = dict(a.get("schema") or {}), dict(b.get("schema") or {})
    special = {"schema_migration", "source_revision", LEDGER_TABLE}
    if expect == "deletion":
        special |= set(PRUNED_TABLES)

    # the schema (sqlite_master): identical, except what the migration changes
    added, removed, changed = _schema_delta(sa, sb)
    if expect == "migration":
        # the ledger table, and whatever index the code declares on it (schema entries are [tbl_name, sql])
        want_added = {k for k in added if list(sb[k])[0] == LEDGER_TABLE}
        if f"table:{LEDGER_TABLE}" not in added:
            ck.fail(f"schema: table {LEDGER_TABLE} was not added")
        ck.eq("schema: objects added", sorted(added), sorted(want_added))
        ck.eq("schema: objects removed", sorted(removed), sorted(MIGRATION_DROPPED_OBJECTS))
        ck.eq("schema: objects changed", sorted(changed), sorted(k for k in MIGRATION_CHANGED_OBJECTS if k in changed))
        if not any(k in changed for k in MIGRATION_CHANGED_OBJECTS):
            ck.fail("schema: the v_paper_inputs_head text did not change")
    else:
        ck.eq("schema: objects added", sorted(added), [])
        ck.eq("schema: objects removed", sorted(removed), [])
        ck.eq("schema: objects changed", sorted(changed), [])

    # every table outside the special ones: equal digests
    if expect == "migration":
        ck.eq("tables: set", sorted(set(tb) - {LEDGER_TABLE}), sorted(ta))
    else:
        ck.eq("tables: set", sorted(tb), sorted(ta))
    for t in sorted(set(ta) & set(tb)):
        if t in special and expect != "identity":
            continue
        ck.eq(f"table {t}", dsum(ta[t]), dsum(tb[t]))
    # every v_* object: equal digests (the migration's changed text keeps the value: v_paper_inputs_head is compared too), under one
    # projection of the clock-derived columns (``prune_archive.CLOCK_DERIVED_COLUMNS``)
    ck.eq("objects: the clock-derived columns left out", a.get("projection"), b.get("projection"))
    ck.eq("objects: set", sorted(ob), sorted(oa))
    for o in sorted(set(oa) & set(ob)):
        ck.eq(f"object {o}", dsum(oa[o]), dsum(ob[o]))
    if expect == "identity":
        for k in sorted(set(xa) | set(xb)):
            ck.eq(f"extras {k}", xa.get(k), xb.get(k))
        return {"ok": not ck.failures, "expect": expect, "failures": ck.failures, "checked": len(ck.checked)}

    # source_revision: equal apart from `reclaimed`
    ra, rb = dict(xa.get("source_revision") or {}), dict(xb.get("source_revision") or {})
    if not ra or not rb:
        ck.fail("source_revision: a digest document lacks its extras")
    else:
        ck.eq("source_revision without reclaimed", dsum(ra["without_reclaimed"]), dsum(rb["without_reclaimed"]))
        if expect == "migration":
            ck.eq("source_revision: rows with reclaimed <> 0 after the migration", rb.get("reclaimed_nonzero"), 0)
        else:
            # a digest taken by code before the key was written has none: it fails here, and is taken again
            ck.eq("source_revision: rows whose reclaimed is 1 but not accepted AND superseded, or 0 on an accepted, superseded revision "
                  "with no item row left, after the deletion", rb.get("reclaimed_unexplained"), 0)

    # schema_migration: the migration adds exactly one row; the deletion none
    ma = [tuple(r) for r in (xa.get("schema_migration") or [])]
    mb = [tuple(r) for r in (xb.get("schema_migration") or [])]
    if expect == "migration":
        extra = [r for r in mb if r not in ma]
        missing = [r for r in ma if r not in mb]
        ck.eq("schema_migration: the rows before the migration kept", missing, [])
        ok_new = len(extra) == 1 and int(extra[0][0]) == MIGRATION_VERSION and str(extra[0][2]) == MIGRATION_SLUG
        ck.checked.append("schema_migration: one new row")
        if not ok_new:
            ck.fail(f"schema_migration: expected exactly one new row ({MIGRATION_VERSION}, {MIGRATION_SLUG}); got {extra!r}")
        if LEDGER_TABLE in ta:
            ck.fail(f"{LEDGER_TABLE}: present before the migration")
        if LEDGER_TABLE not in tb:
            ck.fail(f"{LEDGER_TABLE}: absent after the migration")
        else:
            ck.eq(f"{LEDGER_TABLE}: empty after the migration", int(tb[LEDGER_TABLE]["rows"]), 0)
        return {"ok": not ck.failures, "expect": expect, "failures": ck.failures, "checked": len(ck.checked)}

    # deletion
    ck.eq("schema_migration", sorted(ma), sorted(mb))
    sums = archive_table_sums(archive_manifest or {})
    for t in PRUNED_TABLES:
        if t not in ta or t not in tb:
            ck.fail(f"table {t}: absent from a digest document")
            continue
        ck.eq(f"table {t}: before = after + archive", dsum(ta[t]), dadd(dsum(tb[t]), sums.get(t, (0, 0))))
    groups = sorted((str(g["table"]), str(g["revision_id"]), int(g["record_file_id"]), int(g["rows"])) for g in (archive_groups or [])
                    if g.get("class", "deletable") == "deletable")
    ledger_a = [tuple(r) for r in (xa.get(LEDGER_TABLE) or [])]
    ledger_b = sorted((str(r[0]), str(r[1]), int(r[2]), int(r[3])) for r in (xb.get(LEDGER_TABLE) or []))
    ck.eq(f"{LEDGER_TABLE}: empty before the deletion", ledger_a, [])
    ck.eq(f"{LEDGER_TABLE}: equal to the archive's group list", ledger_b, groups)
    return {"ok": not ck.failures, "expect": expect, "failures": ck.failures, "checked": len(ck.checked)}
