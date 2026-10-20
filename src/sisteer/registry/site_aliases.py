"""Site aliases: the private map from a deployment's own names for its machines to the registry's neutral labels.

The registry names where a run ran, where an object was fitted and which copy of the record log a mirror reached with a
small vocabulary (``vocab.SUBSTRATES``, ``vocab.FIT_SEARCH_SUBSTRATES``, ``vocab.MIRROR_DESTINATIONS``): ``cluster_a``,
``cluster_b``, ``grading_host`` and the generic ``modal`` / ``runpod`` / ``slurm``. A deployment that wrote records
before those labels existed spelled its machines its own way, and its records are immutable. This module carries that
spelling without putting it in code: the site lists each legacy spelling once, in its own file beside the registry's
``config.toml``::

    # <root>/site_aliases.toml
    [site_aliases]
    oldsite_a = "cluster_a"

and every reader that meets a legacy spelling maps it here: the validator before its vocabulary check, the fold before
it writes a live row, migration 0032 before it rebuilds the five tables that carry a label, and the readers of
persisted state (``legacy_keys`` / ``get_compat``) for keys that embed a label.

Rules (each refused by name, never repaired):

* the file holds one table, ``[site_aliases]``, of ``spelling = "label"`` pairs and nothing else
* a spelling matches ``^[a-z][a-z0-9_]*$`` and is not itself a label (no shadowing)
* a label is one of ``vocab.SUBSTRATES`` and is never also a spelling (no chains)
* ``config.toml`` never carries a ``[site_aliases]`` table: the alias file is the only source, and no tool writes it
  (``registry config --write-default`` and ``replica init`` rewrite ``config.toml`` only, so an alias table there could
  be erased while immutable records still need it)

An absent file is the empty table, whose digest is :data:`EMPTY_DIGEST`. With the empty table a legacy spelling is
refused exactly as any unknown label is: there is no silent fallback.

The process holds one ACTIVE table (:func:`install`, :func:`install_table`). Every fold builds an ``Ingestor`` over a
root, and the Ingestor installs that root's file, so the grading host's harvester, a rebuild and the replica's fold each
map through the table of the root they fold. A ``--db`` migration takes an explicit file (:func:`load_table`) checked
against a declared digest.
"""
from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple

from .config import CONFIG_NAME, ConfigError

#: The file name in a registry root (the grading host's ``~/registry`` and the cluster replica root alike).
FILE_NAME = ALIASES_NAME = "site_aliases.toml"
#: The one table the file carries; ``config.toml`` may not carry it.
TABLE_NAME = TABLE = "site_aliases"
#: The shape of a legacy spelling: a lowercase identifier, so it is safe as a TOML bare key, a dict key and a path part.
SPELLING_RE = re.compile(r"^[a-z][a-z0-9_]*$")
#: The vocabularies whose values are site labels: every column typed by one of them is mapped (:func:`substrate_columns`).
LABEL_VOCABS: Tuple[str, ...] = ("SUBSTRATES", "FIT_SEARCH_SUBSTRATES", "MIRROR_DESTINATIONS")


class SiteAliasError(ConfigError):
    """An alias table, or a file that should hold one, refused by name: the message names the file, the entry and the rule."""


def canonical_json(table: Mapping[str, str]) -> str:
    """The table's canonical JSON: keys sorted, no whitespace. The digest is the sha256 of these bytes."""
    return json.dumps({str(k): str(v) for k, v in table.items()}, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(table: Mapping[str, str]) -> str:
    """sha256 of :func:`canonical_json`; the empty table's is :data:`EMPTY_DIGEST`."""
    return hashlib.sha256(canonical_json(table).encode("ascii")).hexdigest()


EMPTY_DIGEST = digest({})


def labels() -> Tuple[str, ...]:
    """Every value a label column admits (the union of :data:`LABEL_VOCABS`), in vocabulary order."""
    from . import vocab as V
    out: List[str] = []
    for name in LABEL_VOCABS:
        for v in V.VOCABULARIES[name]:
            if v not in out:
                out.append(v)
    return tuple(out)


def validate(table: Any, source: str = "<site aliases>") -> Dict[str, str]:
    """The table as a plain dict sorted by spelling, or :class:`SiteAliasError` naming EVERY problem: not a table, a spelling
    that is not ``^[a-z][a-z0-9_]*$``, a spelling that is itself a label (shadowing), a value that is not a string or not one
    of ``vocab.SUBSTRATES``, and a value that is also a spelling (a chain)."""
    from . import vocab as V
    if table is None:
        return {}
    if not isinstance(table, Mapping):
        raise SiteAliasError(f"{source}: [{TABLE_NAME}] must be a table of spelling = \"label\" pairs, got {type(table).__name__}")
    known = set(labels())
    problems: List[str] = []
    for k in sorted(table, key=str):
        v = table[k]
        if not isinstance(k, str) or not SPELLING_RE.match(k):
            problems.append(f"spelling {k!r} does not match {SPELLING_RE.pattern}")
            continue
        if k in known:
            problems.append(f"spelling {k!r} is itself a label: an alias may not shadow a label")
        if not isinstance(v, str):
            problems.append(f"{k} = {v!r}: the label must be a string")
            continue
        if v not in V.SUBSTRATES:
            problems.append(f"{k} = {v!r}: {v!r} is not a label ({'|'.join(V.SUBSTRATES)})")
        if v in table:
            problems.append(f"{k} = {v!r}: {v!r} is itself a spelling in this table: aliases do not chain")
    if problems:
        raise SiteAliasError(f"{source}: the [{TABLE_NAME}] table is refused: " + "; ".join(problems))
    return {str(k): str(table[k]) for k in sorted(table)}


def _read_toml(path: str) -> Dict[str, Any]:
    import tomllib
    try:
        with open(path, "rb") as fh:
            return tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise SiteAliasError(f"{path}: not valid TOML ({exc}); nothing is read from it") from exc


def load_table(path: str) -> Dict[str, str]:
    """The validated ``[site_aliases]`` table of the TOML file at ``path``. The file must exist and carry that table and
    nothing else (another top-level key is refused by name, so a misspelt table never reads as the empty one)."""
    if not os.path.isfile(path):
        raise SiteAliasError(f"{path}: no such alias file")
    data = _read_toml(path)
    other = sorted(k for k in data if k != TABLE_NAME)
    if other:
        raise SiteAliasError(f"{path}: an alias file carries only the [{TABLE_NAME}] table; found {', '.join(other)}")
    if TABLE_NAME not in data:
        raise SiteAliasError(f"{path}: no [{TABLE_NAME}] table (an empty table is written as a bare [{TABLE_NAME}] header)")
    return validate(data[TABLE_NAME], path)


def root_file(root: str) -> str:
    return os.path.join(root, FILE_NAME)


def refuse_config_table(path: str) -> None:
    """Refuse, by name, a ``config.toml`` at ``path`` that carries a ``[site_aliases]`` table. An absent file passes."""
    if not os.path.isfile(path):
        return
    if TABLE_NAME in _read_toml(path):
        raise SiteAliasError(config_table_refusal(path))


def config_table_refusal(path: str) -> str:
    return (f"{path}: carries a [{TABLE_NAME}] table; aliases live only in {FILE_NAME} beside it, which no tool rewrites. "
            f"Move the table there and remove it from {os.path.basename(path)}")


def load_root(root: str) -> Dict[str, str]:
    """The validated table of ``<root>/site_aliases.toml``; an absent file is the empty table. A ``[site_aliases]`` table in
    ``<root>/config.toml`` is refused by name (one source only)."""
    refuse_config_table(os.path.join(root, CONFIG_NAME))
    path = root_file(root)
    if not os.path.exists(path):
        return {}
    return load_table(path)


def pairs(table: Mapping[str, str]) -> List[Tuple[str, str]]:
    """``[(spelling, label)]`` sorted by spelling, the order every printed form uses."""
    return sorted((str(k), str(v)) for k, v in table.items())


# --------------------------------------------------------------------------------------------
# the process's active table
# --------------------------------------------------------------------------------------------

class _Active:
    table: Dict[str, str] = {}
    digest: str = EMPTY_DIGEST
    source: Optional[str] = None
    stamp: Optional[Tuple[str, Optional[Tuple[int, int, int]]]] = None
    inverse: Dict[str, Tuple[str, ...]] = {}


def _file_stamp(root: str) -> Tuple[str, Optional[Tuple[int, int, int]]]:
    path = root_file(root)
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return (os.path.realpath(root), None)
    return (os.path.realpath(root), (st.st_ino, st.st_size, st.st_mtime_ns))


def _set_active(table: Mapping[str, str], source: Optional[str], stamp=None) -> Dict[str, str]:
    t = dict(table)
    _Active.table = t
    _Active.digest = digest(t)
    _Active.source = source
    _Active.stamp = stamp
    inv: Dict[str, List[str]] = {}
    for k, v in pairs(t):
        inv.setdefault(v, []).append(k)
    _Active.inverse = {v: tuple(ks) for v, ks in inv.items()}
    return dict(t)


def install(source: Any) -> Dict[str, str]:
    """Make a table the process's active table and return it. ``source`` is one of:

    * a registry root directory, or a configuration as ``config.load`` returns it (its ``registry.root``): that root's
      :func:`load_root` table, loaded once per process per root (a second call for the same root re-reads the file only when
      it changed: inode, size or mtime), so a fold may call it freely
    * an alias table already loaded (a mapping of spelling to label), validated again here, as :func:`install_table`
    * None: the empty table"""
    if source is None:
        reset()
        return {}
    if isinstance(source, Mapping):
        reg = source.get("registry")
        if not isinstance(reg, Mapping):
            return install_table(source)
        root = str(reg["root"])
    else:
        root = os.fspath(source)
    stamp = _file_stamp(root)
    if _Active.stamp == stamp:
        return dict(_Active.table)
    table = load_root(root)
    return _set_active(table, root_file(root) if stamp[1] is not None else None, stamp)


def install_table(table: Mapping[str, str], source: str = "<explicit>") -> Dict[str, str]:
    """Make an explicit, already loaded table the active one (validated again here)."""
    return _set_active(validate(table, source), source)


def reset() -> None:
    """Back to the empty table (tests, and a process that folds a second root without one)."""
    _set_active({}, None)


def active() -> Tuple[Dict[str, str], str, Optional[str]]:
    """``(table, digest, source)`` of the active table; ``source`` is the file it came from, or None for the empty table."""
    return dict(_Active.table), _Active.digest, _Active.source


def installed() -> Dict[str, str]:
    """A copy of the active table."""
    return dict(_Active.table)


def canonical(value: Any) -> Any:
    """The label a legacy spelling stands for under the active table, else ``value`` unchanged (a label, an unknown value,
    None or a non-string alike: the caller's vocabulary check decides those)."""
    if isinstance(value, str):
        return _Active.table.get(value, value)
    return value


def _label_re(label: str) -> "re.Pattern[str]":
    return re.compile(r"(?<![A-Za-z0-9])" + re.escape(label) + r"(?![A-Za-z0-9])")


def legacy_keys(key: str) -> Iterator[str]:
    """``key`` first, then every spelling of it in which a label that appears as a whole token (delimited by anything but a
    letter or digit, so ``cluster_a`` inside ``verified_cluster_a`` or ``cluster_a_path``) is replaced by a legacy spelling
    the active table maps to it, every combination once. For reading a persisted mapping whose keys an older writer
    spelled with the legacy name."""
    seen = {key}
    yield key
    hits = [(lab, olds) for lab, olds in _Active.inverse.items() if _label_re(lab).search(key)]
    if not hits:
        return
    choices = [[lab] + list(olds) for lab, olds in hits]
    for combo in itertools.product(*choices):
        out = key
        for (lab, _), pick in zip(hits, combo):
            if pick != lab:
                out = _label_re(lab).sub(pick, out)
        if out not in seen:
            seen.add(out)
            yield out


_MISSING = object()


def get_compat(mapping: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """``mapping[k]`` for the first spelling ``k`` of :func:`legacy_keys` present in ``mapping``, else ``default``."""
    for k in legacy_keys(key):
        v = mapping.get(k, _MISSING) if isinstance(mapping, Mapping) else _MISSING
        if v is not _MISSING:
            return v
    return default


# --------------------------------------------------------------------------------------------
# the label columns
# --------------------------------------------------------------------------------------------

def substrate_columns() -> Tuple[Tuple[str, str, str], ...]:
    """``(table, column, vocabulary)`` for every schema column typed by a label vocabulary (:data:`LABEL_VOCABS`), in
    schema order: ``run.substrate``, ``object.fit_substrate``, ``fit_search.substrate``, ``analysis.substrate`` and
    ``mirror_ack.destination``."""
    from . import schema_def as S
    return tuple((t.name, c.name, str(c.vocab)) for t in S.TABLES for c in t.columns if c.vocab in LABEL_VOCABS)


#: ``(table, column)`` of every label column, in schema order (:func:`substrate_columns` without the vocabulary).
LABEL_COLUMNS: Tuple[Tuple[str, str], ...] = tuple((t, c) for t, c, _ in substrate_columns())


def _columns_by_table() -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, List[str]] = {}
    for t, c, _ in substrate_columns():
        out.setdefault(t, []).append(c)
    return {t: tuple(cs) for t, cs in out.items()}


_COLUMNS_BY_TABLE: Optional[Dict[str, Tuple[str, ...]]] = None


def label_columns(table: str) -> Tuple[str, ...]:
    """The label columns of one table (empty for a table without one)."""
    global _COLUMNS_BY_TABLE
    if _COLUMNS_BY_TABLE is None:
        _COLUMNS_BY_TABLE = _columns_by_table()
    return _COLUMNS_BY_TABLE.get(table, ())


def canonical_row(table: str, row: Dict[str, Any]) -> Dict[str, Any]:
    """``row`` with every label column of ``table`` mapped by :func:`canonical`. The input is never changed: a column that
    maps gives a new dict, and ``row`` itself is returned when nothing maps (the empty table, or no label column)."""
    if not _Active.table:
        return row
    changed = {c: canonical(row[c]) for c in label_columns(table) if c in row and canonical(row[c]) != row[c]}
    if not changed:
        return row
    out = dict(row)
    out.update(changed)
    return out


def is_label_column(table: str, column: str) -> bool:
    return column in label_columns(table)


def census(conn, table: Mapping[str, str], columns: Optional[Sequence[Tuple[str, str, str]]] = None) -> List[Dict[str, Any]]:
    """Per label column and stored value of ``conn`` (read only): ``{table, column, vocab, value, rows, label, status}`` with
    ``status`` one of ``label`` (already a label of that column), ``aliased`` (``table`` maps it to a label the column
    admits), ``unaliased`` (neither) or ``alias_not_admitted`` (``table`` maps it to a label the column does not admit, such
    as a host label on a mirror destination). Tables or columns ``conn`` does not have are skipped."""
    from . import vocab as V
    out: List[Dict[str, Any]] = []
    have = {str(r[0]) for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}
    for t, c, vocab in (columns if columns is not None else substrate_columns()):
        if t not in have:
            continue
        cols = {str(r[1]) for r in conn.execute(f"PRAGMA table_info({t})").fetchall()}
        if c not in cols:
            continue
        allowed = set(V.VOCABULARIES[vocab])
        for value, n in conn.execute(f"SELECT {c}, COUNT(*) FROM {t} WHERE {c} IS NOT NULL GROUP BY {c} ORDER BY {c}").fetchall():
            value = str(value)
            if value in allowed:
                status, label = "label", value
            elif value in table:
                label = table[value]
                status = "aliased" if label in allowed else "alias_not_admitted"
            else:
                status, label = "unaliased", None
            out.append({"table": t, "column": c, "vocab": vocab, "value": value, "rows": int(n), "label": label, "status": status})
    return out
